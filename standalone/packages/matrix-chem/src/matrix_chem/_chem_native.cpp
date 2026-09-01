#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#define MATRIX_STRINGIFY_DETAIL(value) #value
#define MATRIX_STRINGIFY(value) MATRIX_STRINGIFY_DETAIL(value)

namespace {

using Edge = std::pair<npy_intp, npy_intp>;
using Cycle = std::vector<npy_intp>;
using BitVector = std::vector<std::uint64_t>;

struct CycleOrder {
    bool operator()(const Cycle& left, const Cycle& right) const {
        if (left.size() != right.size()) {
            return left.size() < right.size();
        }
        return left < right;
    }
};

npy_intp cycle_rank(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges);
std::set<Cycle, CycleOrder> horton_candidates(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& all_edges,
    npy_intp ring_max);
std::vector<Cycle> minimum_cycle_basis(
    const std::set<Cycle, CycleOrder>& candidates,
    const std::vector<Edge>& edges,
    npy_intp target_rank);

template <typename T>
PyObject* numpy_vector(const std::vector<T>& values, int type) {
    npy_intp dimension = static_cast<npy_intp>(values.size());
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, &dimension, type));
    if (array == nullptr) {
        return nullptr;
    }
    std::copy(values.begin(), values.end(), static_cast<T*>(PyArray_DATA(array)));
    return reinterpret_cast<PyObject*>(array);
}

double effective_radius(
    npy_intp atomic_number,
    double coordination,
    const double* standard_radii,
    const double* pyykko,
    npy_intp pyykko_columns) {
    if (atomic_number < 0 || atomic_number > 118) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<npy_intp> keys;
    for (npy_intp key = 0; key < pyykko_columns; ++key) {
        if (std::isfinite(pyykko[atomic_number * pyykko_columns + key])) {
            keys.push_back(key);
        }
    }
    if (keys.empty()) {
        return standard_radii[atomic_number];
    }
    if (keys.size() == 1 || coordination <= keys.front()) {
        return pyykko[atomic_number * pyykko_columns + keys.front()];
    }
    if (coordination >= keys.back()) {
        return pyykko[atomic_number * pyykko_columns + keys.back()];
    }
    auto upper = std::upper_bound(keys.begin(), keys.end(), coordination);
    const npy_intp upper_key = *upper;
    const npy_intp lower_key = *(upper - 1);
    const double lower_radius =
        pyykko[atomic_number * pyykko_columns + lower_key];
    const double upper_radius =
        pyykko[atomic_number * pyykko_columns + upper_key];
    const double t =
        (coordination - lower_key) / static_cast<double>(upper_key - lower_key);
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double interval_slope = upper_radius - lower_radius;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * lower_radius +
        (t3 - 2.0 * t2 + t) * interval_slope +
        (-2.0 * t3 + 3.0 * t2) * upper_radius +
        (t3 - t2) * interval_slope;
}

PyObject* perceive_continuous_graph(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *numbers_object, *standard_object, *pyykko_object;
    double cutoff, cna_alpha, distance_scale, switch_alpha, lambda_strong, lambda_weak;
    if (!PyArg_ParseTuple(
            args, "OOOOdddddd",
            &coordinates_object, &numbers_object, &standard_object, &pyykko_object,
            &cutoff, &cna_alpha, &distance_scale, &switch_alpha,
            &lambda_strong, &lambda_weak)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* numbers = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(numbers_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* standard = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(standard_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* pyykko = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pyykko_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || numbers == nullptr || standard == nullptr ||
        pyykko == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(numbers);
        Py_XDECREF(standard);
        Py_XDECREF(pyykko);
        return nullptr;
    }
    const npy_intp natoms = PyArray_DIM(coordinates, 0);
    if (PyArray_NDIM(coordinates) != 2 || PyArray_DIM(coordinates, 1) != 3 ||
        PyArray_NDIM(numbers) != 1 || PyArray_DIM(numbers, 0) != natoms ||
        PyArray_NDIM(standard) != 1 || PyArray_DIM(standard, 0) < 119 ||
        PyArray_NDIM(pyykko) != 2 || PyArray_DIM(pyykko, 0) < 119) {
        Py_DECREF(coordinates);
        Py_DECREF(numbers);
        Py_DECREF(standard);
        Py_DECREF(pyykko);
        PyErr_SetString(PyExc_ValueError, "native continuous-graph arrays are invalid");
        return nullptr;
    }
    const auto* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const auto* z = static_cast<const npy_intp*>(PyArray_DATA(numbers));
    const auto* radii = static_cast<const double*>(PyArray_DATA(standard));
    const auto* pyykko_data = static_cast<const double*>(PyArray_DATA(pyykko));
    const npy_intp pyykko_columns = PyArray_DIM(pyykko, 1);
    std::vector<npy_intp> pair_left;
    std::vector<npy_intp> pair_right;
    std::vector<double> pair_distance;
    std::vector<double> coordination(natoms, 0.0);
    std::vector<double> effective(natoms, 0.0);
    std::vector<npy_intp> discrete_left;
    std::vector<npy_intp> discrete_right;
    std::vector<double> connectivity;
    std::vector<npy_intp> accepted_left;
    std::vector<npy_intp> accepted_right;
    npy_intp accepted_cycle_rank = 0;
    std::set<Cycle, CycleOrder> accepted_candidates;
    std::vector<Cycle> accepted_cycles;
    Py_BEGIN_ALLOW_THREADS
    const double cutoff2 = cutoff * cutoff;
    for (npy_intp left = 0; left < natoms; ++left) {
        for (npy_intp right = left + 1; right < natoms; ++right) {
            const double dx = xyz[3 * left] - xyz[3 * right];
            const double dy = xyz[3 * left + 1] - xyz[3 * right + 1];
            const double dz = xyz[3 * left + 2] - xyz[3 * right + 2];
            const double squared = dx * dx + dy * dy + dz * dz;
            if (squared > cutoff2) {
                continue;
            }
            const double distance = std::sqrt(squared);
            pair_left.push_back(left);
            pair_right.push_back(right);
            pair_distance.push_back(distance);
            if (z[left] >= 0 && z[left] <= 118 && z[right] >= 0 && z[right] <= 118) {
                const double radius_sum = radii[z[left]] + radii[z[right]];
                if (std::isfinite(radius_sum)) {
                    const double contribution =
                        0.5 * (1.0 + std::erf(cna_alpha * (radius_sum - distance)));
                    coordination[left] += contribution;
                    coordination[right] += contribution;
                }
            }
        }
    }
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        effective[atom] = effective_radius(
            z[atom], coordination[atom], radii, pyykko_data, pyykko_columns);
    }
    for (std::size_t pair = 0; pair < pair_left.size(); ++pair) {
        const npy_intp left = pair_left[pair];
        const npy_intp right = pair_right[pair];
        if (z[left] < 0 || z[left] > 118 || z[right] < 0 || z[right] > 118) {
            continue;
        }
        const double standard_sum = radii[z[left]] + radii[z[right]];
        if (!std::isfinite(standard_sum) ||
            pair_distance[pair] > distance_scale * standard_sum) {
            continue;
        }
        const double reference = effective[left] + effective[right];
        double value = 0.0;
        if (reference > 1.0e-12) {
            const double reduced = (pair_distance[pair] - reference) / reference;
            const double strong_weight =
                0.5 * (1.0 - std::tanh(switch_alpha * reduced));
            const double strong =
                std::exp((reference - pair_distance[pair]) / lambda_strong);
            const double weak =
                std::exp((reference - pair_distance[pair]) / lambda_weak);
            value = strong_weight * strong + (1.0 - strong_weight) * weak;
        }
        discrete_left.push_back(left);
        discrete_right.push_back(right);
        connectivity.push_back(value);
    }
    std::vector<unsigned char> heavy_partner(natoms, 0);
    for (std::size_t pair = 0; pair < connectivity.size(); ++pair) {
        if (connectivity[pair] < 0.2) {
            continue;
        }
        const npy_intp left = discrete_left[pair];
        const npy_intp right = discrete_right[pair];
        if (z[right] != 1) {
            heavy_partner[left] = 1;
        }
        if (z[left] != 1) {
            heavy_partner[right] = 1;
        }
    }
    for (std::size_t pair = 0; pair < connectivity.size(); ++pair) {
        if (connectivity[pair] < 0.2) {
            continue;
        }
        const npy_intp left = discrete_left[pair];
        const npy_intp right = discrete_right[pair];
        if (z[left] == 1 && z[right] == 1 &&
            (heavy_partner[left] || heavy_partner[right])) {
            continue;
        }
        accepted_left.push_back(left);
        accepted_right.push_back(right);
    }
    std::vector<Edge> accepted_edges;
    accepted_edges.reserve(accepted_left.size());
    for (std::size_t index = 0; index < accepted_left.size(); ++index) {
        accepted_edges.emplace_back(accepted_left[index], accepted_right[index]);
    }
    std::vector<unsigned char> all_allowed(natoms, 1);
    accepted_cycle_rank = cycle_rank(natoms, all_allowed, accepted_edges);
    if (accepted_cycle_rank > 0) {
        accepted_candidates = horton_candidates(
            natoms, all_allowed, accepted_edges, -1);
        accepted_cycles = minimum_cycle_basis(
            accepted_candidates, accepted_edges, accepted_cycle_rank);
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(numbers);
    Py_DECREF(standard);
    Py_DECREF(pyykko);
    PyObject* outputs[] = {
        numpy_vector(pair_left, NPY_INTP),
        numpy_vector(pair_right, NPY_INTP),
        numpy_vector(pair_distance, NPY_DOUBLE),
        numpy_vector(coordination, NPY_DOUBLE),
        numpy_vector(effective, NPY_DOUBLE),
        numpy_vector(discrete_left, NPY_INTP),
        numpy_vector(discrete_right, NPY_INTP),
        numpy_vector(connectivity, NPY_DOUBLE),
        numpy_vector(accepted_left, NPY_INTP),
        numpy_vector(accepted_right, NPY_INTP),
    };
    for (PyObject* output : outputs) {
        if (output == nullptr) {
            for (PyObject* owned : outputs) {
                Py_XDECREF(owned);
            }
            return nullptr;
        }
    }
    PyObject* cycles = PyTuple_New(
        static_cast<Py_ssize_t>(accepted_cycles.size()));
    if (cycles == nullptr) {
        for (PyObject* output : outputs) {
            Py_DECREF(output);
        }
        return nullptr;
    }
    for (std::size_t index = 0; index < accepted_cycles.size(); ++index) {
        PyObject* cycle = PyTuple_New(
            static_cast<Py_ssize_t>(accepted_cycles[index].size()));
        if (cycle == nullptr) {
            Py_DECREF(cycles);
            for (PyObject* output : outputs) {
                Py_DECREF(output);
            }
            return nullptr;
        }
        for (std::size_t atom = 0; atom < accepted_cycles[index].size(); ++atom) {
            PyTuple_SET_ITEM(
                cycle,
                static_cast<Py_ssize_t>(atom),
                PyLong_FromSsize_t(accepted_cycles[index][atom]));
        }
        PyTuple_SET_ITEM(cycles, static_cast<Py_ssize_t>(index), cycle);
    }
    PyObject* result = PyTuple_New(13);
    if (result == nullptr) {
        for (PyObject* output : outputs) {
            Py_DECREF(output);
        }
        return nullptr;
    }
    for (int index = 0; index < 10; ++index) {
        PyTuple_SET_ITEM(result, index, outputs[index]);
    }
    PyTuple_SET_ITEM(result, 10, cycles);
    PyTuple_SET_ITEM(
        result,
        11,
        PyLong_FromSsize_t(
            static_cast<Py_ssize_t>(accepted_candidates.size())));
    PyTuple_SET_ITEM(
        result,
        12,
        PyLong_FromSsize_t(static_cast<Py_ssize_t>(accepted_cycle_rank)));
    return result;
}

Cycle canonical_cycle(const Cycle& cycle) {
    if (cycle.empty()) {
        return {};
    }
    auto minimum = std::min_element(cycle.begin(), cycle.end());
    const std::size_t start = static_cast<std::size_t>(minimum - cycle.begin());
    Cycle forward;
    Cycle backward;
    forward.reserve(cycle.size());
    backward.reserve(cycle.size());
    for (std::size_t offset = 0; offset < cycle.size(); ++offset) {
        forward.push_back(cycle[(start + offset) % cycle.size()]);
        backward.push_back(cycle[(start + cycle.size() - offset) % cycle.size()]);
    }
    return std::min(forward, backward);
}

bool chordless(const std::vector<std::vector<npy_intp>>& adjacency, const Cycle& cycle) {
    std::vector<npy_intp> position(adjacency.size(), -1);
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        position[cycle[index]] = static_cast<npy_intp>(index);
    }
    for (std::size_t index = 0; index < cycle.size(); ++index) {
        const npy_intp atom = cycle[index];
        const npy_intp previous = cycle[(index + cycle.size() - 1) % cycle.size()];
        const npy_intp following = cycle[(index + 1) % cycle.size()];
        for (const npy_intp neighbor : adjacency[atom]) {
            if (position[neighbor] >= 0 && neighbor != previous && neighbor != following) {
                return false;
            }
        }
    }
    return true;
}

std::set<Cycle, CycleOrder> split_at_chords(
    const std::vector<std::vector<npy_intp>>& adjacency, const Cycle& initial) {
    std::vector<Cycle> pending{canonical_cycle(initial)};
    std::set<Cycle, CycleOrder> result;
    while (!pending.empty()) {
        Cycle current = std::move(pending.back());
        pending.pop_back();
        std::vector<npy_intp> position(adjacency.size(), -1);
        for (std::size_t index = 0; index < current.size(); ++index) {
            position[current[index]] = static_cast<npy_intp>(index);
        }
        std::pair<npy_intp, npy_intp> chord{-1, -1};
        for (std::size_t index = 0; index < current.size() && chord.first < 0; ++index) {
            const npy_intp atom = current[index];
            const npy_intp previous =
                current[(index + current.size() - 1) % current.size()];
            const npy_intp following = current[(index + 1) % current.size()];
            for (const npy_intp neighbor : adjacency[atom]) {
                if (position[neighbor] >= 0 && neighbor != previous && neighbor != following) {
                    chord = {
                        static_cast<npy_intp>(index),
                        position[neighbor],
                    };
                    break;
                }
            }
        }
        if (chord.first < 0) {
            result.insert(canonical_cycle(current));
            continue;
        }
        npy_intp first = std::min(chord.first, chord.second);
        npy_intp second = std::max(chord.first, chord.second);
        Cycle part_a(
            current.begin() + first,
            current.begin() + second + 1);
        Cycle part_b(current.begin() + second, current.end());
        part_b.insert(part_b.end(), current.begin(), current.begin() + first + 1);
        if (part_a.size() >= 3) {
            pending.push_back(canonical_cycle(part_a));
        }
        if (part_b.size() >= 3) {
            pending.push_back(canonical_cycle(part_b));
        }
    }
    return result;
}

Cycle tree_cycle(
    npy_intp left, npy_intp right,
    const std::vector<npy_intp>& parent,
    const std::vector<npy_intp>& depth) {
    if (parent[left] == -2 || parent[right] == -2) {
        return {};
    }
    Cycle left_path;
    Cycle right_path;
    npy_intp a = left;
    npy_intp b = right;
    while (depth[a] > depth[b]) {
        left_path.push_back(a);
        if (parent[a] < 0) {
            return {};
        }
        a = parent[a];
    }
    while (depth[b] > depth[a]) {
        right_path.push_back(b);
        if (parent[b] < 0) {
            return {};
        }
        b = parent[b];
    }
    while (a != b) {
        left_path.push_back(a);
        right_path.push_back(b);
        if (parent[a] < 0 || parent[b] < 0) {
            return {};
        }
        a = parent[a];
        b = parent[b];
    }
    left_path.push_back(a);
    left_path.insert(left_path.end(), right_path.rbegin(), right_path.rend());
    return left_path;
}

std::vector<unsigned char> cyclic_core(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges) {
    std::vector<std::set<npy_intp>> mutable_adjacency(natoms);
    std::vector<unsigned char> core = allowed;
    for (const auto& [left, right] : edges) {
        mutable_adjacency[left].insert(right);
        mutable_adjacency[right].insert(left);
    }
    std::vector<npy_intp> pending;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (core[atom] && mutable_adjacency[atom].size() < 2) {
            pending.push_back(atom);
        }
    }
    while (!pending.empty()) {
        const npy_intp atom = pending.back();
        pending.pop_back();
        if (!core[atom]) {
            continue;
        }
        core[atom] = 0;
        for (const npy_intp neighbor : mutable_adjacency[atom]) {
            if (!core[neighbor]) {
                continue;
            }
            mutable_adjacency[neighbor].erase(atom);
            if (mutable_adjacency[neighbor].size() < 2) {
                pending.push_back(neighbor);
            }
        }
    }
    return core;
}

npy_intp cycle_rank(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& edges) {
    std::vector<npy_intp> parent(natoms);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](npy_intp atom) {
        npy_intp current = atom;
        while (parent[current] != current) {
            parent[current] = parent[parent[current]];
            current = parent[current];
        }
        return current;
    };
    for (const auto& [left, right] : edges) {
        npy_intp root_left = find(left);
        npy_intp root_right = find(right);
        if (root_left != root_right) {
            parent[root_right] = root_left;
        }
    }
    std::set<npy_intp> components;
    npy_intp allowed_count = 0;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (allowed[atom]) {
            ++allowed_count;
            components.insert(find(atom));
        }
    }
    return std::max<npy_intp>(
        0,
        static_cast<npy_intp>(edges.size()) - allowed_count +
            static_cast<npy_intp>(components.size()));
}

std::set<Cycle, CycleOrder> horton_candidates(
    npy_intp natoms,
    const std::vector<unsigned char>& allowed,
    const std::vector<Edge>& all_edges,
    npy_intp ring_max) {
    const auto core = cyclic_core(natoms, allowed, all_edges);
    std::vector<std::vector<npy_intp>> adjacency(natoms);
    std::vector<Edge> edges;
    for (const auto& [left, right] : all_edges) {
        if (core[left] && core[right]) {
            edges.emplace_back(left, right);
            adjacency[left].push_back(right);
            adjacency[right].push_back(left);
        }
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
    }
    std::set<npy_intp> roots;
    for (npy_intp atom = 0; atom < natoms; ++atom) {
        if (core[atom] && adjacency[atom].size() != 2) {
            roots.insert(atom);
        }
    }
    std::vector<unsigned char> unseen = core;
    for (npy_intp seed = 0; seed < natoms; ++seed) {
        if (!unseen[seed]) {
            continue;
        }
        std::vector<npy_intp> component{seed};
        unseen[seed] = 0;
        for (std::size_t index = 0; index < component.size(); ++index) {
            for (const npy_intp neighbor : adjacency[component[index]]) {
                if (unseen[neighbor]) {
                    unseen[neighbor] = 0;
                    component.push_back(neighbor);
                }
            }
        }
        bool has_root = false;
        for (const npy_intp atom : component) {
            if (roots.count(atom)) {
                has_root = true;
                break;
            }
        }
        if (!has_root) {
            roots.insert(*std::min_element(component.begin(), component.end()));
        }
    }

    std::set<Cycle, CycleOrder> candidates;
    std::set<Cycle, CycleOrder> processed;
    for (const npy_intp root : roots) {
        std::vector<npy_intp> parent(natoms, -2);
        std::vector<npy_intp> depth(natoms, -1);
        std::vector<npy_intp> queue{root};
        parent[root] = -1;
        depth[root] = 0;
        for (std::size_t index = 0; index < queue.size(); ++index) {
            const npy_intp atom = queue[index];
            for (const npy_intp neighbor : adjacency[atom]) {
                if (parent[neighbor] != -2) {
                    continue;
                }
                parent[neighbor] = atom;
                depth[neighbor] = depth[atom] + 1;
                queue.push_back(neighbor);
            }
        }
        for (const auto& [left, right] : edges) {
            if (parent[left] == right || parent[right] == left) {
                continue;
            }
            Cycle cycle = tree_cycle(left, right, parent, depth);
            if (cycle.size() < 3) {
                continue;
            }
            Cycle canonical = canonical_cycle(cycle);
            if (ring_max >= 0 &&
                static_cast<npy_intp>(canonical.size()) > ring_max) {
                continue;
            }
            if (!processed.insert(canonical).second) {
                continue;
            }
            if (chordless(adjacency, canonical)) {
                candidates.insert(canonical);
            } else {
                const auto parts = split_at_chords(adjacency, canonical);
                for (const Cycle& part : parts) {
                    if (ring_max < 0 ||
                        static_cast<npy_intp>(part.size()) <= ring_max) {
                        candidates.insert(part);
                    }
                }
            }
        }
    }
    return candidates;
}

void xor_vector(BitVector& target, const BitVector& source) {
    for (std::size_t word = 0; word < target.size(); ++word) {
        target[word] ^= source[word];
    }
}

npy_intp highest_bit(const BitVector& vector) {
    for (npy_intp word = static_cast<npy_intp>(vector.size()) - 1; word >= 0; --word) {
        if (vector[word] != 0) {
#if defined(__GNUC__) || defined(__clang__)
            const int local = 63 - __builtin_clzll(vector[word]);
#else
            int local = 0;
            std::uint64_t value = vector[word];
            while (value >>= 1U) {
                ++local;
            }
#endif
            return 64 * word + local;
        }
    }
    return -1;
}

std::vector<Cycle> minimum_cycle_basis(
    const std::set<Cycle, CycleOrder>& candidates,
    const std::vector<Edge>& edges,
    npy_intp target_rank) {
    std::map<Edge, npy_intp> edge_index;
    for (std::size_t index = 0; index < edges.size(); ++index) {
        edge_index[edges[index]] = static_cast<npy_intp>(index);
    }
    const std::size_t words = (edges.size() + 63) / 64;
    std::map<npy_intp, BitVector, std::greater<npy_intp>> basis;
    std::vector<Cycle> selected;
    for (const Cycle& cycle : candidates) {
        BitVector reduced(words, 0);
        for (std::size_t index = 0; index < cycle.size(); ++index) {
            Edge edge = std::minmax(
                cycle[index],
                cycle[(index + 1) % cycle.size()]);
            const npy_intp position = edge_index.at(edge);
            reduced[position / 64] ^= std::uint64_t{1} << (position % 64);
        }
        for (const auto& [pivot, row] : basis) {
            if (reduced[pivot / 64] & (std::uint64_t{1} << (pivot % 64))) {
                xor_vector(reduced, row);
            }
        }
        const npy_intp pivot = highest_bit(reduced);
        if (pivot < 0) {
            continue;
        }
        basis[pivot] = reduced;
        selected.push_back(cycle);
        if (static_cast<npy_intp>(selected.size()) >= target_rank) {
            break;
        }
    }
    return selected;
}

bool parse_graph(
    PyObject* edges_object,
    PyObject* allowed_object,
    npy_intp natoms,
    std::vector<Edge>& edges,
    std::vector<unsigned char>& allowed) {
    PyArrayObject* edge_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(edges_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* allowed_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(allowed_object, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (edge_array == nullptr || allowed_array == nullptr) {
        Py_XDECREF(edge_array);
        Py_XDECREF(allowed_array);
        return false;
    }
    if (natoms < 0 || PyArray_NDIM(edge_array) != 2 ||
        PyArray_DIM(edge_array, 1) != 2 || PyArray_NDIM(allowed_array) != 1) {
        Py_DECREF(edge_array);
        Py_DECREF(allowed_array);
        PyErr_SetString(PyExc_ValueError, "native topology graph arrays are invalid");
        return false;
    }
    allowed.assign(natoms, 0);
    const auto* allowed_data =
        static_cast<const npy_intp*>(PyArray_DATA(allowed_array));
    for (npy_intp index = 0; index < PyArray_DIM(allowed_array, 0); ++index) {
        const npy_intp atom = allowed_data[index];
        if (atom < 0 || atom >= natoms) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(PyExc_ValueError, "allowed topology atom is out of range");
            return false;
        }
        allowed[atom] = 1;
    }
    const auto* edge_data = static_cast<const npy_intp*>(PyArray_DATA(edge_array));
    edges.reserve(PyArray_DIM(edge_array, 0));
    Edge previous{-1, -1};
    for (npy_intp index = 0; index < PyArray_DIM(edge_array, 0); ++index) {
        npy_intp left = edge_data[2 * index];
        npy_intp right = edge_data[2 * index + 1];
        if (left > right) {
            std::swap(left, right);
        }
        if (left < 0 || right >= natoms || left == right ||
            !allowed[left] || !allowed[right]) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(PyExc_ValueError, "native topology edge is invalid");
            return false;
        }
        const Edge edge{left, right};
        if (!edges.empty() && edge <= previous) {
            Py_DECREF(edge_array);
            Py_DECREF(allowed_array);
            PyErr_SetString(
                PyExc_ValueError,
                "native topology edges must be unique and canonically sorted");
            return false;
        }
        edges.push_back(edge);
        previous = edge;
    }
    Py_DECREF(edge_array);
    Py_DECREF(allowed_array);
    return true;
}

PyObject* elementary_cycle_basis_native(PyObject*, PyObject* args) {
    Py_ssize_t natoms_raw;
    PyObject* edges_object;
    PyObject* allowed_object;
    PyObject* ring_max_object;
    if (!PyArg_ParseTuple(
            args, "nOOO",
            &natoms_raw, &edges_object, &allowed_object, &ring_max_object)) {
        return nullptr;
    }
    npy_intp ring_max = -1;
    if (ring_max_object != Py_None) {
        ring_max = PyLong_AsSsize_t(ring_max_object);
        if (ring_max == -1 && PyErr_Occurred()) {
            return nullptr;
        }
        if (ring_max < 3) {
            PyErr_SetString(PyExc_ValueError, "ring maximum must be at least three");
            return nullptr;
        }
    }
    const npy_intp natoms = static_cast<npy_intp>(natoms_raw);
    std::vector<Edge> edges;
    std::vector<unsigned char> allowed;
    if (!parse_graph(edges_object, allowed_object, natoms, edges, allowed)) {
        return nullptr;
    }
    npy_intp rank = 0;
    std::set<Cycle, CycleOrder> candidates;
    std::vector<Cycle> selected;
    Py_BEGIN_ALLOW_THREADS
    rank = cycle_rank(natoms, allowed, edges);
    if (rank > 0) {
        candidates = horton_candidates(natoms, allowed, edges, ring_max);
        selected = minimum_cycle_basis(candidates, edges, rank);
    }
    Py_END_ALLOW_THREADS
    PyObject* cycles = PyTuple_New(static_cast<Py_ssize_t>(selected.size()));
    if (cycles == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < selected.size(); ++index) {
        PyObject* cycle = PyTuple_New(static_cast<Py_ssize_t>(selected[index].size()));
        if (cycle == nullptr) {
            Py_DECREF(cycles);
            return nullptr;
        }
        for (std::size_t atom = 0; atom < selected[index].size(); ++atom) {
            PyTuple_SET_ITEM(
                cycle,
                static_cast<Py_ssize_t>(atom),
                PyLong_FromSsize_t(selected[index][atom]));
        }
        PyTuple_SET_ITEM(cycles, static_cast<Py_ssize_t>(index), cycle);
    }
    return Py_BuildValue(
        "Nnn",
        cycles,
        static_cast<Py_ssize_t>(candidates.size()),
        static_cast<Py_ssize_t>(rank));
}

PyObject* build_info(PyObject*, PyObject*) {
#if defined(__aarch64__) || defined(_M_ARM64)
    const char* architecture = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    const char* architecture = "x86_64";
#else
    const char* architecture = "generic";
#endif
#if defined(__clang__)
    const char* compiler =
        "clang-" MATRIX_STRINGIFY(__clang_major__) "." MATRIX_STRINGIFY(__clang_minor__);
#elif defined(__GNUC__)
    const char* compiler =
        "gcc-" MATRIX_STRINGIFY(__GNUC__) "." MATRIX_STRINGIFY(__GNUC_MINOR__);
#elif defined(_MSC_VER)
    const char* compiler = "msvc-" MATRIX_STRINGIFY(_MSC_VER);
#else
    const char* compiler = "unknown";
#endif
    return Py_BuildValue(
        "{s:s,s:s,s:s,s:O,s:s}",
        "implementation", "cpp",
        "architecture", architecture,
        "compiler", compiler,
        "openmp", Py_False,
        "precision", "integer-exact");
}

PyMethodDef methods[] = {
    {"perceive_continuous_graph", perceive_continuous_graph, METH_VARARGS,
     "Continuous geometry-first graph perception primitives."},
    {"elementary_cycle_basis", elementary_cycle_basis_native, METH_VARARGS,
     "Deterministic unweighted Horton minimum cycle basis."},
    {"build_info", build_info, METH_NOARGS, "Describe this native build."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_chem_native",
    "Portable compiled MATRIX chemistry kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__chem_native() {
    import_array();
    return PyModule_Create(&module);
}
