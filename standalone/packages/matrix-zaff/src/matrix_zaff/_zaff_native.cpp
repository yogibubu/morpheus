#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_24_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#define MATRIX_STRINGIFY_DETAIL(value) #value
#define MATRIX_STRINGIFY(value) MATRIX_STRINGIFY_DETAIL(value)

namespace {

constexpr double kCoincidentTolerance = 1.0e-12;
constexpr double kPenetrationArgument = 8.0;

struct Inputs {
    PyArrayObject* coordinates;
    PyArrayObject* charges;
    PyArrayObject* widths;
    npy_intp atoms;
};

struct PairInputs {
    PyArrayObject* pairs;
    npy_intp count;
};

struct RadialInputs {
    PyArrayObject* coordinates;
    PyArrayObject* epsilon;
    PyArrayObject* rmin_half;
    npy_intp atoms;
};

void release_radial_inputs(RadialInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.epsilon);
    Py_XDECREF(input.rmin_half);
}

void release_inputs(Inputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.charges);
    Py_XDECREF(input.widths);
}

bool parse_inputs(PyObject* coordinates, PyObject* charges, PyObject* widths, Inputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.widths = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(widths, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.charges == nullptr || out.widths == nullptr) {
        release_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 || PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.charges) != 1 || PyArray_NDIM(out.widths) != 1) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF arrays must have shapes (N,3), (N,), (N,)");
        release_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    if (PyArray_DIM(out.charges, 0) != out.atoms || PyArray_DIM(out.widths, 0) != out.atoms) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF array dimensions are inconsistent");
        release_inputs(out);
        return false;
    }
    const double* q = static_cast<const double*>(PyArray_DATA(out.charges));
    const double* w = static_cast<const double*>(PyArray_DATA(out.widths));
    const double* xyz = static_cast<const double*>(PyArray_DATA(out.coordinates));
    for (npy_intp atom = 0; atom < out.atoms; ++atom) {
        if (!std::isfinite(q[atom]) || !std::isfinite(w[atom]) || w[atom] <= 0.0) {
            PyErr_SetString(PyExc_ValueError, "native ZAFF charges and widths must be finite and widths positive");
            release_inputs(out);
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                PyErr_SetString(PyExc_ValueError, "native ZAFF coordinates must be finite");
                release_inputs(out);
                return false;
            }
        }
    }
    return true;
}

bool parse_pairs(PyObject* pairs, npy_intp atoms, PairInputs& out) {
    out.pairs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(pairs, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    if (out.pairs == nullptr) {
        return false;
    }
    if (PyArray_NDIM(out.pairs) != 2 || PyArray_DIM(out.pairs, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "native ZAFF pair list must have shape (P,2)");
        Py_DECREF(out.pairs);
        out.pairs = nullptr;
        return false;
    }
    out.count = PyArray_DIM(out.pairs, 0);
    const npy_intp* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.pairs));
    for (npy_intp pair = 0; pair < out.count; ++pair) {
        const npy_intp left = indices[2 * pair];
        const npy_intp right = indices[2 * pair + 1];
        if (left < 0 || right < 0 || left >= atoms || right >= atoms ||
            left >= right) {
            PyErr_SetString(
                PyExc_ValueError,
                "native ZAFF pairs must be canonical distinct in-range indices");
            Py_DECREF(out.pairs);
            out.pairs = nullptr;
            return false;
        }
    }
    return true;
}

bool parse_radial_inputs(
    PyObject* coordinates, PyObject* epsilon, PyObject* rmin_half,
    RadialInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.epsilon = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(epsilon, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.rmin_half = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(rmin_half, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.epsilon == nullptr ||
        out.rmin_half == nullptr) {
        release_radial_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.epsilon) != 1 ||
        PyArray_NDIM(out.rmin_half) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native Exp-PE arrays must have shapes (N,3), (N,), (N,)");
        release_radial_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    if (PyArray_DIM(out.epsilon, 0) != out.atoms ||
        PyArray_DIM(out.rmin_half, 0) != out.atoms) {
        PyErr_SetString(
            PyExc_ValueError, "native Exp-PE array dimensions are inconsistent");
        release_radial_inputs(out);
        return false;
    }
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(out.coordinates));
    const double* eps = static_cast<const double*>(PyArray_DATA(out.epsilon));
    const double* rmin =
        static_cast<const double*>(PyArray_DATA(out.rmin_half));
    for (npy_intp atom = 0; atom < out.atoms; ++atom) {
        if (!std::isfinite(eps[atom]) || eps[atom] <= 0.0 ||
            !std::isfinite(rmin[atom]) || rmin[atom] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError,
                "native Exp-PE parameters must be finite and positive");
            release_radial_inputs(out);
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                PyErr_SetString(
                    PyExc_ValueError, "native Exp-PE coordinates must be finite");
                release_radial_inputs(out);
                return false;
            }
        }
    }
    return true;
}

inline void displacement(
    const double* xyz, npy_intp left, npy_intp right,
    double& dx, double& dy, double& dz, double& r2) {
    dx = xyz[3 * left] - xyz[3 * right];
    dy = xyz[3 * left + 1] - xyz[3 * right + 1];
    dz = xyz[3 * left + 2] - xyz[3 * right + 2];
    r2 = dx * dx + dy * dy + dz * dz;
}

inline void gaussian_correction(
    double distance, double product, double width_left, double width_right,
    double& energy, double& first, double& second) {
    const double beta = 1.0 / std::sqrt(
        2.0 * (width_left * width_left + width_right * width_right));
    const double argument = beta * distance;
    const double gaussian = std::exp(-(argument * argument));
    const double error_function = std::erf(argument);
    const double root_pi = std::sqrt(std::acos(-1.0));
    const double inverse_r = 1.0 / distance;
    const double inverse_r2 = inverse_r * inverse_r;
    const double inverse_r3 = inverse_r2 * inverse_r;
    const double gaussian_energy = product * error_function * inverse_r;
    const double gaussian_first = product * (
        2.0 * beta * gaussian * inverse_r / root_pi -
        error_function * inverse_r2);
    const double gaussian_second = product * (
        -4.0 * beta * beta * beta * gaussian / root_pi -
        4.0 * beta * gaussian * inverse_r2 / root_pi +
        2.0 * error_function * inverse_r3);
    energy = gaussian_energy - product * inverse_r;
    first = gaussian_first + product * inverse_r2;
    second = gaussian_second - 2.0 * product * inverse_r3;
}

bool accumulate_point_energy(
    const Inputs& input, double& energy, double* gradient,
    const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            if (distance <= kCoincidentTolerance) {
                return false;
            }
            const double product = q[left] * q[right];
            const double inverse_r = 1.0 / distance;
            const double inverse_r3 = inverse_r / r2;
            energy += product * inverse_r;
            if (gradient != nullptr) {
                const double factor = -product * inverse_r3;
                const double values[3] = {factor * dx, factor * dy, factor * dz};
                for (int axis = 0; axis < 3; ++axis) {
                    gradient[3 * left + axis] += values[axis];
                    gradient[3 * right + axis] -= values[axis];
                }
            }
            if (hessian_product != nullptr) {
                const double delta[3] = {dx, dy, dz};
                const double inverse_r5 = inverse_r3 / r2;
                double dv[3];
                for (int axis = 0; axis < 3; ++axis) {
                    dv[axis] = direction[3 * left + axis] - direction[3 * right + axis];
                }
                for (int row = 0; row < 3; ++row) {
                    double contribution = 0.0;
                    for (int column = 0; column < 3; ++column) {
                        const double block = product * (
                            3.0 * delta[row] * delta[column] * inverse_r5 -
                            (row == column ? inverse_r3 : 0.0));
                        contribution += block * dv[column];
                    }
                    hessian_product[3 * left + row] += contribution;
                    hessian_product[3 * right + row] -= contribution;
                }
            }
        }
    }
    return true;
}

npy_intp accumulate_penetration(
    const Inputs& input, double& energy, double* gradient,
    const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    npy_intp count = 0;
    for (npy_intp left = 0; left < input.atoms; ++left) {
        for (npy_intp right = left + 1; right < input.atoms; ++right) {
            double dx, dy, dz, r2;
            displacement(xyz, left, right, dx, dy, dz, r2);
            const double distance = std::sqrt(r2);
            const double beta = 1.0 / std::sqrt(
                2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
            if (beta * distance >= kPenetrationArgument) {
                continue;
            }
            double correction_energy, first, second;
            gaussian_correction(
                distance, q[left] * q[right], widths[left], widths[right],
                correction_energy, first, second);
            energy += correction_energy;
            ++count;
            const double inverse_r = 1.0 / distance;
            const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
            if (gradient != nullptr) {
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = first * unit[axis];
                    gradient[3 * left + axis] += value;
                    gradient[3 * right + axis] -= value;
                }
            }
            if (hessian_product != nullptr) {
                double dv[3];
                for (int axis = 0; axis < 3; ++axis) {
                    dv[axis] = direction[3 * left + axis] - direction[3 * right + axis];
                }
                const double transverse = first * inverse_r;
                const double longitudinal = second - transverse;
                for (int row = 0; row < 3; ++row) {
                    double contribution = 0.0;
                    for (int column = 0; column < 3; ++column) {
                        const double block =
                            longitudinal * unit[row] * unit[column] +
                            (row == column ? transverse : 0.0);
                        contribution += block * dv[column];
                    }
                    hessian_product[3 * left + row] += contribution;
                    hessian_product[3 * right + row] -= contribution;
                }
            }
        }
    }
    return count;
}

npy_intp accumulate_penetration_pairs(
    const Inputs& input, const PairInputs& pair_input, double& energy,
    double* gradient, const double* direction, double* hessian_product) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double beta = 1.0 / std::sqrt(
            2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
        if (beta * distance >= kPenetrationArgument) {
            continue;
        }
        double correction_energy, first, second;
        gaussian_correction(
            distance, q[left] * q[right], widths[left], widths[right],
            correction_energy, first, second);
        energy += correction_energy;
        ++count;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value = first * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (hessian_product != nullptr) {
            double dv[3];
            for (int axis = 0; axis < 3; ++axis) {
                dv[axis] =
                    direction[3 * left + axis] - direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int row = 0; row < 3; ++row) {
                double contribution = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    contribution += block * dv[column];
                }
                hessian_product[3 * left + row] += contribution;
                hessian_product[3 * right + row] -= contribution;
            }
        }
    }
    return count;
}

npy_intp accumulate_penetration_potential_pairs(
    const Inputs& input, const PairInputs& pair_input, double* potential) {
    const double* xyz = static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(input.charges));
    const double* widths = static_cast<const double*>(PyArray_DATA(input.widths));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double beta = 1.0 / std::sqrt(
            2.0 * (widths[left] * widths[left] + widths[right] * widths[right]));
        if (beta * distance >= kPenetrationArgument) {
            continue;
        }
        double correction, unused_first, unused_second;
        gaussian_correction(
            distance, 1.0, widths[left], widths[right],
            correction, unused_first, unused_second);
        potential[left] += correction * q[right];
        potential[right] += correction * q[left];
        ++count;
    }
    return count;
}

inline void damped_exppe(
    double distance, double epsilon, double rmin,
    double& energy, double& first, double& second) {
    constexpr double alpha = 12.649110640673517;  // sqrt(160), UFF 12-6 curvature
    const double x = distance / rmin;
    const double exp_full = std::exp(alpha * (1.0 - x));
    const double exp_half = std::exp(0.5 * alpha * (1.0 - x));
    const double x2 = x * x;
    const double polynomial = x2 * x2 - 2.0 * x2 + 3.0;
    const double polynomial_1 = 4.0 * x * x2 - 4.0 * x;
    const double polynomial_2 = 12.0 * x2 - 4.0;
    const double dimensionless_0 = exp_full - polynomial * exp_half;
    const double dimensionless_1 =
        -alpha * exp_full -
        exp_half * (polynomial_1 - 0.5 * alpha * polynomial);
    const double dimensionless_2 =
        alpha * alpha * exp_full +
        exp_half * (
            -polynomial_2 + alpha * polynomial_1 -
            0.25 * alpha * alpha * polynomial);
    const double radial_energy = epsilon * dimensionless_0;
    const double radial_first = epsilon * dimensionless_1 / rmin;
    const double radial_second = epsilon * dimensionless_2 / (rmin * rmin);
    const double ratio_base = 0.72 * rmin / distance;
    const double ratio2 = ratio_base * ratio_base;
    const double ratio4 = ratio2 * ratio2;
    const double ratio8 = ratio4 * ratio4;
    const double sw = 1.0 / (1.0 + ratio8);
    const double sw_first = 8.0 * sw * (1.0 - sw) / distance;
    const double sw_second =
        8.0 * sw * (1.0 - sw) * (8.0 * (1.0 - 2.0 * sw) - 1.0) /
        (distance * distance);
    energy = sw * radial_energy;
    first = sw * radial_first + sw_first * radial_energy;
    second =
        sw * radial_second + 2.0 * sw_first * radial_first +
        sw_second * radial_energy;
}

npy_intp accumulate_damped_exppe_pairs(
    const RadialInputs& input, const PairInputs& pair_input, double cutoff,
    double& energy, double* gradient, const double* direction,
    double* hessian_product) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const double* epsilon =
        static_cast<const double*>(PyArray_DATA(input.epsilon));
    const double* rmin_half =
        static_cast<const double*>(PyArray_DATA(input.rmin_half));
    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    const double cutoff2 = cutoff * cutoff;
    npy_intp count = 0;
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        if (r2 > cutoff2) {
            continue;
        }
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double pair_epsilon =
            std::sqrt(epsilon[left] * epsilon[right]);
        const double pair_rmin = rmin_half[left] + rmin_half[right];
        double pair_energy, first, second;
        damped_exppe(
            distance, pair_epsilon, pair_rmin, pair_energy, first, second);
        energy += pair_energy;
        ++count;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value = first * bohr_to_angstrom * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (hessian_product != nullptr) {
            double dv[3];
            for (int axis = 0; axis < 3; ++axis) {
                dv[axis] =
                    direction[3 * left + axis] - direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            const double scale = bohr_to_angstrom * bohr_to_angstrom;
            for (int row = 0; row < 3; ++row) {
                double contribution = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    contribution += block * dv[column];
                }
                contribution *= scale;
                hessian_product[3 * left + row] += contribution;
                hessian_product[3 * right + row] -= contribution;
            }
        }
    }
    return count;
}

PyObject* direct_gaussian_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates, &charges, &widths)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    npy_intp count = 0;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(input, energy, nullptr, nullptr, nullptr);
    if (valid) {
        count = accumulate_penetration(input, energy, nullptr, nullptr, nullptr);
    }
    Py_END_ALLOW_THREADS
    release_inputs(input);
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* direct_gaussian_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths;
    if (!PyArg_ParseTuple(args, "OOO", &coordinates, &charges, &widths)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    npy_intp count = 0;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(input, energy, gradient_data, nullptr, nullptr);
    if (valid) {
        count = accumulate_penetration(input, energy, gradient_data, nullptr, nullptr);
    }
    Py_END_ALLOW_THREADS
    release_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* direct_gaussian_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &direction_object)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms || PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_inputs(input);
        PyErr_SetString(PyExc_ValueError, "native ZAFF direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    bool valid;
    const double* direction_data = static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_point_energy(
        input, unused_energy, nullptr, direction_data, product_data);
    if (valid) {
        accumulate_penetration(
            input, unused_energy, nullptr, direction_data, product_data);
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(product);
}

PyObject* gaussian_correction_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* gaussian_correction_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, energy, gradient_data, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* gaussian_correction_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs, *direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &charges, &widths, &pairs,
            &direction_object)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native ZAFF direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    const double* direction_data =
        static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_pairs(
        input, pair_input, unused_energy, nullptr, direction_data, product_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* gaussian_correction_potential(PyObject*, PyObject* args) {
    PyObject *coordinates, *charges, *widths, *pairs;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates, &charges, &widths, &pairs)) {
        return nullptr;
    }
    Inputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_inputs(coordinates, charges, widths, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[1] = {input.atoms};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_inputs(input);
        return nullptr;
    }
    npy_intp count;
    double* potential_data = static_cast<double*>(PyArray_DATA(potential));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_penetration_potential_pairs(
        input, pair_input, potential_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_inputs(input);
    if (count < 0) {
        Py_DECREF(potential);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in electrostatic pair");
        return nullptr;
    }
    return Py_BuildValue("NL", potential, static_cast<long long>(count));
}

PyObject* damped_exppe_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOd", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* damped_exppe_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOd", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, energy, gradient_data, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* damped_exppe_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *epsilon, *rmin_half, *pairs, *direction_object;
    double cutoff;
    if (!PyArg_ParseTuple(
            args, "OOOOdO", &coordinates, &epsilon, &rmin_half, &pairs,
            &cutoff, &direction_object)) {
        return nullptr;
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "native Exp-PE cutoff must be positive");
        return nullptr;
    }
    RadialInputs input{nullptr, nullptr, nullptr, 0};
    if (!parse_radial_inputs(coordinates, epsilon, rmin_half, input)) {
        return nullptr;
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs, input.atoms, pair_input)) {
        release_radial_inputs(input);
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native Exp-PE direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        Py_DECREF(pair_input.pairs);
        release_radial_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    const double* direction_data =
        static_cast<const double*>(PyArray_DATA(direction));
    double* product_data = static_cast<double*>(PyArray_DATA(product));
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_damped_exppe_pairs(
        input, pair_input, cutoff, unused_energy, nullptr, direction_data,
        product_data);
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    Py_DECREF(pair_input.pairs);
    release_radial_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Exp-PE pair");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* switched_lj_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *pairs_object;
    double sigma, epsilon, switch_distance, cutoff;
    if (!PyArg_ParseTuple(
            args, "OOdddd", &coordinates_object, &pairs_object, &sigma,
            &epsilon, &switch_distance, &cutoff)) {
        return nullptr;
    }
    if (!std::isfinite(sigma) || sigma <= 0.0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0 ||
        !std::isfinite(switch_distance) || switch_distance <= 0.0 ||
        !std::isfinite(cutoff) || cutoff <= switch_distance) {
        PyErr_SetString(
            PyExc_ValueError,
            "native switched LJ parameters require positive sigma, epsilon "
            "and 0 < switch < cutoff");
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(
            coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || PyArray_NDIM(coordinates) != 2 ||
        PyArray_DIM(coordinates, 1) != 3) {
        Py_XDECREF(coordinates);
        PyErr_SetString(
            PyExc_ValueError,
            "native switched LJ coordinates must have shape (N,3)");
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates, 0);
    const double* xyz =
        static_cast<const double*>(PyArray_DATA(coordinates));
    for (npy_intp atom = 0; atom < atoms; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(xyz[3 * atom + axis])) {
                Py_DECREF(coordinates);
                PyErr_SetString(
                    PyExc_ValueError,
                    "native switched LJ coordinates must be finite");
                return nullptr;
            }
        }
    }
    PairInputs pair_input{nullptr, 0};
    if (!parse_pairs(pairs_object, atoms, pair_input)) {
        Py_DECREF(coordinates);
        return nullptr;
    }
    npy_intp dimensions[2] = {atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        Py_DECREF(pair_input.pairs);
        Py_DECREF(coordinates);
        return nullptr;
    }

    const npy_intp* pairs =
        static_cast<const npy_intp*>(PyArray_DATA(pair_input.pairs));
    double* gradient_data = static_cast<double*>(PyArray_DATA(gradient));
    const double cutoff2 = cutoff * cutoff;
    const double interval = cutoff - switch_distance;
    double energy = 0.0;
    npy_intp count = 0;
    bool coincident = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp pair = 0; pair < pair_input.count; ++pair) {
        const npy_intp left = pairs[2 * pair];
        const npy_intp right = pairs[2 * pair + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        if (r2 > cutoff2) {
            continue;
        }
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            coincident = true;
            break;
        }
        const double ratio = sigma / distance;
        const double ratio2 = ratio * ratio;
        const double ratio6 = ratio2 * ratio2 * ratio2;
        const double ratio12 = ratio6 * ratio6;
        const double bare = 4.0 * epsilon * (ratio12 - ratio6);
        double switching = 1.0;
        double switching_first = 0.0;
        if (distance > switch_distance) {
            const double reduced =
                (distance - switch_distance) / interval;
            const double reduced2 = reduced * reduced;
            const double reduced3 = reduced2 * reduced;
            const double reduced4 = reduced3 * reduced;
            const double reduced5 = reduced4 * reduced;
            switching =
                1.0 - 10.0 * reduced3 + 15.0 * reduced4 -
                6.0 * reduced5;
            switching_first =
                (-30.0 * reduced2 + 60.0 * reduced3 -
                 30.0 * reduced4) /
                interval;
        }
        const double radial_first =
            24.0 * epsilon * (ratio6 - 2.0 * ratio12) /
                distance * switching +
            bare * switching_first;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        energy += bare * switching;
        ++count;
        for (int axis = 0; axis < 3; ++axis) {
            const double value = radial_first * unit[axis];
            gradient_data[3 * left + axis] += value;
            gradient_data[3 * right + axis] -= value;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(pair_input.pairs);
    Py_DECREF(coordinates);
    if (coincident) {
        Py_DECREF(gradient);
        PyErr_SetString(
            PyExc_ValueError, "coincident atoms in switched LJ pair");
        return nullptr;
    }
    return Py_BuildValue(
        "dNL", energy, gradient, static_cast<long long>(count));
}

struct MorseBondInputs {
    PyArrayObject* coordinates;
    PyArrayObject* bonds;
    PyArrayObject* depths;
    PyArrayObject* alphas;
    PyArrayObject* references;
    npy_intp atoms;
    npy_intp count;
};

void release_morse_bond_inputs(MorseBondInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.bonds);
    Py_XDECREF(input.depths);
    Py_XDECREF(input.alphas);
    Py_XDECREF(input.references);
}

bool parse_morse_bond_inputs(
    PyObject* coordinates, PyObject* bonds, PyObject* depths,
    PyObject* alphas, PyObject* references, MorseBondInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.bonds = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(bonds, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.depths = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(depths, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.alphas = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(alphas, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.references = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(references, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (out.coordinates == nullptr || out.bonds == nullptr ||
        out.depths == nullptr || out.alphas == nullptr ||
        out.references == nullptr) {
        release_morse_bond_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.bonds) != 2 || PyArray_DIM(out.bonds, 1) != 2 ||
        PyArray_NDIM(out.depths) != 1 || PyArray_NDIM(out.alphas) != 1 ||
        PyArray_NDIM(out.references) != 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "native Morse bond arrays require (N,3), (B,2), and three (B,) arrays");
        release_morse_bond_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.count = PyArray_DIM(out.bonds, 0);
    if (PyArray_DIM(out.depths, 0) != out.count ||
        PyArray_DIM(out.alphas, 0) != out.count ||
        PyArray_DIM(out.references, 0) != out.count) {
        PyErr_SetString(
            PyExc_ValueError, "native Morse bond parameter dimensions differ");
        release_morse_bond_inputs(out);
        return false;
    }
    const auto* indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.bonds));
    const auto* depth = static_cast<const double*>(PyArray_DATA(out.depths));
    const auto* alpha = static_cast<const double*>(PyArray_DATA(out.alphas));
    const auto* reference =
        static_cast<const double*>(PyArray_DATA(out.references));
    for (npy_intp bond = 0; bond < out.count; ++bond) {
        const npy_intp left = indices[2 * bond];
        const npy_intp right = indices[2 * bond + 1];
        if (left < 0 || right < 0 || left >= out.atoms ||
            right >= out.atoms || left == right ||
            !std::isfinite(depth[bond]) || depth[bond] < 0.0 ||
            !std::isfinite(alpha[bond]) || alpha[bond] <= 0.0 ||
            !std::isfinite(reference[bond]) || reference[bond] <= 0.0) {
            PyErr_SetString(
                PyExc_ValueError, "native Morse bond record is invalid");
            release_morse_bond_inputs(out);
            return false;
        }
    }
    return true;
}

npy_intp accumulate_morse_bonds(
    const MorseBondInputs& input, double& energy, double* gradient,
    const double* direction, double* product) {
    constexpr double bohr_to_angstrom = 0.52917721092;
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* bonds =
        static_cast<const npy_intp*>(PyArray_DATA(input.bonds));
    const auto* depths =
        static_cast<const double*>(PyArray_DATA(input.depths));
    const auto* alphas =
        static_cast<const double*>(PyArray_DATA(input.alphas));
    const auto* references =
        static_cast<const double*>(PyArray_DATA(input.references));
    for (npy_intp bond = 0; bond < input.count; ++bond) {
        const npy_intp left = bonds[2 * bond];
        const npy_intp right = bonds[2 * bond + 1];
        double dx, dy, dz, r2;
        displacement(xyz, left, right, dx, dy, dz, r2);
        const double distance = std::sqrt(r2);
        if (distance <= kCoincidentTolerance) {
            return -1;
        }
        const double exponential = std::exp(
            -alphas[bond] * (distance - references[bond]));
        const double difference = 1.0 - exponential;
        const double first =
            2.0 * depths[bond] * alphas[bond] * exponential * difference;
        const double second =
            2.0 * depths[bond] * alphas[bond] * alphas[bond] *
            (2.0 * exponential * exponential - exponential);
        energy += depths[bond] * difference * difference;
        const double inverse_r = 1.0 / distance;
        const double unit[3] = {
            dx * inverse_r, dy * inverse_r, dz * inverse_r};
        if (gradient != nullptr) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value =
                    first * bohr_to_angstrom * unit[axis];
                gradient[3 * left + axis] += value;
                gradient[3 * right + axis] -= value;
            }
        }
        if (product != nullptr) {
            double delta_direction[3];
            for (int axis = 0; axis < 3; ++axis) {
                delta_direction[axis] =
                    direction[3 * left + axis] -
                    direction[3 * right + axis];
            }
            const double transverse = first * inverse_r;
            const double longitudinal = second - transverse;
            for (int row = 0; row < 3; ++row) {
                double value = 0.0;
                for (int column = 0; column < 3; ++column) {
                    const double block =
                        longitudinal * unit[row] * unit[column] +
                        (row == column ? transverse : 0.0);
                    value += block * delta_direction[column];
                }
                value *= bohr_to_angstrom * bohr_to_angstrom;
                product[3 * left + row] += value;
                product[3 * right + row] -= value;
            }
        }
    }
    return input.count;
}

PyObject* morse_bond_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &depths, &alphas,
            &references)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, energy, nullptr, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_morse_bond_inputs(input);
    if (count < 0) {
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue("dL", energy, static_cast<long long>(count));
}

PyObject* morse_bond_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    if (!PyArg_ParseTuple(
            args, "OOOOO", &coordinates, &bonds, &depths, &alphas,
            &references)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_morse_bond_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)),
        nullptr, nullptr);
    Py_END_ALLOW_THREADS
    release_morse_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue(
        "dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* morse_bond_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *bonds, *depths, *alphas, *references;
    PyObject* direction_object;
    if (!PyArg_ParseTuple(
            args, "OOOOOO", &coordinates, &bonds, &depths, &alphas,
            &references, &direction_object)) {
        return nullptr;
    }
    MorseBondInputs input{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
    if (!parse_morse_bond_inputs(
            coordinates, bonds, depths, alphas, references, input)) {
        return nullptr;
    }
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction == nullptr || PyArray_NDIM(direction) != 2 ||
        PyArray_DIM(direction, 0) != input.atoms ||
        PyArray_DIM(direction, 1) != 3) {
        Py_XDECREF(direction);
        release_morse_bond_inputs(input);
        PyErr_SetString(
            PyExc_ValueError, "native Morse direction must have shape (N,3)");
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        Py_DECREF(direction);
        release_morse_bond_inputs(input);
        return nullptr;
    }
    double unused_energy = 0.0;
    npy_intp count;
    Py_BEGIN_ALLOW_THREADS
    count = accumulate_morse_bonds(
        input, unused_energy, nullptr,
        static_cast<const double*>(PyArray_DATA(direction)),
        static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    Py_DECREF(direction);
    release_morse_bond_inputs(input);
    if (count < 0) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_ValueError, "coincident atoms in Morse bond");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

constexpr int kLocalDimension = 12;
constexpr double kBohrToAngstrom = 0.52917721092;

struct DirectionalJet {
    double value = 0.0;
    std::array<double, kLocalDimension> gradient{};
    double dot = 0.0;
    std::array<double, kLocalDimension> gradient_dot{};
};

DirectionalJet jet_constant(double value) {
    DirectionalJet result;
    result.value = value;
    return result;
}

DirectionalJet jet_variable(double value, int index, double dot) {
    DirectionalJet result;
    result.value = value;
    result.gradient[index] = 1.0;
    result.dot = dot;
    return result;
}

DirectionalJet operator+(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value + right.value;
    result.dot = left.dot + right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = left.gradient[index] + right.gradient[index];
        result.gradient_dot[index] =
            left.gradient_dot[index] + right.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator-(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value - right.value;
    result.dot = left.dot - right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = left.gradient[index] - right.gradient[index];
        result.gradient_dot[index] =
            left.gradient_dot[index] - right.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator*(const DirectionalJet& left, const DirectionalJet& right) {
    DirectionalJet result;
    result.value = left.value * right.value;
    result.dot = left.dot * right.value + left.value * right.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] =
            left.value * right.gradient[index] +
            right.value * left.gradient[index];
        result.gradient_dot[index] =
            left.dot * right.gradient[index] +
            left.value * right.gradient_dot[index] +
            right.dot * left.gradient[index] +
            right.value * left.gradient_dot[index];
    }
    return result;
}

DirectionalJet operator*(double left, const DirectionalJet& right) {
    return jet_constant(left) * right;
}

DirectionalJet operator*(const DirectionalJet& left, double right) {
    return left * jet_constant(right);
}

DirectionalJet operator+(double left, const DirectionalJet& right) {
    return jet_constant(left) + right;
}

DirectionalJet operator-(const DirectionalJet& left, double right) {
    return left - jet_constant(right);
}

DirectionalJet operator-(double left, const DirectionalJet& right) {
    return jet_constant(left) - right;
}

DirectionalJet jet_unary(
    const DirectionalJet& value, double result_value, double first,
    double second) {
    DirectionalJet result;
    result.value = result_value;
    result.dot = first * value.dot;
    for (int index = 0; index < kLocalDimension; ++index) {
        result.gradient[index] = first * value.gradient[index];
        result.gradient_dot[index] =
            second * value.dot * value.gradient[index] +
            first * value.gradient_dot[index];
    }
    return result;
}

DirectionalJet jet_inverse(const DirectionalJet& value) {
    const double inverse = 1.0 / value.value;
    return jet_unary(
        value, inverse, -inverse * inverse,
        2.0 * inverse * inverse * inverse);
}

DirectionalJet operator/(const DirectionalJet& left, const DirectionalJet& right) {
    return left * jet_inverse(right);
}

DirectionalJet operator/(const DirectionalJet& left, double right) {
    return left * (1.0 / right);
}

DirectionalJet jet_sqrt(const DirectionalJet& value) {
    const double root = std::sqrt(value.value);
    return jet_unary(
        value, root, 0.5 / root, -0.25 / (value.value * root));
}

DirectionalJet jet_exp(const DirectionalJet& value) {
    const double exponential = std::exp(value.value);
    return jet_unary(value, exponential, exponential, exponential);
}

DirectionalJet jet_tanh(const DirectionalJet& value) {
    const double hyperbolic = std::tanh(value.value);
    const double first = 1.0 - hyperbolic * hyperbolic;
    return jet_unary(value, hyperbolic, first, -2.0 * hyperbolic * first);
}

DirectionalJet jet_sin(const DirectionalJet& value) {
    return jet_unary(
        value, std::sin(value.value), std::cos(value.value),
        -std::sin(value.value));
}

DirectionalJet jet_cos(const DirectionalJet& value) {
    return jet_unary(
        value, std::cos(value.value), -std::sin(value.value),
        -std::cos(value.value));
}

DirectionalJet jet_acos(const DirectionalJet& value) {
    const double bounded = std::max(-1.0, std::min(1.0, value.value));
    const double denominator = std::sqrt(1.0 - bounded * bounded);
    return jet_unary(
        value, std::acos(bounded), -1.0 / denominator,
        -bounded / (denominator * denominator * denominator));
}

DirectionalJet jet_atan2(
    const DirectionalJet& y, const DirectionalJet& x) {
    DirectionalJet result;
    const double denominator = x.value * x.value + y.value * y.value;
    result.value = std::atan2(y.value, x.value);
    result.dot =
        (x.value * y.dot - y.value * x.dot) / denominator;
    const double denominator_dot =
        2.0 * (x.value * x.dot + y.value * y.dot);
    for (int index = 0; index < kLocalDimension; ++index) {
        const double numerator =
            x.value * y.gradient[index] - y.value * x.gradient[index];
        const double numerator_dot =
            x.dot * y.gradient[index] +
            x.value * y.gradient_dot[index] -
            y.dot * x.gradient[index] -
            y.value * x.gradient_dot[index];
        result.gradient[index] = numerator / denominator;
        result.gradient_dot[index] =
            numerator_dot / denominator -
            numerator * denominator_dot / (denominator * denominator);
    }
    return result;
}

using JetVector = std::array<DirectionalJet, 3>;

JetVector jet_subtract(const JetVector& left, const JetVector& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

DirectionalJet jet_dot(const JetVector& left, const JetVector& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

JetVector jet_cross(const JetVector& left, const JetVector& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

DirectionalJet jet_norm(const JetVector& value) {
    return jet_sqrt(jet_dot(value, value));
}

JetVector jet_divide(const JetVector& value, const DirectionalJet& scale) {
    return {value[0] / scale, value[1] / scale, value[2] / scale};
}

DirectionalJet bond_order_factor(
    const DirectionalJet& distance, double reference, double radius_sum) {
    constexpr double strong_decay = 0.20;
    constexpr double weak_decay = 0.40;
    constexpr double sharpness = 20.0;
    const auto raw = [&](const DirectionalJet& r) {
        const DirectionalJet x = (r - radius_sum) / radius_sum;
        const DirectionalJet weight =
            0.5 * (1.0 - jet_tanh(sharpness * x));
        const DirectionalJet strong =
            jet_exp((radius_sum - r) / strong_decay);
        const DirectionalJet weak =
            jet_exp((radius_sum - r) / weak_decay);
        return weak + weight * (strong - weak);
    };
    const double reference_x = (reference - radius_sum) / radius_sum;
    const double reference_weight =
        0.5 * (1.0 - std::tanh(sharpness * reference_x));
    const double reference_strong =
        std::exp((radius_sum - reference) / strong_decay);
    const double reference_weak =
        std::exp((radius_sum - reference) / weak_decay);
    const double normalization =
        reference_weak +
        reference_weight * (reference_strong - reference_weak);
    return raw(distance) / normalization;
}

struct LocalValenceInputs {
    PyArrayObject* coordinates = nullptr;
    PyArrayObject* angle_atoms = nullptr;
    PyArrayObject* angle_parameters = nullptr;
    PyArrayObject* torsion_atoms = nullptr;
    PyArrayObject* torsion_parameters = nullptr;
    PyArrayObject* term_offsets = nullptr;
    PyArrayObject* terms = nullptr;
    PyArrayObject* direction = nullptr;
    npy_intp atoms = 0;
    npy_intp angle_count = 0;
    npy_intp torsion_count = 0;
    npy_intp term_count = 0;
};

void release_local_valence_inputs(LocalValenceInputs& input) {
    Py_XDECREF(input.coordinates);
    Py_XDECREF(input.angle_atoms);
    Py_XDECREF(input.angle_parameters);
    Py_XDECREF(input.torsion_atoms);
    Py_XDECREF(input.torsion_parameters);
    Py_XDECREF(input.term_offsets);
    Py_XDECREF(input.terms);
    Py_XDECREF(input.direction);
}

bool parse_local_valence_inputs(
    PyObject* coordinates, PyObject* angle_atoms,
    PyObject* angle_parameters, PyObject* torsion_atoms,
    PyObject* torsion_parameters, PyObject* term_offsets, PyObject* terms,
    PyObject* direction, LocalValenceInputs& out) {
    out.coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.angle_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.angle_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(angle_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.torsion_atoms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_atoms, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.torsion_parameters = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(torsion_parameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    out.term_offsets = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(term_offsets, NPY_INTP, NPY_ARRAY_IN_ARRAY));
    out.terms = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(terms, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (direction != Py_None) {
        out.direction = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(direction, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    }
    if (out.coordinates == nullptr || out.angle_atoms == nullptr ||
        out.angle_parameters == nullptr || out.torsion_atoms == nullptr ||
        out.torsion_parameters == nullptr || out.term_offsets == nullptr ||
        out.terms == nullptr || (direction != Py_None && out.direction == nullptr)) {
        release_local_valence_inputs(out);
        return false;
    }
    if (PyArray_NDIM(out.coordinates) != 2 ||
        PyArray_DIM(out.coordinates, 1) != 3 ||
        PyArray_NDIM(out.angle_atoms) != 2 ||
        PyArray_DIM(out.angle_atoms, 1) != 3 ||
        PyArray_NDIM(out.angle_parameters) != 2 ||
        PyArray_DIM(out.angle_parameters, 1) != 6 ||
        PyArray_NDIM(out.torsion_atoms) != 2 ||
        PyArray_DIM(out.torsion_atoms, 1) != 4 ||
        PyArray_NDIM(out.torsion_parameters) != 2 ||
        PyArray_DIM(out.torsion_parameters, 1) != 6 ||
        PyArray_NDIM(out.term_offsets) != 1 ||
        PyArray_NDIM(out.terms) != 2 || PyArray_DIM(out.terms, 1) != 3) {
        PyErr_SetString(
            PyExc_ValueError, "native local-valence array shapes are invalid");
        release_local_valence_inputs(out);
        return false;
    }
    out.atoms = PyArray_DIM(out.coordinates, 0);
    out.angle_count = PyArray_DIM(out.angle_atoms, 0);
    out.torsion_count = PyArray_DIM(out.torsion_atoms, 0);
    out.term_count = PyArray_DIM(out.terms, 0);
    if (PyArray_DIM(out.angle_parameters, 0) != out.angle_count ||
        PyArray_DIM(out.torsion_parameters, 0) != out.torsion_count ||
        PyArray_DIM(out.term_offsets, 0) != out.torsion_count + 1 ||
        (out.direction != nullptr &&
         (PyArray_NDIM(out.direction) != 2 ||
          PyArray_DIM(out.direction, 0) != out.atoms ||
          PyArray_DIM(out.direction, 1) != 3))) {
        PyErr_SetString(
            PyExc_ValueError, "native local-valence dimensions differ");
        release_local_valence_inputs(out);
        return false;
    }
    const auto* angle_indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.angle_atoms));
    const auto* torsion_indices =
        static_cast<const npy_intp*>(PyArray_DATA(out.torsion_atoms));
    const auto* angle_values =
        static_cast<const double*>(PyArray_DATA(out.angle_parameters));
    const auto* torsion_values =
        static_cast<const double*>(PyArray_DATA(out.torsion_parameters));
    const auto* offsets =
        static_cast<const npy_intp*>(PyArray_DATA(out.term_offsets));
    const auto* term_values =
        static_cast<const double*>(PyArray_DATA(out.terms));
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(out.coordinates));
    const auto* direction_values =
        out.direction == nullptr
            ? nullptr
            : static_cast<const double*>(PyArray_DATA(out.direction));
    if (offsets[0] != 0 || offsets[out.torsion_count] != out.term_count) {
        PyErr_SetString(PyExc_ValueError, "native torsion term offsets are invalid");
        release_local_valence_inputs(out);
        return false;
    }
    for (npy_intp item = 0; item < out.atoms * 3; ++item) {
        if (!std::isfinite(xyz[item]) ||
            (direction_values != nullptr &&
             !std::isfinite(direction_values[item]))) {
            PyErr_SetString(
                PyExc_ValueError,
                "native local-valence coordinates and direction must be finite");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp item = 0; item < out.angle_count; ++item) {
        const npy_intp* atoms = angle_indices + 3 * item;
        const double* parameters = angle_values + 6 * item;
        if (atoms[0] < 0 || atoms[1] < 0 || atoms[2] < 0 ||
            atoms[0] >= out.atoms || atoms[1] >= out.atoms ||
            atoms[2] >= out.atoms || atoms[0] == atoms[1] ||
            atoms[1] == atoms[2] || atoms[0] == atoms[2] ||
            !std::isfinite(parameters[0]) || parameters[0] < 0.0 ||
            !std::isfinite(parameters[1])) {
            PyErr_SetString(PyExc_ValueError, "native angle atom is out of range");
            release_local_valence_inputs(out);
            return false;
        }
        for (int parameter = 2; parameter < 6; ++parameter) {
            if (!std::isfinite(parameters[parameter]) ||
                parameters[parameter] <= 0.0) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native angle radial parameters must be finite and positive");
                release_local_valence_inputs(out);
                return false;
            }
        }
    }
    for (npy_intp item = 0; item < out.torsion_count; ++item) {
        const npy_intp* atoms = torsion_indices + 4 * item;
        const double* parameters = torsion_values + 6 * item;
        bool invalid = false;
        for (int atom = 0; atom < 4; ++atom) {
            invalid = invalid || atoms[atom] < 0 || atoms[atom] >= out.atoms;
            for (int other = 0; other < atom; ++other) {
                invalid = invalid || atoms[atom] == atoms[other];
            }
        }
        for (int parameter = 0; parameter < 6; ++parameter) {
            invalid = invalid || !std::isfinite(parameters[parameter]) ||
                parameters[parameter] <= 0.0;
        }
        if (invalid) {
            PyErr_SetString(PyExc_ValueError, "native torsion atom is out of range");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp item = 0; item < out.torsion_count; ++item) {
        if (offsets[item] < 0 || offsets[item] >= offsets[item + 1] ||
            offsets[item + 1] > out.term_count) {
            PyErr_SetString(PyExc_ValueError, "native torsion term span is invalid");
            release_local_valence_inputs(out);
            return false;
        }
    }
    for (npy_intp term = 0; term < out.term_count; ++term) {
        const double amplitude = term_values[3 * term];
        const double periodicity = term_values[3 * term + 1];
        const double phase = term_values[3 * term + 2];
        if (!std::isfinite(amplitude) || !std::isfinite(periodicity) ||
            periodicity <= 0.0 ||
            std::abs(periodicity - std::round(periodicity)) > 1.0e-12 ||
            !std::isfinite(phase)) {
            PyErr_SetString(
                PyExc_ValueError, "native torsion Fourier term is invalid");
            release_local_valence_inputs(out);
            return false;
        }
    }
    return true;
}

bool accumulate_local_valence(
    const LocalValenceInputs& input, double& energy, double* gradient,
    double* product) {
    const auto* xyz =
        static_cast<const double*>(PyArray_DATA(input.coordinates));
    const auto* direction =
        input.direction == nullptr
            ? nullptr
            : static_cast<const double*>(PyArray_DATA(input.direction));
    const auto* angle_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.angle_atoms));
    const auto* angle_parameters =
        static_cast<const double*>(PyArray_DATA(input.angle_parameters));
    const auto* torsion_atoms =
        static_cast<const npy_intp*>(PyArray_DATA(input.torsion_atoms));
    const auto* torsion_parameters =
        static_cast<const double*>(PyArray_DATA(input.torsion_parameters));
    const auto* offsets =
        static_cast<const npy_intp*>(PyArray_DATA(input.term_offsets));
    const auto* terms =
        static_cast<const double*>(PyArray_DATA(input.terms));
    const auto make_positions = [&](const npy_intp* atoms, int count) {
        std::array<JetVector, 4> positions{};
        for (int atom = 0; atom < count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                const npy_intp global = atoms[atom];
                positions[atom][axis] = jet_variable(
                    xyz[3 * global + axis], local,
                    direction == nullptr
                        ? 0.0
                        : kBohrToAngstrom * direction[3 * global + axis]);
            }
        }
        return positions;
    };
    const auto scatter = [&](const DirectionalJet& value, const npy_intp* atoms,
                             int count) {
        energy += value.value;
        for (int atom = 0; atom < count; ++atom) {
            const npy_intp global = atoms[atom];
            for (int axis = 0; axis < 3; ++axis) {
                const int local = 3 * atom + axis;
                if (gradient != nullptr) {
                    gradient[3 * global + axis] +=
                        kBohrToAngstrom * value.gradient[local];
                }
                if (product != nullptr) {
                    product[3 * global + axis] +=
                        kBohrToAngstrom * value.gradient_dot[local];
                }
            }
        }
    };
    for (npy_intp item = 0; item < input.angle_count; ++item) {
        const npy_intp* atoms = angle_atoms + 3 * item;
        const double* parameters = angle_parameters + 6 * item;
        const auto position = make_positions(atoms, 3);
        const JetVector left = jet_subtract(position[0], position[1]);
        const JetVector right = jet_subtract(position[2], position[1]);
        const DirectionalJet r1 = jet_norm(left);
        const DirectionalJet r2 = jet_norm(right);
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance) {
            return false;
        }
        const DirectionalJet cosine = jet_dot(left, right) / (r1 * r2);
        if (std::abs(cosine.value) >= 1.0 - 1.0e-12) {
            return false;
        }
        const DirectionalJet theta = jet_acos(cosine);
        const double force = parameters[0];
        const double theta0 = parameters[1];
        const DirectionalJet radial =
            bond_order_factor(r1, parameters[2], parameters[3]) *
            bond_order_factor(r2, parameters[4], parameters[5]);
        DirectionalJet shape;
        double amplitude;
        if (std::abs(std::sin(theta0)) <= 1.0e-5) {
            const DirectionalJet sine = jet_sin(theta);
            shape = sine * sine;
            amplitude = 0.5 * force;
        } else {
            const DirectionalJet difference =
                jet_cos(theta) - std::cos(theta0);
            shape = difference * difference;
            const double sine0 = std::sin(theta0);
            amplitude = force / (2.0 * sine0 * sine0);
        }
        scatter(amplitude * radial * shape, atoms, 3);
    }
    for (npy_intp item = 0; item < input.torsion_count; ++item) {
        const npy_intp* atoms = torsion_atoms + 4 * item;
        const double* parameters = torsion_parameters + 6 * item;
        const auto position = make_positions(atoms, 4);
        const JetVector b1 = jet_subtract(position[0], position[1]);
        const JetVector b2 = jet_subtract(position[2], position[1]);
        const JetVector b3 = jet_subtract(position[3], position[2]);
        const DirectionalJet r1 = jet_norm(b1);
        const DirectionalJet r2 = jet_norm(b2);
        const DirectionalJet r3 = jet_norm(b3);
        const JetVector n1 = jet_cross(b1, b2);
        const JetVector n2 = jet_cross(b2, b3);
        if (r1.value <= kCoincidentTolerance ||
            r2.value <= kCoincidentTolerance ||
            r3.value <= kCoincidentTolerance ||
            jet_norm(n1).value <= kCoincidentTolerance ||
            jet_norm(n2).value <= kCoincidentTolerance) {
            return false;
        }
        const DirectionalJet phi = jet_atan2(
            jet_dot(jet_cross(n1, n2), jet_divide(b2, r2)),
            jet_dot(n1, n2));
        DirectionalJet fourier = jet_constant(0.0);
        for (npy_intp term = offsets[item]; term < offsets[item + 1]; ++term) {
            const double amplitude = terms[3 * term];
            const double periodicity = terms[3 * term + 1];
            const double phase = terms[3 * term + 2];
            fourier = fourier +
                amplitude * (1.0 + jet_cos(periodicity * phi - phase));
        }
        const DirectionalJet radial =
            bond_order_factor(r1, parameters[0], parameters[1]) *
            bond_order_factor(r2, parameters[2], parameters[3]) *
            bond_order_factor(r3, parameters[4], parameters[5]);
        scatter(radial * fourier, atoms, 4);
    }
    return true;
}

PyObject* local_valence_energy(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms;
    if (!PyArg_ParseTuple(
            args, "OOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, Py_None, input)) {
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(input, energy, nullptr, nullptr);
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue(
        "dL", energy, static_cast<long long>(count));
}

PyObject* local_valence_energy_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms;
    if (!PyArg_ParseTuple(
            args, "OOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, Py_None, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (gradient == nullptr) {
        release_local_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(
        input, energy, static_cast<double*>(PyArray_DATA(gradient)), nullptr);
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue("dNL", energy, gradient, static_cast<long long>(count));
}

PyObject* local_valence_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates, *angle_atoms, *angle_parameters, *torsion_atoms;
    PyObject *torsion_parameters, *term_offsets, *terms, *direction;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOO", &coordinates, &angle_atoms, &angle_parameters,
            &torsion_atoms, &torsion_parameters, &term_offsets, &terms,
            &direction)) {
        return nullptr;
    }
    LocalValenceInputs input;
    if (!parse_local_valence_inputs(
            coordinates, angle_atoms, angle_parameters, torsion_atoms,
            torsion_parameters, term_offsets, terms, direction, input)) {
        return nullptr;
    }
    npy_intp dimensions[2] = {input.atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dimensions, NPY_DOUBLE, 0));
    if (product == nullptr) {
        release_local_valence_inputs(input);
        return nullptr;
    }
    double energy = 0.0;
    bool valid;
    Py_BEGIN_ALLOW_THREADS
    valid = accumulate_local_valence(
        input, energy, nullptr, static_cast<double*>(PyArray_DATA(product)));
    Py_END_ALLOW_THREADS
    const npy_intp count = input.angle_count + input.torsion_count;
    release_local_valence_inputs(input);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_FloatingPointError, "singular native local valence geometry");
        return nullptr;
    }
    return Py_BuildValue("NL", product, static_cast<long long>(count));
}

PyObject* planar_image_potential_gradient(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *charges_object, *origin_object, *normal_object;
    if (!PyArg_ParseTuple(
            args, "OOOO", &coordinates_object, &charges_object, &origin_object,
            &normal_object)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* origin = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(origin_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* normal = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(normal_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || charges == nullptr || origin == nullptr ||
        normal == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(charges);
        Py_XDECREF(origin);
        Py_XDECREF(normal);
        return nullptr;
    }
    const bool valid_shapes =
        PyArray_NDIM(coordinates) == 2 && PyArray_DIM(coordinates, 1) == 3 &&
        PyArray_NDIM(charges) == 1 &&
        PyArray_DIM(charges, 0) == PyArray_DIM(coordinates, 0) &&
        PyArray_NDIM(origin) == 1 && PyArray_DIM(origin, 0) == 3 &&
        PyArray_NDIM(normal) == 1 && PyArray_DIM(normal, 0) == 3;
    if (!valid_shapes) {
        PyErr_SetString(
            PyExc_ValueError,
            "native planar image arrays require (N,3), (N,), (3,), (3,)");
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        return nullptr;
    }
    const npy_intp atoms = PyArray_DIM(coordinates, 0);
    npy_intp potential_dimension[1] = {atoms};
    npy_intp gradient_dimensions[2] = {atoms, 3};
    PyArrayObject* potential = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(1, potential_dimension, NPY_DOUBLE, 0));
    PyArrayObject* gradient = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, gradient_dimensions, NPY_DOUBLE, 0));
    if (potential == nullptr || gradient == nullptr) {
        Py_XDECREF(potential);
        Py_XDECREF(gradient);
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        return nullptr;
    }
    const double* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(charges));
    const double* plane = static_cast<const double*>(PyArray_DATA(origin));
    const double* n = static_cast<const double*>(PyArray_DATA(normal));
    double* output_potential = static_cast<double*>(PyArray_DATA(potential));
    double* output_gradient = static_cast<double*>(PyArray_DATA(gradient));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp target = 0; target < atoms; ++target) {
        double target_potential = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (npy_intp source = 0; source < atoms; ++source) {
            const double height =
                (xyz[3 * source] - plane[0]) * n[0] +
                (xyz[3 * source + 1] - plane[1]) * n[1] +
                (xyz[3 * source + 2] - plane[2]) * n[2];
            const double dx =
                xyz[3 * target] - (xyz[3 * source] - 2.0 * height * n[0]);
            const double dy =
                xyz[3 * target + 1] -
                (xyz[3 * source + 1] - 2.0 * height * n[1]);
            const double dz =
                xyz[3 * target + 2] -
                (xyz[3 * source + 2] - 2.0 * height * n[2]);
            const double radius2 = dx * dx + dy * dy + dz * dz;
            if (radius2 <= kCoincidentTolerance * kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double inverse_radius = 1.0 / std::sqrt(radius2);
            const double weighted = q[source] * inverse_radius;
            const double weighted_inverse_radius2 = weighted / radius2;
            target_potential += weighted;
            gx -= weighted_inverse_radius2 * dx;
            gy -= weighted_inverse_radius2 * dy;
            gz -= weighted_inverse_radius2 * dz;
        }
        output_potential[target] = target_potential;
        output_gradient[3 * target] = gx;
        output_gradient[3 * target + 1] = gy;
        output_gradient[3 * target + 2] = gz;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(charges);
    Py_DECREF(origin);
    Py_DECREF(normal);
    if (!valid) {
        Py_DECREF(potential);
        Py_DECREF(gradient);
        PyErr_SetString(PyExc_FloatingPointError, "singular planar image geometry");
        return nullptr;
    }
    return Py_BuildValue("NN", potential, gradient);
}

PyObject* planar_image_hessian_vector(PyObject*, PyObject* args) {
    PyObject *coordinates_object, *charges_object, *origin_object, *normal_object;
    PyObject* direction_object;
    double factor;
    if (!PyArg_ParseTuple(
            args, "OOOOOd", &coordinates_object, &charges_object, &origin_object,
            &normal_object, &direction_object, &factor)) {
        return nullptr;
    }
    PyArrayObject* coordinates = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(coordinates_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* charges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(charges_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* origin = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(origin_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* normal = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(normal_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* direction = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(direction_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (coordinates == nullptr || charges == nullptr || origin == nullptr ||
        normal == nullptr || direction == nullptr) {
        Py_XDECREF(coordinates);
        Py_XDECREF(charges);
        Py_XDECREF(origin);
        Py_XDECREF(normal);
        Py_XDECREF(direction);
        return nullptr;
    }
    const npy_intp atoms =
        PyArray_NDIM(coordinates) == 2 ? PyArray_DIM(coordinates, 0) : -1;
    const int direction_rank = PyArray_NDIM(direction);
    const npy_intp direction_count =
        direction_rank == 2 ? 1 :
        (direction_rank == 3 ? PyArray_DIM(direction, 0) : -1);
    const npy_intp direction_atoms =
        direction_rank == 2 ? PyArray_DIM(direction, 0) :
        (direction_rank == 3 ? PyArray_DIM(direction, 1) : -1);
    const npy_intp direction_axes =
        direction_rank == 2 ? PyArray_DIM(direction, 1) :
        (direction_rank == 3 ? PyArray_DIM(direction, 2) : -1);
    const bool valid_shapes =
        atoms >= 0 && PyArray_DIM(coordinates, 1) == 3 &&
        PyArray_NDIM(charges) == 1 && PyArray_DIM(charges, 0) == atoms &&
        PyArray_NDIM(origin) == 1 && PyArray_DIM(origin, 0) == 3 &&
        PyArray_NDIM(normal) == 1 && PyArray_DIM(normal, 0) == 3 &&
        direction_count >= 1 && direction_atoms == atoms &&
        direction_axes == 3 && std::isfinite(factor);
    if (!valid_shapes) {
        PyErr_SetString(PyExc_ValueError, "native planar image HVP arrays are inconsistent");
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        Py_DECREF(direction);
        return nullptr;
    }
    npy_intp dimensions[3] = {direction_count, atoms, 3};
    PyArrayObject* product = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(
            direction_rank,
            direction_rank == 2 ? dimensions + 1 : dimensions,
            NPY_DOUBLE,
            0));
    if (product == nullptr) {
        Py_DECREF(coordinates);
        Py_DECREF(charges);
        Py_DECREF(origin);
        Py_DECREF(normal);
        Py_DECREF(direction);
        return nullptr;
    }
    const double* xyz = static_cast<const double*>(PyArray_DATA(coordinates));
    const double* q = static_cast<const double*>(PyArray_DATA(charges));
    const double* plane = static_cast<const double*>(PyArray_DATA(origin));
    const double* n = static_cast<const double*>(PyArray_DATA(normal));
    const double* vector = static_cast<const double*>(PyArray_DATA(direction));
    double* output = static_cast<double*>(PyArray_DATA(product));
    bool valid = true;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp density = 0; density < direction_count; ++density) {
      const npy_intp density_offset = 3 * density * atoms;
      for (npy_intp target = 0; target < atoms; ++target) {
        double hx = 0.0, hy = 0.0, hz = 0.0;
        for (npy_intp source = 0; source < atoms; ++source) {
            const double height =
                (xyz[3 * source] - plane[0]) * n[0] +
                (xyz[3 * source + 1] - plane[1]) * n[1] +
                (xyz[3 * source + 2] - plane[2]) * n[2];
            const double dx =
                xyz[3 * target] - (xyz[3 * source] - 2.0 * height * n[0]);
            const double dy =
                xyz[3 * target + 1] -
                (xyz[3 * source + 1] - 2.0 * height * n[1]);
            const double dz =
                xyz[3 * target + 2] -
                (xyz[3 * source + 2] - 2.0 * height * n[2]);
            const double radius2 = dx * dx + dy * dy + dz * dz;
            if (radius2 <= kCoincidentTolerance * kCoincidentTolerance) {
                valid = false;
                continue;
            }
            const double projection =
                vector[density_offset + 3 * source] * n[0] +
                vector[density_offset + 3 * source + 1] * n[1] +
                vector[density_offset + 3 * source + 2] * n[2];
            const double ex =
                vector[density_offset + 3 * target] -
                (vector[density_offset + 3 * source] -
                 2.0 * projection * n[0]);
            const double ey =
                vector[density_offset + 3 * target + 1] -
                (vector[density_offset + 3 * source + 1] -
                 2.0 * projection * n[1]);
            const double ez =
                vector[density_offset + 3 * target + 2] -
                (vector[density_offset + 3 * source + 2] -
                 2.0 * projection * n[2]);
            const double inverse_radius = 1.0 / std::sqrt(radius2);
            const double inverse_radius3 = inverse_radius / radius2;
            const double dot = dx * ex + dy * ey + dz * ez;
            const double coefficient =
                3.0 * dot * inverse_radius3 / radius2;
            hx += q[source] * (coefficient * dx - inverse_radius3 * ex);
            hy += q[source] * (coefficient * dy - inverse_radius3 * ey);
            hz += q[source] * (coefficient * dz - inverse_radius3 * ez);
        }
        const double scale = factor * q[target];
        output[density_offset + 3 * target] = scale * hx;
        output[density_offset + 3 * target + 1] = scale * hy;
        output[density_offset + 3 * target + 2] = scale * hz;
      }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(coordinates);
    Py_DECREF(charges);
    Py_DECREF(origin);
    Py_DECREF(normal);
    Py_DECREF(direction);
    if (!valid) {
        Py_DECREF(product);
        PyErr_SetString(PyExc_FloatingPointError, "singular planar image geometry");
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(product);
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
        "precision", "float64");
}

PyMethodDef methods[] = {
    {"direct_gaussian_energy", direct_gaussian_energy, METH_VARARGS,
     "Direct Gaussian electrostatic energy."},
    {"direct_gaussian_energy_gradient", direct_gaussian_energy_gradient, METH_VARARGS,
     "Direct Gaussian electrostatic energy and analytic gradient."},
    {"direct_gaussian_hessian_vector", direct_gaussian_hessian_vector, METH_VARARGS,
     "Direct Gaussian electrostatic analytic Hessian-vector product."},
    {"gaussian_correction_energy", gaussian_correction_energy, METH_VARARGS,
     "Gaussian penetration correction energy on an explicit pair list."},
    {"gaussian_correction_energy_gradient", gaussian_correction_energy_gradient,
     METH_VARARGS,
     "Gaussian penetration correction energy and gradient on a pair list."},
    {"gaussian_correction_hessian_vector", gaussian_correction_hessian_vector,
     METH_VARARGS,
     "Gaussian penetration correction Hessian-vector product on a pair list."},
    {"gaussian_correction_potential", gaussian_correction_potential, METH_VARARGS,
     "Gaussian penetration correction potential on a pair list."},
    {"damped_exppe_energy", damped_exppe_energy, METH_VARARGS,
     "Rationally damped Exp-PE energy on a cutoff pair list."},
    {"damped_exppe_energy_gradient", damped_exppe_energy_gradient, METH_VARARGS,
     "Rationally damped Exp-PE energy and analytic gradient."},
    {"damped_exppe_hessian_vector", damped_exppe_hessian_vector, METH_VARARGS,
     "Rationally damped Exp-PE analytic Hessian-vector product."},
    {"switched_lj_energy_gradient", switched_lj_energy_gradient, METH_VARARGS,
     "C2-switched Lennard-Jones energy and analytic gradient."},
    {"morse_bond_energy", morse_bond_energy, METH_VARARGS,
     "Morse bond energy."},
    {"morse_bond_energy_gradient", morse_bond_energy_gradient, METH_VARARGS,
     "Morse bond energy and analytic Cartesian gradient."},
    {"morse_bond_hessian_vector", morse_bond_hessian_vector, METH_VARARGS,
     "Morse bond analytic Cartesian Hessian-vector product."},
    {"local_valence_energy", local_valence_energy, METH_VARARGS,
     "Bond-order-damped angle and torsion energy."},
    {"local_valence_energy_gradient", local_valence_energy_gradient, METH_VARARGS,
     "Bond-order-damped angle and torsion energy and analytic gradient."},
    {"local_valence_hessian_vector", local_valence_hessian_vector, METH_VARARGS,
     "Bond-order-damped angle and torsion analytic Hessian-vector product."},
    {"planar_image_potential_gradient", planar_image_potential_gradient, METH_VARARGS,
     "Direct planar image potential and target gradient."},
    {"planar_image_hessian_vector", planar_image_hessian_vector, METH_VARARGS,
     "Direct planar image analytic Hessian-vector product."},
    {"build_info", build_info, METH_NOARGS, "Describe this native build."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_zaff_native",
    "Portable compiled ZAFF numerical kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__zaff_native() {
    import_array();
    return PyModule_Create(&module);
}
