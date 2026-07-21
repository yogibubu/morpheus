"""Analytic continuous-topology scaling for non-bonded pair potentials."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from typing import Callable, Protocol

import numpy as np

from .topology.covalent_radii import covalent_radius
from .topology.descriptor_parameters import CNA_SCALE


BOHR_TO_ANGSTROM = 0.52917721092


class RadialDerivatives(Protocol):
    energy: float
    first: float
    second: float


@dataclass(frozen=True)
class _Scalar:
    value: float
    gradient: dict[int, float]
    hessian: dict[tuple[int, int], float]


def continuous_topology_scaled_pair_derivatives(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    radial_factory: Callable[[int, int, float], RadialDerivatives | None],
    *,
    one_four_scale: float,
    switch_alpha_per_angstrom: float = 8.0,
    pair_components: dict[
        tuple[int, int], tuple[float, np.ndarray, np.ndarray]
    ] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate all pairs with smooth 1-2/1-3/1-4 path attenuation.

    Edge membership is the same error-function contribution used in ORACLE's
    continuous coordination number.  Soft unions of one-, two- and three-edge
    paths replace discrete topological exclusions.  A sparse second-order
    algebra carries the complete analytic chain rule through the path weights.
    """

    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    natoms = len(numbers)
    pairs = tuple((i, j) for i in range(natoms) for j in range(i + 1, natoms))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    distances: dict[tuple[int, int], float] = {}
    variables: dict[tuple[int, int], _Scalar] = {}
    edges: dict[tuple[int, int], _Scalar] = {}
    alpha_bohr = float(switch_alpha_per_angstrom) * BOHR_TO_ANGSTROM
    if not np.isfinite(alpha_bohr) or alpha_bohr <= 0.0:
        raise ValueError("continuous topology switch alpha must be finite and positive")
    for pair, index in pair_index.items():
        i, j = pair
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance <= 1.0e-12:
            raise ValueError(f"non-bonded atoms {i + 1} and {j + 1} are coincident")
        distances[pair] = distance
        variables[pair] = _variable(index, distance, 1.0, 0.0)
        ri = covalent_radius(int(numbers[i]))
        rj = covalent_radius(int(numbers[j]))
        if ri is None or rj is None:
            edges[pair] = _constant(0.0)
            continue
        radius_sum_bohr = CNA_SCALE * (float(ri) + float(rj)) / BOHR_TO_ANGSTROM
        argument = alpha_bohr * (radius_sum_bohr - distance)
        gaussian = exp(-(argument**2)) / sqrt(pi)
        value = 0.5 * (1.0 + erf(argument))
        first = -alpha_bohr * gaussian
        second = -2.0 * alpha_bohr**2 * argument * gaussian
        edges[pair] = _variable(index, value, first, second)

    total = _constant(0.0)
    one = _constant(1.0)
    for i, j in pairs:
        radial = radial_factory(i, j, distances[(i, j)])
        if radial is None:
            continue
        one_edge = edges[(i, j)]
        two_paths = [
            _multiply(_edge(edges, i, k), _edge(edges, k, j))
            for k in range(natoms)
            if k not in {i, j}
        ]
        three_paths = [
            _multiply(
                _multiply(_edge(edges, i, k), _edge(edges, k, ell)),
                _edge(edges, ell, j),
            )
            for k in range(natoms)
            for ell in range(natoms)
            if k not in {i, j} and ell not in {i, j, k}
        ]
        two_edge = _soft_union(two_paths)
        three_edge = _soft_union(three_paths)
        scale = _multiply(
            _multiply(_subtract(one, one_edge), _subtract(one, two_edge)),
            _subtract(one, _scale(three_edge, 1.0 - float(one_four_scale))),
        )
        radial_scalar = _variable(
            pair_index[(i, j)],
            float(radial.energy),
            float(radial.first),
            float(radial.second),
        )
        contribution = _multiply(scale, radial_scalar)
        if pair_components is not None:
            pair_components[(i, j)] = _to_cartesian(
                contribution,
                coords,
                pairs,
                distances,
            )
        total = _add(total, contribution)
    return _to_cartesian(total, coords, pairs, distances)


def continuous_topology_pair_scales(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    one_four_scale: float = 0.5,
    switch_alpha_per_angstrom: float = 8.0,
) -> dict[tuple[int, int], float]:
    """Return one-based continuous pair weights for diagnostics."""

    values: dict[tuple[int, int], float] = {}

    @dataclass(frozen=True)
    class _UnitRadial:
        energy: float = 1.0
        first: float = 0.0
        second: float = 0.0

    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    for left in range(len(numbers)):
        for right in range(left + 1, len(numbers)):
            selected = (left, right)

            def factory(i: int, j: int, _distance: float):
                return _UnitRadial() if (i, j) == selected else None

            energy, _gradient, _hessian = continuous_topology_scaled_pair_derivatives(
                coords,
                numbers,
                factory,
                one_four_scale=one_four_scale,
                switch_alpha_per_angstrom=switch_alpha_per_angstrom,
            )
            values[(left + 1, right + 1)] = energy
    return values


def _edge(edges: dict[tuple[int, int], _Scalar], i: int, j: int) -> _Scalar:
    return edges[tuple(sorted((i, j)))]


def _constant(value: float) -> _Scalar:
    return _Scalar(float(value), {}, {})


def _variable(index: int, value: float, first: float, second: float) -> _Scalar:
    gradient = {} if first == 0.0 else {index: float(first)}
    hessian = {} if second == 0.0 else {(index, index): float(second)}
    return _Scalar(float(value), gradient, hessian)


def _add(left: _Scalar, right: _Scalar) -> _Scalar:
    gradient = dict(left.gradient)
    hessian = dict(left.hessian)
    for key, value in right.gradient.items():
        gradient[key] = gradient.get(key, 0.0) + value
    for key, value in right.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value
    return _Scalar(left.value + right.value, gradient, hessian)


def _scale(item: _Scalar, factor: float) -> _Scalar:
    return _Scalar(
        factor * item.value,
        {key: factor * value for key, value in item.gradient.items()},
        {key: factor * value for key, value in item.hessian.items()},
    )


def _subtract(left: _Scalar, right: _Scalar) -> _Scalar:
    return _add(left, _scale(right, -1.0))


def _multiply(left: _Scalar, right: _Scalar) -> _Scalar:
    gradient: dict[int, float] = {}
    hessian: dict[tuple[int, int], float] = {}
    for key in left.gradient.keys() | right.gradient.keys():
        gradient[key] = left.gradient.get(key, 0.0) * right.value + left.value * right.gradient.get(
            key, 0.0
        )
    for key, value in left.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value * right.value
    for key, value in right.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value * left.value
    for i, value_i in left.gradient.items():
        for j, value_j in right.gradient.items():
            key = tuple(sorted((i, j)))
            factor = 2.0 if i == j else 1.0
            hessian[key] = hessian.get(key, 0.0) + factor * value_i * value_j
    return _Scalar(left.value * right.value, gradient, hessian)


def _soft_union(items: list[_Scalar]) -> _Scalar:
    complement = _constant(1.0)
    one = _constant(1.0)
    for item in items:
        complement = _multiply(complement, _subtract(one, item))
    return _subtract(one, complement)


def _to_cartesian(
    value: _Scalar,
    coordinates_bohr: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    distances: dict[tuple[int, int], float],
) -> tuple[float, np.ndarray, np.ndarray]:
    size = 3 * len(coordinates_bohr)
    distance_gradients: list[np.ndarray] = []
    gradient = np.zeros(size, dtype=float)
    hessian = np.zeros((size, size), dtype=float)
    for index, (i, j) in enumerate(pairs):
        rvec = coordinates_bohr[i] - coordinates_bohr[j]
        distance = distances[(i, j)]
        unit = rvec / distance
        dq = np.zeros(size, dtype=float)
        dq[3 * i : 3 * i + 3] = unit
        dq[3 * j : 3 * j + 3] = -unit
        distance_gradients.append(dq)
        coefficient = value.gradient.get(index, 0.0)
        gradient += coefficient * dq
        geometric = coefficient / distance * (np.eye(3) - np.outer(unit, unit))
        si = slice(3 * i, 3 * i + 3)
        sj = slice(3 * j, 3 * j + 3)
        hessian[si, si] += geometric
        hessian[sj, sj] += geometric
        hessian[si, sj] -= geometric
        hessian[sj, si] -= geometric
    for (left, right), coefficient in value.hessian.items():
        outer = np.outer(distance_gradients[left], distance_gradients[right])
        hessian += coefficient * (outer if left == right else outer + outer.T)
    return float(value.value), gradient, 0.5 * (hessian + hessian.T)
