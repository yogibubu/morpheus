"""Continuous ORACLE structural corrections on primitive atom pairs.

The numerical models in this module follow the final working equations in
``CV_radial`` and ``JCP_IS1``.  Distances are expressed in angstrom.  The
core--valence exponential field returns electronic derivatives in atomic
units so that LINK can add them directly to a backend result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, erf, exp, pi
from typing import Iterable, Sequence

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
CV_RADIAL_SIGMA_SCALE = 1.3
CV_RADIAL_WEIGHT_THRESHOLD = 0.9

# Radii and valence metadata used by the published CV_radial calibration.
CV_RADIAL_COVALENT_RADII_ANGSTROM = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}
CV_RADIAL_PERIOD = {
    1: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
}
CV_RADIAL_VALENCE_ELECTRONS = {
    1: 1,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
}

# Class-weighted four-parameter fit from CV_radial.  The result of
# Rcov*(intercept+slope*nval) is in milliangstrom.
CV_RADIAL_RADIUS_AWARE_PERIOD_LINE = {
    2: (-4.1889094413, 0.5212770805),
    3: (-7.2988006002, 0.8169634536),
}
# Joint structural reduction of B and C used for geometry optimization.
# B_XY = -exp(b_X+b_Y) Eh and C_XY is the negative harmonic mean of the
# endpoint period decay constants.
CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE = {
    1: -3.810340191539109,
    5: -0.9994686069660973,
    6: -1.3018727451049243,
    7: -1.4848560102102037,
    8: -1.6341504672362506,
    9: -1.9864199158286915,
    13: 0.0045993717286958585,
    14: -0.4757271225882106,
    15: -0.7307785200402656,
    16: -0.935596238766313,
    17: -1.148208616468822,
}
CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM = {
    1: 0.928290388526001,
    2: 2.564105205901616,
    3: 2.1900014878409646,
}

# JCP_IS1 working PL1 parameterization.  Only OH...O(H) is calibrated.
PL1_HBOND_AMPLITUDE_ANGSTROM = 0.030
PL1_HBOND_REFERENCE_ANGSTROM = 2.15
PL1_HBOND_REFERENCE_UFF_SCALE = 0.81
PL1_HBOND_SIGMA_ANGSTROM = 0.50
PL1_PAIR_SWITCH_SIGMA_ANGSTROM = 0.50
PL1_COVALENT_SWITCH_SIGMA_ANGSTROM = 0.057


@dataclass(frozen=True)
class PairCorrection:
    atoms: tuple[int, int]
    delta_angstrom: float
    layer: str
    weight: float
    donor: int | None = None
    hydrogen: int | None = None
    acceptor: int | None = None


@dataclass(frozen=True)
class HydrogenBondContact:
    donor: int
    hydrogen: int
    acceptor: int
    distance_angstrom: float
    angle_radians: float
    coordination_weight: float
    intramolecular: bool
    pl1_calibrated: bool


@dataclass(frozen=True)
class CVExponentialPair:
    atoms: tuple[int, int]
    amplitude_hartree: float
    decay_inv_angstrom: float


@dataclass(frozen=True)
class CVExponentialFieldResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    pairs: tuple[CVExponentialPair, ...]


def cv_radial_radius_aware_alpha_milliangstrom(atomic_number: int) -> float | None:
    z = int(atomic_number)
    if z == 1:
        return 0.0
    period = CV_RADIAL_PERIOD.get(z)
    valence = CV_RADIAL_VALENCE_ELECTRONS.get(z)
    radius = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(z)
    line = CV_RADIAL_RADIUS_AWARE_PERIOD_LINE.get(period)
    if period is None or valence is None or radius is None or line is None:
        return None
    intercept, slope = line
    return radius * (intercept + slope * valence)


def cv_radial_posterior_amplitude_milliangstrom(
    atomic_number_a: int,
    atomic_number_b: int,
) -> float | None:
    """Return the unique radius-aware CV posterior amplitude."""
    left = cv_radial_radius_aware_alpha_milliangstrom(atomic_number_a)
    right = cv_radial_radius_aware_alpha_milliangstrom(atomic_number_b)
    return None if left is None or right is None else left + right


def cv_radial_bond_delta_angstrom(
    atomic_number_a: int,
    atomic_number_b: int,
    distance_angstrom: float,
    *,
    sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    weight_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> float | None:
    radius_a = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(atomic_number_a))
    radius_b = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(atomic_number_b))
    if radius_a is None or radius_b is None:
        return None
    if sigma_scale <= 0.0:
        raise ValueError("CV-radial sigma scale must be positive")
    if not 0.0 <= weight_threshold <= 1.0:
        raise ValueError("CV-radial weight threshold must lie between zero and one")
    r0 = radius_a + radius_b
    weight = exp(-((float(distance_angstrom) - r0) / (sigma_scale * r0)) ** 2)
    if weight < weight_threshold:
        return None
    amplitude = cv_radial_posterior_amplitude_milliangstrom(atomic_number_a, atomic_number_b)
    return None if amplitude is None else weight * amplitude / 1000.0


def cv_exponential_pair_parameters(
    atomic_number_a: int, atomic_number_b: int
) -> tuple[float, float] | None:
    za, zb = int(atomic_number_a), int(atomic_number_b)
    log_a = CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE.get(za)
    log_b = CV_EXPONENTIAL_ATOMIC_LOG_AMPLITUDE.get(zb)
    period_a, period_b = CV_RADIAL_PERIOD.get(za), CV_RADIAL_PERIOD.get(zb)
    if log_a is None or log_b is None or period_a is None or period_b is None:
        return None
    lambda_a = CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM[period_a]
    lambda_b = CV_EXPONENTIAL_PERIOD_DECAY_INV_ANGSTROM[period_b]
    amplitude = -exp(log_a + log_b)
    decay = -2.0 * lambda_a * lambda_b / (lambda_a + lambda_b)
    return amplitude, decay


def evaluate_cv_exponential_field(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
) -> CVExponentialFieldResult:
    """Evaluate the LINK CV field, including exact radial E/G/H terms."""
    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("CV field coordinates must have shape (natoms, 3)")
    gradient_a = np.zeros_like(xyz)
    hessian_a = np.zeros((3 * len(numbers), 3 * len(numbers)), dtype=float)
    energy = 0.0
    records: list[CVExponentialPair] = []
    for raw_i, raw_j in sorted({tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}):
        if raw_i == raw_j or raw_i < 0 or raw_j >= len(numbers):
            raise IndexError(f"invalid CV field atom pair: {(raw_i, raw_j)}")
        parameters = cv_exponential_pair_parameters(numbers[raw_i], numbers[raw_j])
        if parameters is None:
            continue
        amplitude, decay = parameters
        vector = xyz[raw_i] - xyz[raw_j]
        distance = float(np.linalg.norm(vector))
        if distance <= 1.0e-12:
            raise ValueError("CV field is undefined for coincident atoms")
        unit = vector / distance
        value = amplitude * exp(decay * distance)
        first = decay * value
        second = decay * first
        energy += value
        gradient_a[raw_i] += first * unit
        gradient_a[raw_j] -= first * unit
        radial_hessian = second * np.outer(unit, unit) + (first / distance) * (
            np.eye(3) - np.outer(unit, unit)
        )
        si, sj = slice(3 * raw_i, 3 * raw_i + 3), slice(3 * raw_j, 3 * raw_j + 3)
        hessian_a[si, si] += radial_hessian
        hessian_a[sj, sj] += radial_hessian
        hessian_a[si, sj] -= radial_hessian
        hessian_a[sj, si] -= radial_hessian
        records.append(CVExponentialPair((raw_i, raw_j), amplitude, decay))
    return CVExponentialFieldResult(
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=gradient_a.reshape(-1) * BOHR_TO_ANGSTROM,
        hessian_hartree_per_bohr2=hessian_a * BOHR_TO_ANGSTROM**2,
        pairs=tuple(records),
    )


def coordination_switch(distance: float, reference: float, scale: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("coordination switching width must be positive")
    return 0.5 * (1.0 - erf((float(distance) - scale * float(reference)) / width))


def hbond_angular_factor(angle_radians: float) -> float:
    return 0.5 * (1.0 + cos(pi - float(angle_radians)))


def pl1_hbond_delta_angstrom(distance_angstrom: float, dha_angle_radians: float) -> float:
    gaussian = exp(
        -((float(distance_angstrom) - PL1_HBOND_REFERENCE_ANGSTROM) / PL1_HBOND_SIGMA_ANGSTROM)
        ** 2
    )
    return PL1_HBOND_AMPLITUDE_ANGSTROM * gaussian * hbond_angular_factor(dha_angle_radians)


def perceive_hydrogen_bonds(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    selector_threshold: float = 1.0e-3,
    minimum_angle_degrees: float = 110.0,
) -> tuple[HydrogenBondContact, ...]:
    """Perceive continuous X--H...A contacts for ORACLE pseudo-bonds.

    The general perception accepts N/O/P/S donors and acceptors as specified in
    JCP_IS1.  ``pl1_calibrated`` is deliberately narrower: the current PL1
    parameters are valid only for O--H...O--H contacts.
    """
    from .topology.vdw_radii import uff_vdw_radius

    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(numbers), 3):
        raise ValueError("hydrogen-bond perception needs coordinates with shape (natoms, 3)")
    if not 0.0 <= selector_threshold <= 1.0:
        raise ValueError("hydrogen-bond selector threshold must lie between zero and one")
    bonds = {tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _connected_components(adjacency)
    component_by_atom = {
        atom: component for component, atoms in enumerate(components) for atom in atoms
    }
    eligible = {7, 8, 15, 16}
    contacts: list[HydrogenBondContact] = []
    for hydrogen, z_h in enumerate(numbers):
        if z_h != 1 or len(adjacency[hydrogen]) != 1:
            continue
        donor = next(iter(adjacency[hydrogen]))
        if numbers[donor] not in eligible:
            continue
        dh = xyz[donor] - xyz[hydrogen]
        for acceptor, z_acceptor in enumerate(numbers):
            if acceptor in {hydrogen, donor} or z_acceptor not in eligible:
                continue
            if tuple(sorted((hydrogen, acceptor))) in bonds:
                continue
            radius_h = uff_vdw_radius(1)
            radius_a = uff_vdw_radius(z_acceptor)
            if radius_h is None or radius_a is None:
                continue
            ha = xyz[acceptor] - xyz[hydrogen]
            distance = float(np.linalg.norm(ha))
            reference = float(radius_h + radius_a)
            weight = coordination_switch(
                distance, reference, 1.0, PL1_PAIR_SWITCH_SIGMA_ANGSTROM
            )
            if weight < selector_threshold:
                continue
            denominator = float(np.linalg.norm(dh) * np.linalg.norm(ha))
            if denominator <= 1.0e-12:
                continue
            angle = float(np.arccos(np.clip(np.dot(dh, ha) / denominator, -1.0, 1.0)))
            if angle < np.deg2rad(float(minimum_angle_degrees)):
                continue
            if hbond_angular_factor(angle) < selector_threshold:
                continue
            acceptor_bears_hydrogen = any(numbers[item] == 1 for item in adjacency[acceptor])
            contacts.append(
                HydrogenBondContact(
                    donor=donor,
                    hydrogen=hydrogen,
                    acceptor=acceptor,
                    distance_angstrom=distance,
                    angle_radians=angle,
                    coordination_weight=weight,
                    intramolecular=component_by_atom[hydrogen] == component_by_atom[acceptor],
                    pl1_calibrated=(
                        numbers[donor] == 8
                        and numbers[acceptor] == 8
                        and acceptor_bears_hydrogen
                    ),
                )
            )
    return tuple(
        sorted(
            contacts,
            key=lambda item: (item.hydrogen, item.distance_angstrom, item.acceptor),
        )
    )


def pseudo_bond_pairs(
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
    *,
    selector_threshold: float = 1.0e-3,
) -> tuple[tuple[tuple[int, int], str], ...]:
    """Return H-bond pseudo-pairs and an MST joining remaining fragments."""
    numbers = tuple(int(value) for value in atomic_numbers)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    bonds = {tuple(sorted((int(i), int(j)))) for i, j in bonded_pairs}
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _connected_components(adjacency)
    if len(components) <= 1:
        return tuple(
            ((contact.hydrogen, contact.acceptor), "INTRAMOLECULAR_HBOND")
            for contact in perceive_hydrogen_bonds(
                numbers, xyz, bonds, selector_threshold=selector_threshold
            )
            if contact.intramolecular
        )
    component_by_atom = {
        atom: component for component, atoms in enumerate(components) for atom in atoms
    }
    selected: list[tuple[tuple[int, int], str]] = []
    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for contact in perceive_hydrogen_bonds(
        numbers, xyz, bonds, selector_threshold=selector_threshold
    ):
        pair = tuple(sorted((contact.hydrogen, contact.acceptor)))
        selected.append(
            (pair, "INTRAMOLECULAR_HBOND" if contact.intramolecular else "INTERFRAGMENT_HBOND")
        )
        left, right = component_by_atom[pair[0]], component_by_atom[pair[1]]
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    candidates: list[tuple[float, int, int, tuple[int, int]]] = []
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            pair = min(
                (
                    (float(np.linalg.norm(xyz[i] - xyz[j])), tuple(sorted((i, j))))
                    for i in components[left]
                    for j in components[right]
                ),
                key=lambda item: (item[0], item[1]),
            )
            candidates.append((pair[0], left, right, pair[1]))
    for _distance, left, right, pair in sorted(candidates):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        selected.append((pair, "INTERFRAGMENT_CLOSEST"))
    return tuple(dict.fromkeys(selected))


def _connected_components(adjacency: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(adjacency)))
    result: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack, component = [start], set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            stack.extend(sorted(adjacency[atom] - component, reverse=True))
        remaining -= component
        result.append(tuple(sorted(component)))
    return tuple(result)
