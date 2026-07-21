from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_core import kabsch_rotation

from matrix_chem.inertia import center_of_mass, inertia_tensor
from matrix_chem.isotopes_table import get_default_isotope, get_isotope
from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_chem.structure import Structure
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_smith.survibfit.primitives import Primitive, eval_primitives

from .contracts import IsotopologueObservation, QMParameterPredicate


ROTCONST_TO_MOMENT = (
    get_physical_constants()[Phy.PLANCK]
    / (8.0 * np.pi**2 * get_physical_constants()[Phy.TO_KG] * (1.0e-10) ** 2)
    * 1.0e-6
)


@dataclass(frozen=True)
class KraitchmanComparison:
    isotopologue: str
    atom_index: int
    atom: str
    isotope_mass_number: int
    coordinate: str
    kraitchman_abs_angstrom: float
    fitted_abs_angstrom: float
    difference_angstrom: float
    signed_kraitchman_angstrom: float = 0.0
    signed_reference_angstrom: float = 0.0
    substitution_mass_amu: float = 0.0


@dataclass(frozen=True)
class KraitchmanSeedResult:
    coordinates_angstrom: np.ndarray
    rows: tuple[KraitchmanComparison, ...]
    method: str
    fitted_atom_indices: tuple[int, ...]
    rms_atom_displacement_angstrom: float


def kraitchman_seed_predicates(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    *,
    sigma_distance_angstrom: float = 0.01,
    sigma_angle_degree: float = 1.0,
    include_angles: bool = True,
    require_all_atoms_seeded: bool = True,
) -> tuple[QMParameterPredicate, ...]:
    """Build local primitive predicates from the Kraitchman seed geometry.

    Kraitchman substitution coordinates are not exact structural constraints.
    This helper therefore returns weighted predicates, and by default only for
    primitive coordinates whose atoms are all directly determined by single
    substitutions.
    """

    if sigma_distance_angstrom <= 0.0:
        raise ValueError("sigma_distance_angstrom must be positive")
    if sigma_angle_degree <= 0.0:
        raise ValueError("sigma_angle_degree must be positive")
    rows = kraitchman_comparison(atoms, coords, observations)
    seed = kraitchman_seed_geometry(atoms, coords, observations, rows)
    if seed is None:
        return ()
    seeded_atoms = {idx - 1 for idx in seed.fitted_atom_indices}
    z_numbers = np.array([atomic_number(symbol) or 0 for symbol in atoms], dtype=int)
    _continuous, graph, _ringset, _synthons, _aromaticity = build_topology_objects(
        np.asarray(coords, dtype=float), z_numbers
    )
    bonds = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in graph.bonds))
    adjacency = tuple(
        tuple(sorted(int(item) for item in graph.adjacency[index]))
        for index in range(len(atoms))
    )
    predicates: list[QMParameterPredicate] = []
    for i, j in bonds:
        primitive = Primitive("bond", (i, j))
        if require_all_atoms_seeded and not {i, j}.issubset(seeded_atoms):
            continue
        value = float(eval_primitives([primitive], seed.coordinates_angstrom)[0])
        predicates.append(
            QMParameterPredicate(
                f"R({i + 1},{j + 1})",
                value,
                sigma_distance_angstrom,
                "kraitchman_seed_distance",
            )
        )
    if include_angles:
        for center, neighbors in enumerate(adjacency):
            for pos, left in enumerate(neighbors):
                for right in neighbors[pos + 1 :]:
                    atoms_set = {left, center, right}
                    if require_all_atoms_seeded and not atoms_set.issubset(seeded_atoms):
                        continue
                    primitive = Primitive("angle", (left, center, right))
                    value = float(np.rad2deg(eval_primitives([primitive], seed.coordinates_angstrom)[0]))
                    predicates.append(
                        QMParameterPredicate(
                            f"A({left + 1},{center + 1},{right + 1})",
                            value,
                            sigma_angle_degree,
                            "kraitchman_seed_angle",
                        )
                    )
    return tuple(predicates)


def kraitchman_comparison(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> tuple[KraitchmanComparison, ...]:
    parent = next((obs for obs in observations if not obs.substitutions), None)
    if parent is None:
        return ()
    axis_coords = principal_axis_coordinates(atoms, coords)
    parent_moments = np.array(constants_to_moments(parent.corrected.as_tuple()), dtype=float)
    parent_mass = parent_total_mass(atoms, coords)
    rows: list[KraitchmanComparison] = []
    for obs in observations:
        if len(obs.substitutions) != 1:
            continue
        atom_index, isotope_a = next(iter(obs.substitutions.items()))
        atom_pos = atom_index - 1
        if atom_pos < 0 or atom_pos >= len(atoms):
            continue
        substitution_mass = kraitchman_substitution_mass(
            atoms[atom_pos], int(isotope_a), parent_mass
        )
        if substitution_mass <= 0.0:
            continue
        moments = np.array(constants_to_moments(obs.corrected.as_tuple()), dtype=float)
        delta = moments - parent_moments
        squared = (
            (delta[1] + delta[2] - delta[0]) / (2.0 * substitution_mass),
            (delta[0] + delta[2] - delta[1]) / (2.0 * substitution_mass),
            (delta[0] + delta[1] - delta[2]) / (2.0 * substitution_mass),
        )
        for axis, value, fitted in zip(("a", "b", "c"), squared, axis_coords[atom_pos]):
            kraitchman_abs = float(np.sqrt(max(value, 0.0)))
            signed = float(np.copysign(kraitchman_abs, fitted)) if kraitchman_abs else 0.0
            fitted_abs = float(abs(fitted))
            rows.append(
                KraitchmanComparison(
                    isotopologue=obs.label,
                    atom_index=atom_index,
                    atom=str(atoms[atom_pos]),
                    isotope_mass_number=int(isotope_a),
                    coordinate=axis,
                    kraitchman_abs_angstrom=kraitchman_abs,
                    fitted_abs_angstrom=fitted_abs,
                    difference_angstrom=kraitchman_abs - fitted_abs,
                    signed_kraitchman_angstrom=signed,
                    signed_reference_angstrom=float(fitted),
                    substitution_mass_amu=float(substitution_mass),
                )
            )
    return tuple(rows)


def kraitchman_seed_geometry(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    rows: tuple[KraitchmanComparison, ...] | None = None,
) -> KraitchmanSeedResult | None:
    kraitchman_rows = (
        rows if rows is not None else kraitchman_comparison(atoms, coords, observations)
    )
    atom_targets = _atom_targets(kraitchman_rows)
    if not atom_targets:
        return None
    axis_coords = principal_axis_coordinates(atoms, coords)
    seeded = axis_coords.copy()
    atom_indices = tuple(sorted(atom_targets))
    reference = np.array([axis_coords[idx - 1] for idx in atom_indices], dtype=float)
    target = np.array([atom_targets[idx] for idx in atom_indices], dtype=float)
    method = "direct_substitution"
    if len(atom_indices) >= 3 and _rank3(reference) >= 2 and _rank3(target) >= 2:
        rotated, ok = _rigid_kabsch_update(axis_coords, reference, target)
        if ok:
            seeded = rotated
            method = "rigid_kabsch_plus_exact_substitution"
    for atom_index, target_coord in atom_targets.items():
        seeded[atom_index - 1] = target_coord
    displacements = seeded[[idx - 1 for idx in atom_indices]] - reference
    rms = (
        float(np.sqrt(np.mean(np.sum(displacements * displacements, axis=1))))
        if atom_indices
        else 0.0
    )
    return KraitchmanSeedResult(
        coordinates_angstrom=seeded,
        rows=kraitchman_rows,
        method=method,
        fitted_atom_indices=atom_indices,
        rms_atom_displacement_angstrom=rms,
    )


def constants_to_moments(constants: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(ROTCONST_TO_MOMENT / value if value > 0.0 else 0.0 for value in constants)


def principal_axis_coordinates(
    atoms: list[str] | tuple[str, ...], coords: np.ndarray
) -> np.ndarray:
    structure = Structure.from_atoms_coords(
        list(atoms), [tuple(row) for row in np.asarray(coords, dtype=float)]
    )
    inertia = inertia_tensor(structure, isotopic=True)
    eigvals, eigvecs = np.linalg.eigh(inertia)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0.0:
        eigvecs[:, -1] *= -1.0
    centered = np.asarray(coords, dtype=float) - center_of_mass(structure, isotopic=True)
    return centered @ eigvecs


def parent_total_mass(atoms: list[str] | tuple[str, ...], coords: np.ndarray) -> float:
    structure = Structure.from_atoms_coords(
        list(atoms), [tuple(row) for row in np.asarray(coords, dtype=float)]
    )
    return float(sum(structure.mass_isotope))


def kraitchman_substitution_mass(atom: str, isotope_a: int, parent_mass: float) -> float:
    z_number = atomic_number(atom)
    if z_number is None:
        return 0.0
    default_iso = get_default_isotope(int(z_number))
    substituted_iso = get_isotope(int(z_number), int(isotope_a))
    if default_iso is None or substituted_iso is None:
        return 0.0
    delta_mass = float(substituted_iso.mass - default_iso.mass)
    if delta_mass <= 0.0 or parent_mass <= 0.0:
        return 0.0
    return delta_mass * parent_mass / (parent_mass + delta_mass)


def _atom_targets(rows: tuple[KraitchmanComparison, ...]) -> dict[int, np.ndarray]:
    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(row.atom_index, {})[row.coordinate] = row.signed_kraitchman_angstrom
    targets = {}
    for atom_index, axes in grouped.items():
        if {"a", "b", "c"}.issubset(axes):
            targets[atom_index] = np.array([axes["a"], axes["b"], axes["c"]], dtype=float)
    return targets


def _rank3(points: np.ndarray) -> int:
    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not singular.size:
        return 0
    tol = max(centered.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    return int(np.sum(singular > tol))


def _rigid_kabsch_update(
    coords: np.ndarray, reference: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, bool]:
    ref_center = np.mean(reference, axis=0)
    target_center = np.mean(target, axis=0)
    try:
        rotation = kabsch_rotation(reference, target)
    except ValueError:
        return coords.copy(), False
    return (np.asarray(coords, dtype=float) - ref_center) @ rotation + target_center, True
