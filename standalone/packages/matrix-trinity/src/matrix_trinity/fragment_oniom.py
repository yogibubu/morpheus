"""Symmetric fragment-ONIOM energy, derivative, and Hessian assembly."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .optimizer import OptimizerCoordinateModel, optimizer_hessian_from_cartesian


FRAGMENT_ONIOM_SCHEMA = "matrix.trinity.fragment_oniom.v1"


@dataclass(frozen=True)
class CartesianEvaluation:
    """Energy derivatives for one model evaluated at a common geometry."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        hessian = np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        if hessian.shape != (gradient.size, gradient.size):
            raise ValueError("Cartesian Hessian shape must match the gradient dimension")
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise ValueError("Cartesian evaluation contains non-finite derivatives")
        object.__setattr__(self, "energy_hartree", float(self.energy_hartree))
        object.__setattr__(self, "gradient_hartree_per_bohr", gradient)
        object.__setattr__(self, "hessian_hartree_per_bohr2", 0.5 * (hessian + hessian.T))


@dataclass(frozen=True)
class FragmentModelCorrection:
    """High-minus-low correction for one fragment."""

    identifier: str
    atom_indices: tuple[int, ...]
    high: CartesianEvaluation
    low: CartesianEvaluation


@dataclass(frozen=True)
class FragmentOniomEvaluation:
    """Composite full-system energy and Cartesian derivatives."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    fragment_identifiers: tuple[str, ...]
    schema: str = FRAGMENT_ONIOM_SCHEMA


def assemble_fragment_oniom_evaluation(
    low_full: CartesianEvaluation,
    corrections: Sequence[FragmentModelCorrection],
    *,
    natoms: int,
    require_partition: bool = True,
) -> FragmentOniomEvaluation:
    """Assemble E_L(full) + sum_i [E_H(F_i) - E_L(F_i)] and its derivatives.

    Atom indices are zero-based. Fragment high/low derivatives are local
    Cartesian arrays ordered as the fragment atom list. All terms must be
    evaluated at the same full-system geometry.
    """

    ncart = 3 * int(natoms)
    if natoms < 1 or low_full.gradient_hartree_per_bohr.size != ncart:
        raise ValueError("low-full evaluation does not match natoms")
    energy = low_full.energy_hartree
    gradient = low_full.gradient_hartree_per_bohr.copy()
    hessian = low_full.hessian_hartree_per_bohr2.copy()
    assigned: set[int] = set()
    identifiers: list[str] = []
    for correction in corrections:
        atoms = tuple(int(index) for index in correction.atom_indices)
        if not correction.identifier:
            raise ValueError("fragment correction needs an identifier")
        if not atoms or len(set(atoms)) != len(atoms):
            raise ValueError(f"fragment {correction.identifier} has invalid atom indices")
        if min(atoms) < 0 or max(atoms) >= natoms:
            raise ValueError(f"fragment {correction.identifier} atom index is outside the system")
        overlap = assigned.intersection(atoms)
        if overlap:
            raise ValueError("fragment corrections must not overlap")
        assigned.update(atoms)
        local_ncart = 3 * len(atoms)
        if correction.high.gradient_hartree_per_bohr.size != local_ncart:
            raise ValueError(f"high evaluation for {correction.identifier} has wrong dimension")
        if correction.low.gradient_hartree_per_bohr.size != local_ncart:
            raise ValueError(f"low evaluation for {correction.identifier} has wrong dimension")
        cart = _cartesian_indices(atoms)
        energy += correction.high.energy_hartree - correction.low.energy_hartree
        gradient[cart] += (
            correction.high.gradient_hartree_per_bohr
            - correction.low.gradient_hartree_per_bohr
        )
        hessian[np.ix_(cart, cart)] += (
            correction.high.hessian_hartree_per_bohr2
            - correction.low.hessian_hartree_per_bohr2
        )
        identifiers.append(correction.identifier)
    if require_partition and assigned != set(range(natoms)):
        missing = sorted(set(range(natoms)) - assigned)
        raise ValueError(f"fragment corrections do not cover atoms: {missing}")
    hessian = 0.5 * (hessian + hessian.T)
    return FragmentOniomEvaluation(
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        fragment_identifiers=tuple(identifiers),
    )


def assemble_fragment_oniom_from_xyzin(
    xyzin_path: Path | str,
    low_full: CartesianEvaluation,
    *,
    high_by_fragment: dict[str, CartesianEvaluation],
    low_by_fragment: dict[str, CartesianEvaluation],
    require_electronic_states: bool = True,
) -> FragmentOniomEvaluation:
    """Assemble the composite derivatives using the shared #FRAGMENTS contract."""

    from matrix_core import read_xyzin_geometry
    from matrix_fragments import read_fragment_records

    target = Path(xyzin_path)
    geometry = read_xyzin_geometry(target)
    fragments = read_fragment_records(target)
    if not fragments:
        raise ValueError("fragment ONIOM requires a built #FRAGMENTS section")
    identifiers = {fragment.identifier for fragment in fragments}
    if set(high_by_fragment) != identifiers or set(low_by_fragment) != identifiers:
        raise ValueError("high/low fragment evaluations must exactly match #FRAGMENTS")
    corrections: list[FragmentModelCorrection] = []
    for fragment in fragments:
        if require_electronic_states and (
            fragment.charge is None or fragment.multiplicity is None
        ):
            raise ValueError(
                f"fragment {fragment.identifier} needs explicit CHARGE and MULTIPLICITY"
            )
        corrections.append(
            FragmentModelCorrection(
                identifier=fragment.identifier,
                atom_indices=tuple(atom - 1 for atom in fragment.atoms),
                high=high_by_fragment[fragment.identifier],
                low=low_by_fragment[fragment.identifier],
            )
        )
    return assemble_fragment_oniom_evaluation(
        low_full,
        corrections,
        natoms=len(geometry.atoms),
        require_partition=True,
    )


def fragment_oniom_optimizer_hessian(
    evaluation: FragmentOniomEvaluation,
    model: OptimizerCoordinateModel,
) -> np.ndarray:
    """Transform the rigorously assembled Cartesian Hessian to active SONIC."""

    return optimizer_hessian_from_cartesian(evaluation.hessian_hartree_per_bohr2, model)


def assemble_fragment_optimizer_hessian_blocks(
    base_hessian: np.ndarray,
    full_labels: Sequence[str],
    fragment_blocks: Sequence[tuple[Sequence[str], np.ndarray]],
) -> np.ndarray:
    """Replace intrafragment blocks in a full SONIC Hessian by staged results."""

    labels = tuple(str(label) for label in full_labels)
    matrix = np.asarray(base_hessian, dtype=float).copy()
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("base Hessian shape does not match full SONIC labels")
    positions = {label: index for index, label in enumerate(labels)}
    assigned: set[int] = set()
    for block_labels, block_hessian in fragment_blocks:
        names = tuple(str(label) for label in block_labels)
        try:
            indices = [positions[label] for label in names]
        except KeyError as exc:
            raise ValueError(f"fragment Hessian label is absent from full SONIC: {exc}") from exc
        if assigned.intersection(indices):
            raise ValueError("fragment Hessian blocks overlap")
        assigned.update(indices)
        block = np.asarray(block_hessian, dtype=float)
        if block.shape != (len(indices), len(indices)):
            raise ValueError("fragment Hessian block shape does not match its labels")
        matrix[np.ix_(indices, indices)] = 0.5 * (block + block.T)
    return 0.5 * (matrix + matrix.T)


def symmetric_two_oniom_hessian_seed(
    fragment_one_oniom: np.ndarray,
    fragment_two_oniom: np.ndarray,
    *,
    fragment_one_atoms: Sequence[int],
    fragment_two_atoms: Sequence[int],
    natoms: int,
) -> np.ndarray:
    """Build the user's symmetric two-job seed without halving QM blocks.

    The F1--F1 block is taken from the job with F1 high, the F2--F2 block from
    the job with F2 high, and the intermolecular F1--F2 block is the arithmetic
    mean of the two jobs. This is a Hessian *seed*, not the derivative of the
    composite inclusion--exclusion energy.
    """

    shape = (3 * int(natoms), 3 * int(natoms))
    first = np.asarray(fragment_one_oniom, dtype=float)
    second = np.asarray(fragment_two_oniom, dtype=float)
    if first.shape != shape or second.shape != shape:
        raise ValueError(f"two-job Hessians must both have shape {shape}")
    atoms_one = tuple(int(index) for index in fragment_one_atoms)
    atoms_two = tuple(int(index) for index in fragment_two_atoms)
    if set(atoms_one).intersection(atoms_two):
        raise ValueError("two-job fragment atom sets overlap")
    if set(atoms_one).union(atoms_two) != set(range(natoms)):
        raise ValueError("two-job fragment atom sets must partition the system")
    one = _cartesian_indices(atoms_one)
    two = _cartesian_indices(atoms_two)
    seed = np.zeros(shape, dtype=float)
    seed[np.ix_(one, one)] = first[np.ix_(one, one)]
    seed[np.ix_(two, two)] = second[np.ix_(two, two)]
    cross = 0.5 * (
        first[np.ix_(one, two)]
        + second[np.ix_(one, two)]
    )
    seed[np.ix_(one, two)] = cross
    seed[np.ix_(two, one)] = cross.T
    return 0.5 * (seed + seed.T)


def align_cartesian_hessian_to_reference(
    hessian_hartree_per_bohr2: np.ndarray,
    moving_coordinates_angstrom: np.ndarray,
    reference_coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Rotate a full Cartesian Hessian into a reference geometry frame."""

    from matrix_core import kabsch_rotation, rotate_cartesian_derivatives

    moving = np.asarray(moving_coordinates_angstrom, dtype=float)
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("moving and reference geometries must have equal natoms x 3 shape")
    rotation = kabsch_rotation(moving, reference)
    zero_gradient = np.zeros(moving.size, dtype=float)
    _gradient, rotated = rotate_cartesian_derivatives(
        zero_gradient,
        rotation,
        np.asarray(hessian_hartree_per_bohr2, dtype=float),
    )
    assert rotated is not None
    return rotated


def symmetric_two_oniom_geometry_seed(
    reference_coordinates_angstrom: np.ndarray,
    fragment_one_high_coordinates_angstrom: np.ndarray,
    fragment_two_high_coordinates_angstrom: np.ndarray,
    *,
    fragment_one_atoms: Sequence[int],
    fragment_two_atoms: Sequence[int],
) -> np.ndarray:
    """Combine two optimized ONIOM geometries in one common rigid frame.

    Both full geometries are first Kabsch-aligned to the reference. Internal
    shape F1 is taken from the F1-high job and shape F2 from the F2-high job.
    For each fragment, its center is the arithmetic midpoint of the two jobs
    and its rigid orientation is their SO(3) geodesic midpoint. Thus the
    intermolecular pose is averaged only after both jobs share one frame.
    """

    from matrix_core import (
        kabsch_align,
        kabsch_rotation,
        rotation_matrix_from_vector,
        rotation_vector_from_matrix,
    )

    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    first_raw = np.asarray(fragment_one_high_coordinates_angstrom, dtype=float)
    second_raw = np.asarray(fragment_two_high_coordinates_angstrom, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference coordinates must have shape natoms x 3")
    if first_raw.shape != reference.shape or second_raw.shape != reference.shape:
        raise ValueError("two ONIOM geometries must match the reference shape")
    first = kabsch_align(first_raw, reference)
    second = kabsch_align(second_raw, reference)
    groups = (
        (tuple(int(index) for index in fragment_one_atoms), first),
        (tuple(int(index) for index in fragment_two_atoms), second),
    )
    if set(groups[0][0]).intersection(groups[1][0]):
        raise ValueError("two-job fragment atom sets overlap")
    if set(groups[0][0]).union(groups[1][0]) != set(range(reference.shape[0])):
        raise ValueError("two-job fragment atom sets must partition the system")
    seed = np.zeros_like(reference)
    for atoms, selected_geometry in groups:
        indices = np.asarray(atoms, dtype=int)
        ref_fragment = reference[indices]
        first_fragment = first[indices]
        second_fragment = second[indices]
        rotation_first = kabsch_rotation(ref_fragment, first_fragment)
        rotation_second = kabsch_rotation(ref_fragment, second_fragment)
        relative = rotation_first.T @ rotation_second
        midpoint_rotation = rotation_first @ rotation_matrix_from_vector(
            0.5 * rotation_vector_from_matrix(relative)
        )
        selected_rotation = kabsch_rotation(ref_fragment, selected_geometry[indices])
        intrinsic = (
            selected_geometry[indices] - np.mean(selected_geometry[indices], axis=0)
        ) @ selected_rotation.T
        midpoint_center = 0.5 * (
            np.mean(first_fragment, axis=0) + np.mean(second_fragment, axis=0)
        )
        seed[indices] = intrinsic @ midpoint_rotation + midpoint_center
    return seed


def _cartesian_indices(atom_indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [3 * atom + component for atom in atom_indices for component in range(3)],
        dtype=int,
    )
