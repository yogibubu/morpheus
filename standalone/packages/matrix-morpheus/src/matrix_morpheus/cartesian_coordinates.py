from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_chem import (
    MolecularGeometry,
    MolecularSymmetry,
    analyze_molecular_symmetry,
)
from matrix_smith import is_total_symmetric_irrep, symmetry_adapted_cartesian_basis


_GEOMETRY_SYMMETRY_DISTANCE_TOLERANCE_ANGSTROM = 1.0e-2
_GEOMETRY_SYMMETRY_INERTIA_RELATIVE_TOLERANCE = 1.0e-3
_GEOMETRY_SYMMETRY_MAX_ROTATION_ORDER = 6


@dataclass(frozen=True)
class CartesianCoordinateModel:
    """Cartesian displacement basis used as SEfit working coordinates."""

    reference_coordinates_angstrom: np.ndarray
    cartesian_from_q: np.ndarray
    labels: tuple[str, ...]
    names: tuple[str, ...]
    irreps: tuple[str, ...]
    frequencies_cm: np.ndarray
    eigenvalues: np.ndarray
    point_group: str
    model_kind: str = "cartesian_symmetry"

    def values(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        delta = np.asarray(coordinates_angstrom, dtype=float).reshape(
            -1
        ) - self.reference_coordinates_angstrom.reshape(-1)
        return np.linalg.pinv(self.cartesian_from_q, rcond=1.0e-10) @ delta

    @property
    def active_totally_symmetric_mask(self) -> np.ndarray:
        return np.array(
            [is_total_symmetric_irrep(self.point_group, irrep) for irrep in self.irreps],
            dtype=bool,
        )


def cartesian_symmetry_coordinate_model(
    atoms: tuple[str, ...],
    reference_coordinates_angstrom: np.ndarray,
    *,
    symmetry: MolecularSymmetry | None = None,
) -> CartesianCoordinateModel:
    """Build a Hessian-free symmetry-adapted Cartesian SEfit coordinate model."""
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if reference.shape != (len(atoms), 3):
        raise ValueError("Cartesian-symmetry reference geometry has inconsistent dimensions")
    if symmetry is None:
        symmetry = analyze_molecular_symmetry(
            MolecularGeometry(atoms=atoms, coordinates_angstrom=reference),
            distance_tolerance=_GEOMETRY_SYMMETRY_DISTANCE_TOLERANCE_ANGSTROM,
            inertia_tolerance=_GEOMETRY_SYMMETRY_INERTIA_RELATIVE_TOLERANCE,
            max_rotation_order=_GEOMETRY_SYMMETRY_MAX_ROTATION_ORDER,
        )
        symmetry_source = "geometry"
    else:
        if any(len(operation.permutation) != len(atoms) for operation in symmetry.operations):
            raise ValueError("ORACLE symmetry operation size does not match Cartesian geometry")
        symmetry_source = "oracle"
    shared_basis = symmetry_adapted_cartesian_basis(
        atoms,
        reference,
        symmetry=symmetry,
        frame_axes_global=np.asarray(symmetry.orientation, dtype=float).T,
    )
    basis_original = shared_basis.cartesian_from_q
    basis_irreps = shared_basis.irreps
    point_group = shared_basis.point_group
    names = _cartesian_symmetry_names(basis_irreps)
    labels = tuple(
        f"SC{idx:03d} SymmetryCartesian {name} irrep={irrep} source={symmetry_source}"
        for idx, (name, irrep) in enumerate(zip(names, basis_irreps), start=1)
    )
    zeros = np.zeros((basis_original.shape[1],), dtype=float)
    return CartesianCoordinateModel(
        reference_coordinates_angstrom=reference.copy(),
        cartesian_from_q=basis_original,
        labels=labels,
        names=names,
        irreps=basis_irreps,
        frequencies_cm=zeros.copy(),
        eigenvalues=zeros.copy(),
        point_group=point_group,
    )


def _cartesian_symmetry_names(irreps: tuple[str, ...]) -> tuple[str, ...]:
    counters: dict[str, int] = {}
    names: list[str] = []
    for irrep in irreps:
        counters[irrep] = counters.get(irrep, 0) + 1
        names.append(f"{irrep}Cart{counters[irrep]:04d}")
    return tuple(names)
