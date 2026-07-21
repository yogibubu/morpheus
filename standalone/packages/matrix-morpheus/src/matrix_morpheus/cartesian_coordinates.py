from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_core import principal_axis_frame
from matrix_smith.survibfit.symmetry_detector import symmetry_elements_from_geometry
from matrix_smith import irrep_characters_for_operations
from matrix_smith.runtime.gic_symmetry import (
    SYMM_INERTIA_TOL,
    SYMM_TOL,
    _cartesian_operation,
    _canonical_operation_order,
)
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_chem.symmetry import MolecularSymmetry


TOTALLY_SYMMETRIC_IRREPS = {"A1", "A", "Ag", "A'"}


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
        return np.array([irrep in TOTALLY_SYMMETRIC_IRREPS for irrep in self.irreps], dtype=bool)


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
        oriented, rotation = _oriented_coords_and_rotation(atoms, reference)
        op_data = _operation_data_for_cartesians(atoms, oriented)
        irreps = irrep_characters_for_operations([item[0] for item in op_data])
        point_group = _point_group_from_ops([item[0] for item in op_data])
        basis_oriented, basis_irreps = _symmetry_adapted_cartesian_vibrations(
            oriented,
            op_data,
            irreps,
        )
        basis_original = _rotate_cartesian_basis(basis_oriented, rotation.T)
        symmetry_source = "geometry"
    else:
        if any(len(operation.permutation) != len(atoms) for operation in symmetry.operations):
            raise ValueError("ORACLE symmetry operation size does not match Cartesian geometry")
        op_data = [
            (
                operation.label,
                np.asarray(operation.rotation, dtype=float),
                tuple(int(atom) - 1 for atom in operation.permutation),
            )
            for operation in symmetry.operations
        ]
        if not op_data:
            op_data = [("E", np.eye(3), tuple(range(len(atoms))))]
        irreps = irrep_characters_for_operations(
            [item[0] for item in op_data],
            point_group=symmetry.point_group,
            operation_matrices=tuple(item[1] for item in op_data),
        )
        point_group = symmetry.point_group
        basis_original, basis_irreps = _symmetry_adapted_cartesian_vibrations(
            reference,
            op_data,
            irreps,
        )
        symmetry_source = "oracle"
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


def _cartesian_vibrational_projector(coords_angstrom: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_angstrom, dtype=float)
    natoms = len(coords)
    centered = coords - np.mean(coords, axis=0)
    basis: list[np.ndarray] = []
    for axis in np.eye(3):
        vec = np.zeros(3 * natoms, dtype=float)
        for atom_idx in range(natoms):
            vec[3 * atom_idx : 3 * atom_idx + 3] = axis
        basis.append(vec)
    for axis in np.eye(3):
        vec = np.zeros(3 * natoms, dtype=float)
        for atom_idx, xyz in enumerate(centered):
            vec[3 * atom_idx : 3 * atom_idx + 3] = np.cross(axis, xyz)
        basis.append(vec)
    ortho = _orthonormalize_vectors(basis, tol=1.0e-10)
    if not ortho:
        return np.eye(3 * natoms, dtype=float)
    q_matrix = np.column_stack(ortho)
    return np.eye(3 * natoms, dtype=float) - q_matrix @ q_matrix.T


def _symmetry_adapted_cartesian_vibrations(
    oriented_coords_angstrom: np.ndarray,
    op_data,
    irreps: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    coords = np.asarray(oriented_coords_angstrom, dtype=float)
    natoms = len(coords)
    projector_vib = _cartesian_vibrational_projector(coords)
    if not irreps:
        basis = _basis_from_projector(projector_vib)
        return basis, tuple("A" for _ in range(basis.shape[1]))

    cart_ops = [
        _cartesian_operation(rotation, mapping, natoms) for _label, rotation, mapping in op_data
    ]
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for irrep, chars in irreps:
        symmetry_projector = np.zeros((3 * natoms, 3 * natoms), dtype=float)
        for char, op_matrix in zip(chars, cart_ops):
            symmetry_projector += float(char) * op_matrix
        symmetry_projector /= float(len(cart_ops))
        projected = projector_vib @ symmetry_projector @ projector_vib
        for vector in _basis_from_projector(projected).T:
            residual = vector.astype(float, copy=True)
            for existing in columns:
                residual -= float(existing @ residual) * existing
            norm = float(np.linalg.norm(residual))
            if norm <= 1.0e-8:
                continue
            columns.append(_canonicalize_vector_sign(residual / norm))
            labels.append(irrep)
    if not columns:
        basis = _basis_from_projector(projector_vib)
        return basis, tuple("A" for _ in range(basis.shape[1]))
    return np.column_stack(columns), tuple(labels)


def _basis_from_projector(projector: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (np.asarray(projector, dtype=float) + np.asarray(projector, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tol = (
        max(matrix.shape)
        * np.finfo(float).eps
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
        * 100.0
    )
    indices = [idx for idx, value in enumerate(eigenvalues) if value > max(tol, 1.0e-8)]
    columns = [_canonicalize_vector_sign(eigenvectors[:, idx]) for idx in indices]
    columns.sort(key=_vector_sort_key)
    return np.column_stack(columns) if columns else np.zeros((matrix.shape[0], 0), dtype=float)


def _orthonormalize_vectors(vectors: list[np.ndarray], *, tol: float) -> list[np.ndarray]:
    ortho: list[np.ndarray] = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=float).copy()
        for item in ortho:
            residual -= float(item @ residual) * item
        norm = float(np.linalg.norm(residual))
        if norm > tol:
            ortho.append(residual / norm)
    return ortho


def _canonicalize_vector_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    if not result.size:
        return result
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _vector_sort_key(vector: np.ndarray) -> tuple[int, float, tuple[float, ...]]:
    arr = np.asarray(vector, dtype=float)
    pivot = int(np.argmax(np.abs(arr))) if arr.size else 0
    return (pivot, -float(abs(arr[pivot])) if arr.size else 0.0, tuple(np.round(arr, 10)))


def _rotate_cartesian_basis(basis: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    rotated = []
    for column in np.asarray(basis, dtype=float).T:
        rotated.append(_rotate_cartesian_vector(column, rotation))
    return np.column_stack(rotated) if rotated else np.zeros_like(basis)


def _cartesian_symmetry_names(irreps: tuple[str, ...]) -> tuple[str, ...]:
    counters: dict[str, int] = {}
    names: list[str] = []
    for irrep in irreps:
        counters[irrep] = counters.get(irrep, 0) + 1
        names.append(f"{irrep}Cart{counters[irrep]:04d}")
    return tuple(names)


def _operation_data_for_cartesians(atoms: tuple[str, ...], oriented_coords: np.ndarray):
    symbols = [_canonical_symbol(atom) for atom in atoms]
    elements, _classes, permutations = symmetry_elements_from_geometry(
        symbols,
        oriented_coords,
        tol=SYMM_TOL,
        max_n=6,
        tol_H=SYMM_TOL,
        ignore_isotopes=True,
        auto_max_n=True,
        inertia_tol=SYMM_INERTIA_TOL,
    )
    unique = []
    seen = set()
    for element, permutation in zip(elements, permutations):
        mapped = tuple(int(item) for item in permutation)
        op_key = (mapped, tuple(np.round(np.asarray(element[1], dtype=float).reshape(-1), 8)))
        if op_key in seen:
            continue
        seen.add(op_key)
        unique.append((element[0], np.asarray(element[1], dtype=float), mapped))
    identity = tuple(range(len(atoms)))
    op_data = unique or [("E", np.eye(3), identity)]
    return [
        (label, rotation, mapping)
        for label, rotation, mapping, _primitive_op in _canonical_operation_order(
            [(label, rotation, mapping, ((), ())) for label, rotation, mapping in op_data]
        )
    ]


def _oriented_coords_and_rotation(
    atoms: tuple[str, ...], coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([atomic_number(atom) for atom in atoms], dtype=float)
    return principal_axis_frame(coords, weights)


def _rotate_cartesian_vector(vector: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.asarray(vector, dtype=float).reshape((-1, 3)) @ rotation).reshape(-1)


def _point_group_from_ops(labels: list[str]) -> str:
    if len(labels) == 1:
        return "C1"
    if len(labels) == 2:
        if any(label.startswith("sigma") for label in labels):
            return "Cs"
        return "C2"
    if (
        len(labels) == 4
        and any(label.startswith("C2") for label in labels)
        and sum(label.startswith("sigma") for label in labels) == 2
    ):
        return "C2v"
    return "UNKNOWN"


def _canonical_symbol(symbol: str) -> str:
    return atomic_symbol(atomic_number(str(symbol).strip()))
