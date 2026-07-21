"""Molecular point-group detection and serialized symmetry operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
import re

import numpy as np

from .geometry import MolecularGeometry
from .topology.elements import atomic_number


CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM = 5.0e-6


@dataclass(frozen=True)
class SymmetryOperation:
    label: str
    rotation: tuple[tuple[float, float, float], ...]
    permutation: tuple[int, ...]
    max_deviation: float


@dataclass(frozen=True)
class MolecularSymmetry:
    point_group: str
    operations: tuple[SymmetryOperation, ...]
    atom_classes: tuple[tuple[int, ...], ...]
    max_deviation: float
    mean_deviation: float
    orientation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class SymmetryProjectionDiagnostics:
    status: str
    max_displacement_angstrom: float
    rms_displacement_angstrom: float
    noop_tolerance_angstrom: float = CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM


def symmetry_projection_diagnostics(
    original: MolecularGeometry,
    projected: MolecularGeometry,
    symmetry: MolecularSymmetry,
) -> SymmetryProjectionDiagnostics:
    displacement = np.linalg.norm(
        np.asarray(projected.coordinates_angstrom, dtype=float)
        - np.asarray(original.coordinates_angstrom, dtype=float),
        axis=1,
    )
    max_displacement = float(np.max(displacement)) if len(displacement) else 0.0
    rms_displacement = (
        float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0
    )
    point_group = symmetry.point_group.strip().upper()
    expected_order = _finite_point_group_order(point_group)
    if point_group in {"", "C1", "UNKNOWN"}:
        status = "NOT_APPLICABLE_C1"
    elif expected_order is not None and len(symmetry.operations) != expected_order:
        status = "SKIPPED_INCOMPLETE_GROUP"
    elif symmetry.max_deviation <= CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM:
        status = "NOT_REQUIRED_NUMERICAL_NOISE"
    else:
        status = "APPLIED"
    return SymmetryProjectionDiagnostics(
        status=status,
        max_displacement_angstrom=max_displacement,
        rms_displacement_angstrom=rms_displacement,
    )


def symmetrize_molecular_geometry(
    geometry: MolecularGeometry,
    symmetry: MolecularSymmetry,
    *,
    minimum_deviation_angstrom: float = CARTESIAN_SYMMETRIZATION_NOOP_TOLERANCE_ANGSTROM,
) -> MolecularGeometry:
    """Return the exact group-average geometry for an assigned point group."""
    point_group = symmetry.point_group.strip().upper()
    if point_group in {"", "C1", "UNKNOWN"}:
        return geometry
    expected_order = _finite_point_group_order(point_group)
    if expected_order is not None and len(symmetry.operations) != expected_order:
        # A group average is a projector only when it contains the complete
        # group.  Averaging an incomplete operation set makes the Cartesian
        # result depend on the platform-specific choice inside a nearly
        # degenerate inertia eigenspace (the historical spiro/D2 regression).
        return geometry
    if symmetry.max_deviation <= float(minimum_deviation_angstrom):
        # Do not turn few-microangstrom input/printing noise into a new
        # Cartesian reference.  Above this floor, accepted quasi-symmetry is
        # projected exactly by the complete finite-group average below.
        return geometry
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    weights = np.array([atomic_number(symbol) or 1 for symbol in geometry.atoms], dtype=float)
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    projected = np.zeros_like(centered)
    for operation in symmetry.operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        mapping = np.asarray([int(atom) - 1 for atom in operation.permutation], dtype=int)
        inverse_mapping = np.argsort(mapping)
        projected += centered[inverse_mapping] @ rotation
    projected /= float(len(symmetry.operations))
    exact = projected + center
    exact[np.abs(exact) < 5.0e-13] = 0.0
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=exact,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={**geometry.metadata, "cartesian_symmetrized_point_group": symmetry.point_group},
    )


def _finite_point_group_order(point_group: str) -> int | None:
    label = point_group.strip().upper()
    if label == "CI" or label == "CS":
        return 2
    match = re.fullmatch(r"C(\d+)", label)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"C(\d+)[VH]", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)[DH]", label)
    if match:
        return 4 * int(match.group(1))
    if label == "T":
        return 12
    if label in {"TD", "TH"}:
        return 24
    if label == "O":
        return 24
    if label == "OH":
        return 48
    if label == "I":
        return 60
    if label == "IH":
        return 120
    return None


def analyze_molecular_symmetry(
    geometry: MolecularGeometry,
    *,
    distance_tolerance: float,
    inertia_tolerance: float,
    max_rotation_order: int,
) -> MolecularSymmetry:
    symbols = list(geometry.atoms)
    weights = np.array([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    center = np.sum(coordinates * weights[:, None], axis=0) / max(
        float(np.sum(weights)), 1.0e-12
    )
    centered = coordinates - center
    if _fully_degenerate_inertia(symbols, centered, tolerance=inertia_tolerance):
        # A spherical top has no physical principal-axis frame.  Retaining the
        # input Cartesian frame avoids a platform-dependent eigenbasis; the
        # point-set operation enumeration below is covariant under a rigid
        # rotation of this frame.
        oriented = centered
        frame = np.eye(3, dtype=float)
    elif _has_degenerate_inertia(symbols, centered, tolerance=inertia_tolerance):
        # A symmetric top has a physically unique axis but no unique basis in
        # the perpendicular plane.  The legacy eigensolver frame can therefore
        # turn a complete Dnh group into an equally self-consistent Cnh
        # subgroup after a rigid rotation.  Anchor the degenerate plane to the
        # ordered molecular point set before enumerating operations.
        oriented, frame = orient_coords_with_frame(
            geometry.coordinates_angstrom,
            weights=weights,
            degeneracy_tolerance=inertia_tolerance,
        )
    else:
        oriented, frame = _legacy_orient_coords_with_frame(
            coordinates,
            weights=weights,
        )
    elements, atom_classes, permutations = symmetry_elements_from_geometry(
        symbols,
        oriented,
        tol=distance_tolerance,
        max_n=max_rotation_order,
        tol_H=distance_tolerance,
        ignore_isotopes=True,
        auto_max_n=True,
        inertia_tol=inertia_tolerance,
    )
    preliminary_group = group_label(
        elements,
        linear=is_linear(oriented, tol=distance_tolerance),
    )
    expected_order = _finite_point_group_order(preliminary_group)
    if expected_order is not None and len(elements) != expected_order:
        # The historical principal-axis convention preserves established
        # operation labels and SALC snapshots whenever it yields a complete
        # group.  Only a genuinely incomplete result activates the covariant
        # construction for a degenerate inertia subspace.
        oriented, frame = orient_coords_with_frame(
            geometry.coordinates_angstrom,
            weights=weights,
            degeneracy_tolerance=inertia_tolerance,
        )
        elements, atom_classes, permutations = symmetry_elements_from_geometry(
            symbols,
            oriented,
            tol=distance_tolerance,
            max_n=max_rotation_order,
            tol_H=distance_tolerance,
            ignore_isotopes=True,
            auto_max_n=True,
            inertia_tol=inertia_tolerance,
        )
    if not elements:
        elements = [("E", np.eye(3), 0.0)]
        permutations = [tuple(range(len(symbols)))]
        atom_classes = tuple((idx,) for idx in range(len(symbols)))
    point_group = group_label(elements, linear=is_linear(oriented, tol=distance_tolerance))
    operations = tuple(
        SymmetryOperation(
            label=str(label),
            # Persist operations in the original Cartesian frame.  Consumers
            # must not need the arbitrary principal-axis frame to apply the
            # stored matrix to the enriched-XYZ coordinates.
            rotation=tuple(
                tuple(float(value) for value in row)
                for row in frame @ np.asarray(rotation, dtype=float) @ frame.T
            ),
            permutation=tuple(int(item) + 1 for item in permutation),
            max_deviation=float(max_deviation),
        )
        for (label, rotation, max_deviation), permutation in zip(elements, permutations)
    )
    return MolecularSymmetry(
        point_group=point_group,
        operations=operations,
        atom_classes=tuple(tuple(int(atom) + 1 for atom in cls) for cls in atom_classes),
        max_deviation=float(max((op.max_deviation for op in operations), default=0.0)),
        mean_deviation=(
            float(np.mean([op.max_deviation for op in operations])) if operations else 0.0
        ),
        orientation=tuple(tuple(float(value) for value in row) for row in frame),
    )


def symmetry_section_lines(
    symmetry: MolecularSymmetry,
    *,
    thresholds,
    projection: SymmetryProjectionDiagnostics | None = None,
) -> list[str]:
    lines = [
        "SCHEMA oracle.xyz.symmetry.v1",
        f"POINT_GROUP {symmetry.point_group}",
        f"OPERATION_COUNT {len(symmetry.operations)}",
        f"MAX_OPERATION_DEVIATION_ANGSTROM {symmetry.max_deviation:.12g}",
        f"MEAN_OPERATION_DEVIATION_ANGSTROM {symmetry.mean_deviation:.12g}",
        f"THRESHOLD_DISTANCE_ANGSTROM {thresholds.distance_angstrom:.12g}",
        f"THRESHOLD_INERTIA_RELATIVE {thresholds.inertia_relative:.12g}",
        f"MAX_ROTATION_ORDER {thresholds.max_rotation_order}",
    ]
    if projection is not None:
        lines.extend(
            (
                f"CARTESIAN_PROJECTION_STATUS {projection.status}",
                "CARTESIAN_PROJECTION_MAX_DISPLACEMENT_ANGSTROM "
                f"{projection.max_displacement_angstrom:.12g}",
                "CARTESIAN_PROJECTION_RMS_DISPLACEMENT_ANGSTROM "
                f"{projection.rms_displacement_angstrom:.12g}",
                "CARTESIAN_PROJECTION_NOOP_TOLERANCE_ANGSTROM "
                f"{projection.noop_tolerance_angstrom:.12g}",
            )
        )
    lines.append("[OPERATIONS]")
    for idx, operation in enumerate(symmetry.operations, start=1):
        matrix = ",".join(f"{value:.12g}" for row in operation.rotation for value in row)
        permutation = ",".join(str(atom) for atom in operation.permutation)
        lines.append(
            f"{idx} LABEL={operation.label} "
            f"MAX_DEVIATION={operation.max_deviation:.12g} "
            f"PERMUTATION={permutation} MATRIX={matrix}"
        )
    lines.append("[ATOM_CLASSES]")
    if symmetry.atom_classes:
        for idx, atoms in enumerate(symmetry.atom_classes, start=1):
            lines.append(f"{idx} ATOMS=" + ",".join(str(atom) for atom in atoms))
    else:
        lines.append("NONE")
    return lines


def orient_coords(coords, weights=None):
    oriented, _frame = orient_coords_with_frame(coords, weights=weights)
    return oriented


def _legacy_orient_coords_with_frame(coords, weights=None):
    """Return the established principal-axis frame used by frozen snapshots."""
    x = np.asarray(coords, dtype=float)
    w = np.ones(len(x), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    center = np.sum(x * w[:, None], axis=0) / max(float(np.sum(w)), 1.0e-12)
    x = x - center
    squared_radii = np.einsum("ij,ij->i", x, x)
    inertia = float(w @ squared_radii) * np.eye(3) - np.einsum("i,ij,ik->jk", w, x, x)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    frame = eigenvectors[:, np.argsort(eigenvalues)]
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return x @ frame, frame


def orient_coords_with_frame(coords, weights=None, degeneracy_tolerance=1.0e-8):
    x = np.array(coords, dtype=float)
    w = np.ones(len(x), dtype=float) if weights is None else np.array(weights, dtype=float)
    center = np.sum(x * w[:, None], axis=0) / max(float(np.sum(w)), 1.0e-12)
    x = x - center
    squared_radii = np.einsum("ij,ij->i", x, x)
    inertia = float(w @ squared_radii) * np.eye(3) - np.einsum("i,ij,ik->jk", w, x, x)
    evals, evecs = np.linalg.eigh(inertia)
    order = np.argsort(evals)
    evals = evals[order]
    frame = evecs[:, order]
    scale = max(float(np.max(np.abs(evals))), 1.0e-12)
    gap01 = abs(float(evals[0] - evals[1])) / scale
    gap12 = abs(float(evals[1] - evals[2])) / scale
    tolerance = float(degeneracy_tolerance)
    if (gap01 <= tolerance) != (gap12 <= tolerance):
        # A symmetric top has one unique inertia axis and an arbitrary basis
        # in the perpendicular degenerate plane.  Anchor that plane to an atom
        # vector, making the frame covariant under rigid rotations.
        unique_index = 2 if gap01 <= tolerance else 0
        axis = np.asarray(frame[:, unique_index], dtype=float)
        projections = x - np.outer(x @ axis, axis)
        norms = np.linalg.norm(projections, axis=1)
        nonzero = np.flatnonzero(norms > 1.0e-12)
        anchor_index = int(nonzero[0]) if len(nonzero) else 0
        if len(nonzero):
            # Atom order is already part of the molecular contract.  Using
            # its first off-axis vector is deterministic in a degenerate
            # plane and, unlike a largest-radius choice, remains stable when
            # several symmetry-equivalent atoms tie numerically.  Treat the
            # anchor as y so conventional axis-aligned planar inputs retain
            # their historical x/y irrep naming.
            ey = projections[anchor_index] / norms[anchor_index]
            ez = axis
            ex = np.cross(ey, ez)
            ex /= max(float(np.linalg.norm(ex)), 1.0e-12)
            ey = np.cross(ez, ex)
            ey /= max(float(np.linalg.norm(ey)), 1.0e-12)
            frame = np.column_stack((ex, ey, ez))
    elif gap01 > tolerance and gap12 > tolerance:
        # Fix eigenvector signs from the first nonzero atomic projection.  This
        # removes platform-dependent sign flips for asymmetric tops.
        frame = frame.copy()
        for column in range(3):
            projections = x @ frame[:, column]
            nonzero = np.flatnonzero(np.abs(projections) > 1.0e-12)
            if len(nonzero) and projections[int(nonzero[0])] < 0.0:
                frame[:, column] *= -1.0
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return x @ frame, frame


def is_linear(coords, tol=1.0e-3):
    x = np.array(coords, dtype=float)
    inertia = np.zeros((3, 3), dtype=float)
    for vec in x:
        inertia += np.dot(vec, vec) * np.eye(3) - np.outer(vec, vec)
    return bool(np.linalg.eigvalsh(inertia)[0] < tol)


def symmetry_elements_from_geometry(
    symbols,
    coords_oriented,
    tol=1.0e-3,
    max_n=6,
    tol_H=None,
    ignore_isotopes=False,
    auto_max_n=False,
    inertia_tol=1.0e-3,
):
    coords = np.asarray(coords_oriented, dtype=float)
    fully_degenerate = _fully_degenerate_inertia(symbols, coords, tolerance=inertia_tol)
    elements, atom_classes, permutations = _symmetry_elements_in_frame(
        symbols,
        coords,
        tol=tol,
        max_n=max_n,
        tol_H=tol_H,
        ignore_isotopes=ignore_isotopes,
        auto_max_n=auto_max_n,
        inertia_tol=inertia_tol,
    )
    nmax, axis = _highest_cn_axis([element[0] for element in elements])
    if nmax >= 3 and axis in {"x", "y"}:
        alignment = _axis_to_z_alignment(axis)
        aligned_elements, aligned_classes, aligned_permutations = _symmetry_elements_in_frame(
            symbols,
            coords @ alignment,
            tol=tol,
            max_n=max_n,
            tol_H=tol_H,
            ignore_isotopes=ignore_isotopes,
            auto_max_n=auto_max_n,
            inertia_tol=inertia_tol,
        )
        aligned_nmax, _aligned_axis = _highest_cn_axis([element[0] for element in aligned_elements])
        if aligned_nmax >= nmax and len(aligned_elements) >= len(elements):
            elements = tuple(
                (
                    label,
                    alignment @ np.asarray(rotation, dtype=float) @ alignment.T,
                    max_deviation,
                )
                for label, rotation, max_deviation in aligned_elements
            )
            atom_classes = aligned_classes
            permutations = aligned_permutations
    if fully_degenerate:
        direct = _symmetry_elements_from_point_set(
            symbols,
            coords,
            tol=tol,
            tol_H=tol_H,
            ignore_isotopes=ignore_isotopes,
        )
        # A spherical top has no canonical inertia frame.  Prefer point-set
        # enumeration whenever it recovers at least the same group so a
        # platform-dependent eigenbasis cannot select a different, but equally
        # complete, operation contract.
        if len(direct[0]) >= len(elements):
            return direct
    return elements, atom_classes, permutations


def _fully_degenerate_inertia(symbols, coords: np.ndarray, *, tolerance: float) -> bool:
    if len(coords) < 4:
        return False
    weights = np.asarray([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    points = np.asarray(coords, dtype=float)
    squared_radii = np.einsum("ij,ij->i", points, points)
    inertia = float(weights @ squared_radii) * np.eye(3) - np.einsum(
        "i,ij,ik->jk", weights, points, points
    )
    eigenvalues = np.linalg.eigvalsh(inertia)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-12)
    return bool(float(np.ptp(eigenvalues)) / scale <= float(tolerance))


def _has_degenerate_inertia(symbols, coords: np.ndarray, *, tolerance: float) -> bool:
    """Return whether any principal-moment pair is degenerate."""
    if len(coords) < 3:
        return False
    weights = np.asarray([atomic_number(symbol) or 1 for symbol in symbols], dtype=float)
    points = np.asarray(coords, dtype=float)
    squared_radii = np.einsum("ij,ij->i", points, points)
    inertia = float(weights @ squared_radii) * np.eye(3) - np.einsum(
        "i,ij,ik->jk", weights, points, points
    )
    eigenvalues = np.linalg.eigvalsh(inertia)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-12)
    gaps = np.abs(np.diff(eigenvalues)) / scale
    return bool(np.any(gaps <= float(tolerance)))


def _symmetry_elements_from_point_set(
    symbols,
    coords: np.ndarray,
    *,
    tol: float,
    tol_H: float | None,
    ignore_isotopes: bool,
):
    """Recover all point-set isometries without choosing principal axes.

    A non-coplanar source triple fixes an orthogonal transformation uniquely.
    Candidate image triples are pruned by element, radius and Gram matrix, so
    the expensive full-geometry match is performed only for viable isometries.
    This path is used for spherical tops, whose inertia eigenvectors carry no
    orientation information.
    """
    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        return (), (), ()
    source_indices = _independent_point_triple(points)
    if source_indices is None:
        return (), (), ()
    source = points[list(source_indices)]
    source_symbols = [
        _symmetry_symbol_key(symbols[index], ignore_isotopes=ignore_isotopes)
        for index in source_indices
    ]
    match_symbols = [
        _symmetry_symbol_key(symbol, ignore_isotopes=ignore_isotopes) for symbol in symbols
    ]
    radii = np.linalg.norm(points, axis=1)
    max_radius = max(float(np.max(radii)), 1.0)
    candidates: list[tuple[int, ...]] = []
    for source_index, source_symbol in zip(source_indices, source_symbols):
        radius_tolerance = tol_H if tol_H is not None and source_symbol in {"H", "Z1"} else tol
        candidates.append(
            tuple(
                index
                for index, symbol in enumerate(match_symbols)
                if symbol == source_symbol
                and abs(float(radii[index] - radii[source_index])) <= float(radius_tolerance)
            )
        )
    gram = source @ source.T
    gram_tolerance = max(1.0e-10, 4.0 * max_radius * float(max(tol, tol_H or tol)))
    operations: list[tuple[str, np.ndarray, float]] = []
    mappings: list[tuple[int, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for target_indices in product(*candidates):
        if len(set(target_indices)) != 3:
            continue
        target = points[list(target_indices)]
        if not np.allclose(target @ target.T, gram, atol=gram_tolerance, rtol=0.0):
            continue
        row_action = np.linalg.solve(source, target)
        rotation = row_action.T
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8, rtol=0.0):
            continue
        mapping, max_deviation = _match_with_map(
            match_symbols,
            points,
            points @ rotation.T,
            tol,
            tol_H=tol_H,
        )
        if mapping is None:
            continue
        key = tuple(round(float(value), 10) for value in rotation.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        operations.append(("", rotation, float(max_deviation)))
        mappings.append(tuple(int(item) for item in mapping))
    ordered = sorted(
        zip(operations, mappings),
        key=lambda item: _automatic_operation_sort_key(item[0][1]),
    )
    counters: Counter[str] = Counter()
    labeled_operations: list[tuple[str, np.ndarray, float]] = []
    ordered_mappings: list[tuple[int, ...]] = []
    for (_label, rotation, max_deviation), mapping in ordered:
        base = _automatic_operation_label(rotation)
        if base not in {"E", "i"}:
            counters[base] += 1
            label = f"{base}_{counters[base]}"
        else:
            label = base
        labeled_operations.append((label, rotation, max_deviation))
        ordered_mappings.append(mapping)
    return (
        tuple(labeled_operations),
        _atom_classes(len(symbols), ordered_mappings),
        tuple(ordered_mappings),
    )


def _independent_point_triple(points: np.ndarray) -> tuple[int, int, int] | None:
    radii = np.linalg.norm(points, axis=1)
    first = int(np.argmax(radii))
    cross_norms = np.linalg.norm(np.cross(points[first], points), axis=1)
    second = int(np.argmax(cross_norms))
    if cross_norms[second] <= 1.0e-10:
        return None
    determinants = np.abs(points @ np.cross(points[first], points[second]))
    third = int(np.argmax(determinants))
    if determinants[third] <= 1.0e-10:
        return None
    return first, second, third


def _automatic_operation_sort_key(matrix: np.ndarray) -> tuple[int, float, tuple[float, ...]]:
    arr = np.asarray(matrix, dtype=float)
    identity_rank = 0 if np.allclose(arr, np.eye(3), atol=1.0e-8) else 1
    return identity_rank, -float(np.linalg.det(arr)), tuple(np.round(arr.reshape(-1), 10))


def _automatic_operation_label(matrix: np.ndarray) -> str:
    arr = np.asarray(matrix, dtype=float)
    if np.allclose(arr, np.eye(3), atol=1.0e-8):
        return "E"
    if np.allclose(arr, -np.eye(3), atol=1.0e-8):
        return "i"
    determinant = float(np.linalg.det(arr))
    proper = arr if determinant > 0.0 else -arr
    cosine = float(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    order = 2 if angle <= 1.0e-10 else max(2, int(round(2.0 * np.pi / angle)))
    if determinant > 0.0:
        return f"C{order}_auto"
    if abs(float(np.trace(arr)) - 1.0) <= 1.0e-8:
        return "sigma_auto"
    return f"iC{order}_auto"


def _symmetry_symbol_key(symbol, *, ignore_isotopes: bool) -> str:
    text = str(symbol).strip()
    if not ignore_isotopes:
        return text
    number = atomic_number(text)
    return f"Z{number}" if number is not None else re.sub(r"^\d+|\d+$", "", text).capitalize()


def _symmetry_elements_in_frame(
    symbols,
    coords: np.ndarray,
    *,
    tol=1.0e-3,
    max_n=6,
    tol_H=None,
    ignore_isotopes=False,
    auto_max_n=False,
    inertia_tol=1.0e-3,
):
    radii = np.linalg.norm(coords, axis=1)
    max_radius = float(np.max(radii)) if len(radii) else 1.0
    if max_radius <= 0.0:
        max_radius = 1.0
    scaled = coords / max_radius
    # The public ORACLE threshold is an absolute Cartesian distance in
    # angstrom.  Candidate matching is performed on radius-normalized
    # coordinates, so both the acceptance tolerance and the reported residual
    # must be converted explicitly rather than silently becoming relative to
    # molecular size.
    scaled_tol = float(tol) / max_radius
    scaled_tol_H = None if tol_H is None else float(tol_H) / max_radius
    sym_use = [_symmetry_symbol_key(symbol, ignore_isotopes=ignore_isotopes) for symbol in symbols]
    if auto_max_n:
        inertia = np.zeros((3, 3), dtype=float)
        for vec in coords:
            inertia += np.dot(vec, vec) * np.eye(3) - np.outer(vec, vec)
        evals = np.linalg.eigvalsh(inertia)
        max_inertia = float(np.max(evals)) if len(evals) else 0.0
        if max_inertia > 0.0:
            d01 = abs(evals[0] - evals[1]) / max_inertia
            d12 = abs(evals[1] - evals[2]) / max_inertia
            if d01 > inertia_tol and d12 > inertia_tol:
                max_n = min(max_n, 2)
    elements = []
    permutations = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    for label, rotation in candidate_ops(max_n=max_n):
        mapped, max_dev = _match_with_map(
            sym_use,
            scaled,
            scaled @ rotation.T,
            scaled_tol,
            tol_H=scaled_tol_H,
        )
        if mapped is not None:
            unique_key = (
                tuple(int(item) for item in mapped),
                tuple(round(float(value), 10) for value in rotation.reshape(-1)),
            )
            if unique_key in seen:
                continue
            seen.add(unique_key)
            elements.append((label, rotation, float(max_dev) * max_radius))
            permutations.append(tuple(mapped))
    return elements, _atom_classes(len(symbols), permutations), permutations


def group_label(elements, linear=False):
    labels = [item[0] for item in elements]
    polyhedral = _polyhedral_group_label(elements)
    if polyhedral:
        return polyhedral
    nmax, axis = _highest_cn_axis(labels)
    has_i = "i" in labels
    has_sigma = any(label.startswith("sigma") for label in labels)
    has_c2 = any(label.startswith("C2") for label in labels)
    if linear:
        return "Dinfh" if has_i else "Cinfv"
    if nmax >= 2:
        sigma_h = {"x": "sigma_yz", "y": "sigma_xz", "z": "sigma_xy"}.get(axis or "z")
        has_sigma_h = sigma_h in labels
        has_sigma_v = has_sigma and not has_sigma_h
        if has_sigma_h and has_c2:
            return f"D{nmax}h"
        if has_sigma_h:
            return f"C{nmax}h"
        # Check Dnd only after an explicit horizontal plane.  High-order Dnh
        # groups also contain diagonal C2 operations which can match the Dnd
        # label pattern when a degenerate inertia plane is anchored to a
        # different, but equivalent, molecular direction.
        dnd_n = _dnd_group_order(labels, principal_order=nmax)
        if dnd_n:
            return f"D{dnd_n}d"
        if has_sigma_v:
            return f"C{nmax}v"
        if has_c2:
            return f"D{nmax}"
        return f"C{nmax}"
    if has_i:
        return "Ci"
    if has_sigma:
        return "Cs"
    return "C1"


def _match_with_map(symbols, coords1, coords2, tol, tol_H=None):
    used = np.zeros(len(coords2), dtype=bool)
    mapping = [-1] * len(coords1)
    by_symbol: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        by_symbol.setdefault(symbol, []).append(idx)
    points1 = np.asarray(coords1, dtype=float)
    points2 = np.asarray(coords2, dtype=float)
    radii1 = np.linalg.norm(points1, axis=1)
    radii2 = np.linalg.norm(points2, axis=1)
    max_dev = 0.0
    for idx in sorted(range(len(coords1)), key=lambda item: (len(by_symbol[symbols[item]]), item)):
        eff_tol = tol_H if tol_H is not None and symbols[idx] in {"H", "Z1"} else tol
        radius = float(radii1[idx])
        symbol_candidates = np.asarray(by_symbol.get(symbols[idx], ()), dtype=int)
        if len(symbol_candidates):
            available = symbol_candidates[
                (~used[symbol_candidates]) & (np.abs(radii2[symbol_candidates] - radius) <= eff_tol)
            ]
        else:
            available = symbol_candidates
        if len(available):
            order = np.argsort(np.abs(radii2[available] - radius), kind="stable")
            available = available[order]
            deviations = np.linalg.norm(points2[available] - points1[idx], axis=1)
            accepted = np.flatnonzero(deviations < eff_tol)
            if len(accepted):
                selected = int(accepted[0])
                cand = int(available[selected])
                deviation = float(deviations[selected])
                mapping[idx] = cand
                used[cand] = True
                max_dev = max(max_dev, deviation)
        if mapping[idx] < 0:
            return None, None
    return tuple(mapping), max_dev


def _atom_classes(natoms: int, permutations) -> tuple[tuple[int, ...], ...]:
    parent = list(range(natoms))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for permutation in permutations:
        for left, right in enumerate(permutation):
            union(left, int(right))
    classes: dict[int, list[int]] = {}
    for atom in range(natoms):
        classes.setdefault(find(atom), []).append(atom)
    return tuple(tuple(values) for values in classes.values())


def _highest_cn_axis(labels):
    best = 1
    axis = None
    for label in labels:
        match = re.match(r"C(\d+)([xyz])", str(label))
        if match and int(match.group(1)) > best:
            best = int(match.group(1))
            axis = match.group(2)
    return best, axis


def _polyhedral_group_label(elements) -> str | None:
    matrices = [np.asarray(item[1], dtype=float) for item in elements]
    td_classes = Counter(
        operation_class
        for operation_class in (_td_candidate_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if td_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C2": 3,
            "S4": 6,
            "sigma_d": 6,
        }
    ):
        return "Td"
    invariant_oh_classes = Counter(
        operation_class
        for operation_class in (_orientation_invariant_oh_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if invariant_oh_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C4": 6,
            "C2": 9,
            "i*E": 1,
            "i*C3": 8,
            "i*C4": 6,
            "i*C2": 9,
        }
    ):
        return "Oh"
    oh_classes = Counter(
        operation_class
        for operation_class in (_polyhedral_matrix_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if oh_classes == Counter(
        {
            "E": 1,
            "C3": 8,
            "C2_axis": 3,
            "C4": 6,
            "C2_edge": 6,
            "i*E": 1,
            "i*C3": 8,
            "i*C2_axis": 3,
            "i*C4": 6,
            "i*C2_edge": 6,
        }
    ):
        return "Oh"
    ih_classes = Counter(
        operation_class
        for operation_class in (_icosahedral_candidate_class(matrix) for matrix in matrices)
        if operation_class is not None
    )
    if ih_classes == Counter(
        {
            "E": 1,
            "C2": 15,
            "C3": 20,
            "C5": 12,
            "C5_2": 12,
            "i*E": 1,
            "i*C2": 15,
            "i*C3": 20,
            "i*C5": 12,
            "i*C5_2": 12,
        }
    ):
        return "Ih"
    if ih_classes == Counter({"E": 1, "C2": 15, "C3": 20, "C5": 12, "C5_2": 12}):
        return "I"
    return None


def _orientation_invariant_oh_class(matrix: np.ndarray) -> str | None:
    arr = np.asarray(matrix, dtype=float)
    determinant = float(np.linalg.det(arr))
    proper = arr if determinant > 0.0 else -arr
    trace = float(np.trace(proper))
    if abs(trace - 3.0) <= 1.0e-8:
        label = "E"
    elif abs(trace) <= 1.0e-8:
        label = "C3"
    elif abs(trace - 1.0) <= 1.0e-8:
        label = "C4"
    elif abs(trace + 1.0) <= 1.0e-8:
        label = "C2"
    else:
        return None
    return f"i*{label}" if determinant < 0.0 else label


def _polyhedral_matrix_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    det = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))
    if det > 0.0:
        if abs(trace) <= 1.0e-8:
            return "C3"
        if abs(trace - 1.0) <= 1.0e-8:
            return "C4"
        if abs(trace + 1.0) <= 1.0e-8:
            return "C2_axis" if _is_coordinate_axis_c2(matrix) else "C2_edge"
    if det < 0.0:
        if np.allclose(matrix, -np.eye(3), atol=1.0e-8):
            return "i*E"
        proper = -matrix
        proper_class = _polyhedral_matrix_class(proper)
        if proper_class is not None:
            return f"i*{proper_class}"
        if abs(trace - 1.0) <= 1.0e-8:
            return "sigma_d"
        if abs(trace + 1.0) <= 1.0e-8:
            return "S4"
    return None


def _is_coordinate_axis_c2(matrix: np.ndarray) -> bool:
    rounded = np.rint(matrix)
    if not np.allclose(matrix, rounded, atol=1.0e-8):
        return False
    return bool(np.count_nonzero(np.abs(np.diag(rounded)) > 0.5) == 3)


def _dnd_group_order(labels: list[str], *, principal_order: int) -> int | None:
    orders = []
    for label in labels:
        for pattern in (r"Dnd_C(\d+)z\^", r"Dnd_C2_xy_(\d+)_"):
            match = re.search(pattern, str(label))
            if match:
                order = int(match.group(1)) // 2
                if order == principal_order:
                    orders.append(order)
        match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", str(label))
        if match:
            operation_order, index = (int(item) for item in match.groups())
            order = operation_order // 2
            if operation_order == 2 * principal_order and index % 2 == 1:
                orders.append(order)
        match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", str(label))
        if match:
            operation_order, power = (int(item) for item in match.groups())
            order = operation_order // 2
            if operation_order == 2 * principal_order and power % 2 == 1:
                orders.append(order)
    return max(orders) if orders else None


def _axis_to_z_alignment(axis: str) -> np.ndarray:
    if axis == "x":
        return np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    if axis == "y":
        return np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
    return np.eye(3)


@lru_cache(maxsize=16)
def candidate_ops(max_n=6):
    ops = [("E", np.eye(3)), ("i", -np.eye(3))]
    for axis, name in [(0, "sigma_yz"), (1, "sigma_xz"), (2, "sigma_xy")]:
        matrix = np.eye(3)
        matrix[axis, axis] = -1.0
        ops.append((name, matrix))
    for n in range(2, max_n + 1):
        for power in range(1, n):
            theta = 2.0 * np.pi * power / n
            ops.append((f"C{n}z^{power}", _rotation_matrix((0, 0, 1), theta)))
            ops.append((f"C{n}x^{power}", _rotation_matrix((1, 0, 0), theta)))
            ops.append((f"C{n}y^{power}", _rotation_matrix((0, 1, 0), theta)))
            sigma_h = np.diag((1.0, 1.0, -1.0))
            ops.append(
                (
                    f"sigma_h*C{n}z^{power}",
                    sigma_h @ _rotation_matrix((0, 0, 1), theta),
                )
            )
    for n in range(2, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            ops.append(
                (
                    f"C2_xy_{n}_{k}",
                    _rotation_matrix((np.cos(theta), np.sin(theta), 0), np.pi),
                )
            )
    for n in range(3, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            normal = np.array((np.cos(theta), np.sin(theta), 0.0), dtype=float)
            normal /= np.linalg.norm(normal)
            ops.append((f"sigma_v_{n}_{k}", np.eye(3) - 2.0 * np.outer(normal, normal)))
            if n % 2 == 1 and 2 * n > max_n:
                shifted = theta + 0.5 * np.pi
                shifted_normal = np.array((np.cos(shifted), np.sin(shifted), 0.0), dtype=float)
                shifted_normal /= np.linalg.norm(shifted_normal)
                ops.append(
                    (
                        f"sigma_v_{n}_{k + n}",
                        np.eye(3) - 2.0 * np.outer(shifted_normal, shifted_normal),
                    )
                )
    for n in range(3, max_n + 1):
        order = 2 * n
        sigma_h = np.diag((1.0, 1.0, -1.0))
        for power in range(1, order, 2):
            theta = 2.0 * np.pi * power / order
            ops.append(
                (
                    f"sigma_h*C{order}z^{power}",
                    sigma_h @ _rotation_matrix((0, 0, 1), theta),
                )
            )
            ops.append(
                (
                    f"C2_xy_{order}_{power}",
                    _rotation_matrix((np.cos(theta / 2.0), np.sin(theta / 2.0), 0), np.pi),
                )
            )
            normal = np.array((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0), dtype=float)
            normal /= np.linalg.norm(normal)
            ops.append((f"sigma_v_{order}_{power}", np.eye(3) - 2.0 * np.outer(normal, normal)))
    ops.extend(_dnd_candidate_ops(max_n))
    ops.extend(_cubic_candidate_ops())
    ops.extend(_icosahedral_candidate_ops())
    return ops


def _dnd_candidate_ops(max_n: int) -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    sd = _diagonal_reflection_matrix()
    for n in range(2, max_n + 1):
        order = 2 * n
        for power in range(n):
            theta = 2.0 * np.pi * power / n
            ops.append(
                (
                    f"Dnd_C2_xy_{order}_{(2 * power + 1) % order}",
                    sd @ _rotation_matrix((0, 0, 1), theta),
                )
            )
        for k in range(n):
            theta = np.pi * k / n
            c2 = _rotation_matrix((np.cos(theta), np.sin(theta), 0), np.pi)
            ops.append((f"Dnd_C{order}z^{(2 * k + 1) % order}", sd @ c2))
    return ops


def _diagonal_reflection_matrix() -> np.ndarray:
    return np.array(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _cubic_candidate_ops() -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    counters: Counter[str] = Counter()
    seen: set[tuple[float, ...]] = set()

    for matrix in _signed_permutation_matrices():
        if _sign_product(matrix) != 1:
            continue
        label = _unique_cubic_label(_td_candidate_class(matrix), "td", counters)
        if label is not None:
            key = tuple(float(value) for value in matrix.reshape(-1))
            seen.add(key)
            ops.append((label, matrix))

    for matrix in _signed_permutation_matrices():
        key = tuple(float(value) for value in matrix.reshape(-1))
        if key in seen:
            continue
        label = _unique_cubic_label(_oh_candidate_class(matrix), "oh", counters)
        if label is not None:
            ops.append((label, matrix))
    return ops


def _signed_permutation_matrices() -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            matrices.append(matrix)
    return tuple(matrices)


def _sign_product(matrix: np.ndarray) -> int:
    product_value = 1.0
    for row in range(3):
        nonzero = np.flatnonzero(np.abs(matrix[row]) > 0.5)
        if len(nonzero) != 1:
            return 0
        product_value *= float(matrix[row, nonzero[0]])
    return 1 if product_value > 0.0 else -1


def _td_candidate_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    det = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))
    if det > 0.0:
        if abs(trace) <= 1.0e-8:
            return "C3"
        if abs(trace + 1.0) <= 1.0e-8:
            return "C2"
    if det < 0.0:
        if abs(trace + 1.0) <= 1.0e-8:
            return "S4"
        if abs(trace - 1.0) <= 1.0e-8:
            return "sigma_d"
    return None


def _oh_candidate_class(matrix: np.ndarray) -> str | None:
    operation_class = _polyhedral_matrix_class(matrix)
    if operation_class is None:
        return None
    return operation_class.replace("*", "_")


def _unique_cubic_label(
    operation_class: str | None,
    prefix: str,
    counters: Counter[str],
) -> str | None:
    if operation_class is None:
        return None
    if operation_class == "E":
        return f"{prefix}_E"
    if operation_class == "i_E":
        return "i"
    counters[f"{prefix}_{operation_class}"] += 1
    return f"{prefix}_{operation_class}_{counters[f'{prefix}_{operation_class}']}"


def _icosahedral_candidate_ops() -> list[tuple[str, np.ndarray]]:
    ops: list[tuple[str, np.ndarray]] = []
    counters: Counter[str] = Counter()
    for matrix in _icosahedral_operation_matrices():
        operation_class = _icosahedral_candidate_class(matrix)
        if operation_class is None:
            continue
        label = _unique_icosahedral_label(operation_class, counters)
        ops.append((label, matrix))
    return ops


@lru_cache(maxsize=1)
def _icosahedral_operation_matrices() -> tuple[np.ndarray, ...]:
    vertices = _icosahedral_vertices()
    source_indices = _independent_vertex_triple(vertices)
    source = vertices[list(source_indices)]
    source_inverse = np.linalg.inv(source)
    matrices: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for target_indices in permutations(range(len(vertices)), 3):
        target = vertices[list(target_indices)]
        if abs(float(np.linalg.det(target))) <= 1.0e-8:
            continue
        matrix = target.T @ source_inverse.T
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-8):
            continue
        if not _maps_vertices_to_self(matrix, vertices):
            continue
        key = tuple(round(float(value), 10) for value in matrix.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        matrices.append(matrix)
    return tuple(matrices)


def _icosahedral_vertices() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = []
    for y in (-1.0, 1.0):
        for z in (-phi, phi):
            vertices.append((0.0, y, z))
    for x in (-1.0, 1.0):
        for y in (-phi, phi):
            vertices.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-1.0, 1.0):
            vertices.append((x, 0.0, z))
    return np.array(vertices, dtype=float)


def _independent_vertex_triple(vertices: np.ndarray) -> tuple[int, int, int]:
    for candidate in permutations(range(len(vertices)), 3):
        if abs(float(np.linalg.det(vertices[list(candidate)]))) > 1.0e-8:
            return tuple(int(item) for item in candidate)
    raise ValueError("no independent icosahedral vertex triple")


def _maps_vertices_to_self(matrix: np.ndarray, vertices: np.ndarray) -> bool:
    transformed = vertices @ matrix.T
    for vertex in transformed:
        if not np.any(np.all(np.isclose(vertices, vertex, atol=1.0e-8), axis=1)):
            return False
    return True


def _icosahedral_candidate_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    det = float(np.linalg.det(matrix))
    proper = matrix if det > 0.0 else -matrix
    proper_class = _icosahedral_proper_class(proper)
    if proper_class is None:
        return None
    return f"i*{proper_class}" if det < 0.0 else proper_class


def _icosahedral_proper_class(matrix: np.ndarray) -> str | None:
    if np.allclose(matrix, np.eye(3), atol=1.0e-8):
        return "E"
    trace = float(np.trace(matrix))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_bar = (1.0 - np.sqrt(5.0)) / 2.0
    if abs(trace) <= 1.0e-8:
        return "C3"
    if abs(trace + 1.0) <= 1.0e-8:
        return "C2"
    if abs(trace - phi) <= 1.0e-8:
        return "C5"
    if abs(trace - phi_bar) <= 1.0e-8:
        return "C5_2"
    return None


def _unique_icosahedral_label(operation_class: str, counters: Counter[str]) -> str:
    if operation_class == "E":
        return "ih_E"
    if operation_class == "i*E":
        return "i"
    key = operation_class.replace("*", "_")
    counters[key] += 1
    return f"ih_{key}_{counters[key]}"


def _rotation_matrix(axis, theta):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )
