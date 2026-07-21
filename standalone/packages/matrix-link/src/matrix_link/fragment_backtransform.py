"""LINK direct rigid-fragment predictors for SONIC realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from matrix_core import rotation_matrix_from_vector

if TYPE_CHECKING:
    from matrix_smith.definition import GICDefinition


@dataclass(frozen=True)
class FragmentRigidPrediction:
    coordinates_angstrom: np.ndarray
    handled_indices: tuple[int, ...]


@dataclass(frozen=True)
class FragmentRigidTangent:
    """Cartesian tangents consistent with the finite fragment predictor."""

    cartesian_from_q: np.ndarray
    handled_indices: tuple[int, ...]


def direct_fragment_rigid_tangent(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    b_matrix: np.ndarray,
    *,
    fixed_atom_indices: tuple[int, ...] = (),
) -> FragmentRigidTangent:
    """Return ``dx/dq`` columns for complete FTRANS/FROT triplets.

    The columns are the differential of the same rigid translations and
    exponential-map rotations used by :func:`direct_fragment_rigid_prediction`.
    They therefore provide the correct pullback of a Cartesian gradient for
    special fragment coordinates; a global Moore--Penrose inverse need not
    choose the same tangent in this subspace.

    The physical generators are analytic.  Their small dense Jacobian in the
    stored SONIC chart is obtained by contracting them with the analytic B
    matrix, so no finite-difference derivative is introduced here.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    matrix = np.asarray(b_matrix, dtype=float)
    nq = len(definition.gics)
    ncart = coords.size
    if matrix.shape != (nq, ncart):
        raise ValueError(f"fragment tangent needs a B matrix of shape {(nq, ncart)}")
    fixed = {int(index) for index in fixed_atom_indices}
    physical_columns: list[np.ndarray] = []
    handled: list[int] = []

    for (function, atoms, ref_atoms), modes in _unit_fragment_groups(definition).items():
        if set(modes) != {0, 1, 2}:
            continue
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        if fixed.intersection(atom_indices.tolist()) or fixed.intersection(ref_indices.tolist()):
            continue
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))
        generators = np.zeros((ncart, 3), dtype=float)
        if function == "FTRANS":
            for axis in range(3):
                displacement = np.zeros_like(coords)
                displacement[atom_indices, axis] = moving_share
                displacement[ref_indices, axis] = -reference_share
                generators[:, axis] = displacement.reshape(-1)
        else:
            center = np.mean(coords[atom_indices, :], axis=0)
            centered = coords[atom_indices, :] - center
            ref_center = np.mean(coords[ref_indices, :], axis=0)
            ref_centered = coords[ref_indices, :] - ref_center
            for axis in range(3):
                unit = np.zeros(3, dtype=float)
                unit[axis] = 1.0
                skew = np.asarray(
                    [
                        [0.0, -unit[2], unit[1]],
                        [unit[2], 0.0, -unit[0]],
                        [-unit[1], unit[0], 0.0],
                    ],
                    dtype=float,
                )
                displacement = np.zeros_like(coords)
                # MATRIX uses row-vector rotations R(v)=I-[v]_x+O(v^2).
                displacement[atom_indices, :] = moving_share * (centered @ (-skew))
                displacement[ref_indices, :] = reference_share * (ref_centered @ skew)
                generators[:, axis] = displacement.reshape(-1)
        physical_columns.extend(generators[:, axis] for axis in range(3))
        handled.extend(modes[mode][0] for mode in range(3))

    empty = np.zeros((ncart, nq), dtype=float)
    if not handled:
        return FragmentRigidTangent(empty, ())

    physical = np.column_stack(physical_columns)
    handled_array = np.asarray(handled, dtype=int)
    chart_jacobian = matrix[handled_array, :] @ physical
    if np.linalg.matrix_rank(chart_jacobian, tol=1.0e-10) != len(handled):
        return FragmentRigidTangent(empty, ())
    mapped = physical @ np.linalg.solve(chart_jacobian, np.eye(len(handled)))

    selector = np.zeros((nq, len(handled)), dtype=float)
    selector[handled_array, np.arange(len(handled))] = 1.0
    # Direct rigid motions must leave every non-fragment SONIC unchanged.  If
    # a future mixed definition violates that contract, retain the safe
    # general pseudoinverse rather than silently returning inconsistent rows.
    if float(np.max(np.abs(matrix @ mapped - selector))) > 1.0e-7:
        return FragmentRigidTangent(empty, ())
    result = empty
    result[:, handled_array] = mapped
    return FragmentRigidTangent(result, tuple(int(index) for index in handled))


def direct_fragment_rigid_prediction(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate_values: Callable[[np.ndarray], np.ndarray],
) -> FragmentRigidPrediction:
    """Impose complete FTRANS/FROT triplets by fragment rigid motions.

    Only unit, one-primitive GIC rows are eligible.  Incomplete or symmetry
    mixed blocks are left to the general nonlinear back-transformer.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    handled: list[int] = []
    for (function, atoms, ref_atoms), modes in _unit_fragment_groups(definition).items():
        if set(modes) != {0, 1, 2}:
            continue
        indices = [modes[mode][0] for mode in range(3)]
        atom_indices = np.asarray([atom - 1 for atom in atoms], dtype=int)
        ref_indices = np.asarray([atom - 1 for atom in ref_atoms], dtype=int)
        moving_share = len(ref_atoms) / float(len(atoms) + len(ref_atoms))
        reference_share = len(atoms) / float(len(atoms) + len(ref_atoms))
        current = np.asarray(evaluate_values(coords), dtype=float)
        delta = target[indices] - current[indices]
        if function == "FTRANS":
            coords[atom_indices, :] += moving_share * delta
            coords[ref_indices, :] -= reference_share * delta
        else:
            for _iteration in range(4):
                current = np.asarray(evaluate_values(coords), dtype=float)
                delta = target[indices] - current[indices]
                if float(np.linalg.norm(delta)) <= 1.0e-10:
                    break
                center = np.mean(coords[atom_indices, :], axis=0)
                centered = coords[atom_indices, :] - center
                ref_center = np.mean(coords[ref_indices, :], axis=0)
                ref_centered = coords[ref_indices, :] - ref_center

                def rotated(increment: np.ndarray) -> np.ndarray:
                    trial = coords.copy()
                    trial[atom_indices, :] = (
                        centered @ rotation_matrix_from_vector(moving_share * increment) + center
                    )
                    trial[ref_indices, :] = (
                        ref_centered
                        @ rotation_matrix_from_vector(-reference_share * increment)
                        + ref_center
                    )
                    return trial

                jacobian = np.zeros((3, 3), dtype=float)
                epsilon = 1.0e-5
                for axis in range(3):
                    increment = np.zeros(3, dtype=float)
                    increment[axis] = epsilon
                    trial = rotated(increment)
                    jacobian[:, axis] = (
                        np.asarray(evaluate_values(trial), dtype=float)[indices] - current[indices]
                    ) / epsilon
                try:
                    physical_increment = np.linalg.solve(jacobian, delta)
                except np.linalg.LinAlgError:
                    physical_increment = np.linalg.pinv(jacobian, rcond=1.0e-10) @ delta
                candidates = []
                for scale in (1.0, 0.5, 0.25):
                    trial = rotated(scale * physical_increment)
                    residual = target[indices] - np.asarray(evaluate_values(trial), dtype=float)[indices]
                    candidates.append((float(np.linalg.norm(residual)), trial))
                best_norm, best_trial = min(candidates, key=lambda item: item[0])
                if best_norm >= float(np.linalg.norm(delta)):
                    break
                coords = best_trial
        handled.extend(indices)
    return FragmentRigidPrediction(coords, tuple(sorted(set(handled))))


def _unit_fragment_groups(
    definition: "GICDefinition",
) -> dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[int, tuple[int, object]]]:
    """Collect unit, one-primitive fragment-coordinate triplets."""

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[int, tuple[int, object]]] = {}
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or not np.isclose(float(coefficients[0][1]), 1.0):
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None or primitive.function not in {"FTRANS", "FROT"}:
            continue
        key = (primitive.function, tuple(primitive.atoms), tuple(primitive.ref_atoms))
        groups.setdefault(key, {})[int(primitive.mode)] = (index, primitive)
    return groups
