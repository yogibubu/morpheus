"""LINK finite-predictor/constrained-corrector realization of SONIC changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from matrix_core import rotation_matrix_from_vector

from .fragment_backtransform import direct_fragment_rigid_prediction
from .internal_coordinates import (
    InternalCoordinateBackTransform,
    cartesian_from_internal_jacobian,
    constrained_internal_coordinate_step,
)

if TYPE_CHECKING:
    from matrix_smith.definition import FrozenGIC, GICDefinition, GICPrimitive


_SOFT_FUNCTIONS = {"D", "IMPD", "U", "RPCK", "FROT", "FTRANS"}
_SOFT_FAMILY_TOKENS = (
    "TORSION",
    "DIHEDRAL",
    "RING_PUCKERING",
    "PUCKER",
    "BUTTERFLY",
    "OUT_OF_PLANE",
    "INVERSION",
    "FRAG_ROTATION",
    "FRAG_TRANSLATION",
)


@dataclass(frozen=True)
class AcyclicTorsionSpec:
    coordinate_index: int
    primitive_id: str
    atoms: tuple[int, int, int, int]
    coefficient: float
    moving_atoms: tuple[int, ...]


@dataclass(frozen=True)
class HybridInternalCoordinateBackTransform:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool
    cartesian_from_q: np.ndarray
    finite_fragment_indices: tuple[int, ...]
    finite_torsion_indices: tuple[int, ...]
    continuation_indices: tuple[int, ...]
    hard_indices: tuple[int, ...]
    substeps: int
    corrector_iterations: int
    method: str = "FINITE_SOFT_PREDICTOR_CONSTRAINED_HARD_CORRECTOR"


def hybrid_internal_coordinate_step(
    definition: "GICDefinition",
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
    tolerance: float = 1.0e-9,
    max_continuation_increment: float = 0.12,
    max_substeps: int = 32,
) -> HybridInternalCoordinateBackTransform:
    """Realize large SONIC moves with finite soft predictors and a hard corrector.

    Complete fragment translations/rotations are imposed on their rigid groups.
    A one-primitive dihedral whose central bond is a graph bridge is imposed by
    an exact finite rotation of one separated subgraph.  Ring, inversion and
    mixed soft coordinates follow a continuation path.  At every path point,
    the remaining coordinates are corrected in the null space of the already
    imposed finite soft coordinates.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    start_values, start_b = _validated_evaluation(evaluate, coords, target.size)
    initial_residual = target - start_values
    initial_norm = float(np.linalg.norm(initial_residual))
    if initial_norm <= tolerance:
        projector = np.linalg.pinv(start_b, rcond=1.0e-8)
        return HybridInternalCoordinateBackTransform(
            coords,
            start_values,
            initial_residual,
            0,
            True,
            projector,
            (),
            (),
            (),
            tuple(range(target.size)),
            0,
            0,
        )

    torsions = _acyclic_torsion_specs(
        definition,
        natoms=coords.shape[0],
        fixed_atom_indices=fixed_atom_indices,
    )
    torsion_indices = tuple(item.coordinate_index for item in torsions)
    fragment_probe = direct_fragment_rigid_prediction(
        definition,
        coords,
        start_values,
        lambda trial: _validated_evaluation(evaluate, trial, target.size)[0],
    )
    fragment_indices = fragment_probe.handled_indices
    finite_indices = tuple(sorted(set((*fragment_indices, *torsion_indices))))
    soft_indices = _soft_coordinate_indices(definition)
    continuation_indices = tuple(sorted(set(soft_indices) - set(finite_indices)))
    hard_indices = tuple(sorted(set(range(target.size)) - set(soft_indices)))
    solve_indices = tuple(sorted((*hard_indices, *continuation_indices)))

    continuation_delta = initial_residual[list(continuation_indices)] if continuation_indices else ()
    max_delta = (
        float(np.max(np.abs(np.asarray(continuation_delta, dtype=float))))
        if len(continuation_delta)
        else 0.0
    )
    substeps = max(1, int(np.ceil(max_delta / max(max_continuation_increment, 1.0e-6))))
    if _has_near_linear_angle(definition, coords):
        soft_delta = initial_residual[list(soft_indices)] if soft_indices else initial_residual
        near_linear_delta = float(np.max(np.abs(soft_delta))) if soft_delta.size else 0.0
        substeps = max(
            substeps,
            2,
            int(
                np.ceil(
                    near_linear_delta / max(0.5 * max_continuation_increment, 1.0e-6)
                )
            ),
        )
    substeps = min(substeps, max(int(max_substeps), 1))
    corrector_iterations = 0
    total_iterations = 0
    last_result: InternalCoordinateBackTransform | None = None

    for substep in range(1, substeps + 1):
        fraction = substep / float(substeps)
        subtarget = start_values + fraction * initial_residual
        # Reapply the finite predictors after each nonlinear hard correction:
        # this is the outer predictor--corrector loop that prevents soft drift.
        for _outer in range(4):
            fragment = direct_fragment_rigid_prediction(
                definition,
                coords,
                subtarget,
                lambda trial: _validated_evaluation(evaluate, trial, target.size)[0],
            )
            coords = fragment.coordinates_angstrom
            coords = direct_acyclic_torsion_prediction(
                coords,
                subtarget,
                evaluate,
                torsions,
            )
            if project_coordinates is not None:
                coords = np.asarray(project_coordinates(coords), dtype=float)
            protected = tuple(sorted(set((*fragment.handled_indices, *torsion_indices))))
            if solve_indices:
                last_result = constrained_internal_coordinate_step(
                    coords,
                    subtarget,
                    evaluate,
                    solve_indices=solve_indices,
                    protected_indices=protected,
                    max_iterations=24,
                    tolerance=max(tolerance * 0.1, 1.0e-11),
                    fixed_atom_indices=fixed_atom_indices,
                    project_coordinates=project_coordinates,
                )
                coords = last_result.coordinates_angstrom
                corrector_iterations += last_result.iterations
                total_iterations += last_result.iterations
            values, _b = _validated_evaluation(evaluate, coords, target.size)
            sub_residual = subtarget - values
            total_iterations += 1
            if float(np.linalg.norm(sub_residual)) <= tolerance * max(
                1.0, float(np.linalg.norm(subtarget - start_values))
            ):
                break

    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    converged = float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm)
    projector = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8)
    return HybridInternalCoordinateBackTransform(
        coordinates_angstrom=coords,
        values=values,
        residual=residual,
        iterations=total_iterations,
        converged=converged,
        cartesian_from_q=projector,
        finite_fragment_indices=fragment_indices,
        finite_torsion_indices=torsion_indices,
        continuation_indices=continuation_indices,
        hard_indices=hard_indices,
        substeps=substeps,
        corrector_iterations=corrector_iterations,
    )


def direct_acyclic_torsion_prediction(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    torsions: tuple[AcyclicTorsionSpec, ...],
) -> np.ndarray:
    """Apply exact finite rotations for independent acyclic dihedrals."""

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if not torsions:
        return coords
    for _pass in range(3):
        improved = False
        for spec in torsions:
            values, _b = _validated_evaluation(evaluate, coords, target.size)
            current_error = _periodic_difference(target[spec.coordinate_index], values[spec.coordinate_index])
            if abs(current_error) <= 1.0e-11:
                continue
            j = spec.atoms[1] - 1
            k = spec.atoms[2] - 1
            axis = coords[k] - coords[j]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12:
                continue
            axis /= axis_norm
            physical_delta = current_error / spec.coefficient
            moving = np.asarray(spec.moving_atoms, dtype=int)
            candidates: list[tuple[float, np.ndarray]] = []
            for sign in (1.0, -1.0):
                trial = coords.copy()
                centered = trial[moving] - coords[j]
                trial[moving] = (
                    centered @ rotation_matrix_from_vector(sign * physical_delta * axis)
                    + coords[j]
                )
                trial_values, _trial_b = _validated_evaluation(evaluate, trial, target.size)
                error = abs(
                    _periodic_difference(
                        target[spec.coordinate_index], trial_values[spec.coordinate_index]
                    )
                )
                candidates.append((error, trial))
            best_error, best = min(candidates, key=lambda item: item[0])
            if best_error + 1.0e-12 < abs(current_error):
                coords = best
                improved = True
        if not improved:
            break
    return coords


def _acyclic_torsion_specs(
    definition: "GICDefinition",
    *,
    natoms: int,
    fixed_atom_indices: tuple[int, ...],
) -> tuple[AcyclicTorsionSpec, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    adjacency = [set() for _ in range(natoms)]
    for primitive in definition.primitives:
        if primitive.function == "R" and len(primitive.atoms) == 2:
            left, right = (atom - 1 for atom in primitive.atoms)
            if 0 <= left < natoms and 0 <= right < natoms and left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
    fixed = set(int(item) for item in fixed_atom_indices)
    specs: list[AcyclicTorsionSpec] = []
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1:
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        coefficient = float(coefficients[0][1])
        if (
            primitive is None
            or primitive.function != "D"
            or len(primitive.atoms) != 4
            or abs(coefficient) <= 1.0e-12
        ):
            continue
        j, k = primitive.atoms[1] - 1, primitive.atoms[2] - 1
        if k not in adjacency[j]:
            adjacency[j].add(k)
            adjacency[k].add(j)
        component_k = _component_without_edge(adjacency, k, (j, k))
        if j in component_k:
            continue
        component_j = _component_without_edge(adjacency, j, (j, k))
        fourth = primitive.atoms[3] - 1
        moving = component_k if fourth in component_k else component_j
        alternate = component_j if moving is component_k else component_k
        if fixed.intersection(moving):
            if fixed.intersection(alternate):
                continue
            moving = alternate
        specs.append(
            AcyclicTorsionSpec(
                coordinate_index=index,
                primitive_id=primitive.identifier,
                atoms=tuple(int(atom) for atom in primitive.atoms),  # type: ignore[arg-type]
                coefficient=coefficient,
                moving_atoms=tuple(sorted(moving)),
            )
        )
    return tuple(specs)


def soft_coordinate_indices(definition: "GICDefinition") -> tuple[int, ...]:
    """Return SONIC rows requiring finite/continuation rather than hard correction.

    This is the single LINK hard/soft classification used both by the hybrid
    back-transform and by adaptive finite-difference step selection.
    """

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    soft: list[int] = []
    for index, gic in enumerate(definition.gics):
        family = str(gic.family).upper()
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        functions = {
            primitive.function
            for primitive_id, _coefficient in coefficients
            if (primitive := primitive_by_id.get(primitive_id)) is not None
        }
        if functions.intersection(_SOFT_FUNCTIONS) or any(
            token in family for token in _SOFT_FAMILY_TOKENS
        ):
            soft.append(index)
    return tuple(soft)


# Internal spelling retained while callers migrate to the public contract.
_soft_coordinate_indices = soft_coordinate_indices


def _component_without_edge(
    adjacency: list[set[int]],
    start: int,
    removed_edge: tuple[int, int],
) -> set[int]:
    removed = {tuple(removed_edge), tuple(reversed(removed_edge))}
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if (atom, neighbor) in removed or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


def _has_near_linear_angle(definition: "GICDefinition", coords: np.ndarray) -> bool:
    for primitive in definition.primitives:
        if primitive.function != "A" or len(primitive.atoms) != 3:
            continue
        i, j, k = (atom - 1 for atom in primitive.atoms)
        left = coords[i] - coords[j]
        right = coords[k] - coords[j]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-14:
            return True
        cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
        if abs(np.sin(np.arccos(cosine))) < 0.12:
            return True
    return False


def _periodic_difference(target: float, value: float) -> float:
    return float((float(target) - float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _validated_evaluation(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, b_matrix = evaluate(np.asarray(coordinates, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    b_matrix = np.asarray(b_matrix, dtype=float)
    if values.shape != (coordinate_count,) or b_matrix.shape != (
        coordinate_count,
        coordinates.size,
    ):
        raise ValueError("internal-coordinate evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(b_matrix)):
        raise ValueError("internal-coordinate evaluator returned non-finite values")
    return values, b_matrix
