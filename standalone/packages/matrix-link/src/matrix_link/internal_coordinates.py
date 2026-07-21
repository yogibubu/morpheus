"""LINK-owned internal-to-Cartesian realization services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class InternalCoordinateBackTransform:
    coordinates_angstrom: np.ndarray
    values: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool
    cartesian_from_q: np.ndarray


@dataclass(frozen=True)
class ProjectorSecantUpdate:
    cartesian_from_q: np.ndarray | None
    relative_error: float
    accepted: bool


def cartesian_from_internal_jacobian(
    b_matrix: np.ndarray,
    *,
    rcond: float = 1.0e-8,
    fixed_cartesian_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Return LINK's local internal-to-Cartesian Jacobian from a Wilson B matrix.

    Coordinate providers supply ``B``; LINK owns its generalized inverse and
    the policies that realize finite internal-coordinate changes.
    """

    b = np.asarray(b_matrix, dtype=float)
    if b.ndim != 2 or not np.all(np.isfinite(b)):
        raise ValueError("Wilson B matrix must be a finite two-dimensional array")
    if fixed_cartesian_columns is None or np.asarray(fixed_cartesian_columns).size == 0:
        return np.linalg.pinv(b, rcond=float(rcond))
    fixed = np.asarray(fixed_cartesian_columns, dtype=int).reshape(-1)
    if np.any(fixed < 0) or np.any(fixed >= b.shape[1]):
        raise ValueError("fixed Cartesian column is outside the Wilson B matrix")
    free = np.ones(b.shape[1], dtype=bool)
    free[fixed] = False
    projector = np.zeros((b.shape[1], b.shape[0]), dtype=float)
    projector[free, :] = np.linalg.pinv(b[:, free], rcond=float(rcond))
    return projector


def internal_from_cartesian_jacobian(
    cartesian_from_internal: np.ndarray,
    *,
    rcond: float = 1.0e-8,
) -> np.ndarray:
    """Return the minimum-norm Cartesian-to-internal linear map.

    This is LINK's projection service for user or SENTINEL variables supplied
    as Cartesian directions. It does not define new coordinates; it projects
    them into the frozen SONIC tangent space supplied by SMITH.
    """

    matrix = np.asarray(cartesian_from_internal, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Cartesian-from-internal Jacobian must be a finite 2-D array")
    return np.linalg.pinv(matrix, rcond=float(rcond))


def nonlinear_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    cartesian_from_q: np.ndarray | None = None,
    max_iterations: int = 60,
    tolerance: float = 1.0e-9,
    max_cartesian_step_angstrom: float = 0.25,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
) -> InternalCoordinateBackTransform:
    """Solve ``q(x) = target_values`` by damped least squares.

    ``evaluate`` returns the active internal-coordinate values and their Wilson
    matrix at a Cartesian geometry.  The accepted geometry is always evaluated
    explicitly, so callers never need to identify a requested internal step
    with the step actually realized by a nonlinear back-transformation.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    fixed_columns = _fixed_cartesian_columns(fixed_atom_indices, coords.shape[0])
    target = np.asarray(target_values, dtype=float).reshape(-1)
    projector = None if cartesian_from_q is None else np.asarray(cartesian_from_q, dtype=float)
    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    initial_norm = float(np.linalg.norm(residual))
    if initial_norm <= tolerance:
        projector = _cartesian_projector(b_matrix, fixed_columns)
        return InternalCoordinateBackTransform(coords, values, residual, 0, True, projector)

    for iteration in range(1, max_iterations + 1):
        if projector is None or projector.shape != (coords.size, target.size):
            projector = _cartesian_projector(b_matrix, fixed_columns)
        dx = projector @ residual
        dx_norm = float(np.linalg.norm(dx))
        if max_cartesian_step_angstrom > 0.0 and dx_norm > max_cartesian_step_angstrom:
            dx *= max_cartesian_step_angstrom / dx_norm

        current_norm = float(np.linalg.norm(residual))
        accepted = False
        scale = 1.0
        for _ in range(10):
            trial_coords = coords + (scale * dx).reshape(coords.shape)
            if project_coordinates is not None:
                trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
            trial_values, trial_b = _validated_evaluation(evaluate, trial_coords, target.size)
            trial_residual = target - trial_values
            if float(np.linalg.norm(trial_residual)) < current_norm:
                coords = trial_coords
                values = trial_values
                b_matrix = trial_b
                residual = trial_residual
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            projector = _cartesian_projector(b_matrix, fixed_columns)
            fallback = _damped_internal_step(
                b_matrix, residual, coords.shape, fixed_columns=fixed_columns
            )
            if fallback is None:
                return InternalCoordinateBackTransform(
                    coords, values, residual, iteration, False, projector
                )
            for fallback_dx in fallback:
                fallback_norm = float(np.linalg.norm(fallback_dx))
                if max_cartesian_step_angstrom > 0.0 and fallback_norm > max_cartesian_step_angstrom:
                    fallback_dx *= max_cartesian_step_angstrom / fallback_norm
                trial_coords = coords + fallback_dx.reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                trial_values, trial_b = _validated_evaluation(evaluate, trial_coords, target.size)
                trial_residual = target - trial_values
                if float(np.linalg.norm(trial_residual)) < current_norm:
                    coords, values, b_matrix, residual = (
                        trial_coords,
                        trial_values,
                        trial_b,
                        trial_residual,
                    )
                    accepted = True
                    break
            if not accepted:
                return InternalCoordinateBackTransform(
                    coords, values, residual, iteration, False, projector
                )
        projector = _cartesian_projector(b_matrix, fixed_columns)
        if float(np.linalg.norm(residual)) <= tolerance * max(1.0, initial_norm):
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, True, projector
            )
    return InternalCoordinateBackTransform(
        coords, values, residual, max_iterations, False, projector
    )


def constrained_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    solve_indices: tuple[int, ...],
    protected_indices: tuple[int, ...] = (),
    protected_combinations: np.ndarray | None = None,
    max_iterations: int = 30,
    tolerance: float = 1.0e-9,
    max_cartesian_step_angstrom: float = 0.20,
    fixed_atom_indices: tuple[int, ...] = (),
    project_coordinates: Callable[[np.ndarray], np.ndarray] | None = None,
) -> InternalCoordinateBackTransform:
    """Correct selected coordinates without stepping along protected rows.

    At every iteration the Cartesian correction is restricted to the numerical
    null space of the protected Wilson rows.  ``protected_combinations`` may
    additionally contain covectors in the internal-coordinate basis; their
    Cartesian constraint rows are ``C @ B``.  This permits LINK to protect a
    delocalized normal coordinate such as the SONIC Qim mode rather than only
    one named internal coordinate.  A damped least-squares solution is then
    computed for ``B_h Z y = r_h``.  This is the corrector half of the finite-
    predictor/linear-corrector SONIC back-transform.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float).copy()
    target = np.asarray(target_values, dtype=float).reshape(-1)
    solve = np.asarray(tuple(dict.fromkeys(int(item) for item in solve_indices)), dtype=int)
    protected = np.asarray(
        tuple(dict.fromkeys(int(item) for item in protected_indices)), dtype=int
    )
    if np.intersect1d(solve, protected).size:
        raise ValueError("solve and protected internal-coordinate sets must be disjoint")
    if solve.size and (np.min(solve) < 0 or np.max(solve) >= target.size):
        raise ValueError("solve internal-coordinate index is outside target vector")
    if protected.size and (np.min(protected) < 0 or np.max(protected) >= target.size):
        raise ValueError("protected internal-coordinate index is outside target vector")
    combinations = _validated_constraint_combinations(
        protected_combinations, target.size
    )
    fixed_columns = _fixed_cartesian_columns(fixed_atom_indices, coords.shape[0])
    values, b_matrix = _validated_evaluation(evaluate, coords, target.size)
    residual = target - values
    initial_solve_norm = float(np.linalg.norm(residual[solve])) if solve.size else 0.0
    initial_total_norm = float(np.linalg.norm(residual))
    projector = _cartesian_projector(b_matrix, fixed_columns)
    if initial_solve_norm <= tolerance:
        return InternalCoordinateBackTransform(coords, values, residual, 0, True, projector)

    for iteration in range(1, max_iterations + 1):
        candidates = _constrained_damped_steps(
            b_matrix,
            residual,
            solve,
            _protected_constraint_rows(b_matrix, protected, combinations),
            coords.shape,
            fixed_columns=fixed_columns,
        )
        if not candidates:
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, False, projector
            )
        current_hard = float(np.linalg.norm(residual[solve]))
        current_soft = float(
            np.linalg.norm(_protected_constraint_residual(residual, protected, combinations))
        )
        current_score = current_hard**2 + 100.0 * current_soft**2
        accepted = False
        for candidate in candidates:
            step = np.asarray(candidate, dtype=float)
            step_norm = float(np.linalg.norm(step))
            if max_cartesian_step_angstrom > 0.0 and step_norm > max_cartesian_step_angstrom:
                step *= max_cartesian_step_angstrom / step_norm
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial_coords = coords + (scale * step).reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                trial_values, trial_b = _validated_evaluation(evaluate, trial_coords, target.size)
                trial_residual = target - trial_values
                trial_hard = float(np.linalg.norm(trial_residual[solve]))
                trial_soft = float(
                    np.linalg.norm(
                        _protected_constraint_residual(
                            trial_residual, protected, combinations
                        )
                    )
                )
                score = trial_hard**2 + 100.0 * trial_soft**2
                if trial_hard < current_hard and score < current_score:
                    coords, values, b_matrix, residual = (
                        trial_coords,
                        trial_values,
                        trial_b,
                        trial_residual,
                    )
                    accepted = True
                    break
            if accepted:
                break
        projector = _cartesian_projector(b_matrix, fixed_columns)
        if not accepted:
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, False, projector
            )
        hard_norm = float(np.linalg.norm(residual[solve]))
        if hard_norm <= tolerance * max(1.0, initial_solve_norm, initial_total_norm):
            return InternalCoordinateBackTransform(
                coords, values, residual, iteration, True, projector
            )
    return InternalCoordinateBackTransform(
        coords, values, residual, max_iterations, False, projector
    )


def secant_projector_update(
    cartesian_from_q: np.ndarray,
    previous_q: np.ndarray,
    previous_coordinates: np.ndarray,
    current_q: np.ndarray,
    current_coordinates: np.ndarray,
) -> ProjectorSecantUpdate:
    """Apply the shared rank-one secant update to an internal-coordinate projector."""

    matrix = np.asarray(cartesian_from_q, dtype=float)
    q_delta = np.asarray(current_q, dtype=float).reshape(-1) - np.asarray(
        previous_q, dtype=float
    ).reshape(-1)
    x_delta = np.asarray(current_coordinates, dtype=float).reshape(-1) - np.asarray(
        previous_coordinates, dtype=float
    ).reshape(-1)
    if matrix.shape != (x_delta.size, q_delta.size):
        return ProjectorSecantUpdate(None, float("inf"), False)
    denominator = float(q_delta @ q_delta)
    if denominator <= 1.0e-24 or not np.isfinite(denominator):
        return ProjectorSecantUpdate(None, float("inf"), False)
    residual = x_delta - matrix @ q_delta
    relative_error = float(np.linalg.norm(residual) / max(np.linalg.norm(x_delta), 1.0e-12))
    correction = np.outer(residual, q_delta) / denominator
    updated = matrix + correction
    if not np.all(np.isfinite(updated)):
        return ProjectorSecantUpdate(None, relative_error, False)
    if relative_error > 0.75 or np.linalg.norm(correction) > max(0.75 * np.linalg.norm(matrix), 1.0e-8):
        return ProjectorSecantUpdate(None, relative_error, False)
    return ProjectorSecantUpdate(updated, relative_error, True)


def should_refresh_coordinate_model(
    *,
    model_age: int,
    line_search_scale: float,
    trust_ratio: float,
    secant_relative_error: float,
) -> bool:
    """Shared MORPHEUS/TRINITY policy for refreshing a nonlinear coordinate model."""

    return bool(
        model_age >= 3
        or line_search_scale < 0.5
        or not np.isfinite(secant_relative_error)
        or secant_relative_error > 0.5
        or trust_ratio < 0.25
        or trust_ratio > 2.5
    )


def transport_internal_hessian(
    hessian: np.ndarray,
    old_b_matrix: np.ndarray,
    new_cartesian_from_q: np.ndarray,
) -> np.ndarray:
    """Transport an internal Hessian to the tangent space at a new geometry."""

    h = np.asarray(hessian, dtype=float)
    tangent_map = np.asarray(old_b_matrix, dtype=float) @ np.asarray(
        new_cartesian_from_q, dtype=float
    )
    transported = tangent_map.T @ h @ tangent_map
    return 0.5 * (transported + transported.T)


def _cartesian_projector(
    b_matrix: np.ndarray,
    fixed_columns: np.ndarray | None = None,
) -> np.ndarray:
    return cartesian_from_internal_jacobian(
        b_matrix,
        fixed_cartesian_columns=fixed_columns,
    )


def _damped_internal_step(
    b_matrix: np.ndarray,
    residual: np.ndarray,
    coordinate_shape: tuple[int, ...],
    fixed_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, ...] | None:
    b = np.asarray(b_matrix, dtype=float)
    free = np.ones(b.shape[1], dtype=bool)
    if fixed_columns is not None and fixed_columns.size:
        free[fixed_columns] = False
    b_free = b[:, free]
    rhs = np.asarray(residual, dtype=float)
    metric = b_free @ b_free.T
    identity = np.eye(metric.shape[0], dtype=float)
    candidates: list[np.ndarray] = []
    for damping in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0):
        try:
            free_step = b_free.T @ np.linalg.solve(metric + damping * identity, rhs)
        except np.linalg.LinAlgError:
            continue
        step = np.zeros(b.shape[1], dtype=float)
        step[free] = free_step
        if step.shape == (int(np.prod(coordinate_shape)),) and np.all(np.isfinite(step)):
            candidates.append(step)
    return tuple(candidates) if candidates else None


def _constrained_damped_steps(
    b_matrix: np.ndarray,
    residual: np.ndarray,
    solve_indices: np.ndarray,
    protected_rows: np.ndarray,
    coordinate_shape: tuple[int, ...],
    *,
    fixed_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    b = np.asarray(b_matrix, dtype=float)
    free = np.ones(b.shape[1], dtype=bool)
    if fixed_columns is not None and fixed_columns.size:
        free[fixed_columns] = False
    free_count = int(np.count_nonzero(free))
    if free_count == 0 or solve_indices.size == 0:
        return ()
    hard = b[solve_indices, :][:, free]
    if protected_rows.size:
        soft = np.asarray(protected_rows, dtype=float)[:, free]
        _u, singular, vh = np.linalg.svd(soft, full_matrices=True)
        cutoff = 1.0e-10 * max(float(singular[0]) if singular.size else 0.0, 1.0)
        rank = int(np.count_nonzero(singular > cutoff))
        null_basis = vh[rank:, :].T
    else:
        null_basis = np.eye(free_count, dtype=float)
    if null_basis.shape[1] == 0:
        return ()
    reduced = hard @ null_basis
    rhs = np.asarray(residual, dtype=float)[solve_indices]
    metric = reduced @ reduced.T
    identity = np.eye(metric.shape[0], dtype=float)
    candidates: list[np.ndarray] = []
    for damping in (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0):
        try:
            reduced_step = reduced.T @ np.linalg.solve(metric + damping * identity, rhs)
        except np.linalg.LinAlgError:
            continue
        free_step = null_basis @ reduced_step
        step = np.zeros(b.shape[1], dtype=float)
        step[free] = free_step
        if step.shape == (int(np.prod(coordinate_shape)),) and np.all(np.isfinite(step)):
            candidates.append(step)
    return tuple(candidates)


def _validated_constraint_combinations(
    combinations: np.ndarray | None, coordinate_count: int
) -> np.ndarray:
    if combinations is None:
        return np.empty((0, coordinate_count), dtype=float)
    matrix = np.asarray(combinations, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != coordinate_count:
        raise ValueError(
            "protected internal-coordinate combinations must have one column per coordinate"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("protected internal-coordinate combinations must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1.0e-14):
        raise ValueError("protected internal-coordinate combination has zero norm")
    return matrix / norms[:, None]


def _protected_constraint_rows(
    b_matrix: np.ndarray,
    protected_indices: np.ndarray,
    combinations: np.ndarray,
) -> np.ndarray:
    rows = []
    if protected_indices.size:
        rows.append(np.asarray(b_matrix, dtype=float)[protected_indices, :])
    if combinations.size:
        rows.append(combinations @ np.asarray(b_matrix, dtype=float))
    if not rows:
        return np.empty((0, np.asarray(b_matrix).shape[1]), dtype=float)
    return np.vstack(rows)


def _protected_constraint_residual(
    residual: np.ndarray,
    protected_indices: np.ndarray,
    combinations: np.ndarray,
) -> np.ndarray:
    values = []
    if protected_indices.size:
        values.append(np.asarray(residual, dtype=float)[protected_indices])
    if combinations.size:
        values.append(combinations @ np.asarray(residual, dtype=float))
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _fixed_cartesian_columns(atom_indices: tuple[int, ...], natoms: int) -> np.ndarray:
    atoms = tuple(int(index) for index in atom_indices)
    if not atoms:
        return np.asarray([], dtype=int)
    if len(set(atoms)) != len(atoms) or min(atoms) < 0 or max(atoms) >= natoms:
        raise ValueError("fixed atom indices are invalid")
    return np.asarray(
        [3 * atom + component for atom in atoms for component in range(3)],
        dtype=int,
    )


def _validated_evaluation(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, b_matrix = evaluate(coordinates)
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
