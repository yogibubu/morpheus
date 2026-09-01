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
    # Wilson rows can have very different numerical scales.  This is common
    # for nearly linear bends, whose analytic derivatives can be orders of
    # magnitude larger than stretches or fragment coordinates.  Applying the
    # cutoff directly to the raw SVD then discards valid coordinate directions
    # in mixed-scale internal-coordinate models.
    # Equilibrate rows before the SVD and undo that scaling on the right:
    # pinv(W B) W is the minimum-norm inverse of B in the original internal
    # displacement units.
    row_norms = np.linalg.norm(b, axis=1)
    row_scale = np.ones(b.shape[0], dtype=float)
    nonzero = row_norms > np.finfo(float).eps
    row_scale[nonzero] = 1.0 / row_norms[nonzero]
    scaled = row_scale[:, None] * b
    fixed = (
        np.asarray((), dtype=int)
        if fixed_cartesian_columns is None
        else np.asarray(fixed_cartesian_columns, dtype=int).reshape(-1)
    )
    row_unscale = np.diag(row_scale)
    if fixed.size == 0:
        return np.linalg.pinv(scaled, rcond=float(rcond)) @ row_unscale
    if np.any(fixed < 0) or np.any(fixed >= b.shape[1]):
        raise ValueError("fixed Cartesian column is outside the Wilson B matrix")
    free = np.ones(b.shape[1], dtype=bool)
    free[fixed] = False
    projector = np.zeros((b.shape[1], b.shape[0]), dtype=float)
    projector[free, :] = np.linalg.pinv(scaled[:, free], rcond=float(rcond)) @ row_unscale
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
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
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
            if evaluate_values is None:
                trial_values, trial_b = _validated_evaluation(evaluate, trial_coords, target.size)
            else:
                trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                trial_b = None
            trial_residual = target - trial_values
            if float(np.linalg.norm(trial_residual)) < current_norm:
                if trial_b is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
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
                if (
                    max_cartesian_step_angstrom > 0.0
                    and fallback_norm > max_cartesian_step_angstrom
                ):
                    fallback_dx *= max_cartesian_step_angstrom / fallback_norm
                trial_coords = coords + fallback_dx.reshape(coords.shape)
                if project_coordinates is not None:
                    trial_coords = np.asarray(project_coordinates(trial_coords), dtype=float)
                if evaluate_values is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
                else:
                    trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                    trial_b = None
                trial_residual = target - trial_values
                if float(np.linalg.norm(trial_residual)) < current_norm:
                    if trial_b is None:
                        trial_values, trial_b = _validated_evaluation(
                            evaluate, trial_coords, target.size
                        )
                        trial_residual = target - trial_values
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


def gdv_redq2x_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    coordinate_unit_scales: np.ndarray,
    max_iterations: int = 200,
    fixed_atom_indices: tuple[int, ...] = (),
) -> InternalCoordinateBackTransform:
    """Replicate GDV ``RedQ2X`` for generic redundant coordinates.

    The implementation follows ``gdv.j32+/utilnz.F:RedQ2X`` in GDV's
    bohr/radian units: each residual component is initially limited to 0.2,
    the Wilson least-squares displacement is rescaled by the agreement
    between its linear and curvilinear internal changes, and convergence is
    based on the Cartesian correction becoming smaller than 1e-6 bohr RMS.
    As in GDV, a small residual in a redundant coordinate is not by itself a
    back-transformation failure; the caller must retain the internal values
    actually realized by the returned Cartesian geometry.
    """

    coords_angstrom = np.asarray(coordinates_angstrom, dtype=float)
    target_native = np.asarray(target_values, dtype=float).reshape(-1)
    scales = np.asarray(coordinate_unit_scales, dtype=float).reshape(-1)
    if scales.shape != target_native.shape:
        raise ValueError("GDV coordinate-unit scales do not match back-transform target")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("GDV coordinate-unit scales must be positive and finite")
    if max_iterations < 1:
        raise ValueError("GDV RedQ2X max_iterations must be positive")

    # Keep the linear algebra in the same units as RedQ2X.  The evaluator's
    # native convention is q(angstrom/radian) and dq/dx(angstrom).
    from .scan import ANGSTROM_TO_BOHR

    coords_bohr = coords_angstrom.copy() * ANGSTROM_TO_BOHR
    fixed_columns = _fixed_cartesian_columns(
        fixed_atom_indices, coords_angstrom.shape[0]
    )
    free_columns = np.setdiff1d(
        np.arange(coords_bohr.size, dtype=int), fixed_columns, assume_unique=True
    )

    def evaluate_gdv(cartesian_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        native_values, native_b = _validated_evaluation(
            evaluate,
            np.asarray(cartesian_bohr, dtype=float) / ANGSTROM_TO_BOHR,
            target_native.size,
        )
        values = scales * native_values
        b_matrix = (
            scales[:, None]
            * np.asarray(native_b, dtype=float)
            / ANGSTROM_TO_BOHR
        )
        return values, b_matrix

    values0, b_matrix = evaluate_gdv(coords_bohr)
    target = scales * target_native
    requested = target - values0
    values = values0.copy()
    residual = requested.copy()
    dq_limit = 0.2
    cartesian_rms = np.inf
    projector = np.zeros((coords_bohr.size, target.size), dtype=float)
    active_b = b_matrix[:, free_columns]
    inverse_btb = _gdv_frmbtb_diagonal_inverse(active_b)

    for iteration in range(1, int(max_iterations) + 1):
        limited = np.clip(residual, -dq_limit, dq_limit)
        limited_count = int(np.count_nonzero(np.abs(residual) > dq_limit))

        dx = np.zeros(coords_bohr.size, dtype=float)
        if free_columns.size:
            active_b = b_matrix[:, free_columns]
            btb_operator = _gdv_mlpxx1_operator(
                active_b,
                coords_bohr,
                free_columns,
            )
            # SLEqS3 solves (B^T B) dx = B^T dq by iterative refinement of
            # the diagonal inverse guess built once by FrmBTB.  MlPXX1 adds
            # GDV's translation and FormRB rotation terms to B^T B; the
            # refined inverse persists between RedQ2X iterations.
            free_dx, inverse_btb, solve_status = _gdv_sleqs3(
                btb_operator,
                active_b.T @ limited,
                inverse_btb,
            )
            if solve_status > 0:
                inverse_btb = _gdv_frmbtb_diagonal_inverse(active_b)
                free_dx, inverse_btb, solve_status = _gdv_sleqs3(
                    btb_operator,
                    active_b.T @ limited,
                    inverse_btb,
                )
            dx[free_columns] = free_dx
            dx = _gdv_remove_translation_rotation(coords_bohr, dx)
            dx[fixed_columns] = 0.0
            projector[free_columns, :] = np.linalg.pinv(active_b, rcond=1.0e-10)

        # CrdBMl(IOp=3) overwrites DQTmp0 with B*dx before RedQ2X compares
        # the linear and curvilinear changes.  In redundant coordinates this
        # is the representable projection of the requested residual, not the
        # residual itself.
        linear = b_matrix @ dx
        trial_values, _trial_b = evaluate_gdv(coords_bohr + dx.reshape(coords_bohr.shape))
        curvilinear = trial_values - values
        linear_norm = float(np.linalg.norm(linear))
        curvilinear_norm = float(np.linalg.norm(curvilinear))
        scalar_product = float(curvilinear @ linear)
        small = np.finfo(float).tiny
        if (
            curvilinear_norm >= small
            and linear_norm > small
            and abs(scalar_product) > small
        ):
            cosine = scalar_product / (curvilinear_norm * linear_norm)
            if abs(cosine) <= small:
                return InternalCoordinateBackTransform(
                    coords_bohr / ANGSTROM_TO_BOHR,
                    values / scales,
                    residual / scales,
                    iteration,
                    False,
                    projector / ANGSTROM_TO_BOHR * scales[None, :],
                )
            scale = abs(scalar_product) / (curvilinear_norm * curvilinear_norm)
            minimum_scale = 0.1 if iteration == 1 else 0.0
            dx *= max(min(scale, 1.0), minimum_scale)

        cartesian_rms = float(np.sqrt(np.mean(dx * dx)))
        coords_bohr += dx.reshape(coords_bohr.shape)
        values, b_matrix = evaluate_gdv(coords_bohr)
        realized = values - values0
        residual = requested - realized
        internal_rms = float(np.sqrt(np.mean(residual * residual)))
        max_residual = float(np.max(np.abs(residual), initial=0.0))

        if cartesian_rms < 1.0e-6:
            if limited_count:
                # RedQ2X expands DQMax by four after a capped step stalls.
                dq_limit *= 4.0
            elif max_residual <= 1.0 and internal_rms <= 1.0:
                native_values = values / scales
                native_residual = residual / scales
                native_projector = projector / ANGSTROM_TO_BOHR * scales[None, :]
                return InternalCoordinateBackTransform(
                    coords_bohr / ANGSTROM_TO_BOHR,
                    native_values,
                    native_residual,
                    iteration,
                    True,
                    native_projector,
                )
        if iteration > 100 and cartesian_rms > 1.0e-4:
            break

    return InternalCoordinateBackTransform(
        coords_bohr / ANGSTROM_TO_BOHR,
        values / scales,
        residual / scales,
        min(int(max_iterations), iteration),
        False,
        projector / ANGSTROM_TO_BOHR * scales[None, :],
    )


def _gdv_frmbtb_diagonal_inverse(b_matrix: np.ndarray) -> np.ndarray:
    """Return the literal diagonal ``FrmBTB`` inverse guess."""

    b = np.asarray(b_matrix, dtype=float)
    diagonal = np.sum(b * b, axis=0)
    result = np.zeros((b.shape[1], b.shape[1]), dtype=float)
    usable = diagonal > 1.0e-20
    indices = np.flatnonzero(usable)
    result[indices, indices] = 1.0 / diagonal[usable]
    return result


def _gdv_mlpxx1_operator(
    b_matrix: np.ndarray,
    coordinates_bohr: np.ndarray,
    cartesian_columns: np.ndarray,
) -> np.ndarray:
    """Return the literal translation/rotation-augmented ``MlPXX1`` matrix."""

    b = np.asarray(b_matrix, dtype=float)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    columns = np.asarray(cartesian_columns, dtype=int).reshape(-1)
    atom_count = coordinates.shape[0]
    operator = b.T @ b
    if atom_count == 0:
        return operator

    translation_scale = 1.0 / float(atom_count * atom_count)
    for axis in range(3):
        basis = np.zeros_like(coordinates)
        basis[:, axis] = 1.0
        vector = basis.reshape(-1)[columns]
        operator += translation_scale * np.outer(vector, vector)

    centered = coordinates - np.mean(coordinates, axis=0)
    values, axes = np.linalg.eigh(centered.T @ centered)
    value_sum = float(np.sum(values))
    for index in range(3):
        basis = np.cross(centered, axes[:, index])
        scale = value_sum - float(values[index])
        if scale > 1.0e-15:
            basis /= scale
        else:
            basis *= scale
        vector = basis.reshape(-1)[columns]
        operator += np.outer(vector, vector)
    return operator


def _gdv_sleqs3(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    inverse_guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Translate GDV ``SLEqS3`` for the non-projected, zero-guess path."""

    operator = np.asarray(matrix, dtype=float)
    target = np.asarray(right_hand_side, dtype=float).reshape(-1)
    inverse = np.asarray(inverse_guess, dtype=float).copy()
    solution = np.zeros_like(target)
    residual = target.copy()
    error = float(np.max(np.abs(target), initial=0.0))
    if error == 0.0:
        return solution, inverse, 0

    convergence = error * 1.0e-10
    convergence_squared = convergence * convergence
    best_solution = solution.copy()
    best_error = error
    previous_error = error
    error_change = error
    update_count = 0
    small_count = 0
    small_cosine = 0.2
    maximum_small_count = 13
    iteration = 0
    max_iterations = 10 * target.size

    def direction_scale(
        mapped_direction: np.ndarray,
        current_residual: np.ndarray,
    ) -> tuple[float, float] | None:
        mapped_norm2 = float(mapped_direction @ mapped_direction)
        residual_norm2 = float(current_residual @ current_residual)
        if mapped_norm2 == 0.0 or residual_norm2 == 0.0:
            return None
        scalar_product = float(mapped_direction @ current_residual)
        cosine = scalar_product / np.sqrt(mapped_norm2 * residual_norm2)
        return scalar_product / mapped_norm2, cosine

    while error > convergence and iteration <= max_iterations:
        direction = inverse @ residual
        mapped = operator @ direction
        update_status = _gdv_sr1_inverse_update(inverse, mapped, direction)
        if update_status == 0:
            update_count += 1

        scaling = direction_scale(mapped, residual)
        if scaling is None:
            iteration = max_iterations + 1
            break
        step_scale, cosine = scaling

        # SLEqS3 first accepts the current step and then tries the *raw*
        # residual when an unsuccessful SR1 update leaves a nearly orthogonal
        # search direction.  In particular, this is not MInv * residual.
        if abs(cosine) <= small_cosine and update_status != 0:
            solution += step_scale * direction
            mapped_solution = operator @ solution
            if update_status * update_count != 0:
                update_status = _gdv_sr1_inverse_update(
                    inverse, mapped_solution, solution
                )
                if update_status == 0:
                    update_count += 1
            residual = target - mapped_solution
            error = float(np.max(np.abs(residual), initial=0.0))
            direction = residual.copy()
            mapped = operator @ direction
            scaling = direction_scale(mapped, residual)
            if scaling is None:
                iteration = max_iterations + 1
                break
            step_scale, cosine = scaling

        if abs(cosine) <= small_cosine and error > convergence:
            small_count += 1
            if small_count > maximum_small_count and update_count != 0:
                # Literal MSmall recovery block: finish the current direction,
                # retain the best solution, reset X, and rebuild a direction
                # from MInv * Y before returning to the main cycle.
                update_count = 0
                small_cosine = 0.2
                small_count = 0
                solution += step_scale * direction
                mapped_solution = operator @ solution
                residual = target - mapped_solution
                error = float(np.max(np.abs(residual), initial=0.0))
                if error > best_error:
                    solution = best_solution.copy()
                    mapped_solution = operator @ solution
                    _gdv_sr1_inverse_update(inverse, mapped_solution, solution)
                    residual = target - mapped_solution
                    error = float(np.max(np.abs(residual), initial=0.0))
                elif error < best_error:
                    best_solution = solution.copy()
                    best_error = error

                solution.fill(0.0)
                direction = inverse @ target
                mapped = operator @ direction
                update_status = _gdv_sr1_inverse_update(inverse, mapped, direction)
                if update_status == 0:
                    update_count += 1
                else:
                    update_count = 0
                scaling = direction_scale(mapped, residual)
                if scaling is None:
                    iteration = max_iterations + 1
                    break
                step_scale, _cosine = scaling
                solution += step_scale * direction
                residual = target - operator @ solution
                error = float(np.max(np.abs(residual), initial=0.0))
                iteration += 1
                if error < best_error:
                    best_solution = solution.copy()
                    best_error = error
                continue
        else:
            small_count = 0

        if small_count > maximum_small_count and update_count == 0:
            small_cosine = 0.05
            small_count = 0

        solution += step_scale * direction
        if not (
            update_count == 0 and error_change <= convergence_squared
        ) or small_count != 0:
            residual = target - operator @ solution
            error = float(np.max(np.abs(residual), initial=0.0))
            error_change = abs(previous_error - error)
            previous_error = error
            iteration += 1
            if error < best_error:
                best_solution = solution.copy()
                best_error = error
            continue

        if best_error < error:
            solution = best_solution.copy()
            error = best_error
        if error > convergence:
            break

    status = 1 if iteration > max_iterations else 0
    return solution, inverse, status


def _gdv_sr1_inverse_update(
    inverse: np.ndarray,
    displacement: np.ndarray,
    response: np.ndarray,
) -> int:
    """Apply ``utilnz.F:SR1`` to the mutable SLEqS3 inverse guess."""

    dx = np.asarray(displacement, dtype=float).reshape(-1)
    dg = np.asarray(response, dtype=float).reshape(-1)
    dx_norm = float(np.linalg.norm(dx))
    dg_norm = float(np.linalg.norm(dg))
    if dx_norm == 0.0 or dg_norm == 0.0:
        return -1
    epsilon = (dg_norm / dx_norm) * 1.0e-10
    update_limit = (dg_norm / dx_norm) / 1.0e-10
    difference = dg - inverse @ dx
    difference_norm = float(np.linalg.norm(difference))
    if difference_norm / dg_norm <= 1.0e-10:
        return -1
    denominator = float(difference @ dx)
    if abs(denominator) <= epsilon:
        if abs(difference_norm / dg_norm) <= 1.0e-10:
            return 0
        if abs(denominator / (dx_norm * dg_norm)) <= epsilon:
            return 3
        if abs(difference_norm * difference_norm / denominator) > update_limit:
            return 4
    inverse_denominator = 1.0 / denominator
    if np.sqrt(abs(difference_norm * difference_norm * inverse_denominator)) >= update_limit:
        return 5
    inverse += np.outer(difference, difference) * inverse_denominator
    return 0


def _gdv_remove_translation_rotation(
    coordinates_bohr: np.ndarray,
    displacement_bohr: np.ndarray,
) -> np.ndarray:
    """Remove the six equal-weight null vectors used by GDV ``RemTR``."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    displacement = np.asarray(displacement_bohr, dtype=float).reshape(
        coordinates.shape
    ).copy()
    centered = coordinates - np.mean(coordinates, axis=0)
    _values, axes = np.linalg.eigh(centered.T @ centered)
    bases = []
    for axis in np.eye(3):
        bases.append(np.broadcast_to(axis, coordinates.shape))
    for index in range(3):
        bases.append(np.cross(centered, axes[:, index]))
    flat = displacement.reshape(-1)
    for basis in bases:
        vector = np.asarray(basis, dtype=float).reshape(-1)
        norm2 = float(vector @ vector)
        if norm2 > 1.0e-15:
            flat -= vector * float(flat @ vector) / norm2
    return flat


def constrained_internal_coordinate_step(
    coordinates_angstrom: np.ndarray,
    target_values: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    evaluate_values: Callable[[np.ndarray], np.ndarray] | None = None,
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
    protected = np.asarray(tuple(dict.fromkeys(int(item) for item in protected_indices)), dtype=int)
    if np.intersect1d(solve, protected).size:
        raise ValueError("solve and protected internal-coordinate sets must be disjoint")
    if solve.size and (np.min(solve) < 0 or np.max(solve) >= target.size):
        raise ValueError("solve internal-coordinate index is outside target vector")
    if protected.size and (np.min(protected) < 0 or np.max(protected) >= target.size):
        raise ValueError("protected internal-coordinate index is outside target vector")
    combinations = _validated_constraint_combinations(protected_combinations, target.size)
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
                if evaluate_values is None:
                    trial_values, trial_b = _validated_evaluation(
                        evaluate, trial_coords, target.size
                    )
                else:
                    trial_values = _validated_values(evaluate_values, trial_coords, target.size)
                    trial_b = None
                trial_residual = target - trial_values
                trial_hard = float(np.linalg.norm(trial_residual[solve]))
                trial_soft = float(
                    np.linalg.norm(
                        _protected_constraint_residual(trial_residual, protected, combinations)
                    )
                )
                score = trial_hard**2 + 100.0 * trial_soft**2
                if trial_hard < current_hard and score < current_score:
                    if trial_b is None:
                        trial_values, trial_b = _validated_evaluation(
                            evaluate, trial_coords, target.size
                        )
                        trial_residual = target - trial_values
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
    if relative_error > 0.75 or np.linalg.norm(correction) > max(
        0.75 * np.linalg.norm(matrix), 1.0e-8
    ):
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


def _validated_values(
    evaluate_values: Callable[[np.ndarray], np.ndarray],
    coordinates: np.ndarray,
    coordinate_count: int,
) -> np.ndarray:
    values = np.asarray(evaluate_values(coordinates), dtype=float).reshape(-1)
    if values.shape != (coordinate_count,):
        raise ValueError("internal-coordinate value evaluator returned incompatible dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("internal-coordinate value evaluator returned non-finite values")
    return values
