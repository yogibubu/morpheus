"""Numerical trust-region and Levenberg--Marquardt solvers for MORPHEUS."""

from __future__ import annotations

import numpy as np

from matrix_morpheus.numerics import limit_step, objective

from .models import TrustRegionStep


TRUST_REGION_MIN_RADIUS = 1.0e-10
DAMPING_MIN = 1.0e-14
DAMPING_MAX = 1.0e12


def _adaptive_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
    trust_radius: float,
) -> TrustRegionStep:
    if jac_weighted.size == 0 or jac_weighted.shape[1] == 0:
        return TrustRegionStep(
            np.zeros((0,), dtype=float), max(float(damping), 0.0), False, "empty"
        )
    step_result = _svd_trust_region_lm_step(jac_weighted, weighted_residual, damping, trust_radius)
    step = step_result.step
    predicted = _predicted_reduction(
        weighted_residual,
        jac_weighted,
        step,
        scale=1.0,
        current_objective=objective(weighted_residual),
    )
    if predicted <= 0.0 or not np.all(np.isfinite(step)):
        step = limit_step(_cauchy_step(jac_weighted, weighted_residual), trust_radius)
        step_result = TrustRegionStep(
            step,
            max(float(damping), 0.0),
            trust_radius > 0.0 and float(np.linalg.norm(step)) >= 0.99 * trust_radius,
            "cauchy_fallback",
        )
    return step_result


def _svd_trust_region_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
    trust_radius: float,
) -> TrustRegionStep:
    """Solve the rank-revealing LM trust-region subproblem in SVD coordinates.

    The local model is ``min 0.5 ||r - J p||^2`` with ``||p|| <= Delta``.
    When the Gauss-Newton step is inside the region, the minimum-norm
    rank-revealing step is used. Otherwise the Levenberg shift is found by
    robust bisection of the secular equation ``||p(mu)|| = Delta``. This is
    preferable to clipping a full LM step because the returned step is the
    solution of the regularized model for the active trust radius.
    """
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return TrustRegionStep(
            np.zeros((0,), dtype=float), max(float(damping), 0.0), False, "empty"
        )
    delta = float(trust_radius)
    if delta <= 0.0 or not np.isfinite(delta):
        step = _rank_revealing_lm_step(jac, residual, damping)
        return TrustRegionStep(step, max(float(damping), 0.0), False, "svd_rank_revealing_lm")
    delta = max(delta, TRUST_REGION_MIN_RADIUS)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        step = limit_step(_augmented_qr_lm_step(jac, residual, damping), delta)
        return TrustRegionStep(step, max(float(damping), 0.0), True, "augmented_qr_lm_fallback")
    if not singular.size:
        return TrustRegionStep(np.zeros(ncols, dtype=float), 0.0, False, "zero_jacobian")
    s0 = max(float(singular[0]), 1.0)
    tol = max(jac.shape) * np.finfo(float).eps * s0 * 100.0
    beta = u_matrix.T @ residual

    def step_for_shift(shift: float) -> np.ndarray:
        mu = max(float(shift), 0.0)
        factors = np.zeros_like(singular)
        keep = singular > tol
        factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
        return vh.T @ (factors * beta)

    unconstrained = step_for_shift(0.0)
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    if np.all(np.isfinite(unconstrained)) and unconstrained_norm <= delta:
        return TrustRegionStep(unconstrained, 0.0, False, "svd_more_hebden_trust_region")

    low = 0.0
    high = max(float(damping), s0 * s0 * 1.0e-12, DAMPING_MIN)
    high_step = step_for_shift(high)
    high_norm = float(np.linalg.norm(high_step))
    while (not np.isfinite(high_norm) or high_norm > delta) and high < DAMPING_MAX:
        low = high
        high = min(high * 4.0, DAMPING_MAX)
        high_step = step_for_shift(high)
        high_norm = float(np.linalg.norm(high_step))
    if not np.all(np.isfinite(high_step)):
        step = limit_step(_cauchy_step(jac, residual), delta)
        return TrustRegionStep(step, high, True, "cauchy_fallback")
    if high_norm > delta:
        step = limit_step(high_step, delta)
        return TrustRegionStep(step, high, True, "svd_more_hebden_trust_region_clipped")

    best_shift = high
    best_step = high_step
    for _ in range(80):
        mid = 0.5 * (low + high)
        trial = step_for_shift(mid)
        trial_norm = float(np.linalg.norm(trial))
        if not np.isfinite(trial_norm):
            low = mid
            continue
        best_shift = mid
        best_step = trial
        if abs(trial_norm - delta) <= max(1.0e-10 * delta, 1.0e-12):
            break
        if trial_norm > delta:
            low = mid
        else:
            high = mid
    if float(np.linalg.norm(best_step)) > delta * (1.0 + 1.0e-8):
        best_step = limit_step(best_step, delta)
    return TrustRegionStep(best_step, best_shift, True, "svd_more_hebden_trust_region")


def _rank_revealing_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    mu = max(float(damping), 0.0)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return _augmented_qr_lm_step(jac, residual, mu)
    if not singular.size:
        return np.zeros(ncols, dtype=float)
    tol = max(jac.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0) * 100.0
    projected = u_matrix.T @ residual
    factors = np.zeros_like(singular)
    keep = singular > tol
    factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
    step = vh.T @ (factors * projected)
    if step.shape[0] != ncols or not np.all(np.isfinite(step)):
        return _augmented_qr_lm_step(jac, residual, mu)
    return step


def _augmented_qr_lm_step(
    jac_weighted: np.ndarray, weighted_residual: np.ndarray, damping: float
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return np.zeros((0,), dtype=float)
    mu = max(float(damping), 0.0)
    if mu > 0.0:
        lhs = np.vstack([jac, np.sqrt(mu) * np.eye(ncols)])
        rhs = np.concatenate([residual, np.zeros(ncols, dtype=float)])
    else:
        lhs = jac
        rhs = residual
    try:
        step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _cauchy_step(jac, residual)
    return np.asarray(step, dtype=float)


def _cauchy_step(jac_weighted: np.ndarray, weighted_residual: np.ndarray) -> np.ndarray:
    gradient = jac_weighted.T @ weighted_residual
    if gradient.size == 0:
        return gradient
    jg = jac_weighted @ gradient
    denom = float(jg @ jg)
    if denom <= 0.0 or not np.isfinite(denom):
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 0.0 else gradient
    alpha = float((gradient @ gradient) / denom)
    return alpha * gradient


def _predicted_reduction(
    weighted_residual: np.ndarray,
    jac_weighted: np.ndarray,
    reduced_step: np.ndarray,
    *,
    scale: float,
    current_objective: float,
) -> float:
    if reduced_step.size == 0:
        return 0.0
    predicted_residual = weighted_residual - float(scale) * (jac_weighted @ reduced_step)
    predicted_objective = objective(predicted_residual)
    reduction = float(current_objective - predicted_objective)
    return reduction if np.isfinite(reduction) else 0.0


def _accepted_trust_update(
    damping: float,
    trust_radius: float,
    ratio: float,
    scale: float,
    step_norm: float,
    max_step: float,
) -> tuple[float, float]:
    ratio = float(ratio) if np.isfinite(ratio) else 0.0
    step_norm = float(step_norm) if np.isfinite(step_norm) else 0.0
    scale = float(scale) if np.isfinite(scale) else 0.0
    damping_floor = max(float(damping), DAMPING_MIN)
    if ratio < 0.25 or scale < 0.5:
        new_damping = min(max(damping_floor * 4.0, DAMPING_MIN), DAMPING_MAX)
        new_radius = _contracted_trust_radius(trust_radius, max_step, step_norm, 0.5)
    elif ratio > 0.75 and scale >= 0.9 and _step_near_trust_boundary(step_norm, trust_radius):
        new_damping = max(damping_floor / 3.0, DAMPING_MIN)
        new_radius = _expanded_trust_radius(trust_radius, max_step, step_norm)
    else:
        new_damping = max(damping_floor / 1.5, DAMPING_MIN)
        new_radius = trust_radius
    return new_damping, new_radius


def _rejected_trust_update(
    damping: float, trust_radius: float, max_step: float
) -> tuple[float, float]:
    damping_floor = max(float(damping), DAMPING_MIN)
    return min(damping_floor * 8.0, DAMPING_MAX), _scaled_trust_radius(trust_radius, max_step, 0.35)


def _scaled_trust_radius(trust_radius: float, max_step: float, scale: float) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    return max(float(current) * float(scale), _minimum_trust_radius(max_step))


def _contracted_trust_radius(
    trust_radius: float, max_step: float, step_norm: float, scale: float
) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    floor = _minimum_trust_radius(max_step)
    if step_norm > 0.0:
        proposed = min(float(current) * float(scale), max(2.0 * step_norm, floor))
    else:
        proposed = float(current) * float(scale)
    return max(float(proposed), floor)


def _expanded_trust_radius(trust_radius: float, max_step: float, step_norm: float) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    proposed = max(current * 1.8, step_norm * 2.0, _minimum_trust_radius(max_step))
    return min(float(max_step), float(proposed))


def _minimum_trust_radius(max_step: float) -> float:
    if max_step > 0.0 and np.isfinite(max_step):
        return max(TRUST_REGION_MIN_RADIUS, 1.0e-9 * float(max_step))
    return TRUST_REGION_MIN_RADIUS


def _step_near_trust_boundary(step_norm: float, trust_radius: float) -> bool:
    if trust_radius <= 0.0 or not np.isfinite(trust_radius):
        return False
    return float(step_norm) >= 0.80 * float(trust_radius)


def _trust_region_is_stalled(
    damping: float, trust_radius: float, stalled_rejections: int, max_step: float
) -> bool:
    if stalled_rejections < 5:
        return False
    if damping >= 0.999 * DAMPING_MAX:
        return True
    return trust_radius > 0.0 and trust_radius <= 10.0 * _minimum_trust_radius(max_step)


def _objective_has_stabilized(
    previous_objective: float | None, current_objective: float, tolerance_MHz: float
) -> bool:
    if previous_objective is None:
        return False
    if not np.isfinite(previous_objective) or not np.isfinite(current_objective):
        return False
    absolute = max(float(tolerance_MHz) * float(tolerance_MHz), 1.0e-14)
    relative = 1.0e-10 * max(1.0, abs(float(previous_objective)), abs(float(current_objective)))
    return abs(float(previous_objective) - float(current_objective)) <= max(absolute, relative)
