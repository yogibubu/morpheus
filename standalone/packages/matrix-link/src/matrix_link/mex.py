"""Minimum-energy crossing (MEX) solver for LINK.

The solver is deliberately backend-neutral: a surface needs only an energy
and a gradient. Hessians are optional and are used only to seed the local
Lagrangian model; subsequent updates are quasi-Newton, so xTB E/G surfaces
are sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


MEX_SCHEMA = "matrix.link.mex_result.v1"


@dataclass(frozen=True)
class MEXSurfaceEvaluation:
    energy_hartree: float
    gradient: np.ndarray
    hessian: np.ndarray | None = None

    def __post_init__(self) -> None:
        energy = float(self.energy_hartree)
        gradient = np.asarray(self.gradient, dtype=float).reshape(-1)
        if not np.isfinite(energy) or gradient.size == 0 or not np.all(np.isfinite(gradient)):
            raise ValueError("MEX surface evaluation must contain finite energy and gradient")
        hessian = None if self.hessian is None else np.asarray(self.hessian, dtype=float)
        if hessian is not None and (
            hessian.shape != (gradient.size, gradient.size) or not np.all(np.isfinite(hessian))
        ):
            raise ValueError("MEX surface Hessian has the wrong shape or non-finite values")
        object.__setattr__(self, "energy_hartree", energy)
        object.__setattr__(self, "gradient", gradient.copy())
        object.__setattr__(self, "hessian", None if hessian is None else 0.5 * (hessian + hessian.T))


MEXSurface = Callable[[np.ndarray], MEXSurfaceEvaluation]


def mex_surface_from_link_service(
    service: object,
    *,
    request_hessian: bool = False,
    tag_prefix: str = "mex-surface",
) -> MEXSurface:
    """Adapt any LINK QM/MM ``GeometryEvaluationService`` to a MEX surface."""

    from .optimizer import optimizer_hessian_from_cartesian
    from .scan import ANGSTROM_TO_BOHR

    evaluation_counter = 0

    def evaluate(q: np.ndarray) -> MEXSurfaceEvaluation:
        nonlocal evaluation_counter
        evaluation_counter += 1
        properties = ("energy", "gradient", "hessian") if request_hessian else ("energy", "gradient")
        result = service.evaluate(
            np.asarray(q, dtype=float),
            tag=f"{tag_prefix}-{evaluation_counter:06d}",
            use_cache=True,
            persist_cache=True,
            requested_properties=properties,
        )
        if result.gradient_hartree_per_bohr is None:
            raise RuntimeError("MEX surface backend did not return a gradient")
        directions_bohr = service.coordinate_directions(result.coordinates_angstrom) / (
            1.0 / ANGSTROM_TO_BOHR
        )
        gradient = directions_bohr @ np.asarray(result.gradient_hartree_per_bohr, dtype=float)
        hessian = None
        if request_hessian and result.hessian_hartree_per_bohr2 is not None:
            hessian = optimizer_hessian_from_cartesian(
                result.hessian_hartree_per_bohr2,
                service.coordinate_model,
            )
        return MEXSurfaceEvaluation(result.energy_hartree, gradient, hessian)

    return evaluate


@dataclass(frozen=True)
class MEXIteration:
    iteration: int
    energy_hartree: float
    gap_hartree: float
    lagrange_multiplier: float
    gradient_inf_norm: float
    step_norm: float


@dataclass(frozen=True)
class MEXResult:
    converged: bool
    coordinates: np.ndarray
    energy_a_hartree: float
    energy_b_hartree: float
    gap_hartree: float
    lagrange_multiplier: float
    approximate_hessian: np.ndarray
    iterations: tuple[MEXIteration, ...]
    schema: str = MEX_SCHEMA


def optimize_mex(
    surface_a: MEXSurface,
    surface_b: MEXSurface,
    initial_coordinates: Sequence[float] | np.ndarray,
    *,
    max_iterations: int = 100,
    gradient_tolerance: float = 1.0e-5,
    gap_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-6,
    penalty: float = 10.0,
    active_indices: Sequence[int] | None = None,
    periodic_periods: Mapping[int, float] | None = None,
    inactive_minimizer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MEXResult:
    """Minimize two surfaces on an active seam subspace.

    ``inactive_minimizer`` may relax all non-active coordinates at each trial.
    Periodic active coordinates are wrapped to ``[-period/2, period/2)``.
    """

    q = np.asarray(initial_coordinates, dtype=float).reshape(-1).copy()
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("MEX initial coordinates must be finite and non-empty")
    if max_iterations <= 0 or penalty <= 0.0:
        raise ValueError("MEX iteration and penalty settings must be positive")
    active = np.arange(q.size, dtype=int) if active_indices is None else np.asarray(active_indices, dtype=int)
    if active.ndim != 1 or active.size == 0 or np.any(active < 0) or np.any(active >= q.size):
        raise ValueError("MEX active_indices must contain valid coordinate indices")
    if np.unique(active).size != active.size:
        raise ValueError("MEX active_indices must be unique")
    periods = {int(index): float(period) for index, period in dict(periodic_periods or {}).items()}
    if any(index not in set(active.tolist()) for index in periods):
        raise ValueError("periodic MEX coordinates must be active")
    if any(period <= 0.0 or not np.isfinite(period) for period in periods.values()):
        raise ValueError("periodic MEX periods must be finite and positive")

    def prepare(vector: np.ndarray) -> np.ndarray:
        candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
        for index, period in periods.items():
            candidate[index] = (candidate[index] + 0.5 * period) % period - 0.5 * period
        if inactive_minimizer is not None:
            candidate = np.asarray(inactive_minimizer(candidate), dtype=float).reshape(-1)
            if candidate.shape != q.shape or not np.all(np.isfinite(candidate)):
                raise ValueError("inactive_minimizer returned an invalid MEX coordinate vector")
        return candidate

    q = prepare(q)
    first = surface_a(q)
    second = surface_b(q)
    if first.gradient.shape != second.gradient.shape or first.gradient.shape != q.shape:
        raise ValueError("MEX surfaces must return gradients with the initial-coordinate shape")
    hessian = _initial_hessian(first, second)
    inactive = np.ones(q.size, dtype=bool)
    inactive[active] = False
    hessian[inactive, :] = 0.0
    hessian[:, inactive] = 0.0
    hessian[inactive, inactive] = np.eye(int(np.sum(inactive)))
    multiplier = 0.0
    records: list[MEXIteration] = []
    converged = False
    for iteration in range(max_iterations):
        average_energy = 0.5 * (first.energy_hartree + second.energy_hartree)
        gap = first.energy_hartree - second.energy_hartree
        lagrangian_gradient = 0.5 * (first.gradient + second.gradient) + (
            multiplier + penalty * gap
        ) * (first.gradient - second.gradient)
        lagrangian_gradient[inactive] = 0.0
        gradient_norm = float(np.max(np.abs(lagrangian_gradient)))
        if abs(gap) <= gap_tolerance and gradient_norm <= gradient_tolerance:
            converged = True
            records.append(MEXIteration(iteration, average_energy, gap, multiplier, gradient_norm, 0.0))
            break
        try:
            step = -np.linalg.solve(hessian, lagrangian_gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian) @ lagrangian_gradient
        step = np.asarray(step, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if step_norm > 0.5:
            step *= 0.5 / step_norm
            step_norm = 0.5
        accepted = False
        merit = average_energy + multiplier * gap + 0.5 * penalty * gap * gap
        scale = 1.0
        previous_q = q.copy()
        previous_gradient = lagrangian_gradient.copy()
        while scale >= 1.0e-5:
            trial_q = prepare(q + scale * step)
            trial_a = surface_a(trial_q)
            trial_b = surface_b(trial_q)
            trial_gap = trial_a.energy_hartree - trial_b.energy_hartree
            trial_merit = 0.5 * (trial_a.energy_hartree + trial_b.energy_hartree)
            trial_merit += multiplier * trial_gap + 0.5 * penalty * trial_gap * trial_gap
            if trial_merit < merit:
                q, first, second = trial_q, trial_a, trial_b
                trial_gradient = 0.5 * (first.gradient + second.gradient) + (
                    multiplier + penalty * trial_gap
                ) * (first.gradient - second.gradient)
                trial_gradient[inactive] = 0.0
                delta_q = q - previous_q
                delta_gradient = trial_gradient - previous_gradient
                hessian = _bfgs_update(hessian, delta_q, delta_gradient)
                accepted = True
                step_norm *= scale
                break
            scale *= 0.5
        if not accepted:
            step_norm = 0.0
        multiplier += penalty * (first.energy_hartree - second.energy_hartree)
        records.append(
            MEXIteration(
                iteration,
                0.5 * (first.energy_hartree + second.energy_hartree),
                first.energy_hartree - second.energy_hartree,
                multiplier,
                gradient_norm,
                step_norm,
            )
        )
        if (
            step_norm <= step_tolerance
            and abs(first.energy_hartree - second.energy_hartree) <= gap_tolerance
            and gradient_norm <= gradient_tolerance
        ):
            converged = True
            break
    return MEXResult(
        converged=converged,
        coordinates=q,
        energy_a_hartree=first.energy_hartree,
        energy_b_hartree=second.energy_hartree,
        gap_hartree=first.energy_hartree - second.energy_hartree,
        lagrange_multiplier=multiplier,
        approximate_hessian=hessian,
        iterations=tuple(records),
    )


def _initial_hessian(first: MEXSurfaceEvaluation, second: MEXSurfaceEvaluation) -> np.ndarray:
    if first.hessian is not None and second.hessian is not None:
        return 0.5 * (first.hessian + second.hessian)
    return np.eye(first.gradient.size, dtype=float)


def _bfgs_update(hessian: np.ndarray, delta_q: np.ndarray, delta_gradient: np.ndarray) -> np.ndarray:
    curvature = float(delta_gradient @ delta_q)
    if curvature <= 1.0e-12:
        return hessian
    h_delta = hessian @ delta_q
    denominator = float(delta_q @ h_delta)
    if denominator <= 1.0e-12:
        return hessian
    updated = hessian - np.outer(h_delta, h_delta) / denominator
    updated += np.outer(delta_gradient, delta_gradient) / curvature
    return 0.5 * (updated + updated.T)


__all__ = [
    "MEX_SCHEMA",
    "MEXIteration",
    "MEXResult",
    "MEXSurfaceEvaluation",
    "mex_surface_from_link_service",
    "optimize_mex",
]
