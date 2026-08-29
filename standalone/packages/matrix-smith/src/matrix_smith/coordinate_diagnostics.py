"""SONIC coordinate conditioning and singularity diagnostics."""

from __future__ import annotations

import numpy as np

from matrix_numerics import singular_spectrum

SONIC_CONDITION_DIAGNOSTICS_SCHEMA = "matrix.smith.sonic_condition_diagnostics.v2"


def sonic_condition_diagnostics(
    jacobian: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    absolute_tolerance: float = 0.0,
    warning_condition_number: float = 1.0e8,
) -> dict[str, object]:
    """Report scale-aware rank, conditioning and singularity state.

    The rank cutoff is ``max(abs_tol, rel_tol * s_max)``.  The legacy
    ``tolerance`` argument remains the relative tolerance so existing callers
    retain their numerical meaning.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("SONIC Jacobian must be a finite nonempty matrix")
    relative = float(tolerance)
    absolute = float(absolute_tolerance)
    warning = float(warning_condition_number)
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("SONIC relative tolerance must be positive and finite")
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("SONIC absolute tolerance must be finite and non-negative")
    if not np.isfinite(warning) or warning <= 1.0:
        raise ValueError("SONIC warning condition number must be greater than one")

    spectrum = singular_spectrum(
        matrix,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )
    singular = spectrum.singular_values
    cutoff = spectrum.cutoff
    rank = spectrum.rank
    minimum_active = spectrum.minimum_active
    condition = spectrum.condition_number
    if not rank:
        status = "SINGULAR"
    elif rank < min(matrix.shape):
        status = "RANK_DEFICIENT"
    elif condition >= warning:
        status = "ILL_CONDITIONED"
    else:
        status = "FULL_RANK"
    margin = float(minimum_active / cutoff) if cutoff and minimum_active else 0.0
    return {
        "schema": SONIC_CONDITION_DIAGNOSTICS_SCHEMA,
        "shape": list(matrix.shape),
        "rank": rank,
        "maximum_rank": min(matrix.shape),
        "singular_values": singular.tolist(),
        "singular_value_cutoff": cutoff,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "condition_number": condition,
        "minimum_active_singular_value": minimum_active,
        "cutoff_margin": margin,
        "status": status,
        "near_dependent": bool(status in {"RANK_DEFICIENT", "ILL_CONDITIONED", "SINGULAR"}),
    }


def normalized_sonic_condition_diagnostics(
    jacobian: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
    maximum_condition_number: float,
) -> dict[str, object]:
    """Audit a SONIC basis independently of heterogeneous coordinate units."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("SONIC Jacobian must be a finite nonempty matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("SONIC Jacobian contains a singular coordinate row")
    return sonic_condition_diagnostics(
        matrix / norms[:, None],
        tolerance=tolerance,
        warning_condition_number=maximum_condition_number,
    )
