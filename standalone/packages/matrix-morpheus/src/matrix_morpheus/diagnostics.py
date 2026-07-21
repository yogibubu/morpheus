"""Public diagnostics helpers for SEfit outputs."""

from __future__ import annotations

import numpy as np

from .fit import (
    SemiexperimentalIterationTrace,
    iteration_trace_csv_rows,
    _svd_diagnostics_csv,
    _uncertainty_diagnostics_csv,
)


def svd_diagnostics_csv(labels: tuple[str, ...], weighted_jacobian: np.ndarray | None) -> str:
    return _svd_diagnostics_csv(labels, weighted_jacobian)


def uncertainty_diagnostics_csv(
    labels: tuple[str, ...],
    weighted_jacobian: np.ndarray | None,
    weighted_residual: np.ndarray | None,
) -> str:
    return _uncertainty_diagnostics_csv(labels, weighted_jacobian, weighted_residual)


__all__ = [
    "SemiexperimentalIterationTrace",
    "iteration_trace_csv_rows",
    "svd_diagnostics_csv",
    "uncertainty_diagnostics_csv",
]
