"""Finite-difference audits of the production analytic SONIC Wilson matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .definition import GICDefinition, build_gic_b_matrix, evaluate_gic_values


@dataclass(frozen=True)
class GICDerivativeAuditRow:
    identifier: str
    name: str
    family: str
    max_abs_error: float
    rms_error: float
    passed: bool


@dataclass(frozen=True)
class GICDerivativeAudit:
    step_angstrom: float
    tolerance: float
    rows: tuple[GICDerivativeAuditRow, ...]
    max_abs_error: float
    passed: bool

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({row.family for row in self.rows}))


def audit_gic_b_matrix_derivatives(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray | None = None,
    *,
    step_angstrom: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GICDerivativeAudit:
    """Compare every analytic SONIC B row with a Cartesian central difference.

    The audit differentiates the complete frozen SONIC definition rather than
    isolated primitives, so it also covers SALCs and ring-coordinate linear
    combinations.  Coordinate differences are reduced to the principal
    angular branch; Cartesian perturbations are sufficiently small that this
    is a no-op for distance-like coordinates.
    """
    step = float(step_angstrom)
    threshold = float(tolerance)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("SONIC derivative-audit step must be positive and finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("SONIC derivative-audit tolerance must be positive and finite")
    coords = np.asarray(
        definition.reference_coordinates_angstrom
        if coordinates_angstrom is None
        else coordinates_angstrom,
        dtype=float,
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("SONIC derivative-audit coordinates must have shape (natoms, 3)")

    analytic = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coords).rows,
        dtype=float,
    )
    numeric = np.zeros_like(analytic)
    for column in range(coords.size):
        plus = coords.reshape(-1).copy()
        minus = coords.reshape(-1).copy()
        plus[column] += step
        minus[column] -= step
        plus_values = evaluate_gic_values(
            definition, coordinates_angstrom=plus.reshape(coords.shape)
        )
        minus_values = evaluate_gic_values(
            definition, coordinates_angstrom=minus.reshape(coords.shape)
        )
        delta = np.asarray(plus_values - minus_values, dtype=float)
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        numeric[:, column] = delta / (2.0 * step)

    row_errors = analytic - numeric
    rows = tuple(
        GICDerivativeAuditRow(
            identifier=gic.identifier,
            name=gic.name,
            family=gic.family,
            max_abs_error=float(np.max(np.abs(error), initial=0.0)),
            rms_error=float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
            passed=bool(np.max(np.abs(error), initial=0.0) <= threshold),
        )
        for gic, error in zip(definition.gics, row_errors, strict=True)
    )
    maximum = max((row.max_abs_error for row in rows), default=0.0)
    return GICDerivativeAudit(
        step_angstrom=step,
        tolerance=threshold,
        rows=rows,
        max_abs_error=maximum,
        passed=all(row.passed for row in rows),
    )
