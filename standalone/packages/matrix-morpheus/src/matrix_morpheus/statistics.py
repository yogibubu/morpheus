from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Protocol

import numpy as np


class MeasurementLike(Protocol):
    labels: tuple[tuple[str, str], ...]
    observed: np.ndarray
    weights: np.ndarray
    n_experimental_rows: int


@dataclass(frozen=True)
class SemiexperimentalWeightDiagnostic:
    row: int
    kind: str
    isotopologue: str
    observable: str
    observed: float
    calculated: float
    residual: float
    sigma: float
    base_weight: float
    robust_weight: float
    effective_weight: float
    weighted_residual: float
    chi_square_contribution: float
    leverage: float
    studentized_residual: float
    cooks_distance: float
    total_weight_fraction: float


def leverage_values(weighted_jac: np.ndarray) -> np.ndarray:
    jac = np.asarray(weighted_jac, dtype=float)
    if jac.ndim != 2 or jac.size == 0:
        return np.zeros((jac.shape[0] if jac.ndim == 2 else 0,), dtype=float)
    normal_inv = np.linalg.pinv(jac.T @ jac, rcond=1.0e-10)
    return np.einsum("ij,jk,ik->i", jac, normal_inv, jac)


def weight_diagnostic_rows(
    model: MeasurementLike | None,
    calculated: np.ndarray,
    residual: np.ndarray,
    weighted_jac: np.ndarray | None,
    weighted_residual: np.ndarray | None,
    robust_sqrt_weights: np.ndarray | None,
) -> tuple[SemiexperimentalWeightDiagnostic, ...]:
    if model is None:
        return ()
    observed = np.asarray(model.observed, dtype=float)
    calculated_values = np.asarray(calculated, dtype=float)
    residual_values = np.asarray(residual, dtype=float)
    base_weights = np.asarray(model.weights, dtype=float)
    nrows = min(
        observed.size,
        calculated_values.size,
        residual_values.size,
        base_weights.size,
        len(model.labels),
    )
    if nrows == 0:
        return ()
    robust_weights = np.ones(nrows, dtype=float)
    if robust_sqrt_weights is not None:
        robust = np.asarray(robust_sqrt_weights, dtype=float)
        robust_weights[: min(nrows, robust.size)] = robust[: min(nrows, robust.size)] ** 2
    effective_weights = base_weights[:nrows] * robust_weights
    weighted = np.asarray(weighted_residual if weighted_residual is not None else (), dtype=float)
    if weighted.size < nrows:
        weighted_values = residual_values[:nrows] * np.sqrt(np.maximum(effective_weights, 0.0))
    else:
        weighted_values = weighted[:nrows]
    leverage = leverage_values(
        np.asarray(weighted_jac if weighted_jac is not None else np.zeros((0, 0)), dtype=float)
    )
    leverage_values_padded = np.zeros(nrows, dtype=float)
    leverage_values_padded[: min(nrows, leverage.size)] = leverage[: min(nrows, leverage.size)]
    total_effective_weight = float(np.sum(effective_weights[effective_weights > 0.0]))
    n_parameters = (
        int(weighted_jac.shape[1])
        if weighted_jac is not None and np.asarray(weighted_jac).ndim == 2
        else 0
    )
    rows: list[SemiexperimentalWeightDiagnostic] = []
    for idx in range(nrows):
        base_weight = float(base_weights[idx])
        robust_weight = float(robust_weights[idx])
        effective_weight = float(effective_weights[idx])
        weighted_value = float(weighted_values[idx])
        leverage_value = float(leverage_values_padded[idx])
        leverage_gap = max(1.0 - leverage_value, np.finfo(float).eps)
        studentized = weighted_value / np.sqrt(leverage_gap)
        cooks = (
            (weighted_value * weighted_value * leverage_value)
            / (max(n_parameters, 1) * leverage_gap * leverage_gap)
        )
        sigma = 1.0 / np.sqrt(base_weight) if base_weight > 0.0 else np.inf
        iso, obs = model.labels[idx]
        rows.append(
            SemiexperimentalWeightDiagnostic(
                row=idx + 1,
                kind="experimental" if idx < model.n_experimental_rows else "predicate",
                isotopologue=iso,
                observable=obs,
                observed=float(observed[idx]),
                calculated=float(calculated_values[idx]),
                residual=float(residual_values[idx]),
                sigma=float(sigma),
                base_weight=base_weight,
                robust_weight=robust_weight,
                effective_weight=effective_weight,
                weighted_residual=weighted_value,
                chi_square_contribution=weighted_value * weighted_value,
                leverage=leverage_value,
                studentized_residual=float(studentized),
                cooks_distance=float(cooks),
                total_weight_fraction=(
                    effective_weight / total_effective_weight
                    if total_effective_weight > 0.0
                    else 0.0
                ),
            )
        )
    return tuple(rows)


def weight_diagnostics_csv(rows: tuple[SemiexperimentalWeightDiagnostic, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "row",
            "kind",
            "isotopologue",
            "observable",
            "observed",
            "calculated",
            "residual",
            "sigma",
            "base_weight",
            "robust_weight",
            "effective_weight",
            "weighted_residual",
            "chi_square_contribution",
            "leverage",
            "studentized_residual",
            "cooks_distance",
            "total_weight_fraction",
        ]
    )
    for item in rows:
        writer.writerow(
            [
                item.row,
                item.kind,
                item.isotopologue,
                item.observable,
                f"{item.observed:.12g}",
                f"{item.calculated:.12g}",
                f"{item.residual:.12g}",
                f"{item.sigma:.12g}",
                f"{item.base_weight:.12g}",
                f"{item.robust_weight:.12g}",
                f"{item.effective_weight:.12g}",
                f"{item.weighted_residual:.12g}",
                f"{item.chi_square_contribution:.12g}",
                f"{item.leverage:.12g}",
                f"{item.studentized_residual:.12g}",
                f"{item.cooks_distance:.12g}",
                f"{item.total_weight_fraction:.12g}",
            ]
        )
    return stream.getvalue()
