"""Curvilinear rovibrational corrections in nonredundant SONIC coordinates.

This module owns the representation-level result of the curvilinear
rovibrational calculation.  Geometry fitting remains in MORPHEUS and
electronic-structure execution remains in the corresponding backend adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


CURVILINEAR_DELTABVIB_SCHEMA = "matrix.trinity.curvilinear-deltabvib.v1"


@dataclass(frozen=True)
class CurvilinearAlphaComponents:
    """Mode-resolved vibration--rotation constants in MHz.

    Each array has shape ``(nvib, 3)`` in the principal-axis order A, B, C.
    Separating the metric, Coriolis and potential-curvature contributions makes
    the result auditable and avoids hiding a Cartesian approximation behind a
    nominally curvilinear label.
    """

    metric_MHz: np.ndarray
    coriolis_MHz: np.ndarray
    potential_MHz: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(value, dtype=float)
            for name, value in (
                ("metric_MHz", self.metric_MHz),
                ("coriolis_MHz", self.coriolis_MHz),
                ("potential_MHz", self.potential_MHz),
            )
        }
        shape = arrays["metric_MHz"].shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("curvilinear alpha components must have shape (nvib, 3)")
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("all curvilinear alpha components must have the same shape")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("curvilinear alpha components must be finite")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def total_MHz(self) -> np.ndarray:
        return self.metric_MHz + self.coriolis_MHz + self.potential_MHz


@dataclass(frozen=True)
class CurvilinearDeltaBVibResult:
    """One isotope-specific curvilinear Delta Bvib result.

    ``delta_MHz`` follows the MORPHEUS correction convention: it is subtracted
    from experimental ground-state constants to obtain semiexperimental
    equilibrium constants.
    """

    label: str
    substitutions: Mapping[int, int]
    frequencies_cm1: tuple[float, ...]
    alpha: CurvilinearAlphaComponents
    delta_MHz: tuple[float, float, float]
    representation: str
    source: str
    excluded_modes: tuple[int, ...] = ()
    schema: str = CURVILINEAR_DELTABVIB_SCHEMA


def curvilinear_deltabvib_from_alpha(
    label: str,
    substitutions: Mapping[int, int],
    frequencies_cm1,
    alpha: CurvilinearAlphaComponents,
    *,
    representation: str,
    source: str,
    exclude_modes: tuple[int, ...] = (),
    invert_imaginary_modes: bool = True,
) -> CurvilinearDeltaBVibResult:
    """Contract mode-resolved curvilinear alpha constants into Delta Bvib."""

    frequencies = np.asarray(frequencies_cm1, dtype=float)
    if frequencies.shape != (alpha.total_MHz.shape[0],):
        raise ValueError("one harmonic frequency is required for every SONIC normal mode")
    if representation not in {"Ir", "IIIr"}:
        raise ValueError("asymmetric-top representation must be Ir or IIIr")
    excluded = {int(mode) for mode in exclude_modes}
    if any(mode < 1 or mode > frequencies.size for mode in excluded):
        raise ValueError("excluded mode index lies outside the vibrational space")
    rows = alpha.total_MHz.copy()
    if invert_imaginary_modes:
        rows[frequencies < 0.0] *= -1.0
    if excluded:
        rows[np.asarray([index + 1 in excluded for index in range(frequencies.size)])] = 0.0
    delta = 0.5 * np.sum(rows, axis=0)
    return CurvilinearDeltaBVibResult(
        label=str(label),
        substitutions={int(key): int(value) for key, value in substitutions.items()},
        frequencies_cm1=tuple(float(value) for value in frequencies),
        alpha=alpha,
        delta_MHz=tuple(float(value) for value in delta),
        representation=representation,
        source=str(source),
        excluded_modes=tuple(sorted(excluded)),
    )


def curvilinear_deltabvib_to_dict(result: CurvilinearDeltaBVibResult) -> dict[str, object]:
    """Serialize the versioned, mode-resolved result without losing provenance."""

    return {
        "schema": result.schema,
        "label": result.label,
        "substitutions": {str(key): value for key, value in result.substitutions.items()},
        "frequencies_cm1": list(result.frequencies_cm1),
        "representation": result.representation,
        "source": result.source,
        "excluded_modes": list(result.excluded_modes),
        "alpha_components_MHz": {
            "metric": result.alpha.metric_MHz.tolist(),
            "coriolis": result.alpha.coriolis_MHz.tolist(),
            "potential": result.alpha.potential_MHz.tolist(),
            "total": result.alpha.total_MHz.tolist(),
        },
        "delta_MHz": list(result.delta_MHz),
    }


def curvilinear_deltabvib_from_dict(payload: Mapping[str, object]) -> CurvilinearDeltaBVibResult:
    if payload.get("schema") != CURVILINEAR_DELTABVIB_SCHEMA:
        raise ValueError("unsupported curvilinear DeltaBvib schema")
    components = payload.get("alpha_components_MHz")
    if not isinstance(components, Mapping):
        raise ValueError("curvilinear DeltaBvib payload lacks alpha components")
    alpha = CurvilinearAlphaComponents(
        metric_MHz=np.asarray(components["metric"], dtype=float),
        coriolis_MHz=np.asarray(components["coriolis"], dtype=float),
        potential_MHz=np.asarray(components["potential"], dtype=float),
    )
    substitutions = payload.get("substitutions", {})
    if not isinstance(substitutions, Mapping):
        raise ValueError("curvilinear isotope substitutions must be a mapping")
    stored_delta = tuple(float(value) for value in payload.get("delta_MHz", ()))
    if len(stored_delta) != 3:
        raise ValueError("curvilinear DeltaBvib payload requires three axis values")
    result = CurvilinearDeltaBVibResult(
        label=str(payload.get("label", "")),
        substitutions={int(key): int(value) for key, value in substitutions.items()},
        frequencies_cm1=tuple(float(value) for value in payload.get("frequencies_cm1", ())),
        alpha=alpha,
        delta_MHz=stored_delta,
        representation=str(payload.get("representation", "")),
        source=str(payload.get("source", "")),
        excluded_modes=tuple(int(value) for value in payload.get("excluded_modes", ())),
    )
    if len(result.frequencies_cm1) != alpha.total_MHz.shape[0]:
        raise ValueError("serialized frequencies and alpha rows disagree")
    return result


def read_curvilinear_deltabvib_results(path: Path | str) -> tuple[CurvilinearDeltaBVibResult, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "curvilinear" in payload:
        rows = payload["curvilinear"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = [payload]
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("curvilinear DeltaBvib file must contain an object or a list of objects")
    return tuple(curvilinear_deltabvib_from_dict(row) for row in rows)


__all__ = [
    "CURVILINEAR_DELTABVIB_SCHEMA",
    "CurvilinearAlphaComponents",
    "CurvilinearDeltaBVibResult",
    "curvilinear_deltabvib_from_alpha",
    "curvilinear_deltabvib_from_dict",
    "curvilinear_deltabvib_to_dict",
    "read_curvilinear_deltabvib_results",
]
