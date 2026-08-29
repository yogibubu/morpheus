"""MORPHEUS cache-or-calculate handoff for isotope-specific Delta Bvib."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Protocol

from matrix_chem.isotopologues import (
    XyzinIsotopologueRecord,
    read_xyzin_isotopologue_records,
    write_xyzin_isotopologue_records,
)

from .contracts import IsotopologueObservation, SemiexperimentalFitRequest
from .models import SemiexperimentalFitResult
from .xyzin_observations import read_xyzin_isotopologues


class DeltaBVibProvider(Protocol):
    def calculate(self, label: str, substitutions: dict[int, int]): ...


@dataclass(frozen=True)
class DeltaBVibResolutionResult:
    xyzin: Path
    observations: tuple[IsotopologueObservation, ...]
    trinity_invoked: bool
    cached_labels: tuple[str, ...]
    calculated_labels: tuple[str, ...]


@dataclass(frozen=True)
class TrinityAssistedSemiexperimentalFitResult:
    fit: SemiexperimentalFitResult
    deltabvib: DeltaBVibResolutionResult


def resolve_xyzin_deltabvib(
    xyzin: Path | str,
    trinity_job: DeltaBVibProvider | None = None,
) -> DeltaBVibResolutionResult:
    """Use complete XYZin corrections; otherwise request only missing rows from TRINITY."""

    target = Path(xyzin)
    records = read_xyzin_isotopologue_records(target)
    missing = tuple(record for record in records if record.deltavib_MHz is None)
    cached = tuple(record.label for record in records if record.deltavib_MHz is not None)
    if missing and trinity_job is None:
        labels = ", ".join(record.label for record in missing)
        raise ValueError(
            "XYZin lacks DELTAVIB_MHZ for "
            f"{labels}; supply a TRINITY DeltaBvib service before MORPHEUS fitting"
        )

    calculated: list[str] = []
    updated: list[XyzinIsotopologueRecord] = []
    for record in records:
        if record.deltavib_MHz is not None:
            updated.append(record)
            continue
        assert trinity_job is not None
        result = trinity_job.calculate(record.label, dict(record.substitutions))
        values = tuple(float(value) for value in result.delta_MHz)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("TRINITY must return finite DeltaBvib values for A, B and C")
        updated.append(
            replace(
                record,
                deltavib_MHz=values,
                deltavib_source=str(result.source),
                deltavib_convention="subtract",
            )
        )
        calculated.append(record.label)

    if calculated:
        write_xyzin_isotopologue_records(target, tuple(updated))
    observations = read_xyzin_isotopologues(target)
    return DeltaBVibResolutionResult(
        xyzin=target,
        observations=observations,
        trinity_invoked=bool(calculated),
        cached_labels=cached,
        calculated_labels=tuple(calculated),
    )


def fit_semiexperimental_with_trinity(
    request: SemiexperimentalFitRequest,
    *,
    xyzin: Path | str,
    trinity_job: DeltaBVibProvider | None = None,
    **fit_options,
) -> TrinityAssistedSemiexperimentalFitResult:
    """Resolve DeltaBvib first, then run the ordinary MORPHEUS fit unchanged."""

    resolution = resolve_xyzin_deltabvib(xyzin, trinity_job)
    fit_request = replace(
        request,
        initial_geometry=Path(xyzin),
        observations=resolution.observations,
    )
    from . import integrated_workflow

    fit = integrated_workflow.fit_semiexperimental_geometry(fit_request, **fit_options)
    return TrinityAssistedSemiexperimentalFitResult(fit=fit, deltabvib=resolution)


__all__ = [
    "DeltaBVibProvider",
    "DeltaBVibResolutionResult",
    "TrinityAssistedSemiexperimentalFitResult",
    "fit_semiexperimental_with_trinity",
    "resolve_xyzin_deltabvib",
]
