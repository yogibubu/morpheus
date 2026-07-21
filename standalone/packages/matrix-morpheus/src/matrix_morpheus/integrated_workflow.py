from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .contracts import (
    IsotopologueObservation,
    SemiexperimentalFitRequest,
    VibrationalCorrection,
)
from .fit import SemiexperimentalFitResult, fit_semiexperimental_geometry


REFERENCE_GEOMETRY_LEVELS = ("L1", "PL1", "PL2", "L2")


@dataclass(frozen=True)
class MorpheusGeometryLevels:
    """Keep the inexpensive rovibrational and accurate structural geometries distinct."""

    deltabvib_l0_geometry: Path
    structural_reference_geometry: Path
    structural_reference_level: str

    def validate(self) -> None:
        level = self.structural_reference_level.strip().upper()
        if level not in REFERENCE_GEOMETRY_LEVELS:
            raise ValueError(
                "structural reference level must be L1, PL1, PL2 or L2"
            )
        if not Path(self.deltabvib_l0_geometry).is_file():
            raise ValueError(f"L0 DeltaVib geometry not found: {self.deltabvib_l0_geometry}")
        if not Path(self.structural_reference_geometry).is_file():
            raise ValueError(
                f"structural reference geometry not found: {self.structural_reference_geometry}"
            )


@dataclass(frozen=True)
class R0PreflightResult:
    fit: SemiexperimentalFitResult
    observations: tuple[IsotopologueObservation, ...]
    warnings: tuple[str, ...]


def fit_ground_state_r0_geometry(
    request: SemiexperimentalFitRequest,
    **fit_options,
) -> R0PreflightResult:
    """Fit raw B0 constants while DeltaVib is running.

    This is deliberately a diagnostic/preflight structure, not the final
    semiexperimental equilibrium structure.  It exposes rank, conditioning,
    parameter uncertainties and unstable isotopologues early.
    """

    raw_observations = tuple(
        replace(
            observation,
            correction=VibrationalCorrection(
                0.0,
                0.0,
                0.0,
                source="R0 preflight: uncorrected ground-state constants",
                convention="subtract",
            ),
        )
        for observation in request.observations
    )
    raw_request = replace(request, observations=raw_observations)
    result = fit_semiexperimental_geometry(raw_request, **fit_options)
    warnings: list[str] = []
    diagnostics = result.diagnostics
    if diagnostics.rank < diagnostics.n_optimized_parameters:
        warnings.append(
            "R0 fit is rank deficient: some structural combinations are not determined "
            "by the supplied isotopologues."
        )
    if diagnostics.condition_number > 1.0e8:
        warnings.append(
            f"R0 fit is ill-conditioned (condition number {diagnostics.condition_number:.3g})."
        )
    if diagnostics.robust_downweighted_isotopologues:
        warnings.append(
            f"R0 fit downweighted {diagnostics.robust_downweighted_isotopologues} isotopologue(s)."
        )
    return R0PreflightResult(result, raw_observations, tuple(warnings))


def prepare_structural_reference(
    geometry: Path | str,
    *,
    level: str,
    output: Path | str | None = None,
) -> Path:
    """Return a predicate/reference geometry, refining L1 to PL1 when needed."""

    source = Path(geometry).expanduser().resolve()
    normalized = level.strip().upper()
    if normalized not in REFERENCE_GEOMETRY_LEVELS:
        raise ValueError("reference level must be L1, PL1, PL2 or L2")
    if normalized != "L1":
        return source
    if output is None:
        output = source.with_name(f"{source.stem}.pl1.xyzin")
    from matrix_oracle import refine_l1_geometry

    return refine_l1_geometry(source, output).output


__all__ = [
    "MorpheusGeometryLevels",
    "REFERENCE_GEOMETRY_LEVELS",
    "R0PreflightResult",
    "fit_ground_state_r0_geometry",
    "prepare_structural_reference",
]
