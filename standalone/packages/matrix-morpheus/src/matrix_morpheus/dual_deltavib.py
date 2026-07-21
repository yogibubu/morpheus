"""MORPHEUS integration of Cartesian and curvilinear Delta Bvib channels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from matrix_trinity import CurvilinearDeltaBVibResult, curvilinear_deltabvib_to_dict

from .contracts import IsotopologueObservation, VibrationalCorrection
from .cubic_corrections import CartesianCubicCorrectionResult
from .io import write_observations_csv


MORPHEUS_DUAL_DELTABVIB_SCHEMA = "matrix.morpheus.isotopic-deltavib.v2"


@dataclass(frozen=True)
class DeltaBVibChannelComparison:
    label: str
    curvilinear_MHz: tuple[float, float, float]
    cartesian_MHz: tuple[float, float, float] | None
    difference_MHz: tuple[float, float, float] | None
    max_abs_difference_MHz: float | None
    within_tolerance: bool | None


@dataclass(frozen=True)
class DualDeltaBVibCorrectionResult:
    observations: tuple[IsotopologueObservation, ...]
    curvilinear: tuple[CurvilinearDeltaBVibResult, ...]
    comparisons: tuple[DeltaBVibChannelComparison, ...]
    output_csv: Path | None = None
    report_json: Path | None = None
    schema: str = MORPHEUS_DUAL_DELTABVIB_SCHEMA

    @property
    def validation_passed(self) -> bool:
        checked = [row.within_tolerance for row in self.comparisons if row.within_tolerance is not None]
        return all(checked) if checked else True


def combine_isotopic_deltabvib_channels(
    observations: tuple[IsotopologueObservation, ...],
    curvilinear: tuple[CurvilinearDeltaBVibResult, ...],
    *,
    cartesian: CartesianCubicCorrectionResult | None = None,
    comparison_tolerance_MHz: float = 5.0,
    output_csv: Path | str | None = None,
    report_json: Path | str | None = None,
) -> DualDeltaBVibCorrectionResult:
    """Use the SONIC result for refinement and Cartesian data for validation.

    The two representations are intentionally not averaged.  Curvilinear SONIC
    values define the MORPHEUS correction, while Cartesian values remain an
    independent diagnostic and retain the property-derivative/intensity route.
    """

    if comparison_tolerance_MHz < 0.0:
        raise ValueError("DeltaBvib comparison tolerance must be non-negative")
    by_label = {row.label: row for row in curvilinear}
    if len(by_label) != len(curvilinear):
        raise ValueError("duplicate curvilinear isotopologue labels")
    expected = {row.label for row in observations}
    if set(by_label) != expected:
        raise ValueError("curvilinear DeltaBvib labels must match the observations exactly")
    cartesian_by_label = {}
    if cartesian is not None:
        cartesian_by_label = {row.label: row for row in cartesian.corrections}
        if set(cartesian_by_label) != expected:
            raise ValueError("Cartesian DeltaBvib labels must match the observations exactly")

    updated: list[IsotopologueObservation] = []
    comparisons: list[DeltaBVibChannelComparison] = []
    for observation in observations:
        internal = by_label[observation.label]
        if dict(internal.substitutions) != dict(observation.substitutions):
            raise ValueError(f"isotope substitutions disagree for {observation.label}")
        delta = tuple(float(value) for value in internal.delta_MHz)
        updated.append(
            replace(
                observation,
                correction=VibrationalCorrection(
                    *delta,
                    source=(
                        f"curvilinear SONIC DeltaBvib ({internal.representation}); "
                        f"{internal.source}"
                    ),
                    convention="subtract",
                ),
            )
        )
        if cartesian is None:
            cart_delta = difference = None
            maximum = None
            passed = None
        else:
            cart_delta = tuple(
                float(value) for value in cartesian_by_label[observation.label].result.total_MHz
            )
            diff_array = np.asarray(delta) - np.asarray(cart_delta)
            difference = tuple(float(value) for value in diff_array)
            maximum = float(np.max(np.abs(diff_array)))
            passed = maximum <= comparison_tolerance_MHz
        comparisons.append(
            DeltaBVibChannelComparison(
                label=observation.label,
                curvilinear_MHz=delta,
                cartesian_MHz=cart_delta,
                difference_MHz=difference,
                max_abs_difference_MHz=maximum,
                within_tolerance=passed,
            )
        )

    csv_path = Path(output_csv) if output_csv is not None else None
    if csv_path is not None:
        write_observations_csv(csv_path, tuple(updated))
    json_path = Path(report_json) if report_json is not None else None
    result = DualDeltaBVibCorrectionResult(
        observations=tuple(updated),
        curvilinear=tuple(by_label[row.label] for row in observations),
        comparisons=tuple(comparisons),
        output_csv=csv_path,
        report_json=json_path,
    )
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(dual_deltabvib_to_dict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def dual_deltabvib_to_dict(result: DualDeltaBVibCorrectionResult) -> dict[str, object]:
    return {
        "schema": result.schema,
        "authoritative_channel": "curvilinear-sonic",
        "cartesian_channel_role": "intensities-and-independent-validation",
        "validation_passed": result.validation_passed,
        "curvilinear": [curvilinear_deltabvib_to_dict(row) for row in result.curvilinear],
        "comparisons": [
            {
                "label": row.label,
                "curvilinear_MHz": list(row.curvilinear_MHz),
                "cartesian_MHz": None if row.cartesian_MHz is None else list(row.cartesian_MHz),
                "difference_MHz": None if row.difference_MHz is None else list(row.difference_MHz),
                "max_abs_difference_MHz": row.max_abs_difference_MHz,
                "within_tolerance": row.within_tolerance,
            }
            for row in result.comparisons
        ],
    }


__all__ = [
    "MORPHEUS_DUAL_DELTABVIB_SCHEMA",
    "DeltaBVibChannelComparison",
    "DualDeltaBVibCorrectionResult",
    "combine_isotopic_deltabvib_channels",
    "dual_deltabvib_to_dict",
]
