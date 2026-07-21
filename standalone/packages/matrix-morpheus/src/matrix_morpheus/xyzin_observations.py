from __future__ import annotations

from pathlib import Path

from matrix_core.isotopologues import (
    XYZIN_ISOTOPOLOGUES_SCHEMA,
    XyzinIsotopologueRecord,
    format_xyzin_isotopologue_issues,
    has_xyzin_isotopologues,
    parse_xyzin_isotopologue_records,
    read_xyzin_isotopologue_records,
    validate_xyzin_isotopologue_records,
    write_xyzin_isotopologue_records,
    xyzin_isotopologue_validation_errors,
    xyzin_isotopologue_section_lines,
)
from matrix_core.xyzin_geometry import read_xyzin_geometry

from .contracts import (
    ElectronicCorrection,
    IsotopologueObservation,
    RotationalConstants,
    VibrationalCorrection,
)


def read_xyzin_isotopologues(path: Path) -> tuple[IsotopologueObservation, ...]:
    target = Path(path)
    records = read_xyzin_isotopologue_records(target)
    atom_count = None
    try:
        atom_count = len(read_xyzin_geometry(target).atoms)
    except Exception:
        atom_count = None
    return observations_from_xyzin_records(records, atom_count=atom_count)


def write_xyzin_isotopologues(
    path: Path, observations: tuple[IsotopologueObservation, ...]
) -> Path:
    records = xyzin_records_from_observations(observations)
    return write_xyzin_isotopologue_records(Path(path), records)


def parse_xyzin_isotopologues_lines(lines: list[str]) -> tuple[IsotopologueObservation, ...]:
    return observations_from_xyzin_records(parse_xyzin_isotopologue_records(lines))


def observations_from_xyzin_records(
    records: tuple[XyzinIsotopologueRecord, ...],
    *,
    atom_count: int | None = None,
) -> tuple[IsotopologueObservation, ...]:
    issues = validate_xyzin_isotopologue_records(
        records, atom_count=atom_count, require_rotational=True
    )
    errors = xyzin_isotopologue_validation_errors(issues)
    if errors:
        raise ValueError(format_xyzin_isotopologue_issues(errors))
    observations: list[IsotopologueObservation] = []
    incomplete = [record.label for record in records if record.rotational_MHz is None]
    if incomplete:
        labels = ", ".join(incomplete)
        raise ValueError(
            "#ISOTOPOLOGUES can contain definition-only records, but SEfit "
            f"requires ROTATIONAL_MHZ for: {labels}"
        )
    for record in records:
        observations.append(_observation_from_record(record))
    return tuple(observations)


def xyzin_records_from_observations(
    observations: tuple[IsotopologueObservation, ...],
) -> tuple[XyzinIsotopologueRecord, ...]:
    records: list[XyzinIsotopologueRecord] = []
    for obs in observations:
        records.append(
            XyzinIsotopologueRecord(
                label=obs.label,
                substitutions=dict(obs.substitutions),
                rotational_MHz=obs.constants.as_tuple(),
                deltavib_MHz=obs.correction.as_tuple(),
                deltavib_source=obs.correction.source,
                deltavib_convention=obs.correction.convention,
                deltael_MHz=obs.electronic_correction.as_tuple(),
                deltael_source=obs.electronic_correction.source,
                deltael_convention=obs.electronic_correction.convention,
                sigma_MHz=_sigma_tuple(obs.weights),
            )
        )
    return tuple(records)


def _observation_from_record(record: XyzinIsotopologueRecord) -> IsotopologueObservation:
    assert record.rotational_MHz is not None
    return IsotopologueObservation(
        label=record.label,
        constants=RotationalConstants(*record.rotational_MHz),
        substitutions=dict(record.substitutions),
        correction=VibrationalCorrection(
            *(record.deltavib_MHz or (0.0, 0.0, 0.0)),
            source=record.deltavib_source,
            convention=record.deltavib_convention,
        ),
        electronic_correction=ElectronicCorrection(
            *(record.deltael_MHz or (0.0, 0.0, 0.0)),
            source=record.deltael_source,
            convention=record.deltael_convention,
        ),
        weights=_weights_tuple(record.sigma_MHz),
    )


def _sigma_tuple(weights: RotationalConstants | None) -> tuple[float, float, float] | None:
    if weights is None:
        return None
    return tuple((1.0 / value) ** 0.5 for value in weights.as_tuple())


def _weights_tuple(sigmas: tuple[float, float, float] | None) -> RotationalConstants | None:
    if sigmas is None:
        return None
    return RotationalConstants(*(1.0 / (sigma * sigma) for sigma in sigmas))
