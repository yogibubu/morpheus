from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib

from .contracts import (
    ElectronicCorrection,
    IsotopologueObservation,
    RotationalConstants,
    VibrationalCorrection,
)
from .isotopologue_format import (
    format_substitutions,
    mass_number as _mass_number,
    parse_substitutions,
    sigma_text_from_weight,
)


CSV_FIELDS = (
    "label",
    "A_MHz",
    "B_MHz",
    "C_MHz",
    "delta_A_MHz",
    "delta_B_MHz",
    "delta_C_MHz",
    "correction_source",
    "correction_convention",
    "delta_elec_A_MHz",
    "delta_elec_B_MHz",
    "delta_elec_C_MHz",
    "electronic_correction_source",
    "electronic_correction_convention",
    "substitutions",
    "sigma_A_MHz",
    "sigma_B_MHz",
    "sigma_C_MHz",
)

REQUIRED_CSV_FIELDS = (
    "label",
    "A_MHz",
    "B_MHz",
    "C_MHz",
    "delta_A_MHz",
    "delta_B_MHz",
    "delta_C_MHz",
    "correction_source",
    "substitutions",
)


def read_observations(path: Path) -> tuple[IsotopologueObservation, ...]:
    target = Path(path)
    suffix = target.suffix.lower()
    from .msr_legacy import is_msr_legacy_file, read_msr_legacy_observations
    from .xyzin_observations import has_xyzin_isotopologues, read_xyzin_isotopologues

    if has_xyzin_isotopologues(target):
        return read_xyzin_isotopologues(target)
    if is_msr_legacy_file(target):
        return read_msr_legacy_observations(target)
    if suffix == ".csv":
        return read_observations_csv(target)
    if suffix == ".json":
        return read_observations_json(target)
    if suffix == ".toml":
        return read_observations_toml(target)
    raise ValueError("Semiexp observations must be .csv, .json, .toml, .msr or .msr.inp")


def read_observations_csv(path: Path) -> tuple[IsotopologueObservation, ...]:
    observations: list[IsotopologueObservation] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_CSV_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing semiexp CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            weights = _weights_from_sigmas(row)
            observations.append(
                IsotopologueObservation(
                    label=str(row["label"]).strip(),
                    constants=RotationalConstants(
                        float(row["A_MHz"]),
                        float(row["B_MHz"]),
                        float(row["C_MHz"]),
                    ),
                    correction=VibrationalCorrection(
                        float(row["delta_A_MHz"] or 0.0),
                        float(row["delta_B_MHz"] or 0.0),
                        float(row["delta_C_MHz"] or 0.0),
                        source=str(row["correction_source"] or "unspecified"),
                        convention=str(row.get("correction_convention") or "subtract"),
                    ),
                    electronic_correction=ElectronicCorrection(
                        float(row.get("delta_elec_A_MHz") or 0.0),
                        float(row.get("delta_elec_B_MHz") or 0.0),
                        float(row.get("delta_elec_C_MHz") or 0.0),
                        source=str(row.get("electronic_correction_source") or "unspecified"),
                        convention=str(row.get("electronic_correction_convention") or "subtract"),
                    ),
                    substitutions=parse_substitutions(str(row["substitutions"] or "")),
                    weights=weights,
                )
            )
    return tuple(observations)


def read_observations_json(path: Path) -> tuple[IsotopologueObservation, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return observations_from_mapping(data)


def read_observations_toml(path: Path) -> tuple[IsotopologueObservation, ...]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return observations_from_mapping(data)


def observations_from_mapping(data: dict) -> tuple[IsotopologueObservation, ...]:
    isotopologues = data.get("isotopologues")
    if not isinstance(isotopologues, list) or not isotopologues:
        raise ValueError("Structured semiexp input needs a non-empty isotopologues list")
    return tuple(_observation_from_mapping(item) for item in isotopologues)


def write_observations_csv(path: Path, observations: tuple[IsotopologueObservation, ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for obs in observations:
            writer.writerow(
                {
                    "label": obs.label,
                    "A_MHz": f"{obs.constants.A_MHz:.12g}",
                    "B_MHz": f"{obs.constants.B_MHz:.12g}",
                    "C_MHz": f"{obs.constants.C_MHz:.12g}",
                    "delta_A_MHz": f"{obs.correction.delta_A_MHz:.12g}",
                    "delta_B_MHz": f"{obs.correction.delta_B_MHz:.12g}",
                    "delta_C_MHz": f"{obs.correction.delta_C_MHz:.12g}",
                    "correction_source": obs.correction.source,
                    "correction_convention": obs.correction.convention,
                    "delta_elec_A_MHz": f"{obs.electronic_correction.delta_A_MHz:.12g}",
                    "delta_elec_B_MHz": f"{obs.electronic_correction.delta_B_MHz:.12g}",
                    "delta_elec_C_MHz": f"{obs.electronic_correction.delta_C_MHz:.12g}",
                    "electronic_correction_source": obs.electronic_correction.source,
                    "electronic_correction_convention": obs.electronic_correction.convention,
                    "substitutions": format_substitutions(obs.substitutions),
                    "sigma_A_MHz": _sigma_text(obs.weights.A_MHz) if obs.weights else "",
                    "sigma_B_MHz": _sigma_text(obs.weights.B_MHz) if obs.weights else "",
                    "sigma_C_MHz": _sigma_text(obs.weights.C_MHz) if obs.weights else "",
                }
            )
    return target


def corrected_constants_rows(
    observations: tuple[IsotopologueObservation, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for obs in observations:
        corrected = obs.corrected
        rows.append(
            {
                "label": obs.label,
                "A_e_MHz": corrected.A_MHz,
                "B_e_MHz": corrected.B_MHz,
                "C_e_MHz": corrected.C_MHz,
                "correction_source": obs.correction.source,
                "electronic_correction_source": obs.electronic_correction.source,
            }
        )
    return rows


def _observation_from_mapping(item: dict) -> IsotopologueObservation:
    constants = _rotconst_from_mapping(_required_mapping(item, "constants"))
    return IsotopologueObservation(
        label=str(item["label"]).strip(),
        constants=constants,
        substitutions=_observation_substitutions(item),
        correction=_vibrational_from_mapping(
            item.get("vibrational_correction", item.get("correction", {}))
        ),
        electronic_correction=_electronic_from_mapping(item.get("electronic_correction", {})),
        weights=_weights_from_sigma_mapping(item.get("sigma_MHz", item.get("sigma", {}))),
    )


def _required_mapping(item: dict, key: str) -> dict:
    value = item.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Structured semiexp isotopologue needs {key}")
    return value


def _rotconst_from_mapping(item: dict) -> RotationalConstants:
    return RotationalConstants(float(item["A_MHz"]), float(item["B_MHz"]), float(item["C_MHz"]))


def _vibrational_from_mapping(item: dict) -> VibrationalCorrection:
    return VibrationalCorrection(
        float(item.get("delta_A_MHz", 0.0)),
        float(item.get("delta_B_MHz", 0.0)),
        float(item.get("delta_C_MHz", 0.0)),
        source=str(item.get("source", "unspecified")),
        convention=str(item.get("convention", "subtract")),
    )


def _electronic_from_mapping(item: dict) -> ElectronicCorrection:
    return ElectronicCorrection(
        float(item.get("delta_A_MHz", 0.0)),
        float(item.get("delta_B_MHz", 0.0)),
        float(item.get("delta_C_MHz", 0.0)),
        source=str(item.get("source", "unspecified")),
        convention=str(item.get("convention", "subtract")),
    )


def _weights_from_sigma_mapping(item: dict) -> RotationalConstants | None:
    if not item:
        return None
    sigmas = (float(item["A_MHz"]), float(item["B_MHz"]), float(item["C_MHz"]))
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("Semiexp sigma values must be positive")
    return RotationalConstants(*(1.0 / (sigma * sigma) for sigma in sigmas))


def _observation_substitutions(item: dict) -> dict[int, int]:
    if "substitutions" in item:
        return _substitutions_from_mapping(item.get("substitutions", {}))
    definition = item.get("definition", item.get("isotopologue_definition", {}))
    if definition is None:
        return {}
    if isinstance(definition, str):
        text = definition.strip()
        if not text or text.lower() == "parent":
            return {}
        return _substitutions_from_mapping(text)
    if isinstance(definition, dict):
        if "substitutions" in definition:
            return _substitutions_from_mapping(definition.get("substitutions", {}))
        if "atom" in definition and "mass" in definition:
            return {int(definition["atom"]): _mass_number(definition["mass"])}
        if not definition:
            return {}
    return _substitutions_from_mapping(definition)


def _substitutions_from_mapping(value) -> dict[int, int]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        return parse_substitutions(value)
    if isinstance(value, dict):
        return {int(atom): _mass_number(mass) for atom, mass in value.items()}
    if isinstance(value, list):
        result = {}
        for item in value:
            if isinstance(item, dict):
                result[int(item["atom"])] = _mass_number(item["mass"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result[int(item[0])] = _mass_number(item[1])
            else:
                raise ValueError("Invalid substitution list entry")
        return result
    raise ValueError("Invalid semiexp substitutions format")


def _weights_from_sigmas(row: dict[str, str]) -> RotationalConstants | None:
    keys = ("sigma_A_MHz", "sigma_B_MHz", "sigma_C_MHz")
    raw = tuple(str(row.get(key, "") or "").strip() for key in keys)
    if not any(raw):
        return None
    if not all(raw):
        raise ValueError("Semiexp sigma columns must be all present or all empty")
    sigmas = tuple(float(item) for item in raw)
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("Semiexp sigma columns must be positive when provided")
    return RotationalConstants(*(1.0 / (sigma * sigma) for sigma in sigmas))


def _sigma_text(weight: float) -> str:
    return sigma_text_from_weight(weight)
