from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .contracts import IsotopologueObservation
from .io import observations_from_mapping


MICROWAVE_DATASET_SCHEMA = "matrix.morpheus.microwave_observations.v1"


@dataclass(frozen=True)
class MicrowaveLiteratureProvenance:
    citation: str
    doi: str
    source: str
    locator: str
    extracted_by: str


@dataclass(frozen=True)
class MicrowaveLiteratureDataset:
    path: Path
    provenance: MicrowaveLiteratureProvenance
    observations: tuple[IsotopologueObservation, ...]
    confirmed: bool
    confirmation_actor: str
    notes: tuple[str, ...] = ()


def read_microwave_literature_dataset(
    path: Path | str, *, require_confirmation: bool = True
) -> MicrowaveLiteratureDataset:
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("schema") != MICROWAVE_DATASET_SCHEMA:
        raise ValueError(f"microwave dataset must declare {MICROWAVE_DATASET_SCHEMA}")
    citation = data.get("literature")
    if not isinstance(citation, dict):
        raise ValueError("microwave dataset needs a literature provenance object")
    provenance = MicrowaveLiteratureProvenance(
        citation=str(citation.get("citation", "")).strip(),
        doi=str(citation.get("doi", "")).strip(),
        source=str(citation.get("source", "")).strip(),
        locator=str(citation.get("locator", "")).strip(),
        extracted_by=str(citation.get("extracted_by", "")).strip(),
    )
    missing = [
        name
        for name in ("citation", "source", "locator", "extracted_by")
        if not getattr(provenance, name)
    ]
    if missing:
        raise ValueError("microwave provenance is incomplete: " + ", ".join(missing))
    confirmation = data.get("confirmation", {})
    if not isinstance(confirmation, dict):
        raise ValueError("microwave dataset confirmation must be an object")
    confirmed = bool(confirmation.get("confirmed", False))
    actor = str(confirmation.get("actor", "")).strip()
    if require_confirmation and not confirmed:
        raise PermissionError(
            "experimental constants are staged but not confirmed; review the paper, units, "
            "isotopologue mapping and uncertainties first"
        )
    if confirmed and not actor:
        raise ValueError("a confirmed microwave dataset must record the confirmation actor")
    observations = observations_from_mapping(data)
    return MicrowaveLiteratureDataset(
        path=target,
        provenance=provenance,
        observations=observations,
        confirmed=confirmed,
        confirmation_actor=actor,
        notes=tuple(str(item) for item in data.get("notes", ())),
    )


def write_microwave_literature_template(
    path: Path | str,
    *,
    citation: str,
    doi: str = "",
    source: str,
    locator: str,
    extracted_by: str = "Keymaker deterministic extraction",
    isotopologues: list[dict] | None = None,
) -> Path:
    target = Path(path)
    payload = {
        "schema": MICROWAVE_DATASET_SCHEMA,
        "literature": {
            "citation": citation,
            "doi": doi,
            "source": source,
            "locator": locator,
            "extracted_by": extracted_by,
        },
        "confirmation": {"confirmed": False, "actor": ""},
        "isotopologues": isotopologues or [],
        "notes": [
            "Verify isotope-to-atom mapping, MHz units, quoted uncertainties and table/page."
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "MICROWAVE_DATASET_SCHEMA",
    "MicrowaveLiteratureDataset",
    "MicrowaveLiteratureProvenance",
    "read_microwave_literature_dataset",
    "write_microwave_literature_template",
]
