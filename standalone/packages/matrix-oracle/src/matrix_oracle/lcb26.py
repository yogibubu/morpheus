"""ORACLE-side access to the unified LCB26 reference index.

ORACLE owns perception (including ring counts); this module only selects a
reference geometry/electronic record for a requested construction.  ZAFF
parameter generation remains an ARCHITECT operation.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

from matrix_switch import canonical_smiles, parse_smiles


class LCB26ReferenceError(ValueError):
    """Raised when an LCB26 reference cannot be selected."""


LCB26_L1_GEOMETRY_DATASET = "L1_GEOMETRY"


def _file_revision(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 resource: {path}") from exc
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=2048)
def _read_json_revision(
    path_text: str,
    _mtime_ns: int,
    _size: int,
) -> Any:
    path = Path(path_text)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB26ReferenceError(f"invalid LCB26 JSON resource: {path}") from exc


def _read_revisioned_json(path: Path) -> Any:
    return _read_json_revision(*_file_revision(path))


def _normal(value: str) -> str:
    return " ".join(
        str(value).casefold().replace("_", " ").replace("-", " ").replace(":", " ").split()
    )


def query_lcb26(
    library_root: Path,
    *,
    identifier: str | None = None,
    name: str | None = None,
    alias: str | None = None,
    smiles: str | None = None,
    formula: str | None = None,
    dataset: str | None = None,
    electronic_level: str | None = None,
    elements: Iterable[str] | None = None,
    element_counts: dict[str, int] | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    open_shell: bool | None = None,
    atom_count: int | None = None,
    heavy_atom_count: int | None = None,
    ring_count: int | None = None,
    contains: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select LCB26 records for ORACLE geometry construction."""

    if limit is not None and int(limit) <= 0:
        return ()
    index_path = Path(library_root).expanduser().resolve() / "enriched" / "index.json"
    payload = _read_revisioned_json(index_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records", ()), list):
        raise LCB26ReferenceError(f"invalid LCB26 query index: {index_path}")
    aliases = [_normal(value) for value in (identifier, name, alias) if value is not None]
    canonical = canonical_smiles(parse_smiles(smiles)) if smiles is not None else None
    required = {str(element) for element in elements or ()}
    required_counts = {
        str(element): int(count) for element, count in (element_counts or {}).items()
    }
    normalized_contains = _normal(contains) if contains is not None else None
    selected = []
    for row in payload.get("records", ()):
        if aliases and not all(value in row.get("normalized_aliases", ()) for value in aliases):
            continue
        if canonical is not None and row.get("canonical_smiles") != canonical:
            continue
        if formula is not None and str(row.get("formula", "")).casefold() != formula.casefold():
            continue
        if dataset is not None and str(row.get("dataset", "")).casefold() != dataset.casefold():
            continue
        if required and not required.issubset(set(row.get("elements", ()))):
            continue
        if any(
            row.get("element_counts", {}).get(element, 0) != count
            for element, count in required_counts.items()
        ):
            continue
        scalar_filters = {
            "electronic_level": electronic_level,
            "charge": charge,
            "multiplicity": multiplicity,
            "open_shell": open_shell,
            "atom_count": atom_count,
            "heavy_atom_count": heavy_atom_count,
            "ring_count": ring_count,
        }
        if any(
            expected is not None and str(row.get(key)).casefold() != str(expected).casefold()
            for key, expected in scalar_filters.items()
        ):
            continue
        if normalized_contains is not None and not any(
            normalized_contains in value for value in row.get("normalized_aliases", ())
        ):
            continue
        selected.append(deepcopy(row))
        if limit is not None and len(selected) >= int(limit):
            break
    return tuple(selected)


def load_lcb26_reference(library_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = Path(library_root).expanduser().resolve() / str(row["record_path"])
    payload = _read_revisioned_json(path)
    if not isinstance(payload, dict):
        raise LCB26ReferenceError(f"invalid LCB26 reference: {path}")
    # Public callers receive an isolated mutable mapping, while immutable JSON
    # parsing is shared across repeated molecule preparations.
    return deepcopy(payload)


def query_lcb26_l1_geometry(
    library_root: Path,
    *,
    identifier: str | None = None,
    elements: Iterable[str] | None = None,
    element_counts: dict[str, int] | None = None,
    atom_count: int | None = None,
    dispersion: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Query the geometry-only L1 archive.

    L1 records deliberately live outside :func:`query_lcb26`: they contain
    optimized geometries for paired L1/L2 calibration, but no CM5/Mayer
    populations.  The returned rows are lightweight and can be passed to
    :func:`load_lcb26_l1_geometry`.
    """
    if limit is not None and int(limit) <= 0:
        return ()
    root = Path(library_root).expanduser().resolve()
    manifest_path = root / "l1_geometries" / "manifest.json"
    payload = _read_revisioned_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise LCB26ReferenceError(f"invalid LCB26 L1 geometry archive: {manifest_path}")
    requested = _normal(identifier) if identifier is not None else None
    required = {str(element).capitalize() for element in elements or ()}
    required_counts = {str(element).capitalize(): int(count) for element, count in (element_counts or {}).items()}
    selected: list[dict[str, Any]] = []
    excluded_identifiers = {
        _normal(value) for value in payload.get("excluded_identifiers", ())
    }
    for source in payload["records"]:
        row = dict(source)
        if _normal(row.get("identifier", "")) in excluded_identifiers:
            continue
        row["dataset"] = LCB26_L1_GEOMETRY_DATASET
        row["geometry_path"] = row["file"]
        row["record_path"] = row["file"]
        row["reference_level"] = payload.get("reference_level", "L2")
        if requested is not None and requested not in _normal(row.get("identifier", "")):
            continue
        if dispersion is not None and str(row.get("dispersion", "")).casefold() != str(dispersion).casefold():
            continue
        if atom_count is not None and int(row.get("atom_count", -1)) != int(atom_count):
            continue
        symbols = _xyz_symbols(root / "l1_geometries" / str(row["file"]))
        counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
        if required and not required.issubset(counts):
            continue
        if any(counts.get(element, 0) != count for element, count in required_counts.items()):
            continue
        row["elements"] = tuple(counts)
        row["element_counts"] = counts
        row["normalized_aliases"] = (_normal(row.get("identifier", "")),)
        selected.append(row)
        if limit is not None and len(selected) >= int(limit):
            break
    return tuple(deepcopy(row) for row in selected)


def load_lcb26_l1_geometry(library_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load an L1 XYZ geometry row without inventing electronic properties."""
    root = Path(library_root).expanduser().resolve()
    relative = str(row.get("geometry_path", row.get("file", "")))
    path = (root / "l1_geometries" / relative).resolve()
    archive_root = (root / "l1_geometries").resolve()
    if archive_root not in path.parents:
        raise LCB26ReferenceError(f"L1 geometry escapes archive: {relative}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 L1 geometry: {path}") from exc
    result = deepcopy(row)
    result["geometry_xyz"] = text
    result["cm5_mayer_imported"] = False
    return result


def _xyz_symbols(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LCB26ReferenceError(f"missing LCB26 L1 geometry: {path}") from exc
    if not lines:
        raise LCB26ReferenceError(f"empty LCB26 L1 geometry: {path}")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise LCB26ReferenceError(f"invalid XYZ atom count: {path}") from exc
    symbols = [line.split()[0].capitalize() for line in lines[2:] if line.split()]
    if len(symbols) != count:
        raise LCB26ReferenceError(f"invalid XYZ atom rows in {path}")
    return symbols


__all__ = [
    "LCB26ReferenceError",
    "LCB26_L1_GEOMETRY_DATASET",
    "load_lcb26_l1_geometry",
    "load_lcb26_reference",
    "query_lcb26",
    "query_lcb26_l1_geometry",
]
