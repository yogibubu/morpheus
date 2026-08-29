from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from matrix_chem.topology.elements import atomic_number, atomic_symbol

from .contracts import (
    ElectronicCorrection,
    IsotopologueObservation,
    RotationalConstants,
    VibrationalCorrection,
)
from .geometry_input import (
    SemiexperimentalGeometryInput,
    _looks_like_gic_expression_constraint,
    _modredundant_fixed_patterns,
)


MSR_LEGACY_SUFFIXES = (".msr", ".msr.inp", "_msr.inp")
MSR_SECTION_NAMES = {"bexp", "dbvib", "dvib", "weights", "dbelec", "dbele", "dbelec"}
MSR_CARTESIAN_CONSTRAINT_HEADERS = {
    "constraints",
    "constraint",
    "modredundant",
    "modredundant_constraints",
    "frozen",
    "freeze",
}
MSR_CARTESIAN_CONSTRAINT_END = {"end", "endconstraints", "endmodredundant"}

# Canonical import API. The *_legacy names below remain as input-format
# aliases so old MSR files and scripts continue to work.
MSR_INPUT_SUFFIXES = MSR_LEGACY_SUFFIXES


@dataclass(frozen=True)
class MSRLegacyInput:
    path: Path
    geometry: SemiexperimentalGeometryInput
    observations: tuple[IsotopologueObservation, ...]
    isotope_labels: tuple[str, ...]
    physical_atom_map: tuple[int, ...]
    controls: "MSRLegacyControls"
    diagnostic_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MSRLegacyControls:
    """Run controls carried by the route section of a legacy MSR input."""

    outlier: str = ""
    condition: str = ""
    propagation: str = ""

    @property
    def outlier_active(self) -> bool:
        return self.outlier.lower() == "active"

    @property
    def condition_active(self) -> bool:
        return self.condition.lower() == "active"


MSRInput = MSRLegacyInput
MSRControls = MSRLegacyControls


@dataclass(frozen=True)
class _ZMatrixLine:
    atom: str
    refs: tuple[int, int, int]
    distance_token: str = ""
    angle_token: str = ""
    dihedral_token: str = ""
    distance: float = 0.0
    angle: float = 0.0
    dihedral: float = 0.0


@dataclass(frozen=True)
class _IsotopeBlock:
    label: str
    masses: tuple[int, ...]


@dataclass(frozen=True)
class _SectionRow:
    values: tuple[float, float, float]
    label: str | None = None


def is_msr_legacy_file(path: Path) -> bool:
    target = Path(path)
    name = target.name.lower()
    if any(name.endswith(suffix) for suffix in MSR_LEGACY_SUFFIXES):
        return True
    if not target.is_file():
        return False
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    return _looks_like_msr_legacy_content(lines)


def read_msr_legacy_input(path: Path) -> MSRLegacyInput:
    target = Path(path)
    if not is_msr_legacy_file(target):
        raise ValueError("Input does not contain a recognizable legacy MSR job")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    controls = _parse_msr_legacy_controls(lines)
    geometry_kind, geometry_lines, variables, remainder = _split_msr_legacy_lines(lines)
    if geometry_kind == "cartesian":
        atoms, coords = _parse_cartesian_geometry(geometry_lines)
        physical_map = tuple(range(1, len(atoms) + 1))
        modredundant, remainder = _extract_cartesian_constraints(remainder)
        fixed_parameters = _modredundant_fixed_patterns(modredundant)
        diagnostic_constraints: tuple[str, ...] = ()
        source_format = "msr_legacy_cartesian"
    else:
        zmat = _parse_zmatrix(geometry_lines, variables)
        atoms, coords, physical_map = _zmatrix_to_cartesian(zmat)
        expression_definitions = _zmatrix_gic_definitions(zmat, physical_map)
        diagnostic_constraints, remainder = _extract_cartesian_constraints(remainder)
        fixed_parameters = (
            *expression_definitions,
            *_zmatrix_frozen_constraints(zmat, physical_map),
        )
        source_format = "msr_legacy_zmatrix"
    isotope_blocks, sections = _parse_isotopes_and_sections(remainder, natoms=len(atoms))
    observations = _build_observations(isotope_blocks, sections)
    geometry = SemiexperimentalGeometryInput(
        tuple(atoms),
        np.asarray(coords, dtype=float),
        target.stem,
        fixed_parameters,
        source_format,
    )
    return MSRLegacyInput(
        target,
        geometry,
        observations,
        tuple(block.label for block in isotope_blocks),
        physical_map,
        controls,
        diagnostic_constraints,
    )


def _looks_like_msr_legacy_content(lines: Iterable[str]) -> bool:
    stripped = tuple(line.strip().lower() for line in lines)
    has_route = any(line.startswith("#m") for line in stripped)
    has_observations = any(_is_section_header(line) for line in stripped)
    has_isotopes = any(line.startswith("#d") and "niso" in line for line in stripped)
    return has_route and has_observations and has_isotopes


def _parse_msr_legacy_controls(lines: Iterable[str]) -> MSRLegacyControls:
    header: list[str] = []
    for raw in lines:
        line = raw.strip()
        parts = _split_fields(line)
        if line and not line.startswith("#") and parts and _is_zmatrix_atom_token(parts[0]):
            break
        header.append(line)
    route = " ".join(header).lower()
    values = {
        key: match.group(1)
        for key in ("outlier", "condition", "propagation")
        if (match := re.search(rf"\b{key}\s*=\s*([a-z0-9_-]+)", route)) is not None
    }
    return MSRLegacyControls(**values)


def read_msr_legacy_geometry(path: Path) -> SemiexperimentalGeometryInput:
    return read_msr_legacy_input(path).geometry


def read_msr_legacy_observations(path: Path) -> tuple[IsotopologueObservation, ...]:
    return read_msr_legacy_input(path).observations


is_msr_file = is_msr_legacy_file
read_msr_input = read_msr_legacy_input
read_msr_geometry = read_msr_legacy_geometry
read_msr_observations = read_msr_legacy_observations


def _split_msr_legacy_lines(
    lines: Iterable[str],
) -> tuple[str, list[str], dict[str, float], list[str]]:
    stripped = [line.strip() for line in lines]
    idx = 0
    while idx < len(stripped):
        line = stripped[idx]
        parts = _split_fields(line)
        if line and not line.startswith("#") and parts and _is_zmatrix_atom_token(parts[0]):
            break
        idx += 1
    if idx >= len(stripped):
        raise ValueError("MSR input contains no geometry block")

    if _is_cartesian_line(stripped[idx]):
        cartesian_lines: list[str] = []
        while idx < len(stripped):
            line = stripped[idx]
            if not line:
                idx += 1
                if cartesian_lines:
                    break
                continue
            if not _is_cartesian_line(line):
                break
            cartesian_lines.append(line)
            idx += 1
        return "cartesian", cartesian_lines, {}, stripped[idx:]

    zmat_lines: list[str] = []
    while idx < len(stripped):
        line = stripped[idx]
        if not line:
            idx += 1
            if zmat_lines:
                break
            continue
        if "=" in line or _is_section_header(line) or _first_token_is_float(line):
            break
        parts = _split_fields(line)
        if not parts or not _is_zmatrix_atom_token(parts[0]):
            break
        zmat_lines.append(line)
        idx += 1

    variables: dict[str, float] = {}
    while idx < len(stripped):
        line = stripped[idx]
        if not line:
            idx += 1
            continue
        if (
            _is_section_header(line)
            or _first_token_is_float(line)
            or _is_cartesian_constraint_header(line)
        ):
            break
        parsed = _parse_variable_assignment(line)
        if parsed is None:
            raise ValueError(f"Unexpected MSR line before isotope blocks: {line}")
        name, value = parsed
        variables[name] = value
        idx += 1

    return "zmatrix", zmat_lines, variables, stripped[idx:]


def _parse_zmatrix(lines: list[str], variables: dict[str, float]) -> tuple[_ZMatrixLine, ...]:
    if not lines:
        raise ValueError("MSR Z-matrix block is empty")
    parsed: list[_ZMatrixLine] = []
    for idx, line in enumerate(lines):
        parts = _split_fields(line)
        atom = _normalize_atom_token(parts[0])
        if idx == 0:
            parsed.append(_ZMatrixLine(atom, (0, 0, 0)))
        elif idx == 1:
            _require_fields(parts, 3, line)
            ref1 = _parse_zmatrix_reference(parts[1], idx, line)
            parsed.append(
                _ZMatrixLine(
                    atom,
                    (ref1, 0, 0),
                    distance_token=parts[2],
                    distance=_resolve_value(parts[2], variables),
                )
            )
        elif idx == 2:
            _require_fields(parts, 5, line)
            ref1 = _parse_zmatrix_reference(parts[1], idx, line)
            ref2 = _parse_zmatrix_reference(parts[3], idx, line)
            parsed.append(
                _ZMatrixLine(
                    atom,
                    (ref1, ref2, 0),
                    distance_token=parts[2],
                    angle_token=parts[4],
                    distance=_resolve_value(parts[2], variables),
                    angle=_resolve_value(parts[4], variables),
                )
            )
        else:
            _require_fields(parts, 7, line)
            ref1 = _parse_zmatrix_reference(parts[1], idx, line)
            ref2 = _parse_zmatrix_reference(parts[3], idx, line)
            ref3 = _parse_zmatrix_reference(parts[5], idx, line)
            parsed.append(
                _ZMatrixLine(
                    atom,
                    (ref1, ref2, ref3),
                    distance_token=parts[2],
                    angle_token=parts[4],
                    dihedral_token=parts[6],
                    distance=_resolve_value(parts[2], variables),
                    angle=_resolve_value(parts[4], variables),
                    dihedral=_resolve_value(parts[6], variables),
                )
            )
    _validate_zmatrix_rows(parsed)
    return tuple(parsed)


def _parse_cartesian_geometry(
    lines: list[str],
) -> tuple[list[str], list[tuple[float, float, float]]]:
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in lines:
        parts = _split_fields(line)
        if not _is_cartesian_line(line):
            raise ValueError(f"Invalid MSR Cartesian geometry line: {line}")
        atom = _normalize_atom_token(parts[0])
        if _is_dummy_atom(atom):
            raise ValueError("MSR Cartesian geometry cannot contain dummy atoms")
        atoms.append(atom)
        coords.append((_parse_float(parts[1]), _parse_float(parts[2]), _parse_float(parts[3])))
    if not atoms:
        raise ValueError("MSR Cartesian geometry block is empty")
    return atoms, coords


def _zmatrix_to_cartesian(
    zmat: tuple[_ZMatrixLine, ...],
) -> tuple[list[str], list[tuple[float, float, float]], tuple[int, ...]]:
    coords: list[np.ndarray] = []
    for idx, row in enumerate(zmat):
        if idx == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
        elif idx == 1:
            p1 = coords[row.refs[0]]
            coords.append(p1 + np.array([0.0, 0.0, row.distance], dtype=float))
        elif idx == 2:
            p1 = coords[row.refs[0]]
            p2 = coords[row.refs[1]]
            ez, ey, _normal = _zmatrix_reference_frame(p1, p2)
            theta = math.radians(row.angle)
            coords.append(p1 + row.distance * (math.cos(theta) * ez + math.sin(theta) * ey))
        else:
            p1 = coords[row.refs[0]]
            p2 = coords[row.refs[1]]
            p3 = coords[row.refs[2]]
            ez, ey, normal = _zmatrix_reference_frame(p1, p2, p3)
            theta = math.radians(row.angle)
            phi = math.radians(row.dihedral)
            coords.append(
                p1
                + row.distance
                * (
                    math.cos(theta) * ez
                    + math.sin(theta) * math.cos(phi) * ey
                    + math.sin(theta) * math.sin(phi) * normal
                )
            )

    atoms: list[str] = []
    clean_coords: list[tuple[float, float, float]] = []
    physical_map: list[int] = []
    for zidx, (row, xyz) in enumerate(zip(zmat, coords), start=1):
        if _is_dummy_atom(row.atom):
            continue
        atoms.append(row.atom)
        clean_coords.append(tuple(float(value) for value in xyz))
        physical_map.append(zidx)
    if not atoms:
        raise ValueError("MSR Z-matrix contains no real atoms")
    return atoms, clean_coords, tuple(physical_map)


def _zmatrix_frozen_constraints(
    zmat: tuple[_ZMatrixLine, ...], physical_map: tuple[int, ...]
) -> tuple[str, ...]:
    z_to_physical = {
        z_index: phys_index for phys_index, z_index in enumerate(physical_map, start=1)
    }
    constraints: list[str] = []
    seen: set[str] = set()
    for z_zero, row in enumerate(zmat):
        z_index = z_zero + 1
        if _is_frozen_token(row.distance_token) and z_zero >= 1:
            atoms = _physical_atoms(z_to_physical, z_index, row.refs[0] + 1)
            if atoms is not None:
                i, j = sorted(atoms[:2])
                _append_unique_constraint(constraints, seen, f"R({i},{j}) Frozen")
        if _is_frozen_token(row.angle_token) and z_zero >= 2:
            atoms = _physical_atoms(z_to_physical, z_index, row.refs[0] + 1, row.refs[1] + 1)
            if atoms is not None:
                _append_unique_constraint(
                    constraints, seen, f"A({atoms[0]},{atoms[1]},{atoms[2]}) Frozen"
                )
        if _is_frozen_token(row.dihedral_token) and z_zero >= 3:
            atoms = _physical_atoms(
                z_to_physical, z_index, row.refs[0] + 1, row.refs[1] + 1, row.refs[2] + 1
            )
            if atoms is not None:
                _append_unique_constraint(
                    constraints, seen, f"D({atoms[0]},{atoms[1]},{atoms[2]},{atoms[3]}) Frozen"
                )
    return tuple(constraints)


def _zmatrix_gic_definitions(
    zmat: tuple[_ZMatrixLine, ...], physical_map: tuple[int, ...]
) -> tuple[str, ...]:
    z_to_physical = {
        z_index: phys_index for phys_index, z_index in enumerate(physical_map, start=1)
    }
    definitions: list[str] = []
    seen: set[str] = set()
    for z_zero, row in enumerate(zmat):
        z_index = z_zero + 1
        if z_zero >= 1:
            atoms = _physical_atoms(z_to_physical, z_index, row.refs[0] + 1)
            if atoms is not None:
                i, j = sorted(atoms[:2])
                _append_zmatrix_definition(
                    definitions, seen, row.distance_token, f"R({i},{j})"
                )
        if z_zero >= 2:
            atoms = _physical_atoms(z_to_physical, z_index, row.refs[0] + 1, row.refs[1] + 1)
            if atoms is not None:
                _append_zmatrix_definition(
                    definitions, seen, row.angle_token, f"A({atoms[0]},{atoms[1]},{atoms[2]})"
                )
        if z_zero >= 3:
            atoms = _physical_atoms(
                z_to_physical, z_index, row.refs[0] + 1, row.refs[1] + 1, row.refs[2] + 1
            )
            if atoms is not None:
                _append_zmatrix_definition(
                    definitions,
                    seen,
                    row.dihedral_token,
                    f"D({atoms[0]},{atoms[1]},{atoms[2]},{atoms[3]})",
                )
    return tuple(definitions)


def _append_zmatrix_definition(
    definitions: list[str], seen: set[str], token: str, expression: str
) -> None:
    name = _clean_variable_name(token)
    if not name or name in seen or name.startswith("#"):
        return
    try:
        _parse_float(name)
    except ValueError:
        pass
    else:
        return
    definitions.append(f"{name}={expression}")
    seen.add(name)


def _physical_atoms(z_to_physical: dict[int, int], *z_indices: int) -> tuple[int, ...] | None:
    atoms: list[int] = []
    for z_index in z_indices:
        phys_index = z_to_physical.get(z_index)
        if phys_index is None:
            return None
        atoms.append(phys_index)
    return tuple(atoms)


def _append_unique_constraint(constraints: list[str], seen: set[str], item: str) -> None:
    if item not in seen:
        constraints.append(item)
        seen.add(item)


def _extract_cartesian_constraints(lines: list[str]) -> tuple[tuple[str, ...], list[str]]:
    idx = 0
    constraints: list[str] = []
    in_explicit_block = False
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if _is_section_header(line) or _first_token_is_float(line):
            break
        if _is_cartesian_constraint_header(line):
            in_explicit_block = True
            idx += 1
            continue
        if _is_cartesian_constraint_end(line):
            idx += 1
            break
        if _is_modredundant_constraint_line(line):
            constraints.append(line)
            idx += 1
            continue
        if _looks_like_gic_expression_constraint(line):
            constraints.append(line)
            idx += 1
            continue
        if in_explicit_block and _looks_like_legacy_expression_constraint(line):
            constraints.append(f"{line} Frozen")
            idx += 1
            continue
        if in_explicit_block:
            raise ValueError(f"Invalid MSR Cartesian constraint line: {line}")
        break
    return tuple(constraints), lines[idx:]


def _is_cartesian_constraint_header(line: str) -> bool:
    parts = _split_fields(line)
    return (
        len(parts) == 1
        and parts[0].lower().rstrip(":") in MSR_CARTESIAN_CONSTRAINT_HEADERS
    )


def _is_cartesian_constraint_end(line: str) -> bool:
    parts = _split_fields(line)
    return len(parts) == 1 and parts[0].lower().rstrip(":") in MSR_CARTESIAN_CONSTRAINT_END


def _looks_like_legacy_expression_constraint(line: str) -> bool:
    text = _strip_inline_comment(line).strip()
    if not text:
        return False
    return bool(
        re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_#'\"]*(?:\s*[+-]\s*[A-Za-z_][A-Za-z0-9_#'\"]*)+",
            text,
        )
    )


def _is_modredundant_constraint_line(line: str) -> bool:
    parts = _split_fields(line)
    if not parts:
        return False
    kind = parts[0].upper()
    expected = {"B": 2, "A": 3, "D": 4, "O": 4, "L": 3}.get(kind)
    if expected is None:
        return False
    atoms = 0
    actions: list[str] = []
    for token in parts[1:]:
        try:
            int(token)
            atoms += 1
        except ValueError:
            actions.append(token.upper())
    return atoms >= expected and "F" in actions


def _parse_isotopes_and_sections(
    lines: list[str], *, natoms: int
) -> tuple[tuple[_IsotopeBlock, ...], dict[str, tuple[_SectionRow, ...]]]:
    rows = [line for line in lines if line.strip()]
    idx = 0
    isotope_blocks: list[_IsotopeBlock] = []
    while idx < len(rows) and not _is_section_header(rows[idx]):
        block_lines = rows[idx : idx + natoms]
        if len(block_lines) < natoms:
            raise ValueError("Incomplete MSR isotope mass block")
        masses: list[int] = []
        label = _optional_label(block_lines[0])
        for raw in block_lines:
            parts = _split_fields(raw)
            try:
                masses.append(_mass_number(_parse_float(parts[0])))
            except Exception as exc:
                raise ValueError(f"Invalid MSR isotope mass line: {raw}") from exc
        if not label:
            label = "parent" if not isotope_blocks else f"iso_{len(isotope_blocks) + 1:03d}"
        isotope_blocks.append(_IsotopeBlock(label, tuple(masses)))
        idx += natoms

    sections: dict[str, tuple[_SectionRow, ...]] = {}
    while idx < len(rows):
        if _is_legacy_trailing_directive(rows[idx]):
            break
        if not _is_section_header(rows[idx]):
            raise ValueError(f"Expected MSR section header, found: {rows[idx]}")
        section = rows[idx].split()[0].lower()
        idx += 1
        section_rows: list[_SectionRow] = []
        while (
            idx < len(rows)
            and not _is_section_header(rows[idx])
            and not _is_legacy_trailing_directive(rows[idx])
        ):
            section_rows.append(_parse_section_row(rows[idx]))
            idx += 1
        sections[section] = tuple(section_rows)
    if not isotope_blocks:
        raise ValueError("MSR input contains no isotope mass blocks")
    if "bexp" not in sections:
        raise ValueError("MSR input needs a bexp section")
    return tuple(isotope_blocks), sections


def _is_legacy_trailing_directive(line: str) -> bool:
    parts = _split_fields(line)
    if not parts:
        return False
    head = parts[0].lower()
    return head in {"propagation", "redundant", "coordinates", "coordinate"}


def _build_observations(
    isotope_blocks: tuple[_IsotopeBlock, ...],
    sections: dict[str, tuple[_SectionRow, ...]],
) -> tuple[IsotopologueObservation, ...]:
    parent = isotope_blocks[0].masses
    bexp = _section_by_label_or_order(sections["bexp"], isotope_blocks)
    dbvib = _section_by_label_or_order(
        sections.get("dbvib") or sections.get("dvib", ()), isotope_blocks
    )
    weights = _section_by_label_or_order(sections.get("weights", ()), isotope_blocks)
    electronic_rows = sections.get("dbelec") or sections.get("dbele") or ()
    dbelec = _section_by_label_or_order(electronic_rows, isotope_blocks)

    observations: list[IsotopologueObservation] = []
    for block in isotope_blocks:
        constants = bexp.get(block.label)
        if constants is None:
            raise ValueError(f"Missing bexp row for MSR isotopologue {block.label}")
        vib = dbvib.get(block.label, (0.0, 0.0, 0.0))
        elec = dbelec.get(block.label, (0.0, 0.0, 0.0))
        weight_values = weights.get(block.label)
        observations.append(
            IsotopologueObservation(
                label=block.label,
                constants=RotationalConstants(*constants),
                substitutions=_substitutions_from_masses(parent, block.masses),
                correction=VibrationalCorrection(*vib, source="MSR dbvib", convention="additive"),
                electronic_correction=ElectronicCorrection(
                    *elec, source="MSR dbelec", convention="additive"
                ),
                weights=RotationalConstants(*weight_values) if weight_values is not None else None,
            )
        )
    return tuple(observations)


def _section_by_label_or_order(
    rows: tuple[_SectionRow, ...],
    isotope_blocks: tuple[_IsotopeBlock, ...],
) -> dict[str, tuple[float, float, float]]:
    if not rows:
        return {}
    by_label = {row.label: row.values for row in rows if row.label}
    result: dict[str, tuple[float, float, float]] = {}
    for idx, block in enumerate(isotope_blocks):
        if block.label in by_label:
            result[block.label] = by_label[block.label]
        elif idx < len(rows):
            result[block.label] = rows[idx].values
    return result


def _substitutions_from_masses(parent: tuple[int, ...], masses: tuple[int, ...]) -> dict[int, int]:
    if len(parent) != len(masses):
        raise ValueError("MSR isotope blocks have inconsistent lengths")
    return {
        idx: mass for idx, (base, mass) in enumerate(zip(parent, masses), start=1) if mass != base
    }


def _parse_section_row(line: str) -> _SectionRow:
    parts = _split_fields(line)
    if len(parts) < 3:
        raise ValueError(f"Invalid MSR data row: {line}")
    values = (_parse_float(parts[0]), _parse_float(parts[1]), _parse_float(parts[2]))
    return _SectionRow(values, _optional_label(line))


def _parse_zmatrix_reference(token: str, current_idx: int, line: str) -> int:
    try:
        ref = int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid MSR Z-matrix reference in line: {line}") from exc
    if ref < 1 or ref > current_idx:
        raise ValueError(f"MSR Z-matrix references must point to previous atoms: {line}")
    return ref - 1


def _validate_zmatrix_rows(rows: list[_ZMatrixLine]) -> None:
    for idx, row in enumerate(rows):
        if idx >= 1 and (not math.isfinite(row.distance) or row.distance <= 0.0):
            raise ValueError(f"MSR Z-matrix distance must be positive for atom {idx + 1}")
        if idx >= 2:
            if not math.isfinite(row.angle) or row.angle < 0.0 or row.angle > 180.0:
                raise ValueError(
                    f"MSR Z-matrix angle must be between 0 and 180 degrees for atom {idx + 1}"
                )
            if row.refs[0] == row.refs[1]:
                raise ValueError(
                    f"MSR Z-matrix bond and angle references must be distinct for atom {idx + 1}"
                )
        if idx >= 3:
            if not math.isfinite(row.dihedral):
                raise ValueError(f"MSR Z-matrix dihedral must be finite for atom {idx + 1}")
            if len({row.refs[0], row.refs[1], row.refs[2]}) != 3:
                raise ValueError(f"MSR Z-matrix references must be distinct for atom {idx + 1}")


def _parse_variable_assignment(line: str) -> tuple[str, float] | None:
    clean = _strip_inline_comment(line)
    if not clean:
        return None
    if "=" in clean:
        name, value = clean.split("=", 1)
        fields = _split_fields(value)
        if not fields:
            return None
        return _clean_variable_name(name), _parse_float(fields[0])
    parts = _split_fields(clean)
    if len(parts) < 2:
        return None
    if _first_token_is_float(clean) or _is_section_header(clean):
        return None
    try:
        value = _parse_float(parts[1])
    except ValueError:
        return None
    return _clean_variable_name(parts[0]), value


def _optional_label(line: str) -> str | None:
    if "\\" not in line:
        return None
    label = line.split("\\", 1)[1].split()[0].strip()
    return label or None


def _is_section_header(line: str) -> bool:
    parts = _split_fields(line)
    return len(parts) == 1 and parts[0].lower() in MSR_SECTION_NAMES


def _first_token_is_float(line: str) -> bool:
    parts = _split_fields(line)
    if not parts:
        return False
    try:
        _parse_float(parts[0])
    except ValueError:
        return False
    return True


def _split_fields(line: str) -> list[str]:
    return line.replace(",", " ").split()


def _strip_inline_comment(line: str) -> str:
    text = line.split("!", 1)[0]
    text = text.split("//", 1)[0]
    return text.strip()


def _parse_float(token: str) -> float:
    text = str(token).strip().replace("D", "E").replace("d", "e")
    return float(text)


def _is_cartesian_line(line: str) -> bool:
    parts = _split_fields(line)
    if len(parts) < 4 or not _is_zmatrix_atom_token(parts[0]):
        return False
    try:
        _parse_float(parts[1])
        _parse_float(parts[2])
        _parse_float(parts[3])
    except ValueError:
        return False
    return True


def _is_zmatrix_atom_token(token: str) -> bool:
    return _is_dummy_atom(token) or _valid_atom_symbol(_normalize_atom_token(token))


def _valid_atom_symbol(symbol: str) -> bool:
    number = atomic_number(symbol)
    return number is not None and number > 0


def _normalize_atom_token(token: str) -> str:
    text = token.strip().strip(",")
    if _is_dummy_atom(text):
        return "X"
    if text.isdigit():
        symbol = atomic_symbol(int(text))
        if symbol == "??":
            raise ValueError(f"Invalid MSR atomic number: {token}")
        return symbol
    clean = text.split("(", 1)[0].split("-", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z]{1,3}", clean):
        symbol = clean[0].upper() + clean[1:].lower()
        return symbol
    match = re.fullmatch(r"([A-Za-z]{1,2})_?\d+", clean)
    if match:
        letters = match.group(1)
        for size in (len(letters), 1):
            symbol = letters[:size][0].upper() + letters[:size][1:].lower()
            if _valid_atom_symbol(symbol):
                return symbol
    return clean[0].upper() + clean[1:].lower()


def _is_dummy_atom(token: str) -> bool:
    return str(token).strip().upper() in {"X", "XX", "-1"}


def _is_frozen_token(token: str) -> bool:
    return str(token or "").strip().startswith("#")


def _clean_variable_name(name: str) -> str:
    return name.strip().lstrip("#").upper()


def _resolve_value(token: str, variables: dict[str, float]) -> float:
    try:
        return _parse_float(token)
    except ValueError:
        key = _clean_variable_name(token)
        if key not in variables:
            raise ValueError(f"Unresolved MSR Z-matrix variable: {token}")
        return variables[key]


def _require_fields(parts: list[str], nfields: int, line: str) -> None:
    if len(parts) < nfields:
        raise ValueError(f"Invalid MSR Z-matrix line: {line}")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("MSR Z-matrix generated a zero-length reference vector")
    return vector / norm


def _zmatrix_reference_frame(
    bond_atom: np.ndarray,
    angle_atom: np.ndarray,
    dihedral_atom: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local frame used by MSR/Gaussian-style Z-matrices.

    The first axis points from the new atom's bond reference to its angle
    reference.  The second axis lies in the reference plane, and the third is
    the out-of-plane normal.  With this convention D=0/180 keeps planar
    Z-matrix rows in the reference plane.
    """
    ez = _unit(angle_atom - bond_atom)
    if dihedral_atom is None:
        ref = (
            np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(ez[2]) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=float)
        )
        normal = _unit(np.cross(ez, ref))
    else:
        normal_raw = np.cross(ez, dihedral_atom - angle_atom)
        if np.linalg.norm(normal_raw) <= 1.0e-8:
            ref = (
                np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(ez[2]) < 0.9
                else np.array([0.0, 1.0, 0.0], dtype=float)
            )
            normal_raw = np.cross(ez, ref)
        normal = _unit(normal_raw)
    ey = _unit(np.cross(normal, ez))
    return ez, ey, normal


def _mass_number(value: float) -> int:
    rounded = int(round(value))
    if rounded <= 0:
        raise ValueError(f"Invalid MSR isotope mass number: {value}")
    return rounded
