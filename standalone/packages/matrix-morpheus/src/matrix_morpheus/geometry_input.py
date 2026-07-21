from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from matrix_chem.geometry_io import read_xyz_atoms_coords
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_core.xyzin_geometry import read_xyzin_geometry


@dataclass(frozen=True)
class SemiexperimentalGeometryInput:
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    comment: str
    fixed_parameters: tuple[str, ...] = ()
    source_format: str = "xyz"


def read_geometry_input(path: Path) -> SemiexperimentalGeometryInput:
    target = Path(path)
    suffix = target.suffix.lower()
    from .msr_legacy import is_msr_legacy_file, read_msr_legacy_geometry

    if is_msr_legacy_file(target):
        return read_msr_legacy_geometry(target)
    if target.name.lower() == "xyzin" or suffix in {"", ".xyzin"}:
        try:
            geometry = read_xyzin_geometry(target)
        except (OSError, ValueError):
            if target.name.lower() == "xyzin" or suffix == ".xyzin":
                raise
        else:
            return SemiexperimentalGeometryInput(
                geometry.atoms,
                geometry.coordinates_angstrom,
                geometry.comment or target.name,
                (),
                "xyzin",
            )
    if suffix == ".xyz":
        atoms, coords, comment = read_xyz_atoms_coords(target)
        return SemiexperimentalGeometryInput(
            tuple(atoms), np.asarray(coords, dtype=float), comment, (), "xyz"
        )
    if suffix in {".com", ".gjf"}:
        return read_gaussian_cartesian_input(target)
    if suffix == ".mfit" or target.name.lower().endswith((".mse.toml", ".semiexp.toml")):
        from .job_input import read_semiexperimental_job_geometry

        return read_semiexperimental_job_geometry(target)
    if suffix == ".toml":
        from .job_input import is_semiexperimental_job_file, read_semiexperimental_job_geometry

        if is_semiexperimental_job_file(target):
            return read_semiexperimental_job_geometry(target)
    raise ValueError(
        "Semiexperimental geometry input must be .xyz, .xyzin, .com, .gjf, "
        ".msr, .msr.inp or a MATRIX/MORPHEUS semiexp job TOML"
    )


def read_gaussian_cartesian_input(path: Path) -> SemiexperimentalGeometryInput:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    route_end = _route_end(lines)
    if route_end is None:
        raise ValueError("Gaussian input needs a route section starting with #")
    route_lines = _route_lines(lines)
    if _route_requests_zmatrix(route_lines):
        raise ValueError(
            "Semiexperimental Gaussian input must contain Cartesian coordinates; "
            "Z-matrix/MSR inputs are not supported"
        )
    idx = route_end + 1 if route_end < len(lines) else route_end
    title_start = idx
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    title = " ".join(line.strip() for line in lines[title_start:idx] if line.strip())
    idx = _next_nonblank(lines, idx)
    if idx >= len(lines) or not _is_charge_multiplicity(lines[idx]):
        raise ValueError(
            "Gaussian input needs a charge/multiplicity line before Cartesian coordinates"
        )
    idx += 1

    atoms: list[str] = []
    coords: list[list[float]] = []
    while idx < len(lines) and lines[idx].strip():
        atom, xyz = _parse_cartesian_line(lines[idx])
        atoms.append(atom)
        coords.append(xyz)
        idx += 1
    if not atoms:
        raise ValueError("Gaussian input contains no Cartesian coordinate block")

    modredundant = lines[idx + 1 :] if idx < len(lines) else ()
    fixed = _modredundant_fixed_patterns(modredundant)
    return SemiexperimentalGeometryInput(
        tuple(atoms),
        np.asarray(coords, dtype=float),
        title or Path(path).stem,
        fixed,
        "gaussian_cartesian_modredundant",
    )


def _route_end(lines: list[str]) -> int | None:
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith("#"):
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            return idx
        idx += 1
    return None


def _route_lines(lines: list[str]) -> tuple[str, ...]:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
            route: list[str] = []
            while idx < len(lines) and lines[idx].strip():
                route.append(lines[idx].strip())
                idx += 1
            return tuple(route)
        idx += 1
    return ()


def _route_requests_zmatrix(route_lines: tuple[str, ...]) -> bool:
    route = " ".join(route_lines).lower()
    return "zmat" in route or "z-matrix" in route


def _next_nonblank(lines: list[str], idx: int) -> int:
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _is_charge_multiplicity(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _parse_cartesian_line(line: str) -> tuple[str, list[float]]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid Gaussian Cartesian line: {line}")
    atom = _atom_symbol(parts[0])
    try:
        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
    except ValueError as exc:
        raise ValueError(f"Invalid Gaussian Cartesian coordinates: {line}") from exc
    return atom, coords


def _atom_symbol(token: str) -> str:
    cleaned = token.split("(", 1)[0].split("-", 1)[0].strip()
    if cleaned.isdigit():
        number = int(cleaned)
        symbol = atomic_symbol(number)
        if number <= 0 or symbol == "??":
            raise ValueError(f"Invalid atomic number in Gaussian Cartesian block: {token}")
        return symbol
    match = re.match(r"([A-Za-z]{1,3})", cleaned)
    if not match:
        raise ValueError(f"Invalid atom token in Gaussian Cartesian block: {token}")
    text = match.group(1)
    symbol = text[0].upper() + text[1:].lower()
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"Invalid atom token in Gaussian Cartesian block: {token}")
    return symbol


def _modredundant_fixed_patterns(lines: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    patterns: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        kind = parts[0].upper()
        expected = {"B": 2, "A": 3, "D": 4, "O": 4, "L": 3}.get(kind)
        if expected is None:
            if _looks_like_gic_expression_constraint(line) and line not in seen:
                patterns.append(line)
                seen.add(line)
            continue
        atoms: list[int] = []
        actions: list[str] = []
        for token in parts[1:]:
            try:
                atoms.append(int(token))
            except ValueError:
                actions.append(token.upper())
        if len(atoms) < expected or "F" not in actions:
            continue
        for pattern in _fixed_patterns_for_coordinate(kind, tuple(atoms[:expected])):
            if pattern not in seen:
                patterns.append(pattern)
                seen.add(pattern)
    return tuple(patterns)


def _looks_like_gic_expression_constraint(line: str) -> bool:
    text = str(line).strip()
    low = text.lower()
    if low.startswith(("gic(", "constraint(", "freeze(", "fixed(")):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_'\"]*(?:\([^)]*\))?\s*=", text):
        return True
    if re.search(r"(?i)\bvalue\s*=", text):
        return True
    if re.search(r"(?i)\b(?:freeze|frozen|fixed)\b", text):
        return True
    if re.search(r"(?i)\((?:[^)]*,\s*)?f(?:reeze|rozen|ixed)?(?:\s*,[^)]*)?\)\s*=", text):
        return True
    if "=[" in text and "]" in text:
        return True
    return False


def _fixed_patterns_for_coordinate(kind: str, atoms: tuple[int, ...]) -> tuple[str, ...]:
    if kind == "B":
        i, j = atoms
        return (f"R({i},{j}) Frozen",)
    if kind == "A":
        i, j, k = atoms
        return (f"A({i},{j},{k}) Frozen",)
    if kind == "D":
        i, j, k, l = atoms
        return (f"D({i},{j},{k},{l}) Frozen",)
    if kind == "O":
        i, j, k, l = atoms
        return (f"U({i},{j},{k},{l}) Frozen",)
    if kind == "L":
        i, j, k = atoms
        return (f"L({i},{j},{k},0,-1) Frozen", f"L({i},{j},{k},0,-2) Frozen")
    return ()
