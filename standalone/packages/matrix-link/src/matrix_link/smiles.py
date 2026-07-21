from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matrix_chem.geometry import MolecularGeometry
from matrix_chem.geometry_io import GeometryParseError


SMILES_SOURCE_FORMAT = "smiles_rdkit"
SMILES_MARKER = "SMILES"
DEFAULT_RDKIT_RANDOM_SEED = 61453


class RDKitUnavailableError(GeometryParseError):
    """Raised when a SMILES import requires RDKit but RDKit is unavailable."""


@dataclass(frozen=True)
class SmilesInput:
    smiles: str
    title: str = ""
    charge: int | None = None
    multiplicity: int | None = None
    route_lines: tuple[str, ...] = ()
    source_path: Path | None = None


def rdkit_available() -> bool:
    try:
        _rdkit_modules()
    except RDKitUnavailableError:
        return False
    return True


def is_legacy_smiles_input(path: Path) -> bool:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    return _looks_like_legacy_smiles_text(text)


def extract_legacy_smiles_input(path: Path) -> SmilesInput:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    route_lines = _route_lines(lines)
    if not _route_requests_smiles(route_lines):
        raise GeometryParseError("legacy input is not marked as SMILES")

    route_end = _route_end(lines)
    if route_end is None:
        raise GeometryParseError("SMILES input needs a route section starting with #")
    idx = _next_nonblank(lines, route_end)
    title_start = idx
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    title = " ".join(line.strip() for line in lines[title_start:idx] if line.strip())
    idx = _next_nonblank(lines, idx)
    if idx >= len(lines) or not _is_charge_multiplicity(lines[idx]):
        raise GeometryParseError("SMILES input needs charge and multiplicity before SMILES")
    charge, multiplicity = (int(value) for value in lines[idx].split()[:2])
    idx = _next_nonblank(lines, idx + 1)
    if idx >= len(lines) or not lines[idx].strip():
        raise GeometryParseError("SMILES input contains no SMILES line")
    smiles = lines[idx].strip()
    return SmilesInput(
        smiles=smiles,
        title=title or target.stem,
        charge=charge,
        multiplicity=multiplicity,
        route_lines=route_lines,
        source_path=target,
    )


def read_legacy_smiles_input(path: Path) -> MolecularGeometry:
    smiles_input = extract_legacy_smiles_input(path)
    return smiles_to_geometry(
        smiles_input.smiles,
        title=smiles_input.title,
        charge=smiles_input.charge,
        multiplicity=smiles_input.multiplicity,
        source_path=smiles_input.source_path,
        route_lines=smiles_input.route_lines,
    )


def smiles_to_geometry(
    smiles: str,
    *,
    title: str = "",
    charge: int | None = None,
    multiplicity: int | None = None,
    source_path: Path | None = None,
    route_lines: tuple[str, ...] = (),
    random_seed: int = DEFAULT_RDKIT_RANDOM_SEED,
) -> MolecularGeometry:
    Chem, AllChem = _rdkit_modules()
    normalized_smiles = _normalize_legacy_smiles(smiles)
    mol = Chem.MolFromSmiles(normalized_smiles)
    if mol is None:
        raise GeometryParseError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=int(random_seed))
    if embed_status != 0:
        raise GeometryParseError(f"RDKit could not embed SMILES in 3D: {smiles}")

    optimize_status: int | None = None
    try:
        optimize_status = int(AllChem.UFFOptimizeMolecule(mol, maxIters=200))
    except Exception:
        optimize_status = None

    conformer = mol.GetConformer()
    atoms: list[str] = []
    coords: list[list[float]] = []
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([float(position.x), float(position.y), float(position.z)])

    job_charge = int(Chem.GetFormalCharge(mol)) if charge is None else charge
    return MolecularGeometry(
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coords, dtype=float),
        comment=title or smiles,
        source_format=SMILES_SOURCE_FORMAT,
        source_path=source_path,
        charge=job_charge,
        multiplicity=multiplicity,
        metadata={
            "smiles": smiles,
            "normalized_smiles": normalized_smiles,
            "route": route_lines,
            "rdkit_embed_status": embed_status,
            "rdkit_uff_optimize_status": optimize_status,
        },
    )


def _rdkit_modules() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RDKitUnavailableError(
            "RDKit is required for SMILES import; install an ORACLE environment with rdkit"
        ) from exc
    return Chem, AllChem


def _normalize_legacy_smiles(smiles: str) -> str:
    return smiles.strip().replace("[C@@H2]", "C").replace("[C@H2]", "C")


def _looks_like_legacy_smiles_text(text: str) -> bool:
    return _route_requests_smiles(_route_lines(text.splitlines()))


def _route_end(lines: list[str]) -> int | None:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
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


def _route_requests_smiles(route_lines: tuple[str, ...]) -> bool:
    return SMILES_MARKER in " ".join(route_lines).upper().split()


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
