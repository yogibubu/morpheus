"""Gaussian-like public input deck for LINK geometry optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from matrix_core.manifest import matrix_version

from .optimizer import OptimizerResult, OptimizerSettings, optimize_geometry
from .scan import QMScanBackend


@dataclass(frozen=True)
class OptimizationInputDeck:
    path: Path
    title: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray | None
    smiles: str
    charge: int
    multiplicity: int
    backend: str
    executable: str | None
    model: str
    route_keywords: tuple[str, ...]
    coordinate_kind: str
    orientation_weights: str
    max_steps: int
    trust_radius: float
    max_trust_radius: float
    include_cv_exponential_field: bool


def read_optimization_input(path: Path | str) -> OptimizationInputDeck:
    """Read a Gaussian-shaped MATRIX optimization input.

    Supported MATRIX link-zero keys are ``%Backend``, ``%Executable``,
    ``%MaxSteps``, ``%TrustRadius`` and ``%MaxTrustRadius``.  The molecular
    specification is either ordinary Cartesian rows or one ``SMILES ...`` row.
    """

    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    options: dict[str, str] = {}
    route_parts: list[str] = []
    cursor = 0
    while cursor < len(lines) and lines[cursor].lstrip().startswith("%"):
        key, _, value = lines[cursor].lstrip()[1:].partition("=")
        options[key.strip().lower()] = value.strip()
        cursor += 1
    while cursor < len(lines) and not lines[cursor].lstrip().startswith("#"):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("MATRIX optimization input needs a # route line")
    while cursor < len(lines):
        text = lines[cursor].strip()
        if not text:
            break
        route_parts.append(text)
        cursor += 1
    route = " ".join(route_parts)
    cursor += 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    title = lines[cursor].strip() if cursor < len(lines) else target.stem
    cursor += 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("MATRIX optimization input lacks charge/multiplicity")
    charge_mult = lines[cursor].split()
    if len(charge_mult) != 2:
        raise ValueError("charge/multiplicity line must contain two integers")
    charge, multiplicity = int(charge_mult[0]), int(charge_mult[1])
    cursor += 1
    atoms: list[str] = []
    coordinates: list[list[float]] = []
    smiles = ""
    while cursor < len(lines):
        text = lines[cursor].strip()
        cursor += 1
        if not text:
            break
        if text.upper().startswith("SMILES "):
            smiles = text.split(None, 1)[1].strip()
            continue
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"invalid Cartesian row: {text}")
        atoms.append(parts[0])
        coordinates.append([float(value) for value in parts[1:]])
    if bool(smiles) == bool(atoms):
        raise ValueError("provide exactly one of Cartesian coordinates or SMILES")
    route_tokens = _route_tokens(route)
    model = next((token for token in route_tokens if "/" in token), "HF/STO-3G")
    backend = options.get("backend", _route_value(route_tokens, "backend") or "gaussian")
    coordinate_kind = (_opt_value(route) or "SONIC").lower()
    if coordinate_kind not in {"sonic", "cartesian"}:
        coordinate_kind = "sonic"
    orientation_weights = options.get("orientation", "mass").lower().replace("_", "-")
    aliases = {
        "masses": "mass",
        "atomic-mass": "mass",
        "z": "atomic-number",
        "number": "atomic-number",
    }
    orientation_weights = aliases.get(orientation_weights, orientation_weights)
    if orientation_weights not in {"mass", "atomic-number"}:
        raise ValueError("%Orientation must be mass or atomic-number")
    keywords = tuple(
        token
        for token in route_tokens
        if token.upper() != "MATRIX"
        and token != model
        and not token.lower().startswith(("backend=", "opt=", "cv="))
    )
    return OptimizationInputDeck(
        path=target,
        title=title,
        atoms=tuple(atoms),
        coordinates_angstrom=None if not coordinates else np.asarray(coordinates, dtype=float),
        smiles=smiles,
        charge=charge,
        multiplicity=multiplicity,
        backend=backend.lower(),
        executable=options.get("executable") or None,
        model=model,
        route_keywords=keywords,
        coordinate_kind=coordinate_kind,
        orientation_weights=orientation_weights,
        max_steps=int(options.get("maxsteps", 100)),
        trust_radius=float(options.get("trustradius", 0.2)),
        max_trust_radius=float(options.get("maxtrustradius", 0.5)),
        include_cv_exponential_field=(
            options.get("corevalence", "").strip().lower() in {"exponential", "cv-radial", "on", "true"}
            or (_route_value(route_tokens, "cv").strip().lower() in {"exponential", "cv-radial"})
        ),
    )


def run_optimization_input(
    path: Path | str,
    *,
    run_dir: Path | str,
) -> OptimizerResult:
    """Translate a public input deck into MATRIX state and run LINK."""

    deck = read_optimization_input(path)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    xyz = root / "input.xyz"
    if deck.smiles:
        from matrix_link import smiles_to_geometry

        geometry = smiles_to_geometry(
            deck.smiles,
            title=deck.title,
            charge=deck.charge,
            multiplicity=deck.multiplicity,
        )
        atoms = geometry.atoms
        coordinates = geometry.coordinates_angstrom
    else:
        assert deck.coordinates_angstrom is not None
        atoms = deck.atoms
        coordinates = deck.coordinates_angstrom
    from matrix_core import principal_axis_orientation

    coordinates = principal_axis_orientation(
        coordinates,
        _orientation_weights(atoms, deck.orientation_weights),
    )
    _write_xyz(xyz, atoms, coordinates, deck.title)
    xyzin = root / "input.xyzin"
    from matrix_chem import preprocess_to_enriched_xyz, write_validation_section

    preprocess_to_enriched_xyz(xyz, xyzin)
    fragmented = _topology_component_count(xyzin) > 1
    if fragmented:
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(xyzin)
    validation = write_validation_section(xyzin, require_fragments=fragmented)
    if validation.status != "PASS":
        raise ValueError("generated MATRIX input did not pass topology validation")
    from matrix_smith import write_gicforge_build_sections

    write_gicforge_build_sections(
        xyzin,
        fragment_mode="special-coordinates" if fragmented else None,
    )
    method, basis = _method_basis(deck.model)
    backend = QMScanBackend(
        name=deck.backend,
        route=_backend_route(deck),
        method=method,
        basis=basis,
        charge=deck.charge,
        multiplicity=deck.multiplicity,
        executable=deck.executable,
    )
    result = optimize_geometry(
        xyzin,
        run_dir=root / "optimizer",
        coordinate_kind=deck.coordinate_kind,
        backend=backend,
        settings=OptimizerSettings(
            max_steps=deck.max_steps,
            trust_radius=deck.trust_radius,
            max_trust_radius=deck.max_trust_radius,
            include_cv_exponential_field=deck.include_cv_exponential_field,
        ),
    )
    write_optimization_log(root / "optimization.log", deck, result)
    return result


def write_optimization_log(
    path: Path | str,
    deck: OptimizationInputDeck,
    result: OptimizerResult,
) -> Path:
    """Write a compact Gaussian-inspired, non-Gaussian MATRIX log."""

    lines = [
        " Entering MATRIX LINK geometry optimizer",
        f" MATRIX version: {matrix_version()}",
        f" Backend: {deck.backend}",
        f" Model: {deck.model}",
        f" Coordinate system: {deck.coordinate_kind.upper()}",
        f" Core-valence field: {'ORACLE CV_radial exponential' if deck.include_cv_exponential_field else 'none'}",
        f" Reference orientation: center={deck.orientation_weights}; axes=principal inertia",
        "",
        " Initial Cartesian orientation (Angstrom)",
        _orientation_table(result.atoms, result.initial_coordinates_angstrom),
        "",
        " Step       Energy (Eh)       dE          Max Force    RMS Force    Max Disp     RMS Disp      Trust       Q       Status",
    ]
    for item in result.iterations:
        lines.append(
            f" {item.iteration:4d} {item.energy_hartree:18.10f} {item.energy_change_hartree:11.3e} "
            f"{item.gradient_inf_norm:11.3e} {item.gradient_rms_norm:11.3e} "
            f"{item.step_inf_norm:11.3e} {item.step_rms_norm:11.3e} "
            f"{item.trust_radius:10.3e} {item.trust_ratio:8.3f} {item.status}"
        )
    lines.extend(
        [
            "",
            f" Optimization status: {result.status}",
            f" Electronic evaluations: energy={result.energy_evaluations} gradient={result.gradient_evaluations} hessian={result.hessian_evaluations}",
            f" Final energy: {result.final_energy_hartree:.12f} Eh",
            " Final Cartesian orientation (Angstrom)",
            _orientation_table(result.atoms, result.final_coordinates_angstrom),
            "",
            " Normal termination of MATRIX LINK" if result.converged else " MATRIX LINK did not converge",
        ]
    )
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _backend_route(deck: OptimizationInputDeck) -> str:
    keywords = " ".join(deck.route_keywords)
    if deck.backend in {"gaussian", "g16", "gdv"}:
        return " ".join(item for item in ("#p", deck.model, "Force", keywords) if item)
    if deck.backend == "orca":
        return " ".join(item for item in (deck.model.replace("/", " "), "EnGrad", keywords) if item)
    return keywords


def _method_basis(model: str) -> tuple[str, str]:
    method, separator, basis = model.partition("/")
    return method, basis if separator else "STO-3G"


def _route_tokens(route: str) -> tuple[str, ...]:
    body = re.sub(r"^#\s*[A-Za-z]*\s*", "", route.strip())
    return tuple(body.split())


def _route_value(tokens: Sequence[str], key: str) -> str:
    prefix = key.lower() + "="
    return next((token.split("=", 1)[1] for token in tokens if token.lower().startswith(prefix)), "")


def _opt_value(route: str) -> str:
    match = re.search(r"\bOpt\s*=\s*\(?\s*([A-Za-z_-]+)", route, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _topology_component_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].strip())
    parent = list(range(natoms))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    in_bonds = False
    for line in lines:
        if line.strip() == "[BONDS]":
            in_bonds = True
            continue
        if in_bonds and line.startswith("["):
            break
        if in_bonds:
            parts = line.split()
            if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                left, right = int(parts[0]) - 1, int(parts[1]) - 1
                a, b = find(left), find(right)
                parent[a] = b
    return len({find(index) for index in range(natoms)})


def _write_xyz(path: Path, atoms: Sequence[str], coords: np.ndarray, title: str) -> None:
    lines = [str(len(atoms)), title]
    lines.extend(
        f"{atom:2s} {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}"
        for atom, row in zip(atoms, np.asarray(coords, dtype=float), strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _orientation_weights(atoms: Sequence[str], mode: str) -> np.ndarray:
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number

    numbers = [atomic_number(atom) for atom in atoms]
    if any(number is None for number in numbers):
        raise ValueError("reference orientation needs recognized atomic symbols")
    if mode == "atomic-number":
        return np.asarray(numbers, dtype=float)
    return np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)


def _orientation_table(atoms: Sequence[str], coords: np.ndarray) -> str:
    return "\n".join(
        f" {index:5d} {atom:3s} {row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}"
        for index, (atom, row) in enumerate(zip(atoms, np.asarray(coords), strict=True), start=1)
    )
