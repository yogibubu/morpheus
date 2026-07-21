from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from matrix_qm import (
    ElectronicSection,
    ElectronicStateRecord,
    ElectronicTransitionRecord,
    TransitionsSection,
    merge_orbitals_section,
    orbital_file_record_from_path,
    write_electronic_section,
    write_transitions_section,
)

from .parsers import summarize_gaussian_log


_EXCITED_STATE_RE = re.compile(
    r"Excited State\s+(?P<index>\d+):\s+"
    r"(?P<label>\S+)\s+"
    r"(?P<energy>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+eV\s+"
    r"(?P<wavelength>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+nm\s+"
    r"f=(?P<osc>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)"
)


@dataclass(frozen=True)
class GaussianElectronicData:
    log_path: Path
    electronic: ElectronicSection
    transitions: TransitionsSection


@dataclass(frozen=True)
class GaussianElectronicPromotion:
    xyzin: Path
    log_path: Path
    wrote_electronic: bool
    wrote_transitions: bool
    wrote_orbitals: bool


def parse_gaussian_electronic_log(path: Path | str) -> GaussianElectronicData:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="ignore")
    summary = summarize_gaussian_log(target)
    states: list[ElectronicStateRecord] = []
    if summary.scf_energies_hartree:
        states.append(
            ElectronicStateRecord(
                label="S0",
                energy_hartree=summary.scf_energies_hartree[-1],
                energy_ev=0.0,
                source="gaussian-log",
            )
        )
    transitions: list[ElectronicTransitionRecord] = []
    for match in _EXCITED_STATE_RE.finditer(text):
        index = int(match.group("index"))
        to_state = f"S{index}"
        label = match.group("label")
        energy_ev = float(match.group("energy"))
        multiplicity, symmetry = _split_excited_label(label)
        states.append(
            ElectronicStateRecord(
                label=to_state,
                energy_ev=energy_ev,
                multiplicity=multiplicity,
                symmetry=symmetry,
                source="gaussian-log",
            )
        )
        transitions.append(
            ElectronicTransitionRecord(
                from_state="S0",
                to_state=to_state,
                energy_ev=energy_ev,
                wavelength_nm=float(match.group("wavelength")),
                oscillator_strength=float(match.group("osc").replace("D", "E")),
                source="gaussian-log",
            )
        )
    return GaussianElectronicData(
        log_path=target,
        electronic=ElectronicSection(tuple(states)),
        transitions=TransitionsSection(tuple(transitions)),
    )


def promote_gaussian_electronic_log_to_xyzin(
    log_path: Path | str,
    xyzin: Path | str,
    *,
    write_electronic: bool = True,
    write_transitions: bool = True,
    orbital_files: tuple[Path | str, ...] = (),
) -> GaussianElectronicPromotion:
    target = Path(xyzin)
    data = parse_gaussian_electronic_log(log_path)
    wrote_electronic = False
    wrote_transitions = False
    wrote_orbitals = False
    if write_electronic and data.electronic.states:
        write_electronic_section(target, data.electronic)
        wrote_electronic = True
    if write_transitions and data.transitions.transitions:
        write_transitions_section(target, data.transitions)
        wrote_transitions = True
    if orbital_files:
        merge_orbitals_section(
            target,
            tuple(
                orbital_file_record_from_path(path, source="gaussian-log") for path in orbital_files
            ),
        )
        wrote_orbitals = True
    return GaussianElectronicPromotion(
        xyzin=target,
        log_path=data.log_path,
        wrote_electronic=wrote_electronic,
        wrote_transitions=wrote_transitions,
        wrote_orbitals=wrote_orbitals,
    )


def _split_excited_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        return label, ""
    multiplicity, symmetry = label.split("-", 1)
    return multiplicity, symmetry
