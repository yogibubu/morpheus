from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MolecularGeometry:
    """Canonical ORACLE Cartesian molecular geometry.

    Coordinates are stored in Angstrom. Atom labels are normalized element
    symbols. Program-specific parser details belong in `metadata`.
    """

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    comment: str = ""
    source_format: str = "unknown"
    source_path: Path | None = None
    charge: int | None = None
    multiplicity: int | None = None
    fixed_parameters: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_angstrom, dtype=float)
        if coords.shape != (len(self.atoms), 3):
            raise ValueError(
                f"coordinates shape must be ({len(self.atoms)}, 3), got {coords.shape}"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must be finite")
        object.__setattr__(self, "coordinates_angstrom", coords)
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "fixed_parameters", tuple(self.fixed_parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    def xyz_lines(self) -> list[str]:
        lines = [str(self.natoms), self.comment]
        for atom, (x, y, z) in zip(self.atoms, self.coordinates_angstrom):
            lines.append(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}")
        return lines
