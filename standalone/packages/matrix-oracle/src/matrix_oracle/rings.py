"""Public ORACLE ring-perception contract.

ORACLE owns molecular cycle perception.  SMITH consumes the resulting ordered
cycles to construct ring coordinates; it does not define a second ring model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from matrix_chem.topology.ringset import RingSet


ORACLE_RING_PERCEPTION_SCHEMA = "matrix.oracle.rings.v1"


@dataclass(frozen=True)
class OracleRingPerception:
    schema: str
    rings: tuple[tuple[int, ...], ...]
    atom_to_rings: tuple[tuple[int, ...], ...]
    bond_to_rings: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perceive_rings(
    graph,
    coordinates=None,
    *,
    maximum_size: int | None = None,
) -> OracleRingPerception:
    """Perceive a complete non-metal minimum cycle basis.

    ``maximum_size=None`` is the production default and admits macrocycles.
    A finite maximum is an explicit user truncation; completeness is exposed
    in the returned diagnostics and is never silently assumed.
    """

    if maximum_size is not None and int(maximum_size) < 3:
        raise ValueError("maximum ring size must be at least three")
    ring_set = RingSet(
        graph,
        coords=coordinates,
        ring_max=None if maximum_size is None else int(maximum_size),
    )
    natoms = int(getattr(graph, "natoms", getattr(graph, "n_atoms", 0)))
    diagnostics = asdict(ring_set.cycle_basis_diagnostics)
    diagnostics["maximum_size_requested"] = maximum_size
    return OracleRingPerception(
        schema=ORACLE_RING_PERCEPTION_SCHEMA,
        rings=tuple(tuple(int(atom) for atom in ring.atoms) for ring in ring_set.rings),
        atom_to_rings=tuple(
            tuple(int(index) for index in ring_set.rings_of_atom(atom))
            for atom in range(natoms)
        ),
        bond_to_rings=tuple(
            (tuple(int(atom) for atom in bond), tuple(int(index) for index in indices))
            for bond, indices in sorted(ring_set.bond_to_rings.items())
        ),
        diagnostics=diagnostics,
    )


__all__ = [
    "ORACLE_RING_PERCEPTION_SCHEMA",
    "OracleRingPerception",
    "perceive_rings",
]
