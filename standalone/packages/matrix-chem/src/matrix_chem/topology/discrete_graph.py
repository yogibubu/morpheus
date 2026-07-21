"""
discrete_graph.py
=================

Construction of a discrete molecular graph from a ContinuousGraph.

Design principles:
- geometry-first
- no explicit valence rules
- no electronic assumptions
- hydrogens must have exactly one bond
"""

import numpy as np

from .covalent_radii import covalent_radius

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

BOND_THRESHOLD = 0.2  # minimum continuous connectivity weight
REFF_SCALE = 1.25  # distance sanity factor


class DiscreteGraph:
    """
    Discrete molecular graph derived from a ContinuousGraph.
    """

    def __init__(self, graph):
        self.graph = graph
        self.Z = graph.Z
        self.coords = graph.coords
        self.connectivity = graph.CONNECTIVITY
        self.natoms = len(self.Z)

        self.adjacency = [set() for _ in range(self.natoms)]
        self.bonds = []

        self._build()
        self._validate_hydrogens()

    # --------------------------------------------------------
    # Graph construction
    # --------------------------------------------------------

    def _build(self):
        for i in range(self.natoms):
            Zi = int(self.Z[i])
            ri = covalent_radius(Zi)
            if ri is None:
                continue

            for j in range(i + 1, self.natoms):
                Zj = int(self.Z[j])
                rj = covalent_radius(Zj)
                if rj is None:
                    continue
                if (
                    Zi == 1
                    and Zj == 1
                    and (self._has_heavy_partner(i) or self._has_heavy_partner(j))
                ):
                    continue

                if self.connectivity[i, j] < BOND_THRESHOLD:
                    continue

                rij = np.linalg.norm(self.coords[i] - self.coords[j])

                if rij > REFF_SCALE * (ri + rj):
                    continue

                self._add_bond(i, j)

    def _add_bond(self, i, j):
        self.adjacency[i].add(j)
        self.adjacency[j].add(i)
        self.bonds.append((i, j))

    def _has_heavy_partner(self, i):
        ri = covalent_radius(int(self.Z[i]))
        if ri is None:
            return False
        for j in range(self.natoms):
            if i == j or int(self.Z[j]) == 1:
                continue
            rj = covalent_radius(int(self.Z[j]))
            if rj is None:
                continue
            if self.connectivity[i, j] < BOND_THRESHOLD:
                continue
            rij = np.linalg.norm(self.coords[i] - self.coords[j])
            if rij <= REFF_SCALE * (ri + rj):
                return True
        return False

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_hydrogens(self):
        for i, Zi in enumerate(self.Z):
            if Zi == 1 and len(self.adjacency[i]) != 1:
                raise ValueError(f"Hydrogen atom {i + 1} has {len(self.adjacency[i])} bonds")

    # --------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------

    def neighbors(self, i):
        return self.adjacency[i]
