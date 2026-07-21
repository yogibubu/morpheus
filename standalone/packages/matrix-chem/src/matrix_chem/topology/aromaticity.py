"""
Aromaticity detection based on:
- ring topology
- planarity
- atomic eligibility derived from synthons

This module does NOT infer electronic structure.
It classifies rings and atoms using available descriptors.
"""

import numpy as np

from .metals import is_metal_atomic_number


class Aromaticity:
    """
    Aromaticity detector.
    """

    def __init__(
        self,
        graph,
        discrete_graph,
        ring_set,
        synthons=None,
        force_aromatic=False,
        planarity_tol=0.1,
    ):
        self.graph = graph
        self.dgraph = discrete_graph
        self.ringset = ring_set
        self.synthons = synthons
        self.force_aromatic = force_aromatic
        self.planarity_tol = planarity_tol

        self.Z = graph.Z
        self.coords = graph.coords

        self.aromatic_atoms = set()
        self.aromatic_bonds = set()

        self._analyze()

    # --------------------------------------------------------
    # Core logic
    # --------------------------------------------------------

    def _analyze(self):
        for ring in self.ringset:
            if not self._is_planar(ring):
                continue

            if not self._is_ring_aromatic(ring):
                continue

            for i in ring.atoms:
                self.aromatic_atoms.add(i)

            for i, j in ring.bonds:
                self.aromatic_bonds.add((i, j))

    # --------------------------------------------------------
    # Planarity
    # --------------------------------------------------------

    def _is_planar(self, ring):
        coords = np.array([self.coords[i] for i in ring.atoms])
        centroid = coords.mean(axis=0)
        coords -= centroid

        _, _, vh = np.linalg.svd(coords, full_matrices=False)
        normal = vh[-1]

        distances = np.abs(coords @ normal)
        return distances.max() < self.planarity_tol

    # --------------------------------------------------------
    # Aromaticity decision
    # --------------------------------------------------------

    def _is_ring_aromatic(self, ring):
        if self.force_aromatic:
            return True

        # Without an explicit electronic-state model, rings smaller than five
        # atoms are conservatively excluded.  This prevents planar saturated
        # cage faces and cyclopropyl rings from being promoted solely because
        # a continuous delocalization proxy is non-zero under strain.
        if len(ring) < 5:
            return False

        for i in ring.atoms:
            if not self._is_atom_aromatic(i):
                return False

        return True

    def _is_atom_aromatic(self, i):
        """
        Decide whether atom i can participate in aromaticity.

        Aromaticity is treated as a discrete attribute derived from:
        - ring membership
        - planarity (checked at ring level)
        - atomic ability to sustain pi delocalization
        """

        # Conservative element/coordination eligibility.  The continuous
        # synthon delocalization component is deliberately not used as a
        # binary gate: it is non-zero in strained saturated rings and can be
        # very small at fused junctions of otherwise aromatic systems.
        Z = int(self.Z[i])
        # Metal--ring contacts are coordination interactions, not additional
        # sigma substituents on the conjugated non-metal cycle.
        deg = sum(
            not is_metal_atomic_number(int(self.Z[neighbor]))
            for neighbor in self.dgraph.neighbors(i)
        )

        if Z == 6:
            return deg == 3
        if Z in (7, 15):
            return deg in (2, 3)
        if Z in (8, 16, 34):
            return deg == 2

        return False
