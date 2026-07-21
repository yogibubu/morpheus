"""
Topology construction pipeline for ORACLE.

This module centralizes the geometry-first construction of all
topological objects and descriptor/synthon objects, without performing any I/O
or workflow decisions.

Frozen contracts:
- Does NOT write files.
- Does NOT call RDKit.
- Does NOT alter topology semantics.
"""

from .continuous_graph import ContinuousGraph
from .discrete_graph import DiscreteGraph
from .ringset import RingSet
from .atomic_synthons import AtomicSynthons
from .aromaticity import Aromaticity

# ============================================================
# Public API
# ============================================================


def build_topology_objects(
    coords,
    Z,
    *,
    bond_order_overrides=None,
    external_charges=None,
    charge_source="ORACLE electronegativity estimate",
    bond_order_source="ORACLE Pauling estimate",
    force_aromatic=False,
):
    """
    Build all topology-related objects from Cartesian coordinates.

    Parameters
    ----------
    coords : array-like, shape (N,3)
        Cartesian coordinates.
    Z : array-like, shape (N,)
        Atomic numbers.
    force_aromatic : bool, optional
        Passed to Aromaticity (no topology effect).

    Returns
    -------
    cg : ContinuousGraph
    dg : DiscreteGraph
    ringset : RingSet
    synthons : AtomicSynthons
    aromaticity : Aromaticity
    """

    # --------------------------------------------------------
    # Continuous topology (geometry-first)
    # --------------------------------------------------------
    # Perceive connectivity from geometry alone. Electronic observables never
    # create or delete graph edges.
    cg = ContinuousGraph(coords, Z)

    # --------------------------------------------------------
    # Discrete topology (H-robust)
    # --------------------------------------------------------
    dg = DiscreteGraph(cg)

    # A new ORACLE state never mixes Mayer and Pauling values, nor CM5 and
    # electronegativity charges, within one structure. Incomplete QM vectors
    # fall back as a whole to the geometry-only observable level.
    candidate_orders = dict(bond_order_overrides or {})
    required_bonds = {tuple(sorted(pair)) for pair in dg.bonds}
    candidate_charges = dict(external_charges or {})
    qm_observables_complete = required_bonds.issubset(candidate_orders) and set(
        candidate_charges
    ) == set(range(len(Z)))
    selected_orders = candidate_orders if qm_observables_complete else {}
    selected_charges = candidate_charges if qm_observables_complete else {}
    cg = ContinuousGraph(coords, Z, bond_order_overrides=selected_orders)
    dg = DiscreteGraph(cg)

    # --------------------------------------------------------
    # Ring detection
    # --------------------------------------------------------
    ringset = RingSet(dg, coords=cg.coords)

    # --------------------------------------------------------
    # Atomic synthons (continuous descriptors)
    # --------------------------------------------------------
    neighbors = [list(dg.adjacency[i]) for i in range(dg.natoms)]
    synthons = AtomicSynthons(
        Z=cg.Z,
        coords=cg.coords,
        neighbors=neighbors,
    )
    synthons._external_charges = selected_charges or None
    synthons._external_bond_orders = selected_orders or None
    synthons._charge_source = (
        charge_source if selected_charges else "ORACLE electronegativity estimate"
    )
    synthons._bond_order_source = (
        bond_order_source if selected_orders else "ORACLE Pauling estimate"
    )

    # --------------------------------------------------------
    # Aromaticity (ring-driven, discrete + geometry)
    # --------------------------------------------------------
    aromaticity = Aromaticity(
        graph=cg,
        discrete_graph=dg,
        ring_set=ringset,
        synthons=synthons,
        force_aromatic=force_aromatic,
    )

    return cg, dg, ringset, synthons, aromaticity
