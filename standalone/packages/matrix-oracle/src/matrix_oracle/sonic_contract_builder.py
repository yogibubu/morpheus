"""ORACLE compatibility frontend for the frozen ONIC/SONIC contract.

The builder is the only stage that converts ORACLE's frozen topology,
symmetry, multicenter, structural-site, and auxiliary-contact evidence into a
typed primitive pool.  SMITH receives the serialized result and performs no
chemical reperception.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from matrix_chem import (
    FragmentMembership,
    OracleSonicContract,
    ORACLE_SONIC_CONTRACT_OWNER,
    ORACLE_SONIC_CONTRACT_SCHEMA,
    PrimaryTopology,
    Primitive,
    PrimitiveCandidate,
    StructuralSite,
    build_primitives,
    grad_primitive,
    graph_cycle_rank,
    geometry_identity_payload_sha256,
    admits_dative_pair,
    is_bridging_ligand,
    is_chalcogen_linkage,
    is_structural_center,
    primary_topology_hash,
    read_enriched_xyz,
    read_geometry_identity_certificate,
    read_molecular_symmetry,
    topology_snapshot_from_xyzin,
    validate_geometry_identity_certificate,
    validate_oracle_sonic_contract,
    write_oracle_sonic_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_fragments import (
    fragment_frame_anchor_atoms,
    fragment_local_frame,
    read_interaction_center_definition,
)

from .auxiliary_contacts import (
    StructuralSiteContactRequest,
    perceive_auxiliary_contact_evidence,
    qualified_vdw_radius,
)
from .contact_graph import complete_and_classify_contact_orbits
from .multicenter_domains import perceive_multicenter_domains
from .local_perception import (
    perceive_local_perception_domains,
    read_frozen_effective_atomic_numbers,
)
from .perception_policy import chemical_perception_policy_manifest


ORACLE_SONIC_CONTRACT_BUILDER = "ORACLE_SONIC_CONTRACT_BUILDER"
ORACLE_SONIC_CONTRACT_BUILDER_VERSION = "2"


def build_oracle_sonic_contract_from_xyzin(
    path: Path,
    *,
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    explicit_multicenter_bridges: Iterable[tuple[int, int, int]] = (),
) -> OracleSonicContract:
    """Build a complete frozen contract from existing ORACLE xyzin sections.

    Explicit bridge indices follow the zero-based provider API. Every
    serialized atom index remains one based. Fragment-pose candidates are
    original finite distance functions (center--center, center--anchor, and
    anchor--anchor), so their Wilson rows are analytic and no projected row is
    serialized as a nonlinear coordinate.
    """

    target = Path(path)
    geometry = read_enriched_xyz(target)
    geometry_identity = read_geometry_identity_certificate(target)
    validate_geometry_identity_certificate(
        geometry_identity,
        canonical_atoms=geometry.atoms,
        canonical_coordinates_angstrom=geometry.coordinates_angstrom,
    )
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(symbol)) for symbol in geometry.atoms)
    snapshot = topology_snapshot_from_xyzin(target)
    bonds = tuple(tuple(int(atom) for atom in pair) for pair in snapshot["bonds"])
    rings = tuple(tuple(int(atom) for atom in item["atoms"]) for item in snapshot["rings"])
    fragments = tuple(
        FragmentMembership(f"F{index:03d}", tuple(int(atom) for atom in atoms))
        for index, atoms in enumerate(snapshot["fragments"], start=1)
    )
    symmetry = read_molecular_symmetry(target)
    cycle_rank = graph_cycle_rank(len(numbers), bonds)
    topology = PrimaryTopology(
        natoms=len(numbers),
        atomic_numbers=numbers,
        bonds=bonds,
        rings=rings,
        fragments=fragments,
        topology_hash=primary_topology_hash(numbers, bonds, rings, fragments),
        cycle_rank=cycle_rank,
        symmetry_permutations=tuple(
            dict.fromkeys(operation.permutation for operation in symmetry.operations)
        ),
    )

    zero_bonds = tuple((left - 1, right - 1) for left, right in bonds)
    zero_rings = tuple(tuple(atom - 1 for atom in ring) for ring in rings)
    effective_atomic_numbers = read_frozen_effective_atomic_numbers(
        target,
        natoms=len(numbers),
    )
    local_perception_domains = perceive_local_perception_domains(
        numbers,
        coordinates,
        zero_bonds,
        zero_rings,
        effective_atomic_numbers=effective_atomic_numbers,
    )
    domains, domain_candidates = perceive_multicenter_domains(
        numbers,
        coordinates,
        zero_bonds,
        explicit_multicenter_bridges=explicit_multicenter_bridges,
    )
    domain_signatures = {
        _candidate_signature(candidate.function, candidate.atoms)
        for candidate in domain_candidates
    }
    candidates: list[PrimitiveCandidate] = []

    graph = _frozen_graph(numbers, coordinates, zero_bonds)
    for primitive in build_primitives(
        graph,
        coordinates,
        include_pseudo_bonds=False,
    ):
        one_based_atoms = tuple(atom + 1 for atom in primitive.atoms)
        if _candidate_signature(primitive.function, one_based_atoms) in domain_signatures:
            continue
        candidate = _finite_primary_candidate(
            primitive,
            coordinates,
            candidate_id=_next_candidate_id(candidates),
            function=primitive.function,
            family=_primary_family(primitive),
            units=_primitive_units(primitive.function),
            owner_id="PRIMARY_TOPOLOGY",
            domain_id=_primary_coordinate_domain(primitive, zero_rings),
            provenance=(
                f"{ORACLE_SONIC_CONTRACT_BUILDER}@"
                f"{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}"
            ),
            refs=_primary_chemical_role_refs(primitive, numbers),
        )
        if candidate is not None:
            candidates.append(candidate)

    remapped_domains = []
    for domain in domains:
        id_map: dict[str, str] = {}
        for candidate in domain_candidates:
            if candidate.domain_id != domain.domain_id:
                continue
            new_id = _next_candidate_id(candidates)
            id_map[candidate.candidate_id] = new_id
            candidates.append(replace(candidate, candidate_id=new_id))
        remapped_domains.append(
            replace(
                domain,
                primitive_candidate_ids=tuple(
                    id_map[candidate_id]
                    for candidate_id in domain.primitive_candidate_ids
                ),
            )
        )

    sites, site_candidates = _structural_sites_and_candidates(
        target,
        coordinates,
        topology,
        candidate_offset=len(candidates),
    )
    candidates.extend(site_candidates)

    evidence = perceive_auxiliary_contact_evidence(
        numbers,
        coordinates,
        zero_bonds,
        configured_site_contacts=configured_site_contacts,
    )
    classified = complete_and_classify_contact_orbits(
        evidence,
        topology,
        structural_sites=sites,
    )
    contacts = []
    site_by_id = {site.site_id: site for site in sites}
    for contact in classified.contacts:
        candidate = _contact_distance_candidate(
            contact,
            coordinates,
            site_by_id,
            candidate_id=_next_candidate_id(candidates),
        )
        if candidate is None:
            contacts.append(contact)
            continue
        candidates.append(candidate)
        contacts.append(
            replace(contact, primitive_candidate_ids=(candidate.candidate_id,))
        )

    candidates.extend(
        _fragment_pose_candidates(
            coordinates,
            fragments,
            candidate_offset=len(candidates),
        )
    )
    contract = OracleSonicContract(
        schema=ORACLE_SONIC_CONTRACT_SCHEMA,
        owner=ORACLE_SONIC_CONTRACT_OWNER,
        primary_topology=topology,
        multicenter_domains=tuple(remapped_domains),
        structural_sites=sites,
        auxiliary_contacts=tuple(contacts),
        primitive_candidates=tuple(candidates),
        primary_cycle_rank=cycle_rank,
        auxiliary_cycle_rank=classified.auxiliary_cycle_rank,
        provenance=(
            f"{ORACLE_SONIC_CONTRACT_BUILDER}@"
            f"{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}"
        ),
        local_perception_domains=local_perception_domains,
        chemical_policy_sha256=chemical_perception_policy_manifest()["sha256"],
        reference_geometry_sha256=geometry_identity.canonical_geometry_sha256,
        geometry_identity_payload_sha256=geometry_identity_payload_sha256(
            geometry_identity
        ),
    )
    validate_oracle_sonic_contract(contract)
    return contract


def write_oracle_sonic_contract_from_xyzin(
    path: Path,
    *,
    configured_site_contacts: Iterable[StructuralSiteContactRequest] = (),
    explicit_multicenter_bridges: Iterable[tuple[int, int, int]] = (),
) -> OracleSonicContract:
    """Build, validate, and serialize the ORACLE-owned frozen contract."""

    contract = build_oracle_sonic_contract_from_xyzin(
        Path(path),
        configured_site_contacts=configured_site_contacts,
        explicit_multicenter_bridges=explicit_multicenter_bridges,
    )
    write_oracle_sonic_contract(Path(path), contract)
    # Freeze the scientific MINIMUM prescription in the same transaction.
    # Downstream SMITH must consume this atlas instead of inferring fragment
    # contacts or chart modes from geometry.
    from .coordinate_atlas import write_minimum_coordinate_atlas_contract

    write_minimum_coordinate_atlas_contract(Path(path), contract)
    return contract


def _frozen_graph(
    numbers: tuple[int, ...],
    coordinates: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> SimpleNamespace:
    adjacency = [set() for _ in numbers]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return SimpleNamespace(
        natoms=len(numbers),
        Z=np.asarray(numbers, dtype=int),
        coords=coordinates,
        bonds=list(bonds),
        adjacency=adjacency,
    )


def _candidate_from_primitive(
    primitive: Primitive,
    coordinates: np.ndarray,
    *,
    candidate_id: str,
    function: str,
    family: str,
    units: str,
    owner_id: str,
    domain_id: str,
    provenance: str,
    refs: tuple[str, ...] = (),
) -> PrimitiveCandidate:
    row = np.asarray(grad_primitive(primitive, coordinates), dtype=float).reshape(-1)
    if np.any(~np.isfinite(row)) or float(np.linalg.norm(row)) <= 1.0e-12:
        raise ValueError(f"ORACLE primitive {candidate_id} has no finite Wilson row")
    return PrimitiveCandidate(
        candidate_id=candidate_id,
        function=function,
        atoms=tuple(atom + 1 for atom in primitive.atoms),
        mode=primitive.mode,
        ref_atoms=tuple(atom + 1 for atom in primitive.ref),
        refs=tuple(str(value) for value in refs),
        family=family,
        units=units,
        owner_id=owner_id,
        domain_id=domain_id,
        analytic_wilson_row=tuple(float(value) for value in row),
        provenance=provenance,
    )


def _finite_primary_candidate(
    primitive: Primitive,
    coordinates: np.ndarray,
    **metadata,
) -> PrimitiveCandidate | None:
    """Discard only mathematically undefined generated valence candidates."""

    try:
        return _candidate_from_primitive(primitive, coordinates, **metadata)
    except ValueError as exc:
        if "has no finite Wilson row" not in str(exc):
            raise
        return None


def _structural_sites_and_candidates(
    path: Path,
    coordinates: np.ndarray,
    topology: PrimaryTopology,
    *,
    candidate_offset: int,
) -> tuple[tuple[StructuralSite, ...], tuple[PrimitiveCandidate, ...]]:
    definition = read_interaction_center_definition(Path(path))
    fragment_by_atom = {
        atom: fragment.fragment_id
        for fragment in topology.fragments
        for atom in fragment.atoms
    }
    sites = []
    for record in definition.centers:
        radius = float(
            np.mean(
                [
                    qualified_vdw_radius(topology.atomic_numbers[atom - 1])[0]
                    for atom in record.atoms
                ]
            )
        )
        sites.append(
            StructuralSite(
                site_id=record.identifier,
                kind=record.kind,
                members=record.atoms,
                fragment_ids=tuple(
                    dict.fromkeys(fragment_by_atom[atom] for atom in record.atoms)
                ),
                center_angstrom=tuple(float(value) for value in record.center),
                frame=fragment_local_frame(coordinates, record.atoms),
                exposed=True,
                effective_radius_angstrom=radius,
                provider="ORACLE_INTERACTION_CENTERS",
                provider_version="2",
                provenance=f"{record.source}:MEMBERWISE_QUALIFIED_VDW_MEAN",
            )
        )
    site_by_id = {site.site_id: site for site in sites}
    candidates = []
    for interaction in definition.interactions:
        site = site_by_id[interaction.center_id]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(interaction.atom - 1,),
        )
        candidates.append(
            _candidate_from_primitive(
                primitive,
                coordinates,
                candidate_id=f"P{candidate_offset + len(candidates) + 1:05d}",
                function="CENTER_ATOM_DIST",
                family="CENTER_ATOM_DISTANCE",
                units="ANGSTROM",
                owner_id=site.site_id,
                domain_id="STRUCTURAL_SITES",
                provenance=f"ORACLE_INTERACTION_CENTERS@2:{interaction.source}",
                refs=(site.site_id,),
            )
        )
    return tuple(sites), tuple(candidates)


def _contact_distance_candidate(
    contact,
    coordinates: np.ndarray,
    sites: dict[str, StructuralSite],
    *,
    candidate_id: str,
) -> PrimitiveCandidate | None:
    left, right = contact.endpoint_a, contact.endpoint_b
    if left.kind == right.kind == "ATOM":
        primitive = Primitive(
            "hbond_dist",
            (int(left.identifier) - 1, int(right.identifier) - 1),
        )
        function = "R"
    elif left.kind == "ATOM" and right.kind == "STRUCTURAL_SITE":
        site = sites[right.identifier]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(int(left.identifier) - 1,),
        )
        function = "FCA_DIST"
    elif left.kind == "STRUCTURAL_SITE" and right.kind == "ATOM":
        site = sites[left.identifier]
        primitive = Primitive(
            "frag_atom_dist",
            tuple(atom - 1 for atom in site.members),
            ref=(int(right.identifier) - 1,),
        )
        function = "FCA_DIST"
    else:
        return None
    return _candidate_from_primitive(
        primitive,
        coordinates,
        candidate_id=candidate_id,
        function=function,
        family=f"{contact.kind}_DISTANCE",
        units="ANGSTROM",
        owner_id=contact.contact_id,
        domain_id="AUXILIARY_CONTACTS",
        provenance=f"{contact.provider}@{contact.provider_version}:{contact.provenance}",
        refs=(site.site_id,) if function == "FCA_DIST" else (),
    )


def _fragment_pose_candidates(
    coordinates: np.ndarray,
    fragments: tuple[FragmentMembership, ...],
    *,
    candidate_offset: int,
) -> tuple[PrimitiveCandidate, ...]:
    candidates = []
    for reference, moving in combinations(fragments, 2):
        owner = f"FRAGMENT_PAIR:{reference.fragment_id}|{moving.fragment_id}"
        provenance = (
            f"{ORACLE_SONIC_CONTRACT_BUILDER}@{ORACLE_SONIC_CONTRACT_BUILDER_VERSION}:"
            "ANALYTIC_FINITE_DISTANCE_POSE_POOL"
        )
        specifications: list[tuple[Primitive, str, str, tuple[str, ...]]] = [
            (
                Primitive(
                    "frag_dist",
                    tuple(atom - 1 for atom in moving.atoms),
                    ref=tuple(atom - 1 for atom in reference.atoms),
                ),
                "FC_DIST",
                "FRAG_DISTANCE",
                (moving.fragment_id, reference.fragment_id),
            )
        ]
        reference_anchors = fragment_frame_anchor_atoms(coordinates, reference.atoms)
        moving_anchors = fragment_frame_anchor_atoms(coordinates, moving.atoms)
        specifications.extend(
            (
                Primitive(
                    "frag_atom_dist",
                    tuple(atom - 1 for atom in moving.atoms),
                    ref=(atom - 1,),
                ),
                "FCA_DIST",
                "FRAG_CENTER_ATOM_DISTANCE",
                (moving.fragment_id,),
            )
            for atom in reference_anchors
        )
        specifications.extend(
            (
                Primitive(
                    "frag_atom_dist",
                    tuple(atom - 1 for atom in reference.atoms),
                    ref=(atom - 1,),
                ),
                "FCA_DIST",
                "FRAG_CENTER_ATOM_DISTANCE",
                (reference.fragment_id,),
            )
            for atom in moving_anchors
        )
        specifications.extend(
            (
                Primitive("bond", (left - 1, right - 1)),
                "R",
                "FRAGMENT_POSE_DISTANCE",
                (),
            )
            for left in reference_anchors
            for right in moving_anchors
        )
        seen = set()
        for primitive, function, family, refs in specifications:
            signature = (function, tuple(primitive.atoms), tuple(primitive.ref))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                _candidate_from_primitive(
                    primitive,
                    coordinates,
                    candidate_id=f"P{candidate_offset + len(candidates) + 1:05d}",
                    function=function,
                    family=family,
                    units="ANGSTROM",
                    owner_id=owner,
                    domain_id="FRAGMENT_POSE",
                    provenance=provenance,
                    refs=refs,
                )
            )
    return tuple(candidates)


def _primary_family(primitive: Primitive) -> str:
    return {
        "bond": "STRETCH",
        "angle": "BEND",
        "linear_bend": "LINEAR_BEND",
        "dihedral": "TORSION",
        "out_of_plane": "OUT_OF_PLANE",
        "out_of_plane_height": "OUT_OF_PLANE_HEIGHT",
    }.get(primitive.kind, primitive.kind.upper())


def _primary_coordinate_domain(
    primitive: Primitive,
    rings: tuple[tuple[int, ...], ...],
) -> str:
    """Return a topology-local compatibility domain for a valence primitive."""

    atoms = tuple(atom + 1 for atom in primitive.atoms)
    if primitive.kind == "bond" and len(atoms) == 2:
        return f"PRIMARY_TOPOLOGY::BOND:{min(atoms)}-{max(atoms)}"
    if primitive.kind in {"angle", "linear_bend"} and len(atoms) == 3:
        ring_owners = tuple(
            index
            for index, ring in enumerate(rings, start=1)
            if set(primitive.atoms).issubset(ring)
        )
        if len(ring_owners) == 1:
            return f"PRIMARY_TOPOLOGY::RING:{ring_owners[0]}::BEND"
        return f"PRIMARY_TOPOLOGY::CENTER:{atoms[1]}::BEND"
    if primitive.kind == "dihedral" and len(atoms) == 4:
        center = tuple(sorted(atoms[1:3]))
        return f"PRIMARY_TOPOLOGY::BOND:{center[0]}-{center[1]}::TORSION"
    if primitive.kind in {"out_of_plane", "out_of_plane_height"} and atoms:
        return f"PRIMARY_TOPOLOGY::CENTER:{atoms[0]}::OUT_OF_PLANE"
    return "PRIMARY_TOPOLOGY::GLOBAL"


def _primary_chemical_role_refs(
    primitive: Primitive,
    numbers: tuple[int, ...],
) -> tuple[str, ...]:
    """Attach periodic-role semantics without changing primitive families."""

    if primitive.kind != "bond" or len(primitive.atoms) != 2:
        return ()
    left, right = primitive.atoms
    left_number, right_number = numbers[left], numbers[right]
    if is_chalcogen_linkage(left_number, right_number):
        return ("CHEMICAL_ROLE=CHALCOGEN_LINKAGE",)
    if (
        is_structural_center(left_number) and is_bridging_ligand(right_number)
    ) or (
        is_structural_center(right_number) and is_bridging_ligand(left_number)
    ):
        return ("CHEMICAL_ROLE=STRUCTURAL_CENTER_LIGAND",)
    if admits_dative_pair(left_number, right_number) or admits_dative_pair(
        right_number, left_number
    ):
        return ("CHEMICAL_ROLE=DATIVE_DONOR_ACCEPTOR",)
    return ()


def _primitive_units(function: str) -> str:
    return (
        "ANGSTROM"
        if function in {"R", "FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}
        else "RADIAN"
    )


def _candidate_signature(
    function: str,
    atoms: tuple[int, ...],
) -> tuple[str, tuple[int, ...]]:
    values = tuple(int(atom) for atom in atoms)
    if function == "R":
        values = tuple(sorted(values))
    elif function == "A" and len(values) == 3:
        values = min(values, tuple(reversed(values)))
    elif function == "D" and len(values) == 4:
        values = min(values, tuple(reversed(values)))
    return function, values


def _next_candidate_id(candidates: list[PrimitiveCandidate]) -> str:
    return f"P{len(candidates) + 1:05d}"


__all__ = [
    "ORACLE_SONIC_CONTRACT_BUILDER",
    "ORACLE_SONIC_CONTRACT_BUILDER_VERSION",
    "build_oracle_sonic_contract_from_xyzin",
    "write_oracle_sonic_contract_from_xyzin",
]
