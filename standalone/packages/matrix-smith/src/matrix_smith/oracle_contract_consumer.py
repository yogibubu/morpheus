"""Fail-closed SMITH consumer for the frozen ORACLE SONIC contract.

This module contains no chemical perception. It validates ORACLE's frozen
geometry, topology, local semantics and finite Wilson rows. Production chart
construction is then delegated to SMITH's single primitive/SALC builder;
missing chemistry is an error, never an invitation to reconstruct it here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from matrix_chem import (
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    GEOMETRY_IDENTICAL,
    GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION,
    GEOMETRY_RIGID_FRAME_ONLY,
    GEOMETRY_TRUE_CHANGE,
    GeometryIdentityError,
    OracleSonicContract,
    PrimitiveCandidate,
    expected_vibrational_mode_count,
    read_enriched_xyz,
    read_geometry_identity_certificate,
    read_molecular_symmetry,
    read_oracle_coordinate_atlas_contract,
    topology_snapshot_from_xyzin,
    validate_geometry_identity_certificate,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_numerics import select_rank_revealing_rows

from .contracts import (
    GICForgeContractError,
    load_frozen_oracle_sonic_contract,
    validate_complete_frozen_oracle_semantics,
)
from .models import GICDefinition, GICPrimitive
from .numerics import _analytic_b_row
from .onic_taxonomy import onic_branch_for_role
from .policy import PRIMITIVE_FAMILY_ORDER, RANK_TOLERANCE, SPECIAL_PRIMITIVE_FAMILIES
from .symmetrization import _partition_quota_rank_revealing_selection


SMITH_ORACLE_CONSUMER_SCHEMA = "matrix.smith.oracle_contract_consumer.v1"
_ROW_VALIDATION_ATOL = 2.0e-7


@dataclass(frozen=True)
class FrozenOracleCandidateSelection:
    """Numerical selection made exclusively from a frozen ORACLE contract."""

    contract: OracleSonicContract
    candidates: tuple[GICPrimitive, ...]
    selected: tuple[GICPrimitive, ...]
    selected_rows: tuple[tuple[float, ...], ...]
    rank: int
    target_rank: int
    family_quotas: tuple[tuple[str, int], ...]
    provenance: str = SMITH_ORACLE_CONSUMER_SCHEMA


@dataclass(frozen=True)
class FrozenOracleCandidatePool:
    """Validated finite primitive functions and their ORACLE-owned rows."""

    contract: OracleSonicContract
    candidates: tuple[GICPrimitive, ...]
    rows: tuple[tuple[float, ...], ...]
    atom_symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]


def load_validated_oracle_candidate_pool(path: Path) -> FrozenOracleCandidatePool:
    """Load all frozen candidates without selecting or interpreting chemistry."""

    contract = load_frozen_oracle_sonic_contract(Path(path))
    validate_complete_frozen_oracle_semantics(contract)
    geometry = read_enriched_xyz(Path(path))
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    _validate_frozen_geometry(Path(path), contract, tuple(geometry.atoms), coords)
    _validate_candidate_ownership(contract)
    records = tuple(
        (candidate, _primitive_from_candidate(candidate))
        for candidate in contract.primitive_candidates
    )
    rows = tuple(
        tuple(float(value) for value in _validated_supplied_row(candidate, primitive, coords))
        for candidate, primitive in records
    )
    return FrozenOracleCandidatePool(
        contract=contract,
        candidates=tuple(primitive for _candidate, primitive in records),
        rows=rows,
        atom_symbols=tuple(geometry.atoms),
        coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
    )


def select_frozen_oracle_candidates(
    path: Path,
    *,
    target_rank: int | None = None,
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
    allow_closing_contacts: bool = False,
) -> FrozenOracleCandidateSelection:
    """Validate and select only ORACLE-supplied primitive candidates.

    Closing contact orbits require the dedicated contact--pose selector and
    are rejected here by default.  This prevents the generic selector from
    accidentally treating closing pseudobonds as ordinary graph edges.
    """

    pool = load_validated_oracle_candidate_pool(Path(path))
    contract = pool.contract
    coords = np.asarray(pool.coordinates_angstrom, dtype=float)

    closing_candidate_ids = {
        candidate_id
        for contact in contract.auxiliary_contacts
        if contact.open_or_closing == "CLOSING"
        for candidate_id in contact.primitive_candidate_ids
    }
    if closing_candidate_ids and not allow_closing_contacts:
        raise GICForgeContractError(
            "closing ORACLE contact candidates require the contact-pose selector"
        )

    records_all = tuple(zip(contract.primitive_candidates, pool.candidates, strict=True))
    rows_all = tuple(np.asarray(row, dtype=float) for row in pool.rows)
    # ORACLE may carry historical linear-bend candidates for rank recovery.
    # They are invalid for any polyhedral center (CN >= 3), even if a later
    # rank-revealing selector would otherwise choose them.
    adjacency: dict[int, set[int]] = {}
    for first, second in contract.primary_topology.bonds:
        first -= 1
        second -= 1
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    records_rows = tuple(
        (record, row)
        for record, row in zip(records_all, rows_all, strict=True)
        if not (
            record[1].family == "LINEAR_BEND"
            and len(adjacency.get(record[1].atoms[1], ())) >= 3
        )
    )
    records = tuple(record for record, _row in records_rows)
    rows = tuple(row for _record, row in records_rows)
    requested = (
        expected_vibrational_mode_count(coords)
        if target_rank is None
        else int(target_rank)
    )
    if requested < 0:
        raise GICForgeContractError("requested Wilson rank must be non-negative")

    quotas = _normalized_family_quotas(family_quotas)
    selection_contract = replace(
        contract,
        primitive_candidates=tuple(record[0] for record, _row in records_rows),
    )
    selected_indices = _select_indices(
        selection_contract,
        rows,
        target_rank=requested,
        family_quotas=quotas,
        rank_tolerance=float(rank_tolerance),
    )
    if len(selected_indices) != requested:
        raise GICForgeContractError(
            "frozen ORACLE primitive pool has insufficient Wilson rank: "
            f"need {requested}, selected {len(selected_indices)}"
        )
    selected = tuple(records[index][1] for index in selected_indices)
    selected_rows = tuple(tuple(float(value) for value in rows[index]) for index in selected_indices)
    return FrozenOracleCandidateSelection(
        contract=contract,
        candidates=tuple(primitive for _candidate, primitive in records),
        selected=selected,
        selected_rows=selected_rows,
        rank=len(selected_indices),
        target_rank=requested,
        family_quotas=tuple(sorted(quotas.items())),
    )


def build_gic_definition_from_oracle_contract(
    path: Path,
    *,
    symmetrize: bool = False,
    orientation: str = "SONIC",
    scientific_path: str = "MINIMUM",
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
) -> GICDefinition:
    """Validate ORACLE, then realize one ONIC chart with the single SMITH builder."""

    branch = onic_branch_for_role(orientation)
    path_role = _normalized_scientific_path(scientific_path)
    if family_quotas:
        raise GICForgeContractError(
            "the canonical ORACLE-to-SMITH chart does not accept family quotas"
        )
    pool = load_validated_oracle_candidate_pool(Path(path))
    _validate_frozen_fragment_handoff(Path(path), pool.contract)
    try:
        atlas = read_oracle_coordinate_atlas_contract(Path(path))
        validate_oracle_coordinate_atlas_contract(atlas)
    except ValueError as exc:
        raise GICForgeContractError(
            "canonical ORACLE-to-SMITH construction requires a valid frozen coordinate atlas"
        ) from exc
    expected_task = (
        ATLAS_TASK_MINIMUM
        if path_role == "MINIMUM"
        else ATLAS_TASK_TRANSITION_STATE
    )
    if atlas.task_regime != expected_task:
        raise GICForgeContractError(
            "ORACLE coordinate atlas task regime contradicts scientific_path"
        )
    if atlas.topology_hash != pool.contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE coordinate atlas topology fingerprint is stale")
    has_closing_contact = any(
        item.graph_role == "CLOSING" for item in atlas.interactions
    )
    if has_closing_contact and path_role != "MINIMUM":
        raise GICForgeContractError(
            "closed noncovalent contacts belong to the MINIMUM scientific path; "
            "a transition-state chart requires ORACLE's explicit TS contract"
        )

    # Primitive generation, exact-rank reduction and SALCs have one SMITH
    # implementation. This consumer validates ORACLE's frozen state and only
    # routes the explicitly distinct minimum/transition-state context.
    from .definition import build_gic_definition_from_xyzin

    definition = build_gic_definition_from_xyzin(
        Path(path),
        symmetrize=symmetrize,
        fragment_context=(
            "minimum" if path_role == "MINIMUM" else "transition_state"
        ),
        rank_tolerance=rank_tolerance,
        coordinate_atlas_contract=atlas,
    )
    if definition.primitive_source != "ORACLE_CONTRACT":
        raise GICForgeContractError(
            "canonical ORACLE-to-SMITH construction requires the frozen #PRIMITIVES contract"
        )

    controlled_prefixes = (
        "ONIC_CORE ",
        "CHART_ORIENTATION ",
        "CHART_ROLE ",
        "LOCAL_PERCEPTION ",
        "CLOSED_CONTACT_POLICY ",
        "SCIENTIFIC_ATLAS ",
    )
    diagnostics = tuple(
        record
        for record in definition.semantic_diagnostics
        if not record.startswith(controlled_prefixes)
    )
    contract_diagnostics = (
        "ONIC_CORE COMMON_TYPED_NONREDUNDANT_ALGEBRA",
        f"CHART_ORIENTATION {branch.acronym}",
        f"CHART_ROLE {_chart_role(branch.acronym)}",
        *_local_perception_diagnostics(pool.contract),
        f"SCIENTIFIC_ATLAS SCHEMA={atlas.schema} POLICY={atlas.policy_id}@"
        f"{atlas.policy_version} TASK={atlas.task_regime}",
    )
    if has_closing_contact:
        contract_diagnostics = (
            *contract_diagnostics,
            "CLOSED_CONTACT_POLICY "
            + (
                "PSEUDOBOND_CONTACT_NATURAL_COORDINATES"
                if definition.fragment_mode == "PSEUDO_BONDS"
                else "SPECIAL_COORDINATES"
            ),
        )
    return replace(
        definition,
        semantic_diagnostics=(*diagnostics, *contract_diagnostics),
    )


def write_gic_definition_from_oracle_contract(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    orientation: str = "SONIC",
    scientific_path: str = "MINIMUM",
    family_quotas: Mapping[str, int] | None = None,
    rank_tolerance: float = RANK_TOLERANCE,
) -> GICDefinition:
    """Serialize a strict ONIC chart without invoking chemical perception."""

    from matrix_core import read_sectioned_lines, replace_section, section_content

    target = Path(path)
    if not section_content(read_sectioned_lines(target), "FRAGMENTS"):
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(target)

    from .definition import gic_definition_section_lines

    definition = build_gic_definition_from_oracle_contract(
        target,
        symmetrize=symmetrize,
        orientation=orientation,
        scientific_path=scientific_path,
        family_quotas=family_quotas,
        rank_tolerance=rank_tolerance,
    )
    replace_section(target, "GIC", gic_definition_section_lines(definition))
    if sycart:
        from .definition import (
            build_sycart_definition_from_xyzin,
            sycart_definition_section_lines,
        )

        sycart_definition = build_sycart_definition_from_xyzin(target)
        replace_section(
            target,
            "SYCART",
            sycart_definition_section_lines(sycart_definition),
        )
    return definition


def _validate_frozen_fragment_handoff(
    path: Path,
    contract: OracleSonicContract,
) -> None:
    """Require the built fragment transport to match ORACLE's frozen partition."""

    from matrix_fragments import read_fragment_records

    observed = tuple(
        (record.identifier, tuple(int(atom) for atom in record.atoms))
        for record in read_fragment_records(Path(path))
    )
    expected = tuple(
        (record.fragment_id, tuple(int(atom) for atom in record.atoms))
        for record in contract.primary_topology.fragments
    )
    if observed != expected:
        raise GICForgeContractError(
            "#FRAGMENTS must be built and must match the frozen ORACLE partition"
        )


def _normalized_scientific_path(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_")
    if normalized not in {"MINIMUM", "TRANSITION_STATE"}:
        raise ValueError("scientific_path must be 'MINIMUM' or 'TRANSITION_STATE'")
    return normalized


def _chart_role(orientation: str) -> str:
    return {
        "TONIC": "GENERAL",
        "CONIC": "EXPLORATION",
        "SONIC": "EXPLOITATION",
    }[str(orientation).strip().upper()]


def _local_perception_diagnostics(contract: OracleSonicContract) -> tuple[str, ...]:
    """Expose frozen ORACLE decisions without re-perceiving local chemistry."""

    domains_by_id = {domain.domain_id: domain for domain in contract.local_perception_domains}
    records = []
    for domain_id in sorted(domains_by_id):
        domain = domains_by_id[domain_id]
        classes = ",".join(
            "+".join(str(atom) for atom in item.members)
            for item in domain.equivalence_classes
        )
        template = domain.template_decision
        records.append(
            "LOCAL_PERCEPTION "
            f"ID={domain.domain_id} KIND={domain.kind} GROUP={domain.proposed_group} "
            f"CONFIDENCE={domain.confidence} CLASSES={classes or 'NONE'} "
            f"TEMPLATE={(template.selected_template if template else None) or 'NONE'} "
            f"STATUS={(template.status if template else 'NOT_APPLICABLE')} "
            f"PROVIDER={domain.provider}@{domain.provider_version}"
        )
    return tuple(records)


def _validate_frozen_geometry(
    path: Path,
    contract: OracleSonicContract,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
) -> None:
    topology = contract.primary_topology
    if coords.shape != (topology.natoms, 3):
        raise GICForgeContractError("ORACLE contract atom count does not match the geometry")
    try:
        numbers = tuple(atomic_number(symbol) for symbol in atom_symbols)
    except (KeyError, ValueError) as exc:
        raise GICForgeContractError("geometry contains an unsupported atomic symbol") from exc
    if numbers != topology.atomic_numbers:
        raise GICForgeContractError("ORACLE contract atomic numbers do not match the geometry")
    try:
        identity = read_geometry_identity_certificate(Path(path))
        validate_geometry_identity_certificate(
            identity,
            canonical_atoms=atom_symbols,
            canonical_coordinates_angstrom=coords,
        )
    except GeometryIdentityError as exc:
        raise GICForgeContractError(
            f"invalid frozen ORACLE Cartesian provenance: {exc}"
        ) from exc
    if identity.canonical_geometry_sha256 != contract.reference_geometry_sha256:
        raise GICForgeContractError(
            "ORACLE SONIC reference geometry contradicts Cartesian provenance"
        )
    permitted_relation = identity.relation in {
        GEOMETRY_IDENTICAL,
        GEOMETRY_RIGID_FRAME_ONLY,
    } or (
        identity.relation == GEOMETRY_TRUE_CHANGE
        and identity.geometry_change_authorization
        == GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION
    )
    if not permitted_relation:
        raise GICForgeContractError(
            "production SONIC requires a proper rigid frame or an explicitly "
            "ORACLE-authorized symmetry projection; "
            f"ORACLE classified {identity.relation} with authorization "
            f"{identity.geometry_change_authorization}"
        )
    try:
        snapshot = topology_snapshot_from_xyzin(Path(path))
        symmetry = read_molecular_symmetry(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(f"invalid frozen ORACLE topology/symmetry: {exc}") from exc
    snapshot_bonds = tuple(tuple(int(atom) for atom in pair) for pair in snapshot["bonds"])
    snapshot_rings = tuple(tuple(int(atom) for atom in row["atoms"]) for row in snapshot["rings"])
    if snapshot_bonds != topology.bonds or snapshot_rings != topology.rings:
        raise GICForgeContractError(
            "ORACLE SONIC PRIMARY_TOPOLOGY contradicts the frozen #TOPOLOGY section"
        )
    operation_permutations = tuple(operation.permutation for operation in symmetry.operations)
    expected_permutations = topology.symmetry_permutations
    if operation_permutations and set(operation_permutations) != set(expected_permutations):
        raise GICForgeContractError(
            "ORACLE SONIC symmetry permutations contradict the frozen #SYMMETRY section"
        )


def _validate_candidate_ownership(contract: OracleSonicContract) -> None:
    domains = {record.domain_id: record for record in contract.multicenter_domains}
    contacts = {record.contact_id: record for record in contract.auxiliary_contacts}
    sites = {record.site_id for record in contract.structural_sites}
    referenced: dict[str, str] = {}
    for domain in domains.values():
        for candidate_id in domain.primitive_candidate_ids:
            referenced[candidate_id] = domain.domain_id
    for contact in contacts.values():
        for candidate_id in contact.primitive_candidate_ids:
            previous = referenced.setdefault(candidate_id, contact.contact_id)
            if previous != contact.contact_id:
                raise GICForgeContractError(
                    f"primitive candidate {candidate_id} is claimed by multiple domains"
                )
    for candidate in contract.primitive_candidates:
        owner = candidate.owner_id
        if owner == "PRIMARY_TOPOLOGY":
            if candidate.domain_id != "PRIMARY_TOPOLOGY" and not (
                candidate.domain_id.startswith("PRIMARY_TOPOLOGY::")
            ):
                raise GICForgeContractError(
                    f"primitive candidate {candidate.candidate_id} has contradictory topology ownership"
                )
            continue
        if owner in domains:
            if referenced.get(candidate.candidate_id) != owner:
                raise GICForgeContractError(
                    f"multicenter primitive {candidate.candidate_id} is not claimed by its domain"
                )
            continue
        if owner in contacts:
            if referenced.get(candidate.candidate_id) != owner:
                raise GICForgeContractError(
                    f"contact primitive {candidate.candidate_id} is not claimed by its contact"
                )
            continue
        if owner in sites and candidate.domain_id == "STRUCTURAL_SITES":
            continue
        if candidate.domain_id == "FRAGMENT_POSE" and owner.startswith("FRAGMENT_PAIR:"):
            continue
        raise GICForgeContractError(
            f"primitive candidate {candidate.candidate_id} has an unknown ORACLE owner"
        )


def _primitive_from_candidate(candidate: PrimitiveCandidate) -> GICPrimitive:
    frame_atoms, ref_frame_atoms, refs = _split_frame_refs(candidate.refs)
    return GICPrimitive(
        identifier=candidate.candidate_id,
        name=candidate.candidate_id,
        family=candidate.family,
        function=candidate.function,
        atoms=candidate.atoms,
        mode=candidate.mode,
        ref_atoms=candidate.ref_atoms,
        refs=refs,
        frame_atoms=frame_atoms,
        ref_frame_atoms=ref_frame_atoms,
        provenance=candidate.provenance,
        semantic_id=candidate.owner_id,
        semantic_type=candidate.domain_id,
    )


def _split_frame_refs(
    refs: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    frame: tuple[int, ...] = ()
    ref_frame: tuple[int, ...] = ()
    retained: list[str] = []
    for item in refs:
        if item.startswith("FRAME_ATOMS="):
            frame = _parse_atom_csv(item.split("=", 1)[1])
        elif item.startswith("REF_FRAME_ATOMS="):
            ref_frame = _parse_atom_csv(item.split("=", 1)[1])
        else:
            retained.append(item)
    return frame, ref_frame, tuple(retained)


def _parse_atom_csv(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(",") if value)
    except ValueError as exc:
        raise GICForgeContractError("invalid ORACLE primitive frame metadata") from exc
    if len(set(values)) != len(values):
        raise GICForgeContractError("ORACLE primitive frame repeats atom indices")
    return values


def _validated_supplied_row(
    candidate: PrimitiveCandidate,
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    supplied = np.asarray(candidate.analytic_wilson_row, dtype=float)
    if float(np.linalg.norm(supplied)) <= RANK_TOLERANCE:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} has a null Wilson row"
        )
    try:
        evaluated = _analytic_b_row(
            primitive,
            coords,
            reference_coords=coords if primitive.function == "FROT" else None,
        )
    except (ArithmeticError, FloatingPointError, ValueError) as exc:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} has no finite SMITH realization"
        ) from exc
    scale = max(1.0, float(np.linalg.norm(supplied)), float(np.linalg.norm(evaluated)))
    residual = float(np.linalg.norm(supplied - evaluated))
    if residual > _ROW_VALIDATION_ATOL * scale:
        raise GICForgeContractError(
            f"ORACLE primitive {candidate.candidate_id} Wilson row contradicts its finite function "
            f"(residual {residual:.6g})"
        )
    return supplied


def _normalized_family_quotas(quotas: Mapping[str, int] | None) -> dict[str, int]:
    if quotas is None:
        return {}
    result = {str(family): int(count) for family, count in quotas.items()}
    if any(not family or count < 0 for family, count in result.items()):
        raise GICForgeContractError("family quotas must be named non-negative integers")
    return result


def _select_indices(
    contract: OracleSonicContract,
    rows: tuple[np.ndarray, ...],
    *,
    target_rank: int,
    family_quotas: dict[str, int],
    rank_tolerance: float,
) -> tuple[int, ...]:
    if target_rank == 0:
        return ()
    norms = np.asarray([np.linalg.norm(row) for row in rows], dtype=float)
    normalized = np.vstack(
        [row / norm if norm > rank_tolerance else row for row, norm in zip(rows, norms, strict=True)]
    )
    if family_quotas:
        if sum(family_quotas.values()) != target_rank:
            raise GICForgeContractError("exact family quotas must sum to the target Wilson rank")
        records = [
            (
                candidate.family,
                "",
                (),
                np.zeros(0),
                normalized[index],
                index,
                index,
            )
            for index, candidate in enumerate(contract.primitive_candidates)
        ]
        selected = _partition_quota_rank_revealing_selection(
            records,
            quotas={family: count for family, count in family_quotas.items() if count},
        )
        if selected is None:
            raise GICForgeContractError("ORACLE primitive pool cannot satisfy exact family quotas")
        return tuple(int(record[6]) for record in selected)

    family_order = {family: index for index, family in enumerate(PRIMITIVE_FAMILY_ORDER)}
    priorities = tuple(
        0
        if candidate.family in SPECIAL_PRIMITIVE_FAMILIES
        else 1 + family_order.get(candidate.family, len(family_order))
        for candidate in contract.primitive_candidates
    )
    selection = select_rank_revealing_rows(
        normalized,
        target_rank=target_rank,
        tolerance=rank_tolerance,
        priorities=priorities,
        tie_tolerance=1.0e-12,
    )
    return selection.indices


__all__ = [
    "FrozenOracleCandidateSelection",
    "FrozenOracleCandidatePool",
    "SMITH_ORACLE_CONSUMER_SCHEMA",
    "build_gic_definition_from_oracle_contract",
    "load_validated_oracle_candidate_pool",
    "select_frozen_oracle_candidates",
    "write_gic_definition_from_oracle_contract",
]
