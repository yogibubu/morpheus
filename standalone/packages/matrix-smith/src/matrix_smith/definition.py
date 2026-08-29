from __future__ import annotations
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_MINIMUM,
    ATLAS_TASK_TRANSITION_STATE,
    BODY_NONLINEAR,
    COORDINATE_PHYSICAL_CONTACT_DISTANCE,
    COORDINATE_REACTION_DISTANCE,
    COORDINATE_REACTION_DISTANCE_ONLY,
    COORDINATE_TS_SUPPORT,
    GRAPH_ROLE_REACTIVE_SUPPORT,
    PSEUDOBOND_REQUIRED,
    OracleCoordinateAtlasContract,
    read_oracle_coordinate_atlas_contract,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.topology.elements import atomic_number
from matrix_chem import (
    MATRIX_XYZ_PRIMITIVES_SCHEMA,
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_molecular_symmetry,
    read_oracle_sonic_contract,
    read_oracle_transition_state_geometry_contract,
    read_primitive_contract,
    TS_CHART_REACTIVE_DISTANCE,
    TS_CHART_REACTIVE_PSEUDOBOND,
    transition_state_descriptor,
    validate_primitive_contract,
    write_validation_section,
)
from matrix_chem.primitive_coordinates import (
    LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA,
    LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA,
)
from matrix_core import read_sectioned_lines, replace_section, section_content

from .contracts import (
    ORACLE_XYZ_GIC_SCHEMA,
    ORACLE_XYZ_SYCART_SCHEMA,
    GICForgeContractError,
    GICForgeRankDeficiencyError,
    validate_gicforge_prerequisites,
)
from .basis_validation import validate_frozen_sonic_basis
from .component_invariants import validate_indivisible_gic_components
from .definition_io import (
    _apply_symmetry_group_limit,
    _key_values,
    _parse_atom_list,
    _point_group,
    _subsection,
    _symmetry_operations,
)
from .periodic_estimates import (
    parse_periodic_coordinate_estimate,
    periodic_coordinate_estimate_line,
)
from .semantic import (
    AUTO_PROVENANCE,
    SEMANTIC_GRAMMAR_VERSION,
    semantic_contract_from_sectioned_lines,
)
from .policy import (
    AROMATIC_LOCAL_MODEL_DIAGNOSTIC,
    FRAGMENT_MODE_NONE,
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
    FRAGMENT_MODES,
    GIC_BACKEND,
    LOCAL_SYMMETRIZATION_METHOD,
    POINT_GROUP_PROJECTOR_METHOD,
    RANK_METHOD,
    RANK_TOLERANCE,
    REDUCTION_POLICY,
    SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SYMMETRIZATION_POLICY,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
    SYCART_BACKEND,
    XH_STRETCH_CLASSES,
    XH_STRETCH_POLICIES,
    XH_STRETCH_POLICY_LOCAL_ALL,
    XH_STRETCH_POLICY_LOCAL_SELECTED,
    XH_STRETCH_POLICY_SYMMETRIZE,
)
from .symmetry_labels import (
    is_total_symmetric_irrep,
    total_symmetric_irrep,
)
from .models import (
    FallbackEvent,
    FrozenGIC,
    GICBMatrix,
    GICDefinition,
    GICPointGroupOperation,
    GICPrimitive,
    GICReductionDiagnostics,
    GICSymmetrizationDiagnostics,
    GICSymmetrizedGroup,
    SYCartDefinition,
)
from .minimum_domain_conditioning import condition_minimum_pseudobond_chart
from .ring_salc_materialization import materialize_ring_out_of_plane_salcs
from .topology_io import (
    _topology_bonds,
    _topology_rings,
    topology_aromatic_atoms_from_lines,
    topology_bond_order_components_from_lines,
    topology_bond_orders_from_lines,
)
from .symmetrization import (
    _apply_local_symmetrization,
    _empty_symmetry_diagnostics,
    _protected_gic_count,
    _reduction_diagnostics_lines,
    _renumber_frozen_gic,
    _skipped_dependent_count,
    _skipped_singular_count,
    _symmetry_closed_projector_primitives,
    _symmetry_diagnostics_lines,
)
from .cartesian_blocks import CARTESIAN_BLOCK_GAUGE, symmetry_adapted_cartesian_basis
from .coordinate_registry import MODE_BEARING_PRIMITIVE_FUNCTIONS
from .atlas_realization import (
    apply_reactive_zone_exclusions,
    validate_atlas_chart_realization,
)
from .fallback_ledger import (
    build_fallback_ledger,
    fallback_ledger_section_lines,
    fallback_provenance_from_lines,
    make_fallback_event,
    merge_fallback_events,
)


_GAUSSIAN_DEPENDENCY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_PHASE_ZERO_TOLERANCE = 1.0e-10
_FRAGMENT_CONTEXT_MINIMUM = "minimum"
_FRAGMENT_CONTEXT_OPTIMIZATION = "optimization"
_FRAGMENT_CONTEXT_EXPLORATION = "exploration"
_FRAGMENT_CONTEXT_TRANSITION_STATE = "transition_state"


@dataclass(frozen=True)
class _ChartExecutionContext:
    coordinate_atlas_contract: OracleCoordinateAtlasContract
    topology_bonds: tuple[tuple[int, int], ...]
    rings: tuple[tuple[int, tuple[int, ...]], ...]
    coords: np.ndarray
    atom_symbols: tuple[str, ...]
    xh_stretch_policy: str
    local_xh_bonds: tuple[tuple[int, int], ...]
    local_xh_classes: tuple[str, ...]
    body_prescriptions: tuple[object, ...]
    fragment_contacts: tuple[tuple[int, int, str], ...]
    semantic_contract: object
    symmetry_operations: tuple[GICPointGroupOperation, ...]
    prescribed_ts_reaction_distances: tuple[tuple[int, int], ...]
    reactive_support_edges: tuple[tuple[int, int], ...]
    bond_orders: dict[tuple[int, int], float]
    aromatic_atoms: frozenset[int]
    ring_puckering_model: str
    target_rank: int
    rank_tolerance: float
    task_regime: str
    condition_ordinary: bool
    reactive_ts: bool
    pseudobond_out_of_plane_support: bool
    condition_transition_state_exact_chart: bool




def build_gic_definition_from_xyzin(
    path: Path,
    *,
    symmetrize: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    rank_tolerance: float = RANK_TOLERANCE,
    coordinate_atlas_contract: OracleCoordinateAtlasContract | None = None,
) -> GICDefinition:
    """Build a frozen SONIC coordinate definition from saved xyzin state."""
    context = _fragment_context(fragment_context)
    separate_exocyclic_torsions = _resolved_separate_exocyclic_torsions(
        Path(path),
        context=context,
        requested=separate_exocyclic_torsions,
    )
    coordinate_atlas_contract = _required_coordinate_atlas_contract(
        Path(path),
        context=context,
        supplied=coordinate_atlas_contract,
    )
    prescribed_fragment_mode = _atlas_fragment_mode(coordinate_atlas_contract)
    if (
        fragment_mode is not None
        and _fragment_mode(fragment_mode) != prescribed_fragment_mode
    ):
        raise GICForgeContractError(
            "fragment_mode cannot override the frozen ORACLE coordinate atlas"
        )
    fragment_mode = prescribed_fragment_mode
    construct_kwargs = {
        "symmetry_group": symmetry_group,
        "improper_dihedrals": improper_dihedrals,
        "fragment_mode": fragment_mode,
        "fragment_context": fragment_context,
        "xh_stretch_policy": xh_stretch_policy,
        "local_xh_bonds": local_xh_bonds,
        "local_xh_classes": local_xh_classes,
        "local_salc": local_salc,
        "local_salc_settings": local_salc_settings,
        "xy3_torsions": xy3_torsions,
        "xy2_torsions": xy2_torsions,
        "separate_exocyclic_torsions": separate_exocyclic_torsions,
        "ring_puckering_model": ring_puckering_model,
        "rank_tolerance": rank_tolerance,
        "retain_candidate_primitives": (
            symmetrize or bool(coordinate_atlas_contract.local_domains)
        ),
        "coordinate_atlas_contract": coordinate_atlas_contract,
        # Preserve the established local/SALC chart whenever its final frozen
        # Jacobian passes. Exact-rank primitive pivoting is a fail-closed
        # recovery for a final C1 TS chart that actually exceeds the gate.
        "condition_transition_state_exact_chart": False,
    }
    definition, atom_symbols, operations = construct_gic_definition_from_xyzin(
        path, **construct_kwargs
    )
    if not symmetrize:
        definition = condition_minimum_pseudobond_chart(
            definition,
            coordinate_atlas_contract,
            sonic_contract=read_oracle_sonic_contract(Path(path)),
            rank_tolerance=rank_tolerance,
        )
        frozen = _with_periodic_coordinate_estimates(
            definition,
            Path(path),
            native_definition=definition,
        )
        validate_indivisible_gic_components(frozen)
        validate_frozen_sonic_basis(frozen, rank_tolerance=rank_tolerance)
        validate_atlas_chart_realization(frozen, coordinate_atlas_contract)
        return _attach_chart_atlas(frozen)
    sonic_contract = read_oracle_sonic_contract(Path(path))

    def finalize_symmetrized_chart(
        native_definition: GICDefinition,
        native_atom_symbols: tuple[str, ...],
        native_operations: tuple[GICPointGroupOperation, ...],
    ) -> GICDefinition:
        candidate = _condition_transition_state_salc_alternatives(
            native_definition,
            context=context,
            atom_symbols=native_atom_symbols,
            symmetry_operations=native_operations,
            coordinate_atlas_contract=coordinate_atlas_contract,
            separate_exocyclic_torsions=separate_exocyclic_torsions,
        )
        candidate = condition_minimum_pseudobond_chart(
            candidate,
            coordinate_atlas_contract,
            sonic_contract=sonic_contract,
            rank_tolerance=rank_tolerance,
        )
        return _with_periodic_coordinate_estimates(
            candidate,
            Path(path),
            native_definition=native_definition,
        )

    frozen = finalize_symmetrized_chart(definition, atom_symbols, operations)
    validate_indivisible_gic_components(frozen)
    try:
        validate_frozen_sonic_basis(frozen, rank_tolerance=rank_tolerance)
    except GICForgeContractError as local_error:
        if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
            raise
        conditioned_definition, conditioned_symbols, conditioned_operations = (
            construct_gic_definition_from_xyzin(
                path,
                **{
                    **construct_kwargs,
                    "condition_transition_state_exact_chart": True,
                },
            )
        )
        conditioned = finalize_symmetrized_chart(
            conditioned_definition,
            conditioned_symbols,
            conditioned_operations,
        )
        try:
            validate_indivisible_gic_components(conditioned)
            validate_frozen_sonic_basis(conditioned, rank_tolerance=rank_tolerance)
        except GICForgeContractError:
            raise local_error
        frozen = replace(
            conditioned,
            semantic_diagnostics=(
                *conditioned.semantic_diagnostics,
                "TS_EXACT_CHART_CONDITIONING FINAL_SALC_GATE_REBUILD=CONDITIONED",
            ),
        )
    validate_atlas_chart_realization(frozen, coordinate_atlas_contract)
    return _attach_chart_atlas(frozen)


def _required_transition_state_contract(path: Path):
    """Read ORACLE's complete TS prescription and verify its frozen topology."""

    try:
        contract = read_oracle_transition_state_geometry_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "an explicit transition_state build requires a valid ORACLE single-geometry TS contract"
        ) from exc
    if contract.natoms != len(read_enriched_xyz(Path(path)).atoms):
        raise GICForgeContractError("ORACLE TS contract atom count is stale")
    try:
        sonic_contract = read_oracle_sonic_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "the ORACLE TS contract requires its frozen SONIC topology"
        ) from exc
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE TS contract topology fingerprint is stale")
    return contract


def _resolved_separate_exocyclic_torsions(
    path: Path,
    *,
    context: str,
    requested: bool,
) -> bool:
    """Consume ORACLE's torsion prescription for TS; retain the non-TS option."""

    if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
        return bool(requested)
    contract = _required_transition_state_contract(path)
    prescribed = (
        transition_state_descriptor(contract, "SEPARATE_EXOCYCLIC_TORSIONS") == "TRUE"
    )
    if requested and not prescribed:
        raise GICForgeContractError(
            "separate_exocyclic_torsions cannot override the ORACLE transition-state contract"
        )
    return prescribed


def _fragment_context(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == _FRAGMENT_CONTEXT_OPTIMIZATION:
        return _FRAGMENT_CONTEXT_MINIMUM
    if normalized not in {
        _FRAGMENT_CONTEXT_MINIMUM,
        _FRAGMENT_CONTEXT_EXPLORATION,
        _FRAGMENT_CONTEXT_TRANSITION_STATE,
    }:
        raise ValueError("fragment_context must be 'minimum', 'exploration' or 'transition_state'")
    return normalized


def _required_coordinate_atlas_contract(
    path: Path,
    *,
    context: str,
    supplied: OracleCoordinateAtlasContract | None = None,
) -> OracleCoordinateAtlasContract:
    """Load the sole ORACLE scientific prescription and validate its scope."""

    try:
        contract = supplied or read_oracle_coordinate_atlas_contract(Path(path))
        validate_oracle_coordinate_atlas_contract(contract)
    except ValueError as exc:
        raise GICForgeContractError(
            "SMITH requires a valid frozen ORACLE coordinate atlas; "
            "scientific chart choices cannot be reconstructed downstream"
        ) from exc
    expected_task = (
        ATLAS_TASK_TRANSITION_STATE
        if context == _FRAGMENT_CONTEXT_TRANSITION_STATE
        else ATLAS_TASK_MINIMUM
    )
    if contract.task_regime != expected_task:
        raise GICForgeContractError(
            "ORACLE coordinate atlas task regime contradicts the requested scientific path"
        )
    try:
        sonic_contract = read_oracle_sonic_contract(Path(path))
    except ValueError as exc:
        raise GICForgeContractError(
            "the coordinate atlas requires its frozen ORACLE SONIC contract"
        ) from exc
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE coordinate atlas topology fingerprint is stale")
    return contract


def _atlas_fragment_mode(contract: OracleCoordinateAtlasContract) -> str:
    """Translate ORACLE's frozen graph effects to an execution mode only."""

    validate_oracle_coordinate_atlas_contract(contract)
    if any(
        item.pseudobond_policy == PSEUDOBOND_REQUIRED
        for item in contract.interactions
    ):
        return FRAGMENT_MODE_PSEUDO_BONDS
    return (
        FRAGMENT_MODE_SPECIAL_COORDINATES
        if len(contract.bodies) > 1
        else FRAGMENT_MODE_NONE
    )


def _atlas_needs_fragment_geometry(
    contract: OracleCoordinateAtlasContract,
) -> bool:
    """Return whether mathematical chart construction needs fragment atoms."""

    return bool(contract.bodies) or (
        any(
            item.pseudobond_policy == PSEUDOBOND_REQUIRED
            for item in contract.interactions
        )
    )


def _atlas_declares_pseudobond_out_of_plane_support(
    contract: OracleCoordinateAtlasContract,
) -> bool:
    """Return whether the frozen atlas admits U in local contact completion."""

    return any(
        item.block_id == "TS_REACTIVE_COMPLETION"
        and "OUT_OF_PLANE" in item.families
        for item in contract.family_compatibility
    )


def _atlas_atom_contacts(
    contract: OracleCoordinateAtlasContract,
    *,
    pseudobond_policy: str | None = None,
    coordinate_role: str | None = None,
) -> tuple[tuple[int, int, str], ...]:
    """Return exact atom contacts; no geometry or chemistry is interpreted."""

    records = []
    for item in contract.interactions:
        if pseudobond_policy is not None and item.pseudobond_policy != pseudobond_policy:
            continue
        if coordinate_role is not None and item.coordinate_role != coordinate_role:
            continue
        if item.endpoint_a[0] != "ATOM" or item.endpoint_b[0] != "ATOM":
            continue
        left, right = sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1])))
        records.append(
            (
                left,
                right,
                item.semantic_kind,
            )
        )
    return tuple(records)


def _atlas_graph_edges(
    contract: OracleCoordinateAtlasContract,
    *,
    graph_role: str,
) -> tuple[tuple[int, int], ...]:
    """Return atom edges carrying one explicit ORACLE graph role."""

    return tuple(
        dict.fromkeys(
            tuple(sorted((int(item.endpoint_a[1]), int(item.endpoint_b[1]))))
            for item in contract.interactions
            if item.graph_role == graph_role
            and item.endpoint_a[0] == item.endpoint_b[0] == "ATOM"
        )
    )


def _task_context_diagnostics(
    context: str,
    *,
    transition_state_contract: object | None,
) -> tuple[str, ...]:
    records = [f"SCIENTIFIC_PATH {context.upper()}"]
    if context != _FRAGMENT_CONTEXT_TRANSITION_STATE:
        return tuple(records)
    if transition_state_contract is None:
        raise GICForgeContractError("missing ORACLE transition-state contract")
    records.extend(
        (
            f"TS_SOURCE {getattr(transition_state_contract, 'source')}",
            f"TS_GEOMETRY_CLASS {getattr(transition_state_contract, 'category_id')}",
            f"TS_CATALOG {getattr(transition_state_contract, 'catalog_id')}@"
            f"{getattr(transition_state_contract, 'catalog_version')}",
            f"TS_CHART_POLICY {getattr(transition_state_contract, 'chart_policy')}",
            f"TS_FROM_ENDPOINTS {getattr(transition_state_contract, 'endpoints_route_status')}",
        )
    )
    return tuple(records)


def _retain_individual_exocyclic_torsions(
    symmetrized: GICDefinition,
    unsymmetrized: GICDefinition,
) -> GICDefinition:
    """Restore one-dihedral exocyclic rows after retained-group adaptation."""

    source = tuple(gic for gic in unsymmetrized.gics if gic.family == "TORSION")
    if not source:
        return symmetrized
    output: list[FrozenGIC] = []
    inserted = False
    for gic in symmetrized.gics:
        if gic.family == "TORSION":
            if not inserted:
                output.extend(source)
                inserted = True
            continue
        output.append(gic)
    if not inserted:
        output.extend(source)
    renumbered = tuple(_renumber_frozen_gic(gic, index) for index, gic in enumerate(output, 1))
    diagnostics = symmetrized.symmetry_diagnostics
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            status=f"{diagnostics.status}_EXOCYCLIC_TORSIONS_SEPARATE",
            total_symmetric_gics=tuple(
                gic.name
                for gic in renumbered
                if gic.family != "TORSION"
                and is_total_symmetric_irrep(symmetrized.point_group, gic.irrep)
            ),
            groups=tuple(group for group in diagnostics.groups if group.family != "TORSION"),
        )
    return replace(
        symmetrized,
        gics=renumbered,
        symmetry_diagnostics=diagnostics,
    )


def construct_gic_definition_from_xyzin(
    path: Path,
    *,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    rank_tolerance: float = RANK_TOLERANCE,
    retain_candidate_primitives: bool = False,
    coordinate_atlas_contract: OracleCoordinateAtlasContract | None = None,
    condition_transition_state_exact_chart: bool = True,
) -> tuple[GICDefinition, tuple[str, ...], tuple[GICPointGroupOperation, ...]]:
    """Construct and reduce GICs without applying symmetry adaptation."""
    target = Path(path)
    validate_gicforge_prerequisites(target)
    lines = read_sectioned_lines(target)
    semantic_contract = semantic_contract_from_sectioned_lines(lines)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    point_group = _point_group(lines)
    symmetry_operations = _symmetry_operations(lines)
    point_group, symmetry_operations = _apply_symmetry_group_limit(
        point_group,
        symmetry_operations,
        symmetry_group,
    )
    context = _fragment_context(fragment_context)
    coordinate_atlas_contract = _required_coordinate_atlas_contract(
        target,
        context=context,
        supplied=coordinate_atlas_contract,
    )
    transition_state_contract = (
        _required_transition_state_contract(target)
        if context == _FRAGMENT_CONTEXT_TRANSITION_STATE
        else None
    )
    separate_exocyclic_torsions = _resolved_separate_exocyclic_torsions(
        target,
        context=context,
        requested=separate_exocyclic_torsions,
    )
    mode = _atlas_fragment_mode(coordinate_atlas_contract)
    if fragment_mode is not None and _fragment_mode(fragment_mode) != mode:
        raise GICForgeContractError(
            "fragment_mode cannot override the frozen ORACLE coordinate atlas"
        )
    fragment_records = _fragment_records(target)
    # A complete one-component fragment contract is required before
    # validation, but it must be coordinate-neutral for a connected molecule.
    if len(fragment_records) <= 1:
        fragment_records = ()
    planned_xh_policy = _planned_xh_stretch_policy(lines)
    explicit_local_xh_selection = local_xh_bonds is not None or local_xh_classes is not None
    resolved_xh_policy = _xh_stretch_policy(
        XH_STRETCH_POLICY_LOCAL_SELECTED
        if xh_stretch_policy is None and explicit_local_xh_selection
        else planned_xh_policy
        if xh_stretch_policy is None
        else xh_stretch_policy
    )
    resolved_local_xh_bonds = _normalize_pairs(
        _planned_local_xh_bonds(lines) if local_xh_bonds is None else local_xh_bonds
    )
    resolved_local_xh_classes = _normalize_xh_classes(
        _planned_local_xh_classes(lines) if local_xh_classes is None else local_xh_classes
    )
    interaction_centers = _interaction_center_definition(target)
    bonds = _topology_bonds(lines, natoms=geometry.natoms)
    prescribed_pseudobonds = _atlas_atom_contacts(
        coordinate_atlas_contract,
        pseudobond_policy=PSEUDOBOND_REQUIRED,
    )
    prescribed_ts_reaction_distances = tuple(
        dict.fromkeys(
            tuple(sorted((left, right)))
            for role in (
                COORDINATE_REACTION_DISTANCE,
                COORDINATE_REACTION_DISTANCE_ONLY,
                COORDINATE_TS_SUPPORT,
            )
            for left, right, _kind in _atlas_atom_contacts(
                coordinate_atlas_contract,
                coordinate_role=role,
            )
        )
    )
    reactive_support_edges = _atlas_graph_edges(
        coordinate_atlas_contract,
        graph_role=GRAPH_ROLE_REACTIVE_SUPPORT,
    )
    task_context_diagnostics = _task_context_diagnostics(
        context,
        transition_state_contract=transition_state_contract,
    )
    rings = _topology_rings(lines, natoms=geometry.natoms)
    aromatic_atoms = topology_aromatic_atoms_from_lines(lines, natoms=geometry.natoms)
    bond_orders = topology_bond_orders_from_lines(lines, natoms=geometry.natoms)
    bond_order_components = topology_bond_order_components_from_lines(lines, natoms=geometry.natoms)
    primitive_source, primitive_source_schema, primitive_b_matrix_sha256 = (
        _consume_oracle_primitive_contract(target, coords, bonds=bonds)
    )
    fragment_index_by_atom = _fragment_index_by_atom(fragment_records)
    physical_fragment_contacts: dict[tuple[int, int], str] = {}
    atlas_distance_contacts = tuple(
        contact
        for contact in _atlas_atom_contacts(
            coordinate_atlas_contract,
            coordinate_role=COORDINATE_PHYSICAL_CONTACT_DISTANCE,
        )
        if tuple(sorted((int(contact[0]), int(contact[1]))))
        not in {
            tuple(sorted((int(left), int(right))))
            for left, right, _kind in prescribed_pseudobonds
        }
    )
    for left, right, kind in atlas_distance_contacts:
        pair = tuple(sorted((int(left), int(right))))
        if pair in bonds or pair[0] == pair[1]:
            continue
        if (
            pair[0] not in fragment_index_by_atom
            or pair[1] not in fragment_index_by_atom
            or fragment_index_by_atom[pair[0]] == fragment_index_by_atom[pair[1]]
        ):
            continue
        physical_fragment_contacts.setdefault(pair, str(kind))
    fragment_contact_candidates = tuple(
        (left, right, physical_fragment_contacts[(left, right)])
        for left, right in sorted(physical_fragment_contacts)
    )

    if (
        context != _FRAGMENT_CONTEXT_TRANSITION_STATE
        and not semantic_contract.protect_coordinates
        and not fragment_records
        and not _requires_interaction_center_coordinates(interaction_centers)
        and mode != FRAGMENT_MODE_PSEUDO_BONDS
    ):
        definition = _connected_minimum_definition(
            atom_symbols=tuple(geometry.atoms),
            coords=coords,
            bonds=bonds,
            point_group=point_group,
            xh_stretch_policy=resolved_xh_policy,
            local_xh_bonds=resolved_local_xh_bonds,
            local_xh_classes=resolved_local_xh_classes,
            local_salc=local_salc,
            local_salc_settings=local_salc_settings,
            xy3_torsions=xy3_torsions,
            xy2_torsions=xy2_torsions,
            separate_exocyclic_torsions=separate_exocyclic_torsions,
            ring_puckering_model=ring_puckering_model,
            rings=rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            primitive_source=primitive_source,
            primitive_source_schema=primitive_source_schema,
            primitive_b_matrix_sha256=primitive_b_matrix_sha256,
            task_context_diagnostics=task_context_diagnostics,
        )
        return definition, tuple(geometry.atoms), symmetry_operations

    pseudo_bonds: tuple[tuple[int, int], ...] = ()
    pseudo_bond_kinds: tuple[str, ...] = ()
    target_rank = _vibrational_rank(coords)
    topology_bonds = bonds
    constrained_disconnected_chart = bool(
        fragment_records
        and any(
            body.dimension != BODY_NONLINEAR
            for body in coordinate_atlas_contract.bodies
        )
    )
    execution_context = _ChartExecutionContext(
        coordinate_atlas_contract=coordinate_atlas_contract,
        topology_bonds=topology_bonds,
        rings=rings,
        coords=coords,
        atom_symbols=tuple(geometry.atoms),
        xh_stretch_policy=resolved_xh_policy,
        local_xh_bonds=resolved_local_xh_bonds,
        local_xh_classes=resolved_local_xh_classes,
        body_prescriptions=coordinate_atlas_contract.bodies,
        fragment_contacts=fragment_contact_candidates,
        semantic_contract=semantic_contract,
        symmetry_operations=symmetry_operations,
        prescribed_ts_reaction_distances=prescribed_ts_reaction_distances,
        reactive_support_edges=reactive_support_edges,
        bond_orders=bond_orders,
        aromatic_atoms=aromatic_atoms,
        ring_puckering_model=ring_puckering_model,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
        task_regime=coordinate_atlas_contract.task_regime,
        condition_ordinary=(
            mode == FRAGMENT_MODE_PSEUDO_BONDS
            and constrained_disconnected_chart
            and not retain_candidate_primitives
        ),
        reactive_ts=(
            transition_state_contract is not None
            and transition_state_contract.chart_policy
            in {TS_CHART_REACTIVE_DISTANCE, TS_CHART_REACTIVE_PSEUDOBOND}
        ),
        pseudobond_out_of_plane_support=(
            _atlas_declares_pseudobond_out_of_plane_support(
                coordinate_atlas_contract
            )
        ),
        condition_transition_state_exact_chart=bool(
            condition_transition_state_exact_chart
        ),
    )

    normalized_contacts = tuple(
        _normalize_pseudo_contact(contact) for contact in prescribed_pseudobonds
    )
    (
        candidate_bonds,
        pseudo_bonds,
        pseudo_bond_kinds,
        candidates,
        semantic_contract,
        semantic_diagnostics,
        fallback_events,
        selected,
        rank,
        reduction_diagnostics,
    ) = _evaluate_atlas_candidate_chart(
        execution_context,
        fragment_records if _atlas_needs_fragment_geometry(coordinate_atlas_contract) else (),
        interaction_centers,
        normalized_contacts,
    )
    bonds = candidate_bonds
    if rank != target_rank:
        raise GICForgeRankDeficiencyError(
            target_rank=target_rank,
            selected_rank=rank,
            candidate_count=len(candidates),
        )

    definition = _selected_chart_definition(
        candidates=tuple(candidates),
        selected=tuple(selected),
        retain_candidate_primitives=retain_candidate_primitives,
        candidate_bonds=candidate_bonds,
        coords=coords,
        atom_symbols=tuple(geometry.atoms),
        point_group=point_group,
        target_rank=target_rank,
        rank=rank,
        reduction_diagnostics=reduction_diagnostics,
        fragment_mode=mode if pseudo_bonds or fragment_records else FRAGMENT_MODE_NONE,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        xh_stretch_policy=resolved_xh_policy,
        local_xh_bonds=resolved_local_xh_bonds,
        local_xh_classes=resolved_local_xh_classes,
        rings=rings,
        bond_orders=bond_orders,
        bond_order_components=bond_order_components,
        aromatic_atoms=aromatic_atoms,
        ring_puckering_model=ring_puckering_model,
        semantic_contract=semantic_contract,
        semantic_diagnostics=semantic_diagnostics,
        fallback_events=fallback_events,
        task_context_diagnostics=task_context_diagnostics,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
    )
    return definition, tuple(geometry.atoms), symmetry_operations


def _connected_minimum_definition(
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    point_group: str,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    local_salc: bool,
    local_salc_settings: object | None,
    xy3_torsions: bool,
    xy2_torsions: bool,
    separate_exocyclic_torsions: bool,
    ring_puckering_model: str,
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict,
    primitive_source: str,
    primitive_source_schema: str,
    primitive_b_matrix_sha256: str,
    task_context_diagnostics: tuple[str, ...],
) -> GICDefinition:
    definition = _construct_merlino_python_definition(
        atom_symbols,
        coords,
        topology_bonds=bonds,
        point_group=point_group,
        improper_dihedrals=False,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
        ring_puckering_diagnostics=_ring_puckering_diagnostics(
            rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            ring_puckering_model=ring_puckering_model,
        ),
    )
    definition = replace(
        definition,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
        semantic_diagnostics=(
            *definition.semantic_diagnostics,
            *task_context_diagnostics,
        ),
        **_wilson_tangent_diagnostics(definition, coords),
    )
    return _migrate_legacy_u_definition(definition)


def _selected_chart_definition(
    *,
    candidates: tuple[GICPrimitive, ...],
    selected: tuple[GICPrimitive, ...],
    retain_candidate_primitives: bool,
    candidate_bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    atom_symbols: tuple[str, ...],
    point_group: str,
    target_rank: int,
    rank: int,
    reduction_diagnostics: GICReductionDiagnostics,
    fragment_mode: str,
    pseudo_bonds: tuple[tuple[int, int], ...],
    pseudo_bond_kinds: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict,
    aromatic_atoms: frozenset[int],
    ring_puckering_model: str,
    semantic_contract: object,
    semantic_diagnostics: tuple[str, ...],
    fallback_events: tuple[FallbackEvent, ...],
    task_context_diagnostics: tuple[str, ...],
    primitive_source: str,
    primitive_source_schema: str,
    primitive_b_matrix_sha256: str,
) -> GICDefinition:
    primitives = candidates if retain_candidate_primitives else selected
    gics = tuple(
        FrozenGIC(
            identifier=f"GIC{idx:03d}",
            name=primitive.name,
            family=primitive.family,
            irrep="A" if point_group.upper() == "C1" else "UNASSIGNED",
            primitive_id=primitive.identifier,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive.identifier, 1.0),),
        )
        for idx, primitive in enumerate(selected, start=1)
    )
    primitives, gics = materialize_ring_out_of_plane_salcs(primitives, gics)
    primitives, gics = _freeze_balanced_out_of_plane_coordinates(
        primitives,
        gics,
        selected=selected,
        bonds=candidate_bonds,
        coords=coords,
        atom_symbols=atom_symbols,
        bond_orders=bond_orders,
    )
    definition = GICDefinition(
        backend=GIC_BACKEND,
        point_group=point_group,
        symmetrize=False,
        target_rank=target_rank,
        rank=rank,
        candidate_count=len(candidates),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
        primitives=primitives,
        gics=gics,
        reduction_diagnostics=reduction_diagnostics,
        symmetry_diagnostics=_empty_symmetry_diagnostics(point_group, requested=False),
        fragment_mode=fragment_mode,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        ring_puckering_diagnostics=_ring_puckering_diagnostics(
            rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
            ring_puckering_model=ring_puckering_model,
            aromatic_atoms=aromatic_atoms,
        ),
        semantic_grammar_version=(
            semantic_contract.grammar_version if semantic_contract.coordinates else ""
        ),
        semantic_diagnostics=(*semantic_diagnostics, *task_context_diagnostics),
        fallback_events=fallback_events,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
    )
    definition = replace(definition, **_wilson_tangent_diagnostics(definition, coords))
    return _migrate_legacy_u_definition(definition)


def _candidate_chart_for_bonds(
    context: _ChartExecutionContext,
    fragment_records: tuple[object, ...],
    interaction_centers: object | None,
    *,
    bonds: tuple[tuple[int, int], ...],
    pseudo_bonds: tuple[tuple[int, int], ...],
    pseudo_bond_kinds: tuple[str, ...],
):
    """Generate and rank one mechanical realization of frozen atlas edges."""

    candidates = _primitive_candidates(
        bonds,
        rings=context.rings,
        coords=context.coords,
        natoms=len(context.atom_symbols),
        atom_symbols=context.atom_symbols,
        xh_stretch_policy=context.xh_stretch_policy,
        local_xh_bonds=context.local_xh_bonds,
        local_xh_classes=context.local_xh_classes,
        improper_dihedrals=False,
        fragment_records=fragment_records,
        body_prescriptions=context.body_prescriptions,
        fragment_contacts=context.fragment_contacts if fragment_records else (),
        interaction_centers=interaction_centers,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        pseudobond_out_of_plane_support=context.pseudobond_out_of_plane_support,
        bond_orders=context.bond_orders,
        aromatic_atoms=context.aromatic_atoms,
        ring_puckering_model=context.ring_puckering_model,
    )
    candidates = _with_prescribed_distance_only_primitives(
        candidates,
        context.prescribed_ts_reaction_distances,
    )
    candidates = _with_transition_state_reaction_distance_family(
        candidates,
        context.prescribed_ts_reaction_distances,
    )
    candidates, reactive_zone_diagnostics = apply_reactive_zone_exclusions(
        candidates,
        context.coordinate_atlas_contract,
    )
    candidates, semantic_contract, semantic_diagnostics, fallback_events = (
        _apply_semantic_contract_to_candidates(
            candidates,
            context.semantic_contract,
            symmetry_operations=context.symmetry_operations,
        )
    )
    selected, rank, reduction_diagnostics = _select_rank_aware_chart(
        candidates,
        context.coords,
        target_rank=context.target_rank,
        rank_tolerance=context.rank_tolerance,
        condition_ordinary=context.condition_ordinary,
        transition_state=(context.task_regime == ATLAS_TASK_TRANSITION_STATE),
        coordinate_atlas_contract=context.coordinate_atlas_contract,
        condition_pseudobond_support=(
            context.task_regime == ATLAS_TASK_MINIMUM
            and len(context.symmetry_operations) <= 1
        ),
        condition_transition_state_exact_chart=(
            context.condition_transition_state_exact_chart
            and len(context.symmetry_operations) <= 1
        ),
    )
    return (
        candidates,
        semantic_contract,
        (*reactive_zone_diagnostics, *semantic_diagnostics),
        fallback_events,
        selected,
        rank,
        reduction_diagnostics,
    )


def _evaluate_atlas_candidate_chart(
    context: _ChartExecutionContext,
    fragment_records: tuple[object, ...],
    interaction_centers: object | None,
    normalized_contacts: tuple[tuple[int, int, str], ...],
):
    pseudo_bonds = tuple((left, right) for left, right, _kind in normalized_contacts)
    pseudo_bond_kinds = tuple(kind for _left, _right, kind in normalized_contacts)
    primary_bonds = (
        tuple(sorted(set(context.topology_bonds + pseudo_bonds)))
        if context.task_regime == ATLAS_TASK_TRANSITION_STATE
        else context.topology_bonds
    )
    local_completion_available = bool(
        context.pseudobond_out_of_plane_support
        and pseudo_bonds
        and fragment_records
    )
    attempt = _candidate_chart_for_bonds(
        context,
        () if local_completion_available else fragment_records,
        interaction_centers,
        bonds=primary_bonds,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
    )
    if attempt[-2] == context.target_rank:
        return (primary_bonds, pseudo_bonds, pseudo_bond_kinds, *attempt)

    support_bonds = tuple(
        sorted(set(primary_bonds + context.reactive_support_edges))
    )
    best_bonds, best = primary_bonds, attempt
    if support_bonds != primary_bonds:
        graph_completed = _candidate_chart_for_bonds(
            context,
            () if local_completion_available else fragment_records,
            interaction_centers,
            bonds=support_bonds,
            pseudo_bonds=pseudo_bonds,
            pseudo_bond_kinds=pseudo_bond_kinds,
        )
        if graph_completed[-2] > best[-2]:
            edge_text = ",".join(
                f"{left}-{right}" for left, right in context.reactive_support_edges
            )
            completion_event = make_fallback_event(
                stage="SMITH_RANK_COMPLETION",
                algorithm_id="ATLAS_REACTIVE_SUPPORT_GRAPH",
                trigger="PRIMARY_TS_CHART_BELOW_EXACT_RANK_AFTER_REACTIVE_ZONE_EXCLUSIONS",
                domain="TS_REACTIVE_ZONE",
                macrofamily="REACTIVE_LOCAL_VALENCE",
                rank_before=best[-2],
                rank_after=graph_completed[-2],
            )
            best_bonds = support_bonds
            best = (
                graph_completed[0],
                graph_completed[1],
                (
                    f"ATLAS_REACTIVE_SUPPORT_COMPLETION EDGES={edge_text} "
                    f"RANK={attempt[-2]}/{context.target_rank}->"
                    f"{graph_completed[-2]}/{context.target_rank}",
                    *graph_completed[2],
                ),
                merge_fallback_events(graph_completed[3], (completion_event,)),
                *graph_completed[4:],
            )
    if best[-2] == context.target_rank or not local_completion_available:
        return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *best)

    local_completed = _candidate_chart_for_bonds(
        context,
        fragment_records,
        interaction_centers,
        bonds=best_bonds,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
    )
    if local_completed[-2] <= best[-2]:
        return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *best)
    completion_event = make_fallback_event(
        stage="SMITH_RANK_COMPLETION",
        algorithm_id="ATLAS_TS_REACTIVE_LOCAL_COMPLETION",
        trigger="SUPPORT_GRAPH_BELOW_EXACT_RANK_AFTER_REACTIVE_ZONE_EXCLUSIONS",
        domain="TS_REACTIVE_ZONE",
        macrofamily="TS_REACTIVE_COMPLETION",
        rank_before=best[-2],
        rank_after=local_completed[-2],
    )
    local_completed = (
        local_completed[0],
        local_completed[1],
        (
            "ATLAS_TS_REACTIVE_LOCAL_COMPLETION "
            f"RANK={best[-2]}/{context.target_rank}->"
            f"{local_completed[-2]}/{context.target_rank}",
            *best[2],
            *local_completed[2],
        ),
        merge_fallback_events(
            best[3],
            local_completed[3],
            (completion_event,),
        ),
        *local_completed[4:],
    )
    return (best_bonds, pseudo_bonds, pseudo_bond_kinds, *local_completed)


def _requires_interaction_center_coordinates(interaction_centers: object | None) -> bool:
    """Return whether declared atom--center interactions require special SONICs.

    Materialized bond and ring centers are reference data. Their mere
    presence must not replace the ordinary, full-rank molecular SONIC path;
    only an actual atom--center interaction consumes those centers.
    """

    if interaction_centers is None:
        return False
    interactions = tuple(getattr(interaction_centers, "interactions", ()) or ())
    return bool(interactions)


def _migrate_legacy_u_definition(definition: GICDefinition) -> GICDefinition:
    """Preserve frozen v1 U values while exposing only Gaussian-order primitives."""
    if definition.primitive_source_schema not in {
        LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA,
        LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA,
    }:
        return definition
    replacements: dict[str, tuple[str, str]] = {}
    primitives: list[GICPrimitive] = []
    for primitive in definition.primitives:
        if primitive.function != "U" or len(primitive.atoms) != 4:
            primitives.append(primitive)
            continue
        i, j, k, ell = primitive.atoms
        migrated = replace(primitive, atoms=(j, ell, k, i))
        replacements[primitive.identifier] = (
            primitive.gaussian_expression(),
            migrated.gaussian_expression(),
        )
        primitives.append(migrated)
    if not replacements:
        return definition
    gics: list[FrozenGIC] = []
    for gic in definition.gics:
        expression = gic.gaussian_expression
        for old, new in replacements.values():
            expression = expression.replace(old, new)
        gics.append(replace(gic, gaussian_expression=expression))
    return replace(definition, primitives=tuple(primitives), gics=tuple(gics))


def _consume_oracle_primitive_contract(
    path: Path,
    coordinates_angstrom: np.ndarray,
    *,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[str, str, str]:
    """Validate the required ORACLE primitive/B contract without reperception."""
    if not section_content(read_sectioned_lines(Path(path)), "PRIMITIVES"):
        raise GICForgeContractError(
            "SMITH requires the frozen ORACLE #PRIMITIVES contract"
        )
    try:
        contract = read_primitive_contract(Path(path))
        validate_primitive_contract(contract, coordinates_angstrom)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid ORACLE #PRIMITIVES contract: {exc}") from exc
    contract_bonds = {
        tuple(sorted(atom + 1 for atom in primitive.atoms))
        for primitive in contract.primitives
        if primitive.kind == "bond"
    }
    topology_bonds = {tuple(sorted(pair)) for pair in bonds}
    if contract_bonds != topology_bonds:
        raise GICForgeContractError(
            "ORACLE #PRIMITIVES bond rows do not match the frozen #TOPOLOGY graph"
        )
    return (
        "ORACLE_CONTRACT",
        contract.schema or MATRIX_XYZ_PRIMITIVES_SCHEMA,
        contract.b_matrix_sha256,
    )


def _construct_merlino_python_definition(
    atom_symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    *,
    topology_bonds: tuple[tuple[int, int], ...],
    point_group: str,
    improper_dihedrals: bool,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
    ring_puckering_diagnostics: tuple[str, ...] = (),
) -> GICDefinition:
    from matrix_smith.runtime.gicforge_python import build_gicforge_python_model

    model = build_gicforge_python_model(
        atom_symbols,
        coordinates_angstrom,
        topology_bonds=tuple((left - 1, right - 1) for left, right in topology_bonds),
        impdih=improper_dihedrals,
        onedih=True,
        svd_local=False,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
    )
    primitive_ids: dict[tuple[object, str], str] = {}
    primitive_indices: dict[str, int] = {}
    primitives: list[GICPrimitive] = []
    gics: list[FrozenGIC] = []
    xh_class_by_bond = _merlino_xh_class_by_bond(model.primitive_candidates, atom_symbols)

    def primitive_id(
        primitive: object,
        family: str,
        *,
        refs: tuple[str, ...] = (),
    ) -> str:
        key = (primitive, family)
        existing = primitive_ids.get(key)
        if existing is not None:
            if refs:
                primitive_index = primitive_indices[existing]
                current = primitives[primitive_index]
                merged_refs = current.refs + tuple(ref for ref in refs if ref not in current.refs)
                if merged_refs != current.refs:
                    primitives[primitive_index] = replace(current, refs=merged_refs)
            return existing
        identifier = f"P{len(primitives) + 1:03d}"
        primitive_ids[key] = identifier
        primitive_indices[identifier] = len(primitives)
        primitives.append(_merlino_runtime_primitive(identifier, primitive, family, refs=refs))
        return identifier

    for index, coordinate in enumerate(model.coordinates, start=1):
        family = _merlino_coordinate_family(coordinate.name, coordinate.block)
        family = _merlino_local_xh_family(
            coordinate,
            family,
            atom_symbols,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
            xh_class_by_bond=xh_class_by_bond,
        )
        refs = _merlino_coordinate_primitive_refs(coordinate, family)
        coefficients = tuple(
            (primitive_id(primitive, family, refs=refs), float(coefficient))
            for coefficient, primitive in coordinate.terms
            if abs(float(coefficient)) > 1.0e-14
        )
        coefficients = _normalized_coefficients(coefficients)
        if not coefficients:
            raise GICForgeContractError(f"empty Merlino Python coordinate {coordinate.name!r}")
        gics.append(
            FrozenGIC(
                identifier=f"GIC{index:03d}",
                name=coordinate.name,
                family=family,
                irrep="A" if point_group.upper() == "C1" else "UNASSIGNED",
                primitive_id=coefficients[0][0],
                gaussian_expression="MERLINO_ACTIVE",
                coefficients=coefficients,
            )
        )

    for candidate in model.primitive_candidates:
        family = _merlino_coordinate_family(candidate.name, candidate.block)
        family = _merlino_local_xh_family(
            candidate,
            family,
            atom_symbols,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
            xh_class_by_bond=xh_class_by_bond,
        )
        refs = _merlino_coordinate_primitive_refs(candidate, family)
        for _coefficient, primitive in candidate.terms:
            primitive_id(primitive, family, refs=refs)

    diagnostics = model.diagnostics
    ring_puckering_diagnostics = _effective_ring_puckering_diagnostics(
        ring_puckering_diagnostics,
        model.primitive_candidates,
    )
    return GICDefinition(
        backend="merlino-python-gicforge.v1",
        point_group=point_group,
        symmetrize=False,
        target_rank=model.target_rank,
        rank=len(model.coordinates),
        candidate_count=len(model.primitive_candidates),
        reference_coordinates_angstrom=model.coordinates_angstrom,
        primitives=tuple(primitives),
        gics=tuple(gics),
        reduction_diagnostics=GICReductionDiagnostics(
            rank_method="merlino_python_type_local_pruning",
            reduction_policy="MERLINO_FORTRAN_COMPATIBLE_BLOCKS",
            selected=tuple(gic.identifier for gic in gics),
            skipped_singular=(),
            skipped_dependent=tuple(
                f"{block}:{count}"
                for block, count in sorted(
                    dict(diagnostics.get("removed_counts_by_block", {})).items()
                )
                if int(count) > 0
            ),
            skipped_dependent_details=tuple(
                str(item) for item in diagnostics.get("local_equivalence", ())
            ),
        ),
        symmetry_diagnostics=_empty_symmetry_diagnostics(point_group, requested=False),
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        ring_puckering_diagnostics=ring_puckering_diagnostics,
        fallback_events=model.fallback_events,
    )


def _effective_ring_puckering_diagnostics(
    report_lines: tuple[str, ...],
    candidates: tuple[object, ...],
) -> tuple[str, ...]:
    """Make the public report describe the primitive model actually built."""

    aromatic_rings: set[int] = set()
    for coordinate in candidates:
        if str(getattr(coordinate, "block", "")) != "RPck":
            continue
        diagnostic = str(getattr(coordinate, "diagnostic", ""))
        if AROMATIC_LOCAL_MODEL_DIAGNOSTIC not in diagnostic:
            continue
        match = re.search(r"\bDOMAIN=RING:(\d+)\b", diagnostic)
        if match is not None:
            aromatic_rings.add(int(match.group(1)))

    if not aromatic_rings:
        return report_lines

    output: list[str] = []
    for line in report_lines:
        match = re.match(r"RING\s+(\d+)\b", line)
        if match is None or int(match.group(1)) not in aromatic_rings:
            output.append(line)
            continue
        prefix = line.split(" MODEL=", 1)[0]
        output.append(
            f"{prefix} {AROMATIC_LOCAL_MODEL_DIAGNOSTIC} PRIMITIVE=GAUSSIAN_U WEIGHTING=LOCAL_SALC"
        )
    return tuple(output)


def _merlino_runtime_primitive(
    identifier: str,
    primitive: object,
    family: str,
    *,
    refs: tuple[str, ...] = (),
) -> GICPrimitive:
    kind = str(getattr(primitive, "kind"))
    atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms"))
    ref_atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "ref", ()))
    mode = int(getattr(primitive, "mode", 0))
    if kind == "bond":
        return GICPrimitive(identifier, identifier, family, "R", atoms, refs=refs)
    if kind == "angle":
        return GICPrimitive(identifier, identifier, family, "A", atoms, refs=refs)
    if kind == "linear_bend":
        return GICPrimitive(
            identifier,
            identifier,
            family,
            "L",
            atoms,
            mode=mode,
            ref_atoms=ref_atoms,
            refs=refs,
        )
    if kind == "dihedral":
        return GICPrimitive(identifier, identifier, family, "D", atoms, refs=refs)
    if kind == "out_of_plane":
        return GICPrimitive(identifier, identifier, family, "U", atoms, refs=refs)
    if kind == "out_of_plane_height":
        return GICPrimitive(identifier, identifier, family, "H", atoms, refs=refs)
    raise GICForgeContractError(f"unsupported Merlino Python primitive kind: {kind}")


def _normalized_coefficients(
    coefficients: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    norm = float(np.sqrt(sum(float(coefficient) ** 2 for _identifier, coefficient in coefficients)))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        return coefficients
    return tuple(
        (identifier, float(coefficient) / norm) for identifier, coefficient in coefficients
    )


def _merlino_coordinate_primitive_refs(
    coordinate: object,
    family: str,
) -> tuple[str, ...]:
    if family != "RING_PUCKER_COMPONENT":
        return ()
    ring = _merlino_coordinate_ring_atoms(coordinate)
    if not ring:
        return ()
    return (_ring_ref_text(ring),)


def _merlino_local_xh_family(
    coordinate: object,
    family: str,
    atom_symbols: tuple[str, ...],
    *,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    xh_class_by_bond: dict[tuple[int, int], str],
) -> str:
    if family != "STRETCH":
        return family
    terms = tuple(getattr(coordinate, "terms", ()))
    if not terms:
        return family
    for _coefficient, primitive in terms:
        if str(getattr(primitive, "kind", "")) != "bond":
            return family
        atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms", ()))
        if len(atoms) != 2 or not _use_local_xh_stretch(
            atoms[0],
            atoms[1],
            atom_symbols,
            xh_stretch_policy,
            local_xh_bonds,
            local_xh_classes,
            xh_class_by_bond,
        ):
            return family
    return "LOCAL_XH_STRETCH"


def _use_local_xh_stretch(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    xh_class_by_bond: dict[tuple[int, int], str],
) -> bool:
    if not _is_xh_bond(left, right, atom_symbols):
        return False
    policy = _xh_stretch_policy(xh_stretch_policy)
    if policy == XH_STRETCH_POLICY_SYMMETRIZE:
        return False
    if policy == XH_STRETCH_POLICY_LOCAL_ALL:
        return True
    pair = _pair_key(left, right)
    if pair in set(local_xh_bonds):
        return True
    return xh_class_by_bond.get(pair, "") in set(local_xh_classes)


def _is_xh_bond(left: int, right: int, atom_symbols: tuple[str, ...]) -> bool:
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return False
    try:
        left_z = atomic_number(str(atom_symbols[left - 1]))
        right_z = atomic_number(str(atom_symbols[right - 1]))
    except KeyError:
        return False
    return (left_z == 1) ^ (right_z == 1)


def _merlino_xh_class_by_bond(
    candidates: tuple[object, ...],
    atom_symbols: tuple[str, ...],
) -> dict[tuple[int, int], str]:
    bonds: set[tuple[int, int]] = set()
    for candidate in candidates:
        for _coefficient, primitive in getattr(candidate, "terms", ()):
            if str(getattr(primitive, "kind", "")) != "bond":
                continue
            atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms", ()))
            if len(atoms) == 2:
                bonds.add(_pair_key(atoms[0], atoms[1]))
    return _xh_class_by_bond(tuple(sorted(bonds)), atom_symbols)


def _xh_class_by_bond(
    bonds: tuple[tuple[int, int], ...],
    atom_symbols: tuple[str, ...],
) -> dict[tuple[int, int], str]:
    hydrogens_by_heavy: dict[int, set[int]] = {}
    for left, right in bonds:
        if not _is_xh_bond(left, right, atom_symbols):
            continue
        heavy, hydrogen = _xh_heavy_and_hydrogen(left, right, atom_symbols)
        if heavy is None or hydrogen is None:
            continue
        hydrogens_by_heavy.setdefault(heavy, set()).add(hydrogen)
    result: dict[tuple[int, int], str] = {}
    for heavy, hydrogens in hydrogens_by_heavy.items():
        count = len(hydrogens)
        if count <= 1:
            xh_class = "XH"
        elif count == 2:
            xh_class = "XH2"
        else:
            xh_class = "XH3"
        for hydrogen in hydrogens:
            result[_pair_key(heavy, hydrogen)] = xh_class
    return result


def _xh_heavy_and_hydrogen(
    left: int,
    right: int,
    atom_symbols: tuple[str, ...],
) -> tuple[int | None, int | None]:
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return None, None
    try:
        left_z = atomic_number(str(atom_symbols[left - 1]))
        right_z = atomic_number(str(atom_symbols[right - 1]))
    except KeyError:
        return None, None
    if left_z == 1 and right_z != 1:
        return right, left
    if right_z == 1 and left_z != 1:
        return left, right
    return None, None


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))


def _xh_stretch_policy(value: str | None) -> str:
    if value is None:
        return XH_STRETCH_POLICY_SYMMETRIZE
    text = str(value).strip().replace("-", "_").upper()
    if not text:
        return XH_STRETCH_POLICY_SYMMETRIZE
    aliases = {
        "YES": XH_STRETCH_POLICY_SYMMETRIZE,
        "TRUE": XH_STRETCH_POLICY_SYMMETRIZE,
        "ALL": XH_STRETCH_POLICY_LOCAL_ALL,
        "LOCAL": XH_STRETCH_POLICY_LOCAL_ALL,
        "LOCAL_ALL": XH_STRETCH_POLICY_LOCAL_ALL,
        "SELECTED": XH_STRETCH_POLICY_LOCAL_SELECTED,
        "LOCAL_SELECTED": XH_STRETCH_POLICY_LOCAL_SELECTED,
        "NO": XH_STRETCH_POLICY_LOCAL_ALL,
        "FALSE": XH_STRETCH_POLICY_LOCAL_ALL,
    }
    normalized = aliases.get(text, text)
    if normalized not in XH_STRETCH_POLICIES:
        raise GICForgeContractError(
            "invalid X-H stretch policy: "
            f"{value!r}; expected SYMMETRIZE, LOCAL_ALL or LOCAL_SELECTED"
        )
    return normalized


def _normalize_pairs(raw_pairs: object | None) -> tuple[tuple[int, int], ...]:
    if raw_pairs is None:
        return ()
    pairs: list[tuple[int, int]] = []
    if isinstance(raw_pairs, str):
        items: object = re.split(r"[,;]\s*", raw_pairs.strip()) if raw_pairs.strip() else ()
    else:
        items = raw_pairs
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text or text.upper() in {"NA", "NONE"}:
                continue
            parts = re.split(r"[-:,/]", text)
            if len(parts) != 2:
                raise GICForgeContractError(f"invalid X-H bond selector: {item!r}")
            left, right = int(parts[0]), int(parts[1])
        else:
            values = tuple(item)
            if len(values) != 2:
                raise GICForgeContractError(f"invalid X-H bond selector: {item!r}")
            left, right = int(values[0]), int(values[1])
        pairs.append(_pair_key(left, right))
    return tuple(dict.fromkeys(pairs))


def _normalize_xh_classes(raw_classes: object | None) -> tuple[str, ...]:
    if raw_classes is None:
        return ()
    if isinstance(raw_classes, str):
        items: object = re.split(r"[,;]\s*", raw_classes.strip()) if raw_classes.strip() else ()
    else:
        items = raw_classes
    classes: list[str] = []
    aliases = {"XH1": "XH", "H1": "XH", "H2": "XH2", "H3": "XH3"}
    for item in items:
        text = str(item).strip().upper().replace("-", "")
        if not text or text in {"NA", "NONE"}:
            continue
        normalized = aliases.get(text, text)
        if normalized not in XH_STRETCH_CLASSES:
            raise GICForgeContractError(
                f"invalid X-H stretch class: {item!r}; expected XH, XH2 or XH3"
            )
        classes.append(normalized)
    return tuple(dict.fromkeys(classes))


def _merlino_coordinate_ring_atoms(coordinate: object) -> tuple[int, ...]:
    atoms: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for _coefficient, primitive in getattr(coordinate, "terms", ()):
        if str(getattr(primitive, "kind", "")) != "dihedral":
            return ()
        term_atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms"))
        if len(term_atoms) != 4:
            return ()
        atoms.update(term_atoms)
        for left, right in zip(term_atoms, term_atoms[1:]):
            edges.add(tuple(sorted((left, right))))
    if len(atoms) < 4:
        return ()
    return _ordered_cycle_from_edges(tuple(sorted(atoms)), edges)


def _ordered_cycle_from_edges(
    atoms: tuple[int, ...],
    edges: set[tuple[int, int]],
) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {atom: [] for atom in atoms}
    for left, right in edges:
        if left not in adjacency or right not in adjacency:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return tuple(sorted(atoms))
    start = min(atoms)
    candidates: list[tuple[int, ...]] = []
    for second in sorted(adjacency[start]):
        path = [start, second]
        previous = start
        current = second
        while len(path) < len(atoms):
            next_atoms = [atom for atom in adjacency[current] if atom != previous]
            if not next_atoms:
                break
            next_atom = next_atoms[0]
            if next_atom in path:
                break
            path.append(next_atom)
            previous, current = current, next_atom
        if len(path) == len(atoms) and start in adjacency[path[-1]]:
            candidates.append(tuple(path))
    if not candidates:
        return tuple(sorted(atoms))
    return min(candidates)


def _merlino_coordinate_family(name: str, block: str) -> str:
    prefix = str(block or name[:4])
    if prefix == "Stre":
        return "STRETCH"
    if prefix in {
        "Rock",
        "Bend",
        "SymD",
        "Scis",
        "SciL",
        "Wagg",
        "Twst",
        "AsyD",
        "EEee",
        "T2xx",
        "T2yy",
        "T2zz",
        "B1GE",
        "EUU",
        "HCAn",
        "XAng",
    }:
        return "BEND"
    if prefix == "Spir":
        return "SPIRO_BEND"
    if prefix == "RDef":
        return "CYCLIC_BEND"
    if prefix == "LAng":
        return "LINEAR_BEND"
    if prefix == "BtFl":
        return "BUTTERFLY"
    if prefix == "RPck":
        return "RING_PUCKER_COMPONENT"
    if prefix in {"Tors", "Dihe"}:
        return "TORSION"
    if prefix == "OuPl":
        return "OUT_OF_PLANE"
    if prefix == "ImpD":
        return "IMPROPER_DIHEDRAL"
    return "TORSION"


def symmetrize_gic_definition(
    definition: GICDefinition,
    *,
    atom_symbols: tuple[str, ...],
    symmetry_operations: tuple[GICPointGroupOperation, ...] = (),
) -> GICDefinition:
    """Apply the frozen GIC symmetrization utility to a reduced definition."""
    # A reactive TS pseudobond is a chart edge prescribed by ORACLE, not an
    # intermolecular body decomposition.  Its selected local analytic SALCs
    # must therefore remain local.  The residual B-metric torsion eigenbasis
    # is retained only as the established safeguard for genuine weak-complex
    # minimum charts, where it was introduced.
    is_transition_state_chart = any(
        gic.family == "TS_REACTION_DISTANCE" for gic in definition.gics
    )
    is_intermolecular = (
        definition.fragment_mode != FRAGMENT_MODE_NONE
        and not is_transition_state_chart
    )
    has_cartesian_intercomponent_coordinates = any(
        primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        for primitive in definition.primitives
    )
    closed_primitives = _symmetry_closed_projector_primitives(
        definition.primitives,
        symmetry_operations=tuple(symmetry_operations),
        include_cartesian_interfragment_orbits=(
            (is_intermolecular or has_cartesian_intercomponent_coordinates)
            and definition.point_group.strip().upper() not in {"C1", "UNKNOWN"}
        ),
    )
    gics, symmetry_diagnostics = _apply_local_symmetrization(
        definition.gics,
        closed_primitives,
        atom_symbols=tuple(atom_symbols),
        point_group=definition.point_group,
        requested=True,
        symmetry_operations=tuple(symmetry_operations),
        reference_coordinates_angstrom=definition.reference_coordinates_angstrom,
        intermolecular=(is_intermolecular or has_cartesian_intercomponent_coordinates),
        pseudobond_mode=(
            definition.fragment_mode == FRAGMENT_MODE_PSEUDO_BONDS
            and not is_transition_state_chart
        ),
    )
    return GICDefinition(
        backend=definition.backend,
        point_group=definition.point_group,
        symmetrize=True,
        target_rank=definition.target_rank,
        rank=definition.rank,
        candidate_count=definition.candidate_count,
        reference_coordinates_angstrom=definition.reference_coordinates_angstrom,
        primitives=closed_primitives,
        gics=gics,
        reduction_diagnostics=definition.reduction_diagnostics,
        symmetry_diagnostics=symmetry_diagnostics,
        fragment_mode=definition.fragment_mode,
        pseudo_bonds=definition.pseudo_bonds,
        pseudo_bond_kinds=definition.pseudo_bond_kinds,
        xh_stretch_policy=definition.xh_stretch_policy,
        local_xh_bonds=definition.local_xh_bonds,
        local_xh_classes=definition.local_xh_classes,
        ring_puckering_diagnostics=definition.ring_puckering_diagnostics,
        contract_schema_version=definition.contract_schema_version,
        semantic_grammar_version=definition.semantic_grammar_version,
        semantic_diagnostics=definition.semantic_diagnostics,
        fallback_events=definition.fallback_events,
        primitive_source=definition.primitive_source,
        primitive_source_schema=definition.primitive_source_schema,
        primitive_b_matrix_sha256=definition.primitive_b_matrix_sha256,
    )


def build_sycart_definition_from_xyzin(path: Path) -> SYCartDefinition:
    """Build the shared SMITH-owned symmetry-adapted Cartesian basis."""
    target = Path(path)
    validate_gicforge_prerequisites(target)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    symmetry = read_molecular_symmetry(target)
    basis = symmetry_adapted_cartesian_basis(
        geometry.atoms,
        coords,
        symmetry=symmetry,
        # MolecularSymmetry.orientation stores principal axes by columns;
        # OnicSiteFrame stores local axes by rows.
        frame_axes_global=np.asarray(symmetry.orientation, dtype=float).T,
    )
    return SYCartDefinition(
        backend=SYCART_BACKEND,
        point_group=basis.point_group,
        target_rank=basis.target_rank,
        vectors=tuple(tuple(float(value) for value in row) for row in basis.cartesian_from_q.T),
        irreps=basis.irreps,
        external_mode_count=basis.external_mode_count,
        linearity=basis.linearity,
        gauge_policy=CARTESIAN_BLOCK_GAUGE,
    )


def gic_definition_section_lines(definition: GICDefinition) -> list[str]:
    if not definition.primitive_source or definition.primitive_source == "LEGACY_RECONSTRUCTED":
        raise GICForgeContractError("GIC serialization requires an explicit primitive source")
    if definition.primitive_source == "ORACLE_CONTRACT" and (
        not definition.primitive_source_schema
        or not definition.primitive_b_matrix_sha256
    ):
        raise GICForgeContractError("ORACLE primitive source metadata is incomplete")
    primitive_dependency = (
        definition.primitive_source_schema
        if definition.primitive_source == "ORACLE_CONTRACT"
        else definition.primitive_source
    )
    fallback_events = build_fallback_ledger(definition)
    lines = [
        f"SCHEMA {definition.contract_schema_version}",
        f"CONTRACT_SCHEMA_VERSION {definition.contract_schema_version}",
        "STATUS BUILT",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA} "
        f"SYMMETRY=oracle.xyz.symmetry.v1 PRIMITIVES={primitive_dependency}",
        "OWNERSHIP ORACLE_PERCEPTION=ORACLE SMITH_CONSTRUCTION=SMITH "
        "GAUSSIAN_SERIALIZATION=GAUSSIAN_READALLGIC",
        "CONSTRUCTION_GEOMETRY_REFERENCE FROZEN_ORACLE_INPUT",
        f"PRIMITIVE_SOURCE {definition.primitive_source}",
        f"PRIMITIVE_SOURCE_SCHEMA {definition.primitive_source_schema or 'NONE'}",
        f"PRIMITIVE_B_MATRIX_SHA256 {definition.primitive_b_matrix_sha256 or 'NONE'}",
        f"FROZEN_SONIC_IDENTITY_SHA256 {sonic_definition_identity_sha256(definition)}",
        f"SEMANTIC_GRAMMAR {definition.semantic_grammar_version or 'NONE'}",
        f"FALLBACK_COUNT {len(fallback_events)}",
        "INDEXING ATOMS=ONE_BASED",
        f"BACKEND {definition.backend}",
        f"POINT_GROUP {definition.point_group}",
        f"SYMMETRY_GROUP {definition.point_group}",
        f"TOTAL_SYMMETRIC_IRREP {total_symmetric_irrep(definition.point_group)}",
        f"TOTAL_SYMMETRIC_GIC_COUNT {len(total_symmetric_gics(definition))}",
        f"TOTAL_SYMMETRIC_GICS {_csv_or_none(total_symmetric_gic_names(definition))}",
        f"SYMMETRIZE {_bool_text(definition.symmetrize)}",
        f"SYMMETRY_MODE {_symmetry_mode(definition)}",
        f"OUT_OF_PLANE_MODE {_out_of_plane_mode(definition.primitives)}",
        f"FRAGMENT_MODE {definition.fragment_mode}",
        f"XH_STRETCH_POLICY {definition.xh_stretch_policy}",
        f"LOCAL_XH_BONDS {_pairs_text(definition.local_xh_bonds)}",
        f"LOCAL_XH_CLASSES {_csv_or_none_from_strings(definition.local_xh_classes)}",
        f"PSEUDO_BOND_COUNT {len(definition.pseudo_bonds)}",
        f"TARGET_RANK {definition.target_rank}",
        f"RANK {definition.rank}",
        f"CANDIDATE_COUNT {definition.candidate_count}",
        f"PRIMITIVE_COUNT {len(definition.primitives)}",
        f"GIC_COUNT {len(definition.gics)}",
        f"WILSON_TANGENT_RANK {definition.wilson_tangent_rank}",
        f"WILSON_TANGENT_SINGULAR_MIN {definition.wilson_tangent_singular_min:.12g}",
        f"WILSON_TANGENT_SINGULAR_MAX {definition.wilson_tangent_singular_max:.12g}",
        f"PERIODIC_COORDINATE_ESTIMATE_COUNT {len(definition.periodic_coordinate_estimates)}",
        f"PROTECTED_GIC_COUNT {_protected_gic_count(definition.primitives)}",
        f"SKIPPED_SINGULAR_COUNT {_skipped_singular_count(definition)}",
        f"SKIPPED_DEPENDENT_COUNT {_skipped_dependent_count(definition)}",
        f"RANK_METHOD {RANK_METHOD}",
        f"REDUCTION_POLICY {REDUCTION_POLICY}",
        "B_MATRIX_DERIVATIVE_MODE ANALYTIC",
        f"RANK_TOLERANCE {RANK_TOLERANCE:.12g}",
        "[PRIMITIVES]",
    ]
    if definition.primitives:
        lines.extend(_primitive_line(primitive) for primitive in definition.primitives)
    else:
        lines.append("NONE")
    lines.append("[FROZEN_GICS]")
    if definition.gics:
        lines.extend(_frozen_gic_line(gic) for gic in definition.gics)
    else:
        lines.append("NONE")
    lines.append("[REDUCTION_DIAGNOSTICS]")
    lines.extend(_reduction_diagnostics_lines(definition))
    lines.append("[SYMMETRY_DIAGNOSTICS]")
    lines.extend(_symmetry_diagnostics_lines(definition))
    lines.append("[RING_PUCKERING_DIAGNOSTICS]")
    lines.extend(definition.ring_puckering_diagnostics or ("NONE",))
    lines.append("[SEMANTIC_DIAGNOSTICS]")
    lines.extend(definition.semantic_diagnostics or ("NONE",))
    lines.extend(fallback_ledger_section_lines(fallback_events))
    lines.append("[PSEUDO_BONDS]")
    if definition.pseudo_bonds:
        kinds = definition.pseudo_bond_kinds or (
            ("INTERFRAGMENT_CLOSEST",) * len(definition.pseudo_bonds)
        )
        if len(kinds) != len(definition.pseudo_bonds):
            kinds = ("INTERFRAGMENT_CLOSEST",) * len(definition.pseudo_bonds)
        for index, ((left, right), kind) in enumerate(
            zip(definition.pseudo_bonds, kinds),
            start=1,
        ):
            lines.append(f"{index} {left} {right} KIND={kind}")
    else:
        lines.append("NONE")
    lines.append("[GAUSSIAN_GIC]")
    gaussian_gics = _gaussian_gic_block_lines(definition)
    if gaussian_gics:
        lines.extend(gaussian_gics)
    else:
        lines.append("NONE")
    lines.append("[PERIODIC_COORDINATE_ESTIMATES]")
    if definition.periodic_coordinate_estimates:
        lines.extend(
            periodic_coordinate_estimate_line(record)
            for record in definition.periodic_coordinate_estimates
        )
    else:
        lines.append("NONE")
    return lines


def sycart_definition_section_lines(definition: SYCartDefinition) -> list[str]:
    lines = [
        f"SCHEMA {ORACLE_XYZ_SYCART_SCHEMA}",
        "STATUS BUILT",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} GIC=oracle.xyz.gic.v1 "
        "SYMMETRY=oracle.xyz.symmetry.v1",
        "INDEXING ATOMS=ONE_BASED",
        f"BACKEND {definition.backend}",
        f"POINT_GROUP {definition.point_group}",
        "SYMMETRY_MODE COMPLETE_ISOTYPIC_PROJECTORS",
        f"EXTERNAL_MODE_COUNT {definition.external_mode_count}",
        f"LINEARITY {definition.linearity or 'UNDECLARED'}",
        f"COMPONENT_GAUGE {definition.gauge_policy or 'UNDECLARED'}",
        f"TARGET_RANK {definition.target_rank}",
        f"COORD_COUNT {len(definition.vectors)}",
        "[SYCART]",
    ]
    if definition.vectors:
        irreps = definition.irreps or ("A",) * len(definition.vectors)
        lines.extend(
            f"SYC{idx:03d} IRREP={irrep} " + _sycart_components(vector)
            for idx, (vector, irrep) in enumerate(zip(definition.vectors, irreps), start=1)
        )
    else:
        lines.append("NONE")
    return lines


def write_gicforge_build_sections(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
) -> GICDefinition:
    target = Path(path)
    # Disconnected molecular components are an intermolecular system even when
    # the caller did not explicitly materialize #FRAGMENTS.  The SONIC API
    # must not silently fall back to one local coordinate set per component:
    # build the shared fragment contract automatically, while leaving
    # covalently connected molecules unchanged.
    if fragment_mode is None or _fragment_mode(fragment_mode) != FRAGMENT_MODE_NONE:
        build_lines = read_sectioned_lines(target)
        if not section_content(build_lines, "FRAGMENTS"):
            from matrix_fragments import (
                build_fragment_definition_from_xyzin,
                write_fragment_build_section,
            )

            fragment_definition = build_fragment_definition_from_xyzin(target)
            if len(fragment_definition.fragments) > 1:
                write_fragment_build_section(target)
    if not section_content(
        read_sectioned_lines(target),
        "ORACLE_COORDINATE_ATLAS",
    ):
        if _fragment_context(fragment_context) == _FRAGMENT_CONTEXT_TRANSITION_STATE:
            raise GICForgeContractError(
                "a transition-state build requires a valid ORACLE TS coordinate atlas"
            )
        from matrix_oracle import write_oracle_sonic_contract_from_xyzin

        write_oracle_sonic_contract_from_xyzin(target)
    definition = build_gic_definition_from_xyzin(
        target,
        symmetrize=symmetrize,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        fragment_context=fragment_context,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        ring_puckering_model=ring_puckering_model,
    )
    replace_section(target, "GIC", gic_definition_section_lines(definition))
    if sycart:
        sycart_definition = build_sycart_definition_from_xyzin(target)
        replace_section(target, "SYCART", sycart_definition_section_lines(sycart_definition))
    return definition


def freeze_sonic_definition_from_xyzin(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Read the existing frozen SONIC identity or build it exactly once."""

    target = Path(path)
    if section_content(read_sectioned_lines(target), "GIC"):
        return read_gic_definition_from_xyzin(target)
    return write_gicforge_build_sections(target, **kwargs)


def sonic_definition_identity_sha256(definition: GICDefinition) -> str:
    """Return the geometry-independent identity of a frozen SONIC chart."""

    payload = {
        "primitives": [
            {
                "identifier": item.identifier,
                "family": item.family,
                "function": item.function,
                "atoms": list(item.atoms),
                "refs": list(item.refs),
            }
            for item in definition.primitives
        ],
        "gics": [
            {
                "identifier": item.identifier,
                "family": item.family,
                "primitive_id": item.primitive_id,
                "coefficients": [
                    [identifier, f"{float(coefficient):.12g}"]
                    for identifier, coefficient in item.coefficients
                ],
            }
            for item in definition.gics
        ],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_pes_exploration_gic_definition_from_xyzin(
    path: Path,
    *,
    retained_group: str = "C1",
    fragment_mode: str | None = None,
) -> GICDefinition:
    """Build the transient SONIC contract shared by scans and PES exploration.

    The retained group is the symmetry common to the complete path, not the
    point group perceived at the reference geometry.  C1 is therefore the
    conservative default.  Local SALCs remain available for stretches, bends
    and rings, while every exocyclic torsion is forced to the one-dihedral form
    so that a scan variable is never hidden inside a torsional combination.
    """

    target = Path(path)
    retained = str(retained_group).strip().upper() or "C1"
    base = read_gic_definition_from_xyzin(target)
    diagnostics = base.reduction_diagnostics
    diagnostic_rows = ()
    if diagnostics is not None:
        diagnostic_rows = (
            diagnostics.selected_by_family
            + diagnostics.skipped_singular_details
            + diagnostics.skipped_dependent_details
        )
    uses_local_salc = any("LOCAL_SALC" in row.upper() for row in diagnostic_rows)
    return build_gic_definition_from_xyzin(
        target,
        symmetrize=retained not in {"C1", "UNKNOWN"},
        symmetry_group=retained,
        improper_dihedrals=False,
        fragment_mode=fragment_mode,
        fragment_context=_FRAGMENT_CONTEXT_EXPLORATION,
        xh_stretch_policy=base.xh_stretch_policy,
        local_xh_bonds=base.local_xh_bonds,
        local_xh_classes=base.local_xh_classes,
        local_salc=uses_local_salc,
        separate_exocyclic_torsions=True,
    )


def build_sonic_definition_from_xyzin(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Public SONIC alias for the legacy GICForge builder."""
    return build_gic_definition_from_xyzin(path, **kwargs)


def write_sonic_build_sections(
    path: Path,
    **kwargs: object,
) -> GICDefinition:
    """Public SONIC alias for writing the frozen GIC and optional SYCART sections."""
    return write_gicforge_build_sections(path, **kwargs)


def write_sonic_build_sections_from_cartesian(
    source: Path,
    target: Path | None = None,
    *,
    source_kind: str = "auto",
    symmetrize: bool = False,
    sycart: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    fragment_context: str = _FRAGMENT_CONTEXT_OPTIMIZATION,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
) -> GICDefinition:
    """Build a standalone SONIC contract from Cartesian input.

    If the input already contains MATRIX topology, synthon, symmetry and
    validation sections, they are consumed as the frozen molecular state.  A
    plain Cartesian file is first enriched with the bundled MATRIX perception
    tools and then passed to the same coordinate builder.
    """
    source_path = Path(source)
    build_path = Path(target) if target is not None else source_path
    lines = read_sectioned_lines(source_path)
    required = ("VALIDATION", "TOPOLOGY", "SYNTHONS", "SYMMETRY")
    if not all(section_content(lines, name) for name in required):
        if target is None:
            build_path = source_path.with_suffix(source_path.suffix + ".xyzin")
        preprocess_to_enriched_xyz(source_path, build_path, source_kind=source_kind)  # type: ignore[arg-type]
    elif build_path != source_path:
        build_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    build_lines = read_sectioned_lines(build_path)
    if not section_content(build_lines, "FRAGMENTS"):
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(build_path)
    write_validation_section(build_path)
    return write_gicforge_build_sections(
        build_path,
        symmetrize=symmetrize,
        sycart=sycart,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        fragment_context=fragment_context,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
    )


build_gsnic_definition_from_xyzin = build_sonic_definition_from_xyzin
write_gsnic_build_sections = write_sonic_build_sections
write_gsnic_build_sections_from_cartesian = write_sonic_build_sections_from_cartesian


from .evaluation import (
    build_gic_b_matrix,
)


def recenter_sonic_definition_from_xyzin(
    path: Path,
    definition: GICDefinition,
) -> GICDefinition:
    """Recenter one frozen SONIC identity on a new ORACLE reference geometry.

    The primitive/GIC identities, families, coefficients and endocyclic/
    exocyclic classification are preserved verbatim.  Only the reference
    Cartesian geometry, geometry-dependent primitive-B provenance and periodic
    reference values are refreshed after ORACLE has rewritten its sections.
    """

    target = Path(path)
    geometry = read_enriched_xyz(target)
    contract = read_primitive_contract(target)
    if len(definition.reference_coordinates_angstrom) != geometry.natoms:
        raise GICForgeContractError(
            "cannot recenter a frozen SONIC definition on a different atom count"
        )
    recentered = replace(
        definition,
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in geometry.coordinates_angstrom
        ),
        primitive_source="ORACLE_CONTRACT",
        primitive_source_schema=contract.schema,
        primitive_b_matrix_sha256=contract.b_matrix_sha256,
    )
    periodic_values = _selected_gic_values(
        recentered,
        families={"TORSION", "PSEUDO_CYCLE_TORSION", "IMPROPER_DIHEDRAL"},
    )
    recentered = replace(
        recentered,
        periodic_coordinate_estimates=tuple(
            replace(
                estimate,
                reference_value_radian=periodic_values.get(
                    estimate.coordinate_identifier,
                    estimate.reference_value_radian,
                ),
            )
            for estimate in recentered.periodic_coordinate_estimates
        ),
    )
    if sonic_definition_identity_sha256(recentered) != sonic_definition_identity_sha256(definition):
        raise GICForgeContractError("recentring changed the frozen SONIC coordinate identity")
    replace_section(target, "GIC", gic_definition_section_lines(recentered))
    return recentered


def build_gic_b_matrix_from_xyzin(path: Path) -> GICBMatrix:
    """Evaluate the B matrix from the frozen #GIC section of an enriched XYZ."""
    target = Path(path)
    definition = read_gic_definition_from_xyzin(target)
    geometry = read_enriched_xyz(target)
    return build_gic_b_matrix(
        definition,
        coordinates_angstrom=geometry.coordinates_angstrom,
    )


def total_symmetric_gics(definition: GICDefinition) -> tuple[FrozenGIC, ...]:
    """Return frozen GICs active in symmetry-preserving optimization/fitting."""
    return tuple(
        gic
        for gic in definition.gics
        if is_total_symmetric_irrep(definition.point_group, gic.irrep)
    )


def total_symmetric_gic_names(definition: GICDefinition) -> tuple[str, ...]:
    return tuple(gic.name for gic in total_symmetric_gics(definition))


def read_gic_definition_from_xyzin(path: Path) -> GICDefinition:
    """Read a frozen ORACLE GIC definition without regenerating coordinates."""
    target = Path(path)
    lines = read_sectioned_lines(target)
    geometry = read_enriched_xyz(target)
    section = section_content(lines, "GIC")
    if not section:
        raise GICForgeContractError("missing #GIC section")
    if section[0].strip() != f"SCHEMA {ORACLE_XYZ_GIC_SCHEMA}":
        raise GICForgeContractError("invalid #GIC schema")
    status = _section_value(section, "STATUS")
    if (status or "").upper() != "BUILT":
        raise GICForgeContractError(f"#GIC status must be BUILT; found {status or 'UNKNOWN'}")
    primitives = tuple(
        _parse_primitive_line(line)
        for line in _subsection(section, "PRIMITIVES")
        if line.strip() and line.strip().upper() != "NONE"
    )
    gics = tuple(
        _parse_frozen_gic_line(line)
        for line in _subsection(section, "FROZEN_GICS")
        if line.strip() and line.strip().upper() != "NONE"
    )
    diagnostics = _parse_reduction_diagnostics(
        section,
        selected=tuple(p.identifier for p in primitives),
    )
    symmetry_diagnostics = _parse_symmetry_diagnostics(section)
    primitive_source = (_section_value(section, "PRIMITIVE_SOURCE") or "").strip()
    if not primitive_source or primitive_source == "LEGACY_RECONSTRUCTED":
        raise GICForgeContractError("#GIC requires an explicit primitive source")
    primitive_source_schema = (
        _section_value(section, "PRIMITIVE_SOURCE_SCHEMA") or ""
    ).strip()
    primitive_b_matrix_sha256 = (
        _section_value(section, "PRIMITIVE_B_MATRIX_SHA256") or ""
    ).strip()
    if primitive_source == "ORACLE_CONTRACT" and (
        not primitive_source_schema
        or primitive_source_schema == "NONE"
        or not primitive_b_matrix_sha256
        or primitive_b_matrix_sha256 == "NONE"
    ):
        raise GICForgeContractError("#GIC ORACLE primitive metadata is incomplete")
    try:
        fallback_diagnostics, fallback_events = fallback_provenance_from_lines(
            _subsection(section, "FALLBACK_DIAGNOSTICS"),
            _subsection(section, "FALLBACK_LEDGER"),
        )
    except ValueError as exc:
        raise GICForgeContractError("malformed #GIC fallback ledger") from exc
    definition = GICDefinition(
        backend=_section_value(section, "BACKEND") or GIC_BACKEND,
        point_group=_section_value(section, "POINT_GROUP") or _point_group(lines),
        symmetrize=_parse_bool(_section_value(section, "SYMMETRIZE")),
        target_rank=_parse_int(_section_value(section, "TARGET_RANK")),
        rank=_parse_int(_section_value(section, "RANK")),
        candidate_count=_parse_int(_section_value(section, "CANDIDATE_COUNT")),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in geometry.coordinates_angstrom
        ),
        primitives=primitives,
        gics=gics,
        reduction_diagnostics=diagnostics,
        symmetry_diagnostics=symmetry_diagnostics,
        fragment_mode=_fragment_mode(_section_value(section, "FRAGMENT_MODE")),
        pseudo_bonds=_parse_pseudo_bonds(section),
        pseudo_bond_kinds=_parse_pseudo_bond_kinds(section),
        xh_stretch_policy=_xh_stretch_policy(_section_value(section, "XH_STRETCH_POLICY")),
        local_xh_bonds=_normalize_pairs(_section_value(section, "LOCAL_XH_BONDS")),
        local_xh_classes=_normalize_xh_classes(_section_value(section, "LOCAL_XH_CLASSES")),
        ring_puckering_diagnostics=tuple(
            line
            for line in _subsection(section, "RING_PUCKERING_DIAGNOSTICS")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        periodic_coordinate_estimates=tuple(
            parse_periodic_coordinate_estimate(line)
            for line in _subsection(section, "PERIODIC_COORDINATE_ESTIMATES")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        contract_schema_version=_section_value(section, "CONTRACT_SCHEMA_VERSION")
        or ORACLE_XYZ_GIC_SCHEMA,
        semantic_grammar_version=(
            ""
            if (_section_value(section, "SEMANTIC_GRAMMAR") or "NONE").upper() == "NONE"
            else (_section_value(section, "SEMANTIC_GRAMMAR") or SEMANTIC_GRAMMAR_VERSION)
        ),
        semantic_diagnostics=tuple(
            line
            for line in _subsection(section, "SEMANTIC_DIAGNOSTICS")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        fallback_diagnostics=fallback_diagnostics,
        fallback_events=fallback_events,
        primitive_source=primitive_source,
        primitive_source_schema=(
            ""
            if primitive_source_schema == "NONE"
            else primitive_source_schema
        ),
        primitive_b_matrix_sha256=(
            ""
            if primitive_b_matrix_sha256 == "NONE"
            else primitive_b_matrix_sha256
        ),
        wilson_tangent_rank=_parse_int(_section_value(section, "WILSON_TANGENT_RANK")),
        wilson_tangent_singular_min=float(
            _section_value(section, "WILSON_TANGENT_SINGULAR_MIN") or 0.0
        ),
        wilson_tangent_singular_max=float(
            _section_value(section, "WILSON_TANGENT_SINGULAR_MAX") or 0.0
        ),
    )
    declared_fallback_count = _parse_int(_section_value(section, "FALLBACK_COUNT"))
    if declared_fallback_count != len(definition.fallback_events):
        raise GICForgeContractError(
            "#GIC fallback count does not match its structured ledger"
        )
    declared_identity = _section_value(section, "FROZEN_SONIC_IDENTITY_SHA256")
    actual_identity = sonic_definition_identity_sha256(definition)
    if declared_identity and declared_identity != actual_identity:
        raise GICForgeContractError(
            "#GIC frozen SONIC identity checksum does not match its coordinates"
        )
    validate_indivisible_gic_components(definition)
    return definition


def gic_b_matrix_lines(matrix: GICBMatrix) -> list[str]:
    """Serialize a GIC B matrix in a compact machine-readable text format."""
    lines = [
        "SCHEMA oracle.gic.bmatrix.v1",
        f"BACKEND {matrix.backend}",
        "UNITS MIXED_GIC_PER_ANGSTROM",
        "DERIVATIVE_MODE ANALYTIC",
        f"ROW_COUNT {len(matrix.rows)}",
        f"COLUMN_COUNT {len(matrix.cartesian_columns)}",
        "[COLUMNS]",
        " ".join(matrix.cartesian_columns) if matrix.cartesian_columns else "NONE",
        "[ROWS]",
    ]
    if not matrix.rows:
        lines.append("NONE")
        return lines
    for label, name, irrep, row in zip(
        matrix.coordinate_labels,
        matrix.coordinate_names,
        matrix.irreps,
        matrix.rows,
    ):
        values = ",".join(f"{value:.12g}" for value in row)
        lines.append(f"{label} NAME={name} IRREP={irrep} VALUES={values}")
    return lines


def write_gic_b_matrix(path: Path, output: Path) -> GICBMatrix:
    matrix = build_gic_b_matrix_from_xyzin(path)
    target = Path(output)
    target.write_text("\n".join(gic_b_matrix_lines(matrix)) + "\n", encoding="utf-8")
    return matrix


from .gaussian_export import (
    _gaussian_gic_block_lines,
    _ring_ref_text,
)

from .primitive_generation import (
    _apply_semantic_contract_to_candidates,
    _freeze_balanced_out_of_plane_coordinates,
    _fragment_records,
    _fragment_index_by_atom,
    _interaction_center_definition,
    _normalize_pseudo_contact,
    _primitive_candidates,
    _ring_puckering_diagnostics,
    _vibrational_rank,
)


def _section_value(section_lines: list[str], key: str) -> str | None:
    key_upper = key.upper()
    for line in section_lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].upper() == key_upper:
            return parts[1].strip()
    return None


def _parse_primitive_line(line: str) -> GICPrimitive:
    parts = line.split()
    if not parts:
        raise GICForgeContractError("empty primitive line")
    fields = _key_values(parts[1:])
    try:
        return GICPrimitive(
            identifier=parts[0],
            name=fields["NAME"],
            family=fields["FAMILY"],
            function=fields["FUNCTION"],
            atoms=_parse_atom_list(fields["ATOMS"]),
            mode=int(fields.get("MODE", "0")),
            ref_atoms=_parse_atom_list(fields.get("REF_ATOMS", "")),
            refs=_parse_text_list(fields.get("REFS", "")),
            frame_atoms=_parse_atom_list(fields.get("FRAME_ATOMS", "")),
            ref_frame_atoms=_parse_atom_list(fields.get("REF_FRAME_ATOMS", "")),
            provenance=fields.get("PROVENANCE", AUTO_PROVENANCE).upper(),
            semantic_id=fields.get("SEMANTIC_ID", ""),
            semantic_type=fields.get("SEMANTIC_TYPE", "").upper(),
            chart=fields.get("CHART", "PRINCIPAL").upper(),
            chart_reference_radian=(
                float(fields["CHART_REFERENCE_RADIAN"])
                if "CHART_REFERENCE_RADIAN" in fields
                else None
            ),
        )
    except KeyError as exc:
        raise GICForgeContractError(f"invalid primitive line: {line}") from exc
    except ValueError as exc:
        raise GICForgeContractError(f"invalid primitive numeric field: {line}") from exc


def _parse_frozen_gic_line(line: str) -> FrozenGIC:
    parts = line.split()
    if not parts:
        raise GICForgeContractError("empty frozen GIC line")
    fields = _key_values(parts[1:])
    coefficients = _parse_coefficients(fields.get("COEFFS", ""))
    if not coefficients:
        primitive_id = fields.get("PRIMITIVE")
        if not primitive_id:
            raise GICForgeContractError(f"invalid frozen GIC coefficients: {line}")
        coefficients = ((primitive_id, 1.0),)
    try:
        return FrozenGIC(
            identifier=parts[0],
            name=fields["NAME"],
            family=fields["FAMILY"],
            irrep=fields["IRREP"],
            primitive_id=coefficients[0][0],
            gaussian_expression=fields.get("GAUSSIAN", "NONE"),
            coefficients=coefficients,
        )
    except KeyError as exc:
        raise GICForgeContractError(f"invalid frozen GIC line: {line}") from exc


def _parse_reduction_diagnostics(
    section_lines: list[str],
    *,
    selected: tuple[str, ...],
) -> GICReductionDiagnostics:
    lines = _subsection(section_lines, "REDUCTION_DIAGNOSTICS")
    if not lines:
        return GICReductionDiagnostics(
            rank_method=_section_value(section_lines, "RANK_METHOD") or RANK_METHOD,
            reduction_policy=_section_value(section_lines, "REDUCTION_POLICY") or REDUCTION_POLICY,
            selected=selected,
        )
    return GICReductionDiagnostics(
        rank_method=_section_value(lines, "RANK_METHOD") or RANK_METHOD,
        reduction_policy=_section_value(lines, "REDUCTION_POLICY") or REDUCTION_POLICY,
        selected=_parse_text_list(_section_value(lines, "SELECTED") or "") or selected,
        selected_by_family=_parse_text_list(_section_value(lines, "SELECTED_BY_FAMILY") or ""),
        skipped_singular=_parse_text_list(_section_value(lines, "SKIPPED_SINGULAR") or ""),
        skipped_dependent=_parse_text_list(_section_value(lines, "SKIPPED_DEPENDENT") or ""),
        skipped_singular_details=_parse_text_list(
            _section_value(lines, "SKIPPED_SINGULAR_DETAILS") or ""
        ),
        skipped_dependent_details=_parse_text_list(
            _section_value(lines, "SKIPPED_DEPENDENT_DETAILS") or ""
        ),
        conditioning_decisions=_parse_text_list(
            _section_value(lines, "CONDITIONING_DECISIONS") or ""
        ),
    )


def _parse_symmetry_diagnostics(
    section_lines: list[str],
) -> GICSymmetrizationDiagnostics | None:
    lines = _subsection(section_lines, "SYMMETRY_DIAGNOSTICS")
    if not lines:
        return None
    groups: list[GICSymmetrizedGroup] = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0].upper() != "GROUP":
            continue
        fields = _key_values(parts[2:] if len(parts) > 1 else parts[1:])
        try:
            groups.append(
                GICSymmetrizedGroup(
                    block=fields["BLOCK"],
                    family=fields["FAMILY"],
                    signature=fields["SIGNATURE"],
                    source_gics=_parse_text_list(fields.get("SOURCES", "")),
                    output_gics=_parse_text_list(fields.get("OUTPUTS", "")),
                )
            )
        except KeyError as exc:
            raise GICForgeContractError(f"invalid symmetry diagnostic group line: {line}") from exc
    return GICSymmetrizationDiagnostics(
        method=_section_value(lines, "METHOD") or "UNKNOWN",
        policy=_section_value(lines, "POLICY") or SYMMETRIZATION_POLICY,
        status=_section_value(lines, "STATUS") or "UNKNOWN",
        point_group=_section_value(lines, "POINT_GROUP")
        or _section_value(section_lines, "POINT_GROUP")
        or "UNKNOWN",
        symmetry_group=_section_value(lines, "SYMMETRY_GROUP")
        or _section_value(section_lines, "SYMMETRY_GROUP")
        or _section_value(lines, "POINT_GROUP")
        or _section_value(section_lines, "POINT_GROUP")
        or "UNKNOWN",
        total_symmetric_irrep=_section_value(lines, "TOTAL_SYMMETRIC_IRREP")
        or _section_value(section_lines, "TOTAL_SYMMETRIC_IRREP")
        or total_symmetric_irrep(_section_value(section_lines, "POINT_GROUP")),
        total_symmetric_gics=_parse_text_list(
            _section_value(lines, "TOTAL_SYMMETRIC_GICS")
            or _section_value(section_lines, "TOTAL_SYMMETRIC_GICS")
            or ""
        ),
        groups=tuple(groups),
        sign_gauge_policy=_section_value(lines, "SIGN_GAUGE_POLICY")
        or "largest_abs_coefficient_pivot",
        path_gauge_policy=_section_value(lines, "PATH_GAUGE_POLICY")
        or "subspace_overlap_procrustes",
        path_overlap_warning_threshold=_parse_float_value(
            _section_value(lines, "PATH_OVERLAP_WARNING_THRESHOLD"),
            default=SALC_PATH_OVERLAP_WARNING_THRESHOLD,
        ),
        operation_tolerance_angstrom=_parse_float_value(
            _section_value(lines, "OPERATION_TOLERANCE_ANGSTROM"),
            default=SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
        ),
        max_operation_residual_angstrom=_parse_float_value(
            _section_value(lines, "MAX_OPERATION_RESIDUAL_ANGSTROM"),
            default=0.0,
        ),
        min_operation_margin_angstrom=_parse_float_value(
            _section_value(lines, "MIN_OPERATION_MARGIN_ANGSTROM"),
            default=0.0,
        ),
        near_threshold_operations=_parse_text_list(
            _section_value(lines, "NEAR_THRESHOLD_OPERATIONS") or ""
        ),
    )


def _parse_float_value(text: str | None, *, default: float) -> float:
    if text is None or text == "":
        return float(default)
    try:
        return float(text)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid float value: {text}") from exc


def _parse_text_list(text: str) -> tuple[str, ...]:
    if not text or text.upper() == "NONE":
        return ()
    return tuple(item for item in text.split(",") if item)


def _csv_or_none(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"


def _parse_coefficients(text: str) -> tuple[tuple[str, float], ...]:
    if not text:
        return ()
    coefficients: list[tuple[str, float]] = []
    for item in text.replace(";", ",").split(","):
        if not item:
            continue
        if ":" not in item:
            raise GICForgeContractError(f"invalid GIC coefficient: {item}")
        primitive_id, value = item.split(":", 1)
        try:
            coefficients.append((primitive_id, float(value)))
        except ValueError as exc:
            raise GICForgeContractError(f"invalid GIC coefficient value: {item}") from exc
    return tuple(coefficients)


def _parse_bool(text: str | None) -> bool:
    return bool((text or "").strip().upper() == "TRUE")






def _planned_xh_stretch_policy(lines: list[str]) -> str:
    return _xh_stretch_policy(_section_value(section_content(lines, "GIC"), "XH_STRETCH_POLICY"))


def _planned_local_xh_bonds(lines: list[str]) -> tuple[tuple[int, int], ...]:
    return _normalize_pairs(_section_value(section_content(lines, "GIC"), "LOCAL_XH_BONDS"))


def _planned_local_xh_classes(lines: list[str]) -> tuple[str, ...]:
    return _normalize_xh_classes(_section_value(section_content(lines, "GIC"), "LOCAL_XH_CLASSES"))


def _fragment_mode(value: str | None) -> str:
    text = (value or FRAGMENT_MODE_NONE).strip().upper().replace("-", "_")
    aliases = {
        "PSEUDO": FRAGMENT_MODE_PSEUDO_BONDS,
        "PSEUDOBONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "HBOND": FRAGMENT_MODE_PSEUDO_BONDS,
        "HBONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "H_BONDS": FRAGMENT_MODE_PSEUDO_BONDS,
        "SPECIAL": FRAGMENT_MODE_SPECIAL_COORDINATES,
        "FRAGMENT": FRAGMENT_MODE_SPECIAL_COORDINATES,
        "FRAGMENT_COORDINATES": FRAGMENT_MODE_SPECIAL_COORDINATES,
    }
    normalized = aliases.get(text, text)
    if normalized not in FRAGMENT_MODES:
        raise GICForgeContractError(f"unsupported fragment mode: {value}")
    return normalized


def _out_of_plane_mode(primitives: tuple[GICPrimitive, ...]) -> str:
    if any(
        primitive.function == "IMPD" or primitive.family == "IMPROPER_DIHEDRAL"
        for primitive in primitives
    ):
        return "IMPROPER_DIHEDRAL"
    return "OUT_OF_PLANE"


def _pairs_text(pairs: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{left}-{right}" for left, right in pairs) if pairs else "NONE"


def _csv_or_none_from_strings(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"


def _parse_int(text: str | None) -> int:
    if text is None:
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid integer field: {text}") from exc


def _parse_pseudo_bonds(section: list[str]) -> tuple[tuple[int, int], ...]:
    bonds: list[tuple[int, int]] = []
    for line in _subsection(section, "PSEUDO_BONDS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 3:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}")
        try:
            left = int(parts[1])
            right = int(parts[2])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}") from exc
        if left == right or left < 1 or right < 1:
            raise GICForgeContractError(f"invalid pseudo-bond indexes: {line}")
        bonds.append(tuple(sorted((left, right))))
    return tuple(sorted(set(bonds)))


def _parse_pseudo_bond_kinds(section: list[str]) -> tuple[str, ...]:
    kinds: list[tuple[tuple[int, int], str]] = []
    for line in _subsection(section, "PSEUDO_BONDS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 3:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}")
        fields = _key_values(parts[3:])
        try:
            pair = tuple(sorted((int(parts[1]), int(parts[2]))))
        except ValueError as exc:
            raise GICForgeContractError(f"invalid pseudo-bond line: {line}") from exc
        kinds.append((pair, fields.get("KIND", "INTERFRAGMENT_CLOSEST").upper()))
    return tuple(kind for _pair, kind in sorted(kinds))


def _cartesian_column_labels(natoms: int) -> tuple[str, ...]:
    axes = ("X", "Y", "Z")
    return tuple(f"{atom}:{axis}" for atom in range(1, natoms + 1) for axis in axes)


def _primitive_line(primitive: GICPrimitive) -> str:
    atoms = ",".join(str(atom) for atom in primitive.atoms)
    mode = (
        f" MODE={primitive.mode}"
        if primitive.function in MODE_BEARING_PRIMITIVE_FUNCTIONS
        else ""
    )
    ref_atoms = (
        " REF_ATOMS=" + ",".join(str(atom) for atom in primitive.ref_atoms)
        if primitive.ref_atoms
        else ""
    )
    refs = " REFS=" + ",".join(primitive.refs) if primitive.refs else ""
    frame_atoms = (
        " FRAME_ATOMS=" + ",".join(str(atom) for atom in primitive.frame_atoms)
        if primitive.frame_atoms
        else ""
    )
    ref_frame_atoms = (
        " REF_FRAME_ATOMS=" + ",".join(str(atom) for atom in primitive.ref_frame_atoms)
        if primitive.ref_frame_atoms
        else ""
    )
    provenance = (
        f" PROVENANCE={primitive.provenance}" if primitive.provenance != AUTO_PROVENANCE else ""
    )
    semantic_id = f" SEMANTIC_ID={primitive.semantic_id}" if primitive.semantic_id else ""
    semantic_type = f" SEMANTIC_TYPE={primitive.semantic_type}" if primitive.semantic_type else ""
    chart = f" CHART={primitive.chart}" if primitive.chart != "PRINCIPAL" else ""
    chart_reference = (
        f" CHART_REFERENCE_RADIAN={primitive.chart_reference_radian:.17g}"
        if primitive.chart_reference_radian is not None
        else ""
    )
    return (
        f"{primitive.identifier} NAME={primitive.name} FAMILY={primitive.family} "
        f"CLASS={primitive.reduction_class} FUNCTION={primitive.function} "
        f"ATOMS={atoms}{ref_atoms}{refs}"
        f"{frame_atoms}{ref_frame_atoms}{mode}"
        f"{provenance}{semantic_id}{semantic_type}{chart}{chart_reference} "
        f"GAUSSIAN={primitive.gaussian_expression()}"
    )


def _frozen_gic_line(gic: FrozenGIC) -> str:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    coeffs = ",".join(
        # Seventeen significant digits guarantee an exact binary-float
        # round trip.  Twelve fixed decimals changed SONIC identities for
        # highly symmetric ring combinations such as cubane and cyclohexane.
        f"{primitive_id}:{coefficient:.17g}"
        for primitive_id, coefficient in coefficients
    )
    return f"{gic.identifier} NAME={gic.name} FAMILY={gic.family} IRREP={gic.irrep} COEFFS={coeffs}"


def _sycart_components(vector: tuple[float, ...]) -> str:
    parts = []
    axes = ("X", "Y", "Z")
    for idx, value in enumerate(vector):
        if abs(value) <= 1.0e-12:
            continue
        atom = idx // 3 + 1
        axis = axes[idx % 3]
        parts.append(f"{atom}:{axis}={value:.12g}")
    return "COMPONENTS=" + (";".join(parts) if parts else "NONE")


def _symmetry_mode(definition: GICDefinition) -> str:
    diagnostics = definition.symmetry_diagnostics
    if (
        definition.symmetrize
        and diagnostics is not None
        and diagnostics.method == POINT_GROUP_PROJECTOR_METHOD
        and diagnostics.groups
    ):
        return "POINT_GROUP_PROJECTOR"
    if (
        definition.symmetrize
        and diagnostics is not None
        and diagnostics.method == LOCAL_SYMMETRIZATION_METHOD
        and diagnostics.groups
    ):
        return "LOCAL_BLOCK_C1" if definition.point_group.upper() == "C1" else "LOCAL_BLOCK"
    if definition.point_group.upper() == "C1":
        return "IDENTITY_C1" if definition.symmetrize else "UNSYMMETRIZED_C1"
    return "SYMMETRIZED" if definition.symmetrize else "UNSYMMETRIZED"


def _bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


from .transition_state_conditioning import (
    _wilson_tangent_diagnostics,
    _condition_transition_state_salc_alternatives,
    _select_rank_aware_chart,
    _with_prescribed_distance_only_primitives,
    _with_transition_state_reaction_distance_family,
)

from .chart_freezing import (
    _with_periodic_coordinate_estimates,
    _selected_gic_values,
    _attach_chart_atlas,
)
