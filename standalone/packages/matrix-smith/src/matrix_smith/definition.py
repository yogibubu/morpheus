from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
import re

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.vdw_radii import descriptor_vdw_radius
from matrix_chem import (
    MATRIX_XYZ_PRIMITIVES_SCHEMA,
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_primitive_contract,
    validate_primitive_contract,
    write_validation_section,
)
from matrix_chem.topology.contracts import SUPPORTED_TOPOLOGY_SCHEMAS, schema_line_supported
from matrix_core.parameters.bdpcs3 import load_bdpcs3_parameters
from matrix_core import read_sectioned_lines, replace_section, section_content

from .contracts import (
    ORACLE_XYZ_GIC_SCHEMA,
    ORACLE_XYZ_SYCART_SCHEMA,
    GICForgeContractError,
    validate_gicforge_prerequisites,
)
from .bmatrix import SparseBMatrix, SparseBRow
from .coordinate_registry import CoordinateSignature
from .generators import generate_stretch_coordinates
from .semantic import (
    AUTO_PROVENANCE,
    SEMANTIC_GRAMMAR_VERSION,
    USER_PROVENANCE,
    SemanticContract,
    SemanticCoordinate,
    semantic_contract_from_sectioned_lines,
    semantic_signature_for_primitive_like,
)
from .policy import (
    B_MATRIX_BACKEND,
    DIAGNOSTIC_FINITE_DIFFERENCE_STEP,
    FRAGMENT_MODE_NONE,
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
    FRAGMENT_MODES,
    GIC_BACKEND,
    LINEAR_ANGLE_DEGREES,
    LOCAL_SYMMETRIZATION_METHOD,
    POINT_GROUP_PROJECTOR_METHOD,
    PRIMITIVE_FAMILY_ORDER,
    PSEUDO_BOND_EFFECTIVE_ORDER,
    PSEUDO_CYCLE_CLOSURE_VDW_SCALE,
    PROJECTOR_SYMMETRIZATION_POLICY,
    RANK_METHOD,
    RANK_TOLERANCE,
    REDUCTION_POLICY,
    SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SPECIAL_REDUCTION_CLASS,
    SYMMETRIZATION_POLICY,
    SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
    SYCART_BACKEND,
    XH_STRETCH_CLASSES,
    XH_STRETCH_POLICIES,
    XH_STRETCH_POLICY_LOCAL_ALL,
    XH_STRETCH_POLICY_LOCAL_SELECTED,
    XH_STRETCH_POLICY_SYMMETRIZE,
    primitive_prefix,
    primitive_reduction_class,
    primitive_symmetry_block,
)
from .symmetry_labels import (
    irrep_characters_for_operations,
    irrep_name_prefix,
    is_total_symmetric_irrep,
    non_total_irrep_sequence,
    total_symmetric_irrep,
)


_GAUSSIAN_DEPENDENCY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")


@dataclass(frozen=True)
class GICPrimitive:
    identifier: str
    name: str
    family: str
    function: str
    atoms: tuple[int, ...]
    mode: int = 0
    ref_atoms: tuple[int, ...] = ()
    refs: tuple[str, ...] = ()
    frame_atoms: tuple[int, ...] = ()
    ref_frame_atoms: tuple[int, ...] = ()
    provenance: str = AUTO_PROVENANCE
    semantic_id: str = ""
    semantic_type: str = ""

    def gaussian_expression(self) -> str:
        if self.function == "IMPD":
            atoms = ",".join(str(atom) for atom in _improper_dihedral_atoms(self.atoms))
            return f"D({atoms})"
        if not self.is_gaussian_native:
            return "NONE"
        atoms = ",".join(str(atom) for atom in self.atoms)
        if self.function == "L":
            return f"L({atoms},0,{self.mode})"
        return f"{self.function}({atoms})"

    @property
    def is_gaussian_native(self) -> bool:
        return self.function in {"R", "A", "L", "D", "U"}

    @property
    def reduction_class(self) -> str:
        return primitive_reduction_class(self.family)


@dataclass(frozen=True)
class FrozenGIC:
    identifier: str
    name: str
    family: str
    irrep: str
    primitive_id: str
    gaussian_expression: str
    coefficients: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class GICReductionDiagnostics:
    rank_method: str
    reduction_policy: str
    selected: tuple[str, ...] = ()
    skipped_singular: tuple[str, ...] = ()
    skipped_dependent: tuple[str, ...] = ()
    selected_by_family: tuple[str, ...] = ()
    skipped_singular_details: tuple[str, ...] = ()
    skipped_dependent_details: tuple[str, ...] = ()


@dataclass(frozen=True)
class GICSymmetrizedGroup:
    block: str
    family: str
    signature: str
    source_gics: tuple[str, ...]
    output_gics: tuple[str, ...]


@dataclass(frozen=True)
class GICSymmetrizationDiagnostics:
    method: str
    policy: str
    status: str
    point_group: str
    symmetry_group: str
    total_symmetric_irrep: str
    total_symmetric_gics: tuple[str, ...] = ()
    groups: tuple[GICSymmetrizedGroup, ...] = ()
    sign_gauge_policy: str = "largest_abs_coefficient_pivot"
    path_gauge_policy: str = "subspace_overlap_procrustes"
    path_overlap_warning_threshold: float = SALC_PATH_OVERLAP_WARNING_THRESHOLD
    operation_tolerance_angstrom: float = SYMMETRY_OPERATION_TOLERANCE_ANGSTROM
    max_operation_residual_angstrom: float = 0.0
    min_operation_margin_angstrom: float = 0.0
    near_threshold_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GICPointGroupOperation:
    label: str
    rotation: tuple[tuple[float, float, float], ...]
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class GICDefinition:
    backend: str
    point_group: str
    symmetrize: bool
    target_rank: int
    rank: int
    candidate_count: int
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    primitives: tuple[GICPrimitive, ...]
    gics: tuple[FrozenGIC, ...]
    reduction_diagnostics: GICReductionDiagnostics | None = None
    symmetry_diagnostics: GICSymmetrizationDiagnostics | None = None
    fragment_mode: str = FRAGMENT_MODE_NONE
    pseudo_bonds: tuple[tuple[int, int], ...] = ()
    pseudo_bond_kinds: tuple[str, ...] = ()
    xh_stretch_policy: str = XH_STRETCH_POLICY_SYMMETRIZE
    local_xh_bonds: tuple[tuple[int, int], ...] = ()
    local_xh_classes: tuple[str, ...] = ()
    ring_puckering_diagnostics: tuple[str, ...] = ()
    contract_schema_version: str = ORACLE_XYZ_GIC_SCHEMA
    semantic_grammar_version: str = ""
    semantic_diagnostics: tuple[str, ...] = ()
    primitive_source: str = "LEGACY_RECONSTRUCTED"
    primitive_source_schema: str = ""
    primitive_b_matrix_sha256: str = ""


@dataclass(frozen=True)
class GICBMatrix:
    backend: str
    coordinate_labels: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    irreps: tuple[str, ...]
    cartesian_columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    sparse_rows: tuple[SparseBRow, ...] = ()

    def sparse_matrix(self) -> SparseBMatrix:
        sparse_rows = self.sparse_rows or tuple(SparseBRow.from_dense(row) for row in self.rows)
        return SparseBMatrix(
            rows=sparse_rows,
            column_count=len(self.cartesian_columns),
            row_labels=self.coordinate_labels,
            backend=self.backend,
        )


@dataclass(frozen=True)
class SYCartDefinition:
    backend: str
    point_group: str
    target_rank: int
    vectors: tuple[tuple[float, ...], ...]


def build_gic_definition_from_xyzin(
    path: Path,
    *,
    symmetrize: bool = False,
    symmetry_group: str | None = None,
    improper_dihedrals: bool | None = None,
    fragment_mode: str | None = None,
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    separate_exocyclic_torsions: bool = False,
    rank_tolerance: float = RANK_TOLERANCE,
) -> GICDefinition:
    """Build a frozen SONIC coordinate definition from saved xyzin state."""
    definition, atom_symbols, operations = construct_gic_definition_from_xyzin(
        path,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        rank_tolerance=rank_tolerance,
        retain_candidate_primitives=symmetrize,
    )
    if not symmetrize:
        return definition
    symmetrized = symmetrize_gic_definition(
        definition,
        atom_symbols=atom_symbols,
        symmetry_operations=operations,
    )
    if separate_exocyclic_torsions:
        symmetrized = _retain_individual_exocyclic_torsions(symmetrized, definition)
    return symmetrized


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
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    separate_exocyclic_torsions: bool = False,
    rank_tolerance: float = RANK_TOLERANCE,
    retain_candidate_primitives: bool = False,
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
    # ORACLE/SMITH always own a native out-of-plane coordinate.  Gaussian
    # improper dihedrals are an export representation and must never alter the
    # frozen primitive or SONIC contracts.
    improper_dihedrals = False
    mode = _planned_fragment_mode(lines) if fragment_mode is None else _fragment_mode(fragment_mode)
    fragment_records = _fragment_records(target)
    planned_xh_policy = _planned_xh_stretch_policy(lines)
    explicit_local_xh_selection = local_xh_bonds is not None or local_xh_classes is not None
    resolved_xh_policy = _xh_stretch_policy(
        XH_STRETCH_POLICY_LOCAL_SELECTED
        if xh_stretch_policy is None and explicit_local_xh_selection
        else planned_xh_policy if xh_stretch_policy is None else xh_stretch_policy
    )
    resolved_local_xh_bonds = _normalize_pairs(
        _planned_local_xh_bonds(lines) if local_xh_bonds is None else local_xh_bonds
    )
    resolved_local_xh_classes = _normalize_xh_classes(
        _planned_local_xh_classes(lines) if local_xh_classes is None else local_xh_classes
    )
    interaction_centers = _interaction_center_definition(target)
    bonds = _topology_bonds(lines, natoms=geometry.natoms)
    rings = _topology_rings(lines, natoms=geometry.natoms)
    bond_orders = topology_bond_orders_from_lines(lines, natoms=geometry.natoms)
    bond_order_components = topology_bond_order_components_from_lines(
        lines, natoms=geometry.natoms
    )
    primitive_source, primitive_source_schema, primitive_b_matrix_sha256 = (
        _consume_oracle_primitive_contract(target, coords, bonds=bonds)
    )

    if (
        not semantic_contract.protect_coordinates
        and not fragment_records
        and _empty_interaction_centers(interaction_centers)
        and mode != FRAGMENT_MODE_PSEUDO_BONDS
    ):
        definition = _construct_merlino_python_definition(
            geometry.atoms,
            coords,
            point_group=point_group,
            improper_dihedrals=improper_dihedrals,
            xh_stretch_policy=resolved_xh_policy,
            local_xh_bonds=resolved_local_xh_bonds,
            local_xh_classes=resolved_local_xh_classes,
            local_salc=local_salc,
            local_salc_settings=local_salc_settings,
            separate_exocyclic_torsions=separate_exocyclic_torsions,
            ring_puckering_diagnostics=_ring_puckering_diagnostics(
                rings,
                bond_orders=bond_orders,
                bond_order_components=bond_order_components,
            ),
        )
        definition = replace(
            definition,
            primitive_source=primitive_source,
            primitive_source_schema=primitive_source_schema,
            primitive_b_matrix_sha256=primitive_b_matrix_sha256,
        )
        return definition, tuple(geometry.atoms), symmetry_operations

    pseudo_bonds: tuple[tuple[int, int], ...] = ()
    pseudo_bond_kinds: tuple[str, ...] = ()
    target_rank = _vibrational_rank(coords)
    topology_bonds = bonds
    base_semantic_contract = semantic_contract
    additional_contacts = 0
    previous_contacts: tuple[tuple[int, int, str], ...] | None = None
    while True:
        candidate_fragment_records = fragment_records
        candidate_interaction_centers = interaction_centers
        candidate_bonds = topology_bonds
        pseudo_bonds = ()
        pseudo_bond_kinds = ()
        normalized_contacts: tuple[tuple[int, int, str], ...] = ()
        if mode == FRAGMENT_MODE_PSEUDO_BONDS:
            # The minimum physical contact graph is the default.  If and only
            # if its complete SONIC candidate matrix is rank deficient, the
            # loop requests one further deterministic conditioning contact.
            candidate_fragment_records = ()
            pseudo_contacts = tuple(
                _pseudo_bonds_for_fragments(
                    fragment_records,
                    bonds=topology_bonds,
                    coords=coords,
                    atom_symbols=tuple(geometry.atoms),
                    additional_contacts=additional_contacts,
                )
                if fragment_records
                else _intramolecular_hbond_contacts(
                    atom_symbols=tuple(geometry.atoms),
                    coords=coords,
                    bonds=topology_bonds,
                )
                or _intramolecular_pseudo_bond_contacts(
                    atom_symbols=tuple(geometry.atoms),
                    coords=coords,
                    bonds=topology_bonds,
                )
            )
            normalized_contacts = tuple(
                _normalize_pseudo_contact(contact) for contact in pseudo_contacts
            )
            pseudo_bonds = tuple((left, right) for left, right, _kind in normalized_contacts)
            pseudo_bond_kinds = tuple(kind for _left, _right, kind in normalized_contacts)
            candidate_bonds = tuple(sorted(set(topology_bonds + pseudo_bonds)))
        candidates = _primitive_candidates(
            candidate_bonds,
            rings=rings,
            coords=coords,
            natoms=geometry.natoms,
            atom_symbols=tuple(geometry.atoms),
            xh_stretch_policy=resolved_xh_policy,
            local_xh_bonds=resolved_local_xh_bonds,
            local_xh_classes=resolved_local_xh_classes,
            improper_dihedrals=improper_dihedrals,
            fragment_records=candidate_fragment_records,
            interaction_centers=candidate_interaction_centers,
            pseudo_bonds=pseudo_bonds,
            bond_orders=bond_orders,
        )
        candidates, semantic_contract, semantic_diagnostics = (
            _apply_semantic_contract_to_candidates(
                candidates,
                base_semantic_contract,
                symmetry_operations=symmetry_operations,
            )
        )
        selected, rank, reduction_diagnostics = _select_ranked_primitives_with_diagnostics(
            candidates,
            coords,
            target_rank=target_rank,
            rank_tolerance=rank_tolerance,
        )
        if rank == target_rank or mode != FRAGMENT_MODE_PSEUDO_BONDS or not fragment_records:
            break
        if normalized_contacts == previous_contacts:
            break
        previous_contacts = normalized_contacts
        additional_contacts += 1
    bonds = candidate_bonds
    if rank != target_rank:
        raise GICForgeContractError(
            "insufficient independent primitive coordinates: "
            f"need {target_rank}, selected rank {rank} from {len(candidates)} candidates"
        )

    definition_primitives = tuple(candidates) if retain_candidate_primitives else tuple(selected)
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
        primitives=definition_primitives,
        gics=gics,
        reduction_diagnostics=reduction_diagnostics,
        symmetry_diagnostics=_empty_symmetry_diagnostics(point_group, requested=False),
        fragment_mode=mode if pseudo_bonds or fragment_records else FRAGMENT_MODE_NONE,
        pseudo_bonds=pseudo_bonds,
        pseudo_bond_kinds=pseudo_bond_kinds,
        xh_stretch_policy=resolved_xh_policy,
        local_xh_bonds=resolved_local_xh_bonds,
        local_xh_classes=resolved_local_xh_classes,
        ring_puckering_diagnostics=_ring_puckering_diagnostics(
            rings,
            bond_orders=bond_orders,
            bond_order_components=bond_order_components,
        ),
        semantic_grammar_version=(
            semantic_contract.grammar_version if semantic_contract.coordinates else ""
        ),
        semantic_diagnostics=semantic_diagnostics,
        primitive_source=primitive_source,
        primitive_source_schema=primitive_source_schema,
        primitive_b_matrix_sha256=primitive_b_matrix_sha256,
    )
    return definition, tuple(geometry.atoms), symmetry_operations


def _empty_interaction_centers(interaction_centers: object | None) -> bool:
    if interaction_centers is None:
        return True
    centers = tuple(getattr(interaction_centers, "centers", ()) or ())
    interactions = tuple(getattr(interaction_centers, "interactions", ()) or ())
    strategy = str(getattr(interaction_centers, "strategy", "NONE")).upper()
    return strategy in {"", "NONE"} and not centers and not interactions


def _consume_oracle_primitive_contract(
    path: Path,
    coordinates_angstrom: np.ndarray,
    *,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[str, str, str]:
    """Validate and consume ORACLE's primitive/B contract when available."""
    if not section_content(read_sectioned_lines(Path(path)), "PRIMITIVES"):
        return "LEGACY_RECONSTRUCTED", "", ""
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
    return "ORACLE_CONTRACT", contract.schema or MATRIX_XYZ_PRIMITIVES_SCHEMA, contract.b_matrix_sha256


def _construct_merlino_python_definition(
    atom_symbols: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    *,
    point_group: str,
    improper_dihedrals: bool,
    xh_stretch_policy: str,
    local_xh_bonds: tuple[tuple[int, int], ...],
    local_xh_classes: tuple[str, ...],
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    separate_exocyclic_torsions: bool = False,
    ring_puckering_diagnostics: tuple[str, ...] = (),
) -> GICDefinition:
    from matrix_smith.runtime.gicforge_python import build_gicforge_python_model

    model = build_gicforge_python_model(
        atom_symbols,
        coordinates_angstrom,
        impdih=improper_dihedrals,
        onedih=True,
        svd_local=False,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
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
    )


def _merlino_runtime_primitive(
    identifier: str,
    primitive: object,
    family: str,
    *,
    refs: tuple[str, ...] = (),
) -> GICPrimitive:
    kind = str(getattr(primitive, "kind"))
    atoms = tuple(int(atom) + 1 for atom in getattr(primitive, "atoms"))
    mode = int(getattr(primitive, "mode", 0))
    if kind == "bond":
        return GICPrimitive(identifier, identifier, family, "R", atoms, refs=refs)
    if kind == "angle":
        return GICPrimitive(identifier, identifier, family, "A", atoms, refs=refs)
    if kind == "linear_bend":
        return GICPrimitive(identifier, identifier, family, "L", atoms, mode=mode, refs=refs)
    if kind == "dihedral":
        return GICPrimitive(identifier, identifier, family, "D", atoms, refs=refs)
    if kind == "out_of_plane":
        return GICPrimitive(identifier, identifier, family, "U", atoms, refs=refs)
    raise GICForgeContractError(f"unsupported Merlino Python primitive kind: {kind}")


def _normalized_coefficients(
    coefficients: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    norm = float(np.sqrt(sum(float(coefficient) ** 2 for _identifier, coefficient in coefficients)))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        return coefficients
    return tuple((identifier, float(coefficient) / norm) for identifier, coefficient in coefficients)


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
    closed_primitives = _symmetry_closed_projector_primitives(
        definition.primitives,
        symmetry_operations=tuple(symmetry_operations),
    )
    gics, symmetry_diagnostics = _apply_local_symmetrization(
        definition.gics,
        closed_primitives,
        atom_symbols=tuple(atom_symbols),
        point_group=definition.point_group,
        requested=True,
        symmetry_operations=tuple(symmetry_operations),
        reference_coordinates_angstrom=definition.reference_coordinates_angstrom,
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
        primitive_source=definition.primitive_source,
        primitive_source_schema=definition.primitive_source_schema,
        primitive_b_matrix_sha256=definition.primitive_b_matrix_sha256,
    )


def build_sycart_definition_from_xyzin(path: Path) -> SYCartDefinition:
    """Build an external-mode-free Cartesian basis for #SYCART."""
    target = Path(path)
    validate_gicforge_prerequisites(target)
    lines = read_sectioned_lines(target)
    geometry = read_enriched_xyz(target)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    target_rank = _vibrational_rank(coords)
    basis = _cartesian_vibrational_basis(coords, target_rank=target_rank)
    return SYCartDefinition(
        backend=SYCART_BACKEND,
        point_group=_point_group(lines),
        target_rank=target_rank,
        vectors=tuple(tuple(float(value) for value in row) for row in basis.T),
    )


def gic_definition_section_lines(definition: GICDefinition) -> list[str]:
    primitive_dependency = (
        definition.primitive_source_schema
        if definition.primitive_source == "ORACLE_CONTRACT"
        else definition.primitive_source
    )
    lines = [
        f"SCHEMA {definition.contract_schema_version}",
        f"CONTRACT_SCHEMA_VERSION {definition.contract_schema_version}",
        "STATUS BUILT",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA} "
        f"SYMMETRY=oracle.xyz.symmetry.v1 PRIMITIVES={primitive_dependency}",
        f"PRIMITIVE_SOURCE {definition.primitive_source}",
        f"PRIMITIVE_SOURCE_SCHEMA {definition.primitive_source_schema or 'NONE'}",
        f"PRIMITIVE_B_MATRIX_SHA256 {definition.primitive_b_matrix_sha256 or 'NONE'}",
        f"SEMANTIC_GRAMMAR {definition.semantic_grammar_version or 'NONE'}",
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
        (
            "SYMMETRY_MODE IDENTITY_C1"
            if definition.point_group.upper() == "C1"
            else "SYMMETRY_MODE UNSYMMETRIZED"
        ),
        f"TARGET_RANK {definition.target_rank}",
        f"COORD_COUNT {len(definition.vectors)}",
        "[SYCART]",
    ]
    if definition.vectors:
        lines.extend(
            f"SYC{idx:03d} IRREP=A " + _sycart_components(vector)
            for idx, vector in enumerate(definition.vectors, start=1)
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
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
    separate_exocyclic_torsions: bool = False,
) -> GICDefinition:
    target = Path(path)
    definition = build_gic_definition_from_xyzin(
        target,
        symmetrize=symmetrize,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
    )
    replace_section(target, "GIC", gic_definition_section_lines(definition))
    if sycart:
        sycart_definition = build_sycart_definition_from_xyzin(target)
        replace_section(target, "SYCART", sycart_definition_section_lines(sycart_definition))
    return definition


def build_pes_exploration_gic_definition_from_xyzin(
    path: Path,
    *,
    retained_group: str = "C1",
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
        fragment_mode=base.fragment_mode,
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
    xh_stretch_policy: str | None = None,
    local_xh_bonds: tuple[tuple[int, int], ...] | None = None,
    local_xh_classes: tuple[str, ...] | None = None,
    local_salc: bool = False,
    local_salc_settings: object | None = None,
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
        write_validation_section(build_path)
    elif build_path != source_path:
        build_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    if fragment_mode is not None and _fragment_mode(fragment_mode) != FRAGMENT_MODE_NONE:
        build_lines = read_sectioned_lines(build_path)
        if not section_content(build_lines, "FRAGMENTS"):
            from matrix_fragments import write_fragment_build_section

            write_fragment_build_section(build_path)
    return write_gicforge_build_sections(
        build_path,
        symmetrize=symmetrize,
        sycart=sycart,
        symmetry_group=symmetry_group,
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
        local_salc=local_salc,
        local_salc_settings=local_salc_settings,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
    )


build_gsnic_definition_from_xyzin = build_sonic_definition_from_xyzin
write_gsnic_build_sections = write_sonic_build_sections
write_gsnic_build_sections_from_cartesian = write_sonic_build_sections_from_cartesian


def build_gic_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> GICBMatrix:
    """Evaluate the Wilson B matrix for a frozen GIC definition."""
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    if definition.backend == "merlino-python-gicforge.v1":
        return _build_merlino_python_b_matrix(definition, coords)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    reference_coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic in definition.gics:
        weighted_rows: list[tuple[float, SparseBRow]] = []
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise GICForgeContractError(
                    f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                )
            weighted_rows.append(
                (
                    float(coefficient),
                    _sparse_analytic_b_row(
                        primitive,
                        coords,
                        reference_coords=reference_coords,
                    ),
                )
            )
        sparse_row = SparseBRow.combine(*tuple(weighted_rows))
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def evaluate_gic_values(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the values of a frozen SONIC/GIC definition."""
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.primitives import eval_primitives

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive)
            for primitive in definition.primitives
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        values_by_id = {
            primitive.identifier: _continuous_primitive_value(
                primitive,
                float(raw_values[index]),
                float(reference_values[index]),
            )
            for index, primitive in enumerate(definition.primitives)
        }
    else:
        values_by_id = {
            identifier: _continuous_primitive_value(
                primitive,
                float(
                    _primitive_value(
                        primitive,
                        coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
                float(
                    _primitive_value(
                        primitive,
                        rotation_reference if primitive.function == "FROT" else reference_coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
            )
            for identifier, primitive in primitive_by_id.items()
        }
    values: list[float] = []
    for gic in definition.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        values.append(
            sum(float(coefficient) * values_by_id[primitive_id] for primitive_id, coefficient in coefficients)
        )
    return np.asarray(values, dtype=float)


def evaluate_gic_value(
    definition: GICDefinition,
    coordinate_index: int,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> float:
    """Evaluate one frozen SONIC without evaluating unrelated primitive rows."""
    index = int(coordinate_index)
    if index < 0 or index >= len(definition.gics):
        raise IndexError(f"SONIC coordinate index {coordinate_index} is outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    gic = definition.gics[index]
    components = gic.coefficients or ((gic.primitive_id, 1.0),)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitives = tuple(primitive_by_id[primitive_id] for primitive_id, _coefficient in components)
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.primitives import eval_primitives

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive) for primitive in primitives
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(raw_values[primitive_index]),
                float(reference_values[primitive_index]),
            )
            for primitive_index, primitive in enumerate(primitives)
        )
    else:
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(
                    _primitive_value(
                        primitive,
                        coords,
                        reference_coords=(
                            rotation_reference
                            if primitive.function == "FROT"
                            else reference_coords
                        ),
                    )
                ),
                float(
                    _primitive_value(
                        primitive,
                        rotation_reference
                        if primitive.function == "FROT"
                        else reference_coords,
                        reference_coords=(
                            rotation_reference
                            if primitive.function == "FROT"
                            else reference_coords
                        ),
                    )
                ),
            )
            for primitive in primitives
        )
    return float(
        sum(
            float(coefficient) * value
            for (_primitive_id, coefficient), value in zip(components, values, strict=True)
        )
    )


def _continuous_primitive_value(
    primitive: GICPrimitive,
    value: float,
    reference: float,
) -> float:
    if primitive.function not in {"D", "IMPD", "FROT", "RPCK"}:
        return float(value)
    delta = (float(value) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
    return float(reference) + delta


def _build_merlino_python_b_matrix(
    definition: GICDefinition,
    coords: np.ndarray,
) -> GICBMatrix:
    from matrix_smith.survibfit.pipeline import b_matrix_analytic

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitive_index_by_id = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    primitive_basis = tuple(
        _survibfit_primitive_from_gic_primitive(primitive) for primitive in definition.primitives
    )
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    primitive_sparse_b = tuple(SparseBRow.from_dense(row) for row in primitive_b)
    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic in definition.gics:
        weighted_rows: list[tuple[float, SparseBRow]] = []
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise GICForgeContractError(
                    f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                )
            primitive_index = primitive_index_by_id[primitive.identifier]
            weighted_rows.append((float(coefficient), primitive_sparse_b[primitive_index]))
        sparse_row = SparseBRow.combine(*tuple(weighted_rows))
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend="merlino-python-survibfit-bmatrix.v1",
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def _survibfit_primitive_from_gic_primitive(primitive: GICPrimitive):
    from matrix_smith.survibfit.primitives import Primitive

    atoms = tuple(atom - 1 for atom in primitive.atoms)
    if primitive.function == "R":
        return Primitive("bond", atoms)
    if primitive.function == "A":
        return Primitive("angle", atoms)
    if primitive.function == "L":
        return Primitive("linear_bend", atoms, mode=primitive.mode)
    if primitive.function == "D":
        return Primitive("dihedral", atoms)
    if primitive.function == "U":
        return Primitive("out_of_plane", atoms)
    raise GICForgeContractError(
        f"unsupported Merlino Python primitive function for B matrix: {primitive.function}"
    )


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
    return GICDefinition(
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
        contract_schema_version=_section_value(section, "CONTRACT_SCHEMA_VERSION")
        or ORACLE_XYZ_GIC_SCHEMA,
        semantic_grammar_version=(
            "" if (_section_value(section, "SEMANTIC_GRAMMAR") or "NONE").upper() == "NONE"
            else (_section_value(section, "SEMANTIC_GRAMMAR") or SEMANTIC_GRAMMAR_VERSION)
        ),
        semantic_diagnostics=tuple(
            line
            for line in _subsection(section, "SEMANTIC_DIAGNOSTICS")
            if line.strip() and line.strip().upper() != "NONE"
        ),
        primitive_source=_section_value(section, "PRIMITIVE_SOURCE") or "LEGACY_RECONSTRUCTED",
        primitive_source_schema=(
            "" if (_section_value(section, "PRIMITIVE_SOURCE_SCHEMA") or "NONE") == "NONE"
            else (_section_value(section, "PRIMITIVE_SOURCE_SCHEMA") or "")
        ),
        primitive_b_matrix_sha256=(
            "" if (_section_value(section, "PRIMITIVE_B_MATRIX_SHA256") or "NONE") == "NONE"
            else (_section_value(section, "PRIMITIVE_B_MATRIX_SHA256") or "")
        ),
    )


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


def gaussian_gic_lines_from_xyzin(
    path: Path,
    *,
    total_symmetric_only: bool = False,
    freeze_non_total: bool = True,
) -> list[str]:
    gic = section_content(read_sectioned_lines(Path(path)), "GIC")
    block = _subsection(gic, "GAUSSIAN_GIC")
    lines = [line for line in block if line.strip() and line.strip().upper() != "NONE"]
    total_names = set(_parse_text_list(_section_value(gic, "TOTAL_SYMMETRIC_GICS")))
    if freeze_non_total:
        lines = _freeze_non_total_gaussian_lines(
            lines,
            final_names=_frozen_gic_names(gic),
            total_names=total_names,
        )
    if not total_symmetric_only:
        return lines
    if not total_names:
        return lines
    return _gaussian_dependency_closed_lines(lines, total_names)


def _freeze_non_total_gaussian_lines(
    lines: list[str],
    *,
    final_names: set[str],
    total_names: set[str],
) -> list[str]:
    if not final_names or not total_names:
        return lines
    return [
        _gaussian_line_with_frozen_label(line)
        if (label := _gaussian_definition_label(line)) in final_names and label not in total_names
        else line
        for line in lines
    ]


def _gaussian_line_with_frozen_label(line: str) -> str:
    if "=" not in line:
        return line
    label, expression = line.split("=", 1)
    base = label.strip()
    if "(" in base:
        return line
    return f"{base}(Frozen) = {expression.strip()}"


def _frozen_gic_names(gic_section: list[str]) -> set[str]:
    names: set[str] = set()
    for line in _subsection(gic_section, "FROZEN_GICS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        fields = _key_values(text.split()[1:])
        name = fields.get("NAME")
        if name:
            names.add(name)
    return names


def _gaussian_dependency_closed_lines(lines: list[str], wanted_names: set[str]) -> list[str]:
    definitions: dict[str, str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for line in lines:
        label = _gaussian_definition_label(line)
        if label is None:
            continue
        definitions[label] = line
        dependencies[label] = tuple(_gaussian_definition_dependencies(line))

    selected: set[str] = set()
    stack = [name for name in wanted_names if name in definitions]
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        stack.extend(
            dependency for dependency in dependencies.get(name, ()) if dependency in definitions
        )

    return [
        line
        for line in lines
        if (label := _gaussian_definition_label(line)) is not None and label in selected
    ]


def _gaussian_definition_label(line: str) -> str | None:
    if "=" not in line:
        return None
    label = line.split("=", 1)[0].strip()
    if not label:
        return None
    return label.split("(", 1)[0].strip()


def _gaussian_definition_dependencies(line: str) -> tuple[str, ...]:
    if "=" not in line:
        return ()
    label = _gaussian_definition_label(line)
    expression = line.split("=", 1)[1]
    dependencies: list[str] = []
    seen: set[str] = set()
    for token in _GAUSSIAN_DEPENDENCY_RE.findall(expression):
        if token == label or token in seen:
            continue
        seen.add(token)
        dependencies.append(token)
    return tuple(dependencies)


def _topology_bonds(lines: list[str], *, natoms: int) -> tuple[tuple[int, int], ...]:
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    bond_lines = _subsection(topology, "BONDS")
    if not bond_lines or any(line.strip().upper() == "NONE" for line in bond_lines):
        raise GICForgeContractError("#TOPOLOGY contains no bonds")
    bonds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for line in bond_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond line: {line}") from exc
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond indexes: {line}")
        bond = tuple(sorted((i, j)))
        if bond not in seen:
            seen.add(bond)
            bonds.append(bond)
    return tuple(sorted(bonds))


def topology_bond_orders_from_lines(
    lines: list[str],
    *,
    natoms: int,
) -> dict[tuple[int, int], float]:
    """Read optional one-based #TOPOLOGY [BOND_ORDERS] rows."""
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        return {}
    bond_order_lines = _subsection(topology, "BOND_ORDERS")
    orders: dict[tuple[int, int], float] = {}
    for line in bond_order_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("(", " ").replace(")", " ").split()
        if len(parts) < 3:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
            value = float(parts[2])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond-order line: {line}") from exc
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid #TOPOLOGY bond-order indexes: {line}")
        orders[tuple(sorted((i, j)))] = value
    return orders


def topology_bond_order_components_from_lines(
    lines: list[str],
    *,
    natoms: int,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Read one-based canonical ``(sigma, pi, pi_pi)`` bond indices."""

    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        return {}
    rows = _subsection(topology, "BOND_ORDER_COMPONENTS")
    result: dict[tuple[int, int], tuple[float, float, float]] = {}
    for line in rows:
        upper = line.strip().upper()
        if upper == "NONE" or upper.startswith("COLUMNS "):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
            numeric = tuple(float(value) for value in parts[2:7])
        except ValueError as exc:
            raise GICForgeContractError(f"invalid bond-order-component line: {line}") from exc
        if len(numeric) == 5:
            _topology, multiplicity, sigma, pi, pi_pi = numeric
        else:
            multiplicity, sigma, pi, pi_pi = numeric
        if i == j or i < 1 or j < 1 or i > natoms or j > natoms:
            raise GICForgeContractError(f"invalid bond-order-component indexes: {line}")
        if min(multiplicity, sigma, pi, pi_pi) < 0.0 or not np.isclose(
            multiplicity, sigma + pi + pi_pi, rtol=1.0e-8, atol=1.0e-10
        ):
            raise GICForgeContractError(f"inconsistent bond-order components: {line}")
        result[tuple(sorted((i, j)))] = (sigma, pi, pi_pi)
    return result


def _topology_rings(lines: list[str], *, natoms: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    topology = section_content(lines, "TOPOLOGY")
    if not topology or not schema_line_supported(topology[0], SUPPORTED_TOPOLOGY_SCHEMAS):
        raise GICForgeContractError("missing valid #TOPOLOGY section")
    ring_lines = _subsection(topology, "RINGS")
    rings: list[tuple[int, tuple[int, ...]]] = []
    for line in ring_lines:
        if line.strip().upper() == "NONE":
            continue
        parts = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if not parts:
            continue
        try:
            ring_index = int(parts[0])
        except ValueError:
            continue
        atoms: list[int] = []
        reading_atoms = False
        for part in parts[1:]:
            token = part.strip()
            if token.upper().startswith("ATOMS="):
                reading_atoms = True
                token = token.split("=", 1)[1]
            elif "=" in token and reading_atoms:
                break
            if not reading_atoms or not token:
                continue
            try:
                atoms.append(int(token))
            except ValueError as exc:
                raise GICForgeContractError(f"invalid #TOPOLOGY ring line: {line}") from exc
        if len(atoms) < 3:
            continue
        if any(atom < 1 or atom > natoms for atom in atoms):
            raise GICForgeContractError(f"invalid #TOPOLOGY ring atom indexes: {line}")
        rings.append((ring_index, tuple(dict.fromkeys(atoms))))
    return tuple(rings)


def _primitive_candidates(
    bonds: tuple[tuple[int, int], ...],
    *,
    rings: tuple[tuple[int, tuple[int, ...]], ...] = (),
    coords: np.ndarray,
    natoms: int,
    atom_symbols: tuple[str, ...] = (),
    xh_stretch_policy: str = XH_STRETCH_POLICY_SYMMETRIZE,
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
    improper_dihedrals: bool = False,
    fragment_records: tuple[object, ...] = (),
    interaction_centers: object | None = None,
    pseudo_bonds: tuple[tuple[int, int], ...] = (),
    bond_orders: dict[tuple[int, int], float] | None = None,
) -> tuple[GICPrimitive, ...]:
    adjacency = _adjacency(bonds, natoms=natoms)
    pseudo_bond_set = {tuple(sorted(pair)) for pair in pseudo_bonds}
    pseudo_cycles = _pseudo_cycles_from_pseudo_bonds(
        bonds,
        pseudo_bonds=tuple(pseudo_bond_set),
        natoms=natoms,
    )
    pseudo_cycle_angle_set = _cycle_angle_triplet_set(pseudo_cycles)
    counters: dict[str, int] = {family: 0 for family in PRIMITIVE_FAMILY_ORDER}
    candidates: list[GICPrimitive] = []

    for stretch in generate_stretch_coordinates(
        bonds,
        atom_symbols=atom_symbols,
        xh_stretch_policy=xh_stretch_policy,
        local_xh_bonds=local_xh_bonds,
        local_xh_classes=local_xh_classes,
    ):
        candidates.append(
            _make_primitive(stretch.family, stretch.function, stretch.atoms, counters)
        )

    candidates.extend(
        _fragment_primitive_candidates(fragment_records, coords=coords, counters=counters)
    )
    candidates.extend(
        _interfragment_hbond_distance_candidates(
            fragment_records,
            atom_symbols=atom_symbols,
            coords=coords,
            bonds=bonds,
            counters=counters,
        )
    )
    intramolecular_hbond_contacts = _intramolecular_hbond_contacts(
        atom_symbols=atom_symbols,
        coords=coords,
        bonds=bonds,
    )
    candidates.extend(
        _hbond_distance_primitives(
            intramolecular_hbond_contacts,
            counters=counters,
            ref="INTRAMOLECULAR_HBOND",
        )
    )
    hbond_replaced_angles = _hbond_replaced_valence_angles(
        intramolecular_hbond_contacts,
        bonds=bonds,
        natoms=natoms,
    )
    candidates.extend(
        _interaction_center_primitive_candidates(interaction_centers, counters=counters)
    )

    linear_angle_keys: set[tuple[int, int, int]] = set()
    linear_angles: list[tuple[int, int, int]] = []
    for center in range(1, natoms + 1):
        for i, k in combinations(sorted(adjacency[center]), 2):
            if (i, center, k) in pseudo_cycle_angle_set:
                continue
            if _angle_key((i, center, k)) in hbond_replaced_angles:
                continue
            angle = _angle_value(coords, (i, center, k))
            if np.degrees(angle) >= LINEAR_ANGLE_DEGREES:
                linear_angle_keys.add(_angle_key((i, center, k)))
                linear_angles.append((i, center, k))
                candidates.append(
                    _make_primitive("LINEAR_BEND", "L", (i, center, k), counters, mode=-1)
                )
                candidates.append(
                    _make_primitive("LINEAR_BEND", "L", (i, center, k), counters, mode=-2)
                )
            else:
                family = (
                    "CYCLIC_BEND"
                    if _ring_index_for_atoms((i, center, k), rings) is not None
                    else "BEND"
                )
                candidates.append(_make_primitive(family, "A", (i, center, k), counters))

    # Pseudobonds augment connectivity but do not create chemical rings.
    # Artificial closure cycles therefore use the ordinary angle/torsion
    # candidates above and must not receive protected ring-puckering rows.

    seen_torsions: set[tuple[int, int, int, int]] = set()
    torsion_candidate = tuple[str, tuple[int, int, int, int], tuple[str, ...]]
    butterfly_torsions: list[torsion_candidate] = []
    condensed_torsions: list[torsion_candidate] = []
    ordinary_torsions: list[torsion_candidate] = []
    for j, k in bonds:
        for i in sorted(adjacency[j] - {k}):
            for ell in sorted(adjacency[k] - {j}):
                torsion = (i, j, k, ell)
                canonical = min(torsion, tuple(reversed(torsion)))
                if canonical in seen_torsions:
                    continue
                seen_torsions.add(canonical)
                if (
                    _angle_key((i, j, k)) in linear_angle_keys
                    or _angle_key((j, k, ell)) in linear_angle_keys
                ):
                    continue
                family = _torsion_family(canonical, rings)
                if family == "CYCLIC_TORSION":
                    continue
                if family == "BUTTERFLY":
                    butterfly_torsions.append((family, canonical, ()))
                elif family == "CONDENSED_RING_TORSION":
                    condensed_torsions.append((family, canonical, ()))
                else:
                    ordinary_torsions.append((family, canonical, ()))

    # A dihedral containing a linear triple has a singular derivative,
    # independently of whether either edge is covalent or a pseudobond.
    # Collapse the linear center and define the equivalent torsion across the
    # two endpoints, as in Gaussian DefRed's special linear-case construction.
    for left_endpoint, linear_center, right_endpoint in linear_angles:
        for left in sorted(adjacency[left_endpoint] - {linear_center}):
            for right in sorted(adjacency[right_endpoint] - {linear_center}):
                if len({left, left_endpoint, right_endpoint, right}) != 4:
                    continue
                torsion = (left, left_endpoint, right_endpoint, right)
                canonical = min(torsion, tuple(reversed(torsion)))
                if canonical in seen_torsions:
                    continue
                seen_torsions.add(canonical)
                ordinary_torsions.append(("TORSION", canonical, ()))

    for family, torsion, refs in butterfly_torsions:
        candidates.append(_make_primitive(family, "D", torsion, counters, refs=refs))
    candidates.extend(
        _ring_pucker_component_candidates(rings, counters=counters, bond_orders=bond_orders or {})
    )
    for family, torsion, refs in condensed_torsions:
        candidates.append(_make_primitive(family, "D", torsion, counters, refs=refs))
    for family, torsion, refs in ordinary_torsions:
        function = "RPCK" if refs else "D"
        candidates.append(_make_primitive(family, function, torsion, counters, refs=refs))

    cyclic_atoms = _cyclic_atom_set(rings)
    out_of_plane_family = "IMPROPER_DIHEDRAL" if improper_dihedrals else "OUT_OF_PLANE"
    out_of_plane_function = "IMPD" if improper_dihedrals else "U"
    for center in range(1, natoms + 1):
        neighbors = sorted(adjacency[center])
        if len(neighbors) < 3:
            continue
        for n1, n2, n3 in combinations(neighbors, 3):
            if {center, n1, n2, n3}.issubset(cyclic_atoms):
                continue
            candidates.append(
                _make_primitive(
                    out_of_plane_family,
                    out_of_plane_function,
                    (center, n1, n2, n3),
                    counters,
                )
            )

    return tuple(candidates)


def _torsion_family(
    atoms: tuple[int, int, int, int],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> str:
    if _butterfly_torsion(atoms, rings):
        return "BUTTERFLY"
    if _ring_index_for_atoms(atoms, rings) is not None:
        return "CYCLIC_TORSION"
    component = _ring_component_for_atoms(atoms, rings)
    if component is not None and len(component) > 1:
        return "CONDENSED_RING_TORSION"
    return "TORSION"


def _pseudo_cycles_from_pseudo_bonds(
    bonds: tuple[tuple[int, int], ...],
    *,
    pseudo_bonds: tuple[tuple[int, int], ...],
    natoms: int,
) -> tuple[tuple[int, ...], ...]:
    pseudo_set = {tuple(sorted(pair)) for pair in pseudo_bonds}
    if not pseudo_set:
        return ()
    graph_bonds = tuple(tuple(sorted(pair)) for pair in bonds)
    cycles: list[tuple[int, ...]] = []
    seen: set[frozenset[int]] = set()
    for left, right in sorted(pseudo_set):
        path_bonds = tuple(pair for pair in graph_bonds if pair != (left, right))
        path = _shortest_graph_path(left, right, _adjacency(path_bonds, natoms=natoms))
        if len(path) < 4:
            continue
        key = frozenset(path)
        if key in seen:
            continue
        seen.add(key)
        cycles.append(_canonical_cycle_order(path))
    return tuple(cycles)


def _canonical_cycle_order(cycle: tuple[int, ...]) -> tuple[int, ...]:
    if not cycle:
        return ()
    variants: list[tuple[int, ...]] = []
    ncycle = len(cycle)
    for base in (cycle, tuple(reversed(cycle))):
        for shift in range(ncycle):
            variants.append(base[shift:] + base[:shift])
    return min(variants)


def _cycle_angle_triplet_set(cycles: tuple[tuple[int, ...], ...]) -> set[tuple[int, int, int]]:
    triplets: set[tuple[int, int, int]] = set()
    for cycle in cycles:
        ncycle = len(cycle)
        for index, center in enumerate(cycle):
            left = cycle[(index - 1) % ncycle]
            right = cycle[(index + 1) % ncycle]
            triplets.add((left, center, right))
            triplets.add((right, center, left))
    return triplets


def _shortest_graph_path(
    start: int,
    stop: int,
    adjacency: dict[int, set[int]],
) -> tuple[int, ...]:
    queue: list[tuple[int, ...]] = [(int(start),)]
    seen = {int(start)}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == stop:
            return path
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(path + (neighbor,))
    return ()


def _pseudo_cycle_bend_component_candidates(
    cycles: tuple[tuple[int, ...], ...],
    *,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    for cycle in cycles:
        for terms in _cycle_angle_component_terms(cycle):
            candidates.append(
                _make_primitive(
                    "PSEUDO_CYCLE_BEND",
                    "RPCB",
                    tuple(cycle),
                    counters,
                    refs=tuple(
                        _encode_angle_component_term(coefficient, atoms)
                        for coefficient, atoms in terms
                    ),
                )
            )
    return tuple(candidates)


def _pseudo_cycle_torsion_component_candidates(
    cycles: tuple[tuple[int, ...], ...],
    *,
    counters: dict[str, int],
    pseudo_bonds: set[tuple[int, int]],
    bond_orders: dict[tuple[int, int], float],
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    effective_orders = dict(bond_orders)
    for pair in pseudo_bonds:
        effective_orders[tuple(sorted(pair))] = PSEUDO_BOND_EFFECTIVE_ORDER
    for cycle in cycles:
        for terms in _ring_pucker_component_terms(cycle, bond_orders=effective_orders):
            candidates.append(
                _make_primitive(
                    "PSEUDO_CYCLE_TORSION",
                    "RPCK",
                    tuple(cycle),
                    counters,
                    refs=tuple(
                        _encode_ring_pucker_term(coefficient, atoms) for coefficient, atoms in terms
                    ),
                )
            )
    return tuple(candidates)


def _cycle_angle_component_terms(
    cycle_atoms: tuple[int, ...],
) -> tuple[tuple[tuple[float, tuple[int, int, int]], ...], ...]:
    ncycle = len(cycle_atoms)
    if ncycle <= 3:
        return ()
    components: list[tuple[tuple[float, tuple[int, int, int]], ...]] = []
    for terms in _cyclic_component_coefficients(ncycle):
        components.append(
            tuple(
                (
                    coefficient,
                    (
                        cycle_atoms[(index - 1) % ncycle],
                        cycle_atoms[index],
                        cycle_atoms[(index + 1) % ncycle],
                    ),
                )
                for index, coefficient in enumerate(terms)
            )
        )
    return tuple(components)


def _cyclic_component_coefficients(ncycle: int) -> tuple[tuple[float, ...], ...]:
    basis = []
    for mode in range(1, ncycle):
        angle = 2.0 * np.pi * float(mode) * np.arange(ncycle, dtype=float) / float(ncycle)
        basis.append(np.cos(angle))
        basis.append(np.sin(angle))
    accepted: list[np.ndarray] = []
    ones = np.ones(ncycle, dtype=float)
    for vector in basis:
        residual = np.asarray(vector, dtype=float)
        residual = residual - float(np.dot(residual, ones) / np.dot(ones, ones)) * ones
        for previous in accepted:
            residual = residual - float(np.dot(residual, previous)) * previous
        norm = float(np.linalg.norm(residual))
        if norm <= 1.0e-10:
            continue
        accepted.append(residual / norm)
        if len(accepted) == ncycle - 3:
            break
    return tuple(tuple(float(value) for value in vector) for vector in accepted)


def _ring_pucker_component_candidates(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    counters: dict[str, int],
    bond_orders: dict[tuple[int, int], float],
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    for _ring_index, ring_atoms in rings:
        for terms in _ring_pucker_component_terms(ring_atoms, bond_orders=bond_orders):
            candidates.append(
                _make_primitive(
                    "RING_PUCKER_COMPONENT",
                    "RPCK",
                    tuple(ring_atoms),
                    counters,
                    refs=tuple(
                        _encode_ring_pucker_term(coefficient, atoms) for coefficient, atoms in terms
                    ),
                )
            )
    return tuple(candidates)


def _ring_pucker_component_terms(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float] | None = None,
) -> tuple[tuple[tuple[float, tuple[int, int, int, int]], ...], ...]:
    """Return ORACLE RPck linear combinations for one ordered ring."""
    ncyc = len(ring_atoms)
    if ncyc <= 3:
        return ()
    vnorm = float(np.sqrt(2.0 / float(ncyc)))
    vnorm1 = float(np.sqrt(1.0 / float(ncyc)))
    istart = ncyc
    components: list[tuple[tuple[float, tuple[int, int, int, int]], ...]] = []
    for ivar in range(1, ncyc - 2):
        even = ivar == 2 * (ivar // 2)
        terms: list[tuple[float, tuple[int, int, int, int]]] = []
        for iterm in range(1, ncyc + 1):
            iang1 = _cyclic_index(iterm + istart - 1, ncyc)
            iang2 = _cyclic_index(iterm + istart, ncyc)
            iang3 = _cyclic_index(iterm + istart + 1, ncyc)
            iang4 = _cyclic_index(iterm + istart + 2, ncyc)
            ivar1 = ivar
            if ivar == 1:
                ivar1 = 2
            elif ivar == 4:
                ivar1 = 3
            elif ivar in {5, 6}:
                ivar1 = 4
            value = np.pi * (2.0 * float(ivar1) * float(iterm - 1)) / float(ncyc)
            if even:
                coefficient = vnorm * float(np.sin(value))
            elif ivar < ncyc - 3:
                coefficient = vnorm * float(np.cos(value))
            else:
                coefficient = vnorm1 * float(np.cos(float(iterm - 1) * np.pi))
            if abs(coefficient) <= 1.0e-14:
                coefficient = 0.0
            terms.append(
                (
                    coefficient
                    * _ring_dihedral_flexibilities_one_based(
                        ring_atoms,
                        bond_orders=bond_orders or {},
                    )[iterm - 1],
                    (
                        ring_atoms[iang1],
                        ring_atoms[iang2],
                        ring_atoms[iang3],
                        ring_atoms[iang4],
                    ),
                )
            )
        components.append(_normalize_ring_pucker_terms(terms))
    return tuple(components)


def _ring_dihedral_flexibilities_one_based(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float],
    contrast_tolerance: float = 0.50,
) -> tuple[float, ...]:
    orders = _ring_dihedral_bond_orders_one_based(ring_atoms, bond_orders=bond_orders)
    finite = [order for order in orders if order is not None and order > 1.0e-12]
    if len(finite) != len(orders):
        return tuple(1.0 for _order in orders)
    reference = min(float(order) for order in finite)
    maximum = max(float(order) for order in finite)
    if reference <= 0.0 or maximum / reference <= 1.0 + float(contrast_tolerance):
        return tuple(1.0 for _order in orders)
    return tuple(float(np.sqrt(reference / float(order))) for order in finite)


def _ring_dihedral_bond_orders_one_based(
    ring_atoms: tuple[int, ...],
    *,
    bond_orders: dict[tuple[int, int], float],
) -> tuple[float | None, ...]:
    ncyc = len(ring_atoms)
    orders: list[float | None] = []
    for iterm in range(1, ncyc + 1):
        iang2 = _cyclic_index(iterm + ncyc, ncyc)
        iang3 = _cyclic_index(iterm + ncyc + 1, ncyc)
        orders.append(bond_orders.get(tuple(sorted((ring_atoms[iang2], ring_atoms[iang3])))))
    return tuple(orders)


def _ring_puckering_diagnostics(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    bond_orders: dict[tuple[int, int], float],
    bond_order_components: dict[tuple[int, int], tuple[float, float, float]] | None = None,
) -> tuple[str, ...]:
    lines: list[str] = []
    for ring_index, ring_atoms in rings:
        orders = _ring_dihedral_bond_orders_one_based(ring_atoms, bond_orders=bond_orders)
        flex = _ring_dihedral_flexibilities_one_based(ring_atoms, bond_orders=bond_orders)
        bonds = []
        for iterm in range(1, len(ring_atoms) + 1):
            iang2 = _cyclic_index(iterm + len(ring_atoms), len(ring_atoms))
            iang3 = _cyclic_index(iterm + len(ring_atoms) + 1, len(ring_atoms))
            bonds.append(f"{ring_atoms[iang2]}-{ring_atoms[iang3]}")
        order_text = ",".join("NA" if order is None else f"{float(order):.8g}" for order in orders)
        flex_text = ",".join(f"{value:.8g}" for value in flex)
        components = bond_order_components or {}
        component_values = [components.get(tuple(sorted(tuple(int(value) for value in bond.split("-"))))) for bond in bonds]
        pi_text = ",".join(
            "NA" if value is None else f"{value[1]:.8g}" for value in component_values
        )
        pi_pi_text = ",".join(
            "NA" if value is None else f"{value[2]:.8g}" for value in component_values
        )
        lines.append(
            f"RING {ring_index} ATOMS={','.join(str(atom) for atom in ring_atoms)} "
            f"CENTRAL_BONDS={','.join(bonds)} BOND_ORDERS={order_text} "
            f"PI_INDICES={pi_text} PI_PI_INDICES={pi_pi_text} FLEX={flex_text}"
        )
    return tuple(lines)


def _normalize_ring_pucker_terms(
    terms: list[tuple[float, tuple[int, int, int, int]]],
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    norm = float(np.sqrt(sum(float(coefficient) ** 2 for coefficient, _atoms in terms)))
    if norm <= 1.0e-14:
        return tuple(terms)
    return tuple((float(coefficient) / norm, atoms) for coefficient, atoms in terms)


def _cyclic_index(index_1based: int, ncyc: int) -> int:
    while index_1based > ncyc:
        index_1based -= ncyc
    while index_1based <= 0:
        index_1based += ncyc
    return index_1based - 1


def _encode_ring_pucker_term(
    coefficient: float,
    atoms: tuple[int, int, int, int],
) -> str:
    atom_text = "-".join(str(atom) for atom in atoms)
    return f"{float(coefficient):.17g}:{atom_text}"


def _encode_angle_component_term(
    coefficient: float,
    atoms: tuple[int, int, int],
) -> str:
    atom_text = "-".join(str(atom) for atom in atoms)
    return f"{float(coefficient):.17g}:{atom_text}"


def _angle_component_terms_from_refs(
    primitive: GICPrimitive,
) -> tuple[tuple[float, tuple[int, int, int]], ...]:
    terms: list[tuple[float, tuple[int, int, int]]] = []
    for ref in primitive.refs:
        if ":" not in ref:
            raise GICForgeContractError(
                f"invalid RPCB term {ref!r} in primitive {primitive.identifier}"
            )
        coefficient_text, atom_text = ref.split(":", 1)
        try:
            coefficient = float(coefficient_text)
            atoms = tuple(int(atom) for atom in atom_text.split("-") if atom)
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid RPCB term {ref!r} in primitive {primitive.identifier}"
            ) from exc
        if len(atoms) != 3:
            raise GICForgeContractError(
                f"invalid RPCB angle term {ref!r} in primitive {primitive.identifier}"
            )
        terms.append((coefficient, atoms))
    return tuple(terms)


def _ring_pucker_terms_from_refs(
    primitive: GICPrimitive,
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    terms: list[tuple[float, tuple[int, int, int, int]]] = []
    for ref in primitive.refs:
        if ":" not in ref:
            raise GICForgeContractError(
                f"invalid RPck term {ref!r} in primitive {primitive.identifier}"
            )
        coefficient_text, atom_text = ref.split(":", 1)
        try:
            coefficient = float(coefficient_text)
            atoms = tuple(int(atom) for atom in atom_text.split("-") if atom)
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid RPck term {ref!r} in primitive {primitive.identifier}"
            ) from exc
        if len(atoms) != 4:
            raise GICForgeContractError(
                f"invalid RPck dihedral term {ref!r} in primitive {primitive.identifier}"
            )
        terms.append((coefficient, atoms))
    return tuple(terms)


def _ring_index_for_atoms(
    atoms: tuple[int, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> int | None:
    atom_set = set(atoms)
    best_index: int | None = None
    best_size: int | None = None
    for ring_index, ring_atoms in rings:
        ring_set = set(ring_atoms)
        if not atom_set.issubset(ring_set):
            continue
        ring_size = len(ring_atoms)
        if (
            best_size is None
            or ring_size < best_size
            or (ring_size == best_size and ring_index < (best_index or ring_index))
        ):
            best_index = ring_index
            best_size = ring_size
    return best_index


def _ring_component_for_atoms(
    atoms: tuple[int, ...],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[int, ...] | None:
    atom_set = set(atoms)
    for component in _ring_components(rings):
        component_atoms: set[int] = set()
        for ring_index in component:
            component_atoms.update(dict(rings)[ring_index])
        if atom_set.issubset(component_atoms):
            return component
    return None


def _ring_components(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    if not rings:
        return ()
    ring_ids = tuple(ring_index for ring_index, _atoms in rings)
    bond_to_rings = _ring_bond_to_rings(rings)
    neighbors: dict[int, set[int]] = {ring_index: set() for ring_index in ring_ids}
    for ring_indices in bond_to_rings.values():
        if len(ring_indices) < 2:
            continue
        for left, right in combinations(ring_indices, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for ring_index in ring_ids:
        if ring_index in seen:
            continue
        stack = [ring_index]
        seen.add(ring_index)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _butterfly_torsion(
    atoms: tuple[int, int, int, int],
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> bool:
    if not rings:
        return False
    i, j, k, ell = atoms
    central_bond = tuple(sorted((j, k)))
    ring_by_index = {ring_index: set(ring_atoms) for ring_index, ring_atoms in rings}
    sharing_rings = _ring_bond_to_rings(rings).get(central_bond, ())
    if len(sharing_rings) < 2:
        return False
    for left_index, right_index in combinations(sharing_rings, 2):
        left_only = ring_by_index[left_index] - ring_by_index[right_index]
        right_only = ring_by_index[right_index] - ring_by_index[left_index]
        if (i in left_only and ell in right_only) or (i in right_only and ell in left_only):
            return True
    return False


def _ring_bond_to_rings(
    rings: tuple[tuple[int, tuple[int, ...]], ...],
) -> dict[tuple[int, int], tuple[int, ...]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    for ring_index, ring_atoms in rings:
        if len(ring_atoms) < 2:
            continue
        for left, right in zip(ring_atoms, ring_atoms[1:] + ring_atoms[:1]):
            bond = tuple(sorted((left, right)))
            mapping.setdefault(bond, []).append(ring_index)
    return {bond: tuple(indices) for bond, indices in mapping.items()}


def _make_primitive(
    family: str,
    function: str,
    atoms: tuple[int, ...],
    counters: dict[str, int],
    *,
    mode: int = 0,
    ref_atoms: tuple[int, ...] = (),
    refs: tuple[str, ...] = (),
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
) -> GICPrimitive:
    counters[family] += 1
    prefix = primitive_prefix(family)
    serial = sum(counters.values())
    return GICPrimitive(
        identifier=f"P{serial:03d}",
        name=f"{prefix}{counters[family]:04d}",
        family=family,
        function=function,
        atoms=tuple(int(atom) for atom in atoms),
        mode=int(mode),
        ref_atoms=tuple(int(atom) for atom in ref_atoms),
        refs=tuple(str(ref) for ref in refs),
        frame_atoms=tuple(int(atom) for atom in frame_atoms),
        ref_frame_atoms=tuple(int(atom) for atom in ref_frame_atoms),
    )


def _apply_semantic_contract_to_candidates(
    candidates: tuple[GICPrimitive, ...],
    contract: SemanticContract,
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> tuple[tuple[GICPrimitive, ...], SemanticContract, tuple[str, ...]]:
    diagnostics = list(contract.diagnostics)
    if not contract.protect_coordinates:
        return candidates, contract, tuple(diagnostics)

    generated: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    user_primitives: list[GICPrimitive] = []
    primitive_substitutions: dict[CoordinateSignature, SemanticCoordinate] = {}
    auto_semantic_requests = [
        coordinate
        for coordinate in contract.protect_coordinates
        if coordinate.semantic_type not in _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS
    ]

    for serial, coordinate in enumerate(contract.protect_coordinates, start=1):
        primitive = _semantic_primitive_from_coordinate(coordinate, serial=serial)
        if primitive is None:
            diagnostics.append(
                f"SEMANTIC_STORED_ONLY ID={coordinate.identifier} TYPE={coordinate.semantic_type}"
            )
            continue
        primitive_substitutions[coordinate.signature] = coordinate
        user_primitives.append(primitive)
        generated[coordinate.identifier] = ((primitive.identifier,), ())
        diagnostics.append(
            f"SEMANTIC_USER_PROTECT ID={coordinate.identifier} PRIMITIVE={primitive.identifier}"
        )

    filtered_candidates: list[GICPrimitive] = []
    matched_auto_requests: set[str] = set()
    for primitive in candidates:
        signature = _semantic_signature_for_gic_primitive(primitive)
        substitute = primitive_substitutions.get(signature)
        if substitute is not None:
            diagnostics.append(
                "SEMANTIC_SUBSTITUTION "
                f"USER={substitute.identifier} REPLACES={primitive.identifier}"
            )
            continue
        matched_coordinate = _semantic_auto_request_for_primitive(
            primitive,
            auto_semantic_requests,
        )
        if matched_coordinate is not None:
            primitive = replace(
                primitive,
                provenance=USER_PROVENANCE,
                semantic_id=matched_coordinate.identifier,
                semantic_type=matched_coordinate.semantic_type,
            )
            matched_auto_requests.add(matched_coordinate.identifier)
            generated[matched_coordinate.identifier] = (
                generated.get(matched_coordinate.identifier, ((), ()))[0]
                + (primitive.identifier,),
                (),
            )
            diagnostics.append(
                "SEMANTIC_AUTO_PROTECT "
                f"ID={matched_coordinate.identifier} PRIMITIVE={primitive.identifier}"
            )
        filtered_candidates.append(primitive)

    for coordinate in auto_semantic_requests:
        if coordinate.identifier not in matched_auto_requests:
            diagnostics.append(
                "SEMANTIC_PENDING_GENERATOR "
                f"ID={coordinate.identifier} TYPE={coordinate.semantic_type}"
            )

    diagnostics.extend(
        _semantic_partial_orbit_diagnostics(
            contract.protect_coordinates,
            symmetry_operations=symmetry_operations,
        )
    )
    return (
        tuple(user_primitives + filtered_candidates),
        contract.with_coordinate_generation(generated),
        tuple(diagnostics),
    )


_SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS = {
    "DISTANCE": ("STRETCH", "R"),
    "ANGLE": ("BEND", "A"),
    "TORSION": ("TORSION", "D"),
    "OUT_OF_PLANE": ("OUT_OF_PLANE", "U"),
}


def _semantic_primitive_from_coordinate(
    coordinate: SemanticCoordinate,
    *,
    serial: int,
) -> GICPrimitive | None:
    mapping = _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS.get(coordinate.semantic_type)
    if mapping is None:
        return None
    family, function = mapping
    try:
        atoms = tuple(int(argument) for argument in coordinate.arguments)
    except ValueError as exc:
        raise GICForgeContractError(
            f"semantic {coordinate.semantic_type} needs atom-index arguments"
        ) from exc
    return GICPrimitive(
        identifier=f"PU{serial:03d}",
        name=_semantic_user_coordinate_name(coordinate.identifier, serial=serial),
        family=family,
        function=function,
        atoms=atoms,
        provenance=USER_PROVENANCE,
        semantic_id=coordinate.identifier,
        semantic_type=coordinate.semantic_type,
    )


def _semantic_user_coordinate_name(identifier: str, *, serial: int) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", identifier)
    if not text:
        text = "Coord"
    return f"Usr{text[:18]}{serial:03d}"


def _semantic_signature_for_gic_primitive(primitive: GICPrimitive) -> CoordinateSignature:
    return semantic_signature_for_primitive_like(
        primitive.function,
        primitive.atoms,
        mode=primitive.mode,
        ref_atoms=primitive.ref_atoms,
    )


def _semantic_auto_request_for_primitive(
    primitive: GICPrimitive,
    coordinates: list[SemanticCoordinate],
) -> SemanticCoordinate | None:
    for coordinate in coordinates:
        if coordinate.semantic_type == "RING_PUCKERING":
            if (
                primitive.family == "RING_PUCKER_COMPONENT"
                and set(primitive.atoms) == {int(atom) for atom in coordinate.arguments}
            ):
                return coordinate
        elif coordinate.semantic_type == "BUTTERFLY":
            if (
                primitive.family == "BUTTERFLY"
                and set(primitive.atoms) == {int(atom) for atom in coordinate.arguments}
            ):
                return coordinate
    return None


def _semantic_partial_orbit_diagnostics(
    coordinates: tuple[SemanticCoordinate, ...],
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> tuple[str, ...]:
    if len(symmetry_operations) <= 1:
        return ()
    user_signatures = {coordinate.signature for coordinate in coordinates}
    diagnostics: list[str] = []
    for coordinate in coordinates:
        mapping = _SEMANTIC_PRIMITIVE_PROTECT_FUNCTIONS.get(coordinate.semantic_type)
        if mapping is None:
            continue
        _family, function = mapping
        atoms = tuple(int(argument) for argument in coordinate.arguments)
        orbit: set[CoordinateSignature] = set()
        for operation in symmetry_operations:
            mapped = tuple(operation.permutation[atom - 1] for atom in atoms)
            orbit.add(semantic_signature_for_primitive_like(function, mapped))
        if len(orbit) > 1 and not orbit.issubset(user_signatures):
            diagnostics.append(
                "SEMANTIC_PARTIAL_ORBIT "
                f"ID={coordinate.identifier} SIZE={len(orbit)} "
                "FALLBACK=LOCAL_SALC"
            )
    return tuple(diagnostics)


def _select_ranked_primitives(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], int]:
    selected, rank, _ = _select_ranked_primitives_with_diagnostics(
        candidates,
        coords,
        target_rank=target_rank,
        rank_tolerance=rank_tolerance,
    )
    return selected, rank


def _select_ranked_primitives_with_diagnostics(
    candidates: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    rank_tolerance: float,
) -> tuple[tuple[GICPrimitive, ...], int, GICReductionDiagnostics]:
    skipped_singular: list[str] = []
    skipped_dependent: list[str] = []
    skipped_singular_details: list[str] = []
    skipped_dependent_details: list[str] = []
    if target_rank == 0:
        return (
            (),
            0,
            GICReductionDiagnostics(
                rank_method=RANK_METHOD,
                reduction_policy=REDUCTION_POLICY,
            ),
        )
    selected: list[GICPrimitive] = []
    basis: list[np.ndarray] = []
    rank = 0

    user_candidates = [
        primitive for primitive in candidates if _is_user_protected_primitive(primitive)
    ]
    pseudo_cycle_candidates = [
        primitive
        for primitive in candidates
        if not _is_user_protected_primitive(primitive)
        and _is_pseudo_cycle_primitive(primitive)
    ]
    special_candidates = [
        primitive
        for primitive in candidates
        if _is_special_primitive(primitive)
        and not _is_user_protected_primitive(primitive)
        and not _is_pseudo_cycle_primitive(primitive)
    ]
    ordinary_candidates = [
        primitive
        for primitive in candidates
        if not _is_user_protected_primitive(primitive)
        and not _is_special_primitive(primitive)
        and not _is_pseudo_cycle_primitive(primitive)
    ]
    ordinary_candidates = list(
        _tricoordinated_ordinary_candidate_order(tuple(ordinary_candidates))
    )
    def select_candidates(
        group: tuple[GICPrimitive, ...],
        *,
        protected: bool,
    ) -> tuple[bool, int]:
        nonlocal rank
        for index, primitive in enumerate(group):
            if rank == target_rank:
                if protected:
                    singular, dependent, singular_details, dependent_details = (
                        _raise_if_remaining_special_independent(
                            tuple(group[index:]),
                            coords,
                            basis,
                            rank,
                            rank_tolerance=rank_tolerance,
                        )
                    )
                    skipped_singular.extend(singular)
                    skipped_dependent.extend(dependent)
                    skipped_singular_details.extend(singular_details)
                    skipped_dependent_details.extend(dependent_details)
                    return True, rank
                return False, rank
            rank, status = _try_select_ranked_primitive(
                primitive,
                coords,
                selected,
                basis,
                rank,
                rank_tolerance=rank_tolerance,
            )
            _record_skip(
                primitive,
                status,
                skipped_singular,
                skipped_dependent,
                skipped_singular_details,
                skipped_dependent_details,
            )
        return False, rank

    should_return, rank = select_candidates(tuple(user_candidates), protected=True)
    if should_return:
        return (
            tuple(selected),
            rank,
            _make_reduction_diagnostics(
                selected,
                skipped_singular=skipped_singular,
                skipped_dependent=skipped_dependent,
                skipped_singular_details=skipped_singular_details,
                skipped_dependent_details=skipped_dependent_details,
            ),
        )

    should_return, rank = select_candidates(tuple(pseudo_cycle_candidates), protected=True)
    if should_return:
        return (
            tuple(selected),
            rank,
            _make_reduction_diagnostics(
                selected,
                skipped_singular=skipped_singular,
                skipped_dependent=skipped_dependent,
                skipped_singular_details=skipped_singular_details,
                skipped_dependent_details=skipped_dependent_details,
            ),
        )

    has_tric_fragment_block = not pseudo_cycle_candidates and any(
        primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        for primitive in special_candidates
    )
    if has_tric_fragment_block:
        # A fragmented coordinate chart is the direct sum of complete
        # intrafragment SONIC spaces and the relative rigid-body TRIC block.
        # Protecting intermolecular rows before ordinary rows can otherwise
        # replace chemically meaningful intrafragment coordinates while still
        # producing a formally full-rank but poorly conditioned basis.
        select_candidates(tuple(ordinary_candidates), protected=False)
        tric_candidates = tuple(
            primitive
            for primitive in special_candidates
            if primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}
        )
        semantic_interfragment_candidates = tuple(
            primitive for primitive in special_candidates if primitive not in tric_candidates
        )
        projected_distance_candidates = tuple(
            primitive
            for primitive in semantic_interfragment_candidates
            if primitive.family == "FRAG_DISTANCE"
        )
        should_return, rank = select_candidates(tric_candidates, protected=True)
        if should_return:
            singular, dependent, singular_details, dependent_details = (
                _raise_if_remaining_special_independent(
                    projected_distance_candidates,
                    coords,
                    basis,
                    rank,
                    rank_tolerance=rank_tolerance,
                )
            )
            skipped_singular.extend(singular)
            skipped_dependent.extend(dependent)
            skipped_singular_details.extend(singular_details)
            skipped_dependent_details.extend(dependent_details)
            return (
                tuple(selected),
                rank,
                _make_reduction_diagnostics(
                    selected,
                    skipped_singular=skipped_singular,
                    skipped_dependent=skipped_dependent,
                    skipped_singular_details=skipped_singular_details,
                    skipped_dependent_details=skipped_dependent_details,
                ),
            )
        if rank == target_rank and projected_distance_candidates:
            singular, dependent, singular_details, dependent_details = (
                _raise_if_remaining_special_independent(
                    projected_distance_candidates,
                    coords,
                    basis,
                    rank,
                    rank_tolerance=rank_tolerance,
                )
            )
            skipped_singular.extend(singular)
            skipped_dependent.extend(dependent)
            skipped_singular_details.extend(singular_details)
            skipped_dependent_details.extend(dependent_details)
        select_candidates(semantic_interfragment_candidates, protected=False)
    elif pseudo_cycle_candidates:
        select_candidates(tuple(ordinary_candidates), protected=False)
        select_candidates(tuple(special_candidates), protected=False)
    else:
        should_return, rank = select_candidates(tuple(special_candidates), protected=True)
        if should_return:
            return (
                tuple(selected),
                rank,
                _make_reduction_diagnostics(
                    selected,
                    skipped_singular=skipped_singular,
                    skipped_dependent=skipped_dependent,
                    skipped_singular_details=skipped_singular_details,
                    skipped_dependent_details=skipped_dependent_details,
                ),
            )
        select_candidates(tuple(ordinary_candidates), protected=False)
    return (
        tuple(selected),
        rank,
        _make_reduction_diagnostics(
            selected,
            skipped_singular=skipped_singular,
            skipped_dependent=skipped_dependent,
            skipped_singular_details=skipped_singular_details,
            skipped_dependent_details=skipped_dependent_details,
        ),
    )


def _tricoordinated_ordinary_candidate_order(
    candidates: tuple[GICPrimitive, ...],
) -> tuple[GICPrimitive, ...]:
    """Keep two valence-angle rows and U at every tricoordinated center.

    This is the non-symmetrized reduction corresponding to Merlino
    ``C2V3At`` plus ``MkGNCO``: the three pair angles contain only two local
    deformation coordinates, while the genuine out-of-plane primitive U is
    retained separately.  The rule is topological and does not depend on a
    planarity threshold.  Improper dihedrals enter only when explicitly
    requested for Gaussian compatibility.
    """

    out_of_plane_groups: dict[int, list[GICPrimitive]] = {}
    for primitive in candidates:
        if (
            primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"}
            and len(primitive.atoms) == 4
        ):
            out_of_plane_groups.setdefault(primitive.atoms[0], []).append(primitive)
    out_of_plane_by_center = {
        center: rows[0] for center, rows in out_of_plane_groups.items() if len(rows) == 1
    }
    bend_count: dict[int, int] = {}
    inserted: set[int] = set()
    ordered: list[GICPrimitive] = []
    for primitive in candidates:
        if primitive in out_of_plane_by_center.values():
            continue
        if primitive.function == "A" and len(primitive.atoms) == 3:
            center = primitive.atoms[1]
            bend_count[center] = bend_count.get(center, 0) + 1
            if bend_count[center] == 3 and center in out_of_plane_by_center:
                ordered.append(out_of_plane_by_center[center])
                inserted.add(center)
        ordered.append(primitive)
    ordered.extend(
        primitive
        for center, primitive in out_of_plane_by_center.items()
        if center not in inserted
    )
    return tuple(ordered)


def _is_special_primitive(primitive: GICPrimitive) -> bool:
    return primitive.reduction_class == SPECIAL_REDUCTION_CLASS or _is_user_protected_primitive(
        primitive
    )


def _is_user_protected_primitive(primitive: GICPrimitive) -> bool:
    return primitive.provenance.upper() == USER_PROVENANCE


def _is_pseudo_cycle_primitive(primitive: GICPrimitive) -> bool:
    return primitive.family in {"PSEUDO_CYCLE_BEND", "PSEUDO_CYCLE_TORSION"}


def _protected_gic_count(primitives: tuple[GICPrimitive, ...]) -> int:
    return sum(
        1
        for primitive in primitives
        if _is_special_primitive(primitive) or _is_pseudo_cycle_primitive(primitive)
    )


def _skipped_singular_count(definition: GICDefinition) -> int:
    if definition.reduction_diagnostics is None:
        return 0
    return len(definition.reduction_diagnostics.skipped_singular)


def _skipped_dependent_count(definition: GICDefinition) -> int:
    if definition.reduction_diagnostics is None:
        return 0
    return len(definition.reduction_diagnostics.skipped_dependent)


def _reduction_diagnostics_lines(definition: GICDefinition) -> list[str]:
    diagnostics = definition.reduction_diagnostics or GICReductionDiagnostics(
        rank_method=RANK_METHOD,
        reduction_policy=REDUCTION_POLICY,
        selected=tuple(primitive.identifier for primitive in definition.primitives),
    )
    return [
        f"RANK_METHOD {diagnostics.rank_method}",
        f"REDUCTION_POLICY {diagnostics.reduction_policy}",
        f"SELECTED {_csv_or_none(diagnostics.selected)}",
        f"SELECTED_BY_FAMILY {_csv_or_none(diagnostics.selected_by_family)}",
        f"SKIPPED_SINGULAR {_csv_or_none(diagnostics.skipped_singular)}",
        f"SKIPPED_DEPENDENT {_csv_or_none(diagnostics.skipped_dependent)}",
        f"SKIPPED_SINGULAR_DETAILS {_csv_or_none(diagnostics.skipped_singular_details)}",
        f"SKIPPED_DEPENDENT_DETAILS {_csv_or_none(diagnostics.skipped_dependent_details)}",
    ]


def _symmetry_diagnostics_lines(definition: GICDefinition) -> list[str]:
    diagnostics = definition.symmetry_diagnostics or _empty_symmetry_diagnostics(
        definition.point_group,
        requested=definition.symmetrize,
    )
    lines = [
        f"METHOD {diagnostics.method}",
        f"POLICY {diagnostics.policy}",
        f"STATUS {diagnostics.status}",
        f"POINT_GROUP {diagnostics.point_group}",
        f"SYMMETRY_GROUP {diagnostics.symmetry_group}",
        f"TOTAL_SYMMETRIC_IRREP {diagnostics.total_symmetric_irrep}",
        f"TOTAL_SYMMETRIC_GICS {_csv_or_none(diagnostics.total_symmetric_gics)}",
        f"SIGN_GAUGE_POLICY {diagnostics.sign_gauge_policy}",
        f"PATH_GAUGE_POLICY {diagnostics.path_gauge_policy}",
        f"PATH_OVERLAP_WARNING_THRESHOLD {diagnostics.path_overlap_warning_threshold:.12g}",
        f"OPERATION_TOLERANCE_ANGSTROM {diagnostics.operation_tolerance_angstrom:.12g}",
        f"MAX_OPERATION_RESIDUAL_ANGSTROM {diagnostics.max_operation_residual_angstrom:.12g}",
        f"MIN_OPERATION_MARGIN_ANGSTROM {diagnostics.min_operation_margin_angstrom:.12g}",
        f"NEAR_THRESHOLD_OPERATIONS {_csv_or_none(diagnostics.near_threshold_operations)}",
        f"GROUP_COUNT {len(diagnostics.groups)}",
        f"SYMMETRIZED_GIC_COUNT {_symmetrized_gic_count(diagnostics)}",
    ]
    for index, group in enumerate(diagnostics.groups, start=1):
        lines.append(
            f"GROUP {index} BLOCK={group.block} FAMILY={group.family} "
            f"SIGNATURE={group.signature} "
            f"SOURCES={_csv_or_none(group.source_gics)} "
            f"OUTPUTS={_csv_or_none(group.output_gics)}"
        )
    return lines


def _empty_symmetry_diagnostics(
    point_group: str,
    *,
    requested: bool,
) -> GICSymmetrizationDiagnostics:
    return GICSymmetrizationDiagnostics(
        method="NONE",
        policy=SYMMETRIZATION_POLICY,
        status="NOT_REQUESTED" if not requested else "NO_ELIGIBLE_GROUPS",
        point_group=point_group,
        symmetry_group=point_group,
        total_symmetric_irrep=total_symmetric_irrep(point_group),
        total_symmetric_gics=(),
    )


def _symmetrized_gic_count(diagnostics: GICSymmetrizationDiagnostics) -> int:
    return sum(len(group.output_gics) for group in diagnostics.groups)


def _make_reduction_diagnostics(
    selected: list[GICPrimitive],
    *,
    skipped_singular: list[str],
    skipped_dependent: list[str],
    skipped_singular_details: list[str] | None = None,
    skipped_dependent_details: list[str] | None = None,
) -> GICReductionDiagnostics:
    return GICReductionDiagnostics(
        rank_method=RANK_METHOD,
        reduction_policy=REDUCTION_POLICY,
        selected=tuple(primitive.identifier for primitive in selected),
        skipped_singular=tuple(skipped_singular),
        skipped_dependent=tuple(skipped_dependent),
        selected_by_family=_family_count_tokens(selected),
        skipped_singular_details=tuple(skipped_singular_details or ()),
        skipped_dependent_details=tuple(skipped_dependent_details or ()),
    )


def _record_skip(
    primitive: GICPrimitive,
    status: str,
    skipped_singular: list[str],
    skipped_dependent: list[str],
    skipped_singular_details: list[str],
    skipped_dependent_details: list[str],
) -> None:
    if status == "singular":
        skipped_singular.append(primitive.identifier)
        skipped_singular_details.append(_primitive_diagnostic_token(primitive))
    elif status == "dependent":
        skipped_dependent.append(primitive.identifier)
        skipped_dependent_details.append(_primitive_diagnostic_token(primitive))


def _family_count_tokens(primitives: list[GICPrimitive]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for primitive in primitives:
        counts[primitive.family] = counts.get(primitive.family, 0) + 1
    return tuple(f"{family}:{count}" for family, count in sorted(counts.items()))


def _primitive_diagnostic_token(primitive: GICPrimitive) -> str:
    return f"{primitive.identifier}:{primitive.family}:{primitive.name}"


def _apply_local_symmetrization(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    atom_symbols: tuple[str, ...],
    point_group: str,
    requested: bool,
    symmetry_operations: tuple[GICPointGroupOperation, ...] = (),
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics]:
    def with_operation_margins(
        result: tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics],
    ) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics]:
        projected_rank = _frozen_gic_b_matrix_rank(
            result[0],
            primitive_by_id={primitive.identifier: primitive for primitive in primitives},
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        has_native_ring_out_of_plane = any(
            primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "U"
            for primitive in primitives
        )
        if (
            has_native_ring_out_of_plane
            and projected_rank is not None
            and projected_rank < len(result[0])
        ):
            raise GICForgeContractError(
                "symmetry projection produced a rank-deficient global SONIC B matrix: "
                f"rank {projected_rank} for {len(result[0])} coordinates"
            )
        return (
            result[0],
            _diagnostics_with_operation_margins(
                result[1],
                operations=symmetry_operations,
                reference_coordinates_angstrom=reference_coordinates_angstrom,
            ),
        )

    if not requested:
        return gics, _empty_symmetry_diagnostics(point_group, requested=False)

    projected = _apply_point_group_projector(
        gics,
        primitives,
        point_group=point_group,
        symmetry_operations=symmetry_operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
    )
    if projected is not None:
        return with_operation_margins(projected)

    source_groups = _local_symmetry_groups(gics, primitives, atom_symbols=atom_symbols)
    if not source_groups:
        prefixed_gics = _prefix_symmetrized_singletons(gics, point_group=point_group)
        return with_operation_margins((
            prefixed_gics,
            GICSymmetrizationDiagnostics(
                method=LOCAL_SYMMETRIZATION_METHOD,
                policy=SYMMETRIZATION_POLICY,
                status="NO_ELIGIBLE_GROUPS",
                point_group=point_group,
                symmetry_group=point_group,
                total_symmetric_irrep=total_symmetric_irrep(point_group),
                total_symmetric_gics=tuple(
                    gic.name
                    for gic in prefixed_gics
                    if is_total_symmetric_irrep(point_group, gic.irrep)
                ),
            ),
        ))

    groups_by_first = {group[0].identifier: (key, group) for key, group in source_groups.items()}
    grouped_ids = {gic.identifier for group in source_groups.values() for gic in group}
    name_counters: dict[tuple[str, str, str], int] = {}
    output: list[FrozenGIC] = []
    diagnostics: list[GICSymmetrizedGroup] = []

    for gic in gics:
        if gic.identifier in groups_by_first:
            key, group = groups_by_first[gic.identifier]
            new_gics = _symmetrized_group_gics(
                key,
                group,
                first_index=len(output) + 1,
                name_counters=name_counters,
                point_group=point_group,
            )
            output.extend(new_gics)
            diagnostics.append(
                GICSymmetrizedGroup(
                    block=key[0],
                    family=key[1],
                    signature=key[2],
                    source_gics=tuple(source.name for source in group),
                    output_gics=tuple(new_gic.name for new_gic in new_gics),
                )
            )
            continue
        if gic.identifier in grouped_ids:
            continue
        output.append(
            _renumber_frozen_gic(
                _prefix_symmetrized_gic(gic, point_group=point_group),
                len(output) + 1,
            )
        )

    output_tuple = tuple(output)
    return with_operation_margins((
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=LOCAL_SYMMETRIZATION_METHOD,
            policy=SYMMETRIZATION_POLICY,
            status="APPLIED",
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=tuple(diagnostics),
        ),
    ))


def _frozen_gic_b_matrix_rank(
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> int | None:
    """Return the Cartesian rank of a complete frozen SONIC set when evaluable."""
    if reference_coordinates_angstrom is None:
        return None
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    rows: list[np.ndarray] = []
    for gic in gics:
        row = np.zeros(coords.size, dtype=float)
        for primitive_id, coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        ):
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                return None
            try:
                row += float(coefficient) * _analytic_b_row(primitive, coords)
            except (FloatingPointError, GICForgeContractError, ValueError):
                return None
        rows.append(row)
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=1.0e-8))


def _apply_point_group_projector(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics] | None:
    operations = _valid_projector_operations(symmetry_operations)
    if len(operations) <= 1 or point_group.upper() in {"C1", "UNKNOWN"}:
        return None
    group_key = point_group.strip().upper()
    coords_for_global = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if reference_coordinates_angstrom is not None
        else None
    )
    has_local_xh = any(gic.family == "LOCAL_XH_STRETCH" for gic in gics)
    if group_key in {"D5H", "D5D"} and coords_for_global is not None and not has_local_xh:
        projected = _apply_rank_revealing_global_projector(
            gics,
            primitives,
            point_group=point_group,
            operations=operations,
            reference_coordinates_angstrom=coords_for_global,
        )
        if projected is not None:
            return projected
    has_native_ring_out_of_plane = any(
        primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "U"
        for primitive in primitives
    )
    use_global_b_selection = has_native_ring_out_of_plane or group_key in {
        "T",
        "TD",
        "O",
        "OH",
        "I",
        "IH",
    }
    coords = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if use_global_b_selection and reference_coordinates_angstrom is not None
        else None
    )

    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    blocks: dict[tuple[str, str], list[FrozenGIC]] = {}
    for gic in gics:
        key = (primitive_symmetry_block(gic.family), gic.family)
        blocks.setdefault(key, []).append(gic)

    output: list[FrozenGIC] = []
    diagnostics: list[GICSymmetrizedGroup] = []
    name_counters: dict[tuple[str, str], int] = {}
    global_b_basis: tuple[np.ndarray, ...] = ()
    for key, block_gics in sorted(
        blocks.items(),
        key=lambda item: _projector_block_sort_key(
            item,
            protect_special=use_global_b_selection,
        ),
    ):
        if key[1] == "LOCAL_XH_STRETCH":
            block_output = tuple(
                _renumber_frozen_gic(gic, len(output) + offset)
                for offset, gic in enumerate(block_gics, start=1)
            )
            output.extend(block_output)
            diagnostics.append(
                GICSymmetrizedGroup(
                    block=key[0],
                    family=key[1],
                    signature="UNSYMMETRIZED_LOCAL_XH",
                    source_gics=tuple(gic.name for gic in block_gics),
                    output_gics=tuple(gic.name for gic in block_output),
                )
            )
            continue
        ring_pucker_coords = (
            np.asarray(reference_coordinates_angstrom, dtype=float)
            if key[1] == "RING_PUCKER_COMPONENT" and reference_coordinates_angstrom is not None
            else None
        )
        projected = _project_gic_block(
            key,
            tuple(block_gics),
            primitive_by_id=primitive_by_id,
            operations=operations,
            point_group=point_group,
            first_index=len(output) + 1,
            name_counters=name_counters,
            reference_coordinates_angstrom=coords,
            ring_pucker_coordinates_angstrom=ring_pucker_coords,
            global_b_basis=global_b_basis,
        )
        if projected is None:
            return None
        block_output, block_diagnostics, global_b_basis = projected
        output.extend(block_output)
        diagnostics.append(block_diagnostics)

    output_tuple = tuple(output)
    if coords_for_global is not None:
        projected_rank = _frozen_gic_b_matrix_rank(
            output_tuple,
            primitive_by_id=primitive_by_id,
            reference_coordinates_angstrom=reference_coordinates_angstrom,
        )
        if (
            has_native_ring_out_of_plane
            and projected_rank is not None
            and projected_rank < len(output_tuple)
        ):
            return None
    return (
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=POINT_GROUP_PROJECTOR_METHOD,
            policy=PROJECTOR_SYMMETRIZATION_POLICY,
            status="APPLIED",
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=tuple(diagnostics),
        ),
    )


def _apply_rank_revealing_global_projector(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: np.ndarray,
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizationDiagnostics] | None:
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    blocks: dict[tuple[str, str], list[FrozenGIC]] = {}
    for gic in gics:
        key = (primitive_symmetry_block(gic.family), gic.family)
        blocks.setdefault(key, []).append(gic)

    candidates: list[
        tuple[
            bool,
            tuple[str, str],
            str,
            np.ndarray,
            tuple[GICPrimitive, ...],
            np.ndarray,
            int,
        ]
    ] = []
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    sequence = 0
    for key, block_gics in sorted(
        blocks.items(),
        key=lambda item: _projector_block_sort_key(item, protect_special=True),
    ):
        block_primitives = _block_primitives_for_gics(
            tuple(block_gics),
            all_primitives=tuple(primitive_by_id.values()),
            primitive_by_id=primitive_by_id,
            key=key,
        )
        if block_primitives is None:
            continue
        primitive_index = _projector_primitive_index_with_aliases(
            block_primitives,
            primitive_by_id=primitive_by_id,
            key=key,
        )
        source_vectors = tuple(
            _gic_coefficient_vector(
                gic,
                primitive_index=primitive_index,
                vector_size=len(block_primitives),
            )
            for gic in block_gics
        )
        if any(vector is None for vector in source_vectors):
            continue
        source_vectors = _extend_projector_source_vectors_with_primitive_basis(
            source_vectors,
            vector_size=len(block_primitives),
        )
        primitive_key_index = _primitive_projector_key_index(block_primitives)
        if primitive_key_index is None:
            continue
        transforms = tuple(
            _operation_primitive_transform(
                block_primitives,
                operation=operation,
                primitive_key_index=primitive_key_index,
            )
            for operation in operations
        )
        if any(transform is None for transform in transforms):
            continue
        seen_vectors: set[tuple[float, ...]] = set()
        for irrep, characters in irrep_characters_for_operations(
            operation_labels,
            point_group,
            operation_matrices=operation_matrices,
        ):
            if len(characters) != len(operations):
                return None
            if all(abs(character) <= 1.0e-14 for character in characters):
                continue
            for source_vector in source_vectors:
                projected = _project_vector_for_irrep(
                    source_vector,
                    characters=characters,
                    transforms=transforms,
                )
                normalized = _normalized_coefficient_vector_or_none(projected)
                if normalized is None:
                    continue
                vector_key = tuple(round(float(value), 10) for value in normalized)
                opposite_key = tuple(round(float(-value), 10) for value in normalized)
                if vector_key in seen_vectors or opposite_key in seen_vectors:
                    continue
                seen_vectors.add(vector_key)
                b_row = _projected_vector_b_row(
                    block_primitives,
                    normalized,
                    coords=reference_coordinates_angstrom,
                )
                normalized_b = _normalized_coefficient_vector_or_none(b_row)
                if normalized_b is None:
                    continue
                candidates.append(
                    (
                        primitive_reduction_class(key[1]) == SPECIAL_REDUCTION_CLASS,
                        key,
                        irrep,
                        normalized,
                        block_primitives,
                        normalized_b,
                        sequence,
                    )
                )
                sequence += 1
    if not candidates:
        return None

    selected: list[tuple[tuple[str, str], str, np.ndarray, tuple[GICPrimitive, ...]]] = []
    q_basis: list[np.ndarray] = []
    remaining = list(candidates)
    for special_phase in (True, False):
        while len(selected) < len(gics):
            best_index: int | None = None
            best_residual: np.ndarray | None = None
            best_score = -1.0
            for idx, candidate in enumerate(remaining):
                is_special, _key, _irrep, _vector, _block_primitives, b_row, _sequence = candidate
                if special_phase and not is_special:
                    continue
                residual = _b_row_residual_against_basis(b_row, q_basis)
                score = float(np.linalg.norm(residual))
                if score > best_score + 1.0e-12:
                    best_index = idx
                    best_residual = residual
                    best_score = score
            if best_index is None or best_residual is None or best_score <= 1.0e-8:
                break
            is_special, key, irrep, vector, block_primitives, _b_row, _sequence = remaining.pop(
                best_index
            )
            q_basis.append(best_residual / best_score)
            selected.append((key, irrep, vector, block_primitives))
            if special_phase and not any(candidate[0] for candidate in remaining):
                break
        if len(selected) == len(gics):
            break
    if len(selected) != len(gics):
        return None

    output: list[FrozenGIC] = []
    name_counters: dict[tuple[str, str], int] = {}
    diagnostics_by_key: dict[tuple[str, str], list[str]] = {}
    for key, irrep, vector, block_primitives in selected:
        _block, family = key
        coefficients = _coefficients_from_vector(block_primitives, vector)
        if not coefficients:
            return None
        gic = FrozenGIC(
            identifier=f"GIC{len(output) + 1:03d}",
            name=_next_projected_name(family, irrep, name_counters),
            family=family,
            irrep=irrep,
            primitive_id=coefficients[0][0],
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=coefficients,
        )
        output.append(gic)
        diagnostics_by_key.setdefault(key, []).append(gic.name)

    diagnostics = tuple(
        GICSymmetrizedGroup(
            block=key[0],
            family=key[1],
            signature="OPS=" + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in blocks[key]),
            output_gics=tuple(output_names),
        )
        for key, output_names in diagnostics_by_key.items()
    )
    output_tuple = tuple(output)
    return (
        output_tuple,
        GICSymmetrizationDiagnostics(
            method=POINT_GROUP_PROJECTOR_METHOD,
            policy=PROJECTOR_SYMMETRIZATION_POLICY,
            status="APPLIED",
            point_group=point_group,
            symmetry_group=point_group,
            total_symmetric_irrep=total_symmetric_irrep(point_group),
            total_symmetric_gics=tuple(
                gic.name for gic in output_tuple if is_total_symmetric_irrep(point_group, gic.irrep)
            ),
            groups=diagnostics,
        ),
    )


def _b_row_residual_against_basis(
    row: np.ndarray,
    q_basis: list[np.ndarray],
) -> np.ndarray:
    residual = np.array(row, dtype=float, copy=True)
    for _pass in range(2):
        for basis_row in q_basis:
            residual -= float(np.dot(residual, basis_row)) * basis_row
    return residual


def _projector_block_sort_key(
    item: tuple[tuple[str, str], list[FrozenGIC]],
    *,
    protect_special: bool = False,
) -> tuple[int, int]:
    _block, family = item[0]
    if protect_special and primitive_reduction_class(family) == SPECIAL_REDUCTION_CLASS:
        try:
            order = PRIMITIVE_FAMILY_ORDER.index(family)
        except ValueError:
            order = len(PRIMITIVE_FAMILY_ORDER)
        return (-1, order)
    priority = {
        "STRETCH": 0,
        "BEND": 1,
        "BUTTERFLY": 2,
        "CYCLIC_BEND": 3,
    }
    if family in priority:
        return (priority[family], len(item[1]))
    try:
        order = PRIMITIVE_FAMILY_ORDER.index(family)
    except ValueError:
        order = len(PRIMITIVE_FAMILY_ORDER)
    return (10 + order, len(item[1]))


def _valid_projector_operations(
    operations: tuple[GICPointGroupOperation, ...],
) -> tuple[GICPointGroupOperation, ...]:
    if not operations:
        return ()
    natoms = len(operations[0].permutation)
    expected = tuple(range(1, natoms + 1))
    if natoms == 0:
        return ()
    validated: list[GICPointGroupOperation] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    for operation in operations:
        if len(operation.permutation) != natoms:
            return ()
        if tuple(sorted(operation.permutation)) != expected:
            return ()
        unique_key = (
            operation.permutation,
            tuple(round(float(value), 10) for row in operation.rotation for value in row),
        )
        if unique_key in seen:
            continue
        seen.add(unique_key)
        validated.append(operation)
    identity_index = next(
        (
            idx
            for idx, operation in enumerate(validated)
            if operation.label == "E" and operation.permutation == expected
        ),
        None,
    )
    if identity_index is None:
        return ()
    if identity_index:
        identity = validated.pop(identity_index)
        validated.insert(0, identity)
    return tuple(validated)


def _diagnostics_with_operation_margins(
    diagnostics: GICSymmetrizationDiagnostics,
    *,
    operations: tuple[GICPointGroupOperation, ...],
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> GICSymmetrizationDiagnostics:
    operation_margins = _symmetry_operation_margins(
        operations,
        reference_coordinates_angstrom=reference_coordinates_angstrom,
    )
    if operation_margins is None:
        return diagnostics
    max_residual, min_margin, near_threshold = operation_margins
    return replace(
        diagnostics,
        operation_tolerance_angstrom=SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
        max_operation_residual_angstrom=max_residual,
        min_operation_margin_angstrom=min_margin,
        near_threshold_operations=near_threshold,
    )


def _symmetry_operation_margins(
    operations: tuple[GICPointGroupOperation, ...],
    *,
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None,
) -> tuple[float, float, tuple[str, ...]] | None:
    validated = _valid_projector_operations(operations)
    if not validated or reference_coordinates_angstrom is None:
        return None
    coords = np.asarray(reference_coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(validated[0].permutation):
        return None
    residuals: list[tuple[str, float]] = []
    for operation in validated:
        rotation = np.asarray(operation.rotation, dtype=float)
        target = coords[np.asarray(operation.permutation, dtype=int) - 1]
        rotated_right = coords @ rotation.T
        rotated_left = coords @ rotation
        residual = min(
            float(np.max(np.linalg.norm(rotated_right - target, axis=1))),
            float(np.max(np.linalg.norm(rotated_left - target, axis=1))),
        )
        residuals.append((operation.label, residual))
    if not residuals:
        return None
    tolerance = float(SYMMETRY_OPERATION_TOLERANCE_ANGSTROM)
    max_residual = max(residual for _label, residual in residuals)
    margins = tuple(tolerance - residual for _label, residual in residuals)
    min_margin = min(margins)
    near_width = tolerance * float(SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION)
    near_threshold = tuple(
        f"{label}:{residual:.6g}"
        for label, residual in residuals
        if tolerance - residual <= near_width
    )
    return max_residual, min_margin, near_threshold


def _symmetry_closed_projector_primitives(
    primitives: tuple[GICPrimitive, ...],
    *,
    symmetry_operations: tuple[GICPointGroupOperation, ...],
) -> tuple[GICPrimitive, ...]:
    operations = _valid_projector_operations(symmetry_operations)
    if len(operations) <= 1:
        return primitives

    output = list(primitives)
    keys = {key for primitive in output if (key := _primitive_projector_key(primitive)) is not None}
    cursor = 0
    while cursor < len(output):
        primitive = output[cursor]
        cursor += 1
        for operation in operations:
            mapped = _mapped_projector_primitive(primitive, operation)
            if mapped is None:
                continue
            key = _primitive_projector_key(mapped)
            if key is None or key in keys:
                continue
            keys.add(key)
            index = len(output) + 1
            output.append(
                replace(
                    mapped,
                    identifier=f"P{index:03d}",
                    name=f"P{index:03d}",
                )
            )
    return tuple(output)


def _mapped_projector_primitive(
    primitive: GICPrimitive,
    operation: GICPointGroupOperation,
) -> GICPrimitive | None:
    if primitive.family == "LINEAR_BEND":
        return None
    if primitive.family in {"FRAG_TRANSLATION", "FRAG_ORIENTATION"}:
        return None

    mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in primitive.atoms)
    mapped_refs = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_atoms)
    mapped_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.frame_atoms)
    mapped_ref_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_frame_atoms)
    if any(atom < 1 for atom in mapped_atoms + mapped_refs + mapped_frame + mapped_ref_frame):
        return None
    return replace(
        primitive,
        atoms=mapped_atoms,
        ref_atoms=mapped_refs,
        frame_atoms=mapped_frame,
        ref_frame_atoms=mapped_ref_frame,
    )


def _project_gic_block(
    key: tuple[str, str],
    gics: tuple[FrozenGIC, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    operations: tuple[GICPointGroupOperation, ...],
    point_group: str,
    first_index: int,
    name_counters: dict[tuple[str, str], int],
    reference_coordinates_angstrom: np.ndarray | None = None,
    ring_pucker_coordinates_angstrom: np.ndarray | None = None,
    global_b_basis: tuple[np.ndarray, ...] = (),
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None:
    block, family = key
    block_primitives = _block_primitives_for_gics(
        gics,
        all_primitives=tuple(primitive_by_id.values()),
        primitive_by_id=primitive_by_id,
        key=key,
    )
    if block_primitives is None:
        return None

    primitive_index = _projector_primitive_index_with_aliases(
        block_primitives,
        primitive_by_id=primitive_by_id,
        key=key,
    )
    source_vectors = tuple(
        _gic_coefficient_vector(
            gic,
            primitive_index=primitive_index,
            vector_size=len(block_primitives),
        )
        for gic in gics
    )
    if any(vector is None for vector in source_vectors):
        return None

    ring_brow_projected: (
        tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None
    ) = None
    ring_brow_counters: dict[tuple[str, str], int] | None = None
    ring_brow_attempted = False

    def ring_brow_result() -> (
        tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None
    ):
        nonlocal ring_brow_attempted, ring_brow_projected, ring_brow_counters
        if ring_brow_attempted:
            return ring_brow_projected
        ring_brow_attempted = True
        if (
            family != "RING_PUCKER_COMPONENT"
            or ring_pucker_coordinates_angstrom is None
            or len(source_vectors) != len(gics)
        ):
            return None
        candidate_counters = dict(name_counters)
        ring_brow_projected = _project_ring_pucker_source_block(
            key,
            gics,
            block_primitives,
            tuple(
                np.asarray(vector, dtype=float) for vector in source_vectors if vector is not None
            ),
            operations=operations,
            point_group=point_group,
            first_index=first_index,
            name_counters=candidate_counters,
            reference_coordinates_angstrom=ring_pucker_coordinates_angstrom,
            global_b_basis=global_b_basis,
        )
        if ring_brow_projected is not None:
            ring_brow_counters = candidate_counters
        return ring_brow_projected

    def use_result(
        result: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
        counters: dict[tuple[str, str], int] | None,
    ) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]]:
        if counters is not None:
            name_counters.clear()
            name_counters.update(counters)
        return result

    primitive_key_index = _primitive_projector_key_index(block_primitives)
    if primitive_key_index is None:
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None
    transforms = tuple(
        _operation_primitive_transform(
            block_primitives,
            operation=operation,
            primitive_key_index=primitive_key_index,
        )
        for operation in operations
    )
    if any(transform is None for transform in transforms):
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None

    projected_vectors: list[tuple[str, np.ndarray]] = []
    basis: list[np.ndarray] = []
    b_basis = list(global_b_basis)
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(operations):
            return None
        if all(abs(character) <= 1.0e-14 for character in characters):
            continue
        # For polyhedral groups, canonicalize the projected source span before
        # selecting rows by Cartesian rank.  Incoming GICs can carry an
        # arbitrary basis inside a degenerate SALC subspace (for example from
        # a degenerate inertia eigensystem); the span is physical, that basis
        # is not.
        canonical_polyhedral_basis = point_group.strip().upper() in {
            "T",
            "TD",
            "TH",
            "O",
            "OH",
            "I",
            "IH",
        }
        irrep_candidates: list[np.ndarray] = []
        for source_vector in source_vectors:
            assert source_vector is not None
            projected = _project_vector_for_irrep(
                source_vector,
                characters=characters,
                transforms=transforms,
            )
            normalized = _normalized_coefficient_vector_or_none(projected)
            if normalized is None:
                continue
            irrep_candidates.append(normalized)
        projection_candidates = (
            _canonical_basis_for_span(irrep_candidates)
            if canonical_polyhedral_basis
            else tuple(irrep_candidates)
        )
        for normalized in projection_candidates:
            independent = _orthonormal_coefficient_residual_or_none(basis, normalized)
            if independent is None:
                continue
            independent = _canonical_vector_sign(independent)
            b_independent = None
            if reference_coordinates_angstrom is not None:
                b_row = _projected_vector_b_row(
                    block_primitives,
                    independent,
                    coords=reference_coordinates_angstrom,
                )
                normalized_b = _normalized_coefficient_vector_or_none(b_row)
                if normalized_b is None:
                    continue
                b_independent = _orthonormal_coefficient_residual_or_none(
                    b_basis,
                    normalized_b,
                )
                if b_independent is None:
                    continue
            basis.append(independent)
            if b_independent is not None:
                b_basis.append(b_independent)
            projected_vectors.append((irrep, independent))
            if len(projected_vectors) == len(gics):
                break
        if len(projected_vectors) == len(gics):
            break
    if len(projected_vectors) != len(gics):
        projected_ring = ring_brow_result()
        return use_result(projected_ring, ring_brow_counters) if projected_ring else None

    primitive_counters = dict(name_counters)
    output: list[FrozenGIC] = []
    for offset, (irrep, vector) in enumerate(projected_vectors):
        coefficients = _coefficients_from_vector(block_primitives, vector)
        if not coefficients:
            return None
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + offset:03d}",
                name=_next_projected_name(family, irrep, primitive_counters),
                family=family,
                irrep=irrep,
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )

    primitive_result = (
        tuple(output),
        GICSymmetrizedGroup(
            block=block,
            family=family,
            signature="OPS=" + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in gics),
            output_gics=tuple(gic.name for gic in output),
        ),
        tuple(b_basis),
    )
    projected_ring = ring_brow_result()
    if projected_ring is not None and _same_projected_irrep_sequence(
        projected_ring,
        primitive_result,
    ):
        return use_result(projected_ring, ring_brow_counters)
    return use_result(primitive_result, primitive_counters)


def _same_projected_irrep_sequence(
    left: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
    right: tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]],
) -> bool:
    return tuple(gic.irrep for gic in left[0]) == tuple(gic.irrep for gic in right[0])


def _project_ring_pucker_source_block(
    key: tuple[str, str],
    block_gics: tuple[FrozenGIC, ...],
    block_primitives: tuple[GICPrimitive, ...],
    source_vectors: tuple[np.ndarray, ...],
    *,
    operations: tuple[GICPointGroupOperation, ...],
    point_group: str,
    first_index: int,
    name_counters: dict[tuple[str, str], int],
    reference_coordinates_angstrom: np.ndarray,
    global_b_basis: tuple[np.ndarray, ...],
) -> tuple[tuple[FrozenGIC, ...], GICSymmetrizedGroup, tuple[np.ndarray, ...]] | None:
    if not source_vectors:
        return None
    try:
        source_b_rows = np.vstack(
            [
                _projected_vector_b_row(
                    block_primitives,
                    source_vector,
                    coords=reference_coordinates_angstrom,
                )
                for source_vector in source_vectors
            ]
        )
    except (FloatingPointError, ValueError):
        return None
    if np.linalg.matrix_rank(source_b_rows, tol=1.0e-8) < len(source_vectors):
        return None
    transforms = tuple(
        _source_b_row_transform(
            source_b_rows,
            operation=operation,
            natoms=len(reference_coordinates_angstrom),
        )
        for operation in operations
    )
    if any(transform is None for transform in transforms):
        return None

    projected_vectors: list[tuple[str, np.ndarray]] = []
    source_basis: list[np.ndarray] = []
    b_basis = list(global_b_basis)
    operation_labels = tuple(operation.label for operation in operations)
    operation_matrices = tuple(operation.rotation for operation in operations)
    source_units = tuple(np.eye(len(source_vectors), dtype=float))
    for irrep, characters in irrep_characters_for_operations(
        operation_labels,
        point_group,
        operation_matrices=operation_matrices,
    ):
        if len(characters) != len(operations):
            return None
        if all(abs(character) <= 1.0e-14 for character in characters):
            continue
        for source_unit in source_units:
            projected_source = _project_vector_for_irrep(
                source_unit,
                characters=characters,
                transforms=transforms,
            )
            normalized_source = _normalized_coefficient_vector_or_none(projected_source)
            if normalized_source is None:
                continue
            independent_source = _orthonormal_coefficient_residual_or_none(
                source_basis,
                normalized_source,
            )
            if independent_source is None:
                continue
            primitive_vector = np.zeros_like(source_vectors[0], dtype=float)
            for source_coefficient, source_vector in zip(independent_source, source_vectors):
                primitive_vector += float(source_coefficient) * source_vector
            try:
                b_row = _projected_vector_b_row(
                    block_primitives,
                    primitive_vector,
                    coords=reference_coordinates_angstrom,
                )
            except (FloatingPointError, ValueError):
                return None
            normalized_b = _normalized_coefficient_vector_or_none(b_row)
            if normalized_b is None:
                continue
            b_independent = _orthonormal_coefficient_residual_or_none(b_basis, normalized_b)
            if b_independent is None:
                continue
            source_basis.append(independent_source)
            b_basis.append(b_independent)
            projected_vectors.append((irrep, primitive_vector))
            if len(projected_vectors) == len(block_gics):
                break
        if len(projected_vectors) == len(block_gics):
            break
    if len(projected_vectors) != len(block_gics):
        return None

    _block, family = key
    output: list[FrozenGIC] = []
    for offset, (irrep, primitive_vector) in enumerate(projected_vectors):
        normalized = _normalized_coefficient_vector_or_none(primitive_vector)
        if normalized is None:
            return None
        coefficients = _coefficients_from_vector(block_primitives, normalized)
        if not coefficients:
            return None
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + offset:03d}",
                name=_next_projected_name(family, irrep, name_counters),
                family=family,
                irrep=irrep,
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )
    return (
        tuple(output),
        GICSymmetrizedGroup(
            block=key[0],
            family=family,
            signature="BROWS_OPS=" + ",".join(operation_labels),
            source_gics=tuple(gic.name for gic in block_gics),
            output_gics=tuple(gic.name for gic in output),
        ),
        tuple(b_basis),
    )


def _source_b_row_transform(
    source_b_rows: np.ndarray,
    *,
    operation: GICPointGroupOperation,
    natoms: int,
) -> np.ndarray | None:
    cartesian = _cartesian_operation_matrix(operation, natoms=natoms)
    matrix = np.zeros((source_b_rows.shape[0], source_b_rows.shape[0]), dtype=float)
    for source_index, row in enumerate(source_b_rows):
        transformed = row @ cartesian
        values, *_ = np.linalg.lstsq(source_b_rows.T, transformed.T, rcond=1.0e-10)
        residual = float(np.linalg.norm(values @ source_b_rows - transformed))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _cartesian_operation_matrix(operation: GICPointGroupOperation, *, natoms: int) -> np.ndarray:
    rotation = np.asarray(operation.rotation, dtype=float)
    matrix = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    for source_index, target_atom in enumerate(operation.permutation):
        target_index = int(target_atom) - 1
        if target_index < 0 or target_index >= natoms:
            continue
        matrix[
            3 * source_index : 3 * source_index + 3,
            3 * target_index : 3 * target_index + 3,
        ] = rotation
    return matrix


def _projected_vector_b_row(
    primitives: tuple[GICPrimitive, ...],
    vector: np.ndarray,
    *,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for primitive, coefficient in zip(primitives, vector):
        if abs(float(coefficient)) <= 1.0e-12:
            continue
        row += float(coefficient) * _analytic_b_row(primitive, coords)
    return row


def _independent_b_row_or_none(
    basis: list[np.ndarray],
    normalized: np.ndarray,
) -> np.ndarray | None:
    if not basis:
        return normalized
    current = np.vstack(basis)
    candidate = np.vstack((*basis, normalized))
    if np.linalg.matrix_rank(candidate, tol=1.0e-8) <= np.linalg.matrix_rank(
        current,
        tol=1.0e-8,
    ):
        return None
    return normalized


def _block_primitives_for_gics(
    gics: tuple[FrozenGIC, ...],
    *,
    all_primitives: tuple[GICPrimitive, ...],
    primitive_by_id: dict[str, GICPrimitive],
    key: tuple[str, str],
) -> tuple[GICPrimitive, ...] | None:
    block, family = key
    ordered: list[GICPrimitive] = []
    index_by_key: dict[tuple[object, ...], int] = {}
    included_ids: set[str] = set()
    required_ids = {
        primitive_id
        for gic in gics
        for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    }

    def add_primitive(primitive: GICPrimitive) -> bool:
        primitive_key = (primitive_symmetry_block(primitive.family), primitive.family)
        if primitive_key != (block, family):
            return True
        projector_key = _primitive_projector_key(primitive)
        if projector_key is None:
            return False
        duplicate_index = index_by_key.get(projector_key)
        if duplicate_index is None:
            index_by_key[projector_key] = len(ordered)
            included_ids.add(primitive.identifier)
            ordered.append(primitive)
            return True
        existing = ordered[duplicate_index]
        if existing.identifier == primitive.identifier:
            return True
        existing_required = existing.identifier in required_ids
        current_required = primitive.identifier in required_ids
        if current_required and existing_required:
            return True
        if current_required and not existing_required:
            included_ids.discard(existing.identifier)
            included_ids.add(primitive.identifier)
            ordered[duplicate_index] = primitive
        return True

    for gic in gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, _coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                return None
            if not add_primitive(primitive):
                return None

    for primitive in all_primitives:
        if primitive.identifier in included_ids:
            continue
        if not add_primitive(primitive):
            return None
    return tuple(ordered)


def _projector_primitive_index_with_aliases(
    block_primitives: tuple[GICPrimitive, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    key: tuple[str, str],
) -> dict[str, int]:
    block, family = key
    key_to_index = {
        _primitive_projector_key(primitive): index
        for index, primitive in enumerate(block_primitives)
    }
    index = {
        primitive.identifier: primitive_index
        for primitive_index, primitive in enumerate(block_primitives)
    }
    for primitive in primitive_by_id.values():
        primitive_key = (primitive_symmetry_block(primitive.family), primitive.family)
        if primitive_key != (block, family):
            continue
        projector_key = _primitive_projector_key(primitive)
        primitive_index = key_to_index.get(projector_key)
        if primitive_index is None:
            continue
        index.setdefault(primitive.identifier, primitive_index)
    return index


def _gic_coefficient_vector(
    gic: FrozenGIC,
    *,
    primitive_index: dict[str, int],
    vector_size: int | None = None,
) -> np.ndarray | None:
    vector = np.zeros(vector_size or len(primitive_index), dtype=float)
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    for primitive_id, coefficient in coefficients:
        idx = primitive_index.get(primitive_id)
        if idx is None:
            return None
        vector[idx] += float(coefficient)
    return vector


def _extend_projector_source_vectors_with_primitive_basis(
    source_vectors: tuple[np.ndarray | None, ...],
    *,
    vector_size: int,
) -> tuple[np.ndarray, ...]:
    out = [np.asarray(vector, dtype=float) for vector in source_vectors if vector is not None]
    seen = {tuple(round(float(value), 12) for value in vector) for vector in out}
    for idx in range(vector_size):
        unit = np.zeros(vector_size, dtype=float)
        unit[idx] = 1.0
        key = tuple(round(float(value), 12) for value in unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(unit)
    return tuple(out)


def _primitive_projector_key_index(
    primitives: tuple[GICPrimitive, ...],
) -> dict[tuple[object, ...], int] | None:
    out: dict[tuple[object, ...], int] = {}
    for idx, primitive in enumerate(primitives):
        key = _primitive_projector_key(primitive)
        if key is None or key in out:
            return None
        out[key] = idx
    return out


def _primitive_projector_orientation_sign(primitive: GICPrimitive) -> float:
    """Return the phase relating an oriented primitive to its canonical key."""
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(primitive.atoms) == 4
    ):
        center, first, second, third = primitive.atoms
        _canonical, sign = _canonical_torsion_key_and_sign(
            (first, center, third, second)
        )
        return sign
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and primitive.function == "D"
        and len(primitive.atoms) == 4
    ):
        _canonical, sign = _canonical_torsion_key_and_sign(primitive.atoms)
        return sign
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        substituents = primitive.atoms[1:]
        return _permutation_parity_sign(substituents, tuple(sorted(substituents)))
    return 1.0


def _operation_primitive_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
    primitive_key_index: dict[tuple[object, ...], int],
) -> np.ndarray | None:
    if all(
        primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK"
        for primitive in primitives
    ):
        return _operation_ring_pucker_transform(primitives, operation=operation)
    if all(
        primitive.family == "PSEUDO_CYCLE_BEND" and primitive.function == "RPCB"
        for primitive in primitives
    ):
        return _operation_angle_component_transform(primitives, operation=operation)
    if all(
        primitive.family == "PSEUDO_CYCLE_TORSION" and primitive.function == "RPCK"
        for primitive in primitives
    ):
        return _operation_ring_pucker_transform(primitives, operation=operation)

    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, primitive in enumerate(primitives):
        terms = _mapped_primitive_projector_terms(primitive, operation)
        if terms is None:
            return None
        for target_key, coefficient in terms:
            if abs(float(coefficient)) <= 1.0e-12:
                continue
            target_index = primitive_key_index.get(target_key)
            if target_index is None:
                return None
            target_orientation = _primitive_projector_orientation_sign(
                primitives[target_index]
            )
            matrix[target_index, source_index] += float(coefficient) * target_orientation
    return matrix


def _operation_ring_pucker_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
) -> np.ndarray | None:
    source_coeffs = [
        _ring_pucker_canonical_torsion_coefficients(_ring_pucker_terms_from_refs(primitive))
        for primitive in primitives
    ]
    pseudoscalar_sign = _operation_pseudoscalar_sign(operation)
    mapped_coeffs = []
    for primitive in primitives:
        mapped_terms = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_atoms):
                return None
            mapped_terms.append((coefficient * pseudoscalar_sign, mapped_atoms))
        mapped_coeffs.append(_ring_pucker_canonical_torsion_coefficients(tuple(mapped_terms)))

    torsion_keys = sorted({key for coeffs in (*source_coeffs, *mapped_coeffs) for key in coeffs})
    if not torsion_keys:
        return None
    basis = np.array(
        [[coeffs.get(key, 0.0) for coeffs in source_coeffs] for key in torsion_keys],
        dtype=float,
    )
    if np.linalg.matrix_rank(basis, tol=1.0e-10) == 0:
        return None
    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, coeffs in enumerate(mapped_coeffs):
        target = np.array([coeffs.get(key, 0.0) for key in torsion_keys], dtype=float)
        values, *_ = np.linalg.lstsq(basis, target, rcond=1.0e-10)
        residual = float(np.linalg.norm(basis @ values - target))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _operation_angle_component_transform(
    primitives: tuple[GICPrimitive, ...],
    *,
    operation: GICPointGroupOperation,
) -> np.ndarray | None:
    source_coeffs = [
        _angle_component_canonical_coefficients(_angle_component_terms_from_refs(primitive))
        for primitive in primitives
    ]
    mapped_coeffs = []
    for primitive in primitives:
        mapped_terms = []
        for coefficient, atoms in _angle_component_terms_from_refs(primitive):
            mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_atoms):
                return None
            mapped_terms.append((coefficient, mapped_atoms))
        mapped_coeffs.append(_angle_component_canonical_coefficients(tuple(mapped_terms)))

    angle_keys = sorted({key for coeffs in (*source_coeffs, *mapped_coeffs) for key in coeffs})
    if not angle_keys:
        return None
    basis = np.array(
        [[coeffs.get(key, 0.0) for coeffs in source_coeffs] for key in angle_keys],
        dtype=float,
    )
    if np.linalg.matrix_rank(basis, tol=1.0e-10) == 0:
        return None
    matrix = np.zeros((len(primitives), len(primitives)), dtype=float)
    for source_index, coeffs in enumerate(mapped_coeffs):
        target = np.array([coeffs.get(key, 0.0) for key in angle_keys], dtype=float)
        values, *_ = np.linalg.lstsq(basis, target, rcond=1.0e-10)
        residual = float(np.linalg.norm(basis @ values - target))
        if residual > 1.0e-8:
            return None
        matrix[:, source_index] = values
    return matrix


def _project_vector_for_irrep(
    vector: np.ndarray,
    *,
    characters: tuple[float, ...],
    transforms: tuple[np.ndarray | None, ...],
) -> np.ndarray:
    projected = np.zeros_like(vector, dtype=float)
    for character, transform in zip(characters, transforms):
        assert transform is not None
        projected += float(character) * (transform @ vector)
    return projected / float(len(transforms))


def _normalized_coefficient_vector_or_none(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return vector / norm


def _canonical_vector_sign(vector: np.ndarray) -> np.ndarray:
    """Choose the sign whose first significant coefficient is positive."""

    significant = np.flatnonzero(np.abs(vector) > 1.0e-12)
    if significant.size and float(vector[int(significant[0])]) < 0.0:
        return -vector
    return vector


def _canonical_basis_for_span(vectors: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return a basis fixed by coordinate order, not by the input basis."""

    if not vectors:
        return ()
    rows = np.vstack(vectors)
    gram = rows @ rows.T
    projector = rows.T @ np.linalg.pinv(gram, rcond=1.0e-12) @ rows
    projector = 0.5 * (projector + projector.T)
    rank = int(np.linalg.matrix_rank(projector, tol=1.0e-10))
    basis: list[np.ndarray] = []
    for axis in range(projector.shape[0]):
        normalized = _normalized_coefficient_vector_or_none(projector[:, axis])
        if normalized is None:
            continue
        independent = _orthonormal_coefficient_residual_or_none(basis, normalized)
        if independent is None:
            continue
        basis.append(_canonical_vector_sign(independent))
        if len(basis) == rank:
            break
    return tuple(basis)


def _orthonormal_coefficient_residual_or_none(
    basis: list[np.ndarray],
    normalized: np.ndarray,
) -> np.ndarray | None:
    residual = np.array(normalized, dtype=float, copy=True)
    for _pass in range(2):
        for vector in basis:
            residual -= float(np.dot(residual, vector)) * vector
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return residual / norm


def _coefficients_from_vector(
    primitives: tuple[GICPrimitive, ...],
    vector: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (primitive.identifier, float(value))
        for primitive, value in zip(primitives, vector)
        if abs(float(value)) > 1.0e-12
    )


def _next_projected_name(
    family: str,
    irrep: str,
    counters: dict[tuple[str, str], int],
) -> str:
    key = (family, irrep)
    counters[key] = counters.get(key, 0) + 1
    return f"{irrep_name_prefix(irrep)}{primitive_prefix(family)}{counters[key]:03d}"


def _primitive_projector_key(primitive: GICPrimitive) -> tuple[object, ...] | None:
    if primitive.family in {"STRETCH", "HBOND_DISTANCE"} and len(primitive.atoms) == 2:
        return (primitive.family, tuple(sorted(primitive.atoms)))
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(primitive.atoms) == 3:
        return (
            primitive.family,
            primitive.atoms[1],
            tuple(sorted((primitive.atoms[0], primitive.atoms[2]))),
        )
    if primitive.family == "LINEAR_BEND" and len(primitive.atoms) == 3:
        return (
            "LINEAR_BEND",
            primitive.atoms[1],
            tuple(sorted((primitive.atoms[0], primitive.atoms[2]))),
            primitive.mode,
        )
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(primitive.atoms) == 4
    ):
        center, first, second, third = primitive.atoms
        canonical, _sign = _canonical_torsion_key_and_sign(
            (first, center, third, second)
        )
        return ("RING_PUCKER_COMPONENT", canonical)
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and len(primitive.atoms) == 4
    ):
        canonical, _sign = _canonical_torsion_key_and_sign(primitive.atoms)
        return (primitive.family, canonical)
    if primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK":
        signature = _ring_pucker_projector_signature(primitive)
        if signature is None:
            return None
        key, _sign = signature
        return ("RING_PUCKER_COMPONENT", key)
    if primitive.family == "PSEUDO_CYCLE_BEND" and primitive.function == "RPCB":
        key = _angle_component_projector_key(primitive)
        return ("PSEUDO_CYCLE_BEND", key) if key is not None else None
    if primitive.family == "PSEUDO_CYCLE_TORSION" and primitive.function == "RPCK":
        signature = _ring_pucker_projector_signature(primitive)
        if signature is None:
            return None
        key, _sign = signature
        return ("PSEUDO_CYCLE_TORSION", key)
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        return (
            primitive.family,
            primitive.atoms[0],
            tuple(sorted(primitive.atoms[1:])),
        )
    if primitive.family == "FRAG_DISTANCE":
        pair = tuple(sorted((_atom_set_key(primitive.atoms), _atom_set_key(primitive.ref_atoms))))
        return ("FRAG_DISTANCE", pair)
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            "FRAG_CENTER_ATOM_DISTANCE",
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            "CENTER_ATOM_DISTANCE",
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "FRAG_TRANSLATION":
        return (
            "FRAG_TRANSLATION",
            primitive.mode,
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
        )
    if primitive.family == "FRAG_ORIENTATION":
        return (
            "FRAG_ORIENTATION",
            primitive.mode,
            _atom_set_key(primitive.atoms),
            _atom_set_key(primitive.ref_atoms),
            _atom_set_key(primitive.frame_atoms),
            _atom_set_key(primitive.ref_frame_atoms),
        )
    return None


def _mapped_primitive_projector_terms(
    primitive: GICPrimitive,
    operation: GICPointGroupOperation,
) -> tuple[tuple[tuple[object, ...], float], ...] | None:
    mapped_atoms = tuple(_mapped_atom(operation, atom) for atom in primitive.atoms)
    mapped_refs = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_atoms)
    mapped_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.frame_atoms)
    mapped_ref_frame = tuple(_mapped_atom(operation, atom) for atom in primitive.ref_frame_atoms)
    if any(atom < 1 for atom in mapped_atoms + mapped_refs + mapped_frame + mapped_ref_frame):
        return None

    if primitive.family in {"STRETCH", "HBOND_DISTANCE"} and len(mapped_atoms) == 2:
        return (((primitive.family, tuple(sorted(mapped_atoms))), 1.0),)
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(mapped_atoms) == 3:
        return (
            (
                (
                    primitive.family,
                    mapped_atoms[1],
                    tuple(sorted((mapped_atoms[0], mapped_atoms[2]))),
                ),
                1.0,
            ),
        )
    if primitive.family == "LINEAR_BEND":
        if not _is_identity_operation(operation):
            return None
        key = _primitive_projector_key(primitive)
        return ((key, 1.0),) if key is not None else None
    if (
        primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "U"
        and len(mapped_atoms) == 4
    ):
        center, first, second, third = mapped_atoms
        canonical, sign = _canonical_torsion_key_and_sign(
            (first, center, third, second)
        )
        return (
            (
                ("RING_PUCKER_COMPONENT", canonical),
                sign * _operation_pseudoscalar_sign(operation),
            ),
        )
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
            "RING_PUCKER_COMPONENT",
        }
        and len(mapped_atoms) == 4
    ):
        canonical, sign = _canonical_torsion_key_and_sign(mapped_atoms)
        return (
            (
                (primitive.family, canonical),
                sign * _operation_pseudoscalar_sign(operation),
            ),
        )
    if primitive.family == "RING_PUCKER_COMPONENT" and primitive.function == "RPCK":
        pseudoscalar_sign = _operation_pseudoscalar_sign(operation)
        mapped_terms = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            mapped_term_atoms = tuple(_mapped_atom(operation, atom) for atom in atoms)
            if any(atom < 1 for atom in mapped_term_atoms):
                return None
            mapped_terms.append((coefficient * pseudoscalar_sign, mapped_term_atoms))
        signature = _ring_pucker_projector_signature_from_terms(tuple(mapped_terms))
        if signature is None:
            return None
        key, sign = signature
        return ((("RING_PUCKER_COMPONENT", key), sign),)
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(mapped_atoms) == 4:
        center = mapped_atoms[0]
        substituents = mapped_atoms[1:]
        sorted_substituents = tuple(sorted(substituents))
        return (
            (
                (
                    primitive.family,
                    center,
                    sorted_substituents,
                ),
                _permutation_parity_sign(substituents, sorted_substituents)
                * _operation_pseudoscalar_sign(operation),
            ),
        )
    if primitive.family == "FRAG_DISTANCE":
        pair = tuple(sorted((_atom_set_key(mapped_atoms), _atom_set_key(mapped_refs))))
        return ((("FRAG_DISTANCE", pair), 1.0),)
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            (
                (
                    "FRAG_CENTER_ATOM_DISTANCE",
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                1.0,
            ),
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            (
                (
                    "CENTER_ATOM_DISTANCE",
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                1.0,
            ),
        )
    if primitive.family == "FRAG_TRANSLATION":
        return tuple(
            (
                (
                    "FRAG_TRANSLATION",
                    target_mode,
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                ),
                coefficient,
            )
            for target_mode, coefficient in _vector_component_terms(
                operation,
                source_mode=primitive.mode,
                axial=False,
            )
        )
    if primitive.family == "FRAG_ORIENTATION":
        return tuple(
            (
                (
                    "FRAG_ORIENTATION",
                    target_mode,
                    _atom_set_key(mapped_atoms),
                    _atom_set_key(mapped_refs),
                    _atom_set_key(mapped_frame),
                    _atom_set_key(mapped_ref_frame),
                ),
                coefficient,
            )
            for target_mode, coefficient in _vector_component_terms(
                operation,
                source_mode=primitive.mode,
                axial=True,
            )
        )
    return None


def _mapped_atom(operation: GICPointGroupOperation, atom: int) -> int:
    if 1 <= atom <= len(operation.permutation):
        return operation.permutation[atom - 1]
    return -1


def _is_identity_operation(operation: GICPointGroupOperation) -> bool:
    return operation.permutation == tuple(range(1, len(operation.permutation) + 1))


def _vector_component_terms(
    operation: GICPointGroupOperation,
    *,
    source_mode: int,
    axial: bool,
) -> tuple[tuple[int, float], ...]:
    if source_mode not in {0, 1, 2}:
        return ()
    rotation = np.asarray(operation.rotation, dtype=float)
    if rotation.shape != (3, 3):
        return ()
    if axial:
        rotation = float(np.linalg.det(rotation)) * rotation
    return tuple(
        (target_mode, float(rotation[source_mode, target_mode]))
        for target_mode in range(3)
        if abs(float(rotation[source_mode, target_mode])) > 1.0e-10
    )


def _operation_pseudoscalar_sign(operation: GICPointGroupOperation) -> float:
    rotation = np.asarray(operation.rotation, dtype=float)
    if rotation.shape != (3, 3):
        return 1.0
    return -1.0 if float(np.linalg.det(rotation)) < 0.0 else 1.0


def _atom_set_key(atoms: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(atom) for atom in atoms))


def _canonical_torsion_key_and_sign(atoms: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    forward = tuple(atoms)
    backward = tuple(reversed(atoms))
    if backward < forward:
        return backward, -1.0
    return forward, 1.0


def _ring_pucker_projector_signature(
    primitive: GICPrimitive,
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], float] | None:
    return _ring_pucker_projector_signature_from_terms(_ring_pucker_terms_from_refs(primitive))


def _ring_pucker_projector_signature_from_terms(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], float] | None:
    by_torsion = _ring_pucker_canonical_torsion_coefficients(terms)
    compact = {
        atoms: coefficient
        for atoms, coefficient in by_torsion.items()
        if abs(float(coefficient)) > 1.0e-12
    }
    if not compact:
        return None
    _first_atoms, first_coefficient = next(iter(sorted(compact.items())))
    overall_sign = -1.0 if first_coefficient < 0.0 else 1.0
    key = tuple(
        (atoms, round(float(coefficient) * overall_sign, 12))
        for atoms, coefficient in sorted(compact.items())
    )
    return key, overall_sign


def _ring_pucker_canonical_torsion_coefficients(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> dict[tuple[int, ...], float]:
    by_torsion: dict[tuple[int, ...], float] = {}
    for coefficient, atoms in terms:
        if len(atoms) != 4:
            continue
        canonical, sign = _canonical_torsion_key_and_sign(atoms)
        by_torsion[canonical] = by_torsion.get(canonical, 0.0) + float(coefficient) * sign
    return {
        atoms: coefficient
        for atoms, coefficient in by_torsion.items()
        if abs(float(coefficient)) > 1.0e-12
    }


def _angle_component_projector_key(
    primitive: GICPrimitive,
) -> tuple[tuple[tuple[object, ...], float], ...] | None:
    coefficients = _angle_component_canonical_coefficients(
        _angle_component_terms_from_refs(primitive)
    )
    compact = {
        key: coefficient
        for key, coefficient in coefficients.items()
        if abs(float(coefficient)) > 1.0e-12
    }
    if not compact:
        return None
    _first_key, first_coefficient = next(iter(sorted(compact.items())))
    overall_sign = -1.0 if first_coefficient < 0.0 else 1.0
    return tuple(
        (key, round(float(coefficient) * overall_sign, 12))
        for key, coefficient in sorted(compact.items())
    )


def _angle_component_canonical_coefficients(
    terms: tuple[tuple[float, tuple[int, ...]], ...],
) -> dict[tuple[object, ...], float]:
    by_angle: dict[tuple[object, ...], float] = {}
    for coefficient, atoms in terms:
        if len(atoms) != 3:
            continue
        key = (atoms[1], tuple(sorted((atoms[0], atoms[2]))))
        by_angle[key] = by_angle.get(key, 0.0) + float(coefficient)
    return {
        key: coefficient
        for key, coefficient in by_angle.items()
        if abs(float(coefficient)) > 1.0e-12
    }


def _permutation_parity_sign(
    order: tuple[int, ...],
    target_order: tuple[int, ...],
) -> float:
    positions = {atom: idx for idx, atom in enumerate(target_order)}
    try:
        indexes = [positions[atom] for atom in order]
    except KeyError:
        return 1.0
    inversions = 0
    for left in range(len(indexes)):
        for right in range(left + 1, len(indexes)):
            if indexes[left] > indexes[right]:
                inversions += 1
    return -1.0 if inversions % 2 else 1.0


def _local_symmetry_groups(
    gics: tuple[FrozenGIC, ...],
    primitives: tuple[GICPrimitive, ...],
    *,
    atom_symbols: tuple[str, ...],
) -> dict[tuple[str, str, str], tuple[FrozenGIC, ...]]:
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    grouped: dict[tuple[str, str, str], list[FrozenGIC]] = {}
    for gic in gics:
        primitive = _single_source_primitive(gic, primitive_by_id)
        if primitive is None:
            continue
        signature = _local_symmetry_signature(primitive, atom_symbols=atom_symbols)
        if not signature:
            continue
        key = (primitive_symmetry_block(primitive.family), primitive.family, signature)
        grouped.setdefault(key, []).append(gic)
    return {key: tuple(group) for key, group in grouped.items() if len(group) > 1}


def _single_source_primitive(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICPrimitive | None:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    if len(coefficients) != 1:
        return None
    primitive_id, coefficient = coefficients[0]
    if abs(float(coefficient) - 1.0) > 1.0e-12:
        return None
    return primitive_by_id.get(primitive_id)


def _local_symmetry_signature(
    primitive: GICPrimitive,
    *,
    atom_symbols: tuple[str, ...],
) -> str | None:
    if primitive.family == "STRETCH" and len(primitive.atoms) == 2:
        return "R:" + "-".join(_sorted_atom_symbols(primitive.atoms, atom_symbols))
    if primitive.family in {"BEND", "CYCLIC_BEND", "SPIRO_BEND"} and len(primitive.atoms) == 3:
        end_symbols = _sorted_atom_symbols(
            (primitive.atoms[0], primitive.atoms[2]),
            atom_symbols,
        )
        return (
            f"{primitive.family}:A:"
            f"{_atom_symbol(primitive.atoms[1], atom_symbols)}:"
            f"{'-'.join(end_symbols)}"
        )
    if primitive.family == "LINEAR_BEND" and len(primitive.atoms) == 3:
        end_symbols = _sorted_atom_symbols(
            (primitive.atoms[0], primitive.atoms[2]),
            atom_symbols,
        )
        return (
            f"L:{primitive.mode}:{_atom_symbol(primitive.atoms[1], atom_symbols)}:"
            f"{'-'.join(end_symbols)}"
        )
    if (
        primitive.family
        in {
            "TORSION",
            "CYCLIC_TORSION",
            "CONDENSED_RING_TORSION",
            "BUTTERFLY",
        }
        and len(primitive.atoms) == 4
    ):
        return f"{primitive.family}:D:" + "-".join(
            _atom_symbol(atom, atom_symbols) for atom in primitive.atoms
        )
    if primitive.family in {"OUT_OF_PLANE", "IMPROPER_DIHEDRAL"} and len(primitive.atoms) == 4:
        substituents = _sorted_atom_symbols(primitive.atoms[1:], atom_symbols)
        return (
            f"{primitive.function}:{_atom_symbol(primitive.atoms[0], atom_symbols)}:"
            f"{'-'.join(substituents)}"
        )
    if primitive.family == "FRAG_DISTANCE":
        left = _atom_multiset_signature(primitive.atoms, atom_symbols)
        right = _atom_multiset_signature(primitive.ref_atoms, atom_symbols)
        pair = tuple(sorted((left, right)))
        return f"FC_DIST:{pair[0]}:{pair[1]}"
    if primitive.family == "FRAG_CENTER_ATOM_DISTANCE":
        return (
            "FCA_DIST:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    if primitive.family == "FRAG_TRANSLATION":
        return (
            f"FTRANS:{primitive.mode}:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    if primitive.family == "FRAG_ORIENTATION":
        return (
            f"FROT:{primitive.mode}:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.frame_atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_frame_atoms, atom_symbols)}"
        )
    if primitive.family == "CENTER_ATOM_DISTANCE":
        return (
            "CENTER_ATOM_DIST:"
            f"{_atom_multiset_signature(primitive.atoms, atom_symbols)}:"
            f"{_atom_multiset_signature(primitive.ref_atoms, atom_symbols)}"
        )
    return None


def _symmetrized_group_gics(
    key: tuple[str, str, str],
    group: tuple[FrozenGIC, ...],
    *,
    first_index: int,
    name_counters: dict[tuple[str, str, str], int],
    point_group: str,
) -> tuple[FrozenGIC, ...]:
    _block, family, _signature = key
    size = len(group)
    output: list[FrozenGIC] = []
    symmetric_weight = 1.0 / np.sqrt(float(size))
    symmetric_irrep = _local_symmetry_irrep(point_group=point_group, kind="S", index=0)
    output.append(
        FrozenGIC(
            identifier=f"GIC{first_index:03d}",
            name=_next_symmetrized_name(family, "S", name_counters, irrep=symmetric_irrep),
            family=family,
            irrep=symmetric_irrep,
            primitive_id=group[0].primitive_id,
            gaussian_expression="LINEAR_COMBINATION",
            coefficients=_combine_gic_coefficients(
                group,
                tuple(symmetric_weight for _idx in range(size)),
            ),
        )
    )
    for group_index in range(1, size):
        weights = [0.0 for _idx in group]
        weights[group_index - 1] = 1.0 / np.sqrt(2.0)
        weights[group_index] = -1.0 / np.sqrt(2.0)
        difference_irrep = _local_symmetry_irrep(
            point_group=point_group,
            kind="D",
            index=group_index - 1,
        )
        output.append(
            FrozenGIC(
                identifier=f"GIC{first_index + group_index:03d}",
                name=_next_symmetrized_name(
                    family,
                    "D",
                    name_counters,
                    irrep=difference_irrep,
                ),
                family=family,
                irrep=difference_irrep,
                primitive_id=group[group_index - 1].primitive_id,
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=_combine_gic_coefficients(group, tuple(weights)),
            )
        )
    return tuple(output)


def _combine_gic_coefficients(
    gics: tuple[FrozenGIC, ...],
    weights: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = {}
    order: list[str] = []
    for gic, weight in zip(gics, weights):
        if abs(weight) <= 1.0e-14:
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            if primitive_id not in totals:
                order.append(primitive_id)
                totals[primitive_id] = 0.0
            totals[primitive_id] += float(weight) * float(coefficient)
    return tuple(
        (primitive_id, totals[primitive_id])
        for primitive_id in order
        if abs(totals[primitive_id]) > 1.0e-14
    )


def _renumber_frozen_gic(gic: FrozenGIC, index: int) -> FrozenGIC:
    return FrozenGIC(
        identifier=f"GIC{index:03d}",
        name=gic.name,
        family=gic.family,
        irrep=gic.irrep,
        primitive_id=gic.primitive_id,
        gaussian_expression=gic.gaussian_expression,
        coefficients=gic.coefficients,
    )


def _prefix_symmetrized_singletons(
    gics: tuple[FrozenGIC, ...],
    *,
    point_group: str,
) -> tuple[FrozenGIC, ...]:
    return tuple(_prefix_symmetrized_gic(gic, point_group=point_group) for gic in gics)


def _prefix_symmetrized_gic(gic: FrozenGIC, *, point_group: str) -> FrozenGIC:
    if gic.family == "LOCAL_XH_STRETCH":
        return gic
    irrep = (
        gic.irrep if gic.irrep and gic.irrep != "UNASSIGNED" else total_symmetric_irrep(point_group)
    )
    prefix = irrep_name_prefix(irrep)
    name = gic.name if gic.name.startswith(prefix) else f"{prefix}{gic.name}"
    return FrozenGIC(
        identifier=gic.identifier,
        name=name,
        family=gic.family,
        irrep=irrep,
        primitive_id=gic.primitive_id,
        gaussian_expression=gic.gaussian_expression,
        coefficients=gic.coefficients,
    )


def _next_symmetrized_name(
    family: str,
    kind: str,
    counters: dict[tuple[str, str, str], int],
    *,
    irrep: str,
) -> str:
    key = (family, kind, irrep)
    counters[key] = counters.get(key, 0) + 1
    return f"{irrep_name_prefix(irrep)}{primitive_prefix(family)}{kind}{counters[key]:03d}"


def _local_symmetry_irrep(
    *,
    point_group: str,
    kind: str,
    index: int,
) -> str:
    if kind == "S":
        return total_symmetric_irrep(point_group)
    non_total = non_total_irrep_sequence(point_group)
    if not non_total:
        return total_symmetric_irrep(point_group)
    return non_total[index % len(non_total)]


def _sorted_atom_symbols(
    atoms: tuple[int, ...],
    atom_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(_atom_symbol(atom, atom_symbols) for atom in atoms))


def _atom_multiset_signature(
    atoms: tuple[int, ...],
    atom_symbols: tuple[str, ...],
) -> str:
    if not atoms:
        return "NONE"
    return ".".join(_sorted_atom_symbols(atoms, atom_symbols))


def _atom_symbol(atom: int, atom_symbols: tuple[str, ...]) -> str:
    if 1 <= atom <= len(atom_symbols):
        return atom_symbols[atom - 1].upper()
    return f"A{atom}"


def _try_select_ranked_primitive(
    primitive: GICPrimitive,
    coords: np.ndarray,
    selected: list[GICPrimitive],
    basis: list[np.ndarray],
    rank: int,
    *,
    rank_tolerance: float,
) -> tuple[int, str]:
    normalized = _normalized_b_row_or_none(
        primitive,
        coords,
        rank_tolerance=rank_tolerance,
    )
    if normalized is None:
        return rank, "singular"
    orthonormal = _orthonormal_residual_or_none(
        basis,
        normalized,
        rank_tolerance=rank_tolerance,
    )
    if orthonormal is None:
        return rank, "dependent"
    selected.append(primitive)
    basis.append(orthonormal)
    return rank + 1, "selected"


def _raise_if_remaining_special_independent(
    primitives: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    basis: list[np.ndarray],
    rank: int,
    *,
    rank_tolerance: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    skipped_singular: list[str] = []
    skipped_dependent: list[str] = []
    skipped_singular_details: list[str] = []
    skipped_dependent_details: list[str] = []
    for primitive in primitives:
        normalized = _normalized_b_row_or_none(
            primitive,
            coords,
            rank_tolerance=rank_tolerance,
        )
        if normalized is None:
            skipped_singular.append(primitive.identifier)
            skipped_singular_details.append(_primitive_diagnostic_token(primitive))
            continue
        if (
            _orthonormal_residual_or_none(
                basis,
                normalized,
                rank_tolerance=rank_tolerance,
            )
            is not None
        ):
            raise GICForgeContractError(
                "protected special primitive set exceeds the vibrational rank: "
                f"{primitive.identifier} {primitive.name} would add an independent "
                "row after the target rank was reached"
            )
        skipped_dependent.append(primitive.identifier)
        skipped_dependent_details.append(_primitive_diagnostic_token(primitive))
    return (
        tuple(skipped_singular),
        tuple(skipped_dependent),
        tuple(skipped_singular_details),
        tuple(skipped_dependent_details),
    )


def _normalized_b_row_or_none(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    rank_tolerance: float,
) -> np.ndarray | None:
    try:
        row = _analytic_b_row(primitive, coords)
    except FloatingPointError:
        return None
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm) or norm <= rank_tolerance:
        return None
    return row / norm


def _orthonormal_residual_or_none(
    basis: list[np.ndarray],
    normalized: np.ndarray,
    *,
    rank_tolerance: float,
) -> np.ndarray | None:
    residual = np.array(normalized, dtype=float, copy=True)
    for vector in basis:
        residual -= float(np.dot(residual, vector)) * vector
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm <= rank_tolerance:
        return None
    return residual / norm


def _sparse_analytic_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> SparseBRow:
    coords = np.asarray(coords, dtype=float)
    natoms = int(coords.shape[0])
    if primitive.function == "R":
        i, j = (atom - 1 for atom in primitive.atoms)
        delta = coords[i] - coords[j]
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("zero-length distance coordinate")
        unit = delta / distance
        return SparseBRow.from_atom_gradients(
            natoms,
            ((primitive.atoms[0], unit), (primitive.atoms[1], -unit)),
        )
    if primitive.function in {"FC_DIST"}:
        center = _fragment_center(coords, primitive.atoms)
        ref_center = _fragment_center(coords, primitive.ref_atoms)
        delta = center - ref_center
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("coincident fragment centers")
        unit = delta / distance
        gradients = tuple((atom, unit / len(primitive.atoms)) for atom in primitive.atoms) + tuple(
            (atom, -unit / len(primitive.ref_atoms)) for atom in primitive.ref_atoms
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    if primitive.function in {"FCA_DIST", "CENTER_ATOM_DIST"}:
        if len(primitive.ref_atoms) != 1:
            raise FloatingPointError("center-atom distance needs exactly one reference atom")
        atom = primitive.ref_atoms[0]
        delta = _fragment_center(coords, primitive.atoms) - coords[atom - 1]
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("fragment center and atom are coincident")
        unit = delta / distance
        gradients = tuple((item, unit / len(primitive.atoms)) for item in primitive.atoms) + (
            (atom, -unit),
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    if primitive.function == "FTRANS":
        axis = np.zeros(3, dtype=float)
        axis[primitive.mode] = 1.0
        gradients = tuple((atom, axis / len(primitive.atoms)) for atom in primitive.atoms) + tuple(
            (atom, -axis / len(primitive.ref_atoms)) for atom in primitive.ref_atoms
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    return SparseBRow.from_dense(
        _analytic_b_row(primitive, coords, reference_coords=reference_coords)
    )


def _analytic_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if primitive.function == "R":
        return _distance_b_row(coords, primitive.atoms)
    if primitive.function == "A":
        return _angle_b_row(coords, primitive.atoms)
    if primitive.function == "FC_DIST":
        return _fragment_center_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FCA_DIST":
        return _fragment_center_atom_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "CENTER_ATOM_DIST":
        return _fragment_center_atom_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FTRANS":
        return _fragment_translation_b_row(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
        )
    if primitive.function == "RPCB":
        return _angle_component_b_row(primitive, coords)
    if primitive.function == "RPCK":
        return _ring_pucker_component_b_row(primitive, coords)
    return _dual_b_row(primitive, coords, reference_coords=reference_coords)


def _distance_b_row(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    i, j = (atom - 1 for atom in atoms)
    delta = coords[i] - coords[j]
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length distance coordinate")
    row = np.zeros(coords.size, dtype=float)
    unit = delta / distance
    row[3 * i : 3 * i + 3] = unit
    row[3 * j : 3 * j + 3] = -unit
    return row


def _angle_b_row(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    i, j, k = (atom - 1 for atom in atoms)
    rji = coords[i] - coords[j]
    rjk = coords[k] - coords[j]
    dji = float(np.linalg.norm(rji))
    djk = float(np.linalg.norm(rjk))
    if dji <= RANK_TOLERANCE or djk <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length angle arm")
    eji = rji / dji
    ejk = rjk / djk
    cosine = float(np.clip(np.dot(eji, ejk), -1.0, 1.0))
    sine = float(np.sqrt(max(1.0 - cosine * cosine, 0.0)))
    if sine <= RANK_TOLERANCE:
        raise FloatingPointError("linear angle has no ordinary bend derivative")
    gi = (cosine * eji - ejk) / (dji * sine)
    gk = (cosine * ejk - eji) / (djk * sine)
    row = np.zeros(coords.size, dtype=float)
    row[3 * i : 3 * i + 3] = gi
    row[3 * k : 3 * k + 3] = gk
    row[3 * j : 3 * j + 3] = -(gi + gk)
    return row


def _fragment_center_distance_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> np.ndarray:
    center = _fragment_center(coords, atoms)
    ref_center = _fragment_center(coords, ref_atoms)
    delta = center - ref_center
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("coincident fragment centers")
    unit = delta / distance
    row = np.zeros(coords.size, dtype=float)
    _accumulate_center_gradient(row, atoms, unit / len(atoms))
    _accumulate_center_gradient(row, ref_atoms, -unit / len(ref_atoms))
    return row


def _fragment_center_atom_distance_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> np.ndarray:
    if len(ref_atoms) != 1:
        raise FloatingPointError("center-atom distance needs exactly one reference atom")
    atom = ref_atoms[0]
    delta = _fragment_center(coords, atoms) - coords[atom - 1]
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("fragment center and atom are coincident")
    unit = delta / distance
    row = np.zeros(coords.size, dtype=float)
    _accumulate_center_gradient(row, atoms, unit / len(atoms))
    row[3 * (atom - 1) : 3 * atom] -= unit
    return row


def _fragment_translation_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    axis = np.zeros(3, dtype=float)
    axis[mode] = 1.0
    _accumulate_center_gradient(row, atoms, axis / len(atoms))
    _accumulate_center_gradient(row, ref_atoms, -axis / len(ref_atoms))
    return row


def _ring_pucker_component_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        if coefficient == 0.0:
            continue
        term = GICPrimitive(
            identifier=f"{primitive.identifier}_D",
            name="RPckD",
            family="TORSION",
            function="D",
            atoms=atoms,
        )
        row += coefficient * _dual_b_row(term, coords)
    return row


def _angle_component_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for coefficient, atoms in _angle_component_terms_from_refs(primitive):
        if coefficient == 0.0:
            continue
        row += coefficient * _angle_b_row(coords, atoms)
    return row


def _accumulate_center_gradient(
    row: np.ndarray,
    atoms: tuple[int, ...],
    gradient: np.ndarray,
) -> None:
    for atom in atoms:
        start = 3 * (atom - 1)
        row[start : start + 3] += gradient


def _dual_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> np.ndarray:
    dcoords = _dual_coordinates(coords)
    value = _dual_primitive_value(
        primitive,
        dcoords,
        coords,
        reference_coords=reference_coords,
    )
    if not np.isfinite(value.val):
        raise FloatingPointError("non-finite analytic derivative value")
    return np.asarray(value.der, dtype=float)


def _finite_difference_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    step_angstrom: float = DIAGNOSTIC_FINITE_DIFFERENCE_STEP,
) -> np.ndarray:
    flat = np.asarray(coords, dtype=float).reshape(-1)
    base = _primitive_value(primitive, coords)
    row = np.zeros_like(flat)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += step_angstrom
        minus[idx] -= step_angstrom
        try:
            value_plus = _primitive_value(primitive, plus.reshape(coords.shape))
            value_minus = _primitive_value(primitive, minus.reshape(coords.shape))
        except FloatingPointError:
            row[idx] = np.nan
            continue
        if primitive.function == "RPCK":
            delta = _ring_pucker_component_periodic_delta(
                primitive,
                plus.reshape(coords.shape),
                minus.reshape(coords.shape),
            )
        elif primitive.function in {"D", "IMPD"}:
            delta = _periodic_delta(value_plus, value_minus)
        else:
            delta = value_plus - value_minus
        if not np.isfinite(delta) or not np.isfinite(base):
            row[idx] = np.nan
        else:
            row[idx] = delta / (2.0 * step_angstrom)
    return row


def _ring_pucker_component_periodic_delta(
    primitive: GICPrimitive,
    plus_coords: np.ndarray,
    minus_coords: np.ndarray,
) -> float:
    delta = 0.0
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        delta += coefficient * _periodic_delta(
            _dihedral_value(plus_coords, atoms),
            _dihedral_value(minus_coords, atoms),
        )
    return float(delta)


class _Dual:
    __slots__ = ("val", "der")

    def __init__(self, val: float, der: np.ndarray):
        self.val = float(val)
        self.der = np.asarray(der, dtype=float)

    def _coerce(self, other: object) -> "_Dual":
        if isinstance(other, _Dual):
            return other
        return _Dual(float(other), np.zeros_like(self.der))

    def __add__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val + rhs.val, self.der + rhs.der)

    def __radd__(self, other: object) -> "_Dual":
        return self.__add__(other)

    def __sub__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val - rhs.val, self.der - rhs.der)

    def __rsub__(self, other: object) -> "_Dual":
        lhs = self._coerce(other)
        return _Dual(lhs.val - self.val, lhs.der - self.der)

    def __mul__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val * rhs.val, self.val * rhs.der + rhs.val * self.der)

    def __rmul__(self, other: object) -> "_Dual":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        inv = 1.0 / rhs.val
        return _Dual(self.val * inv, (self.der - self.val * rhs.der * inv) * inv)

    def __rtruediv__(self, other: object) -> "_Dual":
        lhs = self._coerce(other)
        inv = 1.0 / self.val
        return _Dual(lhs.val * inv, (lhs.der - lhs.val * self.der * inv) * inv)

    def __neg__(self) -> "_Dual":
        return _Dual(-self.val, -self.der)


def _dual_coordinates(coords: np.ndarray) -> list[list[_Dual]]:
    flat = np.asarray(coords, dtype=float).reshape(-1)
    dim = flat.size
    out: list[list[_Dual]] = []
    for atom in range(coords.shape[0]):
        row = []
        for axis in range(3):
            idx = 3 * atom + axis
            der = np.zeros(dim, dtype=float)
            der[idx] = 1.0
            row.append(_Dual(float(coords[atom, axis]), der))
        out.append(row)
    return out


def _dual_primitive_value(
    primitive: GICPrimitive,
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> _Dual:
    if primitive.function == "L":
        return _dual_linear_bend_value(dcoords, primitive.atoms, mode=primitive.mode)
    if primitive.function == "D":
        return _dual_dihedral_value(dcoords, primitive.atoms)
    if primitive.function == "U":
        return _dual_out_of_plane_value(dcoords, primitive.atoms)
    if primitive.function == "IMPD":
        return _dual_dihedral_value(dcoords, _improper_dihedral_atoms(primitive.atoms))
    if primitive.function == "FROT":
        return _dual_fragment_rotation_value(
            dcoords,
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
            reference_coords=reference_coords,
        )
    raise GICForgeContractError(
        f"analytic B row is not implemented for function {primitive.function}"
    )


def _dual_linear_bend_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    *,
    mode: int,
) -> _Dual:
    i, j, k = (atom - 1 for atom in atoms)
    left = _d_unit(_d_vec_sub(dcoords[i], dcoords[j]))
    right = _d_unit(_d_vec_sub(dcoords[k], dcoords[j]))
    axis = _d_unit(_d_vec_sub(right, left))
    e1, e2 = _d_orthogonal_frame(axis)
    bend = _d_vec_add(left, right)
    return _d_dot(bend, e1 if mode == -1 else e2)


def _dual_dihedral_value(dcoords: list[list[_Dual]], atoms: tuple[int, ...]) -> _Dual:
    i, j, k, ell = (atom - 1 for atom in atoms)
    p0, p1, p2, p3 = dcoords[i], dcoords[j], dcoords[k], dcoords[ell]
    b0 = _d_vec_neg(_d_vec_sub(p1, p0))
    b1 = _d_vec_sub(p2, p1)
    b2 = _d_vec_sub(p3, p2)
    b1 = _d_unit(b1)
    v = _d_vec_sub(b0, _d_vec_scale(b1, _d_dot(b0, b1)))
    w = _d_vec_sub(b2, _d_vec_scale(b1, _d_dot(b2, b1)))
    x = _d_dot(v, w)
    y = _d_dot(_d_cross(b1, v), w)
    return _d_atan2(y, x)


def _dual_out_of_plane_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
) -> _Dual:
    center, n1, n2, n3 = (atom - 1 for atom in atoms)
    r1 = _d_vec_sub(dcoords[n1], dcoords[center])
    r2 = _d_vec_sub(dcoords[n2], dcoords[center])
    r3 = _d_vec_sub(dcoords[n3], dcoords[center])
    normal = _d_unit(_d_cross(r2, r3))
    return _d_asin(_d_dot(_d_unit(r1), normal))


def _dual_fragment_rotation_value(
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
    reference_coords: np.ndarray | None = None,
) -> _Dual:
    frame_atoms = frame_atoms or _fragment_frame_anchor_atoms(atoms, coords=coords)
    ref_frame_atoms = ref_frame_atoms or _fragment_frame_anchor_atoms(ref_atoms, coords=coords)
    frame_frag = _d_fragment_frame(dcoords, atoms, frame_atoms=frame_atoms)
    frame_ref = _d_fragment_frame(dcoords, ref_atoms, frame_atoms=ref_frame_atoms)
    rotation = [
        [_d_dot(frame_frag[left], frame_ref[right]) for right in range(3)] for left in range(3)
    ]
    if reference_coords is not None:
        reference_frag = _fragment_frame(
            np.asarray(reference_coords, dtype=float),
            atoms,
            frame_atoms=frame_atoms,
        )
        reference_ref = _fragment_frame(
            np.asarray(reference_coords, dtype=float),
            ref_atoms,
            frame_atoms=ref_frame_atoms,
        )
        reference_rotation = reference_frag.T @ reference_ref
        rotation = [
            [
                sum(rotation[left][axis] * reference_rotation[right, axis] for axis in range(3))
                for right in range(3)
            ]
            for left in range(3)
        ]
    return _d_rotation_vector(rotation)[mode]


def _d_fragment_frame(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...],
) -> list[list[_Dual]]:
    p_atom, q_atom = frame_atoms
    center = _d_fragment_center(dcoords, atoms)
    p_axis = _d_unit(_d_vec_sub(dcoords[p_atom - 1], center))
    q_raw = _d_cross(p_axis, _d_vec_sub(dcoords[q_atom - 1], center))
    q_axis = _d_unit(q_raw)
    s_axis = _d_unit(_d_cross(p_axis, q_axis))
    return [p_axis, q_axis, s_axis]


def _d_fragment_center(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
) -> list[_Dual]:
    if not atoms:
        raise FloatingPointError("fragment has no atoms")
    total = [_d_zero_like(dcoords[atoms[0] - 1][0]) for _axis in range(3)]
    for atom in atoms:
        total = _d_vec_add(total, dcoords[atom - 1])
    return _d_vec_scale(total, 1.0 / len(atoms))


def _d_quaternion_vector(rotation: list[list[_Dual]]) -> tuple[_Dual, _Dual, _Dual]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace.val <= -1.0 + RANK_TOLERANCE:
        raise FloatingPointError("fragment quaternion is singular near 180 degrees")
    kw = 0.5 * _d_sqrt(trace + 1.0)
    denom = 4.0 * kw
    return (
        (rotation[1][2] - rotation[2][1]) / denom,
        (rotation[2][0] - rotation[0][2]) / denom,
        (rotation[0][1] - rotation[1][0]) / denom,
    )


def _d_rotation_vector(rotation: list[list[_Dual]]) -> tuple[_Dual, _Dual, _Dual]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace.val <= -1.0 + RANK_TOLERANCE:
        raise FloatingPointError("fragment exponential map is singular near 180 degrees")
    kw = 0.5 * _d_sqrt(trace + 1.0)
    kx, ky, kz = _d_quaternion_vector(rotation)
    kn2 = kx * kx + ky * ky + kz * kz
    if kn2.val <= RANK_TOLERANCE * RANK_TOLERANCE:
        return 2.0 * kx, 2.0 * ky, 2.0 * kz
    kn = _d_sqrt(kn2)
    factor = (2.0 * _d_atan2(kn, kw)) / kn
    return factor * kx, factor * ky, factor * kz


def _d_zero_like(value: _Dual) -> _Dual:
    return _Dual(0.0, np.zeros_like(value.der))


def _d_vec_add(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [left[idx] + right[idx] for idx in range(3)]


def _d_vec_sub(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [left[idx] - right[idx] for idx in range(3)]


def _d_vec_neg(vector: list[_Dual]) -> list[_Dual]:
    return [-item for item in vector]


def _d_vec_scale(vector: list[_Dual], scale: float | _Dual) -> list[_Dual]:
    return [scale * item for item in vector]


def _d_dot(left: list[_Dual], right: list[_Dual]) -> _Dual:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _d_cross(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _d_norm(vector: list[_Dual]) -> _Dual:
    return _d_sqrt(_d_dot(vector, vector))


def _d_unit(vector: list[_Dual]) -> list[_Dual]:
    norm = _d_norm(vector)
    if norm.val <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length vector")
    return [item / norm for item in vector]


def _d_orthogonal_frame(axis: list[_Dual]) -> tuple[list[_Dual], list[_Dual]]:
    trial = [1.0, 0.0, 0.0]
    if abs(axis[0].val) > 0.9:
        trial = [0.0, 1.0, 0.0]
    e1 = _d_unit(_d_cross(axis, [_d_constant(axis[0], value) for value in trial]))
    e2 = _d_unit(_d_cross(axis, e1))
    return e1, e2


def _d_constant(template: _Dual, value: float) -> _Dual:
    return _Dual(value, np.zeros_like(template.der))


def _d_sqrt(value: _Dual) -> _Dual:
    if value.val <= 0.0:
        raise FloatingPointError("square root of non-positive value")
    root = float(np.sqrt(value.val))
    return _Dual(root, value.der / (2.0 * root))


def _d_acos(value: _Dual) -> _Dual:
    clipped = float(np.clip(value.val, -1.0, 1.0))
    denom = float(np.sqrt(max(1.0 - clipped * clipped, 0.0)))
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("acos derivative is singular")
    return _Dual(float(np.arccos(clipped)), -value.der / denom)


def _d_asin(value: _Dual) -> _Dual:
    clipped = float(np.clip(value.val, -1.0, 1.0))
    denom = float(np.sqrt(max(1.0 - clipped * clipped, 0.0)))
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("asin derivative is singular")
    return _Dual(float(np.arcsin(clipped)), value.der / denom)


def _d_atan2(y_value: _Dual, x_value: _Dual) -> _Dual:
    denom = x_value.val * x_value.val + y_value.val * y_value.val
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("atan2 derivative is singular")
    der = (x_value.val * y_value.der - y_value.val * x_value.der) / denom
    return _Dual(float(np.arctan2(y_value.val, x_value.val)), der)


def _primitive_value(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> float:
    if primitive.function == "R":
        return _distance_value(coords, primitive.atoms)
    if primitive.function == "A":
        return _angle_value(coords, primitive.atoms)
    if primitive.function == "L":
        return _linear_bend_value(coords, primitive.atoms, mode=primitive.mode)
    if primitive.function == "D":
        return _dihedral_value(coords, primitive.atoms)
    if primitive.function == "U":
        return _out_of_plane_value(coords, primitive.atoms)
    if primitive.function == "IMPD":
        return _dihedral_value(coords, _improper_dihedral_atoms(primitive.atoms))
    if primitive.function == "FC_DIST":
        return _fragment_center_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FCA_DIST":
        return _fragment_center_atom_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "CENTER_ATOM_DIST":
        return _fragment_center_atom_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FTRANS":
        return _fragment_translation_value(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
        )
    if primitive.function == "RPCB":
        return _angle_component_value(primitive, coords)
    if primitive.function == "RPCK":
        return _ring_pucker_component_value(primitive, coords)
    if primitive.function == "FROT":
        value = _fragment_rotation_value(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
        )
        if reference_coords is None:
            return value
        reference_frag = _fragment_frame(
            np.asarray(reference_coords, dtype=float),
            primitive.atoms,
            frame_atoms=primitive.frame_atoms,
        )
        reference_ref = _fragment_frame(
            np.asarray(reference_coords, dtype=float),
            primitive.ref_atoms,
            frame_atoms=primitive.ref_frame_atoms,
        )
        current_frag = _fragment_frame(
            coords,
            primitive.atoms,
            frame_atoms=primitive.frame_atoms,
        )
        current_ref = _fragment_frame(
            coords,
            primitive.ref_atoms,
            frame_atoms=primitive.ref_frame_atoms,
        )
        delta_rotation = (current_frag.T @ current_ref) @ (reference_frag.T @ reference_ref).T
        return float(_rotation_vector(delta_rotation)[primitive.mode])
    raise GICForgeContractError(f"unsupported primitive function: {primitive.function}")


def _ring_pucker_component_value(primitive: GICPrimitive, coords: np.ndarray) -> float:
    value = 0.0
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        value += coefficient * _dihedral_value(coords, atoms)
    return float(value)


def _angle_component_value(primitive: GICPrimitive, coords: np.ndarray) -> float:
    value = 0.0
    for coefficient, atoms in _angle_component_terms_from_refs(primitive):
        value += coefficient * _angle_value(coords, atoms)
    return float(value)


def _distance_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j = (atom - 1 for atom in atoms)
    return float(np.linalg.norm(coords[i] - coords[j]))


def _angle_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j, k = (atom - 1 for atom in atoms)
    u = coords[i] - coords[j]
    v = coords[k] - coords[j]
    return float(np.arccos(np.clip(_dot_unit(u, v), -1.0, 1.0)))


def _linear_bend_value(coords: np.ndarray, atoms: tuple[int, ...], *, mode: int) -> float:
    i, j, k = (atom - 1 for atom in atoms)
    left = _unit(coords[i] - coords[j])
    right = _unit(coords[k] - coords[j])
    axis = _unit(right - left)
    e1, e2 = _orthogonal_frame(axis)
    bend = left + right
    return float(np.dot(bend, e1 if mode == -1 else e2))


def _dihedral_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j, k, ell = (atom - 1 for atom in atoms)
    p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[ell]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _out_of_plane_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    center, n1, n2, n3 = (atom - 1 for atom in atoms)
    r1 = coords[n1] - coords[center]
    r2 = coords[n2] - coords[center]
    r3 = coords[n3] - coords[center]
    normal = _unit(np.cross(r2, r3))
    return float(np.arcsin(np.clip(np.dot(_unit(r1), normal), -1.0, 1.0)))


def _fragment_center_distance_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> float:
    delta = _fragment_center(coords, atoms) - _fragment_center(coords, ref_atoms)
    return float(np.linalg.norm(delta))


def _fragment_center_atom_distance_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> float:
    if len(ref_atoms) != 1:
        raise FloatingPointError("center-atom distance needs exactly one reference atom")
    return float(np.linalg.norm(_fragment_center(coords, atoms) - coords[ref_atoms[0] - 1]))


def _fragment_translation_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
) -> float:
    delta = _fragment_center(coords, atoms) - _fragment_center(coords, ref_atoms)
    return float(delta[mode])


def _fragment_rotation_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
) -> float:
    frame_frag = _fragment_frame(coords, atoms, frame_atoms=frame_atoms)
    frame_ref = _fragment_frame(coords, ref_atoms, frame_atoms=ref_frame_atoms)
    rotation = frame_frag.T @ frame_ref
    return float(_rotation_vector(rotation)[mode])


def _cartesian_vibrational_basis(coords: np.ndarray, *, target_rank: int) -> np.ndarray:
    flat_dim = int(coords.shape[0] * 3)
    if target_rank == 0:
        return np.zeros((flat_dim, 0), dtype=float)
    centered = coords - np.mean(coords, axis=0)
    rows: list[np.ndarray] = []
    for axis in range(3):
        row = np.zeros((coords.shape[0], 3), dtype=float)
        row[:, axis] = 1.0
        rows.append(row.reshape(-1))
    for axis_vector in np.eye(3):
        row = np.cross(axis_vector, centered).reshape(-1)
        if np.linalg.norm(row) > RANK_TOLERANCE:
            rows.append(row)
    external = np.asarray(rows, dtype=float)
    _, singular_values, vh = np.linalg.svd(external, full_matrices=True)
    rank = int(np.sum(singular_values > RANK_TOLERANCE))
    basis = vh[rank:].T
    if basis.shape[1] < target_rank:
        raise GICForgeContractError(
            f"cannot build SYCART basis: need {target_rank}, got {basis.shape[1]}"
        )
    return basis[:, :target_rank]


def _fragment_primitive_candidates(
    fragment_records: tuple[object, ...],
    *,
    coords: np.ndarray,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    if len(fragment_records) <= 1:
        return ()
    records = sorted(fragment_records, key=lambda item: getattr(item, "identifier"))
    reference = max(
        records,
        key=lambda item: (len(getattr(item, "atoms")), -_fragment_number(item)),
    )
    candidates: list[GICPrimitive] = []

    for record in records:
        if getattr(record, "identifier") == getattr(reference, "identifier"):
            continue
        atoms = tuple(getattr(record, "atoms"))
        ref_atoms = tuple(getattr(reference, "atoms"))
        candidates.append(
            _make_primitive(
                "FRAG_DISTANCE",
                "FC_DIST",
                atoms,
                counters,
                ref_atoms=ref_atoms,
                refs=(getattr(record, "identifier"), getattr(reference, "identifier")),
            )
        )
        for axis in range(3):
            candidates.append(
                _make_primitive(
                    "FRAG_TRANSLATION",
                    "FTRANS",
                    atoms,
                    counters,
                    mode=axis,
                    ref_atoms=ref_atoms,
                    refs=(getattr(record, "identifier"), getattr(reference, "identifier")),
                )
            )
        if (
            _fragment_frame_rank(coords, atoms) >= 2
            and _fragment_frame_rank(coords, ref_atoms) >= 2
        ):
            frame_atoms = _fragment_frame_anchor_atoms(atoms, coords=coords)
            ref_frame_atoms = _fragment_frame_anchor_atoms(ref_atoms, coords=coords)
            for axis in range(3):
                candidates.append(
                    _make_primitive(
                        "FRAG_ORIENTATION",
                        "FROT",
                        atoms,
                        counters,
                        mode=axis,
                        ref_atoms=ref_atoms,
                        refs=(getattr(record, "identifier"), getattr(reference, "identifier")),
                        frame_atoms=frame_atoms,
                        ref_frame_atoms=ref_frame_atoms,
                    )
                )
    return tuple(candidates)


def _interfragment_hbond_distance_candidates(
    fragment_records: tuple[object, ...],
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    records = tuple(
        record
        for record in sorted(fragment_records, key=lambda item: getattr(item, "identifier"))
        if tuple(getattr(record, "atoms", ()))
    )
    if len(records) <= 1:
        return ()
    covalent = {tuple(sorted(bond)) for bond in bonds}
    contacts = _interfragment_hbond_contacts(
        records,
        atom_symbols=atom_symbols,
        coords=coords,
        covalent=covalent,
        fragment_index_by_atom=_fragment_index_by_atom(records),
    )
    candidates: list[GICPrimitive] = []
    for _distance, pair, kind in contacts:
        if kind != "HBOND":
            continue
        candidates.extend(
            _hbond_distance_primitives(((pair, "INTERFRAGMENT_HBOND"),), counters=counters)
        )
    return tuple(candidates)


def _hbond_distance_primitives(
    contacts: tuple[tuple[tuple[int, int], str], ...],
    *,
    counters: dict[str, int],
    ref: str | None = None,
) -> tuple[GICPrimitive, ...]:
    candidates: list[GICPrimitive] = []
    for item in contacts:
        if len(item) == 2 and isinstance(item[0], tuple):
            pair, source = item
        else:
            pair = item[0]  # type: ignore[index]
            source = ref or "HBOND"
        candidates.append(
            _make_primitive(
                "HBOND_DISTANCE",
                "R",
                tuple(sorted(int(atom) for atom in pair)),
                counters,
                refs=(str(source),),
            )
        )
    return tuple(candidates)


def _intramolecular_hbond_contacts(
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], str], ...]:
    if not atom_symbols or len(atom_symbols) != int(coords.shape[0]):
        return ()
    parameters = load_bdpcs3_parameters().hbond
    atomic_numbers = tuple(int(atomic_number(symbol) or 0) for symbol in atom_symbols)
    covalent = {tuple(sorted(bond)) for bond in bonds}
    adjacency = _adjacency(tuple(covalent), natoms=len(atom_symbols))
    selected: list[tuple[int, int, int, float]] = []
    for hydrogen, z_hydrogen in enumerate(atomic_numbers, start=1):
        if z_hydrogen != 1:
            continue
        donors = tuple(
            atom
            for atom in adjacency[hydrogen]
            if atomic_numbers[atom - 1] in parameters.donor_atomic_numbers
        )
        if len(donors) != 1:
            continue
        donor = donors[0]
        best: tuple[int, int, int, float] | None = None
        for acceptor, z_acceptor in enumerate(atomic_numbers, start=1):
            if acceptor in {hydrogen, donor}:
                continue
            if z_acceptor not in parameters.acceptor_atomic_numbers:
                continue
            if tuple(sorted((hydrogen, acceptor))) in covalent:
                continue
            if tuple(sorted((donor, acceptor))) in covalent:
                continue
            if not _same_covalent_component(hydrogen, acceptor, adjacency):
                continue
            distance = float(np.linalg.norm(coords[hydrogen - 1] - coords[acceptor - 1]))
            if distance >= parameters.search_cutoff_ang:
                continue
            angle = float(np.degrees(_angle_value(coords, (donor, hydrogen, acceptor))))
            if angle < parameters.angle_threshold_deg:
                continue
            candidate = (donor, hydrogen, acceptor, distance)
            if best is None or (distance, acceptor) < (best[3], best[2]):
                best = candidate
        if best is not None:
            selected.append(best)
    contacts: list[tuple[tuple[int, int], str]] = []
    seen: set[tuple[int, int]] = set()
    for donor, hydrogen, acceptor, _distance in sorted(
        selected,
        key=lambda item: (item[3], item[0], item[1], item[2]),
    ):
        donor_acceptor = (donor, acceptor)
        if donor_acceptor in seen:
            continue
        seen.add(donor_acceptor)
        contacts.append((tuple(sorted((hydrogen, acceptor))), "INTRAMOLECULAR_HBOND"))
    return tuple(contacts)


def _normalize_pseudo_contact(contact) -> tuple[int, int, str]:
    if len(contact) == 3:
        left, right, kind = contact
        return int(left), int(right), str(kind)
    if len(contact) == 2:
        pair, kind = contact
        left, right = pair
        return int(left), int(right), str(kind)
    raise ValueError(f"invalid pseudo-contact record: {contact!r}")


def _intramolecular_pseudo_bond_contacts(
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], str], ...]:
    if not atom_symbols or len(atom_symbols) != int(coords.shape[0]):
        return ()
    parameters = load_bdpcs3_parameters().hbond
    atomic_numbers = tuple(int(atomic_number(symbol) or 0) for symbol in atom_symbols)
    adjacency = _adjacency(bonds, natoms=len(atom_symbols))
    candidates: list[tuple[float, int, int, int, bool]] = []
    for hydrogen, z_hydrogen in enumerate(atomic_numbers, start=1):
        if z_hydrogen != 1:
            continue
        donors = tuple(adjacency[hydrogen])
        if len(donors) != 1:
            continue
        donor = donors[0]
        for acceptor, z_acceptor in enumerate(atomic_numbers, start=1):
            if acceptor in {hydrogen, donor}:
                continue
            if z_acceptor not in parameters.acceptor_atomic_numbers:
                continue
            if tuple(sorted((hydrogen, acceptor))) in {tuple(sorted(bond)) for bond in bonds}:
                continue
            if not _same_covalent_component(hydrogen, acceptor, adjacency):
                continue
            distance = float(np.linalg.norm(coords[hydrogen - 1] - coords[acceptor - 1]))
            if distance >= min(parameters.distance_cutoff_ang, 2.50):
                continue
            polar_donor = atomic_numbers[donor - 1] in parameters.donor_atomic_numbers
            candidates.append((distance, donor, hydrogen, acceptor, polar_donor))
    selected: list[tuple[tuple[int, int], str]] = []
    seen_donor_acceptor: set[tuple[int, int]] = set()
    seen_pair: set[tuple[int, int]] = set()
    primary_acceptors: set[int] = set()
    for _distance, donor, hydrogen, acceptor, polar_donor in sorted(candidates):
        if not polar_donor:
            continue
        donor_acceptor = tuple(sorted((donor, acceptor)))
        pair = tuple(sorted((hydrogen, acceptor)))
        if donor_acceptor in seen_donor_acceptor or pair in seen_pair:
            continue
        seen_donor_acceptor.add(donor_acceptor)
        seen_pair.add(pair)
        primary_acceptors.add(acceptor)
        selected.append((pair, "INTRAMOLECULAR_PSEUDO_HBOND"))
    for _distance, donor, hydrogen, acceptor, polar_donor in sorted(candidates):
        if polar_donor or acceptor not in primary_acceptors:
            continue
        donor_acceptor = tuple(sorted((donor, acceptor)))
        pair = tuple(sorted((hydrogen, acceptor)))
        if donor_acceptor in seen_donor_acceptor or pair in seen_pair:
            continue
        seen_donor_acceptor.add(donor_acceptor)
        seen_pair.add(pair)
        selected.append((pair, "INTRAMOLECULAR_PSEUDO_HBOND"))
    return tuple(selected)


def _hbond_replaced_valence_angles(
    contacts: tuple[tuple[tuple[int, int], str], ...],
    *,
    bonds: tuple[tuple[int, int], ...],
    natoms: int,
) -> set[tuple[int, int, int]]:
    adjacency = _adjacency(bonds, natoms=natoms)
    replaced: set[tuple[int, int, int]] = set()
    for pair, _source in contacts:
        hydrogen, acceptor = tuple(int(atom) for atom in pair)
        path = _shortest_covalent_path(hydrogen, acceptor, adjacency)
        if len(path) >= 3:
            replaced.add(_angle_key((path[0], path[1], path[2])))
    return replaced


def _angle_key(atoms: tuple[int, int, int]) -> tuple[int, int, int]:
    left, center, right = (int(atom) for atom in atoms)
    ends = tuple(sorted((left, right)))
    return (ends[0], center, ends[1])


def _same_covalent_component(left: int, right: int, adjacency: dict[int, set[int]]) -> bool:
    return bool(_shortest_covalent_path(left, right, adjacency))


def _shortest_covalent_path(
    start: int,
    stop: int,
    adjacency: dict[int, set[int]],
) -> tuple[int, ...]:
    queue: list[tuple[int, ...]] = [(int(start),)]
    seen = {int(start)}
    while queue:
        path = queue.pop(0)
        atom = path[-1]
        if atom == stop:
            return path
        for neighbor in sorted(adjacency.get(atom, ())):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((*path, neighbor))
    return ()


def _interaction_center_primitive_candidates(
    definition: object | None,
    *,
    counters: dict[str, int],
) -> tuple[GICPrimitive, ...]:
    if definition is None:
        return ()
    centers = {
        getattr(center, "identifier"): center for center in getattr(definition, "centers", ())
    }
    candidates: list[GICPrimitive] = []
    for interaction in getattr(definition, "interactions", ()):
        center = centers.get(getattr(interaction, "center_id"))
        if center is None:
            continue
        atom = int(getattr(interaction, "atom"))
        atoms = tuple(int(item) for item in getattr(center, "atoms"))
        candidates.append(
            _make_primitive(
                "CENTER_ATOM_DISTANCE",
                "CENTER_ATOM_DIST",
                atoms,
                counters,
                ref_atoms=(atom,),
                refs=(getattr(center, "identifier"), f"A{atom}"),
            )
        )
    return tuple(candidates)


def _fragment_records(path: Path) -> tuple[object, ...]:
    try:
        from matrix_fragments import read_fragment_records
    except ImportError:
        return ()
    return tuple(read_fragment_records(Path(path)))


def _pseudo_bonds_for_fragments(
    fragment_records: tuple[object, ...],
    *,
    bonds: tuple[tuple[int, int], ...],
    coords: np.ndarray,
    atom_symbols: tuple[str, ...],
    additional_contacts: int = 0,
) -> tuple[tuple[int, int, str], ...]:
    records = tuple(
        record
        for record in sorted(fragment_records, key=lambda item: getattr(item, "identifier"))
        if tuple(getattr(record, "atoms", ()))
    )
    if len(records) <= 1:
        return ()
    covalent = {tuple(sorted(bond)) for bond in bonds}
    fragment_index_by_atom = _fragment_index_by_atom(records)
    hbond_contacts = _interfragment_hbond_contacts(
        records,
        atom_symbols=atom_symbols,
        coords=coords,
        covalent=covalent,
        fragment_index_by_atom=fragment_index_by_atom,
    )
    hbond_edges: list[tuple[int, float, int, int, tuple[int, int], str]] = []
    for distance, pair, kind in hbond_contacts:
        key = tuple(
            sorted(
                (
                    fragment_index_by_atom[pair[0]],
                    fragment_index_by_atom[pair[1]],
                )
            )
        )
        if key[0] == key[1]:
            continue
        hbond_edges.append((0, distance, key[0], key[1], pair, kind))
    fragment_edges: list[tuple[int, float, int, int, tuple[int, int], str]] = []
    closure_edges: list[tuple[int, float, int, int, tuple[int, int], str]] = []
    for left_index, left in enumerate(records):
        left_atoms = tuple(int(atom) for atom in getattr(left, "atoms"))
        for right_index, right in enumerate(records[left_index + 1 :], start=left_index + 1):
            right_atoms = tuple(int(atom) for atom in getattr(right, "atoms"))
            candidates = _interfragment_atom_pair_candidates(
                left_atoms,
                right_atoms,
                covalent=covalent,
                coords=coords,
            )
            if not candidates:
                continue
            distance, pair = candidates[0]
            fragment_edges.append(
                (1, distance, left_index, right_index, pair, "INTERFRAGMENT_CLOSEST")
            )
            closure_edges.extend(
                (2, distance, left_index, right_index, candidate_pair, "PSEUDO_CYCLE_CLOSURE")
                for distance, candidate_pair in candidates
            )
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    selected: list[tuple[int, int, str]] = []
    for _priority, _distance, left_index, right_index, pair, kind in sorted(hbond_edges):
        selected.append((pair[0], pair[1], kind))
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root
    for _priority, _distance, left_index, right_index, pair, kind in sorted(fragment_edges):
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        selected.append((pair[0], pair[1], kind))
        if len(selected) == len(records) - 1:
            break
    roots = {find(index) for index in range(len(records))}
    if len(roots) != 1:
        raise GICForgeContractError(
            "cannot connect all fragments with pseudo-bonds from H-bonds/closest atom pairs"
        )
    selected_set = {tuple(sorted((left, right))) for left, right, _kind in selected}
    graph_bonds = tuple(sorted(covalent | selected_set))
    if not _pseudo_cycles_from_pseudo_bonds(
        graph_bonds,
        pseudo_bonds=tuple(selected_set),
        natoms=coords.shape[0],
    ):
        for _priority, distance, _left_index, _right_index, pair, kind in sorted(
            closure_edges
        ):
            canonical = tuple(sorted(pair))
            if canonical in selected_set:
                continue
            if not _pseudo_cycle_closure_allowed(
                canonical,
                distance=distance,
                atom_symbols=atom_symbols,
            ):
                continue
            trial_selected = selected_set | {canonical}
            graph_bonds = tuple(sorted(covalent | trial_selected))
            if _pseudo_cycles_from_pseudo_bonds(
                graph_bonds,
                pseudo_bonds=tuple(trial_selected),
                natoms=coords.shape[0],
            ):
                selected.append((canonical[0], canonical[1], kind))
                selected_set = trial_selected
                break
    admissible: list[tuple[float, tuple[int, int], str]] = []
    for _priority, distance, _left_index, _right_index, pair, kind in sorted(closure_edges):
        canonical = tuple(sorted(pair))
        if canonical in selected_set:
            continue
        if not _pseudo_cycle_closure_allowed(
            canonical,
            distance=distance,
            atom_symbols=atom_symbols,
        ):
            continue
        admissible.append((distance, canonical, kind))
    for _ in range(max(0, int(additional_contacts))):
        if not admissible:
            break
        best_index = max(
            range(len(admissible)),
            key=lambda index: _pseudo_bond_conditioning_score(
                tuple((left, right) for left, right, _kind in selected)
                + (admissible[index][1],),
                coords=coords,
                natoms=len(atom_symbols),
            ),
        )
        _distance, pair, kind = admissible.pop(best_index)
        if pair in selected_set:
            continue
        selected.append((pair[0], pair[1], kind))
        selected_set.add(pair)
    return tuple(sorted(set(selected)))


def _pseudo_bond_conditioning_score(
    pairs: tuple[tuple[int, int], ...],
    *,
    coords: np.ndarray,
    natoms: int,
) -> tuple[int, float]:
    rows: list[np.ndarray] = []
    for left, right in pairs:
        delta = np.asarray(coords[right - 1] - coords[left - 1], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-12:
            continue
        unit = delta / norm
        row = np.zeros(3 * natoms, dtype=float)
        row[3 * (left - 1) : 3 * left] = -unit
        row[3 * (right - 1) : 3 * right] = unit
        rows.append(row)
    if not rows:
        return (0, 0.0)
    singular_values = np.linalg.svd(np.asarray(rows), compute_uv=False)
    nonzero = singular_values[singular_values > 1.0e-10]
    return (int(nonzero.size), float(nonzero[-1]) if nonzero.size else 0.0)


def _pseudo_cycle_closure_allowed(
    pair: tuple[int, int],
    *,
    distance: float,
    atom_symbols: tuple[str, ...],
) -> bool:
    left, right = pair
    if left < 1 or right < 1 or left > len(atom_symbols) or right > len(atom_symbols):
        return False
    try:
        z_left = atomic_number(atom_symbols[left - 1])
        z_right = atomic_number(atom_symbols[right - 1])
    except Exception:
        return False
    r_left = descriptor_vdw_radius(z_left)
    r_right = descriptor_vdw_radius(z_right)
    if r_left is None or r_right is None:
        return False
    return float(distance) <= PSEUDO_CYCLE_CLOSURE_VDW_SCALE * (float(r_left) + float(r_right))


def _closest_interfragment_atom_pair(
    left_atoms: tuple[int, ...],
    right_atoms: tuple[int, ...],
    *,
    covalent: set[tuple[int, int]],
    coords: np.ndarray,
) -> tuple[float, tuple[int, int]] | None:
    best: tuple[float, tuple[int, int]] | None = None
    for left in left_atoms:
        for right in right_atoms:
            pair = tuple(sorted((int(left), int(right))))
            if pair in covalent:
                continue
            distance = float(np.linalg.norm(coords[pair[0] - 1] - coords[pair[1] - 1]))
            if best is None or (distance, pair) < best:
                best = (distance, pair)
    return best


def _interfragment_atom_pair_candidates(
    left_atoms: tuple[int, ...],
    right_atoms: tuple[int, ...],
    *,
    covalent: set[tuple[int, int]],
    coords: np.ndarray,
) -> tuple[tuple[float, tuple[int, int]], ...]:
    candidates: list[tuple[float, tuple[int, int]]] = []
    for left in left_atoms:
        for right in right_atoms:
            pair = tuple(sorted((int(left), int(right))))
            if pair in covalent:
                continue
            distance = float(np.linalg.norm(coords[pair[0] - 1] - coords[pair[1] - 1]))
            candidates.append((distance, pair))
    return tuple(sorted(candidates))


def _fragment_index_by_atom(records: tuple[object, ...]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, record in enumerate(records):
        for atom in getattr(record, "atoms", ()):
            mapping[int(atom)] = index
    return mapping


def _interfragment_hbond_contacts(
    records: tuple[object, ...],
    *,
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    covalent: set[tuple[int, int]],
    fragment_index_by_atom: dict[int, int],
) -> tuple[tuple[float, tuple[int, int], str], ...]:
    if len(records) <= 1:
        return ()
    parameters = load_bdpcs3_parameters().hbond
    atomic_numbers = tuple(int(atomic_number(symbol) or 0) for symbol in atom_symbols)
    adjacency = _adjacency(tuple(covalent), natoms=len(atom_symbols))
    selected: list[tuple[int, int, int, float, float]] = []
    for hydrogen, z_hydrogen in enumerate(atomic_numbers, start=1):
        if z_hydrogen != 1:
            continue
        donors = tuple(
            atom
            for atom in adjacency[hydrogen]
            if atomic_numbers[atom - 1] in parameters.donor_atomic_numbers
        )
        if len(donors) != 1:
            continue
        donor = donors[0]
        donor_fragment = fragment_index_by_atom.get(donor)
        if donor_fragment is None:
            continue
        best: tuple[int, int, int, float, float] | None = None
        for acceptor, z_acceptor in enumerate(atomic_numbers, start=1):
            if acceptor in {hydrogen, donor}:
                continue
            if z_acceptor not in parameters.acceptor_atomic_numbers:
                continue
            acceptor_fragment = fragment_index_by_atom.get(acceptor)
            if acceptor_fragment is None or acceptor_fragment == donor_fragment:
                continue
            if tuple(sorted((hydrogen, acceptor))) in covalent:
                continue
            if tuple(sorted((donor, acceptor))) in covalent:
                continue
            distance = float(np.linalg.norm(coords[hydrogen - 1] - coords[acceptor - 1]))
            if distance >= parameters.search_cutoff_ang:
                continue
            angle = float(np.degrees(_angle_value(coords, (donor, hydrogen, acceptor))))
            if angle < parameters.angle_threshold_deg:
                continue
            candidate = (donor, hydrogen, acceptor, distance, angle)
            if best is None or (distance, acceptor) < (best[3], best[2]):
                best = candidate
        if best is not None:
            selected.append(best)

    unique: list[tuple[float, tuple[int, int], str]] = []
    seen_donor_acceptor: set[tuple[int, int]] = set()
    for donor, hydrogen, acceptor, distance, _angle in sorted(
        selected,
        key=lambda item: (item[3], item[0], item[1], item[2]),
    ):
        donor_acceptor = (donor, acceptor)
        if donor_acceptor in seen_donor_acceptor:
            continue
        seen_donor_acceptor.add(donor_acceptor)
        unique.append((distance, tuple(sorted((hydrogen, acceptor))), "HBOND"))
    return tuple(unique)


def _interaction_center_definition(path: Path) -> object | None:
    try:
        from matrix_fragments import read_interaction_center_definition
    except ImportError:
        return None
    return read_interaction_center_definition(Path(path))


def _fragment_number(record: object) -> int:
    identifier = str(getattr(record, "identifier"))
    try:
        return int(identifier[1:])
    except ValueError:
        return 0


def _fragment_anchor_atoms(
    record: object,
    *,
    coords: np.ndarray,
    limit: int = 3,
) -> tuple[int, ...]:
    atoms = tuple(getattr(record, "atoms"))
    if len(atoms) <= limit:
        return atoms
    center = _fragment_center(coords, atoms)
    ranked = sorted(
        atoms,
        key=lambda atom: (-float(np.linalg.norm(coords[atom - 1] - center)), atom),
    )
    return tuple(sorted(ranked[:limit]))


def _fragment_center(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    if not atoms:
        raise FloatingPointError("fragment has no atoms")
    return np.mean(coords[[atom - 1 for atom in atoms]], axis=0)


def _fragment_frame_rank(coords: np.ndarray, atoms: tuple[int, ...]) -> int:
    if len(atoms) < 2:
        return 0
    centered = coords[[atom - 1 for atom in atoms]] - _fragment_center(coords, atoms)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return int(np.sum(singular_values > 1.0e-8))


def _fragment_frame(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...] = (),
) -> np.ndarray:
    if _fragment_frame_rank(coords, atoms) < 2:
        raise FloatingPointError("fragment orientation is underdefined")
    p_atom, q_atom = (
        tuple(frame_atoms) if frame_atoms else _fragment_frame_anchor_atoms(atoms, coords=coords)
    )
    center = _fragment_center(coords, atoms)
    p_axis = _unit(coords[p_atom - 1] - center)
    q_raw = np.cross(p_axis, coords[q_atom - 1] - center)
    q_axis = _unit(q_raw)
    s_axis = _unit(np.cross(p_axis, q_axis))
    return np.column_stack([p_axis, q_axis, s_axis])


def _fragment_frame_anchor_atoms(
    atoms: tuple[int, ...],
    *,
    coords: np.ndarray,
) -> tuple[int, int]:
    center = _fragment_center(coords, atoms)
    ranked = sorted(
        atoms,
        key=lambda atom: (-float(np.linalg.norm(coords[atom - 1] - center)), atom),
    )
    p_atom = ranked[0]
    p_axis = _unit(coords[p_atom - 1] - center)
    q_candidates = []
    for atom in atoms:
        if atom == p_atom:
            continue
        vector = coords[atom - 1] - center
        norm = float(np.linalg.norm(vector))
        if norm <= RANK_TOLERANCE:
            continue
        dot = abs(float(np.dot(p_axis, vector / norm)))
        q_candidates.append((dot, -norm, atom))
    if not q_candidates:
        raise FloatingPointError("fragment has no second orientation anchor")
    _dot, _norm, q_atom = min(q_candidates)
    return p_atom, q_atom


def _canonicalize_frame(frame: np.ndarray) -> np.ndarray:
    out = np.array(frame, dtype=float, copy=True)
    for axis in range(3):
        column = out[:, axis]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            out[:, axis] *= -1.0
    if np.linalg.det(out) < 0.0:
        out[:, -1] *= -1.0
    return out


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[1, 2] - rotation[2, 1]) * scale
        qy = (rotation[2, 0] - rotation[0, 2]) * scale
        qz = (rotation[0, 1] - rotation[1, 0]) * scale
    else:
        qw, qx, qy, qz = _fallback_quaternion(rotation)
    quat = np.array([qw, qx, qy, qz], dtype=float)
    if quat[0] < 0.0:
        quat = -quat
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    quat /= norm
    vector = quat[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(vector_norm, quat[0])
    return vector / vector_norm * angle


def _quaternion_vector(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[1, 2] - rotation[2, 1]) * scale
        qy = (rotation[2, 0] - rotation[0, 2]) * scale
        qz = (rotation[0, 1] - rotation[1, 0]) * scale
    else:
        qw, qx, qy, qz = _fallback_quaternion(rotation)
    quat = np.array([qw, qx, qy, qz], dtype=float)
    if quat[0] < 0.0:
        quat = -quat
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    quat /= norm
    return quat[1:]


def _fallback_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        return (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        )
    if rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
        return (
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        )
    scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
    return (
        (rotation[1, 0] - rotation[0, 1]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
    )


def _vibrational_rank(coords: np.ndarray) -> int:
    natoms = int(coords.shape[0])
    if natoms <= 1:
        return 0
    return max(0, 3 * natoms - (5 if _is_linear(coords) else 6))


def _is_linear(coords: np.ndarray) -> bool:
    if coords.shape[0] <= 2:
        return True
    centered = coords - np.mean(coords, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size < 2 or singular_values[0] <= RANK_TOLERANCE:
        return False
    return bool(singular_values[1] / singular_values[0] <= 1.0e-6)


def _adjacency(bonds: tuple[tuple[int, int], ...], *, natoms: int) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(1, natoms + 1)}
    for i, j in bonds:
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _cyclic_atom_set(rings: tuple[tuple[int, tuple[int, ...]], ...]) -> set[int]:
    return {atom for _ring_index, atoms in rings for atom in atoms}


def _improper_dihedral_atoms(atoms: tuple[int, ...]) -> tuple[int, ...]:
    center, n1, n2, n3 = atoms
    return (n1, center, n3, n2)


def _point_group(lines: list[str]) -> str:
    for line in section_content(lines, "SYMMETRY"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "POINT_GROUP":
            return parts[1]
    return "UNKNOWN"


def _apply_symmetry_group_limit(
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    symmetry_group: str | None,
) -> tuple[str, tuple[GICPointGroupOperation, ...]]:
    target = (symmetry_group or "").strip()
    if not target:
        return point_group, operations
    if point_group.upper() == target.upper():
        return point_group, operations
    if not operations:
        if target.upper() == "C1":
            return "C1", ()
        raise GICForgeContractError(
            f"cannot reduce {point_group} to {target}: no stored symmetry operations"
        )
    identity = _identity_symmetry_operation(operations)
    target_key = target.upper()
    if target_key == "C1":
        return "C1", (identity,)
    keep = [identity]
    if target_key == "CS":
        match = next((operation for operation in operations if _is_sigma_operation(operation)), None)
    elif target_key == "CI":
        match = next((operation for operation in operations if operation.label == "i"), None)
    elif target_key == "C2":
        match = next((operation for operation in operations if operation.label.startswith("C2")), None)
    else:
        raise GICForgeContractError(
            f"unsupported reduced symmetry group {target!r}; "
            "supported limits are C1, Cs, Ci, C2, or the full point group"
        )
    if match is None:
        raise GICForgeContractError(
            f"cannot reduce {point_group} to {target}: no compatible stored operation"
        )
    keep.append(match)
    return target, tuple(keep)


def _identity_symmetry_operation(
    operations: tuple[GICPointGroupOperation, ...],
) -> GICPointGroupOperation:
    match = next((operation for operation in operations if operation.label == "E"), None)
    if match is not None:
        return match
    natoms = len(operations[0].permutation) if operations else 0
    return GICPointGroupOperation(
        label="E",
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        permutation=tuple(range(1, natoms + 1)),
    )


def _is_sigma_operation(operation: GICPointGroupOperation) -> bool:
    return operation.label.startswith("sigma")


def _symmetry_operations(lines: list[str]) -> tuple[GICPointGroupOperation, ...]:
    symmetry = section_content(lines, "SYMMETRY")
    if not symmetry:
        return ()
    operation_lines = _subsection(symmetry, "OPERATIONS")
    if not operation_lines:
        return ()
    operations: list[GICPointGroupOperation] = []
    for line in operation_lines:
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        fields = _key_values(parts[1:])
        try:
            matrix_values = _parse_float_list(fields["MATRIX"])
            if len(matrix_values) != 9:
                raise ValueError("operation matrix must have 9 values")
            permutation = _parse_atom_list(fields["PERMUTATION"])
            operations.append(
                GICPointGroupOperation(
                    label=fields["LABEL"],
                    rotation=tuple(
                        tuple(float(value) for value in matrix_values[start : start + 3])
                        for start in (0, 3, 6)
                    ),
                    permutation=permutation,
                )
            )
        except KeyError as exc:
            raise GICForgeContractError(f"invalid #SYMMETRY operation line: {line}") from exc
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid #SYMMETRY operation numeric field: {line}"
            ) from exc
    return tuple(operations)


def _subsection(section_lines: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for idx in range(start, len(section_lines)):
        text = section_lines[idx].strip()
        if text.startswith("[") and text.endswith("]"):
            end = idx
            break
    return list(section_lines[start:end])


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


def _key_values(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.upper()] = value
    return fields


def _parse_atom_list(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    try:
        return tuple(int(item) for item in text.split(",") if item)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid atom list: {text}") from exc


def _parse_float_list(text: str) -> tuple[float, ...]:
    if not text:
        return ()
    try:
        return tuple(float(item) for item in text.split(",") if item)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid float list: {text}") from exc


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


def _planned_improper_dihedrals(lines: list[str]) -> bool:
    mode = _section_value(section_content(lines, "GIC"), "OUT_OF_PLANE_MODE")
    return (mode or "").strip().upper() in {"IMPROPER_DIHEDRAL", "IMPROPER", "IMPD", "G16"}


def _planned_fragment_mode(lines: list[str]) -> str:
    gic = section_content(lines, "GIC")
    mode = _section_value(gic, "FRAGMENT_MODE")
    if mode is None:
        return FRAGMENT_MODE_SPECIAL_COORDINATES
    return _fragment_mode(mode)


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
    mode = f" MODE={primitive.mode}" if primitive.function == "L" else ""
    if primitive.function in {"FTRANS", "FROT"}:
        mode = f" MODE={primitive.mode}"
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
        f" PROVENANCE={primitive.provenance}"
        if primitive.provenance != AUTO_PROVENANCE
        else ""
    )
    semantic_id = f" SEMANTIC_ID={primitive.semantic_id}" if primitive.semantic_id else ""
    semantic_type = (
        f" SEMANTIC_TYPE={primitive.semantic_type}" if primitive.semantic_type else ""
    )
    return (
        f"{primitive.identifier} NAME={primitive.name} FAMILY={primitive.family} "
        f"CLASS={primitive.reduction_class} FUNCTION={primitive.function} "
        f"ATOMS={atoms}{ref_atoms}{refs}"
        f"{frame_atoms}{ref_frame_atoms}{mode}"
        f"{provenance}{semantic_id}{semantic_type} "
        f"GAUSSIAN={primitive.gaussian_expression()}"
    )


def _gaussian_gic_block_lines(definition: GICDefinition) -> list[str]:
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    lines: list[str] = []
    virtual_centers = _gaussian_virtual_center_atoms(definition.primitives)
    if virtual_centers:
        for center_id, atoms in sorted(virtual_centers.items()):
            lines.extend(_gaussian_virtual_center_lines(center_id, atoms))
    fragment_atoms = _gaussian_fragment_atoms(definition.primitives)
    if fragment_atoms:
        for fragment_id, atoms in sorted(fragment_atoms.items()):
            lines.append(f"{fragment_id}=Fragment({_atom_interval(atoms)})")
        for fragment_id in sorted(fragment_atoms):
            lines.extend(_gaussian_center_lines(fragment_id))
        frot_pairs = _gaussian_frot_pairs(definition.primitives)
        frame_atoms = _gaussian_fragment_frame_atoms(definition.primitives)
        frame_fragments = sorted({fragment for pair in frot_pairs for fragment in pair})
        for fragment_id in frame_fragments:
            atoms = fragment_atoms[fragment_id]
            lines.extend(
                _gaussian_frame_lines(
                    fragment_id,
                    atoms,
                    coords=coords,
                    frame_atoms=frame_atoms.get(fragment_id, ()),
                )
            )
        for frag_id, ref_id in sorted(frot_pairs):
            lines.extend(_gaussian_quaternion_lines(frag_id, ref_id))

    for gic in definition.gics:
        expression = _gaussian_expression_for_gic(definition, gic)
        if expression:
            label = _gaussian_label_for_gic(definition, gic)
            lines.append(f"{label} = {expression}")
    return lines


def _gaussian_label_for_gic(definition: GICDefinition, gic: FrozenGIC) -> str:
    if gic.family == "RING_PUCKER_COMPONENT":
        return gic.name
    return gic.name if definition.symmetrize else gic.identifier


def _ring_pucker_group_key_for_gic(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[int, ...] | None:
    primitive = _single_source_primitive(gic, primitive_by_id)
    if primitive is not None and primitive.function == "RPCK":
        return primitive.atoms
    atoms: set[int] = set()
    for primitive_id, _coefficient in gic.coefficients or ():
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            return None
        if primitive.family != "RING_PUCKER_COMPONENT" or primitive.function not in {"D", "U"}:
            return None
        atoms.update(primitive.atoms)
    if len(atoms) < 4:
        return None
    return tuple(sorted(atoms))


def _ring_ref_text(ring: tuple[int, ...]) -> str:
    return "RING:" + "-".join(str(atom) for atom in ring)


def _ring_ref_atoms(ref: str) -> tuple[int, ...] | None:
    if not ref.startswith("RING:"):
        return None
    try:
        atoms = tuple(int(atom) for atom in ref[5:].split("-") if atom)
    except ValueError:
        return None
    if len(atoms) < 4:
        return None
    return atoms


def _ring_refs_from_primitive(primitive: GICPrimitive) -> tuple[tuple[int, ...], ...]:
    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for ref in primitive.refs:
        ring = _ring_ref_atoms(ref)
        if ring is None or ring in seen:
            continue
        seen.add(ring)
        rings.append(ring)
    return tuple(rings)


def _ring_pucker_source_ring_keys_for_gic(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[tuple[int, ...], ...]:
    primitive = _single_source_primitive(gic, primitive_by_id)
    if (
        primitive is not None
        and primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "RPCK"
    ):
        return (primitive.atoms,)

    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for primitive_id, _coefficient in gic.coefficients or ():
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None or primitive.family != "RING_PUCKER_COMPONENT":
            continue
        primitive_rings = _ring_refs_from_primitive(primitive)
        if not primitive_rings and primitive.function == "RPCK":
            primitive_rings = (primitive.atoms,)
        for ring in primitive_rings:
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    if rings:
        return tuple(rings)

    group_key = _ring_pucker_group_key_for_gic(gic, primitive_by_id)
    return (group_key,) if group_key is not None else ()


def _ring_edges(ring: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(sorted((atom, ring[(index + 1) % len(ring)]))) for index, atom in enumerate(ring)
    )


def _all_ring_pucker_source_ring_keys(
    definition: GICDefinition,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[tuple[int, ...], ...]:
    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for primitive in definition.primitives:
        if primitive.family != "RING_PUCKER_COMPONENT":
            continue
        primitive_rings = _ring_refs_from_primitive(primitive)
        if not primitive_rings and primitive.function == "RPCK":
            primitive_rings = (primitive.atoms,)
        for ring in primitive_rings:
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    if rings:
        return tuple(rings)

    for gic in definition.gics:
        if gic.family != "RING_PUCKER_COMPONENT":
            continue
        for ring in _ring_pucker_source_ring_keys_for_gic(gic, primitive_by_id):
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    return tuple(rings)


def _condensed_ring_pucker_keys(
    definition: GICDefinition,
    primitive_by_id: dict[str, GICPrimitive],
) -> set[tuple[int, ...]]:
    rings = _all_ring_pucker_source_ring_keys(definition, primitive_by_id)
    edge_to_rings: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for ring in rings:
        for edge in _ring_edges(ring):
            edge_to_rings.setdefault(edge, []).append(ring)
    condensed: set[tuple[int, ...]] = set()
    for shared_rings in edge_to_rings.values():
        if len(shared_rings) > 1:
            condensed.update(shared_rings)
    return condensed


def _ring_pucker_gic_is_condensed(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
    condensed_ring_keys: set[tuple[int, ...]],
) -> bool:
    source_rings = _ring_pucker_source_ring_keys_for_gic(gic, primitive_by_id)
    return any(ring in condensed_ring_keys for ring in source_rings)


def _gaussian_expression_for_gic(
    definition: GICDefinition,
    gic: FrozenGIC,
) -> str | None:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    if len(coefficients) == 1 and abs(coefficients[0][1] - 1.0) <= 1.0e-12:
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None:
            raise GICForgeContractError(
                f"unknown primitive {coefficients[0][0]!r} in frozen GIC {gic.identifier}"
            )
        return _gaussian_expression_for_primitive(primitive)

    terms: list[str] = []
    for primitive_id, coefficient in coefficients:
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
            )
        expression = _gaussian_expression_for_primitive(primitive)
        if expression is None:
            return None
        terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
    return "".join(terms) if terms else None


def _gaussian_linear_term(coefficient: float, expression: str, *, first: bool) -> str:
    sign = "-" if coefficient < 0.0 else "+"
    magnitude = abs(float(coefficient))
    coefficient_text = f"{magnitude:.12g}"
    term = f"{coefficient_text}*({_strip_gaussian_outer_parentheses(expression)})"
    if first:
        return f"-{term}" if sign == "-" else term
    return f"{sign}{term}"


def _strip_gaussian_outer_parentheses(expression: str) -> str:
    text = expression.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


def _gaussian_expression_for_primitive(primitive: GICPrimitive) -> str | None:
    if primitive.is_gaussian_native:
        return primitive.gaussian_expression()
    if primitive.function == "FC_DIST":
        frag_id, ref_id = primitive.refs
        return (
            f"SQRT((Cx{frag_id}-Cx{ref_id})**2+"
            f"(Cy{frag_id}-Cy{ref_id})**2+"
            f"(Cz{frag_id}-Cz{ref_id})**2)"
        )
    if primitive.function == "FCA_DIST":
        frag_id, atom_ref = primitive.refs
        atom = atom_ref[1:]
        return (
            f"SQRT((Cx{frag_id}-X({atom}))**2+"
            f"(Cy{frag_id}-Y({atom}))**2+"
            f"(Cz{frag_id}-Z({atom}))**2)"
        )
    if primitive.function == "CENTER_ATOM_DIST":
        center_id, atom_ref = primitive.refs
        atom = atom_ref[1:]
        return (
            f"SQRT((Cx{center_id}-X({atom}))**2+"
            f"(Cy{center_id}-Y({atom}))**2+"
            f"(Cz{center_id}-Z({atom}))**2)"
        )
    if primitive.function == "FTRANS":
        frag_id, ref_id = primitive.refs
        axis = ("x", "y", "z")[primitive.mode]
        return f"C{axis}{frag_id}-C{axis}{ref_id}"
    if primitive.function == "FROT":
        frag_id, ref_id = primitive.refs
        axis = ("x", "y", "z")[primitive.mode]
        return f"E{axis}{frag_id}{ref_id}"
    if primitive.function == "RPCB":
        terms: list[str] = []
        for coefficient, atoms in _angle_component_terms_from_refs(primitive):
            expression = "A(" + ",".join(str(atom) for atom in atoms) + ")"
            terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
        return "".join(terms) if terms else None
    if primitive.function == "RPCK":
        terms: list[str] = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            expression = "D(" + ",".join(str(atom) for atom in atoms) + ")"
            terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
        return "".join(terms) if terms else None
    if primitive.function == "IMPD":
        return primitive.gaussian_expression()
    return None


def _gaussian_fragment_atoms(
    primitives: tuple[GICPrimitive, ...],
) -> dict[str, tuple[int, ...]]:
    fragments: dict[str, tuple[int, ...]] = {}
    for primitive in primitives:
        if not primitive.refs:
            continue
        first = primitive.refs[0]
        if first.startswith("F"):
            fragments.setdefault(first, primitive.atoms)
        if len(primitive.refs) < 2:
            continue
        second = primitive.refs[1]
        if second.startswith("F"):
            fragments.setdefault(second, primitive.ref_atoms)
    return fragments


def _gaussian_virtual_center_atoms(
    primitives: tuple[GICPrimitive, ...],
) -> dict[str, tuple[int, ...]]:
    centers: dict[str, tuple[int, ...]] = {}
    for primitive in primitives:
        if primitive.function != "CENTER_ATOM_DIST" or not primitive.refs:
            continue
        center_id = primitive.refs[0]
        if center_id.startswith("C"):
            centers.setdefault(center_id, primitive.atoms)
    return centers


def _gaussian_frot_pairs(primitives: tuple[GICPrimitive, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for primitive in primitives:
        if primitive.function == "FROT" and len(primitive.refs) >= 2:
            pairs.add((primitive.refs[0], primitive.refs[1]))
    return pairs


def _gaussian_fragment_frame_atoms(
    primitives: tuple[GICPrimitive, ...],
) -> dict[str, tuple[int, ...]]:
    frame_atoms: dict[str, tuple[int, ...]] = {}
    for primitive in primitives:
        if primitive.function != "FROT" or len(primitive.refs) < 2:
            continue
        if primitive.frame_atoms:
            frame_atoms.setdefault(primitive.refs[0], primitive.frame_atoms)
        if primitive.ref_frame_atoms:
            frame_atoms.setdefault(primitive.refs[1], primitive.ref_frame_atoms)
    return frame_atoms


def _gaussian_center_lines(fragment_id: str) -> list[str]:
    return [
        f"Cx{fragment_id}(Inactive)=XCntr({fragment_id})",
        f"Cy{fragment_id}(Inactive)=YCntr({fragment_id})",
        f"Cz{fragment_id}(Inactive)=ZCntr({fragment_id})",
    ]


def _gaussian_virtual_center_lines(center_id: str, atoms: tuple[int, ...]) -> list[str]:
    if not atoms:
        return []
    denominator = len(atoms)
    return [
        f"Cx{center_id}(Inactive)=({_gaussian_axis_sum('X', atoms)})/{denominator}",
        f"Cy{center_id}(Inactive)=({_gaussian_axis_sum('Y', atoms)})/{denominator}",
        f"Cz{center_id}(Inactive)=({_gaussian_axis_sum('Z', atoms)})/{denominator}",
    ]


def _gaussian_axis_sum(axis: str, atoms: tuple[int, ...]) -> str:
    return "+".join(f"{axis}({atom})" for atom in atoms)


def _gaussian_frame_lines(
    fragment_id: str,
    atoms: tuple[int, ...],
    *,
    coords: np.ndarray,
    frame_atoms: tuple[int, ...] = (),
) -> list[str]:
    p_atom, q_atom = (
        tuple(frame_atoms) if frame_atoms else _fragment_frame_anchor_atoms(atoms, coords=coords)
    )
    return [
        f"RP{fragment_id}(Inactive)=SQRT((X({p_atom})-Cx{fragment_id})**2+"
        f"(Y({p_atom})-Cy{fragment_id})**2+(Z({p_atom})-Cz{fragment_id})**2)",
        f"Px{fragment_id}(Inactive)=[X({p_atom})-Cx{fragment_id}]/RP{fragment_id}",
        f"Py{fragment_id}(Inactive)=[Y({p_atom})-Cy{fragment_id}]/RP{fragment_id}",
        f"Pz{fragment_id}(Inactive)=[Z({p_atom})-Cz{fragment_id}]/RP{fragment_id}",
        f"QQx{fragment_id}(Inactive)=Py{fragment_id}*[Z({q_atom})-Cz{fragment_id}]-"
        f"Pz{fragment_id}*[Y({q_atom})-Cy{fragment_id}]",
        f"QQy{fragment_id}(Inactive)=Pz{fragment_id}*[X({q_atom})-Cx{fragment_id}]-"
        f"Px{fragment_id}*[Z({q_atom})-Cz{fragment_id}]",
        f"QQz{fragment_id}(Inactive)=Px{fragment_id}*[Y({q_atom})-Cy{fragment_id}]-"
        f"Py{fragment_id}*[X({q_atom})-Cx{fragment_id}]",
        f"RQ{fragment_id}(Inactive)=SQRT(QQx{fragment_id}**2+"
        f"QQy{fragment_id}**2+QQz{fragment_id}**2)",
        f"Qx{fragment_id}(Inactive)=QQx{fragment_id}/RQ{fragment_id}",
        f"Qy{fragment_id}(Inactive)=QQy{fragment_id}/RQ{fragment_id}",
        f"Qz{fragment_id}(Inactive)=QQz{fragment_id}/RQ{fragment_id}",
        f"Sx{fragment_id}(Inactive)=Py{fragment_id}*Qz{fragment_id}-"
        f"Pz{fragment_id}*Qy{fragment_id}",
        f"Sy{fragment_id}(Inactive)=Pz{fragment_id}*Qx{fragment_id}-"
        f"Px{fragment_id}*Qz{fragment_id}",
        f"Sz{fragment_id}(Inactive)=Px{fragment_id}*Qy{fragment_id}-"
        f"Py{fragment_id}*Qx{fragment_id}",
    ]


def _gaussian_quaternion_lines(frag_id: str, ref_id: str) -> list[str]:
    pair = f"{frag_id}{ref_id}"
    rows = []
    for left_axis, left_prefix in (("1", "P"), ("2", "Q"), ("3", "S")):
        for right_axis, right_prefix in (("1", "P"), ("2", "Q"), ("3", "S")):
            rows.append(
                f"R{left_axis}{right_axis}{pair}(Inactive)="
                f"{left_prefix}x{frag_id}*{right_prefix}x{ref_id}+"
                f"{left_prefix}y{frag_id}*{right_prefix}y{ref_id}+"
                f"{left_prefix}z{frag_id}*{right_prefix}z{ref_id}"
            )
    return [
        *rows,
        f"Kw{pair}(Inactive)=0.5*SQRT(R11{pair}+R22{pair}+R33{pair}+1)",
        f"Kx{pair}(Inactive)=(R23{pair}-R32{pair})/(4*Kw{pair})",
        f"Ky{pair}(Inactive)=(R31{pair}-R13{pair})/(4*Kw{pair})",
        f"Kz{pair}(Inactive)=(R12{pair}-R21{pair})/(4*Kw{pair})",
        f"Kn{pair}(Inactive)=SQRT(Kx{pair}**2+Ky{pair}**2+Kz{pair}**2+1.0D-24)",
        f"Th{pair}(Inactive)=2*ATAN(Kn{pair}/Kw{pair})",
        f"Ex{pair}(Inactive)=Th{pair}*Kx{pair}/Kn{pair}",
        f"Ey{pair}(Inactive)=Th{pair}*Ky{pair}/Kn{pair}",
        f"Ez{pair}(Inactive)=Th{pair}*Kz{pair}/Kn{pair}",
    ]


def _primitive_by_id(
    primitives: tuple[GICPrimitive, ...],
    identifier: str,
) -> GICPrimitive:
    for primitive in primitives:
        if primitive.identifier == identifier:
            return primitive
    raise GICForgeContractError(f"unknown primitive id in frozen GIC: {identifier}")


def _atom_interval(atoms: tuple[int, ...]) -> str:
    sorted_atoms = sorted(atoms)
    if not sorted_atoms:
        return ""
    ranges: list[str] = []
    start = previous = sorted_atoms[0]
    for atom in sorted_atoms[1:]:
        if atom == previous + 1:
            previous = atom
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = atom
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def _frozen_gic_line(gic: FrozenGIC) -> str:
    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    coeffs = ",".join(
        f"{primitive_id}:{coefficient:.12f}" for primitive_id, coefficient in coefficients
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


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length vector")
    return vector / norm


def _dot_unit(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))


def _orthogonal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, trial))
    e2 = _unit(np.cross(axis, e1))
    return e1, e2


def _periodic_delta(value_plus: float, value_minus: float) -> float:
    delta = float(value_plus - value_minus)
    while delta > np.pi:
        delta -= 2.0 * np.pi
    while delta < -np.pi:
        delta += 2.0 * np.pi
    return delta
