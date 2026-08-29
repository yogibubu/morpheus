"""Separated intra/inter conditioning for stable pseudobond MINIMUM charts.

ORACLE owns the chemistry, domains, compatibility blocks, and thresholds.
SMITH only realizes the frozen prescription by exact-rank linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from matrix_chem import (
    CoordinateComponent,
    coordinate_selection_units,
    validate_coordinate_component_transform,
)
from matrix_chem.coordinate_atlas_contract import (
    ATLAS_TASK_MINIMUM,
    CANDIDATE_OPTIONAL,
    PSEUDOBOND_REQUIRED,
    OracleCoordinateAtlasContract,
    validate_oracle_coordinate_atlas_contract,
)
from matrix_chem.oracle_sonic_contract import (
    OracleSonicContract,
    validate_oracle_sonic_contract,
)

from .component_invariants import validate_indivisible_gic_components
from .contracts import GICForgeContractError
from .evaluation import build_gic_and_primitive_b_matrices
from .minimum_domain_sampling import local_domain_samples
from .models import GICDefinition, GICPrimitive, GICReductionDiagnostics
from .policy import (
    FRAGMENT_MODE_PSEUDO_BONDS,
    MAX_NORMALIZED_SONIC_CONDITION,
    RANK_TOLERANCE,
)


MINIMUM_DOMAIN_SELECTION_SCHEMA = "matrix.smith.minimum_domain_partition.v2"
_CONTACT_SUPPORT_REF = "PSEUDOBOND_CONTACT_SUPPORT"
_INTER_FAMILIES = frozenset(
    {
        "PSEUDO_BOND_DISTANCE",
        "FRAG_DISTANCE",
        "PSEUDO_BOND_BEND",
        "PSEUDO_BOND_TORSION",
    }
)
_MAX_INTERNAL_EXCHANGE_ORDER = 2


@dataclass(frozen=True)
class _Audit:
    maximum_condition: float
    mean_condition: float
    worst_sample: str
    full_rank: bool


@dataclass(frozen=True)
class _InternalExchange:
    slot: int
    source_id: str
    alternative_id: str
    block_id: str


@dataclass(frozen=True)
class _PrimitiveAtlasEntry:
    prescription: object
    domain_id: str


@dataclass(frozen=True)
class _SupportCandidate:
    primitive_ids: tuple[str, ...]
    inter_audit: _Audit
    locality: int


@dataclass(frozen=True)
class _SampleRows:
    label: str
    gic_rows: np.ndarray
    primitive_rows: np.ndarray


def condition_minimum_pseudobond_chart(
    definition: GICDefinition,
    contract: OracleCoordinateAtlasContract,
    *,
    sonic_contract: OracleSonicContract,
    rank_tolerance: float = RANK_TOLERANCE,
) -> GICDefinition:
    """Realize one exact fixed MINIMUM chart over ORACLE's local domain."""

    if not _requires_conditioning(contract, sonic_contract=sonic_contract):
        return definition
    _validate_input_chart(definition)
    samples = local_domain_samples(
        definition.reference_coordinates_angstrom,
        contract.local_domains,
    )
    internal, internal_exchange, intra_before, intra_after = _condition_internal_chart(
        definition,
        contract=contract,
        sonic_contract=sonic_contract,
        rank_tolerance=rank_tolerance,
    )
    conditioned, support_ids, inter_before, inter_after, global_after = (
        _condition_interfragment_chart(
            internal,
            contract=contract,
            sonic_contract=sonic_contract,
            samples=samples,
            rank_tolerance=rank_tolerance,
        )
    )
    validate_indivisible_gic_components(conditioned)
    if not global_after.full_rank:
        raise GICForgeContractError(
            "separated MINIMUM conditioning did not retain exact global rank"
        )
    return _with_partition_diagnostics(
        conditioned,
        sample_count=len(samples),
        internal_exchange=internal_exchange,
        support_ids=support_ids,
        intra_before=intra_before,
        intra_after=intra_after,
        inter_before=inter_before,
        inter_after=inter_after,
        global_after=global_after,
    )


def _requires_conditioning(
    contract: OracleCoordinateAtlasContract,
    *,
    sonic_contract: OracleSonicContract,
) -> bool:
    validate_oracle_coordinate_atlas_contract(contract)
    validate_oracle_sonic_contract(sonic_contract)
    if contract.topology_hash != sonic_contract.primary_topology.topology_hash:
        raise GICForgeContractError("ORACLE SONIC and coordinate-atlas topology hashes differ")
    return contract.task_regime == ATLAS_TASK_MINIMUM and any(
        item.pseudobond_policy == PSEUDOBOND_REQUIRED for item in contract.interactions
    )


def _validate_input_chart(definition: GICDefinition) -> None:
    if definition.fragment_mode != FRAGMENT_MODE_PSEUDO_BONDS:
        raise GICForgeContractError(
            "a required MINIMUM pseudobond domain needs a PSEUDO_BONDS chart"
        )
    if len(definition.gics) != definition.target_rank:
        raise GICForgeContractError(
            "MINIMUM domain conditioning requires an exact nonredundant chart"
        )


def _condition_internal_chart(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    rank_tolerance: float,
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit]:
    internal_indices = _internal_gic_indices(definition)
    reference = (("REFERENCE", np.asarray(definition.reference_coordinates_angstrom)),)
    cache = _row_cache(definition, reference)
    baseline = _audit_cached(
        cache,
        row_indices=internal_indices,
        projected_against=(),
        rank_tolerance=rank_tolerance,
    )
    blocks = _substitution_blocks(contract, prefix="INTERNAL::")
    trigger = min((item.condition_trigger for item in blocks.values()), default=None)
    if trigger is None or baseline.maximum_condition <= trigger:
        return definition, (), baseline, baseline

    primitive_by_id = {item.identifier: item for item in definition.primitives}
    prescriptions = _primitive_prescriptions(
        definition,
        contract=contract,
        sonic_contract=sonic_contract,
    )
    exchanges = _eligible_internal_exchanges(
        definition,
        primitive_by_id=primitive_by_id,
        prescription_by_id=prescriptions,
        substitution_blocks=frozenset(blocks),
    )
    candidates: list[tuple[_Audit, tuple[_InternalExchange, ...]]] = []
    for order in range(1, _MAX_INTERNAL_EXCHANGE_ORDER + 1):
        for exchange_set in combinations(exchanges, order):
            if not _compatible_exchange_set(exchange_set):
                continue
            audit = _audit_cached(
                cache,
                row_indices=internal_indices,
                projected_against=(),
                rank_tolerance=rank_tolerance,
                replacement_by_slot={item.slot: item.alternative_id for item in exchange_set},
                primitive_index={
                    item.identifier: index for index, item in enumerate(definition.primitives)
                },
            )
            if audit.full_rank:
                candidates.append((audit, exchange_set))
    if not candidates:
        return definition, (), baseline, baseline
    audit, selected = min(
        candidates,
        key=lambda item: (
            item[0].maximum_condition,
            item[0].mean_condition,
            tuple(exchange.alternative_id for exchange in item[1]),
        ),
    )
    minimum_gain = max(
        float(blocks[item.block_id].minimum_relative_gain or 0.0) for item in selected
    )
    if not _material_gain(
        baseline.maximum_condition,
        audit.maximum_condition,
        minimum_gain=minimum_gain,
    ):
        return definition, (), baseline, baseline
    candidate = _apply_internal_exchanges(
        definition,
        selected,
        primitive_by_id=primitive_by_id,
    )
    labels = tuple(f"{item.source_id}->{item.alternative_id}" for item in selected)
    return candidate, labels, baseline, audit


def _condition_interfragment_chart(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    samples: tuple[tuple[str, np.ndarray], ...],
    rank_tolerance: float,
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit, _Audit]:
    internal_indices = _internal_gic_indices(definition)
    fragment_by_atom = {
        atom: fragment.fragment_id
        for fragment in sonic_contract.primary_topology.fragments
        for atom in fragment.atoms
    }
    pairs = tuple(
        sorted(
            {
                _support_pair(primitive)
                for primitive in definition.primitives
                if primitive.refs and primitive.refs[0] == _CONTACT_SUPPORT_REF
            }
        )
    )
    current = definition
    primitive_cache = _row_cache(definition, samples)
    support_ids: list[str] = []
    before_audits: list[_Audit] = []
    after_audits: list[_Audit] = []
    for pair in pairs:
        current, selected_ids, before, after = _condition_contact_pair(
            current,
            pair=pair,
            fragment_by_atom=fragment_by_atom,
            contract=contract,
            sonic_contract=sonic_contract,
            internal_indices=internal_indices,
            rank_tolerance=rank_tolerance,
            defer_global_selection=len(pairs) > 1,
            cache=_recombine_row_cache(current, primitive_cache),
        )
        support_ids.extend(selected_ids)
        before_audits.append(before)
        after_audits.append(after)
    cache = _recombine_row_cache(current, primitive_cache)
    global_audit = _audit_cached(
        cache,
        row_indices=tuple(range(len(current.gics))),
        projected_against=(),
        rank_tolerance=rank_tolerance,
    )
    return (
        current,
        tuple(support_ids),
        _combine_audits(before_audits),
        _combine_audits(after_audits),
        global_audit,
    )


def _condition_contact_pair(
    definition: GICDefinition,
    *,
    pair: tuple[str, str],
    fragment_by_atom: dict[int, str],
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
    internal_indices: tuple[int, ...],
    rank_tolerance: float,
    defer_global_selection: bool,
    cache: tuple[_SampleRows, ...],
) -> tuple[GICDefinition, tuple[str, ...], _Audit, _Audit]:
    support_slots = _selected_support_slots(definition, pair=pair)
    required_slots = _required_contact_slots(
        definition,
        pair=pair,
        fragment_by_atom=fragment_by_atom,
    )
    inter_indices = tuple(sorted((*required_slots, *support_slots)))
    support_count = len(support_slots)
    if support_count <= 0:
        raise GICForgeContractError("a pseudobond fragment chart has no selectable contact support")
    primitive_index = {item.identifier: index for index, item in enumerate(definition.primitives)}
    baseline = _audit_cached(
        cache,
        row_indices=inter_indices,
        projected_against=internal_indices,
        rank_tolerance=rank_tolerance,
    )
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    support_pool = tuple(
        item
        for item in definition.primitives
        if item.refs and item.refs[0] == _CONTACT_SUPPORT_REF and _support_pair(item) == pair
    )
    units = _support_selection_units(support_pool)
    selections = _support_id_sets(units, target_count=support_count)
    if not selections:
        raise GICForgeContractError(
            "ORACLE contact-support pool cannot span the required exact dimension"
        )

    trigger, minimum_gain = _contact_condition_policy(contract)
    baseline_complete = _selected_support_is_component_complete(
        definition,
        support_slots=support_slots,
        support_pool=support_pool,
    )
    if baseline_complete and baseline.full_rank and baseline.maximum_condition <= trigger:
        return definition, _support_ids(definition, support_slots), baseline, baseline

    candidates = _contact_support_candidates(
        cache=cache,
        selections=selections,
        support_slots=support_slots,
        inter_indices=inter_indices,
        internal_indices=internal_indices,
        primitive_index=primitive_index,
        primitive_by_id=primitive_by_id,
        sonic_contract=sonic_contract,
        contract=contract,
        pair=pair,
        rank_tolerance=rank_tolerance,
    )
    if not candidates:
        raise GICForgeContractError(
            "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: no complete exact MINIMUM interfragment "
            "chart is full rank over the ORACLE local domain"
        )
    selected = _choose_contact_support(
        candidates,
        cache=cache,
        support_slots=support_slots,
        primitive_index=primitive_index,
        gic_count=len(definition.gics),
        rank_tolerance=rank_tolerance,
        minimum_gain=minimum_gain,
        defer_global_selection=defer_global_selection,
    )
    selected_definition = _replace_support_slots(
        definition,
        support_slots=support_slots,
        primitive_ids=selected.primitive_ids,
        primitive_by_id=primitive_by_id,
    )
    return (
        selected_definition,
        selected.primitive_ids,
        baseline,
        selected.inter_audit,
    )


def _contact_support_candidates(
    *,
    cache: tuple[_SampleRows, ...],
    selections: tuple[tuple[str, ...], ...],
    support_slots: tuple[int, ...],
    inter_indices: tuple[int, ...],
    internal_indices: tuple[int, ...],
    primitive_index: dict[str, int],
    primitive_by_id: dict[str, GICPrimitive],
    sonic_contract: OracleSonicContract,
    contract: OracleCoordinateAtlasContract,
    pair: tuple[str, str],
    rank_tolerance: float,
) -> tuple[_SupportCandidate, ...]:
    audits = _audit_support_candidates(
        cache,
        selections=selections,
        support_slots=support_slots,
        inter_indices=inter_indices,
        internal_indices=internal_indices,
        primitive_index=primitive_index,
        rank_tolerance=rank_tolerance,
    )
    return tuple(
        _SupportCandidate(
            primitive_ids=primitive_ids,
            inter_audit=audit,
            locality=_support_locality(
                primitive_ids,
                primitive_by_id=primitive_by_id,
                sonic_contract=sonic_contract,
                contract=contract,
                pair=pair,
            ),
        )
        for primitive_ids, audit in zip(selections, audits, strict=True)
        if audit.full_rank
    )


def _choose_contact_support(
    candidates: tuple[_SupportCandidate, ...],
    *,
    cache: tuple[_SampleRows, ...],
    support_slots: tuple[int, ...],
    primitive_index: dict[str, int],
    gic_count: int,
    rank_tolerance: float,
    minimum_gain: float,
    defer_global_selection: bool,
) -> _SupportCandidate:
    best_condition = min(item.inter_audit.maximum_condition for item in candidates)
    locality_band = best_condition / max(1.0e-12, 1.0 - minimum_gain)
    eligible = tuple(
        item for item in candidates if item.inter_audit.maximum_condition <= locality_band
    )
    if defer_global_selection:
        return min(
            eligible,
            key=lambda item: (
                item.locality,
                item.inter_audit.maximum_condition,
                item.primitive_ids,
            ),
        )
    selected, _global_audit = _select_global_safe_support(
        eligible,
        cache=cache,
        support_slots=support_slots,
        primitive_index=primitive_index,
        gic_count=gic_count,
        rank_tolerance=rank_tolerance,
        minimum_gain=minimum_gain,
    )
    return selected


def _support_selection_units(
    primitives: tuple[GICPrimitive, ...],
) -> tuple[tuple[str, ...], ...]:
    components = tuple(
        CoordinateComponent(
            operator=item.function,
            atoms=item.atoms,
            mode=item.mode,
            ref_atoms=item.ref_atoms,
            context=(item.family, *item.refs),
        )
        for item in primitives
    )
    try:
        units = coordinate_selection_units(components)
    except ValueError as exc:
        raise GICForgeContractError(
            f"ORACLE contact-support component pool is invalid: {exc}"
        ) from exc
    return tuple(tuple(primitives[index].identifier for index in unit) for unit in units)


def _support_id_sets(
    units: tuple[tuple[str, ...], ...],
    *,
    target_count: int,
) -> tuple[tuple[str, ...], ...]:
    selected: list[tuple[str, ...]] = []

    def visit(index: int, current: tuple[str, ...]) -> None:
        if len(current) == target_count:
            selected.append(current)
            return
        if index == len(units) or len(current) > target_count:
            return
        visit(index + 1, current)
        visit(index + 1, current + units[index])

    visit(0, ())
    return tuple(sorted(set(selected)))


def _replace_support_slots(
    definition: GICDefinition,
    *,
    support_slots: tuple[int, ...],
    primitive_ids: tuple[str, ...],
    primitive_by_id: dict[str, GICPrimitive],
) -> GICDefinition:
    if len(support_slots) != len(primitive_ids):
        raise GICForgeContractError("contact-support replacement changed chart size")
    gics = list(definition.gics)
    for slot, primitive_id in zip(support_slots, primitive_ids, strict=True):
        source = gics[slot]
        primitive = primitive_by_id[primitive_id]
        gics[slot] = replace(
            source,
            name=f"{source.irrep}{primitive.name}",
            family=primitive.family,
            primitive_id=primitive_id,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive_id, 1.0),),
        )
    candidate = replace(definition, gics=tuple(gics))
    diagnostics = _synchronize_reduction_selection(
        candidate,
        primitive_by_id=primitive_by_id,
    )
    return replace(candidate, reduction_diagnostics=diagnostics)


def _row_cache(
    definition: GICDefinition,
    samples: tuple[tuple[str, np.ndarray], ...],
) -> tuple[_SampleRows, ...]:
    cached: list[_SampleRows] = []
    for label, coordinates in samples:
        gic_matrix, primitive_matrix = build_gic_and_primitive_b_matrices(
            definition,
            coordinates_angstrom=coordinates,
        )
        gic_rows = np.asarray(gic_matrix.rows, dtype=float)
        primitive_rows = np.asarray(primitive_matrix.rows, dtype=float)
        cached.append(
            _SampleRows(
                label=label,
                gic_rows=gic_rows,
                primitive_rows=primitive_rows,
            )
        )
    return tuple(cached)


def _recombine_row_cache(
    definition: GICDefinition,
    cache: tuple[_SampleRows, ...],
) -> tuple[_SampleRows, ...]:
    """Rebuild only GIC combinations over an unchanged primitive tangent cache."""

    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    recombined: list[_SampleRows] = []
    for sample in cache:
        rows: list[np.ndarray] = []
        for gic in definition.gics:
            row = np.zeros(sample.primitive_rows.shape[1], dtype=float)
            for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
                try:
                    primitive_row = sample.primitive_rows[primitive_index[primitive_id]]
                except KeyError as exc:
                    raise GICForgeContractError(
                        f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                    ) from exc
                row += float(coefficient) * primitive_row
            rows.append(row)
        recombined.append(
            _SampleRows(
                label=sample.label,
                gic_rows=np.asarray(rows, dtype=float),
                primitive_rows=sample.primitive_rows,
            )
        )
    return tuple(recombined)


def _audit_cached(
    cache: tuple[_SampleRows, ...],
    *,
    row_indices: tuple[int, ...],
    projected_against: tuple[int, ...],
    rank_tolerance: float,
    replacement_by_slot: dict[int, str] | None = None,
    primitive_index: dict[str, int] | None = None,
) -> _Audit:
    replacements = replacement_by_slot or {}
    if replacements and primitive_index is None:
        raise ValueError("primitive indices are required for row replacements")
    conditions: list[float] = []
    worst_sample = cache[0].label
    full_rank = True
    for sample in cache:
        try:
            all_rows = np.array(sample.gic_rows, copy=True)
            for slot, primitive_id in replacements.items():
                all_rows[slot] = sample.primitive_rows[primitive_index[primitive_id]]
            rows = all_rows[np.asarray(row_indices, dtype=int)]
            if projected_against:
                fixed = all_rows[np.asarray(projected_against, dtype=int)]
                rows = rows - rows @ np.linalg.pinv(fixed) @ fixed
            condition, rank = _normalized_condition(rows, rank_tolerance=rank_tolerance)
        except (FloatingPointError, ValueError, GICForgeContractError):
            condition, rank = float("inf"), 0
        conditions.append(condition)
        if condition >= max(conditions):
            worst_sample = sample.label
        full_rank = full_rank and rank == len(row_indices)
    return _Audit(
        maximum_condition=max(conditions),
        mean_condition=float(np.mean(conditions)),
        worst_sample=worst_sample,
        full_rank=full_rank,
    )


def _select_global_safe_support(
    candidates: tuple[_SupportCandidate, ...],
    *,
    cache: tuple[_SampleRows, ...],
    support_slots: tuple[int, ...],
    primitive_index: dict[str, int],
    gic_count: int,
    rank_tolerance: float,
    minimum_gain: float,
) -> tuple[_SupportCandidate, _Audit]:
    safe: list[tuple[_SupportCandidate, _Audit]] = []
    for item in candidates:
        audit = _audit_cached(
            cache,
            row_indices=tuple(range(gic_count)),
            projected_against=(),
            rank_tolerance=rank_tolerance,
            replacement_by_slot=dict(zip(support_slots, item.primitive_ids, strict=True)),
            primitive_index=primitive_index,
        )
        if audit.full_rank and audit.maximum_condition <= MAX_NORMALIZED_SONIC_CONDITION:
            safe.append((item, audit))
    if not safe:
        raise GICForgeContractError(
            "LOCAL_NONREDUNDANT_DOMAIN_UNSAFE: no MINIMUM interfragment chart "
            "passes the global condition gate"
        )
    best_global = min(pair[1].maximum_condition for pair in safe)
    global_band = best_global / max(1.0e-12, 1.0 - minimum_gain)
    comparable = tuple(pair for pair in safe if pair[1].maximum_condition <= global_band)
    return min(
        comparable,
        key=lambda pair: (
            pair[0].locality,
            pair[0].inter_audit.maximum_condition,
            pair[1].maximum_condition,
            pair[0].primitive_ids,
        ),
    )


def _audit_support_candidates(
    cache: tuple[_SampleRows, ...],
    *,
    selections: tuple[tuple[str, ...], ...],
    support_slots: tuple[int, ...],
    inter_indices: tuple[int, ...],
    internal_indices: tuple[int, ...],
    primitive_index: dict[str, int],
    rank_tolerance: float,
) -> tuple[_Audit, ...]:
    count = len(selections)
    maximum = np.zeros(count, dtype=float)
    total = np.zeros(count, dtype=float)
    full_rank = np.ones(count, dtype=bool)
    worst = np.zeros(count, dtype=int)
    selection_indices = np.asarray(
        tuple(
            tuple(primitive_index[primitive_id] for primitive_id in selection)
            for selection in selections
        ),
        dtype=int,
    )
    inter_position = {gic_index: position for position, gic_index in enumerate(inter_indices)}
    support_positions = tuple(inter_position[slot] for slot in support_slots)
    for sample_index, sample in enumerate(cache):
        rows = np.broadcast_to(
            sample.gic_rows[np.asarray(inter_indices, dtype=int)],
            (count, len(inter_indices), sample.gic_rows.shape[1]),
        ).copy()
        for column, position in enumerate(support_positions):
            rows[:, position, :] = sample.primitive_rows[selection_indices[:, column]]
        fixed = sample.gic_rows[np.asarray(internal_indices, dtype=int)]
        projector = np.linalg.pinv(fixed) @ fixed
        rows -= rows @ projector
        norms = np.linalg.norm(rows, axis=2)
        regular = np.all(np.isfinite(norms) & (norms > 1.0e-12), axis=1)
        normalized = np.divide(
            rows,
            norms[:, :, None],
            out=np.zeros_like(rows),
            where=norms[:, :, None] > 1.0e-12,
        )
        singular = np.linalg.svd(normalized, compute_uv=False)
        ranks = np.count_nonzero(
            singular > rank_tolerance * singular[:, :1],
            axis=1,
        )
        conditions = np.divide(
            singular[:, 0],
            singular[:, -1],
            out=np.full(count, np.inf),
            where=singular[:, -1] > 0.0,
        )
        conditions[~regular] = np.inf
        improved_worst = conditions >= maximum
        worst[improved_worst] = sample_index
        maximum = np.maximum(maximum, conditions)
        total += conditions
        full_rank &= regular & (ranks == len(inter_indices))
    return tuple(
        _Audit(
            maximum_condition=float(maximum[index]),
            mean_condition=float(total[index] / len(cache)),
            worst_sample=cache[int(worst[index])].label,
            full_rank=bool(full_rank[index]),
        )
        for index in range(count)
    )


def _normalized_condition(
    rows: np.ndarray,
    *,
    rank_tolerance: float,
) -> tuple[float, int]:
    norms = np.linalg.norm(rows, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        return float("inf"), 0
    singular = np.linalg.svd(rows / norms[:, None], compute_uv=False)
    rank = int(np.count_nonzero(singular > rank_tolerance * singular[0]))
    condition = float(singular[0] / singular[-1]) if singular[-1] else float("inf")
    return condition, rank


def _internal_gic_indices(definition: GICDefinition) -> tuple[int, ...]:
    return tuple(
        index for index, gic in enumerate(definition.gics) if gic.family not in _INTER_FAMILIES
    )




def _selected_support_slots(
    definition: GICDefinition,
    *,
    pair: tuple[str, str] | None = None,
) -> tuple[int, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    slots = []
    for index, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        primitives = tuple(primitive_by_id[item[0]] for item in coefficients)
        if primitives and all(
            item.refs
            and item.refs[0] == _CONTACT_SUPPORT_REF
            and (pair is None or _support_pair(item) == pair)
            for item in primitives
        ):
            slots.append(index)
    return tuple(slots)


def _selected_support_is_component_complete(
    definition: GICDefinition,
    *,
    support_slots: tuple[int, ...],
    support_pool: tuple[GICPrimitive, ...],
) -> bool:
    try:
        primitive_index = {
            primitive.identifier: index for index, primitive in enumerate(support_pool)
        }
        transform = np.zeros((len(support_pool), len(support_slots)))
        for column, slot in enumerate(support_slots):
            gic = definition.gics[slot]
            for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
                transform[primitive_index[primitive_id], column] += float(coefficient)
        components = tuple(
            CoordinateComponent(
                operator=primitive.function,
                atoms=primitive.atoms,
                mode=primitive.mode,
                ref_atoms=primitive.ref_atoms,
                context=(primitive.family, *primitive.refs),
            )
            for primitive in support_pool
        )
        validate_coordinate_component_transform(components, transform)
    except (KeyError, ValueError):
        return False
    return bool(support_slots)


def _support_pair(primitive: GICPrimitive) -> tuple[str, str]:
    fragments = tuple(sorted(ref for ref in primitive.refs[1:] if ref.startswith("F")))
    if len(fragments) != 2:
        raise GICForgeContractError(
            f"contact-support primitive {primitive.identifier} lacks one fragment pair"
        )
    return fragments


def _required_contact_slots(
    definition: GICDefinition,
    *,
    pair: tuple[str, str],
    fragment_by_atom: dict[int, str],
) -> tuple[int, ...]:
    primitive_by_id = {item.identifier: item for item in definition.primitives}
    slots: list[int] = []
    for index, gic in enumerate(definition.gics):
        if gic.family != "PSEUDO_BOND_DISTANCE":
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        owners = {
            tuple(sorted(fragment_by_atom[atom] for atom in primitive_by_id[primitive_id].atoms))
            for primitive_id, _coefficient in coefficients
        }
        if owners == {pair}:
            slots.append(index)
    return tuple(slots)


def _combine_audits(audits: list[_Audit]) -> _Audit:
    if not audits:
        raise GICForgeContractError("a pseudobond chart has no fragment-pair audit")
    worst = max(audits, key=lambda item: item.maximum_condition)
    return _Audit(
        maximum_condition=worst.maximum_condition,
        mean_condition=float(np.mean([item.mean_condition for item in audits])),
        worst_sample=worst.worst_sample,
        full_rank=all(item.full_rank for item in audits),
    )


def _support_ids(
    definition: GICDefinition,
    support_slots: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(
        (definition.gics[index].coefficients or ((definition.gics[index].primitive_id, 1.0),))[0][0]
        for index in support_slots
    )


def _substitution_blocks(
    contract: OracleCoordinateAtlasContract,
    *,
    prefix: str,
) -> dict[str, object]:
    return {
        item.block_id: item
        for item in contract.family_compatibility
        if item.substitutions
        and item.block_id.startswith(prefix)
        and item.condition_trigger is not None
        and item.minimum_relative_gain is not None
    }


def _contact_condition_policy(
    contract: OracleCoordinateAtlasContract,
) -> tuple[float, float]:
    matches = tuple(
        item
        for item in contract.family_compatibility
        if item.block_id == "MINIMUM_CONTACT::PSEUDOBOND" and item.substitutions
    )
    if len(matches) != 1:
        raise GICForgeContractError(
            "ORACLE atlas must define one MINIMUM pseudobond compatibility block"
        )
    item = matches[0]
    if item.condition_trigger is None or item.minimum_relative_gain is None:
        raise GICForgeContractError("ORACLE MINIMUM pseudobond block lacks conditioning thresholds")
    return float(item.condition_trigger), float(item.minimum_relative_gain)


def _primitive_prescriptions(
    definition: GICDefinition,
    *,
    contract: OracleCoordinateAtlasContract,
    sonic_contract: OracleSonicContract,
) -> dict[str, _PrimitiveAtlasEntry]:
    atlas_by_candidate_id = {item.candidate_id: item for item in contract.candidates}
    oracle_by_signature = {
        (item.function, item.atoms, item.mode, item.ref_atoms): item
        for item in sonic_contract.primitive_candidates
    }
    prescriptions: dict[str, _PrimitiveAtlasEntry] = {}
    for primitive in definition.primitives:
        candidate = oracle_by_signature.get(
            (primitive.function, primitive.atoms, primitive.mode, primitive.ref_atoms)
        )
        if candidate is None:
            continue
        prescription = atlas_by_candidate_id.get(candidate.candidate_id)
        if prescription is not None:
            prescriptions[primitive.identifier] = _PrimitiveAtlasEntry(
                prescription=prescription,
                domain_id=candidate.domain_id,
            )
    return prescriptions


def _eligible_internal_exchanges(
    definition: GICDefinition,
    *,
    primitive_by_id: dict[str, GICPrimitive],
    prescription_by_id: dict[str, _PrimitiveAtlasEntry],
    substitution_blocks: frozenset[str],
) -> tuple[_InternalExchange, ...]:
    selected_ids = {
        primitive_id
        for gic in definition.gics
        for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
    }
    alternatives: dict[tuple[str, str, str], list[str]] = {}
    for primitive in definition.primitives:
        entry = prescription_by_id.get(primitive.identifier)
        if (
            entry is None
            or entry.prescription.requirement != CANDIDATE_OPTIONAL
            or entry.prescription.mixing_block not in substitution_blocks
        ):
            continue
        alternatives.setdefault(
            (
                entry.prescription.mixing_block,
                primitive.family,
                entry.domain_id,
            ),
            [],
        ).append(primitive.identifier)
    exchanges: list[_InternalExchange] = []
    for slot, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or abs(float(coefficients[0][1]) - 1.0) > 1.0e-12:
            continue
        source_id = coefficients[0][0]
        source = primitive_by_id[source_id]
        if source.family in _INTER_FAMILIES:
            continue
        entry = prescription_by_id.get(source_id)
        if entry is None or entry.prescription.mixing_block not in substitution_blocks:
            continue
        for alternative_id in alternatives.get(
            (
                entry.prescription.mixing_block,
                source.family,
                entry.domain_id,
            ),
            (),
        ):
            if alternative_id == source_id or alternative_id in selected_ids:
                continue
            exchanges.append(
                _InternalExchange(
                    slot=slot,
                    source_id=source_id,
                    alternative_id=alternative_id,
                    block_id=entry.prescription.mixing_block,
                )
            )
    return tuple(
        sorted(
            exchanges,
            key=lambda item: (item.slot, item.alternative_id, item.block_id),
        )
    )


def _compatible_exchange_set(exchanges: tuple[_InternalExchange, ...]) -> bool:
    return len({item.slot for item in exchanges}) == len(exchanges) and len(
        {item.alternative_id for item in exchanges}
    ) == len(exchanges)


def _apply_internal_exchanges(
    definition: GICDefinition,
    exchanges: tuple[_InternalExchange, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICDefinition:
    gics = list(definition.gics)
    replacements: dict[str, str] = {}
    for exchange in exchanges:
        source = gics[exchange.slot]
        primitive = primitive_by_id[exchange.alternative_id]
        replacements[exchange.source_id] = exchange.alternative_id
        gics[exchange.slot] = replace(
            source,
            name=f"{source.irrep}{primitive.name}",
            family=primitive.family,
            primitive_id=primitive.identifier,
            gaussian_expression=primitive.gaussian_expression(),
            coefficients=((primitive.identifier, 1.0),),
        )
    diagnostics = _replace_reduction_selection(
        definition.reduction_diagnostics,
        replacements=replacements,
        primitive_by_id=primitive_by_id,
    )
    return replace(definition, gics=tuple(gics), reduction_diagnostics=diagnostics)


def _replace_reduction_selection(
    diagnostics: GICReductionDiagnostics | None,
    *,
    replacements: dict[str, str],
    primitive_by_id: dict[str, GICPrimitive],
) -> GICReductionDiagnostics | None:
    if diagnostics is None or not replacements:
        return diagnostics
    selected = tuple(replacements.get(item, item) for item in diagnostics.selected)
    skipped = set(diagnostics.skipped_dependent)
    skipped.difference_update(replacements.values())
    skipped.update(replacements)
    details = tuple(
        f"{identifier}:{primitive_by_id[identifier].family}:{primitive_by_id[identifier].name}"
        for identifier in sorted(skipped)
    )
    return replace(
        diagnostics,
        selected=selected,
        skipped_dependent=tuple(sorted(skipped)),
        skipped_dependent_details=details,
    )


def _synchronize_reduction_selection(
    definition: GICDefinition,
    *,
    primitive_by_id: dict[str, GICPrimitive],
) -> GICReductionDiagnostics | None:
    diagnostics = definition.reduction_diagnostics
    if diagnostics is None:
        return None
    support_pool = {
        identifier
        for identifier, primitive in primitive_by_id.items()
        if primitive.refs and primitive.refs[0] == _CONTACT_SUPPORT_REF
    }
    active_support = tuple(
        dict.fromkeys(
            primitive_id
            for gic in definition.gics
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            if primitive_id in support_pool
        )
    )
    previous = list(diagnostics.selected)
    insertion = next(
        (index for index, identifier in enumerate(previous) if identifier in support_pool),
        len(previous),
    )
    retained = [identifier for identifier in previous if identifier not in support_pool]
    retained[insertion:insertion] = active_support
    selected = tuple(retained)
    skipped_set = set(diagnostics.skipped_dependent)
    skipped_set.update(support_pool.difference(active_support))
    skipped_set.difference_update(active_support)
    skipped = tuple(sorted(skipped_set))
    details = tuple(
        f"{identifier}:{primitive_by_id[identifier].family}:{primitive_by_id[identifier].name}"
        for identifier in skipped
    )
    return replace(
        diagnostics,
        selected=selected,
        skipped_dependent=skipped,
        skipped_dependent_details=details,
    )


def _material_gain(
    baseline: float,
    candidate: float,
    *,
    minimum_gain: float,
) -> bool:
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(baseline):
        return True
    return (baseline - candidate) / max(baseline, 1.0e-12) >= minimum_gain


def _support_locality(
    primitive_ids: tuple[str, ...],
    *,
    primitive_by_id: dict[str, GICPrimitive],
    sonic_contract: OracleSonicContract,
    contract: OracleCoordinateAtlasContract,
    pair: tuple[str, str],
) -> int:
    contact_atoms = {
        int(endpoint[1])
        for item in contract.interactions
        if item.pseudobond_policy == PSEUDOBOND_REQUIRED
        and tuple(sorted(item.fragment_ids)) == pair
        for endpoint in (item.endpoint_a, item.endpoint_b)
        if endpoint[0] == "ATOM"
    }
    adjacency = _adjacency(
        sonic_contract.primary_topology.natoms,
        sonic_contract.primary_topology.bonds,
    )
    distances = _graph_distances(contact_atoms, adjacency)
    score = 0
    for primitive_id in primitive_ids:
        primitive = primitive_by_id[primitive_id]
        score += sum(distances.get(atom, len(adjacency)) for atom in primitive.atoms)
        score += sum(distances.get(atom, len(adjacency)) for atom in primitive.ref_atoms)
    return score


def _adjacency(
    natoms: int,
    bonds: tuple[tuple[int, int], ...],
) -> dict[int, set[int]]:
    adjacency = {atom: set() for atom in range(1, natoms + 1)}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _graph_distances(
    roots: set[int],
    adjacency: dict[int, set[int]],
) -> dict[int, int]:
    distances = {root: 0 for root in roots}
    frontier = list(sorted(roots))
    for atom in frontier:
        for neighbor in sorted(adjacency.get(atom, ())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[atom] + 1
            frontier.append(neighbor)
    return distances


def _with_partition_diagnostics(
    definition: GICDefinition,
    *,
    sample_count: int,
    internal_exchange: tuple[str, ...],
    support_ids: tuple[str, ...],
    intra_before: _Audit,
    intra_after: _Audit,
    inter_before: _Audit,
    inter_after: _Audit,
    global_after: _Audit,
) -> GICDefinition:
    status = "SEPARATED_SELECTION" if internal_exchange else "SEPARATED_CANONICAL_INTRA"
    diagnostics = tuple(
        item
        for item in definition.semantic_diagnostics
        if not item.startswith("MINIMUM_LOCAL_DOMAIN ")
        and not item.startswith("MINIMUM_DOMAIN_PARTITION ")
    )
    diagnostics += (
        f"MINIMUM_DOMAIN_PARTITION SCHEMA={MINIMUM_DOMAIN_SELECTION_SCHEMA} "
        f"STATUS={status} L_COMPONENTS=COMPLETE",
        f"MINIMUM_DOMAIN_PARTITION INTRA_SAMPLES=1 "
        f"BASE_MAX_K={intra_before.maximum_condition:.12g} "
        f"SELECTED_MAX_K={intra_after.maximum_condition:.12g} "
        f"EXCHANGES={','.join(internal_exchange) if internal_exchange else 'NONE'}",
        f"MINIMUM_DOMAIN_PARTITION INTER_SAMPLES={sample_count} "
        f"BASE_MAX_K={inter_before.maximum_condition:.12g} "
        f"SELECTED_MAX_K={inter_after.maximum_condition:.12g} "
        f"WORST_SAMPLE={inter_after.worst_sample} "
        f"SUPPORT={','.join(support_ids)}",
        f"MINIMUM_DOMAIN_PARTITION GLOBAL_MAX_K={global_after.maximum_condition:.12g} "
        f"GLOBAL_WORST_SAMPLE={global_after.worst_sample}",
    )
    return replace(definition, semantic_diagnostics=diagnostics)


__all__ = ["MINIMUM_DOMAIN_SELECTION_SCHEMA", "condition_minimum_pseudobond_chart"]
