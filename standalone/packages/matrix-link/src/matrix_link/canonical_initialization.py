"""Approved adaptive initialization v2 for the canonical QM ladder.

The historical initialization protocol v1.1 remains frozen and is neither
loaded nor reinterpreted by this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol_manifest import LINK_PROTOCOL_ID, LINK_PROTOCOL_VERSION


CANONICAL_INITIALIZATION_SCHEMA = "matrix.link.canonical_initialization_protocol.v2"
CANONICAL_INITIALIZATION_ID = "matrix-canonical-initialization-v2"
CANONICAL_INITIALIZATION_VERSION = "2.3.11"


@dataclass(frozen=True)
class CanonicalSeedRoute:
    target: str
    electronic_state: str
    spin: str
    primary: str
    progressive_candidates: tuple[str, ...]
    target_hessian_policy: str
    xtb_available: bool


@dataclass(frozen=True)
class CanonicalInitializationProtocol:
    payload: dict[str, Any]
    sha256: str
    source: str

    def geometry_route(self, geometry_class: str) -> tuple[str, ...]:
        try:
            route = self.payload["geometry_routes"][geometry_class]
        except KeyError as exc:
            raise ValueError(
                "geometry_class must be credible_xyz, noncredible_or_generated, or invalid"
            ) from exc
        return tuple(str(stage) for stage in route)

    def seed_route(
        self,
        target: str,
        *,
        electronic_state: str,
        spin: str,
        xtb_available: bool,
    ) -> CanonicalSeedRoute:
        target_key = str(target).strip().upper()
        state_key = _state_route_key(electronic_state, spin)
        try:
            record = self.payload["seed_routes"][state_key][target_key]
        except KeyError as exc:
            raise RuntimeError(
                f"canonical initialization is unsupported for "
                f"{target_key}/{electronic_state}/{spin}"
            ) from exc
        availability = "with_xtb" if xtb_available else "without_xtb"
        primary = record.get(f"primary_{availability}", record.get("primary"))
        if primary is None:
            raise RuntimeError("canonical initialization route has no primary seed")
        return CanonicalSeedRoute(
            target=target_key,
            electronic_state=electronic_state,
            spin=spin,
            primary=str(primary),
            progressive_candidates=tuple(
                str(item) for item in record[f"progressive_{availability}"]
            ),
            target_hessian_policy=str(record["target_hessian"]),
            xtb_available=bool(xtb_available),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def load_canonical_initialization_protocol() -> CanonicalInitializationProtocol:
    resource = resources.files("matrix_link").joinpath(
        "data/canonical_initialization_protocol_manifest.json"
    )
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical MATRIX initialization v2 is unreadable") from exc
    validate_canonical_initialization_protocol(payload)
    return CanonicalInitializationProtocol(
        payload=deepcopy(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_link:data/canonical_initialization_protocol_manifest.json",
    )


def canonical_initialization_schema_path() -> Path:
    return Path(
        str(
            resources.files("matrix_link").joinpath(
                "data/canonical_initialization_protocol_manifest.schema.json"
            )
        )
    )


def validate_canonical_initialization_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CANONICAL_INITIALIZATION_SCHEMA:
        raise RuntimeError("unsupported canonical MATRIX initialization schema")
    if (
        payload.get("protocol_id") != CANONICAL_INITIALIZATION_ID
        or payload.get("manifest_version") != CANONICAL_INITIALIZATION_VERSION
        or payload.get("status") != "approved_validation"
        or payload.get("replaces_legacy_protocol") is not False
    ):
        raise RuntimeError("canonical MATRIX initialization v2 identity or status changed")
    if payload.get("canonical_ladder") != {
        "protocol_id": "matrix-canonical-qm-ladder-v1",
        "manifest_version": "1.0.11",
    }:
        raise RuntimeError("canonical initialization ladder authority changed")
    if payload.get("optimizer_contract") != {
        "protocol_id": LINK_PROTOCOL_ID,
        "manifest_version": LINK_PROTOCOL_VERSION,
        "settings_duplicated": False,
        "required_observation": "GDIIS_first_eligible_at_optimizer_iteration_3",
    }:
        raise RuntimeError("canonical initialization optimizer authority changed")
    if payload.get("owners") != {
        "geometry_quality_and_symmetry": "ORACLE",
        "structural_preparation": "ARCHITECT",
        "connectivity_seed": "SWITCH",
        "coordinate_definition": "SMITH",
        "hessian_audit_and_optimization": "LINK",
        "scientific_and_backend_resolution": "MATRIX_QM",
        "electronic_state_identity": "APOC",
        "human_orchestration": "KEYMAKER",
    }:
        raise RuntimeError("canonical initialization owner boundaries changed")
    if payload.get("governance") != {
        "inspect_existing_implementation_before_change": True,
        "reuse_existing_algorithms": "mandatory",
        "parallel_reimplementation": "forbidden",
        "silent_scientific_substitution": False,
        "molecule_specific_patches": False,
        "backend_specific_optimizer_patches": False,
        "method_specific_optimizer_patches": False,
        "change_policy": "new_manifest_version_and_explicit_approval",
    }:
        raise RuntimeError("canonical initialization governance changed")

    geometry = payload.get("geometry_routes", {})
    if set(geometry) != {"credible_xyz", "noncredible_or_generated", "invalid"}:
        raise RuntimeError("canonical initial-geometry route set changed")
    if not geometry["credible_xyz"] or geometry["credible_xyz"][0] != (
        "ORACLE_detect_and_symmetrize"
    ):
        raise RuntimeError("credible XYZ must always begin with ORACLE symmetry")
    if geometry["noncredible_or_generated"] != [
        "SWITCH_connectivity_and_crude_seed_if_source_is_SMILES",
        "ORACLE_detect_topology_quality_and_symmetrize",
        "ARCHITECT_refine_with_GFN_FF_or_UFF_when_xTB_unavailable",
        "ORACLE_reanalyse_and_symmetrize",
        "SMITH_build_frozen_symmetry_adapted_SONIC",
        "LINK_seed_hessian_audit",
    ]:
        raise RuntimeError("noncredible geometry owner chain changed")
    if geometry["invalid"] != ["stop_before_electronic_structure_evaluation"]:
        raise RuntimeError("invalid geometry must stop before QM evaluation")

    routes = payload.get("seed_routes", {})
    if set(routes) != {"ground", "excited_closed_shell", "excited_open_shell"}:
        raise RuntimeError("canonical seed-route state classes changed")
    if set(routes["ground"]) != {
        "PRE0",
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
        "DL2",
        "FL4",
        "PL4",
    }:
        raise RuntimeError("canonical ground-state seed targets changed")
    if set(routes["excited_closed_shell"]) != {"L0", "L1", "L3", "L4"}:
        raise RuntimeError("canonical closed-shell excited seed targets changed")
    if set(routes["excited_open_shell"]) != {"L0"}:
        raise RuntimeError("canonical open-shell excited seed targets changed")
    for state_routes in routes.values():
        for target, record in state_routes.items():
            required = {
                "progressive_with_xtb",
                "progressive_without_xtb",
                "target_hessian",
            }
            if not required <= set(record) or not (
                "primary" in record or {"primary_with_xtb", "primary_without_xtb"} <= set(record)
            ):
                raise RuntimeError(f"canonical seed route is incomplete: {target}")
            if not record["progressive_with_xtb"] or not record["progressive_without_xtb"]:
                raise RuntimeError(f"canonical seed route has an empty candidate chain: {target}")

    pre0 = routes["ground"]["PRE0"]
    if pre0["primary"] != "Lindh_1995_ANC_plus_Swart_2006_special" or pre0["progressive_with_xtb"] != [
        "Lindh_1995_ANC_plus_Swart_2006_special",
        "Almloef",
    ]:
        raise RuntimeError(
            "GFN2-xTB target must try Lindh-1995 ANC plus Swart-special and audit Almlöf only as fallback"
        )
    l0 = routes["ground"]["L0"]
    if l0["primary_with_xtb"] != "PRE0" or l0["primary_without_xtb"] != "Lindh_1995_ANC_plus_Swart_2006_special":
        raise RuntimeError("L0 seed route must not label L0 as its own lower level")
    expected_ground_primary = {
        "L1": "L0",
        "L2": "L1",
        "L3": "L2",
        "L4": "L3",
        "DL2": "L1",
        "FL4": "L3",
        "PL4": "L3",
    }
    for target, primary in expected_ground_primary.items():
        if routes["ground"][target]["primary"] != primary:
            raise RuntimeError(f"canonical {target} primary seed must remain {primary}")
    for target in ("L1", "L2", "L3", "L4", "DL2", "FL4", "PL4"):
        record = routes["ground"][target]
        if record["progressive_with_xtb"][0] != "PRE0":
            raise RuntimeError(f"{target} must retain PRE0 as preferred minimum preparation")
        if record["progressive_without_xtb"][0] != "L0":
            raise RuntimeError(f"{target} must use L0 when xTB is unavailable")
    if routes["ground"]["FL4"]["primary"] != "L3":
        raise RuntimeError("FL4 must use the L3 initial Hessian")
    if routes["ground"]["PL4"]["primary"] != "L3":
        raise RuntimeError("PL4 must use the L3 initial Hessian")

    hessian = payload.get("seed_hessian_contract", {})
    required_source_contract = {
        "canonical_internal_model": "Lindh_1995_ANC_plus_Swart_2006_special",
        "internal_model_fallback": (
            "Fischer_Almloef_only_after_explicit_seed_audit_failure"
        ),
        "model_hessian_source_coordinates": (
            "Lindh_cartesian_base_plus_SMITH_declared_special_interactions_on_"
            "pseudobond_source_independent_of_active_optimization_chart"
        ),
        "force_field_hessian_source_coordinates": (
            "SMITH_pseudobonds_independent_of_active_optimization_chart"
        ),
        "qm_hessian_source_coordinates": "active_frozen_SONIC",
        "special_edge_policy": (
            "any_ORACLE_SMITH_declared_special_edge_or_center_uses_declared_"
            "effective_order_independent_of_molecule_or_hapticity"
        ),
    }
    if any(hessian.get(key) != value for key, value in required_source_contract.items()):
        raise RuntimeError("canonical seed-Hessian source-coordinate contract changed")
    if hessian.get("transform") != "linear_congruence_to_frozen_SONIC_without_B_prime":
        raise RuntimeError("seed Hessian transformation must omit B-prime")
    if hessian.get("final_physical_transform") != "include_B_prime":
        raise RuntimeError("final physical Hessian transformation must include B-prime")
    if hessian.get("soft_stiff_policy") != (
        "diagnose_and_scale_but_never_delete_physical_cross_couplings"
    ):
        raise RuntimeError("physical soft/stiff Hessian couplings may not be deleted")
    if hessian.get("threshold_status") != "provisional_requires_dedicated_validation":
        raise RuntimeError("unvalidated seed-Hessian thresholds may not be frozen silently")

    outcomes = payload.get("audit_outcomes", {})
    if outcomes.get("ambiguous") != "stop_without_guessing":
        raise RuntimeError("ambiguous seed-Hessian audits must fail closed")
    if outcomes.get("provider_inadequate") != ("advance_to_next_declared_lower_level_candidate"):
        raise RuntimeError("seed-Hessian escalation policy changed")
    stage = payload.get("stage_contract", {})
    if stage.get("symmetry") != (
        "ORACLE_symmetrizes_every_geometry_and_every_gradient_is_symmetrized_at_each_iteration"
    ):
        raise RuntimeError("canonical symmetry contract changed")
    if stage.get("unsupported") != (
        "fail_closed_without_method_basis_reference_or_backend_substitution"
    ):
        raise RuntimeError("unsupported canonical initialization must fail closed")


def _state_route_key(electronic_state: str, spin: str) -> str:
    state = str(electronic_state).strip().casefold()
    spin_key = str(spin).strip().casefold()
    if state == "ground" and spin_key in {"closed_shell", "open_shell"}:
        return "ground"
    if state == "excited" and spin_key == "closed_shell":
        return "excited_closed_shell"
    if state == "excited" and spin_key == "open_shell":
        return "excited_open_shell"
    raise ValueError("electronic_state/spin must use canonical ground/excited shell labels")


__all__ = [
    "CANONICAL_INITIALIZATION_ID",
    "CANONICAL_INITIALIZATION_SCHEMA",
    "CANONICAL_INITIALIZATION_VERSION",
    "CanonicalInitializationProtocol",
    "CanonicalSeedRoute",
    "canonical_initialization_schema_path",
    "load_canonical_initialization_protocol",
    "validate_canonical_initialization_protocol",
]
