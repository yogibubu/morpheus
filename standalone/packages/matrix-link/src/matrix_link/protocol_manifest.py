"""Frozen runtime contract for the LINK optimizer.

The package resource is the sole protocol authority.  Every LINK run validates
it fail-closed and writes an invocation manifest that records the exact backend,
method, basis and gradient mode without changing the scientific protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping


LINK_PROTOCOL_SCHEMA = "matrix.link.optimizer.protocol_manifest.v2"
LINK_RUNTIME_METHOD_SCHEMA = "matrix.link.runtime_method_manifest.v1"
LINK_PROTOCOL_ID = "link-sonic-optimizer-v2"
LINK_PROTOCOL_VERSION = "2.32.0"
LINK_METHOD_MANIFEST_SCHEMA = "matrix.link.method_manifest.v1"
LINK_METHOD_MANIFEST_DIR = "config/link_optimizer_method_manifests"


@dataclass(frozen=True)
class LinkOptimizerProtocol:
    payload: dict[str, Any]
    sha256: str
    source: str

    @property
    def protocol_id(self) -> str:
        return str(self.payload["protocol_id"])

    @property
    def manifest_version(self) -> str:
        return str(self.payload["manifest_version"])


def load_link_optimizer_protocol() -> LinkOptimizerProtocol:
    """Load and strictly validate the single packaged LINK protocol."""

    resource = resources.files("matrix_link").joinpath(
        "data/optimizer_protocol_manifest.json"
    )
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LINK optimizer protocol manifest is unreadable") from exc
    _validate_link_optimizer_protocol(payload)
    return LinkOptimizerProtocol(
        payload=dict(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_link:data/optimizer_protocol_manifest.json",
    )


def _validate_link_optimizer_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != LINK_PROTOCOL_SCHEMA:
        raise RuntimeError(f"unsupported LINK optimizer protocol schema: {payload.get('schema')!r}")
    if payload.get("protocol_id") != LINK_PROTOCOL_ID or payload.get("status") != "frozen":
        raise RuntimeError("LINK optimizer protocol must be the frozen canonical protocol")
    if payload.get("manifest_version") != LINK_PROTOCOL_VERSION:
        raise RuntimeError(
            f"LINK optimizer protocol version is not the confirmed "
            f"{LINK_PROTOCOL_VERSION} contract"
        )
    algorithm_governance = payload.get("algorithm_governance", {})
    if algorithm_governance != {
        "inspect_existing_implementation_before_any_change": True,
        "reuse_existing_algorithms_and_shared_components": "mandatory",
        "parallel_reimplementation_of_existing_capability": "forbidden",
        "molecule_specific_optimizer_patches": False,
        "molecule_identity_in_optimizer_policy": "forbidden",
        "backend_specific_optimizer_patches": False,
        "generality_requirement": (
            "algorithm_depends_only_on_declared_mathematical_and_physical_properties"
        ),
        "exception_policy": "new_manifest_version_and_explicit_user_confirmation",
    }:
        raise RuntimeError(
            "LINK algorithm-governance contract differs from the confirmed frozen contract"
        )
    if payload.get("coordinate_contract") != {
        "optimizer_variables": "active_nonredundant_SONIC_within_chart_epoch",
        "derivative_projection": "authoritative_current_geometry_Jacobian_covector_projection",
        "geometry_realization": "authoritative_nonlinear_SONIC_backtransform",
        "metric": "complete_physical_active_Wilson_metric_in_bohr",
        "coordinate_family_metadata": "sole_source_for_soft_stiff_classification",
        "hbond_pseudobond_provenance": (
            "SMITH_HBOND_DISTANCE_soft_family_without_covalent_STRETCH_mixing"
        ),
        "molecule_identity_allowed": False,
        "chart_identity_scope": "frozen_within_each_accepted_state_epoch",
        "finite_motion_validity": (
            "successful_nonlinear_realization_plus_full_rank_bounded_condition_current_Jacobian"
        ),
        "cumulative_cartesian_drift": "telemetry_only_not_a_chart_invalidity_criterion",
    }:
        raise RuntimeError("LINK coordinate contract differs from the confirmed frozen contract")
    if payload.get("convergence_contract") != {
        "logic": (
            "GDV_four_force_displacement_criteria_for_transition_states_and_"
            "five_LINK_criteria_for_minima"
        ),
        "energy_unit": "hartree",
        "gradient_unit": "hartree_per_bohr",
        "transition_state_gradient_metric": "GDV_active_internal_coordinate_components",
        "transition_state_energy_change": (
            "diagnostic_only_not_a_GDV_convergence_condition"
        ),
        "displacement_unit": "bohr",
        "displacement_metric": "Kabsch_aligned_cartesian_components",
        "accepted_state_source": (
            "current_evaluated_geometry_plus_GDV_proposed_displacement_for_"
            "transition_states"
        ),
        "prospective_unevaluated_step_may_converge": (
            "GDV_transition_state_displacement_test_only"
        ),
        "prospective_geometry_energy_or_gradient_used": False,
        "stationary_zero_step_policy": "accept_repeat_of_current_evaluated_state",
        "rejected_trial_may_replace_last_accepted_displacement": False,
        "trace_summary_and_runtime_use_identical_values": True,
        "geometry_seed": {
            "role": "lower_level_geometry_preparation",
            "reference_profile": "xTB_normal_equivalent",
            "energy_tolerance_hartree": 5.0e-6,
            "gradient_norm_tolerance_hartree_per_bohr": 1.0e-3,
            "final_energy_rise_tolerance_hartree": 1.0e-10,
        },
    }:
        raise RuntimeError("LINK convergence contract differs from the confirmed frozen contract")
    if payload.get("cartesian_trust_region") != {
        "norm": "Kabsch_aligned_cartesian_atomic_RMS_angstrom",
        "realization": "authoritative_nonlinear_backtransform_at_every_boundary_iteration",
        "boundary_solver": (
            "Wilson_tangent_seeded_exact_nonlinear_safeguarded_bracketed_scalar_root_search"
        ),
        "coverage": [
            "restricted_RFO",
            "preconditioned_Cauchy",
            "controlled_GDIIS",
            "inter_intra_microiteration",
        ],
        "radius_update_fraction": "minimum_realized_cartesian_atomic_RMS_over_applied_radius",
            "rejected_trial_radius": (
                "max_min_radius_and_half_minimum_of_current_radius_and_"
                "rejected_realized_cartesian_atomic_RMS"
            ),
            "pre_evaluation_rejection_scale": (
                "authoritative_nonlinear_cartesian_RMS_of_rejected_candidate_"
                "not_current_state_alias"
            ),
            "unrealizable_step": (
                "reject_before_QM_evaluation_on_backtransform_failure_nonfinite_geometry_"
                "rank_loss_or_unbounded_Jacobian_condition"
            ),
        "coordinate_or_backend_specific_radius": False,
    }:
        raise RuntimeError(
            "LINK Cartesian trust-region contract differs from the confirmed frozen contract"
        )
    if payload.get("transition_state_trust_region") != {
        "step_basis": "raw_active_GIC_DiagFC_eigenbasis_in_GDV_bohr_radian_units",
        "norm": (
            "GDV_DXOpt_raw_internal_bohr_radian_norm_OptDX_for_native_GIC_or_"
            "CrdGRo_IOp5_force_constant_weights_for_generic_GIC"
        ),
        "initial_radius": 0.3,
        "minimum_radius": 0.3,
        "maximum_radius": 0.3,
        "radius_update": "GDV_l103_default_TS_UpTrus_false_fixed_DXMaxT",
        "cartesian_realization": (
            "GDV_RedQ2X_after_internal_scaling_with_GrdOpt_MxITry_10_"
            "and_half_ScFact_after_each_RedCar_failure"
        ),
    }:
        raise RuntimeError(
            "LINK transition-state trust-region contract differs from GDV"
        )
    if payload.get("execution_efficiency_contract") != {
        "scientific_model_change": False,
        "cartesian_realization_reuse": (
            "exact_optimizer_vector_within_immutable_chart_and_projector_state"
        ),
        "cartesian_realization_consumers": [
            "trust_region",
            "geometry_validity",
            "backend_dispatch",
        ],
        "differential_reuse": (
            "exact_cartesian_geometry_within_immutable_chart_and_projector_state"
        ),
        "trust_boundary_probe": (
            "primary_nonlinear_realization_without_exhaustive_failure_recovery_then_"
            "unchanged_geometry_validity_gate"
        ),
        "trust_boundary_initial_scale": (
            "Wilson_cartesian_tangent_prediction_clipped_to_unit_interval_then_"
            "authoritative_exact_nonlinear_verification"
        ),
        "production_realization": (
            "exhaustive_failure_recovery_remains_available_outside_trust_boundary_screening"
        ),
        "failed_continuation_waypoint": "stop_before_later_waypoints",
        "cache_invalidation": (
            "mandatory_on_chart_projector_or_rotation_chart_mutation"
        ),
        "backend_evaluation_change": False,
    }:
        raise RuntimeError(
            "LINK execution-efficiency contract differs from the confirmed frozen contract"
        )
    if payload.get("hessian_contract") != {
        "primitive_to_SONIC": (
            "physical_cartesian_tangent_congruence_without_coefficient_pseudoinverse"
        ),
        "initial_seed_order": [
            "Lindh_1995_ANC_cartesian_plus_Swart_special_on_SMITH_pseudobond_source",
            (
                "Fischer_Almloef_internal_model_on_SMITH_pseudobond_source_"
                "after_seed_audit_failure"
            ),
            "explicitly_declared_external_provider",
        ],
        "internal_model_source_coordinates": (
            "Lindh_cartesian_base_plus_SMITH_declared_special_interactions_on_"
            "pseudobond_source_independent_of_active_optimization_chart"
        ),
        "force_field_source_coordinates": (
            "SMITH_pseudobonds_independent_of_active_optimization_chart"
        ),
        "qm_source_coordinates": "active_SONIC_chart_epoch",
        "intermolecular_active_coordinates": (
            "user_selected_pseudobonds_or_exponential_mapping"
        ),
        "special_interaction_effective_order": (
            "any_ORACLE_SMITH_declared_special_edge_or_center_uses_declared_"
            "effective_order_independent_of_molecule_or_hapticity"
        ),
        "internal_model_fallback_audit": (
            "finite_positive_curvature_and_bounded_condition_number"
        ),
        "external_initial_guess_transform": (
            "linear_congruence_then_automatic_exact_sparse_B_prime_for_every_"
            "SONIC_QM_seed"
        ),
        "external_initial_guess_B_prime": (
            "initial_gradient_reuse_active_SONIC_rows_only_parallel_sparse_"
            "analytic_primitives_no_extra_QM_evaluation"
        ),
        "physical_final_hessian_transform": (
            "exact_coordinate_transform_with_B_prime_for_curvilinear_SONIC"
        ),
        "stored_model": "unconditioned_symmetric_physical_or_secant_Hessian",
        "stored_model_spectral_ceiling": (
            "minimum_Hessian_eigenvalue_floor_times_maximum_Hessian_condition_number"
        ),
        "step_model_conditioning": (
            "minimum_only_ephemeral_full_metric_orthonormal_spectral_regularization_"
            "transition_state_raw_active_GIC_spectrum_in_GDV_bohr_radian_units"
        ),
        "minimum_conditioning": (
            "step_model_only_positive_spectral_floor_with_bounded_condition_number"
        ),
        "transition_state_conditioning": (
            "none_use_raw_ordered_active_GIC_Hessian_in_GDV_bohr_radian_units_for_GDV_DXRFO"
        ),
        "transition_state_prediction_model": (
            "same_unconditioned_physical_or_secant_Hessian_used_by_GDV_DXRFO"
        ),
        "transition_seed_mode_selection": (
            "first_ordered_raw_active_GIC_Hessian_mode_in_GDV_bohr_radian_units_"
            "ModMax_1_independent_of_"
            "ORACLE_SMITH_reaction_labels"
        ),
        "transition_mode_tracking": (
            "diagnostic_overlap_only_never_changes_GDV_first_ordered_mode"
        ),
        "transition_state_index_policy": (
            "observe_and_preserve_any_raw_Hessian_index_DXRFO_maximizes_first_ordered_"
            "mode_and_uses_bounded_downhill_steps_for_additional_negative_modes"
        ),
        "transition_state_index_refresh": (
            "targeted_symmetric_multisecant_not_final_exact_Hessian_two_analytic_"
            "gradient_probes_per_independent_audited_mode"
        ),
        "far_from_minimum_update": "damped_BFGS",
        "near_minimum_update": (
            "full_damped_BFGS_minimum_GDV_D2CorX_method_4_"
            "multipoint_MSP_Bofill_transition_state"
        ),
        "minimum_BFGS_base": (
            "stored_Hessian_or_ephemeral_positive_step_model_if_"
            "full_metric_generalized_spectrum_is_not_numerically_positive_definite"
        ),
        "secant_informativity_gate": (
            "GDV_D2CorX_RMax_0.6_RMin_4x_RMS_force_tolerance_"
            "GrdErr_1e-6_immediate_previous_RMax_exempt"
        ),
        "rejected_transition_state_secant": (
            "none_DXRFO_accepts_every_finite_valid_evaluated_geometry_without_model_"
            "ratio_rejection"
        ),
        "unusable_transition_state_secant": (
            "backend_or_geometry_failure_contracts_trust_radius_without_Hessian_update"
        ),
        "rejected_transition_state_secant_guards": (
            "not_applicable_to_GDV_DXRFO_model_ratio"
        ),
        "secant_preservation": "mandatory_no_post_update_spectral_conditioning",
        "soft_stiff_decoupling": (
            "forbidden_preserve_all_physical_seed_and_secant_couplings"
        ),
        "bad_model_reset": (
            "minimum_only_rebuild_declared_chemical_seed_at_current_geometry_"
            "transition_state_preserves_D2CorX_never_metric_diagonal"
        ),
        "source_name_or_backend_identity_allowed": False,
        "molecule_identity_allowed": False,
    }:
        raise RuntimeError("LINK Hessian contract differs from the confirmed frozen contract")
    symmetry = payload.get("symmetry_contract", {})
    expected_symmetry = {
        "initial_geometry": "ORACLE_detect_and_symmetrize_before_SMITH_definition",
        "sonic_definition": "SMITH_symmetrized_from_ORACLE_geometry",
        "gradient_every_iteration": "symmetrize_after_backend_or_finite_difference_for_all_QM_backends",
        "analytic_gradient_symmetry": (
            "Cartesian_invariant_subspace_projector_before_SONIC_covector_projection"
        ),
        "numerical_gradient_symmetry": (
            "totally_symmetric_SONIC_stencil_by_construction"
        ),
        "workflow_split": (
            "exploitation_and_exploration_have_distinct_symmetry_contracts"
        ),
        "parent_group_use": (
            "exploitation_freezes_parent_group_for_minima_and_transition_states"
        ),
        "reduced_symmetry_trials": (
            "exploitation_forbids_numerical_symmetry_loss_and_restores_parent_group"
        ),
        "exploration_symmetry": (
            "instantaneous_ORACLE_reperception_without_parent_group_constraint"
        ),
        "coordinate_space": "totally_symmetric_SONIC_for_minimum_and_energy_only_optimization",
        "failure_policy": "fail_closed_if_required_symmetry_contract_is_missing",
        "immutable": True,
    }
    if symmetry != expected_symmetry:
        raise RuntimeError(
            "LINK optimizer protocol symmetry contract differs from the confirmed frozen contract"
        )
    standard_step = payload.get("standard_step", {})
    if standard_step != {
        "model": "restricted_step_rational_function_optimization",
        "transition_state_step": (
            "GDV_DXRFO_Neg_1_dual_independent_shifts_and_bounded_extra_negative_modes_"
            "on_raw_ordered_active_GIC_Hessian_in_bohr_radian_units_then_"
            "GDV_DXOpt_dispatch"
        ),
        "coordinates": "complete_physical_active_Wilson_metric_in_bohr",
        "coordinate_schedule": "joint_complete_active_space",
        "analytic_gradient_projection": "backend_cartesian_to_active_SONIC",
        "hessian_update": (
            "damped_BFGS_minimum_GDV_D2CorX_method_4_"
            "multipoint_MSP_Bofill_transition_state"
        ),
        "hessian_seed_order": [
            "Lindh_1995_ANC_cartesian_plus_Swart_special_on_SMITH_pseudobond_source",
            (
                "Fischer_Almloef_internal_model_on_SMITH_pseudobond_source_"
                "after_seed_audit_failure"
            ),
            "explicitly_declared_external_provider",
        ],
        "analytic_gradient_trial": (
            "request_energy_and_gradient_on_trial_reuse_every_finite_valid_DXRFO_"
            "geometry_without_model_ratio_rejection"
        ),
    }:
        raise RuntimeError(
            "LINK standard-step contract differs from the confirmed frozen contract"
        )
    gdiis = payload.get("gdiis", {})
    if gdiis != {
        "enabled_for": ["minimum"],
        "start_after_iterations": 3,
        "history_length": 6,
        "minimum_accepted_stable_points": 3,
        "max_condition": 5000.0,
        "max_coefficient": 2.0,
        "residual": (
            "minus_ephemeral_full_metric_conditioned_Hessian_inverse_times_gradient"
        ),
        "history_filter": "accepted_points_only_rank_revealed_stable_suffix",
        "candidate_safeguards": (
            "coefficient_bound_non_descent_and_nonpositive_model_reject_"
            "candidate_without_pruning_stable_history"
        ),
        "trust_region": "same_realized_cartesian_trust_contract_as_RFO",
        "fallback": "restricted_RFO_or_current_frozen_hessian_update",
        "safeguarded": True,
    }:
        raise RuntimeError("LINK GDIIS contract differs from the confirmed frozen contract")
    if payload.get("chart_lifecycle") != {
        "availability": "native_LINK_optimizer_only",
        "gaussian_readallgic": "fixed_chart_for_entire_external_optimization",
        "trigger": "accepted_optimizer_states_only",
        "identity_scope": (
            "persistent_basin_semantics_excluding_atom_equivalence_and_"
            "local_point_group_jitter"
        ),
        "scientific_decision_owner": "ORACLE",
        "coordinate_build_owner": "SMITH",
        "task_regimes": ["MINIMUM", "TRANSITION_STATE"],
        "persistence": "ORACLE_PerceptionBasinPolicy_persistent_change_window",
        "geometry_contract": "accepted_post_ORACLE_cartesian_geometry_unchanged_by_rebuild",
        "rank_contract": "complete_exact_vibrational_target_rank_chart_preserved",
        "minimum_task_subspace": (
            "totally_symmetric_irreps_for_non_C1_and_complete_chart_for_C1"
        ),
        "transition_state_task_subspace": "complete_exact_rank_chart",
        "transition_state_chart_policy": (
            "fixed_while_current_chart_is_numerically_valid"
        ),
        "transition_state_semantic_change": (
            "record_current_ORACLE_identity_and_outside_anchor_candidates_without_"
            "changing_frozen_reaction_kernel_or_SMITH_definition"
        ),
        "transition_state_rebuild_gate": [
            "primitive_domain_evaluation_failure",
            "ordinary_angle_near_linear_domain_failure",
            "zero_or_nonfinite_Wilson_row",
            "normalized_Wilson_rank_loss",
            "normalized_Wilson_condition_above_lifecycle_limit",
        ],
        "transition_state_linear_bend_policy": (
            "forbidden_in_frozen_reactive_zone_for_initial_and_rebuilt_charts"
        ),
        "runtime_task_view": (
            "complete_chart_retained_by_lifecycle_and_compact_task_GIC_"
            "definition_used_by_LINK"
        ),
        "omitted_irrep_policy": (
            "not_optimizer_variables_and_not_frozen_nonlinear_equations"
        ),
        "initial_hessian_boundary": (
            "complete_chart_seed_transported_by_tangent_congruence_to_runtime_task_view"
        ),
        "task_subspace_coverage": (
            "nonlinear_realization_gradient_Hessian_step_phase_convergence_secant_and_GDIIS"
        ),
        "rebuild_symmetry_policy": "inherit_initial_SMITH_symmetrize_contract",
        "epoch_contract": "monotonic_increment_only_after_validated_rebuild",
        "gradient_contract": "reproject_at_accepted_geometry_in_new_chart",
        "hessian_contract": (
            "certified_tangent_congruence_from_a_numerically_valid_previous_"
            "chart_else_canonical_chemical_seed"
        ),
        "candidate_rejection": (
            "any_SMITH_contract_failure_is_typed_unavailable_and_deferred_only_"
            "while_the_fixed_chart_remains_numerically_valid"
        ),
        "cross_epoch_secant_update": False,
        "history_contract": "reset_GDIIS_selective_and_transition_mode_history",
        "cache_contract": "chart_epoch_scoped",
        "molecule_specific_rules": False,
    }:
        raise RuntimeError("LINK chart-lifecycle contract differs from the frozen contract")
    if payload.get("far_from_minimum") != {
        "criterion": (
            "minimum_and_projected_force_above_10x_threshold_and_"
            "unrestricted_metric_step_above_2x_trust_radius"
        ),
        "step": "single_full_metric_Cauchy_trial",
        "hessian_update": "damped_bfgs_only",
        "bofill": "disabled",
        "gdiis": "enabled",
        "line_search": "forbidden",
        "trial_evaluations": "one",
    }:
        raise RuntimeError("LINK far-from-minimum protocol differs from the frozen contract")
    if payload.get("energy_only_gradient") != {
        "coordinate_space": "SONIC",
        "backend_request": "energy_only_without_gradient_or_hessian",
        "discard_unrequested_derivatives": True,
        "cost_accounting": {
            "primary_metric": "optimization_energy_evaluations_excluding_final_hessian",
            "secondary_metric": "accepted_optimizer_iterations",
            "slightly_increased_iterations_allowed": True,
            "required_conditions": [
                "converged",
                "same_convergence_profile",
                "no_protocol_or_backend_specific_patch",
            ],
        },
        "initial_stencil": "one_sided",
        "initial_class_screening": {
            "enabled_for": "never",
            "reason": "production_protocol_uses_only_exact_symmetry_mask",
            "molecule_specific_rules": False,
        },
        "adaptive_step": "coordinate_family_curvature_and_energy_noise",
        "parallel_displaced_energies": True,
        "stencil_profiles": {
            "geometry_seed_energy_only": {
                "stencil": "one_sided_only",
                "automatic_two_sided_transition": False,
                "final_gradient_verification": "none",
                "convergence_source": "one_sided_gradient",
                "explicit_hessian_request_may_evaluate_both_sides": True,
            },
            "stationary_energy_only": {
                "stencil": "adaptive_two_sided",
                "two_sided_switch": {
                    "criterion": "max_abs_coordinate_gradient",
                    "threshold_hartree_per_protocol_coordinate": 7.0e-4,
                    "irreversible": True,
                    "applies_from_next_iteration": True,
                },
                "final_gradient_verification": "two_sided",
                "mismatch_policy": "continue_two_sided",
            },
        },
    }:
        raise RuntimeError(
            "LINK energy-only gradient contract differs from the confirmed frozen contract"
        )
    if payload.get("trial_contract") != {
        "ordinary_step_QM_trials_per_optimizer_iteration": 1,
        "analytic_trial_properties": ["energy", "gradient"],
        "accepted_analytic_trial": (
            "reuse_energy_and_gradient_without_duplicate_backend_call"
        ),
        "energy_converged_minimum_trial": (
            "accept_even_when_model_ratio_is_poor_and_contract_radius_if_ratio_"
            "is_below_shrink_threshold"
        ),
        "energy_only_trial_properties": ["energy"],
        "accepted_energy_only_trial": (
            "finite_difference_gradient_only_at_accepted_geometry"
        ),
        "transition_state_acceptance": (
            "GDV_DXRFO_accept_every_finite_valid_evaluated_geometry_and_keep_raw_actual_"
            "over_predicted_ratio_diagnostic_with_default_TS_UpTrus_false"
        ),
        "rejected_trial": (
            "minimum_model_failure_or_backend_geometry_failure_only_no_transition_state_"
            "model_ratio_rejection"
        ),
        "in_cycle_geometric_backtracking": False,
        "far_from_minimum_trial": (
            "one_full_metric_Cauchy_QM_trial_without_line_search"
        ),
    }:
        raise RuntimeError("LINK trial contract differs from the confirmed frozen contract")
    if payload.get("backend_contract") != {
        "optimizer_algorithm": "identical_for_all_QM_backends",
        "backend_responsibility": "declared_energy_gradient_and_Hessian_evaluation_only",
        "backend_identity_in_optimizer_decisions": False,
        "QM_program_source_modification": False,
    }:
        raise RuntimeError("LINK backend contract differs from the confirmed frozen contract")
    if payload.get("runtime_contract") != {
        "manifest_required_for_every_LINK_run": True,
        "record_exact_program_method_basis_and_gradient_mode": True,
        "record_protocol_sha256": True,
        "keymaker_uses_same_LINK_entry_point": True,
        "backend_specific_optimizer_patches": False,
        "qm_code_modified": False,
        "final_hessian_request": "explicit_opt_in_only",
    }:
        raise RuntimeError("LINK runtime contract differs from the confirmed frozen contract")
    if payload.get("optional_accelerators") != {
        "selective_fd_refresh": "explicit_runtime_setting_recorded_in_run_manifest",
        "sparse_hessian_updates": (
            "forbidden_because_projection_breaks_the_secant_equation"
        ),
        "backend_identity_may_select_accelerator": False,
    }:
        raise RuntimeError(
            "LINK optional-accelerator contract differs from the confirmed frozen contract"
        )
    if payload.get("scientific_basis") != [
        "Banerjee_Adams_Simons_Shepard_JPhysChem_1985_DOI_10.1021/j100247a015",
        "Lindh_Bernhardsson_Karlstroem_Malmqvist_ChemPhysLett_1995_"
        "DOI_10.1016/0009-2614(95)00646-L",
        "Swart_Bickelhaupt_IntJQuantumChem_2006_DOI_10.1002/qua.21049",
        "Fischer_Almloef_JPhysChem_1992_DOI_10.1021/j100203a036",
        "Csaszar_Pulay_JMolStruct_1984_DOI_10.1016/S0022-2860(84)87198-7",
        "Bofill_JComputChem_1994_DOI_10.1002/jcc.540150102",
        "Farkas_Schlegel_PCCP_2002_DOI_10.1039/B108658H",
        "Wang_Song_JChemPhys_2016_DOI_10.1063/1.4952956",
    ]:
        raise RuntimeError("LINK scientific basis differs from the confirmed frozen contract")
    if payload.get("scope") != "LINK minimum and transition-state geometry optimization":
        raise RuntimeError("LINK optimizer protocol scope differs from the frozen contract")
    if payload.get("change_policy") != "new_manifest_version_and_explicit_user_confirmation":
        raise RuntimeError("LINK optimizer protocol change policy differs from the frozen contract")


def runtime_method_payload(
    protocol: LinkOptimizerProtocol,
    *,
    backend: object | None,
    engine_command: str,
    coordinate_kind: str,
    stationary_point: str,
    prefer_analytic_gradient: bool,
    optimizer_settings: object,
    symmetry_verification: str,
) -> dict[str, Any]:
    """Build the exact per-run method record governed by the frozen protocol."""

    program = "external"
    method = "unspecified"
    basis = "unspecified"
    gradient_mode = "auto_from_returned_properties"
    charge: int | None = None
    multiplicity: int | None = None
    if backend is not None:
        program = str(getattr(backend, "name", "unspecified"))
        method = str(getattr(backend, "method", "unspecified") or "unspecified")
        basis = str(getattr(backend, "basis", "unspecified") or "unspecified")
        gradient_mode = str(
            getattr(backend, "gradient_mode", "analytic") or "analytic"
        )
        charge = int(getattr(backend, "charge", 0))
        multiplicity = int(getattr(backend, "multiplicity", 1))
    return {
        "schema": LINK_RUNTIME_METHOD_SCHEMA,
        "protocol": {
            "id": protocol.protocol_id,
            "version": protocol.manifest_version,
            "sha256": protocol.sha256,
            "source": protocol.source,
            "status": "validated",
        },
        "symmetry_contract": dict(protocol.payload["symmetry_contract"]),
        "algorithm_governance": dict(protocol.payload["algorithm_governance"]),
        "symmetry_verification": str(symmetry_verification),
        "optimizer_settings": {
            "convergence_profile": str(
                getattr(optimizer_settings, "convergence_profile")
            ),
            "initial_hessian_model": str(
                getattr(optimizer_settings, "initial_hessian_model")
            ),
            "enable_gdiis": bool(getattr(optimizer_settings, "enable_gdiis")),
            "gdiis_start": int(getattr(optimizer_settings, "gdiis_start")),
            "gdiis_history": int(getattr(optimizer_settings, "gdiis_history")),
            "gdiis_max_condition": float(
                getattr(optimizer_settings, "gdiis_max_condition")
            ),
            "gdiis_max_coefficient": float(
                getattr(optimizer_settings, "gdiis_max_coefficient")
            ),
            "compute_final_hessian": bool(
                getattr(optimizer_settings, "compute_final_hessian")
            ),
            "transition_index_probe_rms_angstrom": float(
                getattr(optimizer_settings, "transition_index_probe_rms_angstrom")
            ),
            "fd_stencil_policy": str(
                getattr(optimizer_settings, "fd_stencil_policy")
            ),
            "symmetrize_analytic_gradients": bool(
                getattr(optimizer_settings, "symmetrize_analytic_gradients")
            ),
            "adaptive_fd_mode": bool(getattr(optimizer_settings, "adaptive_fd_mode")),
            "fd_totally_symmetric_only": bool(
                getattr(optimizer_settings, "fd_totally_symmetric_only")
            ),
            "fd_initial_class_threshold_fraction": float(
                getattr(optimizer_settings, "fd_initial_class_threshold_fraction")
            ),
            "fd_class_threshold_release_factor": float(
                getattr(optimizer_settings, "fd_class_threshold_release_factor")
            ),
            "fd_class_screen_audit_interval": int(
                getattr(optimizer_settings, "fd_class_screen_audit_interval")
            ),
            "selective_fd_refresh": bool(
                getattr(optimizer_settings, "selective_fd_refresh")
            ),
            "sparse_hessian_updates": bool(
                getattr(optimizer_settings, "sparse_hessian_updates")
            ),
            "coordinate_schedule": str(getattr(optimizer_settings, "coordinate_schedule")),
            "hessian_update": str(getattr(optimizer_settings, "hessian_update")),
        },
        "invocation": {
            "program": program,
            "method": method,
            "basis": basis,
            "gradient_mode": gradient_mode,
            "prefer_analytic_gradient": bool(prefer_analytic_gradient),
            "coordinate_kind": str(coordinate_kind),
            "stationary_point": str(stationary_point),
            "charge": charge,
            "multiplicity": multiplicity,
            "external_engine_command_declared": bool(str(engine_command).strip()),
        },
        "backend_specific_optimizer_patch": False,
    }


def validate_runtime_optimizer_settings(settings: object) -> None:
    """Reject invocation-level changes to confirmed protocol invariants."""

    expected = {
        "far_from_minimum_force_factor": 10.0,
        "far_from_minimum_displacement_factor": 2.0,
        "fd_two_sided_switch_force": 7.0e-4,
        "initial_hessian_model": "lindh_swart_special",
        "hessian_update": "auto",
        "coordinate_schedule": "joint",
        "adaptive_fd_mode": True,
        "fd_initial_class_threshold_fraction": 0.10,
        "fd_class_threshold_release_factor": 10.0,
        "fd_class_screen_audit_interval": 3,
        "enable_gdiis": True,
        "gdiis_start": 3,
        "gdiis_history": 6,
        "gdiis_max_condition": 5.0e3,
        "gdiis_max_coefficient": 2.0,
        "symmetrize_analytic_gradients": True,
        "line_search_reductions": 0,
        "sparse_hessian_updates": False,
    }
    stencil_policy = str(getattr(settings, "fd_stencil_policy", ""))
    stencil_expected = {
        "adaptive_two_sided": {
            "two_sided": True,
            "one_sided_until_convergence": True,
            "final_gradient_verification": True,
        },
        "one_sided_only": {
            "two_sided": False,
            "one_sided_until_convergence": True,
            "final_gradient_verification": False,
        },
    }
    if stencil_policy not in stencil_expected:
        raise ValueError(
            "LINK frozen optimizer protocol requires a declared finite-difference "
            "stencil profile"
        )
    expected.update(stencil_expected[stencil_policy])
    mismatches = {
        name: {"required": required, "received": getattr(settings, name, None)}
        for name, required in expected.items()
        if getattr(settings, name, None) != required
    }
    if mismatches:
        details = ", ".join(
            f"{name}={values['received']!r} (required {values['required']!r})"
            for name, values in mismatches.items()
        )
        raise ValueError(
            "LINK frozen optimizer protocol cannot be overridden at runtime: " + details
        )


def validate_method_manifest_collection(directory: Path | str) -> tuple[dict[str, Any], ...]:
    """Validate every checked-in backend method manifest against LINK's protocol.

    The repository manifests describe the backend boundary and gradient mode;
    the actual QM method and basis are required at invocation time unless a
    concrete, qualified campaign is explicitly recorded in that manifest.
    This prevents silent backend-specific optimizer variants.
    """

    root = Path(directory)
    paths = tuple(sorted(root.glob("*.json")))
    if not paths:
        raise RuntimeError(f"no LINK method manifests found in {root}")
    required = {
        "xtb-gfn2-analytic",
        "xtb-gfn2-energy-only",
        "orca-runtime-analytic",
        "orca-runtime-energy-only",
        "pyscf-runtime-analytic",
        "pyscf-runtime-energy-only",
        "et-runtime-analytic",
        "et-runtime-energy-only",
    }
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable LINK method manifest: {path}") from exc
        if payload.get("schema") != LINK_METHOD_MANIFEST_SCHEMA:
            raise RuntimeError(f"unsupported LINK method manifest schema: {path}")
        if payload.get("status") != "frozen":
            raise RuntimeError(f"LINK method manifest is not frozen: {path}")
        if payload.get("optimizer_protocol_id") != LINK_PROTOCOL_ID:
            raise RuntimeError(f"method manifest is not bound to LINK protocol: {path}")
        if payload.get("optimizer_protocol_version") != LINK_PROTOCOL_VERSION:
            raise RuntimeError(f"method manifest has the wrong LINK protocol version: {path}")
        if payload.get("reuse_existing_link_algorithms") is not True:
            raise RuntimeError(f"method manifest permits algorithm reimplementation: {path}")
        if payload.get("molecule_specific_optimizer_patches") is not False:
            raise RuntimeError(f"molecule-specific optimizer patch enabled: {path}")
        if payload.get("symmetry_protocol") != (
            "frozen_global_oracle_initial_and_per_iteration_gradient_symmetrization"
        ):
            raise RuntimeError(f"invalid symmetry protocol in method manifest: {path}")
        if payload.get("backend_specific_optimizer_patches") is not False:
            raise RuntimeError(f"backend-specific optimizer patch enabled: {path}")
        if payload.get("qm_code_untouched") is not True:
            raise RuntimeError(f"QM-code modification is not forbidden in: {path}")
        if payload.get("gradient_mode") not in {
            "analytic",
            "energy_only_coordinate_finite_difference",
        }:
            raise RuntimeError(f"invalid gradient mode in method manifest: {path}")
        if payload.get("method_declaration") not in {
            "required_at_invocation",
            "qualified_campaign",
        }:
            raise RuntimeError(f"method declaration policy missing in: {path}")
        if payload.get("method_id") == "xtb-gfn2-energy-only":
            required_xtb_profile = {
                "convergence_profile": "geometry_seed",
                "convergence_reference": "xTB_6.7.1_opt_normal",
                "finite_difference_stencil": "one_sided_only",
                "automatic_two_sided_transition": False,
                "final_central_gradient_verification": False,
                "displaced_point_requested_properties": ["energy"],
                "xtb_gradient_flag_for_displaced_points": False,
                "xtb_hessian_flag_for_displaced_points": False,
            }
            mismatches = {
                key: {"required": value, "received": payload.get(key)}
                for key, value in required_xtb_profile.items()
                if payload.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"xTB energy-only profile differs from the frozen contract: {mismatches}"
                )
        payloads.append(dict(payload))
    present = {str(payload.get("method_id")) for payload in payloads}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError("missing required LINK method manifests: " + ", ".join(missing))
    if len(present) != len(payloads):
        raise RuntimeError("duplicate LINK method_id in method manifest collection")
    return tuple(payloads)


def write_runtime_method_manifest(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "LINK_PROTOCOL_ID",
    "LINK_PROTOCOL_VERSION",
    "LINK_PROTOCOL_SCHEMA",
    "LINK_RUNTIME_METHOD_SCHEMA",
    "LINK_METHOD_MANIFEST_SCHEMA",
    "LINK_METHOD_MANIFEST_DIR",
    "LinkOptimizerProtocol",
    "load_link_optimizer_protocol",
    "runtime_method_payload",
    "validate_runtime_optimizer_settings",
    "validate_method_manifest_collection",
    "write_runtime_method_manifest",
]
