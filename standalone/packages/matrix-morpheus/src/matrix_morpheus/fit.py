from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_chem.geometry_io import write_xyz
from matrix_core import ScientificValidationError, build_run_manifest
from matrix_morpheus.numerics import objective
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive

from .contracts import (
    SemiexperimentalFitRequest,
)
from .constraints import (
    _fixed_primitive_targets,
    _fixed_primitives_from_patterns,
    _gic_expression_constraints_from_patterns,
    _gic_expression_definitions_from_patterns,
    _gic_expression_constraint_targets,
    _gic_expression_uses_gic_names,
    _gic_values,
    _linear_primitive_constraints_from_patterns,
    _project_fixed_primitives,
)
from .geometry_input import read_geometry_input
from .kraitchman import (
    KraitchmanComparison,
    KraitchmanSeedResult,
    kraitchman_comparison,
    kraitchman_seed_geometry,
)
from .statistics import (
    SemiexperimentalWeightDiagnostic,
    weight_diagnostic_rows as _weight_diagnostic_rows,
    weight_diagnostics_csv as _weight_diagnostics_csv,
)
from .cartesian_coordinates import cartesian_symmetry_coordinate_model
from .models import (
    GICProjectorState,
    MeasurementModel,
    SemiexperimentalDiagnosticWarning,
    SemiexperimentalFitDiagnostics,
    SemiexperimentalFitResult,
    SemiexperimentalGeometryParameter,
    SemiexperimentalIterationTrace,
    SemiexperimentalLeaveOneOutRow,
    SemiexperimentalParameter,
    SemiexperimentalResidual,
    SemiexperimentalRotationalConstantComparison,
)
from .solver import (
    _accepted_trust_update,
    _adaptive_lm_step,
    _objective_has_stabilized,
    _rejected_trust_update,
    _trust_region_is_stalled,
)


from .coordinate_model import (
    _active_coordinate_jacobian,
    _active_mask,
    _atomic_number,
    _auto_pruned_active_mask,
    _cartesian_from_reduced_coordinates,
    _constraint_primitive_pool,
    _dynamic_parameter_scales,
    _fragment_atom_sets_from_primitives,
    _fragment_internal_fixed_primitives,
    _fragment_internal_gic_patterns,
    _gic_fixed_patterns,
    _gic_model,
    _gic_model_signature,
    _gicforge_a1_mask,
    _gicforge_sycart_coordinates,
    _hydrogen_fixed_primitives,
    _make_gicforge_backend,
    _mark_auto_pruned_classes,
    _merge_primitives,
    _oracle_cartesian_symmetry_state,
    _parameter_class_transform,
    _primitive_constrained_cartesian_transform,
    _primitive_constrained_transform,
    _reduced_parameter_scales,
    _resolve_max_iterations,
    _robust_sqrt_weights_for_model,
    _symmetry_expanded_fixed_primitives,
    _validate_gic_model_signature,
    _validate_observation_budget,
    _weak_parameter_patterns,
)


from .measurement_model import (
    _build_measurement_model,
    _build_measurement_model_cartesian_basis,
    _correlation,
    _covariance,
    _diagnostics,
    _gic_projector_state,
    _iteration_trace_row,
    _jacobian_constants_wrt_cartesian_basis,
    _jacobian_constants_wrt_gics,
    _least_squares_hessian,
    _line_search_update,
    _line_search_update_cartesian_basis,
    _measurement_vector,
    _parameters,
    _residual_rows,
    _rotational_constant_rows,
    _secant_projector_update,
    _should_refresh_gic_model,
    _sonic_parameters_from_cartesian_covariance,
    _stationary_point_type,
    _topology_lock,
    _validate_locked_topology,
    _validate_observations,
)


from .fit_outputs import (
    _checkpoint_path,
    _combined_fixed_parameters,
    _constraint_summary_lines,
    _constraints_csv,
    _diagnostics_csv,
    _effective_parameter_names,
    _eigenvalues_csv,
    _geometry_parameters,
    _high_correlations_csv,
    _influence_csv,
    _isotopic_mapping_warning_rows,
    _leave_one_out_csv,
    _matrix_csv,
    _read_semiexp_checkpoint,
    _request_with_auto_resolved_isotopologues,
    _rotational_residual_manifest_stats,
    _semiexp_warning_rows,
    _svd_diagnostics_csv,
    _svd_summary_lines,
    _uncertainty_diagnostics_csv,
    _warnings_csv,
    _write_semiexp_checkpoint,
    geometry_parameters_csv,
    iteration_trace_csv_rows,
    kraitchman_csv_rows,
    parameters_csv,
    residuals_csv,
    rotational_constants_csv,
    semiexperimental_text_report,
)

SEMIEXP_CHECKPOINT_SCHEMA = "oracle.semiexp.checkpoint.v1"
LEGACY_SEMIEXP_CHECKPOINT_SCHEMA = "merlino.semiexp.checkpoint.v1"
SUPPORTED_SEMIEXP_CHECKPOINT_SCHEMAS = (
    SEMIEXP_CHECKPOINT_SCHEMA,
    LEGACY_SEMIEXP_CHECKPOINT_SCHEMA,
)
ROTATIONAL_COMPONENTS = ("A", "B", "C")
MOMENT_COMPONENTS = ("Ia", "Ib", "Ic")
ROTATIONAL_TO_MOMENT_COMPONENT = {"A": "Ia", "B": "Ib", "C": "Ic"}
ROTCONST_TO_MOMENT = (
    get_physical_constants()[Phy.PLANCK]
    / (8.0 * np.pi**2 * get_physical_constants()[Phy.TO_KG] * (1.0e-10) ** 2)
    * 1.0e-6
)
DIAGNOSTIC_CONDITION_WARNING = 1.0e8
DIAGNOSTIC_RELATIVE_SINGULAR_WARNING = 1.0e-8
DIAGNOSTIC_ROBUST_WEIGHT_WARNING = 0.50
DIAGNOSTIC_ROBUST_WEIGHT_SEVERE = 0.25
DIAGNOSTIC_BOND_SIGMA_WARNING_ANGSTROM = 5.0e-3
DIAGNOSTIC_ANGLE_SIGMA_WARNING_DEGREE = 0.50
DIAGNOSTIC_ISOTOPE_SHIFT_WARNING_MHZ = 5.0
DIAGNOSTIC_ISOTOPE_SHIFT_IMPROVEMENT_RATIO = 0.50
DIAGNOSTIC_REDUCED_CHI_SQUARE_WARNING = 9.0
DIAGNOSTIC_REJECTED_STEP_FRACTION_WARNING = 0.50
DIAGNOSTIC_TRUST_RADIUS_WARNING = 1.0e-7
DIAGNOSTIC_LINE_SEARCH_SCALE_WARNING = 1.0e-3
DIAGNOSTIC_PARAMETER_SCALE_RATIO_WARNING = 1.0e6
DIAGNOSTIC_WEIGHTED_RESIDUAL_WARNING = 5.0
DIAGNOSTIC_LEVERAGE_WARNING = 0.95
DIAGNOSTIC_CORRELATION_WARNING = 0.98


def fit_semiexperimental_geometry(
    request: SemiexperimentalFitRequest,
    *,
    max_iter: int | None = None,
    step: float = 1.0e-4,
    damping: float = 1.0e-8,
    max_step: float = 0.25,
    prune_condition: float = 0.0,
    tolerance_MHz: float = 1.0e-6,
    gradient_tolerance: float = 1.0e-8,
    checkpoint: Path | None = None,
    restart: Path | None = None,
    outdir: Path | None = None,
) -> SemiexperimentalFitResult:
    """Fit equilibrium geometry to semiexperimental rotational constants.

    The default working coordinates are MATRIX/SMITH non-redundant GICs. As an
    alternative, `coordinate_model="cartesian_symmetry"` uses a Hessian-free
    translation/rotation-free symmetry-adapted Cartesian displacement basis.
    """
    request.validate()
    if request.coordinate_model == "cartesian_symmetry":
        return _fit_semiexperimental_geometry_cartesian_symmetry(
            request,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            tolerance_MHz=tolerance_MHz,
            gradient_tolerance=gradient_tolerance,
            checkpoint=checkpoint,
            restart=restart,
            outdir=outdir,
        )
    geometry_input = read_geometry_input(Path(request.initial_geometry))
    atoms = list(geometry_input.atoms)
    coords = np.asarray(geometry_input.coordinates_angstrom, dtype=float)
    request, preflight_warnings = _request_with_auto_resolved_isotopologues(request, atoms, coords)
    if restart is not None:
        coords = _read_semiexp_checkpoint(Path(restart), expected_atoms=len(atoms))
    coords0 = coords.copy()
    fixed_parameters = _combined_fixed_parameters(
        request.fixed_parameters, geometry_input.fixed_parameters
    )
    fixed_gic_patterns = _gic_fixed_patterns(fixed_parameters)
    fixed_primitives = _fixed_primitives_from_patterns(fixed_parameters)
    linear_constraints = _linear_primitive_constraints_from_patterns(fixed_parameters)
    expression_constraints = _gic_expression_constraints_from_patterns(fixed_parameters)
    expression_definitions = _gic_expression_definitions_from_patterns(fixed_parameters)
    if fixed_primitives or linear_constraints:
        coords = _project_fixed_primitives(
            coords,
            fixed_primitives,
            _fixed_primitive_targets(fixed_primitives, coords),
            linear_constraints=linear_constraints,
        )
    z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
    _validate_observations(request.observations, len(atoms))
    gicforge_backend = _make_gicforge_backend(tuple(atoms), outdir)

    prims, u_matrix, labels = _gic_model(coords, z_numbers, request, gicforge_backend)
    fragment_atom_sets = _fragment_atom_sets_from_primitives(prims)
    if fragment_atom_sets:
        fixed_gic_patterns = _combined_fixed_parameters(
            fixed_gic_patterns,
            _fragment_internal_gic_patterns(prims, u_matrix, labels, fragment_atom_sets),
        )
    fixed_primitives = _merge_primitives(
        fixed_primitives,
        _hydrogen_fixed_primitives(atoms, prims, fixed_parameters, coords=coords),
    )
    if fragment_atom_sets:
        fixed_primitives = _merge_primitives(
            fixed_primitives,
            _fragment_internal_fixed_primitives(atoms, coords, prims, fragment_atom_sets),
        )
    fixed_primitives = _symmetry_expanded_fixed_primitives(atoms, coords, prims, fixed_primitives)
    fixed_primitive_targets = _fixed_primitive_targets(fixed_primitives, coords)
    expression_targets = _gic_expression_constraint_targets(
        expression_constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=expression_definitions,
    )
    if fixed_primitives or linear_constraints or expression_constraints:
        coords = _project_fixed_primitives(
            coords,
            fixed_primitives,
            fixed_primitive_targets,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
    topology_lock = _topology_lock(atoms, coords)
    reference_gic_signature = _gic_model_signature(labels)
    measurement_model = _build_measurement_model(request, atoms, coords, prims, u_matrix, labels)
    active_mask = _active_mask(
        labels, fixed_gic_patterns, request.parameter_classes
    ) & _gicforge_a1_mask(labels)
    initial_transform, _initial_names, _initial_classes = _parameter_class_transform(
        labels, active_mask, request.parameter_classes
    )
    initial_transform, _initial_names = _primitive_constrained_transform(
        coords,
        prims,
        u_matrix,
        active_mask,
        initial_transform,
        _initial_names,
        fixed_primitives,
        linear_constraints=linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        expression_definitions=expression_definitions,
        labels=labels,
    )
    auto_pruned_patterns: tuple[str, ...] = ()
    if prune_condition > 0.0 and initial_transform.shape[1] > 1:
        try:
            initial_jac_gic = _jacobian_constants_wrt_gics(
                atoms,
                coords,
                request,
                prims,
                u_matrix,
                active_mask,
                labels,
                measurement_model,
                step=step,
            )
            initial_weighted_jac = (initial_jac_gic @ initial_transform) * np.sqrt(
                measurement_model.weights
            )[:, None]
            auto_pruned_patterns = _weak_parameter_patterns(
                _initial_names, initial_weighted_jac, prune_condition
            )
            if auto_pruned_patterns:
                active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
                initial_transform, _initial_names, _initial_classes = _parameter_class_transform(
                    labels, active_mask, request.parameter_classes
                )
                initial_transform, _initial_names = _primitive_constrained_transform(
                    coords,
                    prims,
                    u_matrix,
                    active_mask,
                    initial_transform,
                    _initial_names,
                    fixed_primitives,
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    expression_definitions=expression_definitions,
                    labels=labels,
                )
        except Exception:
            # Pruning is an observability refinement; unsupported mock/legacy primitives must not block the fit.
            auto_pruned_patterns = ()
    n_optimized_parameters = initial_transform.shape[1]
    _validate_observation_budget(
        measurement_model,
        n_optimized_parameters,
        coordinate_model=request.coordinate_model,
    )
    loop_max_iter = (
        _resolve_max_iterations(max_iter, n_optimized_parameters) if n_optimized_parameters else 0
    )

    current_damping = max(float(damping), 0.0)
    trust_radius = float(max_step) if max_step > 0.0 else 0.0
    accepted_steps = 0
    rejected_steps = 0
    stalled_rejections = 0
    model_age = 0
    coordinate_model_reuse_steps = 0
    q_initial = _gic_values(prims, u_matrix, coords)
    projector_state = _gic_projector_state(prims, u_matrix, coords, q_initial)
    b_projector_analytic_refreshes = 1
    b_projector_secant_updates = 0
    b_projector_secant_rejections = 0
    last_b_projector_secant_error = 0.0
    parameter_scale_min = 1.0
    parameter_scale_max = 1.0
    last_trust_ratio = 0.0
    last_line_search_scale = 0.0
    convergence_reason = "max_iter" if loop_max_iter else "no_active_totally_symmetric_parameters"
    robust_scale_used = 0.0
    robust_downweighted_observations = 0
    robust_downweighted_isotopologues = 0
    robust_sqrt = np.ones_like(measurement_model.observed, dtype=float)
    checkpoint_file = _checkpoint_path(outdir, checkpoint)
    previous_objective = None
    iteration_traces: list[SemiexperimentalIterationTrace] = []
    iteration = 0
    for iteration in range(1, loop_max_iter + 1):
        active_mask = _active_mask(labels, fixed_gic_patterns, request.parameter_classes)
        active_mask &= _gicforge_a1_mask(labels)
        active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
        q = _gic_values(prims, u_matrix, coords)
        calc = _measurement_vector(atoms, coords, request, q, labels, measurement_model)
        obs = measurement_model.observed
        weights = measurement_model.weights
        sqrt_weights = np.sqrt(weights)
        residual = obs - calc
        base_weighted_residual = residual * sqrt_weights
        (
            robust_sqrt,
            robust_scale_used,
            robust_downweighted_observations,
            robust_downweighted_isotopologues,
        ) = _robust_sqrt_weights_for_model(
            base_weighted_residual,
            request.robust_loss,
            request.robust_scale,
            measurement_model,
        )
        effective_sqrt_weights = sqrt_weights * robust_sqrt
        weighted_residual = residual * effective_sqrt_weights
        current_objective = objective(weighted_residual)
        jac_gic = _jacobian_constants_wrt_gics(
            atoms,
            coords,
            request,
            prims,
            u_matrix,
            active_mask,
            labels,
            measurement_model,
            step=step,
            cartesian_from_q=projector_state.cartesian_from_q,
        )
        transform, _reduced_names, _class_by_gic = _parameter_class_transform(
            labels, active_mask, request.parameter_classes
        )
        transform, _reduced_names = _primitive_constrained_transform(
            coords,
            prims,
            u_matrix,
            active_mask,
            transform,
            _reduced_names,
            fixed_primitives,
            cartesian_from_q=projector_state.cartesian_from_q,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            expression_definitions=expression_definitions,
            labels=labels,
        )
        jac = jac_gic @ transform
        base_scales = _reduced_parameter_scales(labels, active_mask, transform)
        jac_weighted = jac * effective_sqrt_weights[:, None]
        reduced_scales = _dynamic_parameter_scales(jac_weighted, base_scales)
        if reduced_scales.size:
            parameter_scale_min = min(parameter_scale_min, float(np.min(reduced_scales)))
            parameter_scale_max = max(parameter_scale_max, float(np.max(reduced_scales)))
        if np.sqrt(np.mean(residual * residual)) < tolerance_MHz:
            convergence_reason = "rms_tolerance"
            break
        jac_weighted_scaled = (
            jac_weighted * reduced_scales[None, :] if reduced_scales.size else jac_weighted
        )
        gradient = jac_weighted_scaled.T @ weighted_residual
        gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if float(np.linalg.norm(gradient, ord=np.inf)) < gradient_tolerance:
            convergence_reason = "gradient_tolerance"
            break
        trust_step = _adaptive_lm_step(
            jac_weighted_scaled, weighted_residual, current_damping, trust_radius
        )
        dq_scaled = trust_step.step
        current_damping = trust_step.shift
        dq_reduced = reduced_scales * dq_scaled if reduced_scales.size else dq_scaled
        dq_active = transform @ dq_reduced
        dq = np.zeros_like(q)
        dq[np.where(active_mask)[0]] = dq_active
        line_search = _line_search_update(
            atoms,
            coords,
            request,
            labels,
            measurement_model,
            prims,
            u_matrix,
            dq,
            current_objective=current_objective,
            base_q=q,
            cartesian_from_q=projector_state.cartesian_from_q,
            weighted_residual=weighted_residual,
            jac_weighted=jac_weighted_scaled,
            reduced_step=dq_scaled,
            robust_sqrt_weights=robust_sqrt,
            fixed_primitives=fixed_primitives,
            fixed_primitive_targets=fixed_primitive_targets,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            expression_definitions=expression_definitions,
            sonic_definition=gicforge_backend.link_definition,
        )
        last_trust_ratio = line_search.ratio
        last_line_search_scale = line_search.scale
        if line_search.accepted:
            previous_coords = coords
            previous_q = q
            next_model_age = model_age + 1
            secant_update = _secant_projector_update(
                projector_state.cartesian_from_q,
                previous_coords,
                previous_q,
                line_search.coords,
                line_search.q_values,
            )
            last_b_projector_secant_error = secant_update.relative_error
            try:
                _validate_locked_topology(
                    atoms, line_search.coords, topology_lock, context="GIC semiexperimental fit"
                )
                validation_model = _gic_model(
                    line_search.coords, z_numbers, request, gicforge_backend
                )
                _validate_gic_model_signature(validation_model[2], reference_gic_signature)
            except Exception:
                rejected_steps += 1
                stalled_rejections += 1
                current_damping, trust_radius = _rejected_trust_update(
                    current_damping, trust_radius, max_step
                )
                iteration_traces.append(
                    _iteration_trace_row(
                        iteration,
                        "topology_rejected",
                        current_objective,
                        line_search,
                        current_damping,
                        trust_radius,
                        line_search.scale * float(np.linalg.norm(dq_scaled)),
                        gradient_inf_norm,
                        jac_weighted_scaled,
                        line_search.coords,
                        fixed_primitives,
                        fixed_primitive_targets,
                        linear_constraints=linear_constraints,
                        expression_constraints=expression_constraints,
                        expression_targets=expression_targets,
                        prims=prims,
                        u_matrix=u_matrix,
                        labels=labels,
                        expression_definitions=expression_definitions,
                        robust_scale=robust_scale_used,
                        robust_downweighted_observations=robust_downweighted_observations,
                        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                        coordinate_model_age=model_age,
                        b_projector_secant_error=secant_update.relative_error,
                        linear_solver=trust_step.solver,
                    )
                )
                continue
            refresh_required = _should_refresh_gic_model(
                line_search,
                next_model_age,
                secant_relative_error=secant_update.relative_error,
                tolerance_MHz=tolerance_MHz,
                n_observations=len(weighted_residual),
            )
            refreshed_model = None
            if refresh_required:
                refreshed_model = validation_model
            coords = line_search.coords
            accepted_steps += 1
            stalled_rejections = 0
            current_damping, trust_radius = _accepted_trust_update(
                current_damping,
                trust_radius,
                line_search.ratio,
                line_search.scale,
                float(line_search.scale * np.linalg.norm(dq_scaled)),
                max_step,
            )
            iteration_traces.append(
                _iteration_trace_row(
                    iteration,
                    "accepted",
                    current_objective,
                    line_search,
                    current_damping,
                    trust_radius,
                    line_search.scale * float(np.linalg.norm(dq_scaled)),
                    gradient_inf_norm,
                    jac_weighted_scaled,
                    line_search.coords,
                    fixed_primitives,
                    fixed_primitive_targets,
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=prims,
                    u_matrix=u_matrix,
                    labels=labels,
                    expression_definitions=expression_definitions,
                    robust_scale=robust_scale_used,
                    robust_downweighted_observations=robust_downweighted_observations,
                    robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                    coordinate_model_age=next_model_age,
                    b_projector_secant_error=secant_update.relative_error,
                    linear_solver=trust_step.solver,
                )
            )
            if (
                previous_objective is not None
                and abs(previous_objective - line_search.objective) < tolerance_MHz * tolerance_MHz
            ):
                convergence_reason = "objective_tolerance"
                break
            previous_objective = line_search.objective
            if refresh_required and refreshed_model is not None:
                prims, u_matrix, labels = refreshed_model
                refreshed_q = _gic_values(prims, u_matrix, coords)
                projector_state = _gic_projector_state(prims, u_matrix, coords, refreshed_q)
                b_projector_analytic_refreshes += 1
                model_age = 0
            else:
                if not secant_update.accepted or secant_update.cartesian_from_q is None:
                    projector_state = _gic_projector_state(
                        prims, u_matrix, coords, line_search.q_values
                    )
                    b_projector_analytic_refreshes += 1
                    b_projector_secant_rejections += 1
                else:
                    projector_state = GICProjectorState(
                        coords=coords.copy(),
                        q_values=line_search.q_values.copy(),
                        cartesian_from_q=secant_update.cartesian_from_q,
                    )
                    b_projector_secant_updates += 1
                model_age = next_model_age
                coordinate_model_reuse_steps += 1
            _write_semiexp_checkpoint(
                checkpoint_file,
                atoms,
                coords,
                iteration=iteration,
                damping=current_damping,
                trust_radius=trust_radius,
                labels=labels,
                active_mask=active_mask,
                robust_sqrt_weights=robust_sqrt,
                robust_scale=robust_scale_used,
                robust_downweighted_rows=robust_downweighted_observations,
                robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                coordinate_model=request.coordinate_model,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
            )
        else:
            rejected_steps += 1
            stalled_rejections += 1
            current_damping, trust_radius = _rejected_trust_update(
                current_damping, trust_radius, max_step
            )
            iteration_traces.append(
                _iteration_trace_row(
                    iteration,
                    "rejected",
                    current_objective,
                    line_search,
                    current_damping,
                    trust_radius,
                    line_search.scale * float(np.linalg.norm(dq_scaled)),
                    gradient_inf_norm,
                    jac_weighted_scaled,
                    coords,
                    fixed_primitives,
                    fixed_primitive_targets,
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=prims,
                    u_matrix=u_matrix,
                    labels=labels,
                    expression_definitions=expression_definitions,
                    robust_scale=robust_scale_used,
                    robust_downweighted_observations=robust_downweighted_observations,
                    robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                    coordinate_model_age=model_age,
                    b_projector_secant_error=last_b_projector_secant_error,
                    linear_solver=trust_step.solver,
                )
            )
            if model_age:
                prims, u_matrix, labels = _gic_model(coords, z_numbers, request, gicforge_backend)
                _validate_gic_model_signature(labels, reference_gic_signature)
                refreshed_q = _gic_values(prims, u_matrix, coords)
                projector_state = _gic_projector_state(prims, u_matrix, coords, refreshed_q)
                b_projector_analytic_refreshes += 1
                model_age = 0
                stalled_rejections = 0
            if _trust_region_is_stalled(
                current_damping, trust_radius, stalled_rejections, max_step
            ):
                convergence_reason = (
                    "step_tolerance"
                    if _objective_has_stabilized(
                        previous_objective, current_objective, tolerance_MHz
                    )
                    else "line_search_stalled"
                )
                break
    else:
        iteration = loop_max_iter

    _validate_locked_topology(
        atoms, coords, topology_lock, context="final GIC semiexperimental fit"
    )
    try:
        final_model = _gic_model(coords, z_numbers, request, gicforge_backend)
        _validate_gic_model_signature(final_model[2], reference_gic_signature)
        prims, u_matrix, labels = final_model
    except Exception:
        # Some nearly converged geometries lower only the detected point group at
        # post-processing tolerance.  The last validated GIC model remains the
        # chemically intended coordinate frame for reporting and covariance.
        pass
    active_mask = _active_mask(labels, fixed_gic_patterns, request.parameter_classes)
    active_mask &= _gicforge_a1_mask(labels)
    active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
    q_final = _gic_values(prims, u_matrix, coords)
    bq = u_matrix.T @ b_matrix_analytic(prims, coords)
    # The selected observable components are part of the least-squares problem.
    # In auto mode they must not be reselected at the final geometry, otherwise
    # diagnostics and covariance can refer to a different fit target.
    calc = _measurement_vector(atoms, coords, request, q_final, labels, measurement_model)
    obs = measurement_model.observed
    residual = obs - calc
    jac_gic = _jacobian_constants_wrt_gics(
        atoms, coords, request, prims, u_matrix, active_mask, labels, measurement_model, step=step
    )
    transform, reduced_names, class_by_gic = _parameter_class_transform(
        labels, active_mask, request.parameter_classes
    )
    transform, reduced_names = _primitive_constrained_transform(
        coords,
        prims,
        u_matrix,
        active_mask,
        transform,
        reduced_names,
        fixed_primitives,
        linear_constraints=linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        expression_definitions=expression_definitions,
        labels=labels,
    )
    jac = jac_gic @ transform
    sqrt_weights = np.sqrt(measurement_model.weights)
    base_weighted_residual = residual * sqrt_weights
    (
        robust_sqrt,
        robust_scale_used,
        robust_downweighted_observations,
        robust_downweighted_isotopologues,
    ) = _robust_sqrt_weights_for_model(
        base_weighted_residual,
        request.robust_loss,
        request.robust_scale,
        measurement_model,
    )
    effective_sqrt_weights = sqrt_weights * robust_sqrt
    weighted_jac = jac * effective_sqrt_weights[:, None]
    weighted_residual = residual * effective_sqrt_weights
    hessian = _least_squares_hessian(weighted_jac)
    covariance = _covariance(weighted_jac, weighted_residual)
    correlation = _correlation(covariance)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian) if hessian.size else np.array(())
    stationary_point = _stationary_point_type(hessian_eigenvalues)
    diagnostics = _diagnostics(
        weighted_jac,
        weighted_residual,
        convergence_reason=convergence_reason,
        damping=current_damping,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        max_iterations=loop_max_iter,
        n_optimized_parameters=jac.shape[1],
        observable=measurement_model.observable,
        components=measurement_model.components,
        planar=measurement_model.planar,
        auto_pruned_parameters=auto_pruned_patterns,
        prune_condition_target=prune_condition,
        gicforge_calls=gicforge_backend.counter,
        coordinate_model_reuse_steps=coordinate_model_reuse_steps,
        trust_radius=trust_radius,
        last_trust_ratio=last_trust_ratio,
        last_line_search_scale=last_line_search_scale,
        b_projector_analytic_refreshes=b_projector_analytic_refreshes,
        b_projector_secant_updates=b_projector_secant_updates,
        b_projector_secant_rejections=b_projector_secant_rejections,
        last_b_projector_secant_error=last_b_projector_secant_error,
        parameter_scale_min=parameter_scale_min,
        parameter_scale_max=parameter_scale_max,
        robust_loss=request.robust_loss,
        robust_scale=robust_scale_used,
        robust_downweighted_observations=robust_downweighted_observations,
        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
    )
    class_by_gic = _mark_auto_pruned_classes(labels, class_by_gic, auto_pruned_patterns)
    parameters = _parameters(
        labels,
        q_final,
        active_mask,
        transform=transform,
        covariance=covariance,
        class_by_gic=class_by_gic,
    )
    residual_rows = _residual_rows(measurement_model, calc, obs)
    rotational_constant_rows = _rotational_constant_rows(
        atoms,
        coords,
        request.observations,
        request.excluded_rotational_constants,
    )
    geometry_parameters = _geometry_parameters(
        atoms,
        coords,
        initial_coords=coords0,
        fit_prims=prims,
        fit_u_matrix=u_matrix,
        active_mask=active_mask,
        transform=transform,
        covariance=covariance,
        topology_lock=topology_lock,
    )
    kraitchman_rows = kraitchman_comparison(atoms, coords, request.observations)
    kraitchman_seed = kraitchman_seed_geometry(atoms, coords, request.observations, kraitchman_rows)
    rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    weight_diagnostic_rows = _weight_diagnostic_rows(
        measurement_model,
        calc,
        residual,
        weighted_jac,
        weighted_residual,
        robust_sqrt,
    )
    leave_one_out_rows = (
        _leave_one_out_refits(
            request,
            atoms,
            coords,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            tolerance_MHz=tolerance_MHz,
            gradient_tolerance=gradient_tolerance,
        )
        if request.leave_one_out
        else ()
    )
    _write_semiexp_checkpoint(
        checkpoint_file,
        atoms,
        coords,
        iteration=iteration,
        damping=current_damping,
        trust_radius=trust_radius,
        labels=labels,
        active_mask=active_mask,
        robust_sqrt_weights=robust_sqrt,
        robust_scale=robust_scale_used,
        robust_downweighted_rows=robust_downweighted_observations,
        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
        coordinate_model=request.coordinate_model,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )
    manifest = None
    if outdir is not None:
        manifest = write_semiexperimental_outputs(
            Path(outdir),
            request,
            atoms,
            coords,
            parameters,
            residual_rows,
            kraitchman_rows,
            rotational_constants=rotational_constant_rows,
            geometry_parameters=geometry_parameters,
            kraitchman_seed=kraitchman_seed,
            input_fixed_parameters=geometry_input.fixed_parameters,
            fixed_primitives=fixed_primitives,
            leave_one_out=leave_one_out_rows,
            checkpoint_path=checkpoint_file,
            effective_parameter_names=reduced_names,
            covariance=covariance,
            correlation=correlation,
            hessian=hessian,
            hessian_eigenvalues=hessian_eigenvalues,
            stationary_point=stationary_point,
            diagnostics=diagnostics,
            measurement_model=measurement_model,
            weighted_jacobian=weighted_jac,
            weighted_residual=weighted_residual,
            robust_sqrt_weights=robust_sqrt,
            weight_diagnostics=weight_diagnostic_rows,
            iteration_trace=tuple(iteration_traces),
            preflight_warnings=preflight_warnings,
        )
    return SemiexperimentalFitResult(
        atoms=tuple(atoms),
        initial_coordinates_angstrom=np.asarray(coords0, dtype=float),
        final_coordinates_angstrom=coords,
        parameters=parameters,
        geometry_parameters=geometry_parameters,
        residuals=residual_rows,
        rotational_constants=rotational_constant_rows,
        kraitchman=kraitchman_rows,
        kraitchman_seed=kraitchman_seed,
        covariance=covariance,
        correlation=correlation,
        jacobian=jac,
        hessian=hessian,
        hessian_eigenvalues=hessian_eigenvalues,
        stationary_point=stationary_point,
        gic_labels=labels,
        b_matrix=bq,
        iterations=iteration,
        rms_MHz=rms,
        diagnostics=diagnostics,
        leave_one_out=leave_one_out_rows,
        iteration_trace=tuple(iteration_traces),
        weight_diagnostics=weight_diagnostic_rows,
        manifest=manifest,
        sonic_parameters=parameters,
    )


def _fit_semiexperimental_geometry_cartesian_symmetry(
    request: SemiexperimentalFitRequest,
    *,
    max_iter: int | None,
    step: float,
    damping: float,
    max_step: float,
    prune_condition: float,
    tolerance_MHz: float,
    gradient_tolerance: float,
    checkpoint: Path | None,
    restart: Path | None,
    outdir: Path | None,
) -> SemiexperimentalFitResult:
    geometry_input = read_geometry_input(Path(request.initial_geometry))
    atoms = list(geometry_input.atoms)
    coords = np.asarray(geometry_input.coordinates_angstrom, dtype=float)
    request, preflight_warnings = _request_with_auto_resolved_isotopologues(request, atoms, coords)
    if restart is not None:
        coords = _read_semiexp_checkpoint(Path(restart), expected_atoms=len(atoms))
    coords, sycart_workdir = _gicforge_sycart_coordinates(tuple(atoms), coords, outdir)
    coords, oracle_symmetry = _oracle_cartesian_symmetry_state(tuple(atoms), coords, sycart_workdir)
    coords0 = coords.copy()
    fixed_parameters = _combined_fixed_parameters(
        request.fixed_parameters, geometry_input.fixed_parameters
    )
    fixed_mode_patterns = _gic_fixed_patterns(fixed_parameters)
    fixed_primitives = _fixed_primitives_from_patterns(fixed_parameters)
    linear_constraints = _linear_primitive_constraints_from_patterns(fixed_parameters)
    expression_constraints = _gic_expression_constraints_from_patterns(fixed_parameters)
    expression_definitions = _gic_expression_definitions_from_patterns(fixed_parameters)
    if (
        expression_constraints
        and any(_gic_expression_uses_gic_names(item.expression) for item in expression_constraints)
    ) or any(_gic_expression_uses_gic_names(item.expression) for item in expression_definitions):
        raise ScientificValidationError(
            "GIC### expression constraints require coordinate_model='gic'"
        )
    expression_targets = _gic_expression_constraint_targets(
        expression_constraints,
        coords,
        (),
        np.zeros((0, 0), dtype=float),
        (),
        definitions=expression_definitions,
    )
    if fixed_primitives or linear_constraints or expression_constraints:
        coords = _project_fixed_primitives(
            coords,
            fixed_primitives,
            _fixed_primitive_targets(fixed_primitives, coords),
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=(),
            u_matrix=np.zeros((0, 0), dtype=float),
            labels=(),
            expression_definitions=expression_definitions,
        )
    topology_lock = _topology_lock(atoms, coords)
    _validate_observations(request.observations, len(atoms))
    mode_model = cartesian_symmetry_coordinate_model(
        tuple(atoms), coords0, symmetry=oracle_symmetry
    )
    labels = mode_model.labels
    constraint_prims = tuple(_constraint_primitive_pool(atoms, (), coords))
    fixed_primitives = _merge_primitives(
        fixed_primitives,
        _hydrogen_fixed_primitives(atoms, constraint_prims, fixed_parameters, coords=coords),
    )
    fixed_primitives = _symmetry_expanded_fixed_primitives(
        atoms,
        coords,
        constraint_prims,
        fixed_primitives,
        symmetry=oracle_symmetry,
    )
    fixed_primitive_targets = _fixed_primitive_targets(fixed_primitives, coords)

    measurement_model = _build_measurement_model_cartesian_basis(
        request,
        atoms,
        coords,
        labels,
        mode_model.cartesian_from_q,
    )
    active_mask = _active_mask(labels, fixed_mode_patterns, request.parameter_classes)
    active_mask &= mode_model.active_totally_symmetric_mask
    transform, reduced_names, class_by_mode = _parameter_class_transform(
        labels, active_mask, request.parameter_classes
    )
    transform, reduced_names = _primitive_constrained_cartesian_transform(
        coords,
        mode_model.cartesian_from_q,
        active_mask,
        transform,
        reduced_names,
        fixed_primitives,
        linear_constraints=linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        expression_definitions=expression_definitions,
    )
    auto_pruned_patterns: tuple[str, ...] = ()
    if prune_condition > 0.0 and transform.shape[1] > 1:
        jac_modes = _jacobian_constants_wrt_cartesian_basis(
            atoms,
            coords,
            request,
            labels,
            measurement_model,
            mode_model.cartesian_from_q,
        )
        weighted = (_active_coordinate_jacobian(jac_modes, active_mask) @ transform) * np.sqrt(
            measurement_model.weights
        )[:, None]
        auto_pruned_patterns = _weak_parameter_patterns(reduced_names, weighted, prune_condition)
        if auto_pruned_patterns:
            active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
            transform, reduced_names, class_by_mode = _parameter_class_transform(
                labels, active_mask, request.parameter_classes
            )
            transform, reduced_names = _primitive_constrained_cartesian_transform(
                coords,
                mode_model.cartesian_from_q,
                active_mask,
                transform,
                reduced_names,
                fixed_primitives,
                linear_constraints=linear_constraints,
                expression_constraints=expression_constraints,
                expression_targets=expression_targets,
                expression_definitions=expression_definitions,
            )

    n_optimized_parameters = transform.shape[1]
    _validate_observation_budget(
        measurement_model,
        n_optimized_parameters,
        coordinate_model=request.coordinate_model,
    )
    loop_max_iter = (
        _resolve_max_iterations(max_iter, n_optimized_parameters) if n_optimized_parameters else 0
    )
    current_damping = max(float(damping), 0.0)
    trust_radius = float(max_step) if max_step > 0.0 else 0.0
    accepted_steps = 0
    rejected_steps = 0
    stalled_rejections = 0
    parameter_scale_min = 1.0
    parameter_scale_max = 1.0
    last_trust_ratio = 0.0
    last_line_search_scale = 0.0
    robust_scale_used = 0.0
    robust_downweighted_observations = 0
    robust_downweighted_isotopologues = 0
    robust_sqrt = np.ones_like(measurement_model.observed, dtype=float)
    checkpoint_file = _checkpoint_path(outdir, checkpoint)
    previous_objective = None
    convergence_reason = (
        "max_iter" if loop_max_iter else "no_active_totally_symmetric_cartesian_coordinates"
    )
    iteration_traces: list[SemiexperimentalIterationTrace] = []
    iteration = 0

    for iteration in range(1, loop_max_iter + 1):
        active_mask = _active_mask(labels, fixed_mode_patterns, request.parameter_classes)
        active_mask &= mode_model.active_totally_symmetric_mask
        active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
        q = mode_model.values(coords)
        calc = _measurement_vector(atoms, coords, request, q, labels, measurement_model)
        obs = measurement_model.observed
        weights = measurement_model.weights
        sqrt_weights = np.sqrt(weights)
        residual = obs - calc
        base_weighted_residual = residual * sqrt_weights
        (
            robust_sqrt,
            robust_scale_used,
            robust_downweighted_observations,
            robust_downweighted_isotopologues,
        ) = _robust_sqrt_weights_for_model(
            base_weighted_residual,
            request.robust_loss,
            request.robust_scale,
            measurement_model,
        )
        effective_sqrt_weights = sqrt_weights * robust_sqrt
        weighted_residual = residual * effective_sqrt_weights
        current_objective = objective(weighted_residual)
        jac_modes = _jacobian_constants_wrt_cartesian_basis(
            atoms,
            coords,
            request,
            labels,
            measurement_model,
            mode_model.cartesian_from_q,
        )
        transform, reduced_names, class_by_mode = _parameter_class_transform(
            labels, active_mask, request.parameter_classes
        )
        transform, reduced_names = _primitive_constrained_cartesian_transform(
            coords,
            mode_model.cartesian_from_q,
            active_mask,
            transform,
            reduced_names,
            fixed_primitives,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            expression_definitions=expression_definitions,
        )
        jac = _active_coordinate_jacobian(jac_modes, active_mask) @ transform
        base_scales = np.ones(jac.shape[1], dtype=float)
        jac_weighted = jac * effective_sqrt_weights[:, None]
        reduced_scales = _dynamic_parameter_scales(jac_weighted, base_scales)
        if reduced_scales.size:
            parameter_scale_min = min(parameter_scale_min, float(np.min(reduced_scales)))
            parameter_scale_max = max(parameter_scale_max, float(np.max(reduced_scales)))
        if np.sqrt(np.mean(residual * residual)) < tolerance_MHz:
            convergence_reason = "rms_tolerance"
            break
        jac_weighted_scaled = (
            jac_weighted * reduced_scales[None, :] if reduced_scales.size else jac_weighted
        )
        gradient = jac_weighted_scaled.T @ weighted_residual
        gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_inf_norm < gradient_tolerance:
            convergence_reason = "gradient_tolerance"
            break
        trust_step = _adaptive_lm_step(
            jac_weighted_scaled, weighted_residual, current_damping, trust_radius
        )
        dq_scaled = trust_step.step
        current_damping = trust_step.shift
        dq_reduced = reduced_scales * dq_scaled if reduced_scales.size else dq_scaled
        dq_active = transform @ dq_reduced
        dq = np.zeros_like(q)
        dq[np.where(active_mask)[0]] = dq_active
        line_search = _line_search_update_cartesian_basis(
            atoms,
            coords,
            request,
            labels,
            measurement_model,
            mode_model,
            dq,
            current_objective=current_objective,
            base_q=q,
            weighted_residual=weighted_residual,
            jac_weighted=jac_weighted_scaled,
            reduced_step=dq_scaled,
            robust_sqrt_weights=robust_sqrt,
            fixed_primitives=fixed_primitives,
            fixed_primitive_targets=fixed_primitive_targets,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            expression_definitions=expression_definitions,
        )
        last_trust_ratio = line_search.ratio
        last_line_search_scale = line_search.scale
        if line_search.accepted:
            try:
                _validate_locked_topology(
                    atoms,
                    line_search.coords,
                    topology_lock,
                    context="symmetry-Cartesian semiexperimental fit",
                )
            except Exception:
                rejected_steps += 1
                stalled_rejections += 1
                current_damping, trust_radius = _rejected_trust_update(
                    current_damping, trust_radius, max_step
                )
                iteration_traces.append(
                    _iteration_trace_row(
                        iteration,
                        "topology_rejected",
                        current_objective,
                        line_search,
                        current_damping,
                        trust_radius,
                        line_search.scale * float(np.linalg.norm(dq_scaled)),
                        gradient_inf_norm,
                        jac_weighted_scaled,
                        line_search.coords,
                        fixed_primitives,
                        fixed_primitive_targets,
                        linear_constraints=linear_constraints,
                        expression_constraints=expression_constraints,
                        expression_targets=expression_targets,
                        prims=(),
                        u_matrix=np.zeros((0, 0), dtype=float),
                        labels=(),
                        expression_definitions=expression_definitions,
                        robust_scale=robust_scale_used,
                        robust_downweighted_observations=robust_downweighted_observations,
                        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                        coordinate_model_age=0,
                        b_projector_secant_error=0.0,
                        linear_solver=trust_step.solver,
                    )
                )
                if _trust_region_is_stalled(
                    current_damping, trust_radius, stalled_rejections, max_step
                ):
                    convergence_reason = "line_search_stalled"
                    break
                continue
            coords = line_search.coords
            accepted_steps += 1
            stalled_rejections = 0
            current_damping, trust_radius = _accepted_trust_update(
                current_damping,
                trust_radius,
                line_search.ratio,
                line_search.scale,
                float(line_search.scale * np.linalg.norm(dq_scaled)),
                max_step,
            )
            iteration_traces.append(
                _iteration_trace_row(
                    iteration,
                    "accepted",
                    current_objective,
                    line_search,
                    current_damping,
                    trust_radius,
                    line_search.scale * float(np.linalg.norm(dq_scaled)),
                    gradient_inf_norm,
                    jac_weighted_scaled,
                    line_search.coords,
                    fixed_primitives,
                    fixed_primitive_targets,
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=(),
                    u_matrix=np.zeros((0, 0), dtype=float),
                    labels=(),
                    expression_definitions=expression_definitions,
                    robust_scale=robust_scale_used,
                    robust_downweighted_observations=robust_downweighted_observations,
                    robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                    coordinate_model_age=0,
                    b_projector_secant_error=0.0,
                    linear_solver=trust_step.solver,
                )
            )
            if (
                previous_objective is not None
                and abs(previous_objective - line_search.objective) < tolerance_MHz * tolerance_MHz
            ):
                convergence_reason = "objective_tolerance"
                break
            previous_objective = line_search.objective
            _write_semiexp_checkpoint(
                checkpoint_file,
                atoms,
                coords,
                iteration=iteration,
                damping=current_damping,
                trust_radius=trust_radius,
                labels=labels,
                active_mask=active_mask,
                robust_sqrt_weights=robust_sqrt,
                robust_scale=robust_scale_used,
                robust_downweighted_rows=robust_downweighted_observations,
                robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                coordinate_model=request.coordinate_model,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
            )
        else:
            rejected_steps += 1
            stalled_rejections += 1
            current_damping, trust_radius = _rejected_trust_update(
                current_damping, trust_radius, max_step
            )
            iteration_traces.append(
                _iteration_trace_row(
                    iteration,
                    "rejected",
                    current_objective,
                    line_search,
                    current_damping,
                    trust_radius,
                    line_search.scale * float(np.linalg.norm(dq_scaled)),
                    gradient_inf_norm,
                    jac_weighted_scaled,
                    coords,
                    fixed_primitives,
                    fixed_primitive_targets,
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=(),
                    u_matrix=np.zeros((0, 0), dtype=float),
                    labels=(),
                    expression_definitions=expression_definitions,
                    robust_scale=robust_scale_used,
                    robust_downweighted_observations=robust_downweighted_observations,
                    robust_downweighted_isotopologues=robust_downweighted_isotopologues,
                    coordinate_model_age=0,
                    b_projector_secant_error=0.0,
                    linear_solver=trust_step.solver,
                )
            )
            if _trust_region_is_stalled(
                current_damping, trust_radius, stalled_rejections, max_step
            ):
                convergence_reason = (
                    "step_tolerance"
                    if _objective_has_stabilized(
                        previous_objective, current_objective, tolerance_MHz
                    )
                    else "line_search_stalled"
                )
                break
    else:
        iteration = loop_max_iter

    _validate_locked_topology(
        atoms, coords, topology_lock, context="final symmetry-Cartesian semiexperimental fit"
    )
    active_mask = _active_mask(labels, fixed_mode_patterns, request.parameter_classes)
    active_mask &= mode_model.active_totally_symmetric_mask
    active_mask &= _auto_pruned_active_mask(labels, auto_pruned_patterns)
    q_final = mode_model.values(coords)
    # Keep the initial auto-selected component pair fixed for the full fit.
    calc = _measurement_vector(atoms, coords, request, q_final, labels, measurement_model)
    obs = measurement_model.observed
    residual = obs - calc
    jac_modes = _jacobian_constants_wrt_cartesian_basis(
        atoms,
        coords,
        request,
        labels,
        measurement_model,
        mode_model.cartesian_from_q,
    )
    transform, reduced_names, class_by_mode = _parameter_class_transform(
        labels, active_mask, request.parameter_classes
    )
    transform, reduced_names = _primitive_constrained_cartesian_transform(
        coords,
        mode_model.cartesian_from_q,
        active_mask,
        transform,
        reduced_names,
        fixed_primitives,
        linear_constraints=linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        expression_definitions=expression_definitions,
    )
    jac = _active_coordinate_jacobian(jac_modes, active_mask) @ transform
    sqrt_weights = np.sqrt(measurement_model.weights)
    base_weighted_residual = residual * sqrt_weights
    (
        robust_sqrt,
        robust_scale_used,
        robust_downweighted_observations,
        robust_downweighted_isotopologues,
    ) = _robust_sqrt_weights_for_model(
        base_weighted_residual,
        request.robust_loss,
        request.robust_scale,
        measurement_model,
    )
    effective_sqrt_weights = sqrt_weights * robust_sqrt
    weighted_jac = jac * effective_sqrt_weights[:, None]
    weighted_residual = residual * effective_sqrt_weights
    hessian = _least_squares_hessian(weighted_jac)
    covariance = _covariance(weighted_jac, weighted_residual)
    correlation = _correlation(covariance)
    hessian_eigenvalues = np.linalg.eigvalsh(hessian) if hessian.size else np.array(())
    stationary_point = _stationary_point_type(hessian_eigenvalues)
    diagnostics = _diagnostics(
        weighted_jac,
        weighted_residual,
        convergence_reason=convergence_reason,
        damping=current_damping,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        max_iterations=loop_max_iter,
        n_optimized_parameters=jac.shape[1],
        observable=measurement_model.observable,
        components=measurement_model.components,
        planar=measurement_model.planar,
        auto_pruned_parameters=auto_pruned_patterns,
        prune_condition_target=prune_condition,
        trust_radius=trust_radius,
        last_trust_ratio=last_trust_ratio,
        last_line_search_scale=last_line_search_scale,
        parameter_scale_min=parameter_scale_min,
        parameter_scale_max=parameter_scale_max,
        robust_loss=request.robust_loss,
        robust_scale=robust_scale_used,
        robust_downweighted_observations=robust_downweighted_observations,
        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
        coordinate_model=request.coordinate_model,
    )
    class_by_mode = _mark_auto_pruned_classes(labels, class_by_mode, auto_pruned_patterns)
    parameters = _parameters(
        labels,
        q_final,
        active_mask,
        transform=transform,
        covariance=covariance,
        class_by_gic=class_by_mode,
    )
    residual_rows = _residual_rows(measurement_model, calc, obs)
    rotational_constant_rows = _rotational_constant_rows(
        atoms,
        coords,
        request.observations,
        request.excluded_rotational_constants,
    )
    cartesian_from_parameters = _cartesian_from_reduced_coordinates(
        mode_model.cartesian_from_q, active_mask, transform
    )
    geometry_parameters = _geometry_parameters(
        atoms,
        coords,
        initial_coords=coords0,
        cartesian_from_parameters=cartesian_from_parameters,
        covariance=covariance,
        topology_lock=topology_lock,
    )
    sonic_parameters = _sonic_parameters_from_cartesian_covariance(
        atoms,
        coords,
        cartesian_from_parameters,
        covariance,
    )
    kraitchman_rows = kraitchman_comparison(atoms, coords, request.observations)
    kraitchman_seed = kraitchman_seed_geometry(atoms, coords, request.observations, kraitchman_rows)
    rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    weight_diagnostic_rows = _weight_diagnostic_rows(
        measurement_model,
        calc,
        residual,
        weighted_jac,
        weighted_residual,
        robust_sqrt,
    )
    leave_one_out_rows = (
        _leave_one_out_refits(
            request,
            atoms,
            coords,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            tolerance_MHz=tolerance_MHz,
            gradient_tolerance=gradient_tolerance,
        )
        if request.leave_one_out
        else ()
    )
    _write_semiexp_checkpoint(
        checkpoint_file,
        atoms,
        coords,
        iteration=iteration,
        damping=current_damping,
        trust_radius=trust_radius,
        labels=labels,
        active_mask=active_mask,
        robust_sqrt_weights=robust_sqrt,
        robust_scale=robust_scale_used,
        robust_downweighted_rows=robust_downweighted_observations,
        robust_downweighted_isotopologues=robust_downweighted_isotopologues,
        coordinate_model=request.coordinate_model,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )
    manifest = None
    if outdir is not None:
        manifest = write_semiexperimental_outputs(
            Path(outdir),
            request,
            atoms,
            coords,
            parameters,
            residual_rows,
            kraitchman_rows,
            rotational_constants=rotational_constant_rows,
            geometry_parameters=geometry_parameters,
            kraitchman_seed=kraitchman_seed,
            input_fixed_parameters=geometry_input.fixed_parameters,
            fixed_primitives=fixed_primitives,
            leave_one_out=leave_one_out_rows,
            checkpoint_path=checkpoint_file,
            effective_parameter_names=reduced_names,
            covariance=covariance,
            correlation=correlation,
            hessian=hessian,
            hessian_eigenvalues=hessian_eigenvalues,
            stationary_point=stationary_point,
            diagnostics=diagnostics,
            measurement_model=measurement_model,
            weighted_jacobian=weighted_jac,
            weighted_residual=weighted_residual,
            robust_sqrt_weights=robust_sqrt,
            weight_diagnostics=weight_diagnostic_rows,
            iteration_trace=tuple(iteration_traces),
            preflight_warnings=preflight_warnings,
        )
    return SemiexperimentalFitResult(
        atoms=tuple(atoms),
        initial_coordinates_angstrom=np.asarray(coords0, dtype=float),
        final_coordinates_angstrom=coords,
        parameters=parameters,
        geometry_parameters=geometry_parameters,
        residuals=residual_rows,
        rotational_constants=rotational_constant_rows,
        kraitchman=kraitchman_rows,
        kraitchman_seed=kraitchman_seed,
        covariance=covariance,
        correlation=correlation,
        jacobian=jac,
        hessian=hessian,
        hessian_eigenvalues=hessian_eigenvalues,
        stationary_point=stationary_point,
        gic_labels=labels,
        b_matrix=mode_model.cartesian_from_q.T,
        iterations=iteration,
        rms_MHz=rms,
        diagnostics=diagnostics,
        leave_one_out=leave_one_out_rows,
        iteration_trace=tuple(iteration_traces),
        weight_diagnostics=weight_diagnostic_rows,
        manifest=manifest,
        sonic_parameters=sonic_parameters,
    )


def write_semiexperimental_outputs(
    outdir: Path,
    request: SemiexperimentalFitRequest,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    parameters: tuple[SemiexperimentalParameter, ...],
    residuals: tuple[SemiexperimentalResidual, ...],
    kraitchman: tuple[KraitchmanComparison, ...] = (),
    rotational_constants: tuple[SemiexperimentalRotationalConstantComparison, ...] | None = None,
    geometry_parameters: tuple[SemiexperimentalGeometryParameter, ...] | None = None,
    kraitchman_seed: KraitchmanSeedResult | None = None,
    effective_parameter_names: tuple[str, ...] = (),
    covariance: np.ndarray | None = None,
    correlation: np.ndarray | None = None,
    hessian: np.ndarray | None = None,
    hessian_eigenvalues: np.ndarray | None = None,
    stationary_point: str = "not_checked",
    diagnostics: SemiexperimentalFitDiagnostics | None = None,
    input_fixed_parameters: tuple[str, ...] = (),
    fixed_primitives: tuple[Primitive, ...] = (),
    leave_one_out: tuple[SemiexperimentalLeaveOneOutRow, ...] = (),
    checkpoint_path: Path | None = None,
    measurement_model: MeasurementModel | None = None,
    weighted_jacobian: np.ndarray | None = None,
    weighted_residual: np.ndarray | None = None,
    robust_sqrt_weights: np.ndarray | None = None,
    weight_diagnostics: tuple[SemiexperimentalWeightDiagnostic, ...] = (),
    iteration_trace: tuple[SemiexperimentalIterationTrace, ...] = (),
    preflight_warnings: tuple[SemiexperimentalDiagnosticWarning, ...] = (),
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    xyz = outdir / "semiexp_geometry.xyz"
    params = outdir / "semiexp_parameters.csv"
    geometry_params = outdir / "semiexp_geometry_parameters.csv"
    residual_csv = outdir / "semiexp_residuals.csv"
    rotconst_csv = outdir / "semiexp_rotational_constants.csv"
    text_report = outdir / "semiexp_report.txt"
    kraitchman_csv = outdir / "semiexp_kraitchman.csv"
    kraitchman_xyz = outdir / "semiexp_kraitchman_geometry.xyz"
    covariance_csv = outdir / "semiexp_covariance.csv"
    correlation_csv = outdir / "semiexp_correlation.csv"
    hessian_csv = outdir / "semiexp_hessian.csv"
    hessian_eigs_csv = outdir / "semiexp_hessian_eigenvalues.csv"
    diagnostics_csv = outdir / "semiexp_diagnostics.csv"
    influence_csv = outdir / "semiexp_influence.csv"
    weight_diagnostics_csv = outdir / "semiexp_weight_diagnostics.csv"
    high_correlation_csv = outdir / "semiexp_high_correlations.csv"
    svd_diagnostics_csv = outdir / "semiexp_svd_diagnostics.csv"
    uncertainty_diagnostics_csv = outdir / "semiexp_uncertainty_diagnostics.csv"
    iteration_trace_csv = outdir / "semiexp_iteration_trace.csv"
    constraints_csv = outdir / "semiexp_constraints.csv"
    warnings_csv = outdir / "semiexp_warnings.csv"
    leave_one_out_csv = outdir / "semiexp_leave_one_out.csv"
    active_names = effective_parameter_names or _effective_parameter_names(parameters)
    geometry_rows = (
        geometry_parameters
        if geometry_parameters is not None
        else _geometry_parameters(atoms, coords)
    )
    rotconst_rows = (
        rotational_constants
        if rotational_constants is not None
        else _rotational_constant_rows(
            atoms,
            np.asarray(coords, dtype=float),
            request.observations,
            request.excluded_rotational_constants,
        )
    )
    fixed_parameters = _combined_fixed_parameters(request.fixed_parameters, input_fixed_parameters)
    svd_summary = _svd_summary_lines(active_names, weighted_jacobian)
    constraint_summary = _constraint_summary_lines(
        fixed_parameters, fixed_primitives, request.parameter_classes, parameters
    )
    diagnostic_warnings = (
        preflight_warnings
        + _isotopic_mapping_warning_rows(atoms, coords, request.observations)
        + _semiexp_warning_rows(
            diagnostics,
            active_names,
            parameters,
            geometry_rows,
            weighted_jacobian,
            measurement_model,
            robust_sqrt_weights,
            weighted_residual,
            weight_diagnostics,
            iteration_trace=iteration_trace,
            correlation=correlation,
        )
    )
    write_xyz(xyz, atoms, coords, comment="MATRIX/MORPHEUS semiexperimental equilibrium geometry")
    params.write_text(parameters_csv(parameters), encoding="utf-8")
    geometry_params.write_text(geometry_parameters_csv(geometry_rows), encoding="utf-8")
    residual_csv.write_text(residuals_csv(residuals), encoding="utf-8")
    rotconst_csv.write_text(rotational_constants_csv(rotconst_rows), encoding="utf-8")
    text_report.write_text(
        semiexperimental_text_report(
            request,
            parameters,
            geometry_rows,
            residuals,
            rotconst_rows,
            diagnostics=diagnostics,
            stationary_point=stationary_point,
            fixed_parameters=fixed_parameters,
            diagnostic_warnings=diagnostic_warnings,
            svd_summary=svd_summary,
            constraint_summary=constraint_summary,
            iteration_trace=iteration_trace,
            leave_one_out=leave_one_out,
        ),
        encoding="utf-8",
    )
    kraitchman_csv.write_text(kraitchman_csv_rows(kraitchman), encoding="utf-8")
    if kraitchman_seed is not None:
        write_xyz(
            kraitchman_xyz,
            atoms,
            kraitchman_seed.coordinates_angstrom,
            comment=f"MATRIX/MORPHEUS Kraitchman substitution geometry; method={kraitchman_seed.method}; principal-axis frame",
        )
    covariance_csv.write_text(_matrix_csv(active_names, covariance), encoding="utf-8")
    correlation_csv.write_text(_matrix_csv(active_names, correlation), encoding="utf-8")
    hessian_csv.write_text(_matrix_csv(active_names, hessian), encoding="utf-8")
    hessian_eigs_csv.write_text(_eigenvalues_csv(hessian_eigenvalues), encoding="utf-8")
    diagnostics_csv.write_text(_diagnostics_csv(diagnostics), encoding="utf-8")
    influence_csv.write_text(
        _influence_csv(measurement_model, residuals, weighted_jacobian, weighted_residual),
        encoding="utf-8",
    )
    weight_diagnostics_csv.write_text(
        _weight_diagnostics_csv(weight_diagnostics),
        encoding="utf-8",
    )
    high_correlation_csv.write_text(
        _high_correlations_csv(active_names, correlation), encoding="utf-8"
    )
    svd_diagnostics_csv.write_text(
        _svd_diagnostics_csv(active_names, weighted_jacobian), encoding="utf-8"
    )
    uncertainty_diagnostics_csv.write_text(
        _uncertainty_diagnostics_csv(active_names, weighted_jacobian, weighted_residual),
        encoding="utf-8",
    )
    iteration_trace_csv.write_text(iteration_trace_csv_rows(iteration_trace), encoding="utf-8")
    constraints_csv.write_text(
        _constraints_csv(fixed_parameters, fixed_primitives, request.parameter_classes, parameters),
        encoding="utf-8",
    )
    warnings_csv.write_text(_warnings_csv(diagnostic_warnings), encoding="utf-8")
    if leave_one_out:
        leave_one_out_csv.write_text(_leave_one_out_csv(leave_one_out), encoding="utf-8")
    manifest_inputs = {"initial_geometry": request.initial_geometry}
    if request.coordinate_model == "cartesian_symmetry":
        coordinate_generation = {
            "primitive_source": "SMITH SYCART symmetrized Cartesian parent geometry",
            "reduction": "ordinary Cartesian translations and rotations projected out from the SMITH-symmetrized geometry",
            "symmetry": "SMITH writes symmetrized Cartesian coordinates; Cartesian displacement basis is projected with the same detected point-group irreps",
            "active_subspace": "totally symmetric symmetry-adapted Cartesian displacements only",
            "ring_coordinates": "not used as working coordinates; final primitive internals are reported from final Cartesian geometry",
            "gicforge_sycart": str(outdir / "gicforge_sycart"),
            "line_search": "SVD More-Hebden trust-region least-squares with fixed symmetry-Cartesian basis; no GIC B projector is required",
            "restart_policy": "restart jobs first call SMITH SYCART, then rebuild the symmetry-Cartesian basis from the symmetrized parent geometry",
        }
        backend_coordinate_model = "gicforge-sycart-symmetry-cartesian"
        b_matrix_description = "not required for working-coordinate updates"
    else:
        coordinate_generation = {
            "primitive_source": "SMITH definition service run once per fresh fit; frozen SONIC schema reused for all iterations",
            "reduction": "primitive stretches plus non-redundant non-stretch GIC transform",
            "symmetry": "frozen ORACLE point group with deterministic SMITH irrep assignment",
            "active_subspace": "SMITH-assigned totally symmetric coordinates only",
            "ring_coordinates": "SMITH triangular-flap ring coordinates (U-based local out-of-plane compatibility mode available)",
            "gicforge_iterations": str(outdir / "gicforge_iterations"),
            "line_search": "SVD More-Hebden trust-region least-squares with frozen GIC schema, analytic B rebuilds when required, and secant-updated B projector between rebuilds",
            "restart_policy": "restart jobs rebuild the GIC schema; ordinary iterations reuse the saved schema",
        }
        backend_coordinate_model = "gicforge-frozen-definition"
        b_matrix_description = "analytic from frozen GIC definition"
    outputs = {
        "geometry": xyz,
        "parameters": params,
        "geometry_parameters": geometry_params,
        "residuals": residual_csv,
        "rotational_constants": rotconst_csv,
        "text_report": text_report,
        "kraitchman": kraitchman_csv,
        "covariance": covariance_csv,
        "correlation": correlation_csv,
        "hessian": hessian_csv,
        "hessian_eigenvalues": hessian_eigs_csv,
        "diagnostics": diagnostics_csv,
        "influence": influence_csv,
        "weight_diagnostics": weight_diagnostics_csv,
        "high_correlations": high_correlation_csv,
        "svd_diagnostics": svd_diagnostics_csv,
        "uncertainty_diagnostics": uncertainty_diagnostics_csv,
        "iteration_trace": iteration_trace_csv,
        "constraints": constraints_csv,
        "warnings": warnings_csv,
    }
    if leave_one_out:
        outputs["leave_one_out"] = leave_one_out_csv
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        outputs["checkpoint"] = Path(checkpoint_path)
    if kraitchman_seed is not None:
        outputs["kraitchman_geometry"] = kraitchman_xyz
    manifest = build_run_manifest(
        workflow="semiexperimental_geometry",
        status="completed",
        run_dir=outdir,
        inputs=manifest_inputs,
        outputs=outputs,
        parameters={
            "fixed_parameters": fixed_parameters,
            "input_fixed_parameters": input_fixed_parameters,
            "parameter_classes": tuple(
                {"name": item.name, "patterns": item.patterns, "mode": item.mode}
                for item in request.parameter_classes
            ),
            "stationary_point": stationary_point,
            "convergence_reason": diagnostics.convergence_reason if diagnostics else "not_reported",
            "coordinate_model": diagnostics.coordinate_model
            if diagnostics
            else request.coordinate_model,
            "observable": diagnostics.observable if diagnostics else request.observable,
            "rotational_components": diagnostics.components
            if diagnostics
            else request.rotational_components,
            "isotopologues": tuple(obs.label for obs in request.observations),
            "n_isotopologues": len(request.observations),
            "excluded_rotational_constants": request.excluded_rotational_constants,
            "n_qm_predicates": len(request.qm_predicates),
            "n_gic_parameters": len(parameters),
            "n_working_parameters": len(parameters),
            "n_effective_parameters": len(active_names),
            "n_active_gic_parameters": sum(1 for item in parameters if item.active),
            "n_active_working_parameters": sum(1 for item in parameters if item.active),
            "auto_pruned_parameters": diagnostics.auto_pruned_parameters if diagnostics else (),
            "prune_condition_target": diagnostics.prune_condition_target if diagnostics else 0.0,
            "max_iterations": diagnostics.max_iterations if diagnostics else None,
            "n_kraitchman_rows": len(kraitchman),
            "kraitchman_seed_method": kraitchman_seed.method
            if kraitchman_seed
            else "not_available",
            "n_kraitchman_seed_atoms": len(kraitchman_seed.fitted_atom_indices)
            if kraitchman_seed
            else 0,
            "rank": diagnostics.rank if diagnostics else None,
            "incremental_rank": diagnostics.incremental_rank if diagnostics else None,
            "condition_number": diagnostics.condition_number if diagnostics else None,
            **_rotational_residual_manifest_stats(rotconst_rows),
            "weighted_rms": diagnostics.weighted_rms if diagnostics else None,
            "reduced_chi_square": diagnostics.reduced_chi_square if diagnostics else None,
            "n_warnings": len(diagnostic_warnings),
            "warning_codes": tuple(item.code for item in diagnostic_warnings),
            "gicforge_calls": diagnostics.gicforge_calls if diagnostics else None,
            "coordinate_model_reuse_steps": diagnostics.coordinate_model_reuse_steps
            if diagnostics
            else None,
            "trust_radius": diagnostics.trust_radius if diagnostics else None,
            "last_trust_ratio": diagnostics.last_trust_ratio if diagnostics else None,
            "last_line_search_scale": diagnostics.last_line_search_scale if diagnostics else None,
            "b_projector_analytic_refreshes": diagnostics.b_projector_analytic_refreshes
            if diagnostics
            else None,
            "b_projector_secant_updates": diagnostics.b_projector_secant_updates
            if diagnostics
            else None,
            "b_projector_secant_rejections": diagnostics.b_projector_secant_rejections
            if diagnostics
            else None,
            "last_b_projector_secant_error": diagnostics.last_b_projector_secant_error
            if diagnostics
            else None,
            "parameter_scale_min": diagnostics.parameter_scale_min if diagnostics else None,
            "parameter_scale_max": diagnostics.parameter_scale_max if diagnostics else None,
            "robust_loss": diagnostics.robust_loss if diagnostics else request.robust_loss,
            "robust_scale": diagnostics.robust_scale if diagnostics else request.robust_scale,
            "robust_downweighted_observations": diagnostics.robust_downweighted_observations
            if diagnostics
            else 0,
            "robust_downweighted_isotopologues": diagnostics.robust_downweighted_isotopologues
            if diagnostics
            else 0,
            "linear_solver": diagnostics.linear_solver
            if diagnostics
            else "svd_more_hebden_trust_region",
            "n_iteration_trace_rows": len(iteration_trace),
            "leave_one_out": bool(leave_one_out),
            "n_leave_one_out_rows": len(leave_one_out),
            "coordinate_generation": coordinate_generation,
        },
        backend={
            "solver": "python-orchestrated adaptive SVD More-Hebden trust-region least-squares with QR/Cauchy fallbacks",
            "coordinate_model": backend_coordinate_model,
            "b_matrix": b_matrix_description,
            "backtransform": (
                "LINK hybrid typed SONIC with adaptive line-search and generic fallback"
                if request.coordinate_model == "gic"
                else "Cartesian symmetry displacement"
            ),
            "fortran77_role": "validated numerical kernels only",
            "fortran77_source": "fortran/semiexp/semiexp_core.f",
        },
        messages=[
            "Semiexperimental workflow is orchestrated in Python.",
            (
                "GIC displacements use LINK's coordinate-aware hybrid SONIC back-transformation."
                if request.coordinate_model == "gic"
                else "Cartesian symmetry coordinates use direct Cartesian displacements."
            ),
            "Fortran77 semiexp code is kept as an independent validated numerical-kernel layer.",
        ],
    )
    return manifest.write(outdir / "semiexp_manifest.json")
















def _leave_one_out_refits(
    request: SemiexperimentalFitRequest,
    atoms: list[str] | tuple[str, ...],
    full_coords: np.ndarray,
    *,
    max_iter: int | None,
    step: float,
    damping: float,
    max_step: float,
    prune_condition: float,
    tolerance_MHz: float,
    gradient_tolerance: float,
) -> tuple[SemiexperimentalLeaveOneOutRow, ...]:
    if len(request.observations) < 2:
        return ()
    rows: list[SemiexperimentalLeaveOneOutRow] = []
    for omitted in request.observations:
        training = tuple(obs for obs in request.observations if obs.label != omitted.label)
        if not training:
            continue
        sub_request = replace(request, observations=training, leave_one_out=False)
        try:
            sub_result = fit_semiexperimental_geometry(
                sub_request,
                max_iter=max_iter,
                step=step,
                damping=damping,
                max_step=max_step,
                prune_condition=prune_condition,
                tolerance_MHz=tolerance_MHz,
                gradient_tolerance=gradient_tolerance,
                outdir=None,
            )
        except Exception:
            continue
        omitted_rows = _rotational_constant_rows(
            atoms,
            sub_result.final_coordinates_angstrom,
            (omitted,),
            request.excluded_rotational_constants,
        )
        diffs = np.asarray([row.difference_MHz for row in omitted_rows], dtype=float)
        delta = np.asarray(sub_result.final_coordinates_angstrom, dtype=float) - np.asarray(
            full_coords, dtype=float
        )
        sigmas = np.asarray(
            [parameter.sigma for parameter in sub_result.parameters if parameter.active],
            dtype=float,
        )
        rows.append(
            SemiexperimentalLeaveOneOutRow(
                omitted.label,
                len(training),
                sub_result.rms_MHz,
                float(np.sqrt(np.mean(diffs * diffs))) if diffs.size else 0.0,
                float(np.max(np.abs(diffs))) if diffs.size else 0.0,
                float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
                float(np.max(np.abs(delta))) if delta.size else 0.0,
                float(np.mean(sigmas)) if sigmas.size else 0.0,
                float(np.max(sigmas)) if sigmas.size else 0.0,
                sub_result.iterations,
                sub_result.diagnostics.convergence_reason,
                sub_result.diagnostics.rank,
                sub_result.diagnostics.condition_number,
            )
        )
    return tuple(rows)
