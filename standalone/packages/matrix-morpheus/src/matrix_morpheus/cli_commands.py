"""Morpheus command dispatch for the installed MATRIX suite."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .cli_support import (
    UNHANDLED,
    _append_manifest_output,
    _compile_semiexperimental_latex,
    _ensemble_output_paths,
    _job_default,
    _merge_unique,
    _parse_fixed_parameters,
    _parse_parameter_classes,
    _parse_qm_predicates,
    _primitive_class_budget,
    _prune_semiexp_delivery_artifacts,
    _semiexp_aligned_displacements,
    _semiexp_components_for_budget,
    _semiexp_fit_comparison_contract,
    _semiexp_synthon_auto_score,
    _sensitivity_min_fit_count,
    _sensitivity_safe_apply_gate,
    _write_sensitivity_gate_summary,
)

def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser, root: Path):
    if args.command == "semiexp":
        from matrix_morpheus import (
            DEFAULT_SEMIEXP_OBSERVABLE,
            DEFAULT_SEMIEXP_ROBUST_LOSS,
            DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS,
            HYDROGEN_PARAMETER_CONSTRAINT,
            ParameterClassConstraint,
            QMParameterPredicate,
            SemiexperimentalFinalValidationOptions,
            SemiexperimentalFitRequest,
            SYNTHON_CLASS_LEVELS,
            advise_semiexperimental_gic_sensitivity,
            derive_primitive_class_plan,
            fit_semiexperimental_geometry,
            fit_ground_state_r0_geometry,
            initial_geometry_predicates,
            is_msr_file,
            kraitchman_seed_predicates,
            parse_primitive_class_spec,
            primitive_class_decision_lines,
            prepare_semiexperimental_xyzin,
            preview_semiexperimental_conditioning,
            preview_semiexperimental_gics,
            read_geometry_input,
            read_morpheus_input_config,
            read_msr_input,
            read_observations,
            read_semiexperimental_job,
            run_semiexperimental_final_validation,
            synthon_primitive_class_specs,
            write_morpheus_section_from_result,
            write_semiexperimental_html_report,
            write_semiexperimental_standalone_latex,
        )

        legacy_msr_job = bool(args.job and is_msr_file(args.job))
        legacy_input = read_msr_input(args.job) if legacy_msr_job else None
        job = None if legacy_msr_job or not args.job else read_semiexperimental_job(args.job)
        xyzin_config = read_morpheus_input_config(args.xyzin) if args.xyzin is not None else None
        geometry_path = args.xyz or (job.path if job is not None else None) or args.xyzin
        observations_inline = job.observations_inline if job is not None else ()
        observations_path = (
            args.observations
            or (None if observations_inline else (job.observations if job is not None else None))
            or args.xyzin
        )
        if legacy_msr_job:
            geometry_path = args.xyz or args.job
            observations_path = args.observations or args.job
            observations_inline = ()
        if geometry_path is None:
            raise ValueError("semiexp needs --geometry, --job or --xyzin")
        if observations_path is None and not observations_inline:
            raise ValueError(
                "semiexp needs --observations, inline [[isotopologues]], "
                "a [files].observations entry in --job, or --xyzin"
            )
        preprocess = prepare_semiexperimental_xyzin(
            Path(geometry_path),
            observations_source=Path(observations_path) if observations_path is not None else None,
            observations_inline=observations_inline,
            xyzin_path=args.xyzin,
            workdir=args.outdir,
        )
        geometry_path = preprocess.xyzin
        observations = read_observations(preprocess.xyzin)
        print(f"semiexp_xyzin: {preprocess.xyzin}")
        if preprocess.created_or_updated_geometry:
            print("semiexp_xyzin_geometry: updated")
        if preprocess.updated_isotopologues:
            print("semiexp_xyzin_isotopologues: updated")

        fixed = _merge_unique(
            preprocess.source_fixed_parameters, job.fixed_parameters if job else ()
        )
        if xyzin_config is not None:
            fixed = _merge_unique(fixed, xyzin_config.fixed_parameters)
        fixed = _merge_unique(fixed, _parse_fixed_parameters(args.fixed))
        if args.fix_hydrogens:
            fixed = _merge_unique(fixed, (HYDROGEN_PARAMETER_CONSTRAINT,))
        xyzin_observable = xyzin_config.observable if xyzin_config is not None else None
        observable = _job_default(
            args.observable,
            DEFAULT_SEMIEXP_OBSERVABLE,
            job.observable if job else xyzin_observable,
        )
        xyzin_coordinate_model = xyzin_config.coordinate_model if xyzin_config is not None else None
        coordinate_model = _job_default(
            args.coordinate_model,
            "gic",
            job.coordinate_model if job else xyzin_coordinate_model,
        )
        xyzin_rotational_components = xyzin_config.components if xyzin_config is not None else None
        rotational_components = _job_default(
            args.rotational_components,
            DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS,
            job.rotational_components if job else xyzin_rotational_components,
        )
        qm_predicates = _merge_unique(
            job.qm_predicates if job else (),
            _parse_qm_predicates(args.qm_predicate, QMParameterPredicate),
        )
        if (
            xyzin_config is not None
            and xyzin_config.initial_geometry_predicates.enabled
            and not args.qm_predicate
        ):
            geometry_input_for_predicates = read_geometry_input(Path(geometry_path))
            spec = xyzin_config.initial_geometry_predicates
            generated_predicates = initial_geometry_predicates(
                tuple(geometry_input_for_predicates.atoms),
                geometry_input_for_predicates.coordinates_angstrom,
                distance_sigma_angstrom=spec.distance_sigma_angstrom,
                angle_sigma_degree=spec.angle_sigma_degree,
                dihedral_sigma_degree=spec.dihedral_sigma_degree,
                scope=spec.scope,
            )
            qm_predicates = _merge_unique(qm_predicates, generated_predicates)
            print(
                "xyzin_initial_geometry_predicates: "
                f"count={len(generated_predicates)} "
                f"sigma_R={spec.distance_sigma_angstrom:g} "
                f"sigma_A={spec.angle_sigma_degree:g} "
                f"sigma_D={spec.dihedral_sigma_degree:g}"
            )
        if args.kraitchman_predicates:
            geometry_input_for_kraitchman = read_geometry_input(Path(geometry_path))
            kraitchman_predicates = kraitchman_seed_predicates(
                tuple(geometry_input_for_kraitchman.atoms),
                geometry_input_for_kraitchman.coordinates_angstrom,
                observations,
                sigma_distance_angstrom=args.kraitchman_distance_sigma,
                sigma_angle_degree=args.kraitchman_angle_sigma,
                require_all_atoms_seeded=not args.kraitchman_partial_predicates,
            )
            qm_predicates = _merge_unique(qm_predicates, kraitchman_predicates)
            print(
                "kraitchman_predicates: "
                f"count={len(kraitchman_predicates)} "
                f"sigma_R={args.kraitchman_distance_sigma:g} "
                f"sigma_A={args.kraitchman_angle_sigma:g}"
            )
        parameter_classes = _merge_unique(
            job.parameter_classes if job else (),
            _parse_parameter_classes(args.parameter_class, ParameterClassConstraint),
        )
        primitive_classes = tuple(
            parse_primitive_class_spec(item) for item in getattr(args, "primitive_class", ())
        )
        if xyzin_config is not None:
            primitive_classes = _merge_unique(primitive_classes, xyzin_config.primitive_classes)
            synthon_spec = xyzin_config.synthon_primitive_classes
            if synthon_spec.enabled:
                synthon_budget = _primitive_class_budget(
                    xyzin_config.primitive_class_budget or args.primitive_class_budget,
                    observations=observations,
                    rotational_components=rotational_components,
                )
                geometry_input_for_classes = read_geometry_input(Path(geometry_path))
                if synthon_spec.level == "auto":
                    preview_for_auto = preview_semiexperimental_gics(
                        Path(geometry_path),
                        observations,
                    )
                    primitive_class_min_for_auto = (
                        xyzin_config.primitive_class_min
                        if xyzin_config.primitive_class_min is not None
                        else args.primitive_class_min
                    )
                    primitive_class_cross_for_auto = (
                        xyzin_config.primitive_class_cross_max
                        if xyzin_config.primitive_class_cross_max is not None
                        else args.primitive_class_cross_max
                    )
                    candidate_records = []
                    for candidate_level in ("coarse", "medium", "fine"):
                        candidate_generated = synthon_primitive_class_specs(
                            tuple(geometry_input_for_classes.atoms),
                            geometry_input_for_classes.coordinates_angstrom,
                            level=candidate_level,
                            include_bonds=synthon_spec.include_bonds,
                            include_angles=synthon_spec.include_angles,
                            min_group_size=synthon_spec.min_group_size,
                            bond_order_bins=synthon_spec.bond_order_bins,
                        )
                        candidate_classes = _merge_unique(
                            primitive_classes,
                            candidate_generated,
                        )
                        candidate_plan = derive_primitive_class_plan(
                            preview_for_auto.gic_labels,
                            candidate_classes,
                            min_fraction=primitive_class_min_for_auto,
                            cross_fraction_max=primitive_class_cross_for_auto,
                            max_classes=synthon_budget,
                        )
                        candidate_fixed = _merge_unique(fixed, candidate_plan.fixed_patterns)
                        candidate_parameter_classes = _merge_unique(
                            parameter_classes,
                            candidate_plan.parameter_classes,
                        )
                        candidate_request = SemiexperimentalFitRequest(
                            initial_geometry=geometry_path,
                            observations=observations,
                            fixed_parameters=candidate_fixed,
                            observable=observable,
                            rotational_components=rotational_components,
                            qm_predicates=qm_predicates,
                            parameter_classes=candidate_parameter_classes,
                            coordinate_model=coordinate_model,
                            robust_loss=(
                                job.robust_loss
                                if job and args.robust_loss == DEFAULT_SEMIEXP_ROBUST_LOSS
                                else args.robust_loss
                            ),
                            robust_scale=args.robust_scale,
                            leave_one_out=False,
                            excluded_rotational_constants=tuple(
                                args.exclude_rotational_constant
                            ),
                        )
                        candidate_outdir = (
                            args.outdir / "_synthon_auto_candidates" / candidate_level
                        )
                        candidate_result = fit_semiexperimental_geometry(
                            candidate_request,
                            max_iter=(
                                args.max_iter
                                if args.max_iter is not None
                                else (job.max_iter if job else None)
                            ),
                            step=(
                                args.step
                                if args.step != 1.0e-4
                                else (job.step if job and job.step is not None else 1.0e-4)
                            ),
                            damping=(
                                args.damping
                                if args.damping != 1.0e-8
                                else (job.damping if job and job.damping is not None else 1.0e-8)
                            ),
                            max_step=(
                                args.max_step
                                if args.max_step != 0.25
                                else (job.max_step if job and job.max_step is not None else 0.25)
                            ),
                            prune_condition=(
                                args.prune_condition
                                if args.prune_condition != 0.0
                                else (
                                    job.prune_condition
                                    if job and job.prune_condition is not None
                                    else 0.0
                                )
                            ),
                            outdir=candidate_outdir,
                        )
                        score = _semiexp_synthon_auto_score(candidate_result)
                        candidate_records.append(
                            (
                                score,
                                candidate_level,
                                candidate_generated,
                                candidate_plan,
                                candidate_result,
                            )
                        )
                        print(
                            "synthon_auto_candidate: "
                            f"level={candidate_level} "
                            f"classes={len(candidate_plan.parameter_classes)} "
                            f"active={candidate_result.diagnostics.n_optimized_parameters} "
                            f"rank={candidate_result.diagnostics.rank} "
                            f"cond={candidate_result.diagnostics.condition_number:.6g} "
                            f"max_sigma_XY={score[3]:.6g} "
                            f"max_sigma_XH={score[4]:.6g} "
                            f"max_sigma_CH={score[5]:.6g} "
                            f"max_sigma_A={score[6]:.6g} "
                            f"violations={score[0]}"
                        )
                    selected = min(candidate_records, key=lambda item: item[0])
                    synthon_level = selected[1]
                    generated_classes = selected[2]
                    print(
                        "synthon_auto_selected: "
                        f"level={synthon_level} "
                        f"classes={len(selected[3].parameter_classes)} "
                        f"score={selected[0]}"
                    )
                else:
                    synthon_level = synthon_spec.level
                    if synthon_level not in SYNTHON_CLASS_LEVELS:
                        raise ValueError(f"Unknown SYNTHON_LEVEL: {synthon_level}")
                    generated_classes = synthon_primitive_class_specs(
                        tuple(geometry_input_for_classes.atoms),
                        geometry_input_for_classes.coordinates_angstrom,
                        level=synthon_level,
                        include_bonds=synthon_spec.include_bonds,
                        include_angles=synthon_spec.include_angles,
                        min_group_size=synthon_spec.min_group_size,
                        bond_order_bins=synthon_spec.bond_order_bins,
                    )
                primitive_classes = _merge_unique(primitive_classes, generated_classes)
                print(
                    "xyzin_synthon_primitive_classes: "
                    f"count={len(generated_classes)} "
                    f"level={synthon_level} "
                    f"include_bonds={synthon_spec.include_bonds} "
                    f"include_angles={synthon_spec.include_angles}"
                )
        backend = _job_default(args.backend, "python", job.backend if job else None)
        max_iter = args.max_iter if args.max_iter is not None else (job.max_iter if job else None)
        step = _job_default(args.step, 1.0e-4, job.step if job else None)
        damping = _job_default(args.damping, 1.0e-8, job.damping if job else None)
        max_step = _job_default(args.max_step, 0.25, job.max_step if job else None)
        prune_condition = _job_default(
            args.prune_condition,
            0.0,
            job.prune_condition if job else None,
        )
        robust_loss = _job_default(
            args.robust_loss,
            DEFAULT_SEMIEXP_ROBUST_LOSS,
            job.robust_loss if job else None,
        )
        robust_scale = _job_default(args.robust_scale, 0.0, job.robust_scale if job else None)
        leave_one_out = bool(args.leave_one_out or (job.leave_one_out if job else False))
        checkpoint = (
            args.checkpoint if args.checkpoint is not None else (job.checkpoint if job else None)
        )
        restart = args.restart if args.restart is not None else (job.restart if job else None)
        legacy_robust_profile = bool(
            legacy_input
            and legacy_input.controls.condition_active
            and not args.no_auto_stabilize
        )
        if legacy_input and legacy_input.controls.outlier_active:
            if args.robust_loss == DEFAULT_SEMIEXP_ROBUST_LOSS:
                robust_loss = "huber"
            if args.robust_scale == 0.0:
                robust_scale = 0.1
        if legacy_robust_profile:
            if args.damping == 1.0e-8:
                damping = 1.0e-3
            if args.max_step == 0.25:
                max_step = 5.0e-3
            if args.max_iter is None:
                max_iter = 500
            print(
                "morpheus_legacy_automatic_profile: "
                f"outlier={legacy_input.controls.outlier or 'inactive'} "
                f"condition={legacy_input.controls.condition or 'inactive'} "
                f"propagation={legacy_input.controls.propagation or 'default'} "
                f"robust_loss={robust_loss} robust_scale={robust_scale:g} "
                f"damping={damping:g} max_step={max_step:g}"
            )
        if primitive_classes:
            if coordinate_model != "gic":
                raise ValueError("--primitive-class is only supported with --coordinate-model gic")
            primitive_class_budget_raw = args.primitive_class_budget
            if (
                xyzin_config is not None
                and xyzin_config.primitive_class_budget is not None
                and args.primitive_class_budget == "auto"
            ):
                primitive_class_budget_raw = xyzin_config.primitive_class_budget
            primitive_class_min = (
                xyzin_config.primitive_class_min
                if xyzin_config is not None
                and xyzin_config.primitive_class_min is not None
                and args.primitive_class_min == 0.70
                else args.primitive_class_min
            )
            primitive_class_cross_max = (
                xyzin_config.primitive_class_cross_max
                if xyzin_config is not None
                and xyzin_config.primitive_class_cross_max is not None
                and args.primitive_class_cross_max == 0.20
                else args.primitive_class_cross_max
            )
            class_budget = _primitive_class_budget(
                primitive_class_budget_raw,
                observations=observations,
                rotational_components=rotational_components,
            )
            preview = preview_semiexperimental_gics(Path(geometry_path), observations)
            class_plan = derive_primitive_class_plan(
                preview.gic_labels,
                primitive_classes,
                min_fraction=primitive_class_min,
                cross_fraction_max=primitive_class_cross_max,
                max_classes=class_budget,
            )
            fixed = _merge_unique(fixed, class_plan.fixed_patterns)
            parameter_classes = _merge_unique(parameter_classes, class_plan.parameter_classes)
            print(
                "primitive_class_plan: "
                f"classes={len(class_plan.parameter_classes)} "
                f"fixed={len(class_plan.fixed_patterns)} "
                f"rejected={len(class_plan.rejected_labels)}"
            )
            for item in class_plan.parameter_classes:
                print(
                    f"primitive_class: {item.name} "
                    f"gics={len(item.patterns)} patterns={'|'.join(item.patterns)}"
                )
            for line in primitive_class_decision_lines(class_plan):
                print(line)
        sensitivity_advisor_enabled = bool(args.sensitivity_advisor or legacy_robust_profile)
        sensitivity_apply_enabled = bool(args.apply_sensitivity_advisor or legacy_robust_profile)
        sensitivity_force_enabled = bool(args.force_sensitivity_advisor or legacy_robust_profile)
        sensitivity_fit_threshold = (
            1.1
            if legacy_robust_profile and args.sensitivity_fit_threshold == 0.15
            else args.sensitivity_fit_threshold
        )
        sensitivity_min_fit = (
            "none"
            if legacy_robust_profile and args.sensitivity_min_fit == "auto"
            else args.sensitivity_min_fit
        )
        advisor = None
        if sensitivity_advisor_enabled:
            if coordinate_model != "gic":
                raise ValueError(
                    "--sensitivity-advisor is only supported with --coordinate-model gic"
                )
            advisor_request = SemiexperimentalFitRequest(
                initial_geometry=geometry_path,
                observations=observations,
                fixed_parameters=fixed,
                observable=observable,
                rotational_components=rotational_components,
                qm_predicates=qm_predicates,
                parameter_classes=parameter_classes,
                coordinate_model=coordinate_model,
                robust_loss=robust_loss,
                robust_scale=robust_scale,
                leave_one_out=leave_one_out,
                excluded_rotational_constants=tuple(args.exclude_rotational_constant),
            )
            advisor = advise_semiexperimental_gic_sensitivity(
                advisor_request,
                step=step,
                fit_relative_threshold=sensitivity_fit_threshold,
                fixed_relative_threshold=args.sensitivity_fixed_threshold,
                min_fit_count=_sensitivity_min_fit_count(sensitivity_min_fit),
                distance_sigma_angstrom=args.sensitivity_distance_sigma,
                angle_sigma_degree=args.sensitivity_angle_sigma,
                torsion_sigma_degree=args.sensitivity_torsion_sigma,
                soft_predicate_scale=args.sensitivity_soft_predicate_scale,
                null_predicate_scale=args.sensitivity_null_predicate_scale,
                fit_regularization_scale=args.sensitivity_fit_regularization_scale,
            )
            args.outdir.mkdir(parents=True, exist_ok=True)
            advisor_path = args.outdir / "semiexp_sensitivity_advisor.csv"
            advisor_path.write_text(advisor.csv, encoding="utf-8")
            advisor_applied = False
            if sensitivity_apply_enabled:
                candidate_fixed = _merge_unique(fixed, advisor.fixed_patterns)
                candidate_qm_predicates = _merge_unique(qm_predicates, advisor.predicates)
                if sensitivity_force_enabled:
                    fixed = candidate_fixed
                    qm_predicates = candidate_qm_predicates
                    advisor_applied = True
                    _write_sensitivity_gate_summary(
                        args.outdir / "semiexp_sensitivity_gate.json",
                        {"accepted": True, "reason": "forced"},
                    )
                else:
                    gate = _sensitivity_safe_apply_gate(
                        base_request=advisor_request,
                        candidate_request=SemiexperimentalFitRequest(
                            initial_geometry=geometry_path,
                            observations=observations,
                            fixed_parameters=candidate_fixed,
                            observable=observable,
                            rotational_components=rotational_components,
                            qm_predicates=candidate_qm_predicates,
                            parameter_classes=parameter_classes,
                            coordinate_model=coordinate_model,
                            robust_loss=robust_loss,
                            robust_scale=robust_scale,
                            leave_one_out=leave_one_out,
                            excluded_rotational_constants=tuple(
                                args.exclude_rotational_constant
                            ),
                        ),
                        fit_semiexperimental_geometry=fit_semiexperimental_geometry,
                        outdir=args.outdir / "_sensitivity_gate",
                        max_iter=max_iter,
                        step=step,
                        damping=damping,
                        max_step=max_step,
                        prune_condition=prune_condition,
                        rot_rel_tol=args.sensitivity_gate_rot_rel_tol,
                        rot_abs_tol=args.sensitivity_gate_rot_abs_tol,
                        condition_factor=args.sensitivity_gate_condition_factor,
                        max_bond_delta=args.sensitivity_gate_max_bond_delta,
                        max_angle_delta=args.sensitivity_gate_max_angle_delta,
                    )
                    _write_sensitivity_gate_summary(
                        args.outdir / "semiexp_sensitivity_gate.json",
                        gate,
                    )
                    if gate["accepted"]:
                        fixed = candidate_fixed
                        qm_predicates = candidate_qm_predicates
                        advisor_applied = True
            print(
                "morpheus_sensitivity_advisor: "
                f"fit={advisor.fit_count} "
                f"predicate={advisor.predicate_count} "
                f"fixed={advisor.fixed_count} "
                f"applied={advisor_applied} "
                f"csv={advisor_path}"
            )
        if (
            coordinate_model == "gic"
            and not args.no_auto_stabilize
            and not qm_predicates
            and not parameter_classes
        ):
            preview = preview_semiexperimental_gics(Path(geometry_path), observations)
            if preview.suggested_classes:
                parameter_classes = _merge_unique(parameter_classes, preview.suggested_classes)
                print(f"morpheus_auto_advisor: parameter_classes={len(preview.suggested_classes)}")
                for item in preview.suggested_classes:
                    print(
                        f"morpheus_auto_class: {item.name} "
                        f"mode={item.mode} patterns={'|'.join(item.patterns)}"
                    )
            row_budget = len(observations) * len(
                _semiexp_components_for_budget(rotational_components)
            )
            preview_request = SemiexperimentalFitRequest(
                initial_geometry=geometry_path,
                observations=observations,
                fixed_parameters=fixed,
                observable=observable,
                rotational_components=rotational_components,
                coordinate_model=coordinate_model,
            )
            effective_parameters = preview_semiexperimental_conditioning(
                preview_request
            ).n_effective_parameters
            if (
                effective_parameters > row_budget
                and HYDROGEN_PARAMETER_CONSTRAINT not in fixed
            ):
                fixed = _merge_unique(fixed, (HYDROGEN_PARAMETER_CONSTRAINT,))
                print(
                    "morpheus_auto_stabilize: "
                    f"free_gic_parameters={effective_parameters} "
                    f"fit_rows={row_budget}; action=fix_hydrogens"
                )
        request = SemiexperimentalFitRequest(
            initial_geometry=geometry_path,
            observations=observations,
            fixed_parameters=fixed,
            observable=observable,
            rotational_components=rotational_components,
            qm_predicates=qm_predicates,
            parameter_classes=parameter_classes,
            coordinate_model=coordinate_model,
            robust_loss=robust_loss,
            robust_scale=robust_scale,
            leave_one_out=leave_one_out,
            excluded_rotational_constants=tuple(args.exclude_rotational_constant),
        )
        fit_options = {
            "max_iter": max_iter,
            "step": step,
            "damping": damping,
            "max_step": max_step,
            "prune_condition": prune_condition,
            "checkpoint": checkpoint,
            "restart": restart,
            "outdir": args.outdir,
        }
        free_request = None
        regularization_predicates = tuple(
            predicate
            for predicate in request.qm_predicates
            if predicate.source == "morpheus_sensitivity_advisor_fit_regularization"
        )
        if args.compare_free_fit:
            if args.r0_preflight:
                raise ValueError("--compare-free-fit is incompatible with --r0-preflight")
            from dataclasses import replace as dataclass_replace

            free_request = (
                dataclass_replace(
                    request,
                    qm_predicates=tuple(
                        predicate
                        for predicate in request.qm_predicates
                        if predicate.source
                        != "morpheus_sensitivity_advisor_fit_regularization"
                    ),
                )
                if regularization_predicates
                else request
            )
        r0_report_result = None
        if args.r0_preflight:
            from dataclasses import replace as dataclass_replace

            preflight = fit_ground_state_r0_geometry(request, **fit_options)
            result = preflight.fit
            request = dataclass_replace(request, observations=preflight.observations)
            print("morpheus_fit_kind: R0_PRELIMINARY")
            for warning in preflight.warnings:
                print(f"morpheus_r0_warning: {warning}")
        else:
            if args.include_r0_report:
                r0_options = dict(fit_options)
                r0_options["checkpoint"] = None
                r0_options["restart"] = None
                r0_options["outdir"] = args.outdir / "_r0_report"
                r0_preflight = fit_ground_state_r0_geometry(request, **r0_options)
                r0_report_result = r0_preflight.fit
                print("morpheus_structural_path: INPUT -> R0 -> RS(KRAITCHMAN) -> RE(SE)")
                for warning in r0_preflight.warnings:
                    print(f"morpheus_r0_warning: {warning}")
            result = fit_semiexperimental_geometry(request, **fit_options)
        free_result = None
        if free_request is not None:
            if free_request is request:
                free_result = result
            else:
                free_options = dict(fit_options)
                free_options["checkpoint"] = None
                free_options["restart"] = None
                free_options["outdir"] = args.outdir / "_free_fit_comparison"
                free_result = fit_semiexperimental_geometry(free_request, **free_options)
        displacement_limit = args.max_atom_displacement
        if displacement_limit is None and legacy_robust_profile:
            displacement_limit = 3.0e-3
        safety: dict[str, object] | None = None
        if displacement_limit is not None:
            if displacement_limit <= 0.0:
                raise ValueError("--max-atom-displacement must be positive")
            max_displacement, rms_displacement = _semiexp_aligned_displacements(result)
            full_rank = result.diagnostics.rank == result.diagnostics.n_optimized_parameters
            well_conditioned = math.isfinite(result.diagnostics.condition_number) and (
                result.diagnostics.condition_number <= 1.0e8
            )
            reliable = bool(
                max_displacement <= float(displacement_limit)
                and result.stationary_point == "minimum"
                and full_rank
                and well_conditioned
            )
            safety = {
                "accepted": max_displacement <= float(displacement_limit),
                "reliable": reliable,
                "max_atom_displacement_A": max_displacement,
                "rms_atom_displacement_A": rms_displacement,
                "limit_A": float(displacement_limit),
                "stationary_point": result.stationary_point,
                "full_rank": full_rank,
                "condition_number": result.diagnostics.condition_number,
                "condition_limit": 1.0e8,
            }
            _write_sensitivity_gate_summary(
                args.outdir / "semiexp_geometry_safety.json",
                safety,
            )
            print(
                "morpheus_geometry_safety: "
                f"max_displacement_A={max_displacement:.9g} "
                f"rms_displacement_A={rms_displacement:.9g} "
                f"limit_A={float(displacement_limit):.9g} "
                f"accepted={safety['accepted']}"
            )
            if not safety["accepted"]:
                raise ValueError(
                    "MORPHEUS rejected the fitted structure: maximum aligned atom "
                    f"displacement {max_displacement:.6g} A exceeds the "
                    f"{float(displacement_limit):.6g} A safety limit"
                )
        fit_comparison = None
        if free_result is not None:
            fit_comparison = _semiexp_fit_comparison_contract(
                free_result=free_result,
                constrained_result=result,
                displacement_limit=float(displacement_limit),
                regularization_predicates=regularization_predicates,
                regularization_scale=float(args.sensitivity_fit_regularization_scale),
                excluded_rotational_constants=tuple(args.exclude_rotational_constant),
                advisor_rows=tuple(advisor.rows) if advisor is not None else (),
            )
            comparison_path = args.outdir / "semiexp_fit_comparison.json"
            _write_sensitivity_gate_summary(comparison_path, fit_comparison)
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json", "fit_comparison", comparison_path
            )
        report_path = write_semiexperimental_html_report(
            args.outdir / "semiexp_report.html",
            result,
            request,
            r0_result=r0_report_result,
            fit_comparison=fit_comparison,
        )
        tables_path = write_semiexperimental_standalone_latex(
            args.outdir / "semiexp_results.tex",
            result,
            request=request,
            safety=safety,
            r0_result=r0_report_result,
            fit_comparison=fit_comparison,
        )
        latex_pdf_path = _compile_semiexperimental_latex(tables_path)
        _append_manifest_output(args.outdir / "semiexp_manifest.json", "html_report", report_path)
        _append_manifest_output(args.outdir / "semiexp_manifest.json", "latex_tables", tables_path)
        _append_manifest_output(
            args.outdir / "semiexp_manifest.json", "latex_pdf", latex_pdf_path
        )
        delivery_input_path: Path | None = None
        delivery_geometry_input_path: Path | None = None
        if legacy_msr_job and args.job is not None:
            import shutil

            delivery_input_path = args.outdir / Path(args.job).name
            if delivery_input_path.resolve() != Path(args.job).resolve():
                shutil.copy2(args.job, delivery_input_path)
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json", "input_msr", delivery_input_path
            )
            if args.xyz is not None:
                delivery_geometry_input_path = args.outdir / Path(args.xyz).name
                if delivery_geometry_input_path.resolve() != Path(args.xyz).resolve():
                    shutil.copy2(args.xyz, delivery_geometry_input_path)
                _append_manifest_output(
                    args.outdir / "semiexp_manifest.json",
                    "input_geometry",
                    delivery_geometry_input_path,
                )
        if args.final_validation:
            validation_scales = (
                tuple(float(item) for item in args.validation_sigma_scale)
                if args.validation_sigma_scale
                else (0.5, 2.0)
            )
            if args.validation_no_predicate_scan:
                validation_scales = ()
            validation = run_semiexperimental_final_validation(
                request,
                result,
                args.outdir / "semiexp_final_validation",
                options=SemiexperimentalFinalValidationOptions(
                    coordinate_check=not args.validation_no_coordinate_check,
                    huber_check=not args.validation_no_huber_check,
                    predicate_scan_scales=validation_scales,
                    leave_predicate_groups=not args.validation_no_leave_predicate_groups,
                    max_predicate_groups=args.validation_max_predicate_groups,
                    multistart=args.validation_multistart,
                    multistart_sigma_angstrom=args.validation_multistart_sigma,
                    random_seed=args.validation_random_seed,
                ),
                max_iter=max_iter,
                step=step,
                damping=damping,
                max_step=max_step,
                prune_condition=prune_condition,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_summary",
                validation.summary_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_runs",
                validation.runs_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "predicate_audit",
                validation.predicate_audit_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_issues",
                validation.issues_path,
            )
            print(f"final_validation: {validation.summary_path}")
            print(f"final_validation_runs: {validation.runs_path}")
            print(f"final_validation_issues: {len(validation.issues)}")
        if args.xyzin is not None and not args.no_write_section:
            write_morpheus_section_from_result(
                preprocess.xyzin,
                result,
                outdir=args.outdir,
                backend=backend,
                source_path=args.job or args.xyz or observations_path,
                html_report_path=report_path,
                latex_tables_path=tables_path,
            )
            print(f"updated_morpheus_section: {preprocess.xyzin}")
        if safety and safety.get("reliable") and not args.keep_all_artifacts:
            delivery_files = {
                "semiexp_geometry.xyz",
                "semiexp_report.html",
                "semiexp_results.tex",
                "semiexp_results.pdf",
                "semiexp_manifest.json",
                "semiexp_geometry_safety.json",
            }
            extra_outputs: dict[str, Path] = {}
            if fit_comparison is not None:
                delivery_files.add("semiexp_fit_comparison.json")
                extra_outputs["fit_comparison"] = args.outdir / "semiexp_fit_comparison.json"
            if delivery_input_path is not None:
                delivery_files.add(delivery_input_path.name)
                extra_outputs["input_msr"] = delivery_input_path
            if delivery_geometry_input_path is not None:
                delivery_files.add(delivery_geometry_input_path.name)
                extra_outputs["input_geometry"] = delivery_geometry_input_path
            retained = _prune_semiexp_delivery_artifacts(
                args.outdir,
                delivery_files,
                extra_outputs=extra_outputs,
            )
            print(f"morpheus_delivery_cleanup: retained={','.join(retained)}")
        print(f"manifest: {result.manifest}")
        print(f"report: {report_path}")
        rms_label = (
            "rms_MHz"
            if result.diagnostics.observable == "rotational_constants"
            else "rms_observable"
        )
        print(f"{rms_label}: {result.rms_MHz:.8g}")
        rot_diffs = [row.difference_MHz for row in result.rotational_constants]
        rotational_rms = (
            math.sqrt(sum(diff * diff for diff in rot_diffs) / len(rot_diffs)) if rot_diffs else 0.0
        )
        rotational_mse = (
            sum(diff * diff for diff in rot_diffs) / len(rot_diffs) if rot_diffs else 0.0
        )
        print(f"rotational_rms_MHz: {rotational_rms:.8g}")
        print(f"rotational_mean_square_MHz2: {rotational_mse:.8g}")
        print(f"rotational_mean_square_1e3_MHz2: {1000.0 * rotational_mse:.8g}")
        print(f"iterations: {result.iterations}")
        print(f"stationary_point: {result.stationary_point}")
        print(f"convergence: {result.diagnostics.convergence_reason}")
        print(f"rank: {result.diagnostics.rank}")
        print(f"condition_number: {result.diagnostics.condition_number:.8g}")
        print(f"observable: {result.diagnostics.observable}")
        print(f"components: {','.join(result.diagnostics.components)}")
        print(f"backend: {backend}")
        print(f"coordinate_model: {result.diagnostics.coordinate_model}")
        return 0
    if args.command == "semiexp-ensemble":
        from matrix_core.manifest import build_run_manifest
        from matrix_morpheus import fit_ensemble_job

        result = fit_ensemble_job(args.job, outdir=args.outdir)
        outputs = _ensemble_output_paths(args.outdir)
        build_run_manifest(
            workflow="semiexp_ensemble",
            status=result.acceptance.status,
            run_dir=args.outdir,
            inputs={"job": args.job},
            outputs=outputs,
            parameters={
                "classes": len(result.classes),
                "molecules": len(result.molecule_blocks),
                "rank": result.rank,
                "scaled_condition_number": result.condition_number,
                "weighted_rms_before": result.weighted_rms_before,
                "weighted_rms_after": result.weighted_rms_after,
                "accepted": result.acceptance.accepted,
            },
            backend={"solver": "python", "model": "linearized shared class corrections"},
            messages=list(result.acceptance.reasons) + list(result.acceptance.review_items),
        ).write(args.outdir / "run_manifest.json")
        print(f"report: {args.outdir / 'ensemble_class_corrections.txt'}")
        print(f"manifest: {args.outdir / 'run_manifest.json'}")
        print(f"classes: {len(result.classes)}")
        print(f"molecules: {len(result.molecule_blocks)}")
        print(f"rank: {result.rank}")
        print(f"scaled_condition_number: {result.condition_number:.8g}")
        print(f"acceptance_status: {result.acceptance.status}")
        if result.acceptance.reasons:
            print("acceptance_failures: " + " | ".join(result.acceptance.reasons))
        if result.acceptance.review_items:
            print("acceptance_review: " + " | ".join(result.acceptance.review_items))
        print(f"weighted_rms_before: {result.weighted_rms_before:.8g}")
        print(f"weighted_rms_after: {result.weighted_rms_after:.8g}")
        for item in result.classes:
            print(
                f"class:{item.name}: correction={result.corrections[item.name]:.10g} "
                f"sigma={result.sigma[item.name]:.4g}"
            )
        return 0
    if args.command == "semiexp-ensemble-paper":
        from matrix_morpheus import write_ensemble_jpcl_artifacts

        artifacts = write_ensemble_jpcl_artifacts(
            args.job,
            args.paper_dir,
            outdir=args.outdir,
            soft_prior_sigma=args.soft_prior_sigma,
        )
        print(f"paper_dir: {args.paper_dir}")
        if args.outdir is not None:
            print(f"analysis_dir: {args.outdir}")
        for name, path in sorted(artifacts.items()):
            print(f"{name}: {path}")
        return 0
    if args.command == "semiexp-ensemble-prior-scan":
        from matrix_morpheus import run_ensemble_prior_scan

        kwargs = {}
        if args.sigma:
            kwargs["sigmas"] = tuple(args.sigma)
        rows = run_ensemble_prior_scan(args.job, args.outdir, **kwargs)
        print(f"rows: {len(rows)}")
        print(f"csv: {args.outdir / 'prior_sigma_scan.csv'}")
        print(f"json: {args.outdir / 'prior_sigma_scan.json'}")
        return 0
    if args.command == "semiexp-ensemble-synthon-scan":
        from matrix_morpheus import run_ensemble_synthon_threshold_scan

        kwargs = {}
        if args.threshold:
            kwargs["thresholds"] = tuple(args.threshold)
        rows = run_ensemble_synthon_threshold_scan(args.job, args.outdir, **kwargs)
        print(f"rows: {len(rows)}")
        print(f"csv: {args.outdir / 'synthon_threshold_scan.csv'}")
        print(f"json: {args.outdir / 'synthon_threshold_scan.json'}")
        return 0
    if args.command == "semiexp-benchmark":
        from matrix_morpheus import generate_paper_benchmark_artifacts

        snapshot, artifacts = generate_paper_benchmark_artifacts(
            snapshot_path=args.snapshot,
            outdir=args.outdir,
            refresh_from_outputs=not args.no_refresh,
            update_snapshot=args.update_snapshot,
        )
        print(f"cases: {len(snapshot.get('cases', {}))}")
        print(f"planar_diagnostics: {len(snapshot.get('planar_pair_diagnostics', {}))}")
        for name, path in sorted(artifacts.items()):
            print(f"{name}: {path}")
        return 0
    return UNHANDLED
