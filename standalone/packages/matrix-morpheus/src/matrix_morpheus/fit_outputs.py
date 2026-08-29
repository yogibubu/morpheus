from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
from io import StringIO

import numpy as np

from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_chem.rotational import rotational_constants_MHz
from matrix_chem.structure import Structure
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_core import ScientificValidationError
from matrix_link import (
    cartesian_from_internal_jacobian,
)
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive, eval_primitives

from .contracts import (
    IsotopologueObservation,
    ParameterClassConstraint,
    SemiexperimentalFitRequest,
)
from .constraints import (
    _gic_expression_definitions_from_patterns,
    _parse_gic_expression_constraint_pattern,
    _parse_gic_expression_definition_pattern,
    _primitive_constraint_key,
    _primitives_from_fixed_pattern,
)
from .kraitchman import (
    KraitchmanComparison,
)
from .statistics import (
    SemiexperimentalWeightDiagnostic,
    leverage_values as _leverage_values,
)
from .models import (
    GICExpressionDefinition,
    MeasurementModel,
    SemiexperimentalDiagnosticWarning,
    SemiexperimentalFitDiagnostics,
    SemiexperimentalGeometryParameter,
    SemiexperimentalIterationTrace,
    SemiexperimentalLeaveOneOutRow,
    SemiexperimentalParameter,
    SemiexperimentalResidual,
    SemiexperimentalRotationalConstantComparison,
    TopologyLock,
)
from .solver import (
    DAMPING_MAX,
)


from .coordinate_model import (
    _atomic_number,
    _class_matches,
    _experimental_isotopologue_row_groups,
)


from .measurement_model import (
    _primitive_text,
    _validate_locked_topology,
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


def parameters_csv(parameters: tuple[SemiexperimentalParameter, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["name", "value", "sigma", "active", "parameter_class"])
    for p in parameters:
        writer.writerow(
            [p.name, f"{p.value:.12g}", f"{p.sigma:.12g}", int(p.active), p.parameter_class]
        )
    return stream.getvalue()

def geometry_parameters_csv(parameters: tuple[SemiexperimentalGeometryParameter, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "kind",
            "label",
            "atoms",
            "symbols",
            "value_angstrom",
            "initial_value_angstrom",
            "delta_angstrom",
            "sigma_angstrom",
            "value_degree",
            "initial_value_degree",
            "delta_degree",
            "sigma_degree",
            "value",
            "initial_value",
            "delta",
            "sigma",
            "unit",
        ]
    )
    for item in parameters:
        writer.writerow(
            [
                item.kind,
                item.label,
                "-".join(str(idx) for idx in item.atom_indices),
                "-".join(item.atom_symbols),
                "" if item.value_angstrom is None else f"{item.value_angstrom:.12g}",
                ""
                if item.initial_value_angstrom is None
                else f"{item.initial_value_angstrom:.12g}",
                "" if item.delta_angstrom is None else f"{item.delta_angstrom:.12g}",
                "" if item.sigma_angstrom is None else f"{item.sigma_angstrom:.12g}",
                "" if item.value_degree is None else f"{item.value_degree:.12g}",
                "" if item.initial_value_degree is None else f"{item.initial_value_degree:.12g}",
                "" if item.delta_degree is None else f"{item.delta_degree:.12g}",
                "" if item.sigma_degree is None else f"{item.sigma_degree:.12g}",
                "" if item.value is None else f"{item.value:.12g}",
                "" if item.initial_value is None else f"{item.initial_value:.12g}",
                "" if item.delta is None else f"{item.delta:.12g}",
                "" if item.sigma is None else f"{item.sigma:.12g}",
                item.unit,
            ]
        )
    return stream.getvalue()

def _effective_parameter_names(
    parameters: tuple[SemiexperimentalParameter, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for parameter in parameters:
        if not parameter.active:
            continue
        name = parameter.parameter_class or parameter.name
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)

def _combined_fixed_parameters(
    explicit_fixed: tuple[str, ...],
    input_fixed: tuple[str, ...] = (),
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in (*explicit_fixed, *input_fixed):
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)

def _checkpoint_path(outdir: Path | None, checkpoint: Path | None) -> Path | None:
    if checkpoint is not None:
        return Path(checkpoint)
    if outdir is None:
        return None
    return Path(outdir) / "semiexp_checkpoint.json"

def _write_semiexp_checkpoint(
    path: Path | None,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    *,
    iteration: int,
    damping: float,
    trust_radius: float,
    labels: tuple[str, ...],
    active_mask: np.ndarray,
    robust_sqrt_weights: np.ndarray,
    robust_scale: float,
    robust_downweighted_rows: int,
    robust_downweighted_isotopologues: int,
    coordinate_model: str,
    accepted_steps: int,
    rejected_steps: int,
) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    active_labels = [
        label for label, active in zip(labels, np.asarray(active_mask, dtype=bool)) if active
    ]
    payload = {
        "schema": SEMIEXP_CHECKPOINT_SCHEMA,
        "coordinate_model": coordinate_model,
        "iteration": int(iteration),
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "damping": float(damping),
        "trust_radius": float(trust_radius),
        "atoms": list(atoms),
        "coordinates_angstrom": np.asarray(coords, dtype=float).tolist(),
        "labels": list(labels),
        "active_labels": active_labels,
        "robust_sqrt_weights": np.asarray(robust_sqrt_weights, dtype=float).tolist(),
        "robust_scale": float(robust_scale),
        "robust_downweighted_rows": int(robust_downweighted_rows),
        "robust_downweighted_isotopologues": int(robust_downweighted_isotopologues),
        "restart_policy": "GIC or symmetry-Cartesian definitions are rebuilt deterministically from these Cartesian coordinates.",
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _read_semiexp_checkpoint(path: Path, *, expected_atoms: int) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") not in SUPPORTED_SEMIEXP_CHECKPOINT_SCHEMAS:
        raise ScientificValidationError(f"Invalid SEfit checkpoint schema in {path}")
    coords = np.asarray(data.get("coordinates_angstrom"), dtype=float)
    if coords.shape != (expected_atoms, 3):
        raise ScientificValidationError(
            f"SEfit checkpoint atom count mismatch: expected {expected_atoms}, got {coords.shape[0] if coords.ndim == 2 else 'invalid'}"
        )
    return coords

def residuals_csv(residuals: tuple[SemiexperimentalResidual, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["isotopologue", "observable", "observed", "calculated", "residual"])
    for r in residuals:
        writer.writerow(
            [
                r.isotopologue,
                r.constant,
                f"{r.observed_equilibrium_MHz:.12g}",
                f"{r.calculated_MHz:.12g}",
                f"{r.residual_MHz:.12g}",
            ]
        )
    return stream.getvalue()

def rotational_constants_csv(rows: tuple[SemiexperimentalRotationalConstantComparison, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "isotopologue",
            "component",
            "corrected_experimental_MHz",
            "calculated_MHz",
            "difference_MHz",
        ]
    )
    for item in rows:
        writer.writerow(
            [
                item.isotopologue,
                item.component,
                f"{item.corrected_experimental_MHz:.12g}",
                f"{item.calculated_MHz:.12g}",
                f"{item.difference_MHz:.12g}",
            ]
        )
    return stream.getvalue()

def _rotational_residual_stats(
    rows: tuple[SemiexperimentalRotationalConstantComparison, ...],
) -> tuple[int, float, float, float, float]:
    diffs = np.asarray([row.difference_MHz for row in rows], dtype=float)
    if diffs.size == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    mean_square = float(np.mean(diffs * diffs))
    return (
        int(diffs.size),
        float(np.sqrt(mean_square)),
        mean_square,
        1000.0 * mean_square,
        float(np.max(np.abs(diffs))),
    )

def _rotational_residual_manifest_stats(
    rows: tuple[SemiexperimentalRotationalConstantComparison, ...],
) -> dict[str, float | int]:
    nrows, rms, mean_square, scaled_mean_square, max_abs = _rotational_residual_stats(rows)
    return {
        "n_rotational_constant_residuals": nrows,
        "rotational_rms_MHz": rms,
        "rotational_mean_square_MHz2": mean_square,
        "rotational_mean_square_1e3_MHz2": scaled_mean_square,
        "rotational_max_abs_MHz": max_abs,
    }

def semiexperimental_text_report(
    request: SemiexperimentalFitRequest,
    parameters: tuple[SemiexperimentalParameter, ...],
    geometry_parameters: tuple[SemiexperimentalGeometryParameter, ...],
    residuals: tuple[SemiexperimentalResidual, ...],
    rotational_constants: tuple[SemiexperimentalRotationalConstantComparison, ...],
    *,
    diagnostics: SemiexperimentalFitDiagnostics | None = None,
    stationary_point: str = "not_checked",
    fixed_parameters: tuple[str, ...] = (),
    diagnostic_warnings: tuple[SemiexperimentalDiagnosticWarning, ...] = (),
    svd_summary: tuple[str, ...] = (),
    constraint_summary: tuple[str, ...] = (),
    iteration_trace: tuple[SemiexperimentalIterationTrace, ...] = (),
    leave_one_out: tuple[SemiexperimentalLeaveOneOutRow, ...] = (),
) -> str:
    lines: list[str] = [
        "SEFIT TEXT OUTPUT v1",
        "MATRIX/MORPHEUS semiexperimental equilibrium-geometry fit",
        "=" * 72,
        "",
        "[method]",
        "program = MATRIX/MORPHEUS",
        "method = semiexperimental equilibrium-geometry least squares",
        f"solver = {diagnostics.solver if diagnostics is not None else 'adaptive_lm_trust_region'}",
        f"coordinate_model = {request.coordinate_model}",
        f"coordinate_basis = {_coordinate_model_description(request.coordinate_model)}",
        "excluded_rotational_constants = "
        + (", ".join(request.excluded_rotational_constants) or "none"),
        f"initial_geometry = {request.initial_geometry}",
        f"observable = {diagnostics.observable if diagnostics is not None else request.observable}",
        f"components = {','.join(diagnostics.components) if diagnostics is not None else request.rotational_components}",
        f"isotopologues = {', '.join(obs.label for obs in request.observations)}",
        f"stationary_point = {stationary_point}",
        "",
        "[constraints]",
    ]
    if fixed_parameters:
        lines.extend(f"fixed_parameter = {item}" for item in fixed_parameters)
    else:
        lines.append("fixed_parameter = none")
    if request.parameter_classes:
        for item in request.parameter_classes:
            lines.append(
                f"parameter_class = {item.name}; mode={item.mode}; patterns={'|'.join(item.patterns)}"
            )
    else:
        lines.append("parameter_class = none")
    if request.qm_predicates:
        for item in request.qm_predicates:
            lines.append(
                f"qm_predicate = {item.label_pattern}; value={item.value:.12g}; "
                f"sigma={item.sigma:.12g}; source={item.source}"
            )
    else:
        lines.append("qm_predicate = none")
    lines.extend(["", "[constraint_diagnostics]"])
    lines.extend(constraint_summary or ("constraint_diagnostics = not_available",))
    lines.extend(["", "[fit_statistics]"])
    if diagnostics is not None:
        (
            nrot,
            rotational_rms,
            rotational_mean_square,
            rotational_mean_square_scaled,
            rotational_max,
        ) = _rotational_residual_stats(rotational_constants)
        lines.extend(
            [
                f"convergence = {diagnostics.convergence_reason}",
                f"iterations = {diagnostics.accepted_steps + diagnostics.rejected_steps}",
                f"accepted_steps = {diagnostics.accepted_steps}",
                f"rejected_steps = {diagnostics.rejected_steps}",
                f"max_iterations = {diagnostics.max_iterations}",
                f"n_optimized_parameters = {diagnostics.n_optimized_parameters}",
                f"objective = {diagnostics.objective:.12g}",
                f"weighted_rms = {diagnostics.weighted_rms:.12g}",
                f"reduced_chi_square = {diagnostics.reduced_chi_square:.12g}",
                f"rank = {diagnostics.rank}",
                f"incremental_rank = {diagnostics.incremental_rank}",
                f"condition_number = {diagnostics.condition_number:.12g}",
                f"damping = {diagnostics.damping:.12g}",
                f"linear_solver = {diagnostics.linear_solver}",
                f"robust_loss = {diagnostics.robust_loss}",
                f"robust_scale = {diagnostics.robust_scale:.12g}",
                f"robust_downweighted_observations = {diagnostics.robust_downweighted_observations}",
                f"robust_downweighted_isotopologues = {diagnostics.robust_downweighted_isotopologues}",
                f"trust_radius = {diagnostics.trust_radius:.12g}",
                f"last_trust_ratio = {diagnostics.last_trust_ratio:.12g}",
                f"last_line_search_scale = {diagnostics.last_line_search_scale:.12g}",
                f"gicforge_calls = {diagnostics.gicforge_calls}",
                f"coordinate_model_reuse_steps = {diagnostics.coordinate_model_reuse_steps}",
                f"b_projector_analytic_refreshes = {diagnostics.b_projector_analytic_refreshes}",
                f"b_projector_secant_updates = {diagnostics.b_projector_secant_updates}",
                f"b_projector_secant_rejections = {diagnostics.b_projector_secant_rejections}",
                f"last_b_projector_secant_error = {diagnostics.last_b_projector_secant_error:.12g}",
                f"parameter_scale_min = {diagnostics.parameter_scale_min:.12g}",
                f"parameter_scale_max = {diagnostics.parameter_scale_max:.12g}",
            ]
        )
        lines.extend(
            [
                "",
                "[rotational_residual_statistics]",
                f"n_rotational_constants = {nrot}",
                f"rotational_rms_MHz = {rotational_rms:.12g}",
                f"rotational_mean_square_MHz2 = {rotational_mean_square:.12g}",
                f"rotational_mean_square_1e3_MHz2 = {rotational_mean_square_scaled:.12g}",
                f"rotational_max_abs_MHz = {rotational_max:.12g}",
                (
                    "note = RMS is sqrt(mean(diff_MHz^2)); mean_square_1e3 is printed "
                    "to compare unambiguously with legacy residual conventions."
                ),
            ]
        )
    else:
        lines.append("statistics = not_available")

    lines.extend(["", "[warnings]"])
    if diagnostic_warnings:
        for item in diagnostic_warnings:
            context = item.context or "-"
            lines.append(
                f"warning = severity:{item.severity}; code:{item.code}; "
                f"message:{item.message}; context:{context}"
            )
    else:
        lines.append("warning = none")

    lines.extend(["", "[rank_diagnostics]"])
    lines.extend(svd_summary or ("svd_diagnostics = not_available",))

    lines.extend(
        [
            "",
            "[iteration_trace]",
            "iter status objective_before objective_after rho damping trust_radius step_norm rank smin rel_smin constraint_max",
        ]
    )
    if iteration_trace:
        selected_trace = iteration_trace[-min(12, len(iteration_trace)) :]
        for item in selected_trace:
            lines.append(
                " ".join(
                    (
                        str(item.iteration),
                        item.status,
                        f"{item.objective_before:.12g}",
                        f"{item.objective_after:.12g}",
                        f"{item.trust_ratio:.12g}",
                        f"{item.damping:.12g}",
                        f"{item.trust_radius:.12g}",
                        f"{item.step_norm:.12g}",
                        str(item.rank),
                        f"{item.smallest_singular_value:.12g}",
                        f"{item.relative_smallest_singular_value:.12g}",
                        f"{item.constraint_max_abs:.12g}",
                    )
                )
            )
    else:
        lines.append("iteration_trace = not_available")

    lines.extend(["", "[working_coordinates]", f"coordinate_count = {len(parameters)}"])
    lines.append("index active class value sigma label")
    for idx, item in enumerate(parameters, start=1):
        lines.append(
            " ".join(
                (
                    str(idx),
                    "yes" if item.active else "no",
                    item.parameter_class or "-",
                    f"{item.value:.12g}",
                    f"{item.sigma:.12g}",
                    item.name,
                )
            )
        )

    lines.extend(
        [
            "",
            "[primitive_internal_coordinates]",
            "Final topological geometry with propagated errors and initial-to-final shifts",
        ]
    )
    lines.append("kind label atoms symbols final initial delta sigma unit")
    for item in geometry_parameters:
        atoms = "-".join(str(idx) for idx in item.atom_indices)
        symbols = "-".join(item.atom_symbols)
        if item.value_angstrom is not None:
            value = item.value_angstrom
            initial = item.initial_value_angstrom
            delta = item.delta_angstrom
            sigma = item.sigma_angstrom
            unit = "Angstrom"
        else:
            value = item.value_degree
            initial = item.initial_value_degree
            delta = item.delta_degree
            sigma = item.sigma_degree
            unit = "degree"
        lines.append(
            " ".join(
                (
                    item.kind,
                    item.label,
                    atoms,
                    symbols,
                    "" if value is None else f"{value:.12g}",
                    "" if initial is None else f"{initial:.12g}",
                    "" if delta is None else f"{delta:.12g}",
                    "" if sigma is None else f"{sigma:.12g}",
                    unit,
                )
            )
        )

    lines.extend(["", "[rotational_constants]", "Rotational constants (MHz)"])
    lines.append(
        "isotopologue component corrected_experimental_MHz calculated_MHz exp_minus_calc_MHz"
    )
    for item in rotational_constants:
        lines.append(
            " ".join(
                (
                    item.isotopologue,
                    item.component,
                    f"{item.corrected_experimental_MHz:.12g}",
                    f"{item.calculated_MHz:.12g}",
                    f"{item.difference_MHz:.12g}",
                )
            )
        )

    lines.extend(["", "[fit_residuals]"])
    lines.append("isotopologue observable observed calculated residual")
    for item in residuals:
        lines.append(
            " ".join(
                (
                    item.isotopologue,
                    item.constant,
                    f"{item.observed_equilibrium_MHz:.12g}",
                    f"{item.calculated_MHz:.12g}",
                    f"{item.residual_MHz:.12g}",
                )
            )
        )
    if leave_one_out:
        lines.extend(
            [
                "",
                "[leave_one_out]",
                "omitted training_rms omitted_rms max_abs cart_rms_shift convergence",
            ]
        )
        for item in leave_one_out:
            lines.append(
                " ".join(
                    (
                        item.omitted_isotopologue,
                        f"{item.training_rms:.12g}",
                        f"{item.omitted_rotational_rms_MHz:.12g}",
                        f"{item.omitted_rotational_max_abs_MHz:.12g}",
                        f"{item.cartesian_rms_shift_angstrom:.12g}",
                        item.convergence_reason,
                    )
                )
            )
    return "\n".join(lines) + "\n"

def _coordinate_model_description(coordinate_model: str) -> str:
    if coordinate_model == "cartesian_symmetry":
        return "SMITH SYCART symmetrized Cartesians with totally symmetric Hessian-free Cartesian displacements"
    return "SMITH non-redundant symmetry-adapted SONICs; active subspace is totally symmetric"

def _geometry_parameters(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    *,
    initial_coords: np.ndarray | None = None,
    fit_prims: object | None = None,
    fit_u_matrix: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    transform: np.ndarray | None = None,
    cartesian_from_parameters: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    topology_lock: TopologyLock | None = None,
) -> tuple[SemiexperimentalGeometryParameter, ...]:
    coords = np.asarray(coords, dtype=float)
    initial_array = (
        np.asarray(initial_coords, dtype=float)
        if initial_coords is not None and np.asarray(initial_coords).shape == coords.shape
        else None
    )
    if topology_lock is None:
        z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
        try:
            _continuous, graph, _ringset, _synthons, _aromaticity = build_topology_objects(
                coords, z_numbers
            )
        except Exception as exc:
            raise ScientificValidationError(
                f"Cannot build final geometry parameter table: {exc}"
            ) from exc
        bonds = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in graph.bonds))
        adjacency = tuple(
            tuple(sorted(int(item) for item in graph.adjacency[index]))
            for index in range(len(atoms))
        )
    else:
        _validate_locked_topology(atoms, coords, topology_lock, context="final geometry reporting")
        bonds = topology_lock.bonds
        adjacency = topology_lock.adjacency
    dx_dr = _cartesian_derivatives_wrt_effective_parameters(
        coords,
        fit_prims=fit_prims,
        fit_u_matrix=fit_u_matrix,
        active_mask=active_mask,
        transform=transform,
        cartesian_from_parameters=cartesian_from_parameters,
        covariance=covariance,
    )

    specs: list[tuple[str, str, tuple[int, ...], tuple[str, ...], Primitive, float]] = []
    for i, j in bonds:
        label = f"R({i + 1},{j + 1})"
        symbols = (str(atoms[i]), str(atoms[j]))
        specs.append(("bond", label, (i + 1, j + 1), symbols, Primitive("bond", (i, j)), 1.0))

    for center in range(len(atoms)):
        neighbors = sorted(adjacency[center])
        for pos, left in enumerate(neighbors):
            for right in neighbors[pos + 1 :]:
                label = f"A({left + 1},{center + 1},{right + 1})"
                symbols = (str(atoms[left]), str(atoms[center]), str(atoms[right]))
                primitive = Primitive("angle", (left, center, right))
                specs.append(
                    (
                        "angle",
                        label,
                        (left + 1, center + 1, right + 1),
                        symbols,
                        primitive,
                        180.0 / np.pi,
                    )
                )

    for center_left, center_right in bonds:
        left_neighbors = sorted(atom for atom in adjacency[center_left] if atom != center_right)
        right_neighbors = sorted(atom for atom in adjacency[center_right] if atom != center_left)
        for left in left_neighbors:
            for right in right_neighbors:
                if left == right:
                    continue
                label = f"D({left + 1},{center_left + 1},{center_right + 1},{right + 1})"
                symbols = (
                    str(atoms[left]),
                    str(atoms[center_left]),
                    str(atoms[center_right]),
                    str(atoms[right]),
                )
                primitive = Primitive("dihedral", (left, center_left, center_right, right))
                specs.append(
                    (
                        "dihedral",
                        label,
                        (left + 1, center_left + 1, center_right + 1, right + 1),
                        symbols,
                        primitive,
                        180.0 / np.pi,
                    )
                )

    if fit_prims is not None:
        _append_fit_primitive_geometry_specs(specs, fit_prims, atoms)

    primitives = [item[4] for item in specs]
    values = eval_primitives(primitives, coords) if primitives else np.array(())
    initial_values = (
        eval_primitives(primitives, initial_array)
        if primitives and initial_array is not None
        else np.array(())
    )
    sigmas = _geometry_parameter_sigmas(
        primitives,
        coords,
        cartesian_derivatives=dx_dr,
        covariance=covariance,
    )
    rows: list[SemiexperimentalGeometryParameter] = []
    for idx, (kind, label, atom_indices, symbols, _primitive, angular_scale) in enumerate(specs):
        value = float(values[idx])
        initial_value = float(initial_values[idx]) if idx < len(initial_values) else None
        sigma = sigmas[idx] if sigmas is not None and idx < len(sigmas) else None
        if kind in {"bond", "smith_bond"}:
            rows.append(
                SemiexperimentalGeometryParameter(
                    kind,
                    label,
                    atom_indices,
                    symbols,
                    value_angstrom=value,
                    initial_value_angstrom=initial_value,
                    delta_angstrom=None if initial_value is None else value - initial_value,
                    sigma_angstrom=sigma,
                )
            )
        else:
            value_degree = value * angular_scale
            initial_degree = None if initial_value is None else initial_value * angular_scale
            rows.append(
                SemiexperimentalGeometryParameter(
                    kind,
                    label,
                    atom_indices,
                    symbols,
                    value_degree=value_degree,
                    initial_value_degree=initial_degree,
                    delta_degree=(
                        None
                        if initial_degree is None
                        else _angular_delta_degree(value_degree, initial_degree)
                    ),
                    sigma_degree=None if sigma is None else sigma * angular_scale,
                )
            )
    rows.extend(
        _ring_puckering_geometry_parameters(
            atoms,
            coords,
            bonds,
            initial_coords=initial_array,
            cartesian_derivatives=dx_dr,
            covariance=covariance,
        )
    )
    return tuple(rows)

def _append_fit_primitive_geometry_specs(
    specs: list[tuple[str, str, tuple[int, ...], tuple[str, ...], Primitive, float]],
    fit_prims: object,
    atoms: list[str] | tuple[str, ...],
) -> None:
    """Append SMITH primitives absent from the covalent topology report.

    Non-covalent MORPHEUS fits may use pseudo-bond GICs.  Those primitives are
    intentionally absent from the ordinary covalent graph, but they are the
    chemically relevant coordinates for the fit and must appear in the final
    geometry table and uncertainty diagnostics.
    """

    seen = {_primitive_constraint_key(item[4]) for item in specs}
    for primitive in tuple(fit_prims):
        if primitive.kind not in {"bond", "angle", "dihedral", "out_of_plane", "linear_bend"}:
            continue
        key = _primitive_constraint_key(primitive)
        if key in seen:
            continue
        seen.add(key)
        atom_indices = tuple(int(atom) + 1 for atom in primitive.atoms)
        symbols = tuple(str(atoms[int(atom)]) for atom in primitive.atoms)
        label = f"SMITH:{_primitive_text(primitive)}"
        if primitive.kind == "bond":
            specs.append(("smith_bond", label, atom_indices, symbols, primitive, 1.0))
        elif primitive.kind == "angle":
            specs.append(("smith_angle", label, atom_indices, symbols, primitive, 180.0 / np.pi))
        elif primitive.kind == "dihedral":
            specs.append(("smith_dihedral", label, atom_indices, symbols, primitive, 180.0 / np.pi))
        elif primitive.kind == "out_of_plane":
            specs.append(
                ("smith_out_of_plane", label, atom_indices, symbols, primitive, 180.0 / np.pi)
            )
        elif primitive.kind == "linear_bend":
            specs.append(
                ("smith_linear_bend", label, atom_indices, symbols, primitive, 180.0 / np.pi)
            )

def _geometry_parameter_sigmas(
    geometry_prims: list[Primitive],
    coords: np.ndarray,
    *,
    cartesian_derivatives: np.ndarray | None,
    covariance: np.ndarray | None,
) -> list[float | None] | None:
    if not geometry_prims or covariance is None or covariance.size == 0:
        return None
    covariance = np.asarray(covariance, dtype=float)
    if cartesian_derivatives is None:
        return None
    dx_dr = np.asarray(cartesian_derivatives, dtype=float)
    if covariance.shape != (dx_dr.shape[1], dx_dr.shape[1]):
        return None
    b_geom = b_matrix_analytic(geometry_prims, coords)
    jac = b_geom @ dx_dr
    variances = np.einsum("ij,jk,ik->i", jac, covariance, jac, optimize=True)
    return [float(np.sqrt(max(value, 0.0))) if np.isfinite(value) else None for value in variances]

def _angular_delta_degree(value: float, reference: float) -> float:
    return float((float(value) - float(reference) + 180.0) % 360.0 - 180.0)

def _angular_delta_radian(value: float, reference: float) -> float:
    return float(np.deg2rad(_angular_delta_degree(np.degrees(value), np.degrees(reference))))

def _cartesian_derivatives_wrt_effective_parameters(
    coords: np.ndarray,
    *,
    fit_prims: object | None,
    fit_u_matrix: np.ndarray | None,
    active_mask: np.ndarray | None,
    transform: np.ndarray | None,
    covariance: np.ndarray | None,
    cartesian_from_parameters: np.ndarray | None = None,
) -> np.ndarray | None:
    if covariance is None or np.asarray(covariance).size == 0:
        return None
    covariance = np.asarray(covariance, dtype=float)
    if cartesian_from_parameters is not None:
        dx_dr = np.asarray(cartesian_from_parameters, dtype=float)
        return dx_dr if covariance.shape == (dx_dr.shape[1], dx_dr.shape[1]) else None
    if (
        fit_prims is None
        or fit_u_matrix is None
        or active_mask is None
        or transform is None
        or transform.size == 0
    ):
        return None
    b_fit = np.asarray(fit_u_matrix, dtype=float).T @ b_matrix_analytic(fit_prims, coords)
    active_indices = np.where(active_mask)[0]
    dq_dr = np.zeros((b_fit.shape[0], transform.shape[1]), dtype=float)
    dq_dr[active_indices, :] = transform
    if covariance.shape != (dq_dr.shape[1], dq_dr.shape[1]):
        return None
    return cartesian_from_internal_jacobian(b_fit, rcond=1.0e-8) @ dq_dr

def _ring_puckering_geometry_parameters(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    bonds: tuple[tuple[int, int], ...],
    *,
    initial_coords: np.ndarray | None = None,
    cartesian_derivatives: np.ndarray | None,
    covariance: np.ndarray | None,
) -> list[SemiexperimentalGeometryParameter]:
    rings = _six_membered_ring_cycles(len(atoms), bonds)
    rows: list[SemiexperimentalGeometryParameter] = []
    for ring_index, ring in enumerate(rings, start=1):
        primitives = _ring_puckering_dihedrals(ring)
        values = np.asarray(eval_primitives(primitives, coords), dtype=float)
        initial_values = (
            np.asarray(eval_primitives(primitives, initial_coords), dtype=float)
            if initial_coords is not None
            else None
        )
        rp1_coeffs, rp2_coeffs, rp3_coeffs = _six_membered_ring_puckering_coefficients()
        rp1 = float(rp1_coeffs @ values)
        rp2 = float(rp2_coeffs @ values)
        rp3 = float(rp3_coeffs @ values)
        q = float(np.hypot(rp1, rp2))
        phi = float(np.arctan2(rp2, rp1))
        initial_rp1 = initial_rp2 = initial_rp3 = initial_q = initial_phi = None
        if initial_values is not None:
            initial_rp1 = float(rp1_coeffs @ initial_values)
            initial_rp2 = float(rp2_coeffs @ initial_values)
            initial_rp3 = float(rp3_coeffs @ initial_values)
            initial_q = float(np.hypot(initial_rp1, initial_rp2))
            initial_phi = float(np.arctan2(initial_rp2, initial_rp1))
        sigma_rp1 = sigma_rp2 = sigma_rp3 = sigma_q = sigma_phi = None
        if (
            cartesian_derivatives is not None
            and covariance is not None
            and np.asarray(covariance).size
        ):
            b_dihedrals = b_matrix_analytic(primitives, coords)
            jac_rp1 = rp1_coeffs @ b_dihedrals @ cartesian_derivatives
            jac_rp2 = rp2_coeffs @ b_dihedrals @ cartesian_derivatives
            jac_rp3 = rp3_coeffs @ b_dihedrals @ cartesian_derivatives
            covariance = np.asarray(covariance, dtype=float)
            sigma_rp1 = _linear_error_sigma(jac_rp1, covariance)
            sigma_rp2 = _linear_error_sigma(jac_rp2, covariance)
            sigma_rp3 = _linear_error_sigma(jac_rp3, covariance)
            if q > 1.0e-14:
                jac_q = (rp1 * jac_rp1 + rp2 * jac_rp2) / q
                jac_phi = (-rp2 * jac_rp1 + rp1 * jac_rp2) / (q * q)
                sigma_q = _linear_error_sigma(jac_q, covariance)
                sigma_phi = _linear_error_sigma(jac_phi, covariance)
        atom_indices = tuple(item + 1 for item in ring)
        atom_symbols = tuple(str(atoms[item]) for item in ring)
        rows.extend(
            [
                SemiexperimentalGeometryParameter(
                    "ring_puckering",
                    f"RPck{ring_index:04d}_1",
                    atom_indices,
                    atom_symbols,
                    value=rp1,
                    initial_value=initial_rp1,
                    delta=None if initial_rp1 is None else rp1 - initial_rp1,
                    sigma=sigma_rp1,
                    unit="rad",
                ),
                SemiexperimentalGeometryParameter(
                    "ring_puckering",
                    f"RPck{ring_index:04d}_2",
                    atom_indices,
                    atom_symbols,
                    value=rp2,
                    initial_value=initial_rp2,
                    delta=None if initial_rp2 is None else rp2 - initial_rp2,
                    sigma=sigma_rp2,
                    unit="rad",
                ),
                SemiexperimentalGeometryParameter(
                    "ring_puckering",
                    f"RPck{ring_index:04d}_3",
                    atom_indices,
                    atom_symbols,
                    value=rp3,
                    initial_value=initial_rp3,
                    delta=None if initial_rp3 is None else rp3 - initial_rp3,
                    sigma=sigma_rp3,
                    unit="rad",
                ),
                SemiexperimentalGeometryParameter(
                    "ring_puckering",
                    f"QPck{ring_index:04d}",
                    atom_indices,
                    atom_symbols,
                    value=q,
                    initial_value=initial_q,
                    delta=None if initial_q is None else q - initial_q,
                    sigma=sigma_q,
                    unit="rad",
                ),
                SemiexperimentalGeometryParameter(
                    "ring_puckering",
                    f"PhiP{ring_index:04d}",
                    atom_indices,
                    atom_symbols,
                    value=phi,
                    initial_value=initial_phi,
                    delta=None if initial_phi is None else _angular_delta_radian(phi, initial_phi),
                    sigma=sigma_phi,
                    unit="rad",
                    value_degree=float(np.degrees(phi)),
                    initial_value_degree=(
                        None if initial_phi is None else float(np.degrees(initial_phi))
                    ),
                    delta_degree=(
                        None
                        if initial_phi is None
                        else _angular_delta_degree(
                            float(np.degrees(phi)),
                            float(np.degrees(initial_phi)),
                        )
                    ),
                    sigma_degree=None if sigma_phi is None else float(np.degrees(sigma_phi)),
                ),
            ]
        )
    return rows

def _linear_error_sigma(jacobian_row: np.ndarray, covariance: np.ndarray) -> float | None:
    row = np.asarray(jacobian_row, dtype=float)
    if covariance.shape != (row.size, row.size):
        return None
    variance = float(row @ covariance @ row)
    return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else None

def _six_membered_ring_puckering_coefficients() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0.57735027, -0.28867513, -0.28867513, 0.57735027, -0.28867513, -0.28867513]),
        np.array([0.0, 0.5, -0.5, 0.0, 0.5, -0.5]),
        np.array([0.40824829, -0.40824829, 0.40824829, -0.40824829, 0.40824829, -0.40824829]),
    )

def _ring_puckering_dihedrals(ring: tuple[int, ...]) -> list[Primitive]:
    size = len(ring)
    return [
        Primitive(
            "dihedral",
            (
                ring[(idx - 1) % size],
                ring[idx % size],
                ring[(idx + 1) % size],
                ring[(idx + 2) % size],
            ),
        )
        for idx in range(size)
    ]

def _six_membered_ring_cycles(
    atom_count: int, bonds: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], ...]:
    adjacency = {idx: set() for idx in range(atom_count)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    cycles: set[tuple[int, ...]] = set()
    for start in range(atom_count):
        stack = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            if len(path) == 6:
                if start in adjacency[current]:
                    cycles.add(_canonical_cycle(path))
                continue
            for neighbor in adjacency[current]:
                if neighbor <= start or neighbor in path:
                    continue
                stack.append((neighbor, (*path, neighbor)))
    return tuple(sorted(cycles))

def _canonical_cycle(path: tuple[int, ...]) -> tuple[int, ...]:
    variants = []
    nitems = len(path)
    for seq in (path, tuple(reversed(path))):
        for offset in range(nitems):
            rotated = seq[offset:] + seq[:offset]
            variants.append(rotated)
    return min(variants)

def kraitchman_csv_rows(rows: tuple[KraitchmanComparison, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "isotopologue",
            "atom_index",
            "atom",
            "isotope_A",
            "axis",
            "kraitchman_abs_A",
            "fitted_abs_A",
            "difference_A",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.isotopologue,
                row.atom_index,
                row.atom,
                row.isotope_mass_number,
                row.coordinate,
                f"{row.kraitchman_abs_angstrom:.12g}",
                f"{row.fitted_abs_angstrom:.12g}",
                f"{row.difference_angstrom:.12g}",
            ]
        )
    return stream.getvalue()

def _matrix_csv(labels: tuple[str, ...], matrix: np.ndarray | None) -> str:
    mat = np.asarray(matrix if matrix is not None else np.zeros((0, 0)), dtype=float)
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["parameter", *labels])
    for label, row in zip(labels, mat):
        writer.writerow([label, *[f"{value:.12g}" for value in row]])
    return stream.getvalue()

def _svd_diagnostics_csv(labels: tuple[str, ...], weighted_jac: np.ndarray | None) -> str:
    rows = _svd_diagnostic_rows(labels, weighted_jac)
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "index",
            "singular_value",
            "relative_singular_value",
            "near_null",
            "dominant_coordinate_combination",
        ]
    )
    for idx, singular, relative, near_null, combination in rows:
        writer.writerow([idx, f"{singular:.12g}", f"{relative:.12g}", int(near_null), combination])
    return stream.getvalue()

def _svd_summary_lines(labels: tuple[str, ...], weighted_jac: np.ndarray | None) -> tuple[str, ...]:
    rows = _svd_diagnostic_rows(labels, weighted_jac)
    if not rows:
        return ("svd_diagnostics = not_available",)
    selected = [row for row in rows if row[3]][:5] or rows[-min(5, len(rows)) :]
    lines = ["index singular_value relative near_null dominant_coordinate_combination"]
    lines.extend(
        f"{idx} {singular:.6g} {relative:.6g} {int(near_null)} {combination}"
        for idx, singular, relative, near_null, combination in selected
    )
    return tuple(lines)

def _svd_diagnostic_rows(
    labels: tuple[str, ...],
    weighted_jac: np.ndarray | None,
) -> tuple[tuple[int, float, float, bool, str], ...]:
    jac = np.asarray(
        weighted_jac if weighted_jac is not None else np.zeros((0, len(labels))), dtype=float
    )
    if jac.ndim != 2 or jac.shape[1] == 0:
        return ()
    try:
        _u, singular, vh = np.linalg.svd(jac, full_matrices=True)
    except np.linalg.LinAlgError:
        return ()
    ncols = jac.shape[1]
    s0 = float(singular[0]) if singular.size else 0.0
    threshold = max(jac.shape) * np.finfo(float).eps * max(s0, 1.0) * 100.0
    rows = []
    for col in range(ncols):
        singular_value = float(singular[col]) if col < singular.size else 0.0
        relative = singular_value / s0 if s0 > 0.0 else 0.0
        vector = vh[col, :] if col < vh.shape[0] else np.zeros(ncols, dtype=float)
        near_null = singular_value <= max(threshold, s0 * 1.0e-8)
        rows.append(
            (
                col + 1,
                singular_value,
                relative,
                bool(near_null),
                _format_svd_combination(labels, vector),
            )
        )
    return tuple(rows)

def _uncertainty_diagnostics_csv(
    labels: tuple[str, ...],
    weighted_jac: np.ndarray | None,
    weighted_residual: np.ndarray | None,
) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        ["cutoff", "relative_cutoff", "rank", "parameter", "sigma", "sigma_ratio_to_default"]
    )
    for (
        cutoff_label,
        relative_cutoff,
        rank,
        parameter,
        sigma,
        ratio,
    ) in _uncertainty_diagnostic_rows(
        labels,
        weighted_jac,
        weighted_residual,
    ):
        writer.writerow(
            [
                cutoff_label,
                f"{relative_cutoff:.12g}",
                rank,
                parameter,
                f"{sigma:.12g}",
                f"{ratio:.12g}",
            ]
        )
    return stream.getvalue()

def _uncertainty_cutoff_sensitivity(
    labels: tuple[str, ...],
    weighted_jac: np.ndarray | None,
    weighted_residual: np.ndarray | None,
) -> float:
    ratios = [
        max(ratio, 1.0 / ratio)
        for _cutoff_label, _relative_cutoff, _rank, _parameter, _sigma, ratio in _uncertainty_diagnostic_rows(
            labels,
            weighted_jac,
            weighted_residual,
        )
        if np.isfinite(ratio) and ratio > 0.0
    ]
    return max(ratios) if ratios else 1.0

def _uncertainty_diagnostic_rows(
    labels: tuple[str, ...],
    weighted_jac: np.ndarray | None,
    weighted_residual: np.ndarray | None,
) -> tuple[tuple[str, float, int, str, float, float], ...]:
    jac = np.asarray(
        weighted_jac if weighted_jac is not None else np.zeros((0, len(labels))), dtype=float
    )
    residual = np.asarray(
        weighted_residual if weighted_residual is not None else np.zeros(0), dtype=float
    )
    if jac.ndim != 2 or jac.shape[1] == 0:
        return ()
    try:
        _u, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return ()
    if not singular.size:
        return ()
    sigma2 = (
        float(residual @ residual) / max(jac.shape[0] - jac.shape[1], 1) if residual.size else 0.0
    )
    s0 = max(float(singular[0]), 1.0)
    default_relative = max(jac.shape) * np.finfo(float).eps * 100.0
    cutoffs = (
        ("default", default_relative),
        ("rel_1e-12", 1.0e-12),
        ("rel_1e-10", 1.0e-10),
        ("rel_1e-8", 1.0e-8),
        ("rel_1e-6", 1.0e-6),
    )
    sigma_by_cutoff: dict[str, np.ndarray] = {}
    rank_by_cutoff: dict[str, int] = {}
    for label, relative in cutoffs:
        keep = singular > max(float(relative), 0.0) * s0
        inv_s2 = np.zeros_like(singular)
        inv_s2[keep] = 1.0 / (singular[keep] * singular[keep])
        covariance = sigma2 * ((vh.T * inv_s2) @ vh)
        diag = np.diag(covariance) if covariance.size else np.zeros(jac.shape[1], dtype=float)
        sigma_by_cutoff[label] = np.sqrt(np.maximum(diag, 0.0))
        rank_by_cutoff[label] = int(np.sum(keep))
    default_sigma = sigma_by_cutoff["default"]
    rows: list[tuple[str, float, int, str, float, float]] = []
    for label, relative in cutoffs:
        sigmas = sigma_by_cutoff[label]
        for idx, sigma in enumerate(sigmas):
            base = float(default_sigma[idx]) if idx < default_sigma.size else 0.0
            ratio = float(sigma / base) if base > 0.0 else (1.0 if sigma == 0.0 else float("inf"))
            rows.append(
                (
                    label,
                    float(relative),
                    rank_by_cutoff[label],
                    labels[idx] if idx < len(labels) else f"q{idx + 1}",
                    float(sigma),
                    ratio,
                )
            )
    return tuple(rows)

def _format_svd_combination(labels: tuple[str, ...], vector: np.ndarray, nterms: int = 5) -> str:
    if vector.size == 0:
        return ""
    order = np.argsort(-np.abs(vector))[: min(nterms, vector.size)]
    parts = []
    for idx in order:
        label = labels[idx] if idx < len(labels) else f"q{idx + 1}"
        parts.append(f"{float(vector[idx]):+.4f}*{label}")
    return " ".join(parts)

def _constraints_csv(
    fixed_parameters: tuple[str, ...],
    fixed_primitives: tuple[Primitive, ...],
    parameter_classes: tuple[ParameterClassConstraint, ...],
    parameters: tuple[SemiexperimentalParameter, ...],
) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "kind",
            "name",
            "mode",
            "pattern_or_primitive",
            "matched_active_parameters",
            "matched_labels",
        ]
    )
    active_labels = tuple(item.name for item in parameters if item.active)
    expression_definitions = _gic_expression_definitions_from_patterns(fixed_parameters)
    for item in fixed_parameters:
        matches = _matched_labels(item, active_labels)
        kind, mode = _input_constraint_record_kind(item, expression_definitions)
        writer.writerow([kind, item, mode, item, len(matches), ";".join(matches)])
    for primitive in fixed_primitives:
        writer.writerow(
            [
                "expanded_primitive",
                _primitive_text(primitive),
                "fixed",
                _primitive_text(primitive),
                "",
                "",
            ]
        )
    for parameter_class in parameter_classes:
        matches = tuple(label for label in active_labels if _class_matches(parameter_class, label))
        writer.writerow(
            [
                "parameter_class",
                parameter_class.name,
                parameter_class.mode,
                "|".join(parameter_class.patterns),
                len(matches),
                ";".join(matches),
            ]
        )
    return stream.getvalue()

def _constraint_summary_lines(
    fixed_parameters: tuple[str, ...],
    fixed_primitives: tuple[Primitive, ...],
    parameter_classes: tuple[ParameterClassConstraint, ...],
    parameters: tuple[SemiexperimentalParameter, ...],
) -> tuple[str, ...]:
    active_labels = tuple(item.name for item in parameters if item.active)
    expression_definitions = _gic_expression_definitions_from_patterns(fixed_parameters)
    n_expression_constraints = sum(
        1
        for item in fixed_parameters
        if _parse_gic_expression_constraint_pattern(item, definitions=expression_definitions)
    )
    n_definitions = sum(
        1
        for item in fixed_parameters
        if _parse_gic_expression_definition_pattern(item) is not None
        and _parse_gic_expression_constraint_pattern(item, definitions=expression_definitions)
        is None
    )
    lines = [
        f"input_records = {len(fixed_parameters)}",
        f"input_expression_constraints = {n_expression_constraints}",
        f"input_coordinate_definitions = {n_definitions}",
        f"symmetry_expanded_fixed_primitives = {len(fixed_primitives)}",
        f"parameter_classes = {len(parameter_classes)}",
    ]
    for item in fixed_parameters:
        matches = _matched_labels(item, active_labels)
        kind, _mode = _input_constraint_record_kind(item, expression_definitions)
        lines.append(f"{kind} = {item}; active_label_matches={len(matches)}")
    for parameter_class in parameter_classes:
        matches = tuple(label for label in active_labels if _class_matches(parameter_class, label))
        lines.append(
            f"parameter_class = {parameter_class.name}; mode={parameter_class.mode}; "
            f"patterns={'|'.join(parameter_class.patterns)}; active_label_matches={len(matches)}"
        )
    return tuple(lines)

def _input_constraint_record_kind(
    item: str,
    definitions: tuple[GICExpressionDefinition, ...],
) -> tuple[str, str]:
    if _parse_gic_expression_constraint_pattern(item, definitions=definitions) is not None:
        return "constraint_record", "constraint"
    if _parse_gic_expression_definition_pattern(item) is not None:
        return "definition_record", "definition"
    if _primitives_from_fixed_pattern(item):
        return "primitive_record", "fixed"
    return "input_fixed", "fixed"

def _matched_labels(pattern: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    low = str(pattern).lower()
    return tuple(label for label in labels if low in label.lower())

def _leave_one_out_csv(rows: tuple[SemiexperimentalLeaveOneOutRow, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "omitted_isotopologue",
            "training_isotopologues",
            "training_rms",
            "omitted_rotational_rms_MHz",
            "omitted_rotational_max_abs_MHz",
            "cartesian_rms_shift_angstrom",
            "cartesian_max_shift_angstrom",
            "mean_parameter_sigma",
            "max_parameter_sigma",
            "iterations",
            "convergence_reason",
            "rank",
            "condition_number",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.omitted_isotopologue,
                row.training_isotopologues,
                f"{row.training_rms:.12g}",
                f"{row.omitted_rotational_rms_MHz:.12g}",
                f"{row.omitted_rotational_max_abs_MHz:.12g}",
                f"{row.cartesian_rms_shift_angstrom:.12g}",
                f"{row.cartesian_max_shift_angstrom:.12g}",
                f"{row.mean_parameter_sigma:.12g}",
                f"{row.max_parameter_sigma:.12g}",
                row.iterations,
                row.convergence_reason,
                row.rank,
                f"{row.condition_number:.12g}",
            ]
        )
    return stream.getvalue()

def _eigenvalues_csv(values: np.ndarray | None) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["index", "eigenvalue"])
    for idx, value in enumerate(
        np.asarray(values if values is not None else (), dtype=float), start=1
    ):
        writer.writerow([idx, f"{value:.12g}"])
    return stream.getvalue()

def _diagnostics_csv(diagnostics: SemiexperimentalFitDiagnostics | None) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["key", "value"])
    if diagnostics is None:
        writer.writerow(["status", "not_reported"])
        return stream.getvalue()
    for key, value in diagnostics.__dict__.items():
        writer.writerow([key, value])
    return stream.getvalue()

def iteration_trace_csv_rows(rows: tuple[SemiexperimentalIterationTrace, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "iteration",
            "status",
            "objective_before",
            "objective_after",
            "actual_reduction",
            "predicted_reduction",
            "trust_ratio",
            "line_search_scale",
            "damping",
            "trust_radius",
            "step_norm",
            "gradient_inf_norm",
            "rank",
            "smallest_singular_value",
            "relative_smallest_singular_value",
            "constraint_max_abs",
            "robust_scale",
            "robust_downweighted_observations",
            "robust_downweighted_isotopologues",
            "coordinate_model_age",
            "b_projector_secant_error",
            "linear_solver",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.iteration,
                row.status,
                f"{row.objective_before:.12g}",
                f"{row.objective_after:.12g}",
                f"{row.actual_reduction:.12g}",
                f"{row.predicted_reduction:.12g}",
                f"{row.trust_ratio:.12g}",
                f"{row.line_search_scale:.12g}",
                f"{row.damping:.12g}",
                f"{row.trust_radius:.12g}",
                f"{row.step_norm:.12g}",
                f"{row.gradient_inf_norm:.12g}",
                row.rank,
                f"{row.smallest_singular_value:.12g}",
                f"{row.relative_smallest_singular_value:.12g}",
                f"{row.constraint_max_abs:.12g}",
                f"{row.robust_scale:.12g}",
                row.robust_downweighted_observations,
                row.robust_downweighted_isotopologues,
                row.coordinate_model_age,
                f"{row.b_projector_secant_error:.12g}",
                row.linear_solver,
            ]
        )
    return stream.getvalue()

def _warnings_csv(rows: tuple[SemiexperimentalDiagnosticWarning, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["severity", "code", "message", "context"])
    for row in rows:
        writer.writerow([row.severity, row.code, row.message, row.context])
    return stream.getvalue()

def _request_with_auto_resolved_isotopologues(
    request: SemiexperimentalFitRequest,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
) -> tuple[SemiexperimentalFitRequest, tuple[SemiexperimentalDiagnosticWarning, ...]]:
    observations, warnings = _auto_resolve_isotopic_substitutions(
        atoms, coords, request.observations
    )
    if observations == request.observations:
        return request, warnings
    return replace(request, observations=observations), warnings

def _auto_resolve_isotopic_substitutions(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> tuple[tuple[IsotopologueObservation, ...], tuple[SemiexperimentalDiagnosticWarning, ...]]:
    if not observations:
        return observations, ()
    parent = next((obs for obs in observations if not obs.substitutions), observations[0])
    try:
        parent_exp = np.asarray(parent.corrected.as_tuple(), dtype=float)
        parent_calc = _rotational_constants_for_substitution(atoms, coords, parent.substitutions)
    except Exception:
        return observations, ()
    atom_symbols = tuple(str(atom).strip().capitalize() for atom in atoms)
    resolved: list[IsotopologueObservation] = []
    warnings: list[SemiexperimentalDiagnosticWarning] = []
    used_single_substitutions: set[tuple[int, int, str]] = set()
    for obs in observations:
        replacement = obs
        if len(obs.substitutions) == 1:
            atom_index, isotope = next(iter(obs.substitutions.items()))
            best = _best_single_isotopic_substitution(
                atom_symbols,
                coords,
                parent_exp,
                parent_calc,
                obs,
                int(atom_index),
                int(isotope),
            )
            if best is not None:
                used_atom, input_rms, used_rms = best
                if used_atom != atom_index:
                    key = (used_atom, int(isotope), obs.label)
                    if key not in used_single_substitutions:
                        replacement = replace(obs, substitutions={used_atom: int(isotope)})
                        used_single_substitutions.add(key)
                        warnings.append(
                            SemiexperimentalDiagnosticWarning(
                                "warning",
                                "isotopologue_mapping_autocorrected",
                                "Single-substitution isotopologue was reassigned to the atom that best reproduces the observed isotopic shift.",
                                (
                                    f"isotopologue={obs.label};isotope={int(isotope)};input_atom={int(atom_index)};"
                                    f"used_atom={used_atom};input_shift_rms_MHz={input_rms:.6g};"
                                    f"used_shift_rms_MHz={used_rms:.6g}"
                                ),
                            )
                        )
        resolved.append(replacement)
    return tuple(resolved), tuple(warnings)

def _best_single_isotopic_substitution(
    atom_symbols: tuple[str, ...],
    coords: np.ndarray,
    parent_exp: np.ndarray,
    parent_calc: np.ndarray,
    obs: IsotopologueObservation,
    atom_index: int,
    isotope: int,
) -> tuple[int, float, float] | None:
    if atom_index < 1 or atom_index > len(atom_symbols):
        return None
    symbol = atom_symbols[atom_index - 1]
    candidates = tuple(idx + 1 for idx, item in enumerate(atom_symbols) if item == symbol)
    if len(candidates) < 2:
        return None
    try:
        exp_shift = np.asarray(obs.corrected.as_tuple(), dtype=float) - parent_exp
    except Exception:
        return None
    candidate_rms: list[tuple[float, int]] = []
    for candidate in candidates:
        try:
            candidate_calc = _rotational_constants_for_substitution(
                atom_symbols,
                coords,
                {candidate: isotope},
            )
        except Exception:
            continue
        candidate_shift = candidate_calc - parent_calc
        rms = float(np.sqrt(np.mean((exp_shift - candidate_shift) ** 2)))
        candidate_rms.append((rms, candidate))
    if not candidate_rms:
        return None
    candidate_rms.sort()
    best_rms, best_atom = candidate_rms[0]
    input_rms = next((rms for rms, candidate in candidate_rms if candidate == atom_index), best_rms)
    if best_atom == atom_index:
        return (best_atom, input_rms, best_rms)
    clear_absolute = input_rms - best_rms >= DIAGNOSTIC_ISOTOPE_SHIFT_WARNING_MHZ
    clear_relative = best_rms <= DIAGNOSTIC_ISOTOPE_SHIFT_IMPROVEMENT_RATIO * input_rms
    if not (clear_absolute or clear_relative):
        return (atom_index, input_rms, input_rms)
    if len(candidate_rms) > 1:
        second_rms = candidate_rms[1][0]
        if second_rms > 0.0 and best_rms > 0.90 * second_rms:
            return (atom_index, input_rms, input_rms)
    return (best_atom, input_rms, best_rms)

def _isotopic_mapping_warning_rows(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    if not observations:
        return ()
    parent = next((obs for obs in observations if not obs.substitutions), observations[0])
    try:
        parent_exp = np.asarray(parent.corrected.as_tuple(), dtype=float)
        parent_calc = _rotational_constants_for_substitution(atoms, coords, parent.substitutions)
    except Exception:
        return ()
    rows: list[SemiexperimentalDiagnosticWarning] = []
    atom_symbols = tuple(str(atom).strip().capitalize() for atom in atoms)
    for obs in observations:
        if len(obs.substitutions) != 1:
            continue
        atom_index, isotope = next(iter(obs.substitutions.items()))
        if atom_index < 1 or atom_index > len(atom_symbols):
            continue
        symbol = atom_symbols[atom_index - 1]
        candidates = tuple(idx + 1 for idx, item in enumerate(atom_symbols) if item == symbol)
        if len(candidates) < 2:
            continue
        try:
            exp_shift = np.asarray(obs.corrected.as_tuple(), dtype=float) - parent_exp
            current_calc = _rotational_constants_for_substitution(
                atoms, coords, {atom_index: isotope}
            )
        except Exception:
            continue
        current_shift = current_calc - parent_calc
        current_rms = float(np.sqrt(np.mean((exp_shift - current_shift) ** 2)))
        best_atom = atom_index
        best_rms = current_rms
        for candidate in candidates:
            if candidate == atom_index:
                continue
            try:
                candidate_calc = _rotational_constants_for_substitution(
                    atoms, coords, {candidate: isotope}
                )
            except Exception:
                continue
            candidate_shift = candidate_calc - parent_calc
            candidate_rms = float(np.sqrt(np.mean((exp_shift - candidate_shift) ** 2)))
            if candidate_rms < best_rms:
                best_atom = candidate
                best_rms = candidate_rms
        if best_atom != atom_index and (
            current_rms - best_rms >= DIAGNOSTIC_ISOTOPE_SHIFT_WARNING_MHZ
            or best_rms <= DIAGNOSTIC_ISOTOPE_SHIFT_IMPROVEMENT_RATIO * current_rms
        ):
            rows.append(
                SemiexperimentalDiagnosticWarning(
                    "warning",
                    "isotopologue_mapping_suspicious",
                    "Single-substitution isotopic shift is much better reproduced by another atom of the same element.",
                    (
                        f"isotopologue={obs.label};isotope={isotope};input_atom={atom_index};"
                        f"suggested_atom={best_atom};input_shift_rms_MHz={current_rms:.6g};"
                        f"suggested_shift_rms_MHz={best_rms:.6g}"
                    ),
                )
            )
        elif current_rms >= 10.0 * DIAGNOSTIC_ISOTOPE_SHIFT_WARNING_MHZ:
            rows.append(
                SemiexperimentalDiagnosticWarning(
                    "info",
                    "large_isotopic_shift_mismatch",
                    "Single-substitution isotopic shift is poorly reproduced by the current geometry and atom mapping.",
                    (
                        f"isotopologue={obs.label};isotope={isotope};input_atom={atom_index};"
                        f"shift_rms_MHz={current_rms:.6g}"
                    ),
                )
            )
    return tuple(rows)

def _rotational_constants_for_substitution(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    substitutions: dict[int, int],
) -> np.ndarray:
    isotopes: list[int | None] = [None] * len(atoms)
    for atom_index, isotope in substitutions.items():
        if 1 <= atom_index <= len(isotopes):
            isotopes[atom_index - 1] = int(isotope)
    structure = Structure.from_atoms_coords(
        list(atoms),
        [tuple(row) for row in np.asarray(coords, dtype=float)],
        isotopes=isotopes,
    )
    return np.asarray(rotational_constants_MHz(structure, isotopic=True), dtype=float)

def _semiexp_warning_rows(
    diagnostics: SemiexperimentalFitDiagnostics | None,
    active_names: tuple[str, ...],
    parameters: tuple[SemiexperimentalParameter, ...],
    geometry_parameters: tuple[SemiexperimentalGeometryParameter, ...],
    weighted_jacobian: np.ndarray | None,
    measurement_model: MeasurementModel | None,
    robust_sqrt_weights: np.ndarray | None,
    weighted_residual: np.ndarray | None = None,
    weight_diagnostics: tuple[SemiexperimentalWeightDiagnostic, ...] = (),
    *,
    iteration_trace: tuple[SemiexperimentalIterationTrace, ...] = (),
    correlation: np.ndarray | None = None,
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    rows: list[SemiexperimentalDiagnosticWarning] = []
    seen: set[tuple[str, str]] = set()

    def add(severity: str, code: str, message: str, context: str = "") -> None:
        key = (code, context)
        if key in seen:
            return
        seen.add(key)
        rows.append(SemiexperimentalDiagnosticWarning(severity, code, message, context))

    if diagnostics is not None:
        if diagnostics.convergence_reason == "max_iter":
            add(
                "warning",
                "not_converged_max_iter",
                "SEfit reached the maximum number of iterations before a convergence criterion was satisfied.",
                f"max_iterations={diagnostics.max_iterations};accepted={diagnostics.accepted_steps};rejected={diagnostics.rejected_steps}",
            )
        if diagnostics.rank < diagnostics.n_optimized_parameters:
            add(
                "warning",
                "rank_deficient",
                f"Numerical rank {diagnostics.rank} is lower than {diagnostics.n_optimized_parameters} fitted parameters.",
                f"rank={diagnostics.rank};n_parameters={diagnostics.n_optimized_parameters}",
            )
        if diagnostics.incremental_rank < diagnostics.n_optimized_parameters:
            add(
                "warning",
                "incremental_rank_deficient",
                f"Incremental column rank {diagnostics.incremental_rank} is lower than the fitted dimensionality.",
                f"incremental_rank={diagnostics.incremental_rank};n_parameters={diagnostics.n_optimized_parameters}",
            )
        if (
            not np.isfinite(diagnostics.condition_number)
            or diagnostics.condition_number > DIAGNOSTIC_CONDITION_WARNING
        ):
            add(
                "warning",
                "ill_conditioned_jacobian",
                "Final weighted Jacobian is ill-conditioned.",
                f"condition_number={diagnostics.condition_number:.6g}",
            )
        if diagnostics.planar:
            component_text = ",".join(diagnostics.components)
            if len(diagnostics.components) != 2:
                add(
                    "warning",
                    "planar_component_count",
                    "Planar refinement should use exactly two independent rotational components.",
                    f"components={component_text}",
                )
            elif (
                not np.isfinite(diagnostics.condition_number)
                or diagnostics.condition_number > DIAGNOSTIC_CONDITION_WARNING
            ):
                add(
                    "warning",
                    "planar_pair_ill_conditioned",
                    "Selected planar component pair is numerically ill-conditioned.",
                    f"components={component_text};condition_number={diagnostics.condition_number:.6g}",
                )
        if diagnostics.robust_loss != "none" and diagnostics.robust_downweighted_isotopologues:
            add(
                "info",
                "robust_downweighted_isotopologues",
                "Robust loss downweighted one or more complete isotopologue blocks.",
                f"count={diagnostics.robust_downweighted_isotopologues};loss={diagnostics.robust_loss}",
            )
        total_steps = diagnostics.accepted_steps + diagnostics.rejected_steps
        if total_steps and diagnostics.rejected_steps >= 3:
            rejected_fraction = diagnostics.rejected_steps / total_steps
            if rejected_fraction >= DIAGNOSTIC_REJECTED_STEP_FRACTION_WARNING:
                add(
                    "warning",
                    "many_rejected_steps",
                    "Trust-region globalization rejected a large fraction of proposed steps.",
                    f"rejected_fraction={rejected_fraction:.6g};accepted={diagnostics.accepted_steps};rejected={diagnostics.rejected_steps}",
                )
        if 0.0 < diagnostics.trust_radius < DIAGNOSTIC_TRUST_RADIUS_WARNING:
            add(
                "warning",
                "small_trust_radius",
                "Final trust radius is close to the numerical lower range.",
                f"trust_radius={diagnostics.trust_radius:.6g}",
            )
        if 0.0 < diagnostics.last_line_search_scale < DIAGNOSTIC_LINE_SEARCH_SCALE_WARNING:
            add(
                "warning",
                "line_search_stagnation",
                "Final accepted step required a very small line-search scale.",
                f"line_search_scale={diagnostics.last_line_search_scale:.6g}",
            )
        if (
            np.isfinite(diagnostics.reduced_chi_square)
            and diagnostics.reduced_chi_square > DIAGNOSTIC_REDUCED_CHI_SQUARE_WARNING
        ):
            add(
                "warning",
                "large_reduced_chi_square",
                "Reduced chi-square is larger than expected for the supplied uncertainties.",
                f"reduced_chi_square={diagnostics.reduced_chi_square:.6g}",
            )
        scale_min = max(float(diagnostics.parameter_scale_min), np.finfo(float).tiny)
        scale_ratio = float(diagnostics.parameter_scale_max) / scale_min
        if np.isfinite(scale_ratio) and scale_ratio > DIAGNOSTIC_PARAMETER_SCALE_RATIO_WARNING:
            add(
                "info",
                "large_parameter_scale_range",
                "Dynamic column scaling spans a very large range.",
                f"scale_min={diagnostics.parameter_scale_min:.6g};scale_max={diagnostics.parameter_scale_max:.6g};ratio={scale_ratio:.6g}",
            )
        if diagnostics.damping >= 0.1 * DAMPING_MAX:
            add(
                "warning",
                "large_lm_damping",
                "Levenberg-Marquardt damping is close to the configured upper bound.",
                f"damping={diagnostics.damping:.6g}",
            )

    singular_rows = _svd_diagnostic_rows(active_names, weighted_jacobian)
    near_null = [row for row in singular_rows if row[3]]
    if near_null:
        smallest = min(near_null, key=lambda item: item[2])
        add(
            "warning",
            "small_singular_value",
            "One or more final weighted-Jacobian singular values are near the numerical null space.",
            f"min_relative={smallest[2]:.6g};combination={smallest[4]}",
        )
    elif singular_rows:
        smallest = min(singular_rows, key=lambda item: item[2])
        if smallest[2] < DIAGNOSTIC_RELATIVE_SINGULAR_WARNING:
            add(
                "warning",
                "small_singular_value",
                "Smallest final weighted-Jacobian singular value is below the diagnostic threshold.",
                f"min_relative={smallest[2]:.6g};combination={smallest[4]}",
            )

    for label, weight in _robust_group_weights(measurement_model, robust_sqrt_weights):
        if weight < DIAGNOSTIC_ROBUST_WEIGHT_WARNING:
            severity = "warning" if weight >= DIAGNOSTIC_ROBUST_WEIGHT_SEVERE else "severe"
            add(
                severity,
                "low_robust_isotopologue_weight",
                "Robust loss assigned a low weight to a complete isotopologue block.",
                f"isotopologue={label};weight={weight:.6g}",
            )

    for warning in _large_geometry_uncertainty_warnings(geometry_parameters):
        add(warning.severity, warning.code, warning.message, warning.context)

    sensitivity = _uncertainty_cutoff_sensitivity(
        active_names, weighted_jacobian, weighted_residual
    )
    if sensitivity > 10.0:
        add(
            "warning",
            "uncertainty_cutoff_sensitive",
            "At least one parameter uncertainty is strongly sensitive to the SVD rank cutoff.",
            f"max_sigma_ratio={sensitivity:.6g}",
        )

    for warning in _weighted_residual_warnings(measurement_model, weighted_residual):
        add(warning.severity, warning.code, warning.message, warning.context)

    for warning in _leverage_warnings(measurement_model, weighted_jacobian):
        add(warning.severity, warning.code, warning.message, warning.context)

    for warning in _weight_model_warnings(weight_diagnostics):
        add(warning.severity, warning.code, warning.message, warning.context)

    for warning in _high_correlation_warnings(active_names, correlation):
        add(warning.severity, warning.code, warning.message, warning.context)

    for warning in _iteration_trace_warnings(iteration_trace):
        add(warning.severity, warning.code, warning.message, warning.context)

    active_sigmas = [item.sigma for item in parameters if item.active and np.isfinite(item.sigma)]
    if active_sigmas:
        max_sigma = max(active_sigmas)
        median_sigma = float(np.median(np.asarray(active_sigmas, dtype=float)))
        if median_sigma > 0.0 and max_sigma > 25.0 * median_sigma:
            add(
                "info",
                "large_relative_parameter_sigma",
                "At least one active working coordinate has a much larger uncertainty than the median.",
                f"max_sigma={max_sigma:.6g};median_sigma={median_sigma:.6g}",
            )
    return tuple(rows)

def _weighted_residual_warnings(
    measurement_model: MeasurementModel | None,
    weighted_residual: np.ndarray | None,
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    if measurement_model is None or weighted_residual is None:
        return ()
    residual = np.asarray(weighted_residual, dtype=float)
    rows: list[SemiexperimentalDiagnosticWarning] = []
    for idx, value in enumerate(residual[: measurement_model.n_experimental_rows]):
        if not np.isfinite(value) or abs(float(value)) < DIAGNOSTIC_WEIGHTED_RESIDUAL_WARNING:
            continue
        isotopologue, observable = (
            measurement_model.labels[idx]
            if idx < len(measurement_model.labels)
            else (f"row_{idx + 1}", "unknown")
        )
        rows.append(
            SemiexperimentalDiagnosticWarning(
                "warning",
                "large_weighted_residual",
                "A fitted observable has a large normalized residual.",
                f"row={idx + 1};isotopologue={isotopologue};observable={observable};weighted_residual={float(value):.6g}",
            )
        )
    return tuple(rows)

def _leverage_warnings(
    measurement_model: MeasurementModel | None,
    weighted_jacobian: np.ndarray | None,
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    if measurement_model is None or weighted_jacobian is None:
        return ()
    leverage = _leverage_values(np.asarray(weighted_jacobian, dtype=float))
    rows: list[SemiexperimentalDiagnosticWarning] = []
    for idx, value in enumerate(leverage[: measurement_model.n_experimental_rows]):
        if not np.isfinite(value) or float(value) < DIAGNOSTIC_LEVERAGE_WARNING:
            continue
        isotopologue, observable = (
            measurement_model.labels[idx]
            if idx < len(measurement_model.labels)
            else (f"row_{idx + 1}", "unknown")
        )
        rows.append(
            SemiexperimentalDiagnosticWarning(
                "info",
                "high_leverage_observation",
                "A fitted observable has high statistical leverage in the final linearized model.",
                f"row={idx + 1};isotopologue={isotopologue};observable={observable};leverage={float(value):.6g}",
            )
        )
    return tuple(rows)

def _high_correlation_warnings(
    labels: tuple[str, ...],
    correlation: np.ndarray | None,
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    corr = np.asarray(correlation if correlation is not None else np.zeros((0, 0)), dtype=float)
    if corr.ndim != 2 or corr.size == 0:
        return ()
    rows: list[SemiexperimentalDiagnosticWarning] = []
    n = min(corr.shape[0], corr.shape[1], len(labels))
    for i in range(n):
        for j in range(i + 1, n):
            value = float(corr[i, j])
            if np.isfinite(value) and abs(value) >= DIAGNOSTIC_CORRELATION_WARNING:
                rows.append(
                    SemiexperimentalDiagnosticWarning(
                        "info",
                        "high_parameter_correlation",
                        "Two fitted parameters are very strongly correlated.",
                        f"left={labels[i]};right={labels[j]};correlation={value:.6g}",
                    )
                )
    return tuple(rows[:20])

def _iteration_trace_warnings(
    iteration_trace: tuple[SemiexperimentalIterationTrace, ...],
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    if not iteration_trace:
        return ()
    rows: list[SemiexperimentalDiagnosticWarning] = []
    tail = iteration_trace[-min(5, len(iteration_trace)) :]
    if len(tail) >= 3 and all(item.status == "rejected" for item in tail[-3:]):
        rows.append(
            SemiexperimentalDiagnosticWarning(
                "warning",
                "repeated_final_rejections",
                "The final iterations were rejected by the trust-region acceptance test.",
                f"iterations={','.join(str(item.iteration) for item in tail[-3:])}",
            )
        )
    last = iteration_trace[-1]
    if last.gradient_inf_norm > 0.0 and last.step_norm < DIAGNOSTIC_TRUST_RADIUS_WARNING:
        rows.append(
            SemiexperimentalDiagnosticWarning(
                "info",
                "small_final_step_with_gradient",
                "The final step is very small while the gradient remains non-zero.",
                f"gradient_inf_norm={last.gradient_inf_norm:.6g};step_norm={last.step_norm:.6g}",
            )
        )
    return tuple(rows)

def _robust_group_weights(
    measurement_model: MeasurementModel | None,
    robust_sqrt_weights: np.ndarray | None,
) -> tuple[tuple[str, float], ...]:
    if measurement_model is None or robust_sqrt_weights is None:
        return ()
    sqrt_weights = np.asarray(robust_sqrt_weights, dtype=float)
    if sqrt_weights.size == 0:
        return ()
    result: list[tuple[str, float]] = []
    for group in _experimental_isotopologue_row_groups(measurement_model):
        valid = tuple(
            idx
            for idx in group
            if 0 <= idx < min(measurement_model.n_experimental_rows, sqrt_weights.size)
        )
        if not valid:
            continue
        label = (
            measurement_model.labels[valid[0]][0]
            if valid[0] < len(measurement_model.labels)
            else f"row_{valid[0] + 1}"
        )
        weights = sqrt_weights[np.asarray(valid, dtype=int)] ** 2
        result.append((label, float(np.mean(weights))))
    return tuple(result)

def _large_geometry_uncertainty_warnings(
    geometry_parameters: tuple[SemiexperimentalGeometryParameter, ...],
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    rows: list[SemiexperimentalDiagnosticWarning] = []
    for item in geometry_parameters:
        if item.value_angstrom is not None and item.sigma_angstrom is not None:
            sigma = float(item.sigma_angstrom)
            if np.isfinite(sigma) and sigma > DIAGNOSTIC_BOND_SIGMA_WARNING_ANGSTROM:
                rows.append(
                    SemiexperimentalDiagnosticWarning(
                        "warning",
                        "large_geometry_uncertainty",
                        "Propagated bond-length uncertainty exceeds the diagnostic threshold.",
                        f"{item.label};sigma_A={sigma:.6g}",
                    )
                )
        elif item.value_degree is not None and item.sigma_degree is not None:
            sigma = float(item.sigma_degree)
            if np.isfinite(sigma) and sigma > DIAGNOSTIC_ANGLE_SIGMA_WARNING_DEGREE:
                rows.append(
                    SemiexperimentalDiagnosticWarning(
                        "warning",
                        "large_geometry_uncertainty",
                        "Propagated angular-coordinate uncertainty exceeds the diagnostic threshold.",
                        f"{item.label};sigma_deg={sigma:.6g}",
                    )
                )
    return tuple(sorted(rows, key=lambda item: item.context))

def _influence_csv(
    model: MeasurementModel | None,
    residuals: tuple[SemiexperimentalResidual, ...],
    weighted_jac: np.ndarray | None,
    weighted_residual: np.ndarray | None,
) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "row",
            "isotopologue",
            "observable",
            "residual",
            "weighted_residual",
            "chi_square_contribution",
            "leverage",
        ]
    )
    labels = (
        model.labels
        if model is not None
        else tuple((item.isotopologue, item.constant) for item in residuals)
    )
    weighted = np.asarray(weighted_residual if weighted_residual is not None else (), dtype=float)
    leverage = _leverage_values(
        np.asarray(weighted_jac if weighted_jac is not None else np.zeros((0, 0)), dtype=float)
    )
    for idx, item in enumerate(residuals):
        weighted_value = float(weighted[idx]) if idx < weighted.size else 0.0
        leverage_value = float(leverage[idx]) if idx < leverage.size else 0.0
        iso, obs = labels[idx] if idx < len(labels) else (item.isotopologue, item.constant)
        writer.writerow(
            [
                idx + 1,
                iso,
                obs,
                f"{item.residual_MHz:.12g}",
                f"{weighted_value:.12g}",
                f"{weighted_value * weighted_value:.12g}",
                f"{leverage_value:.12g}",
            ]
        )
    return stream.getvalue()

def _weight_model_warnings(
    rows: tuple[SemiexperimentalWeightDiagnostic, ...],
) -> tuple[SemiexperimentalDiagnosticWarning, ...]:
    if not rows:
        return ()
    warnings: list[SemiexperimentalDiagnosticWarning] = []
    positive = np.array([item.effective_weight for item in rows if item.effective_weight > 0.0])
    if positive.size:
        ratio = float(np.max(positive) / np.min(positive))
        if np.isfinite(ratio) and ratio > 1.0e8:
            warnings.append(
                SemiexperimentalDiagnosticWarning(
                    "info",
                    "large_weight_dynamic_range",
                    "Effective least-squares weights span a very large range.",
                    f"ratio={ratio:.6g}",
                )
            )
    dominant = max(rows, key=lambda item: item.total_weight_fraction)
    if dominant.total_weight_fraction > 0.50:
        warnings.append(
            SemiexperimentalDiagnosticWarning(
                "warning",
                "dominant_weight_row",
                "One fit row carries more than half of the total effective weight.",
                f"row={dominant.row};kind={dominant.kind};label={dominant.isotopologue}:{dominant.observable};fraction={dominant.total_weight_fraction:.6g}",
            )
        )
    predicate_fraction = sum(
        item.total_weight_fraction for item in rows if item.kind == "predicate"
    )
    experimental_fraction = sum(
        item.total_weight_fraction for item in rows if item.kind == "experimental"
    )
    if predicate_fraction > experimental_fraction and experimental_fraction > 0.0:
        warnings.append(
            SemiexperimentalDiagnosticWarning(
                "info",
                "predicate_weight_dominates",
                "QM predicate rows carry more total effective weight than experimental rows.",
                f"predicate_fraction={predicate_fraction:.6g};experimental_fraction={experimental_fraction:.6g}",
            )
        )
    return tuple(warnings)

def _high_correlations_csv(
    labels: tuple[str, ...], correlation: np.ndarray | None, threshold: float = 0.90
) -> str:
    corr = np.asarray(correlation if correlation is not None else np.zeros((0, 0)), dtype=float)
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["parameter_i", "parameter_j", "correlation_abs", "correlation"])
    rows = []
    for i in range(min(corr.shape[0], len(labels))):
        for j in range(i + 1, min(corr.shape[1], len(labels))):
            value = float(corr[i, j])
            abs_value = abs(value)
            if abs_value >= threshold:
                rows.append((abs_value, value, labels[i], labels[j]))
    for abs_value, value, left, right in sorted(rows, reverse=True)[:50]:
        writer.writerow([left, right, f"{abs_value:.12g}", f"{value:.12g}"])
    return stream.getvalue()
