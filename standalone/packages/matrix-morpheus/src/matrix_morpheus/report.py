from __future__ import annotations

from dataclasses import dataclass
import csv
from html import escape
import io
from pathlib import Path
import re

import numpy as np

from matrix_morpheus.numerics import rank_condition

from .contracts import (
    IsotopologueObservation,
    ParameterClassConstraint,
    QMParameterPredicate,
    SemiexperimentalFitRequest,
)
from .coordinate_model import (
    _active_mask,
    _atomic_number,
    _gic_model,
    _gic_fixed_patterns,
    _gicforge_a1_mask,
    _hydrogen_fixed_primitives,
    _merge_primitives,
    _parameter_class_transform,
    _primitive_constrained_transform,
    _symmetry_expanded_fixed_primitives,
)
from .constraints import (
    _fixed_primitive_targets,
    _fixed_primitives_from_patterns,
    _gic_expression_constraint_targets,
    _gic_expression_constraints_from_patterns,
    _gic_expression_definitions_from_patterns,
    _gic_values,
    _linear_primitive_constraints_from_patterns,
    _project_fixed_primitives,
)
from .fit import fit_semiexperimental_geometry
from .fit_outputs import _combined_fixed_parameters, _semiexp_warning_rows
from .measurement_model import (
    _build_measurement_model,
    _jacobian_constants_wrt_gics,
    _topology_lock,
)
from .models import SemiexperimentalFitResult
from .geometry_input import read_geometry_input


@dataclass(frozen=True)
class SemiexperimentalGICPreview:
    atoms: tuple[str, ...]
    gic_labels: tuple[str, ...]
    rows: tuple["SemiexperimentalGICPreviewRow", ...]
    suggested_classes: tuple[ParameterClassConstraint, ...]
    warnings: tuple[str, ...]

    @property
    def text(self) -> str:
        lines = [
            "MATRIX/MORPHEUS semiexperimental GIC preview",
            f"Atoms: {len(self.atoms)}",
            f"Non-redundant GICs: {len(self.gic_labels)}",
            "",
            "Suggested parameter classes:",
        ]
        lines.extend(
            f"  {item.name}:{item.mode}:{'|'.join(item.patterns)}"
            for item in self.suggested_classes
        )
        if not self.suggested_classes:
            lines.append("  none")
        lines.extend(["", "GIC labels:"])
        lines.extend(
            f"  {row.label} [{row.kind}] class={row.suggested_class or '-'}" for row in self.rows
        )
        if self.warnings:
            lines.extend(["", "Warnings:", *[f"  {item}" for item in self.warnings]])
        return "\n".join(lines)


@dataclass(frozen=True)
class SemiexperimentalGICPreviewRow:
    label: str
    kind: str
    atoms: tuple[int, ...]
    suggested_class: str
    state: str


@dataclass(frozen=True)
class SemiexperimentalValidationIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class SemiexperimentalConditioningPreview:
    rank: int
    condition_number: float
    n_observations: int
    n_effective_parameters: int
    components: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def text(self) -> str:
        lines = [
            "Semiexperimental conditioning preview",
            f"observations: {self.n_observations}",
            f"effective parameters: {self.n_effective_parameters}",
            f"rank: {self.rank}",
            f"condition number: {self.condition_number:.8g}",
            f"components: {','.join(self.components)}",
        ]
        if self.warnings:
            lines.extend(["Warnings:", *[f"  {item}" for item in self.warnings]])
        return "\n".join(lines)


@dataclass(frozen=True)
class SemiexperimentalSensitivityAdvisorRow:
    label: str
    value: float
    sensitivity: float
    relative_sensitivity: float
    chemical_role: str
    current_state: str
    suggested_state: str
    predicate_sigma: float
    reason: str


@dataclass(frozen=True)
class SemiexperimentalSensitivityAdvisor:
    rows: tuple[SemiexperimentalSensitivityAdvisorRow, ...]
    predicates: tuple[QMParameterPredicate, ...]
    fixed_patterns: tuple[str, ...]
    components: tuple[str, ...]

    @property
    def fit_count(self) -> int:
        return sum(1 for row in self.rows if row.suggested_state == "fit")

    @property
    def predicate_count(self) -> int:
        return sum(1 for row in self.rows if row.suggested_state == "predicate")

    @property
    def fixed_count(self) -> int:
        return sum(1 for row in self.rows if row.suggested_state == "fixed")

    @property
    def csv(self) -> str:
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "value",
                "sensitivity",
                "relative_sensitivity",
                "chemical_role",
                "current_state",
                "suggested_state",
                "predicate_sigma",
                "reason",
            ]
        )
        for row in self.rows:
            writer.writerow(
                [
                    row.label,
                    f"{row.value:.12g}",
                    f"{row.sensitivity:.12g}",
                    f"{row.relative_sensitivity:.12g}",
                    row.chemical_role,
                    row.current_state,
                    row.suggested_state,
                    f"{row.predicate_sigma:.12g}",
                    row.reason,
                ]
            )
        return stream.getvalue()

    @property
    def text(self) -> str:
        lines = [
            "MATRIX/MORPHEUS sensitivity advisor",
            f"components: {','.join(self.components)}",
            f"fit: {self.fit_count}",
            f"predicate: {self.predicate_count}",
            f"fixed: {self.fixed_count}",
        ]
        lines.extend(
            f"  {row.current_state}->{row.suggested_state:9s} "
            f"role={row.chemical_role} rel={row.relative_sensitivity:.4g} {row.label}"
            for row in self.rows
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class SemiexperimentalBenchmarkCase:
    label: str
    request: SemiexperimentalFitRequest
    max_iter: int | None = None
    step: float = 1.0e-4
    damping: float = 1.0e-8
    max_step: float = 0.25
    prune_condition: float = 0.0


@dataclass(frozen=True)
class SemiexperimentalBenchmarkRow:
    label: str
    rms_MHz: float
    rotational_rms_MHz: float
    iterations: int
    rank: int
    incremental_rank: int
    condition_number: float
    stationary_point: str
    n_parameters: int
    n_kraitchman: int
    gicforge_calls: int
    coordinate_model_reuse_steps: int
    b_projector_secant_updates: int


def preview_semiexperimental_gics(
    xyz: Path,
    observations: tuple[IsotopologueObservation, ...] = (),
) -> SemiexperimentalGICPreview:
    geometry_input = read_geometry_input(Path(xyz))
    atoms = tuple(geometry_input.atoms)
    coords = geometry_input.coordinates_angstrom
    z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
    _prims, _u_matrix, labels = _gic_model(np.asarray(coords, dtype=float), z_numbers)
    suggestions = suggest_parameter_classes(tuple(atoms), labels, observations)
    rows = _preview_rows(labels, suggestions, geometry_input.fixed_parameters)
    warnings = _preview_warnings(labels, suggestions)
    return SemiexperimentalGICPreview(tuple(atoms), labels, rows, suggestions, warnings)


def validate_semiexperimental_request(
    request: SemiexperimentalFitRequest,
) -> tuple[SemiexperimentalValidationIssue, ...]:
    issues: list[SemiexperimentalValidationIssue] = []
    try:
        geometry_input = read_geometry_input(Path(request.initial_geometry))
        atoms = geometry_input.atoms
    except Exception as exc:
        return (SemiexperimentalValidationIssue("error", f"Cannot read parent geometry: {exc}"),)
    labels = [obs.label for obs in request.observations]
    if len(labels) != len(set(labels)):
        issues.append(SemiexperimentalValidationIssue("error", "Duplicate isotopologue labels"))
    for obs in request.observations:
        if any(value <= 0.0 for value in obs.constants.as_tuple()):
            issues.append(
                SemiexperimentalValidationIssue(
                    "error", f"{obs.label}: rotational constants must be positive"
                )
            )
        if obs.weights is not None and any(value <= 0.0 for value in obs.weights.as_tuple()):
            issues.append(
                SemiexperimentalValidationIssue(
                    "error", f"{obs.label}: sigma-derived weights must be positive"
                )
            )
        seen_atoms = set()
        for atom_index, mass in obs.substitutions.items():
            if atom_index in seen_atoms:
                issues.append(
                    SemiexperimentalValidationIssue(
                        "error", f"{obs.label}: duplicate substitution at atom {atom_index}"
                    )
                )
            seen_atoms.add(atom_index)
            if atom_index < 1 or atom_index > len(atoms):
                issues.append(
                    SemiexperimentalValidationIssue(
                        "error", f"{obs.label}: substitution atom {atom_index} is out of range"
                    )
                )
            elif mass == 2 and atoms[atom_index - 1].upper() != "H":
                issues.append(
                    SemiexperimentalValidationIssue(
                        "warning", f"{obs.label}: deuterium substitution on non-H atom {atom_index}"
                    )
                )
        if any(
            abs(value) > 0.25 * max(abs(base), 1.0)
            for value, base in zip(obs.correction.as_tuple(), obs.constants.as_tuple())
        ):
            issues.append(
                SemiexperimentalValidationIssue(
                    "warning", f"{obs.label}: unusually large vibrational correction"
                )
            )
    try:
        preview = preview_semiexperimental_gics(request.initial_geometry, request.observations)
    except Exception as exc:
        issues.append(
            SemiexperimentalValidationIssue("error", f"Cannot generate GIC preview: {exc}")
        )
        return tuple(issues)
    for parameter_class in request.parameter_classes:
        matches = [
            label
            for label in preview.gic_labels
            if any(pattern.lower() in label.lower() for pattern in parameter_class.patterns)
        ]
        if not matches:
            issues.append(
                SemiexperimentalValidationIssue(
                    "error", f"Parameter class {parameter_class.name} matches no GIC"
                )
            )
        kinds = {_gic_kind(label) for label in matches}
        if len(kinds) > 1:
            issues.append(
                SemiexperimentalValidationIssue(
                    "error",
                    f"Parameter class {parameter_class.name} mixes coordinate types: {', '.join(sorted(kinds))}",
                )
            )
    return tuple(issues)


def preview_semiexperimental_conditioning(
    request: SemiexperimentalFitRequest,
    *,
    step: float = 1.0e-4,
) -> SemiexperimentalConditioningPreview:
    geometry_input = read_geometry_input(Path(request.initial_geometry))
    atoms = geometry_input.atoms
    coords_arr = np.asarray(geometry_input.coordinates_angstrom, dtype=float)
    z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
    prims, u_matrix, labels = _gic_model(coords_arr, z_numbers)
    fixed_parameters = _combined_fixed_parameters(
        request.fixed_parameters, geometry_input.fixed_parameters
    )
    fixed_gic_patterns = _gic_fixed_patterns(fixed_parameters)
    linear_constraints = _linear_primitive_constraints_from_patterns(fixed_parameters)
    expression_constraints = _gic_expression_constraints_from_patterns(fixed_parameters)
    expression_definitions = _gic_expression_definitions_from_patterns(fixed_parameters)
    fixed_primitives = _merge_primitives(
        _fixed_primitives_from_patterns(fixed_parameters),
        _hydrogen_fixed_primitives(atoms, prims, fixed_parameters, coords=coords_arr),
    )
    fixed_primitives = _symmetry_expanded_fixed_primitives(
        atoms, coords_arr, prims, fixed_primitives
    )
    fixed_targets = _fixed_primitive_targets(fixed_primitives, coords_arr)
    expression_targets = _gic_expression_constraint_targets(
        expression_constraints,
        coords_arr,
        prims,
        u_matrix,
        labels,
        definitions=expression_definitions,
    )
    if fixed_primitives or linear_constraints or expression_constraints:
        coords_arr = _project_fixed_primitives(
            coords_arr,
            fixed_primitives,
            fixed_targets,
            linear_constraints=linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
    measurement = _build_measurement_model(
        request, atoms, coords_arr, prims, u_matrix, labels
    )
    active = _active_mask(
        labels, fixed_gic_patterns, request.parameter_classes
    ) & _gicforge_a1_mask(labels)
    jac_gic = _jacobian_constants_wrt_gics(
        atoms, coords_arr, request, prims, u_matrix, active, labels, measurement, step=step
    )
    transform, _names, _class_by_gic = _parameter_class_transform(
        labels, active, request.parameter_classes
    )
    transform, _names = _primitive_constrained_transform(
        coords_arr,
        prims,
        u_matrix,
        active,
        transform,
        _names,
        fixed_primitives,
        linear_constraints=linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        expression_definitions=expression_definitions,
        labels=labels,
    )
    jac = jac_gic @ transform
    weighted = jac * np.sqrt(measurement.weights)[:, None]
    conditioning = rank_condition(weighted)
    warnings = []
    if conditioning.rank < weighted.shape[1]:
        warnings.append("rank deficient for the current isotopologues/classes")
    if not np.isfinite(conditioning.condition_number) or conditioning.condition_number > 1.0e8:
        warnings.append("ill-conditioned parameter set")
    return SemiexperimentalConditioningPreview(
        conditioning.rank,
        conditioning.condition_number,
        int(weighted.shape[0]),
        int(weighted.shape[1]),
        measurement.components,
        tuple(warnings),
    )


def advise_semiexperimental_gic_sensitivity(
    request: SemiexperimentalFitRequest,
    *,
    step: float = 1.0e-4,
    fit_relative_threshold: float = 0.15,
    fixed_relative_threshold: float = 1.0e-6,
    min_fit_count: int | None = None,
    distance_sigma_angstrom: float = 0.003,
    angle_sigma_degree: float = 0.3,
    torsion_sigma_degree: float = 0.5,
    soft_predicate_scale: float = 1.0,
    null_predicate_scale: float = 1.0,
    fit_regularization_scale: float = 0.0,
) -> SemiexperimentalSensitivityAdvisor:
    """Partition non-redundant GICs from rotational-constant sensitivity.

    Large-sensitivity coordinates are left free. Smaller but measurable
    coordinates receive QM predicates, and numerically null coordinates are
    fixed. The advisor also keeps a minimum number of ranked coordinates free
    when several isotopologues are available, because low first-order
    sensitivity does not imply that the coordinate is scientifically
    disposable. The sensitivities are computed with the same SMITH GICs and
    MORPHEUS measurement model used by the fit, but with any pre-existing
    predicates removed so that the ranking is driven only by experiment.
    """

    if request.coordinate_model != "gic":
        raise ValueError("GIC sensitivity advisor requires coordinate_model='gic'")
    geometry_input = read_geometry_input(Path(request.initial_geometry))
    atoms = geometry_input.atoms
    coords_arr = np.asarray(geometry_input.coordinates_angstrom, dtype=float)
    z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
    prims, u_matrix, labels = _gic_model(coords_arr, z_numbers)
    experimental_request = SemiexperimentalFitRequest(
        initial_geometry=request.initial_geometry,
        observations=request.observations,
        fixed_parameters=request.fixed_parameters,
        observable=request.observable,
        rotational_components=request.rotational_components,
        qm_predicates=(),
        parameter_classes=request.parameter_classes,
        coordinate_model=request.coordinate_model,
        robust_loss=request.robust_loss,
        robust_scale=request.robust_scale,
        leave_one_out=request.leave_one_out,
        excluded_rotational_constants=request.excluded_rotational_constants,
    )
    measurement = _build_measurement_model(
        experimental_request, atoms, coords_arr, prims, u_matrix, labels
    )
    fixed_parameters = _combined_fixed_parameters(
        request.fixed_parameters, geometry_input.fixed_parameters
    )
    active = _active_mask(
        labels,
        _gic_fixed_patterns(fixed_parameters),
        request.parameter_classes,
    ) & _gicforge_a1_mask(labels)
    jac = _jacobian_constants_wrt_gics(
        atoms,
        coords_arr,
        experimental_request,
        prims,
        u_matrix,
        active,
        labels,
        measurement,
        step=step,
    )
    active_indices = np.where(active)[0]
    weighted = jac * np.sqrt(measurement.weights)[:, None]
    sensitivities = np.linalg.norm(weighted, axis=0) if weighted.size else np.zeros(0)
    max_sensitivity = float(np.max(sensitivities)) if sensitivities.size else 0.0
    if not np.isfinite(max_sensitivity) or max_sensitivity <= 0.0:
        max_sensitivity = 1.0
    q_values = _gic_values(prims, u_matrix, coords_arr)
    candidates: list[tuple[str, float, float, float, str]] = []
    for column, label_index in enumerate(active_indices):
        label = labels[int(label_index)]
        sensitivity = float(sensitivities[column])
        relative = sensitivity / max_sensitivity
        label_id = _gic_id(label)
        if relative < fixed_relative_threshold:
            candidates.append(
                (
                    label,
                    float(q_values[int(label_index)]),
                    sensitivity,
                    relative,
                    "below_fixed_threshold",
                )
            )
        else:
            candidates.append(
                (
                    label,
                    float(q_values[int(label_index)]),
                    sensitivity,
                    relative,
                    "",
                )
            )
    candidates.sort(key=lambda item: item[2], reverse=True)
    free_budget = max(1, int(measurement.n_experimental_rows))
    required_fit = _sensitivity_min_fit_count(
        min_fit_count,
        n_active=len(candidates),
        n_isotopologues=len(request.observations),
    )
    selected_labels = _sensitivity_selected_fit_labels(
        candidates,
        free_budget=free_budget,
        required_fit=required_fit,
        fit_relative_threshold=fit_relative_threshold,
    )
    rows: list[SemiexperimentalSensitivityAdvisorRow] = []
    predicates: list[QMParameterPredicate] = []
    fixed_patterns: list[str] = []
    for label, value, sensitivity, relative, fixed_reason in candidates:
        label_id = _gic_id(label)
        chemical_role = _sensitivity_chemical_role(label)
        current_state = _sensitivity_current_state(request, label_id, label)
        if fixed_reason:
            suggested_state = "fixed"
            sigma = 0.0
            reason = fixed_reason
            fixed_patterns.append(label_id)
        elif label_id in selected_labels:
            suggested_state = "fit"
            sigma = _sensitivity_fit_regularization_sigma(
                label,
                relative=relative,
                fit_regularization_scale=fit_regularization_scale,
                distance_sigma_angstrom=distance_sigma_angstrom,
                angle_sigma_degree=angle_sigma_degree,
                torsion_sigma_degree=torsion_sigma_degree,
            )
            reason = (
                "experiment_sensitive"
                if relative >= fit_relative_threshold
                else _sensitivity_fit_reason(label)
            )
            if sigma > 0.0:
                predicates.append(
                    QMParameterPredicate(
                        label_id,
                        value,
                        sigma,
                        "morpheus_sensitivity_advisor_fit_regularization",
                    )
                )
        else:
            suggested_state = "predicate"
            sigma = _sensitivity_predicate_sigma(
                label,
                relative=relative,
                fit_relative_threshold=fit_relative_threshold,
                distance_sigma_angstrom=distance_sigma_angstrom,
                angle_sigma_degree=angle_sigma_degree,
                torsion_sigma_degree=torsion_sigma_degree,
                soft_predicate_scale=soft_predicate_scale,
                null_predicate_scale=null_predicate_scale,
            )
            reason = (
                "below_fit_threshold"
                if relative < fit_relative_threshold
                else "outside_experimental_budget"
            )
            predicates.append(
                QMParameterPredicate(
                    label_id,
                    value,
                    sigma,
                    "morpheus_sensitivity_advisor",
                )
            )
        rows.append(
            SemiexperimentalSensitivityAdvisorRow(
                label,
                value,
                sensitivity,
                relative,
                chemical_role,
                current_state,
                suggested_state,
                sigma,
                reason,
            )
        )
    return SemiexperimentalSensitivityAdvisor(
        tuple(rows), tuple(predicates), tuple(fixed_patterns), measurement.components
    )


def _sensitivity_current_state(
    request: SemiexperimentalFitRequest,
    label_id: str,
    label: str,
) -> str:
    low_label = str(label).lower()
    low_id = str(label_id).lower()
    if any(pattern.lower() in low_label or pattern.lower() == low_id for pattern in request.fixed_parameters):
        return "fixed"
    if any(pattern.lower() in low_label or pattern.lower() == low_id for pattern in (predicate.label_pattern for predicate in request.qm_predicates)):
        return "predicate"
    for parameter_class in request.parameter_classes:
        if any(pattern.lower() in low_label or pattern.lower() == low_id for pattern in parameter_class.patterns):
            return f"class:{parameter_class.mode}"
    return "free"


def _sensitivity_chemical_role(label: str) -> str:
    if _is_inter_or_special_gic(label):
        return "inter_soft"
    if _is_soft_gic(label):
        return "soft"
    return "intra_hard"


def _sensitivity_selected_fit_labels(
    candidates: list[tuple[str, float, float, float, str]],
    *,
    free_budget: int,
    required_fit: int,
    fit_relative_threshold: float,
) -> set[str]:
    eligible = [item for item in candidates if not item[4]]
    selected: list[str] = []
    inter_soft = [item for item in eligible if _sensitivity_priority(item[0]) == 0]
    inter_min = min(
        len(inter_soft),
        int(free_budget),
        max(0, min(6, max(3, int(np.ceil(required_fit / 2.0))))),
    )
    for item in _sensitivity_selection_order(inter_soft):
        if len(selected) >= inter_min:
            break
        selected.append(_gic_id(item[0]))
    for item in _sensitivity_selection_order(eligible):
        if len(selected) >= free_budget:
            break
        label_id = _gic_id(item[0])
        if label_id in selected:
            continue
        relative = item[3]
        if relative >= fit_relative_threshold or len(selected) < required_fit:
            selected.append(label_id)
    return set(selected)


def _sensitivity_selection_order(
    candidates: list[tuple[str, float, float, float, str]]
) -> list[tuple[str, float, float, float, str]]:
    return sorted(candidates, key=lambda item: (_sensitivity_priority(item[0]), -item[2]))


def _sensitivity_priority(label: str) -> int:
    text = str(label)
    if _is_inter_or_special_gic(text):
        return 0
    if _is_soft_gic(text):
        return 1
    return 2


def _is_inter_or_special_gic(label: str) -> bool:
    text = label.upper()
    if " SMITH " not in text:
        return False
    special_markers = (
        "PSAN",
        "PSTO",
        "STRD",
        "BENDD",
        "FC_",
        "FRAG",
    )
    return any(marker in text for marker in special_markers)


def _is_soft_gic(label: str) -> bool:
    kind = _gic_kind(label)
    if kind in {"dihedral", "out_of_plane", "linear_bend", "ring", "mixed"}:
        return True
    text = str(label).upper()
    return any(marker in text for marker in ("PSAN", "PSTO", "TORS", "OOP", "PUCK"))


def _sensitivity_fit_reason(label: str) -> str:
    if _is_inter_or_special_gic(label):
        return "intermolecular_soft_priority"
    if _is_soft_gic(label):
        return "soft_coordinate_priority"
    return "minimum_isotopologue_coverage"


def _sensitivity_min_fit_count(
    min_fit_count: int | None,
    *,
    n_active: int,
    n_isotopologues: int,
) -> int:
    if n_active <= 0:
        return 0
    if min_fit_count is not None:
        return max(0, min(int(min_fit_count), int(n_active)))
    isotope_floor = max(int(n_isotopologues), 1)
    return min(int(n_active), max(3, isotope_floor))


def _sensitivity_predicate_sigma(
    label: str,
    *,
    relative: float = 1.0,
    fit_relative_threshold: float = 0.15,
    distance_sigma_angstrom: float,
    angle_sigma_degree: float,
    torsion_sigma_degree: float,
    soft_predicate_scale: float = 0.5,
    null_predicate_scale: float = 0.25,
) -> float:
    scale = 1.0
    if _is_soft_gic(label) or _is_inter_or_special_gic(label):
        scale *= max(float(soft_predicate_scale), 1.0e-12)
    if relative < max(float(fit_relative_threshold), 1.0e-12) * 0.1:
        scale *= max(float(null_predicate_scale), 1.0e-12)
    return scale * _sensitivity_base_sigma(
        label,
        distance_sigma_angstrom=distance_sigma_angstrom,
        angle_sigma_degree=angle_sigma_degree,
        torsion_sigma_degree=torsion_sigma_degree,
    )


def _sensitivity_fit_regularization_sigma(
    label: str,
    *,
    relative: float,
    fit_regularization_scale: float,
    distance_sigma_angstrom: float,
    angle_sigma_degree: float,
    torsion_sigma_degree: float,
) -> float:
    if fit_regularization_scale <= 0.0:
        return 0.0
    if not (_is_soft_gic(label) or _is_inter_or_special_gic(label)):
        return 0.0
    base = _sensitivity_base_sigma(
        label,
        distance_sigma_angstrom=distance_sigma_angstrom,
        angle_sigma_degree=angle_sigma_degree,
        torsion_sigma_degree=torsion_sigma_degree,
    )
    sensitivity_factor = 1.0 + max(float(relative), 0.0)
    return float(base) * float(fit_regularization_scale) * sensitivity_factor


def _sensitivity_base_sigma(
    label: str,
    *,
    distance_sigma_angstrom: float,
    angle_sigma_degree: float,
    torsion_sigma_degree: float,
) -> float:
    kind = _gic_kind(label)
    if kind == "distance":
        return float(distance_sigma_angstrom)
    if kind in {"torsion", "ring"} or _is_soft_gic(label):
        return float(np.deg2rad(torsion_sigma_degree))
    return float(np.deg2rad(angle_sigma_degree))


def suggest_parameter_classes(
    atoms: tuple[str, ...],
    gic_labels: tuple[str, ...],
    observations: tuple[IsotopologueObservation, ...] = (),
) -> tuple[ParameterClassConstraint, ...]:
    suggestions: list[ParameterClassConstraint] = list(
        _dominant_primitive_parameter_classes(atoms, gic_labels)
    )
    h_substituted = _substituted_hydrogens(atoms, observations)
    if not h_substituted:
        xh_angle_gics = tuple(
            _gic_id(label)
            for label in gic_labels
            if _gic_kind(label) == "angle" and _dominant_angle_has_h(atoms, label)
        )
        if xh_angle_gics:
            suggestions.append(ParameterClassConstraint("XH_angle_directions", xh_angle_gics, "fixed"))
    return _deduplicate_parameter_classes(tuple(suggestions))


def _dominant_primitive_parameter_classes(
    atoms: tuple[str, ...],
    gic_labels: tuple[str, ...],
    *,
    min_abs_coeff: float = 0.55,
    min_group_size: int = 2,
) -> tuple[ParameterClassConstraint, ...]:
    groups: dict[str, list[str]] = {}
    for label in gic_labels:
        dominant = _dominant_primitive(label)
        if dominant is None:
            continue
        coeff, primitive = dominant
        if coeff < min_abs_coeff:
            continue
        name = _primitive_class_name(atoms, primitive)
        if not name:
            continue
        groups.setdefault(name, []).append(_gic_id(label))
    classes = []
    for name, patterns in sorted(groups.items()):
        unique = tuple(dict.fromkeys(patterns))
        if len(unique) >= min_group_size:
            classes.append(ParameterClassConstraint(name, unique, "shared"))
    return tuple(classes)


def _deduplicate_parameter_classes(
    classes: tuple[ParameterClassConstraint, ...],
) -> tuple[ParameterClassConstraint, ...]:
    merged: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    for item in classes:
        key = (item.name, item.mode)
        if key not in merged:
            merged[key] = []
            order.append(key)
        merged[key].extend(item.patterns)
    return tuple(
        ParameterClassConstraint(name, _unique_patterns(tuple(merged[(name, mode)])), mode)
        for name, mode in order
        if _unique_patterns(tuple(merged[(name, mode)]))
    )


def _dominant_angle_has_h(atoms: tuple[str, ...], label: str) -> bool:
    dominant = _dominant_primitive(label)
    if dominant is None:
        return False
    _coeff, primitive = dominant
    kind, indices = _primitive_kind_atoms(primitive)
    return kind == "A" and any(atoms[idx - 1].upper() == "H" for idx in indices)


def _dominant_primitive(label: str) -> tuple[float, str] | None:
    terms = _primitive_terms(label)
    if not terms:
        return None
    return max(terms, key=lambda item: item[0])


def _primitive_terms(label: str) -> tuple[tuple[float, str], ...]:
    terms = []
    for value, primitive in re.findall(
        r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\*\s*"
        r"([A-Za-z][A-Za-z0-9_]*\(\s*\d+(?:\s*,\s*\d+)*\s*\))",
        str(label),
    ):
        terms.append((abs(float(value)), _canonical_primitive_expression(primitive)))
    return tuple(terms)


def _canonical_primitive_expression(primitive: str) -> str:
    kind, atoms = _primitive_kind_atoms(primitive)
    if kind == "R" and len(atoms) == 2:
        atoms = tuple(sorted(atoms))
    elif kind in {"A", "B"} and len(atoms) == 3:
        atoms = min(atoms, tuple(reversed(atoms)))
    elif kind in {"D", "T"} and len(atoms) == 4:
        atoms = min(atoms, tuple(reversed(atoms)))
    return f"{kind}({','.join(str(item) for item in atoms)})"


def _primitive_kind_atoms(primitive: str) -> tuple[str, tuple[int, ...]]:
    text = re.sub(r"\s+", "", str(primitive))
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\(([^)]*)\)", text)
    if match is None:
        return "", ()
    kind = match.group(1).upper()
    atoms = tuple(int(item) for item in match.group(2).split(",") if item)
    return kind, atoms


def _primitive_class_name(atoms: tuple[str, ...], primitive: str) -> str:
    kind, indices = _primitive_kind_atoms(primitive)
    if not indices:
        return ""
    symbols = tuple(atoms[idx - 1].upper() for idx in indices)
    if kind == "R" and len(symbols) == 2:
        pair = "".join(sorted(symbols))
        return f"{pair}_stretches"
    if kind in {"A", "B"} and len(symbols) == 3:
        endpoints = sorted((symbols[0], symbols[2]))
        return f"{endpoints[0]}{symbols[1]}{endpoints[1]}_bends"
    if kind in {"D", "T"} and len(symbols) == 4:
        forward = "".join(symbols)
        backward = "".join(reversed(symbols))
        return f"{min(forward, backward)}_torsions"
    if kind in {"U", "OOP"} and len(symbols) >= 4:
        return f"{''.join(sorted(symbols))}_oop"
    return ""


def _gic_id(label: str) -> str:
    return str(label).split(None, 1)[0]


def _unique_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for pattern in patterns:
        normalized = pattern.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def write_semiexperimental_html_report(
    path: Path,
    result: SemiexperimentalFitResult,
    request: SemiexperimentalFitRequest,
    *,
    r0_result: SemiexperimentalFitResult | None = None,
    fit_comparison: dict[str, object] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MATRIX/MORPHEUS semiexperimental geometry report</title>",
        "<style>body{font-family:Helvetica,Arial,sans-serif;margin:32px;line-height:1.35}"
        "table{border-collapse:collapse;margin:16px 0;width:100%}"
        "th,td{border:1px solid #ccc;padding:5px 7px;text-align:left;font-size:13px}"
        "th{background:#f3f3f3} code{background:#f7f7f7;padding:1px 3px}</style>",
        "</head><body>",
        "<h1>MATRIX/MORPHEUS Semiexperimental Geometry Report</h1>",
        "<h2>Publishable Summary</h2>",
        _publishable_summary_table(result),
        _fit_comparison_html(fit_comparison),
        "<h2>Diagnostics</h2>",
        "<table><tr><th>Quantity</th><th>Value</th></tr>",
        _row("RMS", f"{result.rms_MHz:.8g}"),
        _row("Iterations", str(result.iterations)),
        _row("Stationary point", result.stationary_point),
        _row("Convergence", result.diagnostics.convergence_reason),
        _row("Linear solver", result.diagnostics.linear_solver),
        _row("Robust loss", result.diagnostics.robust_loss),
        _row("Robust scale", f"{result.diagnostics.robust_scale:.8g}"),
        _row("Downweighted rows", str(result.diagnostics.robust_downweighted_observations)),
        _row(
            "Downweighted isotopologues", str(result.diagnostics.robust_downweighted_isotopologues)
        ),
        _row("Rank", str(result.diagnostics.rank)),
        _row("Condition number", f"{result.diagnostics.condition_number:.8g}"),
        _row("Observable", result.diagnostics.observable),
        _row("Components", ",".join(result.diagnostics.components)),
        "</table>",
        "<h2>Warnings</h2>",
        _diagnostic_warnings_table(result),
        "<h2>Parameter Classes</h2>",
        _classes_table(request.parameter_classes),
        "<h2>GIC Parameters</h2>",
        _parameters_table(result),
        "<h2>Structural Path: input, r0, rs and reSE</h2>",
        _structural_path_pic_table(result, r0_result),
        "<h3>rs substitution coordinates (Kraitchman)</h3>",
        _kraitchman_table(result, _fully_excluded_isotopologues(request)),
        "<h2>Final Cartesian Geometry Parameters</h2>",
        _geometry_parameters_table(result),
        "<h2>Rotational Constants</h2>",
        _rotational_constants_table(result),
        "<h2>Residuals</h2>",
        _residuals_table(result),
        "<h2>Weight and Influence Diagnostics</h2>",
        _weight_diagnostics_table(result),
        "</body></html>",
    ]
    target.write_text("\n".join(html) + "\n", encoding="utf-8")
    return target


def _fit_comparison_html(comparison: dict[str, object] | None) -> str:
    if not comparison:
        return ""
    free = dict(comparison.get("free_fit", {}))
    constrained = dict(comparison.get("constrained_fit", {}))
    model = dict(comparison.get("constraint_model", {}))
    predicates = tuple(dict(item) for item in model.get("predicates", ()))
    excluded = tuple(str(item) for item in comparison.get("excluded_rotational_constants", ()))
    rows = [
        "<h2>Free and constrained fits</h2>",
        "<p>The free fit and the constrained fit use the same retained experimental "
        "constants and the same SONIC parameterization. The second fit adds only the "
        "explicit soft-coordinate priors listed below.</p>",
        "<table><tr><th>Quantity</th><th>Free fit</th><th>Constrained fit</th></tr>",
        _comparison_html_row("Rotational RMS / MHz", free, constrained, "rotational_rms_MHz"),
        _comparison_html_row(
            "Maximum aligned displacement / A",
            free,
            constrained,
            "max_atom_displacement_A",
        ),
        _comparison_html_row("Rank", free, constrained, "rank"),
        _comparison_html_row("Condition number", free, constrained, "condition_number"),
        "</table>",
        f"<p>Acceptance limit: {float(comparison['displacement_limit_A']):.6g} A. "
        f"Explicitly excluded constants: {escape(', '.join(excluded) or 'none')}.</p>",
        "<p>Constraint model: Gaussian priors centered at the input SONIC values; "
        f"scale {float(model.get('scale', 0.0)):.6g}; "
        f"{int(model.get('count', 0))} soft-coordinate priors.</p>",
    ]
    if predicates:
        rows.append(
            "<table><tr><th>SONIC coordinate</th><th>Prior center</th>"
            "<th>Sigma</th><th>Unit</th></tr>"
        )
        for predicate in predicates:
            rows.append(
                "<tr>"
                f"<td>{escape(str(predicate.get('definition', predicate.get('label', ''))))}</td>"
                f"<td>{float(predicate.get('center', 0.0)):.8g}</td>"
                f"<td>{float(predicate.get('sigma', 0.0)):.8g}</td>"
                f"<td>{escape(str(predicate.get('unit', 'native')))}</td>"
                "</tr>"
            )
        rows.append("</table>")
    return "".join(rows)


def _comparison_html_row(
    label: str,
    free: dict[str, object],
    constrained: dict[str, object],
    key: str,
) -> str:
    return (
        f"<tr><td>{escape(label)}</td><td>{escape(str(free.get(key, '')))}</td>"
        f"<td>{escape(str(constrained.get(key, '')))}</td></tr>"
    )


def run_semiexperimental_benchmark(
    cases: tuple[SemiexperimentalBenchmarkCase, ...],
    *,
    outdir: Path | None = None,
    max_iter: int | None = None,
) -> tuple[SemiexperimentalBenchmarkRow, ...]:
    rows = []
    for case in cases:
        case_out = Path(outdir) / case.label if outdir is not None else None
        result = fit_semiexperimental_geometry(
            case.request,
            max_iter=max_iter if max_iter is not None else case.max_iter,
            step=case.step,
            damping=case.damping,
            max_step=case.max_step,
            prune_condition=case.prune_condition,
            outdir=case_out,
        )
        rot_diffs = [item.difference_MHz for item in result.rotational_constants]
        rotational_rms = (
            float(np.sqrt(np.mean(np.asarray(rot_diffs, dtype=float) ** 2))) if rot_diffs else 0.0
        )
        rows.append(
            SemiexperimentalBenchmarkRow(
                case.label,
                result.rms_MHz,
                rotational_rms,
                result.iterations,
                result.diagnostics.rank,
                result.diagnostics.incremental_rank,
                result.diagnostics.condition_number,
                result.stationary_point,
                len(result.parameters),
                len(result.kraitchman),
                result.diagnostics.gicforge_calls,
                result.diagnostics.coordinate_model_reuse_steps,
                result.diagnostics.b_projector_secant_updates,
            )
        )
    return tuple(rows)


def benchmark_csv(rows: tuple[SemiexperimentalBenchmarkRow, ...]) -> str:
    lines = [
        "label,rms,rotational_rms_MHz,iterations,rank,incremental_rank,condition_number,"
        "stationary_point,n_parameters,n_kraitchman,gicforge_calls,coordinate_model_reuse_steps,"
        "b_projector_secant_updates"
    ]
    for row in rows:
        lines.append(
            f"{row.label},{row.rms_MHz:.12g},{row.rotational_rms_MHz:.12g},{row.iterations},{row.rank},"
            f"{row.incremental_rank},{row.condition_number:.12g},{row.stationary_point},"
            f"{row.n_parameters},{row.n_kraitchman},{row.gicforge_calls},{row.coordinate_model_reuse_steps},"
            f"{row.b_projector_secant_updates}"
        )
    return "\n".join(lines) + "\n"


def semiexperimental_latex_tables(result: SemiexperimentalFitResult) -> dict[str, str]:
    return {
        "summary": _latex_publishable_summary_table(result),
        "parameters": _latex_parameter_table(result),
        "rotational_constants": _latex_rotational_constants_table(result),
        "residuals": _latex_residual_table(result),
        "weight_diagnostics": _latex_weight_diagnostics_table(result),
        "kraitchman": _latex_kraitchman_table(result),
    }


def write_semiexperimental_standalone_latex(
    path: Path,
    result: SemiexperimentalFitResult,
    *,
    request: SemiexperimentalFitRequest | None = None,
    safety: dict[str, object] | None = None,
    r0_result: SemiexperimentalFitResult | None = None,
    fit_comparison: dict[str, object] | None = None,
) -> Path:
    """Write a compact, compilable coauthor-facing summary of a reliable SE fit."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rot_diffs = np.asarray(
        [float(item.difference_MHz) for item in result.rotational_constants], dtype=float
    )
    rot_rms = float(np.sqrt(np.mean(rot_diffs * rot_diffs))) if rot_diffs.size else 0.0
    downweighted = sorted(
        {
            item.isotopologue
            for item in result.weight_diagnostics
            if item.kind == "experimental" and item.robust_weight < 0.999
        }
    )
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{tikz}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{microtype}",
        r"\title{MORPHEUS semiexperimental equilibrium structure}",
        r"\author{MATRIX/MORPHEUS}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Result and reliability}",
        (
            "The semiexperimental structure was obtained in symmetry-adapted SONIC "
            "coordinates using the coordinate-aware LINK hybrid back-transformation "
            "and robust least squares."
        ),
        r"\begin{center}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Rotational RMS / MHz & {rot_rms:.6g} \\\\",
        f"Rank / parameters & {result.diagnostics.rank}/{result.diagnostics.n_optimized_parameters} \\\\",
        f"Condition number & {result.diagnostics.condition_number:.6g} \\\\",
        f"Stationary point & {_tex(result.stationary_point)} \\\\",
        f"Convergence criterion & {_tex(result.diagnostics.convergence_reason)} \\\\",
        f"Robust loss & {_tex(result.diagnostics.robust_loss)} \\\\",
    ]
    if safety:
        lines.extend(
            [
                f"Maximum atom displacement / m\\AA & {1000.0 * float(safety['max_atom_displacement_A']):.4f} \\\\",
                f"Displacement limit / m\\AA & {1000.0 * float(safety['limit_A']):.4f} \\\\",
                f"Reliability gate & {'passed' if safety.get('reliable') else 'failed'} \\\\",
            ]
        )
    lines.extend(
        [
            f"Downweighted isotopologues & {_tex(', '.join(downweighted) or 'none')} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            *_latex_fit_comparison(fit_comparison),
            r"\section*{Molecular structure}",
            *_latex_molecule_figure(result),
            r"\section*{Structural path: input, $r_0$, $r_s$, and $r_e^{\rm SE}$}",
            (
                r"Primitive internal coordinates (PICs) provide a common, chemically "
                r"readable comparison. Posterior one-standard-deviation uncertainties "
                r"are reported for the diagnostic $r_0$ and final $r_e^{\rm SE}$ fits."
            ),
            _latex_structural_path_pic_table(result, r0_result),
            r"\subsection*{$r_s$ substitution coordinates}",
            (
                r"The $r_s$ entries are absolute principal-axis substitution coordinates "
                r"from the Kraitchman equations; they are not presented as a global PIC fit."
            ),
            r"\begin{center}",
            _latex_kraitchman_table(
                result,
                _fully_excluded_isotopologues(request) if request is not None else (),
            ),
            r"\end{center}",
            r"\section*{Final Cartesian geometry}",
            r"Coordinates are given in \AA.",
            r"\begin{center}",
            r"\begin{tabular}{rlrrr}",
            r"\toprule",
            r"Index & Atom & $x$ & $y$ & $z$ \\",
            r"\midrule",
        ]
    )
    for index, (atom, coord) in enumerate(
        zip(result.atoms, result.final_coordinates_angstrom), start=1
    ):
        lines.append(
            f"{index} & {_tex(atom)} & {coord[0]:.8f} & {coord[1]:.8f} & {coord[2]:.8f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\section*{Final SONIC geometry and uncertainties}",
            (
                r"The fitted geometry is reported in the symmetry-adapted SONIC "
                r"coordinates used by MORPHEUS. Angular coordinates and their "
                r"uncertainties are expressed in degrees; stretches are in \AA. "
                r"Posterior standard deviations include the declared experimental "
                r"weights and any active structural priors."
            ),
            r"\begingroup\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{@{}llrrll@{}}",
            r"\toprule",
            r"ID & SONIC & Value & $\sigma$ & Unit & Uncertainty \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"ID & SONIC & Value & $\sigma$ & Unit & Uncertainty \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    sonic_definitions: list[tuple[str, str, str]] = []
    reported_sonics = result.sonic_parameters or result.parameters
    for parameter in reported_sonics:
        identifier, sonic_name, definition = _sonic_latex_fields(parameter.name)
        sonic_definitions.append((identifier, sonic_name, definition))
        value, sigma, unit = _sonic_display_value(parameter.name, parameter.value, parameter.sigma)
        uncertainty = "posterior s.d."
        if parameter.sigma <= 0.0:
            prior_sigma = _matching_predicate_sigma(request, parameter.name)
            if prior_sigma is not None and prior_sigma > 0.0:
                _unused, sigma, _unused_unit = _sonic_display_value(
                    parameter.name, parameter.value, prior_sigma
                )
                uncertainty = "prior-equivalent"
            else:
                uncertainty = "fixed" if not parameter.active else "not estimable"
        sigma_text = f"{sigma:.6g}" if sigma > 0.0 else "--"
        lines.append(
            f"{_tex(identifier)} & {_tex(sonic_name)} & {value:.9g} & {sigma_text} & "
            f"{_tex(unit)} & {_tex(uncertainty)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
            r"\subsection*{Exact SONIC definitions}",
            (
                r"The following expressions define the symmetry-adapted coordinates "
                r"as linear combinations of primitive internal coordinates."
            ),
            r"\begingroup\small\sloppy",
        ]
    )
    for identifier, sonic_name, definition in sonic_definitions:
        lines.extend(
            [
                f"\\noindent\\textbf{{{_tex(identifier)} ({_tex(sonic_name)})}}\\par",
                f"{_tex(_wrappable_sonic_definition(definition))}\\par\\smallskip",
            ]
        )
    lines.extend(
        [
            r"\endgroup",
            r"\section*{Rotational constants}",
            r"All constants and differences are in MHz.",
            r"\begin{center}",
            _latex_rotational_constants_table(result),
            r"\end{center}",
            r"\end{document}",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _latex_fit_comparison(comparison: dict[str, object] | None) -> list[str]:
    if not comparison:
        return []
    free = dict(comparison.get("free_fit", {}))
    constrained = dict(comparison.get("constrained_fit", {}))
    model = dict(comparison.get("constraint_model", {}))
    exclusions = tuple(str(item) for item in comparison.get("excluded_rotational_constants", ()))
    predicates = tuple(dict(item) for item in model.get("predicates", ()))
    lines = [
        r"\section*{Free versus constrained refinement}",
        (
            "Both refinements use the same retained rotational constants and the same "
            "SONIC parameterization. The constrained result adds only explicit Gaussian "
            "priors on sensitivity-selected soft coordinates, centered at their input "
            "SONIC values."
        ),
        r"\begin{center}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Quantity & Free fit & Constrained fit \\",
        r"\midrule",
        (
            f"Rotational RMS / MHz & {float(free.get('rotational_rms_MHz', 0.0)):.6g} & "
            f"{float(constrained.get('rotational_rms_MHz', 0.0)):.6g} \\\\"
        ),
        (
            f"Maximum displacement / m\\AA & "
            f"{1000.0 * float(free.get('max_atom_displacement_A', 0.0)):.4f} & "
            f"{1000.0 * float(constrained.get('max_atom_displacement_A', 0.0)):.4f} \\\\"
        ),
        (
            f"Rank / parameters & {int(free.get('rank', 0))}/"
            f"{int(free.get('n_optimized_parameters', 0))} & "
            f"{int(constrained.get('rank', 0))}/"
            f"{int(constrained.get('n_optimized_parameters', 0))} \\\\"
        ),
        (
            f"Condition number & {float(free.get('condition_number', 0.0)):.6g} & "
            f"{float(constrained.get('condition_number', 0.0)):.6g} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        (
            f"The fixed acceptance limit is "
            f"{1000.0 * float(comparison['displacement_limit_A']):.4f}~m\\AA. "
            f"Explicitly excluded constants: {_tex(', '.join(exclusions) or 'none')}. "
            f"The regularization scale is {float(model.get('scale', 0.0)):.6g}, "
            f"corresponding to {len(predicates)} soft-coordinate priors."
        ),
    ]
    if predicates:
        lines.extend(
            [
                r"\begin{center}\small",
                r"\begin{tabular}{p{6.2cm}rrl}",
                r"\toprule",
                r"SONIC coordinate & Prior center & $\sigma$ & Unit \\",
                r"\midrule",
            ]
        )
        for predicate in predicates:
            lines.append(
                f"{_tex(str(predicate.get('definition', predicate.get('label', ''))))} & "
                f"{float(predicate.get('center', 0.0)):.8g} & "
                f"{float(predicate.get('sigma', 0.0)):.8g} & "
                f"{_tex(str(predicate.get('unit', 'native')))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    lines.append(r"\clearpage")
    return lines


def _sonic_latex_fields(label: str) -> tuple[str, str, str]:
    text = str(label).strip()
    tokens = text.split()
    identifier = tokens[0] if tokens else "SONIC"
    name_match = re.search(
        r"\b(A(?:Str|Ang|Tor|Oop|Bend|Lin|Puck|Frag)[A-Za-z0-9_-]*)\b",
        text,
        flags=re.IGNORECASE,
    )
    sonic_name = name_match.group(1) if name_match else identifier
    bracket = re.search(r"\[\s*(.*?)\s*\]", text)
    if bracket:
        definition = bracket.group(1)
    else:
        definition = re.sub(r"^\S+(?:\s+GICForge|\s+SMITH)?\s*", "", text)
        definition = re.sub(r"\birrep=\S+\s*", "", definition)
    definition = re.sub(
        r"(?<![A-Za-z])[-+]?\d+\.\d{6,}",
        _format_signed_sonic_coefficient,
        definition,
    )
    return identifier, sonic_name, definition or "--"


def _format_signed_sonic_coefficient(match: re.Match[str]) -> str:
    raw = match.group(0)
    formatted = f"{float(raw):.6g}"
    if raw.startswith("+") and not formatted.startswith("+"):
        return "+" + formatted
    return formatted


def _wrappable_sonic_definition(definition: str) -> str:
    text = re.sub(r"(?<![Ee])\+", " + ", str(definition))
    return re.sub(r"(?<![Ee])-(?=\d)", " - ", text)


def _sonic_display_value(label: str, value: float, sigma: float) -> tuple[float, float, str]:
    lower = str(label).casefold()
    angular_markers = (
        "aang",
        "ator",
        "aoop",
        "abend",
        "alin",
        "torsion",
        "out_of_plane",
        "angle",
    )
    if any(marker in lower for marker in angular_markers):
        factor = 180.0 / np.pi
        return float(value) * factor, float(sigma) * factor, "deg"
    stretch_markers = ("astr", "stretch", "hbond_distance", "frag_dist")
    if any(marker in lower for marker in stretch_markers):
        return float(value), float(sigma), "angstrom"
    return float(value), float(sigma), "native"


def _matching_predicate_sigma(
    request: SemiexperimentalFitRequest | None,
    label: str,
) -> float | None:
    if request is None:
        return None
    lowered = str(label).casefold()
    identifier = lowered.split()[0] if lowered.split() else lowered
    matches = [
        float(predicate.sigma)
        for predicate in request.qm_predicates
        if predicate.label_pattern.casefold() in lowered
        or predicate.label_pattern.casefold() == identifier
    ]
    return min(matches) if matches else None


def _latex_molecule_figure(result: SemiexperimentalFitResult) -> list[str]:
    coords = np.asarray(result.final_coordinates_angstrom, dtype=float)
    centered = coords - np.mean(coords, axis=0)
    projected, depth = _best_molecular_projection(centered)
    span = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1])), 1.0)
    projected *= 9.0 / span
    topology = _topology_lock(result.atoms, coords, validate_contacts=False)
    colors = {
        "H": "white",
        "C": "carbon",
        "N": "nitrogen",
        "O": "oxygen",
        "F": "green!65!black",
        "S": "yellow!75!orange",
        "CL": "green!55!black",
    }
    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\definecolor{carbon}{RGB}{75,75,78}",
        r"\definecolor{oxygen}{RGB}{210,45,40}",
        r"\definecolor{nitrogen}{RGB}{45,80,190}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
    ]
    for left, right in topology.bonds:
        x1, y1 = projected[left]
        x2, y2 = projected[right]
        lines.append(
            f"\\draw[line width=1.1pt,gray!70] ({x1:.4f},{y1:.4f}) -- ({x2:.4f},{y2:.4f});"
        )
    for index in np.argsort(depth):
        atom = str(result.atoms[int(index)])
        x, y = projected[int(index)]
        color = colors.get(atom.upper(), "orange!70")
        radius = "2.2pt" if atom.upper() == "H" else "4.2pt"
        lines.append(
            f"\\filldraw[fill={color},draw=black,line width=0.35pt] "
            f"({x:.4f},{y:.4f}) circle ({radius});"
        )
        lines.append(
            f"\\node[font=\\scriptsize,anchor=center,inner sep=0.5pt] "
            f"at ({_molecule_label_position(projected, depth, int(index))[0]:.4f},"
            f"{_molecule_label_position(projected, depth, int(index))[1]:.4f}) "
            f"{{{int(index) + 1}}};"
        )
    lines.extend(
        [
            r"\end{tikzpicture}",
            (
                r"\caption{Final semiexperimental structure. Atom numbers correspond "
                r"to the Cartesian and SONIC definitions below; carbon is gray, "
                r"oxygen red and hydrogen white.}"
            ),
            r"\end{figure}",
        ]
    )
    return lines


def _best_molecular_projection(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic oblique view that reduces projected atom overlaps."""

    best_score = -1.0
    best_projected: np.ndarray | None = None
    best_depth: np.ndarray | None = None
    for elevation in (0.35, 0.60, 0.82):
        radial = float(np.sqrt(max(1.0 - elevation * elevation, 0.0)))
        for azimuth in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            direction = np.array(
                [radial * np.cos(azimuth), radial * np.sin(azimuth), elevation],
                dtype=float,
            )
            reference = (
                np.array([0.0, 0.0, 1.0])
                if abs(float(direction[2])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            axis_x = np.cross(direction, reference)
            axis_x /= max(float(np.linalg.norm(axis_x)), np.finfo(float).eps)
            axis_y = np.cross(direction, axis_x)
            candidate = centered @ np.stack((axis_x, axis_y)).T
            span = max(float(np.ptp(candidate[:, 0])), float(np.ptp(candidate[:, 1])), 1.0e-12)
            normalized = candidate / span
            pairwise = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
            pairwise += np.eye(len(candidate)) * 1.0e6
            score = float(np.min(pairwise))
            if score > best_score:
                best_score = score
                best_projected = candidate
                best_depth = centered @ direction
    if best_projected is None or best_depth is None:
        return centered[:, :2], centered[:, 2]
    return best_projected, best_depth


def _molecule_label_position(
    projected: np.ndarray,
    depth: np.ndarray,
    index: int,
) -> tuple[float, float]:
    point = np.asarray(projected[index], dtype=float)
    norm = float(np.linalg.norm(point))
    radial = point / norm if norm > 1.0e-12 else np.array([1.0, 0.0])
    tangent = np.array([-radial[1], radial[0]])
    depth_sign = 1.0 if float(depth[index]) >= 0.0 else -1.0
    label = point + 0.16 * radial + 0.07 * depth_sign * tangent
    return float(label[0]), float(label[1])


def _preview_warnings(
    labels: tuple[str, ...],
    suggestions: tuple[ParameterClassConstraint, ...],
) -> tuple[str, ...]:
    warnings = []
    if not labels:
        warnings.append("No non-redundant GIC labels were generated")
    if not suggestions:
        warnings.append("No automatic parameter-class suggestion was found")
    return tuple(warnings)


def _preview_rows(
    labels: tuple[str, ...],
    suggestions: tuple[ParameterClassConstraint, ...],
    fixed_parameters: tuple[str, ...] = (),
) -> tuple[SemiexperimentalGICPreviewRow, ...]:
    rows = []
    for label in labels:
        assigned = next(
            (
                item.name
                for item in suggestions
                if any(pattern.lower() in label.lower() for pattern in item.patterns)
            ),
            "",
        )
        state = (
            "fixed_by_input"
            if any(pattern.lower() in label.lower() for pattern in fixed_parameters)
            else "active"
        )
        rows.append(
            SemiexperimentalGICPreviewRow(
                label, _gic_kind(label), _gic_atoms(label), assigned, state
            )
        )
    return tuple(rows)


def _gic_kind(label: str) -> str:
    # Native SMITH labels carry the coordinate family in the SONIC name even
    # when the compact report omits the expanded primitive expression.
    name_kind_patterns = (
        ("bond", r"\b[A-Za-z0-9_]*Str\d+\b"),
        ("angle", r"\b[A-Za-z0-9_]*(?:Ang|Bend)\d+\b"),
        ("dihedral", r"\b[A-Za-z0-9_]*Tor\d+\b"),
        ("out_of_plane", r"\b[A-Za-z0-9_]*Oop\d+\b"),
        ("linear_bend", r"\b[A-Za-z0-9_]*Lin\d+\b"),
        ("ring", r"\b[A-Za-z0-9_]*Puck\d+\b"),
    )
    for kind, pattern in name_kind_patterns:
        if re.search(pattern, label, flags=re.IGNORECASE):
            return kind
    for kind, markers in _GIC_KIND_MARKERS.items():
        if any(marker in label for marker in markers):
            return kind
    if "ring" in label.lower():
        return "ring"
    return "mixed"


def _gic_atoms(label: str) -> tuple[int, ...]:
    atoms = []
    for markers in _GIC_KIND_MARKERS.values():
        for marker in markers:
            start = 0
            while True:
                pos = label.find(marker, start)
                if pos < 0:
                    break
                end = label.find(")", pos)
                if end < 0:
                    break
                atoms.extend(
                    int(part.strip())
                    for part in label[pos + len(marker) : end].split(",")
                    if part.strip().lstrip("-").isdigit() and int(part.strip()) > 0
                )
                start = end + 1
    return tuple(sorted(set(atoms)))


_GIC_KIND_MARKERS = {
    "bond": ("R(", "B(", "Bond(", "Stretch(", "bond("),
    "angle": ("A(", "Angle(", "Bend(", "angle("),
    "dihedral": ("D(", "Dihedral(", "Torsion(", "dihedral("),
    "out_of_plane": ("U(", "out_of_plane("),
    "linear_bend": ("L(", "Linear(", "LinearBend(", "linear_bend("),
}




def _substituted_hydrogens(
    atoms: tuple[str, ...], observations: tuple[IsotopologueObservation, ...]
) -> set[int]:
    result = set()
    for obs in observations:
        for atom_index in obs.substitutions:
            if 1 <= atom_index <= len(atoms) and atoms[atom_index - 1].upper() == "H":
                result.add(atom_index)
    return result










def _row(key: str, value: str) -> str:
    return f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"


def _classes_table(classes: tuple[ParameterClassConstraint, ...]) -> str:
    if not classes:
        return "<p>No parameter classes.</p>"
    rows = ["<table><tr><th>Name</th><th>Mode</th><th>Patterns</th></tr>"]
    for item in classes:
        rows.append(
            f"<tr><td>{escape(item.name)}</td><td>{escape(item.mode)}</td>"
            f"<td><code>{escape('|'.join(item.patterns))}</code></td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _diagnostic_warnings_table(result: SemiexperimentalFitResult) -> str:
    warnings = _semiexp_warning_rows(
        result.diagnostics,
        (),
        result.parameters,
        result.geometry_parameters,
        None,
        None,
        None,
    )
    if not warnings:
        return "<p>No diagnostic warnings.</p>"
    rows = ["<table><tr><th>Severity</th><th>Code</th><th>Message</th><th>Context</th></tr>"]
    for item in warnings:
        rows.append(
            f"<tr><td>{escape(item.severity)}</td><td>{escape(item.code)}</td>"
            f"<td>{escape(item.message)}</td><td><code>{escape(item.context)}</code></td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _publishable_summary_table(result: SemiexperimentalFitResult) -> str:
    warnings = _semiexp_warning_rows(
        result.diagnostics,
        (),
        result.parameters,
        result.geometry_parameters,
        None,
        None,
        None,
        weight_diagnostics=result.weight_diagnostics,
    )
    severe = sum(1 for item in warnings if item.severity == "severe")
    warning = sum(1 for item in warnings if item.severity == "warning")
    info = sum(1 for item in warnings if item.severity == "info")
    return "\n".join(
        [
            "<table><tr><th>Quantity</th><th>Value</th></tr>",
            _row("RMS / MHz", f"{result.rms_MHz:.8g}"),
            _row("Weighted RMS", f"{result.diagnostics.weighted_rms:.8g}"),
            _row("Reduced chi-square", f"{result.diagnostics.reduced_chi_square:.8g}"),
            _row("Rank / parameters", f"{result.diagnostics.rank}/{result.diagnostics.n_optimized_parameters}"),
            _row("Condition number", f"{result.diagnostics.condition_number:.8g}"),
            _row("Stationary point", result.stationary_point),
            _row("Warnings", f"{severe} severe, {warning} warning, {info} info"),
            "</table>",
        ]
    )


def _parameters_table(result: SemiexperimentalFitResult) -> str:
    rows = [
        "<table><tr><th>Name</th><th>Value</th><th>Sigma</th><th>Active</th><th>Class</th></tr>"
    ]
    for item in result.parameters:
        rows.append(
            f"<tr><td>{escape(item.name)}</td><td>{item.value:.10g}</td><td>{item.sigma:.10g}</td>"
            f"<td>{int(item.active)}</td><td>{escape(item.parameter_class)}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _pic_key(item) -> tuple[str, tuple[int, ...]]:
    return str(item.kind).upper(), tuple(int(index) for index in item.atom_indices)


def _pic_value_sigma_unit(item, *, initial: bool = False) -> tuple[float | None, float | None, str]:
    if item.value_angstrom is not None:
        value = item.initial_value_angstrom if initial else item.value_angstrom
        return value, None if initial else item.sigma_angstrom, "A"
    if item.value_degree is not None:
        value = item.initial_value_degree if initial else item.value_degree
        return value, None if initial else item.sigma_degree, "degree"
    value = item.initial_value if initial else item.value
    return value, None if initial else item.sigma, item.unit or "native"


def _structural_path_pic_table(
    result: SemiexperimentalFitResult,
    r0_result: SemiexperimentalFitResult | None,
) -> str:
    """Compare all structures in one stable PIC labelling convention."""

    final_by_key = {_pic_key(item): item for item in result.geometry_parameters}
    r0_by_key = (
        {} if r0_result is None else {_pic_key(item): item for item in r0_result.geometry_parameters}
    )
    ordered = list(final_by_key)
    ordered.extend(key for key in r0_by_key if key not in final_by_key)
    if not ordered:
        return "<p>No primitive-coordinate structural path is available.</p>"
    rows = [
        "<table><tr><th>PIC</th><th>Atoms</th><th>Input</th>"
        "<th>r0</th><th>sigma(r0)</th><th>reSE</th><th>sigma(reSE)</th><th>Unit</th></tr>"
    ]
    for key in ordered:
        final = final_by_key.get(key)
        r0 = r0_by_key.get(key)
        reference = final or r0
        assert reference is not None
        initial_value, _unused, initial_unit = _pic_value_sigma_unit(reference, initial=True)
        r0_value, r0_sigma, r0_unit = (
            (None, None, initial_unit) if r0 is None else _pic_value_sigma_unit(r0)
        )
        final_value, final_sigma, final_unit = (
            (None, None, initial_unit) if final is None else _pic_value_sigma_unit(final)
        )
        unit = final_unit or r0_unit or initial_unit
        rows.append(
            f"<tr><td>{escape(reference.label or reference.kind)}</td>"
            f"<td>{'-'.join(str(index) for index in reference.atom_indices)}</td>"
            f"<td>{_optional_number(initial_value)}</td>"
            f"<td>{_optional_number(r0_value)}</td><td>{_optional_number(r0_sigma)}</td>"
            f"<td>{_optional_number(final_value)}</td><td>{_optional_number(final_sigma)}</td>"
            f"<td>{escape(unit)}</td></tr>"
        )
    rows.append("</table>")
    if r0_result is None:
        rows.append("<p><em>r0 values will be populated when the diagnostic fit is attached.</em></p>")
    return "\n".join(rows)


def _latex_structural_path_pic_table(
    result: SemiexperimentalFitResult,
    r0_result: SemiexperimentalFitResult | None,
) -> str:
    final_by_key = {_pic_key(item): item for item in result.geometry_parameters}
    r0_by_key = (
        {} if r0_result is None else {_pic_key(item): item for item in r0_result.geometry_parameters}
    )
    ordered = list(final_by_key)
    ordered.extend(key for key in r0_by_key if key not in final_by_key)
    if not ordered:
        return "No primitive-coordinate structural path is available."
    lines = [
        r"\begingroup\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}llrrrrrl@{}}",
        r"\toprule",
        r"PIC & Atoms & Input & $r_0$ & $\sigma(r_0)$ & $r_e^{\rm SE}$ & $\sigma(r_e^{\rm SE})$ & Unit \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"PIC & Atoms & Input & $r_0$ & $\sigma(r_0)$ & $r_e^{\rm SE}$ & $\sigma(r_e^{\rm SE})$ & Unit \\",
        r"\midrule",
        r"\endhead",
    ]
    for key in ordered:
        final = final_by_key.get(key)
        r0 = r0_by_key.get(key)
        reference = final or r0
        assert reference is not None
        initial_value, _unused, initial_unit = _pic_value_sigma_unit(reference, initial=True)
        r0_value, r0_sigma, r0_unit = (
            (None, None, initial_unit) if r0 is None else _pic_value_sigma_unit(r0)
        )
        final_value, final_sigma, final_unit = (
            (None, None, initial_unit) if final is None else _pic_value_sigma_unit(final)
        )
        unit = final_unit or r0_unit or initial_unit
        lines.append(
            f"{_tex(reference.label or reference.kind)} & "
            f"{_tex('-'.join(str(index) for index in reference.atom_indices))} & "
            f"{_optional_latex_number(initial_value)} & {_optional_latex_number(r0_value)} & "
            f"{_optional_latex_number(r0_sigma)} & {_optional_latex_number(final_value)} & "
            f"{_optional_latex_number(final_sigma)} & {_tex(unit)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    if r0_result is None:
        lines.append(
            r"\emph{The diagnostic $r_0$ result was not attached to this standalone export.}"
        )
    return "\n".join(lines)


def _optional_number(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.8g}"


def _optional_latex_number(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.8g}"


def _geometry_parameters_table(result: SemiexperimentalFitResult) -> str:
    if not result.geometry_parameters:
        return "<p>No final topological geometry parameter table available.</p>"
    rows = [
        "<table><tr><th>Kind</th><th>Label</th><th>Atoms</th><th>Symbols</th>"
        "<th>Bond final / Angstrom</th><th>Bond initial / Angstrom</th>"
        "<th>Bond delta / Angstrom</th><th>Bond sigma / Angstrom</th>"
        "<th>Angle final / degree</th><th>Angle initial / degree</th>"
        "<th>Angle delta / degree</th><th>Angle sigma / degree</th></tr>"
    ]
    for item in result.geometry_parameters:
        rows.append(
            f"<tr><td>{escape(item.kind)}</td><td>{escape(item.label)}</td>"
            f"<td>{'-'.join(str(idx) for idx in item.atom_indices)}</td>"
            f"<td>{escape('-'.join(item.atom_symbols))}</td>"
            f"<td>{'' if item.value_angstrom is None else f'{item.value_angstrom:.8f}'}</td>"
            f"<td>{'' if item.initial_value_angstrom is None else f'{item.initial_value_angstrom:.8f}'}</td>"
            f"<td>{'' if item.delta_angstrom is None else f'{item.delta_angstrom:.8f}'}</td>"
            f"<td>{'' if item.sigma_angstrom is None else f'{item.sigma_angstrom:.8f}'}</td>"
            f"<td>{'' if item.value_degree is None else f'{item.value_degree:.6f}'}</td>"
            f"<td>{'' if item.initial_value_degree is None else f'{item.initial_value_degree:.6f}'}</td>"
            f"<td>{'' if item.delta_degree is None else f'{item.delta_degree:.6f}'}</td>"
            f"<td>{'' if item.sigma_degree is None else f'{item.sigma_degree:.6f}'}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _residuals_table(result: SemiexperimentalFitResult) -> str:
    rows = [
        "<table><tr><th>Isotopologue</th><th>Observable</th><th>Observed</th><th>Calculated</th><th>Residual</th></tr>"
    ]
    for item in result.residuals:
        rows.append(
            f"<tr><td>{escape(item.isotopologue)}</td><td>{escape(item.constant)}</td>"
            f"<td>{item.observed_equilibrium_MHz:.10g}</td><td>{item.calculated_MHz:.10g}</td>"
            f"<td>{item.residual_MHz:.10g}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _weight_diagnostics_table(result: SemiexperimentalFitResult) -> str:
    if not result.weight_diagnostics:
        return "<p>No weight diagnostics available.</p>"
    rows = [
        "<table><tr><th>Row</th><th>Kind</th><th>Label</th><th>Sigma</th>"
        "<th>Weight</th><th>Robust weight</th><th>Weighted residual</th>"
        "<th>Leverage</th><th>Studentized residual</th><th>Cook distance</th></tr>"
    ]
    for item in result.weight_diagnostics:
        rows.append(
            f"<tr><td>{item.row}</td><td>{escape(item.kind)}</td>"
            f"<td>{escape(item.isotopologue)}:{escape(item.observable)}</td>"
            f"<td>{item.sigma:.8g}</td><td>{item.base_weight:.8g}</td>"
            f"<td>{item.robust_weight:.8g}</td>"
            f"<td>{item.weighted_residual:.8g}</td><td>{item.leverage:.8g}</td>"
            f"<td>{item.studentized_residual:.8g}</td>"
            f"<td>{item.cooks_distance:.8g}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _rotational_constants_table(result: SemiexperimentalFitResult) -> str:
    if not result.rotational_constants:
        return "<p>No rotational-constant comparison available.</p>"
    rows = [
        "<table><tr><th>Isotopologue</th><th>Component</th>"
        "<th>Corrected experimental / MHz</th><th>Calculated / MHz</th><th>Difference / MHz</th></tr>"
    ]
    for item in result.rotational_constants:
        rows.append(
            f"<tr><td>{escape(item.isotopologue)}</td><td>{escape(item.component)}</td>"
            f"<td>{item.corrected_experimental_MHz:.10g}</td><td>{item.calculated_MHz:.10g}</td>"
            f"<td>{item.difference_MHz:.10g}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _kraitchman_table(
    result: SemiexperimentalFitResult,
    excluded_isotopologues: tuple[str, ...] = (),
) -> str:
    retained = tuple(
        item for item in result.kraitchman if item.isotopologue not in excluded_isotopologues
    )
    if not retained:
        return "<p>No single-substitution Kraitchman comparison available.</p>"
    rows = [
        "<table><tr><th>Isotopologue</th><th>Atom</th><th>Axis</th><th>Kraitchman abs A</th><th>Fit abs A</th><th>Difference A</th></tr>"
    ]
    for item in retained:
        rows.append(
            f"<tr><td>{escape(item.isotopologue)}</td><td>{item.atom_index} {escape(item.atom)}</td>"
            f"<td>{escape(item.coordinate)}</td><td>{item.kraitchman_abs_angstrom:.10g}</td>"
            f"<td>{item.fitted_abs_angstrom:.10g}</td><td>{item.difference_angstrom:.10g}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _latex_parameter_table(result: SemiexperimentalFitResult) -> str:
    lines = [
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Parameter & Value & Sigma & Active & Class \\\\",
        "\\midrule",
    ]
    for item in result.parameters:
        lines.append(
            f"{_tex(item.name)} & {item.value:.8g} & {item.sigma:.3g} & {int(item.active)} & {_tex(item.parameter_class)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_publishable_summary_table(result: SemiexperimentalFitResult) -> str:
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        f"RMS / MHz & {result.rms_MHz:.6g} \\\\",
        f"Weighted RMS & {result.diagnostics.weighted_rms:.6g} \\\\",
        f"Reduced $\\chi^2$ & {result.diagnostics.reduced_chi_square:.6g} \\\\",
        (
            f"Rank / parameters & {result.diagnostics.rank}/"
            f"{result.diagnostics.n_optimized_parameters} \\\\"
        ),
        f"Condition number & {result.diagnostics.condition_number:.6g} \\\\",
        f"Stationary point & {_tex(result.stationary_point)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def _latex_residual_table(result: SemiexperimentalFitResult) -> str:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Isotopologue & Observable & Observed & Calculated & Residual \\\\",
        "\\midrule",
    ]
    for item in result.residuals:
        lines.append(
            f"{_tex(item.isotopologue)} & {_tex(item.constant)} & {item.observed_equilibrium_MHz:.8g} & {item.calculated_MHz:.8g} & {item.residual_MHz:.3g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_weight_diagnostics_table(result: SemiexperimentalFitResult) -> str:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Row & Type & Weight & Robust & Weighted residual & Leverage \\\\",
        "\\midrule",
    ]
    for item in result.weight_diagnostics:
        lines.append(
            f"{item.row} & {_tex(item.kind)} & {item.base_weight:.4g} & "
            f"{item.robust_weight:.4g} & {item.weighted_residual:.4g} & "
            f"{item.leverage:.4g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_rotational_constants_table(result: SemiexperimentalFitResult) -> str:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Isotopologue & Component & Corrected exp. & Calculated & Difference \\\\",
        "\\midrule",
    ]
    for item in result.rotational_constants:
        lines.append(
            f"{_tex(item.isotopologue)} & {_tex(item.component)} & "
            f"{item.corrected_experimental_MHz:.8g} & {item.calculated_MHz:.8g} & "
            f"{item.difference_MHz:.3g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_kraitchman_table(
    result: SemiexperimentalFitResult,
    excluded_isotopologues: tuple[str, ...] = (),
) -> str:
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Isotopologue & Atom & Axis & Kraitchman & Fit & Difference \\\\",
        "\\midrule",
    ]
    for item in result.kraitchman:
        if item.isotopologue in excluded_isotopologues:
            continue
        lines.append(
            f"{_tex(item.isotopologue)} & {item.atom_index} {_tex(item.atom)} & {_tex(item.coordinate)} & {item.kraitchman_abs_angstrom:.6g} & {item.fitted_abs_angstrom:.6g} & {item.difference_angstrom:.3g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _fully_excluded_isotopologues(
    request: SemiexperimentalFitRequest | None,
) -> tuple[str, ...]:
    if request is None:
        return ()
    components: dict[str, set[str]] = {}
    for exclusion in request.excluded_rotational_constants:
        label, separator, component = str(exclusion).rpartition(":")
        if separator:
            components.setdefault(label, set()).add(component.upper())
    return tuple(sorted(label for label, values in components.items() if values == {"A", "B", "C"}))


def _tex(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )
