from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import csv
import json
from io import StringIO
import os
import re
import tempfile

import numpy as np

from matrix_chem.average_atomic_masses import atomic_mass
from matrix_chem.isotopes_table import get_default_isotope, get_isotope
from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_chem.rotational import rotational_constants_MHz
from matrix_chem.structure import Structure
from matrix_chem import (
    SymmetryThresholds,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_molecular_symmetry,
    write_validation_section,
)
from matrix_chem.geometry_io import write_xyz
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.elements import atomic_number as geometry_atomic_number
from matrix_chem.topology.elements import atomic_symbol
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_core import ScientificValidationError, build_run_manifest
from matrix_core.numerics import limit_step, objective, rank_condition
from matrix_link import (
    cartesian_from_internal_jacobian,
    hybrid_internal_coordinate_step,
    nonlinear_internal_coordinate_step,
    secant_projector_update,
    should_refresh_coordinate_model,
)
from matrix_smith.definition import _survibfit_primitive_from_gic_primitive
from matrix_smith.definition import write_gicforge_build_sections
from matrix_smith.definition import (
    FrozenGIC as LinkFrozenGIC,
    GICDefinition as LinkGICDefinition,
    GICPrimitive as LinkGICPrimitive,
)
from matrix_smith.runtime import GICDefinition, GICForge, define_gics_from_cartesian, run_gicforge
from matrix_smith.runtime.gic_symmetry import SYMM_INERTIA_TOL as GIC_SYMM_INERTIA_TOL
from matrix_smith.runtime.gic_symmetry import SYMM_TOL as GIC_SYMM_TOL
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive, build_primitives, eval_primitives
from matrix_smith.survibfit.symmetry_detector import orient_coords, symmetry_elements_from_geometry
from matrix_smith.survibfit.symmetry_global import primitive_permutation

from .contracts import (
    HYDROGEN_PARAMETER_CONSTRAINT,
    IsotopologueObservation,
    ParameterClassConstraint,
    QMParameterPredicate,
    SemiexperimentalFitRequest,
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
    leverage_values as _leverage_values,
    weight_diagnostic_rows as _weight_diagnostic_rows,
    weight_diagnostics_csv as _weight_diagnostics_csv,
)
from .cartesian_coordinates import CartesianCoordinateModel, cartesian_symmetry_coordinate_model


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
TRUST_REGION_MIN_RADIUS = 1.0e-10
DAMPING_MIN = 1.0e-14
DAMPING_MAX = 1.0e12


@dataclass(frozen=True)
class SemiexperimentalParameter:
    name: str
    value: float
    sigma: float
    active: bool
    parameter_class: str = ""


@dataclass(frozen=True)
class SemiexperimentalResidual:
    isotopologue: str
    constant: str
    observed_equilibrium_MHz: float
    calculated_MHz: float
    residual_MHz: float


@dataclass(frozen=True)
class SemiexperimentalRotationalConstantComparison:
    isotopologue: str
    component: str
    corrected_experimental_MHz: float
    calculated_MHz: float
    difference_MHz: float


@dataclass(frozen=True)
class SemiexperimentalLeaveOneOutRow:
    omitted_isotopologue: str
    training_isotopologues: int
    training_rms: float
    omitted_rotational_rms_MHz: float
    omitted_rotational_max_abs_MHz: float
    cartesian_rms_shift_angstrom: float
    cartesian_max_shift_angstrom: float
    mean_parameter_sigma: float
    max_parameter_sigma: float
    iterations: int
    convergence_reason: str
    rank: int
    condition_number: float


@dataclass(frozen=True)
class SemiexperimentalGeometryParameter:
    kind: str
    label: str
    atom_indices: tuple[int, ...]
    atom_symbols: tuple[str, ...]
    value_angstrom: float | None = None
    value_degree: float | None = None
    sigma_angstrom: float | None = None
    sigma_degree: float | None = None
    value: float | None = None
    sigma: float | None = None
    unit: str = ""
    initial_value_angstrom: float | None = None
    delta_angstrom: float | None = None
    initial_value_degree: float | None = None
    delta_degree: float | None = None
    initial_value: float | None = None
    delta: float | None = None


@dataclass(frozen=True)
class SemiexperimentalDiagnosticWarning:
    severity: str
    code: str
    message: str
    context: str = ""


@dataclass(frozen=True)
class SemiexperimentalFitDiagnostics:
    convergence_reason: str
    objective: float
    weighted_rms: float
    reduced_chi_square: float
    rank: int
    incremental_rank: int
    condition_number: float
    damping: float
    accepted_steps: int
    rejected_steps: int
    max_iterations: int
    n_optimized_parameters: int
    observable: str
    components: tuple[str, ...]
    planar: bool
    auto_pruned_parameters: tuple[str, ...] = ()
    prune_condition_target: float = 0.0
    gicforge_calls: int = 0
    coordinate_model_reuse_steps: int = 0
    trust_radius: float = 0.0
    last_trust_ratio: float = 0.0
    last_line_search_scale: float = 0.0
    b_projector_analytic_refreshes: int = 0
    b_projector_secant_updates: int = 0
    b_projector_secant_rejections: int = 0
    last_b_projector_secant_error: float = 0.0
    parameter_scale_min: float = 1.0
    parameter_scale_max: float = 1.0
    robust_loss: str = "none"
    robust_scale: float = 0.0
    robust_downweighted_observations: int = 0
    robust_downweighted_isotopologues: int = 0
    linear_solver: str = "svd_more_hebden_trust_region"
    coordinate_model: str = "gic"
    solver: str = "adaptive_lm_trust_region"


@dataclass(frozen=True)
class SemiexperimentalIterationTrace:
    iteration: int
    status: str
    objective_before: float
    objective_after: float
    actual_reduction: float
    predicted_reduction: float
    trust_ratio: float
    line_search_scale: float
    damping: float
    trust_radius: float
    step_norm: float
    gradient_inf_norm: float
    rank: int
    smallest_singular_value: float
    relative_smallest_singular_value: float
    constraint_max_abs: float
    robust_scale: float
    robust_downweighted_observations: int
    robust_downweighted_isotopologues: int
    coordinate_model_age: int
    b_projector_secant_error: float
    linear_solver: str


@dataclass(frozen=True)
class MeasurementModel:
    observable: str
    components: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]
    observed: np.ndarray
    weights: np.ndarray
    n_experimental_rows: int
    planar: bool
    experimental_row_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrimitiveLinearConstraint:
    name: str
    primitives: tuple[Primitive, ...]
    coefficients: tuple[float, ...]
    target: float
    angular: bool = False


@dataclass(frozen=True)
class GICExpressionConstraint:
    name: str
    expression: str
    target: float | None = None


@dataclass(frozen=True)
class GICExpressionDefinition:
    name: str
    expression: str


@dataclass(frozen=True)
class LineSearchResult:
    coords: np.ndarray
    q_values: np.ndarray
    objective: float
    accepted: bool
    actual_reduction: float
    predicted_reduction: float
    ratio: float
    scale: float


@dataclass(frozen=True)
class TrustRegionStep:
    step: np.ndarray
    shift: float
    on_boundary: bool
    solver: str


@dataclass(frozen=True)
class GICProjectorState:
    coords: np.ndarray
    q_values: np.ndarray
    cartesian_from_q: np.ndarray


@dataclass(frozen=True)
class SecantProjectorUpdate:
    cartesian_from_q: np.ndarray | None
    relative_error: float
    accepted: bool


@dataclass(frozen=True)
class TopologyLock:
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    adjacency: tuple[tuple[int, ...], ...]


@dataclass
class GICForgeSEBackend:
    atoms: tuple[str, ...]
    root: Path
    mode: str = "gicsym"
    fragment_mode: str = "none"
    counter: int = 0
    last_workdir: Path | None = None
    point_group: str | None = None
    definition: GICDefinition | None = None
    link_definition: LinkGICDefinition | None = None

    def model(self, coords: np.ndarray):
        if self.definition is not None:
            return self.definition.model()
        self.counter += 1
        workdir = self.root / f"iter_{self.counter:04d}"
        workdir.mkdir(parents=True, exist_ok=True)
        # The native SMITH builder is the canonical MORPHEUS coordinate
        # provider for both single- and multi-fragment systems.  In particular,
        # do not silently fall back to the legacy repository-local Fortran
        # executable for ordinary covalent molecules: that made an installed
        # MORPHEUS wheel depend on the developer's MATRIX checkout.
        definition = _fragment_aware_gic_definition(
            self.atoms,
            coords,
            workdir=workdir,
            symmetrize=self.mode == "gicsym",
            fragment_mode=(
                "special-coordinates" if self.fragment_mode == "auto" else self.fragment_mode
            ),
        )
        point_group = definition.point_group
        if self.point_group is None:
            self.point_group = point_group
        elif point_group != self.point_group:
            if _is_symmetry_refinement(self.point_group, point_group):
                self.point_group = point_group
            else:
                raise ScientificValidationError(
                    f"SMITH point group changed from {self.point_group} to {point_group} in {workdir}"
                )
        self.last_workdir = workdir
        self.definition = definition
        self.link_definition = _link_definition_from_runtime(definition)
        return definition.model()


def _link_definition_from_runtime(definition: GICDefinition) -> LinkGICDefinition:
    """Adapt the frozen runtime SONIC model to LINK's typed realization contract."""

    kind_map = {
        "bond": ("STRETCH", "R"),
        "hbond_dist": ("HBOND_DISTANCE", "R"),
        "angle": ("ANGLE", "A"),
        "linear_bend": ("LINEAR_BEND", "L"),
        "dihedral": ("TORSION", "D"),
        "out_of_plane": ("OUT_OF_PLANE", "U"),
    }
    typed_primitives: list[LinkGICPrimitive] = []
    for index, primitive in enumerate(definition.primitives, start=1):
        family, function = kind_map.get(
            str(primitive.kind), (str(primitive.kind).upper(), str(primitive.kind).upper())
        )
        typed_primitives.append(
            LinkGICPrimitive(
                identifier=f"P{index:04d}",
                name=f"{family}{index:04d}",
                family=family,
                function=function,
                atoms=tuple(int(atom) + 1 for atom in primitive.atoms),
                mode=int(getattr(primitive, "mode", 0)),
                ref_atoms=tuple(int(atom) + 1 for atom in getattr(primitive, "ref", ())),
            )
        )

    family_prefixes = {
        "ASTR": "STRETCH",
        "AANG": "ANGLE",
        "ATOR": "TORSION",
        "AOOP": "OUT_OF_PLANE",
    }
    typed_gics: list[LinkFrozenGIC] = []
    matrix = np.asarray(definition.u_matrix, dtype=float)
    for column, (name, label, irrep) in enumerate(
        zip(definition.names, definition.labels, definition.irreps)
    ):
        coefficients = tuple(
            (typed_primitives[row].identifier, float(matrix[row, column]))
            for row in range(matrix.shape[0])
            if abs(float(matrix[row, column])) > 1.0e-12
        )
        upper_name = str(name).upper()
        family = next(
            (value for prefix, value in family_prefixes.items() if upper_name.startswith(prefix)),
            "GENERAL",
        )
        primitive_id = coefficients[0][0] if coefficients else ""
        typed_gics.append(
            LinkFrozenGIC(
                identifier=str(name),
                name=str(name),
                family=family,
                irrep=str(irrep),
                primitive_id=primitive_id,
                gaussian_expression=str(label),
                coefficients=coefficients,
            )
        )
    return LinkGICDefinition(
        backend="matrix-morpheus-runtime-adapter.v1",
        point_group=str(definition.point_group),
        symmetrize=bool(definition.symmetrized),
        target_rank=len(typed_gics),
        rank=len(typed_gics),
        candidate_count=len(typed_gics),
        reference_coordinates_angstrom=tuple(definition.reference_coordinates_angstrom),
        primitives=tuple(typed_primitives),
        gics=tuple(typed_gics),
    )


@dataclass(frozen=True)
class SemiexperimentalFitResult:
    atoms: tuple[str, ...]
    initial_coordinates_angstrom: np.ndarray
    final_coordinates_angstrom: np.ndarray
    parameters: tuple[SemiexperimentalParameter, ...]
    geometry_parameters: tuple[SemiexperimentalGeometryParameter, ...]
    residuals: tuple[SemiexperimentalResidual, ...]
    rotational_constants: tuple[SemiexperimentalRotationalConstantComparison, ...]
    kraitchman: tuple[KraitchmanComparison, ...]
    kraitchman_seed: KraitchmanSeedResult | None
    covariance: np.ndarray
    correlation: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    hessian_eigenvalues: np.ndarray
    stationary_point: str
    gic_labels: tuple[str, ...]
    b_matrix: np.ndarray
    iterations: int
    rms_MHz: float
    diagnostics: SemiexperimentalFitDiagnostics
    leave_one_out: tuple[SemiexperimentalLeaveOneOutRow, ...] = ()
    iteration_trace: tuple[SemiexperimentalIterationTrace, ...] = ()
    weight_diagnostics: tuple[SemiexperimentalWeightDiagnostic, ...] = ()
    manifest: Path | None = None
    sonic_parameters: tuple[SemiexperimentalParameter, ...] = ()


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
            "ring_coordinates": "SMITH U-based ring deformation and puckering coordinates",
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
        f"program = MATRIX/MORPHEUS",
        f"method = semiexperimental equilibrium-geometry least squares",
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


def _primitive_text(primitive: Primitive) -> str:
    atoms = ",".join(str(idx + 1) for idx in primitive.atoms)
    if primitive.kind == "bond":
        return f"R({atoms})"
    if primitive.kind == "angle":
        return f"A({atoms})"
    if primitive.kind == "dihedral":
        return f"D({atoms})"
    if primitive.kind == "out_of_plane":
        return f"U({atoms})"
    if primitive.kind == "linear_bend":
        return f"L({atoms},0,{primitive.mode})"
    return f"{primitive.kind}({atoms})"


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


def _gic_model(
    coords: np.ndarray,
    z_numbers: np.ndarray,
    request: SemiexperimentalFitRequest | None = None,
    backend: GICForgeSEBackend | None = None,
):
    if backend is None:
        atoms = tuple(atomic_symbol(int(z)) for z in z_numbers)
        backend = _make_gicforge_backend(atoms, outdir=None)
    return backend.model(coords)


def _resolve_max_iterations(max_iter: int | None, n_optimized_parameters: int) -> int:
    if n_optimized_parameters <= 0:
        return 0
    if max_iter is not None and max_iter > 0:
        return int(max_iter)
    return max(8, 2 * int(n_optimized_parameters))


def _validate_observation_budget(
    measurement_model: MeasurementModel,
    n_optimized_parameters: int,
    *,
    coordinate_model: str,
) -> None:
    n_rows = int(len(measurement_model.observed))
    if n_optimized_parameters <= n_rows:
        return
    n_predicate_rows = max(0, n_rows - int(measurement_model.n_experimental_rows))
    raise ScientificValidationError(
        "Underdetermined MORPHEUS refinement rejected: "
        f"{n_optimized_parameters} optimized parameters but only {n_rows} fit rows "
        f"({measurement_model.n_experimental_rows} experimental, "
        f"{n_predicate_rows} predicate). Add constraints, parameter classes, "
        "fixed coordinates, or QM predicates before running this refinement. "
        f"coordinate_model={coordinate_model}"
    )


def _make_gicforge_backend(atoms: tuple[str, ...], outdir: Path | None) -> GICForgeSEBackend:
    mode = os.environ.get("MATRIX_MORPHEUS_GICFORGE_MODE", "gicsym").strip().lower() or "gicsym"
    if mode not in {"gic", "gicsym"}:
        raise ValueError("MATRIX_MORPHEUS_GICFORGE_MODE must be 'gic' or 'gicsym'")
    fragment_mode = (
        os.environ.get("MATRIX_MORPHEUS_FRAGMENT_MODE", "auto").strip().lower().replace("_", "-")
        or "auto"
    )
    if fragment_mode in {"pseudo", "pseudobonds", "hbond", "hbonds", "h-bonds"}:
        fragment_mode = "pseudo-bonds"
    if fragment_mode in {"special", "special-coordinates", "specialcoordinates", "fragments"}:
        fragment_mode = "special-coordinates"
    if fragment_mode not in {"auto", "none", "special-coordinates", "pseudo-bonds"}:
        raise ValueError(
            "MATRIX_MORPHEUS_FRAGMENT_MODE must be 'auto', 'none', "
            "'special-coordinates', or 'pseudo-bonds'"
        )
    if outdir is None:
        root = Path(tempfile.mkdtemp(prefix="matrix_se_gicforge_"))
    else:
        root = Path(outdir) / "gicforge_iterations"
        root.mkdir(parents=True, exist_ok=True)
    return GICForgeSEBackend(atoms=atoms, root=root, mode=mode, fragment_mode=fragment_mode)


def _fragment_aware_gic_definition(
    atoms: tuple[str, ...],
    coords: np.ndarray,
    *,
    workdir: Path,
    symmetrize: bool,
    fragment_mode: str,
) -> GICDefinition:
    source = workdir / "molecule.xyz"
    xyzin = workdir / "molecule.xyzin"
    write_xyz(source, atoms, coords, comment="MATRIX/MORPHEUS fragment-aware GIC build")
    preprocess_to_enriched_xyz(
        source,
        xyzin,
        source_kind="xyz",
        symmetry_thresholds=SymmetryThresholds(
            distance_angstrom=GIC_SYMM_TOL,
            inertia_relative=GIC_SYMM_INERTIA_TOL,
        ),
    )
    write_validation_section(xyzin)
    from matrix_fragments import write_fragment_build_section

    write_fragment_build_section(xyzin)
    smith_definition = write_gicforge_build_sections(
        xyzin,
        symmetrize=symmetrize,
        fragment_mode=fragment_mode,
    )
    symmetry_diagnostics = smith_definition.symmetry_diagnostics
    provenance = {"xyzin": str(xyzin), "fragment_mode": fragment_mode}
    if symmetry_diagnostics is not None:
        provenance.update(
            {
                "smith_symmetry_method": symmetry_diagnostics.method,
                "smith_symmetry_status": symmetry_diagnostics.status,
                "smith_sign_gauge_policy": symmetry_diagnostics.sign_gauge_policy,
                "smith_path_gauge_policy": symmetry_diagnostics.path_gauge_policy,
                "smith_operation_residual_max_angstrom": (
                    f"{symmetry_diagnostics.max_operation_residual_angstrom:.12g}"
                ),
                "smith_operation_margin_min_angstrom": (
                    f"{symmetry_diagnostics.min_operation_margin_angstrom:.12g}"
                ),
                "smith_near_threshold_operations": ",".join(
                    symmetry_diagnostics.near_threshold_operations
                ),
            }
        )
    primitive_by_id = {primitive.identifier: primitive for primitive in smith_definition.primitives}
    primitive_index: dict[tuple[str, int], int] = {}
    primitives: list[Primitive] = []
    columns: list[np.ndarray] = []
    for gic in smith_definition.gics:
        column = np.zeros(len(primitives), dtype=float)
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            source_primitive = primitive_by_id[primitive_id]
            for term_index, (basis_primitive, basis_coefficient) in enumerate(
                _survibfit_basis_terms_from_gic_primitive(source_primitive)
            ):
                key = (primitive_id, term_index)
                if key not in primitive_index:
                    primitive_index[key] = len(primitives)
                    primitives.append(basis_primitive)
                    column = np.pad(column, (0, 1))
                    for index, existing in enumerate(columns):
                        columns[index] = np.pad(existing, (0, 1))
                column[primitive_index[key]] += float(coefficient) * float(basis_coefficient)
        columns.append(column)
    if not columns:
        raise ScientificValidationError(
            f"fragment-aware GIC build produced no coordinates in {xyzin}"
        )
    labels = tuple(
        f"{gic.identifier} SMITH {gic.name} irrep={gic.irrep or 'UNK'} {gic.gaussian_expression}"
        for gic in smith_definition.gics
    )
    return GICDefinition(
        atom_symbols=atoms,
        atomic_numbers=tuple(geometry_atomic_number(atom) for atom in atoms),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
        primitives=tuple(primitives),
        u_matrix=np.column_stack(columns),
        labels=labels,
        names=tuple(gic.identifier for gic in smith_definition.gics),
        irreps=tuple(gic.irrep or "UNK" for gic in smith_definition.gics),
        point_group=smith_definition.point_group,
        symmetrized=bool(symmetrize),
        symmetry_source="matrix-smith-fragment-aware",
        source="matrix-smith-native",
        gaussian_input="\n".join(gic.gaussian_expression for gic in smith_definition.gics),
        generation_workdir=str(workdir),
        provenance=provenance,
    )


def _survibfit_basis_terms_from_gic_primitive(
    source_primitive,
) -> tuple[tuple[Primitive, float], ...]:
    if source_primitive.family == "HBOND_DISTANCE" and source_primitive.function == "R":
        return (
            (
                Primitive("hbond_dist", tuple(atom - 1 for atom in source_primitive.atoms)),
                1.0,
            ),
        )
    if source_primitive.function == "RPCB":
        return tuple(
            (Primitive("angle", tuple(atom - 1 for atom in atoms)), coefficient)
            for coefficient, atoms in _linear_component_terms_from_refs(
                source_primitive.refs,
                arity=3,
                function="RPCB",
            )
        )
    if source_primitive.function == "RPCK" and source_primitive.refs:
        return tuple(
            (Primitive("dihedral", tuple(atom - 1 for atom in atoms)), coefficient)
            for coefficient, atoms in _linear_component_terms_from_refs(
                source_primitive.refs,
                arity=4,
                function="RPCK",
            )
        )
    if source_primitive.function == "FC_DIST":
        return (
            (
                Primitive(
                    "frag_dist",
                    tuple(atom - 1 for atom in source_primitive.atoms),
                    ref=tuple(atom - 1 for atom in source_primitive.ref_atoms),
                ),
                1.0,
            ),
        )
    if source_primitive.function in {"FCA_DIST", "CENTER_ATOM_DIST"}:
        return (
            (
                Primitive(
                    "frag_atom_dist",
                    tuple(atom - 1 for atom in source_primitive.atoms),
                    ref=tuple(atom - 1 for atom in source_primitive.ref_atoms),
                ),
                1.0,
            ),
        )
    if source_primitive.function == "FTRANS":
        return (
            (
                Primitive(
                    "frag_trans",
                    tuple(atom - 1 for atom in source_primitive.atoms),
                    mode=int(source_primitive.mode),
                    ref=tuple(atom - 1 for atom in source_primitive.ref_atoms),
                ),
                1.0,
            ),
        )
    if source_primitive.function == "FROT":
        return (
            (
                Primitive(
                    "frag_rot",
                    tuple(atom - 1 for atom in source_primitive.atoms),
                    mode=int(source_primitive.mode),
                    ref=tuple(atom - 1 for atom in source_primitive.ref_atoms),
                ),
                1.0,
            ),
        )
    return ((_survibfit_primitive_from_gic_primitive(source_primitive), 1.0),)


def _linear_component_terms_from_refs(
    refs: tuple[str, ...],
    *,
    arity: int,
    function: str,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    terms: list[tuple[float, tuple[int, ...]]] = []
    for ref in refs:
        if ":" not in ref:
            raise ScientificValidationError(f"invalid {function} component term: {ref}")
        coefficient_text, atom_text = ref.split(":", 1)
        atoms = tuple(int(atom) for atom in atom_text.split("-") if atom)
        if len(atoms) != arity:
            raise ScientificValidationError(f"invalid {function} component atom count: {ref}")
        terms.append((float(coefficient_text), atoms))
    return tuple(terms)


def _gicforge_sycart_coordinates(
    atoms: tuple[str, ...],
    coords: np.ndarray,
    outdir: Path | None,
) -> tuple[np.ndarray, Path]:
    if outdir is None:
        root = Path(tempfile.mkdtemp(prefix="matrix_se_sycart_"))
    else:
        root = Path(outdir) / "gicforge_sycart"
        root.mkdir(parents=True, exist_ok=True)
    workdir = root / "iter_0001"
    computation = GICForge(runner=run_gicforge).compute(
        atoms,
        coords,
        workdir=workdir,
        mode="sycart",
    )
    if computation.sycart_coordinates_angstrom is None:
        raise ScientificValidationError(f"SMITH SYCART did not produce {workdir / 'sycart.xyz'}")
    return np.asarray(computation.sycart_coordinates_angstrom, dtype=float), workdir


def _oracle_cartesian_symmetry_state(
    atoms: tuple[str, ...],
    coords: np.ndarray,
    workdir: Path,
):
    """Materialize and consume the ORACLE symmetry state used by MORPHEUS."""
    source = Path(workdir) / "morpheus_oracle_reference.xyz"
    xyzin = Path(workdir) / "morpheus_oracle_reference.xyzin"
    write_xyz(source, atoms, coords, comment="MORPHEUS Cartesian-symmetry reference")
    preprocess_to_enriched_xyz(
        source,
        xyzin,
        source_kind="xyz",
        symmetry_thresholds=SymmetryThresholds(
            distance_angstrom=GIC_SYMM_TOL,
            inertia_relative=GIC_SYMM_INERTIA_TOL,
        ),
    )
    return (
        np.asarray(read_enriched_xyz(xyzin).coordinates_angstrom, dtype=float),
        read_molecular_symmetry(xyzin),
    )


def _gicforge_point_group(provout: Path) -> str:
    text = provout.read_text(encoding="utf-8", errors="replace") if provout.exists() else ""
    match = re.search(r"Point Group from symm\.f:\s*([A-Za-z0-9]+)", text)
    return match.group(1) if match else "UNKNOWN"


def _is_symmetry_refinement(previous: str, current: str) -> bool:
    previous_order = _point_group_order(previous)
    current_order = _point_group_order(current)
    return current_order > previous_order >= 1


def _point_group_order(point_group: str) -> int:
    normalized = str(point_group).strip().lower()
    explicit = {
        "c1": 1,
        "cs": 2,
        "ci": 2,
        "c2": 2,
        "c2v": 4,
        "c2h": 4,
        "d2": 4,
        "d2h": 8,
    }
    if normalized in explicit:
        return explicit[normalized]
    match = re.match(r"([cd])(\d+)([a-z]*)$", normalized)
    if not match:
        return 0
    family, order_text, suffix = match.groups()
    nfold = int(order_text)
    if family == "c":
        return 2 * nfold if suffix in {"v", "h"} else nfold
    return 4 * nfold if suffix == "h" else 2 * nfold


def _gicforge_a1_mask(labels: tuple[str, ...]) -> np.ndarray:
    irreps = []
    for label in labels:
        match = re.search(r"\birrep=([A-Za-z0-9'\"+-]+)", label)
        irreps.append(match.group(1) if match else None)
    if not any(irrep is not None and irrep not in {"UNK", "UNASSIGNED"} for irrep in irreps):
        return np.ones(len(labels), dtype=bool)
    return np.array([irrep in {"A1", "A", "Ag", "A'"} for irrep in irreps], dtype=bool)


def _gic_model_signature(
    labels: tuple[str, ...],
) -> tuple[int, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    irrep_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for label in labels:
        irrep_match = re.search(r"\birrep=([A-Za-z0-9'\"+-]+)", label)
        irrep = irrep_match.group(1) if irrep_match else "UNK"
        family = _gic_label_family(label)
        irrep_counts[irrep] = irrep_counts.get(irrep, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
    return (len(labels), tuple(sorted(irrep_counts.items())), tuple(sorted(family_counts.items())))


def _validate_gic_model_signature(
    labels: tuple[str, ...],
    reference: tuple[int, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]],
) -> None:
    current = _gic_model_signature(labels)
    if current != reference:
        raise ScientificValidationError(
            f"SMITH coordinate model changed from {reference} to {current}"
        )


def _gic_label_family(label: str) -> str:
    name_match = re.search(r"\bGICForge\s+([A-Za-z0-9'\"+-]+)", label)
    name = name_match.group(1) if name_match else label
    if "Str" in name:
        return "bond"
    if "Ang" in name:
        return "angle"
    if "Lin" in name:
        return "linear_bend"
    if "Tor" in name:
        return "dihedral"
    if "Oop" in name:
        return "out_of_plane"
    return "gic"


def _gic_values(prims: object, u_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    return u_matrix.T @ eval_primitives(prims, coords)


def _active_mask(
    labels: tuple[str, ...],
    fixed: tuple[str, ...],
    parameter_classes: tuple[ParameterClassConstraint, ...] = (),
) -> np.ndarray:
    mask = []
    fixed_l = tuple(item.lower() for item in fixed)
    fixed_classes = tuple(item for item in parameter_classes if item.mode == "fixed")
    for label in labels:
        low = label.lower()
        explicit_fixed = any(item and item in low for item in fixed_l)
        class_fixed = any(_class_matches(item, label) for item in fixed_classes)
        mask.append(not explicit_fixed and not class_fixed)
    return np.array(mask, dtype=bool)


def _gic_fixed_patterns(fixed: tuple[str, ...]) -> tuple[str, ...]:
    """Return fixed patterns that target whole GICs, not primitive coordinates."""
    return tuple(
        item
        for item in fixed
        if not _is_hydrogen_parameter_constraint(item)
        and not _is_linear_constraint_pattern(item)
        and not _is_gic_expression_constraint_pattern(item)
        and not _is_gaussian_gic_definition_record(item)
        and not _primitives_from_fixed_pattern(item)
    )


def _fixed_primitives_from_patterns(fixed: tuple[str, ...]) -> tuple[Primitive, ...]:
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()
    for item in fixed:
        for primitive in _primitives_from_fixed_pattern(item):
            key = _primitive_constraint_key(primitive)
            if key in seen:
                continue
            primitives.append(primitive)
            seen.add(key)
    return tuple(primitives)


_FRAGMENT_PRIMITIVE_KINDS = {"frag_dist", "frag_atom_dist", "frag_trans", "frag_rot"}
_INTERNAL_PRIMITIVE_KINDS = {"bond", "angle", "dihedral", "linear_bend", "out_of_plane"}


def _fragment_atom_sets_from_primitives(
    primitives: object,
) -> tuple[frozenset[int], ...]:
    fragments: list[frozenset[int]] = []
    for primitive in primitives:
        if not isinstance(primitive, Primitive) or primitive.kind not in _FRAGMENT_PRIMITIVE_KINDS:
            continue
        atoms = frozenset(int(atom) for atom in primitive.atoms)
        if len(atoms) > 1:
            fragments.append(atoms)
        ref = frozenset(int(atom) for atom in primitive.ref)
        if primitive.kind in {"frag_dist", "frag_trans", "frag_rot"} and len(ref) > 1:
            fragments.append(ref)
    return _merged_fragment_sets(tuple(fragments))


def _merged_fragment_sets(fragments: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    merged: list[set[int]] = []
    for fragment in fragments:
        if not fragment:
            continue
        current = set(fragment)
        changed = True
        while changed:
            changed = False
            kept: list[set[int]] = []
            for existing in merged:
                if current & existing:
                    current |= existing
                    changed = True
                else:
                    kept.append(existing)
            merged = kept
        merged.append(current)
    return tuple(frozenset(fragment) for fragment in sorted(merged, key=lambda item: min(item)))


def _fragment_internal_fixed_primitives(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    available_prims: object,
    fragment_atom_sets: tuple[frozenset[int], ...],
) -> tuple[Primitive, ...]:
    if not fragment_atom_sets:
        return ()
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()
    for primitive in _constraint_primitive_pool(atoms, available_prims, coords):
        if not _is_fragment_internal_primitive(primitive, fragment_atom_sets):
            continue
        key = _primitive_constraint_key(primitive)
        if key in seen:
            continue
        primitives.append(primitive)
        seen.add(key)
    return tuple(primitives)


def _fragment_internal_gic_patterns(
    prims: tuple[Primitive, ...],
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    fragment_atom_sets: tuple[frozenset[int], ...],
) -> tuple[str, ...]:
    if not fragment_atom_sets or not len(labels):
        return ()
    patterns: list[str] = []
    matrix = np.asarray(u_matrix, dtype=float)
    for col, label in enumerate(labels):
        rows = np.flatnonzero(np.abs(matrix[:, col]) > 1.0e-12)
        if rows.size == 0:
            continue
        column_primitives = tuple(prims[int(row)] for row in rows)
        if all(
            _is_fragment_internal_primitive(primitive, fragment_atom_sets)
            for primitive in column_primitives
        ):
            patterns.append(label)
    return tuple(patterns)


def _is_fragment_internal_primitive(
    primitive: Primitive,
    fragment_atom_sets: tuple[frozenset[int], ...],
) -> bool:
    if primitive.kind not in _INTERNAL_PRIMITIVE_KINDS:
        return False
    atoms = frozenset(int(atom) for atom in primitive.atoms)
    if not atoms:
        return False
    return any(atoms <= fragment for fragment in fragment_atom_sets)


def _linear_primitive_constraints_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[PrimitiveLinearConstraint, ...]:
    constraints: list[PrimitiveLinearConstraint] = []
    for item in fixed:
        parsed = _parse_linear_constraint_pattern(item)
        if parsed is not None:
            constraints.append(parsed)
    return tuple(constraints)


def _gic_expression_constraints_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[GICExpressionConstraint, ...]:
    constraints: list[GICExpressionConstraint] = []
    definitions = _gic_expression_definitions_from_patterns(fixed)
    for item in fixed:
        parsed = _parse_gic_expression_constraint_pattern(item, definitions=definitions)
        if parsed is not None:
            constraints.append(parsed)
    return tuple(constraints)


def _gic_expression_definitions_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[GICExpressionDefinition, ...]:
    definitions: list[GICExpressionDefinition] = []
    seen: set[str] = set()
    for item in fixed:
        parsed = _parse_gic_expression_definition_pattern(item)
        if parsed is None:
            continue
        key = parsed.name.lower()
        if key in seen:
            definitions = [
                definition for definition in definitions if definition.name.lower() != key
            ]
        definitions.append(parsed)
        seen.add(key)
    return tuple(definitions)


def _hydrogen_fixed_primitives(
    atoms: list[str] | tuple[str, ...],
    available_prims: object,
    fixed: tuple[str, ...],
    *,
    coords: np.ndarray | None = None,
) -> tuple[Primitive, ...]:
    """Return a deterministic local coordinate frame for each H/D/T atom."""
    if not any(_is_hydrogen_parameter_constraint(item) for item in fixed):
        return ()
    h_atoms = {
        idx for idx, atom in enumerate(atoms) if str(atom).strip().upper() in {"H", "D", "T"}
    }
    if not h_atoms:
        return ()
    supported = {"bond", "angle", "dihedral", "out_of_plane", "linear_bend"}
    prims = [
        primitive
        for primitive in _constraint_primitive_pool(atoms, available_prims, coords)
        if primitive.kind in supported
    ]
    adjacency = _bond_adjacency(prims)
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()

    def add(primitive: Primitive | None) -> None:
        if primitive is None:
            return
        key = _primitive_constraint_key(primitive)
        if key in seen:
            return
        primitives.append(primitive)
        seen.add(key)

    for h_atom in sorted(h_atoms):
        anchors = sorted(atom for atom in adjacency.get(h_atom, ()) if atom not in h_atoms)
        if not anchors:
            # Last-resort fallback for unusual inputs: keep only directly
            # available primitives, rather than silently ignoring the H atom.
            for primitive in _hydrogen_fallback_primitives(prims, h_atom):
                add(primitive)
            continue
        anchor = anchors[0]
        add(_bond_primitive(prims, h_atom, anchor))
        linear_pair = _hydrogen_linear_pair(prims, h_atom, anchor, h_atoms)
        if linear_pair:
            for primitive in linear_pair:
                add(primitive)
            continue
        first_angle = _hydrogen_angle_primitive(prims, h_atom, anchor, h_atoms)
        add(first_angle)
        orientation = _hydrogen_orientation_primitive(prims, h_atom, anchor, h_atoms)
        if orientation is None:
            orientation = _hydrogen_angle_primitive(
                prims,
                h_atom,
                anchor,
                h_atoms,
                exclude={_primitive_constraint_key(first_angle)}
                if first_angle is not None
                else set(),
            )
        add(orientation)
    return tuple(primitives)


def _constraint_primitive_pool(
    atoms: list[str] | tuple[str, ...],
    available_prims: object,
    coords: np.ndarray | None,
) -> tuple[Primitive, ...]:
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()

    def add(primitive: Primitive) -> None:
        key = _primitive_constraint_key(primitive)
        if key in seen:
            return
        primitives.append(primitive)
        seen.add(key)

    if coords is not None:
        try:
            z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
            _continuous, graph, _ringset, _synthons, _aromaticity = build_topology_objects(
                np.asarray(coords, dtype=float),
                z_numbers,
            )
            for primitive in build_primitives(graph, np.asarray(coords, dtype=float)):
                add(primitive)
        except Exception:
            pass
    for primitive in available_prims:
        add(primitive)
    return tuple(primitives)


def _bond_adjacency(prims: list[Primitive]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for primitive in prims:
        if primitive.kind != "bond" or len(primitive.atoms) != 2:
            continue
        i, j = primitive.atoms
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    return adjacency


def _bond_primitive(prims: list[Primitive], atom_a: int, atom_b: int) -> Primitive | None:
    wanted = {atom_a, atom_b}
    return next(
        (
            primitive
            for primitive in prims
            if primitive.kind == "bond" and set(primitive.atoms) == wanted
        ),
        None,
    )


def _hydrogen_linear_pair(
    prims: list[Primitive],
    h_atom: int,
    anchor: int,
    h_atoms: set[int],
) -> tuple[Primitive, ...]:
    groups: dict[tuple[int, int, int], list[Primitive]] = {}
    for primitive in prims:
        if primitive.kind != "linear_bend" or len(primitive.atoms) != 3:
            continue
        i, j, k = primitive.atoms
        if j != anchor or h_atom not in {i, k}:
            continue
        other = k if i == h_atom else i
        key = (1 if other in h_atoms else 0, other, min(i, k))
        groups.setdefault(key, []).append(primitive)
    for _key, items in sorted(groups.items()):
        modes = {primitive.mode: primitive for primitive in items}
        if -1 in modes and -2 in modes:
            return (modes[-1], modes[-2])
    return ()


def _hydrogen_angle_primitive(
    prims: list[Primitive],
    h_atom: int,
    anchor: int,
    h_atoms: set[int],
    *,
    exclude: set[tuple[str, tuple[int, ...], int]] | None = None,
) -> Primitive | None:
    excluded = exclude or set()
    candidates = []
    for primitive in prims:
        if primitive.kind != "angle" or len(primitive.atoms) != 3:
            continue
        i, j, k = primitive.atoms
        if j != anchor or h_atom not in {i, k}:
            continue
        if _primitive_constraint_key(primitive) in excluded:
            continue
        other = k if i == h_atom else i
        candidates.append(
            (1 if other in h_atoms else 0, other, _primitive_constraint_key(primitive), primitive)
        )
    return min(candidates, default=(None, None, None, None))[3]


def _hydrogen_orientation_primitive(
    prims: list[Primitive],
    h_atom: int,
    anchor: int,
    h_atoms: set[int],
) -> Primitive | None:
    dihedrals = []
    for primitive in prims:
        if primitive.kind != "dihedral" or len(primitive.atoms) != 4:
            continue
        atoms = primitive.atoms
        terminal_h = (atoms[0] == h_atom and atoms[1] == anchor) or (
            atoms[3] == h_atom and atoms[2] == anchor
        )
        if not terminal_h:
            continue
        other_h_count = sum(1 for atom in atoms if atom != h_atom and atom in h_atoms)
        dihedrals.append((other_h_count, _primitive_constraint_key(primitive), primitive))
    if dihedrals:
        return min(dihedrals)[2]
    oops = []
    for primitive in prims:
        if primitive.kind != "out_of_plane" or len(primitive.atoms) != 4:
            continue
        atoms = primitive.atoms
        if atoms[0] != h_atom or atoms[1] != anchor:
            continue
        other_h_count = sum(1 for atom in atoms[2:] if atom in h_atoms)
        oops.append((other_h_count, _primitive_constraint_key(primitive), primitive))
    return min(oops, default=(None, None, None))[2]


def _hydrogen_fallback_primitives(prims: list[Primitive], h_atom: int) -> tuple[Primitive, ...]:
    candidates = [primitive for primitive in prims if h_atom in primitive.atoms]
    candidates.sort(
        key=lambda primitive: (
            {"bond": 0, "angle": 1, "linear_bend": 2, "dihedral": 3, "out_of_plane": 4}.get(
                primitive.kind, 9
            ),
            _primitive_constraint_key(primitive),
        )
    )
    return tuple(candidates[:3])


def _is_hydrogen_parameter_constraint(item: str) -> bool:
    text = str(item).strip().lower().replace("-", "_").replace(" ", "_")
    return text in {
        HYDROGEN_PARAMETER_CONSTRAINT,
        "hydrogen_parameters",
        "hydrogen_primitives",
        "all_hydrogen_parameters",
        "all_hydrogen_primitives",
        "@hydrogen",
    }


def _is_linear_constraint_pattern(item: str) -> bool:
    return str(item).strip().lower().startswith("linear(")


def _is_gic_expression_constraint_pattern(item: str) -> bool:
    text = str(item).strip()
    low = text.lower()
    if low.startswith(("gic(", "constraint(", "freeze(", "fixed(")):
        return True
    if _parse_gaussian_named_expression(text) is not None:
        return True
    if _parse_gaussian_expression_options(text) is not None:
        return True
    return _legacy_expression_target_split(text) is not None


def _merge_primitives(*groups: tuple[Primitive, ...]) -> tuple[Primitive, ...]:
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()
    for group in groups:
        for primitive in group:
            key = _primitive_constraint_key(primitive)
            if key in seen:
                continue
            primitives.append(primitive)
            seen.add(key)
    return tuple(primitives)


def _symmetry_expanded_fixed_primitives(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    available_prims: object,
    fixed_primitives: tuple[Primitive, ...],
    *,
    symmetry=None,
) -> tuple[Primitive, ...]:
    """Expand fixed primitive constraints to the full molecular symmetry orbit."""
    if not fixed_primitives:
        return ()
    if symmetry is not None:
        permutations = tuple(
            tuple(int(atom) - 1 for atom in operation.permutation)
            for operation in symmetry.operations
        )
    else:
        try:
            z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
            symbols = [atomic_symbol(int(z)) for z in z_numbers]
            oriented = orient_coords(coords, weights=z_numbers)
            _elements, _classes, permutations = symmetry_elements_from_geometry(
                symbols,
                oriented,
                tol=GIC_SYMM_TOL,
                max_n=6,
                tol_H=GIC_SYMM_TOL,
                ignore_isotopes=True,
                auto_max_n=True,
                inertia_tol=GIC_SYMM_INERTIA_TOL,
            )
        except Exception:
            return fixed_primitives
    if not permutations:
        return fixed_primitives

    basis = list(available_prims)
    available_by_key: dict[tuple[str, tuple[int, ...], int], Primitive] = {}
    for primitive in basis:
        available_by_key.setdefault(_primitive_constraint_key(primitive), primitive)
    basis_keys = set(available_by_key)
    for primitive in fixed_primitives:
        key = _primitive_constraint_key(primitive)
        if key not in basis_keys:
            basis_keys.add(key)
            basis.append(primitive)

    basis_positions: dict[tuple[str, tuple[int, ...], int], int] = {}
    for idx, primitive in enumerate(basis):
        basis_positions.setdefault(_primitive_constraint_key(primitive), idx)

    expanded: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()

    def add(primitive: Primitive) -> None:
        key = _primitive_constraint_key(primitive)
        if key in seen:
            return
        expanded.append(available_by_key.get(key, primitive))
        seen.add(key)

    for primitive in fixed_primitives:
        add(primitive)
        seed_key = _primitive_constraint_key(primitive)
        seed_index = basis_positions.get(seed_key)
        for mapping in permutations:
            mapped = _map_primitive_by_atoms(primitive, mapping)
            mapped_key = _primitive_constraint_key(mapped)
            candidate = available_by_key.get(mapped_key, mapped)
            if seed_index is not None:
                try:
                    perm_idx, _sign = primitive_permutation(basis, mapping)
                    permuted = basis[perm_idx[seed_index]]
                    if (
                        permuted.kind == primitive.kind
                        and _primitive_constraint_key(permuted) == mapped_key
                    ):
                        candidate = available_by_key.get(mapped_key, permuted)
                except Exception:
                    pass
            add(candidate)
    return tuple(expanded)


def _map_primitive_by_atoms(primitive: Primitive, atom_map: object) -> Primitive:
    mapped_atoms = tuple(int(atom_map[atom]) for atom in primitive.atoms)
    return Primitive(primitive.kind, mapped_atoms, mode=primitive.mode, ref=primitive.ref)


def _primitives_from_fixed_pattern(pattern: str) -> tuple[Primitive, ...]:
    frozen_primitive = _primitives_from_gaussian_current_freeze(pattern)
    if frozen_primitive:
        return frozen_primitive
    text = str(pattern).strip().lower()
    if _top_level_value_marker(text) is not None:
        return ()
    if _first_top_level_equals(text) is not None:
        return ()
    match = re.match(
        r"^(r|b|bond|stretch|a|angle|bend|d|dihedral|torsion|u|out_of_plane|l|linear|linear_bend)\(([^)]*)",
        text,
    )
    if not match:
        return ()
    kind, args_text = match.groups()
    kind = {
        "r": "bond",
        "b": "bond",
        "stretch": "bond",
        "a": "angle",
        "bend": "angle",
        "d": "dihedral",
        "torsion": "dihedral",
        "u": "out_of_plane",
        "l": "linear_bend",
        "linear": "linear_bend",
    }.get(kind, kind)
    args = [part.strip() for part in re.split(r"[,;]", args_text) if part.strip()]
    values: list[int] = []
    mode: int | None = None
    for arg in args:
        if arg.startswith("mode="):
            try:
                mode = int(arg.split("=", 1)[1])
            except ValueError:
                return ()
            continue
        try:
            values.append(int(arg))
        except ValueError:
            return ()
    if kind == "bond" and len(values) >= 2:
        atoms = tuple(value - 1 for value in values[:2])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("bond", atoms),)
    if kind == "angle" and len(values) >= 3:
        atoms = tuple(value - 1 for value in values[:3])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("angle", atoms),)
    if kind == "dihedral" and len(values) >= 4:
        atoms = tuple(value - 1 for value in values[:4])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("dihedral", atoms),)
    if kind == "out_of_plane" and len(values) >= 4:
        atoms = tuple(value - 1 for value in values[:4])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("out_of_plane", atoms),)
    if kind == "linear_bend" and len(values) >= 3:
        atoms = tuple(value - 1 for value in values[:3])
        if any(atom < 0 for atom in atoms):
            return ()
        if len(values) >= 5:
            mode = values[4]
        elif len(values) == 4 and values[3] in {-1, -2}:
            mode = values[3]
        if mode in {-1, -2}:
            return (Primitive("linear_bend", atoms, mode=mode),)
        return (
            Primitive("linear_bend", atoms, mode=-1),
            Primitive("linear_bend", atoms, mode=-2),
        )
    return ()


def _parse_linear_constraint_pattern(pattern: str) -> PrimitiveLinearConstraint | None:
    text = str(pattern).strip()
    if not _is_linear_constraint_pattern(text):
        return None
    if not text.endswith(")"):
        raise ValueError(f"Invalid linear primitive constraint: {pattern}")
    body = text[text.find("(") + 1 : -1].strip()
    if "=" not in body:
        raise ValueError(f"Linear primitive constraint needs '=': {pattern}")
    expr_text, target_text = body.rsplit("=", 1)
    terms = _parse_linear_constraint_terms(expr_text)
    if not terms:
        raise ValueError(f"Linear primitive constraint has no primitive terms: {pattern}")
    primitives = tuple(item[1] for item in terms)
    coefficients = tuple(item[0] for item in terms)
    angular = any(
        primitive.kind in {"angle", "dihedral", "out_of_plane", "linear_bend"}
        for primitive in primitives
    )
    if angular and any(primitive.kind == "bond" for primitive in primitives):
        raise ValueError(
            f"Linear primitive constraint cannot mix bond and angular primitives: {pattern}"
        )
    target = _parse_linear_constraint_target(target_text, angular=angular)
    return PrimitiveLinearConstraint(text, primitives, coefficients, target, angular)


def _parse_linear_constraint_terms(expr_text: str) -> list[tuple[float, Primitive]]:
    expr = re.sub(r"\s+", "", expr_text)
    if not expr:
        return []
    term_re = re.compile(
        r"(?P<sign>[+-]?)"
        r"(?:(?P<coeff>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*)?"
        r"(?P<kind>bond|angle|dihedral|out_of_plane|linear_bend)"
        r"\((?P<args>[^)]*)\)"
    )
    terms: list[tuple[float, Primitive]] = []
    pos = 0
    for match in term_re.finditer(expr):
        if match.start() != pos:
            raise ValueError(f"Invalid linear primitive expression near {expr[pos:]!r}")
        sign = -1.0 if match.group("sign") == "-" else 1.0
        coeff = float(match.group("coeff")) if match.group("coeff") else 1.0
        primitive_text = f"{match.group('kind')}({match.group('args')})"
        primitives = _primitives_from_fixed_pattern(primitive_text)
        if len(primitives) != 1:
            raise ValueError(
                f"Linear primitive terms must resolve to one primitive: {primitive_text}"
            )
        terms.append((sign * coeff, primitives[0]))
        pos = match.end()
    if pos != len(expr):
        raise ValueError(f"Invalid linear primitive expression near {expr[pos:]!r}")
    return terms


def _parse_linear_constraint_target(target_text: str, *, angular: bool) -> float:
    text = str(target_text).strip().lower()
    if not text:
        raise ValueError("Linear primitive constraint target cannot be empty")
    unit = ""
    if text.endswith("deg"):
        unit = "deg"
        text = text[:-3].strip()
    elif text.endswith("rad"):
        unit = "rad"
        text = text[:-3].strip()
    try:
        value = float(text.replace("d", "e").replace("D", "E"))
    except ValueError as exc:
        raise ValueError(f"Invalid linear primitive constraint target: {target_text}") from exc
    if angular and unit != "rad":
        return float(np.deg2rad(value))
    if not angular and unit in {"deg", "rad"}:
        raise ValueError("Bond linear constraints cannot use angular units")
    return value


def _parse_gic_expression_constraint_pattern(
    pattern: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    text = str(pattern).strip()
    low = text.lower()
    if _is_linear_constraint_pattern(text):
        return None
    if _primitives_from_gaussian_current_freeze(text):
        return None
    wrapper = re.match(
        r"^(gic|constraint|freeze|fixed)\((.*)\)$", text, flags=re.IGNORECASE | re.DOTALL
    )
    if wrapper:
        return _parse_gic_expression_constraint_body(
            wrapper.group(2).strip(), text, definitions=definitions
        )
    named = _parse_gaussian_named_expression(text, definitions=definitions)
    if named is not None:
        return named
    expression_options = _parse_gaussian_expression_options(text, definitions=definitions)
    if expression_options is not None:
        return expression_options
    if _legacy_expression_target_split(text) is not None:
        return _parse_gic_expression_constraint_body(text, text, definitions=definitions)
    return None


def _parse_gic_expression_definition_pattern(pattern: str) -> GICExpressionDefinition | None:
    text = str(pattern).strip()
    if _is_linear_constraint_pattern(text):
        return None
    return _parse_gaussian_named_definition(text)


def _primitives_from_gaussian_current_freeze(pattern: str) -> tuple[Primitive, ...]:
    text = str(pattern).strip()
    if not text or _top_level_value_marker(text) is not None:
        return ()
    parsed = _parse_gaussian_named_expression(text) or _parse_gaussian_expression_options(text)
    if parsed is None or parsed.target is not None:
        return ()
    return _simple_primitives_from_gic_expression(parsed.expression)


def _simple_primitives_from_gic_expression(expression: str) -> tuple[Primitive, ...]:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return ()
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        return ()
    try:
        primitive = _primitive_from_gic_expression_call(
            tree.body.func.id,
            tree.body.args,
            tree.body.keywords,
            np.zeros((10000, 3), dtype=float),
            {},
        )
    except ValueError:
        return ()
    if primitive.kind == "linear_bend" and primitive.mode not in {-1, -2}:
        return (
            Primitive("linear_bend", primitive.atoms, mode=-1),
            Primitive("linear_bend", primitive.atoms, mode=-2),
        )
    return (primitive,)


def _parse_gic_expression_constraint_body(
    body: str,
    name: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint:
    named = _parse_gaussian_named_expression(body, definitions=definitions)
    if named is not None:
        return named
    expression_options = _parse_gaussian_expression_options(body, definitions=definitions)
    if expression_options is not None:
        return expression_options
    value_split = _split_value_option_from_expression(body, definitions=definitions)
    if value_split is not None:
        expression, target = value_split
    else:
        split_at = _legacy_expression_target_split(body)
        if split_at is None:
            expression = body
            target = None
        else:
            expression = body[:split_at].strip()
            target = _parse_expression_constraint_target(
                body[split_at + 1 :],
                angular_default=_gic_expression_uses_angular_default_units(
                    expression, definitions=definitions
                ),
            )
    expression = _strip_outer_square_brackets(expression)
    if not expression:
        raise ValueError(f"GIC expression constraint has no expression: {name}")
    _validate_gic_expression(expression)
    return GICExpressionConstraint(name=name, expression=expression, target=target)


def _split_value_option_from_expression(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> tuple[str, float] | None:
    marker = _top_level_value_marker(text)
    if marker is None:
        return None
    expression = text[:marker].strip(" \t,;")
    target, has_constraint = _parse_gaussian_constraint_options(
        text[marker:],
        angular_default=_gic_expression_uses_angular_default_units(
            expression, definitions=definitions
        ),
    )
    if not has_constraint or target is None:
        raise ValueError(f"Gaussian Value= constraint needs a numeric target: {text}")
    return expression, target


def _top_level_value_marker(text: str) -> int | None:
    lower = str(text).lower()
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "(":
            round_depth += 1
        elif char == ")" and round_depth > 0:
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        if (
            round_depth == 0
            and square_depth == 0
            and brace_depth == 0
            and lower.startswith("value", idx)
        ):
            before_ok = idx == 0 or not (lower[idx - 1].isalnum() or lower[idx - 1] == "_")
            after = idx + len("value")
            probe = after
            while probe < len(text) and text[probe].isspace():
                probe += 1
            if before_ok and probe < len(text) and text[probe] == "=":
                return idx
        idx += 1
    return None


_GAUSSIAN_FREEZE_OPTIONS = {"f", "freeze", "frozen", "fixed"}
_GAUSSIAN_NONCONSTRAINT_OPTIONS = {
    "a",
    "active",
    "activate",
    "add",
    "d",
    "diff",
    "r",
    "remove",
    "inactive",
    "k",
    "kill",
    "removeall",
    "printonly",
    "modify",
    "unfreeze",
    "unfrozen",
}


def _parse_gaussian_constraint_options(
    rest: str, *, angular_default: bool = False
) -> tuple[float | None, bool]:
    text = str(rest).strip()
    if not text:
        return None, False
    value_re = re.compile(
        r"(?i)\bvalue\s*=\s*"
        r"(?P<target>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][+-]?\d+)?(?:\s*(?:deg|rad))?)"
    )
    match = value_re.search(text)
    target: float | None = None
    cleaned = text
    if match:
        target = _parse_expression_constraint_target(
            match.group("target"), angular_default=angular_default
        )
        cleaned = text[: match.start()] + text[match.end() :]
    elif text.startswith("="):
        target = _parse_expression_constraint_target(text[1:], angular_default=angular_default)
        cleaned = ""
    has_constraint = target is not None
    saw_freeze = False
    saw_nonconstraint_action = False
    cleaned = cleaned.replace(",", " ").replace(";", " ").strip()
    leftovers: list[str] = []
    for token in cleaned.split():
        low = token.lower()
        option_name = low.split("=", 1)[0]
        if option_name in _GAUSSIAN_FREEZE_OPTIONS:
            saw_freeze = True
            has_constraint = True
            continue
        if option_name in _GAUSSIAN_NONCONSTRAINT_OPTIONS:
            saw_nonconstraint_action = True
            continue
        if option_name in {"fc", "forceconstant", "stepsize", "nsteps", "min", "max"}:
            continue
        leftovers.append(token)
    if leftovers:
        raise ValueError(f"Unsupported Gaussian GIC constraint option(s): {rest}")
    if saw_nonconstraint_action and not saw_freeze:
        has_constraint = False
    return target, has_constraint


def _parse_gaussian_expression_options(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    raw = str(text).strip()
    option_at = _first_top_level_gaussian_option(raw)
    if option_at is None:
        return None
    expression = _strip_outer_square_brackets(raw[:option_at].strip(" \t,;"))
    if not expression:
        return None
    try:
        target, has_constraint = _parse_gaussian_constraint_options(
            raw[option_at:],
            angular_default=_gic_expression_uses_angular_default_units(
                expression, definitions=definitions
            ),
        )
    except ValueError:
        return None
    if not has_constraint:
        return None
    _validate_gic_expression(expression)
    return GICExpressionConstraint(name=raw, expression=expression, target=target)


def _parse_gaussian_named_expression(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    raw = str(text).strip()
    if not raw or "=" not in raw:
        return None
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return None
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return None
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
    except ValueError:
        return None
    expression = expression.strip()
    if not expression:
        raise ValueError(f"GIC expression constraint has no expression: {text}")
    _validate_gic_expression(expression)
    angular_default = _gic_expression_uses_angular_default_units(
        expression, definitions=definitions
    )
    target = None
    has_constraint = False
    option = (name_match.group("option") or "").strip()
    if option:
        try:
            target, has_constraint = _parse_gaussian_constraint_options(
                option, angular_default=angular_default
            )
        except ValueError:
            return None
    rest = rest.strip()
    if rest:
        try:
            rest_target, rest_has_constraint = _parse_gaussian_constraint_options(
                rest,
                angular_default=angular_default,
            )
        except ValueError:
            return None
        if rest_target is not None:
            target = rest_target
        has_constraint = has_constraint or rest_has_constraint
    if not has_constraint:
        return None
    return GICExpressionConstraint(
        name=name_match.group("name"), expression=expression, target=target
    )


def _parse_gaussian_named_definition(text: str) -> GICExpressionDefinition | None:
    raw = str(text).strip()
    if not raw or "=" not in raw:
        return None
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return None
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return None
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
    except ValueError:
        return None
    expression = _strip_outer_square_brackets(expression.strip())
    if not expression:
        return None
    try:
        _validate_gic_expression(expression)
        option = (name_match.group("option") or "").strip()
        angular_default = _gic_expression_uses_angular_default_units(expression)
        if option:
            _parse_gaussian_constraint_options(option, angular_default=angular_default)
        if rest.strip():
            _parse_gaussian_constraint_options(rest, angular_default=angular_default)
    except ValueError:
        return None
    return GICExpressionDefinition(name=name_match.group("name"), expression=expression)


def _is_gaussian_gic_definition_record(text: str) -> bool:
    raw = str(text).strip()
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return False
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return False
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
        _validate_gic_expression(expression)
        option = (name_match.group("option") or "").strip()
        angular_default = _gic_expression_uses_angular_default_units(expression)
        if option:
            _parse_gaussian_constraint_options(option, angular_default=angular_default)
        if rest.strip():
            _parse_gaussian_constraint_options(rest, angular_default=angular_default)
    except ValueError:
        return False
    return True


def _split_gaussian_named_expression_rhs(text: str) -> tuple[str, str]:
    right = str(text).strip()
    if not right:
        raise ValueError("Gaussian GIC expression is empty")
    if right.startswith("["):
        return _extract_outer_square_brackets(right)
    option_at = _first_top_level_gaussian_option(right)
    if option_at is None:
        return right, ""
    return right[:option_at].strip(), right[option_at:].strip()


def _legacy_expression_target_split(text: str) -> int | None:
    split_at = _last_top_level_equals(text)
    if split_at is None:
        return None
    target_text = str(text)[split_at + 1 :]
    try:
        _parse_expression_constraint_target(target_text)
    except ValueError:
        return None
    return split_at


def _first_top_level_gaussian_option(text: str) -> int | None:
    for token, start, _end in _top_level_token_spans(text):
        if _is_gaussian_option_token(token):
            return start
    return None


def _top_level_token_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    token_start: int | None = None
    for idx, char in enumerate(str(text)):
        top = depth_round == 0 and depth_square == 0 and depth_brace == 0
        if top and (char.isspace() or char in ",;"):
            if token_start is not None:
                spans.append((text[token_start:idx], token_start, idx))
                token_start = None
        else:
            if token_start is None:
                token_start = idx
        if char == "(":
            depth_round += 1
        elif char == ")" and depth_round > 0:
            depth_round -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]" and depth_square > 0:
            depth_square -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}" and depth_brace > 0:
            depth_brace -= 1
    if token_start is not None:
        spans.append((text[token_start:], token_start, len(text)))
    return spans


def _is_gaussian_option_token(token: str) -> bool:
    low = str(token).strip().lower()
    if not low:
        return False
    name = low.split("=", 1)[0]
    return name in (
        _GAUSSIAN_FREEZE_OPTIONS
        | _GAUSSIAN_NONCONSTRAINT_OPTIONS
        | {"value", "fc", "forceconstant", "stepsize", "nsteps", "min", "max"}
    )


def _first_top_level_equals(text: str) -> int | None:
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "=" and depth_round == 0 and depth_square == 0 and depth_brace == 0:
            return idx
    return None


def _last_top_level_equals(text: str) -> int | None:
    positions = []
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "=" and depth_round == 0 and depth_square == 0 and depth_brace == 0:
            positions.append(idx)
    return positions[-1] if positions else None


def _extract_outer_square_brackets(text: str) -> tuple[str, str]:
    if not text.startswith("["):
        raise ValueError(f"Expected '[' in GIC expression: {text}")
    depth = 0
    for idx, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[1:idx], text[idx + 1 :]
    raise ValueError(f"Unclosed '[' in GIC expression: {text}")


def _strip_outer_square_brackets(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("["):
        expression, rest = _extract_outer_square_brackets(stripped)
        if rest.strip():
            return stripped
        return expression.strip()
    return stripped


def _parse_expression_constraint_target(
    target_text: str, *, angular_default: bool = False
) -> float:
    text = str(target_text).strip()
    if not text:
        raise ValueError("GIC expression constraint target cannot be empty")
    lower = text.lower()
    unit = ""
    if lower.endswith("deg"):
        unit = "deg"
        text = text[:-3].strip()
    elif lower.endswith("rad"):
        unit = "rad"
        text = text[:-3].strip()
    try:
        value = float(text.replace("d", "e").replace("D", "E"))
    except ValueError as exc:
        raise ValueError(f"Invalid GIC expression constraint target: {target_text}") from exc
    if unit == "deg" or (angular_default and unit == ""):
        return float(np.deg2rad(value))
    return value


def _validate_gic_expression(expression: str) -> None:
    _parse_gic_expression_ast(expression)


def _parse_gic_expression_ast(expression: str) -> ast.Expression:
    normalized = _normalize_gaussian_gic_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid GIC expression syntax: {expression}") from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Expression
            | ast.Load
            | ast.BinOp
            | ast.UnaryOp
            | ast.Call
            | ast.Name
            | ast.Constant,
        ):
            continue
        if isinstance(node, ast.operator | ast.unaryop | ast.keyword):
            continue
        raise ValueError(f"Unsupported syntax in GIC expression: {expression}")
    return tree


def _normalize_gaussian_gic_expression(expression: str) -> str:
    """Normalize Gaussian grouping/call delimiters to Python AST syntax."""
    normalized = str(expression).strip().replace("^", "**")
    return normalized.translate(str.maketrans({"[": "(", "]": ")", "{": "(", "}": ")"}))


def _gic_expression_uses_angular_default_units(
    expression: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> bool:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return False
    definition_map = _gic_expression_definition_map(definitions)
    visiting: set[str] = set()

    def name_is_angular(name: str) -> bool:
        upper = name.upper()
        if upper in visiting:
            return False
        definition = definition_map.get(name) or definition_map.get(upper)
        if definition is None:
            return False
        visiting.add(upper)
        try:
            return expression_is_angular(definition.body)
        finally:
            visiting.remove(upper)

    def expression_is_angular(root: ast.AST) -> bool:
        has_angular = False
        call_function_nodes = {
            id(node.func)
            for node in ast.walk(root)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                if id(node) in call_function_nodes:
                    continue
                if node.id in {"pi", "PI"}:
                    continue
                if name_is_angular(node.id):
                    has_angular = True
                    continue
                return False
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                return False
            kind = node.func.id.lower()
            if kind in {
                "a",
                "angle",
                "bend",
                "d",
                "dihedral",
                "torsion",
                "u",
                "out_of_plane",
                "l",
                "linear",
                "linear_bend",
            }:
                has_angular = True
                continue
            if kind in {"r", "b", "bond", "stretch"}:
                return False
            if kind in {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "arcsin",
                "arccos",
                "arctan",
                "sqrt",
                "exp",
                "log",
                "abs",
                "min",
                "max",
            }:
                return False
            return False
        return has_angular

    return expression_is_angular(tree.body)


def _gic_expression_uses_gic_names(expression: str) -> bool:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return bool(re.search(r"\bGIC\d+\b", expression, flags=re.IGNORECASE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and re.match(r"^GIC\d+$", node.id, flags=re.IGNORECASE):
            return True
    return False


def _gic_expression_constraint_targets(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    if not constraints:
        return np.zeros(0, dtype=float)
    current = _gic_expression_constraint_values(
        constraints, coords, prims, u_matrix, labels, definitions=definitions
    )
    targets = [
        float(constraint.target) if constraint.target is not None else float(current[idx])
        for idx, constraint in enumerate(constraints)
    ]
    return np.asarray(targets, dtype=float)


def _gic_expression_constraint_values(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    values = [
        _evaluate_gic_expression(
            constraint.expression, coords, prims, u_matrix, labels, definitions=definitions
        )
        for constraint in constraints
    ]
    return np.asarray(values, dtype=float)


def _gic_expression_constraint_residuals(
    constraints: tuple[GICExpressionConstraint, ...],
    targets: np.ndarray,
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    target = np.asarray(targets, dtype=float)
    if target.size != len(constraints):
        raise ValueError("GIC expression target size does not match constraints")
    return target - _gic_expression_constraint_values(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def _gic_expression_constraint_b_matrix(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    step: float = 1.0e-5,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    if not constraints:
        return np.zeros((0, np.asarray(coords, dtype=float).size), dtype=float)
    base = np.asarray(coords, dtype=float)
    flat = base.reshape(-1)
    rows = np.zeros((len(constraints), flat.size), dtype=float)
    for idx in range(flat.size):
        delta = step * max(1.0, abs(float(flat[idx])))
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += delta
        minus[idx] -= delta
        values_plus = _gic_expression_constraint_values(
            constraints,
            plus.reshape(base.shape),
            prims,
            u_matrix,
            labels,
            definitions=definitions,
        )
        values_minus = _gic_expression_constraint_values(
            constraints,
            minus.reshape(base.shape),
            prims,
            u_matrix,
            labels,
            definitions=definitions,
        )
        rows[:, idx] = (values_plus - values_minus) / (2.0 * delta)
    return rows


def _evaluate_gic_expression(
    expression: str,
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> float:
    tree = _parse_gic_expression_ast(expression)
    q_values = _gic_values(prims, u_matrix, coords) if len(labels) else np.zeros(0, dtype=float)
    symbols = _gic_expression_symbol_values(labels, q_values)
    value = _evaluate_gic_expression_node(
        tree.body,
        np.asarray(coords, dtype=float),
        symbols,
        _gic_expression_definition_map(definitions),
    )
    if not np.isfinite(value):
        raise ValueError(f"Non-finite GIC expression value: {expression}")
    return float(value)


def _gic_expression_symbol_values(
    labels: tuple[str, ...], q_values: np.ndarray
) -> dict[str, float]:
    symbols: dict[str, float] = {"pi": float(np.pi), "PI": float(np.pi)}
    for idx, value in enumerate(np.asarray(q_values, dtype=float), start=1):
        default = f"GIC{idx:03d}"
        for key in {default, f"GIC{idx}"}:
            symbols[key] = float(value)
            symbols[key.upper()] = float(value)
        label = labels[idx - 1] if idx - 1 < len(labels) else ""
        label_match = re.match(r"\s*(GIC\d+)\b", label)
        if label_match:
            key = label_match.group(1)
            symbols[key] = float(value)
            symbols[key.upper()] = float(value)
        name_match = re.search(r"\bGICForge\s+([A-Za-z0-9_'\"]+)", label)
        if name_match:
            raw_name = name_match.group(1)
            for key in {raw_name, _safe_gic_symbol(raw_name)}:
                if key and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                    symbols[key] = float(value)
                    symbols[key.upper()] = float(value)
    return symbols


def _safe_gic_symbol(name: str) -> str:
    safe = re.sub(r"\W+", "_", str(name).strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe and safe[0].isdigit():
        safe = f"GIC_{safe}"
    return safe


def _gic_expression_definition_map(
    definitions: tuple[GICExpressionDefinition, ...],
) -> dict[str, ast.Expression]:
    mapped: dict[str, ast.Expression] = {}
    for definition in definitions:
        tree = _parse_gic_expression_ast(definition.expression)
        for key in {definition.name, definition.name.upper(), _safe_gic_symbol(definition.name)}:
            if key:
                mapped[key] = tree
                mapped[key.upper()] = tree
    return mapped


def _evaluate_gic_expression_node(
    node: ast.AST,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("GIC expressions only support numeric constants")
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return float(symbols[node.id])
        upper = node.id.upper()
        if upper in symbols:
            return float(symbols[upper])
        definition = (definitions or {}).get(node.id) or (definitions or {}).get(upper)
        if definition is not None:
            key = upper
            if key in stack:
                cycle = " -> ".join((*stack, key))
                raise ValueError(f"Cyclic GIC expression definition: {cycle}")
            value = _evaluate_gic_expression_node(
                definition.body,
                coords,
                symbols,
                definitions,
                (*stack, key),
            )
            symbols[node.id] = float(value)
            symbols[upper] = float(value)
            return float(value)
        raise ValueError(f"Unknown GIC expression symbol: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_gic_expression_node(node.operand, coords, symbols, definitions, stack)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Unsupported unary operator in GIC expression")
    if isinstance(node, ast.BinOp):
        left = _evaluate_gic_expression_node(node.left, coords, symbols, definitions, stack)
        right = _evaluate_gic_expression_node(node.right, coords, symbols, definitions, stack)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator in GIC expression")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("GIC expression functions must be named functions")
        name = node.func.id
        return _evaluate_gic_expression_call(
            name, node.args, node.keywords, coords, symbols, definitions, stack
        )
    raise ValueError("Unsupported syntax in GIC expression")


def _evaluate_gic_expression_call(
    name: str,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> float:
    lowered = name.lower()
    unary_functions = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "arcsin": "asin",
        "acos": "acos",
        "arccos": "acos",
        "atan": "atan",
        "arctan": "atan",
        "sqrt": "sqrt",
        "exp": "exp",
        "log": "log",
    }
    if lowered in unary_functions:
        if keywords or len(args) != 1:
            raise ValueError(f"Function {name} expects one positional argument")
        value = _evaluate_gic_expression_node(args[0], coords, symbols, definitions, stack)
        return float(getattr(np, unary_functions[lowered])(value))
    if lowered == "abs":
        if keywords or len(args) != 1:
            raise ValueError("Function abs expects one positional argument")
        return abs(_evaluate_gic_expression_node(args[0], coords, symbols, definitions, stack))
    if lowered in {"min", "max"}:
        if keywords or len(args) < 1:
            raise ValueError(f"Function {name} expects positional arguments")
        values = [
            _evaluate_gic_expression_node(arg, coords, symbols, definitions, stack) for arg in args
        ]
        return float(min(values) if lowered == "min" else max(values))
    if lowered in {"x", "y", "z"}:
        if keywords or len(args) != 1:
            raise ValueError(f"Cartesian function {name} expects one atom index")
        atom = _gic_expression_atom_index(args[0], coords, symbols, definitions, stack)
        axis = {"x": 0, "y": 1, "z": 2}[lowered]
        return float(coords[atom, axis])
    if lowered in {"cart", "cartesian"}:
        if keywords or len(args) != 2:
            raise ValueError(f"Cartesian function {name} expects atom index and axis")
        atom = _gic_expression_atom_index(args[0], coords, symbols, definitions, stack)
        axis = _gic_expression_cartesian_axis(args[1], coords, symbols, definitions, stack)
        return float(coords[atom, axis])
    if lowered == "dotdiff":
        if keywords or len(args) != 4:
            raise ValueError("DotDiff expects four atom indices")
        i, j, k, l = [
            _gic_expression_atom_index(arg, coords, symbols, definitions, stack) for arg in args
        ]
        return float(np.dot(coords[i] - coords[j], coords[k] - coords[l]))
    primitive = _primitive_from_gic_expression_call(
        name, args, keywords, coords, symbols, definitions, stack
    )
    return float(eval_primitives([primitive], coords)[0])


def _gic_expression_atom_index(
    node: ast.expr,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> int:
    value = _evaluate_gic_expression_node(node, coords, symbols, definitions, stack)
    index = int(round(value))
    if abs(value - index) > 1.0e-10 or index < 1 or index > len(coords):
        raise ValueError(f"Invalid atom index in GIC expression: {value}")
    return index - 1


def _gic_expression_cartesian_axis(
    node: ast.expr,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> int:
    if isinstance(node, ast.Name):
        axis_name = node.id.lower()
        if axis_name in {"x", "y", "z"}:
            return {"x": 0, "y": 1, "z": 2}[axis_name]
    value = _evaluate_gic_expression_node(node, coords, symbols, definitions, stack)
    axis = int(round(value))
    if abs(value - axis) > 1.0e-10:
        raise ValueError(f"Invalid Cartesian axis in GIC expression: {value}")
    if axis in {-1, 1}:
        return 0
    if axis in {-2, 2}:
        return 1
    if axis in {-3, 3}:
        return 2
    raise ValueError(f"Invalid Cartesian axis in GIC expression: {value}")


def _primitive_from_gic_expression_call(
    name: str,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> Primitive:
    lowered = name.lower()
    numeric_args = [
        _evaluate_gic_expression_node(arg, coords, symbols, definitions, stack) for arg in args
    ]
    int_args = [int(round(value)) for value in numeric_args]
    if any(abs(value - round(value)) > 1.0e-10 for value in numeric_args):
        raise ValueError(f"Primitive function {name} needs integer atom indices")

    def atoms(count: int) -> tuple[int, ...]:
        if len(int_args) < count:
            raise ValueError(f"Primitive function {name} needs {count} atom indices")
        result = tuple(value - 1 for value in int_args[:count])
        if any(atom < 0 for atom in result):
            raise ValueError(f"Primitive function {name} has invalid atom index")
        return result

    if lowered in {"r", "b", "bond", "stretch"}:
        if keywords or len(int_args) != 2:
            raise ValueError(f"Primitive function {name} expects two atoms")
        return Primitive("bond", atoms(2))
    if lowered in {"a", "angle", "bend"}:
        if keywords or len(int_args) != 3:
            raise ValueError(f"Primitive function {name} expects three atoms")
        return Primitive("angle", atoms(3))
    if lowered in {"d", "dihedral", "torsion"}:
        if keywords or len(int_args) != 4:
            raise ValueError(f"Primitive function {name} expects four atoms")
        return Primitive("dihedral", atoms(4))
    if lowered in {"u", "out_of_plane"}:
        if keywords or len(int_args) != 4:
            raise ValueError(f"Primitive function {name} expects four atoms")
        return Primitive("out_of_plane", atoms(4))
    if lowered in {"l", "linear", "linear_bend"}:
        mode = None
        for keyword in keywords:
            if keyword.arg != "mode":
                raise ValueError(f"Unsupported keyword for {name}: {keyword.arg}")
            mode = int(
                round(
                    _evaluate_gic_expression_node(
                        keyword.value, coords, symbols, definitions, stack
                    )
                )
            )
        if len(int_args) == 5:
            mode = int_args[4]
        elif len(int_args) == 4:
            mode = int_args[3]
        elif len(int_args) != 3:
            raise ValueError(f"Primitive function {name} expects three atoms and a mode")
        if mode not in {-1, -2}:
            raise ValueError(f"Linear-bend function {name} needs mode -1 or -2")
        return Primitive("linear_bend", atoms(3), mode=mode)
    raise ValueError(f"Unsupported GIC expression function: {name}")


def _primitive_constraint_key(primitive: Primitive) -> tuple[str, tuple[int, ...], int]:
    atoms = primitive.atoms
    if primitive.kind == "bond" and len(atoms) == 2:
        atoms = tuple(sorted(atoms))
    elif primitive.kind in {"angle", "dihedral"}:
        reverse = tuple(reversed(atoms))
        atoms = min(atoms, reverse)
    elif primitive.kind == "out_of_plane" and len(atoms) == 4:
        atoms = (atoms[1], *tuple(sorted((atoms[0], atoms[2], atoms[3]))))
    elif primitive.kind == "linear_bend" and len(atoms) == 3:
        atoms = (atoms[1], *tuple(sorted((atoms[0], atoms[2]))))
    return (primitive.kind, tuple(atoms), primitive.mode)


def _fixed_primitive_targets(
    fixed_primitives: tuple[Primitive, ...], coords: np.ndarray
) -> np.ndarray:
    if not fixed_primitives:
        return np.zeros(0, dtype=float)
    return np.asarray(
        eval_primitives(list(fixed_primitives), np.asarray(coords, dtype=float)), dtype=float
    )


def _project_fixed_primitives(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    target_values: np.ndarray,
    *,
    tolerance: float = 1.0e-11,
    max_iter: int = 10,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    """Project a candidate geometry back onto fixed primitive values."""
    if not fixed_primitives and not linear_constraints and not expression_constraints:
        return np.asarray(coords, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if target.size != len(fixed_primitives):
        raise ValueError("Fixed-primitive target size does not match constraints")
    expression_target_values = (
        np.asarray(expression_targets, dtype=float)
        if expression_targets is not None
        else _gic_expression_constraint_targets(
            expression_constraints,
            coords,
            prims,
            np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
            labels,
            definitions=expression_definitions,
        )
    )
    projected = np.asarray(coords, dtype=float).copy()
    projection_max_iter = max(max_iter, 20) if expression_constraints else max_iter
    for _ in range(projection_max_iter):
        residual = _combined_primitive_constraint_residual(
            projected,
            fixed_primitives,
            target,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_target_values,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        if not np.all(np.isfinite(residual)):
            raise ValueError("Non-finite fixed-primitive residual")
        if float(np.max(np.abs(residual))) <= tolerance:
            return projected
        b_fixed = _combined_primitive_constraint_b_matrix(
            projected,
            fixed_primitives,
            linear_constraints,
            expression_constraints=expression_constraints,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        dx = cartesian_from_internal_jacobian(b_fixed, rcond=1.0e-10) @ residual
        if not np.all(np.isfinite(dx)):
            raise ValueError("Non-finite fixed-primitive projection step")
        projected = projected + dx.reshape(projected.shape)
    residual = _combined_primitive_constraint_residual(
        projected,
        fixed_primitives,
        target,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_target_values,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=expression_definitions,
    )
    if float(np.max(np.abs(residual))) > 100.0 * tolerance:
        raise ValueError("Fixed-primitive projection did not converge")
    return projected


def _primitive_constraint_residual(
    fixed_primitives: tuple[Primitive, ...],
    current: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    for idx, primitive in enumerate(fixed_primitives):
        if primitive.kind in {"dihedral", "out_of_plane"}:
            residual[idx] = (residual[idx] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def _combined_primitive_constraint_residual(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if fixed_primitives:
        current = np.asarray(eval_primitives(list(fixed_primitives), coords), dtype=float)
        parts.append(_primitive_constraint_residual(fixed_primitives, current, fixed_targets))
    if linear_constraints:
        parts.append(_linear_constraint_residuals(linear_constraints, coords))
    if expression_constraints:
        if expression_targets is None:
            raise ValueError("GIC expression constraints need target values")
        parts.append(
            _gic_expression_constraint_residuals(
                expression_constraints,
                expression_targets,
                coords,
                prims,
                np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
                labels,
                definitions=expression_definitions,
            )
        )
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def _combined_primitive_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    rows: list[np.ndarray] = []
    if fixed_primitives:
        rows.append(b_matrix_analytic(list(fixed_primitives), coords))
    if linear_constraints:
        rows.append(_linear_constraint_b_matrix(linear_constraints, coords))
    if expression_constraints:
        rows.append(
            _gic_expression_constraint_b_matrix(
                expression_constraints,
                coords,
                prims,
                np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
                labels,
                definitions=expression_definitions,
            )
        )
    if not rows:
        return np.zeros((0, np.asarray(coords).size), dtype=float)
    return np.vstack(rows)


def _finite_difference_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    step: float = 1.0e-6,
) -> np.ndarray:
    """Finite-difference derivative of combined constraint values.

    `_combined_primitive_constraint_b_matrix` returns derivatives of the
    constrained values.  The residual is target minus current value, therefore
    the finite-difference residual derivative has the opposite sign.
    """
    base = np.asarray(coords, dtype=float)
    flat = base.reshape(-1)
    residual0 = _combined_primitive_constraint_residual(
        base,
        fixed_primitives,
        fixed_targets,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=expression_definitions,
    )
    rows = np.zeros((residual0.size, flat.size), dtype=float)
    if residual0.size == 0:
        return rows
    for idx in range(flat.size):
        delta = float(step) * max(1.0, abs(float(flat[idx])))
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += delta
        minus[idx] -= delta
        residual_plus = _combined_primitive_constraint_residual(
            plus.reshape(base.shape),
            fixed_primitives,
            fixed_targets,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        residual_minus = _combined_primitive_constraint_residual(
            minus.reshape(base.shape),
            fixed_primitives,
            fixed_targets,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        rows[:, idx] = -(residual_plus - residual_minus) / (2.0 * delta)
    return rows


def _linear_constraint_values(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    values = []
    for constraint in constraints:
        primitive_values = np.asarray(
            eval_primitives(list(constraint.primitives), coords), dtype=float
        )
        coeffs = np.asarray(constraint.coefficients, dtype=float)
        values.append(float(coeffs @ primitive_values))
    return np.asarray(values, dtype=float)


def _linear_constraint_residuals(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    current = _linear_constraint_values(constraints, coords)
    target = np.asarray([constraint.target for constraint in constraints], dtype=float)
    residual = target - current
    for idx, constraint in enumerate(constraints):
        if constraint.angular:
            residual[idx] = (residual[idx] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def _linear_constraint_b_matrix(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    rows = []
    for constraint in constraints:
        primitive_b = b_matrix_analytic(list(constraint.primitives), coords)
        coeffs = np.asarray(constraint.coefficients, dtype=float)
        rows.append(coeffs @ primitive_b)
    if not rows:
        return np.zeros((0, np.asarray(coords).size), dtype=float)
    return np.vstack(rows)


def _primitive_constrained_transform(
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    active_mask: np.ndarray,
    transform: np.ndarray,
    names: tuple[str, ...],
    fixed_primitives: tuple[Primitive, ...],
    *,
    cartesian_from_q: np.ndarray | None = None,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    labels: tuple[str, ...] = (),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Project reduced GIC increments onto the null space of fixed primitives."""
    if (
        not fixed_primitives and not linear_constraints and not expression_constraints
    ) or transform.size == 0:
        return transform, names
    active_indices = np.where(active_mask)[0]
    if not len(active_indices):
        return transform, names
    if cartesian_from_q is None:
        cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    if cartesian_from_q.size == 0:
        return transform, names
    b_fixed = _combined_primitive_constraint_b_matrix(
        coords,
        fixed_primitives,
        linear_constraints,
        expression_constraints=expression_constraints,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=expression_definitions,
    )
    constraints_active = (b_fixed @ cartesian_from_q)[:, active_indices]
    constraints_reduced = constraints_active @ transform
    constraints_reduced = _independent_rows_incremental(constraints_reduced)
    null = _nullspace(constraints_reduced)
    constrained = transform @ null
    if constrained.shape[1] == len(names):
        return constrained, names
    return constrained, tuple(
        f"constrained_{idx:03d}" for idx in range(1, constrained.shape[1] + 1)
    )


def _primitive_constrained_cartesian_transform(
    coords: np.ndarray,
    cartesian_from_q: np.ndarray,
    active_mask: np.ndarray,
    transform: np.ndarray,
    names: tuple[str, ...],
    fixed_primitives: tuple[Primitive, ...],
    *,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    expression_prims: object = (),
    expression_u_matrix: np.ndarray | None = None,
    expression_labels: tuple[str, ...] = (),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Project reduced Cartesian-basis increments onto fixed primitive constraints."""
    if (
        not fixed_primitives and not linear_constraints and not expression_constraints
    ) or transform.size == 0:
        return transform, names
    active_indices = np.where(active_mask)[0]
    if not len(active_indices):
        return transform, names
    basis = np.asarray(cartesian_from_q, dtype=float)
    if basis.size == 0:
        return transform, names
    b_fixed = _combined_primitive_constraint_b_matrix(
        coords,
        fixed_primitives,
        linear_constraints,
        expression_constraints=expression_constraints,
        prims=expression_prims,
        u_matrix=expression_u_matrix,
        labels=expression_labels,
        expression_definitions=expression_definitions,
    )
    constraints_active = (b_fixed @ basis)[:, active_indices]
    constraints_reduced = constraints_active @ transform
    constraints_reduced = _independent_rows_incremental(constraints_reduced)
    null = _nullspace(constraints_reduced)
    constrained = transform @ null
    if constrained.shape[1] == len(names):
        return constrained, names
    return constrained, tuple(
        f"constrained_{idx:03d}" for idx in range(1, constrained.shape[1] + 1)
    )


def _cartesian_from_reduced_coordinates(
    cartesian_from_q: np.ndarray,
    active_mask: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    active_indices = np.where(active_mask)[0]
    if transform.size == 0 or not len(active_indices):
        return np.zeros((np.asarray(cartesian_from_q).shape[0], 0), dtype=float)
    return np.asarray(cartesian_from_q, dtype=float)[:, active_indices] @ transform


def _active_coordinate_jacobian(jacobian: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    return np.asarray(jacobian, dtype=float)[:, np.where(active_mask)[0]]


def _nullspace(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("Null-space input must be a matrix")
    ncols = mat.shape[1]
    if ncols == 0:
        return np.zeros((0, 0), dtype=float)
    if mat.size == 0:
        return np.eye(ncols, dtype=float)
    _u, singular, vh = np.linalg.svd(mat, full_matrices=True)
    if not singular.size:
        return np.eye(ncols, dtype=float)
    tol = max(mat.shape) * np.finfo(float).eps * float(singular[0])
    rank = int(np.sum(singular > tol))
    return vh[rank:, :].T.copy()


def _independent_rows_incremental(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return mat
    row_norms = np.linalg.norm(mat, axis=1)
    scale = float(np.max(row_norms)) if row_norms.size else 0.0
    tol = max(mat.shape) * np.finfo(float).eps * max(scale, 1.0) * 100.0
    basis: list[np.ndarray] = []
    selected: list[int] = []
    for idx, row in enumerate(mat):
        residual = row.astype(float, copy=True)
        for vector in basis:
            residual -= vector * float(vector @ residual)
        norm = float(np.linalg.norm(residual))
        if norm > tol:
            basis.append(residual / norm)
            selected.append(idx)
    return mat[selected, :] if selected else np.zeros((0, mat.shape[1]), dtype=float)


def _incremental_column_rank(matrix: np.ndarray) -> int:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return 0
    col_norms = np.linalg.norm(mat, axis=0)
    scale = float(np.max(col_norms)) if col_norms.size else 0.0
    tol = max(mat.shape) * np.finfo(float).eps * max(scale, 1.0) * 100.0
    basis: list[np.ndarray] = []
    for col in range(mat.shape[1]):
        residual = mat[:, col].astype(float, copy=True)
        for vector in basis:
            residual -= vector * float(vector @ residual)
        norm = float(np.linalg.norm(residual))
        if norm > tol:
            basis.append(residual / norm)
    return len(basis)


def _auto_pruned_active_mask(labels: tuple[str, ...], patterns: tuple[str, ...]) -> np.ndarray:
    if not patterns:
        return np.ones(len(labels), dtype=bool)
    lowered = tuple(pattern.lower() for pattern in patterns)
    return np.array(
        [not any(pattern in label.lower() for pattern in lowered) for label in labels], dtype=bool
    )


def _mark_auto_pruned_classes(
    labels: tuple[str, ...],
    class_by_gic: tuple[str, ...],
    patterns: tuple[str, ...],
) -> tuple[str, ...]:
    if not patterns:
        return class_by_gic
    lowered = tuple(pattern.lower() for pattern in patterns)
    classes = list(class_by_gic)
    if len(classes) < len(labels):
        classes.extend("" for _ in range(len(labels) - len(classes)))
    for idx, label in enumerate(labels):
        if any(pattern in label.lower() for pattern in lowered):
            classes[idx] = "auto_pruned_weak"
    return tuple(classes)


def _weak_parameter_patterns(
    names: tuple[str, ...],
    weighted_jac: np.ndarray,
    condition_target: float,
) -> tuple[str, ...]:
    if weighted_jac.size == 0 or weighted_jac.shape[1] <= 1 or condition_target <= 0.0:
        return ()
    remaining = list(range(weighted_jac.shape[1]))
    pruned: list[str] = []
    while len(remaining) > 1:
        current = weighted_jac[:, remaining]
        conditioning = rank_condition(current)
        if (
            np.isfinite(conditioning.condition_number)
            and conditioning.condition_number <= condition_target
        ):
            break
        best: tuple[float, int] | None = None
        for col in remaining:
            trial = [item for item in remaining if item != col]
            trial_condition = rank_condition(weighted_jac[:, trial]).condition_number
            if not np.isfinite(trial_condition):
                continue
            score = (trial_condition, col)
            if best is None or score < best:
                best = score
        if best is None or best[0] >= conditioning.condition_number:
            break
        removed = best[1]
        pattern = _parameter_prune_pattern(names[removed])
        if pattern:
            pruned.append(pattern)
        remaining.remove(removed)
    return tuple(pruned)


def _parameter_prune_pattern(name: str) -> str:
    parts = str(name).split()
    if len(parts) >= 3 and re.match(r"^[A-Z][0-9][A-Za-z]+[0-9]+$", parts[2]):
        return parts[2]
    if parts:
        return parts[0]
    return str(name)


def _parameter_class_transform(
    labels: tuple[str, ...],
    active_mask: np.ndarray,
    parameter_classes: tuple[ParameterClassConstraint, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    active_indices = np.where(active_mask)[0]
    class_by_gic = [
        next(
            (
                item.name
                for item in parameter_classes
                if item.mode == "fixed" and _class_matches(item, label)
            ),
            "",
        )
        for label in labels
    ]
    if not len(active_indices):
        return np.zeros((0, 0), dtype=float), (), tuple(class_by_gic)
    columns: list[np.ndarray] = []
    names: list[str] = []
    shared_classes = tuple(item for item in parameter_classes if item.mode == "shared")
    assigned = np.zeros(len(active_indices), dtype=bool)
    for parameter_class in shared_classes:
        local = [
            pos
            for pos, idx in enumerate(active_indices)
            if _class_matches(parameter_class, labels[idx])
        ]
        if not local:
            continue
        col = np.zeros(len(active_indices), dtype=float)
        for pos in local:
            col[pos] = 1.0
            assigned[pos] = True
            class_by_gic[active_indices[pos]] = parameter_class.name
        columns.append(col)
        names.append(parameter_class.name)
    for pos, idx in enumerate(active_indices):
        if assigned[pos]:
            continue
        col = np.zeros(len(active_indices), dtype=float)
        col[pos] = 1.0
        columns.append(col)
        names.append(labels[idx])
    return np.column_stack(columns), tuple(names), tuple(class_by_gic)


def _reduced_parameter_scales(
    labels: tuple[str, ...], active_mask: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    if transform.size == 0:
        return np.ones(transform.shape[1], dtype=float)
    active_indices = np.where(active_mask)[0]
    gic_scales = np.array(
        [_gic_coordinate_scale(labels[idx]) for idx in active_indices], dtype=float
    )
    scales = np.ones(transform.shape[1], dtype=float)
    for col in range(transform.shape[1]):
        weights = np.abs(transform[:, col])
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            scales[col] = float(np.sum(weights * gic_scales) / weight_sum)
    return np.clip(scales, 0.05, 2.0)


def _gic_coordinate_scale(label: str) -> float:
    low = label.lower()
    if "bond(" in low or re.search(r"\br\s*\(", low) or "str" in low:
        return 1.0
    if any(
        token in low
        for token in (
            "angle(",
            "dihedral(",
            "out_of_plane(",
            "linear_bend(",
            "ang",
            "dih",
            "oop",
            "lin",
            "pck",
        )
    ):
        return 0.5
    if re.search(r"\b[adul]\s*\(", low):
        return 0.5
    return 1.0


def _dynamic_parameter_scales(jac_weighted: np.ndarray, base_scales: np.ndarray) -> np.ndarray:
    """Equilibrate Jacobian columns without changing the physical coordinates."""
    base = np.asarray(base_scales, dtype=float)
    if base.size == 0:
        return base
    jac = np.asarray(jac_weighted, dtype=float)
    if jac.ndim != 2 or jac.shape[1] != base.size or jac.size == 0:
        return np.clip(base, 1.0e-8, 1.0e8)
    scaled = jac * base[None, :]
    norms = np.linalg.norm(scaled, axis=0)
    positive = np.isfinite(norms) & (norms > 0.0)
    if not np.any(positive):
        return np.clip(base, 1.0e-8, 1.0e8)
    target = float(np.median(norms[positive]))
    if target <= 0.0 or not np.isfinite(target):
        target = 1.0
    extra = np.ones_like(base)
    extra[positive] = np.clip(target / norms[positive], 1.0e-4, 1.0e4)
    return np.clip(base * extra, 1.0e-8, 1.0e8)


def _robust_sqrt_weights(
    weighted_residual: np.ndarray,
    loss: str,
    scale: float,
    experimental_rows: int | None = None,
    row_groups: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[np.ndarray, float, int, int]:
    """Return IRLS sqrt weights for experimental rows only."""
    residual = np.asarray(weighted_residual, dtype=float)
    sqrt_weights = np.ones_like(residual, dtype=float)
    if str(loss).lower() == "none" or residual.size == 0:
        return sqrt_weights, 0.0, 0, 0
    nrows = (
        residual.size
        if experimental_rows is None
        else max(0, min(int(experimental_rows), residual.size))
    )
    if nrows == 0:
        return sqrt_weights, 0.0, 0, 0
    groups = (
        row_groups
        if row_groups is not None
        else _robust_isotopologue_groups(nrows, loss_rows=nrows)
    )
    groups = tuple(tuple(idx for idx in group if 0 <= idx < nrows) for group in groups)
    groups = tuple(group for group in groups if group)
    if not groups:
        return sqrt_weights, 0.0, 0, 0
    group_scores = np.array(
        [float(np.sqrt(np.mean(residual[np.asarray(group, dtype=int)] ** 2))) for group in groups],
        dtype=float,
    )
    robust_scale = float(scale) if float(scale) > 0.0 else _automatic_robust_scale(group_scores)
    if robust_scale <= 0.0 or not np.isfinite(robust_scale):
        return sqrt_weights, 0.0, 0, 0
    z = np.abs(group_scores) / robust_scale
    group_weights = np.ones_like(group_scores, dtype=float)
    text = str(loss).lower()
    if text == "huber":
        mask = z > 1.0
        group_weights[mask] = 1.0 / np.maximum(z[mask], 1.0e-12)
    elif text == "soft_l1":
        group_weights = 1.0 / np.sqrt(1.0 + z * z)
    elif text == "cauchy":
        group_weights = 1.0 / (1.0 + z * z)
    else:
        raise ValueError(f"Unsupported robust loss: {loss}")
    group_weights = np.clip(group_weights, 1.0e-12, 1.0)
    downweighted_rows = 0
    for group, weight in zip(groups, group_weights):
        sqrt_weights[np.asarray(group, dtype=int)] = np.sqrt(weight)
        if weight < 0.999:
            downweighted_rows += len(group)
    downweighted_groups = int(np.sum(group_weights < 0.999))
    return sqrt_weights, robust_scale, downweighted_rows, downweighted_groups


def _robust_sqrt_weights_for_model(
    weighted_residual: np.ndarray,
    loss: str,
    scale: float,
    model: MeasurementModel,
) -> tuple[np.ndarray, float, int, int]:
    return _robust_sqrt_weights(
        weighted_residual,
        loss,
        scale,
        model.n_experimental_rows,
        _experimental_isotopologue_row_groups(model),
    )


def _experimental_isotopologue_row_groups(model: MeasurementModel) -> tuple[tuple[int, ...], ...]:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, (label, _component) in enumerate(model.labels[: model.n_experimental_rows]):
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(idx)
    return tuple(tuple(groups[label]) for label in order)


def _robust_isotopologue_groups(nrows: int, *, loss_rows: int) -> tuple[tuple[int, ...], ...]:
    # This fallback groups consecutive selected components. The public solver
    # always uses MeasurementModel labels below, but this keeps the helper
    # deterministic for direct unit tests.
    if nrows <= 0:
        return ()
    return tuple((idx,) for idx in range(min(nrows, loss_rows)))


def _automatic_robust_scale(weighted_residual: np.ndarray) -> float:
    residual = np.asarray(weighted_residual, dtype=float)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    if mad > 0.0 and np.isfinite(mad):
        return max(1.4826 * mad, 1.0e-12)
    rms = float(np.sqrt(np.mean(finite * finite)))
    return max(rms, 1.0)


def _class_matches(parameter_class: ParameterClassConstraint, label: str) -> bool:
    low = label.lower()
    return any(pattern.lower() in low for pattern in parameter_class.patterns)


def _jacobian_constants_wrt_gics(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    prims: object,
    u_matrix: np.ndarray,
    active_mask: np.ndarray,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    *,
    step: float,
    cartesian_from_q: np.ndarray | None = None,
) -> np.ndarray:
    active_indices = np.where(active_mask)[0]
    base_q = _gic_values(prims, u_matrix, coords)
    jac = np.zeros((len(measurement_model.observed), len(active_indices)), dtype=float)
    if cartesian_from_q is None:
        cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    analytic = _analytic_measurement_jacobian_wrt_gics(
        atoms,
        coords,
        request,
        labels,
        measurement_model,
        cartesian_from_q,
    )
    if analytic is not None and analytic.shape == (len(measurement_model.observed), len(base_q)):
        return analytic[:, active_indices]
    return _finite_difference_measurement_jacobian_wrt_gics(
        atoms,
        coords,
        request,
        prims,
        u_matrix,
        active_indices,
        labels,
        measurement_model,
        step=step,
        cartesian_from_q=cartesian_from_q,
    )


def _analytic_measurement_jacobian_wrt_gics(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    cartesian_from_q: np.ndarray,
) -> np.ndarray | None:
    if measurement_model.observable == "moments":
        cartesian = _moments_cartesian_jacobian(atoms, coords, request.observations)
        selected = _select_raw_components(
            cartesian, MOMENT_COMPONENTS, measurement_model.components
        )
    elif measurement_model.observable == "rotational_constants":
        cartesian = _rotational_constants_cartesian_jacobian(atoms, coords, request.observations)
        selected = _select_raw_components(
            cartesian, ROTATIONAL_COMPONENTS, measurement_model.components
        )
    else:
        return None
    if measurement_model.experimental_row_indices:
        selected = selected[np.asarray(measurement_model.experimental_row_indices, dtype=int)]
    gic_jac = selected @ cartesian_from_q
    predicate = _predicate_jacobian(request.qm_predicates, labels, coords, cartesian_from_q)
    if predicate.size:
        return np.vstack([gic_jac, predicate])
    return gic_jac


def _finite_difference_measurement_jacobian_wrt_gics(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    prims: object,
    u_matrix: np.ndarray,
    active_indices: np.ndarray,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    *,
    step: float,
    cartesian_from_q: np.ndarray,
) -> np.ndarray:
    base_q = _gic_values(prims, u_matrix, coords)
    jac = np.zeros((len(measurement_model.observed), len(active_indices)), dtype=float)

    def column(idx: int) -> np.ndarray:
        dq = np.zeros_like(base_q)
        dq[idx] = step
        plus = _displace_along_gics(coords, prims, u_matrix, dq, cartesian_from_q=cartesian_from_q)
        plus_q = _gic_values(prims, u_matrix, plus)
        dq[idx] = -step
        minus = _displace_along_gics(coords, prims, u_matrix, dq, cartesian_from_q=cartesian_from_q)
        minus_q = _gic_values(prims, u_matrix, minus)
        return (
            _measurement_vector(atoms, plus, request, plus_q, labels, measurement_model)
            - _measurement_vector(atoms, minus, request, minus_q, labels, measurement_model)
        ) / (2.0 * step)

    max_workers = min(len(active_indices), max(1, (os.cpu_count() or 1)))
    if max_workers <= 1 or len(active_indices) < 4:
        for col, idx in enumerate(active_indices):
            jac[:, col] = column(int(idx))
        return jac
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for col, values in enumerate(executor.map(column, [int(idx) for idx in active_indices])):
            jac[:, col] = values
    return jac


def _jacobian_constants_wrt_cartesian_basis(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    cartesian_from_q: np.ndarray,
) -> np.ndarray:
    if measurement_model.observable == "moments":
        cartesian = _moments_cartesian_jacobian(atoms, coords, request.observations)
        selected = _select_raw_components(
            cartesian, MOMENT_COMPONENTS, measurement_model.components
        )
    elif measurement_model.observable == "rotational_constants":
        cartesian = _rotational_constants_cartesian_jacobian(atoms, coords, request.observations)
        selected = _select_raw_components(
            cartesian, ROTATIONAL_COMPONENTS, measurement_model.components
        )
    else:
        raise ScientificValidationError(
            f"Unsupported observable for Cartesian-basis SEfit: {measurement_model.observable}"
        )
    if measurement_model.experimental_row_indices:
        selected = selected[np.asarray(measurement_model.experimental_row_indices, dtype=int)]
    jac = selected @ cartesian_from_q
    predicate = _predicate_jacobian(request.qm_predicates, labels, coords, cartesian_from_q)
    if predicate.size:
        return np.vstack([jac, predicate])
    return jac


def _moments_cartesian_jacobian(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> np.ndarray:
    rows = []
    for obs in observations:
        _moments, jac = _principal_moments_and_cartesian_jacobian(
            atoms,
            coords,
            _isotopes_for_observation(atoms, obs),
        )
        rows.append(jac)
    return np.vstack(rows) if rows else np.zeros((0, np.asarray(coords).size), dtype=float)


def _rotational_constants_cartesian_jacobian(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> np.ndarray:
    rows = []
    for obs in observations:
        moments, moment_jac = _principal_moments_and_cartesian_jacobian(
            atoms,
            coords,
            _isotopes_for_observation(atoms, obs),
        )
        factors = np.zeros(3, dtype=float)
        positive = moments > 0.0
        factors[positive] = -ROTCONST_TO_MOMENT / (moments[positive] * moments[positive])
        rows.append(factors[:, None] * moment_jac)
    return np.vstack(rows) if rows else np.zeros((0, np.asarray(coords).size), dtype=float)


def _principal_moments_and_cartesian_jacobian(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    isotopes: list[int | None],
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=float)
    masses = _mass_vector_for_isotopes(atoms, isotopes)
    centered, inertia = _centered_coords_and_inertia(arr, masses)
    eye = np.eye(3)
    moments, axes = np.linalg.eigh(inertia)
    jac = np.zeros((3, arr.size), dtype=float)
    for atom_idx, (mass, xyz) in enumerate(zip(masses, centered)):
        for axis_idx in range(3):
            unit = eye[axis_idx]
            derivative = mass * (
                2.0 * xyz[axis_idx] * eye - np.outer(unit, xyz) - np.outer(xyz, unit)
            )
            col = 3 * atom_idx + axis_idx
            for moment_idx in range(3):
                vector = axes[:, moment_idx]
                jac[moment_idx, col] = float(vector @ derivative @ vector)
    return moments, jac


def _principal_moments_from_masses(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    _centered, inertia = _centered_coords_and_inertia(
        np.asarray(coords, dtype=float), np.asarray(masses, dtype=float)
    )
    return np.linalg.eigvalsh(inertia)


def _centered_coords_and_inertia(
    coords: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=float)
    mass = np.asarray(masses, dtype=float)
    total_mass = float(np.sum(mass))
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        raise ScientificValidationError("Cannot build inertia tensor with non-positive total mass")
    centered = arr - (mass[:, None] * arr).sum(axis=0) / total_mass
    inertia = np.zeros((3, 3), dtype=float)
    eye = np.eye(3)
    for atom_mass, xyz in zip(mass, centered):
        inertia += atom_mass * ((xyz @ xyz) * eye - np.outer(xyz, xyz))
    return centered, inertia


def _predicate_jacobian(
    predicates: tuple[QMParameterPredicate, ...],
    labels: tuple[str, ...],
    coords: np.ndarray,
    cartesian_from_q: np.ndarray,
) -> np.ndarray:
    rows = []
    for predicate in predicates:
        primitive = _predicate_primitive(predicate)
        if primitive is not None:
            rows.append(b_matrix_analytic([primitive], coords)[0] @ cartesian_from_q)
        else:
            for idx in _predicate_indices(predicate, labels):
                row = np.zeros(cartesian_from_q.shape[1], dtype=float)
                row[idx] = 1.0
                rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, cartesian_from_q.shape[1]), dtype=float)


def _gic_cartesian_projector(prims: object, u_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    bq = u_matrix.T @ b_matrix_analytic(prims, coords)
    return cartesian_from_internal_jacobian(bq, rcond=1.0e-8)


def _gic_projector_state(
    prims: object,
    u_matrix: np.ndarray,
    coords: np.ndarray,
    q_values: np.ndarray,
) -> GICProjectorState:
    return GICProjectorState(
        coords=np.asarray(coords, dtype=float).copy(),
        q_values=np.asarray(q_values, dtype=float).copy(),
        cartesian_from_q=_gic_cartesian_projector(prims, u_matrix, coords),
    )


def _secant_projector_update(
    cartesian_from_q: np.ndarray,
    previous_coords: np.ndarray,
    previous_q: np.ndarray,
    current_coords: np.ndarray,
    current_q: np.ndarray,
) -> SecantProjectorUpdate:
    update = secant_projector_update(
        cartesian_from_q,
        previous_q,
        previous_coords,
        current_q,
        current_coords,
    )
    return SecantProjectorUpdate(update.cartesian_from_q, update.relative_error, update.accepted)


def _displace_along_gics(
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    dq: np.ndarray,
    *,
    cartesian_from_q: np.ndarray | None = None,
    sonic_definition: LinkGICDefinition | None = None,
) -> np.ndarray:
    if cartesian_from_q is None:
        cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    base_q = _gic_values(prims, u_matrix, coords)

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            _gic_values(prims, u_matrix, candidate),
            u_matrix.T @ b_matrix_analytic(prims, candidate),
        )

    target_q = base_q + np.asarray(dq, dtype=float)
    if sonic_definition is not None:
        result = hybrid_internal_coordinate_step(
            sonic_definition,
            coords,
            target_q,
            evaluate,
        )
        if result.converged:
            return result.coordinates_angstrom
        # LINK's generic corrector is a compatibility fallback for unusual
        # mixed coordinates.  Start it from the finite SONIC predictor rather
        # than making the pseudoinverse path the primary back-transform.
        result = nonlinear_internal_coordinate_step(
            result.coordinates_angstrom,
            target_q,
            evaluate,
        )
    else:
        result = nonlinear_internal_coordinate_step(
            coords,
            target_q,
            evaluate,
            cartesian_from_q=cartesian_from_q,
        )
    if not result.converged:
        raise RuntimeError(
            "GIC back-transformation did not converge: "
            f"residual={np.linalg.norm(result.residual):.6g}"
        )
    return result.coordinates_angstrom


def _displace_along_cartesian_basis(
    coords: np.ndarray,
    cartesian_from_q: np.ndarray,
    dq: np.ndarray,
) -> np.ndarray:
    dx = np.asarray(cartesian_from_q, dtype=float) @ np.asarray(dq, dtype=float)
    return np.asarray(coords, dtype=float) + dx.reshape(np.asarray(coords).shape)


def _line_search_update(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    prims: object,
    u_matrix: np.ndarray,
    dq: np.ndarray,
    *,
    current_objective: float,
    base_q: np.ndarray,
    cartesian_from_q: np.ndarray | None = None,
    weighted_residual: np.ndarray,
    jac_weighted: np.ndarray,
    reduced_step: np.ndarray,
    robust_sqrt_weights: np.ndarray | None = None,
    fixed_primitives: tuple[Primitive, ...] = (),
    fixed_primitive_targets: np.ndarray | None = None,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    sonic_definition: LinkGICDefinition | None = None,
) -> LineSearchResult:
    observed = measurement_model.observed
    sqrt_weights = np.sqrt(measurement_model.weights)
    if robust_sqrt_weights is not None:
        sqrt_weights = sqrt_weights * np.asarray(robust_sqrt_weights, dtype=float)
    best = LineSearchResult(
        coords=coords,
        q_values=base_q,
        objective=current_objective,
        accepted=False,
        actual_reduction=0.0,
        predicted_reduction=0.0,
        ratio=0.0,
        scale=0.0,
    )
    if cartesian_from_q is None:
        cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
        predicted_reduction = _predicted_reduction(
            weighted_residual,
            jac_weighted,
            reduced_step,
            scale=scale,
            current_objective=current_objective,
        )
        try:
            candidate = _displace_along_gics(
                coords,
                prims,
                u_matrix,
                scale * dq,
                cartesian_from_q=cartesian_from_q,
                sonic_definition=sonic_definition,
            )
            if fixed_primitives or linear_constraints or expression_constraints:
                candidate = _project_fixed_primitives(
                    candidate,
                    fixed_primitives,
                    fixed_primitive_targets
                    if fixed_primitive_targets is not None
                    else _fixed_primitive_targets(fixed_primitives, coords),
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=prims,
                    u_matrix=u_matrix,
                    labels=labels,
                    expression_definitions=expression_definitions,
                )
            q_candidate = _gic_values(prims, u_matrix, candidate)
            calc = _measurement_vector(
                atoms, candidate, request, q_candidate, labels, measurement_model
            )
        except Exception:
            continue
        if calc.shape != observed.shape:
            continue
        residual = (observed - calc) * sqrt_weights
        candidate_objective = objective(residual)
        if not np.isfinite(candidate_objective):
            continue
        actual_reduction = current_objective - candidate_objective
        ratio = actual_reduction / predicted_reduction if predicted_reduction > 0.0 else 0.0
        if actual_reduction > 0.0 and candidate_objective < best.objective:
            best = LineSearchResult(
                coords=candidate,
                q_values=q_candidate,
                objective=candidate_objective,
                accepted=True,
                actual_reduction=float(actual_reduction),
                predicted_reduction=float(max(predicted_reduction, 0.0)),
                ratio=float(ratio),
                scale=float(scale),
            )
            if predicted_reduction <= 0.0 or actual_reduction >= 1.0e-4 * predicted_reduction:
                break
    return best


def _line_search_update_cartesian_basis(
    atoms: list[str],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    labels: tuple[str, ...],
    measurement_model: "MeasurementModel",
    mode_model: CartesianCoordinateModel,
    dq: np.ndarray,
    *,
    current_objective: float,
    base_q: np.ndarray,
    weighted_residual: np.ndarray,
    jac_weighted: np.ndarray,
    reduced_step: np.ndarray,
    robust_sqrt_weights: np.ndarray | None = None,
    fixed_primitives: tuple[Primitive, ...] = (),
    fixed_primitive_targets: np.ndarray | None = None,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> LineSearchResult:
    observed = measurement_model.observed
    sqrt_weights = np.sqrt(measurement_model.weights)
    if robust_sqrt_weights is not None:
        sqrt_weights = sqrt_weights * np.asarray(robust_sqrt_weights, dtype=float)
    best = LineSearchResult(
        coords=coords,
        q_values=base_q,
        objective=current_objective,
        accepted=False,
        actual_reduction=0.0,
        predicted_reduction=0.0,
        ratio=0.0,
        scale=0.0,
    )
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
        predicted_reduction = _predicted_reduction(
            weighted_residual,
            jac_weighted,
            reduced_step,
            scale=scale,
            current_objective=current_objective,
        )
        candidate = _displace_along_cartesian_basis(coords, mode_model.cartesian_from_q, scale * dq)
        try:
            if fixed_primitives or linear_constraints or expression_constraints:
                candidate = _project_fixed_primitives(
                    candidate,
                    fixed_primitives,
                    fixed_primitive_targets
                    if fixed_primitive_targets is not None
                    else _fixed_primitive_targets(fixed_primitives, coords),
                    linear_constraints=linear_constraints,
                    expression_constraints=expression_constraints,
                    expression_targets=expression_targets,
                    prims=(),
                    u_matrix=np.zeros((0, 0), dtype=float),
                    labels=(),
                    expression_definitions=expression_definitions,
                )
            q_candidate = mode_model.values(candidate)
            calc = _measurement_vector(
                atoms, candidate, request, q_candidate, labels, measurement_model
            )
        except Exception:
            continue
        if calc.shape != observed.shape:
            continue
        residual = (observed - calc) * sqrt_weights
        candidate_objective = objective(residual)
        if not np.isfinite(candidate_objective):
            continue
        actual_reduction = current_objective - candidate_objective
        ratio = actual_reduction / predicted_reduction if predicted_reduction > 0.0 else 0.0
        if actual_reduction > 0.0 and candidate_objective < best.objective:
            best = LineSearchResult(
                coords=candidate,
                q_values=q_candidate,
                objective=candidate_objective,
                accepted=True,
                actual_reduction=float(actual_reduction),
                predicted_reduction=float(max(predicted_reduction, 0.0)),
                ratio=float(ratio),
                scale=float(scale),
            )
            if predicted_reduction <= 0.0 or actual_reduction >= 1.0e-4 * predicted_reduction:
                break
    return best


def _iteration_trace_row(
    iteration: int,
    status: str,
    current_objective: float,
    line_search: LineSearchResult,
    damping: float,
    trust_radius: float,
    step_norm: float,
    gradient_inf_norm: float,
    jac_weighted_scaled: np.ndarray,
    coords_for_constraints: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_primitive_targets: np.ndarray,
    *,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    robust_scale: float = 0.0,
    robust_downweighted_observations: int = 0,
    robust_downweighted_isotopologues: int = 0,
    coordinate_model_age: int = 0,
    b_projector_secant_error: float = 0.0,
    linear_solver: str = "svd_more_hebden_trust_region",
) -> SemiexperimentalIterationTrace:
    rank, smallest, relative = _jacobian_singular_trace(
        np.asarray(jac_weighted_scaled, dtype=float)
    )
    return SemiexperimentalIterationTrace(
        iteration=int(iteration),
        status=str(status),
        objective_before=float(current_objective),
        objective_after=float(line_search.objective),
        actual_reduction=float(line_search.actual_reduction),
        predicted_reduction=float(line_search.predicted_reduction),
        trust_ratio=float(line_search.ratio),
        line_search_scale=float(line_search.scale),
        damping=float(damping),
        trust_radius=float(trust_radius),
        step_norm=float(step_norm),
        gradient_inf_norm=float(gradient_inf_norm),
        rank=int(rank),
        smallest_singular_value=float(smallest),
        relative_smallest_singular_value=float(relative),
        constraint_max_abs=_constraint_max_abs(
            coords_for_constraints,
            fixed_primitives,
            fixed_primitive_targets,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        ),
        robust_scale=float(robust_scale),
        robust_downweighted_observations=int(robust_downweighted_observations),
        robust_downweighted_isotopologues=int(robust_downweighted_isotopologues),
        coordinate_model_age=int(coordinate_model_age),
        b_projector_secant_error=float(b_projector_secant_error),
        linear_solver=str(linear_solver),
    )


def _jacobian_singular_trace(jacobian: np.ndarray) -> tuple[int, float, float]:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or jac.size == 0 or jac.shape[1] == 0:
        return 0, 0.0, 0.0
    try:
        singular = np.linalg.svd(jac, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0, float("nan"), float("nan")
    if not singular.size:
        return 0, 0.0, 0.0
    s0 = max(float(singular[0]), 1.0)
    tol = max(jac.shape) * np.finfo(float).eps * s0 * 100.0
    rank = int(np.sum(singular > tol))
    smallest = float(singular[-1]) if singular.size else 0.0
    return rank, smallest, smallest / s0 if s0 > 0.0 else 0.0


def _constraint_max_abs(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> float:
    if not fixed_primitives and not linear_constraints and not expression_constraints:
        return 0.0
    try:
        residual = _combined_primitive_constraint_residual(
            np.asarray(coords, dtype=float),
            fixed_primitives,
            np.asarray(fixed_targets, dtype=float),
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
    except Exception:
        return float("inf")
    if residual.size == 0:
        return 0.0
    return float(np.max(np.abs(residual)))


def _adaptive_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
    trust_radius: float,
) -> TrustRegionStep:
    if jac_weighted.size == 0 or jac_weighted.shape[1] == 0:
        return TrustRegionStep(
            np.zeros((0,), dtype=float), max(float(damping), 0.0), False, "empty"
        )
    step_result = _svd_trust_region_lm_step(jac_weighted, weighted_residual, damping, trust_radius)
    step = step_result.step
    predicted = _predicted_reduction(
        weighted_residual,
        jac_weighted,
        step,
        scale=1.0,
        current_objective=objective(weighted_residual),
    )
    if predicted <= 0.0 or not np.all(np.isfinite(step)):
        step = limit_step(_cauchy_step(jac_weighted, weighted_residual), trust_radius)
        step_result = TrustRegionStep(
            step,
            max(float(damping), 0.0),
            trust_radius > 0.0 and float(np.linalg.norm(step)) >= 0.99 * trust_radius,
            "cauchy_fallback",
        )
    return step_result


def _svd_trust_region_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
    trust_radius: float,
) -> TrustRegionStep:
    """Solve the rank-revealing LM trust-region subproblem in SVD coordinates.

    The local model is ``min 0.5 ||r - J p||^2`` with ``||p|| <= Delta``.
    When the Gauss-Newton step is inside the region, the minimum-norm
    rank-revealing step is used. Otherwise the Levenberg shift is found by
    robust bisection of the secular equation ``||p(mu)|| = Delta``. This is
    preferable to clipping a full LM step because the returned step is the
    solution of the regularized model for the active trust radius.
    """
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return TrustRegionStep(
            np.zeros((0,), dtype=float), max(float(damping), 0.0), False, "empty"
        )
    delta = float(trust_radius)
    if delta <= 0.0 or not np.isfinite(delta):
        step = _rank_revealing_lm_step(jac, residual, damping)
        return TrustRegionStep(step, max(float(damping), 0.0), False, "svd_rank_revealing_lm")
    delta = max(delta, TRUST_REGION_MIN_RADIUS)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        step = limit_step(_augmented_qr_lm_step(jac, residual, damping), delta)
        return TrustRegionStep(step, max(float(damping), 0.0), True, "augmented_qr_lm_fallback")
    if not singular.size:
        return TrustRegionStep(np.zeros(ncols, dtype=float), 0.0, False, "zero_jacobian")
    s0 = max(float(singular[0]), 1.0)
    tol = max(jac.shape) * np.finfo(float).eps * s0 * 100.0
    beta = u_matrix.T @ residual

    def step_for_shift(shift: float) -> np.ndarray:
        mu = max(float(shift), 0.0)
        factors = np.zeros_like(singular)
        keep = singular > tol
        factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
        return vh.T @ (factors * beta)

    unconstrained = step_for_shift(0.0)
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    if np.all(np.isfinite(unconstrained)) and unconstrained_norm <= delta:
        return TrustRegionStep(unconstrained, 0.0, False, "svd_more_hebden_trust_region")

    low = 0.0
    high = max(float(damping), s0 * s0 * 1.0e-12, DAMPING_MIN)
    high_step = step_for_shift(high)
    high_norm = float(np.linalg.norm(high_step))
    while (not np.isfinite(high_norm) or high_norm > delta) and high < DAMPING_MAX:
        low = high
        high = min(high * 4.0, DAMPING_MAX)
        high_step = step_for_shift(high)
        high_norm = float(np.linalg.norm(high_step))
    if not np.all(np.isfinite(high_step)):
        step = limit_step(_cauchy_step(jac, residual), delta)
        return TrustRegionStep(step, high, True, "cauchy_fallback")
    if high_norm > delta:
        step = limit_step(high_step, delta)
        return TrustRegionStep(step, high, True, "svd_more_hebden_trust_region_clipped")

    best_shift = high
    best_step = high_step
    for _ in range(80):
        mid = 0.5 * (low + high)
        trial = step_for_shift(mid)
        trial_norm = float(np.linalg.norm(trial))
        if not np.isfinite(trial_norm):
            low = mid
            continue
        best_shift = mid
        best_step = trial
        if abs(trial_norm - delta) <= max(1.0e-10 * delta, 1.0e-12):
            break
        if trial_norm > delta:
            low = mid
        else:
            high = mid
    if float(np.linalg.norm(best_step)) > delta * (1.0 + 1.0e-8):
        best_step = limit_step(best_step, delta)
    return TrustRegionStep(best_step, best_shift, True, "svd_more_hebden_trust_region")


def _rank_revealing_lm_step(
    jac_weighted: np.ndarray,
    weighted_residual: np.ndarray,
    damping: float,
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    mu = max(float(damping), 0.0)
    try:
        u_matrix, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return _augmented_qr_lm_step(jac, residual, mu)
    if not singular.size:
        return np.zeros(ncols, dtype=float)
    tol = max(jac.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0) * 100.0
    projected = u_matrix.T @ residual
    factors = np.zeros_like(singular)
    keep = singular > tol
    factors[keep] = singular[keep] / (singular[keep] * singular[keep] + mu)
    step = vh.T @ (factors * projected)
    if step.shape[0] != ncols or not np.all(np.isfinite(step)):
        return _augmented_qr_lm_step(jac, residual, mu)
    return step


def _augmented_qr_lm_step(
    jac_weighted: np.ndarray, weighted_residual: np.ndarray, damping: float
) -> np.ndarray:
    jac = np.asarray(jac_weighted, dtype=float)
    residual = np.asarray(weighted_residual, dtype=float)
    ncols = jac.shape[1]
    if ncols == 0:
        return np.zeros((0,), dtype=float)
    mu = max(float(damping), 0.0)
    if mu > 0.0:
        lhs = np.vstack([jac, np.sqrt(mu) * np.eye(ncols)])
        rhs = np.concatenate([residual, np.zeros(ncols, dtype=float)])
    else:
        lhs = jac
        rhs = residual
    try:
        step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _cauchy_step(jac, residual)
    return np.asarray(step, dtype=float)


def _cauchy_step(jac_weighted: np.ndarray, weighted_residual: np.ndarray) -> np.ndarray:
    gradient = jac_weighted.T @ weighted_residual
    if gradient.size == 0:
        return gradient
    jg = jac_weighted @ gradient
    denom = float(jg @ jg)
    if denom <= 0.0 or not np.isfinite(denom):
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 0.0 else gradient
    alpha = float((gradient @ gradient) / denom)
    return alpha * gradient


def _predicted_reduction(
    weighted_residual: np.ndarray,
    jac_weighted: np.ndarray,
    reduced_step: np.ndarray,
    *,
    scale: float,
    current_objective: float,
) -> float:
    if reduced_step.size == 0:
        return 0.0
    predicted_residual = weighted_residual - float(scale) * (jac_weighted @ reduced_step)
    predicted_objective = objective(predicted_residual)
    reduction = float(current_objective - predicted_objective)
    return reduction if np.isfinite(reduction) else 0.0


def _accepted_trust_update(
    damping: float,
    trust_radius: float,
    ratio: float,
    scale: float,
    step_norm: float,
    max_step: float,
) -> tuple[float, float]:
    ratio = float(ratio) if np.isfinite(ratio) else 0.0
    step_norm = float(step_norm) if np.isfinite(step_norm) else 0.0
    scale = float(scale) if np.isfinite(scale) else 0.0
    damping_floor = max(float(damping), DAMPING_MIN)
    if ratio < 0.25 or scale < 0.5:
        new_damping = min(max(damping_floor * 4.0, DAMPING_MIN), DAMPING_MAX)
        new_radius = _contracted_trust_radius(trust_radius, max_step, step_norm, 0.5)
    elif ratio > 0.75 and scale >= 0.9 and _step_near_trust_boundary(step_norm, trust_radius):
        new_damping = max(damping_floor / 3.0, DAMPING_MIN)
        new_radius = _expanded_trust_radius(trust_radius, max_step, step_norm)
    else:
        new_damping = max(damping_floor / 1.5, DAMPING_MIN)
        new_radius = trust_radius
    return new_damping, new_radius


def _rejected_trust_update(
    damping: float, trust_radius: float, max_step: float
) -> tuple[float, float]:
    damping_floor = max(float(damping), DAMPING_MIN)
    return min(damping_floor * 8.0, DAMPING_MAX), _scaled_trust_radius(trust_radius, max_step, 0.35)


def _scaled_trust_radius(trust_radius: float, max_step: float, scale: float) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    return max(float(current) * float(scale), _minimum_trust_radius(max_step))


def _contracted_trust_radius(
    trust_radius: float, max_step: float, step_norm: float, scale: float
) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    floor = _minimum_trust_radius(max_step)
    if step_norm > 0.0:
        proposed = min(float(current) * float(scale), max(2.0 * step_norm, floor))
    else:
        proposed = float(current) * float(scale)
    return max(float(proposed), floor)


def _expanded_trust_radius(trust_radius: float, max_step: float, step_norm: float) -> float:
    if max_step <= 0.0:
        return trust_radius
    current = trust_radius if trust_radius > 0.0 else max_step
    proposed = max(current * 1.8, step_norm * 2.0, _minimum_trust_radius(max_step))
    return min(float(max_step), float(proposed))


def _minimum_trust_radius(max_step: float) -> float:
    if max_step > 0.0 and np.isfinite(max_step):
        return max(TRUST_REGION_MIN_RADIUS, 1.0e-9 * float(max_step))
    return TRUST_REGION_MIN_RADIUS


def _step_near_trust_boundary(step_norm: float, trust_radius: float) -> bool:
    if trust_radius <= 0.0 or not np.isfinite(trust_radius):
        return False
    return float(step_norm) >= 0.80 * float(trust_radius)


def _trust_region_is_stalled(
    damping: float, trust_radius: float, stalled_rejections: int, max_step: float
) -> bool:
    if stalled_rejections < 5:
        return False
    if damping >= 0.999 * DAMPING_MAX:
        return True
    return trust_radius > 0.0 and trust_radius <= 10.0 * _minimum_trust_radius(max_step)


def _objective_has_stabilized(
    previous_objective: float | None, current_objective: float, tolerance_MHz: float
) -> bool:
    if previous_objective is None:
        return False
    if not np.isfinite(previous_objective) or not np.isfinite(current_objective):
        return False
    absolute = max(float(tolerance_MHz) * float(tolerance_MHz), 1.0e-14)
    relative = 1.0e-10 * max(1.0, abs(float(previous_objective)), abs(float(current_objective)))
    return abs(float(previous_objective) - float(current_objective)) <= max(absolute, relative)


def _should_refresh_gic_model(
    line_search: LineSearchResult,
    model_age: int,
    *,
    secant_relative_error: float,
    tolerance_MHz: float,
    n_observations: int,
) -> bool:
    if model_age <= 0:
        return False
    if should_refresh_coordinate_model(
        model_age=model_age,
        line_search_scale=line_search.scale,
        trust_ratio=line_search.ratio,
        secant_relative_error=secant_relative_error,
    ):
        return True
    objective_scale = max(int(n_observations), 1) * tolerance_MHz * tolerance_MHz
    return line_search.objective <= max(10.0 * objective_scale, 1.0e-24)


def _constants_vector(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> np.ndarray:
    values: list[float] = []
    for obs in observations:
        moments = _principal_moments_from_masses(coords, _mass_vector_for_observation(atoms, obs))
        constants = np.zeros(3, dtype=float)
        positive = moments > 0.0
        constants[positive] = ROTCONST_TO_MOMENT / moments[positive]
        values.extend(constants)
    return np.array(values, dtype=float)


def _moments_vector(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
) -> np.ndarray:
    values: list[float] = []
    for obs in observations:
        values.extend(
            _principal_moments_from_masses(coords, _mass_vector_for_observation(atoms, obs))
        )
    return np.array(values, dtype=float)


def _build_measurement_model(
    request: SemiexperimentalFitRequest,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
) -> MeasurementModel:
    observable = "moments" if request.observable == "auto" else request.observable
    planar = _is_planar(coords)
    components = _select_components(request, observable, atoms, coords, prims, u_matrix, planar)
    row_labels: list[tuple[str, str]] = []
    for obs in request.observations:
        row_labels.extend((obs.label, comp) for comp in components)
    observed = _experimental_observed_vector(request, observable, components)
    weights = _experimental_weights_vector(request, observable, components)
    experimental_row_indices = _retained_experimental_row_indices(request, row_labels)
    observed = observed[np.asarray(experimental_row_indices, dtype=int)]
    weights = weights[np.asarray(experimental_row_indices, dtype=int)]
    row_labels = [row_labels[index] for index in experimental_row_indices]
    n_experimental_rows = int(observed.size)
    predicate_values, predicate_weights, predicate_labels = _predicate_observations(
        request.qm_predicates, labels
    )
    if predicate_values.size:
        observed = np.concatenate([observed, predicate_values])
        weights = np.concatenate([weights, predicate_weights])
        row_labels.extend(predicate_labels)
    return MeasurementModel(
        observable=observable,
        components=components,
        labels=tuple(row_labels),
        observed=observed,
        weights=weights,
        n_experimental_rows=n_experimental_rows,
        planar=planar,
        experimental_row_indices=experimental_row_indices,
    )


def _build_measurement_model_cartesian_basis(
    request: SemiexperimentalFitRequest,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    labels: tuple[str, ...],
    cartesian_from_q: np.ndarray,
) -> MeasurementModel:
    observable = "moments" if request.observable == "auto" else request.observable
    planar = _is_planar(coords)
    components = _select_components_from_cartesian_basis(
        request, observable, atoms, coords, cartesian_from_q, planar
    )
    row_labels: list[tuple[str, str]] = []
    for obs in request.observations:
        row_labels.extend((obs.label, comp) for comp in components)
    observed = _experimental_observed_vector(request, observable, components)
    weights = _experimental_weights_vector(request, observable, components)
    experimental_row_indices = _retained_experimental_row_indices(request, row_labels)
    observed = observed[np.asarray(experimental_row_indices, dtype=int)]
    weights = weights[np.asarray(experimental_row_indices, dtype=int)]
    row_labels = [row_labels[index] for index in experimental_row_indices]
    n_experimental_rows = int(observed.size)
    predicate_values, predicate_weights, predicate_labels = _predicate_observations(
        request.qm_predicates, labels
    )
    if predicate_values.size:
        observed = np.concatenate([observed, predicate_values])
        weights = np.concatenate([weights, predicate_weights])
        row_labels.extend(predicate_labels)
    return MeasurementModel(
        observable=observable,
        components=components,
        labels=tuple(row_labels),
        observed=observed,
        weights=weights,
        n_experimental_rows=n_experimental_rows,
        planar=planar,
        experimental_row_indices=experimental_row_indices,
    )


def _measurement_vector(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    request: SemiexperimentalFitRequest,
    q_values: np.ndarray,
    labels: tuple[str, ...],
    model: MeasurementModel,
) -> np.ndarray:
    if model.observable == "moments":
        raw = _moments_vector(atoms, coords, request.observations)
        selected = _select_raw_components(raw, MOMENT_COMPONENTS, model.components)
    else:
        raw = _constants_vector(atoms, coords, request.observations)
        selected = _select_raw_components(raw, ROTATIONAL_COMPONENTS, model.components)
    if model.experimental_row_indices:
        selected = selected[np.asarray(model.experimental_row_indices, dtype=int)]
    predicate_values = _predicate_values(request.qm_predicates, labels, q_values, coords)
    if predicate_values.size:
        return np.concatenate([selected, predicate_values])
    return selected


def _retained_experimental_row_indices(
    request: SemiexperimentalFitRequest,
    labels: list[tuple[str, str]],
) -> tuple[int, ...]:
    excluded = {
        f"{label}:{component.upper()}"
        for item in request.excluded_rotational_constants
        for label, separator, component in (str(item).rpartition(":"),)
        if separator
    }
    if not excluded:
        return tuple(range(len(labels)))
    reverse_moment = {value: key for key, value in ROTATIONAL_TO_MOMENT_COMPONENT.items()}
    retained = []
    for index, (label, component) in enumerate(labels):
        rotational_component = reverse_moment.get(component, component).upper()
        if f"{label}:{rotational_component}" not in excluded:
            retained.append(index)
    if not retained:
        raise ValueError("All rotational constants were excluded from the fit")
    return tuple(retained)


def _experimental_observed_vector(
    request: SemiexperimentalFitRequest,
    observable: str,
    components: tuple[str, ...],
) -> np.ndarray:
    if observable == "moments":
        raw = []
        for obs in request.observations:
            raw.extend(_constants_to_moments(obs.corrected.as_tuple()))
        return _select_raw_components(np.array(raw, dtype=float), MOMENT_COMPONENTS, components)
    raw = _observed_vector(request.observations)
    return _select_raw_components(raw, ROTATIONAL_COMPONENTS, components)


def _experimental_weights_vector(
    request: SemiexperimentalFitRequest,
    observable: str,
    components: tuple[str, ...],
) -> np.ndarray:
    values: list[float] = []
    for obs in request.observations:
        if observable == "moments":
            values.extend(_moment_weights(obs))
        else:
            values.extend(obs.weights.as_tuple() if obs.weights is not None else (1.0, 1.0, 1.0))
    component_names = MOMENT_COMPONENTS if observable == "moments" else ROTATIONAL_COMPONENTS
    return _select_raw_components(np.array(values, dtype=float), component_names, components)


def _observed_vector(observations: tuple[IsotopologueObservation, ...]) -> np.ndarray:
    values: list[float] = []
    for obs in observations:
        values.extend(obs.corrected.as_tuple())
    return np.array(values, dtype=float)


def _weights_vector(observations: tuple[IsotopologueObservation, ...]) -> np.ndarray:
    values: list[float] = []
    for obs in observations:
        values.extend(obs.weights.as_tuple() if obs.weights is not None else (1.0, 1.0, 1.0))
    return np.array(values, dtype=float)


def _select_components(
    request: SemiexperimentalFitRequest,
    observable: str,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    planar: bool,
) -> tuple[str, ...]:
    if request.rotational_components != "auto":
        return _explicit_component_selection(request.rotational_components, observable, planar)
    if not planar:
        return MOMENT_COMPONENTS if observable == "moments" else ROTATIONAL_COMPONENTS
    if observable == "moments":
        return _best_planar_moment_pair(atoms, coords, request.observations, prims, u_matrix)
    return _best_planar_rotational_pair(atoms, coords, request.observations, prims, u_matrix)


def _select_components_from_cartesian_basis(
    request: SemiexperimentalFitRequest,
    observable: str,
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    cartesian_from_q: np.ndarray,
    planar: bool,
) -> tuple[str, ...]:
    if request.rotational_components != "auto":
        return _explicit_component_selection(request.rotational_components, observable, planar)
    if not planar:
        return MOMENT_COMPONENTS if observable == "moments" else ROTATIONAL_COMPONENTS
    if observable == "moments":
        return _best_planar_moment_pair_from_cartesian_basis(
            atoms,
            coords,
            request.observations,
            cartesian_from_q,
        )
    return _best_planar_rotational_pair_from_cartesian_basis(
        atoms,
        coords,
        request.observations,
        cartesian_from_q,
    )


def _explicit_component_selection(selection: str, observable: str, planar: bool) -> tuple[str, ...]:
    if planar and selection == "ABC":
        raise ScientificValidationError(
            "Planar semiexperimental fits can use only AB, AC or BC; ABC is redundant"
        )
    rot_components = tuple(selection)
    if observable == "moments":
        return tuple(ROTATIONAL_TO_MOMENT_COMPONENT[item] for item in rot_components)
    return rot_components


def _best_planar_rotational_pair(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    prims: object,
    u_matrix: np.ndarray,
) -> tuple[str, ...]:
    candidates = (("A", "B"), ("A", "C"), ("B", "C"))
    cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    full = _rotational_constants_cartesian_jacobian(atoms, coords, observations) @ cartesian_from_q
    best = candidates[0]
    best_score = (-1, float("inf"), float("inf"))
    for pair in candidates:
        subset = _select_raw_components(full, ROTATIONAL_COMPONENTS, pair)
        singular = np.linalg.svd(subset, compute_uv=False)
        rank = int(
            np.sum(
                singular
                > max(subset.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
            )
        )
        cond = (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0.0
            else float("inf")
        )
        stability = _planar_moment_pair_stability(
            observations, _rotational_pair_to_moment_pair(pair)
        )
        score = (rank, stability, cond)
        if _planar_pair_score_is_better(score, best_score):
            best = pair
            best_score = score
    return best


def _best_planar_moment_pair(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    prims: object,
    u_matrix: np.ndarray,
) -> tuple[str, ...]:
    cartesian_from_q = _gic_cartesian_projector(prims, u_matrix, coords)
    return _best_planar_moment_pair_from_cartesian_basis(
        atoms, coords, observations, cartesian_from_q
    )


def _best_planar_moment_pair_from_cartesian_basis(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    cartesian_from_q: np.ndarray,
) -> tuple[str, ...]:
    candidates = (("Ia", "Ib"), ("Ia", "Ic"), ("Ib", "Ic"))
    full = _moments_cartesian_jacobian(atoms, coords, observations) @ cartesian_from_q
    best = candidates[0]
    best_score = (-1, float("inf"), float("inf"))
    for pair in candidates:
        subset = _select_raw_components(full, MOMENT_COMPONENTS, pair)
        singular = np.linalg.svd(subset, compute_uv=False)
        rank = int(
            np.sum(
                singular
                > max(subset.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
            )
        )
        cond = (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0.0
            else float("inf")
        )
        stability = _planar_moment_pair_stability(observations, pair)
        score = (rank, stability, cond)
        if _planar_pair_score_is_better(score, best_score):
            best = pair
            best_score = score
    return best


def _best_planar_rotational_pair_from_cartesian_basis(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    cartesian_from_q: np.ndarray,
) -> tuple[str, ...]:
    candidates = (("A", "B"), ("A", "C"), ("B", "C"))
    full = _rotational_constants_cartesian_jacobian(atoms, coords, observations) @ cartesian_from_q
    best = candidates[0]
    best_score = (-1, float("inf"), float("inf"))
    for pair in candidates:
        subset = _select_raw_components(full, ROTATIONAL_COMPONENTS, pair)
        singular = np.linalg.svd(subset, compute_uv=False)
        rank = int(
            np.sum(
                singular
                > max(subset.shape) * np.finfo(float).eps * (singular[0] if singular.size else 0.0)
            )
        )
        cond = (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0.0
            else float("inf")
        )
        stability = _planar_moment_pair_stability(
            observations, _rotational_pair_to_moment_pair(pair)
        )
        score = (rank, stability, cond)
        if _planar_pair_score_is_better(score, best_score):
            best = pair
            best_score = score
    return best


def _planar_pair_score_is_better(
    score: tuple[int, float, float],
    best_score: tuple[int, float, float],
) -> bool:
    if score[0] != best_score[0]:
        return score[0] > best_score[0]
    if score[1] != best_score[1]:
        return score[1] < best_score[1]
    return score[2] < best_score[2]


def _rotational_pair_to_moment_pair(pair: tuple[str, str]) -> tuple[str, str]:
    return tuple(ROTATIONAL_TO_MOMENT_COMPONENT[item] for item in pair)


def _planar_moment_pair_stability(
    observations: tuple[IsotopologueObservation, ...],
    pair: tuple[str, str],
) -> float:
    pair_set = set(pair)
    values = []
    for obs in observations:
        moments = _constants_to_moments(obs.corrected.as_tuple())
        sigmas = tuple(
            (1.0 / weight) ** 0.5 if weight > 0.0 else float("inf")
            for weight in _moment_weights(obs)
        )
        ia, ib, ic = moments
        sia, sib, sic = sigmas
        if pair_set == {"Ia", "Ib"}:
            omitted = ic
            predicted = ia + ib
            sigma = (sia * sia + sib * sib) ** 0.5
        elif pair_set == {"Ia", "Ic"}:
            omitted = ib
            predicted = ic - ia
            sigma = (sic * sic + sia * sia) ** 0.5
        elif pair_set == {"Ib", "Ic"}:
            omitted = ia
            predicted = ic - ib
            sigma = (sic * sic + sib * sib) ** 0.5
        else:
            return float("inf")
        scale = max(abs(omitted), 1.0e-12)
        consistency = abs(predicted - omitted) / scale
        values.append(consistency + sigma / scale)
    if not values:
        return float("inf")
    # Deterministic RMS score: lower means a more stable omitted planar component.
    return float(np.sqrt(np.mean(np.square(values))))


def _select_raw_components(
    raw: np.ndarray, component_names: tuple[str, ...], selected: tuple[str, ...]
) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, len(component_names)))
        idx = [component_names.index(item) for item in selected]
        return arr[:, idx].reshape(-1)
    idx = []
    for block in range(arr.shape[0] // len(component_names)):
        idx.extend(block * len(component_names) + component_names.index(item) for item in selected)
    return arr[idx, :]


def _constants_to_moments(constants: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(ROTCONST_TO_MOMENT / value if value > 0.0 else 0.0 for value in constants)


def _moment_weights(obs: IsotopologueObservation) -> tuple[float, float, float]:
    if obs.weights is None:
        return (1.0, 1.0, 1.0)
    constants = obs.corrected.as_tuple()
    sigmas_b = tuple((1.0 / weight) ** 0.5 for weight in obs.weights.as_tuple())
    weights = []
    for b_value, sigma_b in zip(constants, sigmas_b):
        sigma_i = abs(ROTCONST_TO_MOMENT * sigma_b / (b_value * b_value))
        weights.append(1.0 / (sigma_i * sigma_i) if sigma_i > 0.0 else 1.0)
    return tuple(weights)


def _predicate_observations(
    predicates: tuple[QMParameterPredicate, ...],
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    values = []
    weights = []
    row_labels = []
    for predicate in predicates:
        primitive = _predicate_primitive(predicate)
        if primitive is not None:
            values.append(_predicate_observed_value(predicate, primitive))
            weights.append(_predicate_weight(predicate, primitive))
            row_labels.append((predicate.source, _primitive_text(primitive)))
        else:
            matches = _predicate_indices(predicate, labels)
            if not matches:
                raise ScientificValidationError(
                    f"QM predicate did not match any GIC: {predicate.label_pattern}"
                )
            for idx in matches:
                values.append(predicate.value)
                weights.append(predicate.weight)
                row_labels.append((predicate.source, labels[idx]))
    return np.array(values, dtype=float), np.array(weights, dtype=float), row_labels


def _predicate_values(
    predicates: tuple[QMParameterPredicate, ...],
    labels: tuple[str, ...],
    q_values: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    values = []
    for predicate in predicates:
        primitive = _predicate_primitive(predicate)
        if primitive is not None:
            values.append(float(eval_primitives([primitive], coords)[0]))
        else:
            for idx in _predicate_indices(predicate, labels):
                values.append(float(q_values[idx]))
    return np.array(values, dtype=float)


def _predicate_indices(predicate: QMParameterPredicate, labels: tuple[str, ...]) -> list[int]:
    pattern = predicate.label_pattern.lower()
    return [idx for idx, label in enumerate(labels) if pattern in label.lower()]


def _predicate_primitive(predicate: QMParameterPredicate) -> Primitive | None:
    primitives = _primitives_from_fixed_pattern(predicate.label_pattern)
    if len(primitives) != 1:
        return None
    return _canonical_predicate_primitive(primitives[0])


def _canonical_predicate_primitive(primitive: Primitive) -> Primitive:
    atoms = tuple(int(atom) for atom in primitive.atoms)
    if primitive.kind == "bond" and len(atoms) == 2:
        return Primitive(primitive.kind, tuple(sorted(atoms)))
    if primitive.kind == "angle" and len(atoms) == 3:
        left, center, right = atoms
        if right < left:
            return Primitive(primitive.kind, (right, center, left))
        return primitive
    if primitive.kind == "dihedral" and len(atoms) == 4:
        reverse = tuple(reversed(atoms))
        if reverse < atoms:
            return Primitive(primitive.kind, reverse)
        return primitive
    if primitive.kind == "linear_bend" and len(atoms) == 3:
        left, center, right = atoms
        if right < left:
            return Primitive(primitive.kind, (right, center, left), primitive.mode)
        return primitive
    return primitive


def _predicate_observed_value(predicate: QMParameterPredicate, primitive: Primitive) -> float:
    value = float(predicate.value)
    if primitive.kind in {"angle", "dihedral", "out_of_plane", "linear_bend"}:
        return float(np.deg2rad(value))
    return value


def _predicate_weight(predicate: QMParameterPredicate, primitive: Primitive) -> float:
    sigma = float(predicate.sigma)
    if primitive.kind in {"angle", "dihedral", "out_of_plane", "linear_bend"}:
        sigma = float(np.deg2rad(sigma))
    if sigma <= 0.0:
        raise ValueError("QM predicate sigma must be positive")
    return 1.0 / (sigma * sigma)


def _is_planar(coords: np.ndarray, tol: float = 1.0e-3) -> bool:
    centered = np.asarray(coords, dtype=float) - np.mean(coords, axis=0)
    if centered.shape[0] < 3:
        return False
    singular = np.linalg.svd(centered, compute_uv=False)
    scale = max(float(singular[0]), 1.0)
    return float(singular[-1]) / scale < tol


def _residual_rows(
    model: MeasurementModel,
    calculated: np.ndarray,
    observed: np.ndarray,
) -> tuple[SemiexperimentalResidual, ...]:
    rows = []
    for idx, (isotopologue, label) in enumerate(model.labels):
        rows.append(
            SemiexperimentalResidual(
                isotopologue,
                label,
                float(observed[idx]),
                float(calculated[idx]),
                float(observed[idx] - calculated[idx]),
            )
        )
    return tuple(rows)


def _rotational_constant_rows(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    observations: tuple[IsotopologueObservation, ...],
    excluded_rotational_constants: tuple[str, ...] = (),
) -> tuple[SemiexperimentalRotationalConstantComparison, ...]:
    calculated = _constants_vector(atoms, coords, observations).reshape(
        (-1, len(ROTATIONAL_COMPONENTS))
    )
    rows: list[SemiexperimentalRotationalConstantComparison] = []
    excluded = {str(item) for item in excluded_rotational_constants}
    for obs, calc_triplet in zip(observations, calculated):
        for component, observed_value, calculated_value in zip(
            ROTATIONAL_COMPONENTS,
            obs.corrected.as_tuple(),
            calc_triplet,
        ):
            if f"{obs.label}:{component}" in excluded:
                continue
            rows.append(
                SemiexperimentalRotationalConstantComparison(
                    obs.label,
                    component,
                    float(observed_value),
                    float(calculated_value),
                    float(observed_value - calculated_value),
                )
            )
    return tuple(rows)


def _parameters(
    labels: tuple[str, ...],
    values: np.ndarray,
    active_mask: np.ndarray,
    transform: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    class_by_gic: tuple[str, ...] = (),
) -> tuple[SemiexperimentalParameter, ...]:
    params = []
    active_positions = {idx: pos for pos, idx in enumerate(np.where(active_mask)[0])}
    for idx, label in enumerate(labels):
        active = bool(active_mask[idx])
        parameter_class = class_by_gic[idx] if idx < len(class_by_gic) else ""
        sigma = 0.0
        if active:
            pos = active_positions[idx]
            if transform is not None and transform.size:
                row = np.asarray(transform[pos, :], dtype=float)
                active = bool(np.linalg.norm(row) > 1.0e-12)
            else:
                row = np.zeros(0, dtype=float)
        if active:
            pos = active_positions[idx]
            if (
                transform is not None
                and covariance is not None
                and transform.size
                and covariance.size
            ):
                if covariance.shape == (row.size, row.size):
                    variance = float(row @ covariance @ row)
                    sigma = float(np.sqrt(max(variance, 0.0)))
            elif covariance is not None and covariance.size and pos < covariance.shape[0]:
                sigma = float(np.sqrt(max(float(covariance[pos, pos]), 0.0)))
        params.append(
            SemiexperimentalParameter(label, float(values[idx]), sigma, active, parameter_class)
        )
    return tuple(params)


def _sonic_parameters_from_cartesian_covariance(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    cartesian_from_parameters: np.ndarray,
    covariance: np.ndarray,
) -> tuple[SemiexperimentalParameter, ...]:
    """Represent any final fit in SONICs and propagate its Cartesian covariance."""

    z_numbers = np.asarray([_atomic_number(symbol) for symbol in atoms], dtype=int)
    prims, u_matrix, labels = _gic_model(np.asarray(coords, dtype=float), z_numbers)
    values = _gic_values(prims, u_matrix, np.asarray(coords, dtype=float))
    cartesian_map = np.asarray(cartesian_from_parameters, dtype=float)
    reduced_covariance = np.asarray(covariance, dtype=float)
    if (
        cartesian_map.ndim != 2
        or reduced_covariance.ndim != 2
        or reduced_covariance.shape[0] != reduced_covariance.shape[1]
        or cartesian_map.shape[1] != reduced_covariance.shape[0]
    ):
        sonic_covariance = np.zeros((len(labels), len(labels)), dtype=float)
    else:
        cartesian_covariance = cartesian_map @ reduced_covariance @ cartesian_map.T
        sonic_b = np.asarray(u_matrix, dtype=float).T @ b_matrix_analytic(prims, coords)
        sonic_covariance = sonic_b @ cartesian_covariance @ sonic_b.T
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        variance = (
            float(sonic_covariance[index, index])
            if index < sonic_covariance.shape[0]
            else 0.0
        )
        sigma = float(np.sqrt(max(variance, 0.0)))
        rows.append(SemiexperimentalParameter(label, float(value), sigma, sigma > 0.0))
    return tuple(rows)


def _covariance(jac: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if jac.size == 0:
        return np.zeros((0, 0), dtype=float)
    try:
        _u, singular, vh = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(jac.T @ jac, rcond=1.0e-10)
    if not singular.size:
        return np.zeros((jac.shape[1], jac.shape[1]), dtype=float)
    tol = max(jac.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0) * 100.0
    inv_s2 = np.zeros_like(singular)
    keep = singular > tol
    inv_s2[keep] = 1.0 / (singular[keep] * singular[keep])
    return (vh.T * inv_s2) @ vh


def _diagnostics(
    weighted_jac: np.ndarray,
    weighted_residual: np.ndarray,
    *,
    convergence_reason: str,
    damping: float,
    accepted_steps: int,
    rejected_steps: int,
    max_iterations: int,
    n_optimized_parameters: int,
    observable: str,
    components: tuple[str, ...],
    planar: bool,
    auto_pruned_parameters: tuple[str, ...] = (),
    prune_condition_target: float = 0.0,
    gicforge_calls: int = 0,
    coordinate_model_reuse_steps: int = 0,
    trust_radius: float = 0.0,
    last_trust_ratio: float = 0.0,
    last_line_search_scale: float = 0.0,
    b_projector_analytic_refreshes: int = 0,
    b_projector_secant_updates: int = 0,
    b_projector_secant_rejections: int = 0,
    last_b_projector_secant_error: float = 0.0,
    parameter_scale_min: float = 1.0,
    parameter_scale_max: float = 1.0,
    robust_loss: str = "none",
    robust_scale: float = 0.0,
    robust_downweighted_observations: int = 0,
    robust_downweighted_isotopologues: int = 0,
    linear_solver: str = "svd_more_hebden_trust_region",
    coordinate_model: str = "gic",
) -> SemiexperimentalFitDiagnostics:
    conditioning = rank_condition(weighted_jac)
    incremental_rank = _incremental_column_rank(weighted_jac)
    obj = objective(weighted_residual)
    dof = max(weighted_residual.size - weighted_jac.shape[1], 1) if weighted_jac.ndim == 2 else 1
    return SemiexperimentalFitDiagnostics(
        convergence_reason=convergence_reason,
        objective=obj,
        weighted_rms=float(np.sqrt(np.mean(weighted_residual * weighted_residual)))
        if weighted_residual.size
        else 0.0,
        reduced_chi_square=float((weighted_residual @ weighted_residual) / dof)
        if weighted_residual.size
        else 0.0,
        rank=conditioning.rank,
        incremental_rank=incremental_rank,
        condition_number=conditioning.condition_number,
        damping=float(damping),
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        max_iterations=int(max_iterations),
        n_optimized_parameters=int(n_optimized_parameters),
        observable=observable,
        components=components,
        planar=planar,
        auto_pruned_parameters=auto_pruned_parameters,
        prune_condition_target=float(prune_condition_target),
        gicforge_calls=int(gicforge_calls),
        coordinate_model_reuse_steps=int(coordinate_model_reuse_steps),
        trust_radius=float(trust_radius),
        last_trust_ratio=float(last_trust_ratio),
        last_line_search_scale=float(last_line_search_scale),
        b_projector_analytic_refreshes=int(b_projector_analytic_refreshes),
        b_projector_secant_updates=int(b_projector_secant_updates),
        b_projector_secant_rejections=int(b_projector_secant_rejections),
        last_b_projector_secant_error=float(last_b_projector_secant_error),
        parameter_scale_min=float(parameter_scale_min),
        parameter_scale_max=float(parameter_scale_max),
        robust_loss=str(robust_loss),
        robust_scale=float(robust_scale),
        robust_downweighted_observations=int(robust_downweighted_observations),
        robust_downweighted_isotopologues=int(robust_downweighted_isotopologues),
        linear_solver=str(linear_solver),
        coordinate_model=coordinate_model,
    )


def _least_squares_hessian(weighted_jac: np.ndarray) -> np.ndarray:
    if weighted_jac.size == 0:
        return np.zeros((0, 0), dtype=float)
    return 2.0 * (weighted_jac.T @ weighted_jac)


def _correlation(covariance: np.ndarray) -> np.ndarray:
    if covariance.size == 0:
        return np.zeros((0, 0), dtype=float)
    diag = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denom = np.outer(diag, diag)
    corr = np.zeros_like(covariance)
    np.divide(covariance, denom, out=corr, where=denom > 0.0)
    return np.clip(corr, -1.0, 1.0)


def _stationary_point_type(eigenvalues: np.ndarray) -> str:
    if eigenvalues.size == 0:
        return "not_checked"
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tol = max(1.0e-10, 10.0 * eigenvalues.size * np.finfo(float).eps * scale)
    if np.all(eigenvalues > tol):
        return "minimum"
    if np.any(eigenvalues < -tol):
        return "transition_state_or_saddle"
    return "flat_or_rank_deficient"


def _isotopes_for_observation(
    atoms: list[str] | tuple[str, ...], obs: IsotopologueObservation
) -> list[int | None]:
    isotopes: list[int | None] = [None] * len(atoms)
    for atom_index, isotope_a in obs.substitutions.items():
        if atom_index < 1 or atom_index > len(atoms):
            raise ScientificValidationError(
                f"Isotopologue {obs.label} substitution atom {atom_index} is out of range"
            )
        isotopes[atom_index - 1] = int(isotope_a)
    return isotopes


def _mass_vector_for_observation(
    atoms: list[str] | tuple[str, ...],
    obs: IsotopologueObservation,
) -> np.ndarray:
    return _mass_vector_for_isotopes(atoms, _isotopes_for_observation(atoms, obs))


def _mass_vector_for_isotopes(
    atoms: list[str] | tuple[str, ...],
    isotopes: list[int | None] | tuple[int | None, ...],
) -> np.ndarray:
    isotope_key = tuple(0 if item is None else int(item) for item in isotopes)
    return np.asarray(
        _cached_mass_tuple(tuple(str(atom) for atom in atoms), isotope_key), dtype=float
    )


@lru_cache(maxsize=4096)
def _cached_mass_tuple(atoms: tuple[str, ...], isotope_key: tuple[int, ...]) -> tuple[float, ...]:
    if len(atoms) != len(isotope_key):
        raise ValueError("Mass-cache atom/isotope length mismatch")
    masses: list[float] = []
    for atom, isotope_a in zip(atoms, isotope_key):
        z_number = geometry_atomic_number(atom)
        if z_number is None:
            raise ValueError(f"Unknown atomic symbol: {atom}")
        if isotope_a == 0:
            try:
                masses.append(float(get_default_isotope(z_number).mass))
            except Exception:
                masses.append(float(atomic_mass(z_number)))
        else:
            try:
                isotope = get_isotope(z_number, int(isotope_a))
                if isotope is None:
                    isotope = get_default_isotope(z_number)
                masses.append(float(isotope.mass))
            except Exception:
                masses.append(float(atomic_mass(z_number)))
    return tuple(masses)


def _validate_observations(observations: tuple[IsotopologueObservation, ...], natoms: int) -> None:
    for obs in observations:
        for atom_index in obs.substitutions:
            if atom_index < 1 or atom_index > natoms:
                raise ScientificValidationError(
                    f"Isotopologue {obs.label} substitution atom {atom_index} is out of range"
                )
        if any(value <= 0.0 for value in obs.corrected.as_tuple()):
            raise ScientificValidationError(
                f"Isotopologue {obs.label} has non-positive equilibrium rotational constants"
            )
        if obs.weights is not None and any(value <= 0.0 for value in obs.weights.as_tuple()):
            raise ScientificValidationError(
                f"Isotopologue {obs.label} has non-positive least-squares weights"
            )


def _topology_lock(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    *,
    validate_contacts: bool = True,
    context: str = "initial topology validation",
) -> TopologyLock:
    coords = np.asarray(coords, dtype=float)
    atomic_numbers = tuple(_atomic_number(symbol) for symbol in atoms)
    try:
        _continuous, graph, _ringset, _synthons, _aromaticity = build_topology_objects(
            coords,
            np.asarray(atomic_numbers, dtype=int),
        )
    except Exception as exc:
        raise ScientificValidationError(f"Initial topology validation failed: {exc}") from exc
    bonds = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in graph.bonds))
    adjacency = tuple(
        tuple(sorted(int(item) for item in graph.adjacency[index]))
        for index in range(len(atomic_numbers))
    )
    lock = TopologyLock(atomic_numbers=atomic_numbers, bonds=bonds, adjacency=adjacency)
    if validate_contacts:
        _validate_spurious_contacts(coords, lock, context=context)
    return lock


def _validate_locked_topology(
    atoms: list[str] | tuple[str, ...],
    coords: np.ndarray,
    reference: TopologyLock,
    *,
    context: str = "semiexperimental fit",
) -> None:
    current = _topology_lock(atoms, coords, validate_contacts=False)
    if (
        current.atomic_numbers == reference.atomic_numbers
        and current.bonds == reference.bonds
        and current.adjacency == reference.adjacency
    ):
        _validate_spurious_contacts(coords, reference, context=context)
        return
    added = sorted(set(current.bonds) - set(reference.bonds))
    removed = sorted(set(reference.bonds) - set(current.bonds))
    details: list[str] = []
    if added:
        details.append("added bonds " + ", ".join(_bond_label(pair) for pair in added[:8]))
    if removed:
        details.append("removed bonds " + ", ".join(_bond_label(pair) for pair in removed[:8]))
    if len(added) > 8:
        details.append(f"{len(added) - 8} additional added bonds")
    if len(removed) > 8:
        details.append(f"{len(removed) - 8} additional removed bonds")
    suffix = "; " + "; ".join(details) if details else ""
    raise ScientificValidationError(
        f"Topology changed during {context}; rejecting geometry{suffix}"
    )


def _validate_spurious_contacts(
    coords: np.ndarray, reference: TopologyLock, *, context: str
) -> None:
    coords = np.asarray(coords, dtype=float)
    bonded = set(reference.bonds)
    contacts: list[tuple[int, int, float]] = []
    for i, zi in enumerate(reference.atomic_numbers):
        if zi != 1:
            continue
        ri = covalent_radius(zi)
        if ri is None:
            continue
        for j in range(i + 1, len(reference.atomic_numbers)):
            zj = reference.atomic_numbers[j]
            if zj != 1 or (i, j) in bonded:
                continue
            rj = covalent_radius(zj)
            if rj is None:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= 1.25 * (float(ri) + float(rj)):
                contacts.append((i, j, distance))
    if not contacts:
        return
    preview = ", ".join(f"{i + 1}-{j + 1} ({distance:.3f} A)" for i, j, distance in contacts[:8])
    extra = f"; {len(contacts) - 8} additional H-H contacts" if len(contacts) > 8 else ""
    raise ScientificValidationError(
        f"Spurious nonbonded H-H contact during {context}: {preview}{extra}"
    )


def _bond_label(pair: tuple[int, int]) -> str:
    return f"{pair[0] + 1}-{pair[1] + 1}"


def _atomic_number(symbol: str) -> int:
    z = geometry_atomic_number(symbol)
    if z is None:
        raise ScientificValidationError(f"Unknown element symbol {symbol}")
    return int(z)
