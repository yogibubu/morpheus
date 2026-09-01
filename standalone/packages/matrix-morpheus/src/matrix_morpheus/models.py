"""Stable data contracts for MORPHEUS fitting and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_smith.survibfit.primitives import Primitive

from .kraitchman import KraitchmanComparison, KraitchmanSeedResult
from .statistics import SemiexperimentalWeightDiagnostic


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
