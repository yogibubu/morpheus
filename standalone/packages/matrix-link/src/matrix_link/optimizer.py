from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from itertools import combinations
import json
import math
from pathlib import Path
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from matrix_core import require_authorized_descendant_calculation
from matrix_chem.geometry_alignment import (
    aligned_cartesian_displacement,
    kabsch_align,
    kabsch_rotation,
    rotate_cartesian_derivatives,
)
from matrix_chem.xyzin_geometry import read_xyzin_geometry
from .internal_coordinates import (
    cartesian_from_internal_jacobian,
    internal_from_cartesian_jacobian,
    transport_internal_hessian,
)

from .scan import (
    ANGSTROM_TO_BOHR,
    PointEvaluationResult,
    PESExplorationPolicy,
    QMScanBackend,
    ScanPoint,
    _coordinate_index,
    _normalized_backend_name,
    _resolve_zaff_backend,
    coordinate_direction_from_gic,
    point_result_to_json,
    read_point_result,
    run_qm_scan_points,
    write_point_result,
    prepare_pes_exploration_geometry,
)
from .optimizer_controller import ControllerSettings, TrustRegionController, TrialDecision
from .optimizer_convergence import (
    GEOMETRY_SEED_ENERGY_TOLERANCE_HARTREE as GEOMETRY_SEED_ENERGY_TOLERANCE_HARTREE,
    GEOMETRY_SEED_FINAL_ENERGY_RISE_TOLERANCE_HARTREE as GEOMETRY_SEED_FINAL_ENERGY_RISE_TOLERANCE_HARTREE,
    GEOMETRY_SEED_GRADIENT_NORM_TOLERANCE_HARTREE_PER_BOHR as GEOMETRY_SEED_GRADIENT_NORM_TOLERANCE_HARTREE_PER_BOHR,
    convergence_force_satisfied as _convergence_force_satisfied,
    convergence_gradient as _convergence_gradient,
    gaussian_like_convergence as _gaussian_like_convergence,
    gdv_transition_state_prospective_convergence as _gdv_transition_state_prospective_convergence,
)
from .optimizer_hessian_updates import (
    bfgs_update as _bfgs_update,
    bofill_update as _bofill_update,
    hessian_is_usable as _hessian_is_usable,
    optimizer_hessian_index as _optimizer_hessian_index,
    sr1_update as _sr1_update,
    stored_hessian_is_numerically_usable as _stored_hessian_is_numerically_usable,
)
from .optimizer_metrics import rms as _rms
from .saddle_rfo import (
    condition_aware_reaction_mode,
    gdv_dxrfo_step,
    symmetric_multisecant_hessian_refresh,
)

if TYPE_CHECKING:
    from .chart_lifecycle import ChartLifecycleController, ChartLifecycleResult
    from .frozen_chart_replay import FrozenChartReference


OPTIMIZER_TRACE_SCHEMA = "matrix.trinity.information_efficient_optimizer.trace.v1"
OPTIMIZER_SUMMARY_SCHEMA = "matrix.trinity.information_efficient_optimizer.summary.v1"
OPTIMIZER_CACHE_SCHEMA = "matrix.trinity.information_efficient_optimizer.cache.v1"
OPTIMIZER_HESSIAN_SCHEMA = "matrix.trinity.information_efficient_optimizer.hessian.v1"
TRANSITION_STATE_VALIDATION_SCHEMA = "matrix.link.transition_state_validation.v1"
IRC_VERIFICATION_SCHEMA = "matrix.link.irc_verification.v1"
OPTIMIZER_DAMPING_MIN = 1.0e-12
GDV_TS_INITIAL_TRUST_RADIUS = 0.3
GDV_D2CORX_MAX_HISTORY_DISTANCE = 0.6
GDV_D2CORX_GRADIENT_ERROR = 1.0e-6
GDV_REDQ2X_HARD_FAILURE_RETRY_FACTORS = tuple(
    0.5**attempt for attempt in range(10)
)
GDV_LENGTH_COORDINATE_FUNCTIONS = frozenset(
    {"R", "H", "FC_DIST", "FCA_DIST", "FTRANS", "CENTER_ATOM_DIST"}
)
GDV_ANGULAR_COORDINATE_FUNCTIONS = frozenset(
    {"A", "L", "D", "U", "IMPD", "RPCB", "RPCK", "FROT"}
)
GDV_DIHEDRAL_COORDINATE_FUNCTIONS = frozenset({"D", "IMPD", "RPCK"})

class ElectronicStateResolutionError(RuntimeError):
    """APOC could not establish one unambiguous state after root expansion."""


@dataclass(frozen=True)
class InitialSymmetryBreakingModeDisplacement:
    """Explicit consent and source mode for a pure symmetry-lowering displacement."""

    source_coordinates_angstrom: np.ndarray
    source_cartesian_mode: np.ndarray
    source_mode_index: int
    allow_symmetry_lowering: bool = False


@dataclass(frozen=True)
class OptimizerSettings:
    max_steps: int = 50
    trust_radius: float = 0.2
    max_trust_radius: float = 0.3
    min_trust_radius: float = 1.0e-4
    cartesian_trust_tolerance: float = 1.0e-3
    cartesian_trust_max_iterations: int = 48
    gradient_tolerance: float = 4.5e-4
    step_tolerance: float = 1.8e-3
    energy_tolerance: float = 1.0e-6
    # ``stationary`` is the production geometry/frequency contract.  The
    # role-based ``geometry_seed`` profile reproduces the normal xTB stopping
    # test for lower-level geometries used only to seed a more expensive PES:
    # |dE| <= 5e-6 Eh, ||g_tangent||_2 <= 1e-3 Eh/bohr, with a non-increasing
    # final accepted energy.  It is never selected from the backend identity.
    convergence_profile: str = "stationary"
    max_force_tolerance: float | None = None
    rms_force_tolerance: float | None = None
    max_displacement_tolerance: float | None = None
    rms_displacement_tolerance: float | None = None
    # Protocol defaults for numerical derivatives of energy-only surfaces.
    # Values are expressed in the native coordinate unit: Å for stretches and
    # radians for angular coordinates.  ``one_sided_only`` is the frozen
    # geometry-seed policy; ``adaptive_two_sided`` remains available for
    # explicitly selected stationary-point protocols.
    fd_stencil_policy: str = "adaptive_two_sided"
    fd_step: float = 0.005
    fd_hard_characteristic_scale: float = 0.05
    fd_soft_characteristic_scale: float = 0.20
    fd_min_step: float = 0.002
    fd_max_step: float = 0.02
    energy_noise: float = 1.0e-7
    numerical_energy_noise_floor: float = 1.0e-7
    energy_noise_samples: int = 0
    two_sided: bool = True
    one_sided_until_convergence: bool = True
    final_gradient_verification: bool = True
    final_hessian_rescale_min: float = 0.1
    final_hessian_rescale_max: float = 10.0
    adaptive_fd_mode: bool = True
    fd_two_sided_switch_force: float = 7.0e-4
    fd_totally_symmetric_only: bool = False
    fd_initial_class_threshold_fraction: float = 0.10
    fd_class_threshold_release_factor: float = 10.0
    fd_class_screen_audit_interval: int = 3
    selective_fd_refresh: bool = False
    fd_refresh_interval: int = 3
    fd_gradient_change_tolerance: float = 1.0e-4
    selective_min_refresh_fraction: float = 0.25
    selective_coupling_threshold: float = 0.05
    selective_fallback_rejections: int = 1
    selective_fallback_gradient_growth: float = 1.5
    surrogate_max_samples: int = 12
    fd_parallel_workers: int = 1
    hessian_coupling_threshold: float = 1.0e-8
    sparse_hessian_updates: bool = False
    symmetry_reduction: bool = False
    symmetrize_analytic_gradients: bool = True
    prefer_analytic_gradient: bool = True
    cache_tolerance: float = 1.0e-10
    resume: bool = False
    min_abs_metric_diagonal: float = 1.0e-8
    acceptance_threshold: float = 0.10
    bfgs_damping: bool = True
    min_hessian_eigenvalue: float = 1.0e-4
    max_hessian_condition: float = 1.0e8
    # Retained as a compatibility field; the frozen protocol forbids
    # in-cycle geometric backtracking and therefore requires zero.
    line_search_reductions: int = 0
    energy_increase_tolerance: float | None = None
    hessian_update: str = "auto"
    initial_hessian_model: str = "lindh_swart_special"
    enable_gdiis: bool = True
    coordinate_drift_warning: float = 0.25
    min_interatomic_distance: float = 0.35
    gdiis_history: int = 6
    gdiis_start: int = 3
    gdiis_max_condition: float = 5.0e3
    gdiis_max_coefficient: float = 2.0
    hessian_bad_ratio_limit: int = 2
    max_stagnation_recoveries: int = 2
    stagnation_step_floor: float = 1.0e-10
    fragment_radial_curvature: float | None = None
    fragment_tangential_curvature: float | None = None
    fragment_rotation_curvature: float | None = None
    fragment_rotation_rebase_threshold: float = 1.0
    coordinate_schedule: str = "joint"
    coordinate_phase_max_steps: int = 8
    coordinate_phase_gradient_factor: float = 3.0
    far_from_minimum_force_factor: float = 10.0
    far_from_minimum_displacement_factor: float = 2.0
    fixed_atoms: tuple[int, ...] = ()
    freeze_inactive_sonic: bool = True
    backtransform_continuation_step: float = 0.12
    backtransform_max_substeps: int = 32
    backtransform_method: str = "hybrid"
    geodesic_jacobian_displacement: float = 1.0e-4
    rigid_reference_groups: tuple[tuple[int, ...], ...] = ()
    include_cv_exponential_field: bool = False
    stationary_point: str = "minimum"
    transition_mode: int = 0
    transition_mode_overlap_threshold: float = 0.50
    transition_index_probe_rms_angstrom: float = 0.005
    compute_final_hessian: bool = False
    require_exact_final_hessian: bool = True
    verify_irc: bool = True
    irc_step_size: float = 0.05
    irc_max_steps: int = 64
    irc_gradient_tolerance: float = 1.0e-4
    coordinate_parallel_workers: int = 1

    def __post_init__(self) -> None:
        for name in (
            "trust_radius",
            "max_trust_radius",
            "min_trust_radius",
            "cartesian_trust_tolerance",
            "gradient_tolerance",
            "step_tolerance",
            "energy_tolerance",
            "fd_step",
            "fd_hard_characteristic_scale",
            "fd_soft_characteristic_scale",
            "fd_min_step",
            "fd_max_step",
            "energy_noise",
            "numerical_energy_noise_floor",
            "fd_gradient_change_tolerance",
            "selective_min_refresh_fraction",
            "selective_coupling_threshold",
            "selective_fallback_gradient_growth",
            "hessian_coupling_threshold",
            "cache_tolerance",
            "min_abs_metric_diagonal",
            "acceptance_threshold",
            "min_hessian_eigenvalue",
            "max_hessian_condition",
            "coordinate_drift_warning",
            "min_interatomic_distance",
            "gdiis_max_condition",
            "gdiis_max_coefficient",
            "fragment_radial_curvature",
            "fragment_tangential_curvature",
            "fragment_rotation_curvature",
            "fragment_rotation_rebase_threshold",
            "coordinate_phase_gradient_factor",
            "far_from_minimum_force_factor",
            "far_from_minimum_displacement_factor",
            "backtransform_continuation_step",
            "geodesic_jacobian_displacement",
            "transition_mode_overlap_threshold",
            "transition_index_probe_rms_angstrom",
            "irc_step_size",
            "irc_gradient_tolerance",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        max_force = (
            self.gradient_tolerance
            if self.max_force_tolerance is None
            else float(self.max_force_tolerance)
        )
        rms_force = (
            (2.0 / 3.0) * max_force
            if self.rms_force_tolerance is None
            else float(self.rms_force_tolerance)
        )
        max_disp = (
            self.step_tolerance
            if self.max_displacement_tolerance is None
            else float(self.max_displacement_tolerance)
        )
        rms_disp = (
            (2.0 / 3.0) * max_disp
            if self.rms_displacement_tolerance is None
            else float(self.rms_displacement_tolerance)
        )
        object.__setattr__(self, "max_force_tolerance", max_force)
        object.__setattr__(self, "rms_force_tolerance", rms_force)
        object.__setattr__(self, "max_displacement_tolerance", max_disp)
        object.__setattr__(self, "rms_displacement_tolerance", rms_disp)
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(
            self, "stationary_point", str(self.stationary_point).strip().lower().replace("-", "_")
        )
        object.__setattr__(
            self,
            "convergence_profile",
            str(self.convergence_profile).strip().lower().replace("-", "_"),
        )
        object.__setattr__(self, "transition_mode", int(self.transition_mode))
        object.__setattr__(self, "energy_noise_samples", int(self.energy_noise_samples))
        object.__setattr__(self, "fd_refresh_interval", int(self.fd_refresh_interval))
        object.__setattr__(
            self, "selective_fallback_rejections", int(self.selective_fallback_rejections)
        )
        object.__setattr__(self, "surrogate_max_samples", int(self.surrogate_max_samples))
        object.__setattr__(self, "fd_parallel_workers", int(self.fd_parallel_workers))
        object.__setattr__(
            self, "coordinate_parallel_workers", int(self.coordinate_parallel_workers)
        )
        object.__setattr__(self, "line_search_reductions", int(self.line_search_reductions))
        object.__setattr__(
            self,
            "cartesian_trust_max_iterations",
            int(self.cartesian_trust_max_iterations),
        )
        object.__setattr__(self, "gdiis_history", int(self.gdiis_history))
        object.__setattr__(self, "gdiis_start", int(self.gdiis_start))
        object.__setattr__(self, "hessian_bad_ratio_limit", int(self.hessian_bad_ratio_limit))
        object.__setattr__(self, "coordinate_phase_max_steps", int(self.coordinate_phase_max_steps))
        object.__setattr__(self, "backtransform_max_substeps", int(self.backtransform_max_substeps))
        object.__setattr__(self, "irc_max_steps", int(self.irc_max_steps))
        object.__setattr__(
            self, "coordinate_schedule", str(self.coordinate_schedule).strip().lower()
        )
        object.__setattr__(
            self, "backtransform_method", str(self.backtransform_method).strip().lower()
        )
        object.__setattr__(self, "fixed_atoms", tuple(int(index) for index in self.fixed_atoms))
        object.__setattr__(
            self,
            "rigid_reference_groups",
            tuple(tuple(int(index) for index in group) for group in self.rigid_reference_groups),
        )
        energy_increase = (
            max(5.0 * self.energy_noise, 0.1 * self.energy_tolerance)
            if self.energy_increase_tolerance is None
            else float(self.energy_increase_tolerance)
        )
        object.__setattr__(self, "energy_increase_tolerance", energy_increase)
        if not self.prefer_analytic_gradient:
            object.__setattr__(
                self,
                "energy_noise",
                max(float(self.energy_noise), float(self.numerical_energy_noise_floor)),
            )
        object.__setattr__(self, "hessian_update", str(self.hessian_update).strip().lower())
        object.__setattr__(
            self, "initial_hessian_model", str(self.initial_hessian_model).strip().lower()
        )
        object.__setattr__(
            self,
            "fd_stencil_policy",
            str(self.fd_stencil_policy).strip().lower().replace("-", "_"),
        )
        if self.fd_stencil_policy not in {"adaptive_two_sided", "one_sided_only"}:
            raise ValueError(
                "fd_stencil_policy must be 'adaptive_two_sided' or 'one_sided_only'"
            )
        if self.fd_stencil_policy == "one_sided_only":
            object.__setattr__(self, "two_sided", False)
            object.__setattr__(self, "one_sided_until_convergence", True)
            object.__setattr__(self, "final_gradient_verification", False)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.stationary_point not in {"minimum", "transition_state", "automatic"}:
            raise ValueError(
                "stationary_point must be 'minimum', 'transition_state' or 'automatic'"
            )
        if self.convergence_profile not in {"stationary", "geometry_seed"}:
            raise ValueError(
                "convergence_profile must be 'stationary' or 'geometry_seed'"
            )
        if (
            self.convergence_profile == "geometry_seed"
            and self.stationary_point != "minimum"
        ):
            raise ValueError(
                "the geometry_seed convergence profile is defined only for minima"
            )
        if self.transition_mode < 0:
            raise ValueError("transition_mode must be a non-negative Hessian eigenmode index")
        if not 0.0 <= self.transition_mode_overlap_threshold <= 1.0:
            raise ValueError("transition_mode_overlap_threshold must be in [0, 1]")
        if self.transition_index_probe_rms_angstrom <= 0.0:
            raise ValueError("transition-index probe RMS displacement must be positive")
        if self.irc_step_size <= 0.0 or self.irc_max_steps <= 0:
            raise ValueError("IRC step size and maximum steps must be positive")
        if self.irc_gradient_tolerance <= 0.0:
            raise ValueError("IRC gradient tolerance must be positive")
        if len(set(self.fixed_atoms)) != len(self.fixed_atoms) or any(
            index < 0 for index in self.fixed_atoms
        ):
            raise ValueError("fixed_atoms must contain unique non-negative indices")
        flattened_groups = [index for group in self.rigid_reference_groups for index in group]
        if any(not group for group in self.rigid_reference_groups):
            raise ValueError("rigid reference groups cannot be empty")
        if any(index < 0 for index in flattened_groups):
            raise ValueError("rigid reference group indices must be non-negative")
        if self.energy_noise_samples < 0:
            raise ValueError("energy_noise_samples must be non-negative")
        if self.fd_two_sided_switch_force <= 0.0:
            raise ValueError("fd_two_sided_switch_force must be positive")
        if not 0.0 <= self.fd_initial_class_threshold_fraction < 1.0:
            raise ValueError("fd_initial_class_threshold_fraction must lie in [0, 1)")
        if self.fd_class_threshold_release_factor <= 1.0:
            raise ValueError("fd_class_threshold_release_factor must exceed one")
        if self.fd_class_screen_audit_interval <= 0:
            raise ValueError("fd_class_screen_audit_interval must be positive")
        if self.fd_refresh_interval <= 0:
            raise ValueError("fd_refresh_interval must be positive")
        if self.surrogate_max_samples < 0:
            raise ValueError("surrogate_max_samples must be non-negative")
        if self.fd_parallel_workers <= 0:
            raise ValueError("fd_parallel_workers must be positive")
        if self.coordinate_parallel_workers <= 0:
            raise ValueError("coordinate_parallel_workers must be positive")
        if self.line_search_reductions != 0:
            raise ValueError("the frozen LINK protocol forbids in-cycle geometric backtracking")
        if self.gdiis_history < 3 or self.gdiis_start < 3:
            raise ValueError("GDIIS history and start must be at least three")
        if self.gdiis_max_condition <= 1.0 or self.gdiis_max_coefficient <= 0.0:
            raise ValueError("invalid GDIIS safeguards")
        if self.hessian_bad_ratio_limit <= 0:
            raise ValueError("hessian_bad_ratio_limit must be positive")
        if self.coordinate_phase_max_steps <= 0 or self.coordinate_phase_gradient_factor <= 0.0:
            raise ValueError("invalid coordinate phase controls")
        if self.far_from_minimum_force_factor <= 1.0:
            raise ValueError("far-from-minimum force factor must exceed one")
        if self.far_from_minimum_displacement_factor <= 1.0:
            raise ValueError("far-from-minimum displacement factor must exceed one")
        if self.backtransform_continuation_step <= 0.0 or self.backtransform_max_substeps <= 0:
            raise ValueError("invalid hybrid back-transform continuation controls")
        if self.backtransform_method not in {"hybrid", "geodesic"}:
            raise ValueError("backtransform_method must be 'hybrid' or 'geodesic'")
        if self.coordinate_schedule not in {
            "joint",
            "inter-intra-joint",
            "inter-intra-micro",
        }:
            raise ValueError(
                "coordinate_schedule must be 'joint', 'inter-intra-joint' or 'inter-intra-micro'"
            )
        fragment_curvatures = tuple(
            value
            for value in (
                self.fragment_radial_curvature,
                self.fragment_tangential_curvature,
                self.fragment_rotation_curvature,
            )
            if value is not None
        )
        if fragment_curvatures and min(fragment_curvatures) <= 0.0:
            raise ValueError("fragment Hessian curvatures must be positive")
        if not 0.0 < self.fragment_rotation_rebase_threshold < math.pi:
            raise ValueError("fragment rotation rebase threshold must lie between zero and pi")
        if (self.fragment_radial_curvature is None) != (self.fragment_tangential_curvature is None):
            raise ValueError("radial and tangential fragment curvatures must be set together")
        if self.trust_radius <= 0.0 or self.max_trust_radius <= 0.0:
            raise ValueError("trust radii must be positive")
        if not 0.0 < self.cartesian_trust_tolerance < 0.1:
            raise ValueError("Cartesian trust tolerance must lie in (0, 0.1)")
        if self.cartesian_trust_max_iterations <= 0:
            raise ValueError("Cartesian trust maximum iterations must be positive")
        if self.min_interatomic_distance <= 0.0:
            raise ValueError("min_interatomic_distance must be positive")
        if self.min_hessian_eigenvalue <= 0.0:
            raise ValueError("min_hessian_eigenvalue must be positive")
        if self.max_hessian_condition <= 1.0:
            raise ValueError("max_hessian_condition must be greater than one")
        if self.hessian_update not in {"auto", "bfgs", "sr1", "bofill"}:
            raise ValueError("hessian_update must be 'auto', 'bfgs', 'sr1' or 'bofill'")
        if self.initial_hessian_model not in {
            "auto",
            "lindh_swart_special",
            "berny",
            "almloef",
        }:
            raise ValueError(
                "initial_hessian_model must be 'auto', 'lindh_swart_special', "
                "'berny' or 'almloef'"
            )
        if self.fd_min_step <= 0.0 or self.fd_max_step < self.fd_min_step:
            raise ValueError("invalid finite-difference step bounds")
        if (
            self.final_hessian_rescale_min <= 0.0
            or self.final_hessian_rescale_max < self.final_hessian_rescale_min
        ):
            raise ValueError("invalid final Hessian rescale bounds")
        if self.fd_hard_characteristic_scale <= 0.0 or self.fd_soft_characteristic_scale <= 0.0:
            raise ValueError("hard and soft characteristic scales must be positive")
        if self.selective_min_refresh_fraction < 0.0 or self.selective_min_refresh_fraction > 1.0:
            raise ValueError("selective_min_refresh_fraction must be between zero and one")
        if self.selective_coupling_threshold < 0.0:
            raise ValueError("selective_coupling_threshold must be non-negative")
        if self.selective_fallback_rejections < 0:
            raise ValueError("selective_fallback_rejections must be non-negative")
        if self.max_stagnation_recoveries < 0:
            raise ValueError("max_stagnation_recoveries must be non-negative")
        if self.stagnation_step_floor <= 0.0:
            raise ValueError("stagnation_step_floor must be positive")
        if self.selective_fallback_gradient_growth < 1.0:
            raise ValueError("selective_fallback_gradient_growth must be at least one")
        for name in (
            "max_force_tolerance",
            "rms_force_tolerance",
            "max_displacement_tolerance",
            "rms_displacement_tolerance",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class OptimizerCoordinateModel:
    kind: str
    labels: tuple[str, ...]
    directions_angstrom: np.ndarray
    metric_diagonal: np.ndarray
    sonic_labels: tuple[str, ...] = ()
    sonic_from_coordinates: np.ndarray | None = None
    reference_values: np.ndarray | None = None
    sonic_definition: object | None = None
    pes_exploration: bool = False
    retained_group: str = ""
    rank_reduced_labels: tuple[str, ...] = ()
    typed_onic_runtime: object | None = None

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions_angstrom, dtype=float)
        if directions.ndim != 2:
            raise ValueError("coordinate directions must have shape ncoord x ncart")
        metric = np.asarray(self.metric_diagonal, dtype=float).reshape(-1)
        if metric.shape != (directions.shape[0],):
            raise ValueError("metric diagonal length must match coordinate count")
        if len(self.labels) != directions.shape[0]:
            raise ValueError("coordinate labels must match coordinate count")
        if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(metric)):
            raise ValueError("coordinate model contains non-finite values")
        sonic_labels = tuple(str(item) for item in self.sonic_labels)
        transform = self.sonic_from_coordinates
        if transform is not None:
            transform = np.asarray(transform, dtype=float)
            if self.kind != "sonic":
                raise ValueError("a SONIC projection requires kind='sonic'")
            if transform.shape != (len(sonic_labels), len(self.labels)):
                raise ValueError("SONIC projection must have shape nsonic x nvariables")
            if not np.all(np.isfinite(transform)) or np.linalg.matrix_rank(transform) < len(
                self.labels
            ):
                raise ValueError("SONIC projection must be finite and full column rank")
        typed_runtime = self.typed_onic_runtime
        if self.kind == "typed_onic":
            from .typed_onic import TypedOnicRuntime

            if not isinstance(typed_runtime, TypedOnicRuntime):
                raise TypeError("typed ONIC coordinate models require a TypedOnicRuntime")
            if self.sonic_definition is not None:
                raise ValueError("typed ONIC models cannot also carry a legacy SONIC definition")
            if tuple(self.labels) != typed_runtime.coordinate_identifiers:
                raise ValueError("typed ONIC model labels must preserve the frozen contract order")
            reference_shape = np.asarray(
                typed_runtime.definition.reference_coordinates_angstrom, dtype=float
            ).size
            if directions.shape[1] != reference_shape:
                raise ValueError("typed ONIC directions do not match the contract atom frame")
        elif typed_runtime is not None:
            raise ValueError("a TypedOnicRuntime requires kind='typed_onic'")
        reference = self.reference_values
        if reference is not None:
            reference = np.asarray(reference, dtype=float).reshape(-1)
            if reference.shape != (len(self.labels),) or not np.all(np.isfinite(reference)):
                raise ValueError("coordinate reference values must match coordinate labels")
            if typed_runtime is not None and not np.allclose(
                reference,
                typed_runtime.reference_values,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("typed ONIC reference values contradict the frozen runtime")
        object.__setattr__(self, "directions_angstrom", directions)
        object.__setattr__(self, "metric_diagonal", metric)
        object.__setattr__(self, "sonic_labels", sonic_labels)
        object.__setattr__(self, "sonic_from_coordinates", transform)
        object.__setattr__(self, "reference_values", reference)
        retained_group = str(self.retained_group).strip().upper()
        if self.pes_exploration and not retained_group:
            retained_group = "C1"
        object.__setattr__(self, "retained_group", retained_group)
        object.__setattr__(
            self,
            "rank_reduced_labels",
            tuple(str(item) for item in self.rank_reduced_labels),
        )


@dataclass(frozen=True)
class OptimizerEvaluation:
    q: np.ndarray
    coordinates_angstrom: np.ndarray
    result: PointEvaluationResult
    cache_hit: bool = False
    chart_epoch: int = 0

    @property
    def energy_hartree(self) -> float:
        if self.result.energy_hartree is None:
            raise ValueError("evaluation result has no energy")
        return float(self.result.energy_hartree)

    @property
    def gradient_hartree_per_bohr(self) -> np.ndarray | None:
        gradient = self.result.gradient_hartree_per_bohr
        if gradient is None:
            return None
        return np.asarray(gradient, dtype=float).reshape(-1)


@dataclass(frozen=True)
class OptimizerIteration:
    iteration: int
    status: str
    energy_hartree: float
    trial_energy_hartree: float
    gradient_inf_norm: float
    gradient_rms_norm: float
    step_norm: float
    step_inf_norm: float
    step_rms_norm: float
    energy_change_hartree: float
    convergence: dict[str, bool]
    trust_radius: float
    trust_ratio: float
    gradient_policy: str
    fd_mode: str
    fd_step_min: float
    fd_step_max: float
    refreshed_coordinate_count: int
    predicted_coordinate_count: int
    active_coordinate_fraction: float
    fd_one_sided_count: int
    fd_two_sided_count: int
    fd_parallel_workers: int
    local_group_count: int
    local_group_sizes: tuple[int, ...]
    surrogate_sample_count: int
    hessian_sparsity: float
    hessian_min_eigenvalue: float
    hessian_condition: float
    hessian_update_status: str
    step_policy: str
    gdiis_attempted: bool
    gdiis_used: bool
    gdiis_status: str
    gdiis_history_size: int
    gdiis_retained_history_size: int
    gdiis_discarded_history_size: int
    rejected_trial_count: int
    geometry_status: str
    coordinate_model_status: str
    selective_fallback_status: str
    symmetry_status: str
    qm_evaluations: int
    energy_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    fd_displacements: int
    cache_hits: int
    avoided_evaluations: int
    proposed_cartesian_rmsd_angstrom: float = 0.0
    trust_step_scale: float = 1.0
    trust_solver_iterations: int = 0
    applied_trust_radius_angstrom: float = 0.0
    class_threshold_fraction: float = 0.0
    class_screen_audit: bool = False
    class_screen_audit_interval: int = 0
    chart_epoch: int = 0
    chart_lifecycle_status: str = "DISABLED"
    predicted_reduction_hartree: float = 0.0
    actual_reduction_hartree: float = 0.0
    current_projected_gradient_norm: float | None = None
    trial_projected_gradient_norm: float | None = None
    trial_cartesian_rmsd_angstrom: float = 0.0
    transition_mode_index: int | None = None
    transition_mode_overlap: float | None = None
    transition_ascending_shift: float | None = None
    transition_descending_shift: float | None = None
    message: str = ""


@dataclass(frozen=True)
class StepProposal:
    step: np.ndarray
    policy: str
    hessian_min_eigenvalue: float
    hessian_condition: float
    damping_shift: float = 0.0
    transition_mode_index: int | None = None
    transition_mode_overlap: float | None = None
    transition_mode_vector: "TransitionModeReference | None" = None
    transition_ascending_shift: float | None = None
    transition_descending_shift: float | None = None
    cartesian_rmsd_angstrom: float = 0.0
    trust_scale: float = 1.0
    trust_iterations: int = 0
    applied_trust_radius_angstrom: float = 0.0
    prediction_hessian: np.ndarray | None = None


@dataclass(frozen=True)
class TransitionModeReference:
    """Reaction-mode tangent stored in a representation-independent frame."""

    cartesian_vector: np.ndarray
    coordinates_angstrom: np.ndarray
    cartesian_subspace: np.ndarray | None = None
    optimizer_vector: np.ndarray | None = None
    selection_policy: str = "ordinal_invariant"
    reaction_overlap: float = 0.0
    isotropic_overlap: float = 0.0


@dataclass(frozen=True)
class GDIISStepResult:
    """Auditable outcome of one mandatory controlled-GDIIS attempt."""

    step: np.ndarray | None
    status: str
    attempted: bool
    history_size: int
    retained_history_size: int
    discarded_history_size: int


@dataclass(frozen=True)
class CoordinateProjectorState:
    q: np.ndarray
    coordinates_angstrom: np.ndarray
    cartesian_from_q: np.ndarray
    age: int = 0
    analytic_refreshes: int = 0
    secant_updates: int = 0
    secant_rejections: int = 0
    last_secant_error: float = 0.0


@dataclass(frozen=True)
class OptimizerResult:
    converged: bool
    status: str
    settings: OptimizerSettings
    atoms: tuple[str, ...]
    initial_coordinates_angstrom: np.ndarray
    final_coordinates_angstrom: np.ndarray
    final_q: np.ndarray
    final_energy_hartree: float
    final_gradient: np.ndarray
    final_energy_change_hartree: float
    final_displacement: np.ndarray
    final_convergence: dict[str, bool]
    energy_noise_hartree: float
    energy_noise_samples: int
    energy_noise_energies_hartree: tuple[float, ...]
    iterations: tuple[OptimizerIteration, ...]
    qm_evaluations: int
    energy_evaluations: int
    final_hessian_energy_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    fd_displacements: int
    cache_hits: int
    avoided_evaluations: int
    cache_path: Path
    trajectory_path: Path
    trace_path: Path
    summary_path: Path
    final_hessian_path: Path | None
    final_hessian_index: int
    final_hessian_kind: str
    initial_hessian_source: str
    exact_final_hessian: bool = False
    final_cartesian_hessian_path: Path | None = None
    final_frequencies_cm: tuple[float, ...] = ()
    transition_mode_overlaps: tuple[float, ...] = ()
    irc_verification: dict[str, object] | None = None
    irc_path: Path | None = None
    final_gradient_verification: dict[str, object] | None = None
    runtime_method_manifest_path: Path | None = None
    chart_lifecycle_events: tuple[ChartLifecycleResult, ...] = ()
    optimization_active_labels: tuple[str, ...] = ()
    optimization_inactive_labels: tuple[str, ...] = ()
    frozen_chart_replay: dict[str, object] | None = None


class EvaluationCache:
    def __init__(
        self,
        *,
        metric_diagonal: Sequence[float],
        tolerance: float,
        path: Path | str | None = None,
        resume: bool = False,
    ) -> None:
        metric = np.asarray(metric_diagonal, dtype=float).reshape(-1)
        if metric.size == 0:
            raise ValueError("cache metric cannot be empty")
        self.metric_diagonal = np.maximum(np.abs(metric), 1.0e-12)
        self.tolerance = float(tolerance)
        self.path = Path(path) if path is not None else None
        self.records: list[OptimizerEvaluation] = []
        self.chart_epoch = 0
        self.hits = 0
        self.misses = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if resume and self.path.is_file():
                self._load(self.path)
            elif not resume:
                self.path.write_text("", encoding="utf-8")

    def lookup(
        self,
        q: np.ndarray,
        *,
        requested_properties: Sequence[str] = (),
    ) -> OptimizerEvaluation | None:
        vector = np.asarray(q, dtype=float).reshape(-1)
        for record in self.records:
            if record.chart_epoch != self.chart_epoch or np.asarray(record.q).size != vector.size:
                continue
            delta = vector - np.asarray(record.q, dtype=float).reshape(-1)
            distance2 = float(np.sum(self.metric_diagonal * delta * delta))
            if distance2 <= self.tolerance * self.tolerance and _evaluation_has_properties(
                record, requested_properties
            ):
                self.hits += 1
                return OptimizerEvaluation(
                    q=record.q,
                    coordinates_angstrom=record.coordinates_angstrom,
                    result=record.result,
                    cache_hit=True,
                    chart_epoch=record.chart_epoch,
                )
        self.misses += 1
        return None

    def add(self, evaluation: OptimizerEvaluation, *, persist: bool = True) -> None:
        if evaluation.chart_epoch != self.chart_epoch:
            raise ValueError("cache evaluation belongs to a different chart epoch")
        self.records.append(evaluation)
        if persist and self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_cache_record_to_json(evaluation), sort_keys=True) + "\n")

    def begin_chart_epoch(
        self,
        epoch: int,
        *,
        metric_diagonal: Sequence[float],
    ) -> None:
        next_epoch = int(epoch)
        if next_epoch != self.chart_epoch + 1:
            raise ValueError("chart cache epochs must increase by exactly one")
        metric = np.asarray(metric_diagonal, dtype=float).reshape(-1)
        if metric.size == 0 or not np.all(np.isfinite(metric)):
            raise ValueError("replacement cache metric must be finite and non-empty")
        self.chart_epoch = next_epoch
        self.metric_diagonal = np.maximum(np.abs(metric), 1.0e-12)

    def _load(self, path: Path) -> None:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if payload.get("schema") != OPTIMIZER_CACHE_SCHEMA:
                continue
            self.records.append(_cache_record_from_json(payload))


def _evaluation_has_properties(
    evaluation: OptimizerEvaluation,
    requested_properties: Sequence[str],
) -> bool:
    result = evaluation.result
    requested = result.execution.get("_requested_properties")
    if requested:
        requested_set = {str(name).strip().lower() for name in requested}
        if any(str(name).strip().lower() not in requested_set for name in requested_properties):
            return False
    available = {
        "energy": result.energy_hartree is not None,
        "gradient": result.gradient_hartree_per_bohr is not None,
        "hessian": result.hessian_hartree_per_bohr2 is not None,
    }
    return all(available.get(str(name).strip().lower(), False) for name in requested_properties)


class GeometryEvaluationService:
    def __init__(
        self,
        *,
        xyzin_path: Path | str,
        run_dir: Path | str,
        coordinate_model: OptimizerCoordinateModel,
        engine_command: str = "",
        backend: QMScanBackend | None = None,
        timeout: float | None = None,
        settings: OptimizerSettings | None = None,
        pes_exploration_policy: PESExplorationPolicy | None = None,
        periodic_pes_adapter=None,
    ) -> None:
        self.xyzin_path = Path(xyzin_path).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.geometry = read_xyzin_geometry(self.xyzin_path)
        self.atoms = tuple(self.geometry.atoms)
        self.reference_coordinates = np.asarray(self.geometry.coordinates_angstrom, dtype=float)
        self.coordinate_model = coordinate_model
        self.engine_command = engine_command
        self.backend = (
            _resolve_zaff_backend(
                backend,
                self.atoms,
                self.reference_coordinates,
            )
            if (backend is not None and _normalized_backend_name(backend.name) == "zaff")
            else backend
        )
        self.timeout = timeout
        self.settings = settings or OptimizerSettings()
        self.pes_exploration_policy = pes_exploration_policy
        self._periodic_pes_adapter = periodic_pes_adapter
        self._representation_realizing = False
        self._lock = Lock()
        self.cache = EvaluationCache(
            metric_diagonal=np.maximum(
                np.abs(coordinate_model.metric_diagonal),
                self.settings.min_abs_metric_diagonal,
            ),
            tolerance=self.settings.cache_tolerance,
            path=self.run_dir / "optimizer_cache.jsonl",
            resume=self.settings.resume,
        )
        self.qm_evaluations = len(self.cache.records)
        self.energy_evaluations = sum(
            int(
                item.result.execution.get(
                    "energy_evaluations", item.result.energy_hartree is not None
                )
            )
            for item in self.cache.records
        )
        self.final_hessian_energy_evaluations = sum(
            int(
                item.result.execution.get("energy_evaluations", 0)
                if str(item.result.execution.get("link_evaluation_tag", "")).startswith(
                    "final-hessian-"
                )
                else 0
            )
            for item in self.cache.records
        )
        self.gradient_evaluations = sum(
            int(
                item.result.execution.get(
                    "gradient_evaluations", item.result.gradient_hartree_per_bohr is not None
                )
            )
            for item in self.cache.records
        )
        self.hessian_evaluations = sum(
            int(
                item.result.execution.get(
                    "hessian_evaluations", item.result.hessian_hartree_per_bohr2 is not None
                )
            )
            for item in self.cache.records
        )
        self.fd_displacements = sum(
            int(item.result.execution.get("fd_displacements", 0)) for item in self.cache.records
        )
        self._projector_state: CoordinateProjectorState | None = None
        self._last_backtransform_diagnostics: dict[str, object] | None = None
        self._coordinate_realization_cache: OrderedDict[
            bytes, tuple[np.ndarray, dict[str, object] | None]
        ] = OrderedDict()
        self._coordinate_realization_cache_limit = 64
        self._coordinate_directions_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._optimizer_metric_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._coordinate_differential_cache_limit = 8
        self._state_reference = None
        self._cv_atomic_numbers: tuple[int, ...] = ()
        self._cv_bonded_pairs: tuple[tuple[int, int], ...] = ()
        self._fd_initial_max_gradient: float | None = None
        self._fd_class_threshold_fraction: float | None = None
        self._fd_class_retained_indices: set[int] = set()
        if self.settings.include_cv_exponential_field:
            from matrix_chem import read_primitive_contract
            from matrix_chem.topology.elements import atomic_number

            self._cv_atomic_numbers = tuple(int(atomic_number(atom) or 0) for atom in self.atoms)
            contract = read_primitive_contract(self.xyzin_path)
            self._cv_bonded_pairs = tuple(
                primitive.atoms for primitive in contract.primitives if primitive.kind == "bond"
            )
        self._install_coordinate_runtime(coordinate_model, self.reference_coordinates)

    def install_coordinate_model(
        self,
        coordinate_model: OptimizerCoordinateModel,
        reference_coordinates_angstrom: np.ndarray,
    ) -> None:
        """Install a validated chart at an already accepted Cartesian state."""

        if self.settings.include_cv_exponential_field:
            raise RuntimeError(
                "dynamic chart replacement is incompatible with a frozen CV bond field"
            )
        reference = np.asarray(reference_coordinates_angstrom, dtype=float)
        if reference.shape != self.reference_coordinates.shape or not np.all(np.isfinite(reference)):
            raise ValueError("replacement chart changed the Cartesian atom frame")
        self.reference_coordinates = reference.copy()
        self.coordinate_model = coordinate_model
        self._install_coordinate_runtime(coordinate_model, reference)
        self.cache.begin_chart_epoch(
            self.cache.chart_epoch + 1,
            metric_diagonal=np.maximum(
                np.abs(coordinate_model.metric_diagonal),
                self.settings.min_abs_metric_diagonal,
            ),
        )

    def _install_coordinate_runtime(
        self,
        coordinate_model: OptimizerCoordinateModel,
        reference_coordinates: np.ndarray,
    ) -> None:
        self._clear_coordinate_realization_cache()
        self._projector_state = None
        self._sonic_definition = None
        self._typed_onic_runtime = None
        self._typed_onic_reference_values = None
        self._sonic_coordinate_indices = ()
        self._sonic_reference_values = None
        self._sonic_full_reference_values = None
        self._sonic_phase_reference_values = None
        self._sonic_periodic_periods: dict[int, float] = {}
        self._sonic_periodic_primitive_ids: tuple[str, ...] = ()
        self._sonic_periodic_gic_components: dict[
            int, tuple[tuple[str, float], ...]
        ] = {}
        self._sonic_primitive_phase_reference_values: dict[str, float] = {}
        self._sonic_rotation_atlas = None
        self._rigid_pose_model = None
        self._rigid_pose_model_all = None
        self._acyclic_torsions = ()
        self._ring_puckering_blocks = ()
        self._sonic_active_families = ()
        self._assigned_cartesian_symmetry = None
        self._freeze_parent_cartesian_symmetry = False
        self._symmetry_thresholds = None
        self._frozen_cartesian_symmetry_projector = None
        if (
            self.pes_exploration_policy is not None
            and self.pes_exploration_policy.pointwise_oracle_symmetry
        ):
            from matrix_chem import read_symmetry_thresholds

            self._symmetry_thresholds = read_symmetry_thresholds(self.xyzin_path)
        if coordinate_model.kind == "typed_onic":
            from .typed_onic import TypedOnicRuntime

            runtime = coordinate_model.typed_onic_runtime
            if not isinstance(runtime, TypedOnicRuntime):
                raise TypeError("typed ONIC optimizer model has no compiled runtime")
            contract_reference = np.asarray(
                runtime.definition.reference_coordinates_angstrom, dtype=float
            )
            if not np.allclose(
                reference_coordinates,
                contract_reference,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "typed ONIC optimizer geometry differs from the frozen contract reference"
                )
            self._typed_onic_runtime = runtime
            self._typed_onic_reference_values = runtime.reference_values
        elif coordinate_model.kind == "sonic":
            from matrix_smith import (
                FragmentRotationAtlas,
                evaluate_gic_values,
                read_gic_definition_from_xyzin,
            )

            definition = (
                coordinate_model.sonic_definition
                if coordinate_model.sonic_definition is not None
                else read_gic_definition_from_xyzin(self.xyzin_path)
            )
            labels = tuple(gic.identifier for gic in definition.gics)
            names = tuple(gic.name for gic in definition.gics)
            self._sonic_definition = definition
            self._sonic_rotation_atlas = FragmentRotationAtlas(definition)
            sonic_labels = coordinate_model.sonic_labels or coordinate_model.labels
            self._sonic_coordinate_indices = tuple(
                _coordinate_index(label, labels, names) for label in sonic_labels
            )
            if definition.symmetrize:
                from matrix_smith.symmetry_labels import is_total_symmetric_irrep

                self._freeze_parent_cartesian_symmetry = all(
                    is_total_symmetric_irrep(definition.point_group, definition.gics[index].irrep)
                    for index in self._sonic_coordinate_indices
                )
            if definition.symmetrize and str(definition.point_group).strip().upper() not in {
                "",
                "C1",
                "UNKNOWN",
            }:
                from matrix_chem.symmetry import analyze_molecular_symmetry
                from matrix_chem import MolecularGeometry, read_symmetry_thresholds

                reference_geometry = MolecularGeometry(
                    atoms=self.atoms,
                    coordinates_angstrom=reference_coordinates,
                )
                symmetry_thresholds = read_symmetry_thresholds(self.xyzin_path)
                self._symmetry_thresholds = symmetry_thresholds
                assigned = analyze_molecular_symmetry(
                    reference_geometry,
                    distance_tolerance=symmetry_thresholds.distance_angstrom,
                    inertia_tolerance=symmetry_thresholds.inertia_relative,
                    max_rotation_order=symmetry_thresholds.max_rotation_order,
                )
                if assigned.point_group != definition.point_group:
                    raise RuntimeError(
                        "Cartesian geometry/symmetry contract mismatch: "
                        f"geometry={assigned.point_group}, SONIC={definition.point_group}"
                    )
                self._assigned_cartesian_symmetry = assigned
                self._frozen_cartesian_symmetry_projector = (
                    _totally_symmetric_cartesian_projector(
                        assigned,
                        natoms=len(self.atoms),
                    )
                )
                if coordinate_model.sonic_from_coordinates is not None:
                    self._freeze_parent_cartesian_symmetry = (
                        _cartesian_directions_are_totally_symmetric(
                            coordinate_model.directions_angstrom,
                            assigned,
                        )
                    )
            source_families = tuple(
                definition.gics[index].family for index in self._sonic_coordinate_indices
            )
            transform = coordinate_model.sonic_from_coordinates
            if transform is None:
                self._sonic_active_families = source_families
            else:
                matrix = np.asarray(transform, dtype=float)
                expected_shape = (len(source_families), len(coordinate_model.labels))
                if matrix.shape != expected_shape:
                    raise RuntimeError(
                        "projected SONIC variables do not match their source-coordinate basis"
                    )
                active_families: list[str] = []
                for column in matrix.T:
                    support = np.flatnonzero(np.abs(column) > 1.0e-8)
                    families = {str(source_families[index]) for index in support}
                    if len(families) != 1:
                        raise RuntimeError(
                            "a projected SONIC variable mixes incompatible coordinate families"
                        )
                    active_families.append(families.pop())
                self._sonic_active_families = tuple(active_families)
            # A rank-reduced active set is a complete local coordinate model,
            # not a request to constrain the discarded dependent GICs.  Keep
            # the retained rows as a compact definition so the nonlinear
            # backtransform solves only the realizable manifold.  This is
            # essential at symmetry-specialized geometries where a formally
            # valid GIC row can lose rank at the reference.
            if not self.settings.freeze_inactive_sonic or coordinate_model.rank_reduced_labels:
                selected_gics = tuple(
                    definition.gics[index] for index in self._sonic_coordinate_indices
                )
                definition = replace(
                    definition,
                    gics=selected_gics,
                    rank=len(selected_gics),
                    target_rank=len(selected_gics),
                )
                self._sonic_definition = definition
                self._sonic_rotation_atlas = FragmentRotationAtlas(definition)
                self._sonic_coordinate_indices = tuple(range(len(selected_gics)))
            gic_index = {
                str(gic.identifier): index for index, gic in enumerate(definition.gics)
            }
            self._sonic_periodic_periods = {
                gic_index[str(estimate.coordinate_identifier)]: 2.0 * np.pi
                for estimate in definition.periodic_coordinate_estimates
                if str(estimate.coordinate_identifier) in gic_index
                and str(estimate.coordinate_domain).strip().upper() == "PERIODIC_2PI"
            }
            primitive_by_id = {
                str(primitive.identifier): primitive for primitive in definition.primitives
            }
            self._sonic_periodic_primitive_ids = tuple(
                primitive_id
                for primitive_id, primitive in primitive_by_id.items()
                if str(primitive.function).upper() in {"D", "IMPD"}
            )
            periodic_primitive_ids = set(self._sonic_periodic_primitive_ids)
            self._sonic_periodic_gic_components = {
                index: tuple(
                    (str(primitive_id), float(coefficient))
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                    if str(primitive_id) in periodic_primitive_ids
                )
                for index, gic in enumerate(definition.gics)
            }
            self._sonic_periodic_gic_components = {
                index: components
                for index, components in self._sonic_periodic_gic_components.items()
                if components
            }
            from matrix_smith import evaluate_primitive_values
            from .periodic import gdv_principal_dihedral

            reference_primitive_values = evaluate_primitive_values(
                definition,
                primitive_ids=self._sonic_periodic_primitive_ids,
                coordinates_angstrom=reference_coordinates,
            )
            self._sonic_primitive_phase_reference_values = {
                primitive_id: gdv_principal_dihedral(value)
                for primitive_id, value in reference_primitive_values.items()
            }
            self._sonic_full_reference_values = evaluate_gic_values(
                definition, coordinates_angstrom=reference_coordinates
            )
            self._sonic_full_reference_values = self._phase_match_sonic_values(
                self._sonic_full_reference_values,
                primitive_values=reference_primitive_values,
                primitive_reference_values=self._sonic_primitive_phase_reference_values,
            )
            self._sonic_phase_reference_values = (
                self._sonic_full_reference_values.copy()
            )
            self._sonic_reference_values = self._sonic_full_reference_values[
                list(self._sonic_coordinate_indices)
            ]
            from .rigid_pose import RigidComplexModel
            from .hybrid_backtransform import (
                acyclic_torsion_specs,
                ring_puckering_block_specs,
            )

            rigid_model = RigidComplexModel.try_from_definition(definition)
            self._rigid_pose_model_all = rigid_model
            if rigid_model is not None and rigid_model.supports_coordinate_indices(
                self._sonic_coordinate_indices
            ):
                self._rigid_pose_model = rigid_model
            self._acyclic_torsions = acyclic_torsion_specs(
                definition,
                natoms=len(self.atoms),
                fixed_atom_indices=self.settings.fixed_atoms,
            )
            self._ring_puckering_blocks = ring_puckering_block_specs(definition)

    def initialize_coordinate_projector(
        self,
        q: Sequence[float] | np.ndarray,
        coordinates_angstrom: np.ndarray,
    ) -> None:
        if self.coordinate_model.kind not in {"sonic", "typed_onic"}:
            return
        vector = np.asarray(q, dtype=float).reshape(-1).copy()
        coords = np.asarray(coordinates_angstrom, dtype=float).copy()
        if vector.shape != (self.coordinate_model.directions_angstrom.shape[0],):
            raise ValueError("projector q length does not match coordinate model")
        if coords.shape != self.reference_coordinates.shape:
            raise ValueError("projector coordinates do not match reference geometry")
        self._update_sonic_phase_reference(coords)
        self._projector_state = CoordinateProjectorState(
            q=vector,
            coordinates_angstrom=coords,
            cartesian_from_q=self._internal_cartesian_from_q(coords),
            age=0,
            analytic_refreshes=(
                1 if self._projector_state is None else self._projector_state.analytic_refreshes + 1
            ),
            secant_updates=0
            if self._projector_state is None
            else self._projector_state.secant_updates,
            secant_rejections=(
                0 if self._projector_state is None else self._projector_state.secant_rejections
            ),
            last_secant_error=0.0
            if self._projector_state is None
            else self._projector_state.last_secant_error,
        )
        self._clear_coordinate_realization_cache()

    def update_coordinate_projector(
        self,
        *,
        previous_q: np.ndarray,
        previous_coordinates: np.ndarray,
        current_q: np.ndarray,
        current_coordinates: np.ndarray,
        trust_ratio: float,
        line_search_scale: float,
    ) -> None:
        if self.coordinate_model.kind not in {"sonic", "typed_onic"}:
            return
        if self._projector_state is None:
            self.initialize_coordinate_projector(current_q, current_coordinates)
            return
        state = self._projector_state
        from .internal_coordinates import secant_projector_update, should_refresh_coordinate_model

        self._update_sonic_phase_reference(current_coordinates)
        rotation_reset = bool(
            self.coordinate_model.kind == "sonic"
            and self._maybe_rebase_sonic_rotations(current_coordinates)
        )

        secant = secant_projector_update(
            state.cartesian_from_q,
            previous_q,
            previous_coordinates,
            current_q,
            current_coordinates,
        )
        secant_matrix = secant.cartesian_from_q
        secant_error = secant.relative_error
        secant_accepted = secant.accepted
        next_age = state.age + 1
        refresh = should_refresh_coordinate_model(
            model_age=next_age,
            line_search_scale=line_search_scale,
            trust_ratio=trust_ratio,
            secant_relative_error=secant_error,
        )
        if refresh or rotation_reset:
            self._projector_state = CoordinateProjectorState(
                q=np.asarray(current_q, dtype=float).reshape(-1).copy(),
                coordinates_angstrom=np.asarray(current_coordinates, dtype=float).copy(),
                cartesian_from_q=self._internal_cartesian_from_q(current_coordinates),
                age=0,
                analytic_refreshes=state.analytic_refreshes + 1,
                secant_updates=state.secant_updates,
                secant_rejections=state.secant_rejections + int(not secant_accepted),
                last_secant_error=secant_error,
            )
        elif secant_accepted and secant_matrix is not None:
            self._projector_state = CoordinateProjectorState(
                q=np.asarray(current_q, dtype=float).reshape(-1).copy(),
                coordinates_angstrom=np.asarray(current_coordinates, dtype=float).copy(),
                cartesian_from_q=secant_matrix,
                age=next_age,
                analytic_refreshes=state.analytic_refreshes,
                secant_updates=state.secant_updates + 1,
                secant_rejections=state.secant_rejections,
                last_secant_error=secant_error,
            )
        else:
            self._projector_state = CoordinateProjectorState(
                q=np.asarray(current_q, dtype=float).reshape(-1).copy(),
                coordinates_angstrom=np.asarray(current_coordinates, dtype=float).copy(),
                cartesian_from_q=self._internal_cartesian_from_q(current_coordinates),
                age=0,
                analytic_refreshes=state.analytic_refreshes + 1,
                secant_updates=state.secant_updates,
                secant_rejections=state.secant_rejections + 1,
                last_secant_error=secant_error,
            )
        self._clear_coordinate_realization_cache()

    def refresh_coordinate_projector(self, q: np.ndarray, coordinates_angstrom: np.ndarray) -> None:
        if (
            self.coordinate_model.kind in {"sonic", "typed_onic"}
            and self._projector_state is not None
        ):
            self.initialize_coordinate_projector(q, coordinates_angstrom)

    def _direct_q_realization_is_exact(self) -> bool:
        """Return whether the last finite realization needs no chart-drift guard.

        Direct acyclic-torsion and rigid-fragment realizations satisfy their
        requested internal coordinates analytically.  Their cumulative
        Cartesian displacement from the initial geometry is therefore not a
        measure of local chart validity.  Projected, exploratory, ring, and
        fallback realizations retain the ordinary drift safeguard.
        """

        diagnostic = self._last_backtransform_diagnostics
        method = "" if diagnostic is None else str(diagnostic.get("method", ""))
        return bool(
            diagnostic is not None
            and method.startswith("DIRECT_")
            and "RING" not in method
            and not bool(diagnostic.get("linear_fallback", False))
            and not self.settings.rigid_reference_groups
            and self._assigned_cartesian_symmetry is None
            and self.pes_exploration_policy is None
        )

    def coordinate_model_status(self, coordinates: np.ndarray, settings: OptimizerSettings) -> str:
        displacement = np.asarray(coordinates, dtype=float) - self.reference_coordinates
        max_displacement = float(np.max(np.abs(displacement))) if displacement.size else 0.0
        if (
            self.coordinate_model.kind not in {"sonic", "typed_onic"}
            or self._projector_state is None
        ):
            return f"ok:max_drift={max_displacement:.6g}"
        state = self._projector_state
        prefix = (
            f"projector_age={state.age}:refreshes={state.analytic_refreshes}:"
            f"secant={state.secant_updates}:secant_rejected={state.secant_rejections}:"
            f"secant_error={state.last_secant_error:.6g}:max_drift={max_displacement:.6g}"
        )
        if self.coordinate_model.rank_reduced_labels:
            prefix += ":rank_reduced=" + ",".join(self.coordinate_model.rank_reduced_labels)
        if self._last_backtransform_diagnostics is not None:
            diagnostic = self._last_backtransform_diagnostics
            prefix += (
                f":backtransform={diagnostic['method']}:substeps={diagnostic['substeps']}:"
                f"finite_frag={diagnostic['finite_fragment_count']}:"
                f"finite_tors={diagnostic['finite_torsion_count']}:"
                f"finite_ring={diagnostic['finite_ring_count']}:"
                f"finite_ring_phi={diagnostic['finite_ring_phase_count']}:"
                f"continuation={diagnostic['continuation_count']}:"
                f"linear_fallback={int(bool(diagnostic['linear_fallback']))}"
            )
        if (
            max_displacement > settings.coordinate_drift_warning
            and not self._direct_q_realization_is_exact()
        ):
            return f"ok:frozen_{self.coordinate_model.kind}_cumulative_drift_notice:" + prefix
        return "ok:" + prefix

    def coordinate_realization_status(
        self,
        coordinates: np.ndarray,
        settings: OptimizerSettings,
    ) -> str:
        """Audit whether a realized geometry remains a valid local chart.

        The nonlinear back-transform has already certified coordinate
        realization before returning.  This second, representation-neutral
        gate verifies that the current Jacobian is finite, full row rank, and
        acceptably conditioned.  Cumulative Cartesian motion from the initial
        geometry is telemetry, not a chart-validity criterion: finite fragment
        rotations can be large while the local differential chart remains
        perfectly regular.
        """

        coords = np.asarray(coordinates, dtype=float)
        if coords.shape != self.reference_coordinates.shape or not np.all(np.isfinite(coords)):
            return "invalid_coordinates"
        if self.coordinate_model.kind not in {"sonic", "typed_onic"}:
            return "valid_nonsonic"
        try:
            if self.coordinate_model.kind == "typed_onic":
                runtime = self._typed_onic_runtime
                if runtime is None:
                    return "missing_typed_onic_runtime"
                jacobian = np.asarray(runtime.evaluate(coords).b_matrix.to_dense(), dtype=float)
            else:
                _values, jacobian = self._evaluate_active_sonic(coords)
                transform = getattr(
                    self.coordinate_model,
                    "sonic_from_coordinates",
                    None,
                )
                if transform is not None:
                    jacobian = (
                        internal_from_cartesian_jacobian(transform, rcond=1.0e-10)
                        @ jacobian
                    )
            from matrix_numerics import singular_spectrum

            spectrum = singular_spectrum(
                jacobian,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
                normalize_rows=True,
                zero_row_tolerance=1.0e-12,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            return f"jacobian_audit_failed:{type(exc).__name__}"
        expected_rank = len(self.coordinate_model.labels)
        if spectrum.rank != expected_rank:
            return f"rank_deficient:{spectrum.rank}/{expected_rank}"
        if (
            not np.isfinite(spectrum.condition_number)
            or spectrum.condition_number > settings.max_hessian_condition
        ):
            return f"ill_conditioned:{spectrum.condition_number:.6g}"
        return f"valid:rank={spectrum.rank}/{expected_rank}:condition={spectrum.condition_number:.6g}"

    def coordinate_phase_masks(self, schedule: str) -> tuple[tuple[str, np.ndarray], ...]:
        size = len(self.coordinate_model.labels)
        all_mask = np.ones(size, dtype=bool)
        if (
            schedule not in {"inter-intra-joint", "inter-intra-micro"}
            or not self._sonic_active_families
        ):
            return (("joint", all_mask),)
        inter = np.asarray(
            [
                family.startswith("FRAG_")
                or family in {"H_BOND_DISTANCE", "FRAG_CENTER_ATOM_DISTANCE"}
                for family in self._sonic_active_families
            ],
            dtype=bool,
        )
        intra = ~inter
        if not np.any(inter) or not np.any(intra):
            return (("joint", all_mask),)
        return (("inter", inter), ("intra", intra), ("joint", all_mask))

    def coordinate_soft_mask(self, hessian: np.ndarray) -> np.ndarray:
        """Classify active variables with LINK's chemical hard/soft contract."""

        if self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            diagonal = np.abs(np.diag(np.asarray(hessian, dtype=float)))
            return diagonal <= np.median(diagonal)
        from .hybrid_backtransform import soft_coordinate_indices

        soft_rows = set(soft_coordinate_indices(self._sonic_definition))
        selected = np.asarray(
            [index in soft_rows for index in self._sonic_coordinate_indices], dtype=bool
        )
        transform = self.coordinate_model.sonic_from_coordinates
        if transform is None:
            return selected
        weights = np.abs(np.asarray(transform, dtype=float))
        soft_weight = np.sum(weights[selected, :], axis=0)
        hard_weight = np.sum(weights[~selected, :], axis=0)
        return soft_weight >= hard_weight

    def assert_totally_symmetric_active_sonics(self) -> None:
        """Validate the data needed for total-irrep finite-difference masking.

        A complete SONIC model is allowed here: non-totally-symmetric rows are
        excluded by the numerical-gradient driver and their derivatives are
        set to zero.  Rejecting such a model would force callers to construct
        a reduced model themselves and, more importantly, would not guarantee
        that an explicitly supplied full model follows the frozen symmetry
        contract.
        """

        if self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            raise ValueError("adaptive numerical gradients require a SONIC coordinate model")
        from matrix_smith.symmetry_labels import is_total_symmetric_irrep

        definition = self._sonic_definition
        if not str(definition.point_group).strip():
            raise ValueError("total-irrep numerical-gradient masking needs a point group")
        if not any(
            is_total_symmetric_irrep(definition.point_group, definition.gics[index].irrep)
            for index in self._sonic_coordinate_indices
        ):
            raise ValueError("SONIC model contains no totally symmetric coordinate")

    def finite_difference_class_screen(
        self,
        previous_gradient: np.ndarray | None,
        active_mask: np.ndarray,
        settings: OptimizerSettings,
        *,
        enabled: bool,
        iteration: int = 0,
    ) -> tuple[np.ndarray, float, bool]:
        """Return the frozen schedule-based SONIC class screen.

        The first numerical gradient is complete.  Later one-sided gradients
        use 10%, 5%, and 1% same-family thresholds at the prescribed
        iteration stages.  A three-iteration audit is complete, and audited
        active coordinates are retained for the remainder of the run.
        """

        mask = np.asarray(active_mask, dtype=bool).copy()
        if (
            not enabled
            or previous_gradient is None
            or self.coordinate_model.kind != "sonic"
            or self._sonic_definition is None
        ):
            return mask, 0.0 if self._fd_class_threshold_fraction is None else float(
                self._fd_class_threshold_fraction
            ), False
        # Frozen protocol schedule: the first gradient is complete; after it,
        # use 10%, then 5% after the first three-iteration audit, and 1% after
        # the next three-iteration audit. The audit itself is always full.
        target = 0.10 if iteration < 3 else (0.05 if iteration < 6 else 0.01)
        if self._fd_class_threshold_fraction is None:
            self._fd_class_threshold_fraction = float(target)
        else:
            self._fd_class_threshold_fraction = min(
                float(self._fd_class_threshold_fraction), float(target)
            )
        fraction = float(self._fd_class_threshold_fraction)
        if iteration > 0 and iteration % settings.fd_class_screen_audit_interval == 0:
            return mask, fraction, True
        previous = np.asarray(previous_gradient, dtype=float).reshape(-1)
        if fraction <= 0.0:
            return mask, fraction, False
        families = tuple(
            self._sonic_definition.gics[index].family
            for index in self._sonic_coordinate_indices
        )
        for family in sorted(set(families)):
            indices = [
                index
                for index, item in enumerate(families)
                if item == family and bool(mask[index])
            ]
            if not indices:
                continue
            class_max = max(float(abs(previous[index])) for index in indices)
            cutoff = fraction * class_max
            retained = [index for index in indices if abs(previous[index]) >= cutoff]
            if not retained:
                retained = [max(indices, key=lambda index: abs(previous[index]))]
            mask[indices] = False
            mask[retained] = True
        mask[list(self._fd_class_retained_indices)] = True
        return mask, fraction, False

    def retain_audited_finite_difference_coordinates(
        self,
        gradient: np.ndarray,
        active_mask: np.ndarray,
        threshold_fraction: float,
    ) -> None:
        """Persist coordinates that an audit shows to be materially active."""

        if (
            self.coordinate_model.kind != "sonic"
            or self._sonic_definition is None
            or threshold_fraction <= 0.0
        ):
            return
        values = np.asarray(gradient, dtype=float).reshape(-1)
        allowed = np.asarray(active_mask, dtype=bool).reshape(-1)
        families = tuple(
            self._sonic_definition.gics[index].family
            for index in self._sonic_coordinate_indices
        )
        for family in sorted(set(families)):
            indices = [
                index
                for index, item in enumerate(families)
                if item == family and bool(allowed[index])
            ]
            if not indices:
                continue
            class_max = max(float(abs(values[index])) for index in indices)
            cutoff = threshold_fraction * class_max
            self._fd_class_retained_indices.update(
                index for index in indices if abs(values[index]) >= cutoff
            )

    def assert_frozen_symmetry_contract(self) -> None:
        """Fail closed unless the canonical ORACLE--SMITH symmetry contract is present."""

        if self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            return
        definition = self._sonic_definition
        point_group = str(definition.point_group or "C1").strip().upper()
        if point_group in {"", "C1", "UNKNOWN"}:
            return
        if not bool(definition.symmetrize):
            raise RuntimeError(
                "LINK frozen symmetry contract requires an ORACLE-symmetrized SMITH definition"
            )
        if self._assigned_cartesian_symmetry is None:
            raise RuntimeError(
                "LINK frozen symmetry contract could not assign the ORACLE point-group symmetry"
            )
        if not bool(self.settings.symmetrize_analytic_gradients):
            raise RuntimeError(
                "LINK frozen symmetry contract requires gradient symmetrization at every iteration"
            )
        if self.settings.stationary_point == "minimum":
            self.assert_totally_symmetric_active_sonics()
            if (
                not self.settings.fd_totally_symmetric_only
                and not self._freeze_parent_cartesian_symmetry
            ):
                raise RuntimeError(
                    "LINK minimum optimization requires the frozen initial point group"
                )

    def _clear_coordinate_realization_cache(self) -> None:
        """Discard Cartesian and differential data after a chart-state mutation."""

        self._coordinate_realization_cache.clear()
        self._coordinate_directions_cache.clear()
        self._optimizer_metric_cache.clear()

    def _coordinate_cache_key(self, coordinates_angstrom: np.ndarray | None) -> bytes:
        coordinates = np.ascontiguousarray(
            self.reference_coordinates
            if coordinates_angstrom is None
            else np.asarray(coordinates_angstrom, dtype=float)
        )
        if coordinates.shape != self.reference_coordinates.shape:
            raise ValueError("coordinates do not match the optimizer atom frame")
        return coordinates.tobytes()

    @staticmethod
    def _retain_recent_coordinate_value(
        cache: OrderedDict[bytes, np.ndarray],
        key: bytes,
        value: np.ndarray,
        *,
        limit: int,
    ) -> None:
        cache[key] = np.asarray(value, dtype=float).copy()
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def coordinates_from_q(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Realize ``q`` once per immutable chart state.

        Trust enforcement, geometry validation, and QM dispatch consume the
        same Cartesian point.  Keeping a small exact-key cache prevents those
        independent protocol gates from repeating an identical nonlinear
        back-transform.  Projector refreshes and rotation-chart rebases clear
        the cache before a realization can be reused in a different chart.
        """

        return self._coordinates_from_q_cached(q, exhaustive=True)

    def _coordinates_from_q_for_trust(
        self, q: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Probe one trust candidate without exhaustive failure recovery."""

        return self._coordinates_from_q_cached(q, exhaustive=False)

    def _coordinates_from_q_cached(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        exhaustive: bool,
    ) -> np.ndarray:
        vector = np.ascontiguousarray(np.asarray(q, dtype=float).reshape(-1))
        if vector.shape != (self.coordinate_model.directions_angstrom.shape[0],):
            raise ValueError("q length does not match coordinate model")
        key = vector.tobytes()
        cached = self._coordinate_realization_cache.get(key)
        if cached is not None:
            coordinates, diagnostics = cached
            self._coordinate_realization_cache.move_to_end(key)
            self._last_backtransform_diagnostics = (
                None if diagnostics is None else dict(diagnostics)
            )
            return coordinates.copy()

        coordinates = np.asarray(
            self._realize_coordinates_from_q(vector, exhaustive=exhaustive),
            dtype=float,
        )
        diagnostics = (
            None
            if self._last_backtransform_diagnostics is None
            else dict(self._last_backtransform_diagnostics)
        )
        self._coordinate_realization_cache[key] = (coordinates.copy(), diagnostics)
        self._coordinate_realization_cache.move_to_end(key)
        while len(self._coordinate_realization_cache) > self._coordinate_realization_cache_limit:
            self._coordinate_realization_cache.popitem(last=False)
        return coordinates.copy()

    def _realize_coordinates_from_q(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        exhaustive: bool = True,
    ) -> np.ndarray:
        vector = np.asarray(q, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_model.directions_angstrom.shape[0],):
            raise ValueError("q length does not match coordinate model")
        if self._periodic_pes_adapter is not None and not self._representation_realizing:
            embedding = self._periodic_pes_adapter.embed(vector)
            decoded = self._periodic_pes_adapter.decode(embedding, reference=vector)
            self._representation_realizing = True
            try:
                return self._coordinates_from_q_cached(decoded, exhaustive=exhaustive)
            finally:
                self._representation_realizing = False
        if self.coordinate_model.kind == "typed_onic":
            runtime = self._typed_onic_runtime
            reference_values = self._typed_onic_reference_values
            if runtime is None or reference_values is None:
                raise RuntimeError("typed ONIC optimizer runtime is unavailable")
            start = (
                self.reference_coordinates
                if self._projector_state is None
                else self._projector_state.coordinates_angstrom
            )
            initial_projector = (
                None
                if self._projector_state is None
                else self._projector_state.cartesian_from_q
            )
            result = runtime.realize(
                reference_values + vector,
                start_coordinates_angstrom=start,
                initial_cartesian_from_q=initial_projector,
                fixed_atom_indices=self.settings.fixed_atoms,
                project_coordinates=self._project_geometry_constraints,
                max_continuation_increment=self.settings.backtransform_continuation_step,
                max_substeps=self.settings.backtransform_max_substeps,
            )
            self._last_backtransform_diagnostics = {
                "method": result.method,
                "substeps": result.substeps,
                "finite_fragment_count": sum(
                    item.representation == "EXPONENTIAL_MAP" for item in result.block_diagnostics
                ),
                "finite_torsion_count": 0,
                "finite_ring_count": 0,
                "finite_ring_phase_count": 0,
                "continuation_count": sum(
                    item.representation == "INVERSE_DISTANCE_PROJECTOR"
                    for item in result.block_diagnostics
                ),
                "corrector_iterations": result.iterations,
                "linear_fallback": False,
            }
            if not result.converged:
                raise RuntimeError(
                    "typed ONIC back-transformation did not converge: "
                    f"residual={np.linalg.norm(result.residual):.6g}"
                )
            return self._prepare_pes_exploration_geometry(
                self._project_geometry_constraints(result.coordinates_angstrom)
            )
        if self.coordinate_model.kind == "sonic" and self._projector_state is not None:
            from .hybrid_backtransform import (
                direct_rigid_soft_step,
                hybrid_internal_coordinate_step,
            )
            from .geodesic_backtransform import geodesic_internal_coordinate_step
            from .internal_coordinates import nonlinear_internal_coordinate_step

            state = self._projector_state
            if self.settings.freeze_inactive_sonic:
                target_values = self._absolute_sonic_values(vector)
                evaluate = self._evaluate_all_sonic
                evaluate_values = self._evaluate_all_sonic_values
            else:
                if self._sonic_reference_values is None:
                    raise RuntimeError("SONIC reference values are unavailable")
                target_values = self._sonic_reference_values + self._sonic_displacements(vector)
                evaluate = self._evaluate_active_sonic
                evaluate_values = self._evaluate_active_sonic_values
            if self.settings.stationary_point == "transition_state":
                from .internal_coordinates import gdv_redq2x_internal_coordinate_step

                coordinate_scales = self.gdv_backtransform_coordinate_scales(
                    target_values.size
                )
                start_values, _start_b = evaluate(state.coordinates_angstrom)
                requested_delta = target_values - start_values
                failed_iterations: list[int] = []
                result = None
                retry_factor = GDV_REDQ2X_HARD_FAILURE_RETRY_FACTORS[0]
                # gdv.j32+/l103.F:GrdOpt initializes ScFact=2 and NErrs=0.
                # The first RedCar attempt therefore uses 1.0.  A RedQ2X
                # nonconvergence sets IFailR and increments NErrs for a GIC
                # optimization; every following attempt then executes
                # ScFact=0.5*ScFact.  Each attempt restarts from the same C.
                # This is deliberately not the 1.0, 0.9, ... branch used
                # only while no hard RedCar error has occurred.
                for retry_factor in GDV_REDQ2X_HARD_FAILURE_RETRY_FACTORS:
                    result = gdv_redq2x_internal_coordinate_step(
                        state.coordinates_angstrom,
                        start_values + retry_factor * requested_delta,
                        evaluate,
                        coordinate_unit_scales=coordinate_scales,
                        fixed_atom_indices=self.settings.fixed_atoms,
                    )
                    if result.converged:
                        break
                    failed_iterations.append(result.iterations)
                assert result is not None
                self._last_backtransform_diagnostics = {
                    "method": "GDV_REDQ2X",
                    "substeps": result.iterations,
                    "finite_fragment_count": 0,
                    "finite_torsion_count": 0,
                    "finite_ring_count": 0,
                    "finite_ring_phase_count": 0,
                    "continuation_count": 0,
                    "corrector_iterations": result.iterations,
                    "linear_fallback": False,
                    "residual_norm": float(np.linalg.norm(result.residual)),
                    "retry_factor": retry_factor,
                    "failed_attempt_iterations": tuple(failed_iterations),
                }
                if not result.converged:
                    raise RuntimeError(
                        "GDV RedQ2X back-transformation did not converge: "
                        f"residual={np.linalg.norm(result.residual):.6g}"
                    )
                coordinates = self._project_geometry_constraints(
                    result.coordinates_angstrom
                )
                return self._prepare_pes_exploration_geometry(coordinates)
            if self.settings.backtransform_method == "geodesic":
                geodesic = geodesic_internal_coordinate_step(
                    state.coordinates_angstrom,
                    target_values,
                    evaluate,
                    fixed_atom_indices=self.settings.fixed_atoms,
                    max_steps=self.settings.backtransform_max_substeps,
                    jacobian_displacement=self.settings.geodesic_jacobian_displacement,
                )
                self._last_backtransform_diagnostics = {
                    "method": geodesic.method,
                    "substeps": geodesic.steps,
                    "finite_fragment_count": 0,
                    "finite_torsion_count": 0,
                    "finite_ring_count": 0,
                    "finite_ring_phase_count": 0,
                    "continuation_count": 0,
                    "corrector_iterations": 0,
                    "linear_fallback": not geodesic.converged,
                }
                if not geodesic.converged:
                    geodesic_corrected = nonlinear_internal_coordinate_step(
                        geodesic.coordinates_angstrom,
                        target_values,
                        evaluate,
                        evaluate_values=evaluate_values,
                        fixed_atom_indices=self.settings.fixed_atoms,
                        project_coordinates=self._project_rigid_reference_groups,
                    )
                    if not geodesic_corrected.converged:
                        raise RuntimeError(
                            "SONIC geodesic back-transformation did not converge: "
                            f"residual={np.linalg.norm(geodesic_corrected.residual):.6g}"
                        )
                    coordinates = geodesic_corrected.coordinates_angstrom
                else:
                    coordinates = geodesic.coordinates_angstrom
                coordinates = self._project_geometry_constraints(coordinates)
                return self._prepare_pes_exploration_geometry(coordinates)
            rigid_model = self._rigid_pose_model
            if (
                rigid_model is not None
                and not self.settings.fixed_atoms
                and not self.coordinate_model.rank_reduced_labels
                and target_values.shape == (rigid_model.coordinate_count,)
            ):
                coordinates = rigid_model.realize_sonic(self._rigid_model_values(target_values))
                self._last_backtransform_diagnostics = {
                    "method": "DIRECT_RIGID_FRAGMENT_POSE",
                    "substeps": 0,
                    "finite_fragment_count": len(rigid_model.blocks),
                    "finite_torsion_count": 0,
                    "finite_ring_count": 0,
                    "finite_ring_phase_count": 0,
                    "continuation_count": 0,
                    "corrector_iterations": 0,
                    "linear_fallback": False,
                }
                coordinates = self._project_geometry_constraints(coordinates)
                return self._prepare_pes_exploration_geometry(coordinates)
            direct_soft = direct_rigid_soft_step(
                self._sonic_definition,
                state.coordinates_angstrom,
                target_values,
                evaluate_values,
                evaluate_subset=self._evaluate_sonic_subset,
                evaluate_values_subset=self._evaluate_sonic_values_subset,
                rigid_model=self._rigid_pose_model_all,
                rigid_target_transform=self._rigid_model_values,
                torsions=self._acyclic_torsions,
                ring_blocks=self._ring_puckering_blocks,
                current_values=(
                    self._absolute_sonic_values(state.q)
                    if self.settings.freeze_inactive_sonic
                    else self._sonic_reference_values + self._sonic_displacements(state.q)
                ),
                fixed_atom_indices=self.settings.fixed_atoms,
            )
            if direct_soft is not None:
                coordinates = direct_soft.coordinates_angstrom
                projected = bool(
                    self.settings.rigid_reference_groups
                    or self._assigned_cartesian_symmetry is not None
                )
                if projected:
                    coordinates = self._project_geometry_constraints(coordinates)
                if self.pes_exploration_policy is not None:
                    coordinates = self._prepare_pes_exploration_geometry(coordinates)
                    projected = True
                if (
                    not projected
                    or np.max(
                        np.abs(evaluate_values(coordinates) - target_values),
                        initial=0.0,
                    )
                    <= 1.0e-9
                ):
                    self._last_backtransform_diagnostics = {
                        "method": direct_soft.method,
                        "substeps": 0,
                        "finite_fragment_count": len(direct_soft.fragment_indices),
                        "finite_torsion_count": len(direct_soft.torsion_indices),
                        "finite_ring_count": len(direct_soft.ring_indices),
                        "finite_ring_phase_count": 0,
                        "continuation_count": 0,
                        "corrector_iterations": 0,
                        "linear_fallback": False,
                    }
                    return coordinates
            result = hybrid_internal_coordinate_step(
                self._sonic_definition,
                state.coordinates_angstrom,
                target_values,
                evaluate,
                evaluate_subset=self._evaluate_sonic_subset,
                evaluate_values=evaluate_values,
                evaluate_values_subset=self._evaluate_sonic_values_subset,
                rigid_model=self._rigid_pose_model_all,
                rigid_target_transform=self._rigid_model_values,
                fixed_atom_indices=self.settings.fixed_atoms,
                project_coordinates=self._project_rigid_reference_groups,
                max_continuation_increment=self.settings.backtransform_continuation_step,
                max_substeps=self.settings.backtransform_max_substeps,
            )
            hybrid_converged = result.converged
            self._last_backtransform_diagnostics = {
                "method": result.method,
                "substeps": result.substeps,
                "finite_fragment_count": len(result.finite_fragment_indices),
                "finite_torsion_count": len(result.finite_torsion_indices),
                "finite_ring_count": len(result.finite_ring_indices),
                "finite_ring_phase_count": len(result.finite_ring_phase_indices),
                "continuation_count": len(result.continuation_indices),
                "corrector_iterations": result.corrector_iterations,
                "linear_fallback": not hybrid_converged,
            }
            if not result.converged and exhaustive:
                # A valid Wilson tangent can still lead to a failed nonlinear
                # corrector when a mixed target crosses a strongly curved
                # region. Retry generically with shorter continuation steps
                # before falling back to the iterative Wilson corrector.
                for factor in (0.5, 0.25, 0.125):
                    retry = hybrid_internal_coordinate_step(
                        self._sonic_definition,
                        state.coordinates_angstrom,
                        target_values,
                        evaluate,
                        evaluate_subset=self._evaluate_sonic_subset,
                        evaluate_values=evaluate_values,
                        evaluate_values_subset=self._evaluate_sonic_values_subset,
                        rigid_model=self._rigid_pose_model_all,
                        rigid_target_transform=self._rigid_model_values,
                        fixed_atom_indices=self.settings.fixed_atoms,
                        project_coordinates=self._project_rigid_reference_groups,
                        max_continuation_increment=(
                            self.settings.backtransform_continuation_step * factor
                        ),
                        max_substeps=self.settings.backtransform_max_substeps * 2,
                    )
                    if retry.converged:
                        result = retry
                        self._last_backtransform_diagnostics = {
                            "method": retry.method,
                            "substeps": retry.substeps,
                            "finite_fragment_count": len(retry.finite_fragment_indices),
                            "finite_torsion_count": len(retry.finite_torsion_indices),
                            "finite_ring_count": len(retry.finite_ring_indices),
                            "finite_ring_phase_count": len(retry.finite_ring_phase_indices),
                            "continuation_count": len(retry.continuation_indices),
                            "corrector_iterations": retry.corrector_iterations,
                            "linear_fallback": False,
                            "adaptive_retry_factor": factor,
                        }
                        break
            if not result.converged and exhaustive:
                # Compatibility fallback for unusual mixed coordinates not yet
                # classified by the finite predictor.  It starts from the
                # hybrid result, so the old pseudoinverse is never the primary
                # mechanism for a large soft displacement.
                result = nonlinear_internal_coordinate_step(
                    result.coordinates_angstrom,
                    target_values,
                    evaluate,
                    evaluate_values=evaluate_values,
                    fixed_atom_indices=self.settings.fixed_atoms,
                    project_coordinates=self._project_rigid_reference_groups,
                )
            if not result.converged:
                raise RuntimeError(
                    "SONIC back-transformation did not converge: "
                    f"residual={np.linalg.norm(result.residual):.6g}"
                )
            coordinates = self._project_geometry_constraints(result.coordinates_angstrom)
            return self._prepare_pes_exploration_geometry(coordinates)
        flat = (
            self.reference_coordinates.reshape(-1)
            + self.coordinate_model.directions_angstrom.T @ vector
        )
        coordinates = self._project_geometry_constraints(
            flat.reshape(self.reference_coordinates.shape)
        )
        return self._prepare_pes_exploration_geometry(coordinates)

    def coordinates_from_representation(
        self,
        values: Sequence[float] | np.ndarray,
        request,
        *,
        periodic_contracts=(),
        reference_values: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Realize a requested representation through the canonical LINK path.

        Periodic embeddings are decoded to scalar SONIC values and then handed
        to the existing ``coordinates_from_q`` implementation.  This keeps the
        embedding useful for global PES exploration without introducing a second
        Cartesian back-transform algorithm.
        """

        from .representation_adapter import (
            decode_periodic_embedding,
            validate_link_representation,
        )

        validated = validate_link_representation(request)
        vector = np.asarray(values, dtype=float).reshape(-1)
        if validated.mode == "SCALAR":
            return self.coordinates_from_q(vector)
        if validated.mode == "PERIODIC_EMBEDDING":
            realization = decode_periodic_embedding(
                vector,
                tuple(periodic_contracts),
                reference=None
                if reference_values is None
                else np.asarray(reference_values, dtype=float),
            )
            if realization.scalar_values.shape != (
                self.coordinate_model.directions_angstrom.shape[0],
            ):
                raise ValueError(
                    "periodic embedding contracts must cover the LINK coordinate model"
                )
            return self.coordinates_from_q(realization.scalar_values)
        if validated.mode == "CARTESIAN":
            if vector.shape != self.reference_coordinates.reshape(-1).shape:
                raise ValueError("Cartesian representation length does not match geometry")
            return vector.reshape(self.reference_coordinates.shape).copy()
        raise ValueError(
            "QUATERNION_POSE realization requires the existing rigid-pose service; "
            "it is not a generic internal-coordinate representation"
        )

    def coordinates_from_global_pes_embedding(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        periodic_contracts=(),
        reference_values: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Realize a global-PES point using LINK's canonical PES request."""

        from .pes_representation import global_pes_representation

        return self.coordinates_from_representation(
            values,
            global_pes_representation(),
            periodic_contracts=periodic_contracts,
            reference_values=reference_values,
        )

    def coordinates_from_q_batch(
        self,
        q_values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Realize independent candidates, vectorizing rigid fragment poses."""

        values = np.asarray(q_values, dtype=float)
        expected = self.coordinate_model.directions_angstrom.shape[0]
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError("q batch has invalid shape")
        rigid_model = self._rigid_pose_model
        if (
            self.coordinate_model.kind == "sonic"
            and self._projector_state is not None
            and rigid_model is not None
            and not self.settings.fixed_atoms
        ):
            transform = self.coordinate_model.sonic_from_coordinates
            displacements = (
                values if transform is None else values @ np.asarray(transform, dtype=float).T
            )
            if self.settings.freeze_inactive_sonic:
                if (
                    self._sonic_full_reference_values is None
                    or self._sonic_reference_values is None
                ):
                    raise RuntimeError("SONIC reference values are unavailable")
                targets = np.broadcast_to(
                    self._sonic_full_reference_values,
                    (len(values), len(self._sonic_full_reference_values)),
                ).copy()
                targets[:, list(self._sonic_coordinate_indices)] = (
                    self._sonic_reference_values[None, :] + displacements
                )
            else:
                if self._sonic_reference_values is None:
                    raise RuntimeError("SONIC reference values are unavailable")
                targets = self._sonic_reference_values[None, :] + displacements
            coordinates = rigid_model.realize_sonic_batch(
                targets,
                workers=self.settings.coordinate_parallel_workers,
            )
            if (
                self.settings.rigid_reference_groups
                or self._assigned_cartesian_symmetry is not None
                or self.pes_exploration_policy is not None
            ):
                coordinates = np.asarray(
                    [
                        self._prepare_pes_exploration_geometry(
                            self._project_geometry_constraints(candidate)
                        )
                        for candidate in coordinates
                    ]
                )
            return coordinates
        return np.asarray([self.coordinates_from_q(row) for row in values])

    def _prepare_pes_exploration_geometry(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        if self.pes_exploration_policy is None:
            return np.asarray(coordinates_angstrom, dtype=float)
        coordinates, _symmetry = prepare_pes_exploration_geometry(
            self.xyzin_path,
            self.atoms,
            coordinates_angstrom,
            policy=self.pes_exploration_policy,
        )
        return coordinates

    def _project_geometry_constraints(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        coords = self._project_rigid_reference_groups(coordinates_angstrom)
        if (
            self._assigned_cartesian_symmetry is not None
            and self._freeze_parent_cartesian_symmetry
            and self.settings.freeze_inactive_sonic
            and getattr(self, "pes_exploration_policy", None) is None
        ):
            from matrix_chem import atomic_mass
            from matrix_chem.topology.elements import atomic_number

            weights = np.asarray(
                [atomic_mass(int(atomic_number(atom) or 0)) for atom in self.atoms],
                dtype=float,
            )
            center = np.average(coords, axis=0, weights=weights)
            centered = np.asarray(coords, dtype=float) - center
            projector = self._frozen_cartesian_symmetry_projector
            if projector is None:
                projector = _totally_symmetric_cartesian_projector(
                    self._assigned_cartesian_symmetry,
                    natoms=len(self.atoms),
                )
            coords = (projector @ centered.reshape(-1)).reshape((-1, 3)) + center
            coords[np.abs(coords) < 5.0e-13] = 0.0
        return coords

    def coordinate_directions(self, coordinates_angstrom: np.ndarray | None = None) -> np.ndarray:
        key = self._coordinate_cache_key(coordinates_angstrom)
        cached = self._coordinate_directions_cache.get(key)
        if cached is not None:
            self._coordinate_directions_cache.move_to_end(key)
            return cached.copy()
        if self.coordinate_model.kind == "typed_onic":
            directions = self._typed_onic_cartesian_from_q(coordinates_angstrom).T
        elif self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            directions = np.asarray(self.coordinate_model.directions_angstrom, dtype=float)
        else:
            coords = (
                self.reference_coordinates
                if coordinates_angstrom is None
                else np.asarray(coordinates_angstrom, dtype=float)
            )
            directions = self._sonic_cartesian_from_q(coords).T
        directions = np.asarray(directions, dtype=float)
        self._retain_recent_coordinate_value(
            self._coordinate_directions_cache,
            key,
            directions,
            limit=self._coordinate_differential_cache_limit,
        )
        return directions.copy()

    def _internal_cartesian_from_q(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        if self.coordinate_model.kind == "typed_onic":
            return self._typed_onic_cartesian_from_q(coordinates_angstrom)
        return self._sonic_cartesian_from_q(coordinates_angstrom)

    def _typed_onic_cartesian_from_q(
        self,
        coordinates_angstrom: np.ndarray | None = None,
    ) -> np.ndarray:
        runtime = self._typed_onic_runtime
        if runtime is None:
            raise RuntimeError("typed ONIC optimizer runtime is unavailable")
        coordinates = (
            self.reference_coordinates
            if coordinates_angstrom is None
            else np.asarray(coordinates_angstrom, dtype=float)
        )
        b_matrix = runtime.evaluate(coordinates).b_matrix.to_dense()
        fixed_columns = np.asarray(
            [3 * atom + component for atom in self.settings.fixed_atoms for component in range(3)],
            dtype=int,
        )
        return cartesian_from_internal_jacobian(
            b_matrix,
            rcond=1.0e-8,
            fixed_cartesian_columns=fixed_columns,
        )

    def optimizer_metric_diagonal(
        self,
        coordinates_angstrom: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the diagonal of the complete physical optimizer metric."""

        return np.diag(self.optimizer_metric(coordinates_angstrom)).copy()

    def gdv_internal_step_weights(self) -> np.ndarray:
        """Return the literal ``OptDX`` weights for active GDV variables."""

        functions = self._gdv_internal_coordinate_functions()
        # utilnz.F:OptDX uses DotSB + 0.01*DotD.
        return np.asarray(
            [
                0.1 if function in GDV_DIHEDRAL_COORDINATE_FUNCTIONS else 1.0
                for function in functions
            ],
            dtype=float,
        )

    def gdv_uses_force_constant_step_weights(self) -> bool:
        """Return whether GDV selects ``CrdGRo(IOp=5)`` for this GIC set.

        In the source used to build the reference executable, ``IniGic`` sets
        ``ITstSp=10`` whenever at least one active coordinate is a genuinely
        generic expression. ``GrdOpt`` then decodes ``IDoGRo=1`` and replaces
        the fixed ``OptDX`` weights with the force-constant-dependent weights
        from ``CrdGRo``. A named +1 alias of one native primitive is simplified
        by the Gaussian parser and is not generic (as in Baker TS1).
        """

        definition = self._sonic_definition
        if self.coordinate_model.kind != "sonic" or definition is None:
            return False
        primitive_by_id = {
            primitive.identifier: primitive for primitive in definition.primitives
        }
        for index in self._sonic_coordinate_indices:
            gic = definition.gics[index]
            coefficients = tuple(gic.coefficients)
            if len(coefficients) != 1 or coefficients[0][1] != 1.0:
                return True
            primitive = primitive_by_id.get(coefficients[0][0])
            if primitive is None or not primitive.is_gaussian_native:
                return True
        return False

    def gdv_internal_coordinate_scales(self) -> np.ndarray:
        """Map LINK's native SONIC variables to GDV bohr/radian variables."""

        if self.coordinate_model.kind != "sonic":
            return np.ones(len(self.coordinate_model.labels), dtype=float)
        functions = self._gdv_internal_coordinate_functions()
        scales = []
        for function in functions:
            if function in GDV_LENGTH_COORDINATE_FUNCTIONS:
                scales.append(ANGSTROM_TO_BOHR)
            elif function in GDV_ANGULAR_COORDINATE_FUNCTIONS:
                scales.append(1.0)
            else:
                raise RuntimeError(
                    f"no GDV unit contract is defined for SONIC function {function!r}"
                )
        return np.asarray(scales, dtype=float)

    def gdv_backtransform_coordinate_scales(self, size: int) -> np.ndarray:
        """Return GDV units for the active or complete SONIC target vector."""

        active = self.gdv_internal_coordinate_scales()
        if active.size == int(size):
            return active
        definition = self._sonic_definition
        if definition is None or len(definition.gics) != int(size):
            raise RuntimeError("SONIC back-transform target has no GDV unit contract")
        from matrix_smith.policy import PRIMITIVE_POLICY_BY_FAMILY

        scales = []
        for gic in definition.gics:
            policy = PRIMITIVE_POLICY_BY_FAMILY.get(str(gic.family))
            if policy is None:
                raise RuntimeError(
                    "no GDV coordinate-function contract is defined for "
                    f"SONIC family {gic.family!r}"
                )
            function = str(policy.function).upper()
            if function in GDV_LENGTH_COORDINATE_FUNCTIONS:
                scales.append(ANGSTROM_TO_BOHR)
            elif function in GDV_ANGULAR_COORDINATE_FUNCTIONS:
                scales.append(1.0)
            else:
                raise RuntimeError(
                    f"no GDV unit contract is defined for SONIC function {function!r}"
                )
        return np.asarray(scales, dtype=float)

    def _gdv_internal_coordinate_functions(self) -> tuple[str, ...]:
        """Return GDV coordinate kinds in the active frozen-GIC order."""

        size = len(self.coordinate_model.labels)
        if self.coordinate_model.kind != "sonic":
            return tuple("DIMENSIONLESS" for _index in range(size))
        families = tuple(self._sonic_active_families)
        if len(families) != size:
            raise RuntimeError("active SONIC families do not match optimizer variables")
        from matrix_smith.policy import PRIMITIVE_POLICY_BY_FAMILY

        functions = []
        for family in families:
            policy = PRIMITIVE_POLICY_BY_FAMILY.get(str(family))
            if policy is None:
                raise RuntimeError(
                    f"no GDV coordinate-function contract is defined for SONIC family {family!r}"
                )
            functions.append(str(policy.function).upper())
        return tuple(functions)

    def optimizer_metric(
        self,
        coordinates_angstrom: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the complete Wilson metric induced by ``dx/dq``.

        ``coordinate_directions`` contains the Cartesian columns ``dx/dq`` as
        rows in angstrom.  The optimizer gradient and Hessian use bohr-based
        Cartesian derivatives, so the unique compatible displacement metric
        is ``G = (dx/dq)_bohr.T (dx/dq)_bohr``.  Chemical family labels never
        rescale this tensor: doing so changes an RFO step under an otherwise
        invertible reparameterization of the same Cartesian tangent space.
        """

        metric_cache = getattr(self, "_optimizer_metric_cache", None)
        key = (
            None
            if metric_cache is None
            else self._coordinate_cache_key(coordinates_angstrom)
        )
        cached = None if metric_cache is None else metric_cache.get(key)
        if cached is not None and key is not None:
            metric_cache.move_to_end(key)
            return cached.copy()
        directions_bohr = (
            np.asarray(self.coordinate_directions(coordinates_angstrom), dtype=float)
            * ANGSTROM_TO_BOHR
        )
        metric = directions_bohr @ directions_bohr.T
        metric = 0.5 * (metric + metric.T)
        metric = _positive_metric_matrix(metric, self.settings.min_abs_metric_diagonal)
        if metric_cache is not None and key is not None:
            self._retain_recent_coordinate_value(
                metric_cache,
                key,
                metric,
                limit=self._coordinate_differential_cache_limit,
            )
        return metric.copy()

    def sonic_coordinate_directions(
        self, coordinates_angstrom: np.ndarray | None = None
    ) -> np.ndarray:
        """Return dx/dq rows for the underlying SONIC realization contract."""

        if self.coordinate_model.kind != "sonic":
            raise ValueError("underlying SONIC directions require a SONIC coordinate model")
        coords = (
            self.reference_coordinates
            if coordinates_angstrom is None
            else np.asarray(coordinates_angstrom, dtype=float)
        )
        return self._sonic_cartesian_from_q(coords, apply_variable_projection=False).T

    def _sonic_cartesian_from_q(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        apply_variable_projection: bool = True,
    ) -> np.ndarray:
        definition = self._sonic_definition
        if definition is None:
            return np.asarray(self.coordinate_model.directions_angstrom, dtype=float).T
        rigid_model = self._rigid_pose_model
        if (
            rigid_model is not None
            and not self.settings.fixed_atoms
            and not self.settings.rigid_reference_groups
            and self._assigned_cartesian_symmetry is None
            and self.pes_exploration_policy is None
        ):
            values = self._evaluate_all_sonic_values(np.asarray(coordinates_angstrom, dtype=float))
            cartesian_from_all_q = rigid_model.sonic_tangent(self._rigid_model_values(values))
            if self._sonic_rotation_atlas is not None:
                cartesian_from_all_q = self._sonic_rotation_atlas.cartesian_columns_from_local(
                    cartesian_from_all_q
                )
            selected = np.asarray(
                cartesian_from_all_q[:, self._sonic_coordinate_indices],
                dtype=float,
            )
            transform = (
                self.coordinate_model.sonic_from_coordinates if apply_variable_projection else None
            )
            return selected if transform is None else selected @ transform
        from matrix_smith import build_gic_b_matrix

        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
                parallel_workers=self.settings.coordinate_parallel_workers,
                rotation_reference_coordinates=(
                    None
                    if self._sonic_rotation_atlas is None
                    else self._sonic_rotation_atlas.reference_coordinates
                ),
            ).rows,
            dtype=float,
        )
        if self._sonic_rotation_atlas is not None:
            _unused, transformed = self._sonic_rotation_atlas.transform(
                np.zeros(b_matrix.shape[0], dtype=float), b_matrix
            )
            assert transformed is not None
            b_matrix = transformed
        if self.settings.stationary_point == "transition_state":
            # utilam.F:FormGI(IUseBI=4) forms the internal-force map from the
            # raw complete Wilson B matrix by SVD.  Finite torsion/fragment
            # tangents are useful predictors for LINK's hybrid minimum
            # backtransform, but replacing columns here changes G^- B and
            # therefore changes every DXRFO force and D2CorX secant.  TS
            # coordinate realization already follows GDV RedQ2X separately.
            coordinate_indices = (
                tuple(range(b_matrix.shape[0]))
                if self.settings.freeze_inactive_sonic
                else self._sonic_coordinate_indices
            )
            active_b_matrix = b_matrix[list(coordinate_indices), :]
            fixed_columns = np.asarray(
                [
                    3 * atom + component
                    for atom in self.settings.fixed_atoms
                    for component in range(3)
                ],
                dtype=int,
            )
            cartesian_from_all_q = cartesian_from_internal_jacobian(
                active_b_matrix,
                rcond=1.0e-8,
                fixed_cartesian_columns=fixed_columns,
            )
            selected = (
                np.asarray(cartesian_from_all_q, dtype=float)
                if not self.settings.freeze_inactive_sonic
                else np.asarray(
                    cartesian_from_all_q[:, self._sonic_coordinate_indices],
                    dtype=float,
                )
            )
            transform = (
                self.coordinate_model.sonic_from_coordinates
                if apply_variable_projection
                else None
            )
            return selected if transform is None else selected @ transform
        rigid_soft_model = self._rigid_pose_model_all
        if (
            rigid_soft_model is not None
            and not self.settings.fixed_atoms
            and not self.settings.rigid_reference_groups
            and self._assigned_cartesian_symmetry is None
            and self.pes_exploration_policy is None
        ):
            rigid_soft_values = self._evaluate_all_sonic_values(
                np.asarray(coordinates_angstrom, dtype=float)
            )
            fragment_cartesian_from_q = rigid_soft_model.sonic_tangent_from_base(
                self._rigid_model_values(rigid_soft_values),
                coordinates_angstrom,
            )
            if self._sonic_rotation_atlas is not None:
                fragment_cartesian_from_q = self._sonic_rotation_atlas.cartesian_columns_from_local(
                    fragment_cartesian_from_q
                )
            fragment_handled_indices = rigid_soft_model.coordinate_indices
        else:
            from .fragment_backtransform import direct_fragment_rigid_tangent

            fragment_tangent = direct_fragment_rigid_tangent(
                definition,
                coordinates_angstrom,
                b_matrix,
                fixed_atom_indices=self.settings.fixed_atoms,
            )
            fragment_cartesian_from_q = fragment_tangent.cartesian_from_q
            fragment_handled_indices = fragment_tangent.handled_indices
        full_b_matrix = b_matrix
        coordinate_indices = (
            tuple(range(full_b_matrix.shape[0]))
            if self.settings.freeze_inactive_sonic
            else self._sonic_coordinate_indices
        )
        b_matrix = full_b_matrix[list(coordinate_indices), :]
        fixed_columns = np.asarray(
            [3 * atom + component for atom in self.settings.fixed_atoms for component in range(3)],
            dtype=int,
        )
        cartesian_from_all_q = cartesian_from_internal_jacobian(
            b_matrix,
            rcond=1.0e-8,
            fixed_cartesian_columns=fixed_columns,
        )
        local_index = {full_index: index for index, full_index in enumerate(coordinate_indices)}
        from .hybrid_backtransform import acyclic_torsion_cartesian_tangent

        for torsion in self._acyclic_torsions:
            if torsion.coordinate_index not in local_index:
                continue
            torsion_tangent = acyclic_torsion_cartesian_tangent(coordinates_angstrom, torsion)
            if torsion_tangent is not None:
                # The finite mixed predictor reapplies every rigid-fragment
                # pose after rotating the bridge.  If the torsion moves atoms
                # in the reference fragment, its body frame moves too; add
                # the unique rigid-pose response that keeps all FTRANS/FROT
                # values fixed.  This is the analytic chain rule of that same
                # predictor; this is its analytic chain-rule contribution.
                handled = tuple(
                    index for index in fragment_handled_indices if index < full_b_matrix.shape[0]
                )
                if handled:
                    induced = full_b_matrix[list(handled), :] @ torsion_tangent
                    torsion_tangent = (
                        torsion_tangent - fragment_cartesian_from_q[:, list(handled)] @ induced
                    )
                cartesian_from_all_q[:, local_index[torsion.coordinate_index]] = torsion_tangent
        for full_index in fragment_handled_indices:
            if full_index in local_index:
                cartesian_from_all_q[:, local_index[full_index]] = fragment_cartesian_from_q[
                    :, full_index
                ]
        if not self.settings.freeze_inactive_sonic:
            selected = np.asarray(cartesian_from_all_q, dtype=float)
        else:
            selected = np.asarray(
                cartesian_from_all_q[:, self._sonic_coordinate_indices], dtype=float
            )
        transform = (
            self.coordinate_model.sonic_from_coordinates if apply_variable_projection else None
        )
        return selected if transform is None else selected @ transform

    def _project_rigid_reference_groups(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        groups = self.settings.rigid_reference_groups
        coords = np.asarray(coordinates_angstrom, dtype=float).copy()
        if not groups:
            return coords
        from matrix_chem import kabsch_align

        for group in groups:
            indices = np.asarray(group, dtype=int)
            if np.max(indices) >= coords.shape[0]:
                raise ValueError("rigid reference group atom index is outside geometry")
            coords[indices] = kabsch_align(
                coords[indices],
                self.reference_coordinates[indices],
            )
        return coords

    def actual_q(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        if self.coordinate_model.kind == "typed_onic":
            runtime = self._typed_onic_runtime
            reference = self._typed_onic_reference_values
            if runtime is None or reference is None:
                raise RuntimeError("typed ONIC optimizer runtime is unavailable")
            return runtime.evaluate(coordinates_angstrom).values - reference
        if self.coordinate_model.kind != "sonic":
            displacement = (
                np.asarray(coordinates_angstrom, dtype=float) - self.reference_coordinates
            )
            return self.coordinate_model.directions_angstrom @ displacement.reshape(-1)
        values, _ = self._evaluate_active_sonic(np.asarray(coordinates_angstrom, dtype=float))
        reference = self._sonic_reference_values
        if reference is None:
            raise RuntimeError("SONIC reference values are unavailable")
        sonic_displacement = values - reference
        transform = self.coordinate_model.sonic_from_coordinates
        if transform is None:
            return sonic_displacement
        return internal_from_cartesian_jacobian(transform, rcond=1.0e-10) @ sonic_displacement

    def coordinate_reference_values(self) -> np.ndarray:
        """Return the active absolute coordinate values of the frozen contract."""

        if self.coordinate_model.kind == "typed_onic":
            if self._typed_onic_reference_values is None:
                raise RuntimeError("typed ONIC reference values are unavailable")
            return self._typed_onic_reference_values.copy()
        if self.coordinate_model.kind != "sonic":
            return np.zeros(len(self.coordinate_model.labels), dtype=float)
        if self.coordinate_model.reference_values is not None:
            return np.asarray(self.coordinate_model.reference_values, dtype=float).copy()
        if self._sonic_reference_values is None:
            raise RuntimeError("SONIC reference values are unavailable")
        return np.asarray(self._sonic_reference_values, dtype=float).copy()

    def sonic_contract_labels(self) -> tuple[str, ...]:
        """Return the underlying frozen SONIC labels used to realize active variables."""

        return tuple(self.coordinate_model.sonic_labels or self.coordinate_model.labels)

    def sonic_reference_values(self) -> np.ndarray:
        if self._sonic_reference_values is None:
            raise RuntimeError("SONIC reference values are unavailable")
        return np.asarray(self._sonic_reference_values, dtype=float).copy()

    def sonic_values_from_q(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.sonic_reference_values() + self._sonic_displacements(q)

    def _sonic_displacements(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(q, dtype=float).reshape(-1)
        transform = self.coordinate_model.sonic_from_coordinates
        return vector if transform is None else np.asarray(transform, dtype=float) @ vector

    def _rigid_model_values(self, optimizer_values: np.ndarray) -> np.ndarray:
        """Map continuous optimizer FROT values into the current rigid chart."""

        values = np.asarray(optimizer_values, dtype=float)
        atlas = self._sonic_rotation_atlas
        return values.copy() if atlas is None else atlas.to_local_values(values)

    def _absolute_sonic_values(self, q: np.ndarray) -> np.ndarray:
        if self._sonic_full_reference_values is None or self._sonic_reference_values is None:
            raise RuntimeError("SONIC reference values are unavailable")
        if self.coordinate_model.rank_reduced_labels:
            # A rank-reduced active set cannot be combined with the old policy
            # of freezing every omitted GIC at its reference value: a dropped
            # row may be linearly induced by the retained rows.  Construct a
            # locally compatible full SONIC target and let that dependent row
            # follow the independent displacement.
            reference_values, full_b = self._evaluate_all_sonic(self.reference_coordinates)
            active = tuple(self._sonic_coordinate_indices)
            active_delta = self._sonic_displacements(q)
            active_b = np.asarray(full_b, dtype=float)[list(active), :]
            cartesian_delta = np.linalg.pinv(active_b, rcond=1.0e-10) @ active_delta
            return np.asarray(reference_values, dtype=float) + np.asarray(full_b) @ cartesian_delta
        target = self._sonic_full_reference_values.copy()
        target[list(self._sonic_coordinate_indices)] = (
            self._sonic_reference_values + self._sonic_displacements(q)
        )
        return target

    def _evaluate_all_sonic(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import build_sparse_gic_b_matrix, evaluate_gic_values

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        values = self._phase_match_sonic_values(
            values,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ),
        )
        b_matrix = np.asarray(
            build_sparse_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
                parallel_workers=self.settings.coordinate_parallel_workers,
            ).to_dense(),
            dtype=float,
        )
        if self._sonic_rotation_atlas is not None:
            values, transformed = self._sonic_rotation_atlas.transform(values, b_matrix)
            assert transformed is not None
            b_matrix = transformed
        return values, b_matrix

    def _evaluate_all_sonic_values(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import evaluate_gic_values

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        values = self._phase_match_sonic_values(
            values,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ),
        )
        if self._sonic_rotation_atlas is not None:
            values, _rows = self._sonic_rotation_atlas.transform(values)
        return values

    def _evaluate_sonic_subset(
        self, coordinates_angstrom: np.ndarray, coordinate_indices: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate only the requested SONIC values and Wilson-B rows."""
        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import evaluate_gic_subset

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        values, rows = evaluate_gic_subset(
            definition,
            coordinate_indices,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
            parallel_workers=self.settings.coordinate_parallel_workers,
        )
        values = self._phase_match_sonic_values(
            values,
            coordinate_indices=coordinate_indices,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ),
        )
        if self._sonic_rotation_atlas is not None:
            values, transformed = self._sonic_rotation_atlas.transform_subset(
                coordinate_indices, values, rows
            )
            assert transformed is not None
            rows = transformed
        return values, rows

    def _evaluate_sonic_values_subset(
        self, coordinates_angstrom: np.ndarray, coordinate_indices: tuple[int, ...]
    ) -> np.ndarray:
        """Evaluate requested SONIC values without constructing Wilson rows."""

        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import evaluate_gic_values_subset

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        values = evaluate_gic_values_subset(
            definition,
            coordinate_indices,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        values = self._phase_match_sonic_values(
            values,
            coordinate_indices=coordinate_indices,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ),
        )
        if self._sonic_rotation_atlas is None:
            return values
        transformed, _rows = self._sonic_rotation_atlas.transform_subset(coordinate_indices, values)
        return transformed

    def _evaluate_active_sonic(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import (
            build_sparse_gic_b_matrix,
            evaluate_gic_values_subset,
        )

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        selected_indices = tuple(self._sonic_coordinate_indices)
        values = evaluate_gic_values_subset(
            definition,
            selected_indices,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        values = self._phase_match_sonic_values(
            values,
            coordinate_indices=selected_indices,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ),
        )
        b_matrix = np.asarray(
            build_sparse_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
                coordinate_indices=selected_indices,
                parallel_workers=self.settings.coordinate_parallel_workers,
            ).to_dense(),
            dtype=float,
        )
        if self._sonic_rotation_atlas is not None:
            values, transformed = self._sonic_rotation_atlas.transform_subset(
                selected_indices, values, b_matrix
            )
            assert transformed is not None
            b_matrix = transformed
        return values, b_matrix

    def _evaluate_active_sonic_values(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        return self._evaluate_sonic_values_subset(
            coordinates_angstrom, tuple(self._sonic_coordinate_indices)
        )

    def _evaluate_periodic_sonic_primitives(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        rotation_reference_coordinates: np.ndarray | None,
    ) -> dict[str, float]:
        definition = self._sonic_definition
        primitive_ids = getattr(self, "_sonic_periodic_primitive_ids", ())
        if definition is None or not primitive_ids:
            return {}
        from matrix_smith import evaluate_primitive_values

        return evaluate_primitive_values(
            definition,
            primitive_ids=primitive_ids,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference_coordinates,
        )

    def _phase_match_sonic_values(
        self,
        values: np.ndarray,
        *,
        coordinate_indices: tuple[int, ...] | None = None,
        reference_values: np.ndarray | None = None,
        primitive_values: Mapping[str, float] | None = None,
        primitive_reference_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Apply GDV primitive ``CrdVa1/FixCrd`` phases before forming GICs."""

        result = np.asarray(values, dtype=float).reshape(-1).copy()
        indices = (
            tuple(range(result.size))
            if coordinate_indices is None
            else tuple(int(index) for index in coordinate_indices)
        )
        if len(indices) != result.size:
            raise ValueError("SONIC phase indices do not match evaluated values")
        from .periodic import gdv_match_dihedral_phase

        primitive_references = (
            getattr(self, "_sonic_primitive_phase_reference_values", {})
            if primitive_reference_values is None
            else {
                str(identifier): float(value)
                for identifier, value in primitive_reference_values.items()
            }
        )
        primitive_current = {
            str(identifier): float(value)
            for identifier, value in dict(primitive_values or {}).items()
        }
        corrected_indices: set[int] = set()
        components_by_gic = getattr(self, "_sonic_periodic_gic_components", {})
        if primitive_current and primitive_references and components_by_gic:
            for local_index, full_index in enumerate(indices):
                components = components_by_gic.get(full_index, ())
                for primitive_id, coefficient in components:
                    if (
                        primitive_id not in primitive_current
                        or primitive_id not in primitive_references
                    ):
                        continue
                    raw_value = primitive_current[primitive_id]
                    matched_value = gdv_match_dihedral_phase(
                        raw_value, primitive_references[primitive_id]
                    )
                    result[local_index] += coefficient * (matched_value - raw_value)
                    corrected_indices.add(full_index)

        periods = self._sonic_periodic_periods
        reference = (
            self._sonic_phase_reference_values
            if reference_values is None
            else np.asarray(reference_values, dtype=float).reshape(-1)
        )
        if not periods or reference is None:
            return result

        for local_index, full_index in enumerate(indices):
            period = periods.get(full_index)
            if period is None or full_index in corrected_indices:
                continue
            if not np.isclose(period, 2.0 * np.pi, rtol=0.0, atol=1.0e-14):
                raise RuntimeError("GDV SONIC periodic coordinates require a 2-pi domain")
            result[local_index] = gdv_match_dihedral_phase(
                result[local_index], reference[full_index]
            )
        return result

    def _update_sonic_phase_reference(self, coordinates_angstrom: np.ndarray) -> None:
        """Advance periodic GIC phases only at a newly accepted geometry."""

        definition = self._sonic_definition
        reference = self._sonic_phase_reference_values
        if (
            self.coordinate_model.kind != "sonic"
            or definition is None
            or reference is None
            or (
                not self._sonic_periodic_periods
                and not getattr(self, "_sonic_periodic_primitive_ids", ())
            )
        ):
            return
        from matrix_smith import evaluate_gic_values

        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        raw_primitive_values = self._evaluate_periodic_sonic_primitives(
            coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        from .periodic import gdv_match_dihedral_phase

        primitive_references = getattr(
            self, "_sonic_primitive_phase_reference_values", {}
        )
        next_primitive_references = {
            primitive_id: gdv_match_dihedral_phase(
                raw_primitive_values[primitive_id], previous_value
            )
            for primitive_id, previous_value in primitive_references.items()
        }

        raw_values = evaluate_gic_values(
            definition,
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
            rotation_reference_coordinates=rotation_reference,
        )
        self._sonic_phase_reference_values = self._phase_match_sonic_values(
            raw_values,
            reference_values=reference,
            primitive_values=raw_primitive_values,
            primitive_reference_values=next_primitive_references,
        )
        self._sonic_primitive_phase_reference_values = next_primitive_references

    def _maybe_rebase_sonic_rotations(self, coordinates_angstrom: np.ndarray) -> bool:
        atlas = self._sonic_rotation_atlas
        definition = self._sonic_definition
        if atlas is None or definition is None or not atlas.active:
            return False
        from matrix_smith import evaluate_gic_values

        local_values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=atlas.reference_coordinates,
        )
        local_values = self._phase_match_sonic_values(
            local_values,
            primitive_values=self._evaluate_periodic_sonic_primitives(
                coordinates_angstrom,
                rotation_reference_coordinates=atlas.reference_coordinates,
            ),
        )
        # Keep every optimizer step inside a well-conditioned local SO(3)
        # chart.  Waiting until the logarithm is close to pi is too late for
        # strongly deforming or reactive fragments: their canonical frames can
        # become ill-conditioned well before the formal singularity.
        if atlas.max_local_norm(local_values) < self.settings.fragment_rotation_rebase_threshold:
            return False
        atlas.rebase(local_values, coordinates_angstrom)
        # The rigid realization must use the same newly rebased fragment
        # frames as the value/B evaluators.  Recompile only this immutable
        # realization model; the frozen SONIC definition and optimizer
        # coordinate labels remain unchanged.
        from .rigid_pose import RigidComplexModel

        rebased_definition = replace(
            definition,
            reference_coordinates_angstrom=tuple(
                tuple(float(component) for component in row)
                for row in np.asarray(coordinates_angstrom, dtype=float)
            ),
        )
        rigid_model = RigidComplexModel.try_from_definition(rebased_definition)
        self._rigid_pose_model_all = rigid_model
        self._rigid_pose_model = (
            rigid_model
            if rigid_model is not None
            and rigid_model.supports_coordinate_indices(self._sonic_coordinate_indices)
            else None
        )
        return True

    def _point_cartesian_symmetry(
        self,
        coordinates_angstrom: np.ndarray | None,
    ):
        """Return frozen exploitation or instantaneous exploration symmetry."""

        reference = self._assigned_cartesian_symmetry
        policy = getattr(self, "pes_exploration_policy", None)
        if policy is None:
            return reference
        if not policy.pointwise_oracle_symmetry:
            return None
        if coordinates_angstrom is None:
            return reference
        thresholds = self._symmetry_thresholds
        if thresholds is None:
            return reference

        from matrix_chem import MolecularGeometry
        from matrix_chem.symmetry import analyze_molecular_symmetry

        return analyze_molecular_symmetry(
            MolecularGeometry(
                atoms=self.atoms,
                coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
            ),
            distance_tolerance=thresholds.distance_angstrom,
            inertia_tolerance=thresholds.inertia_relative,
            max_rotation_order=thresholds.max_rotation_order,
        )

    def _symmetrize_point_gradient(
        self,
        result: PointEvaluationResult,
        coordinates_angstrom: np.ndarray | None = None,
    ) -> PointEvaluationResult:
        """Project gradients with the symmetry contract of the active workflow."""
        symmetry = self._point_cartesian_symmetry(coordinates_angstrom)
        if (
            symmetry is None
            or not self.settings.symmetrize_analytic_gradients
            or result.gradient_hartree_per_bohr is None
        ):
            return result
        gradient = np.asarray(result.gradient_hartree_per_bohr, dtype=float)
        frozen_contract = getattr(self, "pes_exploration_policy", None) is None
        frozen_projector = getattr(
            self,
            "_frozen_cartesian_symmetry_projector",
            None,
        )
        projector = (
            frozen_projector
            if frozen_contract
            and symmetry is self._assigned_cartesian_symmetry
            and frozen_projector is not None
            else _totally_symmetric_cartesian_projector(
                symmetry,
                natoms=len(self.atoms),
            )
        )
        projected = (projector @ gradient.reshape(-1)).reshape(gradient.shape)
        execution = dict(result.execution)
        execution.update(
            {
                "gradient_symmetrization": "cartesian_invariant_subspace_projector",
                "gradient_symmetry_point_group": symmetry.point_group,
                "gradient_symmetry_operation_count": len(symmetry.operations),
                "gradient_symmetry_contract": (
                    "frozen_initial_group"
                    if frozen_contract
                    else "instantaneous_exploration_group"
                ),
            }
        )
        return replace(
            result,
            gradient_hartree_per_bohr=projected,
            execution=execution,
        )

    def evaluate(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        tag: str,
        use_cache: bool = True,
        persist_cache: bool = True,
        requested_properties: Sequence[str] = (),
        realized_coordinates_angstrom: np.ndarray | None = None,
        restart_artifact: Path | str | None = None,
        restart_projection: str | None = None,
    ) -> OptimizerEvaluation:
        vector = np.asarray(q, dtype=float).reshape(-1)
        if use_cache:
            with self._lock:
                cached = self.cache.lookup(
                    vector,
                    requested_properties=requested_properties,
                )
            if cached is not None:
                # A cache hit returns the evaluated Cartesian point together
                # with its optimizer coordinates.  Re-realizing the requested
                # (merely tolerance-equivalent) q is both slower and less
                # internally consistent than using that stored geometry.
                cached_coordinates = np.asarray(
                    cached.coordinates_angstrom, dtype=float
                )
                projected = (
                    self._symmetrize_point_gradient(
                        cached.result,
                        cached_coordinates,
                    )
                    if not requested_properties or "gradient" in requested_properties
                    else cached.result
                )
                if projected is cached.result:
                    return cached
                return replace(cached, result=projected)
        generated_coordinates = realized_coordinates_angstrom is None
        if generated_coordinates:
            # Capture the realization diagnostic atomically: finite-difference
            # workers may share this service, while the direct path itself is
            # much cheaper than a redundant full SONIC value evaluation.
            with self._lock:
                coords = self.coordinates_from_q(vector)
                direct_q_is_exact = self._direct_q_realization_is_exact()
        else:
            coords = np.asarray(realized_coordinates_angstrom, dtype=float)
            direct_q_is_exact = False
        if coords.shape != self.reference_coordinates.shape or not np.all(np.isfinite(coords)):
            raise ValueError("realized coordinates have invalid shape or values")
        with self._lock:
            point_index = self.qm_evaluations
            self.qm_evaluations += 1
        point = ScanPoint(
            index=point_index,
            displacement=0.0,
            coordinates_angstrom=coords,
            xyz_path=self.run_dir / "points" / f"eval_{point_index:05d}.xyz",
            result_path=self.run_dir / "points" / f"eval_{point_index:05d}.json",
        )
        result = (
            self._evaluate_builtin_backend_with_state_following(
                point,
                tag=tag,
                requested_properties=requested_properties,
                restart_artifact=restart_artifact,
                restart_projection=restart_projection,
            )
            if self.backend is not None
            else self._evaluate_external_command(
                point,
                tag=tag,
                requested_properties=requested_properties,
            )
        )
        if self.backend is None:
            result = self._follow_electronic_state(result)
        if requested_properties:
            execution = dict(result.execution)
            execution["link_evaluation_tag"] = tag
            if "gradient" not in requested_properties:
                # Energy-only is a hard contract.  Some resident or command
                # line backends may return extra derivatives anyway; LINK
                # must discard them so they cannot affect symmetry projection,
                # cache reuse, convergence, or accounting.
                result = replace(
                    result,
                    gradient_hartree_per_bohr=None,
                    execution={**execution, "gradient_evaluations": 0},
                )
            else:
                result = replace(result, execution=execution)
            if "hessian" not in requested_properties:
                result = replace(
                    result,
                    hessian_hartree_per_bohr2=None,
                    execution={**dict(result.execution), "hessian_evaluations": 0},
                )
        if not requested_properties or "gradient" in requested_properties:
            result = self._symmetrize_point_gradient(result, coords)
        if requested_properties:
            execution = dict(result.execution)
            execution["_requested_properties"] = [str(item) for item in requested_properties]
            if "gradient" not in requested_properties:
                execution["gradient_evaluations"] = 0
                # Some command-line backends return a gradient even when the
                # request was energy-only.  Preserve that supplied property
                # for cache reuse, but do not count it as an additional LINK
                # gradient evaluation.
                result = replace(result, execution=execution)
            if "hessian" not in requested_properties:
                execution = dict(result.execution)
                execution["hessian_evaluations"] = 0
                result = replace(result, execution=execution)
        if self.settings.include_cv_exponential_field:
            from .cv_layer import add_cv_exponential_field

            result = add_cv_exponential_field(
                result,
                self._cv_atomic_numbers,
                coords,
                self._cv_bonded_pairs,
            )
        if result.status != "completed" or result.energy_hartree is None:
            raise RuntimeError(result.message or f"QM evaluation failed for {tag}")
        if point.result_path is not None:
            write_point_result(point.result_path, result)
        with self._lock:
            energy_count = int(
                result.execution.get("energy_evaluations", result.energy_hartree is not None)
            )
            if tag.startswith("final-hessian-"):
                self.final_hessian_energy_evaluations += energy_count
            else:
                self.energy_evaluations += energy_count
            self.gradient_evaluations += int(
                result.execution.get(
                    "gradient_evaluations", result.gradient_hartree_per_bohr is not None
                )
            )
            self.hessian_evaluations += int(
                result.execution.get(
                    "hessian_evaluations", result.hessian_hartree_per_bohr2 is not None
                )
            )
            self.fd_displacements += int(result.execution.get("fd_displacements", 0))
            # Count every finite-difference energy point, including
            # directional TS points (fd-ts-*), in the campaign accounting.
            self.fd_displacements += int(tag.startswith("fd-"))
        realized_q = (
            vector.copy()
            if direct_q_is_exact or self.coordinate_model.kind not in {"sonic", "typed_onic"}
            else self.actual_q(coords)
        )
        evaluation = OptimizerEvaluation(
            q=realized_q,
            coordinates_angstrom=coords,
            result=result,
            cache_hit=False,
            chart_epoch=self.cache.chart_epoch,
        )
        if persist_cache:
            with self._lock:
                self.cache.add(evaluation)
        return evaluation

    def _state_fingerprints(self, result: PointEvaluationResult):
        if self.backend is None or int(self.backend.electronic_state) <= 0:
            return ()
        execution = dict(result.execution)
        archive_text = str(execution.get("state_fingerprint_file", "")).strip()
        if not archive_text:
            raise RuntimeError(
                "APOC state following was requested, but the QM adapter returned no state manifold"
            )
        archive = Path(archive_text)
        if not archive.is_file():
            raise FileNotFoundError(f"APOC state-fingerprint archive not found: {archive}")
        from matrix_apoc import StateFingerprint

        with np.load(archive, allow_pickle=False) as payload:
            state_ids = np.asarray(payload["state_ids"], dtype=int).reshape(-1)
            excitations = np.asarray(payload["excitation_energies_hartree"], dtype=float).reshape(
                -1
            )
            vectors = np.asarray(payload["vectors"], dtype=float)
            representation = str(np.asarray(payload["representation"]).reshape(-1)[0])
        if vectors.ndim != 2 or vectors.shape[0] != state_ids.size:
            raise ValueError("APOC state-fingerprint archive has inconsistent dimensions")
        if excitations.shape != state_ids.shape:
            raise ValueError("APOC state energies do not match the saved state manifold")
        ground = float(execution["ground_energy_hartree"])
        return tuple(
            StateFingerprint(
                int(state_id),
                vector,
                representation,
                energy_hartree=ground + float(excitation),
                label=f"root {int(state_id)}",
                source=str(archive),
            )
            for state_id, excitation, vector in zip(state_ids, excitations, vectors, strict=True)
        )

    def _follow_electronic_state(self, result: PointEvaluationResult) -> PointEvaluationResult:
        if self.backend is None or int(self.backend.electronic_state) <= 0:
            return result
        if str(self.backend.state_tracking).strip().casefold() != "apoc":
            raise RuntimeError("excited-state LINK points require APOC state following")
        candidates = self._state_fingerprints(result)
        requested = int(self.backend.electronic_state)
        with self._lock:
            reference = self._state_reference
        if reference is None:
            try:
                selected = next(item for item in candidates if item.state_id == requested)
            except StopIteration as exc:
                raise RuntimeError(
                    f"QM state manifold does not contain requested initial root {requested}"
                ) from exc
            overlap = 1.0
            margin = 1.0
        else:
            from matrix_apoc import follow_state_fingerprint

            selected, match = follow_state_fingerprint(
                reference,
                candidates,
                minimum_overlap=float(self.backend.state_tracking_minimum_overlap),
                ambiguity_margin=float(self.backend.state_tracking_ambiguity_margin),
                strict=True,
            )
            overlap = float(match.overlap)
            margin = float(match.margin)
            continuous = bool(match.continuous)
            ambiguous = bool(match.ambiguous)
        if reference is None:
            continuous = True
            ambiguous = False
        if selected.energy_hartree is None:
            raise RuntimeError("APOC-selected state has no total energy")
        execution = dict(result.execution)
        execution.update(
            {
                "state_tracking": "APOC",
                "requested_electronic_state": requested,
                "selected_electronic_state": int(selected.state_id),
                "state_overlap": overlap,
                "state_assignment_margin": margin,
                "state_assignment_continuous": continuous,
                "state_assignment_ambiguous": ambiguous,
                "state_reference_id": (requested if reference is None else int(reference.state_id)),
            }
        )
        return replace(
            result,
            energy_hartree=float(selected.energy_hartree),
            execution=execution,
        )

    def accept_electronic_state(self, evaluation: OptimizerEvaluation) -> None:
        """Advance the APOC reference only after LINK accepts the geometry."""

        if self.backend is None or int(self.backend.electronic_state) <= 0:
            return
        selected_id = int(
            evaluation.result.execution.get(
                "selected_electronic_state", self.backend.electronic_state
            )
        )
        candidates = self._state_fingerprints(evaluation.result)
        selected = next(
            (item for item in candidates if item.state_id == selected_id),
            None,
        )
        if selected is None:
            raise RuntimeError(f"accepted LINK point lacks APOC-selected root {selected_id}")
        execution = evaluation.result.execution
        continuous = bool(execution.get("state_assignment_continuous", True))
        ambiguous = bool(execution.get("state_assignment_ambiguous", False))
        if not continuous or ambiguous:
            raise ElectronicStateResolutionError(
                "an accepted LINK point cannot advance an ambiguous or discontinuous state"
            )
        with self._lock:
            self._state_reference = selected

    def supports_resident_potential_batch(self) -> bool:
        return bool(
            self.backend is not None
            and _normalized_backend_name(self.backend.name) == "zaff"
            and self.backend.force_field is not None
            and not self.settings.include_cv_exponential_field
        )

    def evaluate_resident_potential_batch(
        self,
        q_values: Sequence[Sequence[float] | np.ndarray],
        *,
        tags: Sequence[str],
        requested_properties: Sequence[str],
        persist_cache: bool = False,
    ) -> tuple[OptimizerEvaluation, ...]:
        """Evaluate one homogeneous block through a resident potential backend."""

        if not self.supports_resident_potential_batch():
            return tuple(
                self.evaluate(
                    q,
                    tag=tag,
                    use_cache=False,
                    persist_cache=persist_cache,
                    requested_properties=requested_properties,
                )
                for q, tag in zip(q_values, tags, strict=True)
            )
        vectors = tuple(np.asarray(q, dtype=float).reshape(-1) for q in q_values)
        labels = tuple(str(tag) for tag in tags)
        if len(vectors) != len(labels):
            raise ValueError("ZAFF batch q/tag counts differ")
        if not vectors:
            return ()
        coordinates = tuple(self.coordinates_from_q_batch(vectors))
        direct_batch_q_is_exact = bool(
            self._rigid_pose_model is not None
            and not self.settings.fixed_atoms
            and not self.settings.rigid_reference_groups
            and self._assigned_cartesian_symmetry is None
            and self.pes_exploration_policy is None
        )
        backend = self.backend
        assert backend is not None and backend.force_field is not None
        properties = tuple(requested_properties or backend.properties or ("energy", "gradient"))
        from matrix_engines import DerivativeOrder, PotentialSystem
        from matrix_zaff import ZaffBackend

        order = (
            DerivativeOrder.HESSIAN
            if "hessian" in properties
            else DerivativeOrder.GRADIENT
            if "gradient" in properties
            else DerivativeOrder.ENERGY
        )
        session = ZaffBackend().prepare(
            PotentialSystem(
                atoms=self.atoms,
                charge=backend.charge,
                multiplicity=backend.multiplicity,
            ),
            model=str(backend.force_field),
            options={
                "zoom_level": backend.zaff_zoom_level,
                "xyzin": self.xyzin_path,
            },
        )
        results = session.evaluate_batch(
            coordinates,
            derivative_order=order,
        )
        result_source = f"ZAFF {backend.force_field}"
        with self._lock:
            first_index = self.qm_evaluations
            self.qm_evaluations += len(results)
        evaluations: list[OptimizerEvaluation] = []
        for offset, (vector, coords, result) in enumerate(
            zip(vectors, coordinates, results, strict=True)
        ):
            point_index = first_index + offset
            tag = labels[offset]
            point_result = PointEvaluationResult(
                point_index=point_index,
                displacement=0.0,
                energy_hartree=result.energy_hartree,
                gradient_hartree_per_bohr=(
                    result.gradient_hartree_per_bohr if "gradient" in properties else None
                ),
                hessian_hartree_per_bohr2=(
                    result.hessian_hartree_per_bohr2 if "hessian" in properties else None
                ),
                backend_coordinates_angstrom=coords,
                source=result_source,
                execution={
                    **dict(result.execution),
                    "link_evaluation_tag": tag,
                    "_requested_properties": list(properties),
                    "gradient_evaluations": int("gradient" in properties),
                    "hessian_evaluations": int("hessian" in properties),
                    **(
                        {"force_field_resolution": dict(backend.resolution)}
                        if backend.resolution is not None
                        else {}
                    ),
                },
            )
            if "gradient" in properties:
                point_result = self._symmetrize_point_gradient(point_result, coords)
            result_path = self.run_dir / "points" / f"eval_{point_index:05d}.json"
            write_point_result(result_path, point_result)
            with self._lock:
                energy_count = int(point_result.energy_hartree is not None)
                if tag.startswith("final-hessian-"):
                    self.final_hessian_energy_evaluations += energy_count
                else:
                    self.energy_evaluations += energy_count
                self.gradient_evaluations += int(point_result.gradient_hartree_per_bohr is not None)
                self.hessian_evaluations += int(point_result.hessian_hartree_per_bohr2 is not None)
            realized_q = (
                np.asarray(vector, dtype=float).copy()
                if direct_batch_q_is_exact
                or self.coordinate_model.kind not in {"sonic", "typed_onic"}
                else self.actual_q(coords)
            )
            evaluation = OptimizerEvaluation(
                q=realized_q,
                coordinates_angstrom=coords,
                result=point_result,
                cache_hit=False,
                chart_epoch=self.cache.chart_epoch,
            )
            if persist_cache:
                with self._lock:
                    self.cache.add(evaluation)
            evaluations.append(evaluation)
        return tuple(evaluations)

    def estimate_energy_noise(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        reference: OptimizerEvaluation,
        samples: int,
    ) -> tuple[float, tuple[float, ...]]:
        energies = [reference.energy_hartree]
        for index in range(int(samples)):
            evaluation = self.evaluate(
                q,
                tag=f"noise-{index + 1}",
                use_cache=False,
                persist_cache=False,
            )
            energies.append(evaluation.energy_hartree)
        if len(energies) < 2:
            return 0.0, tuple(energies)
        return float(np.std(np.asarray(energies, dtype=float), ddof=1)), tuple(
            float(item) for item in energies
        )

    def _evaluate_external_command(
        self,
        point: ScanPoint,
        *,
        tag: str,
        requested_properties: Sequence[str] = (),
    ) -> PointEvaluationResult:
        if not self.engine_command.strip():
            raise ValueError("optimizer needs an engine command or built-in backend")
        assert point.xyz_path is not None
        assert point.result_path is not None
        point.xyz_path.parent.mkdir(parents=True, exist_ok=True)
        point.result_path.parent.mkdir(parents=True, exist_ok=True)
        _write_xyz(point.xyz_path, self.atoms, point.coordinates_angstrom, tag)
        command = _format_command(
            self.engine_command,
            xyz=point.xyz_path,
            result=point.result_path,
            index=point.index,
            workdir=point.xyz_path.parent,
            tag=tag,
            properties=",".join(str(item) for item in requested_properties),
        )
        require_authorized_descendant_calculation(
            backend="LINK/external-point-evaluator",
            input_path=point.xyz_path,
            command=command,
            workdir=point.xyz_path.parent,
        )
        completed = subprocess.run(
            command,
            cwd=point.xyz_path.parent,
            check=False,
            timeout=self.timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            return PointEvaluationResult(
                point_index=point.index,
                displacement=0.0,
                status="failed",
                message=completed.stdout.strip(),
                source=" ".join(command),
            )
        return read_point_result(point.result_path, point=point)

    def _evaluate_builtin_backend_with_state_following(
        self,
        point: ScanPoint,
        *,
        tag: str,
        requested_properties: Sequence[str] = (),
        restart_artifact: Path | str | None = None,
        restart_projection: str | None = None,
    ) -> PointEvaluationResult:
        backend = self.backend
        assert backend is not None
        if restart_artifact is not None:
            backend = replace(
                backend,
                restart_artifact=Path(restart_artifact),
                restart_projection=(
                    None if restart_projection is None else str(restart_projection)
                ),
            )
        if int(backend.electronic_state) <= 0:
            return self._evaluate_builtin_backend(
                point,
                tag=tag,
                requested_properties=requested_properties,
                backend_override=backend,
            )
        if str(backend.state_tracking).strip().casefold() != "apoc":
            raise RuntimeError("excited-state LINK points require APOC state following")
        target_root = int(backend.electronic_state)
        initial_roots = max(
            int(backend.excited_states or 0),
            int(backend.state_tracking_initial_roots or 0),
            6,
            target_root + 3,
        )
        increment = int(backend.state_tracking_root_increment)
        maximum = max(
            int(backend.state_tracking_max_roots or 0),
            12,
            target_root + 7,
        )
        if increment <= 0 or maximum < initial_roots:
            raise ValueError("invalid APOC root-expansion contract")
        attempted: list[int] = []
        roots = initial_roots
        last_error: Exception | None = None
        while True:
            attempted.append(roots)
            expanded = replace(backend, excited_states=roots)
            result = self._evaluate_builtin_backend(
                point,
                tag=tag,
                requested_properties=requested_properties,
                backend_override=expanded,
                state_attempt=len(attempted),
            )
            try:
                followed = self._follow_electronic_state(result)
            except Exception as exc:
                from matrix_apoc import StateContinuityError

                if not isinstance(exc, StateContinuityError):
                    raise
                last_error = exc
                if roots >= maximum:
                    break
                roots = min(maximum, roots + increment)
                continue
            execution = dict(followed.execution)
            execution.update(
                {
                    "state_following_contract": "strict_APOC_root_expansion_v1",
                    "state_root_spaces_attempted": attempted,
                    "state_root_expansion_count": len(attempted) - 1,
                    "energy_evaluations": max(
                        int(execution.get("energy_evaluations", 1)), len(attempted)
                    ),
                    "gradient_evaluations": (
                        max(
                            int(execution.get("gradient_evaluations", 1)),
                            len(attempted),
                        )
                        if followed.gradient_hartree_per_bohr is not None
                        else 0
                    ),
                }
            )
            return replace(followed, execution=execution)
        raise ElectronicStateResolutionError(
            "APOC state identity remained ambiguous or discontinuous after root-space "
            f"expansion {attempted}: {last_error}"
        )

    def _evaluate_builtin_backend(
        self,
        point: ScanPoint,
        *,
        tag: str,
        requested_properties: Sequence[str] = (),
        backend_override: QMScanBackend | None = None,
        state_attempt: int = 1,
    ) -> PointEvaluationResult:
        point_run_dir = self.run_dir / "backend_points" / f"eval_{point.index:05d}"
        if state_attempt > 1:
            point_run_dir = point_run_dir / f"state-root-attempt-{state_attempt:02d}"
        backend = self.backend if backend_override is None else backend_override
        if backend is not None and requested_properties:
            backend = replace(backend, properties=tuple(requested_properties))
        results = run_qm_scan_points(
            self.xyzin_path,
            (point,),
            backend,
            run_dir=point_run_dir,
        )
        result = results[0]
        if point.result_path is not None:
            write_point_result(point.result_path, result)
        return result


def coordinate_model_from_xyzin(
    xyzin_path: Path | str,
    *,
    kind: str = "cartesian",
    coordinates: Sequence[str] = (),
    task_regime: str | None = None,
    metric_diagonal: Sequence[float] | None = None,
    pes_exploration: bool = False,
    retained_group: str = "C1",
    sonic_definition: object | None = None,
) -> OptimizerCoordinateModel:
    geometry = read_xyzin_geometry(Path(xyzin_path))
    ncart = int(np.asarray(geometry.coordinates_angstrom).size)
    coordinate_kind = str(kind).replace("-", "_")
    normalized_task_regime = str(task_regime or "").strip().upper().replace("-", "_")
    if normalized_task_regime not in {"", "MINIMUM", "TRANSITION_STATE"}:
        raise ValueError(
            "task_regime must be 'MINIMUM', 'TRANSITION_STATE', or omitted"
        )
    if coordinate_kind == "cartesian":
        directions = np.eye(ncart, dtype=float)
        labels = tuple(f"X{index + 1}" for index in range(ncart))
    elif coordinate_kind == "sonic":
        from matrix_smith import (
            build_pes_exploration_gic_definition_from_xyzin,
            read_gic_definition_from_xyzin,
            sonic_definition_identity_sha256,
        )

        if sonic_definition is not None and pes_exploration:
            raise ValueError(
                "an explicit frozen SONIC definition cannot be replaced by PES exploration"
            )
        persisted = read_gic_definition_from_xyzin(Path(xyzin_path))
        from .onic_contract_gate import (
            has_frozen_oracle_sonic_contract,
            validate_link_onic_contract,
        )

        if has_frozen_oracle_sonic_contract(Path(xyzin_path)):
            validate_link_onic_contract(
                Path(xyzin_path),
                definition=persisted,
                orientation="SONIC",
            )
        if sonic_definition is not None:
            if sonic_definition_identity_sha256(sonic_definition) != (
                sonic_definition_identity_sha256(persisted)
            ):
                raise ValueError("explicit and serialized frozen SONIC definitions differ")
            definition = sonic_definition
        elif pes_exploration:
            definition = build_pes_exploration_gic_definition_from_xyzin(
                Path(xyzin_path), retained_group=retained_group
            )
        else:
            definition = persisted
        if not coordinates:
            if pes_exploration:
                coordinates = tuple(gic.name or gic.identifier for gic in definition.gics)
            elif normalized_task_regime == "TRANSITION_STATE":
                # A first-order saddle is a full vibrational-space problem.
                # Selecting only the totally symmetric rows here silently
                # changes the frozen SMITH chart and can constrain the search
                # onto a singular ordinary-angle boundary.
                coordinates = tuple(gic.name or gic.identifier for gic in definition.gics)
            else:
                from matrix_smith.symmetry_labels import irrep_sequence

                totally_symmetric = irrep_sequence(definition.point_group)[0]
                coordinates = tuple(
                    gic.name or gic.identifier
                    for gic in definition.gics
                    if not definition.symmetrize or gic.irrep == totally_symmetric
                )
        if not coordinates:
            raise ValueError("SONIC optimizer coordinate model needs explicit coordinate labels")
        rows = [
            coordinate_direction_from_gic(
                xyzin_path,
                item,
                definition=definition,
            ).vector_angstrom
            for item in coordinates
        ]
        directions = np.vstack(rows)
        labels = tuple(str(item) for item in coordinates)
        independent, dropped = _independent_coordinate_rows(directions, labels)
        if dropped:
            directions = directions[list(independent)]
            labels = tuple(labels[index] for index in independent)
    else:
        raise ValueError(f"unsupported optimizer coordinate model: {kind}")
    metric = (
        np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1)
        if metric_diagonal is None
        else np.asarray(metric_diagonal, dtype=float).reshape(-1)
    )
    if metric.shape != (directions.shape[0],):
        raise ValueError("metric diagonal length must match coordinate count")
    return OptimizerCoordinateModel(
        kind=coordinate_kind,
        labels=labels,
        directions_angstrom=directions,
        metric_diagonal=np.maximum(np.abs(metric), 1.0e-12),
        sonic_definition=definition if coordinate_kind == "sonic" else None,
        pes_exploration=bool(pes_exploration),
        retained_group=(str(retained_group).strip().upper() if pes_exploration else ""),
        rank_reduced_labels=tuple(dropped) if coordinate_kind == "sonic" else (),
    )


def coordinate_model_from_typed_onic(
    runtime,
    *,
    metric_diagonal: Sequence[float] | None = None,
) -> OptimizerCoordinateModel:
    """Build the canonical optimizer model from a compiled typed ONIC runtime."""

    from .typed_onic import TypedOnicRuntime

    if not isinstance(runtime, TypedOnicRuntime):
        raise TypeError("typed ONIC optimizer construction requires a TypedOnicRuntime")
    reference = np.asarray(runtime.definition.reference_coordinates_angstrom, dtype=float)
    evaluation = runtime.evaluate(reference)
    cartesian_from_q = cartesian_from_internal_jacobian(
        evaluation.b_matrix.to_dense(),
        rcond=1.0e-8,
    )
    directions = np.asarray(cartesian_from_q.T, dtype=float)
    metric = (
        np.diag((directions * ANGSTROM_TO_BOHR) @ (directions * ANGSTROM_TO_BOHR).T)
        if metric_diagonal is None
        else np.asarray(metric_diagonal, dtype=float).reshape(-1)
    )
    if metric.shape != (runtime.coordinate_count,):
        raise ValueError("typed ONIC metric diagonal length must match coordinate count")
    return OptimizerCoordinateModel(
        kind="typed_onic",
        labels=runtime.coordinate_identifiers,
        directions_angstrom=directions,
        metric_diagonal=np.maximum(np.abs(metric), 1.0e-12),
        reference_values=evaluation.values,
        typed_onic_runtime=runtime,
    )


def _independent_coordinate_rows(
    directions: np.ndarray,
    labels: Sequence[str],
    *,
    relative_tolerance: float = 1.0e-10,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Select a deterministic full-rank SONIC row basis at the reference.

    A symmetry-specialized or planar coordinate definition can contain rows
    that become linearly dependent at one geometry even though the symbolic
    coordinate catalog is valid in general.  Such rows cannot be realized by
    a Cartesian backtransform at that point.  A maximum-residual greedy QR
    equivalent retains the best-conditioned independent subset and reports
    the removed labels for diagnostics.
    """

    matrix = np.asarray(directions, dtype=float)
    if matrix.ndim != 2 or len(labels) != matrix.shape[0]:
        raise ValueError("coordinate directions and labels must have matching rows")
    if matrix.shape[0] <= 1:
        return tuple(range(matrix.shape[0])), ()
    row_scale = max(float(np.max(np.linalg.norm(matrix, axis=1), initial=0.0)), 1.0)
    threshold = float(relative_tolerance) * row_scale
    selected: list[int] = []
    remaining = set(range(matrix.shape[0]))
    while remaining:
        best_index = None
        best_residual = -1.0
        basis = matrix[selected, :] if selected else None
        for index in sorted(remaining):
            row = matrix[index]
            if basis is None:
                residual = row
            else:
                residual = row - row @ np.linalg.pinv(basis, rcond=relative_tolerance) @ basis
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > best_residual:
                best_residual = residual_norm
                best_index = index
        if best_index is None or best_residual <= threshold:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    selected.sort()
    dropped = tuple(str(labels[index]) for index in range(len(labels)) if index not in selected)
    return tuple(selected), dropped


def _validate_transition_state_task_model(model: OptimizerCoordinateModel) -> None:
    """Require the complete, unprojected frozen SMITH chart for a SONIC TS.

    A transition-state search is a full vibrational-space problem.  LINK may
    consume the chart selected by ORACLE/SMITH, but it must not silently turn
    that chart into a totally symmetric subset or another projected variable
    space.  Cartesian and typed-ONIC models have their own explicit contracts
    and are unaffected by this SONIC-specific gate.
    """

    if model.kind != "sonic":
        return
    definition = model.sonic_definition
    if definition is None:
        raise ValueError("a SONIC transition-state model needs a frozen SMITH definition")
    expected_labels = tuple(gic.name or gic.identifier for gic in definition.gics)
    if model.sonic_from_coordinates is not None or tuple(model.labels) != expected_labels:
        raise ValueError(
            "SONIC transition-state optimization requires the complete frozen SMITH "
            "coordinate order without projection"
        )
    directions = np.asarray(model.directions_angstrom, dtype=float)
    if np.linalg.matrix_rank(directions, tol=1.0e-10) != len(expected_labels):
        raise ValueError("the complete frozen SMITH transition-state chart is rank deficient")


def _lifecycle_task_coordinate_model(
    complete_model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None = None,
) -> tuple[OptimizerCoordinateModel, tuple[str, ...]]:
    """Derive LINK's nonlinear task chart from one complete lifecycle chart.

    ORACLE and SMITH own the complete exact-rank chart retained by the lifecycle
    controller.  LINK realizes only the task variables: every row for a TS or
    C1 minimum, and the totally symmetric rows for a non-C1 minimum.  Building
    a compact definition is essential because nonlinear back-transformation
    must not impose omitted irreps as frozen coordinate equations.
    """

    if settings.stationary_point == "transition_state":
        if (
            complete_model.kind != "sonic"
            or complete_model.sonic_definition is None
            or not complete_model.sonic_definition.symmetrize
            or str(complete_model.sonic_definition.point_group).strip().upper()
            in {"", "C1", "UNKNOWN"}
        ):
            return complete_model, ()
        # Match Gaussian ReadAllGIC: all SONIC rows remain in the chart, but
        # only the totally symmetric block is optimized.  The other rows are
        # retained as frozen rank-closing coordinates during realization.
        return _lifecycle_total_symmetric_coordinate_model(
            complete_model,
            settings,
            coordinates_angstrom,
            xyzin_path=xyzin_path,
            preserve_complete_definition=True,
        )
    if settings.stationary_point != "minimum":
        raise ValueError("lifecycle task charts require an explicit minimum or transition state")
    if complete_model.kind != "sonic" or complete_model.sonic_definition is None:
        return complete_model, ()
    if complete_model.sonic_from_coordinates is not None:
        raise ValueError("chart lifecycle does not accept a projected optimizer variable model")

    definition = complete_model.sonic_definition
    point_group = str(definition.point_group or "C1").strip().upper()
    if not definition.symmetrize or point_group in {"", "C1", "UNKNOWN"}:
        return complete_model, ()
    return _lifecycle_total_symmetric_coordinate_model(
        complete_model,
        settings,
        coordinates_angstrom,
        xyzin_path=xyzin_path,
    )


def _lifecycle_total_symmetric_coordinate_model(
    complete_model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None,
    preserve_complete_definition: bool = False,
) -> tuple[OptimizerCoordinateModel, tuple[str, ...]]:
    """Build and audit the exact totally symmetric vibrational subspace.

    SMITH labels ordinary symmetry-adapted rows explicitly, but deliberately
    leaves some periodic torsions ``UNASSIGNED`` so their scalar periodic
    identities remain intact.  Counting only explicit irrep labels can then
    underfill the physical totally symmetric space (Baker TS11), while a
    mislabeled degenerate component can overfill it (Baker TS15).  Derive the
    required rank from the frozen Cartesian group projector for every point
    group, project the eligible rows, and require exact agreement.
    """

    from matrix_chem import read_molecular_symmetry
    from matrix_smith.symmetry_labels import is_total_symmetric_irrep

    if xyzin_path is None:
        raise ValueError("a frozen xyzin path is required to audit molecular symmetry")
    definition = complete_model.sonic_definition
    symmetry = read_molecular_symmetry(Path(xyzin_path))
    if str(symmetry.point_group).strip().upper() != str(definition.point_group).strip().upper():
        raise RuntimeError(
            "LINK symmetry-subspace audit found inconsistent frozen point groups: "
            f"Cartesian={symmetry.point_group}, SONIC={definition.point_group}"
        )
    natoms = np.asarray(coordinates_angstrom, dtype=float).shape[0]
    projector = _totally_symmetric_cartesian_projector(symmetry, natoms=natoms)

    complete_cartesian_from_q = np.asarray(complete_model.directions_angstrom, dtype=float).T
    vibrational_basis = np.linalg.qr(complete_cartesian_from_q, mode="reduced")[0]
    reduced_projector = vibrational_basis.T @ projector @ vibrational_basis
    reduced_projector = 0.5 * (reduced_projector + reduced_projector.T)
    eigenvalues = np.linalg.eigvalsh(reduced_projector)
    eigenvalue_residual = max(
        (min(abs(float(value)), abs(float(value) - 1.0)) for value in eigenvalues),
        default=0.0,
    )
    covariance_residual = float(
        np.linalg.norm(
            projector @ vibrational_basis - vibrational_basis @ reduced_projector,
            ord=2,
        )
    )
    if eigenvalue_residual > 1.0e-7 or covariance_residual > 1.0e-7:
        raise RuntimeError(
            "LINK complete SONIC tangent is not invariant under the frozen point group "
            f"(eigenvalue residual={eigenvalue_residual:.3e}, "
            f"covariance residual={covariance_residual:.3e})"
        )
    expected_rank = int(np.count_nonzero(eigenvalues > 0.5))

    identifiers = tuple(gic.identifier for gic in definition.gics)
    names = tuple(gic.name for gic in definition.gics)
    source_labels = tuple(complete_model.sonic_labels or complete_model.labels)
    source_indices = tuple(_coordinate_index(label, identifiers, names) for label in source_labels)
    candidate_positions = tuple(
        position
        for position, index in enumerate(source_indices)
        if is_total_symmetric_irrep(definition.point_group, definition.gics[index].irrep)
        or str(definition.gics[index].irrep).strip().upper() == "UNASSIGNED"
    )
    if not candidate_positions:
        raise RuntimeError("LINK lifecycle chart has no total-symmetry candidates")
    candidate_directions = complete_cartesian_from_q[:, candidate_positions]
    projected_directions = projector @ candidate_directions
    coefficient_projector = (
        np.linalg.pinv(candidate_directions, rcond=1.0e-10) @ projected_directions
    )
    coefficient_projector[np.abs(coefficient_projector) < 1.0e-7] = 0.0
    realization_residual = float(
        np.linalg.norm(
            candidate_directions @ coefficient_projector - projected_directions,
            ord=2,
        )
    )
    if realization_residual > 1.0e-7:
        raise RuntimeError(
            "LINK total-symmetry candidates are not closed under the frozen projector "
            f"(residual={realization_residual:.3e})"
        )

    basis_columns: list[np.ndarray] = []
    anchor_positions: list[int] = []
    for candidate_index in range(len(candidate_positions)):
        vector = np.asarray(coefficient_projector[:, candidate_index], dtype=float).copy()
        for basis in basis_columns:
            vector -= basis * float(basis @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-8:
            continue
        vector /= norm
        if vector[candidate_index] < 0.0:
            vector *= -1.0
        vector[np.abs(vector) < 1.0e-12] = 0.0
        basis_columns.append(vector)
        anchor_positions.append(candidate_index)
    actual_rank = len(basis_columns)
    if actual_rank != expected_rank:
        raise RuntimeError(
            "LINK total-symmetric coordinate count disagrees with the Cartesian "
            f"vibrational projector for {definition.point_group}: "
            f"coordinates={actual_rank}, expected={expected_rank}"
        )
    transform = np.column_stack(basis_columns)
    runtime_directions = candidate_directions @ transform
    if np.linalg.matrix_rank(runtime_directions, tol=1.0e-9) != expected_rank:
        raise RuntimeError("LINK total-symmetric runtime tangent is rank deficient")
    symmetry_residual = float(
        np.linalg.norm((np.eye(projector.shape[0]) - projector) @ runtime_directions, ord=2)
    )
    if symmetry_residual > 1.0e-7:
        raise RuntimeError(
            "LINK runtime tangent contains non-totally-symmetric displacement "
            f"(residual={symmetry_residual:.3e})"
        )

    selected_candidate_positions: list[int] = []
    selection_only = True
    for column in transform.T:
        nonzero = np.flatnonzero(np.abs(column) > 1.0e-8)
        if len(nonzero) != 1 or abs(abs(float(column[nonzero[0]])) - 1.0) > 1.0e-8:
            selection_only = False
            break
        selected_candidate_positions.append(int(nonzero[0]))
    selected_source_positions = tuple(
        candidate_positions[position] for position in selected_candidate_positions
    )
    if selection_only:
        if len(selected_source_positions) == len(complete_model.labels):
            return complete_model, ()
        return _compact_lifecycle_coordinate_model(
            complete_model,
            settings,
            coordinates_angstrom,
            active_positions=selected_source_positions,
            source_indices=source_indices,
            source_labels=source_labels,
            preserve_complete_definition=preserve_complete_definition,
        )

    candidate_labels = tuple(source_labels[position] for position in candidate_positions)
    active_labels: list[str] = []
    for number, (anchor, column) in enumerate(
        zip(anchor_positions, transform.T, strict=True),
        start=1,
    ):
        members = tuple(
            candidate_labels[index] for index in np.flatnonzero(np.abs(column) > 1.0e-8)
        )
        if len(members) == 1:
            active_labels.append(members[0])
        else:
            active_labels.append(f"TSym{number:03d}[{'+'.join(members)}]")
    active_anchor_set = frozenset(anchor_positions)
    inactive_labels = tuple(
        source_labels[position]
        for position in range(len(source_labels))
        if position not in frozenset(candidate_positions)
    ) + tuple(
        f"NonTSym[{candidate_labels[position]}]"
        for position in range(len(candidate_positions))
        if position not in active_anchor_set
    )
    directions = runtime_directions.T
    runtime_model = OptimizerCoordinateModel(
        kind="sonic",
        labels=tuple(active_labels),
        directions_angstrom=directions,
        metric_diagonal=np.maximum(
            np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1),
            1.0e-12,
        ),
        sonic_labels=candidate_labels,
        sonic_from_coordinates=transform,
        sonic_definition=(
            definition
            if preserve_complete_definition
            else replace(
                definition,
                gics=tuple(
                    definition.gics[source_indices[position]] for position in candidate_positions
                ),
                rank=len(candidate_positions),
                target_rank=len(candidate_positions),
            )
        ),
        pes_exploration=complete_model.pes_exploration,
        retained_group=complete_model.retained_group,
    )
    return runtime_model, inactive_labels


def _totally_symmetric_cartesian_projector(
    symmetry: object,
    *,
    natoms: int,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the orthogonal projector onto vectors fixed by every operation.

    Finite molecular groups permit the familiar arithmetic group average.
    ORACLE's frozen contracts for linear ``Cinfv`` and ``Dinfh`` molecules,
    however, contain a finite diagnostic sample of the continuous group and
    are intentionally not closed under multiplication.  The common invariant
    subspace is instead the intersection of the null spaces of ``R(g)-I``.
    This construction is valid for closed finite groups and sampled continuous
    groups alike and does not depend on how often an operation is sampled.
    """

    from matrix_chem import cartesian_operation_matrix

    operations = tuple(getattr(symmetry, "operations", ()))
    if not operations:
        raise RuntimeError("LINK symmetry-subspace audit needs frozen symmetry operations")
    dimension = 3 * int(natoms)
    identity = np.eye(dimension, dtype=float)
    representations = []
    for operation in operations:
        representation = cartesian_operation_matrix(
            np.asarray(operation.rotation, dtype=float),
            tuple(int(atom) - 1 for atom in operation.permutation),
            natoms=natoms,
        )
        orthogonality_residual = float(
            np.linalg.norm(representation.T @ representation - identity, ord=2)
        )
        if orthogonality_residual > 1.0e-8:
            raise RuntimeError(
                "LINK frozen Cartesian symmetry representation is not orthogonal "
                f"(residual={orthogonality_residual:.3e})"
            )
        representations.append(representation)
    constraints = np.concatenate(
        [representation - identity for representation in representations],
        axis=0,
    )
    _left, singular_values, right_transpose = np.linalg.svd(
        constraints,
        full_matrices=True,
    )
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    threshold = max(float(tolerance) * scale, np.finfo(float).eps * max(constraints.shape) * scale)
    rank = int(np.count_nonzero(singular_values > threshold))
    invariant_basis = right_transpose[rank:].T
    projector = invariant_basis @ invariant_basis.T
    projector = 0.5 * (projector + projector.T)
    idempotency_residual = float(np.linalg.norm(projector @ projector - projector, ord=2))
    invariance_residual = max(
        float(np.linalg.norm((representation - identity) @ projector, ord=2))
        for representation in representations
    )
    if idempotency_residual > 1.0e-8 or invariance_residual > 1.0e-8:
        raise RuntimeError(
            "LINK could not construct the frozen totally symmetric Cartesian projector "
            f"(idempotency={idempotency_residual:.3e}, invariance={invariance_residual:.3e})"
        )
    return projector


def _cartesian_directions_are_totally_symmetric(
    directions_angstrom: np.ndarray,
    symmetry: object,
    *,
    tolerance: float = 1.0e-7,
) -> bool:
    """Return whether every Cartesian tangent lies in the invariant subspace."""

    directions = np.asarray(directions_angstrom, dtype=float)
    if directions.ndim != 2 or directions.shape[1] % 3:
        raise ValueError("Cartesian coordinate directions must have shape ncoord x 3N")
    natoms = directions.shape[1] // 3
    if not tuple(getattr(symmetry, "operations", ())):
        return False
    projector = _totally_symmetric_cartesian_projector(symmetry, natoms=natoms)
    residual = directions - directions @ projector.T
    scale = max(float(np.linalg.norm(directions, ord=2)), 1.0)
    return bool(float(np.linalg.norm(residual, ord=2)) <= float(tolerance) * scale)


def _compact_lifecycle_coordinate_model(
    complete_model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    coordinates_angstrom: np.ndarray,
    *,
    active_positions: tuple[int, ...],
    source_indices: tuple[int, ...],
    source_labels: tuple[str, ...],
    preserve_complete_definition: bool = False,
) -> tuple[OptimizerCoordinateModel, tuple[str, ...]]:
    """Build one exact-rank nonlinear LINK view from selected SMITH rows."""

    from matrix_smith import build_gic_b_matrix

    definition = complete_model.sonic_definition
    selected_indices = tuple(source_indices[position] for position in active_positions)
    selected_gics = tuple(definition.gics[index] for index in selected_indices)
    runtime_definition = (
        definition
        if preserve_complete_definition
        else replace(
            definition,
            gics=selected_gics,
            rank=len(selected_gics),
            target_rank=len(selected_gics),
        )
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(
            replace(
                definition,
                gics=selected_gics,
                rank=len(selected_gics),
                target_rank=len(selected_gics),
            ),
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
            parallel_workers=settings.coordinate_parallel_workers,
        ).rows,
        dtype=float,
    )
    if b_matrix.shape[0] != len(selected_gics) or np.linalg.matrix_rank(
        b_matrix, tol=1.0e-9
    ) != len(selected_gics):
        raise RuntimeError("LINK minimum lifecycle task chart is rank deficient")
    directions = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8).T
    labels = tuple(complete_model.labels[position] for position in active_positions)
    sonic_labels = tuple(source_labels[position] for position in active_positions)
    active_set = frozenset(active_positions)
    inactive_labels = tuple(
        label
        for position, label in enumerate(complete_model.labels)
        if position not in active_set
    )
    runtime_model = OptimizerCoordinateModel(
        kind="sonic",
        labels=labels,
        directions_angstrom=directions,
        metric_diagonal=np.maximum(
            np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1),
            1.0e-12,
        ),
        sonic_labels=sonic_labels,
        sonic_definition=runtime_definition,
        pes_exploration=complete_model.pes_exploration,
        retained_group=complete_model.retained_group,
    )
    return runtime_model, inactive_labels


def _lifecycle_runtime_initial_hessian(
    complete_model: OptimizerCoordinateModel,
    runtime_model: OptimizerCoordinateModel,
    initial_hessian: np.ndarray | None,
) -> tuple[np.ndarray | None, bool]:
    """Transport a complete-chart seed Hessian into LINK's runtime tangent."""

    if initial_hessian is None or runtime_model is complete_model:
        return initial_hessian, False
    matrix = np.asarray(initial_hessian, dtype=float)
    runtime_shape = (len(runtime_model.labels), len(runtime_model.labels))
    if matrix.shape == runtime_shape:
        return matrix, False
    complete_shape = (len(complete_model.labels), len(complete_model.labels))
    if matrix.shape != complete_shape:
        raise ValueError(
            "lifecycle initial Hessian matches neither complete nor runtime chart"
        )
    complete_cartesian_from_q = np.asarray(
        complete_model.directions_angstrom,
        dtype=float,
    ).T
    complete_b_matrix = internal_from_cartesian_jacobian(
        complete_cartesian_from_q,
    )
    runtime_cartesian_from_q = np.asarray(
        runtime_model.directions_angstrom,
        dtype=float,
    ).T
    tangent_map = complete_b_matrix @ runtime_cartesian_from_q
    if np.linalg.matrix_rank(tangent_map, tol=1.0e-10) != len(runtime_model.labels):
        raise RuntimeError("lifecycle initial-Hessian tangent transport is rank deficient")
    transported = transport_internal_hessian(
        matrix,
        complete_b_matrix,
        runtime_cartesian_from_q,
    )
    return np.asarray(transported, dtype=float), True


def _apply_chart_lifecycle_transition(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    hessian: np.ndarray,
    lifecycle_result: ChartLifecycleResult,
    settings: OptimizerSettings,
) -> tuple[
    OptimizerCoordinateModel,
    OptimizerEvaluation,
    np.ndarray,
    np.ndarray,
    str,
    tuple[str, ...],
]:
    """Rebase optimizer tensors after a validated ORACLE/SMITH transition."""

    from matrix_numerics import singular_spectrum

    from .chart_lifecycle import ChartLifecycleError

    candidate = lifecycle_result.current.candidate
    complete_model = candidate.coordinate_model
    if not isinstance(complete_model, OptimizerCoordinateModel):
        raise ChartLifecycleError("SMITH candidate has no LINK optimizer coordinate model")
    new_model, inactive_labels = _lifecycle_task_coordinate_model(
        complete_model,
        settings,
        current.coordinates_angstrom,
        xyzin_path=candidate.source_xyzin_path or service.xyzin_path,
    )
    old_cartesian_from_q = service.coordinate_directions(
        current.coordinates_angstrom
    ).T
    old_b_matrix = internal_from_cartesian_jacobian(old_cartesian_from_q)
    new_cartesian_from_q = np.asarray(new_model.directions_angstrom, dtype=float).T
    tangent_map = old_b_matrix @ new_cartesian_from_q
    spectrum = singular_spectrum(tangent_map, absolute_tolerance=1.0e-10)
    subspace_certified, subspace_status = _chart_tangent_subspace_certificate(
        old_cartesian_from_q,
        new_cartesian_from_q,
        absolute_tolerance=1.0e-10,
    )
    transport_valid = bool(
        lifecycle_result.previous_chart_valid_for_hessian_transport
        and subspace_certified
        and spectrum.rank == len(new_model.labels)
        and np.isfinite(spectrum.condition_number)
        and spectrum.condition_number <= settings.max_hessian_condition
    )
    if transport_valid:
        next_hessian = transport_internal_hessian(
            hessian,
            old_b_matrix,
            new_cartesian_from_q,
        )
        hessian_status = "chart_epoch_hessian_transported"
    else:
        if candidate.source_xyzin_path is None:
            raise ChartLifecycleError(
                "chart tangent transport failed and no fresh ORACLE artifact can seed Hessian"
            )
        try:
            next_hessian = _initial_optimizer_hessian(
                new_model,
                settings,
                initial_hessian=None,
                atoms=service.atoms,
                coordinates_angstrom=current.coordinates_angstrom,
                xyzin_path=candidate.source_xyzin_path,
            )
        except (ImportError, OSError, ValueError, np.linalg.LinAlgError) as exc:
            raise ChartLifecycleError(
                "chart tangent transport and canonical Hessian rebuild both failed"
            ) from exc
        transport_rejection = (
            subspace_status
            if lifecycle_result.previous_chart_valid_for_hessian_transport
            else "previous_chart_numerically_invalid"
        )
        hessian_status = (
            "chart_epoch_canonical_chemical_hessian_rebuilt_" + transport_rejection
        )
    next_hessian = 0.5 * (
        np.asarray(next_hessian, dtype=float) + np.asarray(next_hessian, dtype=float).T
    )
    expected_shape = (len(new_model.labels), len(new_model.labels))
    if next_hessian.shape != expected_shape or not np.all(np.isfinite(next_hessian)):
        raise ChartLifecycleError("replacement chart produced an invalid optimizer Hessian")
    service.install_coordinate_model(new_model, current.coordinates_angstrom)
    next_q = np.zeros(len(new_model.labels), dtype=float)
    next_current = replace(
        current,
        q=next_q.copy(),
        cache_hit=False,
        chart_epoch=service.cache.chart_epoch,
    )
    service.cache.add(next_current)
    service.initialize_coordinate_projector(next_q, current.coordinates_angstrom)
    return (
        new_model,
        next_current,
        next_q,
        next_hessian,
        hessian_status,
        inactive_labels,
    )


def _chart_tangent_subspace_certificate(
    old_cartesian_from_q: np.ndarray,
    new_cartesian_from_q: np.ndarray,
    *,
    absolute_tolerance: float,
) -> tuple[bool, str]:
    """Certify that congruence preserves the physical optimizer subspace.

    A coordinate reparameterization may rescale or mix basis vectors by an
    arbitrary nonsingular matrix, so raw singular values of that matrix do not
    certify a physical Hessian transport.  Congruence is valid only when every
    new tangent direction belongs to the old Cartesian tangent subspace.  If a
    chart was rebuilt because an old coordinate became singular, this exact
    containment test rejects the stale tangent and activates the canonical
    chemical seed already required by the LINK protocol.
    """

    old = np.asarray(old_cartesian_from_q, dtype=float)
    new = np.asarray(new_cartesian_from_q, dtype=float)
    if old.ndim != 2 or new.ndim != 2 or old.shape[0] != new.shape[0]:
        return False, "incompatible_tangent_shapes"
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        return False, "nonfinite_tangent"
    old_u, old_s, _old_vt = np.linalg.svd(old, full_matrices=False)
    new_u, new_s, _new_vt = np.linalg.svd(new, full_matrices=False)
    old_rank = int(np.count_nonzero(old_s > absolute_tolerance))
    new_rank = int(np.count_nonzero(new_s > absolute_tolerance))
    if old_rank != old.shape[1] or new_rank != new.shape[1]:
        return False, "rank_deficient_tangent"
    old_basis = old_u[:, :old_rank]
    new_basis = new_u[:, :new_rank]
    residual = new_basis - old_basis @ (old_basis.T @ new_basis)
    residual_norm = float(np.linalg.norm(residual, ord=2)) if residual.size else 0.0
    numerical_tolerance = max(
        1.0e-8,
        100.0 * np.finfo(float).eps * max(old.shape + new.shape),
    )
    if not np.isfinite(residual_norm) or residual_norm > numerical_tolerance:
        return False, "incompatible_tangent_subspace"
    return True, "certified_tangent_subspace"


def optimize_geometry(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: OptimizerCoordinateModel | None = None,
    coordinate_kind: str = "cartesian",
    coordinates: Sequence[str] = (),
    engine_command: str = "",
    backend: QMScanBackend | None = None,
    settings: OptimizerSettings | None = None,
    timeout: float | None = None,
    initial_hessian: np.ndarray | None = None,
    initial_hessian_source: str = "metric-diagonal",
    initial_symmetry_breaking_mode_displacement: InitialSymmetryBreakingModeDisplacement | None = None,
    refine_initial_qm_hessian_with_b_prime: bool = False,
    initial_hessian_b_prime_workers: int = 0,
    irc_target_geometries: Sequence[np.ndarray] = (),
    representation_request=None,
    periodic_contracts=(),
    require_frozen_symmetry_contract: bool = False,
    chart_lifecycle_controller: ChartLifecycleController | None = None,
    frozen_chart_reference: FrozenChartReference | None = None,
) -> OptimizerResult:
    settings = settings or OptimizerSettings()
    from .protocol_manifest import (
        load_link_optimizer_protocol,
        runtime_method_payload,
        validate_runtime_optimizer_settings,
        write_runtime_method_manifest,
    )

    protocol = load_link_optimizer_protocol()
    if chart_lifecycle_controller is not None:
        from .chart_lifecycle import ChartLifecycleController

        if not isinstance(chart_lifecycle_controller, ChartLifecycleController):
            raise TypeError("chart_lifecycle_controller must be a ChartLifecycleController")
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen_chart_replay = None
    if frozen_chart_reference is not None:
        from .frozen_chart_replay import FrozenChartReference

        if not isinstance(frozen_chart_reference, FrozenChartReference):
            raise TypeError("frozen_chart_reference must be a FrozenChartReference")
        if chart_lifecycle_controller is not None:
            raise ValueError("immutable chart replay forbids dynamic chart lifecycle")
        expected_regime = {
            "minimum": "MINIMUM",
            "transition_state": "TRANSITION_STATE",
        }.get(settings.stationary_point)
        if expected_regime is None:
            raise ValueError("immutable chart replay requires an explicit stationary-point task")
        if frozen_chart_reference.task_regime != expected_regime:
            raise ValueError(
                "immutable chart task regime contradicts optimizer stationary-point task"
            )
        replay_audit = frozen_chart_reference.validate(xyzin_path)
        replay_audit.write(root / "frozen_chart_replay.json")
        frozen_chart_replay = replay_audit.to_json()
    complete_model = coordinate_model or coordinate_model_from_xyzin(
        xyzin_path,
        kind=coordinate_kind,
        coordinates=coordinates,
        task_regime=(
            "TRANSITION_STATE"
            if settings.stationary_point == "transition_state"
            else "MINIMUM"
            if settings.stationary_point == "minimum"
            else None
        ),
    )
    if settings.stationary_point == "transition_state":
        _validate_transition_state_task_model(complete_model)
    if frozen_chart_reference is not None:
        definition = complete_model.sonic_definition
        expected_labels = (
            ()
            if definition is None
            else tuple(gic.name or gic.identifier for gic in definition.gics)
        )
        if complete_model.kind != "sonic" or tuple(complete_model.labels) != expected_labels:
            raise ValueError(
                "immutable transition-state chart replay requires the complete frozen "
                "SMITH coordinate order"
            )
        if len(complete_model.labels) != frozen_chart_reference.target_rank:
            raise ValueError(
                "immutable chart runtime rank differs from the certified frozen rank"
            )
    if settings.stationary_point == "automatic":
        if complete_model.kind != "sonic" or complete_model.sonic_definition is None:
            raise ValueError(
                "automatic stationary-point classification requires a SONIC coordinate model"
            )
        from matrix_smith.symmetry_labels import is_total_symmetric_irrep

        definition = complete_model.sonic_definition
        total_active = all(
            is_total_symmetric_irrep(definition.point_group, gic.irrep)
            for gic in definition.gics
            if (gic.name or gic.identifier) in complete_model.labels
        )
        reduced_model = len(complete_model.labels) < len(definition.gics)
        if not (total_active and reduced_model):
            raise ValueError(
                "automatic SONIC protocol requires a reduced totally symmetric optimization "
                "model; classify a full-space transition state explicitly"
            )
        # Optimize the stationary point in the totally symmetric subspace.
        # The subsequent complete SONIC Hessian performs the automatic TS/minimum
        # classification from the non-totally-symmetric curvatures.
        settings = replace(
            settings,
            stationary_point="minimum",
            adaptive_fd_mode=True,
            fd_totally_symmetric_only=True,
            prefer_analytic_gradient=False,
        )
    validate_runtime_optimizer_settings(settings)
    initial_geometry = read_xyzin_geometry(Path(xyzin_path))
    lifecycle_inactive_labels: tuple[str, ...] = ()
    lifecycle_initial_hessian_transported = False
    if chart_lifecycle_controller is not None:
        chart_lifecycle_controller.validate_initial_chart(
            tuple(initial_geometry.atoms),
            np.asarray(initial_geometry.coordinates_angstrom, dtype=float),
            complete_model,
            task_regime=(
                "TRANSITION_STATE"
                if settings.stationary_point == "transition_state"
                else "MINIMUM"
            ),
        )
    if chart_lifecycle_controller is not None or frozen_chart_reference is not None:
        # The immutable replay certifies the complete frozen SMITH chart, but
        # Gaussian ReadAllGIC optimizes only the totally symmetric block when
        # molecular symmetry is active.  Keep the complete chart as the
        # rank/identity contract and derive the same runtime task subspace for
        # both dynamic lifecycle and immutable replay execution.
        model, lifecycle_inactive_labels = _lifecycle_task_coordinate_model(
            complete_model,
            settings,
            np.asarray(initial_geometry.coordinates_angstrom, dtype=float),
            xyzin_path=xyzin_path,
        )
        initial_hessian, lifecycle_initial_hessian_transported = (
            _lifecycle_runtime_initial_hessian(
                complete_model,
                model,
                initial_hessian,
            )
        )
    else:
        model = complete_model
    periodic_pes_adapter = None
    if representation_request is not None:
        from .representation_adapter import validate_link_representation
        from matrix_smith import RepresentationRequest

        validated_request = validate_link_representation(representation_request)
        if not isinstance(validated_request, RepresentationRequest):
            raise TypeError("representation_request must be a matrix_smith request")
        if validated_request.mode == "PERIODIC_EMBEDDING":
            from .periodic_pes import PeriodicPESAdapter

            if len(tuple(periodic_contracts)) != len(model.labels):
                raise ValueError("periodic contracts must cover the runtime optimizer model")
            periodic_pes_adapter = PeriodicPESAdapter(tuple(periodic_contracts))
        elif validated_request.mode != "SCALAR":
            raise ValueError(
                "optimize_geometry accepts SCALAR or PERIODIC_EMBEDDING internal representations"
            )
    service = GeometryEvaluationService(
        xyzin_path=xyzin_path,
        run_dir=run_dir,
        coordinate_model=model,
        engine_command=engine_command,
        backend=backend,
        timeout=timeout,
        settings=settings,
        pes_exploration_policy=(
            PESExplorationPolicy(retained_group=model.retained_group or "C1")
            if model.pes_exploration
            else None
        ),
        periodic_pes_adapter=periodic_pes_adapter,
    )
    # Backend-driven QM runs are governed by the immutable ORACLE--SMITH
    # symmetry contract. External evaluator tests without a persisted backend
    # manifest remain outside that deployment contract.
    if require_frozen_symmetry_contract and model.pes_exploration:
        raise ValueError("PES exploration cannot request a frozen exploitation symmetry contract")
    if not model.pes_exploration and (backend is not None or require_frozen_symmetry_contract):
        service.assert_frozen_symmetry_contract()
    if settings.fd_totally_symmetric_only:
        service.assert_totally_symmetric_active_sonics()
    runtime_method_manifest_path = write_runtime_method_manifest(
        root / "link_runtime_method_manifest.json",
        runtime_method_payload(
            protocol,
            backend=backend,
            engine_command=engine_command,
            coordinate_kind=model.kind,
            stationary_point=settings.stationary_point,
            prefer_analytic_gradient=settings.prefer_analytic_gradient,
            optimizer_settings=settings,
            symmetry_verification=(
                "instantaneous_ORACLE_reperception_without_parent_group_constraint"
                if model.pes_exploration
                else "validated_frozen_initial_group_and_per_iteration_gradient_projection"
                if backend is not None or require_frozen_symmetry_contract
                else "external_driver_outside_backend_manifest_contract"
            ),
        ),
    )
    trace_path = root / "optimizer_trace.jsonl"
    trajectory_path = root / "optimizer_trajectory.xyz"
    summary_path = root / "optimizer_summary.json"
    final_hessian_path = root / "optimizer_hessian.json" if settings.compute_final_hessian else None
    initial_coordinates_for_result = np.asarray(
        initial_geometry.coordinates_angstrom, dtype=float
    ).copy()
    q = np.zeros(len(model.labels), dtype=float)
    hessian, selected_hessian_source = _initial_optimizer_hessian_selection(
        model,
        settings,
        initial_hessian=initial_hessian,
        atoms=tuple(initial_geometry.atoms),
        coordinates_angstrom=initial_geometry.coordinates_angstrom,
        xyzin_path=xyzin_path,
    )
    # Keep the physical/chemical Hessian model intact.  Positive-definite
    # flooring (minimum) and index conditioning (transition state) belong to
    # the local step model assembled by the trust-region solver; applying
    # them here would overwrite physically meaningful couplings before the
    # first secant update.
    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    transition_mode_reference: TransitionModeReference | None = None
    transition_mode_overlaps: list[float] = []
    hessian_source = (
        initial_hessian_source if initial_hessian is not None else selected_hessian_source
    )
    if lifecycle_initial_hessian_transported:
        hessian_source += " + complete-chart-to-runtime-tangent-transport"
    trust_radius = (
        GDV_TS_INITIAL_TRUST_RADIUS
        if settings.stationary_point == "transition_state"
        else float(settings.trust_radius)
    )
    current_damping = 0.0
    iterations: list[OptimizerIteration] = []
    chart_lifecycle_events = []
    trace_path.write_text("", encoding="utf-8")
    trajectory_path.write_text("", encoding="utf-8")

    current = service.evaluate(
        q,
        tag="initial",
        requested_properties=(
            ("energy", "gradient")
            if settings.prefer_analytic_gradient
            else ("energy",)
        ),
    )
    service.accept_electronic_state(current)
    service.initialize_coordinate_projector(q, current.coordinates_angstrom)
    energy_noise_estimate = 0.0
    energy_noise_energies: tuple[float, ...] = (current.energy_hartree,)
    if settings.energy_noise_samples:
        energy_noise_estimate, energy_noise_energies = service.estimate_energy_noise(
            q,
            reference=current,
            samples=settings.energy_noise_samples,
        )
        settings = replace(settings, energy_noise=max(settings.energy_noise, energy_noise_estimate))
    _append_xyz(trajectory_path, service.atoms, current.coordinates_angstrom, "initial")
    gradient, fd_info = _gradient_in_coordinate_space(
        service,
        current,
        q,
        hessian,
        settings,
        iteration=0,
        previous_gradient=None,
        selective_disabled=False,
        two_sided_latched=False,
    )
    if refine_initial_qm_hessian_with_b_prime:
        correction = _initial_qm_b_prime_correction(
            model,
            current.coordinates_angstrom,
            current.gradient_hartree_per_bohr,
            optimizer_gradient=gradient,
            workers=initial_hessian_b_prime_workers,
        )
        if correction is not None:
            hessian = 0.5 * (hessian + hessian.T) + correction
            hessian = 0.5 * (hessian + hessian.T)
            hessian_source += (
                " + automatic-exact-curvilinear-architect-b-prime-"
                + (
                    "from-initial-analytic-gradient"
                    if current.gradient_hartree_per_bohr is not None
                    else "from-initial-numerical-gradient"
                )
            )
    if initial_symmetry_breaking_mode_displacement is not None:
        if settings.stationary_point != "transition_state":
            raise ValueError(
                "an initial symmetry-breaking displacement requires a transition-state search"
            )
        if initial_symmetry_breaking_mode_displacement.allow_symmetry_lowering is not True:
            raise PermissionError(
                "initial symmetry-breaking displacement requires explicit consent"
            )
        source_coordinates = np.asarray(
            initial_symmetry_breaking_mode_displacement.source_coordinates_angstrom,
            dtype=float,
        )
        source_mode = np.asarray(
            initial_symmetry_breaking_mode_displacement.source_cartesian_mode,
            dtype=float,
        ).reshape(-1)
        current_coordinates = np.asarray(current.coordinates_angstrom, dtype=float)
        if source_coordinates.shape != current_coordinates.shape:
            raise ValueError("source and distorted TS geometries have different shapes")
        if source_mode.shape != (current_coordinates.size,):
            raise ValueError("source symmetry-breaking mode has the wrong dimension")
        from .transition_state_distortion import fit_pure_mode_displacement

        fit = fit_pure_mode_displacement(
            source_coordinates,
            current_coordinates,
            source_mode,
        )
        if (
            not fit.success
            or fit.rms_residual_angstrom > 1.0e-7
            or fit.relative_residual > 1.0e-4
        ):
            raise ValueError(
                "distorted TS geometry is not a pure source-mode displacement: "
                f"rigid-fit RMS {fit.rms_residual_angstrom:.9g} angstrom, "
                f"relative residual {fit.relative_residual:.9g}"
            )
        initial_index = int(
            np.count_nonzero(
                np.linalg.eigvalsh(hessian) < -settings.min_hessian_eigenvalue
            )
        )
        if initial_index != 2:
            raise ValueError(
                "explicit symmetry lowering requires the preserved exact index-two Hessian"
            )
        hessian_source += (
            " + explicit-pure-symmetry-breaking-mode-displacement"
            f"-source-mode-{initial_symmetry_breaking_mode_displacement.source_mode_index}"
            f"-amplitude-{fit.amplitude:.12g}"
            f"-rigid-fit-rms-{fit.rms_residual_angstrom:.9g}"
            "-exact-hessian-preserved-index-2"
        )
    transition_mode_reference = _initial_transition_mode_reference(
        hessian,
        service.optimizer_metric(current.coordinates_angstrom),
        settings,
        coordinate_unit_scales=service.gdv_internal_coordinate_scales(),
        coordinate_directions=service.coordinate_directions(current.coordinates_angstrom),
        coordinates_angstrom=current.coordinates_angstrom,
        reaction_directions=_transition_reaction_cartesian_directions(
            model,
            service.coordinate_directions(current.coordinates_angstrom),
        ),
    )
    local_groups = _local_coordinate_groups(model, hessian, settings)
    hessian_mask = _hessian_sparsity_mask(model, hessian, settings)
    two_sided_latched = bool(
        current.gradient_hartree_per_bohr is None
        and (
            str(fd_info.get("mode", "")) == "two-sided"
            or (
                str(fd_info.get("mode", "")) == "one-sided"
                and gradient.size
                and float(np.max(np.abs(gradient))) <= settings.fd_two_sided_switch_force
            )
        )
    )
    converged = False
    status = "max_steps"
    previous_energy = current.energy_hartree
    last_energy_change = 0.0
    last_cartesian_step_bohr = np.zeros(current.coordinates_angstrom.size, dtype=float)
    selective_disabled = False
    selective_rejection_count = 0
    optimization_history: list[tuple[OptimizerEvaluation, np.ndarray]] = [
        (current, gradient.copy())
    ]
    gdv_hessian_history: list[tuple[OptimizerEvaluation, np.ndarray]] = [
        (current, gradient.copy())
    ]
    bad_model_ratio_count = 0
    stagnation_count = 0
    stagnation_recoveries = 0
    micro_masks = service.coordinate_phase_masks("inter-intra-micro")
    micro_schedule_available = len(micro_masks) == 3
    use_micro_schedule = bool(
        settings.stationary_point == "minimum"
        and micro_schedule_available
        and settings.coordinate_schedule == "inter-intra-micro"
    )
    phase_masks = service.coordinate_phase_masks(
        "joint"
        if use_micro_schedule or settings.stationary_point == "transition_state"
        else settings.coordinate_schedule
    )
    phase_index = 0
    phase_steps = 0
    final_gradient_verification: dict[str, object] | None = None
    final_hessian_refinement_attempted = False
    transition_index_refresh_attempted = False

    for iteration in range(1, settings.max_steps + 1):
        if (
            current.gradient_hartree_per_bohr is None
            and two_sided_latched
            and str(fd_info.get("mode", "")) != "two-sided"
        ):
            # The absolute 7e-4 latch applies from the following iteration.
            # Recompute immediately even when the preceding one-sided model
            # proposed no acceptable step; otherwise a false one-sided zero can
            # keep the optimizer indefinitely on the wrong side of a minimum.
            gradient, fd_info = _gradient_in_coordinate_space(
                service,
                current,
                q,
                hessian,
                settings,
                force_explicit=True,
                iteration=iteration,
                previous_gradient=gradient,
                selective_disabled=selective_disabled,
                two_sided_latched=True,
            )
            optimization_history = [(current, gradient.copy())]
            gdv_hessian_history = [(current, gradient.copy())]
            bad_model_ratio_count = 0
            current_damping = 0.0
        phase_name, phase_mask = phase_masks[phase_index]
        phase_force = float(np.max(np.abs(gradient[phase_mask]))) if np.any(phase_mask) else 0.0
        if phase_index < len(phase_masks) - 1 and (
            phase_steps >= settings.coordinate_phase_max_steps
            or phase_force
            <= settings.coordinate_phase_gradient_factor * settings.max_force_tolerance
        ):
            phase_index += 1
            phase_steps = 0
            phase_name, phase_mask = phase_masks[phase_index]
            trust_radius = min(trust_radius, settings.trust_radius)
        convergence_gradient = _convergence_gradient(current, gradient, settings, service)
        convergence = _gaussian_like_convergence(
            settings, last_energy_change, convergence_gradient, last_cartesian_step_bohr
        )
        if phase_index == len(phase_masks) - 1 and all(convergence.values()):
            if current.gradient_hartree_per_bohr is None:
                (
                    gradient,
                    fd_info,
                    hessian,
                    settings,
                    convergence,
                    final_gradient_verification,
                    final_hessian_refinement_attempted,
                    converged,
                ) = _advance_numerical_convergence_state(
                    service,
                    current,
                    q,
                    gradient,
                    fd_info,
                    hessian,
                    settings,
                    convergence,
                    last_energy_change=last_energy_change,
                    last_cartesian_step_bohr=last_cartesian_step_bohr,
                    selective_disabled=selective_disabled,
                    final_hessian_refinement_attempted=final_hessian_refinement_attempted,
                    transition_mode_reference=transition_mode_reference,
                )
                two_sided_latched = True
                if final_gradient_verification.get("two_sided_mismatch"):
                    optimization_history = [(current, gradient.copy())]
                    gdv_hessian_history = [(current, gradient.copy())]
                    bad_model_ratio_count = 0
                    current_damping = 0.0
            else:
                final_gradient_verification = {
                    "performed": False,
                    "confirmed": True,
                    "reason": "analytic_gradient",
                }
                converged = True
            if converged:
                (
                    converged,
                    transition_index_refresh_attempted,
                    index_status,
                ) = _transition_state_index_convergence_state(
                    hessian,
                    settings,
                    refresh_attempted=transition_index_refresh_attempted,
                )
                if index_status and index_status != "transition_state_index_refresh_pending":
                    status = index_status
                    break
            if converged:
                status = (
                    "converged_transition_state"
                    if settings.stationary_point == "transition_state"
                    else "converged_gaussian"
                )
                break
        phase_gradient, phase_hessian = _restricted_phase_model(
            gradient, hessian, phase_mask, settings
        )
        if use_micro_schedule:
            proposal = _inter_intra_micro_step(
                service,
                current,
                q,
                hessian,
                gradient,
                micro_masks[0][1],
                micro_masks[1][1],
                trust_radius,
                settings,
            )
        else:
            proposal = _geometric_trust_region_step(
                service,
                current,
                q,
                phase_hessian,
                phase_gradient,
                trust_radius,
                settings,
                metric=service.optimizer_metric(current.coordinates_angstrom),
                damping=current_damping,
                transition_mode_reference=transition_mode_reference,
            )
        far_from_minimum = bool(
            settings.stationary_point == "minimum"
            and _far_from_minimum_cauchy_required(
                service,
                current,
                phase_gradient,
                phase_hessian,
                phase_mask,
                trust_radius,
                settings,
            )
        )
        if far_from_minimum and not use_micro_schedule:
            proposal = _preconditioned_cauchy_step(
                service,
                current,
                q,
                phase_hessian,
                phase_gradient,
                trust_radius,
                settings,
            )
        model_proposal = proposal
        if proposal.transition_mode_vector is not None:
            transition_mode_reference = proposal.transition_mode_vector
            transition_mode_overlaps.append(float(proposal.transition_mode_overlap or 0.0))
        gdiis_result = GDIISStepResult(
            None,
            (
                f"not_started:iteration={iteration}:start={settings.gdiis_start}"
                if settings.enable_gdiis and settings.stationary_point == "minimum"
                else "disabled_or_not_minimum"
            ),
            False,
            len(optimization_history),
            len(optimization_history),
            0,
        )
        gdiis_used = False
        if _gdiis_is_active(iteration, settings):
            transported_history = [
                (history_q, history_energy, np.where(phase_mask, history_gradient, 0.0))
                for history_q, history_energy, history_gradient in _transported_optimizer_history(
                    service, optimization_history
                )
            ]
            gdiis_result = _safeguarded_gdiis_step(
                transported_history,
                q,
                phase_gradient,
                phase_hessian,
                settings,
                metric=service.optimizer_metric(current.coordinates_angstrom),
            )
            if gdiis_result.discarded_history_size:
                optimization_history = optimization_history[gdiis_result.discarded_history_size :]
        diis_step = gdiis_result.step
        if diis_step is not None:
            diis_step = np.where(phase_mask, diis_step, 0.0)
            proposal = StepProposal(
                step=diis_step,
                policy=(
                    f"phase_{phase_name}:controlled_gdiis:{gdiis_result.status}"
                    if len(phase_masks) > 1
                    else f"controlled_gdiis:{gdiis_result.status}"
                ),
                hessian_min_eigenvalue=proposal.hessian_min_eigenvalue,
                hessian_condition=proposal.hessian_condition,
                damping_shift=proposal.damping_shift,
            )
            gdiis_used = True
        if len(phase_masks) > 1 and not proposal.policy.startswith("phase_"):
            proposal = replace(
                proposal,
                policy=f"phase_{phase_name}:{proposal.policy}",
            )
        try:
            proposal = (
                _enforce_proposal_gdv_internal_trust(
                    proposal, service, current, q, trust_radius
                )
                if settings.stationary_point == "transition_state"
                else _enforce_proposal_cartesian_trust(
                    proposal,
                    service,
                    current,
                    q,
                    trust_radius,
                    settings,
                )
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            if not gdiis_used:
                raise
            gdiis_result = replace(
                gdiis_result,
                step=None,
                status=f"{gdiis_result.status}:unrealizable_cartesian_trust",
            )
            gdiis_used = False
            proposal = model_proposal
            if len(phase_masks) > 1 and not proposal.policy.startswith("phase_"):
                proposal = replace(proposal, policy=f"phase_{phase_name}:{proposal.policy}")
            proposal = (
                _enforce_proposal_gdv_internal_trust(
                    proposal, service, current, q, trust_radius
                )
                if settings.stationary_point == "transition_state"
                else _enforce_proposal_cartesian_trust(
                    proposal,
                    service,
                    current,
                    q,
                    trust_radius,
                    settings,
                )
            )
        current_damping = proposal.damping_shift
        step = proposal.step
        proposed_step_norm = float(np.linalg.norm(step))
        step_norm = proposed_step_norm
        try:
            proposed_coordinates = service.coordinates_from_q(q + step)
            proposed_cartesian_step_bohr = (
                aligned_cartesian_displacement(
                    current.coordinates_angstrom, proposed_coordinates
                ).reshape(-1)
                * ANGSTROM_TO_BOHR
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            proposed_cartesian_step_bohr = np.full(current.coordinates_angstrom.size, math.inf)
        gdv_prospective_convergence = _gdv_transition_state_prospective_convergence(
            settings,
            convergence_gradient,
            proposed_cartesian_step_bohr,
        )
        gdv_prospective_converged = bool(
            gdv_prospective_convergence is not None
            and all(gdv_prospective_convergence.values())
        )
        stationary_repeat = bool(
            proposed_step_norm <= settings.stagnation_step_floor
            and _convergence_force_satisfied(convergence)
        )
        zero_cartesian_step = bool(
            not stationary_repeat
            and proposed_step_norm > settings.stagnation_step_floor
            and np.all(np.isfinite(proposed_cartesian_step_bohr))
            and float(np.max(np.abs(proposed_cartesian_step_bohr)))
            <= settings.stagnation_step_floor
        )
        if gdv_prospective_converged:
            # GrdOpt/CONVEF tests the current evaluated force together with D,
            # the just-built displacement, before RedCar and before another
            # electronic-structure call. Record that final GDV cycle while
            # retaining the current evaluated geometry as the TS result.
            trial = current
            prediction_hessian = (
                hessian
                if proposal.prediction_hessian is None
                else proposal.prediction_hessian
            )
            predicted = _predicted_reduction(gradient, prediction_hessian, step)
            actual = 0.0
            rho = 0.0
            accepted = False
            message = "converged_transition_state_gdv_prospective_step"
            rejected_trials = 0
            last_cartesian_step_bohr = proposed_cartesian_step_bohr.copy()
            convergence = gdv_prospective_convergence
            final_gradient_verification = {
                "performed": False,
                "confirmed": True,
                "reason": "analytic_gradient",
            }
            converged = True
            status = "converged_transition_state"
        elif stationary_repeat:
            # The current state already carries an evaluated energy and
            # gradient.  When its stationary-point model proposes an exactly
            # null step, accept that evaluated state once more so that the
            # energy and displacement criteria refer to two accepted states.
            # This is not convergence on an unevaluated prospective geometry.
            trial = current
            step = np.zeros_like(step)
            predicted = 0.0
            actual = 0.0
            rho = 1.0
            accepted = True
            message = "accepted_stationary_repeat"
            rejected_trials = 0
        elif zero_cartesian_step:
            trial = current
            step = np.zeros_like(step)
            predicted = 0.0
            actual = 0.0
            rho = -math.inf
            accepted = False
            message = "rejected_zero_cartesian_step"
            rejected_trials = 0
        else:
            trial, step, predicted, actual, rho, accepted, message, rejected_trials = (
                _evaluate_step_trials(
                    service,
                    current,
                    q,
                    gradient,
                    (
                        hessian
                        if proposal.prediction_hessian is None
                        else proposal.prediction_hessian
                    ),
                    step,
                    settings,
                    iteration=iteration,
                    far_from_minimum=far_from_minimum,
                )
            )
        step_norm = float(np.linalg.norm(step))
        current_projected_norm, trial_projected_norm = (
            _transition_state_stationarity_norms(
                service,
                current,
                trial,
                gradient,
                settings,
            )
        )
        trial_cartesian_rmsd = _cartesian_rms_displacement_angstrom(
            current.coordinates_angstrom,
            trial.coordinates_angstrom,
        )
        if accepted and chart_lifecycle_controller is not None:
            chart_valid, chart_reason = chart_lifecycle_controller.validate_proposed_geometry(
                service.atoms,
                trial.coordinates_angstrom,
            )
            if not chart_valid:
                accepted = False
                message = f"rejected_invalid_fixed_chart:{chart_reason}"
        chart_changed = False
        lifecycle_status = "DISABLED"
        lifecycle_epoch = (
            0
            if chart_lifecycle_controller is None
            else chart_lifecycle_controller.snapshot.epoch
        )
        if gdv_prospective_converged:
            update_status = "gdv_convef_prospective_step_no_qm_evaluation"
        elif accepted:
            previous_gradient_inf = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            previous_gradient_policy = str(fd_info["gradient_policy"])
            old_q = q.copy()
            old_gradient = gradient.copy()
            old_energy = current.energy_hartree
            old_coordinates = current.coordinates_angstrom.copy()
            q = np.asarray(trial.q, dtype=float).copy()
            current = trial
            service.accept_electronic_state(current)
            lifecycle_result = None
            if chart_lifecycle_controller is not None:
                lifecycle_result = chart_lifecycle_controller.evaluate_accepted_geometry(
                    service.atoms,
                    current.coordinates_angstrom,
                )
                chart_lifecycle_events.append(lifecycle_result)
                lifecycle_status = lifecycle_result.status
                if lifecycle_result.coordinate_changed:
                    (
                        model,
                        current,
                        q,
                        hessian,
                        chart_hessian_status,
                        lifecycle_inactive_labels,
                    ) = _apply_chart_lifecycle_transition(
                        service,
                        current,
                        hessian,
                        lifecycle_result,
                        settings,
                    )
                    chart_changed = True
                    lifecycle_epoch = lifecycle_result.current.epoch
                    transition_mode_reference = _initial_transition_mode_reference(
                        hessian,
                        service.optimizer_metric(current.coordinates_angstrom),
                        settings,
                        coordinate_unit_scales=service.gdv_internal_coordinate_scales(),
                        coordinate_directions=service.coordinate_directions(
                            current.coordinates_angstrom
                        ),
                        coordinates_angstrom=current.coordinates_angstrom,
                        reaction_directions=_transition_reaction_cartesian_directions(
                            model,
                            service.coordinate_directions(current.coordinates_angstrom),
                        ),
                    )
                    local_groups = _local_coordinate_groups(model, hessian, settings)
                    hessian_mask = _hessian_sparsity_mask(model, hessian, settings)
                    micro_masks = service.coordinate_phase_masks("inter-intra-micro")
                    micro_schedule_available = len(micro_masks) == 3
                    use_micro_schedule = bool(
                        settings.stationary_point == "minimum"
                        and micro_schedule_available
                        and settings.coordinate_schedule == "inter-intra-micro"
                    )
                    phase_masks = service.coordinate_phase_masks(
                        "joint"
                        if use_micro_schedule
                        or settings.stationary_point == "transition_state"
                        else settings.coordinate_schedule
                    )
                    phase_index = 0
                    phase_steps = 0
                if lifecycle_result.requires_commit:
                    chart_lifecycle_controller.commit_transition(lifecycle_result)
                    lifecycle_epoch = lifecycle_result.current.epoch
            # Preserve the sign: geometry-seed separately tests |dE| and a
            # final energy rise. An absolute value made downhill steps look
            # like increases.
            last_energy_change = _signed_energy_change(
                old_energy,
                current.energy_hartree,
            )
            last_cartesian_step_bohr = (
                aligned_cartesian_displacement(
                    old_coordinates, current.coordinates_angstrom
                ).reshape(-1)
                * ANGSTROM_TO_BOHR
            )
            _append_xyz(
                trajectory_path,
                service.atoms,
                current.coordinates_angstrom,
                f"iteration={iteration} energy={current.energy_hartree:.12g}",
            )
            gradient, fd_info = _gradient_in_coordinate_space(
                service,
                current,
                q,
                hessian,
                settings,
                iteration=iteration,
                previous_gradient=old_gradient,
                selective_disabled=selective_disabled,
                two_sided_latched=two_sided_latched,
            )
            if str(fd_info.get("gradient_policy", "")).startswith("coordinate_energy_fd") and (
                str(fd_info.get("mode", "")) == "two-sided"
                or bool(fd_info.get("near_minimum", False))
                or (
                    str(fd_info.get("mode", "")) == "one-sided"
                    and gradient.size
                    and float(np.max(np.abs(gradient))) <= settings.fd_two_sided_switch_force
                )
            ):
                two_sided_latched = True
            new_gradient_inf = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            if chart_changed:
                selective_disabled = False
                selective_rejection_count = 0
                two_sided_latched = bool(
                    current.gradient_hartree_per_bohr is None
                    and str(fd_info.get("mode", "")) == "two-sided"
                )
            elif previous_gradient_policy == "coordinate_energy_fd_selective":
                growth_limit = max(
                    settings.max_force_tolerance,
                    settings.selective_fallback_gradient_growth * previous_gradient_inf,
                )
                if new_gradient_inf > growth_limit:
                    selective_disabled = True
            if chart_changed:
                update_status = f"{chart_hessian_status}_cross_epoch_secant_skipped"
            elif settings.stationary_point == "transition_state":
                transported_hessian_history = _transported_optimizer_history(
                    service,
                    gdv_hessian_history,
                )
                hessian, update_status = _gdv_d2corx_history_update(
                    hessian,
                    current_q=service.actual_q(current.coordinates_angstrom),
                    current_gradient=gradient,
                    history=[
                        (history_q, history_gradient)
                        for history_q, _history_energy, history_gradient in (
                            transported_hessian_history
                        )
                    ],
                    settings=settings,
                    coordinate_unit_scales=service.gdv_internal_coordinate_scales(),
                )
            else:
                hessian, update_status = _update_hessian(
                    hessian,
                    q - old_q,
                    gradient - old_gradient,
                    settings,
                    metric=service.optimizer_metric(current.coordinates_angstrom),
                    coordinate_unit_scales=(
                        service.gdv_internal_coordinate_scales()
                        if settings.stationary_point == "transition_state"
                        else None
                    ),
                    far_from_minimum=far_from_minimum,
                    cartesian_step_bohr=last_cartesian_step_bohr,
                )
            if settings.sparse_hessian_updates:
                hessian = _project_hessian_sparsity(hessian, hessian_mask)
            if chart_changed:
                optimization_history = [(current, gradient.copy())]
                gdv_hessian_history = [(current, gradient.copy())]
                bad_model_ratio_count = 0
                current_damping = 0.0
            else:
                optimization_history.append((current, gradient.copy()))
                del optimization_history[: -settings.gdiis_history]
                gdv_hessian_history.append((current, gradient.copy()))
                del gdv_hessian_history[: -(len(q) + 1)]
            if chart_changed:
                bad_model_ratio_count = 0
            elif settings.stationary_point == "transition_state":
                # GDV DXRFO keeps the D2CorX model independently of the
                # actual/predicted energy ratio; UpTrus is disabled for this
                # transition-state protocol.
                bad_model_ratio_count = 0
            elif not np.isfinite(rho) or rho < 0.10 or rho > 2.5:
                bad_model_ratio_count += 1
            else:
                bad_model_ratio_count = 0
            if bad_model_ratio_count > settings.hessian_bad_ratio_limit:
                hessian, rebuild_status = _rebuild_optimizer_hessian_at_geometry(
                    model,
                    settings,
                    hessian,
                    atoms=tuple(initial_geometry.atoms),
                    coordinates_angstrom=current.coordinates_angstrom,
                    xyzin_path=xyzin_path,
                )
                update_status += f"_untrusted_model_{rebuild_status}"
                current_damping = max(current_damping, settings.min_hessian_eigenvalue)
                bad_model_ratio_count = 0
            line_search_scale = (
                float(np.linalg.norm(step)) / proposed_step_norm
                if proposed_step_norm > 0.0
                else 0.0
            )
            if not chart_changed:
                service.update_coordinate_projector(
                    previous_q=old_q,
                    previous_coordinates=old_coordinates,
                    current_q=q,
                    current_coordinates=current.coordinates_angstrom,
                    trust_ratio=rho,
                    line_search_scale=line_search_scale,
                )
            current_damping, trust_radius = _accepted_optimizer_trust_update(
                current_damping,
                trust_radius,
                rho,
                line_search_scale,
                _cartesian_rms_displacement_angstrom(old_coordinates, current.coordinates_angstrom),
                settings,
            )
            convergence_gradient = _convergence_gradient(current, gradient, settings, service)
            convergence = _gaussian_like_convergence(
                settings, last_energy_change, convergence_gradient, last_cartesian_step_bohr
            )
            if all(convergence.values()):
                index_status = None
                if current.gradient_hartree_per_bohr is None:
                    (
                        gradient,
                        fd_info,
                        hessian,
                        settings,
                        convergence,
                        final_gradient_verification,
                        final_hessian_refinement_attempted,
                        converged,
                    ) = _advance_numerical_convergence_state(
                        service,
                        current,
                        q,
                        gradient,
                        fd_info,
                        hessian,
                        settings,
                        convergence,
                        last_energy_change=last_energy_change,
                        last_cartesian_step_bohr=last_cartesian_step_bohr,
                        selective_disabled=selective_disabled,
                        final_hessian_refinement_attempted=final_hessian_refinement_attempted,
                        transition_mode_reference=transition_mode_reference,
                    )
                    two_sided_latched = True
                    if final_gradient_verification.get("two_sided_mismatch"):
                        optimization_history = [(current, gradient.copy())]
                        gdv_hessian_history = [(current, gradient.copy())]
                        bad_model_ratio_count = 0
                        current_damping = 0.0
                else:
                    final_gradient_verification = {
                        "performed": False,
                        "confirmed": True,
                        "reason": "analytic_gradient",
                    }
                    converged = True
                if converged:
                    if (
                        settings.stationary_point == "transition_state"
                        and current.gradient_hartree_per_bohr is not None
                        and _optimizer_hessian_index(hessian, settings) != 1
                        and not transition_index_refresh_attempted
                    ):
                        hessian, index_refresh_status = (
                            _refresh_transition_index_subspace_from_analytic_gradients(
                                service,
                                current,
                                q,
                                hessian,
                                settings,
                                transition_mode_reference,
                            )
                        )
                        update_status += f"_{index_refresh_status}"
                        transition_index_refresh_attempted = True
                    (
                        converged,
                        transition_index_refresh_attempted,
                        index_status,
                    ) = _transition_state_index_convergence_state(
                        hessian,
                        settings,
                        refresh_attempted=transition_index_refresh_attempted,
                    )
                    if index_status == "transition_state_index_refresh_pending":
                        message = index_status
                    elif index_status:
                        status = index_status
                        message = index_status
                if converged:
                    status = (
                        "converged_transition_state"
                        if settings.stationary_point == "transition_state"
                        else "converged_gaussian"
                    )
                    message = (
                        (
                            "accepted_converged_one_sided"
                            if final_gradient_verification
                            and final_gradient_verification.get("reason")
                            == "one_sided_only_protocol"
                            else "accepted_converged_two_sided_refinement"
                        )
                        if current.gradient_hartree_per_bohr is None
                        else "accepted_converged"
                    )
                else:
                    if index_status is None:
                        message = (
                            "accepted_continue_two_sided_after_mismatch"
                            if final_gradient_verification
                            and final_gradient_verification.get("two_sided_mismatch")
                            else "accepted_final_hessian_step_pending"
                        )
            elif convergence["energy"]:
                message = "accepted_energy_plateau"
            selective_rejection_count = 0
        else:
            update_status = "skipped_rejected_step"
            bad_model_ratio_count += 1
            stagnation_count += 1
            rejected_secant = _rejected_transition_state_secant_data(
                service,
                current,
                trial,
                gradient,
                settings,
            )
            consume_rejected_secant = _rejected_transition_state_secant_policy(
                message,
                rejected_secant is not None,
            )
            if rejected_secant is not None and consume_rejected_secant:
                secant_step, secant_y, secant_cartesian_step = rejected_secant
                candidate_hessian, secant_status = _update_hessian(
                    hessian,
                    secant_step,
                    secant_y,
                    settings,
                    metric=service.optimizer_metric(trial.coordinates_angstrom),
                    coordinate_unit_scales=service.gdv_internal_coordinate_scales(),
                    cartesian_step_bohr=secant_cartesian_step,
                )
                update_status = f"rejected_trial_{secant_status}"
                if _hessian_secant_update_applied(secant_status):
                    hessian = candidate_hessian
                    bad_model_ratio_count = 0
            if bad_model_ratio_count > settings.hessian_bad_ratio_limit:
                hessian, rebuild_status = _rebuild_optimizer_hessian_at_geometry(
                    model,
                    settings,
                    hessian,
                    atoms=tuple(initial_geometry.atoms),
                    coordinates_angstrom=current.coordinates_angstrom,
                    xyzin_path=xyzin_path,
                )
                update_status += f"_untrusted_model_{rebuild_status}"
                optimization_history = optimization_history[-1:]
                gdv_hessian_history = gdv_hessian_history[-1:]
                bad_model_ratio_count = 0
            recovered = False
            if trust_radius <= settings.min_trust_radius and stagnation_count >= 1:
                if stagnation_recoveries < settings.max_stagnation_recoveries:
                    hessian, rebuild_status = _rebuild_optimizer_hessian_at_geometry(
                        model,
                        settings,
                        hessian,
                        atoms=tuple(initial_geometry.atoms),
                        coordinates_angstrom=current.coordinates_angstrom,
                        xyzin_path=xyzin_path,
                    )
                    update_status += f"_stagnation_{rebuild_status}"
                    recovery_settings = replace(
                        settings,
                        one_sided_until_convergence=False,
                        selective_fd_refresh=False,
                    )
                    gradient, fd_info = _gradient_in_coordinate_space(
                        service,
                        current,
                        q,
                        hessian,
                        recovery_settings,
                        force_explicit=True,
                        force_two_sided=(current.gradient_hartree_per_bohr is None),
                        two_sided_latched=True,
                        selective_disabled=True,
                    )
                    settings = recovery_settings
                    trust_radius = float(settings.trust_radius)
                    stagnation_recoveries += 1
                    stagnation_count = 0
                    bad_model_ratio_count = 0
                    update_status += "_stagnation_recovered_two_sided"
                    message = "rejected_stagnation_recovered"
                    recovered = True
                else:
                    status = "stalled"
                    message = "stalled_trust_radius_after_recovery"
            if recovered:
                last_energy_change = 0.0
            service.refresh_coordinate_projector(q, current.coordinates_angstrom)
            if not recovered:
                current_damping, trust_radius = _rejected_optimizer_trust_update(
                    current_damping,
                    trust_radius,
                    _rejected_optimizer_cartesian_step(
                        current,
                        trial,
                        proposal,
                        message,
                    ),
                    settings,
                )
            convergence_gradient = _convergence_gradient(current, gradient, settings, service)
            convergence = _gaussian_like_convergence(
                settings, last_energy_change, convergence_gradient, last_cartesian_step_bohr
            )
            if settings.selective_fd_refresh and not selective_disabled:
                selective_rejection_count += 1
                if (
                    settings.selective_fallback_rejections
                    and selective_rejection_count >= settings.selective_fallback_rejections
                ):
                    selective_disabled = True
        convergence_gradient = _convergence_gradient(current, gradient, settings, service)
        grad_inf = float(np.max(np.abs(convergence_gradient))) if convergence_gradient.size else 0.0
        grad_rms = _rms(convergence_gradient)
        step_max = (
            float(np.max(np.abs(last_cartesian_step_bohr)))
            if last_cartesian_step_bohr.size
            else 0.0
        )
        step_rms = _rms(last_cartesian_step_bohr)
        record = OptimizerIteration(
            iteration=iteration,
            status=message,
            energy_hartree=current.energy_hartree,
            trial_energy_hartree=trial.energy_hartree,
            gradient_inf_norm=grad_inf,
            gradient_rms_norm=grad_rms,
            step_norm=step_norm,
            step_inf_norm=step_max,
            step_rms_norm=step_rms,
            energy_change_hartree=last_energy_change,
            convergence=convergence,
            trust_radius=trust_radius,
            trust_ratio=rho,
            gradient_policy=str(fd_info["gradient_policy"]),
            fd_mode=fd_info["mode"],
            fd_step_min=fd_info["step_min"],
            fd_step_max=fd_info["step_max"],
            class_threshold_fraction=float(fd_info.get("class_threshold_fraction", 0.0)),
            class_screen_audit=bool(fd_info.get("class_screen_audit", False)),
            class_screen_audit_interval=int(fd_info.get("class_screen_audit_interval", 0)),
            chart_epoch=lifecycle_epoch,
            chart_lifecycle_status=lifecycle_status,
            refreshed_coordinate_count=int(fd_info["refreshed_coordinate_count"]),
            predicted_coordinate_count=int(fd_info["predicted_coordinate_count"]),
            active_coordinate_fraction=float(fd_info["active_coordinate_fraction"]),
            fd_one_sided_count=int(fd_info["one_sided_count"]),
            fd_two_sided_count=int(fd_info["two_sided_count"]),
            fd_parallel_workers=int(fd_info["parallel_workers"]),
            local_group_count=len(local_groups),
            local_group_sizes=tuple(len(group) for group in local_groups),
            surrogate_sample_count=int(fd_info["surrogate_sample_count"]),
            hessian_sparsity=_matrix_sparsity(hessian),
            hessian_min_eigenvalue=proposal.hessian_min_eigenvalue,
            hessian_condition=proposal.hessian_condition,
            hessian_update_status=update_status,
            step_policy=proposal.policy,
            gdiis_attempted=gdiis_result.attempted,
            gdiis_used=gdiis_used,
            gdiis_status=gdiis_result.status,
            gdiis_history_size=gdiis_result.history_size,
            gdiis_retained_history_size=gdiis_result.retained_history_size,
            gdiis_discarded_history_size=gdiis_result.discarded_history_size,
            rejected_trial_count=rejected_trials,
            geometry_status=_geometry_status(
                service.reference_coordinates, current.coordinates_angstrom
            ),
            coordinate_model_status=service.coordinate_model_status(
                current.coordinates_angstrom, settings
            ),
            selective_fallback_status=_selective_fallback_status(settings, selective_disabled),
            symmetry_status=_symmetry_status(settings, model),
            qm_evaluations=service.qm_evaluations,
            energy_evaluations=service.energy_evaluations,
            gradient_evaluations=service.gradient_evaluations,
            hessian_evaluations=service.hessian_evaluations,
            fd_displacements=service.fd_displacements,
            cache_hits=service.cache.hits,
            avoided_evaluations=service.cache.hits,
            proposed_cartesian_rmsd_angstrom=proposal.cartesian_rmsd_angstrom,
            trust_step_scale=proposal.trust_scale,
            trust_solver_iterations=proposal.trust_iterations,
            applied_trust_radius_angstrom=proposal.applied_trust_radius_angstrom,
            predicted_reduction_hartree=float(predicted),
            actual_reduction_hartree=float(actual),
            current_projected_gradient_norm=current_projected_norm,
            trial_projected_gradient_norm=trial_projected_norm,
            trial_cartesian_rmsd_angstrom=trial_cartesian_rmsd,
            transition_mode_index=proposal.transition_mode_index,
            transition_mode_overlap=proposal.transition_mode_overlap,
            transition_ascending_shift=proposal.transition_ascending_shift,
            transition_descending_shift=proposal.transition_descending_shift,
            message=message,
        )
        iterations.append(record)
        phase_steps += 1
        _append_trace(trace_path, record, model)
        if converged:
            break
        if status in {
            "stalled",
            "frozen_chart_domain_invalid",
            "stationary_minimum",
            "stationary_higher_order_saddle",
        }:
            break
        if (
            settings.stationary_point == "minimum"
            and current.energy_hartree > previous_energy
            and trust_radius <= settings.min_trust_radius
        ):
            status = "stalled"
            break
        previous_energy = current.energy_hartree

    if (
        settings.final_gradient_verification
        and current.gradient_hartree_per_bohr is None
        and final_gradient_verification is None
    ):
        final_gradient, _final_fd_info = _verify_final_numerical_gradient(
            service,
            current,
            q,
            hessian,
            settings,
            selective_disabled=selective_disabled,
        )
        final_gradient_verification = {
            "performed": True,
            "confirmed": True,
            "gradient_policy": "final_central_verification",
            "fd_two_sided_count": int(_final_fd_info["two_sided_count"]),
        }
    else:
        final_gradient = np.asarray(gradient, dtype=float).copy()
        _final_fd_info = fd_info
    final_convergence = _gaussian_like_convergence(
        settings,
        last_energy_change,
        _convergence_gradient(current, final_gradient, settings, service),
        last_cartesian_step_bohr,
    )
    final_geometry_converged = all(final_convergence.values())
    exact_final_hessian = False
    final_cartesian_hessian_path: Path | None = None
    final_frequencies_cm: tuple[float, ...] = ()
    irc_verification: dict[str, object] | None = None
    irc_path: Path | None = None
    if settings.stationary_point == "transition_state":
        (
            hessian,
            exact_final_hessian,
            final_cartesian_hessian_path,
            final_frequencies_cm,
            irc_verification,
            irc_path,
            validation,
        ) = _validate_final_transition_state(
            service,
            current,
            q,
            model,
            hessian,
            settings,
            transition_mode_reference,
            root,
            irc_target_geometries,
        )
        final_convergence.update(validation)
        if transition_mode_overlaps:
            # A transient loss of overlap is expected when a numerical
            # gradient perturbs or mixes nearby Hessian modes.  It must not
            # invalidate a TS that subsequently recovers the tracked mode and
            # passes the independent final Hessian/gradient validation.  The
            # final exact-mode test above remains mandatory.
            final_convergence["mode_tracking"] = (
                transition_mode_overlaps[-1] >= settings.transition_mode_overlap_threshold
            )
    if settings.compute_final_hessian and converged and not final_frequencies_cm and model.kind == "sonic":
        final_frequencies_cm = _frequencies_from_final_sonic_hessian(service, current, hessian)
    final_hessian_index = _optimizer_hessian_index(hessian, settings)
    final_hessian_kind = "exact" if exact_final_hessian else "approximate"
    # A first-order saddle is not a valid TS unless the model retains exactly
    # one downhill direction.  For minima, however, a quasi-Newton Hessian can
    # remain indefinite when the geometry starts at convergence and no secant
    # update is available; report its index in the summary without turning an
    # minimum already at convergence into a false failure.
    if settings.stationary_point == "transition_state":
        final_convergence["hessian_index"] = final_hessian_index == 1
        if final_geometry_converged and final_hessian_index != 1:
            converged = False
            status = (
                "stationary_minimum"
                if final_hessian_index == 0
                else "stationary_higher_order_saddle"
            )
    if converged and not all(final_convergence.values()):
        converged = False
        status = "not_converged_after_final_refresh"
    if settings.compute_final_hessian:
        assert final_hessian_path is not None
        write_optimizer_hessian(
            final_hessian_path,
            hessian,
            model,
            source=f"final-{settings.stationary_point}; initial={hessian_source}",
            q=q,
        )
    result = OptimizerResult(
        converged=converged,
        status=status if not converged else status,
        settings=settings,
        atoms=service.atoms,
        initial_coordinates_angstrom=initial_coordinates_for_result,
        final_coordinates_angstrom=current.coordinates_angstrom,
        final_q=q,
        final_energy_hartree=current.energy_hartree,
        final_gradient=_convergence_gradient(current, final_gradient, settings, service),
        final_energy_change_hartree=last_energy_change,
        final_displacement=last_cartesian_step_bohr,
        final_convergence=final_convergence,
        energy_noise_hartree=settings.energy_noise,
        energy_noise_samples=settings.energy_noise_samples,
        energy_noise_energies_hartree=energy_noise_energies,
        iterations=tuple(iterations),
        qm_evaluations=service.qm_evaluations,
        energy_evaluations=service.energy_evaluations,
        final_hessian_energy_evaluations=service.final_hessian_energy_evaluations,
        gradient_evaluations=service.gradient_evaluations,
        hessian_evaluations=service.hessian_evaluations,
        fd_displacements=service.fd_displacements,
        cache_hits=service.cache.hits,
        avoided_evaluations=service.cache.hits,
        cache_path=service.cache.path or (root / "optimizer_cache.jsonl"),
        trajectory_path=trajectory_path,
        trace_path=trace_path,
        summary_path=summary_path,
        final_hessian_path=final_hessian_path,
        final_hessian_index=final_hessian_index,
        final_hessian_kind=final_hessian_kind,
        initial_hessian_source=hessian_source,
        exact_final_hessian=exact_final_hessian,
        final_cartesian_hessian_path=final_cartesian_hessian_path,
        final_frequencies_cm=final_frequencies_cm,
        transition_mode_overlaps=tuple(transition_mode_overlaps),
        irc_verification=irc_verification,
        irc_path=irc_path,
        final_gradient_verification=final_gradient_verification,
        runtime_method_manifest_path=runtime_method_manifest_path,
        chart_lifecycle_events=tuple(chart_lifecycle_events),
        optimization_active_labels=tuple(model.labels),
        optimization_inactive_labels=lifecycle_inactive_labels,
        frozen_chart_replay=frozen_chart_replay,
    )
    summary_path.write_text(
        json.dumps(optimizer_result_to_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def optimizer_result_to_json(result: OptimizerResult) -> dict[str, object]:
    from .chart_lifecycle import chart_lifecycle_result_to_json

    return {
        "schema": OPTIMIZER_SUMMARY_SCHEMA,
        "converged": result.converged,
        "status": result.status,
        "stationary_point": result.settings.stationary_point,
        "transition_mode": result.settings.transition_mode,
        "transition_mode_overlaps": list(result.transition_mode_overlaps),
        "transition_mode_tracking": {
            "minimum_overlap": (
                min(result.transition_mode_overlaps) if result.transition_mode_overlaps else None
            ),
            "final_overlap": (
                result.transition_mode_overlaps[-1] if result.transition_mode_overlaps else None
            ),
            "threshold": result.settings.transition_mode_overlap_threshold,
            "transient_loss_allowed": True,
            "recovered_at_final_step": bool(
                result.transition_mode_overlaps
                and result.transition_mode_overlaps[-1]
                >= result.settings.transition_mode_overlap_threshold
            ),
        },
        "exact_final_hessian": result.exact_final_hessian,
        "final_cartesian_hessian": (
            None
            if result.final_cartesian_hessian_path is None
            else str(result.final_cartesian_hessian_path)
        ),
        "final_frequencies_cm-1": list(result.final_frequencies_cm),
        "irc_verification": result.irc_verification,
        "irc_path": None if result.irc_path is None else str(result.irc_path),
        "final_hessian_index": result.final_hessian_index,
        "final_hessian_kind": result.final_hessian_kind,
        "final_energy_hartree": result.final_energy_hartree,
        "final_energy_change_hartree": result.final_energy_change_hartree,
        "final_gradient_inf_norm": float(np.max(np.abs(result.final_gradient))),
        "final_gradient_rms_norm": _rms(result.final_gradient),
        "final_displacement_inf_norm": float(np.max(np.abs(result.final_displacement)))
        if result.final_displacement.size
        else 0.0,
        "final_displacement_rms_norm": _rms(result.final_displacement),
        "final_convergence": result.final_convergence,
        "energy_noise_hartree": result.energy_noise_hartree,
        "energy_noise_samples": result.energy_noise_samples,
        "energy_noise_energies_hartree": list(result.energy_noise_energies_hartree),
        "final_q": np.asarray(result.final_q, dtype=float).tolist(),
        "final_coordinates_angstrom": np.asarray(
            result.final_coordinates_angstrom, dtype=float
        ).tolist(),
        "optimization_steps": len(result.iterations),
        "gaussian_equivalent_steps": len(result.iterations),
        "optimizer_diagnostics": _optimizer_diagnostics(result.iterations),
        "chart_lifecycle": [
            chart_lifecycle_result_to_json(item) for item in result.chart_lifecycle_events
        ],
        "frozen_chart_replay": result.frozen_chart_replay,
        "optimization_task_subspace": {
            "active_labels": list(result.optimization_active_labels),
            "inactive_labels": list(result.optimization_inactive_labels),
            "active_count": len(result.optimization_active_labels),
            "inactive_count": len(result.optimization_inactive_labels),
        },
        "qm_evaluations": result.qm_evaluations,
        "energy_evaluations": result.energy_evaluations,
        "final_hessian_energy_evaluations": result.final_hessian_energy_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "hessian_evaluations": result.hessian_evaluations,
        "fd_displacements": result.fd_displacements,
        "cache_hits": result.cache_hits,
        "avoided_evaluations": result.avoided_evaluations,
        "cache": str(result.cache_path),
        "initial_hessian_source": result.initial_hessian_source,
        "final_gradient_verification": result.final_gradient_verification,
        "final_hessian": (
            str(result.final_hessian_path) if result.final_hessian_path is not None else None
        ),
        "runtime_method_manifest": (
            None
            if result.runtime_method_manifest_path is None
            else str(result.runtime_method_manifest_path)
        ),
        "convergence_thresholds": {
            "energy_tolerance_hartree": result.settings.energy_tolerance,
            "max_force_tolerance": result.settings.max_force_tolerance,
            "rms_force_tolerance": result.settings.rms_force_tolerance,
            "max_displacement_tolerance": result.settings.max_displacement_tolerance,
            "rms_displacement_tolerance": result.settings.rms_displacement_tolerance,
        },
        "iterations": [
            {
                "iteration": item.iteration,
                "status": item.status,
                "energy_hartree": item.energy_hartree,
                "trial_energy_hartree": item.trial_energy_hartree,
                "gradient_inf_norm": item.gradient_inf_norm,
                "gradient_rms_norm": item.gradient_rms_norm,
                "step_norm": item.step_norm,
                "step_inf_norm": item.step_inf_norm,
                "step_rms_norm": item.step_rms_norm,
                "energy_change_hartree": item.energy_change_hartree,
                "convergence": item.convergence,
                "trust_radius": item.trust_radius,
                "trust_ratio": item.trust_ratio,
                "gradient_policy": item.gradient_policy,
                "fd_mode": item.fd_mode,
                "fd_step_min": item.fd_step_min,
                "fd_step_max": item.fd_step_max,
                "class_threshold_fraction": item.class_threshold_fraction,
                "class_screen_audit": item.class_screen_audit,
                "class_screen_audit_interval": item.class_screen_audit_interval,
                "chart_epoch": item.chart_epoch,
                "chart_lifecycle_status": item.chart_lifecycle_status,
                "refreshed_coordinate_count": item.refreshed_coordinate_count,
                "predicted_coordinate_count": item.predicted_coordinate_count,
                "active_coordinate_fraction": item.active_coordinate_fraction,
                "fd_one_sided_count": item.fd_one_sided_count,
                "fd_two_sided_count": item.fd_two_sided_count,
                "fd_parallel_workers": item.fd_parallel_workers,
                "local_group_count": item.local_group_count,
                "local_group_sizes": list(item.local_group_sizes),
                "surrogate_sample_count": item.surrogate_sample_count,
                "hessian_sparsity": item.hessian_sparsity,
                "hessian_min_eigenvalue": item.hessian_min_eigenvalue,
                "hessian_condition": item.hessian_condition,
                "hessian_update_status": item.hessian_update_status,
                "step_policy": item.step_policy,
                "gdiis_attempted": item.gdiis_attempted,
                "gdiis_used": item.gdiis_used,
                "gdiis_status": item.gdiis_status,
                "gdiis_history_size": item.gdiis_history_size,
                "gdiis_retained_history_size": item.gdiis_retained_history_size,
                "gdiis_discarded_history_size": item.gdiis_discarded_history_size,
                "rejected_trial_count": item.rejected_trial_count,
                "geometry_status": item.geometry_status,
                "coordinate_model_status": item.coordinate_model_status,
                "selective_fallback_status": item.selective_fallback_status,
                "symmetry_status": item.symmetry_status,
                "qm_evaluations": item.qm_evaluations,
                "energy_evaluations": item.energy_evaluations,
                "gradient_evaluations": item.gradient_evaluations,
                "hessian_evaluations": item.hessian_evaluations,
                "fd_displacements": item.fd_displacements,
                "cache_hits": item.cache_hits,
                "avoided_evaluations": item.avoided_evaluations,
                "proposed_cartesian_rmsd_angstrom": (item.proposed_cartesian_rmsd_angstrom),
                "trust_step_scale": item.trust_step_scale,
                "trust_solver_iterations": item.trust_solver_iterations,
                "applied_trust_radius_angstrom": item.applied_trust_radius_angstrom,
                "predicted_reduction_hartree": item.predicted_reduction_hartree,
                "actual_reduction_hartree": item.actual_reduction_hartree,
                "current_projected_gradient_norm": item.current_projected_gradient_norm,
                "trial_projected_gradient_norm": item.trial_projected_gradient_norm,
                "trial_cartesian_rmsd_angstrom": item.trial_cartesian_rmsd_angstrom,
                "transition_mode_index": item.transition_mode_index,
                "transition_mode_overlap": item.transition_mode_overlap,
                "transition_ascending_shift": item.transition_ascending_shift,
                "transition_descending_shift": item.transition_descending_shift,
                "message": item.message,
            }
            for item in result.iterations
        ],
        "trajectory": str(result.trajectory_path),
        "trace": str(result.trace_path),
    }


def _validate_final_transition_state(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    model: OptimizerCoordinateModel,
    approximate_hessian: np.ndarray,
    settings: OptimizerSettings,
    tracked_mode: TransitionModeReference | None,
    root: Path,
    irc_target_geometries: Sequence[np.ndarray] = (),
) -> tuple[
    np.ndarray,
    bool,
    Path | None,
    tuple[float, ...],
    dict[str, object] | None,
    Path | None,
    dict[str, bool],
]:
    """Require an exact first-order saddle and verify both IRC branches."""

    validation = {
        "exact_hessian": not settings.require_exact_final_hessian,
        "one_imaginary_frequency": not settings.require_exact_final_hessian,
        "exact_mode_overlap": not settings.require_exact_final_hessian,
        "irc": not settings.verify_irc,
    }
    # Reduced-coordinate TS searches (RAMA rigid torsions, PIC/SONIC scans)
    # deliberately certify the updated internal Hessian without requesting a
    # Cartesian backend Hessian.  This is distinct from the normal exact TS
    # path, which continues below unchanged.
    if not settings.require_exact_final_hessian and not settings.verify_irc:
        return (
            approximate_hessian,
            False,
            None,
            (),
            {
                "schema": TRANSITION_STATE_VALIDATION_SCHEMA,
                "status": "reduced_coordinate_validation",
                "hessian_space": model.kind,
                "exact_cartesian_hessian": False,
            },
            None,
            validation,
        )
    exact = service.evaluate(
        q,
        tag="final-transition-state-hessian",
        use_cache=False,
        persist_cache=True,
        requested_properties=("energy", "gradient", "hessian"),
    )
    cartesian = exact.result.hessian_hartree_per_bohr2
    if cartesian is None:
        return (
            approximate_hessian,
            False,
            None,
            (),
            {
                "schema": TRANSITION_STATE_VALIDATION_SCHEMA,
                "status": "missing_exact_cartesian_hessian",
            },
            None,
            validation,
        )
    cartesian_matrix = np.asarray(cartesian, dtype=float)
    expected = current.coordinates_angstrom.size
    if cartesian_matrix.shape != (expected, expected):
        raise ValueError(
            "final Cartesian Hessian has shape "
            f"{cartesian_matrix.shape}, expected {(expected, expected)}"
        )
    cartesian_gradient = exact.gradient_hartree_per_bohr
    if model.kind == "sonic" and cartesian_gradient is None:
        raise RuntimeError(
            "a physical final Hessian requires the Cartesian gradient for "
            "the exact B-prime transformation"
        )
    cartesian_path = root / "final_cartesian_hessian.json"
    cartesian_path.write_text(
        json.dumps(
            {
                "schema": "matrix.link.final_cartesian_hessian.v1",
                "units": "hartree/bohr^2",
                "atoms": list(service.atoms),
                "coordinates_angstrom": exact.coordinates_angstrom.tolist(),
                "gradient_hartree_per_bohr": (
                    None if cartesian_gradient is None else cartesian_gradient.tolist()
                ),
                "hessian": cartesian_matrix.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exact_optimizer_hessian = (
        optimizer_hessian_from_cartesian(
            cartesian_matrix,
            model,
            cartesian_gradient_hartree_per_bohr=cartesian_gradient,
            coordinates_angstrom=exact.coordinates_angstrom,
        )
        if model.kind == "sonic"
        else optimizer_hessian_from_cartesian(cartesian_matrix, model)
    )
    exact_orthonormal, inverse_sqrt_metric = _orthonormal_optimizer_hessian(
        exact_optimizer_hessian,
        service.optimizer_metric(exact.coordinates_angstrom),
        settings,
    )
    exact_values, exact_vectors = np.linalg.eigh(exact_orthonormal)
    exact_candidates = _cartesian_transition_mode_candidates(
        exact_vectors,
        inverse_sqrt_metric,
        service.coordinate_directions(exact.coordinates_angstrom),
    )
    mode_index, mode_overlap, _exact_reference = _select_transition_mode(
        exact_candidates,
        settings,
        tracked_mode,
        exact.coordinates_angstrom,
        eigenvalues=exact_values,
    )
    exact_mode = inverse_sqrt_metric @ exact_vectors[:, mode_index]
    frequencies = _exact_cartesian_frequencies_cm(
        service.atoms,
        exact.coordinates_angstrom,
        cartesian_matrix,
    )
    imaginary_count = int(np.count_nonzero(np.asarray(frequencies) < -1.0e-6))
    validation["exact_hessian"] = True
    validation["one_imaginary_frequency"] = imaginary_count == 1
    validation["exact_mode_overlap"] = mode_overlap >= settings.transition_mode_overlap_threshold
    irc_report: dict[str, object] | None = None
    irc_path: Path | None = None
    if settings.verify_irc:
        irc_report, irc_path = _verify_transition_state_irc(
            service,
            exact,
            q,
            exact_optimizer_hessian,
            exact_mode,
            settings,
            root,
            irc_target_geometries,
        )
        validation["irc"] = bool(irc_report.get("verified", False))
    return (
        exact_optimizer_hessian,
        True,
        cartesian_path,
        frequencies,
        irc_report,
        irc_path,
        validation,
    )


def _exact_cartesian_frequencies_cm(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    cartesian_hessian: np.ndarray,
) -> tuple[float, ...]:
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_gf import cartesian_normal_modes_from_hessian

    masses = np.asarray(
        [atomic_mass(int(atomic_number(atom) or 0)) for atom in atoms],
        dtype=float,
    )
    try:
        modes = cartesian_normal_modes_from_hessian(
            cartesian_hessian,
            masses,
            np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR,
            project_external=True,
            source="LINK exact final TS Hessian",
        )
    except ValueError as exc:
        if len(atoms) > 1 or "no vibrational Cartesian subspace" not in str(exc):
            raise
        modes = cartesian_normal_modes_from_hessian(
            cartesian_hessian,
            masses,
            np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR,
            project_external=False,
            source="LINK exact final TS Hessian (unprojected atom)",
        )
    return tuple(float(value) for value in modes.frequencies_cm)


def _verify_transition_state_irc(
    service: GeometryEvaluationService,
    saddle: OptimizerEvaluation,
    saddle_q: np.ndarray,
    hessian: np.ndarray,
    mode: np.ndarray,
    settings: OptimizerSettings,
    root: Path,
    target_geometries: Sequence[np.ndarray] = (),
) -> tuple[dict[str, object], Path]:
    """Follow optimizer-coordinate IRC branches to two distinct minima."""

    direction = np.asarray(mode, dtype=float).reshape(-1)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    branches: list[dict[str, object]] = []
    endpoints: list[np.ndarray] = []
    for sign, label in ((-1.0, "reverse"), (1.0, "forward")):
        branch_q = np.asarray(saddle_q, dtype=float) + sign * settings.irc_step_size * direction
        evaluation = service.evaluate(
            branch_q,
            tag=f"irc-{label}-000",
            use_cache=False,
            persist_cache=True,
            requested_properties=("energy", "gradient"),
        )
        points: list[dict[str, object]] = []
        converged = False
        target_reached = False
        for step_index in range(settings.irc_max_steps + 1):
            gradient, _info = _gradient_in_coordinate_space(
                service,
                evaluation,
                np.asarray(evaluation.q, dtype=float),
                hessian,
                settings,
                force_explicit=evaluation.gradient_hartree_per_bohr is None,
                selective_disabled=True,
            )
            gradient_norm = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            target_rmsd = None
            if target_geometries:
                target_rmsd = min(
                    _aligned_rmsd(evaluation.coordinates_angstrom, target)
                    for target in target_geometries
                )
            points.append(
                {
                    "step": step_index,
                    "q": np.asarray(evaluation.q, dtype=float).tolist(),
                    "energy_hartree": evaluation.energy_hartree,
                    "gradient_inf_norm": gradient_norm,
                    "target_rmsd_angstrom": target_rmsd,
                }
            )
            if target_rmsd is not None and target_rmsd <= 0.25:
                converged = True
                target_reached = True
                break
            if gradient_norm <= settings.irc_gradient_tolerance:
                converged = True
                break
            descent = -np.asarray(gradient, dtype=float)
            maximum = float(np.max(np.abs(descent))) if descent.size else 0.0
            step_limit = settings.irc_step_size
            if target_rmsd is not None:
                # Use long IRC strides far from a known minimum and taper them
                # continuously near the RMSD identification threshold.
                step_limit = min(
                    4.0 * settings.irc_step_size,
                    settings.irc_step_size * max(1.0, target_rmsd / 0.25),
                )
            if maximum > step_limit:
                descent *= step_limit / maximum
            accepted: OptimizerEvaluation | None = None
            scale = 1.0
            for reduction in range(10):
                candidate_q = np.asarray(evaluation.q, dtype=float) + scale * descent
                try:
                    candidate = service.evaluate(
                        candidate_q,
                        tag=f"irc-{label}-{step_index + 1:03d}-ls{reduction}",
                        use_cache=True,
                        persist_cache=True,
                        requested_properties=("energy", "gradient"),
                    )
                except RuntimeError:
                    # A long adaptive stride can leave the evaluator's valid
                    # domain; back off and retry the same IRC point.
                    scale *= 0.5
                    continue
                if candidate.energy_hartree < evaluation.energy_hartree - 1.0e-14:
                    accepted = candidate
                    break
                scale *= 0.5
            if accepted is None:
                break
            evaluation = accepted
        endpoints.append(np.asarray(evaluation.q, dtype=float).copy())
        branches.append(
            {
                "direction": label,
                "converged_minimum": converged,
                "target_reached": target_reached,
                "initial_energy_below_saddle": bool(
                    points and float(points[0]["energy_hartree"]) < saddle.energy_hartree
                ),
                "endpoint_energy_hartree": evaluation.energy_hartree,
                "endpoint_q": np.asarray(evaluation.q, dtype=float).tolist(),
                "points": points,
            }
        )
    separated = bool(
        len(endpoints) == 2
        and float(np.linalg.norm(endpoints[1] - endpoints[0])) > settings.irc_step_size
    )
    endpoint_coordinates = tuple(service.coordinates_from_q(endpoint) for endpoint in endpoints)
    target_matches: dict[str, object] = {}
    target_ok = True
    if target_geometries:
        targets = tuple(np.asarray(item, dtype=float) for item in target_geometries)
        if len(targets) != 2 or any(
            item.shape != endpoint_coordinates[0].shape for item in targets
        ):
            raise ValueError("IRC target geometries must contain two n_atoms x 3 structures")
        assignments = ((0, 1), (1, 0))
        scores = []
        for first, second in assignments:
            rmsds = (
                _aligned_rmsd(endpoint_coordinates[0], targets[first]),
                _aligned_rmsd(endpoint_coordinates[1], targets[second]),
            )
            scores.append((sum(rmsds), rmsds, (first, second)))
        _score, rmsds, assignment = min(scores, key=lambda item: item[0])
        target_ok = all(value <= 0.25 for value in rmsds)
        target_matches = {
            "verified": target_ok,
            "rmsd_angstrom": list(rmsds),
            "target_assignment": list(assignment),
            "threshold_angstrom": 0.25,
        }
    verified = bool(
        separated
        and all(bool(branch["converged_minimum"]) for branch in branches)
        and all(bool(branch["initial_energy_below_saddle"]) for branch in branches)
        and target_ok
    )
    report: dict[str, object] = {
        "schema": IRC_VERIFICATION_SCHEMA,
        "verified": verified,
        "endpoints_distinct": separated,
        "branches": branches,
        "target_minima": target_matches,
    }
    path = root / "irc_verification.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, path


def _aligned_rmsd(moving: np.ndarray, reference: np.ndarray) -> float:
    from matrix_chem.geometry_alignment import kabsch_rotation

    left = np.asarray(moving, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise ValueError("IRC endpoint and target geometries must have matching shapes")
    left_centered = left - np.mean(left, axis=0)
    right_centered = right - np.mean(right, axis=0)
    rotation = kabsch_rotation(left_centered, right_centered)
    return float(np.sqrt(np.mean((left_centered @ rotation - right_centered) ** 2)))


def _optimizer_diagnostics(iterations: Sequence[OptimizerIteration]) -> dict[str, object]:
    records = tuple(iterations)
    accepted = sum(1 for item in records if item.status.startswith("accepted"))
    rejected_trials = sum(item.rejected_trial_count for item in records)
    trust_limited = sum(1 for item in records if "trust_limited" in item.step_policy)
    component_limited = sum(1 for item in records if "component_limited" in item.step_policy)
    hessian_updates: dict[str, int] = {}
    for item in records:
        hessian_updates[item.hessian_update_status] = (
            hessian_updates.get(item.hessian_update_status, 0) + 1
        )
    return {
        "accepted_macro_steps": accepted,
        "rejected_trial_count": rejected_trials,
        "trust_limited_steps": trust_limited,
        "component_limited_steps": component_limited,
        "hessian_update_status_counts": hessian_updates,
    }


def read_optimizer_hessian(
    path: Path | str,
    *,
    expected_labels: Sequence[str] | None = None,
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != OPTIMIZER_HESSIAN_SCHEMA:
        raise ValueError(f"unsupported optimizer Hessian schema: {schema}")
    labels = tuple(str(item) for item in payload.get("labels", ()))
    if (
        expected_labels is not None
        and labels
        and labels != tuple(str(item) for item in expected_labels)
    ):
        raise ValueError("optimizer Hessian labels do not match the active coordinate model")
    matrix = np.asarray(payload["hessian"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("optimizer Hessian must be a square matrix")
    if labels and matrix.shape != (len(labels), len(labels)):
        raise ValueError("optimizer Hessian shape does not match stored labels")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("optimizer Hessian contains non-finite values")
    return 0.5 * (matrix + matrix.T)


def read_cartesian_gradient(path: Path | str) -> np.ndarray:
    """Read a Cartesian gradient vector in hartree/bohr for a B-prime transform."""

    target = Path(path)
    if target.suffix.casefold() == ".npy":
        values = np.load(target)
    elif target.suffix.casefold() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in (
                "cartesian_gradient_hartree_per_bohr",
                "gradient_hartree_per_bohr",
                "gradient",
            ):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError("Cartesian-gradient JSON contains no supported gradient key")
        values = np.asarray(payload, dtype=float)
    else:
        values = np.loadtxt(target, dtype=float)
    gradient = np.asarray(values, dtype=float).reshape(-1)
    if gradient.size == 0 or not np.all(np.isfinite(gradient)):
        raise ValueError("Cartesian gradient must be a non-empty finite vector")
    return gradient


def write_optimizer_hessian(
    path: Path | str,
    hessian: np.ndarray,
    model: OptimizerCoordinateModel,
    *,
    source: str,
    q: Sequence[float] | np.ndarray | None = None,
) -> Path:
    target = Path(path)
    matrix = _validate_optimizer_hessian(hessian, len(model.labels))
    payload: dict[str, object] = {
        "schema": OPTIMIZER_HESSIAN_SCHEMA,
        "coordinate_kind": model.kind,
        "labels": list(model.labels),
        "source": str(source),
        "units": "hartree per optimizer-coordinate squared",
        "hessian": matrix.tolist(),
    }
    if q is not None:
        payload["q"] = np.asarray(q, dtype=float).reshape(-1).tolist()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def optimizer_hessian_from_cartesian(
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    model: OptimizerCoordinateModel,
    *,
    cartesian_gradient_hartree_per_bohr: np.ndarray | None = None,
    coordinates_angstrom: np.ndarray | None = None,
    b_prime_parallel_workers: int = 0,
) -> np.ndarray:
    """Transform a Cartesian Hessian into optimizer coordinates.

    External initial guesses use the linear congruence by default. Passing a
    Cartesian gradient explicitly requests the more expensive exact
    curvilinear transformation with ARCHITECT's B-prime tensor; this is
    diagnostic-only for a seed and mandatory for a physical final Hessian.
    """

    cart = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    ncart = model.directions_angstrom.shape[1]
    if cart.shape != (ncart, ncart):
        raise ValueError(f"Cartesian Hessian shape must be {(ncart, ncart)}, got {cart.shape}")
    directions_bohr = np.asarray(model.directions_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    if (
        coordinates_angstrom is not None
        and model.kind == "sonic"
        and model.sonic_definition is not None
    ):
        directions_bohr = _optimizer_directions_at_geometry(
            model,
            model.sonic_definition,
            np.asarray(coordinates_angstrom, dtype=float),
        )
    if cartesian_gradient_hartree_per_bohr is not None:
        if model.kind != "sonic" or model.sonic_definition is None:
            raise ValueError("B-prime Hessian transformation requires a frozen SONIC model")
        from matrix_zaff import curvilinear_internal_hessian_from_cartesian

        definition = model.sonic_definition
        definition_labels = tuple(gic.identifier for gic in definition.gics)
        definition_names = tuple(gic.name for gic in definition.gics)
        source_labels = tuple(model.sonic_labels or model.labels)
        coordinate_indices = tuple(
            _coordinate_index(label, definition_labels, definition_names)
            for label in source_labels
        )
        result = curvilinear_internal_hessian_from_cartesian(
            definition,
            cart,
            cartesian_gradient_hartree_per_bohr,
            coordinates_angstrom=coordinates_angstrom,
            cartesian_from_internal_bohr=directions_bohr.T,
            coordinate_indices=coordinate_indices,
            parallel_workers=b_prime_parallel_workers,
        )
        return _validate_optimizer_hessian(result.hessian_internal, len(model.labels))
    projected = directions_bohr @ (0.5 * (cart + cart.T)) @ directions_bohr.T
    return _validate_optimizer_hessian(projected, len(model.labels))


def _initial_qm_b_prime_correction(
    model: OptimizerCoordinateModel,
    coordinates_angstrom: np.ndarray,
    cartesian_gradient_hartree_per_bohr: np.ndarray | None,
    *,
    optimizer_gradient: np.ndarray | None = None,
    workers: int = 0,
) -> np.ndarray | None:
    """Return the exact geometric-curvature correction for a QM seed.

    The Cartesian Hessian is first imported by linear congruence.  Once LINK
    has evaluated the initial gradient, ARCHITECT adds the exact ``B'`` term
    without another QM call.  Analytic Cartesian gradients are used directly;
    for an energy-only surface the initial SONIC finite-difference gradient is
    lifted to the unique minimum-norm Cartesian tangent covector.
    """

    if model.kind != "sonic" or model.sonic_definition is None:
        return None
    gradient = cartesian_gradient_hartree_per_bohr
    if gradient is None:
        if optimizer_gradient is None:
            return None
        directions_bohr = _optimizer_directions_at_geometry(
            model,
            model.sonic_definition,
            np.asarray(coordinates_angstrom, dtype=float),
        )
        internal = np.asarray(optimizer_gradient, dtype=float).reshape(-1)
        if internal.shape != (directions_bohr.shape[0],):
            raise ValueError("initial optimizer gradient does not match the SONIC model")
        metric = _positive_metric_matrix(
            directions_bohr @ directions_bohr.T,
            1.0e-12,
        )
        gradient = directions_bohr.T @ np.linalg.solve(metric, internal)
    ncart = int(np.asarray(model.directions_angstrom, dtype=float).shape[1])
    return optimizer_hessian_from_cartesian(
        np.zeros((ncart, ncart), dtype=float),
        model,
        cartesian_gradient_hartree_per_bohr=np.asarray(
            gradient, dtype=float
        ),
        coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        b_prime_parallel_workers=workers,
    )


def optimizer_hessian_from_force_field_cartesian(
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    model: OptimizerCoordinateModel,
    *,
    xyzin_path: Path | str,
    coordinates_angstrom: np.ndarray | None = None,
) -> np.ndarray:
    """Project an FF Hessian through a full pseudo-bond SONIC source basis.

    Force-field Hessians have one immutable source-coordinate contract:
    intermolecular connectivity is represented by SMITH pseudo-bonds even
    when the subsequent optimization uses exponential mappings.  The source
    Hessian is reconstructed in its physical Cartesian tangent space and only
    then transformed to the active optimizer SONIC.  QM Hessians must use
    :func:`optimizer_hessian_from_cartesian` directly instead.
    """

    if model.kind == "typed_onic":
        runtime = model.typed_onic_runtime
        if runtime is None:
            raise ValueError("typed-ONIC force-field Hessians require a compiled runtime")
        geometry = np.asarray(
            runtime.definition.reference_coordinates_angstrom
            if coordinates_angstrom is None
            else coordinates_angstrom,
            dtype=float,
        )
        target_directions_bohr = _typed_onic_directions_at_geometry(
            runtime,
            geometry,
        )
        target_definition = None
    elif model.kind == "sonic" and model.sonic_definition is not None:
        target_definition = model.sonic_definition
        geometry = (
            np.asarray(target_definition.reference_coordinates_angstrom, dtype=float)
            if coordinates_angstrom is None
            else np.asarray(coordinates_angstrom, dtype=float)
        )
        target_directions_bohr = _optimizer_directions_at_geometry(
            model, target_definition, geometry
        )
    else:
        raise ValueError("force-field Hessians require a frozen SONIC or typed-ONIC optimizer model")
    from matrix_smith import build_gic_b_matrix, build_gic_definition_from_xyzin
    if model.kind == "typed_onic" and coordinates_angstrom is None:
        geometry = np.asarray(model.typed_onic_runtime.definition.reference_coordinates_angstrom, dtype=float)
    symmetry_group = (
        str(target_definition.point_group)
        if target_definition is not None
        else next(
            (
                str(block.exact_retained_group)
                for block in model.typed_onic_runtime.definition.blocks
                if str(block.exact_retained_group).strip()
            ),
            "C1",
        )
    )
    source_definition = build_gic_definition_from_xyzin(
        Path(xyzin_path),
        symmetrize=bool(symmetry_group.upper() not in {"", "C1", "UNKNOWN"}),
        symmetry_group=symmetry_group,
    )
    source_b_angstrom = np.asarray(
        build_gic_b_matrix(
            source_definition,
            coordinates_angstrom=geometry,
        ).rows,
        dtype=float,
    )
    source_rank = int(np.linalg.matrix_rank(source_b_angstrom, tol=1.0e-9))
    if source_rank != int(source_definition.target_rank):
        raise ValueError(
            "ORACLE-atlas source SONIC is rank deficient: "
            f"rank {source_rank}, target {source_definition.target_rank}"
        )
    source_cartesian_from_internal_bohr = (
        cartesian_from_internal_jacobian(source_b_angstrom, rcond=1.0e-8)
        * ANGSTROM_TO_BOHR
    )
    return _project_cartesian_hessian_through_internal_source(
        cartesian_hessian_hartree_per_bohr2,
        source_cartesian_from_internal_bohr,
        target_directions_bohr,
    )


def _project_cartesian_hessian_through_internal_source(
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    source_cartesian_from_internal_bohr: np.ndarray,
    target_directions_bohr: np.ndarray,
) -> np.ndarray:
    """Apply the source-internal/tangent/target congruence without B-prime."""

    cartesian = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    source_j = np.asarray(source_cartesian_from_internal_bohr, dtype=float)
    target_j_rows = np.asarray(target_directions_bohr, dtype=float)
    if source_j.ndim != 2 or target_j_rows.ndim != 2:
        raise ValueError("source and target Hessian Jacobians must be two-dimensional")
    ncart = source_j.shape[0]
    if cartesian.shape != (ncart, ncart) or target_j_rows.shape[1] != ncart:
        raise ValueError("Cartesian Hessian and source/target Jacobian dimensions differ")
    if not all(np.all(np.isfinite(item)) for item in (cartesian, source_j, target_j_rows)):
        raise ValueError("Hessian source transformation contains non-finite values")
    cartesian = 0.5 * (cartesian + cartesian.T)
    source_hessian = source_j.T @ cartesian @ source_j
    source_b_bohr = internal_from_cartesian_jacobian(source_j, rcond=1.0e-8)
    source_tangent_cartesian = source_b_bohr.T @ source_hessian @ source_b_bohr
    target = target_j_rows @ source_tangent_cartesian @ target_j_rows.T
    return _validate_optimizer_hessian(target, target_j_rows.shape[0])


def _typed_onic_directions_at_geometry(runtime: object, coordinates_angstrom: np.ndarray) -> np.ndarray:
    """Return typed-ONIC Cartesian tangent rows in bohr at one geometry."""

    evaluation = runtime.evaluate(np.asarray(coordinates_angstrom, dtype=float))
    cartesian_from_q = cartesian_from_internal_jacobian(
        evaluation.b_matrix.to_dense(),
        rcond=1.0e-8,
    )
    return np.asarray(cartesian_from_q.T, dtype=float) * ANGSTROM_TO_BOHR


def _optimizer_directions_at_geometry(
    model: OptimizerCoordinateModel,
    definition: object,
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Return ``dx/dq`` rows at the imported-Hessian geometry."""

    from matrix_smith import build_gic_b_matrix

    b_matrix = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    from .fragment_backtransform import direct_fragment_rigid_tangent

    cartesian_from_all = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8)
    fragment_tangent = direct_fragment_rigid_tangent(
        definition,
        np.asarray(coordinates_angstrom, dtype=float),
        b_matrix,
    )
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_all[:, handled_index] = fragment_tangent.cartesian_from_q[:, handled_index]
    labels = tuple(gic.identifier for gic in definition.gics)
    names = tuple(gic.name for gic in definition.gics)
    source_labels = tuple(model.sonic_labels or model.labels)
    indices = tuple(_coordinate_index(label, labels, names) for label in source_labels)
    selected = cartesian_from_all[:, indices]
    transform = model.sonic_from_coordinates
    if transform is not None:
        selected = selected @ np.asarray(transform, dtype=float)
    return np.asarray(selected.T, dtype=float) * ANGSTROM_TO_BOHR


def chemical_optimizer_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str,
    settings: OptimizerSettings | None = None,
) -> np.ndarray:
    """Return the production chemical initial Hessian for an active model."""

    return _initial_optimizer_hessian(
        model,
        settings or OptimizerSettings(),
        initial_hessian=None,
        atoms=tuple(atoms),
        coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        xyzin_path=xyzin_path,
    )


def optimizer_hessian_from_gaussian_hessian(
    path: Path | str,
    model: OptimizerCoordinateModel,
) -> np.ndarray:
    return optimizer_hessian_from_engine_hessian("gaussian", path, model)


def hessian_input_from_engine_hessian(
    engine: str,
    path: Path | str,
    *,
    grd: Path | str | None = None,
    output: Path | str | None = None,
    geometry: Path | str | None = None,
    spectrum: Path | str | None = None,
    input_path: Path | str | None = None,
):
    """Compatibility wrapper around the canonical ``matrix_qm`` dispatcher."""

    from matrix_qm import hessian_input_from_engine

    return hessian_input_from_engine(
        engine,
        path,
        grd=grd,
        output=output,
        geometry=geometry,
        spectrum=spectrum,
        input_path=input_path,
    )


def cartesian_vibrational_hessian_index_from_engine(
    engine: str,
    path: Path | str,
    *,
    grd: Path | str | None = None,
    output: Path | str | None = None,
) -> int:
    """Return the vibrational inertia of an imported Cartesian Hessian.

    Translation and rotation are projected in the canonical mass-weighted
    Cartesian space.  This is the pre-optimization basin test; it is not
    inferred from the reduced optimizer-coordinate Hessian.
    """
    from matrix_qm import cartesian_normal_modes_from_hessian

    data = hessian_input_from_engine_hessian(engine, path, grd=grd, output=output)
    modes = cartesian_normal_modes_from_hessian(
        data.cartesian_hessian,
        data.masses_amu,
        data.cartesian_coordinates_bohr,
        source=f"{engine} initial Cartesian Hessian {path}",
    )
    return int(np.count_nonzero(np.asarray(modes.eigenvalues, dtype=float) < -1.0e-12))


def optimizer_hessian_from_engine_hessian(
    engine: str,
    path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    grd: Path | str | None = None,
    output: Path | str | None = None,
    cartesian_gradient_hartree_per_bohr: np.ndarray | None = None,
    use_b_prime: bool = False,
    b_prime_parallel_workers: int = 0,
    hessian_kind: str = "qm",
    xyzin_path: Path | str | None = None,
) -> np.ndarray:
    normalized_kind = str(hessian_kind).strip().lower().replace("-", "_")
    if normalized_kind not in {"qm", "force_field"}:
        raise ValueError("hessian_kind must be 'qm' or 'force_field'")
    if normalized_kind == "force_field" and xyzin_path is None:
        raise ValueError("force-field Hessian import requires xyzin_path")
    if normalized_kind == "force_field" and use_b_prime:
        raise ValueError("B-prime is not part of the force-field seed contract")
    data = hessian_input_from_engine_hessian(engine, path, grd=grd, output=output)
    hessian = np.asarray(data.cartesian_hessian, dtype=float)
    gradient = (
        None
        if cartesian_gradient_hartree_per_bohr is None
        else np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    )
    coordinates_angstrom = (
        np.asarray(data.cartesian_coordinates_bohr, dtype=float) / ANGSTROM_TO_BOHR
    )
    if model.kind == "sonic" and model.sonic_definition is not None:
        reference = np.asarray(model.sonic_definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom.shape != reference.shape:
            raise ValueError("Imported Hessian geometry does not match the SONIC atom count")
        rotation = kabsch_rotation(coordinates_angstrom, reference)
        rotation_gradient = (
            np.zeros(coordinates_angstrom.size, dtype=float) if gradient is None else gradient
        )
        rotated_gradient, rotated_hessian = rotate_cartesian_derivatives(
            rotation_gradient,
            rotation,
            hessian,
        )
        assert rotated_hessian is not None
        hessian = rotated_hessian
        gradient = None if gradient is None else rotated_gradient
        coordinates_angstrom = kabsch_align(coordinates_angstrom, reference)
    if use_b_prime and gradient is None:
        raise ValueError("B-prime Hessian transformation requires a Cartesian gradient")
    if normalized_kind == "force_field":
        assert xyzin_path is not None
        return optimizer_hessian_from_force_field_cartesian(
            hessian,
            model,
            xyzin_path=xyzin_path,
            coordinates_angstrom=coordinates_angstrom,
        )
    return optimizer_hessian_from_cartesian(
        hessian,
        model,
        cartesian_gradient_hartree_per_bohr=(gradient if use_b_prime else None),
        coordinates_angstrom=coordinates_angstrom,
        b_prime_parallel_workers=b_prime_parallel_workers,
    )


def build_optimizer_hessian_from_gaussian_job(
    xyzin_path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    run_dir: Path | str,
    route: str = "#p HF/STO-3G Freq",
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
) -> tuple[np.ndarray, Path]:
    """Run a Gaussian frequency job and project its Cartesian Hessian into optimizer coordinates."""

    from matrix_gaussian import (
        formchk_checkpoint,
        hessian_input_from_gaussian_log,
        read_gaussian_fchk,
        run_gaussian_job,
        write_gaussian_point_input,
    )

    geometry = read_xyzin_geometry(Path(xyzin_path))
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "optimizer_hessian_seed.chk"
    input_path = write_gaussian_point_input(
        root / "gauin.gjf",
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        route=route,
        title="MATRIX optimizer Hessian seed",
        charge=charge,
        multiplicity=multiplicity,
        ensure_force=False,
        link0=(f"%chk={checkpoint_path.name}",),
    )
    run = run_gaussian_job(
        root,
        executable=executable,
        input_path=input_path,
        timeout=timeout,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    if checkpoint_path.is_file():
        fchk_path = formchk_checkpoint(
            checkpoint_path,
            root / "optimizer_hessian_seed.fchk",
            timeout=timeout,
        )
        fchk = read_gaussian_fchk(fchk_path)
        cartesian_hessian = fchk.to_hessian_input().cartesian_hessian
        fchk_coordinates_angstrom = (
            np.asarray(fchk.cartesian_coordinates_bohr, dtype=float) * 0.529177210903
        )
        target_coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
        rotation = kabsch_rotation(fchk_coordinates_angstrom, target_coordinates)
        _gradient, rotated_hessian = rotate_cartesian_derivatives(
            np.zeros(target_coordinates.size, dtype=float),
            rotation,
            cartesian_hessian,
        )
        assert rotated_hessian is not None
        cartesian_hessian = rotated_hessian
    else:
        # Minimal test doubles may not implement Gaussian checkpoints. Keep the
        # log archive as their compatibility path; production GDV/G16 jobs use
        # the full-precision checkpoint branch above.
        data = hessian_input_from_gaussian_log(run.log_path)
        cartesian_hessian = data.cartesian_hessian
    return optimizer_hessian_from_cartesian(cartesian_hessian, model), run.log_path


def build_optimizer_hessian_seed(
    xyzin_path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    engine: str,
    run_dir: Path | str,
    route: str = "",
    method: str = "",
    basis: str = "",
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    engine_command: str = "",
    hessian_path: Path | str | None = None,
    hessian_kind: str = "qm",
) -> tuple[np.ndarray, Path]:
    """Build a low-level Hessian seed, then project it into optimizer coordinates.

    ``hessian_kind`` is an explicit scientific source declaration. ``qm``
    uses the optimization coordinates directly; ``force_field`` enforces the
    frozen ORACLE-atlas source-coordinate contract before the target transformation.
    """

    name = _normalized_hessian_engine(engine)
    normalized_kind = str(hessian_kind).strip().lower().replace("-", "_")
    if normalized_kind not in {"qm", "force_field"}:
        raise ValueError("hessian_kind must be 'qm' or 'force_field'")
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if engine_command.strip():
        target = (
            Path(hessian_path)
            if hessian_path is not None
            else root / _default_hessian_output_name(name)
        )
        command = _format_command(
            engine_command,
            xyzin=Path(xyzin_path),
            workdir=root,
            hessian=target,
            output=target,
        )
        require_authorized_descendant_calculation(
            backend="LINK/external-hessian-provider",
            input_path=xyzin_path,
            command=command,
            workdir=root,
        )
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout.strip())
        projected = optimizer_hessian_from_engine_hessian(
            name,
            target,
            model,
            hessian_kind=normalized_kind,
            xyzin_path=xyzin_path,
        )
        return projected, target
    if name == "xtb":
        return _build_xtb_optimizer_hessian_seed(
            xyzin_path,
            model,
            run_dir=root,
            route=route or "--gfn 2",
            executable=executable,
            timeout=timeout,
            hessian_kind=normalized_kind,
        )
    if name == "gaussian":
        hessian, log_path = build_optimizer_hessian_from_gaussian_job(
            xyzin_path,
            model,
            run_dir=root,
            route=route or "#p HF/STO-3G Freq",
            charge=charge,
            multiplicity=multiplicity,
            executable=executable,
            timeout=timeout,
        )
        return hessian, log_path
    if name == "orca":
        return _build_orca_optimizer_hessian_seed(
            xyzin_path,
            model,
            run_dir=root,
            route=route or "HF STO-3G Freq",
            charge=charge,
            multiplicity=multiplicity,
            executable=executable,
            timeout=timeout,
        )
    if name == "cfour":
        return _build_cfour_optimizer_hessian_seed(
            xyzin_path,
            model,
            run_dir=root,
            method=method or "HF",
            basis=basis or "STO-3G",
            charge=charge,
            multiplicity=multiplicity,
            executable=executable,
            timeout=timeout,
        )
    raise ValueError(
        f"automatic Hessian seed jobs for {engine!r} need --initial-hessian-seed-command; "
        "MATRIX can still read the resulting Hessian through --initial-hessian-engine/--initial-hessian-file"
    )


def _build_xtb_optimizer_hessian_seed(
    xyzin_path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    run_dir: Path,
    route: str,
    executable: str | None,
    timeout: float | None,
    hessian_kind: str = "qm",
) -> tuple[np.ndarray, Path]:
    """Run xTB Hessian generation on the exact geometry consumed by LINK."""

    from matrix_chem import write_xyz
    from matrix_xtb import hessian_input_from_xtb_files, run_xtb_job

    geometry_path = Path(xyzin_path)
    geometry = read_xyzin_geometry(geometry_path)
    seed_input = run_dir / "input.xyz"
    write_xyz(
        seed_input,
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        comment="MATRIX xTB Hessian seed",
    )
    output_path = run_dir / "xtb.out"
    run = run_xtb_job(
        run_dir,
        executable=executable,
        input_path=seed_input,
        output_path=output_path,
        timeout=timeout,
        extra_args=tuple(shlex.split(route)) + ("--hess",),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    hessian_path = run_dir / "hessian"
    data = hessian_input_from_xtb_files(
        hessian_path,
        geometry=seed_input,
        output=output_path,
    )
    projected = (
        optimizer_hessian_from_force_field_cartesian(
            data.cartesian_hessian,
            model,
            xyzin_path=xyzin_path,
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        )
        if hessian_kind == "force_field"
        else optimizer_hessian_from_cartesian(data.cartesian_hessian, model)
    )
    return projected, hessian_path


def _build_orca_optimizer_hessian_seed(
    xyzin_path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    run_dir: Path,
    route: str,
    charge: int,
    multiplicity: int,
    executable: str | None,
    timeout: float | None,
) -> tuple[np.ndarray, Path]:
    from matrix_orca import hessian_input_from_orca_output, run_orca_job, write_orca_point_input

    geometry = read_xyzin_geometry(Path(xyzin_path))
    input_path = write_orca_point_input(
        run_dir / "orca.inp",
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        route=route,
        charge=charge,
        multiplicity=multiplicity,
    )
    run = run_orca_job(
        run_dir,
        executable=executable,
        input_path=input_path,
        timeout=timeout,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    data = hessian_input_from_orca_output(run.output_path)
    return optimizer_hessian_from_cartesian(data.cartesian_hessian, model), run.output_path


def _build_cfour_optimizer_hessian_seed(
    xyzin_path: Path | str,
    model: OptimizerCoordinateModel,
    *,
    run_dir: Path,
    method: str,
    basis: str,
    charge: int,
    multiplicity: int,
    executable: str | None,
    timeout: float | None,
) -> tuple[np.ndarray, Path]:
    from matrix_cfour import hessian_input_from_cfour_files, run_cfour_job, write_cfour_point_input

    geometry = read_xyzin_geometry(Path(xyzin_path))
    input_path = write_cfour_point_input(
        run_dir / "ZMAT",
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
    )
    text = input_path.read_text(encoding="utf-8")
    input_path.write_text(text.replace("DERIV_LEVEL=FIRST", "DERIV_LEVEL=SECOND"), encoding="utf-8")
    run = run_cfour_job(
        run_dir,
        executable=executable,
        input_path=input_path,
        timeout=timeout,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    fcmfinal = run_dir / "FCMFINAL"
    data = hessian_input_from_cfour_files(
        fcmfinal,
        grd=run_dir / "GRD",
        output=run.output_path,
    )
    return optimizer_hessian_from_cartesian(data.cartesian_hessian, model), fcmfinal


def _normalized_hessian_engine(engine: str) -> str:
    normalized = str(engine).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "xyzin": "xyzin",
        "matrix": "xyzin",
        "gaussian": "gaussian",
        "g16": "gaussian",
        "gdv": "gaussian",
        "gaussian16": "gaussian",
        "orca": "orca",
        "molpro": "molpro",
        "mrcc": "mrcc",
        "cfour": "cfour",
        "cfourfcmfinal": "cfour",
        "xtb": "xtb",
        "pyscf": "pyscf",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Hessian engine: {engine}") from exc


def _default_hessian_output_name(engine: str) -> str:
    if engine == "cfour":
        return "FCMFINAL"
    if engine == "orca":
        return "orca.out"
    if engine == "gaussian":
        return "gau.log"
    if engine == "molpro":
        return "molpro.out"
    if engine == "mrcc":
        return "mrcc.out"
    if engine == "xtb":
        return "hessian"
    return "hessian.out"


def _signed_energy_change(old_energy: float, new_energy: float) -> float:
    """Return ``E_new - E_old`` for convergence and trace diagnostics."""

    old = float(old_energy)
    new = float(new_energy)
    if not math.isfinite(old) or not math.isfinite(new):
        raise ValueError("energy change requires finite energies")
    return new - old


def _positive_metric_matrix(metric: np.ndarray, absolute_floor: float) -> np.ndarray:
    """Return a symmetric positive metric without discarding its couplings."""

    matrix = np.asarray(metric, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("optimizer metric must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("optimizer metric contains non-finite values")
    matrix = 0.5 * (matrix + matrix.T)
    if matrix.size == 0:
        return matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    maximum = max(float(np.max(np.abs(eigenvalues))), float(absolute_floor))
    floor = max(float(absolute_floor), 1.0e-12 * maximum)
    positive = np.maximum(eigenvalues, floor)
    result = eigenvectors @ np.diag(positive) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _initial_optimizer_hessian(
    model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    *,
    initial_hessian: np.ndarray | None,
    atoms: Sequence[str] = (),
    coordinates_angstrom: np.ndarray | None = None,
    xyzin_path: Path | str | None = None,
) -> np.ndarray:
    matrix, _source = _initial_optimizer_hessian_selection(
        model,
        settings,
        initial_hessian=initial_hessian,
        atoms=atoms,
        coordinates_angstrom=coordinates_angstrom,
        xyzin_path=xyzin_path,
    )
    return matrix


def _initial_optimizer_hessian_selection(
    model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    *,
    initial_hessian: np.ndarray | None,
    atoms: Sequence[str] = (),
    coordinates_angstrom: np.ndarray | None = None,
    xyzin_path: Path | str | None = None,
) -> tuple[np.ndarray, str]:
    """Return the initial Hessian and its exact deterministic provenance."""

    if initial_hessian is None:
        if model.kind == "sonic" and atoms and coordinates_angstrom is not None:
            # A rank-one active SONIC space has no meaningful Hessian block to
            # reconstruct: its sole diagonal curvature is sufficient for the
            # first RFO step.  Use the existing force-field seed machinery,
            # transformed onto that coordinate, and avoid an unnecessary
            # complete Hessian acquisition (in particular, never route this
            # case through GDV).  This is a mathematical rank criterion, not a
            # molecule-, backend- or symmetry-specific exception.
            if len(model.labels) == 1:
                almloef = _fischer_almloef_hessian(
                    model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
                )
                if _initial_model_hessian_passes_audit(almloef, settings):
                    assert almloef is not None
                    return almloef, "fischer-almloef-rank-one-active-sonic-seed"
                chemical = _chemical_valence_hessian(
                    model, atoms, coordinates_angstrom, settings, xyzin_path=xyzin_path
                )
                if chemical is not None and _initial_model_hessian_passes_audit(
                    chemical, settings
                ):
                    return chemical, "chemical-valence-rank-one-active-sonic-seed"
            if settings.initial_hessian_model in {"auto", "lindh_swart_special"}:
                lindh_swart = _lindh_swart_special_hessian(
                    model,
                    atoms,
                    coordinates_angstrom,
                    xyzin_path=xyzin_path,
                )
                if _initial_model_hessian_passes_audit(lindh_swart, settings):
                    assert lindh_swart is not None
                    return (
                        lindh_swart,
                        "lindh-1995-anc-plus-swart-special-oracle-atlas-source-congruence",
                    )
                almloef = _fischer_almloef_hessian(
                    model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
                )
                if _initial_model_hessian_passes_audit(almloef, settings):
                    assert almloef is not None
                    return almloef, "fischer-almloef-oracle-atlas-source-fallback"
                raise ValueError(
                    "Lindh--Swart-special and Fischer--Almloef SONIC Hessian seeds failed audit"
                )
            if settings.initial_hessian_model == "berny":
                berny = _berny_geometric_fragment_hessian(
                    model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
                )
                if berny is None:
                    berny = _gaussian_berny_pseudobond_hessian(
                        model,
                        atoms,
                        coordinates_angstrom,
                        xyzin_path=xyzin_path,
                        require_pseudobonds=False,
                    )
                if berny is None:
                    raise ValueError("Berny Hessian needs a frozen SONIC definition")
                return berny, "berny-explicit-initial-hessian"
            if settings.initial_hessian_model == "almloef":
                almloef = _fischer_almloef_hessian(
                    model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
                )
                if almloef is None:
                    raise ValueError("Fischer-Almloef Hessian needs a frozen SONIC definition")
                return almloef, "fischer-almloef-explicit-oracle-atlas-source"
            berny = _gaussian_berny_pseudobond_hessian(
                model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
            )
            if berny is not None:
                return berny, "berny-pseudobond-initial-hessian"
            hybrid = _berny_geometric_fragment_hessian(
                model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
            )
            if hybrid is not None:
                return hybrid, "berny-geometric-fragment-initial-hessian"
            chemical = _chemical_valence_hessian(
                model, atoms, coordinates_angstrom, settings, xyzin_path=xyzin_path
            )
            if chemical is not None:
                return chemical, "chemical-valence-initial-hessian"
        if atoms and coordinates_angstrom is not None:
            chemical = _chemical_valence_hessian(
                model, atoms, coordinates_angstrom, settings, xyzin_path=xyzin_path
            )
            if chemical is not None:
                return chemical, "chemical-valence"
        return (
            np.diag(
                np.maximum(np.abs(model.metric_diagonal), settings.min_abs_metric_diagonal)
            ),
            "metric-diagonal-nonsonic-fallback",
        )
    matrix = _validate_optimizer_hessian(initial_hessian, len(model.labels))
    diagonal = np.diag(matrix).copy()
    small = np.abs(diagonal) < settings.min_abs_metric_diagonal
    if np.any(small):
        matrix = matrix.copy()
        for index in np.flatnonzero(small):
            sign = 1.0 if diagonal[index] >= 0.0 else -1.0
            matrix[index, index] = sign * settings.min_abs_metric_diagonal
    return 0.5 * (matrix + matrix.T), "caller-provided-initial-hessian"


def _initial_model_hessian_passes_audit(
    hessian: np.ndarray | None,
    settings: OptimizerSettings,
) -> bool:
    """Apply the common finite, positive-curvature and conditioning audit."""

    if hessian is None:
        return False
    matrix = np.asarray(hessian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return False
    if not eigenvalues.size:
        return True
    maximum = float(np.max(eigenvalues))
    minimum = float(np.min(eigenvalues))
    if minimum <= 0.0 or maximum <= 0.0:
        return False
    return bool(maximum / minimum <= settings.max_hessian_condition)


def _rebuild_optimizer_hessian_at_geometry(
    model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    previous_hessian: np.ndarray,
    *,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    xyzin_path: Path | str | None,
) -> tuple[np.ndarray, str]:
    """Rebuild the declared chemical seed at the current geometry.

    A model reset must never manufacture a diagonal from the coordinate
    metric: the metric is not a force-constant model.  When an in-memory
    custom coordinate model has no frozen chemical definition, retaining the
    last finite physical model is the only non-invented fallback and is made
    explicit in the returned provenance string.
    """

    previous = 0.5 * (
        np.asarray(previous_hessian, dtype=float) + np.asarray(previous_hessian, dtype=float).T
    )
    try:
        rebuilt = _initial_optimizer_hessian(
            model,
            settings,
            initial_hessian=None,
            atoms=tuple(atoms),
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
            xyzin_path=xyzin_path,
        )
    except (ImportError, OSError, ValueError, np.linalg.LinAlgError):
        return previous, "canonical_seed_unavailable_previous_model_preserved"
    rebuilt = 0.5 * (np.asarray(rebuilt, dtype=float) + np.asarray(rebuilt, dtype=float).T)
    if not _stored_hessian_is_numerically_usable(rebuilt, settings):
        return previous, "canonical_seed_invalid_previous_model_preserved"
    return rebuilt, "canonical_chemical_seed_rebuilt_at_current_geometry"




def _lindh_swart_special_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None,
) -> np.ndarray | None:
    """Build the canonical Lindh ANC seed plus special Swart coordinates.

    Ordinary atom-based curvature is the general Cartesian Lindh-1995 ANC
    model. Swart--Bickelhaupt terms are added only for declared special
    interactions, using the frozen ORACLE-atlas source chart. The combined Cartesian form is point-group averaged
    and transformed once to the active SONIC tangent.
    """

    if xyzin_path is None or model.sonic_definition is None:
        return None
    try:
        from matrix_chem import lindh_1995_cartesian_hessian
        from matrix_smith import (
            build_primitive_b_matrix,
            construct_gic_definition_from_xyzin,
        )

        active_definition = model.sonic_definition
        source_definition, source_atoms, operations = construct_gic_definition_from_xyzin(
            Path(xyzin_path),
            symmetry_group=str(active_definition.point_group),
            fragment_context=_frozen_smith_fragment_context(Path(xyzin_path)),
            retain_candidate_primitives=False,
        )
    except (ImportError, OSError, ValueError):
        return None
    if tuple(source_atoms) != tuple(atoms):
        return None
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3):
        return None
    try:
        lindh = lindh_1995_cartesian_hessian(
            tuple(atoms),
            coords,
            excluded_valence_pairs=_center_interaction_lindh_valence_pairs(
                Path(xyzin_path)
            ),
        )
        special_indices = tuple(
            index
            for index, primitive in enumerate(source_definition.primitives)
            if _is_swart_special_primitive(
                primitive,
                definition=source_definition,
            )
        )
        curvatures = np.asarray(
            [
                _swart_lindh_primitive_curvature(
                    source_definition.primitives[index],
                    atoms=tuple(atoms),
                    coordinates_angstrom=coords,
                    definition=source_definition,
                    xyzin_path=Path(xyzin_path),
                )
                for index in special_indices
            ],
            dtype=float,
        )
        all_primitive_b = np.asarray(
            build_primitive_b_matrix(
                source_definition,
                coordinates_angstrom=coords,
            ).rows,
            dtype=float,
        )
    except (FloatingPointError, OSError, ValueError):
        return None
    primitive_b = all_primitive_b[np.asarray(special_indices, dtype=int)]
    if (
        primitive_b.shape != (len(curvatures), coords.size)
        or not np.all(np.isfinite(curvatures))
        or np.any(curvatures <= 0.0)
    ):
        return None
    special_cartesian_hessian_angstrom = (
        primitive_b.T @ (curvatures[:, None] * primitive_b)
        if special_indices
        else np.zeros((coords.size, coords.size), dtype=float)
    )
    cartesian_hessian = np.asarray(
        lindh.effective_hessian_hartree_per_bohr2,
        dtype=float,
    ) + special_cartesian_hessian_angstrom / (ANGSTROM_TO_BOHR**2)
    cartesian_hessian = _point_group_average_cartesian_hessian(
        cartesian_hessian,
        operations=tuple(operations),
        natoms=len(atoms),
    )
    active_directions_bohr = _optimizer_directions_at_geometry(
        model,
        active_definition,
        coords,
    )
    active_hessian = (
        active_directions_bohr
        @ cartesian_hessian
        @ active_directions_bohr.T
    )
    return _validate_optimizer_hessian(active_hessian, len(model.labels))


def _center_interaction_lindh_valence_pairs(
    xyzin_path: Path,
) -> tuple[tuple[int, int], ...]:
    """Return atom spokes replaced by declared center-based curvature."""

    try:
        from matrix_fragments import read_interaction_center_definition

        definition = read_interaction_center_definition(Path(xyzin_path))
    except (ImportError, OSError, ValueError):
        return ()
    centers = {
        str(center.identifier): center
        for center in getattr(definition, "centers", ())
    }
    pairs: set[tuple[int, int]] = set()
    for interaction in getattr(definition, "interactions", ()):
        kind = str(getattr(interaction, "kind", "")).upper()
        if not (
            kind.startswith("METAL_ETA")
            or kind
            in {
                "ATOM_BOND_CENTER",
                "ATOM_RING_CENTER",
                "ATOM_HAPTIC_CENTER",
            }
        ):
            continue
        center = centers.get(str(getattr(interaction, "center_id", "")))
        if center is None:
            continue
        center_kind = str(getattr(center, "kind", "")).upper()
        if center_kind not in {"BOND_CENTER", "RING_CENTER", "HAPTIC_CENTER"}:
            continue
        atom = int(getattr(interaction, "atom")) - 1
        for member in getattr(center, "atoms", ()):
            donor = int(member) - 1
            if atom != donor:
                pairs.add(tuple(sorted((atom, donor))))
    return tuple(sorted(pairs))


def _is_swart_special_primitive(primitive: object, *, definition: object) -> bool:
    """Select coordinates attached to SMITH's declared special graph."""

    from matrix_smith import SPECIAL_REDUCTION_CLASS, primitive_reduction_class

    family = str(getattr(primitive, "family", ""))
    if primitive_reduction_class(family) == SPECIAL_REDUCTION_CLASS:
        return True
    pseudo_bonds = {
        tuple(sorted((int(left), int(right))))
        for left, right in getattr(definition, "pseudo_bonds", ())
    }
    if not pseudo_bonds:
        return False
    support = {
        int(atom)
        for attribute in ("atoms", "ref_atoms", "frame_atoms", "ref_frame_atoms")
        for atom in getattr(primitive, attribute, ())
    }
    return any(left in support and right in support for left, right in pseudo_bonds)


def _swart_lindh_primitive_curvature(
    primitive: object,
    *,
    atoms: tuple[str, ...],
    coordinates_angstrom: np.ndarray,
    definition: object,
    xyzin_path: Path,
) -> float:
    """Return one Swart--Lindh source curvature in SMITH coordinate units."""

    from matrix_chem import (
        DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER,
        swart_lindh_center_screening,
        swart_lindh_force_constant,
        swart_lindh_screening,
    )
    from matrix_chem.topology.covalent_radii import covalent_radius
    from matrix_chem.topology.elements import atomic_number

    coords = np.asarray(coordinates_angstrom, dtype=float)
    radii = tuple(
        float(covalent_radius(int(atomic_number(symbol) or 0)) or 0.75)
        for symbol in atoms
    )
    pseudo_bonds = {
        tuple(sorted((int(left), int(right))))
        for left, right in getattr(definition, "pseudo_bonds", ())
    }

    def screening(left: int, right: int) -> float:
        pair = tuple(sorted((int(left), int(right))))
        distance = float(np.linalg.norm(coords[left - 1] - coords[right - 1]))
        order = (
            DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER if pair in pseudo_bonds else 1.0
        )
        return swart_lindh_screening(
            distance,
            radii[left - 1] + radii[right - 1],
            effective_order=order,
        )

    def basic(function: str, indices: tuple[int, ...]) -> float:
        if function == "R" and len(indices) == 2:
            return swart_lindh_force_constant("bond", (screening(*indices),)) * (
                ANGSTROM_TO_BOHR**2
            )
        if function in {"A", "L"} and len(indices) >= 3:
            left, center, right = indices[:3]
            return swart_lindh_force_constant(
                "angle",
                (screening(left, center), screening(center, right)),
            )
        if function == "D" and len(indices) >= 4:
            first, second, third, fourth = indices[:4]
            return swart_lindh_force_constant(
                "dihedral",
                (
                    screening(first, second),
                    screening(second, third),
                    screening(third, fourth),
                ),
            )
        if function in {"U", "IMPD"} and len(indices) == 4:
            center, first, second, third = indices
            return swart_lindh_force_constant(
                "improper",
                (
                    screening(center, first),
                    screening(center, second),
                    screening(center, third),
                ),
            )
        raise ValueError(f"unsupported Swart--Lindh primitive {function}/{indices}")

    function = str(getattr(primitive, "function", "")).upper()
    indices = tuple(int(index) for index in getattr(primitive, "atoms", ()))
    if function in {"R", "A", "L", "D", "U", "IMPD"}:
        return basic(function, indices)
    if function == "RPCB":
        return float(
            sum(
                coefficient * coefficient * basic("A", component)
                for coefficient, component in _encoded_component_terms(
                    getattr(primitive, "refs", ()), arity=3
                )
            )
        )
    if function == "RPCK":
        return float(
            sum(
                coefficient * coefficient * basic("D", component)
                for coefficient, component in _encoded_component_terms(
                    getattr(primitive, "refs", ()), arity=4
                )
            )
        )

    special_order = _swart_lindh_special_effective_order(
        primitive,
        xyzin_path=xyzin_path,
        default=DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER,
    )
    special_screening = swart_lindh_center_screening(effective_order=special_order)
    if function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}:
        return swart_lindh_force_constant("bond", (special_screening,)) * (
            ANGSTROM_TO_BOHR**2
        )
    if function == "FTRANS":
        # Preserve the established geomeTRIC/LINK translation scale while
        # allowing the declared special-edge order to tune weak/strong cases.
        return (
            0.025
            * special_order
            / DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
            * ANGSTROM_TO_BOHR**2
        )
    if function == "FROT":
        return 0.025 * special_order / DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
    raise ValueError(f"unsupported Swart--Lindh primitive function {function!r}")


def _swart_lindh_special_effective_order(
    primitive: object,
    *,
    xyzin_path: Path,
    default: float,
) -> float:
    """Resolve an explicit center order, otherwise return the frozen default."""

    try:
        from matrix_fragments import read_interaction_center_definition

        interaction_definition = read_interaction_center_definition(xyzin_path)
    except (ImportError, OSError, ValueError):
        return float(default)
    refs = tuple(str(item) for item in getattr(primitive, "refs", ()))
    center_id = refs[0] if refs else ""
    exact = tuple(
        float(getattr(interaction, "effective_order", default))
        for interaction in getattr(interaction_definition, "interactions", ())
        if str(getattr(interaction, "center_id", "")) == center_id
    )
    if exact:
        return float(max(exact))
    haptic = tuple(
        float(getattr(interaction, "effective_order", default))
        for interaction in getattr(interaction_definition, "interactions", ())
        if str(getattr(interaction, "kind", "")).upper().startswith("METAL_ETA")
        or str(getattr(interaction, "kind", "")).upper() == "ATOM_RING_CENTER"
    )
    return float(np.exp(np.mean(np.log(haptic)))) if haptic else float(default)


def _point_group_average_cartesian_hessian(
    hessian: np.ndarray,
    *,
    operations: tuple[object, ...],
    natoms: int,
) -> np.ndarray:
    """Project a Cartesian quadratic form onto the frozen point group."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if len(operations) <= 1:
        return matrix
    transformed: list[np.ndarray] = []
    for operation in operations:
        rotation = np.asarray(getattr(operation, "rotation"), dtype=float)
        permutation = tuple(int(item) for item in getattr(operation, "permutation"))
        cartesian = np.zeros((3 * natoms, 3 * natoms), dtype=float)
        for source_index, target_atom in enumerate(permutation):
            target_index = target_atom - 1
            cartesian[
                3 * source_index : 3 * source_index + 3,
                3 * target_index : 3 * target_index + 3,
            ] = rotation
        transformed.append(cartesian.T @ matrix @ cartesian)
    averaged = np.mean(np.stack(transformed), axis=0)
    return 0.5 * (averaged + averaged.T)


def _fischer_almloef_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None,
) -> np.ndarray | None:
    """Fischer--Almloef diagonal primitive model transformed to frozen SONIC.

    T. H. Fischer and J. Almloef, J. Phys. Chem. 96 (1992) 9768--9774,
    DOI 10.1021/j100203a036. Distances and covalent radii are evaluated in
    bohr, as in the published atomic-unit parameterization.
    """

    if xyzin_path is None:
        return None
    try:
        from matrix_chem.topology.covalent_radii import covalent_radius
        from matrix_chem.topology.elements import atomic_number
        from matrix_smith import construct_gic_definition_from_xyzin

        active_definition = model.sonic_definition
        if active_definition is None:
            return None
        definition, source_atoms, _operations = construct_gic_definition_from_xyzin(
            Path(xyzin_path),
            symmetry_group=str(active_definition.point_group),
            fragment_context=_frozen_smith_fragment_context(Path(xyzin_path)),
            retain_candidate_primitives=False,
        )
    except (ImportError, OSError, ValueError):
        return None
    if tuple(source_atoms) != tuple(atoms):
        return None
    geometry_angstrom = np.asarray(coordinates_angstrom, dtype=float)
    coords = geometry_angstrom * ANGSTROM_TO_BOHR
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    radii = tuple(float(covalent_radius(z) or 0.75) * ANGSTROM_TO_BOHR for z in numbers)
    connectivity = np.zeros((len(atoms), len(atoms)), dtype=int)
    pseudo_bonds = {tuple(sorted((int(i) - 1, int(j) - 1))) for i, j in definition.pseudo_bonds}
    for primitive in definition.primitives:
        if str(primitive.function).upper() == "R" and len(primitive.atoms) == 2:
            i, j = (int(index) - 1 for index in primitive.atoms)
            connectivity[i, j] = connectivity[j, i] = 1

    def distance(i: int, j: int) -> float:
        return float(np.linalg.norm(coords[i] - coords[j]))

    def rcov(i: int, j: int) -> float:
        return radii[i] + radii[j]

    def curvature(primitive) -> float:
        function = str(primitive.function).upper()
        idx = tuple(int(index) - 1 for index in primitive.atoms)
        edges = tuple(tuple(sorted(pair)) for pair in zip(idx, idx[1:]))
        if pseudo_bonds.intersection(edges):
            return _gaussian_gredfc_primitive_curvature(
                primitive, numbers, np.asarray(coordinates_angstrom, dtype=float)
            )
        if function == "R" and len(idx) == 2:
            i, j = idx
            # Fischer--Almloef parameterizes radial curvature in Eh/bohr^2,
            # whereas frozen SMITH/LINK stretch coordinates are in angstrom.
            # Apply the exact Hessian congruence dq_bohr/dq_angstrom squared.
            return (
                0.3601
                * math.exp(-1.944 * (distance(i, j) - rcov(i, j)))
                * ANGSTROM_TO_BOHR**2
            )
        if function in {"A", "L"} and len(idx) >= 3:
            i, j, k = idx[:3]
            return 0.089 + 0.11 * (rcov(i, j) * rcov(j, k)) ** 0.42 * math.exp(
                -0.44 * (distance(i, j) + distance(j, k) - rcov(i, j) - rcov(j, k))
            )
        if function in {"D", "IMPD"} and len(idx) >= 4:
            _i, j, k, _l = idx[:4]
            degree = max(int(connectivity[j].sum() + connectivity[k].sum() - 2), 1)
            return 0.0015 + 14.0 * degree**0.57 / (distance(j, k) * rcov(j, k)) ** 4 * math.exp(
                -2.85 * (distance(j, k) - rcov(j, k))
            )
        if function == "U" and len(idx) == 4:
            center, left, right, out = idx
            u = coords[out] - coords[center]
            v = coords[left] - coords[center]
            w = coords[right] - coords[center]
            normal = np.cross(v, w)
            sine = float(
                np.dot(u, normal) / max(np.linalg.norm(u) * np.linalg.norm(normal), 1.0e-15)
            )
            value = -math.asin(float(np.clip(sine, -1.0, 1.0)))
            return 0.0025 + 0.0061 * (rcov(center, left) * rcov(center, right)) ** 0.80 * math.cos(
                value
            ) ** 4 * math.exp(-3.0 * (distance(center, out) - rcov(center, out)))
        if function == "RPCB":
            from types import SimpleNamespace

            return float(
                sum(
                    coefficient
                    * coefficient
                    * curvature(SimpleNamespace(function="A", atoms=component))
                    for coefficient, component in _encoded_component_terms(
                        getattr(primitive, "refs", ()), arity=3
                    )
                )
            )
        if function == "RPCK":
            from types import SimpleNamespace

            return float(
                sum(
                    coefficient
                    * coefficient
                    * curvature(SimpleNamespace(function="D", atoms=component))
                    for coefficient, component in _encoded_component_terms(
                        getattr(primitive, "refs", ()), arity=4
                    )
                )
            )
        if function == "FTRANS":
            return 0.025 * ANGSTROM_TO_BOHR**2
        if function == "FROT":
            return 0.025
        if function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}:
            return 0.05 * ANGSTROM_TO_BOHR**2
        raise ValueError(f"Fischer-Almloef has no primitive assignment for {primitive.function}")

    curvatures = np.asarray(
        [curvature(primitive) for primitive in definition.primitives],
        dtype=float,
    )
    return _primitive_diagonal_hessian_to_active(
        model,
        definition,
        curvatures,
        coordinates_angstrom=geometry_angstrom,
    )


def _frozen_smith_fragment_context(xyzin_path: Path) -> str:
    """Translate ORACLE's frozen task regime into SMITH's construction context."""

    from matrix_chem import (
        ATLAS_TASK_MINIMUM,
        ATLAS_TASK_TRANSITION_STATE,
        read_oracle_coordinate_atlas_contract,
    )

    task_regime = read_oracle_coordinate_atlas_contract(xyzin_path).task_regime
    if task_regime == ATLAS_TASK_MINIMUM:
        return "minimum"
    if task_regime == ATLAS_TASK_TRANSITION_STATE:
        return "transition_state"
    raise ValueError(f"unsupported frozen ORACLE task regime: {task_regime}")


def _gaussian_berny_pseudobond_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None,
    require_pseudobonds: bool = True,
) -> np.ndarray | None:
    """Transform Gaussian GRedFC's pseudobond primitive Hessian into SONIC."""

    if xyzin_path is None:
        return None
    try:
        from matrix_chem.topology.elements import atomic_number
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    except (ImportError, OSError, ValueError):
        return None
    if require_pseudobonds and definition.fragment_mode != "PSEUDO_BONDS":
        return None
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3):
        raise ValueError("Berny coordinates do not match the atom list")
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    primitive_hessian = np.diag(
        [_gaussian_gredfc_primitive_curvature(p, numbers, coords) for p in definition.primitives]
    )
    gic_by_label = {label: gic for gic in definition.gics for label in (gic.identifier, gic.name)}
    sonic_labels = model.sonic_labels or model.labels
    sonic_transform = np.zeros((len(sonic_labels), len(definition.primitives)), dtype=float)
    for row, label in enumerate(sonic_labels):
        gic = gic_by_label.get(label)
        if gic is None:
            raise ValueError(f"SONIC coordinate {label!r} is absent from the frozen definition")
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            if primitive_id not in primitive_index:
                raise ValueError(f"unknown primitive {primitive_id!r} in {label!r}")
            sonic_transform[row, primitive_index[primitive_id]] += float(coefficient)
    primitive_from_active = _primitive_from_active_matrix(
        model,
        sonic_transform,
        definition=definition,
        coordinates_angstrom=coords,
    )
    if np.linalg.matrix_rank(primitive_from_active, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-SONIC Berny transform is rank deficient")
    hessian = primitive_from_active.T @ primitive_hessian @ primitive_from_active
    return 0.5 * (hessian + hessian.T)


def _primitive_from_active_matrix(
    model: OptimizerCoordinateModel,
    sonic_from_primitive: np.ndarray,
    *,
    definition: object,
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Return the physical tangent map ``dq_active -> dq_primitive``.

    A coefficient pseudoinverse solves ``C dp = A dq`` in an arbitrary
    Euclidean gauge of the redundant primitive list.  The molecular geometry
    instead supplies the unique realizable change: ``dp = Bp (dx/dq) dq``.
    Both factors below come from the authoritative SMITH/LINK Wilson tangent,
    including fragment translations and exponential-map rotations.
    """

    sonic_transform = np.asarray(sonic_from_primitive, dtype=float)
    if sonic_transform.ndim != 2:
        raise ValueError("SONIC-from-primitive transform must be two-dimensional")
    active_transform = model.sonic_from_coordinates
    if active_transform is None:
        active_transform = np.eye(sonic_transform.shape[0], dtype=float)
    else:
        active_transform = np.asarray(active_transform, dtype=float)
    if active_transform.shape != (sonic_transform.shape[0], len(model.labels)):
        raise ValueError("active SONIC transform is inconsistent with primitive coordinates")
    from matrix_smith import build_primitive_b_matrix

    primitive_b = np.asarray(
        build_primitive_b_matrix(
            definition,
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    cartesian_from_active_angstrom = (
        _optimizer_directions_at_geometry(
            model,
            definition,
            np.asarray(coordinates_angstrom, dtype=float),
        ).T
        / ANGSTROM_TO_BOHR
    )
    primitive_from_active = primitive_b @ cartesian_from_active_angstrom
    residual = sonic_transform @ primitive_from_active - active_transform
    tolerance = 1.0e-8 * max(1.0, float(np.linalg.norm(active_transform)))
    if float(np.linalg.norm(residual)) > tolerance:
        raise ValueError("physical primitive/SONIC tangent contract is inconsistent")
    return np.asarray(primitive_from_active, dtype=float)


def _primitive_diagonal_hessian_to_active(
    model: OptimizerCoordinateModel,
    source_definition: object,
    curvatures: np.ndarray,
    *,
    coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Transform a source-primitive diagonal model to the active SONIC."""

    from matrix_smith import build_primitive_b_matrix

    values = np.asarray(curvatures, dtype=float).reshape(-1)
    primitive_b = np.asarray(
        build_primitive_b_matrix(
            source_definition,
            coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    if primitive_b.shape[0] != len(values) or not np.all(np.isfinite(values)):
        raise ValueError("primitive Hessian diagonal is inconsistent with its source basis")
    if np.any(values <= 0.0):
        raise ValueError("primitive Hessian curvatures must be positive")
    active_definition = model.sonic_definition
    if active_definition is None:
        raise ValueError("primitive Hessian transformation requires a frozen active SONIC")
    cartesian_from_active_angstrom = (
        _optimizer_directions_at_geometry(
            model,
            active_definition,
            np.asarray(coordinates_angstrom, dtype=float),
        ).T
        / ANGSTROM_TO_BOHR
    )
    primitive_from_active = primitive_b @ cartesian_from_active_angstrom
    if np.linalg.matrix_rank(primitive_from_active, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-active SONIC transform is rank deficient")
    hessian = primitive_from_active.T @ (values[:, None] * primitive_from_active)
    return _validate_optimizer_hessian(hessian, len(model.labels))


def _berny_geometric_fragment_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    xyzin_path: Path | str | None,
) -> np.ndarray | None:
    """Berny intramolecular plus geomeTRIC fragment primitive Hessian."""

    if xyzin_path is None:
        return None
    try:
        from matrix_chem.topology.elements import atomic_number
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    except (ImportError, OSError, ValueError):
        return None
    if definition.fragment_mode != "SPECIAL_COORDINATES":
        return None
    coords = np.asarray(coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    curvatures: list[float] = []
    for primitive in definition.primitives:
        function = str(primitive.function).upper()
        if function == "FTRANS":
            # geomeTRIC assigns 0.05 to each absolute fragment primitive.
            # MATRIX stores q_rel=q_a-q_b rather than (q_a-q_b)/sqrt(2).
            # geomeTRIC translations are in bohr; MATRIX FTRANS is in angstrom.
            curvatures.append(0.025 * ANGSTROM_TO_BOHR**2)
        elif function == "FROT":
            # Exponential-map rotations are dimensionless angular coordinates.
            curvatures.append(0.025)
        elif function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}:
            curvatures.append(0.05 * ANGSTROM_TO_BOHR**2)
        else:
            curvatures.append(_gaussian_gredfc_primitive_curvature(primitive, numbers, coords))
    primitive_hessian = np.diag(curvatures)
    gic_by_label = {label: gic for gic in definition.gics for label in (gic.identifier, gic.name)}
    sonic_labels = model.sonic_labels or model.labels
    sonic_transform = np.zeros((len(sonic_labels), len(definition.primitives)), dtype=float)
    for row, label in enumerate(sonic_labels):
        gic = gic_by_label.get(label)
        if gic is None:
            raise ValueError(f"SONIC coordinate {label!r} is absent from the frozen definition")
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            sonic_transform[row, primitive_index[primitive_id]] += float(coefficient)
    primitive_from_active = _primitive_from_active_matrix(
        model,
        sonic_transform,
        definition=definition,
        coordinates_angstrom=coords,
    )
    if np.linalg.matrix_rank(primitive_from_active, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-SONIC fragment transform is rank deficient")
    hessian = primitive_from_active.T @ primitive_hessian @ primitive_from_active
    return 0.5 * (hessian + hessian.T)


def _gaussian_gredfc_primitive_curvature(
    primitive,
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
) -> float:
    """Literal FCStr/FCBend/FCTors assignment in Gaussian GRedFC.

    MATRIX's genuine out-of-plane angle U has no Gaussian GRedFC primitive
    force constant because Gaussian substitutes an improper dihedral.  Its
    transferable curvature is defined halfway between the Berny valence-bend
    and torsional curvatures of the same tricoordinated center.
    """

    function = str(primitive.function).upper()
    indices = tuple(int(index) - 1 for index in primitive.atoms)
    if function == "R" and len(indices) == 2:
        distance = _distance_bohr(coordinates_angstrom, indices[0], indices[1])
        return (
            _gaussian_fcstr(distance, atomic_numbers[indices[0]], atomic_numbers[indices[1]])
            * ANGSTROM_TO_BOHR**2
        )
    if function == "A" and len(indices) == 3:
        return (
            0.160 if atomic_numbers[indices[0]] == 1 or atomic_numbers[indices[2]] == 1 else 0.250
        )
    if function == "L" and len(indices) >= 3:
        central = _distance_bohr(coordinates_angstrom, indices[1], indices[2])
        return _gaussian_fctors(central, atomic_numbers[indices[1]], atomic_numbers[indices[2]])
    if function in {"D", "IMPD"} and len(indices) >= 4:
        central = _distance_bohr(coordinates_angstrom, indices[1], indices[2])
        return _gaussian_fctors(central, atomic_numbers[indices[1]], atomic_numbers[indices[2]])
    if function == "U" and len(indices) == 4:
        center, first, second, third = indices
        neighbours = (first, second, third)
        bend_curvatures = [
            0.160 if atomic_numbers[left] == 1 or atomic_numbers[right] == 1 else 0.250
            for left, right in combinations(neighbours, 2)
        ]
        torsional_curvatures = [
            _gaussian_fctors(
                _distance_bohr(coordinates_angstrom, center, neighbour),
                atomic_numbers[center],
                atomic_numbers[neighbour],
            )
            for neighbour in neighbours
        ]
        return 0.5 * (float(np.mean(bend_curvatures)) + float(np.mean(torsional_curvatures)))
    if function == "RPCB" and primitive.family == "PSEUDO_CYCLE_BEND":
        terms = _encoded_component_terms(primitive.refs, arity=3)
        return float(
            sum(
                coefficient
                * coefficient
                * (
                    0.160
                    if atomic_numbers[atoms[0] - 1] == 1 or atomic_numbers[atoms[2] - 1] == 1
                    else 0.250
                )
                for coefficient, atoms in terms
            )
        )
    if function == "RPCK" and primitive.family in {
        "PSEUDO_CYCLE_TORSION",
        "RING_PUCKER_COMPONENT",
    }:
        # RPCK/CHARM components are normalized linear combinations of
        # ordinary ring torsions.  The diagonal Berny seed therefore follows
        # the exact congruence sum c_i^2 k_i used for pseudo-cycle torsions;
        # no arbitrary generic fallback is needed for genuine ring puckering.
        terms = _encoded_component_terms(primitive.refs, arity=4)
        return float(
            sum(
                coefficient
                * coefficient
                * _gaussian_fctors(
                    _distance_bohr(coordinates_angstrom, atoms[1] - 1, atoms[2] - 1),
                    atomic_numbers[atoms[1] - 1],
                    atomic_numbers[atoms[2] - 1],
                )
                for coefficient, atoms in terms
            )
        )
    raise ValueError(
        f"Gaussian Berny has no exact primitive assignment for {primitive.function}/{primitive.family}"
    )


def _encoded_component_terms(
    refs: Sequence[str],
    *,
    arity: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    terms: list[tuple[float, tuple[int, ...]]] = []
    for ref in refs:
        coefficient_text, separator, atom_text = str(ref).partition(":")
        if not separator:
            raise ValueError(f"invalid encoded primitive component {ref!r}")
        atoms = tuple(int(value) for value in atom_text.split("-") if value)
        if len(atoms) != arity:
            raise ValueError(f"invalid {arity}-atom primitive component {ref!r}")
        terms.append((float(coefficient_text), atoms))
    if not terms:
        raise ValueError("composite primitive has no encoded components")
    return tuple(terms)


def _distance_bohr(coordinates_angstrom: np.ndarray, left: int, right: int) -> float:
    return float(
        np.linalg.norm(coordinates_angstrom[left] - coordinates_angstrom[right]) * ANGSTROM_TO_BOHR
    )


def _gaussian_fcstr(
    distance_bohr: float, left_atomic_number: int, right_atomic_number: int
) -> float:
    shifts = (
        -0.244,
        0.352,
        1.085,
        0.660,
        1.522,
        2.068,
        0.7126,
        1.4725,
        1.8238,
        2.0203,
        0.8335,
        1.6549,
        2.1164,
        2.2137,
        2.3718,
        0.9491,
        1.7190,
        2.3185,
        2.5206,
        2.5110,
        2.5,
    )

    def row(number: int) -> int:
        return (number + 13) // 8 if number <= 18 else (number + 53) // 18

    high = min(max(row(max(left_atomic_number, right_atomic_number)), 1), 6)
    low = min(max(row(min(left_atomic_number, right_atomic_number)), 1), 6)
    denominator = max((distance_bohr - shifts[high * (high - 1) // 2 + low - 1]) ** 3, 0.1)
    return max(1.734 / denominator, 1.0e-3)


def _gaussian_fctors(
    distance_bohr: float, left_atomic_number: int, right_atomic_number: int
) -> float:
    radii = (
        0.32,
        0.60,
        1.20,
        1.05,
        0.81,
        0.77,
        0.74,
        0.74,
        0.72,
        0.72,
        1.50,
        1.40,
        1.30,
        1.17,
        1.10,
        1.04,
        0.99,
        0.99,
        1.80,
        1.60,
        *([1.40] * 11),
        1.30,
        1.20,
        1.20,
        1.10,
        1.10,
    )
    left = min(max(int(left_atomic_number), 1), 36)
    right = min(max(int(right_atomic_number), 1), 36)
    overlap = (radii[left - 1] + radii[right - 1]) / 0.529 - float(distance_bohr)
    return 0.0023 + 0.07 * max(overlap, 0.0)


def _chemical_valence_hessian(
    model: OptimizerCoordinateModel,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    settings: OptimizerSettings,
    *,
    xyzin_path: Path | str | None = None,
) -> np.ndarray | None:
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3) or len(atoms) < 2:
        return None
    try:
        from matrix_chem.topology.covalent_radii import covalent_radius
        from matrix_chem.topology.elements import atomic_number
    except ImportError:
        return None
    ncart = coords.size
    cart_angstrom = np.eye(ncart, dtype=float) * 0.02
    numbers = [atomic_number(atom) or 0 for atom in atoms]
    bond_count = 0
    for i in range(len(atoms)):
        zi = numbers[i]
        ri = covalent_radius(zi) or 0.75
        for j in range(i + 1, len(atoms)):
            zj = numbers[j]
            rj = covalent_radius(zj) or 0.75
            delta = coords[i] - coords[j]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-8:
                continue
            cutoff = 1.25 * (ri + rj) + 0.20
            if distance > cutoff:
                continue
            unit = delta / distance
            stiffness = _bond_stiffness_hartree_per_angstrom2(zi, zj, distance, ri + rj)
            block = stiffness * np.outer(unit, unit)
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)
            cart_angstrom[si, si] += block
            cart_angstrom[sj, sj] += block
            cart_angstrom[si, sj] -= block
            cart_angstrom[sj, si] -= block
            bond_count += 1
    if bond_count == 0:
        return None
    # Convert Eh/A^2 to Eh/bohr^2 before using the standard Cartesian projector.
    cart_bohr = cart_angstrom / (ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR)
    projected = optimizer_hessian_from_cartesian(cart_bohr, model)
    # Complete the central-force Cartesian guess with transferable
    # Schlegel/TRIC curvatures: central springs underestimate bends and cannot
    # supply the inter-fragment rigid-body block.
    projected = projected.copy()
    for index, label in enumerate(model.labels):
        name = str(label).lower()
        if name.startswith("bend"):
            projected[index, index] = max(float(projected[index, index]), 0.160)
        elif name.startswith(("fdis", "fc_dist")):
            # The radial intermolecular coordinate is stiffer than tangential
            # sliding but remains much softer than a covalent stretch.
            projected[index, index] = max(float(projected[index, index]), 0.100)
        elif name.startswith("ftrn"):
            # MATRIX stores a relative translation (fragment minus reference),
            # so two geomeTRIC-like 0.05 primitive blocks project to 0.025.
            projected[index, index] = max(float(projected[index, index]), 0.025)
        elif name.startswith("frot"):
            projected[index, index] = max(
                float(projected[index, index]), settings.fragment_rotation_curvature or 0.025
            )
    if (
        xyzin_path is not None
        and settings.fragment_radial_curvature is not None
        and settings.fragment_tangential_curvature is not None
    ):
        projected = _apply_fragment_hessian_blocks(projected, model, coords, xyzin_path, settings)
    diagonal_floor = np.diag(np.maximum(np.diag(projected), settings.min_abs_metric_diagonal))
    if _hessian_is_usable(projected, settings):
        return projected
    return diagonal_floor


def _apply_fragment_hessian_blocks(
    hessian: np.ndarray,
    model: OptimizerCoordinateModel,
    coordinates_angstrom: np.ndarray,
    xyzin_path: Path | str,
    settings: OptimizerSettings,
) -> np.ndarray:
    """Add radial/tangential curvature in complete relative-translation blocks."""

    try:
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    except (ImportError, OSError, ValueError):
        return hessian
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    coordinate_position = {label: index for index, label in enumerate(model.labels)}
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[int, int]] = {}
    group_reference_frames: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]] = {}
    for gic in definition.gics:
        position = coordinate_position.get(gic.identifier, coordinate_position.get(gic.name))
        if position is None:
            continue
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        if len(coefficients) != 1 or not np.isclose(float(coefficients[0][1]), 1.0):
            continue
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None or primitive.function != "FTRANS":
            continue
        key = (tuple(primitive.atoms), tuple(primitive.ref_atoms))
        groups.setdefault(key, {})[int(primitive.mode)] = int(position)
        ref_frame_atoms = tuple(getattr(primitive, "ref_frame_atoms", ()))
        if ref_frame_atoms:
            group_reference_frames[key] = ref_frame_atoms
    result = np.asarray(hessian, dtype=float).copy()
    coords = np.asarray(coordinates_angstrom, dtype=float)
    for (atoms, ref_atoms), modes in groups.items():
        if set(modes) != {0, 1, 2}:
            continue
        center = np.mean(coords[np.asarray(atoms, dtype=int) - 1], axis=0)
        ref_center = np.mean(coords[np.asarray(ref_atoms, dtype=int) - 1], axis=0)
        radial = center - ref_center
        norm = float(np.linalg.norm(radial))
        if norm <= 1.0e-10:
            continue
        radial /= norm
        ref_frame_atoms = group_reference_frames.get((atoms, ref_atoms), ())
        if ref_frame_atoms:
            from matrix_smith.numerics import _fragment_frame

            reference_frame = _fragment_frame(
                coords,
                ref_atoms,
                frame_atoms=ref_frame_atoms,
            )
            radial = radial @ reference_frame
        tangential_curvature = float(settings.fragment_tangential_curvature)
        radial_curvature = float(settings.fragment_radial_curvature)
        block = tangential_curvature * np.eye(3) + (
            radial_curvature - tangential_curvature
        ) * np.outer(radial, radial)
        indices = [modes[axis] for axis in range(3)]
        result[np.ix_(indices, indices)] = block
    return 0.5 * (result + result.T)


def _bond_stiffness_hartree_per_angstrom2(
    zi: int, zj: int, distance: float, covalent_sum: float
) -> float:
    if zi == 1 or zj == 1:
        base = 0.45
    elif zi in {6, 7, 8} and zj in {6, 7, 8}:
        base = 0.65
    else:
        base = 0.35
    if covalent_sum > 1.0e-8:
        base *= min(1.5, max(0.5, covalent_sum / max(distance, 1.0e-8)))
    return float(base)


def _validate_optimizer_hessian(hessian: np.ndarray, ncoord: int) -> np.ndarray:
    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape != (ncoord, ncoord):
        raise ValueError(f"optimizer Hessian shape must be {(ncoord, ncoord)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("optimizer Hessian contains non-finite values")
    return 0.5 * (matrix + matrix.T)


def _cache_record_to_json(evaluation: OptimizerEvaluation) -> dict[str, object]:
    return {
        "schema": OPTIMIZER_CACHE_SCHEMA,
        "chart_epoch": evaluation.chart_epoch,
        "q": np.asarray(evaluation.q, dtype=float).reshape(-1).tolist(),
        "coordinates_angstrom": np.asarray(evaluation.coordinates_angstrom, dtype=float).tolist(),
        "result": point_result_to_json(evaluation.result),
    }


def _cache_record_from_json(payload: dict[str, object]) -> OptimizerEvaluation:
    result_payload = dict(payload.get("result") or {})
    gradient = result_payload.get("gradient_hartree_per_bohr")
    hessian = result_payload.get("hessian_hartree_per_bohr2")
    backend_coordinates = result_payload.get("backend_coordinates_angstrom")
    result = PointEvaluationResult(
        point_index=int(result_payload.get("point_index", 0)),
        displacement=float(result_payload.get("displacement", 0.0)),
        energy_hartree=result_payload.get("energy_hartree"),
        gradient_hartree_per_bohr=None if gradient is None else np.asarray(gradient, dtype=float),
        hessian_hartree_per_bohr2=None if hessian is None else np.asarray(hessian, dtype=float),
        backend_coordinates_angstrom=(
            None if backend_coordinates is None else np.asarray(backend_coordinates, dtype=float)
        ),
        frame_alignment_rms_angstrom=float(result_payload.get("frame_alignment_rms_angstrom", 0.0)),
        status=str(result_payload.get("status", "completed")),
        message=str(result_payload.get("message", "")),
        source=str(result_payload.get("source", "optimizer-cache")),
        point_group=str(result_payload.get("point_group", "")),
        symmetry_projection_status=str(
            result_payload.get("symmetry_projection_status", "NOT_ANALYZED")
        ),
        symmetry_projection_max_displacement_angstrom=float(
            result_payload.get("symmetry_projection_max_displacement_angstrom", 0.0)
        ),
        symmetry_projection_rms_displacement_angstrom=float(
            result_payload.get("symmetry_projection_rms_displacement_angstrom", 0.0)
        ),
        execution=dict(result_payload.get("execution", {})),
        schema=str(result_payload.get("schema", "oracle.link.point_result.v1")),
    )
    return OptimizerEvaluation(
        q=np.asarray(payload["q"], dtype=float),
        coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
        result=result,
        cache_hit=False,
        chart_epoch=int(payload.get("chart_epoch", 0)),
    )


def _refine_hessian_diagonals_from_two_sided_energies(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    coordinate_steps: Sequence[float] | None = None,
    zero_off_diagonal: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace Hessian diagonals and either rescale or clear the couplings."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    ncoord = matrix.shape[0]
    if matrix.shape != (len(q), len(q)):
        raise ValueError("Hessian and optimizer coordinate dimensions do not match")
    if coordinate_steps is None:
        steps = np.full(
            ncoord,
            min(settings.fd_max_step, max(settings.fd_min_step, settings.fd_step)),
            dtype=float,
        )
    else:
        steps = np.asarray(coordinate_steps, dtype=float).reshape(-1)
        if steps.shape != (ncoord,) or not np.all(np.isfinite(steps)):
            raise ValueError("two-sided Hessian coordinate steps do not match the model")
        steps = np.minimum(
            settings.fd_max_step,
            np.maximum(settings.fd_min_step, np.abs(steps)),
        )
    new_diagonal = np.zeros(ncoord, dtype=float)
    extra_points = 0
    for index in range(ncoord):
        step = float(steps[index])
        plus = np.asarray(q, dtype=float).copy()
        minus = np.asarray(q, dtype=float).copy()
        plus[index] += step
        minus[index] -= step
        plus_eval = service.evaluate(
            plus, tag=f"final-hessian-plus-{index + 1}", requested_properties=("energy",)
        )
        minus_eval = service.evaluate(
            minus, tag=f"final-hessian-minus-{index + 1}", requested_properties=("energy",)
        )
        new_diagonal[index] = (
            plus_eval.energy_hartree - 2.0 * current.energy_hartree + minus_eval.energy_hartree
        ) / (step * step)
        extra_points += int(not plus_eval.cache_hit) + int(not minus_eval.cache_hit)
    old_diagonal = np.diag(matrix).copy()
    refined = matrix.copy()
    np.fill_diagonal(refined, new_diagonal)
    scale_factors = np.ones(ncoord, dtype=float)
    valid = (np.abs(old_diagonal) > settings.min_abs_metric_diagonal) & np.isfinite(new_diagonal)
    scale_factors[valid] = np.sqrt(
        np.minimum(
            settings.final_hessian_rescale_max**2,
            np.maximum(
                settings.final_hessian_rescale_min**2,
                np.abs(new_diagonal[valid] / old_diagonal[valid]),
            ),
        )
    )
    for left in range(ncoord):
        for right in range(left):
            refined[left, right] = refined[right, left] = (
                0.0
                if zero_off_diagonal
                else matrix[left, right] * scale_factors[left] * scale_factors[right]
            )
    refined = 0.5 * (refined + refined.T)
    return refined, {
        "performed": True,
        "steps": steps.tolist(),
        "extra_points": extra_points,
        "old_diagonal": old_diagonal.tolist(),
        "new_diagonal": new_diagonal.tolist(),
        "rescale_factors": scale_factors.tolist(),
        "sign_changes": int(np.count_nonzero(np.signbit(old_diagonal) != np.signbit(new_diagonal))),
        "off_diagonal_policy": "zero" if zero_off_diagonal else "scaled_previous",
    }


def _advance_numerical_convergence_state(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    gradient: np.ndarray,
    fd_info: Mapping[str, object],
    hessian: np.ndarray,
    settings: OptimizerSettings,
    convergence: Mapping[str, bool],
    *,
    last_energy_change: float,
    last_cartesian_step_bohr: np.ndarray,
    selective_disabled: bool,
    final_hessian_refinement_attempted: bool,
    transition_mode_reference: TransitionModeReference | None,
) -> tuple[
    np.ndarray,
    dict[str, object],
    np.ndarray,
    OptimizerSettings,
    dict[str, bool],
    dict[str, object],
    bool,
    bool,
]:
    """Verify one-sided convergence and advance the irreversible central state."""

    active_gradient = np.asarray(gradient, dtype=float)
    active_info = dict(fd_info)
    active_hessian = np.asarray(hessian, dtype=float)
    active_settings = settings
    active_convergence = dict(convergence)
    mode = str(active_info.get("mode", ""))
    verification: dict[str, object]

    if (
        active_settings.fd_stencil_policy == "one_sided_only"
        and not active_settings.compute_final_hessian
    ):
        verification = {
            "performed": False,
            "confirmed": True,
            "reason": "one_sided_only_protocol",
            "gradient_policy": "one_sided_only",
            "fd_two_sided_count": 0,
            "final_hessian_requested": False,
        }
        return (
            active_gradient,
            active_info,
            active_hessian,
            active_settings,
            active_convergence,
            verification,
            final_hessian_refinement_attempted,
            True,
        )

    if mode != "two-sided":
        verified_gradient, verified_info = _verify_final_numerical_gradient(
            service,
            current,
            q,
            active_hessian,
            active_settings,
            selective_disabled=selective_disabled,
        )
        verified_convergence = _gaussian_like_convergence(
            active_settings,
            last_energy_change,
            _convergence_gradient(current, verified_gradient, active_settings, service),
            last_cartesian_step_bohr,
        )
        active_gradient = verified_gradient
        active_info = dict(verified_info)
        active_convergence = verified_convergence
        agreement = bool(all(verified_convergence.values()))
        verification = {
            "performed": True,
            "confirmed": agreement,
            "gradient_policy": "final_central_verification",
            "fd_two_sided_count": int(verified_info["two_sided_count"]),
        }
        if not agreement:
            hessian_info = {"performed": False, "reason": "final_hessian_not_requested"}
            if active_settings.compute_final_hessian:
                active_hessian, hessian_info = _refine_hessian_diagonals_from_two_sided_energies(
                    service,
                    current,
                    q,
                    active_hessian,
                    active_settings,
                    coordinate_steps=verified_info.get("coordinate_steps"),
                    zero_off_diagonal=True,
                )
            active_settings = replace(
                active_settings,
                one_sided_until_convergence=False,
                final_gradient_verification=False,
                selective_fd_refresh=False,
            )
            verification.update(
                {
                    "two_sided_mismatch": True,
                    "continue_two_sided": True,
                    "hessian_refinement": hessian_info,
                }
            )
            return (
                active_gradient,
                active_info,
                active_hessian,
                active_settings,
                active_convergence,
                verification,
                final_hessian_refinement_attempted,
                False,
            )
    else:
        verification = {
            "performed": False,
            "confirmed": True,
            "reason": "already_two_sided",
            "fd_two_sided_count": int(active_info.get("two_sided_count", 0)),
        }

    if not active_settings.compute_final_hessian:
        active_settings = replace(
            active_settings,
            one_sided_until_convergence=False,
            final_gradient_verification=False,
            selective_fd_refresh=False,
        )
        verification["final_hessian_requested"] = False
        return (
            active_gradient,
            active_info,
            active_hessian,
            active_settings,
            active_convergence,
            verification,
            True,
            True,
        )
    if not final_hessian_refinement_attempted:
        active_hessian, hessian_info = _refine_hessian_diagonals_from_two_sided_energies(
            service,
            current,
            q,
            active_hessian,
            active_settings,
            coordinate_steps=active_info.get("coordinate_steps"),
            zero_off_diagonal=False,
        )
        active_settings = replace(
            active_settings,
            one_sided_until_convergence=False,
            final_gradient_verification=False,
            selective_fd_refresh=False,
        )
        verification.update(
            {
                "hessian_refinement": hessian_info,
                "final_step_pending": True,
            }
        )
        return (
            active_gradient,
            active_info,
            active_hessian,
            active_settings,
            active_convergence,
            verification,
            True,
            False,
        )

    verification["final_step_converged"] = True
    return (
        active_gradient,
        active_info,
        active_hessian,
        active_settings,
        active_convergence,
        verification,
        True,
        True,
    )


def _frequencies_from_final_sonic_hessian(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    hessian: np.ndarray,
) -> tuple[float, ...]:
    """Return GF frequencies from a stationary-point SONIC Hessian.

    The transformation is deliberately delegated to MATRIX-GF.  LINK owns the
    optimization Hessian and the convergence contract; GF owns mass weighting,
    external-mode projection and the frequency convention.
    """
    if service.coordinate_model.kind != "sonic":
        return ()
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_gf import cartesian_normal_modes_from_sonic_hessian

    _values, b_matrix = service._evaluate_active_sonic(current.coordinates_angstrom)
    masses = np.asarray(
        [atomic_mass(int(atomic_number(atom) or 0)) for atom in service.atoms],
        dtype=float,
    )
    if np.any(masses <= 0.0):
        raise ValueError("final SONIC GF requires positive atomic masses")
    modes = cartesian_normal_modes_from_sonic_hessian(
        np.asarray(hessian, dtype=float),
        b_matrix,
        masses,
        np.asarray(current.coordinates_angstrom, dtype=float).reshape((-1, 3)) * ANGSTROM_TO_BOHR,
        source="LINK final SONIC Hessian",
    )
    return tuple(float(value) for value in modes.frequencies_cm)


def _verify_final_numerical_gradient(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    selective_disabled: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recompute the final numerical gradient with the cached plus side and new minus side."""

    verification_settings = replace(
        settings,
        one_sided_until_convergence=False,
        adaptive_fd_mode=False,
        two_sided=True,
        selective_fd_refresh=False,
    )
    gradient, info = _gradient_in_coordinate_space(
        service,
        current,
        q,
        hessian,
        verification_settings,
        force_explicit=True,
        selective_disabled=selective_disabled,
        force_two_sided=True,
    )
    info = dict(info)
    info["gradient_policy"] = "final_central_verification"
    return gradient, info


def _restart_evaluation_kwargs(current: object) -> dict[str, object]:
    """Return an optional backend restart contract without changing test doubles.

    Optimizer evaluations produced by resident QM backends carry a point-result
    execution mapping.  Lightweight services and external engines may not.
    Omitting the keywords entirely in that case preserves the generic evaluator
    boundary.
    """

    result = getattr(current, "result", None)
    execution = getattr(result, "execution", None)
    if not isinstance(execution, Mapping):
        return {}
    if execution.get("restart_reuse_for_displacements") is False:
        return {}
    artifact = execution.get("restart_artifact")
    if artifact is None:
        return {}
    projection = execution.get("restart_projection")
    return {
        "restart_artifact": str(artifact),
        "restart_projection": None if projection is None else str(projection),
    }


def _gradient_in_coordinate_space(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    force_explicit: bool = False,
    iteration: int = 0,
    previous_gradient: np.ndarray | None = None,
    selective_disabled: bool = False,
    force_two_sided: bool = False,
    two_sided_latched: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    analytic = current.gradient_hartree_per_bohr
    if analytic is not None and settings.prefer_analytic_gradient:
        directions_bohr = service.coordinate_directions(current.coordinates_angstrom) / (
            1.0 / ANGSTROM_TO_BOHR
        )
        gradient = directions_bohr @ analytic
        return gradient, {
            "gradient_policy": "analytic_cartesian_projected_full",
            "mode": "analytic-gradient",
            "step_min": 0.0,
            "step_max": 0.0,
            "refreshed_coordinate_count": gradient.size,
            "predicted_coordinate_count": 0,
            "active_coordinate_fraction": 1.0,
            "one_sided_count": 0,
            "two_sided_count": 0,
            "parallel_workers": 1,
            "surrogate_sample_count": 0,
        }
    if settings.stationary_point == "transition_state" and not force_explicit:
        negative_subspace_gradient = _gradient_in_negative_mode_subspace(
            service,
            current,
            q,
            hessian,
            settings,
            iteration=iteration,
            two_sided_latched=two_sided_latched,
        )
        if negative_subspace_gradient is not None:
            return negative_subspace_gradient
    ncoord = q.size
    gradient = np.zeros(ncoord, dtype=float)
    steps = np.zeros(ncoord, dtype=float)
    near_minimum = bool(
        force_two_sided
        or two_sided_latched
        or (
            settings.adaptive_fd_mode
            and previous_gradient is not None
            and np.max(np.abs(np.asarray(previous_gradient, dtype=float)))
            <= settings.fd_two_sided_switch_force
        )
    )
    use_central = bool(
        settings.two_sided
        and (
            force_two_sided
            or two_sided_latched
            or (
                not settings.one_sided_until_convergence
                and (near_minimum or not settings.adaptive_fd_mode)
            )
        )
    )
    if force_two_sided:
        use_central = bool(settings.two_sided)
    fd_modes = np.full(ncoord, "two-sided" if use_central else "one-sided", dtype=object)
    refresh_mask = np.ones(ncoord, dtype=bool)
    symmetry_mask = np.ones(ncoord, dtype=bool)
    if settings.fd_totally_symmetric_only:
        if service.coordinate_model.kind != "sonic" or service._sonic_definition is None:
            raise ValueError(
                "totally-symmetric finite differences require a SONIC coordinate model"
            )
        from matrix_smith.symmetry_labels import is_total_symmetric_irrep

        definition = service._sonic_definition
        symmetry_mask = np.asarray(
            [
                is_total_symmetric_irrep(
                    definition.point_group,
                    definition.gics[index].irrep,
                )
                for index in service._sonic_coordinate_indices
            ],
            dtype=bool,
        )
        if symmetry_mask.shape != (ncoord,) or not np.any(symmetry_mask):
            raise ValueError(
                "SONIC total-symmetry mask does not match the numerical coordinate model"
            )
        # At a geometry satisfying the frozen parent group, non-totally-
        # symmetric first derivatives vanish.  Do not sample their displaced
        # energies; retain their exact projected value of zero.
        refresh_mask &= symmetry_mask
    # The frozen production protocol uses only the exact symmetry mask.
    # Same-family magnitude screening is retained only as an isolated
    # diagnostic helper and is never active in production optimization.
    class_screen_enabled = False
    class_screen_audit = bool(force_two_sided)
    if hasattr(service, "finite_difference_class_screen"):
        class_mask, class_threshold_fraction, class_screen_audit = service.finite_difference_class_screen(
            previous_gradient,
            symmetry_mask,
            settings,
            enabled=class_screen_enabled,
            iteration=iteration,
        )
    else:
        class_mask, class_threshold_fraction = symmetry_mask.copy(), 0.0
        class_screen_audit = bool(force_two_sided)
    refresh_mask &= class_mask
    surrogate_samples = 0
    selective_active = settings.selective_fd_refresh and not selective_disabled
    if selective_active and previous_gradient is not None and not force_explicit:
        previous = np.asarray(previous_gradient, dtype=float).reshape(-1)
        predicted, counts = _surrogate_gradient_from_cache(
            service.cache.records, q, previous, settings
        )
        surrogate_samples = counts
        due = iteration <= 0 or (iteration % settings.fd_refresh_interval == 0)
        if not due and surrogate_samples >= 2:
            refresh_mask = _selective_refresh_mask(previous, predicted, hessian, settings)
            refresh_mask &= symmetry_mask
            gradient[:] = predicted
            gradient[~symmetry_mask] = 0.0
    current_energy = current.energy_hartree
    restart_kwargs = _restart_evaluation_kwargs(current)
    soft = service.coordinate_soft_mask(hessian)

    def finite_difference_step(index: int) -> float:
        hii = max(abs(float(hessian[index, index])), settings.min_abs_metric_diagonal)
        two_sided = str(fd_modes[index]) == "two-sided"
        if settings.adaptive_fd_mode and two_sided:
            scale = (
                settings.fd_soft_characteristic_scale
                if soft[index]
                else settings.fd_hard_characteristic_scale
            )
            step = (3.0 * settings.energy_noise * scale / hii) ** (1.0 / 3.0)
        elif settings.adaptive_fd_mode:
            step = math.sqrt(settings.energy_noise / hii)
        elif two_sided:
            step = settings.fd_step
        else:
            step = settings.fd_step
        return min(settings.fd_max_step, max(settings.fd_min_step, step))

    indices = [int(index) for index in np.flatnonzero(refresh_mask)]
    for index in indices:
        steps[index] = finite_difference_step(index)
    point_requests: list[tuple[int, int, np.ndarray, str]] = []
    for index in indices:
        plus_q = q.copy()
        plus_q[index] += steps[index]
        point_requests.append((index, 1, plus_q, f"fd-plus-{index + 1}"))
        if str(fd_modes[index]) == "two-sided":
            minus_q = q.copy()
            minus_q[index] -= steps[index]
            point_requests.append((index, -1, minus_q, f"fd-minus-{index + 1}"))

    def evaluate_fd_point(
        request: tuple[int, int, np.ndarray, str],
    ) -> tuple[int, int, float, float, int]:
        index, sign, _displaced_q, tag = request
        active_step = float(steps[index])
        backend = getattr(service, "backend", None)
        maximum_halvings = (
            int(backend.state_tracking_max_displacement_halvings)
            if backend is not None
            else 0
        )
        for halving in range(maximum_halvings + 1):
            displaced_q = q.copy()
            displaced_q[index] += sign * active_step
            try:
                evaluation = service.evaluate(
                    displaced_q,
                    tag=(tag if halving == 0 else f"{tag}-state-half-{halving}"),
                    requested_properties=("energy",),
                    **restart_kwargs,
                )
            except ElectronicStateResolutionError:
                if halving >= maximum_halvings:
                    raise
                active_step *= 0.5
                continue
            return index, sign, evaluation.energy_hartree, active_step, halving
        raise AssertionError("unreachable APOC displacement-halving state")

    if settings.fd_parallel_workers > 1 and len(point_requests) > 1:
        with ThreadPoolExecutor(max_workers=settings.fd_parallel_workers) as pool:
            point_results = list(pool.map(evaluate_fd_point, point_requests))
    else:
        point_results = [evaluate_fd_point(request) for request in point_requests]
    energies = {(index, sign): energy for index, sign, energy, _step, _halving in point_results}
    realized_steps = {(index, sign): step for index, sign, _energy, step, _halving in point_results}
    state_halvings = {
        (index, sign): halving for index, sign, _energy, _step, halving in point_results
    }
    for index in indices:
        step = realized_steps[(index, 1)]
        plus_energy = energies[(index, 1)]
        if str(fd_modes[index]) == "two-sided":
            minus_step = realized_steps[(index, -1)]
            minus_energy = energies[(index, -1)]
            gradient[index] = (
                -step / (minus_step * (minus_step + step)) * minus_energy
                + (step - minus_step) / (minus_step * step) * current_energy
                + minus_step / (step * (minus_step + step)) * plus_energy
            )
            steps[index] = min(step, minus_step)
        else:
            gradient[index] = (plus_energy - current_energy) / step
            steps[index] = step
    if (
        class_screen_audit
        and hasattr(service, "retain_audited_finite_difference_coordinates")
    ):
        service.retain_audited_finite_difference_coordinates(
            gradient,
            symmetry_mask,
            class_threshold_fraction,
        )
    if (
        previous_gradient is None
        and getattr(getattr(service, "coordinate_model", None), "kind", None) == "sonic"
        and settings.fd_initial_class_threshold_fraction > 0.0
    ):
        initial_max = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        service._fd_initial_max_gradient = initial_max
        service._fd_class_threshold_fraction = float(
            settings.fd_initial_class_threshold_fraction
        )
    refreshed = int(np.count_nonzero(refresh_mask))
    predicted_count = int(ncoord - refreshed)
    one_sided_count = int(np.count_nonzero(fd_modes[refresh_mask] == "one-sided"))
    two_sided_count = int(np.count_nonzero(fd_modes[refresh_mask] == "two-sided"))
    fd_mode = (
        "mixed"
        if one_sided_count and two_sided_count
        else ("one-sided" if one_sided_count else "two-sided")
    )
    return gradient, {
        "gradient_policy": "coordinate_energy_fd_selective"
        if selective_active
        else "coordinate_energy_fd_full",
        "mode": fd_mode,
        "step_min": float(np.min(steps[refresh_mask])) if refreshed else 0.0,
        "step_max": float(np.max(steps[refresh_mask])) if refreshed else 0.0,
        "refreshed_coordinate_count": refreshed,
        "predicted_coordinate_count": predicted_count,
        "symmetry_excluded_coordinate_count": int(np.count_nonzero(~symmetry_mask)),
        "symmetry_mask": symmetry_mask.tolist(),
        "class_threshold_fraction": class_threshold_fraction,
        "class_screen_enabled": class_screen_enabled,
        "class_screen_audit": class_screen_audit,
        "class_screen_audit_interval": int(settings.fd_class_screen_audit_interval),
        "active_coordinate_fraction": float(refreshed / ncoord) if ncoord else 0.0,
        "one_sided_count": one_sided_count,
        "two_sided_count": two_sided_count,
        "parallel_workers": min(settings.fd_parallel_workers, max(len(point_requests), 1)),
        "surrogate_sample_count": surrogate_samples,
        "hard_coordinate_count": int(np.count_nonzero(~soft & refresh_mask)),
        "soft_coordinate_count": int(np.count_nonzero(soft & refresh_mask)),
        "near_minimum": near_minimum,
        "coordinate_steps": steps.tolist(),
        "state_displacement_halvings": {
            f"q{index + 1}:{'plus' if sign > 0 else 'minus'}": count
            for (index, sign), count in state_halvings.items()
            if count
        },
    }


def _gradient_in_negative_mode_subspace(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    iteration: int,
    two_sided_latched: bool,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Estimate a TS gradient with central differences along negative modes.

    The uphill eigenspace is the numerically most sensitive part of a
    transition-state search.  Directional central differences are used there
    while the orthogonal complement retains the information-efficient
    one-sided estimate.  The returned vector is reconstructed in the complete
    orthonormal optimizer-coordinate basis, so no force component is dropped.
    """

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if matrix.shape != (q.size, q.size) or q.size == 0:
        return None
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return None
    negative = np.flatnonzero(eigenvalues < -1.0e-12)
    if negative.size == 0:
        return None
    negative_modes = np.asarray(eigenvectors[:, negative], dtype=float)
    try:
        basis, _ = np.linalg.qr(negative_modes, mode="complete")
    except np.linalg.LinAlgError:
        return None
    nnegative = int(negative_modes.shape[1])
    complement = basis[:, nnegative:]
    current_energy = float(current.energy_hartree)
    noise = max(float(settings.energy_noise), 1.0e-16)

    mode_steps = np.zeros(q.size, dtype=float)
    for column, eigenvalue in enumerate(eigenvalues[negative]):
        curvature = max(abs(float(eigenvalue)), settings.min_abs_metric_diagonal)
        mode_steps[column] = min(
            settings.fd_max_step,
            max(
                settings.fd_min_step,
                (3.0 * noise * settings.fd_soft_characteristic_scale / curvature) ** (1.0 / 3.0),
            ),
        )
    complement_steps = np.zeros(complement.shape[1], dtype=float)
    for column in range(complement.shape[1]):
        direction = complement[:, column]
        curvature = max(
            abs(float(direction @ matrix @ direction)), settings.min_abs_metric_diagonal
        )
        complement_steps[column] = min(
            settings.fd_max_step,
            max(settings.fd_min_step, math.sqrt(noise / curvature)),
        )

    requests: list[tuple[str, int, np.ndarray, float]] = []
    for column in range(nnegative):
        direction = basis[:, column]
        step = mode_steps[column]
        requests.append(("negative-plus", column, q + step * direction, step))
        requests.append(("negative-minus", column, q - step * direction, step))
    for column in range(complement.shape[1]):
        direction = complement[:, column]
        step = complement_steps[column]
        requests.append(("complement-plus", column, q + step * direction, step))

    def evaluate(request: tuple[str, int, np.ndarray, float]) -> tuple[str, int, float]:
        family, column, displaced_q, _step = request
        tag = f"fd-ts-{family}-{column + 1}"
        result = service.evaluate(
            displaced_q,
            tag=tag,
            requested_properties=("energy",),
            **_restart_evaluation_kwargs(current),
        )
        return family, column, float(result.energy_hartree)

    if settings.fd_parallel_workers > 1 and len(requests) > 1:
        with ThreadPoolExecutor(max_workers=settings.fd_parallel_workers) as pool:
            values = list(pool.map(evaluate, requests))
    else:
        values = [evaluate(request) for request in requests]
    energies = {(family, column): energy for family, column, energy in values}

    directional = np.zeros(q.size, dtype=float)
    for column in range(nnegative):
        directional[column] = (
            energies[("negative-plus", column)] - energies[("negative-minus", column)]
        ) / (2.0 * mode_steps[column])
    for column in range(complement.shape[1]):
        directional[nnegative + column] = (
            energies[("complement-plus", column)] - current_energy
        ) / complement_steps[column]
    gradient = basis @ directional
    return gradient, {
        "gradient_policy": "transition_negative_subspace_central_complement_forward",
        "mode": "negative-subspace-mixed",
        "step_min": float(np.min(np.concatenate((mode_steps, complement_steps))))
        if complement_steps.size
        else float(np.min(mode_steps)),
        "step_max": float(np.max(np.concatenate((mode_steps, complement_steps))))
        if complement_steps.size
        else float(np.max(mode_steps)),
        "refreshed_coordinate_count": int(q.size),
        "predicted_coordinate_count": 0,
        "active_coordinate_fraction": 1.0,
        "one_sided_count": int(complement.shape[1]),
        "two_sided_count": nnegative,
        "parallel_workers": int(settings.fd_parallel_workers),
        "surrogate_sample_count": 0,
        "near_minimum": bool(two_sided_latched),
        "negative_mode_count": nnegative,
    }


def _surrogate_gradient_from_cache(
    records: Sequence[OptimizerEvaluation],
    q: np.ndarray,
    fallback: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[np.ndarray, int]:
    base = np.asarray(q, dtype=float).reshape(-1)
    gradient = np.asarray(fallback, dtype=float).reshape(-1).copy()
    usable: list[tuple[np.ndarray, float]] = []
    for record in records[-settings.surrogate_max_samples :]:
        try:
            usable.append((np.asarray(record.q, dtype=float).reshape(-1), record.energy_hartree))
        except ValueError:
            continue
    if len(usable) < 2:
        return gradient, len(usable)
    q_values = np.vstack([item[0] for item in usable])
    energies = np.asarray([item[1] for item in usable], dtype=float)
    centered = q_values - base
    design = np.column_stack([np.ones(centered.shape[0]), centered])
    try:
        coeffs, *_ = np.linalg.lstsq(design, energies, rcond=None)
    except np.linalg.LinAlgError:
        return gradient, len(usable)
    if coeffs.shape[0] == gradient.size + 1 and np.all(np.isfinite(coeffs[1:])):
        gradient[:] = coeffs[1:]
    return gradient, len(usable)


def _selective_refresh_mask(
    previous: np.ndarray,
    predicted: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
) -> np.ndarray:
    previous_gradient = np.asarray(previous, dtype=float).reshape(-1)
    predicted_gradient = np.asarray(predicted, dtype=float).reshape(-1)
    stable = np.abs(predicted_gradient - previous_gradient) <= settings.fd_gradient_change_tolerance
    inactive_force = (np.abs(previous_gradient) <= settings.max_force_tolerance) & (
        np.abs(predicted_gradient) <= settings.max_force_tolerance
    )
    refresh = ~(stable & inactive_force)
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if matrix.shape == (refresh.size, refresh.size) and refresh.size:
        diagonal = np.maximum(np.abs(np.diag(matrix)), settings.min_abs_metric_diagonal)
        denom = np.sqrt(np.outer(diagonal, diagonal))
        normalized_coupling = np.divide(
            np.abs(matrix),
            denom,
            out=np.zeros_like(matrix, dtype=float),
            where=denom > 0.0,
        )
        np.fill_diagonal(normalized_coupling, 0.0)
        active = (
            refresh
            | (np.abs(previous_gradient) > 0.5 * settings.max_force_tolerance)
            | (np.abs(predicted_gradient) > 0.5 * settings.max_force_tolerance)
        )
        coupled_score = normalized_coupling @ active.astype(float)
        refresh |= coupled_score > settings.selective_coupling_threshold
    else:
        coupled_score = np.zeros(refresh.size, dtype=float)
    minimum = int(math.ceil(settings.selective_min_refresh_fraction * refresh.size))
    if minimum > 0 and int(np.count_nonzero(refresh)) < minimum:
        priority = np.maximum(np.abs(previous_gradient), np.abs(predicted_gradient)) + coupled_score
        for index in np.argsort(priority)[::-1][:minimum]:
            refresh[int(index)] = True
    if refresh.size and not np.any(refresh):
        refresh[int(np.argmax(np.abs(predicted_gradient)))] = True
    return np.asarray(refresh, dtype=bool)


def _local_coordinate_groups(
    model: OptimizerCoordinateModel,
    hessian: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[tuple[int, ...], ...]:
    ncoord = len(model.labels)
    if ncoord == 0:
        return ()
    parent = list(range(ncoord))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    directions = np.asarray(model.directions_angstrom, dtype=float)
    support = np.abs(directions) > 1.0e-12
    for i in range(ncoord):
        for j in range(i + 1, ncoord):
            shares_cartesian = bool(np.any(support[i] & support[j]))
            coupled = abs(float(hessian[i, j])) > settings.hessian_coupling_threshold
            if shares_cartesian or coupled:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for index in range(ncoord):
        groups.setdefault(find(index), []).append(index)
    return tuple(tuple(items) for items in groups.values())


def _hessian_sparsity_mask(
    model: OptimizerCoordinateModel,
    hessian: np.ndarray,
    settings: OptimizerSettings,
) -> np.ndarray:
    ncoord = len(model.labels)
    mask = np.eye(ncoord, dtype=bool)
    directions = np.asarray(model.directions_angstrom, dtype=float)
    support = np.abs(directions) > 1.0e-12
    for i in range(ncoord):
        for j in range(i + 1, ncoord):
            keep = (
                bool(np.any(support[i] & support[j]))
                or abs(float(hessian[i, j])) > settings.hessian_coupling_threshold
            )
            mask[i, j] = keep
            mask[j, i] = keep
    return mask


def _project_hessian_sparsity(hessian: np.ndarray, mask: np.ndarray) -> np.ndarray:
    projected = np.asarray(hessian, dtype=float).copy()
    projected[~mask] = 0.0
    return 0.5 * (projected + projected.T)


def _matrix_sparsity(matrix: np.ndarray) -> float:
    array = np.asarray(matrix, dtype=float)
    if array.size == 0:
        return 1.0
    return float(1.0 - np.count_nonzero(np.abs(array) > 1.0e-14) / array.size)


def _symmetry_status(
    settings: OptimizerSettings,
    model: OptimizerCoordinateModel,
) -> str:
    if getattr(model, "pes_exploration", False):
        return "instantaneous_exploration_group_perception"
    definition = model.sonic_definition if model.kind == "sonic" else None
    if (
        definition is not None
        and bool(definition.symmetrize)
        and str(definition.point_group).strip().upper() not in {"", "C1", "UNKNOWN"}
    ):
        return "frozen_initial_group_gradient_projection"
    if not settings.symmetry_reduction:
        return "disabled"
    if model.kind != "sonic":
        return "disabled_non_sonic"
    return "diagnostic_only"


def _selective_fallback_status(settings: OptimizerSettings, selective_disabled: bool) -> str:
    if not settings.selective_fd_refresh:
        return "disabled"
    return "fallback_full_fd" if selective_disabled else "selective_active"


def _active_optimizer_metric(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return active indices, dx/dq in Å and bohr, and the full SONIC metric."""

    directions_angstrom = np.asarray(
        service.coordinate_directions(current.coordinates_angstrom), dtype=float
    )
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if mask.shape != (directions_angstrom.shape[0],):
        mask = np.ones(directions_angstrom.shape[0], dtype=bool)
    indices = np.flatnonzero(mask)
    active_angstrom = directions_angstrom[indices]
    active_bohr = active_angstrom * ANGSTROM_TO_BOHR
    metric = active_bohr @ active_bohr.T
    metric = 0.5 * (metric + metric.T)
    return indices, active_angstrom, active_bohr, metric


def _far_from_minimum_cauchy_required(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    gradient: np.ndarray,
    hessian: np.ndarray,
    active_mask: np.ndarray,
    trust_radius: float,
    settings: OptimizerSettings,
) -> bool:
    """Select Cauchy only for a high-force, oversized quadratic-model step."""

    g = np.asarray(gradient, dtype=float).reshape(-1)
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    indices, directions_angstrom, directions_bohr, metric = _active_optimizer_metric(
        service, current, active_mask
    )
    if indices.size == 0 or matrix.shape != (g.size, g.size):
        return False
    active_gradient = g[indices]
    try:
        metric_inverse = np.linalg.pinv(metric, rcond=1.0e-10, hermitian=True)
        projected_cartesian_gradient = directions_bohr.T @ (metric_inverse @ active_gradient)
        projected_max_force = float(np.max(np.abs(projected_cartesian_gradient)))

        metric_values, metric_vectors = np.linalg.eigh(metric)
        metric_floor = max(
            float(np.max(np.abs(metric_values), initial=0.0)) * 1.0e-10,
            settings.min_abs_metric_diagonal,
        )
        inverse_sqrt_metric = (
            metric_vectors
            @ np.diag(1.0 / np.sqrt(np.maximum(metric_values, metric_floor)))
            @ metric_vectors.T
        )
        active_hessian = matrix[np.ix_(indices, indices)]
        orthonormal_hessian = inverse_sqrt_metric @ active_hessian @ inverse_sqrt_metric
        orthonormal_hessian = 0.5 * (orthonormal_hessian + orthonormal_hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(orthonormal_hessian)
        maximum = max(
            float(np.max(np.abs(eigenvalues), initial=0.0)),
            settings.min_hessian_eigenvalue,
        )
        curvature_floor = max(
            settings.min_hessian_eigenvalue,
            maximum / settings.max_hessian_condition,
        )
        positive_curvature = np.maximum(eigenvalues, curvature_floor)
        orthonormal_gradient = inverse_sqrt_metric @ active_gradient
        orthonormal_step = -eigenvectors @ (
            (eigenvectors.T @ orthonormal_gradient) / positive_curvature
        )
        active_step = inverse_sqrt_metric @ orthonormal_step
        cartesian_step = (directions_angstrom.T @ active_step).reshape(
            current.coordinates_angstrom.shape
        )
        predicted_rmsd = _cartesian_rms_displacement_angstrom(
            current.coordinates_angstrom,
            current.coordinates_angstrom + cartesian_step,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return False

    force_threshold = settings.far_from_minimum_force_factor * float(settings.max_force_tolerance)
    displacement_threshold = settings.far_from_minimum_displacement_factor * float(trust_radius)
    return bool(
        np.isfinite(projected_max_force)
        and np.isfinite(predicted_rmsd)
        and projected_max_force > force_threshold
        and predicted_rmsd > displacement_threshold
    )


def _minimum_effective_hessian_model(
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return the conditioned model in an orthonormal full-metric chart."""

    matrix, inverse_sqrt_metric = _orthonormal_optimizer_hessian(
        hessian,
        metric,
        settings,
    )
    size = matrix.shape[0]
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        eigenvalues = np.ones(size, dtype=float)
        eigenvectors = np.eye(size, dtype=float)
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    maximum = max(float(np.max(np.abs(eigenvalues))), settings.min_hessian_eigenvalue)
    floor = max(settings.min_hessian_eigenvalue, maximum / settings.max_hessian_condition)
    positive = np.maximum(eigenvalues, floor)
    condition = float(np.max(positive) / np.min(positive)) if positive.size else 1.0
    effective_hessian = eigenvectors @ np.diag(positive) @ eigenvectors.T
    return effective_hessian, inverse_sqrt_metric, minimum, condition, floor


def _orthonormal_optimizer_hessian(
    hessian: np.ndarray,
    metric: np.ndarray | None,
    settings: OptimizerSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a covariant Hessian in LINK's physical Wilson metric."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    size = matrix.shape[0]
    metric_matrix = np.eye(size, dtype=float)
    if metric is not None:
        supplied = np.asarray(metric, dtype=float)
        if supplied.ndim == 1 and supplied.shape == (size,):
            supplied = np.diag(supplied)
        if supplied.shape != (size, size):
            raise ValueError("optimizer metric shape does not match the Hessian")
        metric_matrix = _positive_metric_matrix(supplied, settings.min_abs_metric_diagonal)
    metric_values, metric_vectors = np.linalg.eigh(metric_matrix)
    inverse_sqrt_metric = (
        metric_vectors @ np.diag(1.0 / np.sqrt(metric_values)) @ metric_vectors.T
    )
    orthonormal = inverse_sqrt_metric @ matrix @ inverse_sqrt_metric
    return 0.5 * (orthonormal + orthonormal.T), inverse_sqrt_metric


def _minimum_effective_hessian_displacement(
    hessian: np.ndarray,
    gradient: np.ndarray,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``-H_eff^-1`` in the same full-metric chart used by RFO."""

    effective, inverse_sqrt_metric, _minimum, _condition, _floor = _minimum_effective_hessian_model(
        hessian,
        settings,
        metric=metric,
    )
    scaled_gradient = inverse_sqrt_metric @ np.asarray(gradient, dtype=float).reshape(-1)
    try:
        scaled_step = -np.linalg.solve(effective, scaled_gradient)
    except np.linalg.LinAlgError:
        scaled_step = -(np.linalg.pinv(effective, rcond=1.0e-12) @ scaled_gradient)
    return inverse_sqrt_metric @ np.asarray(scaled_step, dtype=float)


def _geometric_trust_region_step(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
    damping: float = 0.0,
    transition_mode_reference: TransitionModeReference | None = None,
) -> StepProposal:
    """Restricted-step RFO with one authoritative nonlinear trust solve.

    The RFO level shift is determined in the orthonormal full-metric chart,
    where the step norm is the local Cartesian norm.  The shared nonlinear
    Cartesian trust enforcer then realizes the resulting direction exactly.
    Keeping nonlinear realization out of the algebraic level-shift search is
    essential for large typed-ONIC contracts: otherwise every trial shift
    rebuilds and back-transforms the complete coordinate model before the
    common trust enforcer repeats the same operation.
    """
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    gradient_vector = np.asarray(gradient, dtype=float).reshape(-1)
    if settings.stationary_point == "transition_state":
        return _transition_state_trust_region_step(
            service,
            current,
            q,
            matrix,
            gradient_vector,
            radius,
            settings,
            metric=metric,
            transition_mode_reference=transition_mode_reference,
        )
    effective_hessian, inverse_sqrt_metric, minimum, condition, floor = (
        _minimum_effective_hessian_model(
            matrix,
            settings,
            metric=metric,
        )
    )
    gradient_vector = inverse_sqrt_metric @ gradient_vector

    def rfo_step(shift: float) -> np.ndarray:
        size = gradient_vector.size
        if size == 0:
            return np.zeros(0, dtype=float)
        augmented = np.zeros((size + 1, size + 1), dtype=float)
        augmented[:-1, :-1] = effective_hessian + max(float(shift), 0.0) * np.eye(size)
        augmented[:-1, -1] = gradient_vector
        augmented[-1, :-1] = gradient_vector
        values, vectors = np.linalg.eigh(augmented)
        root = vectors[:, int(np.argmin(values))]
        if abs(float(root[-1])) <= 1.0e-14:
            raise np.linalg.LinAlgError("singular RS-RFO root")
        step = np.asarray(root[:-1] / root[-1], dtype=float)
        if float(gradient_vector @ step) > 0.0:
            step *= -1.0
        return step

    def physical_step(step: np.ndarray) -> np.ndarray:
        return inverse_sqrt_metric @ np.asarray(step, dtype=float)

    shift = max(float(damping), 0.0)
    step = rfo_step(shift)
    # In the orthonormal metric chart, ||step|| is the local total Cartesian
    # displacement in bohr.  LINK's trust radius is the aligned Cartesian RMS
    # displacement per atom in angstrom, hence the sqrt(natom) conversion.
    # This is only the inexpensive algebraic RS-RFO restriction; the shared
    # nonlinear enforcer below remains authoritative for the finite step.
    metric_target = (
        float(radius)
        * ANGSTROM_TO_BOHR
        * math.sqrt(max(int(current.coordinates_angstrom.shape[0]), 1))
    )
    step_norm = float(np.linalg.norm(step))
    active = not np.isfinite(step_norm) or step_norm > 1.1 * metric_target
    if active:
        lower = shift
        upper = max(1.0e-8, 2.0 * max(shift, floor))
        upper_step = rfo_step(upper)
        for _ in range(32):
            upper_norm = float(np.linalg.norm(upper_step))
            if np.isfinite(upper_norm) and upper_norm <= metric_target:
                break
            lower = upper
            upper *= 2.0
            upper_step = rfo_step(upper)
        else:
            raise RuntimeError("unable to bracket an RS-RFO metric-trust step")
        # The shifted positive quadratic model has a monotonically decreasing
        # norm.  A loose 10% algebraic target mirrors the established
        # geomeTRIC strategy; exact finite-chart enforcement happens once in
        # The caller applies the stationary-point-specific trust contract.
        step = upper_step
        for _ in range(32):
            shift = 0.5 * (lower + upper)
            step = rfo_step(shift)
            step_norm = float(np.linalg.norm(step))
            if np.isfinite(step_norm) and 0.9 * metric_target <= step_norm <= metric_target:
                break
            if not np.isfinite(step_norm) or step_norm > metric_target:
                lower = shift
            else:
                upper = shift
        if not np.isfinite(step_norm) or step_norm > metric_target:
            shift = upper
            step = rfo_step(shift)
    policy = "rs_rfo_cartesian_trust"
    if active:
        policy += "_restricted"
    if minimum < -settings.min_hessian_eigenvalue:
        policy += "_positive_model"
    proposal = StepProposal(
        step=physical_step(step),
        policy=policy,
        hessian_min_eigenvalue=minimum,
        hessian_condition=condition,
        damping_shift=shift,
    )
    return proposal


def _preconditioned_cauchy_step(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
) -> StepProposal:
    """Return the Cauchy point of the conditioned full-metric model.

    In the orthonormal chart ``y = G**(1/2) q``, the Cauchy direction is
    ``-g_y`` and its unconstrained minimizer is
    ``alpha = (g_y.T g_y) / (g_y.T H_y g_y)``.  The common nonlinear
    Cartesian trust solver then restricts that single model step if needed.
    """

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    gradient_vector = np.asarray(gradient, dtype=float).reshape(-1)
    effective, inverse_sqrt_metric, minimum, condition, _floor = _minimum_effective_hessian_model(
        matrix,
        settings,
        metric=service.optimizer_metric(current.coordinates_angstrom),
    )
    scaled_gradient = inverse_sqrt_metric @ gradient_vector
    numerator = float(scaled_gradient @ scaled_gradient)
    denominator = float(scaled_gradient @ effective @ scaled_gradient)
    if numerator <= np.finfo(float).eps or denominator <= np.finfo(float).eps:
        scaled_step = np.zeros_like(scaled_gradient)
    else:
        scaled_step = -(numerator / denominator) * scaled_gradient
    step = inverse_sqrt_metric @ scaled_step
    proposal = StepProposal(
        step=step,
        policy="full_metric_cauchy_far_from_minimum",
        hessian_min_eigenvalue=minimum,
        hessian_condition=condition,
        damping_shift=0.0,
    )
    return proposal


def _transition_state_trust_region_step(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
    transition_mode_reference: TransitionModeReference | None = None,
) -> StepProposal:
    """Return GDV ``DXRFO`` in its literal bohr/radian GIC representation."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    unit_scales = np.asarray(
        service.gdv_internal_coordinate_scales(), dtype=float
    ).reshape(-1)
    if unit_scales.shape != (matrix.shape[0],) or np.any(~np.isfinite(unit_scales)) or np.any(
        unit_scales <= 0.0
    ):
        raise ValueError("GDV coordinate-unit scales do not match the optimizer model")
    inverse_unit_scales = 1.0 / unit_scales
    gdv_matrix = (
        inverse_unit_scales[:, None] * matrix * inverse_unit_scales[None, :]
    )
    # l103.F:DiagFC diagonalizes FC itself.  ReadAllGIC does not solve the
    # generalized Wilson-metric eigenproblem used by LINK's minimum solver.
    # Its distance variables are in bohr, whereas SONIC stores them in
    # angstrom; angular variables are radians in both programs.
    del metric, q, settings
    raw_values, eigenvectors = np.linalg.eigh(gdv_matrix)
    optimizer_candidates = inverse_unit_scales[:, None] * eigenvectors
    mode_candidates = _cartesian_transition_mode_candidates(
        eigenvectors,
        np.diag(inverse_unit_scales),
        service.coordinate_directions(current.coordinates_angstrom),
    )
    # l103.F fixes ModMax=1: after the ordered eigensolve this is always mode
    # zero in Python.  The previous reference is retained only to report an
    # overlap; it must never change the selected DXRFO root.
    mode_index = 0
    mode_overlap, mode_vector = _gdv_first_mode_reference(
        mode_candidates,
        optimizer_candidates,
        current.coordinates_angstrom,
        transition_mode_reference,
    )
    gdv_gradient = inverse_unit_scales * np.asarray(gradient, dtype=float)
    projected = eigenvectors.T @ gdv_gradient
    result = gdv_dxrfo_step(
        raw_values,
        projected,
        maximum_internal_step=float(radius),
    )
    if not result.ok:
        raise np.linalg.LinAlgError("GDV DXRFO did not produce an acceptable step")
    step = inverse_unit_scales * (eigenvectors @ result.step)
    magnitudes = np.abs(raw_values)
    condition = float(
        np.max(magnitudes)
        / max(float(np.min(magnitudes)), np.finfo(float).eps)
    )
    policy = f"gdv_dxrfo_raw_spectrum_index_{result.raw_index}"
    proposal = StepProposal(
        step=np.asarray(step, dtype=float),
        policy=policy,
        hessian_min_eigenvalue=float(np.min(raw_values)) if raw_values.size else 0.0,
        hessian_condition=condition,
        damping_shift=0.0,
        transition_mode_index=mode_index,
        transition_mode_overlap=mode_overlap,
        transition_mode_vector=mode_vector,
        transition_ascending_shift=result.lambda0,
        transition_descending_shift=result.lambda_stable,
        prediction_hessian=matrix.copy(),
    )
    return proposal


def _gdv_first_mode_reference(
    cartesian_candidates: np.ndarray,
    optimizer_candidates: np.ndarray,
    coordinates_angstrom: np.ndarray,
    previous: TransitionModeReference | None,
) -> tuple[float, TransitionModeReference]:
    """Describe GDV's first ordered mode without using tracking to select it."""

    cartesian = np.asarray(cartesian_candidates, dtype=float)
    optimizer = np.asarray(optimizer_candidates, dtype=float)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if cartesian.ndim != 2 or cartesian.shape[1] == 0:
        raise ValueError("GDV transition-mode candidates cannot be empty")
    if optimizer.ndim != 2 or optimizer.shape[1] != cartesian.shape[1]:
        raise ValueError("GDV optimizer-mode candidates have the wrong shape")
    vector = cartesian[:, 0].copy()
    optimizer_vector = optimizer[:, 0].copy()
    overlap = 1.0
    if previous is not None:
        old_coordinates = np.asarray(previous.coordinates_angstrom, dtype=float)
        old_vector = np.asarray(previous.cartesian_vector, dtype=float).reshape(-1)
        if old_coordinates.shape != coordinates.shape or old_vector.shape != vector.shape:
            raise ValueError("GDV first-mode reference has the wrong shape")
        rotation = kabsch_rotation(old_coordinates, coordinates)
        transported = (old_vector.reshape((-1, 3)) @ rotation).reshape(-1)
        transported /= max(float(np.linalg.norm(transported)), 1.0e-30)
        signed_overlap = float(vector @ transported)
        overlap = abs(signed_overlap)
        if signed_overlap < 0.0:
            vector *= -1.0
            optimizer_vector *= -1.0
    return overlap, TransitionModeReference(
        cartesian_vector=vector,
        coordinates_angstrom=coordinates.copy(),
        cartesian_subspace=vector.reshape((-1, 1)),
        optimizer_vector=optimizer_vector,
        selection_policy="gdv_first_ordered_mode",
    )


def _select_transition_mode(
    cartesian_candidates: np.ndarray,
    settings: OptimizerSettings,
    reference: TransitionModeReference | None,
    coordinates_angstrom: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
    reaction_directions: np.ndarray | None = None,
    optimizer_candidates: np.ndarray | None = None,
) -> tuple[int, float, TransitionModeReference]:
    """Select a reaction mode while tracking near-degenerate subspaces."""

    vectors = np.asarray(cartesian_candidates, dtype=float)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != coordinates.size:
        raise ValueError("transition-mode candidates must be Cartesian tangent columns")
    if vectors.shape[1] == 0:
        raise ValueError("transition-mode candidates cannot be empty")
    values = None if eigenvalues is None else np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values is not None and values.shape != (vectors.shape[1],):
        raise ValueError("transition-mode eigenvalues have the wrong dimension")
    if reference is None:
        if settings.transition_mode >= vectors.shape[1]:
            raise ValueError(
                f"transition_mode {settings.transition_mode} exceeds the "
                f"{vectors.shape[1]}-mode optimizer Hessian"
            )
        selection = condition_aware_reaction_mode(
            (
                np.zeros(vectors.shape[1], dtype=float)
                if values is None
                else values
            ),
            vectors,
            reaction_directions,
            settings.transition_mode,
            absolute_floor=settings.min_hessian_eigenvalue,
            maximum_condition=settings.max_hessian_condition,
        )
        index = selection.index
        overlap = 1.0
        vector = vectors[:, index].copy()
        optimizer_vector = None
        if optimizer_candidates is not None:
            optimizer_matrix = np.asarray(optimizer_candidates, dtype=float)
            if optimizer_matrix.ndim != 2 or optimizer_matrix.shape[1] != vectors.shape[1]:
                raise ValueError("optimizer mode candidates have the wrong shape")
            optimizer_vector = optimizer_matrix[:, index].copy()
        cluster = _near_degenerate_mode_indices(values, index, settings)
        subspace = _orthonormal_cartesian_subspace(vectors[:, cluster])
    else:
        selection = None
        previous = np.asarray(reference.cartesian_vector, dtype=float).reshape(-1)
        if previous.shape != (vectors.shape[0],):
            raise ValueError("tracked transition mode has the wrong dimension")
        previous_coordinates = np.asarray(reference.coordinates_angstrom, dtype=float)
        if previous_coordinates.shape != coordinates.shape:
            raise ValueError("tracked transition-mode geometry has the wrong shape")
        rotation = kabsch_rotation(previous_coordinates, coordinates)
        previous = (previous.reshape((-1, 3)) @ rotation).reshape(-1)
        previous /= max(float(np.linalg.norm(previous)), 1.0e-30)
        previous_subspace = (
            previous.reshape((-1, 1))
            if reference.cartesian_subspace is None
            else np.asarray(reference.cartesian_subspace, dtype=float)
        )
        if previous_subspace.ndim != 2 or previous_subspace.shape[0] != vectors.shape[0]:
            raise ValueError("tracked transition-mode subspace has the wrong dimension")
        transported = np.column_stack(
            [
                (column.reshape((-1, 3)) @ rotation).reshape(-1)
                for column in previous_subspace.T
            ]
        )
        transported = _orthonormal_cartesian_subspace(transported)
        # Select the individual eigenvector with the largest overlap with the
        # transported reaction vector.  The previous implementation selected
        # from the overlap of the whole near-degenerate subspace, which is
        # invariant under rotations but does not identify a unique vector.
        # That allowed the reported uphill mode to jump between unrelated
        # roots while the subspace overlap remained high (the TS4/TS10
        # failure mode).  Keep the subspace for degeneracy diagnostics and
        # Cartesian transport, but make the root choice auditable and
        # continuous.
        if optimizer_candidates is not None and reference.optimizer_vector is not None:
            optimizer_matrix = np.asarray(optimizer_candidates, dtype=float)
            previous_optimizer = np.asarray(reference.optimizer_vector, dtype=float).reshape(-1)
            if (
                optimizer_matrix.ndim != 2
                or optimizer_matrix.shape[1] != vectors.shape[1]
                or previous_optimizer.shape != (vectors.shape[1],)
            ):
                raise ValueError("internal transition-mode reference has the wrong shape")
            previous_optimizer /= max(float(np.linalg.norm(previous_optimizer)), 1.0e-30)
            vector_overlaps = np.abs(optimizer_matrix.T @ previous_optimizer)
        else:
            vector_overlaps = np.abs(vectors.T @ previous)
        index = int(np.argmax(vector_overlaps))
        cluster = _near_degenerate_mode_indices(values, index, settings)
        subspace = _orthonormal_cartesian_subspace(vectors[:, cluster])
        overlap = float(np.linalg.norm(subspace.T @ previous))
        if subspace.shape[1] == transported.shape[1]:
            left, _singular, right_t = np.linalg.svd(
                subspace.T @ transported,
                full_matrices=False,
            )
            subspace = subspace @ left @ right_t
        overlaps = vectors.T @ previous
        signed_overlap = float(overlaps[index])
        vector = vectors[:, index].copy()
        optimizer_vector = None
        if optimizer_candidates is not None:
            optimizer_vector = np.asarray(optimizer_candidates, dtype=float)[:, index].copy()
            if float(optimizer_vector @ (np.asarray(reference.optimizer_vector, dtype=float) if reference.optimizer_vector is not None else optimizer_vector)) < 0.0:
                optimizer_vector *= -1.0
        if signed_overlap < 0.0:
            vector *= -1.0
    return index, overlap, TransitionModeReference(
        vector,
        coordinates.copy(),
        cartesian_subspace=subspace,
        optimizer_vector=optimizer_vector,
        selection_policy=(
            reference.selection_policy
            if selection is None
            else selection.policy
        ),
        reaction_overlap=(
            reference.reaction_overlap
            if selection is None
            else selection.selected_overlap
        ),
        isotropic_overlap=(
            reference.isotropic_overlap
            if selection is None
            else selection.isotropic_overlap
        ),
    )


def _near_degenerate_mode_indices(
    eigenvalues: np.ndarray | None,
    index: int,
    settings: OptimizerSettings,
) -> np.ndarray:
    """Return the numerical eigenvalue cluster containing one tracked mode."""

    if eigenvalues is None:
        return np.asarray((index,), dtype=int)
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    scale = max(float(np.max(np.abs(values))), settings.min_hessian_eigenvalue)
    tolerance = max(
        settings.min_hessian_eigenvalue,
        math.sqrt(np.finfo(float).eps) * scale,
    )
    return np.flatnonzero(np.abs(values - values[index]) <= tolerance)


def _orthonormal_cartesian_subspace(vectors: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis for Cartesian mode columns."""

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("transition-mode subspace cannot be empty")
    basis, triangular = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(triangular)
    if np.any(~np.isfinite(diagonal)) or np.any(np.abs(diagonal) <= 1.0e-30):
        raise ValueError("transition-mode subspace is rank deficient")
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return basis * signs


def _cartesian_transition_mode_candidates(
    eigenvectors: np.ndarray,
    inverse_sqrt_metric: np.ndarray,
    coordinate_directions: np.ndarray,
) -> np.ndarray:
    """Map orthonormal-chart modes into normalized Cartesian tangents."""

    vectors = np.asarray(eigenvectors, dtype=float)
    inverse_sqrt = np.asarray(inverse_sqrt_metric, dtype=float)
    directions = np.asarray(coordinate_directions, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != vectors.shape[1]:
        raise ValueError("transition-mode eigenvectors must form a square matrix")
    if inverse_sqrt.shape != vectors.shape:
        raise ValueError("transition-mode metric transform has the wrong shape")
    if directions.ndim != 2 or directions.shape[0] != vectors.shape[0]:
        raise ValueError("transition-mode Cartesian directions have the wrong shape")
    cartesian = directions.T @ (inverse_sqrt @ vectors)
    norms = np.linalg.norm(cartesian, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-30):
        raise ValueError("transition-mode Cartesian tangent is singular")
    return cartesian / norms


def _initial_transition_mode_reference(
    hessian: np.ndarray,
    metric: np.ndarray,
    settings: OptimizerSettings,
    *,
    coordinate_unit_scales: np.ndarray,
    coordinate_directions: np.ndarray,
    coordinates_angstrom: np.ndarray,
    reaction_directions: np.ndarray | None = None,
) -> TransitionModeReference | None:
    """Record GDV's first ordered seed mode independently of final validation."""
    if settings.stationary_point != "transition_state":
        return None
    del metric
    matrix = 0.5 * (
        np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T
    )
    unit_scales = np.asarray(coordinate_unit_scales, dtype=float).reshape(-1)
    if unit_scales.shape != (matrix.shape[0],) or np.any(~np.isfinite(unit_scales)) or np.any(
        unit_scales <= 0.0
    ):
        raise ValueError("GDV coordinate-unit scales do not match the initial Hessian")
    inverse_unit_scales = 1.0 / unit_scales
    gdv_matrix = (
        inverse_unit_scales[:, None] * matrix * inverse_unit_scales[None, :]
    )
    values, vectors = np.linalg.eigh(gdv_matrix)
    optimizer_vectors = inverse_unit_scales[:, None] * vectors
    candidates = _cartesian_transition_mode_candidates(
        vectors,
        np.diag(inverse_unit_scales),
        coordinate_directions,
    )
    _overlap, reference = _gdv_first_mode_reference(
        candidates,
        optimizer_vectors,
        coordinates_angstrom,
        None,
    )
    return reference


def _transition_reaction_cartesian_directions(
    model: OptimizerCoordinateModel,
    coordinate_directions: np.ndarray,
) -> np.ndarray | None:
    """Consume SMITH's frozen TS-reaction family without chemical inference."""

    definition = model.sonic_definition
    if model.kind != "sonic" or definition is None:
        return None
    positions = {label: index for index, label in enumerate(model.labels)}
    selected: list[int] = []
    for gic in definition.gics:
        if gic.family != "TS_REACTION_DISTANCE":
            continue
        position = positions.get(gic.name, positions.get(gic.identifier))
        if position is not None:
            selected.append(position)
    if not selected:
        return None
    directions = np.asarray(coordinate_directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != len(model.labels):
        raise ValueError("reaction-family directions do not match the SONIC chart")
    return directions[np.asarray(sorted(set(selected)), dtype=int)].T


def _cartesian_rms_displacement_angstrom(
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    """Return aligned RMS displacement in Angstrom per atom.

    This is the sole norm used by LINK's Cartesian trust region.  Dividing by
    the atom count, rather than by ``3 * natom``, makes the radius identical to
    the molecular-optimization convention used for Cartesian RMS steps and
    independent of the optimizer-coordinate representation.
    """

    displacement = aligned_cartesian_displacement(before, after)
    return float(np.sqrt(np.sum(displacement * displacement) / max(displacement.shape[0], 1)))


def _safeguarded_bracket_trial(
    lower: float,
    upper: float,
    lower_value: float,
    upper_value: float,
) -> float:
    """Return a secant trial kept strictly inside a valid scalar bracket."""

    left = float(lower)
    right = float(upper)
    span = right - left
    if span <= 0.0 or not np.isfinite(span):
        raise ValueError("invalid scalar-root bracket")
    low = float(lower_value)
    high = float(upper_value)
    if np.isfinite(low) and np.isfinite(high) and high > low:
        fraction = -low / (high - low)
        fraction = float(np.clip(fraction, 0.1, 0.9))
    else:
        fraction = 0.5
    return left + fraction * span


def _restrict_step_to_cartesian_trust(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    step: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
) -> tuple[np.ndarray, float, float, int]:
    """Restrict any optimizer step using its nonlinear Cartesian realization.

    The scalar search is representation agnostic: it follows the proposed ray
    in optimizer space, repeatedly performs the authoritative back-transform,
    and brackets the first boundary of the aligned Cartesian atomic-RMS trust
    region.  A safeguarded secant interpolation accelerates nearly linear
    charts while bisection bounds retain convergence for strongly curved
    coordinates or failed intermediate back-transforms.

    Returns ``(restricted_step, realized_rmsd, scale, iterations)``.  No QM
    evaluations are performed.
    """

    vector = np.asarray(step, dtype=float).reshape(-1)
    trust_radius = float(radius)
    if trust_radius <= 0.0 or not np.isfinite(trust_radius):
        raise ValueError("Cartesian trust radius must be finite and positive")
    if not np.all(np.isfinite(vector)):
        raise ValueError("optimizer step contains non-finite values")
    if vector.size == 0 or float(np.linalg.norm(vector)) <= np.finfo(float).eps:
        return vector.copy(), 0.0, 1.0, 0

    realization_by_scale: dict[float, tuple[float, bool]] = {}
    realize_for_trust = getattr(
        service,
        "_coordinates_from_q_for_trust",
        service.coordinates_from_q,
    )

    def realized_rmsd(scale: float) -> tuple[float, bool]:
        key = float(scale)
        if key not in realization_by_scale:
            try:
                coordinates = realize_for_trust(q + key * vector)
                rejection = _realized_geometry_rejection(
                    service, coordinates, settings
                )
                value = (
                    math.inf
                    if rejection
                    else _cartesian_rms_displacement_angstrom(
                        current.coordinates_angstrom,
                        coordinates,
                    )
                )
                valid = not rejection
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
                value = math.inf
                valid = False
            realization_by_scale[key] = float(value), bool(valid)
        return realization_by_scale[key]

    tolerance = float(settings.cartesian_trust_tolerance)
    target = (1.0 - tolerance) * trust_radius
    lower_scale = 0.0
    lower_rmsd = 0.0
    upper_scale = 1.0
    upper_rmsd = math.inf
    upper_valid = False
    best_step = np.zeros_like(vector)
    best_scale = 0.0
    best_rmsd = 0.0
    iterations = 0

    # The Wilson metric already supplies the Cartesian tangent of the full
    # optimizer step.  Use it to seed the nonlinear scalar solve near the
    # trust boundary instead of first realizing a usually enormous full step.
    # Every accepted scale is still verified by the authoritative finite
    # back-transform; the predictor only chooses the first scalar probe.
    predictor_scale = 1.0
    coordinate_directions = getattr(service, "coordinate_directions", None)
    if callable(coordinate_directions):
        try:
            tangent = np.asarray(
                coordinate_directions(current.coordinates_angstrom), dtype=float
            ).T @ vector
            tangent_coordinates = np.asarray(
                current.coordinates_angstrom, dtype=float
            ) + tangent.reshape(np.asarray(current.coordinates_angstrom).shape)
            tangent_rmsd = _cartesian_rms_displacement_angstrom(
                current.coordinates_angstrom,
                tangent_coordinates,
            )
            if np.isfinite(tangent_rmsd) and tangent_rmsd > target:
                predictor_scale = float(
                    np.clip(target / tangent_rmsd, 8.0 * np.finfo(float).eps, 1.0)
                )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            predictor_scale = 1.0

    initial_rmsd, initial_valid = realized_rmsd(predictor_scale)
    iterations += int(predictor_scale < 1.0)
    if (
        predictor_scale == 1.0
        and initial_valid
        and np.isfinite(initial_rmsd)
        and initial_rmsd <= trust_radius
    ):
        return vector.copy(), float(initial_rmsd), 1.0, 0
    if initial_valid and np.isfinite(initial_rmsd) and initial_rmsd <= target:
        lower_scale = predictor_scale
        lower_rmsd = float(initial_rmsd)
        best_step = predictor_scale * vector
        best_scale = predictor_scale
        best_rmsd = float(initial_rmsd)
        if target - best_rmsd <= tolerance * trust_radius:
            return best_step, best_rmsd, best_scale, max(iterations, 1)
        corrected_scale = min(
            1.0,
            predictor_scale * target / max(best_rmsd, np.finfo(float).eps),
        )
        if corrected_scale > predictor_scale * (1.0 + tolerance):
            corrected_rmsd, corrected_valid = realized_rmsd(corrected_scale)
            iterations += 1
            if (
                corrected_valid
                and np.isfinite(corrected_rmsd)
                and corrected_rmsd <= target
            ):
                lower_scale = corrected_scale
                lower_rmsd = float(corrected_rmsd)
                best_step = corrected_scale * vector
                best_scale = corrected_scale
                best_rmsd = float(corrected_rmsd)
                if target - best_rmsd <= tolerance * trust_radius:
                    return best_step, best_rmsd, best_scale, iterations
            else:
                upper_scale = corrected_scale
                upper_rmsd = float(corrected_rmsd)
                upper_valid = bool(corrected_valid)
    else:
        upper_scale = predictor_scale
        upper_rmsd = float(initial_rmsd)
        upper_valid = bool(initial_valid)

    remaining_iterations = max(
        int(settings.cartesian_trust_max_iterations) - iterations,
        0,
    )
    for _bracket_iteration in range(remaining_iterations):
        iterations += 1
        span = upper_scale - lower_scale
        if span <= 8.0 * np.finfo(float).eps:
            break
        trial_scale = _safeguarded_bracket_trial(
            lower_scale,
            upper_scale,
            lower_rmsd - target,
            upper_rmsd - target,
        )
        trial_step = trial_scale * vector
        trial_rmsd, trial_valid = realized_rmsd(trial_scale)

        if trial_valid and np.isfinite(trial_rmsd) and trial_rmsd <= target:
            lower_scale = trial_scale
            lower_rmsd = float(trial_rmsd)
            best_step = trial_step
            best_scale = trial_scale
            best_rmsd = float(trial_rmsd)
            if not upper_valid:
                # A chart-domain boundary is a hard constraint, not another
                # trust target.  The first bracketed valid interior point is
                # deliberately conservative; chasing the largest admissible
                # scale repeats expensive failed back-transforms without
                # adding scientific information.
                break
            if target - best_rmsd <= tolerance * trust_radius:
                break
        else:
            upper_scale = trial_scale
            upper_rmsd = float(trial_rmsd)
            upper_valid = bool(trial_valid)

    if best_scale <= 0.0:
        raise RuntimeError(
            "unable to realize a nonzero optimizer step inside the Cartesian trust region"
        )
    # ``best_rmsd`` was obtained from the exact same finite realization and is
    # cached above.  Repeating the back-transform here used to double the most
    # expensive successful boundary evaluation for no additional safeguard.
    verified_rmsd = best_rmsd
    if verified_rmsd > trust_radius * (1.0 + 8.0 * np.finfo(float).eps):
        raise RuntimeError("Cartesian trust solver returned an out-of-radius step")
    return best_step, float(verified_rmsd), float(best_scale), int(iterations)


def _enforce_proposal_cartesian_trust(
    proposal: StepProposal,
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
) -> StepProposal:
    """Apply LINK's single Cartesian trust contract to any step proposal."""

    step, rmsd, scale, iterations = _restrict_step_to_cartesian_trust(
        service,
        current,
        q,
        proposal.step,
        radius,
        settings,
    )
    policy = proposal.policy
    if scale < 1.0 and "realized_cartesian_trust_restricted" not in policy:
        policy += ":realized_cartesian_trust_restricted"
    return replace(
        proposal,
        step=step,
        policy=policy,
        cartesian_rmsd_angstrom=rmsd,
        trust_scale=proposal.trust_scale * scale,
        trust_iterations=proposal.trust_iterations + iterations,
        applied_trust_radius_angstrom=float(radius),
    )


def _enforce_proposal_gdv_internal_trust(
    proposal: StepProposal,
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    radius: float,
) -> StepProposal:
    """Apply GDV ``DXOpt`` with its ``OptDX``/``CrdGRo`` dispatch."""

    step = np.asarray(proposal.step, dtype=float).reshape(-1)
    unit_scales = np.asarray(
        service.gdv_internal_coordinate_scales(), dtype=float
    ).reshape(-1)
    if (
        unit_scales.shape != step.shape
        or np.any(~np.isfinite(unit_scales))
        or np.any(unit_scales <= 0.0)
    ):
        raise ValueError("GDV internal coordinate units do not match the optimizer step")
    gdv_step = unit_scales * step
    if service.gdv_uses_force_constant_step_weights():
        if proposal.prediction_hessian is None:
            raise ValueError("GDV CrdGRo trust weights require the DXRFO Hessian")
        hessian = np.asarray(proposal.prediction_hessian, dtype=float)
        inverse_unit_scales = 1.0 / unit_scales
        gdv_hessian = (
            inverse_unit_scales[:, None]
            * hessian
            * inverse_unit_scales[None, :]
        )
        weights = _gdv_crdgro_step_weights(gdv_step, gdv_hessian)
    else:
        weights = np.asarray(service.gdv_internal_step_weights(), dtype=float).reshape(-1)
    if (
        weights.shape != step.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("GDV internal trust weights do not match the optimizer step")
    effective_norm = float(np.linalg.norm(weights * gdv_step))
    target = float(radius)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("GDV internal trust radius must be positive and finite")
    scale = 1.0 if effective_norm <= target else target / effective_norm
    restricted = step * scale
    coordinates = None
    realization_factor = 1.0
    last_realization_error: ValueError | FloatingPointError | None = None
    for factor in GDV_REDQ2X_HARD_FAILURE_RETRY_FACTORS:
        try:
            trial_step = restricted * factor
            coordinates = service.coordinates_from_q(
                np.asarray(q, dtype=float) + trial_step
            )
            restricted = trial_step
            realization_factor = factor
            break
        except (ValueError, FloatingPointError) as exc:
            last_realization_error = exc
    if coordinates is None:
        assert last_realization_error is not None
        raise last_realization_error
    rmsd = _cartesian_rms_displacement_angstrom(
        current.coordinates_angstrom,
        coordinates,
    )
    policy = proposal.policy
    if scale < 1.0 and "gdv_internal_trust_restricted" not in policy:
        policy += ":gdv_internal_trust_restricted"
    if realization_factor < 1.0:
        policy += f":gdv_internal_realization_retry_factor_{realization_factor:.9g}"
    return replace(
        proposal,
        step=restricted,
        policy=policy,
        cartesian_rmsd_angstrom=float(rmsd),
        trust_scale=proposal.trust_scale * scale * realization_factor,
        trust_iterations=proposal.trust_iterations + int(realization_factor < 1.0),
        applied_trust_radius_angstrom=target,
    )


def _gdv_crdgro_step_weights(
    gdv_step: np.ndarray,
    gdv_hessian: np.ndarray,
) -> np.ndarray:
    """Replicate ``gdv.j32+/utilam.F:CrdGRo(IOp=5)`` exactly."""

    step = np.asarray(gdv_step, dtype=float).reshape(-1)
    hessian = np.asarray(gdv_hessian, dtype=float)
    if hessian.shape != (step.size, step.size):
        raise ValueError("GDV CrdGRo Hessian does not match the optimizer step")
    if np.any(~np.isfinite(step)) or np.any(~np.isfinite(hessian)):
        raise ValueError("GDV CrdGRo inputs must be finite")
    row_norms = np.linalg.norm(hessian, axis=1)
    force_norm = float(np.linalg.norm(row_norms))
    if not math.isfinite(force_norm) or force_norm <= 0.0:
        raise ValueError("GDV CrdGRo Hessian must have a finite nonzero norm")
    base = np.clip(row_norms * math.sqrt(float(step.size)) / force_norm, 0.1, 1.0)
    displacement_blend = 0.5 * (1.0 + np.tanh(10.0 * (np.abs(step) - 0.7)))
    return base + (1.0 - base) * displacement_blend


def _accepted_optimizer_trust_update(
    damping: float,
    trust_radius: float,
    ratio: float,
    scale: float,
    cartesian_rmsd: float,
    settings: OptimizerSettings,
) -> tuple[float, float]:
    if settings.stationary_point == "transition_state":
        # l103.F enables UpdDXM by default only when NNeg==0.  Consequently
        # the ordinary Opt=(TS,ReadAllGIC,...) path keeps DXMaxT fixed at its
        # CalcHFFC value (0.300) instead of applying utilnz.F:UpdDXM.  The
        # ratio remains diagnostic, exactly as in the native TS transcript.
        del ratio, scale, cartesian_rmsd
        return 0.0, float(trust_radius)

    controller = TrustRegionController(
        ControllerSettings(
            acceptance_threshold=settings.acceptance_threshold,
            min_radius=settings.min_trust_radius,
            max_radius=settings.max_trust_radius,
            energy_noise=settings.energy_noise,
            energy_tolerance=settings.energy_tolerance,
        )
    )
    decision = controller.assess(1.0, float(ratio))
    # Expansion depends on whether the *realized Cartesian* step reached the
    # trust boundary.  The optimizer-coordinate or trial-reduction scale is
    # representation dependent and cannot be used for this decision.
    del scale
    step_fraction = float(cartesian_rmsd) / max(float(trust_radius), np.finfo(float).eps)
    if (
        decision.accepted
        and float(trust_radius) <= settings.min_trust_radius * (1.0 + 8.0 * np.finfo(float).eps)
    ):
        # An accepted step at the hard floor is evidence that the current
        # radius is usable.  Do not leave the optimizer permanently frozen
        # merely because finite-difference noise makes rho smaller than the
        # ordinary expansion threshold; escape the floor conservatively and
        # let the standard controller take over on subsequent cycles.
        new_radius = min(
            settings.max_trust_radius,
            max(
                settings.min_trust_radius * 2.0,
                settings.min_trust_radius * controller.settings.expansion_factor,
            ),
        )
    else:
        new_radius = controller.radius_after(
            trust_radius,
            decision,
            step_fraction=float(np.clip(step_fraction, 0.0, 1.0)),
        )
    new_damping = 0.0 if decision.ratio >= 0.75 else damping
    if decision.ratio < 0.10:
        new_damping = max(float(new_damping), OPTIMIZER_DAMPING_MIN)
    return new_damping, new_radius


def _rejected_optimizer_trust_update(
    damping: float,
    trust_radius: float,
    cartesian_rmsd: float,
    settings: OptimizerSettings,
) -> tuple[float, float]:
    controller = TrustRegionController(
        ControllerSettings(
            acceptance_threshold=settings.acceptance_threshold,
            min_radius=settings.min_trust_radius,
            max_radius=settings.max_trust_radius,
            energy_noise=settings.energy_noise,
            energy_tolerance=settings.energy_tolerance,
        )
    )
    return (
        max(float(damping), OPTIMIZER_DAMPING_MIN),
        controller.radius_after_rejection(
            trust_radius,
            realized_step=cartesian_rmsd,
        ),
    )


def _rejected_optimizer_cartesian_step(
    current: OptimizerEvaluation,
    trial: OptimizerEvaluation,
    proposal: StepProposal,
    message: str,
) -> float:
    """Return the Cartesian size of the geometry that was actually rejected.

    Geometry screening and backend failures occur before a trial evaluation
    can replace ``current``.  In those paths ``trial`` is deliberately the
    current evaluated state, so aligning the two states measures only floating
    point noise and can collapse the trust radius to its hard floor.  The
    proposal already carries the authoritative nonlinear Cartesian
    realization of the rejected candidate and is the correct contraction
    scale.  Energy/model rejections instead use the evaluated trial geometry.
    """

    if message in {"rejected_invalid_geometry", "rejected_evaluation"}:
        proposed = float(proposal.cartesian_rmsd_angstrom)
        if np.isfinite(proposed) and proposed > 0.0:
            return proposed
    return _cartesian_rms_displacement_angstrom(
        current.coordinates_angstrom,
        trial.coordinates_angstrom,
    )


def _evaluate_step_trials(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    step: np.ndarray,
    settings: OptimizerSettings,
    *,
    iteration: int,
    far_from_minimum: bool = False,
) -> tuple[OptimizerEvaluation, np.ndarray, float, float, float, bool, str, int]:
    current_energy = current.energy_hartree
    candidate_step = np.asarray(step, dtype=float).reshape(-1)
    # One model proposal corresponds to one QM trial.  A rejection contracts
    # the trust region and the next optimizer iteration constructs a new step;
    # LINK never converts one proposal into an in-cycle backtracking scan.
    reductions = 0
    controller = TrustRegionController(
        ControllerSettings(
            acceptance_threshold=settings.acceptance_threshold,
            min_radius=settings.min_trust_radius,
            max_radius=settings.max_trust_radius,
            energy_noise=max(
                float(settings.energy_noise),
                float(settings.energy_increase_tolerance) / 5.0,
            ),
            energy_tolerance=settings.energy_tolerance,
        )
    )
    factor = 1.0
    for attempt in range(reductions + 1):
        realized_request = candidate_step * factor
        candidate_q = q + realized_request
        geometry_message = _candidate_geometry_rejection(service, candidate_q, settings)
        if geometry_message:
            factor *= 0.5
            continue
        try:
            trial = service.evaluate(
                candidate_q,
                tag=f"step-{iteration}",
                requested_properties=(
                    "energy",
                    "gradient",
                )
                if settings.prefer_analytic_gradient
                else ("energy",),
            )
        except RuntimeError:
            return (
                current,
                np.zeros_like(candidate_step),
                0.0,
                0.0,
                -math.inf,
                False,
                "rejected_evaluation",
                1,
            )
        realized_step = np.asarray(trial.q, dtype=float) - np.asarray(q, dtype=float)
        predicted = _predicted_reduction(gradient, hessian, realized_step)
        actual = current_energy - trial.energy_hartree
        decision = controller.assess(
            predicted,
            actual,
            stationary_point=settings.stationary_point,
        )
        if (
            far_from_minimum
            and settings.stationary_point == "minimum"
            and np.isfinite(actual)
            and actual >= 0.0
        ):
            # Far from a minimum the Hessian model is deliberately treated as
            # a preconditioner, not as an acceptance oracle.  A finite
            # downhill Cauchy step is therefore accepted even when its model
            # ratio is poor; this avoids spending extra backend evaluations
            # repairing an untrusted quadratic model.
            decision = TrialDecision(
                True,
                decision.ratio,
                decision.predicted_reduction,
                decision.actual_reduction,
                "accepted_downhill_far_from_minimum",
            )
        if settings.stationary_point == "transition_state":
            rho = decision.ratio
            catastrophic = False
        else:
            rho = decision.ratio
            catastrophic = actual < -max(
                0.05,
                10.0 * abs(predicted),
                100.0 * settings.energy_tolerance,
            )
        if catastrophic:
            return (
                trial,
                realized_step,
                predicted,
                actual,
                rho,
                False,
                ("rejected_catastrophic_energy" if catastrophic else f"rejected_{decision.reason}"),
                1,
            )
        if not decision.accepted:
            if attempt < reductions:
                factor *= 0.5
                continue
            return (
                trial,
                realized_step,
                predicted,
                actual,
                rho,
                False,
                f"rejected_{decision.reason}",
                attempt + 1,
            )
        # Analytic trial results already contain the accepted-point gradient;
        # reuse them.  Energy-only backends still require their ordinary
        # accepted-point refresh before LINK forms the finite-difference
        # gradient.
        if not settings.prefer_analytic_gradient:
            trial = service.evaluate(
                trial.q,
                tag=f"accepted-{iteration}",
                requested_properties=("energy",),
            )
        if settings.stationary_point == "transition_state":
            message = "accepted_transition_state_" + decision.reason
        else:
            message = "accepted" if actual >= 0.0 else "accepted_nonmonotone"
        if attempt:
            message += f"_geometry_reduction_{attempt}"
        return trial, realized_step, predicted, actual, rho, True, message, 0
    return (
        current,
        np.zeros_like(candidate_step),
        0.0,
        0.0,
        -math.inf,
        False,
        "rejected_invalid_geometry",
        0,
    )


def _candidate_geometry_rejection(
    service: GeometryEvaluationService,
    q: np.ndarray,
    settings: OptimizerSettings,
) -> str:
    try:
        coords = service.coordinates_from_q(q)
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return f"coordinate_back_transform:{exc}"
    return _realized_geometry_rejection(service, coords, settings)


def _realized_geometry_rejection(
    service: GeometryEvaluationService,
    coords: np.ndarray,
    settings: OptimizerSettings,
) -> str:
    """Audit an already realized candidate without repeating its back-transform."""
    realization_status = service.coordinate_realization_status(coords, settings)
    if not realization_status.startswith("valid"):
        return f"coordinate_chart_invalid:{realization_status}"
    if coords.shape[0] < 2:
        return ""
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    distances[distances == 0.0] = np.inf
    min_distance = float(np.min(distances))
    if min_distance < settings.min_interatomic_distance:
        return f"short_contact:{min_distance:.6g}"
    return ""


def _predicted_reduction(gradient: np.ndarray, hessian: np.ndarray, step: np.ndarray) -> float:
    return float(-(gradient @ step + 0.5 * step @ (0.5 * (hessian + hessian.T)) @ step))


def _transition_state_stationarity_norms(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    trial: OptimizerEvaluation,
    current_internal_gradient: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[float | None, float | None]:
    """Return coordinate-invariant current/trial TS residual norms."""

    if settings.stationary_point != "transition_state":
        return None, None
    if current.gradient_hartree_per_bohr is None or trial.gradient_hartree_per_bohr is None:
        return None, None
    current_residual = _convergence_gradient(
        current,
        current_internal_gradient,
        settings,
        service,
    )
    trial_residual = _convergence_gradient(
        trial,
        np.zeros_like(np.asarray(current_internal_gradient, dtype=float)),
        settings,
        service,
    )
    if not np.all(np.isfinite(current_residual)) or not np.all(np.isfinite(trial_residual)):
        return None, None
    return float(np.linalg.norm(current_residual)), float(np.linalg.norm(trial_residual))


def _restricted_phase_model(
    gradient: np.ndarray,
    hessian: np.ndarray,
    active_mask: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    grad = np.where(mask, np.asarray(gradient, dtype=float), 0.0)
    matrix = np.zeros_like(np.asarray(hessian, dtype=float))
    active = np.flatnonzero(mask)
    inactive = np.flatnonzero(~mask)
    matrix[np.ix_(active, active)] = np.asarray(hessian, dtype=float)[np.ix_(active, active)]
    matrix[inactive, inactive] = max(1.0, settings.min_hessian_eigenvalue)
    return grad, 0.5 * (matrix + matrix.T)


def _inter_intra_micro_step(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    inter_mask: np.ndarray,
    intra_mask: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
) -> StepProposal:
    """One block Gauss--Seidel sweep followed by one Cartesian trust limit."""

    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    step = np.zeros_like(grad)
    minimum = math.inf
    maximum = 0.0
    inter = np.asarray(inter_mask, dtype=bool)
    intra = np.asarray(intra_mask, dtype=bool)
    for mask in (inter, intra, inter):
        indices = np.flatnonzero(mask)
        if not indices.size:
            continue
        residual = grad + matrix @ step
        block = matrix[np.ix_(indices, indices)]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(block)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(indices.size, dtype=float)
            eigenvectors = np.eye(indices.size, dtype=float)
        minimum = min(minimum, float(np.min(eigenvalues)))
        block_maximum = max(float(np.max(np.abs(eigenvalues))), settings.min_hessian_eigenvalue)
        maximum = max(maximum, block_maximum)
        floor = max(
            settings.min_hessian_eigenvalue,
            block_maximum / settings.max_hessian_condition,
        )
        positive = np.maximum(eigenvalues, floor)
        step[indices] += -(eigenvectors @ ((eigenvectors.T @ residual[indices]) / positive))
    # Strong inter/intra coupling can make the three block corrections nearly
    # cancel even while the full gradient is far from zero.  Accepting that
    # numerically null sweep creates a false energy plateau and poisons
    # subsequent secant updates.  Compare it with the smallest meaningful
    # Newton scale of the joint model and fall back to the safeguarded
    # Cartesian trust-region solve when the block sweep has stalled.
    gradient_norm = float(np.linalg.norm(grad))
    # Use the curvature actually inverted by the block sweep.  A very large
    # off-block coupling is precisely the situation in which the corrections
    # can cancel and must not make the null step look acceptable.
    matrix_scale = max(maximum, settings.min_hessian_eigenvalue)
    meaningful_step = 1.0e-4 * gradient_norm / matrix_scale
    if (
        float(grad @ step) >= 0.0
        or not np.all(np.isfinite(step))
        or float(np.linalg.norm(step)) < meaningful_step
    ):
        fallback = _geometric_trust_region_step(
            service,
            current,
            q,
            matrix,
            grad,
            radius,
            settings,
        )
        return replace(
            fallback,
            policy=f"inter_intra_micro_fallback:{fallback.policy}",
        )
    condition = (
        maximum / max(settings.min_hessian_eigenvalue, minimum)
        if minimum > 0.0
        else settings.max_hessian_condition
    )
    proposal = StepProposal(
        step=step,
        policy="inter_intra_micro_gauss_seidel",
        hessian_min_eigenvalue=0.0 if not np.isfinite(minimum) else minimum,
        hessian_condition=float(condition),
        damping_shift=0.0,
    )
    return proposal


def _safeguarded_gdiis_step(
    history: Sequence[tuple[np.ndarray, float, np.ndarray]],
    current_q: np.ndarray,
    current_gradient: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
) -> GDIISStepResult:
    """Build one controlled-GDIIS step from stable accepted geometries.

    The Pulay residuals are the relaxation vectors ``-H_eff^-1 g``.  Both
    those residuals and the final relaxation of the interpolated gradient use
    the exact positive effective Hessian employed by LINK's minimum RFO step.
    Numerically unstable oldest records are removed permanently; an otherwise
    stable candidate that fails a model safeguard falls back for this step
    without corrupting the retained history.
    """

    history_size = len(history)
    if history_size < 3:
        return GDIISStepResult(
            None,
            "insufficient_stable_history",
            True,
            history_size,
            history_size,
            0,
        )
    available = list(history[-settings.gdiis_history :])
    discarded_before_window = history_size - len(available)
    last_status = "insufficient_stable_history"
    # Three accepted points are the smallest stable Pulay history.
    for first in range(0, len(available) - 2):
        records = available[first:]
        size = len(records)
        gradients = np.vstack([np.asarray(item[2], dtype=float) for item in records])
        errors = np.vstack(
            [
                _minimum_effective_hessian_displacement(
                    hessian,
                    item_gradient,
                    settings,
                    metric=metric,
                )
                for item_gradient in gradients
            ]
        )
        if not np.all(np.isfinite(errors)):
            last_status = "nonfinite_residual"
            continue
        overlap = errors @ errors.T
        overlap = 0.5 * (overlap + overlap.T)
        scale = max(float(np.max(np.abs(np.diag(overlap)))), 1.0e-16)
        overlap /= scale
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(overlap)
        except np.linalg.LinAlgError:
            last_status = "rank_revealing_eigensolve_failed"
            continue
        largest = float(np.max(eigenvalues, initial=0.0))
        rank_floor = max(
            largest / settings.gdiis_max_condition,
            np.finfo(float).eps * max(1, len(records)) * max(largest, 1.0),
        )
        retained_modes = eigenvalues > rank_floor
        rank = int(np.count_nonzero(retained_modes))
        if rank < len(records):
            last_status = f"rank_deficient={rank}/{len(records)}"
            continue
        smallest = float(np.min(eigenvalues[retained_modes]))
        condition = largest / smallest
        ones = np.ones(len(records), dtype=float)
        inverse_times_ones = eigenvectors[:, retained_modes] @ (
            (eigenvectors[:, retained_modes].T @ ones) / eigenvalues[retained_modes]
        )
        denominator = float(ones @ inverse_times_ones)
        if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
            last_status = "invalid_constrained_inverse"
            continue
        coefficients = inverse_times_ones / denominator
        if np.all(np.isfinite(coefficients)) and (
            float(np.max(np.abs(coefficients))) > settings.gdiis_max_coefficient
        ):
            discarded = discarded_before_window + first
            return GDIISStepResult(
                None,
                "coefficient_bound",
                True,
                history_size,
                size,
                discarded,
            )
        if not np.all(np.isfinite(coefficients)):
            last_status = "nonfinite_coefficients"
            continue
        interpolated_q = sum(
            coefficient * item[0] for coefficient, item in zip(coefficients, records, strict=True)
        )
        interpolated_gradient = coefficients @ gradients
        relaxed_q = np.asarray(interpolated_q, dtype=float) + (
            _minimum_effective_hessian_displacement(
                hessian,
                interpolated_gradient,
                settings,
                metric=metric,
            )
        )
        step = relaxed_q - np.asarray(current_q, dtype=float)
        gradient = np.asarray(current_gradient, dtype=float)
        discarded = discarded_before_window + first
        if float(gradient @ step) >= -1.0e-12:
            return GDIISStepResult(
                None,
                "non_descent",
                True,
                history_size,
                size,
                discarded,
            )
        if _predicted_reduction(gradient, hessian, step) <= 0.0:
            return GDIISStepResult(
                None,
                "nonpositive_model",
                True,
                history_size,
                size,
                discarded,
            )
        return GDIISStepResult(
            step,
            f"accepted:n={size}:rank={rank}:condition={condition:.3g}",
            True,
            history_size,
            size,
            discarded,
        )
    # No numerically admissible suffix of three or more records remains.  Reset
    # the unusable Pulay subspace while retaining the current accepted point.
    discarded = max(0, history_size - 1)
    return GDIISStepResult(
        None,
        last_status,
        True,
        history_size,
        min(history_size, 1),
        discarded,
    )


def _gdiis_is_active(iteration: int, settings: OptimizerSettings) -> bool:
    """Return the frozen minimum-protocol activation state for this cycle."""

    return bool(
        settings.enable_gdiis
        and settings.stationary_point == "minimum"
        and int(iteration) >= settings.gdiis_start
    )


def _transported_optimizer_history(
    service: GeometryEvaluationService,
    history: Sequence[tuple[OptimizerEvaluation, np.ndarray]],
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Express saved Cartesian points and forces in the current SONIC chart."""

    transported: list[tuple[np.ndarray, float, np.ndarray]] = []
    for evaluation, saved_gradient in history:
        coordinates = np.asarray(evaluation.coordinates_angstrom, dtype=float)
        q_value = service.actual_q(coordinates)
        cartesian_gradient = evaluation.gradient_hartree_per_bohr
        if cartesian_gradient is None:
            gradient = np.asarray(saved_gradient, dtype=float).copy()
        else:
            directions_bohr = service.coordinate_directions(coordinates) * ANGSTROM_TO_BOHR
            gradient = directions_bohr @ np.asarray(cartesian_gradient, dtype=float).reshape(-1)
        transported.append((q_value, evaluation.energy_hartree, gradient))
    return transported


def _rejected_transition_state_secant_data(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    trial: OptimizerEvaluation,
    current_gradient: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a valid same-epoch analytic secant from a rejected TS trial.

    Rejection keeps the optimizer state fixed, but a finite analytic gradient
    at the evaluated trial still carries local curvature information. This
    helper performs no update itself: the main loop consumes the secant only
    for a model-quality rejection and leaves all update-informativity checks
    to the common Hessian-update path.
    """

    if settings.stationary_point != "transition_state":
        return None
    if current.chart_epoch != trial.chart_epoch:
        return None
    cartesian_gradient = trial.gradient_hartree_per_bohr
    if current.gradient_hartree_per_bohr is None or cartesian_gradient is None:
        return None
    step = np.asarray(trial.q, dtype=float).reshape(-1) - np.asarray(
        current.q, dtype=float
    ).reshape(-1)
    current_covector = np.asarray(current_gradient, dtype=float).reshape(-1)
    if step.shape != current_covector.shape or not np.all(np.isfinite(step)):
        return None
    if float(np.linalg.norm(step)) <= np.finfo(float).eps:
        return None
    directions_bohr = (
        np.asarray(
            service.coordinate_directions(trial.coordinates_angstrom),
            dtype=float,
        )
        * ANGSTROM_TO_BOHR
    )
    trial_covector = directions_bohr @ np.asarray(
        cartesian_gradient, dtype=float
    ).reshape(-1)
    y = trial_covector - current_covector
    cartesian_step_bohr = (
        aligned_cartesian_displacement(
            current.coordinates_angstrom,
            trial.coordinates_angstrom,
        ).reshape(-1)
        * ANGSTROM_TO_BOHR
    )
    if not all(
        np.all(np.isfinite(item)) for item in (trial_covector, y, cartesian_step_bohr)
    ):
        return None
    return step, y, cartesian_step_bohr


def _rejected_transition_state_secant_policy(
    message: str,
    secant_available: bool,
) -> bool:
    """DXRFO has no rejected-model secant path in the GDV protocol."""

    del message, secant_available
    return False


def _hessian_secant_update_applied(status: str) -> bool:
    normalized = str(status).lower()
    return "skipped" not in normalized and not normalized.endswith("_rejected")


def _update_hessian(
    hessian: np.ndarray,
    step: np.ndarray,
    y: np.ndarray,
    settings: OptimizerSettings,
    *,
    metric: np.ndarray | None = None,
    coordinate_unit_scales: np.ndarray | None = None,
    far_from_minimum: bool = False,
    cartesian_step_bohr: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    scheme = "bfgs" if far_from_minimum else settings.hessian_update
    if scheme == "auto":
        scheme = "bofill" if settings.stationary_point == "transition_state" else "bfgs"
    base = np.asarray(hessian, dtype=float)
    automatic = settings.hessian_update == "auto"
    status_prefix = f"{'auto_' if automatic else ''}{scheme}"
    if scheme == "bofill" and cartesian_step_bohr is not None:
        cartesian_step = np.asarray(cartesian_step_bohr, dtype=float).reshape(-1)
        if not np.all(np.isfinite(cartesian_step)):
            return 0.5 * (base + base.T), f"{status_prefix}_skipped_nonfinite_cartesian_step"
        maximum = float(np.max(np.abs(cartesian_step))) if cartesian_step.size else 0.0
        rms = _rms(cartesian_step)
        if (
            maximum <= settings.max_displacement_tolerance
            and rms <= settings.rms_displacement_tolerance
        ):
            return 0.5 * (
                base + base.T
            ), f"{status_prefix}_skipped_subresolution_cartesian_step"
    if scheme == "sr1":
        updated, status = _sr1_update(base, step, y)
    elif scheme == "bofill":
        if coordinate_unit_scales is None:
            updated, status = _bofill_update(base, step, y)
        else:
            unit_scales = np.asarray(coordinate_unit_scales, dtype=float).reshape(-1)
            if (
                unit_scales.shape != np.asarray(step, dtype=float).reshape(-1).shape
                or np.any(~np.isfinite(unit_scales))
                or np.any(unit_scales <= 0.0)
            ):
                raise ValueError("Bofill coordinate-unit scales do not match the secant")
            inverse_unit_scales = 1.0 / unit_scales
            gdv_hessian = (
                inverse_unit_scales[:, None]
                * base
                * inverse_unit_scales[None, :]
            )
            gdv_step = unit_scales * np.asarray(step, dtype=float).reshape(-1)
            gdv_y = inverse_unit_scales * np.asarray(y, dtype=float).reshape(-1)
            gdv_updated, status = _bofill_update(gdv_hessian, gdv_step, gdv_y)
            updated = unit_scales[:, None] * gdv_updated * unit_scales[None, :]
            status += "_gdv_bohr_radian_units"
    else:
        update_base = base
        used_positive_step_base = False
        if settings.stationary_point == "minimum":
            effective, inverse_sqrt_metric, minimum, _condition, _floor = (
                _minimum_effective_hessian_model(base, settings, metric=metric)
            )
            spectral_scale = max(
                1.0,
                float(np.linalg.norm(effective, ord=2)),
            )
            positivity_tolerance = (
                100.0 * np.finfo(float).eps * max(int(base.shape[0]), 1) * spectral_scale
            )
            if minimum <= positivity_tolerance:
                sqrt_metric = np.linalg.inv(inverse_sqrt_metric)
                update_base = sqrt_metric @ effective @ sqrt_metric
                update_base = 0.5 * (update_base + update_base.T)
                used_positive_step_base = True
        updated, status = _bfgs_update(
            update_base,
            step,
            y,
            damp=settings.bfgs_damping,
        )
        if used_positive_step_base:
            if status == "bfgs_skipped_curvature":
                updated = base
            else:
                status += "_from_positive_step_model"
    if automatic:
        status = f"auto_{status}"
    if not _stored_hessian_is_numerically_usable(updated, settings):
        return 0.5 * (
            np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T
        ), f"{status}_rejected"
    return updated, status


def _gdv_d2corx_history_update(
    hessian: np.ndarray,
    *,
    current_q: np.ndarray,
    current_gradient: np.ndarray,
    history: Sequence[tuple[np.ndarray, np.ndarray]],
    settings: OptimizerSettings,
    coordinate_unit_scales: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Apply GDV ``D2CorX`` method 4 to the accepted-point history.

    GDV stores the current point in slot one and up to ``NVar`` previous
    points in reverse chronological slots. ``D2CorX`` visits those slots
    from oldest to newest and applies a Bofill MSP update for every
    admissible secant. The immediately preceding point is exempt from the
    ``RMax`` distance test, exactly as in ``utilam.F``.
    """

    base = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    q_now = np.asarray(current_q, dtype=float).reshape(-1)
    gradient_now = np.asarray(current_gradient, dtype=float).reshape(-1)
    scales = np.asarray(coordinate_unit_scales, dtype=float).reshape(-1)
    if q_now.shape != gradient_now.shape or scales.shape != q_now.shape:
        raise ValueError("GDV D2CorX vectors do not match the Hessian dimension")
    if base.shape != (q_now.size, q_now.size):
        raise ValueError("GDV D2CorX Hessian dimension does not match its vectors")
    if (
        np.any(~np.isfinite(q_now))
        or np.any(~np.isfinite(gradient_now))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("GDV D2CorX requires finite coordinates, gradients, and scales")

    inverse_scales = 1.0 / scales
    gdv_hessian = inverse_scales[:, None] * base * inverse_scales[None, :]
    gdv_q_now = scales * q_now
    gdv_gradient_now = inverse_scales * gradient_now
    retained_history = list(history)[-q_now.size :]
    minimum_distance = 4.0 * float(settings.rms_force_tolerance)
    minimum_distance2 = minimum_distance * minimum_distance
    maximum_distance2 = GDV_D2CORX_MAX_HISTORY_DISTANCE**2
    selected = 0
    selected_weights: list[str] = []
    for history_index, (history_q, history_gradient) in enumerate(retained_history):
        q_old = np.asarray(history_q, dtype=float).reshape(-1)
        gradient_old = np.asarray(history_gradient, dtype=float).reshape(-1)
        if q_old.shape != q_now.shape or gradient_old.shape != q_now.shape:
            raise ValueError("GDV D2CorX history does not match the current point")
        if np.any(~np.isfinite(q_old)) or np.any(~np.isfinite(gradient_old)):
            continue
        step = gdv_q_now - scales * q_old
        gradient_change = gdv_gradient_now - inverse_scales * gradient_old
        distance2 = float(step @ step)
        gradient_distance = float(np.linalg.norm(gradient_change))
        immediate_previous = history_index == len(retained_history) - 1
        if (
            (not immediate_previous and distance2 > maximum_distance2)
            or distance2 < minimum_distance2
            or gradient_distance < GDV_D2CORX_GRADIENT_ERROR
        ):
            continue
        gdv_hessian, status = _bofill_update(
            gdv_hessian,
            step,
            gradient_change,
        )
        selected += 1
        selected_weights.append(status.removeprefix("bofill_msp_"))

    updated = scales[:, None] * gdv_hessian * scales[None, :]
    updated = 0.5 * (updated + updated.T)
    point_count = selected + 1
    weight_status = ",".join(selected_weights) if selected_weights else "none"
    return (
        updated,
        "auto_bofill_d2corx_"
        f"points={point_count}_weights={weight_status}_gdv_bohr_radian_units",
    )


def _refresh_transition_index_subspace_from_analytic_gradients(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    transition_mode_reference: TransitionModeReference | None,
) -> tuple[np.ndarray, str]:
    """Refresh only the subspace needed to determine first-order inertia.

    This is not a final exact Hessian.  Central analytic-gradient probes span
    the tracked reaction direction and every negative direction of the
    approximate Hessian.  A symmetric block-secant update then replaces only
    information constrained by those probes before the stationary point is
    classified.
    """

    if current.gradient_hartree_per_bohr is None:
        raise ValueError("transition-index subspace refresh needs analytic gradients")
    coordinate_directions = service.coordinate_directions(current.coordinates_angstrom)
    matrix, directions, indices = _transition_index_refresh_basis(
        service,
        current,
        hessian,
        settings,
        transition_mode_reference,
        coordinate_directions,
    )
    cartesian_from_q = np.asarray(coordinate_directions, dtype=float).T
    natoms = max(int(current.coordinates_angstrom.shape[0]), 1)
    directional_gradients = np.zeros_like(directions)
    restart = _restart_evaluation_kwargs(current)

    for column in range(directions.shape[1]):
        directional_gradients[:, column] = _transition_index_probe_gradient(
            service,
            q,
            directions[:, column],
            cartesian_from_q,
            natoms,
            settings,
            column,
            restart,
        )

    before = _optimizer_hessian_index(matrix, settings)
    refreshed = symmetric_multisecant_hessian_refresh(
        matrix,
        directions,
        directional_gradients,
    )
    after = _optimizer_hessian_index(refreshed, settings)
    return (
        refreshed,
        f"transition_index_subspace_refresh_{before}_to_{after}_modes_{indices.size}",
    )


def _transition_index_refresh_basis(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    hessian: np.ndarray,
    settings: OptimizerSettings,
    transition_mode_reference: TransitionModeReference | None,
    coordinate_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    orthonormal, inverse_sqrt_metric = _orthonormal_optimizer_hessian(
        matrix,
        service.optimizer_metric(current.coordinates_angstrom),
        settings,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(orthonormal)
    candidates = _cartesian_transition_mode_candidates(
        eigenvectors,
        inverse_sqrt_metric,
        coordinate_directions,
    )
    transition_mode, _overlap, _reference = _select_transition_mode(
        candidates,
        settings,
        transition_mode_reference,
        current.coordinates_angstrom,
        eigenvalues=eigenvalues,
        reaction_directions=_transition_reaction_cartesian_directions(
            service.coordinate_model,
            coordinate_directions,
        ),
    )
    selected = {
        int(index)
        for index in np.flatnonzero(eigenvalues < -settings.min_hessian_eigenvalue)
    }
    selected.add(int(transition_mode))
    indices = np.asarray(sorted(selected), dtype=int)
    return matrix, inverse_sqrt_metric @ eigenvectors[:, indices], indices


def _transition_index_probe_gradient(
    service: GeometryEvaluationService,
    q: np.ndarray,
    direction: np.ndarray,
    cartesian_from_q: np.ndarray,
    natoms: int,
    settings: OptimizerSettings,
    column: int,
    restart: Mapping[str, object],
) -> np.ndarray:
    cartesian_tangent = cartesian_from_q @ direction
    rms_per_unit = float(np.linalg.norm(cartesian_tangent) / math.sqrt(natoms))
    if not np.isfinite(rms_per_unit) or rms_per_unit <= 1.0e-14:
        raise ValueError("transition-index probe direction has zero Cartesian norm")
    displacement = float(settings.transition_index_probe_rms_angstrom) / rms_per_unit
    evaluations = tuple(
        service.evaluate(
            np.asarray(q, dtype=float) + sign * displacement * direction,
            tag=f"transition-index-{label}-{column}",
            requested_properties=("energy", "gradient"),
            **restart,
        )
        for sign, label in ((1.0, "plus"), (-1.0, "minus"))
    )
    gradients = []
    for evaluation in evaluations:
        cartesian_gradient = evaluation.gradient_hartree_per_bohr
        if cartesian_gradient is None:
            raise RuntimeError("backend omitted a transition-index probe gradient")
        gradients.append(
            (service.coordinate_directions(evaluation.coordinates_angstrom) * ANGSTROM_TO_BOHR)
            @ cartesian_gradient
        )
    return (gradients[0] - gradients[1]) / (2.0 * displacement)


def _transition_state_index_convergence_state(
    hessian: np.ndarray,
    settings: OptimizerSettings,
    *,
    refresh_attempted: bool,
) -> tuple[bool, bool, str | None]:
    """Keep raw Hessian inertia diagnostic, not a convergence hard stop.

    LINK follows an index-one *step model* built from the transported
    ORACLE/SMITH reaction direction.  The raw approximate Hessian index may
    therefore be zero or higher than one during the search and is reported as
    diagnostics rather than used to reject a valid stationary-point search.
    """

    if settings.stationary_point != "transition_state":
        return True, refresh_attempted, None
    return True, refresh_attempted, None


def _geometry_status(reference: np.ndarray, coordinates: np.ndarray) -> str:
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape[0] < 2:
        return "ok"
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    distances[distances == 0.0] = np.inf
    min_distance = float(np.min(distances))
    if min_distance < 0.35:
        return f"short_contact:{min_distance:.6g}"
    displacement = aligned_cartesian_displacement(reference, coordinates)
    max_displacement = float(np.max(np.abs(displacement))) if displacement.size else 0.0
    return f"ok:max_cartesian_displacement={max_displacement:.6g}"


def _append_trace(path: Path, record: OptimizerIteration, model: OptimizerCoordinateModel) -> None:
    payload = {
        "schema": OPTIMIZER_TRACE_SCHEMA,
        "iteration": record.iteration,
        "status": record.status,
        "energy_hartree": record.energy_hartree,
        "trial_energy_hartree": record.trial_energy_hartree,
        "gradient_inf_norm": record.gradient_inf_norm,
        "gradient_rms_norm": record.gradient_rms_norm,
        "step_norm": record.step_norm,
        "step_inf_norm": record.step_inf_norm,
        "step_rms_norm": record.step_rms_norm,
        "energy_change_hartree": record.energy_change_hartree,
        "convergence": record.convergence,
        "trust_radius": record.trust_radius,
        "trust_ratio": record.trust_ratio,
        "gradient_policy": record.gradient_policy,
        "fd_mode": record.fd_mode,
        "fd_step_min": record.fd_step_min,
        "fd_step_max": record.fd_step_max,
        "class_threshold_fraction": record.class_threshold_fraction,
        "class_screen_audit": record.class_screen_audit,
        "class_screen_audit_interval": record.class_screen_audit_interval,
        "chart_epoch": record.chart_epoch,
        "chart_lifecycle_status": record.chart_lifecycle_status,
        "refreshed_coordinate_count": record.refreshed_coordinate_count,
        "predicted_coordinate_count": record.predicted_coordinate_count,
        "active_coordinate_fraction": record.active_coordinate_fraction,
        "fd_one_sided_count": record.fd_one_sided_count,
        "fd_two_sided_count": record.fd_two_sided_count,
        "fd_parallel_workers": record.fd_parallel_workers,
        "local_group_count": record.local_group_count,
        "local_group_sizes": list(record.local_group_sizes),
        "surrogate_sample_count": record.surrogate_sample_count,
        "hessian_sparsity": record.hessian_sparsity,
        "hessian_min_eigenvalue": record.hessian_min_eigenvalue,
        "hessian_condition": record.hessian_condition,
        "hessian_update_status": record.hessian_update_status,
        "step_policy": record.step_policy,
        "gdiis_attempted": record.gdiis_attempted,
        "gdiis_used": record.gdiis_used,
        "gdiis_status": record.gdiis_status,
        "gdiis_history_size": record.gdiis_history_size,
        "gdiis_retained_history_size": record.gdiis_retained_history_size,
        "gdiis_discarded_history_size": record.gdiis_discarded_history_size,
        "rejected_trial_count": record.rejected_trial_count,
        "geometry_status": record.geometry_status,
        "coordinate_model_status": record.coordinate_model_status,
        "selective_fallback_status": record.selective_fallback_status,
        "symmetry_status": record.symmetry_status,
        "qm_evaluations": record.qm_evaluations,
        "energy_evaluations": record.energy_evaluations,
        "gradient_evaluations": record.gradient_evaluations,
        "hessian_evaluations": record.hessian_evaluations,
        "fd_displacements": record.fd_displacements,
        "cache_hits": record.cache_hits,
        "avoided_evaluations": record.avoided_evaluations,
        "proposed_cartesian_rmsd_angstrom": record.proposed_cartesian_rmsd_angstrom,
        "trust_step_scale": record.trust_step_scale,
        "trust_solver_iterations": record.trust_solver_iterations,
        "applied_trust_radius_angstrom": record.applied_trust_radius_angstrom,
        "predicted_reduction_hartree": record.predicted_reduction_hartree,
        "actual_reduction_hartree": record.actual_reduction_hartree,
        "current_projected_gradient_norm": record.current_projected_gradient_norm,
        "trial_projected_gradient_norm": record.trial_projected_gradient_norm,
        "trial_cartesian_rmsd_angstrom": record.trial_cartesian_rmsd_angstrom,
        "transition_mode_index": record.transition_mode_index,
        "transition_mode_overlap": record.transition_mode_overlap,
        "active_coordinates": list(model.labels),
        "message": record.message,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_xyz(path: Path, atoms: tuple[str, ...], coords: np.ndarray, comment: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n{comment}\n")
        for atom, xyz in zip(atoms, np.asarray(coords, dtype=float), strict=True):
            handle.write(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}\n")


def _write_xyz(path: Path, atoms: tuple[str, ...], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [str(len(atoms)), comment]
    for atom, xyz in zip(atoms, np.asarray(coords, dtype=float), strict=True):
        text.append(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _format_command(template: str, **values: object) -> list[str]:
    formatted = template.format(**{key: str(value) for key, value in values.items()})
    return shlex.split(formatted)
