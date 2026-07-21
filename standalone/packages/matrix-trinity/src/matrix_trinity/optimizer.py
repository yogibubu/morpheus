from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import json
import math
from pathlib import Path
import shlex
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

from matrix_core.geometry_alignment import (
    aligned_cartesian_displacement,
    kabsch_align,
    kabsch_rotation,
    rotate_cartesian_derivatives,
)
from matrix_core.xyzin_geometry import read_xyzin_geometry
from matrix_link import cartesian_from_internal_jacobian, internal_from_cartesian_jacobian

from .scan import (
    ANGSTROM_TO_BOHR,
    PointEvaluationResult,
    PESExplorationPolicy,
    QMScanBackend,
    ScanPoint,
    _coordinate_index,
    _normalized_backend_name,
    coordinate_direction_from_gic,
    point_result_to_json,
    read_point_result,
    run_qm_scan_points,
    write_point_result,
    prepare_pes_exploration_geometry,
)


OPTIMIZER_TRACE_SCHEMA = "matrix.trinity.information_efficient_optimizer.trace.v1"
OPTIMIZER_SUMMARY_SCHEMA = "matrix.trinity.information_efficient_optimizer.summary.v1"
OPTIMIZER_CACHE_SCHEMA = "matrix.trinity.information_efficient_optimizer.cache.v1"
OPTIMIZER_HESSIAN_SCHEMA = "matrix.trinity.information_efficient_optimizer.hessian.v1"
OPTIMIZER_DAMPING_MIN = 1.0e-12


@dataclass(frozen=True)
class OptimizerSettings:
    max_steps: int = 50
    trust_radius: float = 0.2
    max_trust_radius: float = 0.3
    min_trust_radius: float = 1.0e-4
    gradient_tolerance: float = 4.5e-4
    step_tolerance: float = 1.8e-3
    energy_tolerance: float = 1.0e-6
    max_force_tolerance: float | None = None
    rms_force_tolerance: float | None = None
    max_displacement_tolerance: float | None = None
    rms_displacement_tolerance: float | None = None
    fd_step: float = 0.01
    fd_hard_characteristic_scale: float = 0.05
    fd_soft_characteristic_scale: float = 0.20
    fd_min_step: float = 1.0e-4
    fd_max_step: float = 0.05
    energy_noise: float = 1.0e-8
    energy_noise_samples: int = 0
    two_sided: bool = True
    adaptive_fd_mode: bool = False
    fd_central_gradient_factor: float = 5.0
    fd_totally_symmetric_only: bool = False
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
    prefer_analytic_gradient: bool = True
    cache_tolerance: float = 1.0e-10
    resume: bool = False
    min_abs_metric_diagonal: float = 1.0e-8
    acceptance_threshold: float = 0.10
    bfgs_damping: bool = True
    min_hessian_eigenvalue: float = 1.0e-4
    max_hessian_condition: float = 1.0e8
    max_coordinate_step: float = 0.25
    line_search_reductions: int = 6
    energy_increase_tolerance: float | None = None
    hessian_update: str = "auto"
    initial_hessian_model: str = "auto"
    enable_gdiis: bool = False
    hessian_reset_on_bad_update: bool = True
    coordinate_drift_warning: float = 0.25
    min_interatomic_distance: float = 0.35
    gdiis_history: int = 6
    gdiis_start: int = 6
    gdiis_max_condition: float = 1.0e4
    gdiis_max_coefficient: float = 2.0
    hessian_bad_ratio_limit: int = 2
    fragment_radial_curvature: float | None = None
    fragment_tangential_curvature: float | None = None
    fragment_rotation_curvature: float | None = None
    coordinate_schedule: str = "auto"
    coordinate_phase_max_steps: int = 8
    coordinate_phase_gradient_factor: float = 3.0
    fixed_atoms: tuple[int, ...] = ()
    freeze_inactive_sonic: bool = True
    backtransform_continuation_step: float = 0.12
    backtransform_max_substeps: int = 32
    rigid_reference_groups: tuple[tuple[int, ...], ...] = ()
    include_cv_exponential_field: bool = False

    def __post_init__(self) -> None:
        for name in (
            "trust_radius",
            "max_trust_radius",
            "min_trust_radius",
            "gradient_tolerance",
            "step_tolerance",
            "energy_tolerance",
            "fd_step",
            "fd_hard_characteristic_scale",
            "fd_soft_characteristic_scale",
            "fd_min_step",
            "fd_max_step",
            "energy_noise",
            "fd_gradient_change_tolerance",
            "fd_central_gradient_factor",
            "selective_min_refresh_fraction",
            "selective_coupling_threshold",
            "selective_fallback_gradient_growth",
            "hessian_coupling_threshold",
            "cache_tolerance",
            "min_abs_metric_diagonal",
            "acceptance_threshold",
            "min_hessian_eigenvalue",
            "max_hessian_condition",
            "max_coordinate_step",
            "coordinate_drift_warning",
            "min_interatomic_distance",
            "gdiis_max_condition",
            "gdiis_max_coefficient",
            "fragment_radial_curvature",
            "fragment_tangential_curvature",
            "fragment_rotation_curvature",
            "coordinate_phase_gradient_factor",
            "backtransform_continuation_step",
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
        object.__setattr__(self, "energy_noise_samples", int(self.energy_noise_samples))
        object.__setattr__(self, "fd_refresh_interval", int(self.fd_refresh_interval))
        object.__setattr__(
            self, "selective_fallback_rejections", int(self.selective_fallback_rejections)
        )
        object.__setattr__(self, "surrogate_max_samples", int(self.surrogate_max_samples))
        object.__setattr__(self, "fd_parallel_workers", int(self.fd_parallel_workers))
        object.__setattr__(self, "line_search_reductions", int(self.line_search_reductions))
        object.__setattr__(self, "gdiis_history", int(self.gdiis_history))
        object.__setattr__(self, "gdiis_start", int(self.gdiis_start))
        object.__setattr__(self, "hessian_bad_ratio_limit", int(self.hessian_bad_ratio_limit))
        object.__setattr__(self, "coordinate_phase_max_steps", int(self.coordinate_phase_max_steps))
        object.__setattr__(
            self, "backtransform_max_substeps", int(self.backtransform_max_substeps)
        )
        object.__setattr__(
            self, "coordinate_schedule", str(self.coordinate_schedule).strip().lower()
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
        object.__setattr__(self, "hessian_update", str(self.hessian_update).strip().lower())
        object.__setattr__(
            self, "initial_hessian_model", str(self.initial_hessian_model).strip().lower()
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
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
        if self.fd_refresh_interval <= 0:
            raise ValueError("fd_refresh_interval must be positive")
        if self.surrogate_max_samples < 0:
            raise ValueError("surrogate_max_samples must be non-negative")
        if self.fd_parallel_workers <= 0:
            raise ValueError("fd_parallel_workers must be positive")
        if self.line_search_reductions < 0:
            raise ValueError("line_search_reductions must be non-negative")
        if self.gdiis_history < 2 or self.gdiis_start < 2:
            raise ValueError("GDIIS history and start must be at least two")
        if self.gdiis_max_condition <= 1.0 or self.gdiis_max_coefficient <= 0.0:
            raise ValueError("invalid GDIIS safeguards")
        if self.hessian_bad_ratio_limit <= 0:
            raise ValueError("hessian_bad_ratio_limit must be positive")
        if self.coordinate_phase_max_steps <= 0 or self.coordinate_phase_gradient_factor <= 0.0:
            raise ValueError("invalid coordinate phase controls")
        if self.backtransform_continuation_step <= 0.0 or self.backtransform_max_substeps <= 0:
            raise ValueError("invalid hybrid back-transform continuation controls")
        if self.coordinate_schedule not in {
            "auto",
            "joint",
            "inter-intra-joint",
            "inter-intra-micro",
        }:
            raise ValueError(
                "coordinate_schedule must be 'auto', 'joint', 'inter-intra-joint' or 'inter-intra-micro'"
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
        if (self.fragment_radial_curvature is None) != (self.fragment_tangential_curvature is None):
            raise ValueError("radial and tangential fragment curvatures must be set together")
        if self.trust_radius <= 0.0 or self.max_trust_radius <= 0.0:
            raise ValueError("trust radii must be positive")
        if self.max_coordinate_step <= 0.0:
            raise ValueError("max_coordinate_step must be positive")
        if self.min_interatomic_distance <= 0.0:
            raise ValueError("min_interatomic_distance must be positive")
        if self.min_hessian_eigenvalue <= 0.0:
            raise ValueError("min_hessian_eigenvalue must be positive")
        if self.max_hessian_condition <= 1.0:
            raise ValueError("max_hessian_condition must be greater than one")
        if self.hessian_update not in {"auto", "bfgs", "sr1", "bofill"}:
            raise ValueError("hessian_update must be 'auto', 'bfgs', 'sr1' or 'bofill'")
        if self.initial_hessian_model not in {"auto", "berny", "almloef"}:
            raise ValueError("initial_hessian_model must be 'auto', 'berny' or 'almloef'")
        if self.fd_min_step <= 0.0 or self.fd_max_step < self.fd_min_step:
            raise ValueError("invalid finite-difference step bounds")
        if (
            self.fd_hard_characteristic_scale <= 0.0
            or self.fd_soft_characteristic_scale <= 0.0
        ):
            raise ValueError("hard and soft characteristic scales must be positive")
        if self.fd_central_gradient_factor <= 0.0:
            raise ValueError("fd_central_gradient_factor must be positive")
        if self.selective_min_refresh_fraction < 0.0 or self.selective_min_refresh_fraction > 1.0:
            raise ValueError("selective_min_refresh_fraction must be between zero and one")
        if self.selective_coupling_threshold < 0.0:
            raise ValueError("selective_coupling_threshold must be non-negative")
        if self.selective_fallback_rejections < 0:
            raise ValueError("selective_fallback_rejections must be non-negative")
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
        reference = self.reference_values
        if reference is not None:
            reference = np.asarray(reference, dtype=float).reshape(-1)
            if reference.shape != (len(self.labels),) or not np.all(np.isfinite(reference)):
                raise ValueError("coordinate reference values must match coordinate labels")
        object.__setattr__(self, "directions_angstrom", directions)
        object.__setattr__(self, "metric_diagonal", metric)
        object.__setattr__(self, "sonic_labels", sonic_labels)
        object.__setattr__(self, "sonic_from_coordinates", transform)
        object.__setattr__(self, "reference_values", reference)
        retained_group = str(self.retained_group).strip().upper()
        if self.pes_exploration and not retained_group:
            retained_group = "C1"
        object.__setattr__(self, "retained_group", retained_group)


@dataclass(frozen=True)
class OptimizerEvaluation:
    q: np.ndarray
    coordinates_angstrom: np.ndarray
    result: PointEvaluationResult
    cache_hit: bool = False

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
    message: str = ""


@dataclass(frozen=True)
class StepProposal:
    step: np.ndarray
    policy: str
    hessian_min_eigenvalue: float
    hessian_condition: float
    damping_shift: float = 0.0


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
    gradient_evaluations: int
    hessian_evaluations: int
    fd_displacements: int
    cache_hits: int
    avoided_evaluations: int
    cache_path: Path
    trajectory_path: Path
    trace_path: Path
    summary_path: Path
    final_hessian_path: Path
    initial_hessian_source: str


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
        self.hits = 0
        self.misses = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if resume and self.path.is_file():
                self._load(self.path)
            elif not resume:
                self.path.write_text("", encoding="utf-8")

    def lookup(self, q: np.ndarray) -> OptimizerEvaluation | None:
        vector = np.asarray(q, dtype=float).reshape(-1)
        for record in self.records:
            delta = vector - np.asarray(record.q, dtype=float).reshape(-1)
            distance2 = float(np.sum(self.metric_diagonal * delta * delta))
            if distance2 <= self.tolerance * self.tolerance:
                self.hits += 1
                return OptimizerEvaluation(
                    q=record.q,
                    coordinates_angstrom=record.coordinates_angstrom,
                    result=record.result,
                    cache_hit=True,
                )
        self.misses += 1
        return None

    def add(self, evaluation: OptimizerEvaluation, *, persist: bool = True) -> None:
        self.records.append(evaluation)
        if persist and self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_cache_record_to_json(evaluation), sort_keys=True) + "\n")

    def _load(self, path: Path) -> None:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if payload.get("schema") != OPTIMIZER_CACHE_SCHEMA:
                continue
            self.records.append(_cache_record_from_json(payload))


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
    ) -> None:
        self.xyzin_path = Path(xyzin_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.geometry = read_xyzin_geometry(self.xyzin_path)
        self.atoms = tuple(self.geometry.atoms)
        self.reference_coordinates = np.asarray(self.geometry.coordinates_angstrom, dtype=float)
        self.coordinate_model = coordinate_model
        self.engine_command = engine_command
        self.backend = backend
        self.timeout = timeout
        self.settings = settings or OptimizerSettings()
        self.pes_exploration_policy = pes_exploration_policy
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
            int(item.result.execution.get("fd_displacements", 0))
            for item in self.cache.records
        )
        self._projector_state: CoordinateProjectorState | None = None
        self._sonic_definition = None
        self._sonic_coordinate_indices: tuple[int, ...] = ()
        self._sonic_reference_values: np.ndarray | None = None
        self._sonic_full_reference_values: np.ndarray | None = None
        self._sonic_rotation_atlas = None
        self._sonic_active_families: tuple[str, ...] = ()
        self._assigned_cartesian_symmetry = None
        self._last_backtransform_diagnostics: dict[str, object] | None = None
        self._cv_atomic_numbers: tuple[int, ...] = ()
        self._cv_bonded_pairs: tuple[tuple[int, int], ...] = ()
        if self.settings.include_cv_exponential_field:
            from matrix_chem import read_primitive_contract
            from matrix_chem.topology.elements import atomic_number

            self._cv_atomic_numbers = tuple(int(atomic_number(atom) or 0) for atom in self.atoms)
            contract = read_primitive_contract(self.xyzin_path)
            self._cv_bonded_pairs = tuple(
                primitive.atoms for primitive in contract.primitives if primitive.kind == "bond"
            )
        if coordinate_model.kind == "sonic":
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
            if (
                definition.symmetrize
                and str(definition.point_group).strip().upper() not in {"", "C1", "UNKNOWN"}
                and self.settings.freeze_inactive_sonic
            ):
                from matrix_chem.symmetry import analyze_molecular_symmetry
                from matrix_chem import MolecularGeometry, read_symmetry_thresholds

                reference_geometry = MolecularGeometry(
                    atoms=self.atoms,
                    coordinates_angstrom=self.reference_coordinates,
                )
                symmetry_thresholds = read_symmetry_thresholds(self.xyzin_path)
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
            sonic_labels = coordinate_model.sonic_labels or coordinate_model.labels
            self._sonic_coordinate_indices = tuple(
                _coordinate_index(label, labels, names) for label in sonic_labels
            )
            if coordinate_model.sonic_from_coordinates is None:
                self._sonic_active_families = tuple(
                    definition.gics[index].family for index in self._sonic_coordinate_indices
                )
            if not self.settings.freeze_inactive_sonic:
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
            self._sonic_full_reference_values = evaluate_gic_values(
                definition, coordinates_angstrom=self.reference_coordinates
            )
            self._sonic_reference_values = self._sonic_full_reference_values[
                list(self._sonic_coordinate_indices)
            ]

    def initialize_coordinate_projector(
        self,
        q: Sequence[float] | np.ndarray,
        coordinates_angstrom: np.ndarray,
    ) -> None:
        if self.coordinate_model.kind != "sonic":
            return
        vector = np.asarray(q, dtype=float).reshape(-1).copy()
        coords = np.asarray(coordinates_angstrom, dtype=float).copy()
        if vector.shape != (self.coordinate_model.directions_angstrom.shape[0],):
            raise ValueError("projector q length does not match coordinate model")
        if coords.shape != self.reference_coordinates.shape:
            raise ValueError("projector coordinates do not match reference geometry")
        self._projector_state = CoordinateProjectorState(
            q=vector,
            coordinates_angstrom=coords,
            cartesian_from_q=self._sonic_cartesian_from_q(coords),
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
        if self.coordinate_model.kind != "sonic":
            return
        if self._projector_state is None:
            self.initialize_coordinate_projector(current_q, current_coordinates)
            return
        state = self._projector_state
        from matrix_link import secant_projector_update, should_refresh_coordinate_model

        rotation_reset = self._maybe_rebase_sonic_rotations(current_coordinates)

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
                cartesian_from_q=self._sonic_cartesian_from_q(current_coordinates),
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
                cartesian_from_q=self._sonic_cartesian_from_q(current_coordinates),
                age=0,
                analytic_refreshes=state.analytic_refreshes + 1,
                secant_updates=state.secant_updates,
                secant_rejections=state.secant_rejections + 1,
                last_secant_error=secant_error,
            )

    def refresh_coordinate_projector(self, q: np.ndarray, coordinates_angstrom: np.ndarray) -> None:
        if self.coordinate_model.kind == "sonic" and self._projector_state is not None:
            self.initialize_coordinate_projector(q, coordinates_angstrom)

    def coordinate_model_status(self, coordinates: np.ndarray, settings: OptimizerSettings) -> str:
        displacement = np.asarray(coordinates, dtype=float) - self.reference_coordinates
        max_displacement = float(np.max(np.abs(displacement))) if displacement.size else 0.0
        if self.coordinate_model.kind != "sonic" or self._projector_state is None:
            return f"ok:max_drift={max_displacement:.6g}"
        state = self._projector_state
        prefix = (
            f"projector_age={state.age}:refreshes={state.analytic_refreshes}:"
            f"secant={state.secant_updates}:secant_rejected={state.secant_rejections}:"
            f"secant_error={state.last_secant_error:.6g}:max_drift={max_displacement:.6g}"
        )
        if self._last_backtransform_diagnostics is not None:
            diagnostic = self._last_backtransform_diagnostics
            prefix += (
                f":backtransform={diagnostic['method']}:substeps={diagnostic['substeps']}:"
                f"finite_frag={diagnostic['finite_fragment_count']}:"
                f"finite_tors={diagnostic['finite_torsion_count']}:"
                f"continuation={diagnostic['continuation_count']}:"
                f"linear_fallback={int(bool(diagnostic['linear_fallback']))}"
            )
        if max_displacement > settings.coordinate_drift_warning:
            return "frozen_sonic_drift_warning:" + prefix
        return "ok:" + prefix

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

        size = len(self.coordinate_model.labels)
        if self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            diagonal = np.abs(np.diag(np.asarray(hessian, dtype=float)))
            return diagonal <= np.median(diagonal)
        from matrix_link import soft_coordinate_indices

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
        """Reject numerical-optimization variables outside the total irrep."""

        if self.coordinate_model.kind != "sonic" or self._sonic_definition is None:
            raise ValueError("adaptive numerical gradients require a SONIC coordinate model")
        from matrix_smith.symmetry_labels import is_total_symmetric_irrep

        definition = self._sonic_definition
        non_total = tuple(
            self.coordinate_model.sonic_labels[index]
            if self.coordinate_model.sonic_labels
            else self.coordinate_model.labels[index]
            for index, coordinate_index in enumerate(self._sonic_coordinate_indices)
            if not is_total_symmetric_irrep(
                definition.point_group, definition.gics[coordinate_index].irrep
            )
        )
        if non_total:
            raise ValueError(
                "adaptive numerical gradients accept only totally symmetric SONICs; "
                f"non-total coordinates: {', '.join(non_total)}"
            )

    def coordinates_from_q(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(q, dtype=float).reshape(-1)
        if vector.shape != (self.coordinate_model.directions_angstrom.shape[0],):
            raise ValueError("q length does not match coordinate model")
        if self.coordinate_model.kind == "sonic" and self._projector_state is not None:
            from matrix_link import (
                hybrid_internal_coordinate_step,
                nonlinear_internal_coordinate_step,
            )

            state = self._projector_state
            if self.settings.freeze_inactive_sonic:
                target_values = self._absolute_sonic_values(vector)
                evaluate = self._evaluate_all_sonic
            else:
                if self._sonic_reference_values is None:
                    raise RuntimeError("SONIC reference values are unavailable")
                target_values = self._sonic_reference_values + self._sonic_displacements(vector)
                evaluate = self._evaluate_active_sonic
            result = hybrid_internal_coordinate_step(
                self._sonic_definition,
                state.coordinates_angstrom,
                target_values,
                evaluate,
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
                "continuation_count": len(result.continuation_indices),
                "corrector_iterations": result.corrector_iterations,
                "linear_fallback": not hybrid_converged,
            }
            if not result.converged:
                # Compatibility fallback for unusual mixed coordinates not yet
                # classified by the finite predictor.  It starts from the
                # hybrid result, so the old pseudoinverse is never the primary
                # mechanism for a large soft displacement.
                result = nonlinear_internal_coordinate_step(
                    result.coordinates_angstrom,
                    target_values,
                    evaluate,
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

    def _prepare_pes_exploration_geometry(
        self, coordinates_angstrom: np.ndarray
    ) -> np.ndarray:
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
        if self._assigned_cartesian_symmetry is not None:
            from matrix_chem import MolecularGeometry
            from matrix_chem.symmetry import symmetrize_molecular_geometry

            coords = symmetrize_molecular_geometry(
                MolecularGeometry(atoms=self.atoms, coordinates_angstrom=coords),
                self._assigned_cartesian_symmetry,
                minimum_deviation_angstrom=0.0,
            ).coordinates_angstrom
        return coords

    def coordinate_directions(self, coordinates_angstrom: np.ndarray | None = None) -> np.ndarray:
        if self.coordinate_model.kind != "sonic":
            return np.asarray(self.coordinate_model.directions_angstrom, dtype=float)
        definition = self._sonic_definition
        if definition is None:
            return np.asarray(self.coordinate_model.directions_angstrom, dtype=float)
        coords = (
            self.reference_coordinates
            if coordinates_angstrom is None
            else np.asarray(coordinates_angstrom, dtype=float)
        )
        cartesian_from_q = self._sonic_cartesian_from_q(coords)
        directions = cartesian_from_q.T
        return np.asarray(directions, dtype=float)

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
        from matrix_smith import build_gic_b_matrix

        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
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
        from matrix_link import direct_fragment_rigid_tangent

        fragment_tangent = direct_fragment_rigid_tangent(
            definition,
            coordinates_angstrom,
            b_matrix,
            fixed_atom_indices=self.settings.fixed_atoms,
        )
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
        for full_index in fragment_tangent.handled_indices:
            if full_index in local_index:
                cartesian_from_all_q[:, local_index[full_index]] = (
                    fragment_tangent.cartesian_from_q[:, full_index]
                )
        if not self.settings.freeze_inactive_sonic:
            selected = np.asarray(cartesian_from_all_q, dtype=float)
        else:
            selected = np.asarray(
                cartesian_from_all_q[:, self._sonic_coordinate_indices], dtype=float
            )
        transform = self.coordinate_model.sonic_from_coordinates if apply_variable_projection else None
        return selected if transform is None else selected @ transform

    def _project_rigid_reference_groups(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        groups = self.settings.rigid_reference_groups
        coords = np.asarray(coordinates_angstrom, dtype=float).copy()
        if not groups:
            return coords
        from matrix_core import kabsch_align

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

    def _absolute_sonic_values(self, q: np.ndarray) -> np.ndarray:
        if self._sonic_full_reference_values is None or self._sonic_reference_values is None:
            raise RuntimeError("SONIC reference values are unavailable")
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
        from matrix_smith import build_gic_b_matrix, evaluate_gic_values

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
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ).rows,
            dtype=float,
        )
        if self._sonic_rotation_atlas is not None:
            values, transformed = self._sonic_rotation_atlas.transform(values, b_matrix)
            assert transformed is not None
            b_matrix = transformed
        return values, b_matrix

    def _evaluate_active_sonic(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        definition = self._sonic_definition
        if definition is None:
            raise RuntimeError("SONIC definition is unavailable")
        from matrix_smith import build_gic_b_matrix, evaluate_gic_values

        indices = list(self._sonic_coordinate_indices)
        rotation_reference = (
            None
            if self._sonic_rotation_atlas is None
            else self._sonic_rotation_atlas.reference_coordinates
        )
        all_values = evaluate_gic_values(
            definition,
            coordinates_angstrom=coordinates_angstrom,
            rotation_reference_coordinates=rotation_reference,
        )
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
                rotation_reference_coordinates=rotation_reference,
            ).rows,
            dtype=float,
        )
        if self._sonic_rotation_atlas is not None:
            all_values, transformed = self._sonic_rotation_atlas.transform(all_values, b_matrix)
            assert transformed is not None
            b_matrix = transformed
        return all_values[indices], b_matrix[indices, :]

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
        # Rebase automatically only before the exponential map approaches its
        # pi singularity.  Tools that need shorter local charts can use the
        # public FragmentRotationAtlas directly with their own policy.
        if atlas.max_local_norm(local_values) < 2.5:
            return False
        atlas.rebase(local_values, coordinates_angstrom)
        return True

    def evaluate(
        self,
        q: Sequence[float] | np.ndarray,
        *,
        tag: str,
        use_cache: bool = True,
        persist_cache: bool = True,
        requested_properties: Sequence[str] = (),
    ) -> OptimizerEvaluation:
        vector = np.asarray(q, dtype=float).reshape(-1)
        if use_cache:
            with self._lock:
                cached = self.cache.lookup(vector)
            if cached is not None:
                return cached
        coords = self.coordinates_from_q(vector)
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
            self._evaluate_builtin_backend(
                point,
                tag=tag,
                requested_properties=requested_properties,
            )
            if self.backend is not None
            else self._evaluate_external_command(
                point,
                tag=tag,
                requested_properties=requested_properties,
            )
        )
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
        with self._lock:
            self.energy_evaluations += int(
                result.execution.get("energy_evaluations", result.energy_hartree is not None)
            )
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
            self.fd_displacements += int(
                tag.startswith("fd-plus-") or tag.startswith("fd-minus-")
            )
        realized_q = (
            self.actual_q(coords) if self.coordinate_model.kind == "sonic" else vector.copy()
        )
        evaluation = OptimizerEvaluation(
            q=realized_q,
            coordinates_angstrom=coords,
            result=result,
            cache_hit=False,
        )
        if persist_cache:
            with self._lock:
                self.cache.add(evaluation)
        return evaluation

    def supports_architect_batch(self) -> bool:
        return bool(
            self.backend is not None
            and _normalized_backend_name(self.backend.name) == "architect"
            and self.backend.force_field is not None
            and not self.settings.include_cv_exponential_field
        )

    def evaluate_architect_batch(
        self,
        q_values: Sequence[Sequence[float] | np.ndarray],
        *,
        tags: Sequence[str],
        requested_properties: Sequence[str],
        persist_cache: bool = False,
    ) -> tuple[OptimizerEvaluation, ...]:
        """Evaluate one homogeneous LINK/SENTINEL block through ZION."""

        if not self.supports_architect_batch():
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
            raise ValueError("ARCHITECT batch q/tag counts differ")
        if not vectors:
            return ()
        coordinates = tuple(self.coordinates_from_q(vector) for vector in vectors)
        backend = self.backend
        assert backend is not None and backend.force_field is not None
        from matrix_architect import evaluate_force_field_batch, load_force_field

        field = load_force_field(backend.force_field)
        if field.atoms != self.atoms:
            raise ValueError("LINK geometry atoms differ from the ARCHITECT force field")
        properties = tuple(requested_properties or backend.properties or ("energy", "gradient"))
        results = evaluate_force_field_batch(
            field,
            coordinates,
            properties=properties,
            device=backend.device,
        )
        with self._lock:
            first_index = self.qm_evaluations
            self.qm_evaluations += len(results)
        evaluations: list[OptimizerEvaluation] = []
        for offset, (vector, coords, result) in enumerate(
            zip(vectors, coordinates, results, strict=True)
        ):
            point_index = first_index + offset
            point_result = PointEvaluationResult(
                point_index=point_index,
                displacement=0.0,
                energy_hartree=result.energy_hartree,
                gradient_hartree_per_bohr=result.gradient_hartree_per_bohr,
                hessian_hartree_per_bohr2=result.hessian_hartree_per_bohr2,
                backend_coordinates_angstrom=coords,
                source=f"ARCHITECT {backend.force_field}",
                execution=dict(result.execution),
            )
            result_path = self.run_dir / "points" / f"eval_{point_index:05d}.json"
            write_point_result(result_path, point_result)
            with self._lock:
                self.energy_evaluations += int(point_result.energy_hartree is not None)
                self.gradient_evaluations += int(
                    point_result.gradient_hartree_per_bohr is not None
                )
                self.hessian_evaluations += int(
                    point_result.hessian_hartree_per_bohr2 is not None
                )
            realized_q = self.actual_q(coords) if self.coordinate_model.kind == "sonic" else vector
            evaluation = OptimizerEvaluation(
                q=realized_q,
                coordinates_angstrom=coords,
                result=point_result,
                cache_hit=False,
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

    def _evaluate_builtin_backend(
        self,
        point: ScanPoint,
        *,
        tag: str,
        requested_properties: Sequence[str] = (),
    ) -> PointEvaluationResult:
        point_run_dir = self.run_dir / "backend_points" / f"eval_{point.index:05d}"
        backend = self.backend
        if (
            backend is not None
            and _normalized_backend_name(backend.name) == "architect"
            and requested_properties
        ):
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
    metric_diagonal: Sequence[float] | None = None,
    pes_exploration: bool = False,
    retained_group: str = "C1",
) -> OptimizerCoordinateModel:
    geometry = read_xyzin_geometry(Path(xyzin_path))
    ncart = int(np.asarray(geometry.coordinates_angstrom).size)
    coordinate_kind = str(kind).replace("-", "_")
    if coordinate_kind == "cartesian":
        directions = np.eye(ncart, dtype=float)
        labels = tuple(f"X{index + 1}" for index in range(ncart))
    elif coordinate_kind == "sonic":
        from matrix_smith import (
            build_gic_b_matrix,
            build_pes_exploration_gic_definition_from_xyzin,
            read_gic_definition_from_xyzin,
        )

        definition = (
            build_pes_exploration_gic_definition_from_xyzin(
                Path(xyzin_path), retained_group=retained_group
            )
            if pes_exploration
            else read_gic_definition_from_xyzin(Path(xyzin_path))
        )
        if not coordinates:
            if pes_exploration:
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
    else:
        raise ValueError(f"unsupported optimizer coordinate model: {kind}")
    metric = (
        np.ones(directions.shape[0], dtype=float)
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
    )


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
) -> OptimizerResult:
    settings = settings or OptimizerSettings()
    model = coordinate_model or coordinate_model_from_xyzin(
        xyzin_path,
        kind=coordinate_kind,
        coordinates=coordinates,
    )
    link_numerical_et = bool(
        backend is not None
        and _normalized_backend_name(backend.name) == "et"
        and str(backend.gradient_mode).strip().casefold() in {"numerical", "link-numerical"}
    )
    if link_numerical_et:
        settings = replace(
            settings,
            adaptive_fd_mode=True,
            fd_totally_symmetric_only=True,
            prefer_analytic_gradient=False,
            fd_parallel_workers=max(settings.fd_parallel_workers, int(backend.processors)),
        )
    service = GeometryEvaluationService(
        xyzin_path=xyzin_path,
        run_dir=run_dir,
        coordinate_model=model,
        engine_command=engine_command,
        backend=backend,
        timeout=timeout,
        settings=settings,
    )
    if settings.fd_totally_symmetric_only:
        service.assert_totally_symmetric_active_sonics()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / "optimizer_trace.jsonl"
    trajectory_path = root / "optimizer_trajectory.xyz"
    summary_path = root / "optimizer_summary.json"
    final_hessian_path = root / "optimizer_hessian.json"
    initial_geometry = read_xyzin_geometry(Path(xyzin_path))
    q = np.zeros(len(model.labels), dtype=float)
    hessian = _initial_optimizer_hessian(
        model,
        settings,
        initial_hessian=initial_hessian,
        atoms=tuple(initial_geometry.atoms),
        coordinates_angstrom=initial_geometry.coordinates_angstrom,
        xyzin_path=xyzin_path,
    )
    hessian_source = (
        initial_hessian_source
        if initial_hessian is not None
        else (
            "fischer-almloef-primitive-linear-transform"
            if settings.initial_hessian_model == "almloef"
            else (
                "gaussian-berny-primitive-linear-transform"
                if settings.initial_hessian_model == "berny"
                else _default_hessian_source(xyzin_path)
            )
        )
    )
    local_groups = _local_coordinate_groups(model, hessian, settings)
    hessian_mask = _hessian_sparsity_mask(model, hessian, settings)
    trust_radius = float(settings.trust_radius)
    current_damping = 0.0
    iterations: list[OptimizerIteration] = []
    trace_path.write_text("", encoding="utf-8")
    trajectory_path.write_text("", encoding="utf-8")

    current = service.evaluate(q, tag="initial")
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
    )
    converged = False
    status = "max_steps"
    previous_energy = current.energy_hartree
    last_energy_change = 0.0
    last_step = np.zeros_like(q)
    last_cartesian_step_bohr = np.zeros(current.coordinates_angstrom.size, dtype=float)
    selective_disabled = False
    selective_rejection_count = 0
    optimization_history: list[tuple[OptimizerEvaluation, np.ndarray]] = [
        (current, gradient.copy())
    ]
    bad_model_ratio_count = 0
    micro_masks = service.coordinate_phase_masks("inter-intra-micro")
    use_micro_schedule = settings.coordinate_schedule == "inter-intra-micro" or (
        settings.coordinate_schedule == "auto" and len(micro_masks) == 3
    )
    phase_masks = service.coordinate_phase_masks(
        "joint" if use_micro_schedule else settings.coordinate_schedule
    )
    phase_index = 0
    phase_steps = 0

    for iteration in range(1, settings.max_steps + 1):
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
            converged = True
            status = "converged_gaussian"
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
                damping=current_damping,
                force_rfo=bad_model_ratio_count >= settings.hessian_bad_ratio_limit,
            )
        diis_step, diis_status = (None, "disabled")
        if settings.enable_gdiis:
            diis_step, diis_status = _safeguarded_gdiis_step(
                _transported_optimizer_history(service, optimization_history),
                q,
                gradient,
                hessian,
                settings,
            )
        if diis_step is not None and not use_micro_schedule:
            diis_step = np.where(phase_mask, diis_step, 0.0)
            try:
                diis_coords = service.coordinates_from_q(q + diis_step)
                diis_rms = _cartesian_rms_displacement_angstrom(
                    current.coordinates_angstrom, diis_coords
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                diis_rms = math.inf
            if np.isfinite(diis_rms) and diis_rms <= trust_radius:
                proposal = StepProposal(
                    step=diis_step,
                    policy=(
                        f"phase_{phase_name}:safeguarded_gdiis:{diis_status}"
                        if len(phase_masks) > 1
                        else f"safeguarded_gdiis:{diis_status}"
                    ),
                    hessian_min_eigenvalue=proposal.hessian_min_eigenvalue,
                    hessian_condition=proposal.hessian_condition,
                    damping_shift=proposal.damping_shift,
                )
        current_damping = proposal.damping_shift
        if len(phase_masks) > 1 and not proposal.policy.startswith("phase_"):
            proposal = StepProposal(
                step=proposal.step,
                policy=f"phase_{phase_name}:{proposal.policy}",
                hessian_min_eigenvalue=proposal.hessian_min_eigenvalue,
                hessian_condition=proposal.hessian_condition,
                damping_shift=proposal.damping_shift,
            )
        step = proposal.step
        proposed_step_norm = float(np.linalg.norm(step))
        step_norm = proposed_step_norm
        step_inf = float(np.max(np.abs(step))) if step.size else 0.0
        try:
            proposed_coordinates = service.coordinates_from_q(q + step)
            proposed_cartesian_step_bohr = (
                aligned_cartesian_displacement(
                    current.coordinates_angstrom, proposed_coordinates
                ).reshape(-1)
                * ANGSTROM_TO_BOHR
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            proposed_cartesian_step_bohr = np.full(current.coordinates_angstrom.size, math.inf)
        if (
            float(np.max(np.abs(proposed_cartesian_step_bohr)))
            <= settings.max_displacement_tolerance
            and _rms(proposed_cartesian_step_bohr) <= settings.rms_displacement_tolerance
        ):
            convergence = _gaussian_like_convergence(
                settings, 0.0, convergence_gradient, proposed_cartesian_step_bohr
            )
            if phase_index == len(phase_masks) - 1 and all(convergence.values()):
                converged = True
                status = "converged_gaussian"
                last_energy_change = 0.0
                last_step = step
                last_cartesian_step_bohr = proposed_cartesian_step_bohr
                break
        trial, step, predicted, actual, rho, accepted, message, rejected_trials = (
            _evaluate_step_trials(
                service,
                current,
                q,
                gradient,
                hessian,
                step,
                settings,
                iteration=iteration,
            )
        )
        step_norm = float(np.linalg.norm(step))
        step_inf = float(np.max(np.abs(step))) if step.size else 0.0
        if accepted:
            previous_gradient_inf = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            previous_gradient_policy = str(fd_info["gradient_policy"])
            old_q = q.copy()
            old_gradient = gradient.copy()
            old_energy = current.energy_hartree
            old_coordinates = current.coordinates_angstrom.copy()
            q = np.asarray(trial.q, dtype=float).copy()
            current = trial
            last_energy_change = abs(old_energy - current.energy_hartree)
            last_step = q - old_q
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
            )
            new_gradient_inf = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            if previous_gradient_policy == "coordinate_energy_fd_selective":
                growth_limit = max(
                    settings.max_force_tolerance,
                    settings.selective_fallback_gradient_growth * previous_gradient_inf,
                )
                if new_gradient_inf > growth_limit:
                    selective_disabled = True
            hessian, update_status = _update_hessian(
                hessian,
                q - old_q,
                gradient - old_gradient,
                settings,
                metric_diagonal=model.metric_diagonal,
                analytic_gradient_update=(
                    previous_gradient_policy == "analytic_cartesian_projected_full"
                    and str(fd_info["gradient_policy"]) == "analytic_cartesian_projected_full"
                ),
                prefer_coupled_update=use_micro_schedule,
            )
            if settings.sparse_hessian_updates:
                hessian = _project_hessian_sparsity(hessian, hessian_mask)
            optimization_history.append((current, gradient.copy()))
            del optimization_history[: -settings.gdiis_history]
            if not np.isfinite(rho) or rho < 0.10 or rho > 2.5:
                bad_model_ratio_count += 1
            else:
                bad_model_ratio_count = 0
            if bad_model_ratio_count > settings.hessian_bad_ratio_limit:
                hessian = _initial_optimizer_hessian(
                    model,
                    settings,
                    initial_hessian=None,
                    atoms=tuple(initial_geometry.atoms),
                    coordinates_angstrom=current.coordinates_angstrom,
                    xyzin_path=xyzin_path,
                )
                update_status += "_untrusted_model_rebuilt"
                current_damping = max(current_damping, settings.min_hessian_eigenvalue)
                bad_model_ratio_count = 0
            line_search_scale = (
                float(np.linalg.norm(step)) / proposed_step_norm
                if proposed_step_norm > 0.0
                else 0.0
            )
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
                converged = True
                status = "converged_gaussian"
                message = "accepted_converged"
            elif convergence["energy"]:
                message = "accepted_energy_plateau"
            selective_rejection_count = 0
        else:
            update_status = "skipped_rejected_step"
            bad_model_ratio_count += 1
            if bad_model_ratio_count > settings.hessian_bad_ratio_limit:
                hessian = _initial_optimizer_hessian(
                    model,
                    settings,
                    initial_hessian=None,
                    atoms=tuple(initial_geometry.atoms),
                    coordinates_angstrom=current.coordinates_angstrom,
                    xyzin_path=xyzin_path,
                )
                update_status += "_untrusted_model_rebuilt"
                optimization_history = optimization_history[-1:]
                bad_model_ratio_count = 0
            service.refresh_coordinate_projector(q, current.coordinates_angstrom)
            current_damping, trust_radius = _rejected_optimizer_trust_update(
                current_damping,
                trust_radius,
                _cartesian_rms_displacement_angstrom(
                    current.coordinates_angstrom, trial.coordinates_angstrom
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
            message=message,
        )
        iterations.append(record)
        phase_steps += 1
        _append_trace(trace_path, record, model)
        if converged:
            break
        if current.energy_hartree > previous_energy and trust_radius <= settings.min_trust_radius:
            status = "stalled"
            break
        previous_energy = current.energy_hartree

    final_gradient, _final_fd_info = _gradient_in_coordinate_space(
        service,
        current,
        q,
        hessian,
        settings,
        force_explicit=current.gradient_hartree_per_bohr is None,
        selective_disabled=selective_disabled,
        force_two_sided=settings.adaptive_fd_mode,
    )
    final_convergence = _gaussian_like_convergence(
        settings,
        last_energy_change,
        _convergence_gradient(current, final_gradient, settings, service),
        last_cartesian_step_bohr,
    )
    if converged and not all(final_convergence.values()):
        converged = False
        status = "not_converged_after_final_refresh"
    write_optimizer_hessian(
        final_hessian_path,
        hessian,
        model,
        source=f"final-bfgs; initial={hessian_source}",
        q=q,
    )
    result = OptimizerResult(
        converged=converged,
        status=status if not converged else status,
        settings=settings,
        atoms=service.atoms,
        initial_coordinates_angstrom=service.reference_coordinates,
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
        initial_hessian_source=hessian_source,
    )
    summary_path.write_text(
        json.dumps(optimizer_result_to_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def optimizer_result_to_json(result: OptimizerResult) -> dict[str, object]:
    return {
        "schema": OPTIMIZER_SUMMARY_SCHEMA,
        "converged": result.converged,
        "status": result.status,
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
        "qm_evaluations": result.qm_evaluations,
        "energy_evaluations": result.energy_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "hessian_evaluations": result.hessian_evaluations,
        "fd_displacements": result.fd_displacements,
        "cache_hits": result.cache_hits,
        "avoided_evaluations": result.avoided_evaluations,
        "cache": str(result.cache_path),
        "initial_hessian_source": result.initial_hessian_source,
        "final_hessian": str(result.final_hessian_path),
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
                "message": item.message,
            }
            for item in result.iterations
        ],
        "trajectory": str(result.trajectory_path),
        "trace": str(result.trace_path),
    }


def _optimizer_diagnostics(iterations: Sequence[OptimizerIteration]) -> dict[str, object]:
    records = tuple(iterations)
    accepted = sum(1 for item in records if item.status.startswith("accepted"))
    line_search = sum(1 for item in records if "line_search" in item.status)
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
        "line_search_macro_steps": line_search,
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
    cart = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    ncart = model.directions_angstrom.shape[1]
    if cart.shape != (ncart, ncart):
        raise ValueError(f"Cartesian Hessian shape must be {(ncart, ncart)}, got {cart.shape}")
    directions_bohr = np.asarray(model.directions_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    if cartesian_gradient_hartree_per_bohr is not None:
        if model.kind != "sonic" or model.sonic_definition is None:
            raise ValueError("B-prime Hessian transformation requires a frozen SONIC model")
        from matrix_architect import curvilinear_internal_hessian_from_cartesian

        definition = model.sonic_definition
        if coordinates_angstrom is not None:
            directions_bohr = _optimizer_directions_at_geometry(
                model,
                definition,
                np.asarray(coordinates_angstrom, dtype=float),
            )
        result = curvilinear_internal_hessian_from_cartesian(
            definition,
            cart,
            cartesian_gradient_hartree_per_bohr,
            coordinates_angstrom=coordinates_angstrom,
            cartesian_from_internal_bohr=directions_bohr.T,
            parallel_workers=b_prime_parallel_workers,
        )
        return _validate_optimizer_hessian(result.hessian_internal, len(model.labels))
    projected = directions_bohr @ (0.5 * (cart + cart.T)) @ directions_bohr.T
    return _validate_optimizer_hessian(projected, len(model.labels))


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
    from matrix_link import direct_fragment_rigid_tangent

    cartesian_from_all = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8)
    fragment_tangent = direct_fragment_rigid_tangent(
        definition,
        np.asarray(coordinates_angstrom, dtype=float),
        b_matrix,
    )
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_all[:, handled_index] = fragment_tangent.cartesian_from_q[
            :, handled_index
        ]
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
) -> np.ndarray:
    data = hessian_input_from_engine_hessian(engine, path, grd=grd, output=output)
    hessian = np.asarray(data.cartesian_hessian, dtype=float)
    gradient = (
        None
        if cartesian_gradient_hartree_per_bohr is None
        else np.asarray(cartesian_gradient_hartree_per_bohr, dtype=float).reshape(-1)
    )
    coordinates_angstrom = np.asarray(
        data.cartesian_coordinates_bohr, dtype=float
    ) / ANGSTROM_TO_BOHR
    if model.kind == "sonic" and model.sonic_definition is not None:
        reference = np.asarray(
            model.sonic_definition.reference_coordinates_angstrom, dtype=float
        )
        if coordinates_angstrom.shape != reference.shape:
            raise ValueError("Imported Hessian geometry does not match the SONIC atom count")
        rotation = kabsch_rotation(coordinates_angstrom, reference)
        rotation_gradient = (
            np.zeros(coordinates_angstrom.size, dtype=float)
            if gradient is None
            else gradient
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
        hessian_input_from_gaussian_log,
        run_gaussian_job,
        write_gaussian_point_input,
    )

    geometry = read_xyzin_geometry(Path(xyzin_path))
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_path = write_gaussian_point_input(
        root / "gauin.gjf",
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        route=route,
        title="MATRIX optimizer Hessian seed",
        charge=charge,
        multiplicity=multiplicity,
        ensure_force=False,
    )
    run = run_gaussian_job(
        root,
        executable=executable,
        input_path=input_path,
        timeout=timeout,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    data = hessian_input_from_gaussian_log(run.log_path)
    return optimizer_hessian_from_cartesian(data.cartesian_hessian, model), run.log_path


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
) -> tuple[np.ndarray, Path]:
    """Build a low-level Hessian seed, then project it into optimizer coordinates."""

    name = _normalized_hessian_engine(engine)
    root = Path(run_dir)
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
        return optimizer_hessian_from_engine_hessian(name, target, model), target
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


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))


def _initial_optimizer_hessian(
    model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
    *,
    initial_hessian: np.ndarray | None,
    atoms: Sequence[str] = (),
    coordinates_angstrom: np.ndarray | None = None,
    xyzin_path: Path | str | None = None,
) -> np.ndarray:
    if initial_hessian is None:
        if model.kind == "sonic" and atoms and coordinates_angstrom is not None:
            if settings.initial_hessian_model in {"auto", "berny"}:
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
                return berny
            if settings.initial_hessian_model == "almloef":
                almloef = _fischer_almloef_hessian(
                    model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
                )
                if almloef is None:
                    raise ValueError("Fischer-Almloef Hessian needs a frozen SONIC definition")
                return almloef
            berny = _gaussian_berny_pseudobond_hessian(
                model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
            )
            if berny is not None:
                return berny
            hybrid = _berny_geometric_fragment_hessian(
                model, atoms, coordinates_angstrom, xyzin_path=xyzin_path
            )
            if hybrid is not None:
                return hybrid
            chemical = _chemical_valence_hessian(
                model, atoms, coordinates_angstrom, settings, xyzin_path=xyzin_path
            )
            if chemical is not None:
                return chemical
        return np.diag(np.maximum(np.abs(model.metric_diagonal), settings.min_abs_metric_diagonal))
    matrix = _validate_optimizer_hessian(initial_hessian, len(model.labels))
    diagonal = np.diag(matrix).copy()
    small = np.abs(diagonal) < settings.min_abs_metric_diagonal
    if np.any(small):
        matrix = matrix.copy()
        for index in np.flatnonzero(small):
            sign = 1.0 if diagonal[index] >= 0.0 else -1.0
            matrix[index, index] = sign * settings.min_abs_metric_diagonal
    return 0.5 * (matrix + matrix.T)


def _default_hessian_source(xyzin_path: Path | str) -> str:
    geometry = read_xyzin_geometry(Path(xyzin_path))
    try:
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(Path(xyzin_path))
        if definition.fragment_mode == "PSEUDO_BONDS":
            return "gaussian-berny-pseudobond-linear-transform"
        if definition.fragment_mode == "SPECIAL_COORDINATES":
            return "berny-intramolecular-geometric-fragment-linear-transform"
        return "gaussian-berny-primitive-linear-transform"
    except (ImportError, OSError, ValueError):
        pass
    return "chemical-valence" if len(geometry.atoms) > 1 else "metric-diagonal"


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
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    except (ImportError, OSError, ValueError):
        return None
    coords = np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR
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
            return 0.3601 * math.exp(-1.944 * (distance(i, j) - rcov(i, j)))
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
            center, out, left, right = idx
            u = coords[out] - coords[center]
            v = coords[left] - coords[center]
            w = coords[right] - coords[center]
            normal = np.cross(v, w)
            sine = float(
                np.dot(u, normal) / max(np.linalg.norm(u) * np.linalg.norm(normal), 1.0e-15)
            )
            value = math.asin(float(np.clip(sine, -1.0, 1.0)))
            return 0.0025 + 0.0061 * (rcov(center, left) * rcov(center, right)) ** 0.80 * math.cos(
                value
            ) ** 4 * math.exp(-3.0 * (distance(center, out) - rcov(center, out)))
        if function in {"FTRANS", "FROT"}:
            return 0.025
        if function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}:
            return 0.05 * ANGSTROM_TO_BOHR**2
        raise ValueError(f"Fischer-Almloef has no primitive assignment for {primitive.function}")

    primitive_hessian = np.diag([curvature(primitive) for primitive in definition.primitives])
    primitive_index = {primitive.identifier: i for i, primitive in enumerate(definition.primitives)}
    gic_by_label = {label: gic for gic in definition.gics for label in (gic.identifier, gic.name)}
    sonic_labels = model.sonic_labels or model.labels
    sonic_transform = np.zeros((len(sonic_labels), len(definition.primitives)), dtype=float)
    for row, label in enumerate(sonic_labels):
        gic = gic_by_label.get(label)
        if gic is None:
            raise ValueError(f"SONIC coordinate {label!r} is absent from the frozen definition")
        for primitive_id, coefficient in gic.coefficients or ((gic.primitive_id, 1.0),):
            sonic_transform[row, primitive_index[primitive_id]] += float(coefficient)
    transform = _active_from_sonic_matrix(model) @ sonic_transform
    if np.linalg.matrix_rank(transform, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-SONIC Fischer-Almloef transform is rank deficient")
    hessian = transform @ primitive_hessian @ transform.T
    return 0.5 * (hessian + hessian.T)


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
    transform = _active_from_sonic_matrix(model) @ sonic_transform
    if np.linalg.matrix_rank(transform, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-SONIC Berny transform is rank deficient")
    hessian = transform @ primitive_hessian @ transform.T
    return 0.5 * (hessian + hessian.T)


def _active_from_sonic_matrix(model: OptimizerCoordinateModel) -> np.ndarray:
    """Return the congruence map from underlying SONIC to active variables."""

    transform = model.sonic_from_coordinates
    if transform is None:
        return np.eye(len(model.labels), dtype=float)
    return np.asarray(transform, dtype=float).T


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
        if function in {"FTRANS", "FROT"}:
            # geomeTRIC assigns 0.05 to each absolute fragment primitive.
            # MATRIX stores q_rel=q_a-q_b rather than (q_a-q_b)/sqrt(2).
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
    transform = _active_from_sonic_matrix(model) @ sonic_transform
    if np.linalg.matrix_rank(transform, tol=1.0e-10) != len(model.labels):
        raise ValueError("primitive-to-SONIC fragment transform is rank deficient")
    hessian = transform @ primitive_hessian @ transform.T
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
    if function == "RPCK" and primitive.family == "PSEUDO_CYCLE_TORSION":
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


def _gaussian_like_convergence(
    settings: OptimizerSettings,
    energy_change: float,
    gradient: np.ndarray,
    step: np.ndarray,
) -> dict[str, bool]:
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    disp = np.asarray(step, dtype=float).reshape(-1)
    max_force = float(np.max(np.abs(grad))) if grad.size else 0.0
    max_disp = float(np.max(np.abs(disp))) if disp.size else 0.0
    return {
        "energy": abs(float(energy_change)) <= settings.energy_tolerance,
        "max_force": max_force <= settings.max_force_tolerance,
        "rms_force": _rms(grad) <= settings.rms_force_tolerance,
        "max_displacement": max_disp <= settings.max_displacement_tolerance,
        "rms_displacement": _rms(disp) <= settings.rms_displacement_tolerance,
    }


def _convergence_gradient(
    evaluation: OptimizerEvaluation,
    internal_gradient: np.ndarray,
    settings: OptimizerSettings,
    service: GeometryEvaluationService,
) -> np.ndarray:
    """Use Cartesian forces when supplied by the electronic-structure backend."""
    if service.coordinate_model.kind == "sonic" and settings.freeze_inactive_sonic:
        cartesian = evaluation.gradient_hartree_per_bohr
        if cartesian is None:
            return np.asarray(internal_gradient, dtype=float).reshape(-1)
        directions = service.coordinate_directions(evaluation.coordinates_angstrom).T
        tangent_projector = directions @ np.linalg.pinv(directions, rcond=1.0e-8)
        return tangent_projector @ np.asarray(cartesian, dtype=float).reshape(-1)
    cartesian = evaluation.gradient_hartree_per_bohr
    if cartesian is not None:
        return np.asarray(cartesian, dtype=float).reshape(-1)
    return np.asarray(internal_gradient, dtype=float).reshape(-1)


def _cache_record_to_json(evaluation: OptimizerEvaluation) -> dict[str, object]:
    return {
        "schema": OPTIMIZER_CACHE_SCHEMA,
        "q": np.asarray(evaluation.q, dtype=float).reshape(-1).tolist(),
        "coordinates_angstrom": np.asarray(evaluation.coordinates_angstrom, dtype=float).tolist(),
        "result": point_result_to_json(evaluation.result),
    }


def _cache_record_from_json(payload: dict[str, object]) -> OptimizerEvaluation:
    result_payload = dict(payload.get("result") or {})
    gradient = result_payload.get("gradient_hartree_per_bohr")
    hessian = result_payload.get("hessian_hartree_per_bohr2")
    result = PointEvaluationResult(
        point_index=int(result_payload.get("point_index", 0)),
        displacement=float(result_payload.get("displacement", 0.0)),
        energy_hartree=result_payload.get("energy_hartree"),
        gradient_hartree_per_bohr=None if gradient is None else np.asarray(gradient, dtype=float),
        hessian_hartree_per_bohr2=None if hessian is None else np.asarray(hessian, dtype=float),
        status=str(result_payload.get("status", "completed")),
        message=str(result_payload.get("message", "")),
        source=str(result_payload.get("source", "optimizer-cache")),
        schema=str(result_payload.get("schema", "oracle.link.point_result.v1")),
    )
    return OptimizerEvaluation(
        q=np.asarray(payload["q"], dtype=float),
        coordinates_angstrom=np.asarray(payload["coordinates_angstrom"], dtype=float),
        result=result,
        cache_hit=False,
    )


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
) -> tuple[np.ndarray, dict[str, float | str]]:
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
    ncoord = q.size
    gradient = np.zeros(ncoord, dtype=float)
    steps = np.zeros(ncoord, dtype=float)
    near_minimum = bool(
        force_two_sided
        or (
            settings.adaptive_fd_mode
            and previous_gradient is not None
            and np.max(np.abs(np.asarray(previous_gradient, dtype=float)))
            <= settings.fd_central_gradient_factor * settings.max_force_tolerance
        )
    )
    use_central = bool(settings.two_sided and (near_minimum or not settings.adaptive_fd_mode))
    fd_modes = np.full(ncoord, "two-sided" if use_central else "one-sided", dtype=object)
    refresh_mask = np.ones(ncoord, dtype=bool)
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
            gradient[:] = predicted
    current_energy = current.energy_hartree
    soft = service.coordinate_soft_mask(hessian)

    def evaluate_fd(index: int) -> tuple[int, float, float]:
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
        step = min(settings.fd_max_step, max(settings.fd_min_step, step))
        plus_q = q.copy()
        plus_q[index] += step
        plus = service.evaluate(plus_q, tag=f"fd-plus-{index + 1}")
        if two_sided:
            minus_q = q.copy()
            minus_q[index] -= step
            minus = service.evaluate(minus_q, tag=f"fd-minus-{index + 1}")
            value = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
        else:
            value = (plus.energy_hartree - current_energy) / step
        return index, value, step

    indices = [int(index) for index in np.flatnonzero(refresh_mask)]
    if settings.fd_parallel_workers > 1 and len(indices) > 1:
        with ThreadPoolExecutor(max_workers=settings.fd_parallel_workers) as pool:
            fd_results = list(pool.map(evaluate_fd, indices))
    else:
        fd_results = [evaluate_fd(index) for index in indices]
    for index, value, step in fd_results:
        gradient[index] = value
        steps[index] = step
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
        "active_coordinate_fraction": float(refreshed / ncoord) if ncoord else 0.0,
        "one_sided_count": one_sided_count,
        "two_sided_count": two_sided_count,
        "parallel_workers": min(settings.fd_parallel_workers, max(refreshed, 1)),
        "surrogate_sample_count": surrogate_samples,
        "hard_coordinate_count": int(np.count_nonzero(~soft & refresh_mask)),
        "soft_coordinate_count": int(np.count_nonzero(soft & refresh_mask)),
        "near_minimum": near_minimum,
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


def _symmetry_status(settings: OptimizerSettings, model: OptimizerCoordinateModel) -> str:
    if not settings.symmetry_reduction:
        return "disabled"
    if model.kind != "sonic":
        return "disabled_non_sonic"
    return "diagnostic_only"


def _selective_fallback_status(settings: OptimizerSettings, selective_disabled: bool) -> str:
    if not settings.selective_fd_refresh:
        return "disabled"
    return "fallback_full_fd" if selective_disabled else "selective_active"


def _geometric_trust_region_step(
    service: GeometryEvaluationService,
    current: OptimizerEvaluation,
    q: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    settings: OptimizerSettings,
    *,
    damping: float = 0.0,
    force_rfo: bool = False,
) -> StepProposal:
    """Restricted-step RFO bounded by the realized Cartesian RMSD."""
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    gradient_vector = np.asarray(gradient, dtype=float).reshape(-1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        eigenvalues = np.ones(gradient_vector.size, dtype=float)
        eigenvectors = np.eye(gradient_vector.size, dtype=float)
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    maximum = max(float(np.max(np.abs(eigenvalues))), settings.min_hessian_eigenvalue)
    floor = max(settings.min_hessian_eigenvalue, maximum / settings.max_hessian_condition)
    positive = np.maximum(eigenvalues, floor)
    condition = float(np.max(positive) / np.min(positive)) if positive.size else 1.0
    effective_hessian = eigenvectors @ np.diag(positive) @ eigenvectors.T

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
        maximum_component = float(np.max(np.abs(step))) if step.size else 0.0
        if maximum_component > settings.max_coordinate_step:
            step *= settings.max_coordinate_step / maximum_component
        return step

    def cartesian_rms(step: np.ndarray) -> float:
        try:
            coordinates = service.coordinates_from_q(np.asarray(q, dtype=float) + step)
        except (FloatingPointError, RuntimeError, ValueError):
            return float("inf")
        return _cartesian_rms_displacement_angstrom(current.coordinates_angstrom, coordinates)

    shift = max(float(damping), 0.0)
    step = rfo_step(shift)
    try:
        rmsd = cartesian_rms(step)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        rmsd = math.inf
    active = not np.isfinite(rmsd) or rmsd > float(radius)
    if active:
        lower = shift
        upper = max(1.0e-8, 2.0 * max(shift, floor))
        upper_step = rfo_step(upper)
        for _ in range(80):
            try:
                upper_rmsd = cartesian_rms(upper_step)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                upper_rmsd = math.inf
            if np.isfinite(upper_rmsd) and upper_rmsd <= float(radius):
                break
            lower = upper
            upper *= 2.0
            upper_step = rfo_step(upper)
        else:
            raise RuntimeError("unable to bracket a Cartesian trust-region step")
        # Monotonic in the shifted quadratic model.  A relative 10% Cartesian
        # tolerance follows geomeTRIC and avoids unnecessary back-transforms.
        for _ in range(60):
            shift = 0.5 * (lower + upper)
            step = rfo_step(shift)
            try:
                rmsd = cartesian_rms(step)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                rmsd = math.inf
            if np.isfinite(rmsd) and 0.9 * float(radius) <= rmsd <= float(radius):
                break
            if not np.isfinite(rmsd) or rmsd > float(radius):
                lower = shift
            else:
                upper = shift
        if not np.isfinite(rmsd) or rmsd > float(radius):
            shift = upper
            step = rfo_step(shift)
    policy = "rs_rfo_cartesian_trust"
    if active:
        policy += "_restricted"
    if minimum < -settings.min_hessian_eigenvalue:
        policy += "_positive_model"
    return StepProposal(
        step=np.asarray(step, dtype=float),
        policy=policy,
        hessian_min_eigenvalue=minimum,
        hessian_condition=condition,
        damping_shift=shift,
    )


def _cartesian_rms_displacement_angstrom(
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    displacement = aligned_cartesian_displacement(before, after)
    return float(np.sqrt(np.sum(displacement * displacement) / max(displacement.shape[0], 1)))


def _accepted_optimizer_trust_update(
    damping: float,
    trust_radius: float,
    ratio: float,
    scale: float,
    cartesian_rmsd: float,
    settings: OptimizerSettings,
) -> tuple[float, float]:
    safe_ratio = float(ratio) if np.isfinite(ratio) else 0.0
    safe_rmsd = float(cartesian_rmsd) if np.isfinite(cartesian_rmsd) else float(trust_radius)
    if safe_ratio < 0.10:
        new_radius = 0.5 * min(float(trust_radius), max(safe_rmsd, settings.min_trust_radius))
        new_damping = max(float(damping), OPTIMIZER_DAMPING_MIN)
    elif safe_ratio >= 0.75 and safe_rmsd >= 0.8 * float(trust_radius):
        new_radius = math.sqrt(2.0) * float(trust_radius)
        new_damping = 0.0
    else:
        new_damping = 0.0
        new_radius = float(trust_radius)
    return new_damping, max(settings.min_trust_radius, min(settings.max_trust_radius, new_radius))


def _rejected_optimizer_trust_update(
    damping: float,
    trust_radius: float,
    cartesian_rmsd: float,
    settings: OptimizerSettings,
) -> tuple[float, float]:
    return (
        max(float(damping), OPTIMIZER_DAMPING_MIN),
        max(settings.min_trust_radius, 0.5 * float(trust_radius)),
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
) -> tuple[OptimizerEvaluation, np.ndarray, float, float, float, bool, str, int]:
    current_energy = current.energy_hartree
    candidate_step = np.asarray(step, dtype=float).reshape(-1)
    reductions = max(0, int(settings.line_search_reductions))
    factor = 1.0
    for attempt in range(reductions + 1):
        realized_request = candidate_step * factor
        candidate_q = q + realized_request
        geometry_message = _candidate_geometry_rejection(service, candidate_q, settings)
        if geometry_message:
            factor *= 0.5
            continue
        try:
            trial = service.evaluate(candidate_q, tag=f"step-{iteration}")
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
        rho = float(actual / predicted) if predicted > 0.0 else -math.inf
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
                "rejected_catastrophic_energy",
                1,
            )
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
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return f"coordinate_back_transform:{exc}"
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
    if float(grad @ step) >= 0.0 or not np.all(np.isfinite(step)):
        return _geometric_trust_region_step(service, current, q, matrix, grad, radius, settings)
    maximum_component = float(np.max(np.abs(step))) if step.size else 0.0
    if maximum_component > settings.max_coordinate_step:
        step *= settings.max_coordinate_step / maximum_component
    restricted = False
    try:
        rmsd = _cartesian_rms_displacement_angstrom(
            current.coordinates_angstrom, service.coordinates_from_q(q + step)
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        rmsd = math.inf
    if not np.isfinite(rmsd):
        return _geometric_trust_region_step(service, current, q, matrix, grad, radius, settings)
    if rmsd > radius and rmsd > 0.0:
        step *= radius / rmsd
        restricted = True
    condition = (
        maximum / max(settings.min_hessian_eigenvalue, minimum)
        if minimum > 0.0
        else settings.max_hessian_condition
    )
    policy = "inter_intra_micro_gauss_seidel"
    if restricted:
        policy += "_restricted"
    return StepProposal(
        step=step,
        policy=policy,
        hessian_min_eigenvalue=0.0 if not np.isfinite(minimum) else minimum,
        hessian_condition=float(condition),
        damping_shift=0.0,
    )


def _safeguarded_gdiis_step(
    history: Sequence[tuple[np.ndarray, float, np.ndarray]],
    current_q: np.ndarray,
    current_gradient: np.ndarray,
    hessian: np.ndarray,
    settings: OptimizerSettings,
) -> tuple[np.ndarray | None, str]:
    """Extrapolate from gradient residuals, rejecting unstable DIIS solutions.

    This is the geometry analogue of Pulay GDIIS.  The affine coefficients are
    bounded explicitly because large coefficients are a reliable symptom of a
    nearly singular history, as also guarded against in Gaussian's optimizer.
    """

    if len(history) < settings.gdiis_start:
        return None, "insufficient_history"
    records = list(history[-settings.gdiis_history :])
    gradients = np.vstack([np.asarray(item[2], dtype=float) for item in records])
    overlap = gradients @ gradients.T
    scale = max(float(np.max(np.abs(np.diag(overlap)))), 1.0e-16)
    overlap /= scale
    try:
        condition = float(np.linalg.cond(overlap))
    except np.linalg.LinAlgError:
        return None, "condition_failed"
    if not np.isfinite(condition) or condition > settings.gdiis_max_condition:
        return None, f"ill_conditioned={condition:.3g}"
    size = len(records)
    system = np.zeros((size + 1, size + 1), dtype=float)
    system[:size, :size] = overlap
    system[:size, size] = 1.0
    system[size, :size] = 1.0
    rhs = np.zeros(size + 1, dtype=float)
    rhs[size] = 1.0
    try:
        coefficients = np.linalg.solve(system, rhs)[:size]
    except np.linalg.LinAlgError:
        return None, "singular_system"
    if (
        not np.all(np.isfinite(coefficients))
        or float(np.max(np.abs(coefficients))) > settings.gdiis_max_coefficient
    ):
        return None, "coefficient_bound"
    target = sum(
        coefficient * item[0] for coefficient, item in zip(coefficients, records, strict=True)
    )
    step = np.asarray(target, dtype=float) - np.asarray(current_q, dtype=float)
    gradient = np.asarray(current_gradient, dtype=float)
    if float(gradient @ step) >= -1.0e-12:
        return None, "non_descent"
    if _predicted_reduction(gradient, hessian, step) <= 0.0:
        return None, "nonpositive_model"
    if float(np.max(np.abs(step))) > settings.max_coordinate_step:
        step *= settings.max_coordinate_step / float(np.max(np.abs(step)))
    return step, f"n={size}:condition={condition:.3g}"


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


def _update_hessian(
    hessian: np.ndarray,
    step: np.ndarray,
    y: np.ndarray,
    settings: OptimizerSettings,
    *,
    metric_diagonal: Sequence[float],
    analytic_gradient_update: bool = False,
    prefer_coupled_update: bool = False,
) -> tuple[np.ndarray, str]:
    scheme = settings.hessian_update
    if scheme == "auto":
        scheme = "bofill" if prefer_coupled_update else "bfgs"
    base = np.asarray(hessian, dtype=float)
    if scheme == "sr1":
        updated, status = _sr1_update(base, step, y)
    elif scheme == "bofill":
        updated, status = _bofill_update(base, step, y, damp=settings.bfgs_damping)
    else:
        updated, status = _bfgs_update(base, step, y, damp=settings.bfgs_damping)
    if settings.hessian_update == "auto":
        status = f"auto_{status}"
    if not _hessian_is_usable(updated, settings):
        if settings.hessian_reset_on_bad_update:
            return np.diag(
                np.maximum(np.abs(metric_diagonal), settings.min_abs_metric_diagonal)
            ), f"{status}_reset"
        return 0.5 * (
            np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T
        ), f"{status}_rejected"
    return updated, status


def _bfgs_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray, *, damp: bool
) -> tuple[np.ndarray, str]:
    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    hs = h @ s
    s_h_s = float(s @ hs)
    s_y = float(s @ yvec)
    if damp and s_h_s > 0.0 and s_y < 0.2 * s_h_s:
        theta = 0.8 * s_h_s / (s_h_s - s_y)
        yvec = theta * yvec + (1.0 - theta) * hs
        s_y = float(s @ yvec)
        status = "bfgs_damped"
    else:
        status = "bfgs"
    if s_y <= 1.0e-12 or s_h_s <= 1.0e-12:
        return h, "bfgs_skipped_curvature"
    updated = h - np.outer(hs, hs) / s_h_s + np.outer(yvec, yvec) / s_y
    return 0.5 * (updated + updated.T), status


def _bofill_update(
    hessian: np.ndarray, step: np.ndarray, y: np.ndarray, *, damp: bool
) -> tuple[np.ndarray, str]:
    """Blend SR1 curvature capture with BFGS positive-definite stabilization."""
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    bfgs, bfgs_status = _bfgs_update(h, step, y, damp=damp)
    sr1, sr1_status = _sr1_update(h, step, y)
    if sr1_status.startswith("sr1_skipped"):
        return bfgs, f"bofill_{bfgs_status}"
    if bfgs_status.startswith("bfgs_skipped"):
        return sr1, f"bofill_{sr1_status}"
    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    residual = yvec - h @ s
    numerator = float(residual @ s) ** 2
    denominator = float((residual @ residual) * (s @ s))
    phi = 0.0 if denominator <= 1.0e-24 else min(1.0, max(0.0, numerator / denominator))
    mixed = phi * sr1 + (1.0 - phi) * bfgs
    return 0.5 * (mixed + mixed.T), f"bofill_phi={phi:.3f}"


def _sr1_update(hessian: np.ndarray, step: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str]:
    s = np.asarray(step, dtype=float).reshape(-1)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    h = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    residual = yvec - h @ s
    denom = float(residual @ s)
    threshold = 1.0e-8 * float(np.linalg.norm(residual) * np.linalg.norm(s))
    if abs(denom) <= threshold:
        return h, "sr1_skipped_curvature"
    updated = h + np.outer(residual, residual) / denom
    return 0.5 * (updated + updated.T), "sr1"


def _hessian_is_usable(hessian: np.ndarray, settings: OptimizerSettings) -> bool:
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    if not np.all(np.isfinite(matrix)):
        return False
    try:
        eig = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    if eig.size == 0:
        return True
    if float(np.min(eig)) < -settings.min_hessian_eigenvalue:
        return False
    if (
        float(np.max(np.abs(eig)))
        > settings.max_hessian_condition / settings.min_hessian_eigenvalue
    ):
        return False
    return True


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


def _coordinate_model_status(
    reference: np.ndarray,
    coordinates: np.ndarray,
    model: OptimizerCoordinateModel,
    settings: OptimizerSettings,
) -> str:
    displacement = aligned_cartesian_displacement(reference, coordinates)
    max_displacement = float(np.max(np.abs(displacement))) if displacement.size else 0.0
    if model.kind == "sonic" and max_displacement > settings.coordinate_drift_warning:
        return f"frozen_sonic_drift_warning:{max_displacement:.6g}"
    return f"ok:max_drift={max_displacement:.6g}"


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
