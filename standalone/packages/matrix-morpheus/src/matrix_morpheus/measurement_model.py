from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import numpy as np

from matrix_chem.average_atomic_masses import atomic_mass
from matrix_chem.isotopes_table import get_default_isotope, get_isotope
from matrix_chem.physical_constants import Phy, get_physical_constants
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.elements import atomic_number as geometry_atomic_number
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_core import ScientificValidationError
from matrix_morpheus.numerics import objective, rank_condition
from matrix_link import (
    hybrid_internal_coordinate_step,
    nonlinear_internal_coordinate_step,
    secant_projector_update,
    should_refresh_coordinate_model,
)
from matrix_smith.models import (
    GICDefinition as LinkGICDefinition,
)
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive, eval_primitives

from .contracts import (
    IsotopologueObservation,
    QMParameterPredicate,
    SemiexperimentalFitRequest,
)
from .constraints import (
    _combined_primitive_constraint_residual,
    _fixed_primitive_targets,
    _gic_values,
    _primitives_from_fixed_pattern,
    _project_fixed_primitives,
)
from .cartesian_coordinates import CartesianCoordinateModel
from .models import (
    GICExpressionConstraint,
    GICExpressionDefinition,
    GICProjectorState,
    LineSearchResult,
    MeasurementModel,
    PrimitiveLinearConstraint,
    SecantProjectorUpdate,
    SemiexperimentalFitDiagnostics,
    SemiexperimentalIterationTrace,
    SemiexperimentalParameter,
    SemiexperimentalResidual,
    SemiexperimentalRotationalConstantComparison,
    TopologyLock,
)
from .solver import (
    _predicted_reduction,
)


from .coordinate_model import (
    _atomic_number,
    _gic_cartesian_projector,
    _gic_model,
    _incremental_column_rank,
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
