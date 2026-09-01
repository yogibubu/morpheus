"""Symmetry-reduced SONIC Hessians acquired from electronic energies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from matrix_chem import atomic_mass, read_xyzin_geometry
from matrix_chem.topology.elements import atomic_number
from matrix_gf import gf_from_acquired_force_constants_and_xyzin
from matrix_gf import cartesian_normal_modes_from_sonic_hessian
from matrix_qm import HessianInput
from matrix_smith import build_gic_b_matrix, read_gic_definition_from_xyzin

from .optimizer import (
    GeometryEvaluationService,
    OptimizerSettings,
    coordinate_model_from_xyzin,
)
from .scan import BOHR_TO_ANGSTROM, QMScanBackend


SONIC_HESSIAN_PLAN_SCHEMA = "matrix.link.sonic_hessian_plan.v1"
SONIC_HESSIAN_RESULT_SCHEMA = "matrix.link.sonic_hessian_result.v1"

_LENGTH_FAMILIES = {
    "STRETCH",
    "HBOND_DISTANCE",
    "SPECIAL_HBOND_DISTANCE",
    "FRAGMENT_CENTER_DISTANCE",
    "FRAGMENT_CENTER_ATOM_DISTANCE",
    "FRAG_CENTER_ATOM_DISTANCE",
    "CENTER_ATOM_DISTANCE",
    "FRAG_TRANSLATION",
}


@dataclass(frozen=True)
class SonicHessianCoordinate:
    index: int
    label: str
    family: str
    irrep: str
    unit: str
    step: float


@dataclass(frozen=True)
class SonicHessianPoint:
    key: str
    role: str
    displacement: tuple[float, ...]
    active_coordinates: tuple[int, ...]


@dataclass(frozen=True)
class SonicHessianPlan:
    point_group: str
    coordinates: tuple[SonicHessianCoordinate, ...]
    symmetry_blocks: tuple[tuple[str, tuple[int, ...]], ...]
    points: tuple[SonicHessianPoint, ...]
    property_source: str = "energy"
    displacement_basis: str = "sonic"
    mixed_stencil: str = "one-sided"
    schema: str = SONIC_HESSIAN_PLAN_SCHEMA

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinates)

    @property
    def mixed_coupling_count(self) -> int:
        return sum(point.role == "mixed-plus-plus" for point in self.points)

    @property
    def energy_point_count(self) -> int:
        return len(self.points)

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "coordinate_count": self.coordinate_count,
                "mixed_coupling_count": self.mixed_coupling_count,
                "energy_point_count": self.energy_point_count,
                "cross_irrep_policy": "zero by symmetry",
            }
        )
        return payload


@dataclass(frozen=True)
class SonicHessianResult:
    plan: SonicHessianPlan
    force_constants: np.ndarray
    energies_hartree: Mapping[str, float]
    frequencies_cm: np.ndarray
    g_matrix: np.ndarray
    state_audit: Mapping[str, Mapping[str, object]]
    result_path: Path


def sonic_coordinate_unit(family: str) -> str:
    """Return the finite-difference unit implied by a frozen SONIC family."""

    return "angstrom" if str(family).strip().upper() in _LENGTH_FAMILIES else "radian"


def plan_sonic_hessian(
    xyzin_path: Path | str,
    *,
    stretch_step_angstrom: float = 0.01,
    angular_step_radian: float = 0.01,
    property_source: str = "energy",
    mixed_stencil: str = "one-sided",
) -> SonicHessianPlan:
    """Build the minimal symmetry-block SONIC energy grid for a Hessian.

    Every diagonal uses the default central three-point ``(-h_i,0,+h_i)``
    derivative.  The default mixed derivative adds one ``(+h_i,+h_j)`` point
    only when both coordinates have the same irrep; cross-irrep force
    constants are exactly zero and no forbidden displacement is generated.
    """

    if stretch_step_angstrom <= 0.0 or angular_step_radian <= 0.0:
        raise ValueError("SONIC Hessian displacement steps must be positive")
    source = str(property_source).strip().casefold().replace("_", "-")
    if source not in {"energy", "cartesian-gradient"}:
        raise ValueError("property_source must be 'energy' or 'cartesian-gradient'")
    stencil = str(mixed_stencil).strip().casefold().replace("_", "-")
    if stencil not in {"one-sided"}:
        raise ValueError("mixed_stencil must be 'one-sided'")

    definition = read_gic_definition_from_xyzin(Path(xyzin_path))
    if definition.rank != len(definition.gics):
        raise ValueError(
            "direct SONIC Hessian acquisition requires a frozen nonredundant GIC contract"
        )
    coordinates = tuple(
        SonicHessianCoordinate(
            index=index,
            label=str(gic.name or gic.identifier),
            family=str(gic.family).strip().upper(),
            irrep=str(gic.irrep).strip(),
            unit=sonic_coordinate_unit(gic.family),
            step=(
                float(stretch_step_angstrom)
                if sonic_coordinate_unit(gic.family) == "angstrom"
                else float(angular_step_radian)
            ),
        )
        for index, gic in enumerate(definition.gics)
    )
    if not coordinates:
        raise ValueError("the frozen SONIC contract contains no coordinates")

    block_map: dict[str, list[int]] = {}
    for coordinate in coordinates:
        block_map.setdefault(coordinate.irrep, []).append(coordinate.index)
    symmetry_blocks = tuple(
        (irrep, tuple(indices)) for irrep, indices in block_map.items()
    )

    ncoord = len(coordinates)
    zero = np.zeros(ncoord, dtype=float)
    points = [
        SonicHessianPoint(
            key="center",
            role="center",
            displacement=tuple(zero),
            active_coordinates=(),
        )
    ]
    for coordinate in coordinates:
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            displacement = zero.copy()
            displacement[coordinate.index] = sign * coordinate.step
            points.append(
                SonicHessianPoint(
                    key=f"q{coordinate.index + 1:03d}-{suffix}",
                    role=f"diagonal-{suffix}",
                    displacement=tuple(displacement),
                    active_coordinates=(coordinate.index,),
                )
            )
    if source == "energy":
        for _irrep, indices in symmetry_blocks:
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    displacement = zero.copy()
                    displacement[left] = coordinates[left].step
                    displacement[right] = coordinates[right].step
                    points.append(
                        SonicHessianPoint(
                            key=f"q{left + 1:03d}-q{right + 1:03d}-plus-plus",
                            role="mixed-plus-plus",
                            displacement=tuple(displacement),
                            active_coordinates=(left, right),
                        )
                    )
    return SonicHessianPlan(
        point_group=str(definition.point_group),
        coordinates=coordinates,
        symmetry_blocks=symmetry_blocks,
        points=tuple(points),
        property_source=source,
        mixed_stencil=stencil,
    )


def sonic_hessian_from_energies(
    plan: SonicHessianPlan,
    energies_hartree: Mapping[str, float],
) -> np.ndarray:
    """Reconstruct a symmetry-block Hessian from a completed plan."""

    if plan.property_source != "energy":
        raise ValueError("the SONIC Hessian plan does not acquire energies")

    missing = tuple(point.key for point in plan.points if point.key not in energies_hartree)
    if missing:
        raise ValueError("missing SONIC Hessian energies: " + ", ".join(missing))
    center = float(energies_hartree["center"])
    hessian = np.zeros((plan.coordinate_count, plan.coordinate_count), dtype=float)
    for coordinate in plan.coordinates:
        minus = float(energies_hartree[f"q{coordinate.index + 1:03d}-minus"])
        plus = float(energies_hartree[f"q{coordinate.index + 1:03d}-plus"])
        hessian[coordinate.index, coordinate.index] = (
            plus - 2.0 * center + minus
        ) / coordinate.step**2
    for point in plan.points:
        if point.role != "mixed-plus-plus":
            continue
        left, right = point.active_coordinates
        left_energy = float(energies_hartree[f"q{left + 1:03d}-plus"])
        right_energy = float(energies_hartree[f"q{right + 1:03d}-plus"])
        value = (
            float(energies_hartree[point.key]) - left_energy - right_energy + center
        ) / (plan.coordinates[left].step * plan.coordinates[right].step)
        hessian[left, right] = value
        hessian[right, left] = value
    return hessian


def sonic_hessian_from_cartesian_gradients(
    plan: SonicHessianPlan,
    definition,
    point_coordinates_angstrom: Mapping[str, np.ndarray],
    gradients_hartree_per_bohr: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Differentiate backend Cartesian gradients in the displaced SONIC basis."""

    if plan.property_source != "cartesian-gradient":
        raise ValueError("the SONIC Hessian plan does not acquire Cartesian gradients")
    missing = tuple(
        point.key
        for point in plan.points
        if point.key not in point_coordinates_angstrom
        or point.key not in gradients_hartree_per_bohr
    )
    if missing:
        raise ValueError("missing SONIC Hessian gradients: " + ", ".join(missing))
    internal_gradients: dict[str, np.ndarray] = {}
    for point in plan.points:
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=np.asarray(point_coordinates_angstrom[point.key], dtype=float),
            ).rows,
            dtype=float,
        )
        cartesian = np.asarray(
            gradients_hartree_per_bohr[point.key], dtype=float
        ).reshape(-1)
        if cartesian.size != b_matrix.shape[1]:
            raise ValueError(f"Cartesian gradient size is invalid at {point.key}")
        gradient_per_angstrom = cartesian / BOHR_TO_ANGSTROM
        internal, *_ = np.linalg.lstsq(
            b_matrix.T,
            gradient_per_angstrom,
            rcond=1.0e-10,
        )
        internal_gradients[point.key] = np.asarray(internal, dtype=float)
    hessian = np.zeros((plan.coordinate_count, plan.coordinate_count), dtype=float)
    for coordinate in plan.coordinates:
        minus = internal_gradients[f"q{coordinate.index + 1:03d}-minus"]
        plus = internal_gradients[f"q{coordinate.index + 1:03d}-plus"]
        hessian[:, coordinate.index] = (plus - minus) / (2.0 * coordinate.step)
    hessian = 0.5 * (hessian + hessian.T)
    for left in range(plan.coordinate_count):
        for right in range(plan.coordinate_count):
            if plan.coordinates[left].irrep != plan.coordinates[right].irrep:
                hessian[left, right] = 0.0
    return hessian


def acquire_sonic_hessian(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    backend: QMScanBackend,
    workers: int = 1,
    stretch_step_angstrom: float = 0.01,
    angular_step_radian: float = 0.01,
    property_source: str = "energy",
    mixed_stencil: str = "one-sided",
    timeout: float | None = None,
) -> SonicHessianResult:
    """Execute a planned SONIC Hessian and directly perform the GF analysis."""

    if workers <= 0:
        raise ValueError("SONIC Hessian workers must be positive")
    source = Path(xyzin_path).expanduser().resolve()
    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    plan = plan_sonic_hessian(
        source,
        stretch_step_angstrom=stretch_step_angstrom,
        angular_step_radian=angular_step_radian,
        property_source=property_source,
        mixed_stencil=mixed_stencil,
    )
    (target / "sonic-hessian-plan.json").write_text(
        json.dumps(plan.to_json(), indent=2) + "\n", encoding="utf-8"
    )

    labels = tuple(coordinate.label for coordinate in plan.coordinates)
    model = coordinate_model_from_xyzin(source, kind="sonic", coordinates=labels)
    geometry = read_xyzin_geometry(source)
    service = GeometryEvaluationService(
        xyzin_path=source,
        run_dir=target / "evaluation-service",
        coordinate_model=model,
        backend=backend,
        timeout=timeout,
        settings=OptimizerSettings(
            # Hessian rows span every irrep.  Full reference-symmetry
            # projection is valid only for totally symmetric displacements;
            # non-total rows must be allowed to retain their proper subgroup.
            freeze_inactive_sonic=False,
            symmetrize_analytic_gradients=False,
            resume=True,
            coordinate_parallel_workers=max(1, int(workers)),
        ),
    )
    zero = np.zeros(plan.coordinate_count, dtype=float)
    service.initialize_coordinate_projector(zero, geometry.coordinates_angstrom)

    requested_properties = (
        ("energy", "gradient")
        if plan.property_source == "cartesian-gradient"
        else ("energy",)
    )

    def evaluate(point: SonicHessianPoint):
        return point, service.evaluate(
            np.asarray(point.displacement, dtype=float),
            tag=f"sonic-hessian-{point.key}",
            requested_properties=requested_properties,
        )

    center_point = plan.points[0]
    _point, center_evaluation = evaluate(center_point)
    service.accept_electronic_state(center_evaluation)
    evaluated = [(center_point, center_evaluation)]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        evaluated.extend(executor.map(evaluate, plan.points[1:]))

    energies = {
        point.key: float(evaluation.result.energy_hartree)
        for point, evaluation in evaluated
    }
    state_audit = {
        point.key: {
            key: value
            for key, value in evaluation.result.execution.items()
            if key.startswith("state_") or key in {"selected_electronic_state"}
        }
        for point, evaluation in evaluated
    }
    point_coordinates = {
        point.key: np.asarray(evaluation.coordinates_angstrom, dtype=float)
        for point, evaluation in evaluated
    }
    if plan.property_source == "cartesian-gradient":
        gradients = {}
        for point, evaluation in evaluated:
            gradient = evaluation.result.gradient_hartree_per_bohr
            if gradient is None:
                raise RuntimeError(
                    f"backend returned no Cartesian gradient at {point.key}"
                )
            gradients[point.key] = np.asarray(gradient, dtype=float)
        definition = read_gic_definition_from_xyzin(source)
        force_constants = sonic_hessian_from_cartesian_gradients(
            plan,
            definition,
            point_coordinates,
            gradients,
        )
    else:
        force_constants = sonic_hessian_from_energies(plan, energies)

    numbers = np.asarray([atomic_number(atom) or 0 for atom in geometry.atoms], dtype=int)
    masses = np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)
    coordinates_bohr = np.asarray(geometry.coordinates_angstrom, dtype=float) / BOHR_TO_ANGSTROM
    ncart = coordinates_bohr.size
    geometry_input = HessianInput(
        atomic_numbers=numbers,
        cartesian_coordinates_bohr=coordinates_bohr,
        masses_amu=masses,
        cartesian_hessian=np.zeros((ncart, ncart), dtype=float),
        harmonic_frequencies_cm=np.array((), dtype=float),
        source="LINK direct SONIC energy Hessian",
        provenance={
            "BACKEND": backend.name,
            "METHOD": backend.method,
            "BASIS": backend.basis,
        },
    )
    gf = gf_from_acquired_force_constants_and_xyzin(
        force_constants,
        geometry_input,
        source,
        block_by_irrep=True,
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(
            read_gic_definition_from_xyzin(source),
            coordinates_angstrom=np.asarray(geometry.coordinates_angstrom, dtype=float),
        ).rows,
        dtype=float,
    )
    cartesian_modes = cartesian_normal_modes_from_sonic_hessian(
        force_constants,
        b_matrix,
        masses,
        coordinates_bohr.reshape(len(geometry.atoms), 3),
        source="LINK SONIC Hessian",
    )
    result_path = target / "sonic-hessian-gf.json"
    payload = {
        "schema": SONIC_HESSIAN_RESULT_SCHEMA,
        "xyzin": str(source),
        "plan": plan.to_json(),
        "backend": backend.name,
        "method": backend.method,
        "basis": backend.basis,
        "electronic_state": int(backend.electronic_state),
        "state_tracking": backend.state_tracking,
        "energies_hartree": energies,
        "state_audit": state_audit,
        "force_constants_hartree_per_sonic_unit2": force_constants.tolist(),
        "g_matrix": gf.g_matrix.tolist(),
        "frequencies_cm-1": gf.frequencies_cm.tolist(),
        "cartesian_mass_weighted_normal_modes": cartesian_modes.mass_weighted_modes.tolist(),
        "normal_mode_basis_contract": "mass-weighted orthonormal Cartesian modes; 3N-6 rows",
        "gf_block_labels": list(gf.block_labels),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return SonicHessianResult(
        plan=plan,
        force_constants=force_constants,
        energies_hartree=energies,
        frequencies_cm=gf.frequencies_cm,
        g_matrix=gf.g_matrix,
        state_audit=state_audit,
        result_path=result_path,
    )
