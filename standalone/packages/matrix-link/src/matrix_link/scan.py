from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from matrix_core import CalculationResources, require_authorized_descendant_calculation
from matrix_chem.geometry_alignment import (
    aligned_cartesian_displacement,
    kabsch_rotation,
    rotate_cartesian_derivatives,
)
from matrix_chem.xyzin_geometry import read_xyzin_geometry
from matrix_qm import backend_method_name
from .internal_coordinates import cartesian_from_internal_jacobian


BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
ORACLE_LINK_POINT_RESULT_SCHEMA = "oracle.link.point_result.v1"
ORACLE_LINK_SCAN_MANIFEST_SCHEMA = "oracle.link.scan_manifest.v1"
ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA = "oracle.link.finite_difference_derivatives.v1"
FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA = "matrix.link.full_precision_derivatives.v1"
FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME = "matrix_derivatives.json"


@dataclass(frozen=True)
class PESExplorationPolicy:
    """Symmetry/coordinate policy shared by scans, SENTINEL and PES fitting."""

    retained_group: str = "C1"
    pointwise_oracle_symmetry: bool = True
    pointwise_cartesian_symmetrization: bool = True
    separate_exocyclic_torsions: bool = True

    def __post_init__(self) -> None:
        group = str(self.retained_group).strip().upper() or "C1"
        object.__setattr__(self, "retained_group", group)
        if self.pointwise_cartesian_symmetrization and not self.pointwise_oracle_symmetry:
            raise ValueError("pointwise symmetrization requires ORACLE symmetry perception")
        if not self.separate_exocyclic_torsions:
            raise ValueError("PES exploration requires separate exocyclic torsions")

    def protocol_payload(self) -> dict[str, object]:
        return {
            "mode": "PES_EXPLORATION",
            "retained_group": self.retained_group,
            "sonic_symmetry": (
                "C1_UNSYMMETRIZED" if self.retained_group == "C1" else "RETAINED_GROUP_ONLY"
            ),
            "pointwise_oracle_symmetry": self.pointwise_oracle_symmetry,
            "pointwise_cartesian_symmetrization": self.pointwise_cartesian_symmetrization,
            "separate_exocyclic_torsions": self.separate_exocyclic_torsions,
        }

    @classmethod
    def monte_carlo(cls) -> "PESExplorationPolicy":
        """Fast C1 policy for independent GA/Monte Carlo candidates.

        Random intermolecular poses have no symmetry contract to recover.
        Disabling pointwise perception also prevents a geometry-only operation
        from scaling with the general molecular-symmetry machinery.
        """

        return cls(
            retained_group="C1",
            pointwise_oracle_symmetry=False,
            pointwise_cartesian_symmetrization=False,
            separate_exocyclic_torsions=True,
        )


@dataclass(frozen=True)
class PointSymmetryInfo:
    point_group: str
    operation_count: int
    projection_status: str
    projection_max_displacement_angstrom: float
    projection_rms_displacement_angstrom: float


@dataclass(frozen=True)
class CoordinateDirection:
    """Cartesian displacement basis vector for a LINK scan coordinate."""

    kind: str
    label: str
    vector_angstrom: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_angstrom, dtype=float).reshape(-1)
        if vector.size == 0:
            raise ValueError("coordinate direction cannot be empty")
        if not np.all(np.isfinite(vector)):
            raise ValueError("coordinate direction contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("coordinate direction has zero norm")
        object.__setattr__(self, "vector_angstrom", vector)


@dataclass(frozen=True)
class ScanPoint:
    index: int
    displacement: float
    coordinates_angstrom: np.ndarray
    xyz_path: Path | None = None
    result_path: Path | None = None
    status: str = "prepared"
    symmetry_analyzed: bool = False
    point_group: str = ""
    symmetry_operation_count: int = 0
    symmetry_projection_status: str = "NOT_ANALYZED"
    symmetry_projection_max_displacement_angstrom: float = 0.0
    symmetry_projection_rms_displacement_angstrom: float = 0.0
    retained_group: str = "C1"

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates_angstrom, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("scan point coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coords)):
            raise ValueError("scan point coordinates contain non-finite values")
        object.__setattr__(self, "coordinates_angstrom", coords)
        if self.xyz_path is not None:
            object.__setattr__(self, "xyz_path", Path(self.xyz_path))
        if self.result_path is not None:
            object.__setattr__(self, "result_path", Path(self.result_path))
        object.__setattr__(self, "retained_group", str(self.retained_group).strip().upper() or "C1")


@dataclass(frozen=True)
class PointEvaluationResult:
    point_index: int
    displacement: float
    energy_hartree: float | None = None
    gradient_hartree_per_bohr: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None
    backend_coordinates_angstrom: np.ndarray | None = None
    frame_alignment_rms_angstrom: float = 0.0
    status: str = "completed"
    message: str = ""
    source: str = ""
    point_group: str = ""
    symmetry_projection_status: str = "NOT_ANALYZED"
    symmetry_projection_max_displacement_angstrom: float = 0.0
    symmetry_projection_rms_displacement_angstrom: float = 0.0
    execution: dict[str, object] = field(default_factory=dict)
    schema: str = ORACLE_LINK_POINT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.energy_hartree is not None:
            object.__setattr__(self, "energy_hartree", float(self.energy_hartree))
        for name in ("gradient_hartree_per_bohr", "hessian_hartree_per_bohr2"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains non-finite values")
                object.__setattr__(self, name, array)
        if self.backend_coordinates_angstrom is not None:
            coordinates = np.asarray(self.backend_coordinates_angstrom, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError("backend coordinates must have shape natoms x 3")
            object.__setattr__(self, "backend_coordinates_angstrom", coordinates)
        object.__setattr__(
            self, "frame_alignment_rms_angstrom", float(self.frame_alignment_rms_angstrom)
        )


@dataclass(frozen=True)
class FiniteDifferenceDerivatives:
    coordinate_label: str
    energy_derivatives_hartree: tuple[float, ...]
    gradient_derivatives_hartree_per_bohr: tuple[np.ndarray, ...] = ()
    hessian_derivatives_hartree_per_bohr2: tuple[np.ndarray, ...] = ()
    fit_rank: int = 0
    residual_norm: float = 0.0


@dataclass(frozen=True)
class QMScanBackend:
    """Electronic-structure backend settings for LINK point scans."""

    name: str
    route: str = ""
    method: str = ""
    reference: str | None = None
    basis: str = ""
    dispersion_contract: str | None = None
    basis_file: Path | None = None
    charge: int = 0
    multiplicity: int = 1
    executable: str | None = None
    timeout: float | None = None
    env: dict[str, str] | None = None
    extra_args: tuple[str, ...] = ()
    oniom_high_atoms: tuple[int, ...] = ()
    oniom_atom_types: tuple[str, ...] = ()
    oniom_low_route: str = "#p UFF Force"
    gaussian_connectivity_bonds: tuple[tuple[int, int, float], ...] | None = None
    processors: int = 1
    memory_gb: int | None = None
    force_field: Path | None = None
    zaff_library: Path | None = None
    zaff_gaff_parameters: Path | None = None
    resolution: Mapping[str, object] | None = None
    properties: tuple[str, ...] = ()
    device: str | None = None
    zaff_zoom_level: str | None = None
    gradient_mode: str = "analytic"
    numerical_gradient_step_bohr: float = 1.0e-3
    numerical_gradient_stencil: str = "central"
    electronic_state: int = 0
    excited_states: int | None = None
    state_spin: str = "singlet"
    freeze_core: bool = False
    state_tracking: str = "apoc"
    state_tracking_minimum_overlap: float = 0.70
    state_tracking_ambiguity_margin: float = 0.05
    state_tracking_initial_roots: int | None = None
    state_tracking_root_increment: int = 2
    state_tracking_max_roots: int | None = None
    state_tracking_max_displacement_halvings: int = 3
    restart_artifact: Path | None = None
    restart_projection: str | None = None
    scf_convergence: Mapping[str, float] | None = None
    restart_reuse_for_displacements: bool = True


def prepare_pes_exploration_geometry(
    xyzin_path: Path | str,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    policy: PESExplorationPolicy | None = None,
) -> tuple[np.ndarray, PointSymmetryInfo]:
    """Perceive and optionally project the instantaneous symmetry of one PES point."""

    from matrix_chem import MolecularGeometry, SymmetryThresholds, read_symmetry_thresholds
    from matrix_chem.symmetry import (
        analyze_molecular_symmetry,
        symmetry_projection_diagnostics,
        symmetrize_molecular_geometry,
    )

    resolved = policy or PESExplorationPolicy()
    coords = np.asarray(coordinates_angstrom, dtype=float)
    geometry = MolecularGeometry(
        atoms=tuple(str(atom) for atom in atoms),
        coordinates_angstrom=coords,
    )
    if not resolved.pointwise_oracle_symmetry:
        return coords.copy(), PointSymmetryInfo(
            point_group="NOT_ANALYZED",
            operation_count=0,
            projection_status="DISABLED",
            projection_max_displacement_angstrom=0.0,
            projection_rms_displacement_angstrom=0.0,
        )
    try:
        thresholds = read_symmetry_thresholds(Path(xyzin_path))
    except ValueError:
        # Plain XYZ inputs accepted by the low-level scan API have no frozen
        # ORACLE section yet; use the same documented ORACLE defaults.
        thresholds = SymmetryThresholds()
    symmetry = analyze_molecular_symmetry(
        geometry,
        distance_tolerance=thresholds.distance_angstrom,
        inertia_tolerance=thresholds.inertia_relative,
        max_rotation_order=thresholds.max_rotation_order,
    )
    projected = (
        symmetrize_molecular_geometry(geometry, symmetry)
        if resolved.pointwise_cartesian_symmetrization
        else geometry
    )
    diagnostics = symmetry_projection_diagnostics(geometry, projected, symmetry)
    return np.asarray(projected.coordinates_angstrom, dtype=float), PointSymmetryInfo(
        point_group=symmetry.point_group,
        operation_count=len(symmetry.operations),
        projection_status=(
            diagnostics.status
            if resolved.pointwise_cartesian_symmetrization
            else "PERCEIVED_NOT_PROJECTED"
        ),
        projection_max_displacement_angstrom=diagnostics.max_displacement_angstrom,
        projection_rms_displacement_angstrom=diagnostics.rms_displacement_angstrom,
    )


def _scan_point_with_pointwise_symmetry(
    xyzin_path: Path | str,
    atoms: Sequence[str],
    point: ScanPoint,
    *,
    policy: PESExplorationPolicy,
) -> ScanPoint:
    if point.symmetry_analyzed and point.retained_group == policy.retained_group:
        return point
    coordinates, symmetry = prepare_pes_exploration_geometry(
        xyzin_path,
        atoms,
        point.coordinates_angstrom,
        policy=policy,
    )
    return replace(
        point,
        coordinates_angstrom=coordinates,
        symmetry_analyzed=True,
        point_group=symmetry.point_group,
        symmetry_operation_count=symmetry.operation_count,
        symmetry_projection_status=symmetry.projection_status,
        symmetry_projection_max_displacement_angstrom=(
            symmetry.projection_max_displacement_angstrom
        ),
        symmetry_projection_rms_displacement_angstrom=(
            symmetry.projection_rms_displacement_angstrom
        ),
        retained_group=policy.retained_group,
    )


def coordinate_direction_from_gic(
    xyzin_path: Path | str,
    coordinate: str | int,
    *,
    reference_coordinates_angstrom: np.ndarray | None = None,
    rcond: float = 1.0e-8,
    definition: object | None = None,
) -> CoordinateDirection:
    """Build a linearized Cartesian displacement direction from a frozen SONIC/GIC row."""

    from matrix_smith import build_gic_b_matrix, read_gic_definition_from_xyzin

    target = Path(xyzin_path)
    resolved_definition = (
        read_gic_definition_from_xyzin(target) if definition is None else definition
    )
    coords = (
        np.asarray(reference_coordinates_angstrom, dtype=float)
        if reference_coordinates_angstrom is not None
        else np.asarray(resolved_definition.reference_coordinates_angstrom, dtype=float)
    )
    b_matrix = np.asarray(
        build_gic_b_matrix(resolved_definition, coordinates_angstrom=coords).rows,
        dtype=float,
    )
    labels = tuple(gic.identifier for gic in resolved_definition.gics)
    names = tuple(gic.name for gic in resolved_definition.gics)
    index = _coordinate_index(coordinate, labels, names)
    from .fragment_backtransform import direct_fragment_rigid_tangent

    cartesian_from_q = cartesian_from_internal_jacobian(b_matrix, rcond=rcond)
    fragment_tangent = direct_fragment_rigid_tangent(
        resolved_definition,
        coords,
        b_matrix,
    )
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_q[:, handled_index] = fragment_tangent.cartesian_from_q[:, handled_index]
    direction = cartesian_from_q[:, index]
    return CoordinateDirection(
        kind="sonic",
        label=labels[index],
        vector_angstrom=direction,
        source=(f"#GIC {target}" if definition is None else f"PES-exploration SONIC {target}"),
    )


def coordinate_direction_from_pes_exploration_gic(
    xyzin_path: Path | str,
    coordinate: str | int,
    *,
    retained_group: str = "C1",
    rcond: float = 1.0e-8,
) -> CoordinateDirection:
    """Build a direction in the scan-safe transient SONIC contract."""

    from matrix_smith import build_pes_exploration_gic_definition_from_xyzin

    definition = build_pes_exploration_gic_definition_from_xyzin(
        Path(xyzin_path), retained_group=retained_group
    )
    return coordinate_direction_from_gic(
        xyzin_path,
        coordinate,
        rcond=rcond,
        definition=definition,
    )


def coordinate_direction_from_normal_mode(
    xyzin_path: Path | str,
    mode: int,
) -> CoordinateDirection:
    """Build a Cartesian displacement direction from a stored #NORMAL_MODES row."""

    from matrix_qm import read_normal_modes_section

    modes = read_normal_modes_section(Path(xyzin_path))
    mode_index = int(mode)
    if mode_index < 1 or mode_index > modes.modes.shape[0]:
        raise ValueError(f"normal-mode index {mode_index} outside 1..{modes.modes.shape[0]}")
    vector = np.asarray(modes.modes[mode_index - 1], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"normal-mode {mode_index} has zero norm")
    return CoordinateDirection(
        kind="normal_mode",
        label=f"Q{mode_index}",
        vector_angstrom=vector / norm,
        source=f"#NORMAL_MODES {xyzin_path}",
    )


def coordinate_direction_from_cartesian_vector(
    vector: Sequence[float] | np.ndarray,
    *,
    label: str = "cartesian",
) -> CoordinateDirection:
    """Build a direction from an explicit Cartesian vector."""

    return CoordinateDirection(kind="cartesian", label=label, vector_angstrom=np.asarray(vector))


def prepare_coordinate_scan(
    xyzin_path: Path | str,
    direction: CoordinateDirection,
    displacements: Sequence[float],
    *,
    run_dir: Path | str | None = None,
    write_xyz: bool = True,
    exploration_policy: PESExplorationPolicy | None = None,
) -> tuple[ScanPoint, ...]:
    """Generate displaced geometries for a one-dimensional LINK scan."""

    geometry = read_xyzin_geometry(Path(xyzin_path))
    coords0 = np.asarray(geometry.coordinates_angstrom, dtype=float)
    vector = np.asarray(direction.vector_angstrom, dtype=float)
    if vector.size != coords0.size:
        raise ValueError(
            f"coordinate direction has {vector.size} components, expected {coords0.size}"
        )
    target_dir = Path(run_dir) if run_dir is not None else None
    if target_dir is not None and write_xyz:
        target_dir.mkdir(parents=True, exist_ok=True)
    policy = exploration_policy or PESExplorationPolicy()
    points: list[ScanPoint] = []
    for index, displacement in enumerate(displacements):
        value = float(displacement)
        raw_coords = coords0 + value * vector.reshape(coords0.shape)
        coords, symmetry = prepare_pes_exploration_geometry(
            xyzin_path,
            geometry.atoms,
            raw_coords,
            policy=policy,
        )
        xyz_path = None
        result_path = None
        if target_dir is not None:
            xyz_path = target_dir / f"point_{index:04d}.xyz"
            result_path = target_dir / f"point_{index:04d}.json"
            if write_xyz:
                _write_xyz(
                    xyz_path,
                    geometry.atoms,
                    coords,
                    f"{direction.label} {value:.12g}; ORACLE point group {symmetry.point_group}",
                )
        points.append(
            ScanPoint(
                index=index,
                displacement=value,
                coordinates_angstrom=coords,
                xyz_path=xyz_path,
                result_path=result_path,
                symmetry_analyzed=True,
                point_group=symmetry.point_group,
                symmetry_operation_count=symmetry.operation_count,
                symmetry_projection_status=symmetry.projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    symmetry.projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    symmetry.projection_rms_displacement_angstrom
                ),
                retained_group=policy.retained_group,
            )
        )
    return tuple(points)


def write_scan_manifest(
    path: Path | str,
    *,
    xyzin_path: Path | str,
    direction: CoordinateDirection,
    points: Sequence[ScanPoint],
    engine_command: str = "",
    external_protocol: str = "xyz-energy-gradient-json-v1",
    exploration_policy: PESExplorationPolicy | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    policy = exploration_policy or PESExplorationPolicy(
        retained_group=points[0].retained_group if points else "C1"
    )
    payload = {
        "schema": ORACLE_LINK_SCAN_MANIFEST_SCHEMA,
        "xyzin": str(xyzin_path),
        "coordinate": {
            "kind": direction.kind,
            "label": direction.label,
            "source": direction.source,
        },
        "backend": {
            "engine_command": engine_command,
            "external_protocol": external_protocol,
        },
        "pes_exploration": policy.protocol_payload(),
        "points": [
            {
                "index": point.index,
                "displacement": point.displacement,
                "xyz_path": str(point.xyz_path) if point.xyz_path is not None else None,
                "result_path": str(point.result_path) if point.result_path is not None else None,
                "status": point.status,
                "point_group": point.point_group,
                "symmetry_operation_count": point.symmetry_operation_count,
                "symmetry_projection_status": point.symmetry_projection_status,
                "symmetry_projection_max_displacement_angstrom": (
                    point.symmetry_projection_max_displacement_angstrom
                ),
                "symmetry_projection_rms_displacement_angstrom": (
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            }
            for point in points
        ],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_external_scan_points(
    points: Sequence[ScanPoint],
    *,
    engine_command: str,
    timeout: float | None = None,
) -> tuple[PointEvaluationResult, ...]:
    """Run an external point evaluator for prepared scan geometries.

    The command is a shell-like template.  The placeholders ``{xyz}``,
    ``{result}``, ``{index}``, ``{displacement}`` and ``{workdir}`` are expanded
    for each point.  The evaluator must write a JSON result file following
    ``oracle.link.point_result.v1``.
    """

    if not engine_command.strip():
        raise ValueError("external scan needs an engine command")
    results: list[PointEvaluationResult] = []
    for point in points:
        if point.xyz_path is None or point.result_path is None:
            raise ValueError("external scan points need xyz_path and result_path")
        workdir = point.xyz_path.parent
        result_path = point.result_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = _format_command(
            engine_command,
            xyz=point.xyz_path,
            result=result_path,
            index=point.index,
            displacement=point.displacement,
            workdir=workdir,
        )
        require_authorized_descendant_calculation(
            backend="LINK/external-scan-evaluator",
            input_path=point.xyz_path,
            command=cmd,
            workdir=workdir,
        )
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            check=False,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            results.append(
                PointEvaluationResult(
                    point_index=point.index,
                    displacement=point.displacement,
                    status="failed",
                    message=completed.stdout.strip(),
                    source=" ".join(cmd),
                    point_group=point.point_group,
                    symmetry_projection_status=point.symmetry_projection_status,
                    symmetry_projection_max_displacement_angstrom=(
                        point.symmetry_projection_max_displacement_angstrom
                    ),
                    symmetry_projection_rms_displacement_angstrom=(
                        point.symmetry_projection_rms_displacement_angstrom
                    ),
                )
            )
            continue
        results.append(read_point_result(result_path, point=point))
    return tuple(results)


def run_qm_scan_points(
    xyzin_path: Path | str,
    points: Sequence[ScanPoint],
    backend: QMScanBackend | str,
    *,
    run_dir: Path | str | None = None,
    exploration_policy: PESExplorationPolicy | None = None,
) -> tuple[PointEvaluationResult, ...]:
    """Run prepared scan points with a MATRIX electronic-structure adapter.

    This is the in-process equivalent of ``run_external_scan_points`` for
    supported QM backends.  Each point is evaluated in its own work directory,
    then written back to the standard LINK point-result JSON contract.
    """

    geometry = read_xyzin_geometry(Path(xyzin_path))
    spec = backend if isinstance(backend, QMScanBackend) else QMScanBackend(str(backend))
    name = _normalized_backend_name(spec.name)
    if name == "zaff":
        spec = _resolve_zaff_backend(
            spec,
            geometry.atoms,
            geometry.coordinates_angstrom,
        )
        name = _normalized_backend_name(spec.name)
    root = Path(run_dir) if run_dir is not None else _scan_root_from_points(points)
    root.mkdir(parents=True, exist_ok=True)
    prepared_points = tuple(
        input_point
        if exploration_policy is None
        else _scan_point_with_pointwise_symmetry(
            xyzin_path,
            geometry.atoms,
            input_point,
            policy=exploration_policy,
        )
        for input_point in points
    )
    if name == "zaff":
        try:
            zaff_results = _run_zaff_scan_batch(
                spec,
                geometry.atoms,
                prepared_points,
                xyzin_path=xyzin_path,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed point diagnostics
            zaff_results = tuple(
                PointEvaluationResult(
                    point_index=point.index,
                    displacement=point.displacement,
                    status="failed",
                    message=str(exc),
                    source=name,
                )
                for point in prepared_points
            )
        results: list[PointEvaluationResult] = []
        for point, raw_result in zip(prepared_points, zaff_results, strict=True):
            point_dir = root / f"point_{point.index:04d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            result = replace(
                raw_result,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
            if point.result_path is not None:
                write_point_result(point.result_path, result)
            write_point_result(point_dir / "point.json", result)
            results.append(result)
        return tuple(results)

    results: list[PointEvaluationResult] = []
    for point in prepared_points:
        point_dir = root / f"point_{point.index:04d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_qm_scan_point(name, spec, geometry.atoms, point, point_dir)
            result = _prefer_full_precision_derivative_artifact(result, point, point_dir)
            result = replace(
                result,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed point diagnostics
            result = PointEvaluationResult(
                point_index=point.index,
                displacement=point.displacement,
                status="failed",
                message=str(exc),
                source=name,
                point_group=point.point_group,
                symmetry_projection_status=point.symmetry_projection_status,
                symmetry_projection_max_displacement_angstrom=(
                    point.symmetry_projection_max_displacement_angstrom
                ),
                symmetry_projection_rms_displacement_angstrom=(
                    point.symmetry_projection_rms_displacement_angstrom
                ),
            )
        result = _attach_backend_resolution(result, spec)
        if point.result_path is not None:
            write_point_result(point.result_path, result)
        write_point_result(point_dir / "point.json", result)
        results.append(result)
    return tuple(results)


def write_full_precision_derivative_artifact(
    path: Path | str,
    *,
    energy_hartree: float,
    gradient_hartree_per_bohr: np.ndarray | None = None,
    hessian_hartree_per_bohr2: np.ndarray | None = None,
    coordinates_angstrom: np.ndarray | None = None,
) -> Path:
    """Write the backend-independent, float64-round-trippable derivative sidecar."""

    target = Path(path)
    payload: dict[str, object] = {
        "schema": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "energy_hartree": float(energy_hartree),
    }
    for key, value in (
        ("gradient_hartree_per_bohr", gradient_hartree_per_bohr),
        ("hessian_hartree_per_bohr2", hessian_hartree_per_bohr2),
        ("coordinates_angstrom", coordinates_angstrom),
    ):
        if value is not None:
            array = np.asarray(value, dtype=float)
            if np.any(~np.isfinite(array)):
                raise ValueError(f"full-precision derivative artifact contains non-finite {key}")
            payload[key] = array.tolist()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _prefer_full_precision_derivative_artifact(
    result: PointEvaluationResult,
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    """Apply a backend/wrapper derivative sidecar without decimal truncation."""

    artifact = point_dir / FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME
    if not artifact.is_file():
        return result
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema") != FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported full-precision derivative artifact schema")
    ncart = point.coordinates_angstrom.size
    gradient_value = payload.get("gradient_hartree_per_bohr")
    gradient = None if gradient_value is None else np.asarray(gradient_value, dtype=float).reshape(-1)
    if gradient is not None and gradient.size != ncart:
        raise ValueError("full-precision derivative gradient has the wrong dimension")
    hessian_value = payload.get("hessian_hartree_per_bohr2")
    hessian = None if hessian_value is None else np.asarray(hessian_value, dtype=float)
    if hessian is not None and hessian.shape != (ncart, ncart):
        raise ValueError("full-precision derivative Hessian has the wrong dimension")
    if gradient is not None and np.any(~np.isfinite(gradient)):
        raise ValueError("full-precision derivative gradient is non-finite")
    if hessian is not None and np.any(~np.isfinite(hessian)):
        raise ValueError("full-precision derivative Hessian is non-finite")
    source_coordinates_value = payload.get("coordinates_angstrom")
    if source_coordinates_value is not None:
        source_coordinates = np.asarray(source_coordinates_value, dtype=float)
        if source_coordinates.shape != point.coordinates_angstrom.shape:
            raise ValueError("full-precision derivative coordinates have the wrong dimension")
        rotation = kabsch_rotation(source_coordinates, point.coordinates_angstrom)
        rotation_gradient = np.zeros(ncart, dtype=float) if gradient is None else gradient
        gradient, hessian = rotate_cartesian_derivatives(
            rotation_gradient,
            rotation,
            hessian,
        )
        if gradient_value is None:
            gradient = None
    energy = float(payload["energy_hartree"])
    if not math.isfinite(energy):
        raise ValueError("full-precision derivative energy is non-finite")
    execution = dict(result.execution)
    source_transport = execution.get("derivative_transport")
    source_lossless = execution.get("derivative_transport_lossless_float64")
    execution.update(
        {
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": (
                "backend_full_precision_sidecar"
                if source_transport is None
                else f"backend_derivative_sidecar:{source_transport}"
            ),
            "derivative_transport_lossless_float64": (
                True if source_lossless is None else bool(source_lossless)
            ),
            "derivative_transport_artifact": str(artifact.resolve()),
        }
    )
    return replace(
        result,
        energy_hartree=energy,
        gradient_hartree_per_bohr=(
            result.gradient_hartree_per_bohr if gradient is None else gradient.reshape(-1)
        ),
        hessian_hartree_per_bohr2=(
            result.hessian_hartree_per_bohr2 if hessian is None else hessian
        ),
        execution=execution,
    )


def point_result_to_json(result: PointEvaluationResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": result.schema,
        "point_index": result.point_index,
        "displacement": result.displacement,
        "status": result.status,
        "message": result.message,
        "source": result.source,
        "point_group": result.point_group,
        "symmetry_projection_status": result.symmetry_projection_status,
        "symmetry_projection_max_displacement_angstrom": (
            result.symmetry_projection_max_displacement_angstrom
        ),
        "symmetry_projection_rms_displacement_angstrom": (
            result.symmetry_projection_rms_displacement_angstrom
        ),
        "execution": dict(result.execution),
    }
    if result.energy_hartree is not None:
        payload["energy_hartree"] = result.energy_hartree
    if result.gradient_hartree_per_bohr is not None:
        payload["gradient_hartree_per_bohr"] = (
            np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1).tolist()
        )
    if result.hessian_hartree_per_bohr2 is not None:
        payload["hessian_hartree_per_bohr2"] = np.asarray(
            result.hessian_hartree_per_bohr2, dtype=float
        ).tolist()
    if result.backend_coordinates_angstrom is not None:
        payload["backend_coordinates_angstrom"] = np.asarray(
            result.backend_coordinates_angstrom, dtype=float
        ).tolist()
        payload["frame_alignment_rms_angstrom"] = result.frame_alignment_rms_angstrom
    return payload


def write_point_result(path: Path | str, result: PointEvaluationResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(point_result_to_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_point_result(path: Path | str, *, point: ScanPoint | None = None) -> PointEvaluationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", ORACLE_LINK_POINT_RESULT_SCHEMA)
    if schema != ORACLE_LINK_POINT_RESULT_SCHEMA:
        raise ValueError(f"unsupported LINK point-result schema: {schema}")
    point_index = int(payload.get("point_index", point.index if point is not None else 0))
    displacement = float(
        payload.get("displacement", point.displacement if point is not None else 0.0)
    )
    gradient = payload.get("gradient_hartree_per_bohr")
    hessian = payload.get("hessian_hartree_per_bohr2")
    backend_coordinates = payload.get("backend_coordinates_angstrom")
    return PointEvaluationResult(
        point_index=point_index,
        displacement=displacement,
        energy_hartree=payload.get("energy_hartree"),
        gradient_hartree_per_bohr=None if gradient is None else np.asarray(gradient, dtype=float),
        hessian_hartree_per_bohr2=None if hessian is None else np.asarray(hessian, dtype=float),
        backend_coordinates_angstrom=(
            None if backend_coordinates is None else np.asarray(backend_coordinates, dtype=float)
        ),
        frame_alignment_rms_angstrom=float(payload.get("frame_alignment_rms_angstrom", 0.0)),
        status=str(payload.get("status", "completed")),
        message=str(payload.get("message", "")),
        source=str(payload.get("source", path)),
        point_group=str(payload.get("point_group", point.point_group if point is not None else "")),
        symmetry_projection_status=str(
            payload.get(
                "symmetry_projection_status",
                point.symmetry_projection_status if point is not None else "NOT_ANALYZED",
            )
        ),
        symmetry_projection_max_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_max_displacement_angstrom",
                (point.symmetry_projection_max_displacement_angstrom if point is not None else 0.0),
            )
        ),
        symmetry_projection_rms_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_rms_displacement_angstrom",
                (point.symmetry_projection_rms_displacement_angstrom if point is not None else 0.0),
            )
        ),
        execution=dict(payload.get("execution", {})),
        schema=schema,
    )


def write_point_results_jsonl(path: Path | str, results: Iterable[PointEvaluationResult]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(point_result_to_json(result), sort_keys=True) + "\n")
    return target


def read_point_results_jsonl(path: Path | str) -> tuple[PointEvaluationResult, ...]:
    results: list[PointEvaluationResult] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        results.append(
            PointEvaluationResult(
                point_index=int(payload["point_index"]),
                displacement=float(payload["displacement"]),
                energy_hartree=payload.get("energy_hartree"),
                gradient_hartree_per_bohr=payload.get("gradient_hartree_per_bohr"),
                hessian_hartree_per_bohr2=payload.get("hessian_hartree_per_bohr2"),
                status=str(payload.get("status", "completed")),
                message=str(payload.get("message", "")),
                source=str(payload.get("source", "")),
                point_group=str(payload.get("point_group", "")),
                symmetry_projection_status=str(
                    payload.get("symmetry_projection_status", "NOT_ANALYZED")
                ),
                symmetry_projection_max_displacement_angstrom=float(
                    payload.get("symmetry_projection_max_displacement_angstrom", 0.0)
                ),
                symmetry_projection_rms_displacement_angstrom=float(
                    payload.get("symmetry_projection_rms_displacement_angstrom", 0.0)
                ),
                execution=dict(payload.get("execution", {})),
                schema=str(payload.get("schema", ORACLE_LINK_POINT_RESULT_SCHEMA)),
            )
        )
    return tuple(results)


def finite_difference_derivatives_to_json(
    derivatives: FiniteDifferenceDerivatives,
) -> dict[str, object]:
    return {
        "schema": ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA,
        "coordinate_label": derivatives.coordinate_label,
        "energy_derivatives_hartree": list(derivatives.energy_derivatives_hartree),
        "gradient_derivatives_hartree_per_bohr": [
            np.asarray(item, dtype=float).reshape(-1).tolist()
            for item in derivatives.gradient_derivatives_hartree_per_bohr
        ],
        "hessian_derivatives_hartree_per_bohr2": [
            np.asarray(item, dtype=float).tolist()
            for item in derivatives.hessian_derivatives_hartree_per_bohr2
        ],
        "fit_rank": int(derivatives.fit_rank),
        "residual_norm": float(derivatives.residual_norm),
    }


def write_finite_difference_derivatives(
    path: Path | str,
    derivatives: FiniteDifferenceDerivatives,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(finite_difference_derivatives_to_json(derivatives), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def read_finite_difference_derivatives(path: Path | str) -> FiniteDifferenceDerivatives:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA)
    if schema != ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA:
        raise ValueError(f"unsupported LINK finite-difference derivative schema: {schema}")
    return FiniteDifferenceDerivatives(
        coordinate_label=str(payload["coordinate_label"]),
        energy_derivatives_hartree=tuple(
            float(value) for value in payload.get("energy_derivatives_hartree", ())
        ),
        gradient_derivatives_hartree_per_bohr=tuple(
            np.asarray(item, dtype=float)
            for item in payload.get("gradient_derivatives_hartree_per_bohr", ())
        ),
        hessian_derivatives_hartree_per_bohr2=tuple(
            np.asarray(item, dtype=float)
            for item in payload.get("hessian_derivatives_hartree_per_bohr2", ())
        ),
        fit_rank=int(payload.get("fit_rank", 0)),
        residual_norm=float(payload.get("residual_norm", 0.0)),
    )


def finite_difference_derivatives(
    results: Sequence[PointEvaluationResult],
    *,
    coordinate_label: str,
    max_order: int = 4,
) -> FiniteDifferenceDerivatives:
    """Fit local derivatives with respect to the scan displacement."""

    completed = [item for item in results if item.status == "completed"]
    if len(completed) < 2:
        raise ValueError("at least two completed points are needed for finite differences")
    x = np.asarray([item.displacement for item in completed], dtype=float)
    if len(set(float(value) for value in x)) != x.size:
        raise ValueError("scan displacements must be unique")
    order = max(1, min(int(max_order), x.size - 1))
    design = _taylor_design(x, order)
    energy_derivatives: tuple[float, ...] = ()
    rank = 0
    residual_norm = 0.0
    energies = [item.energy_hartree for item in completed]
    if all(value is not None for value in energies):
        coeff, residual_norm, rank = _least_squares_derivatives(
            design, np.asarray(energies, dtype=float)
        )
        energy_derivatives = tuple(float(value) for value in coeff[1:])
    gradient_derivatives = _array_derivatives(
        completed,
        design,
        attr="gradient_hartree_per_bohr",
    )
    hessian_derivatives = _array_derivatives(
        completed,
        design,
        attr="hessian_hartree_per_bohr2",
    )
    return FiniteDifferenceDerivatives(
        coordinate_label=coordinate_label,
        energy_derivatives_hartree=energy_derivatives,
        gradient_derivatives_hartree_per_bohr=gradient_derivatives,
        hessian_derivatives_hartree_per_bohr2=hessian_derivatives,
        fit_rank=int(rank),
        residual_norm=float(residual_norm),
    )


def symmetric_displacements(step: float, points_each_side: int) -> tuple[float, ...]:
    count = int(points_each_side)
    if count <= 0:
        raise ValueError("points_each_side must be positive")
    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    values = [float(i * h) for i in range(-count, count + 1)]
    return tuple(values)


def _coordinate_index(
    coordinate: str | int, labels: tuple[str, ...], names: tuple[str, ...]
) -> int:
    if isinstance(coordinate, int):
        index = int(coordinate)
        if index < 1 or index > len(labels):
            raise ValueError(f"GIC index {index} outside 1..{len(labels)}")
        return index - 1
    wanted = str(coordinate)
    for index, label in enumerate(labels):
        if label == wanted:
            return index
    for index, name in enumerate(names):
        if name == wanted:
            return index
    raise ValueError(f"unknown GIC coordinate {wanted!r}")


def _write_xyz(path: Path, atoms: tuple[str, ...], coords: np.ndarray, comment: str) -> None:
    lines = [str(len(atoms)), comment]
    for atom, xyz in zip(atoms, coords):
        lines.append(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_root_from_points(points: Sequence[ScanPoint]) -> Path:
    for point in points:
        if point.result_path is not None:
            return Path(point.result_path).parent
        if point.xyz_path is not None:
            return Path(point.xyz_path).parent
    return Path.cwd()


def _normalized_backend_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gaussian": "gaussian",
        "g16": "gaussian",
        "gdv": "gaussian",
        "gaussian16": "gaussian",
        "fragmentoniom": "fragment_oniom",
        "cfour": "cfour",
        "orca": "orca",
        "molpro": "molpro",
        "mrcc": "mrcc",
        "xtb": "xtb",
        "pyscf": "pyscf",
        "et": "et",
        "zaff": "zaff",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported QM scan backend: {name}") from exc


def _resolve_zaff_backend(
    spec: QMScanBackend,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    cm5_charges: Sequence[float] | None = None,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    charge_source: str = "",
    bond_order_source: str = "",
) -> QMScanBackend:
    """Resolve an existing ZAFF artifact without invoking ARCHITECT."""

    if _normalized_backend_name(spec.name) != "zaff" or spec.resolution is not None:
        return spec
    del cm5_charges, mayer_bond_orders, charge_source, bond_order_source
    from matrix_zaff import (
        explicit_zaff_resolution,
        resolve_zaff_force_field,
    )

    resolution = (
        explicit_zaff_resolution(spec.force_field)
        if spec.force_field is not None
        else resolve_zaff_force_field(
            atoms,
            coordinates_angstrom,
            charge=spec.charge,
            multiplicity=spec.multiplicity,
            library=spec.zaff_library,
        )
    )
    provenance = resolution.to_dict()
    if resolution.force_field_path is not None:
        return replace(
            spec,
            force_field=resolution.force_field_path,
            resolution=provenance,
        )
    return replace(
        spec,
        name="xtb",
        route="--gfnff",
        method="",
        basis="",
        force_field=None,
        resolution=provenance,
    )


def _attach_backend_resolution(
    result: PointEvaluationResult,
    spec: QMScanBackend,
) -> PointEvaluationResult:
    if spec.resolution is None:
        return result
    execution = dict(result.execution)
    execution["force_field_resolution"] = dict(spec.resolution)
    source = result.source
    if spec.resolution.get("selection") == "GFN_FF_FALLBACK":
        source = f"{source} [ZAFF fallback: GFN-FF]"
    return replace(result, source=source, execution=execution)


def _run_qm_scan_point(
    name: str,
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    if name == "gaussian":
        return _run_gaussian_scan_point(spec, atoms, point, point_dir)
    if name == "fragment_oniom":
        return _run_fragment_oniom_scan_point(spec, atoms, point, point_dir)
    if name == "orca":
        return _run_orca_scan_point(spec, atoms, point, point_dir)
    if name == "molpro":
        return _run_molpro_scan_point(spec, atoms, point, point_dir)
    if name == "mrcc":
        return _run_mrcc_scan_point(spec, atoms, point, point_dir)
    if name == "cfour":
        return _run_cfour_scan_point(spec, atoms, point, point_dir)
    if name == "xtb":
        return _run_xtb_scan_point(spec, atoms, point, point_dir)
    if name == "pyscf":
        return _run_pyscf_scan_point(spec, atoms, point, point_dir)
    if name == "et":
        return _run_et_scan_point(spec, atoms, point, point_dir)
    if name == "zaff":
        return _run_zaff_scan_point(spec, atoms, point)
    raise ValueError(f"unsupported QM scan backend: {name}")


def _run_zaff_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
) -> PointEvaluationResult:
    """Evaluate one point through the independent resident ZAFF backend."""

    return _run_zaff_scan_batch(spec, atoms, (point,))[0]


def _run_zaff_scan_batch(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    points: Sequence[ScanPoint],
    *,
    xyzin_path: Path | str | None = None,
) -> tuple[PointEvaluationResult, ...]:
    from matrix_engines import DerivativeOrder, PotentialSystem
    from matrix_zaff import ZaffBackend

    if spec.force_field is None:
        raise ValueError("ZAFF backend needs a compiled artifact path")
    properties = tuple(spec.properties or ("energy", "gradient"))
    requested = set(properties)
    unsupported = requested - {"energy", "gradient", "hessian"}
    if unsupported:
        raise ValueError(f"unsupported ZAFF properties: {sorted(unsupported)}")
    order = (
        DerivativeOrder.HESSIAN
        if "hessian" in requested
        else DerivativeOrder.GRADIENT
        if "gradient" in requested
        else DerivativeOrder.ENERGY
    )
    session = ZaffBackend().prepare(
        PotentialSystem(
            atoms=tuple(atoms),
            charge=spec.charge,
            multiplicity=spec.multiplicity,
        ),
        model=str(spec.force_field),
        options={
            "zoom_level": spec.zaff_zoom_level,
            "xyzin": xyzin_path,
        },
    )
    evaluations = session.evaluate_batch(
        tuple(point.coordinates_angstrom for point in points),
        derivative_order=order,
    )
    return tuple(
        _attach_backend_resolution(
            PointEvaluationResult(
                point_index=point.index,
                displacement=point.displacement,
                energy_hartree=(result.energy_hartree if "energy" in requested else None),
                gradient_hartree_per_bohr=(
                    result.gradient_hartree_per_bohr if "gradient" in requested else None
                ),
                hessian_hartree_per_bohr2=(
                    result.hessian_hartree_per_bohr2 if "hessian" in requested else None
                ),
                backend_coordinates_angstrom=point.coordinates_angstrom,
                source=f"ZAFF {Path(spec.force_field)}",
                execution={
                    **dict(result.execution),
                    "derivative_transport_contract": (
                        FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA
                    ),
                    "derivative_transport": "zaff_in_memory_float64",
                    "derivative_transport_lossless_float64": True,
                },
            ),
            spec,
        )
        for point, result in zip(points, evaluations, strict=True)
    )


def _run_gaussian_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_gaussian import (
        read_gaussian_log_geometry,
        read_gaussian_point_result,
        run_gaussian_job,
        write_gaussian_point_input,
        write_gaussian_oniom_point_input,
    )
    from matrix_gaussian.electronic import (
        GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION,
        gaussian_state_fingerprint_data,
        gaussian_state_reference_energy_hartree,
        write_gaussian_state_fingerprint_archive,
    )

    route = _gaussian_route(spec)
    writer = (
        write_gaussian_oniom_point_input if spec.oniom_high_atoms else write_gaussian_point_input
    )
    writer_kwargs = {}
    if spec.oniom_high_atoms:
        writer_kwargs = {
            "high_atoms": spec.oniom_high_atoms,
            "atom_types": spec.oniom_atom_types,
        }
    if spec.gaussian_connectivity_bonds is not None:
        writer_kwargs["connectivity_bonds"] = spec.gaussian_connectivity_bonds
    requested_mode = str(spec.gradient_mode).strip().casefold()
    writer_kwargs["ensure_force"] = requested_mode not in {
        "energy",
        "numerical",
        "link-numerical",
    }
    input_path = writer(
        point_dir / "gauin.gjf",
        atoms,
        point.coordinates_angstrom,
        route=route,
        title=f"MATRIX LINK point {point.index} {point.displacement:.12g}",
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        link0=tuple(
            item
            for item in (
                "%Chk=gauin.chk",
                f"%NProcShared={int(spec.processors)}" if int(spec.processors) > 1 else "",
                f"%Mem={int(spec.memory_gb)}GB" if spec.memory_gb is not None else "",
            )
            if item
        ),
        basis_set_file=spec.basis_file,
        **writer_kwargs,
    )
    run = run_gaussian_job(
        point_dir,
        executable=spec.executable,
        # ``run_gaussian_job`` resolves input_path relative to point_dir;
        # passing the already-prefixed Path duplicates the directory.
        input_path=input_path.name,
        timeout=spec.timeout,
        env=spec.env,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_gaussian_point_result(
        run.log_path,
        require_gradient=writer_kwargs["ensure_force"],
    )
    try:
        returned = read_gaussian_log_geometry(run.log_path).coordinates_angstrom
    except ValueError:
        # Minimal external wrappers may return only energy and gradient.  Full
        # production adapters return geometry as well, allowing frame recovery.
        returned = None
    execution = {
        "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "derivative_transport": "gaussian_restored_force_table",
        "derivative_transport_lossless_float64": False,
    }
    checkpoint = point_dir / "gauin.chk"
    if writer_kwargs["ensure_force"] and checkpoint.is_file():
        from matrix_gaussian import (
            formchk_checkpoint,
            read_gaussian_fchk_point_result,
        )

        fchk_path = formchk_checkpoint(
            checkpoint,
            point_dir / "gauin.fchk",
            timeout=spec.timeout,
        )
        fchk = read_gaussian_fchk_point_result(fchk_path)
        fchk_gradient = np.asarray(
            fchk.cartesian_gradient_hartree_per_bohr, dtype=float
        ).reshape(-1)
        if fchk_gradient.shape != (point.coordinates_angstrom.size,):
            raise RuntimeError("Gaussian checkpoint contains no complete Cartesian gradient")
        fchk_coordinates = (
            np.asarray(fchk.cartesian_coordinates_bohr, dtype=float)
            * BOHR_TO_ANGSTROM
        )
        rotation = kabsch_rotation(fchk_coordinates, point.coordinates_angstrom)
        transported_gradient, _unused_hessian = rotate_cartesian_derivatives(
            fchk_gradient,
            rotation,
        )
        precise_energy = (
            parsed.energy_hartree
            if fchk.total_energy_hartree is None
            else float(fchk.total_energy_hartree)
        )
        parsed = replace(
            parsed,
            energy_hartree=precise_energy,
            gradient_hartree_per_bohr=transported_gradient.reshape(-1),
        )
        derivative_artifact = write_full_precision_derivative_artifact(
            point_dir / FULL_PRECISION_DERIVATIVE_ARTIFACT_NAME,
            energy_hartree=precise_energy,
            gradient_hartree_per_bohr=transported_gradient,
            coordinates_angstrom=point.coordinates_angstrom,
        )
        execution.update(
            {
                "derivative_transport": "gaussian_checkpoint_fchk_decimal",
                "derivative_transport_lossless_float64": False,
                "derivative_transport_significant_digits": 9,
                "derivative_transport_artifact": str(derivative_artifact.resolve()),
                "restart_artifact": str(checkpoint.resolve()),
                "restart_artifact_format": "gaussian_chk",
            }
        )
    if spec.electronic_state > 0 and str(spec.state_tracking).strip().casefold() == "apoc":
        state_data = gaussian_state_fingerprint_data(run.log_path)
        if state_data is None:
            raise RuntimeError(
                "Gaussian excited-state LINK point contains no supported amplitude "
                "manifold for APOC state following"
            )
        state_ids, excitations, _vectors, _representation = state_data
        archive = write_gaussian_state_fingerprint_archive(
            run.log_path, point_dir / "gaussian_state_fingerprints.npz"
        )
        if archive is None:
            raise RuntimeError("Gaussian state-fingerprint archive could not be written")
        reference_energy = gaussian_state_reference_energy_hartree(run.log_path)
        if reference_energy is None:
            raise RuntimeError(
                "Gaussian excited-state point has no method-consistent reference energy"
            )
        execution.update(
            {
                "energy_protocol_id": (
                    "gaussian.eom_ccsd.precise_manifold_energy.v1"
                    if _representation == GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION
                    else "gaussian.tddft.tda.requested_root_energy.v1"
                ),
                "requested_state_energy_hartree": float(parsed.energy_hartree),
                "state_fingerprint_file": str(archive),
                "state_fingerprint_representation": _representation,
                "state_ids": [int(value) for value in state_ids],
                "ground_energy_hartree": float(reference_energy),
                "excitation_energies_hartree": [float(value) for value in excitations],
            }
        )
    # Gaussian prints Cartesian forces only after restoring the original
    # input axes. ``read_gaussian_log_geometry`` may instead return the
    # archive/standard orientation, so rotating the already-restored
    # derivatives with that geometry applies a spurious frame change.
    return _point_result_from_parsed(
        point,
        parsed,
        f"Gaussian {route}",
        returned,
        execution=execution,
        derivatives_in_point_frame=True,
    )


def _gaussian_route(spec: QMScanBackend) -> str:
    """Compose a complete Gaussian route from model and extra keywords.

    ``QMScanBackend.route`` is also used for backend-specific additions such
    as empirical dispersion.  When those additions do not contain a
    method/basis model, retain the explicitly selected ``method`` and
    ``basis`` instead of silently falling back to HF/STO-3G.
    """

    route = str(spec.route).strip()
    method = str(spec.method).strip() or "HF"
    engine_method = backend_method_name(spec.name, method)
    basis = "Gen" if spec.basis_file is not None else (str(spec.basis).strip() or "STO-3G")
    normalized_method = re.sub(r"[^A-Z0-9]", "", method.upper())
    separate_basis_methods = {
        "REVDSDPBEP86D3",
        "REVDSDPBEP86D3BJ",
        "REVDSDPBEP86D4",
    }
    model = (
        f"{engine_method} {basis}"
        if normalized_method in separate_basis_methods
        else f"{engine_method}/{basis}"
    )
    if not route:
        return f"#p {model} Force"
    tokens = route.lstrip("#").split()
    if any("/" in token for token in tokens):
        return route
    prefix = "#p" if not route.startswith("#") else route.split(maxsplit=1)[0]
    extras = route.lstrip("#").removeprefix("p").removeprefix("P").strip()
    return " ".join(item for item in (prefix, model, extras) if item)


def _run_fragment_oniom_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    """Evaluate explicit subtractive E_L(real)+E_H(model)-E_L(model)."""

    high_atoms = tuple(int(index) for index in spec.oniom_high_atoms)
    if not high_atoms or min(high_atoms) < 0 or max(high_atoms) >= len(atoms):
        raise ValueError("fragment ONIOM backend needs valid oniom_high_atoms")
    full_low_spec = QMScanBackend(
        name="gaussian",
        route=spec.oniom_low_route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
        gaussian_connectivity_bonds=spec.gaussian_connectivity_bonds,
    )
    fragment_high_spec = QMScanBackend(
        name="gaussian",
        route=spec.route,
        charge=0,
        multiplicity=1,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
    )
    fragment_low_spec = QMScanBackend(
        name="gaussian",
        route=spec.oniom_low_route,
        charge=0,
        multiplicity=1,
        executable=spec.executable,
        timeout=spec.timeout,
        env=spec.env,
        gaussian_connectivity_bonds=tuple(
            (high_atoms.index(left), high_atoms.index(right), order)
            for left, right, order in (spec.gaussian_connectivity_bonds or ())
            if left in high_atoms and right in high_atoms
        ),
    )
    low_full = _run_gaussian_scan_point(full_low_spec, atoms, point, point_dir / "low_full")
    local_coordinates = point.coordinates_angstrom[list(high_atoms)]
    local_atoms = tuple(atoms[index] for index in high_atoms)
    local_point = ScanPoint(
        index=point.index,
        displacement=point.displacement,
        coordinates_angstrom=local_coordinates,
    )
    high_fragment = _run_gaussian_scan_point(
        fragment_high_spec, local_atoms, local_point, point_dir / "high_fragment"
    )
    low_fragment = _run_gaussian_scan_point(
        fragment_low_spec, local_atoms, local_point, point_dir / "low_fragment"
    )
    components = (low_full, high_fragment, low_fragment)
    if any(
        item.energy_hartree is None or item.gradient_hartree_per_bohr is None for item in components
    ):
        raise RuntimeError("fragment ONIOM component lacks energy or analytic gradient")
    gradient = np.asarray(low_full.gradient_hartree_per_bohr, dtype=float).reshape((-1, 3))
    correction = (
        np.asarray(high_fragment.gradient_hartree_per_bohr, dtype=float)
        - np.asarray(low_fragment.gradient_hartree_per_bohr, dtype=float)
    ).reshape((-1, 3))
    gradient[list(high_atoms)] += correction
    energy = (
        float(low_full.energy_hartree)
        + float(high_fragment.energy_hartree)
        - float(low_fragment.energy_hartree)
    )
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient.reshape(-1),
        backend_coordinates_angstrom=point.coordinates_angstrom,
        status="completed",
        source=(
            f"fragment-ONIOM Gaussian: {spec.oniom_low_route} real + "
            f"{spec.route} model - {spec.oniom_low_route} model"
        ),
        message="three component energy/gradient evaluations",
    )


def _run_orca_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_orca import (
        orca_state_fingerprint_data,
        read_orca_output_geometry,
        read_orca_point_result,
        run_orca_job,
        write_orca_state_fingerprint_archive,
        write_orca_point_input,
    )

    route = _orca_route(spec)
    restart_orbitals = None
    if spec.restart_artifact is not None:
        source = Path(spec.restart_artifact).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ORCA restart artifact does not exist: {source}")
        if source.suffix.casefold() != ".gbw":
            raise ValueError("ORCA restart artifact must be a .gbw file")
        local_restart = point_dir / "central_reference.gbw"
        if source != local_restart.resolve():
            shutil.copy2(source, local_restart)
        restart_orbitals = local_restart.name
    input_path = write_orca_point_input(
        point_dir / "orca.inp",
        atoms,
        point.coordinates_angstrom,
        route=route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        electronic_state=spec.electronic_state,
        excited_states=spec.excited_states,
        processors=spec.processors,
        memory_gb=spec.memory_gb,
        restart_orbitals=restart_orbitals,
        restart_projection=spec.restart_projection,
        scf_convergence=spec.scf_convergence,
    )
    run = run_orca_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    requested_mode = str(spec.gradient_mode).strip().casefold()
    require_gradient = requested_mode not in {"energy", "numerical", "link-numerical"}
    parsed = read_orca_point_result(run.output_path, require_gradient=require_gradient)
    try:
        returned = read_orca_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    # Keep the energy-read contract explicit and stable.  APOC may request a
    # rerun with a different selected root; LINK must be able to prove that
    # the rerun used exactly the same ORCA quantity and not silently fall back
    # to the SCF or excitation energy alone.
    excited = int(spec.electronic_state) > 0
    execution = {
        "energy_protocol_id": (
            "orca.tda.final_single_point_energy.v1"
            if excited
            else "orca.final_single_point_energy.v1"
        ),
        "requested_electronic_state": int(spec.electronic_state),
        "requested_state_energy_hartree": float(parsed.energy_hartree),
        "restart_reuse_for_displacements": bool(
            spec.restart_reuse_for_displacements
        ),
        "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
        "derivative_transport": (
            "orca_engrad_full_precision"
            if run.output_path.with_suffix(".engrad").is_file()
            else "orca_output_fallback"
        ),
        "derivative_transport_lossless_float64": (
            run.output_path.with_suffix(".engrad").is_file()
        ),
    }
    gbw_path = input_path.with_suffix(".gbw")
    if gbw_path.is_file():
        execution.update(
            {
                "restart_artifact": str(gbw_path.resolve()),
                "restart_artifact_format": "orca_gbw",
                "restart_projection": (
                    "CMatrix"
                    if "rohf" in str(spec.reference or "").casefold()
                    else "FMatrix"
                ),
            }
        )
    if spec.electronic_state > 0 and str(spec.state_tracking).strip().casefold() == "apoc":
        state_data = orca_state_fingerprint_data(run.output_path)
        if state_data is None:
            raise RuntimeError("ORCA excited-state point contains no TDA state manifold")
        archive = write_orca_state_fingerprint_archive(
            run.output_path, point_dir / "orca_state_fingerprints.npz"
        )
        if archive is None:
            raise RuntimeError("ORCA state-fingerprint archive could not be written")
        state_ids, excitations, _vectors, representation = state_data
        execution.update(
            {
                "state_fingerprint_file": str(archive),
                "state_fingerprint_representation": representation,
                "state_ids": [int(value) for value in state_ids],
                # Root 1 is the ORCA excited-state total energy; reconstruct
                # the ground-state reference from its excitation energy.
                "ground_energy_hartree": float(parsed.energy_hartree - excitations[0]),
                "excitation_energies_hartree": [float(value) for value in excitations],
            }
        )
    return _point_result_from_parsed(point, parsed, f"ORCA {route}", returned, execution=execution)


def _orca_route(spec: QMScanBackend) -> str:
    """Compose ORCA model keywords with any route-only additions."""

    method = str(spec.method).strip() or "HF"
    # ORCA 6 names the available double hybrid as DSD-PBEP86 and takes D4 as
    # a separate keyword.  Accept the repository-level spelling as well.
    if method.casefold() in {"revsd-pbep86-d4", "revsd-pbep86"}:
        method = "DSD-PBEP86"
    basis = str(spec.basis).strip() or "STO-3G"
    route = str(spec.route).strip()
    if not route:
        route = "D4"
    tokens = route.split()
    normalized = {token.casefold() for token in tokens}
    requested_mode = str(spec.gradient_mode).strip().casefold()
    energy_only = requested_mode in {"energy", "numerical", "link-numerical"}
    additions = route if (energy_only or "engrad" in normalized) else f"{route} EnGrad"
    if spec.electronic_state > 0 and int(spec.electronic_state) >= 1:
        if "d4" not in normalized:
            additions = f"D4 {additions}"
        # Double hybrids require the corresponding RI-C auxiliary basis.
        if method.casefold() == "dsd-pbep86" and "/c" not in normalized:
            additions = f"{additions} {basis}/C"
    # A fully specified route supplied by a backend launcher is authoritative.
    # Do not prepend the generic HF/STO-3G defaults when the route already
    # contains a method/basis model such as ``B3LYP/G 6-31+G*``.
    # ORCA auxiliary-basis tokens (for example def2/J or def2-TZVP/C) contain
    # a slash but do not define the electronic method/orbital-basis model.
    # Only a non-auxiliary slash token makes the supplied route authoritative.
    route_has_model = any(
        "/" in token
        and not token.casefold().endswith(("/c", "/j", "/jk"))
        for token in tokens
    )
    if route_has_model or (method.casefold() in normalized and basis.casefold() in normalized):
        return additions
    return f"{method} {basis} {additions}"


def _run_molpro_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_molpro import (
        read_molpro_output_geometry,
        read_molpro_point_result,
        run_molpro_job,
        write_molpro_point_input,
    )

    method = spec.method or "hf"
    basis = spec.basis or "sto-3g"
    input_path = write_molpro_point_input(
        point_dir / "molpro.com",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    run = run_molpro_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_molpro_point_result(run.output_path)
    try:
        returned = read_molpro_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    return _point_result_from_parsed(
        point,
        parsed,
        f"Molpro {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "molpro_output_fallback",
            "derivative_transport_lossless_float64": False,
        },
    )


def _run_mrcc_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_mrcc import (
        read_mrcc_output_geometry,
        read_mrcc_point_result,
        run_mrcc_job,
        write_mrcc_point_input,
    )

    method = spec.method or "HF"
    basis = spec.basis or "STO-3G"
    input_path = write_mrcc_point_input(
        point_dir / "MINP",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    run = run_mrcc_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_mrcc_point_result(run.output_path)
    try:
        returned = read_mrcc_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    return _point_result_from_parsed(
        point,
        parsed,
        f"MRCC {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "mrcc_output_fallback",
            "derivative_transport_lossless_float64": False,
        },
    )


def _run_cfour_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_cfour import (
        parse_cfour_grd,
        read_cfour_point_result,
        run_cfour_job,
        write_cfour_point_input,
    )

    method = spec.method or "HF"
    basis = spec.basis or "6-31G*"
    input_path = write_cfour_point_input(
        point_dir / "ZMAT",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    _copy_cfour_genbas(point_dir, spec.env)
    run = run_cfour_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_cfour_point_result(point_dir / "GRD", output=run.output_path)
    _numbers, returned_bohr = parse_cfour_grd(point_dir / "GRD")
    returned = np.asarray(returned_bohr, dtype=float) * BOHR_TO_ANGSTROM
    return _point_result_from_parsed(
        point,
        parsed,
        f"CFOUR {method}/{basis}",
        returned,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "cfour_grd_full_precision",
            "derivative_transport_lossless_float64": True,
        },
    )


def _run_xtb_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_xtb import read_xtb_point_result, run_xtb_job, write_xtb_point_input

    input_path = write_xtb_point_input(
        point_dir / "xtb.xyz",
        atoms,
        point.coordinates_angstrom,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    options = list(shlex.split(spec.route)) if spec.route.strip() else ["--gfn", "2"]
    requested = {str(item).strip().lower() for item in spec.properties}
    # A Hessian request also requires the gradient channel for the LINK
    # evaluation contract and for downstream MEX projections.
    needs_gradient = not requested or bool(requested & {"gradient", "hessian"})
    if needs_gradient and "--grad" not in options:
        options.append("--grad")
    if "hessian" in requested and "--hess" not in options:
        options.append("--hess")
    if spec.charge and not any(item in options for item in ("--chrg", "--charge")):
        options.extend(("--chrg", str(spec.charge)))
    if spec.multiplicity > 1 and "--uhf" not in options:
        options.extend(("--uhf", str(spec.multiplicity - 1)))
    options.extend(spec.extra_args)
    resources = CalculationResources(
        process_count=1,
        threads_per_process=max(1, int(spec.processors)),
        memory_per_job_gb=float(spec.memory_gb or 1),
        concurrent_jobs=1,
    )
    run = run_xtb_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=tuple(options),
        resources=resources,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    hessian_run = run
    if "hessian" in requested:
        # xTB accepts --grad and --hess in the command line but only writes
        # the native Hessian bundle when the numerical-Hessian job is run as a
        # separate pass.  Keep the gradient pass for LINK's point contract,
        # then refresh the same work directory with --hess.
        hessian_options = [item for item in options if item != "--grad"]
        hessian_run = run_xtb_job(
            point_dir,
            executable=spec.executable,
            input_path=input_path,
            timeout=spec.timeout,
            env=spec.env,
            extra_args=tuple(hessian_options),
            resources=resources,
        )
        if hessian_run.success is not True:
            raise RuntimeError(hessian_run.message)
    parsed = read_xtb_point_result(
        hessian_run.output_path,
        geometry=input_path,
        require_gradient=needs_gradient,
    )
    if "hessian" in requested:
        from matrix_xtb import hessian_input_from_xtb_files

        hessian = hessian_input_from_xtb_files(
            point_dir / "hessian",
            geometry=input_path,
            spectrum=point_dir / "vibspectrum",
            output=run.output_path,
        ).cartesian_hessian
        parsed = replace(parsed, hessian_hartree_per_bohr2=hessian)
    return _point_result_from_parsed(
        point,
        parsed,
        f"xTB {' '.join(options)}",
        point.coordinates_angstrom,
        execution={
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "xtb_native_derivative_files",
            "derivative_transport_lossless_float64": True,
        },
    )


def _run_pyscf_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_pyscf import (
        hessian_input_from_pyscf_output,
        read_pyscf_output_geometry,
        read_pyscf_point_result,
        run_pyscf_job,
        write_pyscf_hessian_input,
        write_pyscf_point_input,
    )

    method = spec.method or spec.route or "HF"
    basis = spec.basis or "sto-3g"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    if spec.properties:
        requested = set(spec.properties)
    else:
        requested = {"energy"}
        if requested_mode not in {"energy", "numerical", "link-numerical"}:
            requested.add("gradient")
    unsupported = requested - {"energy", "gradient", "hessian"}
    if unsupported:
        raise ValueError(
            "unsupported PySCF point properties: " + ", ".join(sorted(unsupported))
        )
    include_hessian = "hessian" in requested
    include_gradient = "gradient" in requested or include_hessian
    writer = write_pyscf_hessian_input if include_hessian else write_pyscf_point_input
    writer_options = {
        "method": method,
        "reference": spec.reference,
        "basis": basis,
        "charge": spec.charge,
        "multiplicity": spec.multiplicity,
        "dispersion_contract": spec.dispersion_contract,
        "accelerator": (
            str(spec.resolution.get("accelerator", "density_fitting"))
            if isinstance(spec.resolution, Mapping)
            else "density_fitting"
        ),
    }
    if not include_hessian:
        writer_options["_include_gradient"] = include_gradient
    input_path = writer(
        point_dir / "pyscf_job.py",
        atoms,
        point.coordinates_angstrom,
        **writer_options,
    )
    run = run_pyscf_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=max(1, int(spec.processors)),
            memory_per_job_gb=float(spec.memory_gb or 1),
            concurrent_jobs=1,
        ),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_pyscf_point_result(run.output_path)
    if include_hessian:
        parsed = replace(
            parsed,
            hessian_hartree_per_bohr2=hessian_input_from_pyscf_output(
                run.output_path,
                input_path=input_path,
            ).cartesian_hessian,
        )
    returned = read_pyscf_output_geometry(run.output_path).coordinates_angstrom
    dispersion_label = (
        "" if spec.dispersion_contract is None else f" + {spec.dispersion_contract}"
    )
    return _point_result_from_parsed(
        point,
        parsed,
        f"PySCF {method}/{basis}{dispersion_label}",
        returned,
        execution={
            "energy_evaluations": 1,
            "gradient_evaluations": int(include_gradient),
            "hessian_evaluations": int(include_hessian),
            "derivative_transport_contract": FULL_PRECISION_DERIVATIVE_ARTIFACT_SCHEMA,
            "derivative_transport": "pyscf_structured_float64",
            "derivative_transport_lossless_float64": True,
        },
    )


def _run_et_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_et import evaluate_et_point

    method = spec.method or "HF"
    basis = spec.basis or "cc-pVDZ"
    requested_mode = str(spec.gradient_mode).strip().casefold()
    # LINK owns numerical differentiation in optimizer coordinates.  Asking
    # for ``numerical`` therefore makes eT an energy provider; the historical
    # Cartesian 3N finite difference remains available only as the explicit
    # ``cartesian-numerical`` diagnostic mode.
    et_mode = {
        "numerical": "energy",
        "link-numerical": "energy",
        "cartesian-numerical": "numerical",
    }.get(requested_mode, requested_mode)
    et_env = dict(spec.env or {})
    et_threads = str(max(1, int(spec.processors)))
    et_env.update(
        {
            "OMP_NUM_THREADS": et_threads,
            "MKL_NUM_THREADS": et_threads,
            "OPENBLAS_NUM_THREADS": et_threads,
        }
    )
    parsed = evaluate_et_point(
        point_dir,
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        executable=spec.executable,
        timeout=spec.timeout,
        env=et_env,
        extra_args=spec.extra_args,
        gradient_mode=et_mode,
        numerical_step_bohr=spec.numerical_gradient_step_bohr,
        numerical_stencil=spec.numerical_gradient_stencil,
        workers=spec.processors,
        memory_gb=spec.memory_gb or 8,
        electronic_state=spec.electronic_state,
        excited_states=spec.excited_states,
        state_spin=spec.state_spin,
        freeze_core=spec.freeze_core,
        capture_state_vectors=(
            spec.electronic_state > 0 and str(spec.state_tracking).strip().casefold() == "apoc"
        ),
    )
    state_label = (
        "S0"
        if spec.electronic_state == 0
        else f"{spec.state_spin[0].upper()}{spec.electronic_state}"
    )
    return _point_result_from_parsed(
        point,
        parsed,
        f"eT {method}/{basis} state={state_label} gradient={requested_mode}",
        point.coordinates_angstrom,
        # Preserve the backend execution metadata.  In particular, eT stores
        # the APOC state-manifold archive here; dropping it makes every
        # excited-state LINK point fail before state following can start.
        execution=parsed.execution,
    )


def _copy_cfour_genbas(point_dir: Path, env: dict[str, str] | None) -> None:
    if (point_dir / "GENBAS").exists():
        return
    cfour_home = None
    if env and env.get("MATRIX_CFOUR_HOME"):
        cfour_home = env["MATRIX_CFOUR_HOME"]
    elif os.environ.get("MATRIX_CFOUR_HOME"):
        cfour_home = os.environ["MATRIX_CFOUR_HOME"]
    if not cfour_home:
        return
    source = Path(cfour_home) / "basis" / "GENBAS"
    if source.is_file():
        shutil.copy2(source, point_dir / "GENBAS")


def _point_result_from_parsed(
    point: ScanPoint,
    parsed: object,
    source: str,
    backend_coordinates_angstrom: np.ndarray | None = None,
    execution: Mapping[str, object] | None = None,
    *,
    derivatives_in_point_frame: bool = False,
) -> PointEvaluationResult:
    normal = bool(getattr(parsed, "normal_termination", True))
    if not normal:
        raise RuntimeError(f"{source} did not terminate normally")
    raw_gradient = getattr(parsed, "gradient_hartree_per_bohr", None)
    gradient = None if raw_gradient is None else np.asarray(raw_gradient, dtype=float)
    hessian = getattr(parsed, "hessian_hartree_per_bohr2", None)
    returned = None
    alignment_rms = 0.0
    if backend_coordinates_angstrom is not None:
        returned = np.asarray(backend_coordinates_angstrom, dtype=float)
        if returned.shape != point.coordinates_angstrom.shape:
            raise RuntimeError(f"{source} returned incompatible Cartesian coordinates")
        rotation = kabsch_rotation(returned, point.coordinates_angstrom)
        if gradient is not None and not derivatives_in_point_frame:
            gradient, hessian = rotate_cartesian_derivatives(gradient, rotation, hessian)
        displacement = aligned_cartesian_displacement(point.coordinates_angstrom, returned)
        alignment_rms = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
    return PointEvaluationResult(
        point_index=point.index,
        displacement=point.displacement,
        energy_hartree=float(getattr(parsed, "energy_hartree")),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        backend_coordinates_angstrom=returned,
        frame_alignment_rms_angstrom=alignment_rms,
        source=source,
        execution={**dict(getattr(parsed, "execution", {})), **dict(execution or {})},
    )


def _format_command(template: str, **values: object) -> list[str]:
    formatted = template.format(**{key: str(value) for key, value in values.items()})
    return shlex.split(formatted)


def _taylor_design(displacements: np.ndarray, order: int) -> np.ndarray:
    columns = [np.ones_like(displacements, dtype=float)]
    for power in range(1, order + 1):
        columns.append(np.power(displacements, power) / float(math.factorial(power)))
    return np.column_stack(columns)


def _least_squares_derivatives(
    design: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    coeff, residuals, rank, _singular = np.linalg.lstsq(design, values, rcond=None)
    residual_norm = float(np.sqrt(np.sum(residuals))) if residuals.size else 0.0
    return coeff, residual_norm, int(rank)


def _array_derivatives(
    results: Sequence[PointEvaluationResult],
    design: np.ndarray,
    *,
    attr: str,
) -> tuple[np.ndarray, ...]:
    arrays = [getattr(item, attr) for item in results]
    if any(array is None for array in arrays):
        return ()
    flat = np.vstack([np.asarray(array, dtype=float).reshape(-1) for array in arrays])
    coeff, _residual_norm, _rank = _least_squares_derivatives(design, flat)
    shape = np.asarray(arrays[0], dtype=float).shape
    return tuple(np.asarray(row, dtype=float).reshape(shape) for row in coeff[1:])
