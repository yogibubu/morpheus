from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from collections.abc import Iterable, Sequence

import numpy as np

from matrix_core.geometry_alignment import (
    aligned_cartesian_displacement,
    kabsch_rotation,
    rotate_cartesian_derivatives,
)
from matrix_core.xyzin_geometry import read_xyzin_geometry
from matrix_link import cartesian_from_internal_jacobian


BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
ORACLE_LINK_POINT_RESULT_SCHEMA = "oracle.link.point_result.v1"
ORACLE_LINK_SCAN_MANIFEST_SCHEMA = "oracle.link.scan_manifest.v1"
ORACLE_LINK_FINITE_DIFFERENCE_DERIVATIVES_SCHEMA = (
    "oracle.link.finite_difference_derivatives.v1"
)


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
                "C1_UNSYMMETRIZED"
                if self.retained_group == "C1"
                else "RETAINED_GROUP_ONLY"
            ),
            "pointwise_oracle_symmetry": self.pointwise_oracle_symmetry,
            "pointwise_cartesian_symmetrization": self.pointwise_cartesian_symmetrization,
            "separate_exocyclic_torsions": self.separate_exocyclic_torsions,
        }


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
        object.__setattr__(self, "frame_alignment_rms_angstrom", float(self.frame_alignment_rms_angstrom))


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
    basis: str = ""
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
    properties: tuple[str, ...] = ()
    device: str | None = None
    gradient_mode: str = "analytic"
    numerical_gradient_step_bohr: float = 1.0e-3
    numerical_gradient_stencil: str = "central"
    electronic_state: int = 0
    excited_states: int | None = None
    state_spin: str = "singlet"
    freeze_core: bool = False


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
    from matrix_link import direct_fragment_rigid_tangent

    cartesian_from_q = cartesian_from_internal_jacobian(b_matrix, rcond=rcond)
    fragment_tangent = direct_fragment_rigid_tangent(
        resolved_definition,
        coords,
        b_matrix,
    )
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_q[:, handled_index] = fragment_tangent.cartesian_from_q[
            :, handled_index
        ]
    direction = cartesian_from_q[:, index]
    return CoordinateDirection(
        kind="sonic",
        label=labels[index],
        vector_angstrom=direction,
        source=(
            f"#GIC {target}"
            if definition is None
            else f"PES-exploration SONIC {target}"
        ),
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
    if name == "architect":
        try:
            architect_results = _run_architect_scan_batch(
                spec, geometry.atoms, prepared_points
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed point diagnostics
            architect_results = tuple(
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
        for point, raw_result in zip(prepared_points, architect_results, strict=True):
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
        if point.result_path is not None:
            write_point_result(point.result_path, result)
        write_point_result(point_dir / "point.json", result)
        results.append(result)
    return tuple(results)


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
        payload["gradient_hartree_per_bohr"] = np.asarray(
            result.gradient_hartree_per_bohr, dtype=float
        ).reshape(-1).tolist()
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
        point_group=str(
            payload.get("point_group", point.point_group if point is not None else "")
        ),
        symmetry_projection_status=str(
            payload.get(
                "symmetry_projection_status",
                point.symmetry_projection_status if point is not None else "NOT_ANALYZED",
            )
        ),
        symmetry_projection_max_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_max_displacement_angstrom",
                (
                    point.symmetry_projection_max_displacement_angstrom
                    if point is not None
                    else 0.0
                ),
            )
        ),
        symmetry_projection_rms_displacement_angstrom=float(
            payload.get(
                "symmetry_projection_rms_displacement_angstrom",
                (
                    point.symmetry_projection_rms_displacement_angstrom
                    if point is not None
                    else 0.0
                ),
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


def _coordinate_index(coordinate: str | int, labels: tuple[str, ...], names: tuple[str, ...]) -> int:
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
        "architect": "architect",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported QM scan backend: {name}") from exc


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
    if name == "architect":
        return _run_architect_scan_point(spec, atoms, point)
    raise ValueError(f"unsupported QM scan backend: {name}")


def _run_architect_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
) -> PointEvaluationResult:
    """Evaluate a LINK/SENTINEL point with ARCHITECT instead of a QM code."""

    return _run_architect_scan_batch(spec, atoms, (point,))[0]


def _run_architect_scan_batch(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    points: Sequence[ScanPoint],
) -> tuple[PointEvaluationResult, ...]:
    """Evaluate one LINK/SENTINEL block through the shared ZION batch runtime."""

    from matrix_architect import evaluate_force_field_batch, load_force_field

    if spec.force_field is None:
        raise ValueError("ARCHITECT backend needs a force-field model path")
    field = load_force_field(spec.force_field)
    if tuple(atoms) != field.atoms:
        raise ValueError("LINK geometry atoms differ from the ARCHITECT force field")
    properties = tuple(spec.properties or ("energy", "gradient"))
    evaluations = evaluate_force_field_batch(
        field,
        tuple(point.coordinates_angstrom for point in points),
        properties=properties,
        device=spec.device,
    )
    return tuple(
        PointEvaluationResult(
            point_index=point.index,
            displacement=point.displacement,
            energy_hartree=result.energy_hartree,
            gradient_hartree_per_bohr=result.gradient_hartree_per_bohr,
            hessian_hartree_per_bohr2=result.hessian_hartree_per_bohr2,
            backend_coordinates_angstrom=point.coordinates_angstrom,
            source=f"ARCHITECT {Path(spec.force_field)}",
            execution=dict(result.execution),
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

    route = spec.route or "#p HF/STO-3G Force"
    writer = write_gaussian_oniom_point_input if spec.oniom_high_atoms else write_gaussian_point_input
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
            item for item in (
                f"%NProcShared={int(spec.processors)}" if int(spec.processors) > 1 else "",
                f"%Mem={int(spec.memory_gb)}GB" if spec.memory_gb is not None else "",
            ) if item
        ),
        **writer_kwargs,
    )
    run = run_gaussian_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
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
    return _point_result_from_parsed(point, parsed, f"Gaussian {route}", returned)


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
    low_full = _run_gaussian_scan_point(
        full_low_spec, atoms, point, point_dir / "low_full"
    )
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
    if any(item.energy_hartree is None or item.gradient_hartree_per_bohr is None for item in components):
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
        read_orca_output_geometry,
        read_orca_point_result,
        run_orca_job,
        write_orca_point_input,
    )

    route = spec.route or "HF STO-3G EnGrad"
    input_path = write_orca_point_input(
        point_dir / "orca.inp",
        atoms,
        point.coordinates_angstrom,
        route=route,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    run = run_orca_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_orca_point_result(run.output_path)
    try:
        returned = read_orca_output_geometry(run.output_path).coordinates_angstrom
    except ValueError:
        returned = None
    return _point_result_from_parsed(point, parsed, f"ORCA {route}", returned)


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
    return _point_result_from_parsed(point, parsed, f"Molpro {method}/{basis}", returned)


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
    return _point_result_from_parsed(point, parsed, f"MRCC {method}/{basis}", returned)


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
    return _point_result_from_parsed(point, parsed, f"CFOUR {method}/{basis}", returned)


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
    if "--grad" not in options:
        options.append("--grad")
    if spec.charge and not any(item in options for item in ("--chrg", "--charge")):
        options.extend(("--chrg", str(spec.charge)))
    if spec.multiplicity > 1 and "--uhf" not in options:
        options.extend(("--uhf", str(spec.multiplicity - 1)))
    options.extend(spec.extra_args)
    run = run_xtb_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=tuple(options),
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_xtb_point_result(run.output_path, geometry=input_path)
    return _point_result_from_parsed(
        point,
        parsed,
        f"xTB {' '.join(options)}",
        point.coordinates_angstrom,
    )


def _run_pyscf_scan_point(
    spec: QMScanBackend,
    atoms: tuple[str, ...],
    point: ScanPoint,
    point_dir: Path,
) -> PointEvaluationResult:
    from matrix_pyscf import (
        read_pyscf_output_geometry,
        read_pyscf_point_result,
        run_pyscf_job,
        write_pyscf_point_input,
    )

    method = spec.method or spec.route or "HF"
    basis = spec.basis or "sto-3g"
    input_path = write_pyscf_point_input(
        point_dir / "pyscf_job.py",
        atoms,
        point.coordinates_angstrom,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
    )
    run = run_pyscf_job(
        point_dir,
        executable=spec.executable,
        input_path=input_path,
        timeout=spec.timeout,
        env=spec.env,
        extra_args=spec.extra_args,
    )
    if run.success is not True:
        raise RuntimeError(run.message)
    parsed = read_pyscf_point_result(run.output_path)
    returned = read_pyscf_output_geometry(run.output_path).coordinates_angstrom
    return _point_result_from_parsed(point, parsed, f"PySCF {method}/{basis}", returned)


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
        env=spec.env,
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
) -> PointEvaluationResult:
    normal = bool(getattr(parsed, "normal_termination", True))
    if not normal:
        raise RuntimeError(f"{source} did not terminate normally")
    raw_gradient = getattr(parsed, "gradient_hartree_per_bohr", None)
    gradient = (
        None if raw_gradient is None else np.asarray(raw_gradient, dtype=float)
    )
    hessian = getattr(parsed, "hessian_hartree_per_bohr2", None)
    returned = None
    alignment_rms = 0.0
    if backend_coordinates_angstrom is not None:
        returned = np.asarray(backend_coordinates_angstrom, dtype=float)
        if returned.shape != point.coordinates_angstrom.shape:
            raise RuntimeError(f"{source} returned incompatible Cartesian coordinates")
        rotation = kabsch_rotation(returned, point.coordinates_angstrom)
        if gradient is not None:
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
        execution=dict(getattr(parsed, "execution", {})),
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
