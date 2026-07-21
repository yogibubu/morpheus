"""ORACLE-owned redundant primitive coordinates and Wilson B matrix.

The primitive layer is deliberately redundant and geometry local.  SMITH is a
consumer of this layer: it selects, combines and symmetry adapts these rows to
construct SONIC coordinates, but it does not own molecular reperception.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .structural_corrections import pseudo_bond_pairs


MATRIX_XYZ_PRIMITIVES_SCHEMA = "matrix.xyz.primitives.v1"
LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA = "oracle.xyz.primitives.v1"
LINEAR_ANGLE_DEGREES = 165.0


@dataclass(frozen=True)
class Primitive:
    """One ORACLE primitive; atom indices are zero based in memory."""

    kind: str
    atoms: tuple[int, ...]
    mode: int = 0
    ref: tuple[int, ...] = ()

    @property
    def function(self) -> str:
        return {
            "bond": "R",
            "hbond_dist": "R",
            "pseudo_bond": "R",
            "angle": "A",
            "linear_bend": "L",
            "dihedral": "D",
            "out_of_plane": "U",
            "frag_dist": "R",
            "frag_atom_dist": "R",
            "frag_trans": "FTRANS",
            "frag_rot": "FROT",
        }.get(self.kind, self.kind.upper())

    @property
    def label(self) -> str:
        atoms = ",".join(str(atom + 1) for atom in self.atoms)
        if self.kind == "linear_bend":
            return f"L({atoms},0,{self.mode})"
        return f"{self.function}({atoms})"


@dataclass(frozen=True)
class PrimitiveCoordinateContract:
    schema: str
    primitives: tuple[Primitive, ...]
    reference_values: tuple[float, ...]
    reference_geometry_sha256: str
    b_matrix_sha256: str
    b_matrix_rank: int
    b_matrix_columns: int


def build_primitives(
    discrete_graph,
    coords: np.ndarray,
    linear_threshold: float = np.deg2rad(LINEAR_ANGLE_DEGREES),
    *,
    include_pseudo_bonds: bool = True,
    pair_selector_threshold: float = 1.0e-3,
) -> list[Primitive]:
    """Build the redundant local primitive set from frozen ORACLE topology."""
    xyz = np.asarray(coords, dtype=float)
    n = int(discrete_graph.natoms)
    bonds = [Primitive("bond", tuple(sorted((i, j)))) for i, j in discrete_graph.bonds]
    pseudo: list[Primitive] = []
    if include_pseudo_bonds:
        for pair, source in pseudo_bond_pairs(
            tuple(int(value) for value in discrete_graph.Z),
            xyz,
            discrete_graph.bonds,
            selector_threshold=pair_selector_threshold,
        ):
            kind = "hbond_dist" if "HBOND" in source else "pseudo_bond"
            pseudo.append(Primitive(kind, tuple(sorted(pair)), ref=()))

    angles: list[Primitive] = []
    linears: list[Primitive] = []
    linear_keys: set[tuple[int, int, int]] = set()
    for center in range(n):
        neighbors = sorted(discrete_graph.adjacency[center])
        for left, right in combinations(neighbors, 2):
            atoms = (left, center, right)
            if angle(*atoms, xyz) >= linear_threshold:
                linear_keys.add(_angle_key(atoms))
                linears.extend(
                    (Primitive("linear_bend", atoms, mode=-1), Primitive("linear_bend", atoms, mode=-2))
                )
            else:
                angles.append(Primitive("angle", atoms))

    dihedrals: list[Primitive] = []
    seen: set[tuple[int, int, int, int]] = set()
    for j, k in sorted(tuple(sorted(pair)) for pair in discrete_graph.bonds):
        for i in sorted(set(discrete_graph.adjacency[j]) - {k}):
            for ell in sorted(set(discrete_graph.adjacency[k]) - {j}):
                atoms = (i, j, k, ell)
                canonical = min(atoms, tuple(reversed(atoms)))
                if canonical in seen:
                    continue
                seen.add(canonical)
                if _angle_key((i, j, k)) in linear_keys or _angle_key((j, k, ell)) in linear_keys:
                    continue
                dihedrals.append(Primitive("dihedral", canonical))

    out_of_planes: list[Primitive] = []
    for center in range(n):
        neighbors = sorted(discrete_graph.adjacency[center])
        for n1, n2, n3 in combinations(neighbors, 3):
            out_of_planes.append(Primitive("out_of_plane", (center, n1, n2, n3)))
    return bonds + pseudo + angles + linears + dihedrals + out_of_planes


def eval_primitive(primitive: Primitive, coords: np.ndarray) -> float:
    xyz = np.asarray(coords, dtype=float)
    if primitive.kind in {"bond", "hbond_dist", "pseudo_bond"}:
        i, j = primitive.atoms
        return float(np.linalg.norm(xyz[i] - xyz[j]))
    if primitive.kind == "angle":
        return float(angle(*primitive.atoms, xyz))
    if primitive.kind == "dihedral":
        return float(dihedral(*primitive.atoms, xyz))
    if primitive.kind == "out_of_plane":
        return float(out_of_plane(*primitive.atoms, xyz))
    if primitive.kind == "linear_bend":
        first, second = linear_components(*primitive.atoms, xyz)
        return float(first if primitive.mode == -1 else second)
    if primitive.kind in {"frag_dist", "frag_atom_dist", "frag_trans", "frag_rot"}:
        return _eval_fragment_primitive(primitive, xyz)
    raise ValueError(f"unknown primitive kind: {primitive.kind}")


def eval_primitives(primitives: Iterable[Primitive], coords: np.ndarray) -> np.ndarray:
    return np.asarray([eval_primitive(primitive, coords) for primitive in primitives], dtype=float)


def grad_primitive(primitive: Primitive, coords: np.ndarray, fd_step: float = 1.0e-4) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float)
    if primitive.kind in {"bond", "hbond_dist", "pseudo_bond"}:
        return _bond_grad(*primitive.atoms, xyz)
    if primitive.kind == "angle":
        return _angle_grad(*primitive.atoms, xyz)
    if primitive.kind == "dihedral":
        return _dihedral_grad(*primitive.atoms, xyz)
    if primitive.kind == "out_of_plane":
        return _out_of_plane_grad(*primitive.atoms, xyz)
    if primitive.kind == "linear_bend":
        return _linear_grad(*primitive.atoms, xyz, mode=primitive.mode)
    # Fragment exponential-map rows retain their numerical derivative.
    return _finite_difference_gradient(lambda value: eval_primitive(primitive, value), xyz, fd_step)


def primitive_b_matrix(
    primitives: Sequence[Primitive], coords: np.ndarray, *, fd_step: float = 1.0e-5
) -> np.ndarray:
    """Evaluate the Wilson matrix associated with an ORACLE primitive set."""
    xyz = np.asarray(coords, dtype=float)
    rows = [grad_primitive(primitive, xyz, fd_step=fd_step).reshape(-1) for primitive in primitives]
    return np.vstack(rows) if rows else np.zeros((0, 3 * xyz.shape[0]), dtype=float)


def build_primitive_contract(discrete_graph, coords: np.ndarray) -> PrimitiveCoordinateContract:
    xyz = np.asarray(coords, dtype=float)
    primitives = tuple(build_primitives(discrete_graph, xyz))
    values = tuple(float(value) for value in eval_primitives(primitives, xyz))
    b_matrix = primitive_b_matrix(primitives, xyz)
    return PrimitiveCoordinateContract(
        schema=MATRIX_XYZ_PRIMITIVES_SCHEMA,
        primitives=primitives,
        reference_values=values,
        reference_geometry_sha256=_array_sha256(xyz),
        b_matrix_sha256=_array_sha256(b_matrix),
        b_matrix_rank=int(np.linalg.matrix_rank(b_matrix, tol=1.0e-10)),
        b_matrix_columns=int(b_matrix.shape[1]),
    )


def primitive_contract_section_lines(contract: PrimitiveCoordinateContract) -> list[str]:
    lines = [
        f"SCHEMA {contract.schema}",
        "OWNER ORACLE",
        "INDEXING ATOMS=ONE_BASED",
        "B_MATRIX_EVALUATION REFERENCE_GEOMETRY",
        "B_MATRIX_DERIVATIVE ANALYTIC_LOCAL_NUMERICAL_FRAGMENT",
        "B_MATRIX_FD_STEP_ANGSTROM 1e-5",
        f"REFERENCE_GEOMETRY_SHA256 {contract.reference_geometry_sha256}",
        f"PRIMITIVE_COUNT {len(contract.primitives)}",
        f"B_MATRIX_ROWS {len(contract.primitives)}",
        f"B_MATRIX_COLUMNS {contract.b_matrix_columns}",
        f"B_MATRIX_RANK {contract.b_matrix_rank}",
        f"B_MATRIX_SHA256 {contract.b_matrix_sha256}",
        "[DEFINITIONS]",
        "COLUMNS ID KIND FUNCTION ATOMS MODE REF VALUE",
    ]
    for index, (primitive, value) in enumerate(
        zip(contract.primitives, contract.reference_values, strict=True), start=1
    ):
        atoms = ",".join(str(atom + 1) for atom in primitive.atoms)
        ref = ",".join(str(atom + 1) for atom in primitive.ref) or "NONE"
        lines.append(
            f"P{index:04d} {primitive.kind} {primitive.function} {atoms} "
            f"{primitive.mode} {ref} {value:.16g}"
        )
    return lines


def write_primitive_contract(path: Path, contract: PrimitiveCoordinateContract) -> None:
    replace_section(Path(path), "PRIMITIVES", primitive_contract_section_lines(contract))


def read_primitive_contract(path: Path) -> PrimitiveCoordinateContract:
    content = section_content(read_sectioned_lines(Path(path)), "PRIMITIVES")
    if not content:
        raise ValueError(f"missing #PRIMITIVES section in {Path(path)}")
    metadata: dict[str, str] = {}
    primitives: list[Primitive] = []
    values: list[float] = []
    in_definitions = False
    for raw in content:
        text = raw.strip()
        if text == "[DEFINITIONS]":
            in_definitions = True
            continue
        if not text or text.startswith("COLUMNS "):
            continue
        if not in_definitions:
            fields = text.split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
            continue
        fields = text.split()
        if len(fields) != 7 or not fields[0].startswith("P"):
            raise ValueError(f"invalid ORACLE primitive record: {text}")
        atoms = tuple(int(value) - 1 for value in fields[3].split(","))
        ref = () if fields[5] == "NONE" else tuple(int(value) - 1 for value in fields[5].split(","))
        primitives.append(Primitive(fields[1], atoms, int(fields[4]), ref))
        values.append(float(fields[6]))
    schema = metadata.get("SCHEMA", "")
    if schema not in {MATRIX_XYZ_PRIMITIVES_SCHEMA, LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA}:
        raise ValueError(f"unsupported ORACLE primitive schema: {schema or 'missing'}")
    if int(metadata.get("PRIMITIVE_COUNT", "-1")) != len(primitives):
        raise ValueError("#PRIMITIVES count does not match its definitions")
    return PrimitiveCoordinateContract(
        schema=schema,
        primitives=tuple(primitives),
        reference_values=tuple(values),
        reference_geometry_sha256=metadata.get("REFERENCE_GEOMETRY_SHA256", ""),
        b_matrix_sha256=metadata.get("B_MATRIX_SHA256", ""),
        b_matrix_rank=int(metadata.get("B_MATRIX_RANK", "0")),
        b_matrix_columns=int(metadata.get("B_MATRIX_COLUMNS", "0")),
    )


def validate_primitive_contract(
    contract: PrimitiveCoordinateContract, coords: np.ndarray, *, atol: float = 2.0e-8
) -> None:
    xyz = np.asarray(coords, dtype=float)
    if contract.reference_geometry_sha256 and contract.reference_geometry_sha256 != _array_sha256(xyz):
        raise ValueError("#PRIMITIVES reference geometry does not match the Cartesian block")
    values = eval_primitives(contract.primitives, xyz)
    if not np.allclose(values, contract.reference_values, atol=atol, rtol=0.0):
        raise ValueError("#PRIMITIVES reference values do not match the Cartesian block")
    b_matrix = primitive_b_matrix(contract.primitives, xyz)
    if contract.b_matrix_sha256 and contract.b_matrix_sha256 != _array_sha256(b_matrix):
        raise ValueError("#PRIMITIVES Wilson B fingerprint does not match the Cartesian block")


def angle(i: int, j: int, k: int, coords: np.ndarray) -> float:
    u = _unit(coords[i] - coords[j])
    v = _unit(coords[k] - coords[j])
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def dihedral(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> float:
    b1 = coords[i] - coords[j]
    b2 = coords[k] - coords[j]
    b3 = coords[ell] - coords[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return float(np.arctan2(np.dot(np.cross(n1, n2), _unit(b2)), np.dot(n1, n2)))


def out_of_plane(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> float:
    vector = coords[i] - coords[j]
    normal = np.cross(coords[k] - coords[j], coords[ell] - coords[j])
    denominator = np.linalg.norm(vector) * np.linalg.norm(normal)
    if denominator < 1.0e-12:
        return 0.0
    return float(np.arcsin(np.clip(np.dot(vector, normal) / denominator, -1.0, 1.0)))


def linear_components(i: int, j: int, k: int, coords: np.ndarray) -> tuple[float, float]:
    u = _unit(coords[i] - coords[j])
    v = _unit(coords[k] - coords[j])
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, u))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(u, axis))
    second = _unit(np.cross(u, first))
    bending = v + u
    return float(np.dot(bending, first)), float(np.dot(bending, second))


def _eval_fragment_primitive(primitive: Primitive, coords: np.ndarray) -> float:
    fragment = np.asarray(primitive.atoms, dtype=int)
    reference = np.asarray(primitive.ref, dtype=int)
    delta = coords[fragment].mean(axis=0) - coords[reference].mean(axis=0)
    if primitive.kind == "frag_trans":
        return float(delta[primitive.mode])
    if primitive.kind in {"frag_dist", "frag_atom_dist"}:
        return float(np.linalg.norm(delta))
    def frame(indices: np.ndarray) -> np.ndarray:
        centered = coords[indices] - coords[indices].mean(axis=0)
        inertia = np.zeros((3, 3), dtype=float)
        for vector in centered:
            inertia += np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector)
        values, vectors = np.linalg.eigh(inertia)
        result = vectors[:, np.argsort(values)]
        if np.linalg.det(result) < 0.0:
            result[:, -1] *= -1.0
        return result

    rotation = frame(reference).T @ frame(fragment)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                0.25 / scale,
                (rotation[2, 1] - rotation[1, 2]) * scale,
                (rotation[0, 2] - rotation[2, 0]) * scale,
                (rotation[1, 0] - rotation[0, 1]) * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    rotation_vector = (
        np.zeros(3) if norm < 1.0e-12 else vector / norm * (2.0 * np.arctan2(norm, quaternion[0]))
    )
    return float(rotation_vector[primitive.mode])


def _bond_grad(i: int, j: int, coords: np.ndarray) -> np.ndarray:
    delta = coords[i] - coords[j]
    distance = np.linalg.norm(delta)
    gradient = np.zeros_like(coords)
    if distance > 1.0e-12:
        gradient[i], gradient[j] = delta / distance, -delta / distance
    return gradient


def _angle_grad(i: int, j: int, k: int, coords: np.ndarray) -> np.ndarray:
    left, right = coords[i] - coords[j], coords[k] - coords[j]
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    gradient = np.zeros_like(coords)
    if left_norm < 1.0e-12 or right_norm < 1.0e-12:
        return gradient
    u, v = left / left_norm, right / right_norm
    cosine = np.clip(np.dot(u, v), -1.0, 1.0)
    sine = np.sqrt(max(1.0 - cosine * cosine, 1.0e-16))
    gradient[i] = (-v + cosine * u) / (sine * left_norm)
    gradient[k] = (-u + cosine * v) / (sine * right_norm)
    gradient[j] = -(gradient[i] + gradient[k])
    return gradient


def _dihedral_grad(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> np.ndarray:
    """Algorithmic derivative of the signed dihedral definition."""
    class Dual:
        __slots__ = ("value", "derivative")

        def __init__(self, value, derivative):
            self.value = float(value)
            self.derivative = derivative

        def __add__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(self.value + other.value, self.derivative + other.derivative)

        def __sub__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(self.value - other.value, self.derivative - other.derivative)

        def __mul__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(
                self.value * other.value,
                self.value * other.derivative + other.value * self.derivative,
            )

        def __truediv__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            inverse = 1.0 / other.value
            return Dual(
                self.value * inverse,
                (self.derivative - self.value * other.derivative * inverse) * inverse,
            )

    def dot(left, right):
        return sum((left[index] * right[index] for index in range(3)), start=Dual(0.0, np.zeros(12)))

    def cross(left, right):
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    def unit(vector):
        squared = dot(vector, vector)
        length_value = np.sqrt(squared.value)
        length = Dual(length_value, squared.derivative / (2.0 * length_value))
        return [component / length for component in vector]

    base = coords[[i, j, k, ell]].reshape(-1)
    variables: list[Dual] = []
    for index, value in enumerate(base):
        derivative = np.zeros(12)
        derivative[index] = 1.0
        variables.append(Dual(value, derivative))
    vectors = [variables[start : start + 3] for start in (0, 3, 6, 9)]
    ri, rj, rk, rell = vectors
    b1 = [ri[m] - rj[m] for m in range(3)]
    b2 = [rk[m] - rj[m] for m in range(3)]
    b3 = [rell[m] - rk[m] for m in range(3)]
    n1, n2 = cross(b1, b2), cross(b2, b3)
    x = dot(n1, n2)
    y = dot(cross(n1, n2), unit(b2))
    denominator = x.value * x.value + y.value * y.value
    derivative = (x.value * y.derivative - y.value * x.derivative) / denominator
    local = derivative.reshape(4, 3)
    gradient = np.zeros_like(coords)
    gradient[i], gradient[j], gradient[k], gradient[ell] = local
    return gradient


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )


def _out_of_plane_grad(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> np.ndarray:
    vector = coords[i] - coords[j]
    left, right = coords[k] - coords[j], coords[ell] - coords[j]
    vector_norm = np.linalg.norm(vector)
    normal = np.cross(left, right)
    normal_norm = np.linalg.norm(normal)
    if vector_norm < 1.0e-12 or normal_norm < 1.0e-12:
        return np.zeros_like(coords)
    unit_vector, unit_normal = vector / vector_norm, normal / normal_norm
    sine = np.clip(np.dot(unit_vector, unit_normal), -0.999999, 0.999999)
    denominator = np.sqrt(1.0 - sine * sine)
    if denominator < 1.0e-12:
        return np.zeros_like(coords)
    unit_jacobian = (np.eye(3) - np.outer(unit_vector, unit_vector)) / vector_norm
    normal_jacobian = (np.eye(3) - np.outer(unit_normal, unit_normal)) / normal_norm
    gradient_vector = unit_jacobian @ unit_normal
    gradient_normal = normal_jacobian @ unit_vector
    gradient = np.zeros_like(coords)
    gradient[i] = gradient_vector
    gradient[j] = -gradient_vector + (_cross_matrix(right) - _cross_matrix(left)).T @ gradient_normal
    gradient[k] = (-_cross_matrix(right)).T @ gradient_normal
    gradient[ell] = _cross_matrix(left).T @ gradient_normal
    return gradient / denominator


def _linear_grad(i: int, j: int, k: int, coords: np.ndarray, *, mode: int) -> np.ndarray:
    left, right = coords[i] - coords[j], coords[k] - coords[j]
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm < 1.0e-12 or right_norm < 1.0e-12:
        return np.zeros_like(coords)
    u, v = left / left_norm, right / right_norm
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, u))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    cross = np.cross(u, axis)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1.0e-12:
        return np.zeros_like(coords)
    first = cross / cross_norm
    second = np.cross(u, first)
    bending = v + u
    identity = np.eye(3)
    jacobian_u = (identity - np.outer(u, u)) / left_norm
    jacobian_v = (identity - np.outer(v, v)) / right_norm
    projector = identity / cross_norm - np.outer(cross, cross) / cross_norm**3
    first_u = projector @ (-_cross_matrix(axis))
    # d( u x first )/du = -[first]_x + [u]_x d(first)/du.
    second_u = -_cross_matrix(first) + _cross_matrix(u) @ first_u
    if mode == -1:
        derivative_u, derivative_v = first + first_u.T @ bending, first
    else:
        derivative_u, derivative_v = second + second_u.T @ bending, second
    gradient = np.zeros_like(coords)
    gradient[i] = jacobian_u.T @ derivative_u
    gradient[k] = jacobian_v.T @ derivative_v
    gradient[j] = -(gradient[i] + gradient[k])
    return gradient


def _finite_difference_gradient(function, coords: np.ndarray, step: float) -> np.ndarray:
    gradient = np.zeros_like(coords)
    for atom in range(coords.shape[0]):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            delta = float(function(plus) - function(minus))
            if abs(delta) > np.pi and abs(delta) < 2.0 * np.pi + 1.0e-8:
                delta -= np.sign(delta) * 2.0 * np.pi
            gradient[atom, axis] = delta / (2.0 * step)
    return gradient


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return np.zeros_like(vector) if length < 1.0e-12 else vector / length


def _angle_key(atoms: tuple[int, int, int]) -> tuple[int, int, int]:
    left, center, right = atoms
    return min((left, center, right), (right, center, left))


def _array_sha256(values: np.ndarray) -> str:
    # Primitive rows can differ in their final floating-point bits across
    # Python/NumPy platforms even when they implement the same contract.
    # Quantizing below the validation tolerance makes the fingerprint stable
    # without masking a chemically meaningful change.  Normalize signed zero
    # as well, since it has no geometric meaning but a different byte pattern.
    canonical = np.round(np.asarray(values, dtype="<f8"), decimals=12)
    canonical[canonical == 0.0] = 0.0
    canonical = np.ascontiguousarray(canonical, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()
