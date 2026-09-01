"""Coordinate-model construction and constrained parameterization for MORPHEUS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile

import numpy as np

from matrix_chem import (
    SymmetryThresholds,
    build_topology_objects,
    preprocess_to_enriched_xyz,
    read_enriched_xyz,
    read_molecular_symmetry,
    write_validation_section,
)
from matrix_chem.geometry_io import write_xyz
from matrix_chem.topology.elements import atomic_number as geometry_atomic_number
from matrix_chem.topology.elements import atomic_symbol
from matrix_core import ScientificValidationError
from matrix_link import cartesian_from_internal_jacobian
from matrix_morpheus.numerics import rank_condition
from matrix_smith.definition import write_gicforge_build_sections
from matrix_smith.evaluation import _survibfit_primitive_from_gic_primitive
from matrix_smith.models import (
    FrozenGIC as LinkFrozenGIC,
    GICDefinition as LinkGICDefinition,
    GICPrimitive as LinkGICPrimitive,
)
from matrix_smith.runtime import GICDefinition, GICForge, run_gicforge
from matrix_smith.runtime.gic_symmetry import SYMM_INERTIA_TOL as GIC_SYMM_INERTIA_TOL
from matrix_smith.runtime.gic_symmetry import SYMM_TOL as GIC_SYMM_TOL
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive, build_primitives
from matrix_smith.survibfit.symmetry_detector import (
    orient_coords,
    symmetry_elements_from_geometry,
)
from matrix_smith.survibfit.symmetry_global import primitive_permutation

from .constraints import (
    _combined_primitive_constraint_b_matrix,
    _is_gaussian_gic_definition_record,
    _is_linear_constraint_pattern,
    _legacy_expression_target_split,
    _parse_gaussian_expression_options,
    _parse_gaussian_named_expression,
    _primitive_constraint_key,
    _primitives_from_fixed_pattern,
)
from .contracts import (
    HYDROGEN_PARAMETER_CONSTRAINT,
    ParameterClassConstraint,
    SemiexperimentalFitRequest,
)
from .models import (
    GICExpressionConstraint,
    GICExpressionDefinition,
    MeasurementModel,
    PrimitiveLinearConstraint,
)


@dataclass
class GICForgeSEBackend:
    atoms: tuple[str, ...]
    root: Path
    mode: str = "gicsym"
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
    if outdir is None:
        root = Path(tempfile.mkdtemp(prefix="matrix_se_gicforge_"))
    else:
        root = Path(outdir) / "gicforge_iterations"
        root.mkdir(parents=True, exist_ok=True)
    return GICForgeSEBackend(atoms=atoms, root=root, mode=mode)


def _fragment_aware_gic_definition(
    atoms: tuple[str, ...],
    coords: np.ndarray,
    *,
    workdir: Path,
    symmetrize: bool,
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
    )
    symmetry_diagnostics = smith_definition.symmetry_diagnostics
    provenance = {
        "xyzin": str(xyzin),
        "fragment_mode": smith_definition.fragment_mode,
        "fragment_mode_source": "ORACLE_COORDINATE_ATLAS",
    }
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


def _gic_cartesian_projector(prims: object, u_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    bq = u_matrix.T @ b_matrix_analytic(prims, coords)
    return cartesian_from_internal_jacobian(bq, rcond=1.0e-8)


def _atomic_number(symbol: str) -> int:
    z = geometry_atomic_number(symbol)
    if z is None:
        raise ScientificValidationError(f"Unknown element symbol {symbol}")
    return int(z)
