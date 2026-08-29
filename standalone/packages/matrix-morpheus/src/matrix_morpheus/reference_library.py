from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matrix_chem.geometry_io import read_xyz_atoms_coords, write_xyz
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_smith.survibfit.primitives import (
    Primitive,
    build_primitives,
    eval_primitives,
    grad_primitive,
)
from matrix_smith.survibfit.synthon_similarity import compare_against_library

from .coordinate_model import _atomic_number


DEFAULT_SE_GEOMETRY_LIBRARY = Path(__file__).resolve().parent / "data" / "se_geometries"
DEFAULT_FRAGMENT_KINDS = ("bond", "angle", "dihedral", "out_of_plane")


@dataclass(frozen=True)
class ReferenceGeometry:
    slug: str
    name: str
    atoms: int
    level: str
    path: Path


@dataclass(frozen=True)
class ReferenceMatch:
    rank: int
    slug: str
    name: str
    atoms: int
    level: str
    path: Path
    similarity_combined: float
    combined_distance_neglog: float
    similarity_synthon: float
    distance_synthon: float
    similarity_ring: float
    distance_ring: float
    effective_ring_weight: float
    nrings: int


@dataclass(frozen=True)
class ReferenceLibrarySearchResult:
    query_xyz: Path
    library_root: Path
    matches: tuple[ReferenceMatch, ...]
    skipped: tuple[dict[str, str], ...]
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_xyz": str(self.query_xyz),
            "library_root": str(self.library_root),
            "settings": self.settings,
            "matches": [{**asdict(match), "path": str(match.path)} for match in self.matches],
            "skipped": list(self.skipped),
        }

    def write(self, outdir: Path) -> dict[str, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "reference_matches.json"
        csv_path = out / "reference_matches.csv"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_reference_matches_csv(csv_path, self.matches)
        return {"reference_matches_json": json_path, "reference_matches_csv": csv_path}


@dataclass(frozen=True)
class FragmentTarget:
    query_index: int
    kind: str
    atom_indices: tuple[int, ...]
    atom_symbols: tuple[str, ...]
    signature: tuple[str, ...]
    initial_value: float
    target_value: float
    delta: float
    support: int
    mean_similarity: float
    mean_zeff_distance: float
    source_slugs: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceAssistedGeometryResult:
    query_xyz: Path
    library_root: Path
    atoms: tuple[str, ...]
    initial_coordinates: np.ndarray
    assisted_coordinates: np.ndarray
    targets: tuple[FragmentTarget, ...]
    unmatched: tuple[dict[str, Any], ...]
    iterations: int
    rms_target_residual_initial: float
    rms_target_residual_final: float
    max_cartesian_shift_angstrom: float
    settings: dict[str, Any]

    def write(self, outdir: Path) -> dict[str, Path]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        xyz_path = out / "reference_assisted_geometry.xyz"
        csv_path = out / "fragment_targets.csv"
        unmatched_path = out / "unmatched_fragments.csv"
        json_path = out / "reference_assisted_geometry.json"
        classes_path = out / "multiclasses_fragment_summary.csv"
        write_xyz(
            xyz_path,
            self.atoms,
            self.assisted_coordinates,
            comment="reference-library assisted geometry; unmatched fragments kept at query values",
        )
        write_fragment_targets_csv(csv_path, self.targets)
        write_unmatched_fragments_csv(unmatched_path, self.unmatched)
        write_multiclasses_fragment_summary_csv(classes_path, self.targets)
        payload = {
            "query_xyz": str(self.query_xyz),
            "library_root": str(self.library_root),
            "settings": self.settings,
            "iterations": self.iterations,
            "rms_target_residual_initial": self.rms_target_residual_initial,
            "rms_target_residual_final": self.rms_target_residual_final,
            "max_cartesian_shift_angstrom": self.max_cartesian_shift_angstrom,
            "targets": [
                {
                    **asdict(target),
                    "atom_indices": list(target.atom_indices),
                    "atom_symbols": list(target.atom_symbols),
                    "signature": list(target.signature),
                    "source_slugs": list(target.source_slugs),
                }
                for target in self.targets
            ],
            "unmatched": list(self.unmatched),
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "assisted_geometry_xyz": xyz_path,
            "fragment_targets_csv": csv_path,
            "fragment_targets_json": json_path,
            "unmatched_fragments_csv": unmatched_path,
            "multiclasses_fragment_summary_csv": classes_path,
        }


def load_se_reference_library(library_root: Path | None = None) -> tuple[ReferenceGeometry, ...]:
    root = Path(library_root) if library_root is not None else DEFAULT_SE_GEOMETRY_LIBRARY
    manifest = root / "manifest.csv"
    if manifest.exists():
        entries = _load_manifest_entries(root, manifest)
    else:
        entries = _load_xyz_entries(root)
    if not entries:
        raise ValueError(f"no reference XYZ geometries found in {root}")
    return tuple(entries)


def search_reference_library(
    query_xyz: Path,
    *,
    library_root: Path | None = None,
    top_k: int = 10,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    standardize: bool = True,
    include_ring_comparison: bool = True,
    ring_weight: float = 0.25,
    outdir: Path | None = None,
) -> ReferenceLibrarySearchResult:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    root = Path(library_root) if library_root is not None else DEFAULT_SE_GEOMETRY_LIBRARY
    query = Path(query_xyz)
    references = load_se_reference_library(root)
    by_path = {ref.path.resolve(): ref for ref in references}

    report = compare_against_library(
        query,
        [ref.path for ref in references],
        covariance_mode=covariance_mode,
        regularization=regularization,
        standardize=standardize,
        include_ring_comparison=include_ring_comparison,
        ring_weight=ring_weight,
    )

    matches: list[ReferenceMatch] = []
    for rank, row in enumerate(report["ranking"][:top_k], start=1):
        path = Path(row["xyz"])
        ref = by_path.get(path.resolve())
        if ref is None:
            ref = ReferenceGeometry(
                path.stem, path.stem.replace("_", " "), int(row["natoms"]), "", path
            )
        matches.append(
            ReferenceMatch(
                rank=rank,
                slug=ref.slug,
                name=ref.name,
                atoms=ref.atoms,
                level=ref.level,
                path=ref.path,
                similarity_combined=float(row["similarity_combined"]),
                combined_distance_neglog=float(row["combined_distance_neglog"]),
                similarity_synthon=float(row["similarity_exp_minus_db"]),
                distance_synthon=float(row["bhattacharyya_distance"]),
                similarity_ring=float(row["ring_similarity_exp_minus_db"]),
                distance_ring=float(row["ring_bhattacharyya_distance"]),
                effective_ring_weight=float(row["effective_ring_weight"]),
                nrings=int(row["nrings"]),
            )
        )

    result = ReferenceLibrarySearchResult(
        query_xyz=query,
        library_root=root,
        matches=tuple(matches),
        skipped=tuple(
            {"xyz": str(item.get("xyz", "")), "error": str(item.get("error", ""))}
            for item in report["skipped"]
        ),
        settings={
            "top_k": int(top_k),
            "library_size": int(len(references)),
            "compared": int(report["library_size"]),
            "covariance_mode": covariance_mode,
            "regularization": float(regularization),
            "standardize": bool(standardize),
            "include_ring_comparison": bool(include_ring_comparison),
            "ring_weight": float(ring_weight),
            "feature_names": report.get("feature_names", []),
            "ring_feature_names": report.get("ring_feature_names", []),
        },
    )
    if outdir is not None:
        result.write(outdir)
    return result


def build_reference_assisted_geometry(
    query_xyz: Path,
    *,
    library_root: Path | None = None,
    top_library_matches: int = 25,
    max_fragment_matches: int = 8,
    min_fragment_support: int = 1,
    zeff_threshold: float = 0.08,
    apply_kinds: tuple[str, ...] = DEFAULT_FRAGMENT_KINDS,
    max_bond_delta: float = 0.08,
    max_angle_delta: float = np.deg2rad(15.0),
    max_dihedral_delta: float = np.deg2rad(45.0),
    max_out_of_plane_delta: float = np.deg2rad(30.0),
    tether_weight: float = 0.02,
    max_iterations: int = 25,
    step_limit_angstrom: float = 0.05,
    covariance_mode: str = "full",
    regularization: float = 5.0e-2,
    standardize: bool = True,
    include_ring_comparison: bool = True,
    ring_weight: float = 0.25,
    outdir: Path | None = None,
) -> ReferenceAssistedGeometryResult:
    if top_library_matches < 1:
        raise ValueError("top_library_matches must be positive")
    if max_fragment_matches < 1:
        raise ValueError("max_fragment_matches must be positive")
    if min_fragment_support < 1:
        raise ValueError("min_fragment_support must be positive")
    if zeff_threshold < 0.0:
        raise ValueError("zeff_threshold must be non-negative")
    if tether_weight < 0.0:
        raise ValueError("tether_weight must be non-negative")
    value_thresholds = {
        "bond": float(max_bond_delta),
        "angle": float(max_angle_delta),
        "dihedral": float(max_dihedral_delta),
        "out_of_plane": float(max_out_of_plane_delta),
    }
    for name, value in value_thresholds.items():
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} value threshold must be finite and non-negative")

    root = Path(library_root) if library_root is not None else DEFAULT_SE_GEOMETRY_LIBRARY
    query = Path(query_xyz)
    atoms, coords, _comment = read_xyz_atoms_coords(query)
    atoms_tuple = tuple(str(atom) for atom in atoms)
    apply = tuple(
        _normalize_primitive_kind(kind) for kind in apply_kinds if _normalize_primitive_kind(kind)
    )
    if not apply:
        raise ValueError("apply_kinds did not contain any supported primitive kind")

    ranking = search_reference_library(
        query,
        library_root=root,
        top_k=top_library_matches,
        covariance_mode=covariance_mode,
        regularization=regularization,
        standardize=standardize,
        include_ring_comparison=include_ring_comparison,
        ring_weight=ring_weight,
    )
    reference_weights = {
        match.slug: max(float(match.similarity_combined), 1.0e-12) for match in ranking.matches
    }
    reference_descriptors = []
    for match in ranking.matches:
        try:
            reference_descriptors.extend(_primitive_descriptors(match.path, slug=match.slug))
        except Exception:
            continue
    query_descriptors = tuple(
        item for item in _primitive_descriptors(query, slug="query") if item.primitive.kind in apply
    )
    index: dict[tuple[str, tuple[str, ...]], list[_PrimitiveDescriptor]] = {}
    for descriptor in reference_descriptors:
        if descriptor.primitive.kind not in apply:
            continue
        index.setdefault((descriptor.primitive.kind, descriptor.signature), []).append(descriptor)

    targets: list[FragmentTarget] = []
    unmatched: list[dict[str, Any]] = []
    for descriptor in query_descriptors:
        candidates = []
        for candidate in index.get((descriptor.primitive.kind, descriptor.signature), []):
            zeff_distance = _zeff_signature_distance(
                descriptor.zeff_signature, candidate.zeff_signature
            )
            if zeff_distance is None or zeff_distance > zeff_threshold:
                continue
            value_distance = abs(
                _primitive_delta(descriptor.primitive.kind, candidate.value, descriptor.value)
            )
            if value_distance > value_thresholds[descriptor.primitive.kind]:
                continue
            weight = reference_weights.get(candidate.slug, 1.0) / (1.0 + zeff_distance)
            candidates.append((weight, zeff_distance, candidate))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2].slug))
        selected = candidates[:max_fragment_matches]
        if len(selected) < min_fragment_support:
            unmatched.append(
                _unmatched_fragment_row(descriptor, "no supported fragment in reference library")
            )
            continue
        weights = np.array([item[0] for item in selected], dtype=float)
        values = np.array([item[2].value for item in selected], dtype=float)
        target = _weighted_fragment_target(descriptor.primitive.kind, values, weights)
        mean_similarity = float(
            np.mean([reference_weights.get(item[2].slug, 0.0) for item in selected])
        )
        mean_zeff_distance = float(np.mean([item[1] for item in selected]))
        targets.append(
            FragmentTarget(
                query_index=descriptor.index,
                kind=descriptor.primitive.kind,
                atom_indices=tuple(atom + 1 for atom in descriptor.primitive.atoms),
                atom_symbols=descriptor.atom_symbols,
                signature=descriptor.signature,
                initial_value=float(descriptor.value),
                target_value=float(target),
                delta=float(_primitive_delta(descriptor.primitive.kind, target, descriptor.value)),
                support=len(selected),
                mean_similarity=mean_similarity,
                mean_zeff_distance=mean_zeff_distance,
                source_slugs=tuple(item[2].slug for item in selected),
            )
        )

    assisted, iterations, initial_rms, final_rms = _optimize_fragment_targets(
        coords,
        list(query_descriptors),
        targets,
        tether_weight=tether_weight,
        max_iterations=max_iterations,
        step_limit_angstrom=step_limit_angstrom,
    )
    result = ReferenceAssistedGeometryResult(
        query_xyz=query,
        library_root=root,
        atoms=atoms_tuple,
        initial_coordinates=np.asarray(coords, dtype=float),
        assisted_coordinates=assisted,
        targets=tuple(targets),
        unmatched=tuple(unmatched),
        iterations=iterations,
        rms_target_residual_initial=initial_rms,
        rms_target_residual_final=final_rms,
        max_cartesian_shift_angstrom=float(np.max(np.linalg.norm(assisted - coords, axis=1)))
        if len(coords)
        else 0.0,
        settings={
            "top_library_matches": int(top_library_matches),
            "max_fragment_matches": int(max_fragment_matches),
            "min_fragment_support": int(min_fragment_support),
            "zeff_threshold": float(zeff_threshold),
            "apply_kinds": list(apply),
            "max_bond_delta": float(max_bond_delta),
            "max_angle_delta": float(max_angle_delta),
            "max_dihedral_delta": float(max_dihedral_delta),
            "max_out_of_plane_delta": float(max_out_of_plane_delta),
            "tether_weight": float(tether_weight),
            "max_iterations": int(max_iterations),
            "step_limit_angstrom": float(step_limit_angstrom),
            "library_size": ranking.settings["library_size"],
            "reference_matches": [match.slug for match in ranking.matches],
        },
    )
    if outdir is not None:
        result.write(outdir)
    return result


def write_reference_matches_csv(path: Path, matches: tuple[ReferenceMatch, ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "slug",
                "name",
                "atoms",
                "level",
                "path",
                "similarity_combined",
                "combined_distance_neglog",
                "similarity_synthon",
                "distance_synthon",
                "similarity_ring",
                "distance_ring",
                "effective_ring_weight",
                "nrings",
            ]
        )
        for match in matches:
            writer.writerow(
                [
                    match.rank,
                    match.slug,
                    match.name,
                    match.atoms,
                    match.level,
                    match.path,
                    f"{match.similarity_combined:.12g}",
                    f"{match.combined_distance_neglog:.12g}",
                    f"{match.similarity_synthon:.12g}",
                    f"{match.distance_synthon:.12g}",
                    f"{match.similarity_ring:.12g}",
                    f"{match.distance_ring:.12g}",
                    f"{match.effective_ring_weight:.12g}",
                    match.nrings,
                ]
            )
    return target


def write_fragment_targets_csv(path: Path, targets: tuple[FragmentTarget, ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_index",
                "kind",
                "atom_indices",
                "atom_symbols",
                "signature",
                "initial_value",
                "target_value",
                "delta",
                "support",
                "mean_similarity",
                "mean_zeff_distance",
                "source_slugs",
            ]
        )
        for item in targets:
            writer.writerow(
                [
                    item.query_index,
                    item.kind,
                    "-".join(str(value) for value in item.atom_indices),
                    "-".join(item.atom_symbols),
                    "-".join(item.signature),
                    f"{item.initial_value:.12g}",
                    f"{item.target_value:.12g}",
                    f"{item.delta:.12g}",
                    item.support,
                    f"{item.mean_similarity:.12g}",
                    f"{item.mean_zeff_distance:.12g}",
                    "|".join(item.source_slugs),
                ]
            )
    return target


def write_unmatched_fragments_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "atom_indices", "atom_symbols", "signature", "initial_value", "reason"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return target


def write_multiclasses_fragment_summary_csv(
    path: Path, targets: tuple[FragmentTarget, ...]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    groups = _cluster_fragment_targets_for_classes(targets)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_name",
                "kind",
                "signature",
                "value_cluster",
                "query_fragments",
                "mean_initial",
                "mean_target",
                "mean_delta",
                "mean_support",
                "mean_similarity",
            ]
        )
        for (kind, signature, cluster), items in sorted(groups.items()):
            writer.writerow(
                [
                    _fragment_class_name(kind, signature, cluster),
                    kind,
                    "-".join(signature),
                    cluster,
                    len(items),
                    f"{np.mean([item.initial_value for item in items]):.12g}",
                    f"{np.mean([item.target_value for item in items]):.12g}",
                    f"{np.mean([item.delta for item in items]):.12g}",
                    f"{np.mean([item.support for item in items]):.12g}",
                    f"{np.mean([item.mean_similarity for item in items]):.12g}",
                ]
            )
    return target


def _load_manifest_entries(root: Path, manifest: Path) -> list[ReferenceGeometry]:
    entries: list[ReferenceGeometry] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.is_file():
                continue
            atom_count = _parse_manifest_atom_count(row.get("atoms", ""))
            if atom_count <= 0:
                atom_count = _read_xyz_atom_count(path)
            if atom_count <= 0:
                continue
            entries.append(
                ReferenceGeometry(
                    slug=row.get("slug", path.stem).strip() or path.stem,
                    name=row.get("name", path.stem).strip() or path.stem,
                    atoms=atom_count,
                    level=row.get("level", "").strip(),
                    path=path,
                )
            )
    return entries


def _load_xyz_entries(root: Path) -> list[ReferenceGeometry]:
    entries = []
    for path in sorted(root.glob("**/*.xyz")):
        atom_count = _read_xyz_atom_count(path)
        if atom_count <= 0:
            continue
        entries.append(
            ReferenceGeometry(
                slug=path.stem,
                name=path.stem.replace("_", " "),
                atoms=atom_count,
                level="",
                path=path,
            )
        )
    return entries


def _parse_manifest_atom_count(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


@dataclass(frozen=True)
class _PrimitiveDescriptor:
    index: int
    slug: str
    primitive: Primitive
    atom_symbols: tuple[str, ...]
    signature: tuple[str, ...]
    zeff_signature: tuple[float, ...]
    value: float


def _primitive_descriptors(path: Path, *, slug: str) -> tuple[_PrimitiveDescriptor, ...]:
    atoms, coords, _comment = read_xyz_atoms_coords(Path(path))
    z_numbers = np.array([_atomic_number(atom) for atom in atoms], dtype=int)
    _continuous, graph, _ringset, synthons, _aromaticity = build_topology_objects(
        np.asarray(coords, dtype=float), z_numbers
    )
    primitives = build_primitives(graph, np.asarray(coords, dtype=float))
    values = eval_primitives(primitives, np.asarray(coords, dtype=float))
    zeff = tuple(float(synthons.Zeff(i)) for i in range(len(atoms)))
    rows = []
    for idx, primitive in enumerate(primitives):
        if primitive.kind not in DEFAULT_FRAGMENT_KINDS:
            continue
        signature = _primitive_signature(primitive.kind, primitive.atoms, tuple(atoms))
        if not signature:
            continue
        rows.append(
            _PrimitiveDescriptor(
                index=idx,
                slug=slug,
                primitive=primitive,
                atom_symbols=tuple(str(atoms[i]) for i in primitive.atoms),
                signature=signature,
                zeff_signature=_primitive_zeff_signature(primitive.kind, primitive.atoms, zeff),
                value=float(values[idx]),
            )
        )
    return tuple(rows)


def _primitive_signature(
    kind: str, indices_zero_based: tuple[int, ...], atoms: tuple[str, ...]
) -> tuple[str, ...]:
    symbols = tuple(
        _normalize_symbol(atoms[index]) for index in indices_zero_based if 0 <= index < len(atoms)
    )
    if len(symbols) != len(indices_zero_based):
        return ()
    if kind == "angle" and len(symbols) == 3:
        ends = sorted((symbols[0], symbols[2]))
        return (ends[0], symbols[1], ends[1])
    if kind == "dihedral" and len(symbols) == 4:
        return tuple(sorted((symbols[1], symbols[2])))
    if kind == "out_of_plane" and len(symbols) == 4:
        return (symbols[0],)
    if kind == "bond":
        return tuple(sorted(symbols))
    return symbols


def _primitive_zeff_signature(
    kind: str, indices_zero_based: tuple[int, ...], zeff: tuple[float, ...]
) -> tuple[float, ...]:
    values = tuple(float(zeff[index]) for index in indices_zero_based if 0 <= index < len(zeff))
    if len(values) != len(indices_zero_based):
        return ()
    if kind == "angle" and len(values) == 3:
        ends = sorted((values[0], values[2]))
        return (ends[0], values[1], ends[1])
    if kind == "dihedral" and len(values) == 4:
        return tuple(sorted((values[1], values[2])))
    if kind == "out_of_plane" and values:
        return (values[0],)
    if kind == "bond":
        return tuple(sorted(values))
    return values


def _zeff_signature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return float(max(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _weighted_fragment_target(kind: str, values: np.ndarray, weights: np.ndarray) -> float:
    if kind == "dihedral":
        z = np.sum(weights * np.exp(1j * values))
        return float(np.angle(z))
    return float(np.average(values, weights=weights))


def _primitive_delta(kind: str, target: float, value: float) -> float:
    if kind in {"dihedral", "out_of_plane"}:
        return _wrap_angle(float(target) - float(value))
    return float(target) - float(value)


def _optimize_fragment_targets(
    coords: np.ndarray,
    descriptors: list[_PrimitiveDescriptor],
    targets: list[FragmentTarget],
    *,
    tether_weight: float,
    max_iterations: int,
    step_limit_angstrom: float,
) -> tuple[np.ndarray, int, float, float]:
    if not targets:
        current = np.asarray(coords, dtype=float).copy()
        return current, 0, 0.0, 0.0
    by_index = {target.query_index: target for target in targets}
    selected_descriptors = [
        descriptor for descriptor in descriptors if descriptor.index in by_index
    ]
    selected = [descriptor.primitive for descriptor in selected_descriptors]
    target_values = np.array(
        [by_index[descriptor.index].target_value for descriptor in selected_descriptors],
        dtype=float,
    )
    weights = np.array(
        [
            max(by_index[descriptor.index].mean_similarity, 1.0e-3)
            for descriptor in selected_descriptors
        ],
        dtype=float,
    )
    current = np.asarray(coords, dtype=float).copy()
    initial_rms = _target_rms(selected, target_values, weights, current)
    final_rms = initial_rms
    sqrt_tether = np.sqrt(float(tether_weight))
    for iteration in range(1, max(0, int(max_iterations)) + 1):
        values = eval_primitives(selected, current)
        residual = np.array(
            [
                _primitive_delta(primitive.kind, values[i], target_values[i])
                for i, primitive in enumerate(selected)
            ]
        )
        bmat = np.vstack([grad_primitive(primitive, current).reshape(-1) for primitive in selected])
        sqrt_w = np.sqrt(weights)
        lhs = bmat * sqrt_w[:, None]
        rhs = -residual * sqrt_w
        if tether_weight > 0.0:
            lhs = np.vstack([lhs, sqrt_tether * np.eye(current.size)])
            rhs = np.concatenate(
                [
                    rhs,
                    sqrt_tether
                    * (np.asarray(coords, dtype=float).reshape(-1) - current.reshape(-1)),
                ]
            )
        step, *_ = np.linalg.lstsq(lhs, rhs, rcond=1.0e-10)
        step = step.reshape(current.shape)
        max_atom_step = float(np.max(np.linalg.norm(step, axis=1))) if len(step) else 0.0
        if max_atom_step > step_limit_angstrom > 0.0:
            step *= step_limit_angstrom / max_atom_step
        current = current + step
        final_rms = _target_rms(selected, target_values, weights, current)
        if max_atom_step < 1.0e-6 or final_rms < 1.0e-6:
            return current, iteration, initial_rms, final_rms
    return current, int(max_iterations), initial_rms, final_rms


def _target_rms(
    primitives: list[Primitive], targets: np.ndarray, weights: np.ndarray, coords: np.ndarray
) -> float:
    values = eval_primitives(primitives, coords)
    residual = np.array(
        [
            _primitive_delta(primitive.kind, values[i], targets[i])
            for i, primitive in enumerate(primitives)
        ]
    )
    return float(np.sqrt(np.average(residual * residual, weights=weights)))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _unmatched_fragment_row(descriptor: _PrimitiveDescriptor, reason: str) -> dict[str, Any]:
    return {
        "kind": descriptor.primitive.kind,
        "atom_indices": "-".join(str(index + 1) for index in descriptor.primitive.atoms),
        "atom_symbols": "-".join(descriptor.atom_symbols),
        "signature": "-".join(descriptor.signature),
        "initial_value": f"{descriptor.value:.12g}",
        "reason": reason,
    }


def _cluster_fragment_targets_for_classes(
    targets: tuple[FragmentTarget, ...],
) -> dict[tuple[str, tuple[str, ...], str], list[FragmentTarget]]:
    base: dict[tuple[str, tuple[str, ...]], list[FragmentTarget]] = {}
    for item in targets:
        base.setdefault((item.kind, item.signature), []).append(item)
    grouped: dict[tuple[str, tuple[str, ...], str], list[FragmentTarget]] = {}
    for (kind, signature), items in base.items():
        ordered = sorted(items, key=lambda item: item.initial_value)
        threshold = _class_value_cluster_threshold(kind)
        clusters: list[list[FragmentTarget]] = []
        for item in ordered:
            if not clusters:
                clusters.append([item])
                continue
            center = float(np.mean([member.initial_value for member in clusters[-1]]))
            distance = abs(_primitive_delta(kind, item.initial_value, center))
            if distance <= threshold:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster_index, members in enumerate(clusters, start=1):
            label = _class_value_cluster_label(kind, members, cluster_index, len(clusters))
            grouped[(kind, signature, label)] = members
    return grouped


def _class_value_cluster_threshold(kind: str) -> float:
    if kind == "bond":
        return 0.08
    if kind == "angle":
        return np.deg2rad(15.0)
    if kind == "dihedral":
        return np.deg2rad(45.0)
    if kind == "out_of_plane":
        return np.deg2rad(30.0)
    return float("inf")


def _class_value_cluster_label(
    kind: str, items: list[FragmentTarget], cluster_index: int, nclusters: int
) -> str:
    if nclusters <= 1:
        return "all"
    center = float(np.mean([item.initial_value for item in items]))
    if kind == "bond":
        return f"{center:.2f}A"
    return f"{np.rad2deg(center):.0f}deg"


def _fragment_class_name(kind: str, signature: tuple[str, ...], cluster: str) -> str:
    suffix = "" if cluster == "all" else f"_{cluster}"
    return f"{kind}_{'_'.join(signature)}{suffix}".replace("-", "_").replace(".", "p")


def _normalize_primitive_kind(kind: str) -> str:
    text = str(kind or "").strip().lower().replace("-", "_")
    aliases = {
        "r": "bond",
        "bond": "bond",
        "stretch": "bond",
        "a": "angle",
        "angle": "angle",
        "bend": "angle",
        "d": "dihedral",
        "dihedral": "dihedral",
        "torsion": "dihedral",
        "u": "out_of_plane",
        "oop": "out_of_plane",
        "out_of_plane": "out_of_plane",
    }
    return aliases.get(text, "")


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    return text[:1].upper() + text[1:].lower() if text else ""


def _read_xyz_atom_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return 0
    first = lines[0].strip()
    try:
        return int(first)
    except ValueError:
        return 0
