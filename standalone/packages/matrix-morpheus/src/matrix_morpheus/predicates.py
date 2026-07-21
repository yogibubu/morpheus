from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_smith.survibfit.primitives import Primitive, eval_primitives

from .contracts import QMParameterPredicate
from .legacy.merlino_core.parameters.bdpcs3 import load_bdpcs3_parameters


DEFAULT_INITIAL_GEOMETRY_PREDICATE_SCOPE = (
    "xy_bonds",
    "xh_bonds",
    "heavy_angles",
    "coh_angles",
    "xh_angles",
    "ring_torsions",
)

ALL_INITIAL_GEOMETRY_PREDICATE_SCOPE = (
    "xy_bonds",
    "xh_bonds",
    "heavy_angles",
    "coh_angles",
    "xh_angles",
    "dihedrals",
    "oops",
    "ring_torsions",
    "interfragment_distances",
)

REFERENCE_LEVEL_PREDICATE_SIGMAS = {
    "high": (0.0015, 0.15, 0.30),
    "medium": (0.0030, 0.30, 0.50),
    "low": (0.0060, 0.60, 1.00),
    "kraitchman": (0.0100, 1.00, 2.00),
}


def predicate_sigmas_for_reference_level(level: str) -> tuple[float, float, float]:
    """Return default predicate sigmas for a declared reference-geometry level."""

    key = str(level).strip().lower().replace("-", "_")
    aliases = {
        "accurate": "high",
        "cc": "high",
        "coupled_cluster": "high",
        "best": "high",
        "dft": "medium",
        "hf": "medium",
        "semiempirical": "low",
        "mm": "low",
        "kra": "kraitchman",
    }
    key = aliases.get(key, key)
    if key not in REFERENCE_LEVEL_PREDICATE_SIGMAS:
        known = ", ".join(sorted(REFERENCE_LEVEL_PREDICATE_SIGMAS))
        raise ValueError(f"Unknown MORPHEUS predicate reference level {level!r}; known: {known}")
    return REFERENCE_LEVEL_PREDICATE_SIGMAS[key]


def initial_geometry_predicates(
    atoms: tuple[str, ...] | list[str],
    coords_angstrom: np.ndarray,
    *,
    distance_sigma_angstrom: float,
    angle_sigma_degree: float,
    dihedral_sigma_degree: float,
    scope: Iterable[str] = DEFAULT_INITIAL_GEOMETRY_PREDICATE_SCOPE,
    source: str = "initial_geometry",
) -> tuple[QMParameterPredicate, ...]:
    """Build primitive predicates centered on the initial Cartesian geometry."""

    if distance_sigma_angstrom <= 0.0:
        raise ValueError("distance_sigma_angstrom must be positive")
    if angle_sigma_degree <= 0.0:
        raise ValueError("angle_sigma_degree must be positive")
    if dihedral_sigma_degree <= 0.0:
        raise ValueError("dihedral_sigma_degree must be positive")
    symbols = tuple(str(atom) for atom in atoms)
    coords = np.asarray(coords_angstrom, dtype=float)
    z_numbers = np.array([atomic_number(symbol) or 0 for symbol in symbols], dtype=int)
    _continuous, graph, ringset, _synthons, _aromaticity = build_topology_objects(coords, z_numbers)
    scopes = {str(item).strip().lower() for item in scope if str(item).strip()}
    aliases = {
        "heavy_bonds": "xy_bonds",
        "all_bonds": "bonds",
        "all_angles": "angles",
        "torsions": "dihedrals",
        "dihedral": "dihedrals",
        "out_of_plane": "oops",
        "out_of_planes": "oops",
        "oop": "oops",
        "noncovalent_distances": "interfragment_distances",
        "fragment_distances": "interfragment_distances",
        "inter_fragment_distances": "interfragment_distances",
    }
    scopes = {aliases.get(item, item) for item in scopes}
    if "all" in scopes:
        scopes = set(ALL_INITIAL_GEOMETRY_PREDICATE_SCOPE)
    if "heavy_bonds" in scopes:
        scopes.add("xy_bonds")
    known_scopes = set(ALL_INITIAL_GEOMETRY_PREDICATE_SCOPE) | {
        "bonds",
        "angles",
        "oh_bonds",
    }
    unknown_scopes = sorted(scopes - known_scopes)
    if unknown_scopes:
        known = ", ".join(sorted(known_scopes | {"all"}))
        unknown = ", ".join(unknown_scopes)
        raise ValueError(f"Unknown initial-geometry predicate scope(s): {unknown}; known: {known}")
    predicates: list[QMParameterPredicate] = []

    bonds = tuple(sorted(tuple(sorted((int(i), int(j)))) for i, j in graph.bonds))
    if {"bonds", "xy_bonds", "xh_bonds", "oh_bonds"} & scopes:
        for i, j in bonds:
            zi, zj = int(z_numbers[i]), int(z_numbers[j])
            include = (
                "bonds" in scopes
                or (
                ("xy_bonds" in scopes and zi > 1 and zj > 1)
                or ("xh_bonds" in scopes and 1 in {zi, zj} and max(zi, zj) > 1)
                or ("oh_bonds" in scopes and {zi, zj} == {1, 8})
                )
            )
            if not include:
                continue
            value = float(eval_primitives([Primitive("bond", (i, j))], coords)[0])
            predicates.append(
                QMParameterPredicate(
                    f"R({i + 1},{j + 1})",
                    value,
                    distance_sigma_angstrom,
                    f"{source}_distance",
                )
            )

    adjacency = tuple(
        tuple(sorted(int(item) for item in graph.adjacency[index]))
        for index in range(len(symbols))
    )
    if {"angles", "heavy_angles", "coh_angles", "xh_angles"} & scopes:
        for center, neighbors in enumerate(adjacency):
            for pos, left in enumerate(neighbors):
                for right in neighbors[pos + 1 :]:
                    z_triplet = (int(z_numbers[left]), int(z_numbers[center]), int(z_numbers[right]))
                    include = (
                        "angles" in scopes
                    ) or (
                        "heavy_angles" in scopes and all(z > 1 for z in z_triplet)
                    ) or (
                        "coh_angles" in scopes
                        and z_triplet[1] == 8
                        and {z_triplet[0], z_triplet[2]} == {1, 6}
                    ) or (
                        "xh_angles" in scopes
                        and z_triplet[1] > 1
                        and 1 in {z_triplet[0], z_triplet[2]}
                    )
                    if not include:
                        continue
                    primitive = Primitive("angle", (left, center, right))
                    value = float(np.rad2deg(eval_primitives([primitive], coords)[0]))
                    predicates.append(
                        QMParameterPredicate(
                            f"A({left + 1},{center + 1},{right + 1})",
                            value,
                            angle_sigma_degree,
                            f"{source}_angle",
                        )
                    )

    if "dihedrals" in scopes:
        for center_left, center_right in bonds:
            left_neighbors = [item for item in adjacency[center_left] if item != center_right]
            right_neighbors = [item for item in adjacency[center_right] if item != center_left]
            for left in left_neighbors:
                for right in right_neighbors:
                    torsion = (left, center_left, center_right, right)
                    reverse = tuple(reversed(torsion))
                    if reverse < torsion:
                        torsion = reverse
                    value = float(np.rad2deg(eval_primitives([Primitive("dihedral", torsion)], coords)[0]))
                    predicates.append(
                        QMParameterPredicate(
                            "D(" + ",".join(str(atom + 1) for atom in torsion) + ")",
                            value,
                            dihedral_sigma_degree,
                            f"{source}_dihedral",
                        )
                    )

    if "oops" in scopes:
        for center, neighbors in enumerate(adjacency):
            if len(neighbors) < 3:
                continue
            for pivot in neighbors:
                others = [item for item in neighbors if item != pivot]
                for pos, left in enumerate(others):
                    for right in others[pos + 1 :]:
                        primitive = Primitive("out_of_plane", (pivot, center, left, right))
                        value = float(np.rad2deg(eval_primitives([primitive], coords)[0]))
                        predicates.append(
                            QMParameterPredicate(
                                f"U({pivot + 1},{center + 1},{left + 1},{right + 1})",
                                value,
                                dihedral_sigma_degree,
                                f"{source}_out_of_plane",
                            )
                        )

    if "ring_torsions" in scopes:
        for ring in getattr(ringset, "rings", ()):
            ring_atoms = tuple(int(atom) for atom in getattr(ring, "atoms", ()))
            size = len(ring_atoms)
            if size < 4:
                continue
            for idx in range(size):
                torsion = tuple(ring_atoms[(idx + offset) % size] for offset in range(4))
                value = float(np.rad2deg(eval_primitives([Primitive("dihedral", torsion)], coords)[0]))
                predicates.append(
                    QMParameterPredicate(
                        "D(" + ",".join(str(atom + 1) for atom in torsion) + ")",
                        value,
                        dihedral_sigma_degree,
                        f"{source}_ring_torsion",
                    )
                )

    if "interfragment_distances" in scopes:
        components = _covalent_components(adjacency)
        if len(set(components)) > 1:
            cutoff = load_bdpcs3_parameters().hbond.search_cutoff_ang
            for left in range(len(symbols)):
                for right in range(left + 1, len(symbols)):
                    if components[left] == components[right]:
                        continue
                    if z_numbers[left] == 1 and z_numbers[right] == 1:
                        continue
                    value = float(
                        eval_primitives([Primitive("bond", (left, right))], coords)[0]
                    )
                    if value > cutoff:
                        continue
                    predicates.append(
                        QMParameterPredicate(
                            f"R({left + 1},{right + 1})",
                            value,
                            distance_sigma_angstrom,
                            f"{source}_interfragment_distance",
                        )
                    )

    return _unique_predicates(tuple(predicates))


def _unique_predicates(predicates: tuple[QMParameterPredicate, ...]) -> tuple[QMParameterPredicate, ...]:
    result: list[QMParameterPredicate] = []
    seen: set[tuple[str, str]] = set()
    for predicate in predicates:
        key = (predicate.label_pattern, predicate.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(predicate)
    return tuple(result)


def _covalent_components(adjacency: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    components = [-1] * len(adjacency)
    component = 0
    for start in range(len(adjacency)):
        if components[start] >= 0:
            continue
        stack = [start]
        components[start] = component
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if components[neighbor] >= 0:
                    continue
                components[neighbor] = component
                stack.append(neighbor)
        component += 1
    return tuple(components)
