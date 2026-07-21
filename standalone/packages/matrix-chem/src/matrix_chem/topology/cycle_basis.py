from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GraphLike(Protocol):
    def neighbors(self, atom: int): ...


@dataclass(frozen=True)
class CycleBasisDiagnostics:
    candidate_cycle_count: int
    selected_cycle_count: int
    cycle_rank: int
    allowed_atom_count: int
    allowed_edge_count: int
    excluded_atoms: tuple[int, ...]


def canonical_cycle(cycle: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    atoms = tuple(int(atom) for atom in cycle)
    rotations: list[tuple[int, ...]] = []
    for index in range(len(atoms)):
        rotated = atoms[index:] + atoms[:index]
        rotations.append(rotated)
        rotations.append(tuple(reversed(rotated)))
    return min(rotations)


def elementary_cycle_basis(
    graph: GraphLike,
    *,
    allowed_atoms: set[int],
    ring_max: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], CycleBasisDiagnostics]:
    candidate_cycles: set[tuple[int, ...]] = set()
    for start in sorted(allowed_atoms):
        _dfs_cycles(
            graph,
            start=start,
            current=start,
            visited=(start,),
            seen_cycles=candidate_cycles,
            allowed_atoms=allowed_atoms,
            ring_max=ring_max,
        )
    edge_index = _allowed_edge_index(graph, allowed_atoms)
    cycle_rank = _cycle_rank(allowed_atoms, edge_index)
    selected = _minimum_cycle_basis(candidate_cycles, edge_index, target_rank=cycle_rank)
    all_atoms = _all_graph_atoms(graph)
    diagnostics = CycleBasisDiagnostics(
        candidate_cycle_count=len(candidate_cycles),
        selected_cycle_count=len(selected),
        cycle_rank=cycle_rank,
        allowed_atom_count=len(allowed_atoms),
        allowed_edge_count=len(edge_index),
        excluded_atoms=tuple(sorted(all_atoms - allowed_atoms)),
    )
    return selected, diagnostics


def is_chordless_cycle(graph: GraphLike, cycle: tuple[int, ...]) -> bool:
    cycle_set = set(cycle)
    ring_edges = {
        tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        for index in range(len(cycle))
    }
    for atom in cycle:
        for neighbor in graph.neighbors(atom):
            if neighbor not in cycle_set:
                continue
            edge = tuple(sorted((atom, int(neighbor))))
            if edge not in ring_edges:
                return False
    return True


def _dfs_cycles(
    graph: GraphLike,
    *,
    start: int,
    current: int,
    visited: tuple[int, ...],
    seen_cycles: set[tuple[int, ...]],
    allowed_atoms: set[int],
    ring_max: int | None,
) -> None:
    if ring_max is not None and len(visited) > ring_max:
        return
    for neighbor_raw in graph.neighbors(current):
        neighbor = int(neighbor_raw)
        if neighbor not in allowed_atoms:
            continue
        if neighbor == start and len(visited) >= 3:
            cycle = canonical_cycle(visited)
            if cycle not in seen_cycles and is_chordless_cycle(graph, cycle):
                seen_cycles.add(cycle)
            continue
        if neighbor in visited or neighbor < start:
            continue
        _dfs_cycles(
            graph,
            start=start,
            current=neighbor,
            visited=(*visited, neighbor),
            seen_cycles=seen_cycles,
            allowed_atoms=allowed_atoms,
            ring_max=ring_max,
        )


def _minimum_cycle_basis(
    cycles: set[tuple[int, ...]],
    edge_index: dict[tuple[int, int], int],
    *,
    target_rank: int,
) -> tuple[tuple[int, ...], ...]:
    basis: dict[int, tuple[int, tuple[int, ...]]] = {}
    selected: list[tuple[int, ...]] = []
    for cycle in sorted(cycles, key=lambda item: (len(item), item)):
        vector = _cycle_vector(cycle, edge_index)
        reduced = vector
        for pivot in sorted(basis, reverse=True):
            if reduced & (1 << pivot):
                reduced ^= basis[pivot][0]
        if reduced == 0:
            continue
        pivot = reduced.bit_length() - 1
        basis[pivot] = (reduced, cycle)
        selected.append(cycle)
        if len(selected) >= target_rank:
            break
    return tuple(selected)


def _allowed_edge_index(graph: GraphLike, allowed_atoms: set[int]) -> dict[tuple[int, int], int]:
    edges = set()
    for atom in allowed_atoms:
        for neighbor_raw in graph.neighbors(atom):
            neighbor = int(neighbor_raw)
            if neighbor in allowed_atoms and atom < neighbor:
                edges.add((atom, neighbor))
    return {edge: index for index, edge in enumerate(sorted(edges))}


def _cycle_rank(allowed_atoms: set[int], edge_index: dict[tuple[int, int], int]) -> int:
    if not allowed_atoms:
        return 0
    parent = {atom: atom for atom in allowed_atoms}

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edge_index:
        union(left, right)
    components = {find(atom) for atom in allowed_atoms}
    return max(0, len(edge_index) - len(allowed_atoms) + len(components))


def _cycle_vector(cycle: tuple[int, ...], edge_index: dict[tuple[int, int], int]) -> int:
    vector = 0
    for index in range(len(cycle)):
        edge = tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        vector ^= 1 << edge_index[edge]
    return vector


def _all_graph_atoms(graph: GraphLike) -> set[int]:
    natoms = getattr(graph, "natoms", getattr(graph, "n_atoms", None))
    if natoms is None:
        return set()
    return set(range(int(natoms)))
