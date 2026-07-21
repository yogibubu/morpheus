from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SPARSE_ZERO_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class SparseBRow:
    """Sparse Wilson-B row with deterministic sorted entries.

    The row is a runtime object, not part of coordinate construction.  Frozen
    coordinate contracts define which rows must be evaluated; this class stores
    the evaluated derivative row efficiently for iterative optimization and
    refinement steps.
    """

    size: int
    entries: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        size = int(self.size)
        if size < 0:
            raise ValueError("SparseBRow size must be non-negative")
        merged: dict[int, float] = {}
        for raw_index, raw_value in self.entries:
            index = int(raw_index)
            if index < 0 or index >= size:
                raise ValueError(f"SparseBRow index {index} outside row size {size}")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("SparseBRow entries must be finite")
            if abs(value) <= SPARSE_ZERO_TOLERANCE:
                continue
            merged[index] = merged.get(index, 0.0) + value
        canonical = tuple(
            (index, value)
            for index, value in sorted(merged.items())
            if abs(value) > SPARSE_ZERO_TOLERANCE
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "entries", canonical)

    @classmethod
    def zeros(cls, size: int) -> "SparseBRow":
        return cls(size=int(size), entries=())

    @classmethod
    def from_dense(
        cls,
        row: Iterable[float] | np.ndarray,
        *,
        zero_tolerance: float = SPARSE_ZERO_TOLERANCE,
    ) -> "SparseBRow":
        dense = np.asarray(tuple(row) if not isinstance(row, np.ndarray) else row, dtype=float)
        if dense.ndim != 1:
            raise ValueError("SparseBRow.from_dense expects a one-dimensional row")
        entries = tuple(
            (int(index), float(value))
            for index, value in enumerate(dense)
            if abs(float(value)) > float(zero_tolerance)
        )
        return cls(size=int(dense.size), entries=entries)

    @classmethod
    def from_atom_gradients(
        cls,
        natoms: int,
        gradients: Iterable[tuple[int, Iterable[float]]],
        *,
        one_based: bool = True,
    ) -> "SparseBRow":
        """Create a row from atom-indexed 3-vector gradients."""

        size = 3 * int(natoms)
        entries: list[tuple[int, float]] = []
        for atom_index, gradient in gradients:
            atom = int(atom_index) - 1 if one_based else int(atom_index)
            if atom < 0 or atom >= int(natoms):
                raise ValueError(f"atom index {atom_index} outside 1..{natoms}")
            values = tuple(float(value) for value in gradient)
            if len(values) != 3:
                raise ValueError("atom gradient must contain three Cartesian components")
            start = 3 * atom
            entries.extend((start + offset, value) for offset, value in enumerate(values))
        return cls(size=size, entries=tuple(entries))

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.size, dtype=float)
        for index, value in self.entries:
            dense[index] = value
        return dense

    def scaled(self, coefficient: float) -> "SparseBRow":
        factor = float(coefficient)
        if abs(factor) <= SPARSE_ZERO_TOLERANCE:
            return SparseBRow.zeros(self.size)
        return SparseBRow(self.size, tuple((index, factor * value) for index, value in self.entries))

    def add(self, other: "SparseBRow", *, coefficient: float = 1.0) -> "SparseBRow":
        if self.size != other.size:
            raise ValueError("cannot add SparseBRow objects with different sizes")
        return SparseBRow.combine((1.0, self), (float(coefficient), other))

    def dot_dense(self, vector: Iterable[float] | np.ndarray) -> float:
        dense = np.asarray(vector, dtype=float)
        if dense.shape != (self.size,):
            raise ValueError(f"dense vector shape {dense.shape} does not match row size {self.size}")
        return float(sum(value * float(dense[index]) for index, value in self.entries))

    def norm(self) -> float:
        return float(np.sqrt(sum(value * value for _index, value in self.entries)))

    @classmethod
    def combine(cls, *weighted_rows: tuple[float, "SparseBRow"]) -> "SparseBRow":
        rows = tuple((float(coefficient), row) for coefficient, row in weighted_rows)
        if not rows:
            return cls.zeros(0)
        size = rows[0][1].size
        accumulator: dict[int, float] = {}
        for coefficient, row in rows:
            if row.size != size:
                raise ValueError("cannot combine SparseBRow objects with different sizes")
            if abs(coefficient) <= SPARSE_ZERO_TOLERANCE:
                continue
            for index, value in row.entries:
                accumulator[index] = accumulator.get(index, 0.0) + coefficient * value
        return cls(size=size, entries=tuple(accumulator.items()))


@dataclass(frozen=True)
class SparseBMatrix:
    """Sparse Wilson-B matrix with dense conversion for audits."""

    rows: tuple[SparseBRow, ...]
    column_count: int
    row_labels: tuple[str, ...] = ()
    backend: str = "oracle-sparse-bmatrix.v1"

    def __post_init__(self) -> None:
        column_count = int(self.column_count)
        if column_count < 0:
            raise ValueError("SparseBMatrix column_count must be non-negative")
        for row in self.rows:
            if row.size != column_count:
                raise ValueError("SparseBMatrix row size does not match column_count")
        if self.row_labels and len(self.row_labels) != len(self.rows):
            raise ValueError("SparseBMatrix row_labels length must match rows")
        object.__setattr__(self, "column_count", column_count)

    @classmethod
    def from_dense(
        cls,
        matrix: Iterable[Iterable[float]] | np.ndarray,
        *,
        row_labels: tuple[str, ...] = (),
        backend: str = "oracle-sparse-bmatrix.v1",
        zero_tolerance: float = SPARSE_ZERO_TOLERANCE,
    ) -> "SparseBMatrix":
        dense = np.asarray(matrix, dtype=float)
        if dense.ndim != 2:
            raise ValueError("SparseBMatrix.from_dense expects a two-dimensional matrix")
        rows = tuple(
            SparseBRow.from_dense(row, zero_tolerance=zero_tolerance) for row in dense
        )
        return cls(rows=rows, column_count=int(dense.shape[1]), row_labels=row_labels, backend=backend)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def nnz(self) -> int:
        return sum(row.nnz for row in self.rows)

    @property
    def density(self) -> float:
        total = self.row_count * self.column_count
        return 0.0 if total == 0 else float(self.nnz) / float(total)

    def to_dense(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, self.column_count), dtype=float)
        return np.vstack([row.to_dense() for row in self.rows])
