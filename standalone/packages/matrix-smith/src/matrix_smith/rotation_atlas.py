"""Continuous SO(3) charts for frozen SONIC fragment rotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from matrix_core import RotationChart

if TYPE_CHECKING:
    from .definition import GICDefinition


@dataclass(frozen=True)
class FragmentRotationGroup:
    key: tuple[str, ...]
    coordinate_indices: tuple[int, int, int]


class FragmentRotationAtlas:
    """Transport FROT values and B rows across locally rebased SO(3) charts."""

    def __init__(self, definition: "GICDefinition") -> None:
        primitive_by_id = {item.identifier: item for item in definition.primitives}
        grouped: dict[tuple[str, ...], dict[int, int]] = {}
        for index, gic in enumerate(definition.gics):
            coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
            if len(coefficients) != 1 or float(coefficients[0][1]) != 1.0:
                continue
            primitive = primitive_by_id.get(coefficients[0][0])
            if primitive is None or primitive.function != "FROT":
                continue
            key = (
                *primitive.refs,
                *(str(item) for item in primitive.frame_atoms),
                "|",
                *(str(item) for item in primitive.ref_frame_atoms),
            )
            grouped.setdefault(key, {})[int(primitive.mode)] = index
        self.groups = tuple(
            FragmentRotationGroup(key, (modes[0], modes[1], modes[2]))
            for key, modes in grouped.items()
            if set(modes) == {0, 1, 2}
        )
        self.reference_coordinates = np.asarray(
            definition.reference_coordinates_angstrom, dtype=float
        ).copy()
        self._charts = {group.key: RotationChart.identity() for group in self.groups}

    @property
    def active(self) -> bool:
        return bool(self.groups)

    def transform(
        self,
        local_values: np.ndarray,
        local_rows: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = np.asarray(local_values, dtype=float).copy()
        rows = None if local_rows is None else np.asarray(local_rows, dtype=float).copy()
        for group in self.groups:
            indices = list(group.coordinate_indices)
            chart = self._charts[group.key]
            values[indices] = chart.value(values[indices])
            if rows is not None:
                rows[indices, :] = chart.rows(rows[indices, :])
        return values, rows

    def max_local_norm(self, local_values: np.ndarray) -> float:
        values = np.asarray(local_values, dtype=float)
        return max(
            (float(np.linalg.norm(values[list(group.coordinate_indices)])) for group in self.groups),
            default=0.0,
        )

    def rebase(self, local_values: np.ndarray, coordinates_angstrom: np.ndarray) -> None:
        values = np.asarray(local_values, dtype=float)
        for group in self.groups:
            indices = list(group.coordinate_indices)
            self._charts[group.key] = self._charts[group.key].rebase(values[indices])
        self.reference_coordinates = np.asarray(coordinates_angstrom, dtype=float).copy()
