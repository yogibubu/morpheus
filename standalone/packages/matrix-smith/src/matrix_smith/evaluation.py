"""Coordinate-value and B-matrix evaluation for frozen SMITH definitions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .bmatrix import SparseBMatrix, SparseBRow
from .contracts import GICForgeContractError
from .models import GICBMatrix, GICDefinition, GICPrimitive
from .numerics import (
    _dual_coordinates,
    _dual_fragment_rotation_values,
    _fragment_relative_frames,
    _linear_fragment_stereographic_values,
    _primitive_value,
    _rotation_vector,
    _sparse_analytic_b_row,
)
from .policy import B_MATRIX_BACKEND


def _cartesian_column_labels(natoms: int) -> tuple[str, ...]:
    axes = ("X", "Y", "Z")
    return tuple(f"{atom}:{axis}" for atom in range(1, natoms + 1) for axis in axes)


def build_gic_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> GICBMatrix:
    """Evaluate the Wilson B matrix for a frozen GIC definition.

    Rows are independent and may be evaluated concurrently.  Thread workers
    share the immutable Cartesian arrays and therefore avoid copying the
    molecular geometry for every coordinate.
    """
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    if definition.backend == "merlino-python-gicforge.v1":
        return _build_merlino_python_b_matrix(definition, coords)
    reference_coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")

    evaluated_rows = _evaluate_native_sparse_gic_rows(
        definition,
        coords,
        reference_coords=reference_coords,
        coordinate_indices=tuple(range(len(definition.gics))),
        parallel_workers=workers,
    )

    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic, sparse_row in zip(definition.gics, evaluated_rows, strict=True):
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def build_primitive_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> GICBMatrix:
    """Evaluate the primitive Wilson rows underlying a frozen GIC definition.

    This is the authoritative primitive derivative path used by consumers
    that must transform a primitive quadratic model through the *physical*
    Cartesian tangent.  It deliberately reuses the same analytic kernels and
    fragment-rotation gauge as :func:`build_gic_b_matrix`.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("primitive B-matrix coordinates must have shape (natoms, 3)")
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.pipeline import b_matrix_analytic

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive)
            for primitive in definition.primitives
        )
        dense_rows = np.asarray(b_matrix_analytic(primitive_basis, coords), dtype=float)
        sparse_rows = tuple(SparseBRow.from_dense(row) for row in dense_rows)
    else:
        reference_coords = (
            np.asarray(definition.reference_coordinates_angstrom, dtype=float)
            if rotation_reference_coordinates is None
            else np.asarray(rotation_reference_coordinates, dtype=float)
        )
        sparse_rows = _evaluate_native_sparse_primitive_rows(
            definition,
            coords,
            reference_coords=reference_coords,
            primitive_ids=tuple(primitive.identifier for primitive in definition.primitives),
            parallel_workers=workers,
        )
        dense_rows = np.asarray([row.to_dense() for row in sparse_rows], dtype=float)
    if not np.all(np.isfinite(dense_rows)):
        raise GICForgeContractError("primitive B matrix contains non-finite values")
    return GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(primitive.identifier for primitive in definition.primitives),
        coordinate_names=tuple(primitive.identifier for primitive in definition.primitives),
        irreps=tuple("" for _primitive in definition.primitives),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(tuple(float(value) for value in row) for row in dense_rows),
        sparse_rows=sparse_rows,
    )


def build_gic_and_primitive_b_matrices(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> tuple[GICBMatrix, GICBMatrix]:
    """Evaluate frozen GIC and primitive rows through one primitive pass.

    Conditioning needs both representations at the same Cartesian sample.
    Evaluating them independently duplicates every primitive derivative.  This
    paired API preserves the public matrix contracts while sharing the sole
    scientifically meaningful primitive evaluation.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if definition.backend == "merlino-python-gicforge.v1":
        return (
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coords,
                rotation_reference_coordinates=rotation_reference_coordinates,
                parallel_workers=workers,
            ),
            build_primitive_b_matrix(
                definition,
                coordinates_angstrom=coords,
                rotation_reference_coordinates=rotation_reference_coordinates,
                parallel_workers=workers,
            ),
        )

    reference_coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_ids = tuple(primitive.identifier for primitive in definition.primitives)
    primitive_rows = _evaluate_native_sparse_primitive_rows(
        definition,
        coords,
        reference_coords=reference_coords,
        primitive_ids=primitive_ids,
        parallel_workers=workers,
    )
    primitive_row_by_id = dict(zip(primitive_ids, primitive_rows, strict=True))
    gic_rows = tuple(
        SparseBRow.combine(
            *tuple(
                (float(coefficient), primitive_row_by_id[primitive_id])
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        for gic in definition.gics
    )
    if any(not np.all(np.isfinite(row.to_dense())) for row in (*gic_rows, *primitive_rows)):
        raise GICForgeContractError("B matrix contains non-finite values")
    columns = _cartesian_column_labels(coords.shape[0])
    gic_matrix = GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=columns,
        rows=tuple(tuple(float(value) for value in row.to_dense()) for row in gic_rows),
        sparse_rows=gic_rows,
    )
    primitive_matrix = GICBMatrix(
        backend=B_MATRIX_BACKEND,
        coordinate_labels=primitive_ids,
        coordinate_names=primitive_ids,
        irreps=tuple("" for _primitive in definition.primitives),
        cartesian_columns=columns,
        rows=tuple(tuple(float(value) for value in row.to_dense()) for row in primitive_rows),
        sparse_rows=primitive_rows,
    )
    return gic_matrix, primitive_matrix


def build_sparse_gic_b_matrix(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    coordinate_indices: tuple[int, ...] | None = None,
    parallel_workers: int = 1,
) -> SparseBMatrix:
    """Evaluate selected Wilson rows without materializing a dense matrix.

    The legacy :func:`build_gic_b_matrix` keeps dense rows for serialization
    compatibility.  Iterative LINK consumers should use this API when they do
    not require that representation.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    indices = (
        tuple(range(len(definition.gics)))
        if coordinate_indices is None
        else tuple(int(index) for index in coordinate_indices)
    )
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("sparse B-matrix subset contains an invalid coordinate index")
    if definition.backend == "merlino-python-gicforge.v1":
        dense = _build_merlino_python_b_matrix(definition, coords)
        rows = tuple(SparseBRow.from_dense(dense.rows[index]) for index in indices)
    else:
        reference_coords = (
            np.asarray(definition.reference_coordinates_angstrom, dtype=float)
            if rotation_reference_coordinates is None
            else np.asarray(rotation_reference_coordinates, dtype=float)
        )
        rows = _evaluate_native_sparse_gic_rows(
            definition,
            coords,
            reference_coords=reference_coords,
            coordinate_indices=indices,
            parallel_workers=int(parallel_workers),
        )
    return SparseBMatrix(
        rows=rows,
        column_count=coords.size,
        row_labels=tuple(definition.gics[index].identifier for index in indices),
        backend=B_MATRIX_BACKEND,
    )


def _evaluate_native_sparse_gic_rows(
    definition: GICDefinition,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    coordinate_indices: tuple[int, ...],
    parallel_workers: int,
) -> tuple[SparseBRow, ...]:
    """Evaluate each required primitive row once, then form frozen GICs."""

    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected = tuple(definition.gics[index] for index in coordinate_indices)
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in selected
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
        )
    )
    for primitive_id in primitive_ids:
        if primitive_id not in primitive_by_id:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC definition"
            )

    row_by_id = dict(
        zip(
            primitive_ids,
            _evaluate_native_sparse_primitive_rows(
                definition,
                coords,
                reference_coords=reference_coords,
                primitive_ids=primitive_ids,
                parallel_workers=workers,
            ),
            strict=True,
        )
    )
    return tuple(
        SparseBRow.combine(
            *tuple(
                (float(coefficient), row_by_id[primitive_id])
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        for gic in selected
    )


def _evaluate_native_sparse_primitive_rows(
    definition: GICDefinition,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    primitive_ids: tuple[str, ...],
    parallel_workers: int,
) -> tuple[SparseBRow, ...]:
    """Evaluate requested primitive rows once in the frozen rotation gauge."""

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    for primitive_id in primitive_ids:
        if primitive_id not in primitive_by_id:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC definition"
            )
    rotation_groups: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ],
        list[GICPrimitive],
    ] = {}
    for primitive_id in primitive_ids:
        primitive = primitive_by_id[primitive_id]
        if primitive.function == "FROT":
            key = (
                tuple(primitive.atoms),
                tuple(primitive.ref_atoms),
                tuple(primitive.frame_atoms),
                tuple(primitive.ref_frame_atoms),
            )
            rotation_groups.setdefault(key, []).append(primitive)
    shared_rotation_rows: dict[str, SparseBRow] = {}
    for group in rotation_groups.values():
        first = group[0]
        rotation_values = _dual_fragment_rotation_values(
            _dual_coordinates(coords),
            coords,
            first.atoms,
            first.ref_atoms,
            frame_atoms=first.frame_atoms,
            ref_frame_atoms=first.ref_frame_atoms,
            reference_coords=reference_coords,
        )
        for primitive in group:
            shared_rotation_rows[primitive.identifier] = SparseBRow.from_dense(
                rotation_values[primitive.mode].der
            )
    ordinary_ids = tuple(
        primitive_id for primitive_id in primitive_ids if primitive_id not in shared_rotation_rows
    )

    def evaluate_primitive(primitive_id: str) -> SparseBRow:
        return _sparse_analytic_b_row(
            primitive_by_id[primitive_id],
            coords,
            reference_coords=reference_coords,
        )

    workers = int(parallel_workers)
    if workers == 1 or len(ordinary_ids) < 2:
        primitive_rows = tuple(evaluate_primitive(primitive_id) for primitive_id in ordinary_ids)
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(ordinary_ids)),
            thread_name_prefix="smith-b-primitive",
        ) as executor:
            primitive_rows = tuple(executor.map(evaluate_primitive, ordinary_ids))
    row_by_id = dict(zip(ordinary_ids, primitive_rows, strict=True))
    row_by_id.update(shared_rotation_rows)
    return tuple(row_by_id[primitive_id] for primitive_id in primitive_ids)


def evaluate_gic_values(
    definition: GICDefinition,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the values of a frozen SONIC/GIC definition."""
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.primitives import eval_primitives

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive)
            for primitive in definition.primitives
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        values_by_id = {
            primitive.identifier: _continuous_primitive_value(
                primitive,
                float(raw_values[index]),
                float(reference_values[index]),
            )
            for index, primitive in enumerate(definition.primitives)
        }
    else:
        values_by_id = _evaluate_native_primitive_values(
            tuple(primitive_by_id.values()),
            coords,
            reference_coords=reference_coords,
            rotation_reference=rotation_reference,
        )
    values: list[float] = []
    for gic in definition.gics:
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        values.append(
            sum(
                float(coefficient) * values_by_id[primitive_id]
                for primitive_id, coefficient in coefficients
            )
        )
    return np.asarray(values, dtype=float)


def evaluate_primitive_values(
    definition: GICDefinition,
    *,
    primitive_ids: tuple[str, ...] | None = None,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate selected frozen primitive values in definition order.

    The returned values use SMITH's frozen-reference gauges.  Stateful
    consumers such as LINK may then transport periodic primitive phases to
    their latest accepted geometry before forming collective GICs.
    """

    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("primitive-value coordinates must have shape (natoms, 3)")
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected_ids = (
        tuple(primitive_by_id)
        if primitive_ids is None
        else tuple(str(identifier) for identifier in primitive_ids)
    )
    missing = tuple(identifier for identifier in selected_ids if identifier not in primitive_by_id)
    if missing:
        raise KeyError(f"unknown frozen primitive identifiers: {','.join(missing)}")
    selected = tuple(primitive_by_id[identifier] for identifier in selected_ids)
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.primitives import eval_primitives

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive) for primitive in selected
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        return {
            primitive.identifier: _continuous_primitive_value(
                primitive,
                float(raw_values[index]),
                float(reference_values[index]),
            )
            for index, primitive in enumerate(selected)
        }
    return _evaluate_native_primitive_values(
        selected,
        coords,
        reference_coords=reference_coords,
        rotation_reference=rotation_reference,
    )


def evaluate_gic_value(
    definition: GICDefinition,
    coordinate_index: int,
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> float:
    """Evaluate one frozen SONIC without evaluating unrelated primitive rows."""
    index = int(coordinate_index)
    if index < 0 or index >= len(definition.gics):
        raise IndexError(f"SONIC coordinate index {coordinate_index} is outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    gic = definition.gics[index]
    components = gic.coefficients or ((gic.primitive_id, 1.0),)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitives = tuple(primitive_by_id[primitive_id] for primitive_id, _coefficient in components)
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.primitives import eval_primitives

        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive) for primitive in primitives
        )
        raw_values = eval_primitives(primitive_basis, coords)
        reference_values = eval_primitives(primitive_basis, reference_coords)
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(raw_values[primitive_index]),
                float(reference_values[primitive_index]),
            )
            for primitive_index, primitive in enumerate(primitives)
        )
    else:
        values = tuple(
            _continuous_primitive_value(
                primitive,
                float(
                    _primitive_value(
                        primitive,
                        coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
                float(
                    _primitive_value(
                        primitive,
                        rotation_reference if primitive.function == "FROT" else reference_coords,
                        reference_coords=(
                            rotation_reference if primitive.function == "FROT" else reference_coords
                        ),
                    )
                ),
            )
            for primitive in primitives
        )
    return float(
        sum(
            float(coefficient) * value
            for (_primitive_id, coefficient), value in zip(components, values, strict=True)
        )
    )


def evaluate_gic_subset(
    definition: GICDefinition,
    coordinate_indices: tuple[int, ...],
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
    parallel_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate selected SONIC values and B rows without building the full B."""
    workers = int(parallel_workers)
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    indices = tuple(int(index) for index in coordinate_indices)
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("SONIC coordinate subset contains an index outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GICForgeContractError("B-matrix coordinates must have shape (natoms, 3)")
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    selected_gics = tuple(definition.gics[index] for index in indices)
    if definition.backend == "merlino-python-gicforge.v1":
        values = np.asarray(
            [
                evaluate_gic_value(
                    definition,
                    index,
                    coordinates_angstrom=coords,
                    rotation_reference_coordinates=rotation_reference,
                )
                for index in indices
            ],
            dtype=float,
        )
    else:
        primitive_ids_for_values = tuple(
            dict.fromkeys(
                primitive_id
                for gic in selected_gics
                for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        values_by_id = _evaluate_native_primitive_values(
            tuple(primitive_by_id[primitive_id] for primitive_id in primitive_ids_for_values),
            coords,
            reference_coords=reference_coords,
            rotation_reference=rotation_reference,
        )
        values = np.asarray(
            [
                sum(
                    float(coefficient) * values_by_id[primitive_id]
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
                for gic in selected_gics
            ],
            dtype=float,
        )
    if definition.backend == "merlino-python-gicforge.v1":
        from matrix_smith.survibfit.pipeline import b_matrix_analytic

        primitive_ids = tuple(
            dict.fromkeys(
                primitive_id
                for gic in selected_gics
                for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
        )
        primitive_basis = tuple(
            _survibfit_primitive_from_gic_primitive(primitive_by_id[primitive_id])
            for primitive_id in primitive_ids
        )
        primitive_rows = np.asarray(b_matrix_analytic(primitive_basis, coords), dtype=float)
        primitive_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
        rows = np.asarray(
            [
                sum(
                    float(coefficient) * primitive_rows[primitive_index[primitive_id]]
                    for primitive_id, coefficient in (
                        gic.coefficients or ((gic.primitive_id, 1.0),)
                    )
                )
                for gic in selected_gics
            ],
            dtype=float,
        )
    else:
        sparse_rows = _evaluate_native_sparse_gic_rows(
            definition,
            coords,
            reference_coords=rotation_reference,
            coordinate_indices=indices,
            parallel_workers=workers,
        )
        rows = np.asarray([row.to_dense() for row in sparse_rows], dtype=float)
    return values, rows


def evaluate_gic_values_subset(
    definition: GICDefinition,
    coordinate_indices: tuple[int, ...],
    *,
    coordinates_angstrom: tuple[tuple[float, float, float], ...] | np.ndarray | None = None,
    rotation_reference_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate selected frozen SONIC values without constructing Wilson rows."""

    indices = tuple(int(index) for index in coordinate_indices)
    if any(index < 0 or index >= len(definition.gics) for index in indices):
        raise IndexError("SONIC coordinate subset contains an index outside the definition")
    coords = (
        np.asarray(definition.reference_coordinates_angstrom, dtype=float)
        if coordinates_angstrom is None
        else np.asarray(coordinates_angstrom, dtype=float)
    )
    reference_coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    rotation_reference = (
        reference_coords
        if rotation_reference_coordinates is None
        else np.asarray(rotation_reference_coordinates, dtype=float)
    )
    if definition.backend == "merlino-python-gicforge.v1":
        return np.asarray(
            [
                evaluate_gic_value(
                    definition,
                    index,
                    coordinates_angstrom=coords,
                    rotation_reference_coordinates=rotation_reference,
                )
                for index in indices
            ],
            dtype=float,
        )
    selected = tuple(definition.gics[index] for index in indices)
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitive_ids = tuple(
        dict.fromkeys(
            primitive_id
            for gic in selected
            for primitive_id, _coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
        )
    )
    primitive_values = _evaluate_native_primitive_values(
        tuple(primitive_by_id[primitive_id] for primitive_id in primitive_ids),
        coords,
        reference_coords=reference_coords,
        rotation_reference=rotation_reference,
    )
    return np.asarray(
        [
            sum(
                float(coefficient) * primitive_values[primitive_id]
                for primitive_id, coefficient in (gic.coefficients or ((gic.primitive_id, 1.0),))
            )
            for gic in selected
        ],
        dtype=float,
    )


def _evaluate_native_primitive_values(
    primitives: tuple[GICPrimitive, ...],
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    rotation_reference: np.ndarray,
) -> dict[str, float]:
    """Evaluate primitive values once, sharing each three-component FROT frame."""

    values: dict[str, float] = {}
    rotation_groups: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ],
        list[GICPrimitive],
    ] = {}
    for primitive in primitives:
        if primitive.function == "FROT":
            key = (
                tuple(primitive.atoms),
                tuple(primitive.ref_atoms),
                tuple(primitive.frame_atoms),
                tuple(primitive.ref_frame_atoms),
            )
            rotation_groups.setdefault(key, []).append(primitive)
            continue
        raw = float(
            _primitive_value(
                primitive,
                coords,
                reference_coords=reference_coords,
            )
        )
        reference = float(
            _primitive_value(
                primitive,
                reference_coords,
                reference_coords=reference_coords,
            )
        )
        values[primitive.identifier] = _continuous_primitive_value(primitive, raw, reference)
    for (
        atoms,
        ref_atoms,
        frame_atoms,
        ref_frame_atoms,
    ), group in rotation_groups.items():
        if len(frame_atoms) == 1:
            current = _linear_fragment_stereographic_values(
                coords,
                atoms,
                frame_atoms[0],
                ref_atoms,
                ref_frame_atoms,
                selection_coords=rotation_reference,
            )
            reference = _linear_fragment_stereographic_values(
                rotation_reference,
                atoms,
                frame_atoms[0],
                ref_atoms,
                ref_frame_atoms,
                selection_coords=rotation_reference,
            )
            for primitive in group:
                values[primitive.identifier] = float(
                    current[primitive.mode] - reference[primitive.mode]
                )
            continue
        current_frag, current_ref = _fragment_relative_frames(
            coords,
            atoms,
            ref_atoms,
            frame_atoms=frame_atoms,
            ref_frame_atoms=ref_frame_atoms,
            gauge_reference_coords=rotation_reference,
        )
        reference_frag, reference_ref = _fragment_relative_frames(
            rotation_reference,
            atoms,
            ref_atoms,
            frame_atoms=frame_atoms,
            ref_frame_atoms=ref_frame_atoms,
            gauge_reference_coords=rotation_reference,
        )
        delta_rotation = (current_frag.T @ current_ref) @ (reference_frag.T @ reference_ref).T
        vector = _rotation_vector(delta_rotation)
        for primitive in group:
            values[primitive.identifier] = float(vector[primitive.mode])
    return values


def _continuous_primitive_value(
    primitive: GICPrimitive,
    value: float,
    reference: float,
) -> float:
    if primitive.function not in {"D", "IMPD", "FROT", "RPCK"}:
        return float(value)
    delta = (float(value) - float(reference) + np.pi) % (2.0 * np.pi) - np.pi
    if primitive.function == "D" and primitive.chart == "PERIODIC_CONTINUATION":
        return float(delta)
    return float(reference) + delta


def _build_merlino_python_b_matrix(
    definition: GICDefinition,
    coords: np.ndarray,
) -> GICBMatrix:
    from matrix_smith.survibfit.pipeline import b_matrix_analytic

    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    primitive_index_by_id = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    primitive_basis = tuple(
        _survibfit_primitive_from_gic_primitive(primitive) for primitive in definition.primitives
    )
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    primitive_sparse_b = tuple(SparseBRow.from_dense(row) for row in primitive_b)
    rows: list[tuple[float, ...]] = []
    sparse_rows: list[SparseBRow] = []
    for gic in definition.gics:
        weighted_rows: list[tuple[float, SparseBRow]] = []
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                raise GICForgeContractError(
                    f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
                )
            primitive_index = primitive_index_by_id[primitive.identifier]
            weighted_rows.append((float(coefficient), primitive_sparse_b[primitive_index]))
        sparse_row = SparseBRow.combine(*tuple(weighted_rows))
        row = sparse_row.to_dense()
        if not np.all(np.isfinite(row)):
            raise GICForgeContractError(f"non-finite B-matrix row for frozen GIC {gic.identifier}")
        sparse_rows.append(sparse_row)
        rows.append(tuple(float(value) for value in row))
    return GICBMatrix(
        backend="merlino-python-survibfit-bmatrix.v1",
        coordinate_labels=tuple(gic.identifier for gic in definition.gics),
        coordinate_names=tuple(gic.name for gic in definition.gics),
        irreps=tuple(gic.irrep for gic in definition.gics),
        cartesian_columns=_cartesian_column_labels(coords.shape[0]),
        rows=tuple(rows),
        sparse_rows=tuple(sparse_rows),
    )


def _survibfit_primitive_from_gic_primitive(primitive: GICPrimitive):
    from matrix_smith.survibfit.primitives import Primitive

    atoms = tuple(atom - 1 for atom in primitive.atoms)
    if primitive.function == "R":
        return Primitive("bond", atoms)
    if primitive.function == "A":
        return Primitive("angle", atoms)
    if primitive.function == "L":
        return Primitive(
            "linear_bend",
            atoms,
            mode=primitive.mode,
            ref=tuple(atom - 1 for atom in primitive.ref_atoms),
        )
    if primitive.function == "D":
        return Primitive("dihedral", atoms)
    if primitive.function == "U":
        return Primitive("out_of_plane", atoms)
    if primitive.function == "H":
        return Primitive("out_of_plane_height", atoms)
    raise GICForgeContractError(
        f"unsupported Merlino Python primitive function for B matrix: {primitive.function}"
    )
