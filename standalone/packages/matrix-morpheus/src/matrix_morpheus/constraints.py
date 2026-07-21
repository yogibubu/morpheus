"""Public constraint utilities for SEfit.

The solver implementation lives in :mod:`matrix_morpheus.fit`; this module is
the stable import surface for MATRIX packages that need Gaussian-style
constraint parsing, values, and B-matrix checks.
"""

from __future__ import annotations

import numpy as np

from matrix_smith.survibfit.primitives import Primitive

from .fit import (
    GICExpressionConstraint,
    GICExpressionDefinition,
    PrimitiveLinearConstraint,
    _combined_primitive_constraint_b_matrix,
    _finite_difference_constraint_b_matrix,
    _fixed_primitives_from_patterns,
    _gic_expression_constraint_targets,
    _gic_expression_constraint_values,
    _gic_expression_constraints_from_patterns,
    _gic_expression_definitions_from_patterns,
    _linear_primitive_constraints_from_patterns,
)


def parse_gaussian_style_constraints(
    records: tuple[str, ...] | list[str],
) -> tuple[
    tuple[Primitive, ...],
    tuple[PrimitiveLinearConstraint, ...],
    tuple[GICExpressionConstraint, ...],
    tuple[GICExpressionDefinition, ...],
]:
    """Parse MATRIX/Gaussian-style primitive, linear, and expression constraints."""
    items = tuple(str(item) for item in records)
    definitions = _gic_expression_definitions_from_patterns(items)
    return (
        _fixed_primitives_from_patterns(items),
        _linear_primitive_constraints_from_patterns(items),
        _gic_expression_constraints_from_patterns(items),
        definitions,
    )


def gic_expression_constraint_values(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _gic_expression_constraint_values(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def gic_expression_constraint_targets(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _gic_expression_constraint_targets(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def combined_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _combined_primitive_constraint_b_matrix(
        coords,
        fixed_primitives,
        linear_constraints,
        expression_constraints=expression_constraints,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=definitions,
    )


def finite_difference_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    definitions: tuple[GICExpressionDefinition, ...] = (),
    step: float = 1.0e-6,
) -> np.ndarray:
    return _finite_difference_constraint_b_matrix(
        coords,
        fixed_primitives,
        fixed_targets,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=definitions,
        step=step,
    )


__all__ = [
    "GICExpressionConstraint",
    "GICExpressionDefinition",
    "PrimitiveLinearConstraint",
    "combined_constraint_b_matrix",
    "finite_difference_constraint_b_matrix",
    "gic_expression_constraint_targets",
    "gic_expression_constraint_values",
    "parse_gaussian_style_constraints",
]
