"""Compatibility facade for the ORACLE-owned primitive-coordinate kernel."""

from matrix_chem.primitive_coordinates import (
    Primitive,
    build_primitives,
    eval_primitive,
    eval_primitives,
    grad_primitive,
)

__all__ = [
    "Primitive",
    "build_primitives",
    "eval_primitive",
    "eval_primitives",
    "grad_primitive",
]
