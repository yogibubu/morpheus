"""LINK adapter for the ORACLE/CV_radial exponential structural field."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from matrix_chem import evaluate_cv_exponential_field

from .scan import PointEvaluationResult


def add_cv_exponential_field(
    result: PointEvaluationResult,
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    bonded_pairs: Iterable[tuple[int, int]],
) -> PointEvaluationResult:
    """Add CV energy and any available analytic derivatives to a LINK result."""
    field = evaluate_cv_exponential_field(
        atomic_numbers, coordinates_angstrom, bonded_pairs
    )
    energy = (
        None
        if result.energy_hartree is None
        else float(result.energy_hartree) + field.energy_hartree
    )
    gradient = result.gradient_hartree_per_bohr
    if gradient is not None:
        gradient = np.asarray(gradient, dtype=float).reshape(-1) + field.gradient_hartree_per_bohr
    hessian = result.hessian_hartree_per_bohr2
    if hessian is not None:
        hessian = np.asarray(hessian, dtype=float) + field.hessian_hartree_per_bohr2
    source = "+".join(item for item in (result.source, "ORACLE_CV_RADIAL_EXPONENTIAL") if item)
    return replace(
        result,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        source=source,
    )
