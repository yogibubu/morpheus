"""Small reusable performance helpers for semiexperimental fits."""

from __future__ import annotations

import numpy as np

from .contracts import IsotopologueObservation
from .fit import _mass_vector_for_observation, _mass_vector_for_isotopes


def mass_vector_for_observation(
    atoms: list[str] | tuple[str, ...],
    observation: IsotopologueObservation,
) -> np.ndarray:
    """Return cached isotope-aware atomic masses for an observation."""
    return _mass_vector_for_observation(atoms, observation).copy()


def mass_vector_for_isotopes(
    atoms: list[str] | tuple[str, ...],
    isotopes: list[int | None] | tuple[int | None, ...],
) -> np.ndarray:
    """Return cached isotope-aware atomic masses for explicit isotope labels."""
    return _mass_vector_for_isotopes(atoms, isotopes).copy()


__all__ = ["mass_vector_for_isotopes", "mass_vector_for_observation"]
