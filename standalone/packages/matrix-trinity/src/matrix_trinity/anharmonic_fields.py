from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def build_normal_mode_anharmonic_field(
    *,
    reference_coordinates_angstrom: np.ndarray,
    frequencies_cm: np.ndarray,
    eigenvalues: np.ndarray,
    mass_weighted_modes: np.ndarray,
    masses_amu: np.ndarray,
    evaluator: Callable[[np.ndarray], tuple[float, np.ndarray]],
    config=None,
):
    """Build a TRINITY QFF from any direct QM or analytic E/G evaluator."""

    from matrix_vpt2_vci import NormalModeQFFFitConfig, fit_normal_mode_qff

    return fit_normal_mode_qff(
        reference_coordinates_angstrom=reference_coordinates_angstrom,
        frequencies_cm=frequencies_cm,
        eigenvalues=eigenvalues,
        mass_weighted_modes=mass_weighted_modes,
        masses_amu=masses_amu,
        evaluator=evaluator,
        config=NormalModeQFFFitConfig() if config is None else config,
        source="TRINITY_DIRECT_ENERGY_GRADIENT",
    )


def build_normal_mode_anharmonic_field_from_architect(
    xyzin: Path | str,
    force_field_path: Path | str,
    *,
    config=None,
):
    """Build the same TRINITY QFF by sampling an ARCHITECT/ZAFF E/G field."""

    from matrix_vpt2_vci import NormalModeQFFFitConfig, fit_normal_mode_qff_from_zaff

    return fit_normal_mode_qff_from_zaff(
        xyzin,
        force_field_path,
        config=NormalModeQFFFitConfig() if config is None else config,
    )
