from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from matrix_gaussian import (
    SemiDiagonalDeltaBVibResult,
    compute_deltabvib_from_semidiagonal_cubic_data,
    lower_to_symmetric,
    read_gaussian_anharmonic_normal_cubic,
    read_gaussian_cartesian_cubic_force_field,
    read_gaussian_fchk,
    read_gaussian_fchk_geometry,
    semidiagonal_data_from_cartesian_cubic,
    semidiagonal_data_from_parent_normal_cubic,
)

from .contracts import IsotopologueObservation, VibrationalCorrection
from .io import write_observations_csv


@dataclass(frozen=True)
class IsotopologueDeltaBVib:
    label: str
    substitutions: dict[int, int]
    result: SemiDiagonalDeltaBVibResult


@dataclass(frozen=True)
class CartesianCubicCorrectionResult:
    gaussian_log: Path
    observations: tuple[IsotopologueObservation, ...]
    corrections: tuple[IsotopologueDeltaBVib, ...]
    output_csv: Path | None = None


def corrections_from_gaussian_cartesian_cubic(
    gaussian_log: Path | str,
    observations: tuple[IsotopologueObservation, ...],
    *,
    output_csv: Path | str | None = None,
) -> CartesianCubicCorrectionResult:
    """Calculate every isotopologue's Delta Bvib from one ``Freq=Cubic`` job."""

    if not observations:
        raise ValueError("at least one isotopologue observation is required")
    field = read_gaussian_cartesian_cubic_force_field(gaussian_log)
    updated: list[IsotopologueObservation] = []
    rows: list[IsotopologueDeltaBVib] = []
    for observation in observations:
        data = semidiagonal_data_from_cartesian_cubic(
            field.geometry.atoms,
            field.geometry.coordinates_angstrom,
            field.hessian_hartree_per_bohr2,
            None,
            substitutions=observation.substitutions,
            cubic_unique_indices=field.cubic_unique_indices,
            cubic_unique_values_hartree_per_bohr3=field.cubic_unique_values_hartree_per_bohr3,
        )
        correction = compute_deltabvib_from_semidiagonal_cubic_data(data)
        rows.append(
            IsotopologueDeltaBVib(
                label=observation.label,
                substitutions=dict(observation.substitutions),
                result=correction,
            )
        )
        updated.append(
            replace(
                observation,
                correction=VibrationalCorrection(
                    *correction.total_MHz,
                    source=f"Gaussian Freq=Cubic Cartesian F2/F3: {field.path.name}",
                    convention="subtract",
                ),
            )
        )
    target = Path(output_csv) if output_csv is not None else None
    if target is not None:
        write_observations_csv(target, tuple(updated))
    return CartesianCubicCorrectionResult(
        gaussian_log=field.path,
        observations=tuple(updated),
        corrections=tuple(rows),
        output_csv=target,
    )


def corrections_from_gaussian_anharmonic(
    gaussian_log: Path | str,
    gaussian_fchk: Path | str,
    observations: tuple[IsotopologueObservation, ...],
    *,
    output_csv: Path | str | None = None,
) -> CartesianCubicCorrectionResult:
    """Reuse a parent ``Freq=Anharm`` cubic field for isotope Delta Bvib.

    The normal-coordinate F3 is first returned to Cartesian space with the
    full 3N linear transformation (zero translation/rotation potential
    derivatives included).  Each isotopologue then receives its own masses,
    Eckart projector, normal modes, Coriolis constants and rotor axes.  The
    partial quartic field printed by Gaussian remains parent-only.
    """

    if not observations:
        raise ValueError("at least one isotopologue observation is required")
    normal = read_gaussian_anharmonic_normal_cubic(gaussian_log)
    fchk = read_gaussian_fchk(Path(gaussian_fchk))
    geometry = read_gaussian_fchk_geometry(Path(gaussian_fchk))
    dimension = 3 * geometry.natoms
    hessian = lower_to_symmetric(fchk.cartesian_hessian_lower)
    modes = np.asarray(fchk.normal_modes, dtype=float).reshape((-1, dimension))
    updated: list[IsotopologueObservation] = []
    rows: list[IsotopologueDeltaBVib] = []
    for observation in observations:
        data = semidiagonal_data_from_parent_normal_cubic(
            geometry.atoms,
            geometry.coordinates_angstrom,
            hessian,
            normal.cubic_qmw_hartree_amu32_bohr3,
            modes,
            fchk.reduced_masses_amu,
            fchk.masses_amu,
            substitutions=observation.substitutions,
        )
        correction = compute_deltabvib_from_semidiagonal_cubic_data(data)
        rows.append(
            IsotopologueDeltaBVib(
                label=observation.label,
                substitutions=dict(observation.substitutions),
                result=correction,
            )
        )
        updated.append(
            replace(
                observation,
                correction=VibrationalCorrection(
                    *correction.total_MHz,
                    source=(
                        "Gaussian Freq=Anharm parent F3; 3N normal-Cartesian-"
                        f"isotopologue transform: {Path(gaussian_log).name}"
                    ),
                    convention="subtract",
                ),
            )
        )
    target = Path(output_csv) if output_csv is not None else None
    if target is not None:
        write_observations_csv(target, tuple(updated))
    return CartesianCubicCorrectionResult(
        gaussian_log=Path(gaussian_log),
        observations=tuple(updated),
        corrections=tuple(rows),
        output_csv=target,
    )


__all__ = [
    "CartesianCubicCorrectionResult",
    "IsotopologueDeltaBVib",
    "corrections_from_gaussian_cartesian_cubic",
    "corrections_from_gaussian_anharmonic",
]
