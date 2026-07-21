# utils/angular_momentum.py

import numpy as np
from typing import Tuple


def angular_momentum_vector(
    masses: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """
    Compute total angular momentum generated when moving from
    coords_a to coords_b.

    Parameters
    ----------
    masses : (N,) array
        Atomic masses
    coords_a : (N, 3) array
        Initial coordinates
    coords_b : (N, 3) array
        Final coordinates

    Returns
    -------
    L : (3,) array
        Total angular momentum vector
    """
    if coords_a.shape != coords_b.shape:
        raise ValueError("Coordinate arrays must have the same shape")

    if coords_a.shape[0] != masses.shape[0]:
        raise ValueError("Mass array incompatible with coordinates")

    disp = coords_b - coords_a
    L = np.zeros(3)

    for i in range(coords_a.shape[0]):
        L += masses[i] * np.cross(coords_a[i], disp[i])

    return L


def check_angular_momentum(
    masses: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    tol: float = 1.0e-8,
) -> Tuple[bool, np.ndarray]:
    """
    Check whether significant angular momentum is generated
    when going from coords_a to coords_b.

    Parameters
    ----------
    masses : (N,) array
        Atomic masses
    coords_a : (N, 3) array
        Initial coordinates
    coords_b : (N, 3) array
        Final coordinates
    tol : float
        Threshold for angular momentum components

    Returns
    -------
    ok : bool
        True if angular momentum is below tolerance
    L : (3,) array
        Angular momentum vector
    """
    L = angular_momentum_vector(masses, coords_a, coords_b)
    ok = np.all(np.abs(L) < tol)
    return ok, L
