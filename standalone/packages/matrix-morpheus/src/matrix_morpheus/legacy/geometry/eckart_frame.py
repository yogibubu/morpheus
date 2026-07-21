# merlino/geometry/eckart_frame.py

import numpy as np
from typing import Tuple

from .structure import Structure
from .inertia import center_of_mass


# ----------------------------------------------------------------------
# Internal helper: translate to COM (matches previous intent)
# ----------------------------------------------------------------------

def _translate_to_com(structure: Structure, use_isotopic_masses: bool = True) -> np.ndarray:
    """
    Return coordinates translated to the center of mass.
    Units: Å
    """
    com = center_of_mass(structure, isotopic=use_isotopic_masses)
    coords = np.asarray(structure.coords, dtype=float)
    return coords - com[None, :]


# ----------------------------------------------------------------------
# Weighted Kabsch / Procrustes for Eckart frame
# ----------------------------------------------------------------------

def eckart_rotation(
    structure: Structure,
    reference: Structure,
) -> np.ndarray:
    """
    Compute the Eckart rotation matrix aligning `structure`
    to `reference` using isotopic masses.

    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix such that:
        coords_structure @ R.T ≈ coords_reference
    """
    if structure.natoms != reference.natoms:
        raise ValueError("Structure and reference must have same number of atoms")

    # Center both structures at COM (isotopic masses)
    X = _translate_to_com(reference, use_isotopic_masses=True)   # reference
    Y = _translate_to_com(structure, use_isotopic_masses=True)   # structure

    masses = np.asarray(structure.mass_isotope, dtype=float)

    # Build weighted covariance matrix: C = sum_i m_i * (Y_i outer X_i)
    C = np.zeros((3, 3))
    for i in range(structure.natoms):
        C += masses[i] * np.outer(Y[i], X[i])

    # SVD
    U, _, Vt = np.linalg.svd(C)

    # Proper rotation (det = +1)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0.0:
        D[2, 2] = -1.0

    R = U @ D @ Vt
    return R


# ----------------------------------------------------------------------
# Apply Eckart frame
# ----------------------------------------------------------------------

def apply_eckart_frame(
    structure: Structure,
    reference: Structure,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Eckart frame to a structure.

    Returns
    -------
    coords_eckart : ndarray, shape (N,3)
        Eckart-aligned coordinates
    R : ndarray, shape (3,3)
        Eckart rotation matrix
    """
    R = eckart_rotation(structure, reference)

    coords = _translate_to_com(structure, use_isotopic_masses=True)
    coords_eckart = coords @ R.T

    return coords_eckart, R


# ----------------------------------------------------------------------
# Eckart RMSD (optional diagnostic)
# ----------------------------------------------------------------------

def eckart_rmsd(
    structure: Structure,
    reference: Structure,
) -> float:
    """
    Compute the mass-weighted RMSD after Eckart alignment.
    """
    coords_eckart, _ = apply_eckart_frame(structure, reference)
    ref_coords = _translate_to_com(reference, use_isotopic_masses=True)

    masses = np.asarray(structure.mass_isotope, dtype=float)

    diff2 = np.sum(masses[:, None] * (coords_eckart - ref_coords) ** 2)
    return np.sqrt(diff2 / np.sum(masses))
