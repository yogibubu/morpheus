# merlino/geometry/axis_representation.py

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple


@dataclass(frozen=True)
class AxisRep:
    """
    Axis representation.

    Parameters
    ----------
    perm : tuple of int
        Permutation mapping (x, y, z) <- (a, b, c),
        where a, b, c are principal axes (Ia <= Ib <= Ic).

        IMPORTANT:
        - Indices in `perm` are 0-based and are used internally by Merlino.
        - Conversion to 1-based indexing MUST be handled exclusively
          by the GUI or I/O layer when interacting with the user.

    handedness : str
        'R' for right-handed, 'L' for left-handed.
    """
    perm: Tuple[int, int, int]
    handedness: str

    def rotation_matrix(self, principal_axes_abc: np.ndarray) -> np.ndarray:
        """
        Build the rotation matrix corresponding to this axis representation.

        Parameters
        ----------
        principal_axes_abc : ndarray, shape (3,3)
            Columns are the principal axes (a, b, c) in the original frame.

        Returns
        -------
        R : ndarray, shape (3,3)
            Rotation matrix transforming coordinates to (x,y,z)
            according to the selected representation.
        """
        # Reorder axes: (x,y,z) <- (a,b,c)
        R = principal_axes_abc[:, self.perm]

        # Enforce handedness if needed
        det = np.linalg.det(R)
        if self.handedness.upper() == "R" and det < 0.0:
            R[:, 0] *= -1.0
        elif self.handedness.upper() == "L" and det > 0.0:
            R[:, 0] *= -1.0

        return R


# ----------------------------------------------------------------------
# Available axis representations
# ----------------------------------------------------------------------

REPRESENTATIONS: Dict[str, AxisRep] = {
    # Identity / standard
    "Ir": AxisRep((0, 1, 2), "R"),
    "Il": AxisRep((0, 1, 2), "L"),

    # Cyclic permutations
    "IIr": AxisRep((1, 2, 0), "R"),
    "IIl": AxisRep((1, 2, 0), "L"),

    "IIIr": AxisRep((2, 0, 1), "R"),
    "IIIl": AxisRep((2, 0, 1), "L"),
}


# ----------------------------------------------------------------------
# Application helper
# ----------------------------------------------------------------------

def apply_axis_representation(
    coords: np.ndarray,
    principal_axes_abc: np.ndarray,
    rep_label: str,
) -> np.ndarray:
    """
    Apply an axis representation to a set of coordinates.

    Parameters
    ----------
    coords : ndarray, shape (N,3)
        Cartesian coordinates in the principal-axis frame (a,b,c).
    principal_axes_abc : ndarray, shape (3,3)
        Principal axes (a,b,c) as columns.
    rep_label : str
        Key of the desired representation (e.g. 'Ir', 'IIIr').

    Returns
    -------
    coords_xyz : ndarray, shape (N,3)
        Coordinates expressed in the selected (x,y,z) frame.

    Notes
    -----
    - This function operates exclusively with 0-based indices.
    - Any conversion to 1-based indexing for user interaction
      must be handled outside this module.
    """
    try:
        rep = REPRESENTATIONS[rep_label]
    except KeyError:
        raise KeyError(
            f"Unknown axis representation '{rep_label}'. "
            f"Available: {list(REPRESENTATIONS.keys())}"
        ) from None

    R = rep.rotation_matrix(principal_axes_abc)

    # Apply rotation
    return coords @ R
