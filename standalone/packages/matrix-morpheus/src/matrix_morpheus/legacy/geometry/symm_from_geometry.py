"""
symm_from_geometry.py
====================

Determine the rotational symmetry number (sigma) and point group
from molecular geometry.

This module:
- analyzes oriented Cartesian coordinates
- determines symmetry purely from geometry
- uses rotor_type as the ONLY classification authority
- does NOT use quasi-linear flags

Returned symmetry is consistent with rotational spectroscopy usage.
"""

import numpy as np

from .inertia import principal_moments


# ============================================================
# Low-level helpers (UNCHANGED)
# ============================================================
def _maxmin(a):
    return np.max(a), np.min(a)


def _group_by_symbol(symbols):
    groups = {}
    for i, s in enumerate(symbols):
        groups.setdefault(s, []).append(i)
    return groups


def _match(coords1, coords2, tol):
    used = np.zeros(len(coords2), dtype=bool)
    for v in coords1:
        found = False
        for j, w in enumerate(coords2):
            if used[j]:
                continue
            if np.linalg.norm(v - w) < tol:
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


def _match_with_map(symbols, coords1, coords2, tol):
    used = np.zeros(len(coords2), dtype=bool)
    mapping = [-1] * len(coords1)
    for i, v in enumerate(coords1):
        found = False
        for j, w in enumerate(coords2):
            if used[j]:
                continue
            if symbols[i] != symbols[j]:
                continue
            if np.linalg.norm(v - w) < tol:
                used[j] = True
                mapping[i] = j
                found = True
                break
        if not found:
            return None
    return mapping


# ============================================================
# Symmetry tests (UNCHANGED)
# ============================================================
def _has_inversion(coords, tol):
    return _match(coords, -coords, tol)


def _has_sigma_plane(coords, axis, tol):
    R = np.eye(3)
    R[axis, axis] = -1.0
    return _match(coords, coords @ R.T, tol)


def _has_c2_axis(coords, axis, tol):
    R = np.eye(3)
    for a in range(3):
        if a != axis:
            R[a, a] = -1.0
    return _match(coords, coords @ R.T, tol)


def _has_cn_axis(coords, n, tol):
    theta = 2.0 * np.pi / n
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    return _match(coords, coords @ R.T, tol)


def _rotation_matrix(axis, theta):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def _candidate_ops(max_n=6):
    ops = []
    # identity
    ops.append(("E", np.eye(3)))
    # inversion
    ops.append(("i", -np.eye(3)))
    # sigma planes
    for axis, name in [(0, "sigma_yz"), (1, "sigma_xz"), (2, "sigma_xy")]:
        R = np.eye(3)
        R[axis, axis] = -1.0
        ops.append((name, R))
    # C2 axes
    ops.append(("C2x", _rotation_matrix((1, 0, 0), np.pi)))
    ops.append(("C2y", _rotation_matrix((0, 1, 0), np.pi)))
    ops.append(("C2z", _rotation_matrix((0, 0, 1), np.pi)))
    # Cn around z
    for n in range(3, max_n + 1):
        theta = 2.0 * np.pi / n
        ops.append((f"C{n}z", _rotation_matrix((0, 0, 1), theta)))
    return ops


def symmetry_elements_from_geometry(symbols, coords_oriented, tol=1.0e-3, max_n=6):
    """
    Determine symmetry elements and atom equivalence classes from geometry.

    Returns:
        elements: list of (label, R) for valid symmetry operations
        classes: list of lists of atom indices (equivalence classes)
        permutations: list of index maps for each operation
    """
    coords = np.asarray(coords_oriented, dtype=float)
    elements = []
    permutations = []
    for label, R in _candidate_ops(max_n=max_n):
        coords_t = coords @ R.T
        mapping = _match_with_map(symbols, coords, coords_t, tol)
        if mapping is not None:
            elements.append((label, R))
            permutations.append(mapping)

    # build equivalence classes from permutations
    n = len(symbols)
    parent = list(range(n))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for mapping in permutations:
        for i, j in enumerate(mapping):
            _union(i, j)

    classes = {}
    for i in range(n):
        r = _find(i)
        classes.setdefault(r, []).append(i)

    return elements, list(classes.values()), permutations


# ============================================================
# Main API
# ============================================================
def symm_and_point_group_from_geometry(
    symbols,
    coords_oriented,
    rotor_type,
    representation,
    tol=1.0e-3,
):
    """
    Determine rotational symmetry number and point group.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols.
    coords_oriented : (N,3) ndarray
        Oriented Cartesian coordinates.
    rotor_type : str
        One of:
          - 'linear'
          - 'spherical'
          - 'symmetric_prolate'
          - 'symmetric_oblate'
          - 'asymmetric_prolate'
          - 'asymmetric_oblate'
    representation : str
        Axis representation (Ir, IIIr, ...). Currently informational.
    tol : float
        Matching tolerance (Å).

    Returns
    -------
    sigma : int
        Rotational symmetry number.
    point_group : str
        Schoenflies point group.
    """

    coords = np.asarray(coords_oriented, dtype=float)
    rt = rotor_type.lower()

    # --------------------------------------------------------
    # LINEAR MOLECULES
    # --------------------------------------------------------
    if rt == "linear":
        if _has_inversion(coords, tol):
            return 2, "Dinfh"
        return 1, "Cinfv"

    # --------------------------------------------------------
    # SPHERICAL TOPS
    # --------------------------------------------------------
    if rt == "spherical":
        # Highly symmetric cases
        if _has_inversion(coords, tol):
            return 12, "Oh"
        return 12, "Td"

    # --------------------------------------------------------
    # SYMMETRIC TOPS
    # --------------------------------------------------------
    if rt.startswith("symmetric"):
        # Principal axis assumed along z
        max_n = 1
        for n in range(6, 1, -1):
            if _has_cn_axis(coords, n, tol):
                max_n = n
                break

        has_sigma_v = _has_sigma_plane(coords, axis=0, tol=tol)
        has_c2 = _has_c2_axis(coords, axis=2, tol=tol)
        has_inv = _has_inversion(coords, tol)

        if has_inv:
            return max_n, f"D{max_n}h"
        if has_sigma_v:
            return max_n, f"C{max_n}v"
        if has_c2:
            return max_n, f"D{max_n}"
        return max_n, f"C{max_n}"

    # --------------------------------------------------------
    # ASYMMETRIC TOPS
    # --------------------------------------------------------
    if rt.startswith("asymmetric"):
        has_c2x = _has_c2_axis(coords, axis=0, tol=tol)
        has_c2y = _has_c2_axis(coords, axis=1, tol=tol)
        has_c2z = _has_c2_axis(coords, axis=2, tol=tol)
        n_c2 = sum((has_c2x, has_c2y, has_c2z))

        has_inv = _has_inversion(coords, tol)

        if n_c2 == 3:
            return 4, "D2h"
        if n_c2 == 1:
            return 2, "C2"
        if has_inv:
            return 1, "Ci"
        return 1, "C1"

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    return 1, "C1"


# ============================================================
# Convenience wrapper
# ============================================================
def symm_number_from_geometry(
    symbols,
    coords_oriented,
    rotor_type,
    representation,
    tol=1.0e-3,
):
    """
    Return only the rotational symmetry number (sigma).
    """
    sigma, _ = symm_and_point_group_from_geometry(
        symbols,
        coords_oriented,
        rotor_type,
        representation,
        tol=tol,
    )
    return int(sigma)


# ============================================================
# Legacy compatibility wrapper
# ============================================================
def symm_from_geometry(structure, representation="Ir", eps_zero=1.0e-6):
    """
    Backwards-compatible helper used by vibrational.py.

    Returns a minimal dict with:
      - representation: axis representation label
      - linear: True if the smallest principal moment is ~0
    """
    moments = principal_moments(structure, isotopic=True)
    linear = bool(moments[0] <= eps_zero)

    return {
        "representation": representation,
        "linear": linear,
    }
