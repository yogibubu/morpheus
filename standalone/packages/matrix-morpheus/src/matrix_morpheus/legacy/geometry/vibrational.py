#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from .symm_from_geometry import symm_from_geometry
from .physical_constants import Phy, get_physical_constants
from .structure import Structure

phy = get_physical_constants()


# ============================================================
# Helpers
# ============================================================

def read_xyz_from_xyzin(xyzin):
    with open(xyzin, "r", encoding="utf-8") as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    xyz = lines[2:2 + nat]

    symbols = []
    coords = np.zeros((nat, 3))
    for i, l in enumerate(xyz):
        p = l.split()
        symbols.append(p[0])
        coords[i] = [float(p[1]), float(p[2]), float(p[3])]
    return symbols, coords


def read_fchk_masses(fchk):
    """
    Read atomic masses from a Gaussian fchk file.
    """
    with open(fchk, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("Real atomic weights") or "Atomic masses" in l or l.startswith("Vib-AtMass"):
            n = int(l.split()[-1])
            vals = []
            i += 1
            while len(vals) < n and i < len(lines):
                vals += [float(x) for x in lines[i].split()]
                i += 1
            return np.array(vals, dtype=float)
        i += 1

    raise ValueError("FCHK masses not found (Real atomic weights / Atomic masses / Vib-AtMass)")


def read_fchk_hessian_cartesian(fchk):
    """
    Read Cartesian Force Constants from Gaussian fchk (lower-triangular).
    Returns a 1D array of length 3N(3N+1)/2.
    """
    with open(fchk, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("Cartesian Force Constants"):
            n = int(l.split()[-1])
            vals = []
            i += 1
            while len(vals) < n and i < len(lines):
                vals += [float(x) for x in lines[i].split()]
                i += 1
            return np.array(vals, dtype=float)
        i += 1

    raise ValueError("FCHK Hessian not found (Cartesian Force Constants)")


def _hessian_eig_to_freq_cm1(eigvals):
    """
    Convert eigenvalues of mass-weighted Hessian (Hartree/Bohr^2/amu)
    to frequencies in cm^-1.
    """
    c = phy[Phy.C_LIGHT]        # cm / s
    Eh = phy[Phy.HARTREE]       # J / Hartree
    a0 = phy[Phy.M_PER_B]       # m / Bohr
    amu = phy[Phy.TO_KG]        # kg / amu
    factor = (1.0 / (2.0 * np.pi * c)) * np.sqrt(Eh / (a0 * a0 * amu))
    return np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * factor


def _orthonormal_columns(mat, rtol=1e-12):
    """
    Orthonormalize columns of mat, dropping near-dependent vectors.
    """
    if mat.size == 0:
        return mat
    # SVD is stable for rank detection
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return mat[:, :0]
    tol = rtol * s[0]
    r = int(np.sum(s > tol))
    return u[:, :r]


def _tr_basis(coords_bohr, masses_amu):
    """
    Build mass-weighted translation and rotation basis vectors.
    """
    nat = len(masses_amu)
    nd = 3 * nat
    m_sqrt = np.sqrt(masses_amu)

    # Center of mass
    com = np.average(coords_bohr, axis=0, weights=masses_amu)
    r = coords_bohr - com

    # Translations: sqrt(m) along x,y,z
    t = []
    for ax in range(3):
        v = np.zeros((nat, 3), dtype=float)
        v[:, ax] = m_sqrt
        t.append(v.reshape(nd))

    # Rotations: sqrt(m) * (e_alpha x r_i)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    axes = [ex, ey, ez]
    rvec = []
    for ax in axes:
        v = np.cross(ax, r) * m_sqrt[:, None]
        rvec.append(v.reshape(nd))

    basis = np.column_stack(t + rvec)
    return basis


def modes_from_hessian(masses_amu, hess_tri, coords_A, linear=False, project_tr=True):
    """
    Diagonalize the Cartesian Hessian (triangular storage) and return
    frequencies and mass-weighted normal modes.
    """
    nat = len(masses_amu)
    nd = 3 * nat
    expected = nd * (nd + 1) // 2
    if len(hess_tri) != expected:
        raise ValueError(f"Hessian size mismatch: expected {expected}, got {len(hess_tri)}")

    # Build full symmetric matrix
    H = np.zeros((nd, nd), dtype=float)
    idx = 0
    for i in range(nd):
        for j in range(i + 1):
            H[i, j] = hess_tri[idx]
            H[j, i] = hess_tri[idx]
            idx += 1

    # Mass-weight
    m = np.repeat(masses_amu, 3)
    w = 1.0 / np.sqrt(m)
    Hmw = H * (w[:, None] * w[None, :])

    # Project out translations/rotations in mass-weighted space
    coords_bohr = np.array(coords_A, dtype=float) / phy[Phy.TO_ANG]
    basis = _tr_basis(coords_bohr, masses_amu)
    Q = _orthonormal_columns(basis)
    if project_tr and Q.shape[1] > 0:
        P = np.eye(nd) - Q @ Q.T
        Hmw = P @ Hmw @ P

    # Diagonalize projected Hessian
    eigvals, eigvecs = np.linalg.eigh(Hmw)

    # Drop translational/rotational modes (based on projected rank)
    nzero = Q.shape[1] if Q.shape[1] > 0 else (5 if linear else 6)
    if nzero >= len(eigvals):
        raise ValueError("Not enough modes to drop translation/rotation")
    drop_idx = np.argsort(np.abs(eigvals))[:nzero]
    keep = np.ones(len(eigvals), dtype=bool)
    keep[drop_idx] = False

    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    # Sort by eigenvalue (ascending)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    freq_cm1 = _hessian_eig_to_freq_cm1(eigvals)
    modes_mw = eigvecs.T.reshape((-1, nat, 3))

    return freq_cm1, modes_mw


def read_fchk_dipole_au(fchk):
    """
    Read dipole moment (in atomic units) from Gaussian fchk.
    Returns np.array([mux, muy, muz]) or None.
    """
    with open(fchk, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("Dipole Moment") and "num derivs" not in l:
            n = int(l.split()[-1])
            vals = []
            i += 1
            while len(vals) < n and i < len(lines):
                vals += [float(x) for x in lines[i].split()]
                i += 1
            if len(vals) >= 3:
                return np.array(vals[:3], dtype=float)
            return None
        i += 1

    return None


def dipole_au_to_debye(dip_au):
    """
    Convert dipole from atomic units (e·Bohr) to Debye.
    """
    if dip_au is None:
        return None
    e_esu = phy[Phy.TO_E]           # esu / e
    bohr_cm = phy[Phy.TO_ANG] * 1e-8  # cm / Bohr
    au_to_debye = (e_esu * bohr_cm) / 1.0e-18
    return np.array(dip_au, dtype=float) * au_to_debye


def mass_weight_modes(masses_amu, modes):
    m_kg = masses_amu * phy[Phy.TO_KG]
    sqrtm = np.sqrt(m_kg / phy[Phy.E_MASS_KG])
    mw = np.zeros_like(modes)
    for k in range(modes.shape[0]):
        for a in range(modes.shape[1]):
            mw[k, a] = modes[k, a] * sqrtm[a]
    return mw


def didq_sym6(masses_amu, coords_A, modes_mw):
    """
    dI/dQ in symmetric 6-vector form (consistent with qcent Fortran).
    Returns array shape (6, nvib).
    """
    nat = len(masses_amu)
    nvib = modes_mw.shape[0]
    didq = np.zeros((6, nvib), dtype=float)
    for k in range(nvib):
        for a in range(nat):
            mi = masses_amu[a]
            sqrm = np.sqrt(mi)
            x, y, z = coords_A[a]
            dx, dy, dz = modes_mw[k, a] / sqrm  # dr/dQ
            rdotdr = x * dx + y * dy + z * dz
            dI11 = mi * (2.0 * rdotdr - 2.0 * x * dx)
            dI22 = mi * (2.0 * rdotdr - 2.0 * y * dy)
            dI33 = mi * (2.0 * rdotdr - 2.0 * z * dz)
            dI12 = mi * (-x * dy - y * dx)
            dI13 = mi * (-x * dz - z * dx)
            dI23 = mi * (-y * dz - z * dy)
            didq[0, k] += dI11
            didq[1, k] += dI22
            didq[2, k] += dI33
            didq[3, k] += dI12
            didq[4, k] += dI13
            didq[5, k] += dI23
    return didq


# ============================================================
# vibin writer (COMPATIBLE)
# ============================================================

def write_vibin(
    vibin,
    symbols,
    coords_A,
    masses_amu,
    freq_cm1,
    modes_mw,
    representation,
    linear,
    project_tr,
    didq_sym6_vals=None,
):
    nat = len(symbols)
    nvib = len(freq_cm1)

    with open(vibin, "w", encoding="utf-8") as f:
        f.write(f"{nat}\n")
        f.write("vibro-rotational input\n")
        for s, r in zip(symbols, coords_A):
            f.write(f"{s:2s} {r[0]:15.8f} {r[1]:15.8f} {r[2]:15.8f}\n")

        f.write(f"\nrepresentation = {representation}\n")
        f.write(f"linear = {linear}\n")
        f.write(f"project_TR = {bool(project_tr)}\n")
        f.write("normalization = mass-weighted\n\n")

        f.write("masses_amu [\n")
        for i, m in enumerate(masses_amu, 1):
            f.write(f" {i:5d} {m:15.8f}\n")
        f.write("]\n\n")

        f.write("freq_cm1 [\n")
        for i, v in enumerate(freq_cm1, 1):
            f.write(f" {i:5d} {v:15.8f}\n")
        f.write("]\n\n")

        if didq_sym6_vals is not None:
            f.write("didq_sym6 [\n")
            for k in range(nvib):
                v = didq_sym6_vals[:, k]
                f.write(
                    f" {k+1:5d} {v[0]: .10e} {v[1]: .10e} {v[2]: .10e}"
                    f" {v[3]: .10e} {v[4]: .10e} {v[5]: .10e}\n"
                )
            f.write("]\n\n")

        for k in range(nvib):
            f.write(f"MODE {k+1}\n")
            for a in range(nat):
                dx, dy, dz = modes_mw[k, a]
                f.write(f"{a+1:5d} {dx:15.8e} {dy:15.8e} {dz:15.8e}\n")
            f.write("ENDMODE\n\n")


# ============================================================
# Public entry
# ============================================================

def vibrational(xyzin, fchkin=None, workdir=".", project_tr=True):

    symbols, coords = read_xyz_from_xyzin(xyzin)

    if fchkin is None or not os.path.exists(fchkin):
        return None

    try:
        masses_amu = read_fchk_masses(fchkin)
        hess_tri = read_fchk_hessian_cartesian(fchkin)
    except Exception as e:
        print(f"WARNING: Vibrations skipped: {e}")
        return None

    if len(masses_amu) != len(symbols):
        print(
            "WARNING: Vibrations skipped: fchkin atom count does not match xyzin "
            f"({len(masses_amu)} vs {len(symbols)})"
        )
        return None
    dip_au = read_fchk_dipole_au(fchkin)
    dip_debye = dipole_au_to_debye(dip_au)

    structure = Structure(symbols=symbols, coords=coords)
    symm = symm_from_geometry(structure)

    freq_cm1, modes_mw = modes_from_hessian(
        masses_amu=masses_amu,
        hess_tri=hess_tri,
        coords_A=coords,
        linear=symm["linear"],
        project_tr=project_tr,
    )

    vibin = os.path.join(workdir, "vibin")

    didq = didq_sym6(masses_amu, coords, modes_mw)

    write_vibin(
        vibin=vibin,
        symbols=symbols,
        coords_A=coords,
        masses_amu=masses_amu,
        freq_cm1=freq_cm1,
        modes_mw=modes_mw,
        representation=symm["representation"],
        linear=symm["linear"],
        project_tr=project_tr,
        didq_sym6_vals=didq,
    )

    return {
        "vibin": vibin,
        "nvib": len(freq_cm1),
        "linear": symm["linear"],
        "representation": symm["representation"],
        "freq_cm1": freq_cm1,
        "n_imag_like": int(np.sum(freq_cm1 < 0.0)),
        "dipole_debye": None if dip_debye is None else dip_debye,
        "dipole_missing": bool(dip_debye is None),
    }


def vib_from_xyzin(xyzin, structure=None, workdir=None, fchkin=None, project_tr=True):
    """
    Convenience wrapper used by the rotational pipeline.

    - Looks for fchkin alongside xyzin (or in workdir if provided)
    - If missing, returns None (caller will fallback)
    """
    if workdir is None:
        workdir = os.path.dirname(os.path.abspath(xyzin))

    if fchkin is None:
        fchkin = os.path.join(workdir, "fchkin")

    # structure currently unused but kept for compatibility
    _ = structure

    return vibrational(xyzin, fchkin=fchkin, workdir=workdir, project_tr=project_tr)
