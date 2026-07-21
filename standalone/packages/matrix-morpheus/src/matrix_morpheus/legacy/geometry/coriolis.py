#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .physical_constants import Phy, get_physical_constants

phy = get_physical_constants()

# =========================
# unit conversions
# =========================
C_CM_S = phy[Phy.C_LIGHT]
CM1_TO_MHZ = C_CM_S / 1.0e6
MHZ_TO_CM1 = 1.0 / CM1_TO_MHZ


# ============================================================
# vibin reader (xyzin-like vibro-rotational file)
# ============================================================

def read_vibin_xyzinlike(vibin_path):
    """
    Reads vibin written by vibrational.write_vibin_xyzinlike.

    Returns dict with:
      coords_A: (N,3)  final frame
      masses_amu: (N,)
      freq_cm1: (nvib,)
      modes_mw: (nvib,N,3)  mass-weighted, orthonormal
    """
    with open(vibin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("vibin too short")

    nat = int(lines[0].strip())
    xyz_lines = lines[2:2 + nat]

    symbols = []
    coords = np.zeros((nat, 3), dtype=float)
    for i, l in enumerate(xyz_lines):
        p = l.split()
        symbols.append(p[0])
        coords[i, 0] = float(p[1])
        coords[i, 1] = float(p[2])
        coords[i, 2] = float(p[3])

    representation = "Ir"
    linear = False
    project_TR = True
    normalization = "mass-weighted-orthonormal"

    masses_amu = None
    freq_cm1 = None
    modes_mw = []

    i = 2 + nat
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if s.startswith("representation"):
            representation = s.split("=", 1)[1].strip()
            i += 1
            continue
        if s.startswith("linear"):
            linear = s.split("=", 1)[1].strip().lower() in ("true", "1", "yes")
            i += 1
            continue
        if s.startswith("project_TR"):
            project_TR = s.split("=", 1)[1].strip().lower() in ("true", "1", "yes")
            i += 1
            continue
        if s.startswith("normalization"):
            normalization = s.split("=", 1)[1].strip()
            i += 1
            continue

        if s.startswith("masses_amu") and "[" in s:
            arr = []
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("]"):
                    break
                if t:
                    p = t.split()
                    if len(p) >= 2:
                        arr.append(float(p[1]))
                i += 1
            masses_amu = np.array(arr, dtype=float)
            i += 1
            continue

        if s.startswith("freq_cm1") and "[" in s:
            arr = []
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("]"):
                    break
                if t:
                    p = t.split()
                    if len(p) >= 2:
                        arr.append(float(p[1]))
                i += 1
            freq_cm1 = np.array(arr, dtype=float)
            i += 1
            continue

        if s.startswith("MODE"):
            mode = np.zeros((nat, 3), dtype=float)
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("ENDMODE"):
                    break
                p = t.split()
                if len(p) >= 4:
                    a = int(p[0]) - 1
                    mode[a, 0] = float(p[1])
                    mode[a, 1] = float(p[2])
                    mode[a, 2] = float(p[3])
                i += 1
            modes_mw.append(mode)
            i += 1
            continue

        # stop if results already appended
        if s.startswith("BEGIN_RESULTS") or s.startswith("BEGIN_CORIOLIS"):
            break

        i += 1

    if masses_amu is None:
        raise ValueError("masses_amu block not found in vibin")
    if freq_cm1 is None:
        raise ValueError("freq_cm1 block not found in vibin")

    modes_mw = np.array(modes_mw, dtype=float)
    if modes_mw.shape[0] != len(freq_cm1):
        raise ValueError("Mismatch MODE blocks vs freq_cm1")

    return {
        "symbols": symbols,
        "coords_A": coords,
        "masses_amu": masses_amu,
        "freq_cm1": freq_cm1,
        "modes_mw": modes_mw,
        "representation": representation,
        "linear": bool(linear),
        "project_TR": bool(project_TR),
        "normalization": normalization,
    }


# ============================================================
# Core: compute sparse Coriolis entries (i, j, -k)
# ============================================================

def compute_coriolis_sparse_entries(
    masses_amu,
    modes_mw,
    freq_cm1,
    A_cm1, B_cm1, C_cm1,
    Geff_thr_cm1=1.0,
    only_upper=True
):
    """
    Build sparse entries:
      (i, j, -k) with k=1,2,3 Cartesian x,y,z

    Each entry stores:
      zeta(i,j,-k)
      Geff(i,j,-k) in cm^-1 and MHz

    Filtering rule (requested):
      keep ONLY if |Geff_cm1| >= Geff_thr_cm1
    """

    m = np.array(masses_amu, dtype=float)
    freq = np.array(freq_cm1, dtype=float)
    nvib, nat, _ = modes_mw.shape

    # axis mapping k=1,2,3 -> x,y,z -> a,b,c -> A,B,C
    Baxis_cm1 = {
        1: float(A_cm1),
        2: float(B_cm1),
        3: float(C_cm1),
    }

    entries = []

    for i in range(nvib):
        jstart = i + 1 if only_upper else 0
        for j in range(jstart, nvib):
            if i == j:
                continue

            wi = float(freq[i])
            wj = float(freq[j])
            if wi <= 0.0 or wj <= 0.0:
                continue

            # zeta vector (x,y,z):
            # zeta_xyz = sum_a (1/m_a) * (L_ai x L_aj)
            acc = np.zeros(3, dtype=float)
            for a in range(nat):
                acc += np.cross(modes_mw[i, a, :], modes_mw[j, a, :]) / m[a]

            for k in (1, 2, 3):
                zeta_k = float(acc[k - 1])
                Bk = Baxis_cm1[k]

                Geff_cm1 = 2.0 * Bk * zeta_k * (wi + wj) / np.sqrt(wi * wj)
                Geff_MHz = Geff_cm1 * CM1_TO_MHZ

                if abs(Geff_cm1) >= float(Geff_thr_cm1):
                    entries.append({
                        "i": i + 1,
                        "j": j + 1,
                        "kneg": -k,
                        "zeta": zeta_k,
                        "Geff_cm1": float(Geff_cm1),
                        "Geff_MHz": float(Geff_MHz),
                    })

    entries.sort(key=lambda d: abs(d["Geff_cm1"]), reverse=True)
    return entries


# ============================================================
# Output: append to vibin
# ============================================================

def append_coriolis_sparse_to_vibin(
    vibin_path,
    entries,
    A_cm1, B_cm1, C_cm1,
    Geff_thr_cm1
):
    with open(vibin_path, "a", encoding="utf-8") as f:
        f.write("\nBEGIN_CORIOLIS\n")
        f.write("# Sparse Coriolis quantities\n")
        f.write("# index convention: (i, j, -k) with k=1,2,3 Cartesian (x,y,z)\n")
        f.write("# i,j are normal-mode indices (1..nvib)\n")
        f.write("# -k indicates Cartesian index (not a mode)\n")
        f.write("# filtering: keep only |Geff_cm1| >= threshold\n\n")

        f.write("units:\n")
        f.write("  zeta = dimensionless\n")
        f.write("  Geff = cm^-1  (and MHz)\n\n")

        f.write(f"A_cm1 = {A_cm1: .12f}\n")
        f.write(f"B_cm1 = {B_cm1: .12f}\n")
        f.write(f"C_cm1 = {C_cm1: .12f}\n\n")

        f.write(f"threshold_Geff_cm1 = {float(Geff_thr_cm1):.6e}\n\n")

        f.write(" i    j   -k        zeta            Geff(cm^-1)          Geff(MHz)\n")
        for d in entries:
            f.write(
                f"{d['i']:4d} {d['j']:4d} {d['kneg']:4d} "
                f"{d['zeta']:16.8e} "
                f"{d['Geff_cm1']:16.8e} "
                f"{d['Geff_MHz']:16.8e}\n"
            )

        f.write("END_CORIOLIS\n")


# ============================================================
# Driver
# ============================================================

def run_coriolis_from_vibin(
    vibin_path,
    A, B, C,
    units="MHz",
    Geff_thr_cm1=1.0,
    only_upper=True
):
    """
    Compute sparse Coriolis entries (i, j, -k) and append to vibin.

    Required:
      vibin_path
      A,B,C rotational constants

    units:
      "MHz"  -> input A,B,C are in MHz
      "cm-1" -> input A,B,C are in cm^-1
    """

    vib = read_vibin_xyzinlike(vibin_path)

    masses_amu = vib["masses_amu"]
    freq_cm1 = vib["freq_cm1"]
    modes_mw = vib["modes_mw"]

    if units.lower() in ("mhz", "mhz."):
        A_cm1 = float(A) * MHZ_TO_CM1
        B_cm1 = float(B) * MHZ_TO_CM1
        C_cm1 = float(C) * MHZ_TO_CM1
    elif units.lower() in ("cm-1", "cm^-1", "cm1"):
        A_cm1 = float(A)
        B_cm1 = float(B)
        C_cm1 = float(C)
    else:
        raise ValueError("units must be 'MHz' or 'cm-1'")

    entries = compute_coriolis_sparse_entries(
        masses_amu=masses_amu,
        modes_mw=modes_mw,
        freq_cm1=freq_cm1,
        A_cm1=A_cm1,
        B_cm1=B_cm1,
        C_cm1=C_cm1,
        Geff_thr_cm1=Geff_thr_cm1,
        only_upper=only_upper
    )

    append_coriolis_sparse_to_vibin(
        vibin_path=vibin_path,
        entries=entries,
        A_cm1=A_cm1,
        B_cm1=B_cm1,
        C_cm1=C_cm1,
        Geff_thr_cm1=Geff_thr_cm1
    )

    return entries
