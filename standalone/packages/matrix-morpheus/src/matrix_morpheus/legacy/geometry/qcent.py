#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from .physical_constants import Phy, get_physical_constants
from .vibrational_conversions import conversion_factor

phy = get_physical_constants()

# ============================================================
# Constants
# ============================================================

C_CM_S = phy[Phy.C_LIGHT]
CM1_TO_MHZ = C_CM_S / 1.0e6


# ============================================================
# Utilities
# ============================================================

def parse_xyzin_section_keyval(xyzin_path, section_name):
    data = {}
    sec = "#" + section_name.upper()
    in_sec = False
    with open(xyzin_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#"):
                in_sec = (s.split()[0].upper() == sec)
                continue
            if in_sec and "=" in s:
                k, v = [x.strip() for x in s.split("=", 1)]
                data[k] = v
    return data


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


# ============================================================
# vibin reader (STRICT format)
# ============================================================

def read_vibin_xyzinlike(vibin_path):

    with open(vibin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    xyz = lines[2:2 + nat]

    symbols = []
    coords = np.zeros((nat, 3))
    for i, l in enumerate(xyz):
        p = l.split()
        symbols.append(p[0])
        coords[i] = [float(p[1]), float(p[2]), float(p[3])]

    representation = "Ir"
    linear = False
    project_TR = True

    masses_amu = None
    freq_cm1 = None
    modes_mw = []

    i = 2 + nat
    while i < len(lines):
        s = lines[i].strip()

        if s.startswith("representation"):
            representation = s.split("=")[1].strip()
        elif s.startswith("linear"):
            linear = s.split("=")[1].strip().lower() in ("1", "true", "yes")
        elif s.startswith("project_TR"):
            project_TR = s.split("=")[1].strip().lower() in ("1", "true", "yes")

        elif s.startswith("masses_amu"):
            arr = []
            i += 1
            while not lines[i].strip().startswith("]"):
                p = lines[i].split()
                arr.append(float(p[1]))
                i += 1
            masses_amu = np.array(arr)

        elif s.startswith("freq_cm1"):
            arr = []
            i += 1
            while not lines[i].strip().startswith("]"):
                p = lines[i].split()
                arr.append(float(p[1]))
                i += 1
            freq_cm1 = np.array(arr)

        elif s.startswith("MODE"):
            mode = np.zeros((nat, 3))
            i += 1
            while not lines[i].strip().startswith("ENDMODE"):
                p = lines[i].split()
                a = int(p[0]) - 1
                mode[a] = [float(p[1]), float(p[2]), float(p[3])]
                i += 1
            modes_mw.append(mode)

        elif s.startswith("BEGIN_RESULTS"):
            break

        i += 1

    if masses_amu is None or freq_cm1 is None:
        raise RuntimeError("Invalid vibin: missing masses or frequencies")

    return {
        "symbols": symbols,
        "coords_A": coords,
        "masses_amu": masses_amu,
        "freq_cm1": freq_cm1,
        "modes_mw": np.array(modes_mw),
        "representation": representation,
        "linear": linear,
        "project_TR": project_TR,
    }


# ============================================================
# Core math
# ============================================================

def inertia_tensor_amuA2(m, r):
    I = np.zeros((3, 3))
    for a in range(len(m)):
        x, y, z = r[a]
        mi = m[a]
        I[0, 0] += mi * (y*y + z*z)
        I[1, 1] += mi * (x*x + z*z)
        I[2, 2] += mi * (x*x + y*y)
        I[0, 1] -= mi * x*y
        I[0, 2] -= mi * x*z
        I[1, 2] -= mi * y*z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I


def sym6(i, j):
    return [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)].index(tuple(sorted((i,j))))


def didq_sym6(m, r, modes):
    nv, nat = modes.shape[0], r.shape[0]
    didq = np.zeros((6, nv))
    for k in range(nv):
        for a in range(nat):
            mi = m[a]
            x, y, z = r[a]
            dx, dy, dz = modes[k, a]
            didq[0, k] += mi * (2*y*dy + 2*z*dz)
            didq[1, k] += mi * (2*x*dx + 2*z*dz)
            didq[2, k] += mi * (2*x*dx + 2*y*dy)
            didq[3, k] -= mi * (x*dy + y*dx)
            didq[4, k] -= mi * (x*dz + z*dx)
            didq[5, k] -= mi * (y*dz + z*dy)
    return didq


def tau_wilson(freq, pmom, didq):
    factg = conversion_factor("FACTG")
    tau = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    acc = 0.0
                    for m in range(len(freq)):
                        acc += didq[sym6(i, j), m] * didq[sym6(k, l), m] / (freq[m]**2)
                    tau[i, j, k, l] = -0.5 * acc / (factg**3 * pmom[i] * pmom[j] * pmom[k] * pmom[l])
    return tau


def _principal_moments(I0):
    vals = np.linalg.eigvalsh(I0)
    vals = np.sort(vals)
    ax_ok = vals > 1.0e-14
    return vals, ax_ok


def _rep_id(rep: str) -> int:
    rep = (rep or "").strip()
    mapping = {
        "Ir": 1,
        "IIr": 2,
        "IIIr": 3,
        "Il": 4,
        "IIl": 5,
        "IIIl": 6,
    }
    return mapping.get(rep, 2)


def _get_rep_perm(rep_id: int):
    # returns 0-based permutation
    if rep_id == 1:   # Ir
        return [2, 0, 1]
    if rep_id == 2:   # IIr
        return [0, 1, 2]
    if rep_id == 3:   # IIIr
        return [1, 2, 0]
    if rep_id == 4:   # Il
        return [2, 1, 0]
    if rep_id == 5:   # IIl
        return [1, 0, 2]
    if rep_id == 6:   # IIIl
        return [0, 2, 1]
    return [0, 1, 2]


def _permute_tau(tau_in, perm):
    tau_out = np.zeros_like(tau_in)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    tau_out[i, j, k, l] = tau_in[perm[i], perm[j], perm[k], perm[l]]
    return tau_out


def _tau_to_qcent_ared_ir(tau, ax_ok):
    if not (ax_ok[0] and ax_ok[1] and ax_ok[2]):
        return np.zeros(5, dtype=float)
    delta_j = 0.125 * (tau[1,1,1,1] + tau[2,2,2,2] + 2.0 * tau[1,1,2,2])
    delta_k = 0.125 * tau[0,0,0,0]
    delta_jk = -0.25 * (tau[0,0,1,1] + tau[0,0,2,2])
    small_j = 0.125 * (tau[1,1,1,1] + tau[2,2,2,2] - 2.0 * tau[1,1,2,2])
    small_k = -0.25 * (tau[0,1,0,1] + tau[0,2,0,2])
    return np.array([delta_j, delta_jk, delta_k, small_j, small_k], dtype=float)


def _qcent_ared_to_sred_ir(q_a):
    delta_j, delta_jk, delta_k, small_j, small_k = q_a
    q_s = np.zeros(5, dtype=float)
    q_s[0] = delta_j
    q_s[1] = delta_jk + delta_j
    q_s[2] = delta_k + delta_jk + delta_j
    q_s[3] = small_j
    q_s[4] = small_k
    return q_s


# ============================================================
# Public entry (PIPELINE COMPATIBLE)
# ============================================================

def compute_qcent_from_xyzin(xyzin_path, workdir="."):

    vibin = os.path.join(workdir, "vibin")
    if not os.path.exists(vibin):
        return None

    vib = read_vibin_xyzinlike(vibin)

    m = vib["masses_amu"]
    r = vib["coords_A"]
    freq = vib["freq_cm1"]
    modes = vib["modes_mw"]

    rot = parse_xyzin_section_keyval(xyzin_path, "ROTATIONAL")
    rep = rot.get("representation", vib["representation"])

    linear = vib["linear"]
    if "quasi_linear" in rot:
        linear = bool(_safe_int(rot["quasi_linear"]))

    I = inertia_tensor_amuA2(m, r)
    pmom = np.array([I[0, 0], I[1, 1], I[2, 2]])
    ax_ok = pmom > 1.0e-14

    didq = didq_sym6(m, r, modes)
    tau = tau_wilson(freq, pmom, didq)
    tau_mhz = tau * CM1_TO_MHZ

    tauP6 = np.array([
        tau_mhz[0, 0, 0, 0], tau_mhz[0, 0, 1, 1], tau_mhz[0, 0, 2, 2],
        tau_mhz[1, 1, 1, 1], tau_mhz[1, 1, 2, 2], tau_mhz[2, 2, 2, 2]
    ])
    tauP6_MHz = tauP6

    with open(vibin, "a") as f:
        f.write("\nBEGIN_RESULTS\n#QCENT\n")
        f.write(f"representation = {rep}\n")
        f.write(f"linear = {linear}\n\n")
        for l, v in zip(["aaaa","aabb","aacc","bbbb","bbcc","cccc"], tauP6_MHz):
            f.write(f"TauP {l:4s} {v: .10E}\n")
        # Quartic centrifugal distortion constants (Watson A and S)
        rep_id = _rep_id(rep)
        perm = _get_rep_perm(rep_id)
        tau_p = _permute_tau(tau_mhz, perm)
        qa_mhz = _tau_to_qcent_ared_ir(tau_p, ax_ok)
        qs_mhz = _qcent_ared_to_sred_ir(qa_mhz)

        qa_cm1 = qa_mhz / CM1_TO_MHZ
        qs_cm1 = qs_mhz / CM1_TO_MHZ

        f.write("\nA-reduction (DelJ DelJK DelK delJ delK) [MHz]\n")
        f.write("DelJ_MHz  {0: .10E}\n".format(qa_mhz[0]))
        f.write("DelJK_MHz {0: .10E}\n".format(qa_mhz[1]))
        f.write("DelK_MHz  {0: .10E}\n".format(qa_mhz[2]))
        f.write("delJ_MHz  {0: .10E}\n".format(qa_mhz[3]))
        f.write("delK_MHz  {0: .10E}\n".format(qa_mhz[4]))

        f.write("\nS-reduction (DJ DJK DK d1 d2) [MHz]\n")
        f.write("DJ_MHz  {0: .10E}\n".format(qs_mhz[0]))
        f.write("DJK_MHz {0: .10E}\n".format(qs_mhz[1]))
        f.write("DK_MHz  {0: .10E}\n".format(qs_mhz[2]))
        f.write("d1_MHz  {0: .10E}\n".format(qs_mhz[3]))
        f.write("d2_MHz  {0: .10E}\n".format(qs_mhz[4]))
        f.write("END_RESULTS\n")

    return {
        "representation": rep,
        "linear": linear,
        "tauP6_MHz": tauP6_MHz,
        "tauP6_cm1": tauP6_MHz / CM1_TO_MHZ,
        "QA_MHz": qa_mhz,
        "QA_cm1": qa_cm1,
        "QS_MHz": qs_mhz,
        "QS_cm1": qs_cm1,
        "DelJ_MHz": qa_mhz[0],
        "DelJK_MHz": qa_mhz[1],
        "DelK_MHz": qa_mhz[2],
        "delJ_MHz": qa_mhz[3],
        "delK_MHz": qa_mhz[4],
        "DJ_MHz": qs_mhz[0],
        "DJK_MHz": qs_mhz[1],
        "DK_MHz": qs_mhz[2],
        "d1_MHz": qs_mhz[3],
        "d2_MHz": qs_mhz[4],
        "nvib": len(freq),
        "min_principal_moment_amuA2": float(np.min(pmom)),
    }
