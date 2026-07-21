"""
thermo_rot.py
=============

Rotational statistical mechanics module.

This module provides:
- Rotational partition function Qrot(T)
- ln Qrot(T)
- First and second derivatives of ln Qrot(T)
- Rotational thermodynamic functions (U, S, Cv, Cp)

IMPORTANT:
- This module is NOT "only thermodynamics".
- Qrot(T) is used BOTH for rotational spectroscopy
  (level populations, intensities)
  AND for thermodynamics.
- No file writing is performed here.

Design principles:
- Temperature is read ONLY from #BASIC as T_K
  (unless provided explicitly in future extensions).
- Rotational constants A, B, C are provided externally (MHz).
- Rotor classification is NOT decided here and MUST be passed
  as rotor_type.
- Smooth quantum/classical crossover via error-function mixing.
"""

import numpy as np
from math import pi, sqrt, erf, exp, log


# ============================================================
# Physical constants (SI)
# ============================================================
h  = 6.62607015e-34      # J s
kB = 1.380649e-23       # J K-1
NA = 6.02214076e23      # mol-1
R  = NA * kB            # J mol-1 K-1


# ============================================================
# Small safe converters
# ============================================================
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


# ============================================================
# XYzin parsing utilities
# ============================================================
def parse_xyzin_basic_T(xyzin_path):
    """
    Read temperature T_K from the #BASIC section of xyzin.
    """
    in_basic = False
    with open(xyzin_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                in_basic = (s.split()[0].upper() == "#BASIC")
                continue
            if not in_basic:
                continue
            if "=" in s:
                k, v = [x.strip() for x in s.split("=", 1)]
                if k.strip().upper() == "T_K":
                    T = _safe_float(v)
                    if T is None or T <= 0.0:
                        raise ValueError("thermo_rot: invalid T_K value")
                    return T

    raise ValueError("thermo_rot: missing T_K in #BASIC")


def parse_xyzin_rotational_section(xyzin_path):
    """
    Parse #ROTATIONAL section into a dict.
    Used only to retrieve the symmetry number if sigma is not provided.
    """
    data = {}
    in_rot = False
    with open(xyzin_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                in_rot = (s.split()[0].upper() == "#ROTATIONAL")
                continue
            if in_rot and "=" in s:
                k, v = [x.strip() for x in s.split("=", 1)]
                data[k] = v
    return data


def get_sigma(xyzin_path, sigma=None):
    """
    Determine the rotational symmetry number.

    Priority:
      1) explicit sigma argument
      2) 'Symm. Number' in #ROTATIONAL
      3) default = 1 (with warning)
    """
    if sigma is not None:
        return max(int(sigma), 1)

    rotinfo = parse_xyzin_rotational_section(xyzin_path)

    if "Symm. Number" in rotinfo:
        s = _safe_int(rotinfo["Symm. Number"])
        if s is not None and s > 0:
            return s

    print("WARNING (thermo_rot): Symm. Number not found; using sigma = 1")
    return 1


# ============================================================
# Quantum rotational partition functions
# (sigma is applied outside)
# ============================================================
def qrot_quantum_linear(Beff_Hz, T, Jmax=200):
    beta = h / (kB * T)
    return sum(
        (2 * J + 1) * exp(-beta * Beff_Hz * J * (J + 1))
        for J in range(Jmax + 1)
    )


def qrot_quantum_spherical(B_Hz, T, Jmax=200):
    beta = h / (kB * T)
    return sum(
        (2 * J + 1) ** 2 * exp(-beta * B_Hz * J * (J + 1))
        for J in range(Jmax + 1)
    )


def qrot_quantum_symmetric(A_Hz, B_Hz, T, Jmax=150):
    beta = h / (kB * T)
    q = 0.0
    for J in range(Jmax + 1):
        pref = (2 * J + 1)
        JJ1 = J * (J + 1)
        for K in range(-J, J + 1):
            q += pref * exp(
                -beta * (B_Hz * JJ1 + (A_Hz - B_Hz) * K * K)
            )
    return q


# ============================================================
# Classical ln(Q) and derivatives
# ============================================================
def lnq_classical_linear(Beff_Hz, T, sigma):
    theta = h * Beff_Hz / kB
    return log(T / (sigma * theta))


def lnq_classical_spherical(B_Hz, T, sigma):
    theta = h * B_Hz / kB
    return 0.5 * log(pi) + 1.5 * log(T / theta) - log(float(sigma))


def lnq_classical_symmetric(A_Hz, B_Hz, T, sigma):
    thetaA = h * A_Hz / kB
    thetaB = h * B_Hz / kB
    return (
        0.5 * log(pi)
        + 1.5 * log(T)
        - log(float(sigma))
        - 0.5 * log(thetaA * thetaB * thetaB)
    )


def dlnq_classical(T, kind):
    if kind == "linear":
        return 1.0 / T, -1.0 / (T * T)
    return 1.5 / T, -1.5 / (T * T)


# ============================================================
# Numerical derivatives of lnQ(T)
# ============================================================
def dlnQ_num(lnQ_func, T, dT=0.05):
    """
    Numerical first and second derivatives of lnQ(T)
    using a symmetric finite-difference scheme.
    """
    Tp = T + dT
    Tm = max(T - dT, 1.0e-6)

    lnQp = lnQ_func(Tp)
    lnQ0 = lnQ_func(T)
    lnQm = lnQ_func(Tm)

    d1 = (lnQp - lnQm) / (Tp - Tm)
    d2 = 2.0 * (
        (lnQp - lnQ0) / (Tp - T) -
        (lnQ0 - lnQm) / (T - Tm)
    ) / (Tp - Tm)

    return d1, d2


# ============================================================
# Smooth mixing weights for quantum/classical crossover
# ============================================================
def _mix_weights(T, Tq, Tc):
    T0 = 0.5 * (Tq + Tc)
    dTmix = 0.2 * (Tc - Tq)
    x = (T - T0) / dTmix
    ex = exp(-x * x)
    w = 0.5 * (1.0 + erf(x))
    wp = ex / (sqrt(pi) * dTmix)
    wpp = (-2.0 * x) * ex / (sqrt(pi) * dTmix**2)
    return w, wp, wpp


# ============================================================
# Public API
# ============================================================
def thermo_rot(
    xyzin_path,
    A_MHz, B_MHz, C_MHz,
    rotor_type,
    sigma=None,
    Tq=200.0,
    Tc=250.0,
    Jmax_linear=200,
    Jmax_spherical=200,
    Jmax_symmetric=150,
    dT_num=0.05,
):
    """
    Compute rotational statistical mechanics and thermodynamics.

    Returns at least:
      - Qrot(T)
      - ln Qrot(T)

    and, if derivatives are needed:
      - U, S, Cv, Cp
    """

    # --- Temperature and symmetry number
    T = parse_xyzin_basic_T(xyzin_path)
    sigma_i = get_sigma(xyzin_path, sigma=sigma)

    # --- Convert MHz -> Hz
    A_Hz = A_MHz * 1.0e6
    B_Hz = B_MHz * 1.0e6
    C_Hz = C_MHz * 1.0e6
    Beff_Hz = max(B_Hz, C_Hz)

    # --- Map rotor_type to effective model
    if rotor_type == "linear":
        kind = "linear"
    elif rotor_type == "spherical":
        kind = "spherical"
    else:
        kind = "symmetric"

    # --- Quantum lnQ
    def lnQ_quantum(Tloc):
        if kind == "linear":
            Q = qrot_quantum_linear(Beff_Hz, Tloc, Jmax_linear)
        elif kind == "spherical":
            Q = qrot_quantum_spherical(B_Hz, Tloc, Jmax_spherical)
        else:
            Q = qrot_quantum_symmetric(
                A_Hz, Beff_Hz, Tloc, Jmax_symmetric
            )
        return log(Q / sigma_i)

    lnQq = lnQ_quantum(T)

    # --- Classical lnQ
    if kind == "linear":
        lnQc = lnq_classical_linear(Beff_Hz, T, sigma_i)
    elif kind == "spherical":
        lnQc = lnq_classical_spherical(B_Hz, T, sigma_i)
    else:
        lnQc = lnq_classical_symmetric(A_Hz, Beff_Hz, T, sigma_i)

    # --- Regime selection / mixing
    if T <= Tq:
        lnQ = lnQq
        dlnQ, d2lnQ = dlnQ_num(lnQ_quantum, T, dT=dT_num)

    elif T >= Tc:
        lnQ = lnQc
        dlnQ, d2lnQ = dlnq_classical(T, kind)

    else:
        w, wp, wpp = _mix_weights(T, Tq, Tc)
        dlnQq, d2lnQq = dlnQ_num(lnQ_quantum, T, dT=dT_num)
        dlnQc, d2lnQc = dlnq_classical(T, kind)

        lnQ = (1.0 - w) * lnQq + w * lnQc
        dlnQ = (1.0 - w) * dlnQq + w * dlnQc + wp * (lnQc - lnQq)
        d2lnQ = (
            (1.0 - w) * d2lnQq
            + w * d2lnQc
            + 2.0 * wp * (dlnQc - dlnQq)
            + wpp * (lnQc - lnQq)
        )

    # --- Thermodynamic quantities
    U  = R * T * T * dlnQ
    S  = R * (lnQ + T * dlnQ)
    Cv = R * (2.0 * T * dlnQ + T * T * d2lnQ)
    Qrot = float(np.exp(lnQ))

    U_kJ = U / 1000.0
    H_kJ = U / 1000.0
    return {
        # --- contratto UNIFICATO per thermo_pipeline / thermo_writer
        "Q_dimless": Qrot,
        "U_kJmol": U_kJ,
        "H_kJmol": H_kJ,
        "S_JmolK": S,
        "Cv_JmolK": Cv,
        "Cp_JmolK": Cv,

        # --- informazioni estese (USATE ALTROVE, NON RIMUOVERE)
        "Qrot": Qrot,
        "lnQ": lnQ,
        "T_K": T,
    }
