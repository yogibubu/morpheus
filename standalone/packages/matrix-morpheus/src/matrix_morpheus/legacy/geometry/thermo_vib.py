"""
thermo_vib.py
=============

Vibrational thermodynamics in the harmonic approximation.

Computes vibrational contribution to:
- Q_vib
- U_vib
- H_vib
- S_vib
- Cv_vib
- Cp_vib  (Cp = Cv for vibrations)

Input:
- T_K from #BASIC
- harmonic frequencies (cm^-1) from #VIBRATIONAL

Conventions:
- If no usable vibrational modes are present:
    Q_vib = 1.0
    U = H = S = Cv = Cp = 0.0

This guarantees robustness and consistency with thermo_pipeline.
"""

import re
import numpy as np
from typing import Dict

from .physical_constants import get_physical_constants, Phy

R = 8.31446261815324  # J mol^-1 K^-1


# ============================================================
# Helpers
# ============================================================
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Read xyzin blocks
# ============================================================
def _read_blocks(xyzin_path: str):
    with open(xyzin_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    basic_lines = []
    vib_lines = []

    current = None
    for line in lines:
        s = line.strip()
        if s == "#BASIC":
            current = "basic"
            continue
        if s == "#VIBRATIONAL":
            current = "vib"
            continue
        if s.startswith("#") and s not in ("#BASIC", "#VIBRATIONAL"):
            current = None

        if current == "basic":
            basic_lines.append(line)
        elif current == "vib":
            vib_lines.append(line)

    return basic_lines, vib_lines


def _parse_basic_T(basic_lines):
    for line in basic_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        if "=" in s:
            k, v = [x.strip() for x in s.split("=", 1)]
        else:
            parts = s.split()
            if len(parts) < 2:
                continue
            k, v = parts[0], parts[1]

        if k.strip().upper() == "T_K":
            T = _safe_float(v)
            if T is None or T <= 0.0:
                raise ValueError("thermo_vib: invalid T_K in #BASIC")
            return T

    raise ValueError("thermo_vib: missing T_K in #BASIC")


# ============================================================
# Vibrational block parsing
# ============================================================
def _parse_vibrational_block(vib_lines):
    collecting = False
    buf = []

    def _grab_numbers(s):
        return re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    for line in vib_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        low = s.lower()

        if (
            low.startswith("frequencies")
            or low.startswith("freq")
            or low.startswith("freq_cm1")
            or low.startswith("freq_cm-1")
            or low.startswith("freq_cm^-1")
        ):
            collecting = True
            buf += _grab_numbers(s)
            continue

        if collecting:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*\s*[:=]", s):
                break
            buf += _grab_numbers(s)

    if len(buf) == 0:
        raise ValueError("No frequencies found in #VIBRATIONAL block")

    return np.array([float(x) for x in buf], dtype=float)


# ============================================================
# Harmonic vibrational thermodynamics
# ============================================================
def _vib_from_frequencies(freq_cm1: np.ndarray, T: float):
    """
    Returns (Qvib, U, H, S, Cv) for harmonic oscillators.

    Frequencies are in cm^-1.
    Includes ZPE consistently with the harmonic partition function.
    """
    phy = get_physical_constants(0)
    h = phy[Phy.PLANCK]
    c = phy[Phy.C_LIGHT]      # cm / s
    kB = phy[Phy.BOLTZMANN]
    NA = phy[Phy.AVOGADRO]

    # cm^-1 -> Hz
    nu = freq_cm1 * c

    x = (h * nu) / (kB * T)
    x = np.clip(x, 0.0, 700.0)

    emx = np.exp(-x)
    ex = np.exp(x)

    # ---- ln Q_vib
    ln1m = np.log1p(-emx)
    lnQ = np.sum(-0.5 * x - ln1m)
    Qvib = float(np.exp(lnQ))

    # ---- Thermodynamic functions
    U = np.sum(h * nu * (0.5 + 1.0 / (ex - 1.0))) * NA
    H = U

    S = R * np.sum((x * emx) / (1.0 - emx) - ln1m)
    Cv = R * np.sum((x * x * ex) / ((ex - 1.0) ** 2))

    return Qvib, float(U), float(H), float(S), float(Cv)


# ============================================================
# Public API
# ============================================================
def thermo_vib(
    xyzin_path: str,
    cutoff_cm1: float = 10.0,
    keep_low_positive: bool = False,
) -> Dict:

    basic_lines, vib_lines = _read_blocks(xyzin_path)
    T = _parse_basic_T(basic_lines)

    # ---- Default: no vibrational contribution
    Qvib = 1.0
    vib = {"U": 0.0, "H": 0.0, "S": 0.0, "Cv": 0.0}
    available = False
    reason = "no #VIBRATIONAL block"
    nfreq_total = 0
    nfreq_used = 0

    if vib_lines:
        try:
            freq = _parse_vibrational_block(vib_lines)
            nfreq_total = int(freq.size)

            if keep_low_positive:
                freq_used = freq[freq > 0.0]
            else:
                freq_used = freq[freq > float(cutoff_cm1)]

            nfreq_used = int(freq_used.size)

            if nfreq_used > 0:
                Qvib, U, H, S, Cv = _vib_from_frequencies(freq_used, T)
                vib = {"U": U, "H": H, "S": S, "Cv": Cv}
                available = True
                reason = "ok"
            else:
                reason = "no frequencies above cutoff"

        except Exception:
            reason = "no usable frequencies in #VIBRATIONAL block"

    # ---- Unified contract
    U = float(vib["U"])
    H = float(vib["H"])
    Cv = float(vib["Cv"])
    S = float(vib["S"])
    return {
        "Q_dimless": float(Qvib),
        "U_kJmol": U / 1000.0,
        "H_kJmol": H / 1000.0,
        "S_JmolK": S,
        "Cv_JmolK": Cv,
        "Cp_JmolK": Cv,

        # diagnostics
        "T_K": float(T),
        "available": bool(available),
        "reason": str(reason),
        "nfreq_total": int(nfreq_total),
        "nfreq_used": int(nfreq_used),
        "cutoff_cm1": float(cutoff_cm1),
        "keep_low_positive": bool(keep_low_positive),
        "vib": vib,
    }
