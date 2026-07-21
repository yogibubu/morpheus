"""
thermo_trasl.py
===============

Translational thermodynamics for an ideal gas molecule.

This module computes:
- Q_trans
- U_trans
- H_trans
- S_trans
- Cv_trans
- Cp_trans

All data are read from the xyzin file.
No file writing is performed here.
"""

import numpy as np

from .physical_constants import Phy, get_physical_constants
from .structure import Structure

phy = get_physical_constants()


# ============================================================
# BASIC section parsing
# ============================================================
def parse_xyzin_basic_section(xyzin_path):
    data = {}
    in_basic = False

    def _norm_key(k):
        ku = k.strip().upper()
        if ku == "T_K":
            return "T_K"
        if ku == "P_ATM":
            return "P_ATM"
        return ku

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
                data[_norm_key(k)] = v

    return data


# ============================================================
# XYZ reader (minimal, local)
# ============================================================
def read_xyz_from_xyzin(xyzin_path):
    with open(xyzin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    xyz_lines = lines[2:2 + nat]

    symbols = []
    coords = []
    for line in xyz_lines:
        f = line.split()
        symbols.append(f[0])
        coords.append((float(f[1]), float(f[2]), float(f[3])))

    return symbols, coords


# ============================================================
# Public API
# ============================================================
def thermo_trasl(xyzin_path):
    """
    Translational thermodynamic functions.
    """

    kb = phy[Phy.BOLTZMANN]
    h  = phy[Phy.PLANCK]
    NA = phy[Phy.AVOGADRO]
    R  = kb * NA

    basic = parse_xyzin_basic_section(xyzin_path)

    try:
        T = float(basic.get("T_K"))
        P_atm = float(basic.get("P_ATM"))
    except Exception:
        return {}

    if T <= 0.0 or P_atm <= 0.0:
        return {}

    # pressure in Pa
    P = P_atm * 101325.0

    # build structure locally
    try:
        symbols, coords = read_xyz_from_xyzin(xyzin_path)
        mol = Structure(symbols=symbols, coords=coords, isotopes=None)
        mass_kg = float(mol.total_mass_isotope) * phy[Phy.TO_KG]
    except Exception:
        return {}

    # --------------------------------------------------------
    # Partition function
    # --------------------------------------------------------
    lnQ = (
        1.5 * np.log((2.0 * np.pi * mass_kg * kb * T) / (h * h))
        + np.log(kb * T / P)
    )

    Q_trans = float(np.exp(lnQ))

    # --------------------------------------------------------
    # Thermodynamic functions
    # --------------------------------------------------------
    U = 1.5 * R * T
    H = 2.5 * R * T
    Cv = 1.5 * R
    Cp = 2.5 * R
    S = R * (lnQ + 2.5)

    U_kJ = U / 1000.0
    H_kJ = H / 1000.0
    return {
        "Q_dimless": Q_trans,
        "U_kJmol": U_kJ,
        "H_kJmol": H_kJ,
        "S_JmolK": S,
        "Cv_JmolK": Cv,
        "Cp_JmolK": Cp,
    }
