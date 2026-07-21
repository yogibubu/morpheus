# utils/vibrational_conversions.py

import math
from typing import Dict

from .physical_constants import Phy, get_physical_constants


# ---------------------------------------------------------------------
# Conversion factors for vibrational analysis
# Faithful translation of Fortran CnvFct
# ---------------------------------------------------------------------

def conversion_factor(key: str, codata: int = 2010) -> float:
    """
    Return vibrational conversion factor.

    Parameters
    ----------
    key : str
        Identifier of the requested factor (case-insensitive).
    codata : int
        CODATA year for physical constants.

    Returns
    -------
    float
        Conversion factor.

    Raises
    ------
    ValueError
        If the keyword is not recognized.
    """

    if not key:
        raise ValueError("Empty conversion factor key")

    k = key.strip().upper()
    phy = get_physical_constants(codata)

    # Short aliases (exactly as in Fortran)
    TO_ANG = phy[Phy.TO_ANG]
    TO_KG = phy[Phy.TO_KG]
    TO_E = phy[Phy.TO_E]
    PLANCK = phy[Phy.PLANCK]
    AVOG = phy[Phy.AVOGADRO]
    JP_CAL = phy[Phy.JP_CAL]
    HARTREE = phy[Phy.HARTREE]
    C_LIGHT = phy[Phy.C_LIGHT]
    BOLTZ = phy[Phy.BOLTZMANN]
    FINE = phy[Phy.FINE_STRUCT]
    E_MASS = phy[Phy.E_MASS_KG]

    # Constants
    PI = 4.0 * math.atan(1.0)
    F2 = 2.0
    F4 = 4.0
    F10P3 = 1.0e3
    F10P10 = 1.0e10
    F10P18 = 1.0e18

    # m → Å
    m2Ang = F10P10

    # Reduced Planck constant (amu · Å^2 · s^-1)
    hbar = PLANCK * m2Ang**2 / (F2 * PI * TO_KG)

    # 4 π^2 c / h   (cm · amu^-1 · Å^-2)
    FactG = F2 * PI * C_LIGHT / hbar

    # h c  (attoJ · cm == mdyn · Å · cm)
    HC = PLANCK * C_LIGHT * F10P18

    # -----------------------------------------------------------------
    # Keyword dispatch (faithful to Fortran logic)
    # -----------------------------------------------------------------

    if k == "FAC0AU":
        return HARTREE * F10P18 / HC

    elif k in ("FAC1AU", "FACT1"):
        val = 1.0 / (math.sqrt(FactG) * HC)
        if k == "FAC1AU":
            val *= HARTREE * F10P18 / TO_ANG
        return val

    elif k in ("FAC2AU", "FACT2"):
        val = 1.0 / (FactG * HC)
        if k == "FAC2AU":
            val *= HARTREE * F10P18 / TO_ANG**2
        return val

    elif k in ("FAC3AU", "FACT3"):
        val = 1.0 / (FactG * math.sqrt(FactG) * HC)
        if k == "FAC3AU":
            val *= HARTREE * F10P18 / TO_ANG**3
        return val

    elif k in ("FAC4AU", "FACT4"):
        val = 1.0 / (FactG**2 * HC)
        if k == "FAC4AU":
            val *= HARTREE * F10P18 / TO_ANG**4
        return val

    elif k == "FACTG":
        return FactG

    elif k == "FACTA":
        return 1.0 / (F2 * FactG)

    elif k == "FACTB":
        return PLANCK * C_LIGHT / (F2 * BOLTZ)

    elif k == "FACTC":
        return (
            BOLTZ * m2Ang**2
            / (C_LIGHT**2 * TO_KG * TO_ANG**2)
        )

    elif k == "HBAR":
        return hbar

    elif k == "TOUMA":
        return TO_KG / E_MASS

    elif k == "PICH12":
        return PI * math.sqrt(C_LIGHT * TO_KG / PLANCK / m2Ang**2)

    elif k == "AU2DEB":
        return F10P10 * TO_E * TO_ANG

    elif k == "MWQ2Q":
        return 1.0 / (math.sqrt(FactG) * TO_ANG)

    elif k == "AU2CM1":
        return HARTREE / (PLANCK * C_LIGHT)

    elif k == "AU2AMU":
        return E_MASS / TO_KG

    elif k == "AU2KJM":
        return HARTREE * AVOG / F10P3

    elif k == "AU2KCAM":
        return HARTREE * AVOG / (F10P3 * JP_CAL)

    elif k == "FACG0AU":
        return 1.0

    elif k == "FACG1AU":
        return 1.0 / (TO_ANG * math.sqrt(FactG))

    elif k == "FACG2AU":
        return 1.0 / (TO_ANG**2 * FactG)

    else:
        raise ValueError(f"Unrecognized keyword for conversion_factor: {key}")
