from __future__ import annotations

import math


GDV_PERIODIC_TOLERANCE = 1.0e-6


def gdv_principal_dihedral(value: float) -> float:
    """Return GDV ``FixCrd(IOp=2)``'s canonical dihedral value."""

    one_pi = math.pi
    two_pi = 2.0 * one_pi
    result = math.fmod(float(value), two_pi)
    if abs(result) > one_pi:
        result = -math.copysign(two_pi - abs(result), result)
    if abs(result + one_pi) < GDV_PERIODIC_TOLERANCE or abs(
        result - one_pi
    ) < GDV_PERIODIC_TOLERANCE:
        result = one_pi
    if abs(result) < GDV_PERIODIC_TOLERANCE:
        result = 0.0
    return result


def gdv_match_dihedral_phase(value: float, reference: float) -> float:
    """Return GDV ``FixCrd(IOp=3)``'s branch nearest ``reference``."""

    one_pi = math.pi
    two_pi = 2.0 * one_pi
    delta = math.fmod(float(value) - float(reference), two_pi)
    if abs(delta - two_pi) < GDV_PERIODIC_TOLERANCE or abs(
        delta + two_pi
    ) < GDV_PERIODIC_TOLERANCE:
        delta = 0.0
    if abs(delta - one_pi) < GDV_PERIODIC_TOLERANCE:
        delta = one_pi
    if abs(delta + one_pi) < GDV_PERIODIC_TOLERANCE:
        delta = -one_pi

    result = float(reference) + delta
    if delta > one_pi:
        result -= two_pi
    elif delta < -one_pi:
        result += two_pi
    if abs(result) < GDV_PERIODIC_TOLERANCE:
        result = 0.0
    if abs(result - one_pi) < GDV_PERIODIC_TOLERANCE:
        result = one_pi
    if abs(result + one_pi) < GDV_PERIODIC_TOLERANCE:
        result = -one_pi
    return result


__all__ = [
    "GDV_PERIODIC_TOLERANCE",
    "gdv_match_dihedral_phase",
    "gdv_principal_dihedral",
]
