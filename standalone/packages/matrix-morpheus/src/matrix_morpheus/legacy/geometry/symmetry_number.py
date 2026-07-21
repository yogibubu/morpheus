"""
Rotational symmetry number utilities.

Maps molecular point-group labels to the rotational symmetry number σ.
Only proper rotations are considered, following standard
statistical thermodynamics conventions.
"""

import re


def rotational_symmetry_number(point_group: str) -> int:
    """
    Return the rotational symmetry number σ for a given point group.

    Parameters
    ----------
    point_group : str
        Point group label (e.g. 'C2v', 'D2d', 'D3h', 'Td', 'Oh', 'Ih',
        'Cinfv', 'Dinfh').

    Returns
    -------
    sigma : int
        Rotational symmetry number.

    Notes
    -----
    - Improper operations (σ, i, S_n) do not contribute directly to σ.
    - Point groups differing only by v/h/d suffixes have the same σ.
    """
    if not point_group:
        raise ValueError("Point group must be specified to determine σ")

    pg = point_group.strip()

    # ------------------------------------------------------------
    # Linear molecules
    # ------------------------------------------------------------
    if pg == "Cinfv":
        return 1
    if pg == "Dinfh":
        return 2

    # ------------------------------------------------------------
    # Trivial groups
    # ------------------------------------------------------------
    if pg in {"C1", "Cs", "Ci"}:
        return 1

    # ------------------------------------------------------------
    # Platonic solids
    # ------------------------------------------------------------
    if pg == "Td":
        return 12
    if pg == "Oh":
        return 24
    if pg == "Ih":
        return 60

    # ------------------------------------------------------------
    # Dihedral groups: Dn, Dnh, Dnd  → σ = 2n
    # (includes D2d, D3d, etc.)
    # ------------------------------------------------------------
    m = re.match(r"D(\d+)", pg)
    if m:
        n = int(m.group(1))
        return 2 * n

    # ------------------------------------------------------------
    # Cyclic groups: Cn, Cnv, Cnh → σ = n
    # ------------------------------------------------------------
    m = re.match(r"C(\d+)", pg)
    if m:
        return int(m.group(1))

    # ------------------------------------------------------------
    # Improper rotation groups Sn
    # Sn has n/2 proper rotations (n even only)
    # Example: S4 → C2 → σ = 2
    # ------------------------------------------------------------
    m = re.match(r"S(\d+)", pg)
    if m:
        n = int(m.group(1))
        if n % 2 != 0:
            raise ValueError(f"Invalid improper rotation group '{pg}'")
        return n // 2

    # ------------------------------------------------------------
    # Unsupported / unknown
    # ------------------------------------------------------------
    raise ValueError(f"Unsupported or unknown point group '{point_group}'")
