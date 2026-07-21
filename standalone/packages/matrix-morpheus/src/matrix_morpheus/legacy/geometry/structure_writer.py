"""
structure_writer.py
===================

Writer for enrichment sections in xyzin files.

Responsibilities:
- write #ROTATIONAL and #VIBRATIONAL blocks
- adapt output format to rotor_type
- write Q_rot for rotational statistics (spectroscopy)

This module performs NO calculations.
"""

from typing import Dict, Optional


# ============================================================
# Small helpers
# ============================================================
def _write_keyval(f, key: str, value):
    f.write(f"{key} = {value}\n")


def _rotor_type_label(rotor_type: str) -> str:
    mapping = {
        "linear": "linear_top",
        "spherical": "spherical_top",
        "symmetric_prolate": "symmetric_top_prolate",
        "symmetric_oblate": "symmetric_top_oblate",
        "asymmetric_prolate": "asymmetric_top_quasi_prolate",
        "asymmetric_oblate": "asymmetric_top_quasi_oblate",
    }
    return mapping.get(rotor_type, rotor_type)


# ============================================================
# Section stripping
# ============================================================
def _strip_section_lines(lines, section_name):
    tag = "#" + section_name.upper()
    out = []
    in_sec = False
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            if s.upper() == tag:
                in_sec = True
                continue
            if in_sec:
                in_sec = False
        if not in_sec:
            out.append(line)
    return out


def _strip_sections(lines, section_names):
    out = lines[:]
    for sec in section_names:
        out = _strip_section_lines(out, sec)
    return out


# ============================================================
# BASIC parsing
# ============================================================
def _extract_TP_from_basic(lines):
    T = None
    P = None
    in_basic = False

    for raw in lines:
        s = raw.strip()
        if s.upper() == "#BASIC":
            in_basic = True
            continue
        if in_basic and s.startswith("#"):
            break
        if not in_basic or "=" not in s:
            continue

        k, v = [x.strip() for x in s.split("=", 1)]
        if k.upper() == "T_K":
            T = float(v)
        elif k.upper() == "P_ATM":
            P = float(v)

    return T, P


# ============================================================
# DELTABVIB extraction
# ============================================================
def _extract_deltabvib_lines(lines):
    """
    Extract DVib* lines from a #DELTABVIB block (case-insensitive).
    """
    out = []
    in_block = False

    for raw in lines:
        s = raw.strip()
        if s.upper().startswith("#DELTABVIB"):
            in_block = True
            continue
        if in_block and s.startswith("#"):
            break
        if in_block and s.upper().startswith("DVIB"):
            out.append(s)

    return out


# ============================================================
# ROTATIONAL block writer
# ============================================================
def _write_rotational_block(
    f,
    rot,
    T_K,
    P_atm,
    Q_rot,
    deltabvib_lines=None,
    qcent=None,
):
    if rot is None:
        return

    rotor_type = rot.rotor_type
    label = _rotor_type_label(rotor_type)

    f.write("\n#ROTATIONAL\n")

    # 1. rotor_type
    _write_keyval(f, "rotor_type", label)

    # 2. representation
    _write_keyval(f, "representation", rot.representation)

    # 3. Watson Reduction (ONLY non-linear, non-spherical)
    if rotor_type not in ("linear", "spherical"):
        watson = getattr(rot, "watson_reduction", "S")
        _write_keyval(f, "Watson Reduction", watson)

    # 4. Point Group
    if getattr(rot, "point_group", None):
        _write_keyval(f, "Point Group", rot.point_group)

    # 5. Symmetry number
    _write_keyval(f, "Symm. Number", int(rot.sigma))

    # 6. Thermodynamic reference conditions
    _write_keyval(f, "T_K", f"{T_K:.6f}")
    _write_keyval(f, "P_atm", f"{P_atm:.6f}")

    # 7. Rotational constants
    if rotor_type == "linear":
        _write_keyval(f, "B_MHz", f"{rot.B_MHz:.13f}")

    elif rotor_type == "spherical":
        _write_keyval(f, "B_MHz", f"{rot.B_MHz:.13f}")

    else:
        _write_keyval(f, "A_MHz", f"{rot.A_MHz:.13f}")
        _write_keyval(f, "B_MHz", f"{rot.B_MHz:.13f}")
        _write_keyval(f, "C_MHz", f"{rot.C_MHz:.13f}")

    # 8. DeltaBvib (copied from #DELTABVIB)
    if deltabvib_lines:
        for l in deltabvib_lines:
            f.write(l + "\n")

    # 9. Dipole moments
    dip = getattr(rot, "dipole_oriented_debye", None)
    if dip is not None and len(dip) >= 3:
        _write_keyval(f, "Dipole_a_D", f"{dip[0]:.8f}")
        _write_keyval(f, "Dipole_b_D", f"{dip[1]:.8f}")
        _write_keyval(f, "Dipole_c_D", f"{dip[2]:.8f}")

    # 10. Centrifugal distortion (S/A reduction, if available)
    if rotor_type not in ("linear", "spherical") and isinstance(qcent, dict):
        s_keys = ("DJ_MHz", "DJK_MHz", "DK_MHz", "d1_MHz", "d2_MHz")
        a_keys = ("DelJ_MHz", "DelJK_MHz", "DelK_MHz", "delJ_MHz", "delK_MHz")
        if all(k in qcent for k in s_keys):
            for k in s_keys:
                _write_keyval(f, k, f"{float(qcent[k]):.12f}")
        if all(k in qcent for k in a_keys):
            for k in a_keys:
                _write_keyval(f, k, f"{float(qcent[k]):.12f}")

    # 11. Rotational partition function (ALWAYS LAST)
    _write_keyval(f, "Q_rot", f"{Q_rot:.6f}")


# ============================================================
# VIBRATIONAL block writer
# ============================================================
def _write_vibrational_block(fh, vib: Optional[Dict]):
    """
    Minimal VIBRATIONAL block for xyzin.
    """
    if vib is None:
        return

    fh.write("\n#VIBRATIONAL\n")

    def _get(key, default=None):
        if isinstance(vib, dict):
            return vib.get(key, default)
        return getattr(vib, key, default)

    linear = _get("linear", None)
    nvib = _get("nvib", None)
    n_imag_like = _get("n_imag_like", None)
    freq_cm1 = _get("freq_cm1", None)
    symmetry_group = _get("symmetry_group", None)

    if linear is not None:
        _write_keyval(fh, "linear", int(bool(linear)))

    if nvib is not None:
        _write_keyval(fh, "nvib", int(nvib))

    if n_imag_like is not None:
        _write_keyval(fh, "n_imag_like", int(n_imag_like))

    if symmetry_group is not None:
        _write_keyval(fh, "symmetry_group", str(symmetry_group))

    if freq_cm1 is not None:
        try:
            arr = [float(x) for x in freq_cm1]
        except Exception:
            arr = []
        if len(arr) > 0:
            fh.write("freq_cm1 = ")
            fh.write(" ".join(f"{x:.6f}" for x in arr))
            fh.write("\n")

    ir_inten = _get("ir_inten_km_mol", None) or _get("ir_inten", None)
    if ir_inten is not None:
        try:
            arr = [float(x) for x in ir_inten]
        except Exception:
            arr = []
        if len(arr) > 0:
            fh.write("ir_inten_km_mol = ")
            fh.write(" ".join(f"{x:.6f}" for x in arr))
            fh.write("\n")

    chi = _get("chi_cm1", None)
    if chi is not None:
        if hasattr(chi, "tolist"):
            try:
                chi = chi.tolist()
            except Exception:
                pass
        fh.write("chi_cm1 = [\n")
        if isinstance(chi, (list, tuple)) and chi:
            if isinstance(chi[0], (list, tuple)) and len(chi[0]) == 3:
                for row in chi:
                    if len(row) < 3:
                        continue
                    try:
                        i, j, v = int(row[0]), int(row[1]), float(row[2])
                    except Exception:
                        continue
                    fh.write(f"{i:3d} {j:3d} {v: .8f}\n")
            else:
                n = len(chi)
                for i in range(n):
                    row = chi[i]
                    if not isinstance(row, (list, tuple)):
                        continue
                    for j in range(min(i + 1, len(row))):
                        try:
                            v = float(row[j])
                        except Exception:
                            continue
                        fh.write(f"{i+1:3d} {j+1:3d} {v: .8f}\n")
        fh.write("]\n")


# ============================================================
# Public API
# ============================================================
def update_enrichment_sections_in_file(
    xyzin_path: str,
    rot=None,
    vib: Optional[Dict] = None,
    qcent: Optional[Dict] = None,
    Q_rot: Optional[float] = None,
):
    with open(xyzin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract reference conditions
    T_K, P_atm = _extract_TP_from_basic(lines)

    # Extract DeltaBvib BEFORE stripping
    deltabvib_lines = _extract_deltabvib_lines(lines)

    # Strip old sections
    lines = _strip_sections(
        lines,
        section_names=(
            "ROTATIONAL",
            "VIBRATIONAL",
            "QCENT",
            "DELTABVIB",
            "THERMO",
        ),
    )

    with open(xyzin_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l if l.endswith("\n") else l + "\n")

        _write_rotational_block(
            f,
            rot=rot,
            T_K=T_K,
            P_atm=P_atm,
            Q_rot=Q_rot if Q_rot is not None else 1.0,
            deltabvib_lines=deltabvib_lines,
            qcent=qcent,
        )
        _write_vibrational_block(f, vib=vib)
