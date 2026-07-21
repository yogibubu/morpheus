import sys
from pathlib import Path
import numpy as np

from .structure import Structure
from .rotational import rotational_info
from .thermo_trasl import thermo_trasl
from .thermo_rot import thermo_rot
from .thermo_vib import thermo_vib
from .thermo_writer import write_thermo


def read_xyz_from_xyzin(xyzin):
    with open(xyzin, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    nat = int(lines[0].strip())
    comment = lines[1].rstrip("\n")
    xyz_lines = lines[2:2 + nat]
    tail_lines = lines[2 + nat:]

    symbols = []
    coords = []
    for line in xyz_lines:
        f = line.split()
        symbols.append(f[0])
        coords.append([float(f[1]), float(f[2]), float(f[3])])

    return nat, comment, symbols, np.array(coords, dtype=float), tail_lines


def _sum_dicts(dicts, keys):
    out = {}
    for k in keys:
        val = 0.0
        present = False
        for d in dicts:
            if d is not None and k in d and d[k] is not None:
                val += d[k]
                present = True
        out[k] = val if present else None
    return out


def _parse_rotational_block(xyzin_path: str):
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
            if not in_rot:
                continue
            if "=" in s:
                k, v = [x.strip() for x in s.split("=", 1)]
                data[k.strip().lower()] = v.strip()
    return data


def _coerce_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _coerce_int(val):
    try:
        return int(val)
    except Exception:
        return None


def _rotor_type_from_block(rot_block):
    raw = rot_block.get("rotor_type")
    if not raw:
        return None
    label = raw.strip().lower()
    mapping = {
        "linear_top": "linear",
        "spherical_top": "spherical",
        "symmetric_top_prolate": "symmetric_prolate",
        "symmetric_top_oblate": "symmetric_oblate",
        "asymmetric_top_quasi_prolate": "asymmetric_prolate",
        "asymmetric_top_quasi_oblate": "asymmetric_oblate",
    }
    return mapping.get(label, label)


def _write_thermo_report(path: Path, thermo: dict):
    lines = []
    lines.append("THERMO PIPELINE REPORT\n")
    lines.append("keys: Q_dimless U_kJmol H_kJmol S_JmolK Cv_JmolK Cp_JmolK\n")
    for label in ("trasl", "rot", "vib", "tot"):
        block = thermo.get(label)
        if block is None:
            continue
        lines.append(f"\n{label.upper()}\n")
        for key in ("Q_dimless", "U_kJmol", "H_kJmol", "S_JmolK", "Cv_JmolK", "Cp_JmolK"):
            if key in block and block[key] is not None:
                lines.append(f"{key} = {block[key]}\n")
    path.write_text("".join(lines), encoding="utf-8")


def thermo_pipeline(xyzin, report=True):
    xyzin = Path(xyzin)
    if xyzin.is_dir():
        raise IsADirectoryError(
            f"thermo_pipeline expects xyzin FILE, got directory: {xyzin}"
        )

    # ---------------------------------------------------------
    # Read structure
    # ---------------------------------------------------------
    nat, comment, symbols, coords, tail_lines = read_xyz_from_xyzin(xyzin)
    mol = Structure(
        symbols=symbols,
        coords=[tuple(x) for x in coords],
        isotopes=None,
    )

    # ---------------------------------------------------------
    # Translational contribution
    # ---------------------------------------------------------
    t = thermo_trasl(str(xyzin))

    # ---------------------------------------------------------
    # Rotational contribution (EXPLICIT, CLEAN)
    # ---------------------------------------------------------
    rot_block = _parse_rotational_block(str(xyzin))
    rotor_type = _rotor_type_from_block(rot_block)

    A_MHz = _coerce_float(rot_block.get("a_mhz"))
    B_MHz = _coerce_float(rot_block.get("b_mhz"))
    C_MHz = _coerce_float(rot_block.get("c_mhz"))

    if A_MHz is None and B_MHz is None and C_MHz is None:
        rot = rotational_info(mol, isotopic=True)
        A_MHz, B_MHz, C_MHz = rot["A"], rot["B"], rot["C"]
        rotor_type = rotor_type or rot["rotor_type"]
    else:
        if B_MHz is None:
            B_MHz = A_MHz if A_MHz is not None else C_MHz
        if A_MHz is None:
            A_MHz = B_MHz
        if C_MHz is None:
            C_MHz = B_MHz
        if rotor_type is None:
            rotor_type = rotational_info(mol, isotopic=True)["rotor_type"]

    sigma = _coerce_int(rot_block.get("symm. number"))
    if sigma is None:
        sigma = _coerce_int(rot_block.get("sigma"))

    r = thermo_rot(
        str(xyzin),
        A_MHz, B_MHz, C_MHz,
        rotor_type,
        sigma=sigma,
    )

    # ---------------------------------------------------------
    # Vibrational contribution
    # ---------------------------------------------------------
    try:
        v = thermo_vib(str(xyzin))
    except Exception:
        v = None

    # ---------------------------------------------------------
    # Totals
    # ---------------------------------------------------------
    keys_add = ("U_kJmol", "H_kJmol", "S_JmolK", "Cv_JmolK", "Cp_JmolK")
    tot = _sum_dicts([t, r, v], keys_add)

    Q_tot = 1.0
    for d in (t, r, v):
        if d is not None and d.get("Q_dimless") is not None:
            Q_tot *= float(d["Q_dimless"])
    tot["Q_dimless"] = Q_tot

    # ---------------------------------------------------------
    # Write output
    # ---------------------------------------------------------
    thermo = {
        "trasl": t,
        "rot": r,
        "vib": v,
        "tot": tot,
    }

    write_thermo(str(xyzin), thermo)

    if report:
        report_path = xyzin.parent / "thermo.report"
        _write_thermo_report(report_path, thermo)

    return thermo


def run_thermo_on_xyzin(xyzin, report=True):
    return thermo_pipeline(xyzin, report=report)


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    thermo_pipeline(sys.argv[1], report=True)


if __name__ == "__main__":
    main()
