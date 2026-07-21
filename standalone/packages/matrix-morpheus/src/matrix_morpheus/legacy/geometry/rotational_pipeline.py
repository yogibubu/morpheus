import numpy as np
import re
from pathlib import Path
from types import SimpleNamespace

from .physical_constants import Phy, get_physical_constants
from .inertia import principal_moments, inertia_tensor
from .axis_representation import apply_axis_representation
from .structure import Structure
from .symmetry_number import rotational_symmetry_number
from .symm_from_geometry import symm_and_point_group_from_geometry
from .structure_writer import update_enrichment_sections_in_file
from .thermo_rot import thermo_rot
from .vibrational import vib_from_xyzin
from .qcent import compute_qcent_from_xyzin
from .coriolis import run_coriolis_from_vibin


# ============================================================
# Rotational constants
# ============================================================
def rotational_constants_MHz(
    structure: Structure,
    isotopic: bool = True,
):
    """
    Compute rotational constants A, B, C in MHz.

    Returned order is ALWAYS A >= B >= C.
    """
    pc = get_physical_constants()

    # principal moments in amu Å^2
    I = principal_moments(structure, isotopic=isotopic)

    # physical constants
    h = pc[Phy.PLANCK]          # J s
    amu_to_kg = pc[Phy.TO_KG]   # kg / amu
    ang_to_m = 1.0e-10          # m / Å

    Bvals = []
    for Ii in I:
        Ii_SI = Ii * amu_to_kg * ang_to_m**2
        if Ii_SI <= 0.0:
            Bi = 0.0
        else:
            Bi = h / (8.0 * np.pi**2 * Ii_SI) * 1.0e-6  # MHz
        Bvals.append(Bi)

    # Sort so that A >= B >= C
    A, B, C = sorted(Bvals, reverse=True)
    return float(A), float(B), float(C)


# ============================================================
# Ray asymmetry parameter
# ============================================================
def ray_asymmetry(A: float, B: float, C: float) -> float:
    """
    Ray's asymmetry parameter κ.

    κ = (2B - A - C) / (A - C)

    Defined for non-linear rotors.
    """
    if abs(A - C) < 1.0e-12:
        return 0.0
    return (2.0 * B - A - C) / (A - C)


# ============================================================
# Rotor classification
# ============================================================
def classify_rotor(
    A: float,
    B: float,
    C: float,
    eps_zero: float = 1.0e-6,
    eps_rel: float = 1.0e-3,
):
    """
    Classify rotor type from rotational constants (MHz).
    """
    if C < eps_zero:
        return "linear"

    if abs(A - B) / A < eps_rel and abs(B - C) / B < eps_rel:
        return "spherical"

    if abs(B - C) / B < eps_rel:
        return "symmetric_prolate"

    if abs(A - B) / A < eps_rel:
        return "symmetric_oblate"

    kappa = ray_asymmetry(A, B, C)
    return "asymmetric_prolate" if kappa < 0.0 else "asymmetric_oblate"


# ============================================================
# Effective constants helpers
# ============================================================
def effective_B(A: float, B: float, C: float):
    """
    Effective B constant (MHz) for quasi-linear or approximate treatments.
    """
    return max(B, C)


# ============================================================
# High-level convenience wrapper (PIPELINE-AUTHORITATIVE)
# ============================================================
def rotational_info(
    structure: Structure,
    isotopic: bool = True,
):
    """
    Compute rotational constants and classify the rotor.

    Returns a dict SAFE for the full pipeline.
    """
    A, B, C = rotational_constants_MHz(structure, isotopic=isotopic)

    rotor_type = classify_rotor(A, B, C)
    kappa = ray_asymmetry(A, B, C)
    Beff = effective_B(A, B, C)

    linear = (rotor_type == "linear")

    # Default representation (may be overridden downstream)
    representation = "Ir"

    # Default symmetry number (may be overridden)
    sigma = 1

    return {
        # canonical keys
        "A": A,
        "B": B,
        "C": C,

        # explicit MHz aliases (pipeline expects these)
        "A_MHz": A,
        "B_MHz": B,
        "C_MHz": C,

        # classification
        "rotor_type": rotor_type,
        "kappa": kappa,
        "Beff": Beff,
        "linear": linear,

        # pipeline-required metadata
        "representation": representation,
        "sigma": sigma,
    }


# ============================================================
# XYzin helpers
# ============================================================
def _read_basic_kv(xyzin_path: str):
    data = {}
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
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                if len(parts) >= 3:
                    k = " ".join(parts[:-1])
                    v = parts[-1]
                else:
                    k, v = parts

            data[k.strip().upper()] = v.strip()

    return data


def _parse_float_d(token: str):
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except Exception:
        return None


def _read_vibrational_extras_from_xyzin(xyzin_path: str):
    """
    Read extra vibrational data from #VIBRATIONAL:
    - ir_inten_km_mol (list)
    - chi_cm1 (list of (i,j,val))
    """
    vib_lines = []
    in_vib = False
    with open(xyzin_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#"):
                in_vib = (s.split()[0].upper() == "#VIBRATIONAL")
                continue
            if in_vib:
                vib_lines.append(s)

    if not vib_lines:
        return None

    out = {}

    def _grab_numbers(s):
        return re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", s)

    i = 0
    while i < len(vib_lines):
        s = vib_lines[i].strip()
        low = s.lower()

        if low.startswith("ir_inten_km_mol") or low.startswith("ir_inten"):
            nums = _grab_numbers(s)
            j = i + 1
            while j < len(vib_lines):
                nxt = vib_lines[j].strip()
                if re.match(r"^[A-Za-z_][A-Za-z0-9_\\-]*\\s*[:=]", nxt) or nxt.lower().startswith("chi_cm1"):
                    break
                nums += _grab_numbers(nxt)
                j += 1
            arr = []
            for n in nums:
                v = _parse_float_d(n)
                if v is not None:
                    arr.append(v)
            if arr:
                out["ir_inten_km_mol"] = arr
            i = j
            continue

        if low.startswith("chi_cm1"):
            chi = []
            # find opening bracket
            if "[" not in s:
                i += 1
            else:
                i += 1
            while i < len(vib_lines):
                row = vib_lines[i].strip()
                if not row:
                    i += 1
                    continue
                if "]" in row:
                    i += 1
                    break
                parts = _grab_numbers(row)
                if len(parts) >= 3:
                    a = _parse_float_d(parts[0])
                    b = _parse_float_d(parts[1])
                    c = _parse_float_d(parts[2])
                    if a is not None and b is not None and c is not None:
                        chi.append((int(a), int(b), float(c)))
                i += 1
            if chi:
                out["chi_cm1"] = chi
            continue

        i += 1

    return out if out else None


def _read_xyz_from_xyzin(xyzin_path: str):
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


def _safe_sigma(point_group: str) -> int:
    try:
        return int(rotational_symmetry_number(point_group))
    except Exception:
        return 1


def _oriented_coords_for_symmetry(structure: Structure, representation: str, isotopic: bool = True):
    """
    Orient coordinates along principal axes and apply axis representation.
    """
    coords = np.array(structure.coords, dtype=float)
    masses = structure.mass_isotope if isotopic else structure.mass_average
    m = np.array(masses, dtype=float)
    M = float(np.sum(m))
    if M <= 0.0:
        return coords

    com = np.sum(coords * m[:, None], axis=0) / M
    coords0 = coords - com[None, :]

    I = inertia_tensor(structure, isotopic=isotopic)
    w, V = np.linalg.eigh(I)
    idx = np.argsort(w)
    V = V[:, idx]
    if np.linalg.det(V) < 0.0:
        V[:, 2] *= -1.0

    # Apply the requested axis representation to oriented coords
    try:
        return apply_axis_representation(coords0, V, representation)
    except Exception:
        return coords0 @ V


def _update_basic_kv(xyzin_path: str, updates: dict):
    """
    Update or insert key/value lines inside #BASIC.
    """
    with open(xyzin_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "#BASIC")
    except StopIteration:
        lines.append("#BASIC")
        i0 = len(lines) - 1

    i = i0 + 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
        i += 1

    block = lines[i0 + 1:i]
    updated = []
    seen = set()

    for line in block:
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        key_norm = "POINT_GROUP" if key == "GROUP" else key
        if key_norm in updates:
            val = updates[key_norm]
            updated.append(f"{key_norm:16s} {val}")
            seen.add(key_norm)
        else:
            updated.append(line)

    for k, v in updates.items():
        if k not in seen:
            updated.append(f"{k:16s} {v}")

    out = lines[:i0 + 1] + updated + lines[i:]
    with open(xyzin_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _write_rotational_report(path: Path, rot_info: dict, vib, qcent, Q_rot, rot_thermo):
    lines = []
    lines.append("ROTATIONAL PIPELINE REPORT\n")
    lines.append("ROTATIONAL CONSTANTS (MHz)\n")
    lines.append(f"A = {rot_info.get('A_MHz', rot_info.get('A'))}\n")
    lines.append(f"B = {rot_info.get('B_MHz', rot_info.get('B'))}\n")
    lines.append(f"C = {rot_info.get('C_MHz', rot_info.get('C'))}\n")
    lines.append(f"rotor_type = {rot_info.get('rotor_type')}\n")
    lines.append(f"kappa = {rot_info.get('kappa')}\n")
    lines.append(f"representation = {rot_info.get('representation')}\n")
    lines.append(f"sigma = {rot_info.get('sigma')}\n")
    if rot_info.get("point_group"):
        lines.append(f"point_group = {rot_info.get('point_group')}\n")

    if Q_rot is not None:
        lines.append("\nROTATIONAL PARTITION FUNCTION\n")
        lines.append(f"Q_rot = {Q_rot}\n")

    if vib is not None:
        lines.append("\nVIBRATIONAL SUMMARY\n")
        lines.append(f"nvib = {vib.get('nvib')}\n")
        lines.append(f"n_imag_like = {vib.get('n_imag_like')}\n")
        lines.append(f"linear = {vib.get('linear')}\n")

    if qcent is not None:
        lines.append("\nQCENT SUMMARY\n")
        lines.append(f"representation = {qcent.get('representation')}\n")
        lines.append(f"linear = {qcent.get('linear')}\n")

    if rot_thermo is not None:
        lines.append("\nROTATIONAL THERMO (from thermo_rot)\n")
        for k in ("U", "H", "S", "Cv", "Cp"):
            if k in rot_thermo:
                lines.append(f"{k} = {rot_thermo[k]}\n")

    path.write_text("".join(lines), encoding="utf-8")


def _write_vibrational_report(path: Path, vib: dict, coriolis_entries=None):
    lines = []
    lines.append("VIBRATIONAL REPORT\n")
    if vib is None:
        lines.append("No vibrational data available.\n")
        path.write_text("".join(lines), encoding="utf-8")
        return

    linear = vib.get("linear", None)
    nvib = vib.get("nvib", None)
    n_imag = vib.get("n_imag_like", None)
    if linear is not None:
        lines.append(f"linear = {linear}\n")
    if nvib is not None:
        lines.append(f"nvib = {nvib}\n")
    if n_imag is not None:
        lines.append(f"n_imag_like = {n_imag}\n")

    freq = vib.get("freq_cm1", None)
    if freq is not None:
        lines.append("\nFrequencies (cm^-1)\n")
        try:
            for i, v in enumerate(freq, 1):
                lines.append(f"{i:4d} {float(v): .8f}\n")
        except Exception:
            lines.append("Frequencies present but unreadable.\n")

    ir = vib.get("ir_inten_km_mol") or vib.get("ir_inten")
    if ir is not None:
        lines.append("\nIR Intensities (KM/Mole)\n")
        try:
            for i, v in enumerate(ir, 1):
                lines.append(f"{i:4d} {float(v): .8f}\n")
        except Exception:
            lines.append("IR intensities present but unreadable.\n")

    chi = vib.get("chi_cm1")
    if chi is not None:
        lines.append("\nChi Matrix (cm^-1, lower triangle)\n")
        try:
            if isinstance(chi, (list, tuple)) and chi and len(chi[0]) == 3:
                for i, j, v in chi:
                    lines.append(f"{int(i):3d} {int(j):3d} {float(v): .8f}\n")
            elif isinstance(chi, (list, tuple)):
                n = len(chi)
                for i in range(n):
                    row = chi[i]
                    if not isinstance(row, (list, tuple)):
                        continue
                    for j in range(min(i + 1, len(row))):
                        lines.append(f"{i+1:3d} {j+1:3d} {float(row[j]): .8f}\n")
        except Exception:
            lines.append("Chi matrix present but unreadable.\n")

    if coriolis_entries:
        lines.append("\nCoriolis (sparse, |Geff_cm1| >= threshold)\n")
        lines.append(" i    j   -k        zeta            Geff(cm^-1)          Geff(MHz)\n")
        for d in coriolis_entries:
            try:
                lines.append(
                    f"{int(d['i']):4d} {int(d['j']):4d} {int(d['kneg']):4d} "
                    f"{float(d['zeta']):16.8e} "
                    f"{float(d['Geff_cm1']):16.8e} "
                    f"{float(d['Geff_MHz']):16.8e}\n"
                )
            except Exception:
                continue

    path.write_text("".join(lines), encoding="utf-8")


# ============================================================
# Rotational pipeline (authoritative)
# ============================================================
def rotational_pipeline(xyzin, report=True):
    xyzin_path = Path(xyzin)
    if xyzin_path.is_dir():
        raise IsADirectoryError(
            f"rotational_pipeline expects xyzin FILE, got directory: {xyzin_path}"
        )

    # --------------------------------------------------------
    # Read structure
    # --------------------------------------------------------
    symbols, coords = _read_xyz_from_xyzin(str(xyzin_path))
    mol = Structure(
        symbols=symbols,
        coords=coords,
        isotopes=None,
    )

    # --------------------------------------------------------
    # Basic metadata
    # --------------------------------------------------------
    basic = _read_basic_kv(str(xyzin_path))
    representation = basic.get("REPRESENTATION", "Ir") or "Ir"
    point_group = basic.get("POINT_GROUP") or basic.get("GROUP") or "C1"

    watson_reduction = (
        basic.get("WATSON REDUCTION")
        or basic.get("WATSON_REDUCTION")
        or None
    )

    sigma = _safe_sigma(point_group)

    # --------------------------------------------------------
    # Rotational info
    # --------------------------------------------------------
    rot_info = rotational_info(mol, isotopic=True)
    rot_info["representation"] = representation

    # --------------------------------------------------------
    # Symmetry from geometry (override default C1 when possible)
    # --------------------------------------------------------
    symm_sigma = None
    symm_pg = None
    try:
        coords_oriented = _oriented_coords_for_symmetry(mol, representation, isotopic=True)
        symm_sigma, symm_pg = symm_and_point_group_from_geometry(
            mol.symbols,
            coords_oriented,
            rot_info["rotor_type"],
            representation,
            tol=5.0e-3,
        )
    except Exception:
        symm_sigma, symm_pg = None, None

    if (not point_group) or point_group.upper() == "C1":
        if symm_pg is not None:
            point_group = str(symm_pg)
        if symm_sigma is not None:
            sigma = int(symm_sigma)
        else:
            sigma = _safe_sigma(point_group)
    else:
        sigma = _safe_sigma(point_group)

    rot_info["sigma"] = sigma
    rot_info["point_group"] = point_group

    # Keep BASIC aligned with computed symmetry
    try:
        _update_basic_kv(str(xyzin_path), {"POINT_GROUP": point_group})
    except Exception:
        pass
    if watson_reduction:
        rot_info["watson_reduction"] = watson_reduction

    rot_obj = SimpleNamespace(
        A_MHz=rot_info["A_MHz"],
        B_MHz=rot_info["B_MHz"],
        C_MHz=rot_info["C_MHz"],
        rotor_type=rot_info["rotor_type"],
        kappa=rot_info["kappa"],
        representation=rot_info["representation"],
        sigma=rot_info["sigma"],
        linear=rot_info["linear"],
        point_group=point_group,
    )

    if watson_reduction:
        rot_obj.watson_reduction = watson_reduction

    # --------------------------------------------------------
    # Vibrational enrichment (from fchkin, if available)
    # --------------------------------------------------------
    vib = vib_from_xyzin(str(xyzin_path), structure=mol)
    if vib is None:
        nvib = 3 * mol.natoms - (5 if rot_info["linear"] else 6)
        vib = {
            "linear": bool(rot_info["linear"]),
            "nvib": max(int(nvib), 0),
            "n_imag_like": 0,
        }
    else:
        vib.setdefault("linear", bool(rot_info["linear"]))
        vib.setdefault("nvib", int(vib.get("freq_cm1", []).__len__()))
        vib.setdefault("n_imag_like", int(vib.get("n_imag_like", 0)))

    vib_extras = _read_vibrational_extras_from_xyzin(str(xyzin_path))
    if vib_extras:
        if vib is None:
            vib = {}
        for k, v in vib_extras.items():
            if k not in vib or vib[k] is None:
                vib[k] = v

    if isinstance(vib, dict):
        dip = vib.get("dipole_oriented_debye") or vib.get("dipole_debye")
        if dip is not None:
            rot_obj.dipole_oriented_debye = dip

    # --------------------------------------------------------
    # QCENT (optional)
    # --------------------------------------------------------
    qcent = compute_qcent_from_xyzin(str(xyzin_path), workdir=str(xyzin_path.parent))

    # --------------------------------------------------------
    # Coriolis (optional, after QCENT)
    # --------------------------------------------------------
    coriolis_entries = None
    if isinstance(vib, dict) and vib.get("freq_cm1") is not None:
        vibin_path = xyzin_path.parent / "vibin"
        if vibin_path.exists():
            try:
                coriolis_entries = run_coriolis_from_vibin(
                    vibin_path=str(vibin_path),
                    A=rot_info["A_MHz"],
                    B=rot_info["B_MHz"],
                    C=rot_info["C_MHz"],
                    units="MHz",
                )
            except Exception:
                coriolis_entries = None

    # --------------------------------------------------------
    # Rotational partition function (optional, from thermo_rot)
    # --------------------------------------------------------
    Q_rot = None
    rot_thermo = None
    try:
        rot_thermo = thermo_rot(
            str(xyzin_path),
            rot_info["A"], rot_info["B"], rot_info["C"],
            rot_info["rotor_type"],
            sigma=rot_info.get("sigma"),
        )
        Q_rot = rot_thermo.get("Q_dimless")
    except Exception:
        Q_rot = None
        rot_thermo = None

    # --------------------------------------------------------
    # Update xyzin enrichment sections
    # --------------------------------------------------------
    update_enrichment_sections_in_file(
        str(xyzin_path),
        rot=rot_obj,
        vib=vib,
        qcent=qcent,
        Q_rot=Q_rot,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    if report:
        report_path = xyzin_path.parent / "rotational.report"
        _write_rotational_report(report_path, rot_info, vib, qcent, Q_rot, rot_thermo)
        if isinstance(vib, dict) and vib.get("freq_cm1") is not None:
            vib_report_path = xyzin_path.parent / "vibrational.report"
            _write_vibrational_report(vib_report_path, vib, coriolis_entries=coriolis_entries)

    return {
        "structure": mol,
        "rot": rot_info,
        "vib": vib,
        "qcent": qcent,
        "Q_rot": Q_rot,
    }
