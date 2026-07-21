"""
rovib_pipeline.py
=================

Rovibrational DOS and Q(T) pipeline based on:
- Vibrational DOS (from vib_anh.py)
- Rotational energy levels (thermo_rot-like model)
- Convolution in log space
"""

from pathlib import Path
from typing import Dict, Optional

from .vib_anh import (
    read_dos_binned,
    read_rotational_block,
    _rotor_kind,
    _coerce_float,
    _coerce_int,
    rot_dos_logg,
    convolve_log_dos,
    write_dos,
    q_from_dos,
)


def _parse_basic_T(xyzin_path: str) -> float:
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
                    return float(v)
    raise ValueError("rovib_pipeline: missing T_K in #BASIC")


def rovib_pipeline(
    xyzin: str,
    vib_dos: Optional[str] = None,
    out: Optional[str] = None,
    rot_out: Optional[str] = None,
    q_out: Optional[str] = None,
    t_k: Optional[float] = None,
    emax_rot: Optional[float] = None,
    jmax: Optional[int] = None,
    sigma: Optional[int] = None,
    rotor_type: Optional[str] = None,
    A_MHz: Optional[float] = None,
    B_MHz: Optional[float] = None,
    C_MHz: Optional[float] = None,
) -> Dict:
    xyzin_path = Path(xyzin)
    if xyzin_path.is_dir():
        raise IsADirectoryError(
            f"rovib_pipeline expects xyzin FILE, got directory: {xyzin_path}"
        )

    workdir = xyzin_path.parent
    vib_dos = str(workdir / "dos_vib.dat") if vib_dos is None else vib_dos
    out = str(workdir / "dos_rovib.dat") if out is None else out
    q_out = str(workdir / "rovib_qt.dat") if q_out is None else q_out

    vib_dos_b, vib_emin, vib_bin = read_dos_binned(vib_dos)
    if not vib_dos_b:
        raise ValueError("rovib_pipeline: vib DOS is empty")

    energies = sorted(vib_dos_b.keys())
    vib_emax = vib_emin + (energies[-1] + 0.5) * vib_bin

    rot_block = read_rotational_block(str(xyzin_path))
    rotor_type_eff = rotor_type or rot_block.get("rotor_type")
    rotor_kind = _rotor_kind(rotor_type_eff)
    if rotor_kind is None:
        raise ValueError("rovib_pipeline: rotor_type not found")

    A = A_MHz if A_MHz is not None else _coerce_float(rot_block.get("a_mhz"))
    B = B_MHz if B_MHz is not None else _coerce_float(rot_block.get("b_mhz"))
    C = C_MHz if C_MHz is not None else _coerce_float(rot_block.get("c_mhz"))
    if A is None or B is None or C is None:
        raise ValueError("rovib_pipeline: missing rotational constants")

    sigma_eff = sigma
    if sigma_eff is None:
        sigma_eff = _coerce_int(rot_block.get("symm. number"))
    if sigma_eff is None:
        sigma_eff = _coerce_int(rot_block.get("sigma"))
    if sigma_eff is None:
        sigma_eff = 1

    if emax_rot is None and jmax is None:
        emax_rot = vib_emax

    rot_logg = rot_dos_logg(
        rotor_kind,
        A,
        B,
        C,
        sigma_eff,
        emax_rot,
        vib_bin,
        jmax=jmax,
    )

    if rot_out:
        write_dos(rot_out, rot_logg, 0.0, vib_bin)

    rovib_logg = convolve_log_dos(vib_dos_b, rot_logg)
    write_dos(out, rovib_logg, vib_emin, vib_bin)

    if t_k is None:
        t_k = _parse_basic_T(str(xyzin_path))
    q_rovib = q_from_dos({vib_emin + (b + 0.5) * vib_bin: lg for b, lg in rovib_logg.items()}, t_k)

    if q_out:
        Path(q_out).write_text(
            "# T_K Q_rovib\n" + f"{t_k:.6f} {q_rovib:.12e}\n",
            encoding="utf-8",
        )

    return {
        "dos_vib": vib_dos,
        "dos_rot": rot_out,
        "dos_rovib": out,
        "Q_rovib": q_rovib,
        "T_K": t_k,
        "bin_cm1": vib_bin,
        "vib_emin_cm1": vib_emin,
        "vib_emax_cm1": vib_emax,
    }
