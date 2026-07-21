#!/usr/bin/env python3
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

# Physical constant: (h*c)/kB in cm*K
HC_OVER_KB = 1.438776877
# MHz to cm^-1
MHZ_TO_CM1 = 1.0e6 / (2.99792458e10)


# ============================================================
# Basic helpers
# ============================================================
def _coerce_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace("D", "E").replace("d", "e"))
    except Exception:
        return None


def _coerce_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return int(x)
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _rotor_kind(rotor_type: Optional[str]) -> Optional[str]:
    if not rotor_type:
        return None
    s = str(rotor_type).strip().lower()
    if "linear" in s:
        return "linear"
    if "spherical" in s:
        return "spherical"
    if "symmetric" in s:
        return "symmetric"
    if "asymmetric" in s:
        return "asymmetric"
    return None


def _grab_numbers(s: str) -> List[str]:
    return re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", s)


# ============================================================
# XYzin readers
# ============================================================
def read_rotational_block(xyzin_path: str) -> Dict[str, str]:
    """
    Parse #ROTATIONAL block into a dict of lowercase keys.
    """
    lines = Path(xyzin_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    in_rot = False
    out: Dict[str, str] = {}
    for line in lines:
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
        else:
            parts = s.split()
            if len(parts) < 2:
                continue
            if len(parts) >= 3:
                k = " ".join(parts[:-1])
                v = parts[-1]
            else:
                k, v = parts

        key = re.sub(r"\s+", " ", k.strip().lower())
        out[key] = v.strip()

    return out


def read_xyzin(xyzin_path: str):
    """
    Read #VIBRATIONAL block from xyzin and return a minimal object with:
    - omega_cm1: list of harmonic frequencies (cm^-1)
    - chi_cm1:   NxN matrix of anharmonic constants (cm^-1)
    """
    lines = Path(xyzin_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    in_vib = False
    vib_lines: List[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            in_vib = (s.split()[0].upper() == "#VIBRATIONAL")
            continue
        if in_vib:
            vib_lines.append(s)

    freqs: List[float] = []
    chi_list: List[Tuple[int, int, float]] = []

    i = 0
    while i < len(vib_lines):
        s = vib_lines[i].strip()
        if not s:
            i += 1
            continue
        low = s.lower()

        if low.startswith("freq") or low.startswith("frequencies"):
            nums = _grab_numbers(s)
            j = i + 1
            while j < len(vib_lines):
                nxt = vib_lines[j].strip()
                if not nxt:
                    j += 1
                    continue
                if re.match(r"^[A-Za-z_][A-Za-z0-9_\\-]*\\s*[:=]", nxt) or nxt.startswith("#"):
                    break
                if nxt.lower().startswith("chi_cm1"):
                    break
                nums += _grab_numbers(nxt)
                j += 1
            for n in nums:
                v = _coerce_float(n)
                if v is not None:
                    freqs.append(v)
            i = j
            continue

        if low.startswith("chi_cm1"):
            # expect a block within [ ... ]
            if "[" in s:
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
                    a = _coerce_int(parts[0])
                    b = _coerce_int(parts[1])
                    c = _coerce_float(parts[2])
                    if a is not None and b is not None and c is not None:
                        chi_list.append((a, b, c))
                i += 1
            continue

        i += 1

    n = len(freqs)
    chi = [[0.0 for _ in range(n)] for _ in range(n)]
    for a, b, c in chi_list:
        if 1 <= a <= n and 1 <= b <= n:
            ia = a - 1
            ib = b - 1
            chi[ia][ib] = float(c)
            chi[ib][ia] = float(c)

    return SimpleNamespace(omega_cm1=freqs, chi_cm1=chi)


# ============================================================
# DOS utilities
# ============================================================
def direct_sum_dos(
    omega_cm1: List[float],
    chi_cm1: List[List[float]],
    vmax: Iterable[int],
    emax_cm1: float,
    bin_cm1: float,
    ncap: Optional[Iterable[float]] = None,
) -> Dict[int, float]:
    """
    Direct summation of vibrational DOS.
    Returns counts per bin (NOT log).
    """
    n = len(omega_cm1)
    if n == 0:
        return {}

    vmax_list = list(vmax) if not isinstance(vmax, (int, float)) else [int(vmax)] * n
    ncap_list: List[float]
    if ncap is None:
        ncap_list = [0.0] * n
    else:
        ncap_list = list(ncap) if not isinstance(ncap, (int, float)) else [float(ncap)] * n

    if not chi_cm1 or len(chi_cm1) != n:
        chi = [[0.0 for _ in range(n)] for _ in range(n)]
    else:
        chi = chi_cm1

    def v_eff(v: int, cap: float) -> float:
        if cap is None or cap <= 0:
            return float(v)
        # smooth saturation (erf)
        return cap * math.erf(float(v) / cap)

    def energy_cm1(vs: List[int]) -> float:
        ve = [v_eff(vs[i], ncap_list[i]) for i in range(n)]
        e = 0.0
        for i in range(n):
            e += omega_cm1[i] * ve[i]
        for i in range(n):
            for j in range(i, n):
                if chi[i][j] != 0.0:
                    e += chi[i][j] * ve[i] * ve[j]
        return e

    dos: Dict[int, float] = {}

    def rec(i: int, vlist: List[int], e_partial: float):
        if i == n:
            if e_partial < 0.0 or e_partial > emax_cm1:
                return
            b = int(e_partial // bin_cm1)
            dos[b] = dos.get(b, 0.0) + 1.0
            return

        for v in range(int(vmax_list[i]) + 1):
            vlist[i] = v
            e = energy_cm1(vlist)
            if e > emax_cm1:
                continue
            rec(i + 1, vlist, e)

    rec(0, [0] * n, 0.0)
    return dos


def write_dos(path: str, dos_logg: Dict[int, float], emin_cm1: float, bin_cm1: float):
    lines = ["# format: E_cm1 log_g\n"]
    for b in sorted(dos_logg.keys()):
        e = emin_cm1 + (b + 0.5) * bin_cm1
        lines.append(f"{e:.6f}  {dos_logg[b]:.12f}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def read_dos_binned(path: str) -> Tuple[Dict[int, float], float, float]:
    p = Path(path)
    if not p.exists():
        return {}, 0.0, 1.0
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    data: List[Tuple[float, float]] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        e = _coerce_float(parts[0])
        lg = _coerce_float(parts[1])
        if e is None or lg is None:
            continue
        data.append((e, lg))

    if not data:
        return {}, 0.0, 1.0

    data.sort(key=lambda x: x[0])
    if len(data) >= 2:
        bin_cm1 = max(1.0e-12, data[1][0] - data[0][0])
    else:
        bin_cm1 = 1.0
    emin = data[0][0] - 0.5 * bin_cm1

    dos: Dict[int, float] = {}
    for e, lg in data:
        b = int(round((e - emin) / bin_cm1 - 0.5))
        dos[b] = lg

    return dos, emin, bin_cm1


def _logsumexp(a: Optional[float], b: float) -> float:
    if a is None:
        return b
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def convolve_log_dos(dos1_logg: Dict[int, float], dos2_logg: Dict[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for b1, lg1 in dos1_logg.items():
        for b2, lg2 in dos2_logg.items():
            b = b1 + b2
            out[b] = _logsumexp(out.get(b), lg1 + lg2)
    return out


def q_from_dos(dos_e_logg: Dict[float, float], t_k: float) -> float:
    if t_k <= 0:
        return 0.0
    beta = HC_OVER_KB / t_k
    acc: Optional[float] = None
    for e, lg in dos_e_logg.items():
        acc = _logsumexp(acc, lg - beta * float(e))
    if acc is None:
        return 0.0
    return math.exp(acc)


# ============================================================
# Rotational DOS (log)
# ============================================================
def rot_dos_logg(
    rotor_kind: str,
    A_MHz: float,
    B_MHz: float,
    C_MHz: float,
    sigma: int,
    emax_cm1: float,
    bin_cm1: float,
    jmax: Optional[int] = None,
) -> Dict[int, float]:
    """
    Build rotational DOS (log_g) in bins of bin_cm1.
    """
    A = float(A_MHz) * MHZ_TO_CM1
    B = float(B_MHz) * MHZ_TO_CM1
    C = float(C_MHz) * MHZ_TO_CM1
    sigma_eff = max(1, int(sigma)) if sigma else 1

    kind = _rotor_kind(rotor_kind) or rotor_kind
    kind = kind.lower()

    if kind == "linear":
        beff = max(B, C)
        if beff <= 0.0:
            return {}
        if jmax is None:
            jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / beff) - 1.0) / 2.0) + 1
        counts: Dict[int, float] = {}
        for J in range(jmax + 1):
            e = beff * J * (J + 1)
            if e > emax_cm1:
                break
            g = (2 * J + 1) / sigma_eff
            b = int(e // bin_cm1)
            counts[b] = counts.get(b, 0.0) + g
        return {b: math.log(g) for b, g in counts.items() if g > 0}

    if kind == "spherical":
        beff = max(A, B, C)
        if beff <= 0.0:
            return {}
        if jmax is None:
            jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / beff) - 1.0) / 2.0) + 1
        counts = {}
        for J in range(jmax + 1):
            e = beff * J * (J + 1)
            if e > emax_cm1:
                break
            g = ((2 * J + 1) ** 2) / sigma_eff
            b = int(e // bin_cm1)
            counts[b] = counts.get(b, 0.0) + g
        return {b: math.log(g) for b, g in counts.items() if g > 0}

    # symmetric/asymmetric: approximate as symmetric top with Beff
    Aeff = max(A, B, C)
    Beff = max(B, C)
    if Beff <= 0.0:
        return {}
    if jmax is None:
        jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / Beff) - 1.0) / 2.0) + 1

    counts = {}
    for J in range(jmax + 1):
        base = Beff * J * (J + 1)
        if base > emax_cm1:
            break
        for K in range(J + 1):
            e = base + (Aeff - Beff) * (K ** 2)
            if e > emax_cm1:
                continue
            g = (2 * J + 1) * (1.0 if K == 0 else 2.0)
            g /= sigma_eff
            b = int(e // bin_cm1)
            counts[b] = counts.get(b, 0.0) + g

    return {b: math.log(g) for b, g in counts.items() if g > 0}


# ============================================================
# CLI (minimal)
# ============================================================
def _build_energy_logg_from_bins(dos_logg: Dict[int, float], bin_cm1: float, emin_cm1: float) -> Dict[float, float]:
    return {emin_cm1 + (b + 0.5) * bin_cm1: lg for b, lg in dos_logg.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="vib_anh utilities (minimal)")
    parser.add_argument("xyzin", help="xyzin path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_direct = sub.add_parser("direct", help="Direct DOS from #VIBRATIONAL")
    p_direct.add_argument("--vmax", type=int, default=6)
    p_direct.add_argument("--emax", type=float, default=8000.0)
    p_direct.add_argument("--bin", dest="bin_cm1", type=float, default=50.0)
    p_direct.add_argument("--ncap", type=float, default=10.0)
    p_direct.add_argument("--out", default="dos_vib.dat")

    p_conv = sub.add_parser("convrot", help="Convolve vib DOS with rotational DOS")
    p_conv.add_argument("--vib-dos", required=True)
    p_conv.add_argument("--emax-rot", type=float, default=None)
    p_conv.add_argument("--jmax", type=int, default=None)
    p_conv.add_argument("--out", default="dos_rovib.dat")
    p_conv.add_argument("--rot-out", default=None)

    p_q = sub.add_parser("q", help="Partition function from DOS")
    p_q.add_argument("--dos", required=True)
    p_q.add_argument("--t", type=float, required=True)

    args = parser.parse_args()

    if args.cmd == "direct":
        vib = read_xyzin(args.xyzin)
        vmax = [args.vmax] * len(vib.omega_cm1)
        ncap = [args.ncap] * len(vib.omega_cm1)
        dos = direct_sum_dos(vib.omega_cm1, vib.chi_cm1, vmax, args.emax, args.bin_cm1, ncap)
        dos_logg = {b: math.log(c) for b, c in dos.items() if c > 0}
        write_dos(args.out, dos_logg, 0.0, args.bin_cm1)
        return

    if args.cmd == "convrot":
        dos_vib, emin, bin_cm1 = read_dos_binned(args.vib_dos)
        rot_block = read_rotational_block(args.xyzin)
        rk = _rotor_kind(rot_block.get("rotor_type"))
        if rk is None:
            raise SystemExit("rotor_type not found in #ROTATIONAL")
        A = _coerce_float(rot_block.get("a_mhz"))
        B = _coerce_float(rot_block.get("b_mhz"))
        C = _coerce_float(rot_block.get("c_mhz"))
        if A is None or B is None or C is None:
            raise SystemExit("missing rotational constants in #ROTATIONAL")
        sigma = _coerce_int(rot_block.get("symm. number")) or _coerce_int(rot_block.get("sigma")) or 1
        emax_rot = args.emax_rot if args.emax_rot is not None else (emin + 100 * bin_cm1)
        rot_logg = rot_dos_logg(rk, A, B, C, sigma, emax_rot, bin_cm1, jmax=args.jmax)
        if args.rot_out:
            write_dos(args.rot_out, rot_logg, 0.0, bin_cm1)
        rovib_logg = convolve_log_dos(dos_vib, rot_logg)
        write_dos(args.out, rovib_logg, emin, bin_cm1)
        return

    if args.cmd == "q":
        dos_vib, emin, bin_cm1 = read_dos_binned(args.dos)
        dos_e = _build_energy_logg_from_bins(dos_vib, bin_cm1, emin)
        q = q_from_dos(dos_e, args.t)
        print(f"{q:.12e}")
        return


if __name__ == "__main__":
    main()
