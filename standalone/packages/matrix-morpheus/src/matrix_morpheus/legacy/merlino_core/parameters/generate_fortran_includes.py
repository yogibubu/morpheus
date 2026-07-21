from __future__ import annotations

from pathlib import Path

from .bdpcs3 import load_bdpcs3_parameters


def _d(value: float) -> str:
    text = f"{float(value):.12g}"
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e", 1)
        return f"{mantissa}D{int(exponent):+d}"
    if "." not in text:
        text += ".0"
    return f"{text}D0"


def build_bdpcs3_hbond_include() -> str:
    params = load_bdpcs3_parameters()
    hbond = params.hbond
    weights = params.weights
    lines = [
        "C Shared BDPCS3 H-bond and metric parameters.",
        "C Source of truth: merlino_core/parameters/bdpcs3_hbond.toml",
        "      DOUBLE PRECISION BDPCS3_HB_ANGLE_MIN",
        "      DOUBLE PRECISION BDPCS3_HB_DIST_CUTOFF",
        "      DOUBLE PRECISION BDPCS3_HB_DIST_WIDTH",
        "      DOUBLE PRECISION BDPCS3_HB_SEARCH_CUTOFF",
        "      DOUBLE PRECISION BDPCS3_HB_DELTA_OO",
        "      DOUBLE PRECISION BDPCS3_HB_DELTA_NN",
        "      DOUBLE PRECISION BDPCS3_W_STRETCH",
        "      DOUBLE PRECISION BDPCS3_W_ANGLE",
        "      DOUBLE PRECISION BDPCS3_W_HBOND",
        "      DOUBLE PRECISION BDPCS3_W_TORSION_MIN",
        "      DOUBLE PRECISION BDPCS3_W_FRAGMENT",
        f"      PARAMETER (BDPCS3_HB_ANGLE_MIN={_d(hbond.angle_threshold_deg)})",
        f"      PARAMETER (BDPCS3_HB_DIST_CUTOFF={_d(hbond.distance_cutoff_ang)})",
        f"      PARAMETER (BDPCS3_HB_DIST_WIDTH={_d(hbond.distance_width_ang)})",
        f"      PARAMETER (BDPCS3_HB_SEARCH_CUTOFF={_d(hbond.search_cutoff_ang)})",
        f"      PARAMETER (BDPCS3_HB_DELTA_OO={_d(hbond.correction_ang(8, 8))})",
        f"      PARAMETER (BDPCS3_HB_DELTA_NN={_d(hbond.correction_ang(7, 7))})",
        f"      PARAMETER (BDPCS3_W_STRETCH={_d(weights.stretch)})",
        f"      PARAMETER (BDPCS3_W_ANGLE={_d(weights.angle)})",
        f"      PARAMETER (BDPCS3_W_HBOND={_d(weights.hbond)})",
        f"      PARAMETER (BDPCS3_W_TORSION_MIN={_d(weights.torsion_min)})",
        f"      PARAMETER (BDPCS3_W_FRAGMENT={_d(weights.fragment)})",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).resolve().parent / "fortran" / "bdpcs3_hbond_params.inc"
    out.write_text(build_bdpcs3_hbond_include(), encoding="utf-8")


if __name__ == "__main__":
    main()
