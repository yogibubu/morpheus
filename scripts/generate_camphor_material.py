#!/usr/bin/env python3
"""Generate camphor manuscript figure and SI tables from archived CSV data."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "camphor"
FIGURES = ROOT / "figures"
GENERATED = ROOT / "generated"


BLUE = "#245C73"
TEAL = "#2F8F83"
GOLD = "#C88A2D"
SLATE = "#4A5561"
INK = "#263238"
RED = "#B55245"
GREY = "#7A8188"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str | float, digits: int = 3) -> str:
    if isinstance(value, str):
        if value.strip() == "":
            return "--"
        value = float(value)
    if math.isnan(value):
        return "--"
    if math.isinf(value):
        return "$\\infty$"
    if abs(value) != 0.0 and (abs(value) < 0.001 or abs(value) >= 10000):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fixed(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def geometry_by_kind(kind: str) -> tuple[list[float], list[float]]:
    rows = [row for row in read_csv("camphor_geometry_parameters.csv") if row["kind"] == kind]
    if kind == "bond":
        shifts = [1000.0 * float(row["delta_angstrom"]) for row in rows]
        errors = [1000.0 * float(row["sigma_angstrom"]) for row in rows]
    else:
        shifts = [float(row["delta_degree"]) for row in rows]
        errors = [float(row["sigma_degree"]) for row in rows]
    return shifts, errors


def make_distribution_panel(ax, values, errors, color, title, xlabel, bins):
    values_np = np.asarray(values, dtype=float)
    errors_np = np.asarray(errors, dtype=float)
    mu = float(values_np.mean())
    sigma = float(values_np.std(ddof=1))
    span = max(abs(values_np).max() * 1.25, errors_np.mean() * 1.10)
    x = np.linspace(-span, span, 400)
    ax.hist(
        values_np,
        bins=bins,
        density=True,
        color=color,
        alpha=0.74,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.plot(x, normal_pdf(x, mu, sigma), color=INK, lw=1.45)
    ax.axvspan(-errors_np.mean(), errors_np.mean(), color=GOLD, alpha=0.18, lw=0)
    ax.axvline(0.0, color=SLATE, lw=0.8, alpha=0.7)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("Density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", color="#D8DCE0", linewidth=0.6, alpha=0.8)
    text = (
        f"RMS shift = {rms(values):.3f}\n"
        f"max |shift| = {max(abs(v) for v in values):.3f}\n"
        f"mean error = {errors_np.mean():.3f}"
    )
    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D8DCE0", "lw": 0.7},
    )


def make_figure() -> None:
    bond_shifts, bond_errors = geometry_by_kind("bond")
    angle_shifts, angle_errors = geometry_by_kind("angle")
    stability = {row["variant"]: row for row in read_csv("camphor_stability_summary.csv")}
    selected = [
        ("ref_no_kra_full_validation", "reference", TEAL, "o", (18, -16), "left"),
        ("all_sigmas_x3", "all sigma x3", BLUE, "o", (18, 2), "left"),
        ("bdpcs3_plus_kraitchman_partial_broad", "broad Kra", GREY, "s", (18, 18), "left"),
        ("bdpcs3_plus_kraitchman_partial_tight", "tight Kra", RED, "D", (8, -2), "left"),
        ("no_h_predicates", "no H predicates", RED, "^", (-36, -8), "left"),
        ("kraitchman_partial_autostabilized", "Kra partial", RED, "v", (-36, 8), "left"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    make_distribution_panel(
        ax1,
        bond_shifts,
        bond_errors,
        BLUE,
        "A  Bond-length corrections",
        "Final - BDPCS3 / m$\\AA$",
        bins=10,
    )
    make_distribution_panel(
        ax2,
        angle_shifts,
        angle_errors,
        TEAL,
        "B  Valence-angle corrections",
        "Final - BDPCS3 / degree",
        bins=12,
    )

    for key, label, color, marker, offset, ha in selected:
        row = stability[key]
        x = float(row["bond_delta_initial_max_mA"])
        y = float(row["angle_delta_initial_max_deg"])
        ax3.scatter(x, y, s=64, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        ax3.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.5,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": "#8A9197", "lw": 0.55, "shrinkA": 2, "shrinkB": 5},
            zorder=4,
        )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Maximum bond-length deviation from BDPCS3 / m$\\AA$", fontsize=8.5)
    ax3.set_ylabel("Max angle deviation / degree", fontsize=8.5)
    ax3.set_title("C  Validation distinguishes stable models from misleading residuals", loc="left", fontsize=9.5, fontweight="bold", color=INK)
    ax3.grid(which="both", color="#D8DCE0", linewidth=0.55, alpha=0.72)
    ax3.set_xlim(0.45, 220)
    ax3.set_ylim(0.015, 8.5)
    ax3.tick_params(labelsize=7.5)
    ax3.text(
        0.985,
        0.035,
        "Accepted model: no Kraitchman predicates; strict C-H soft predicates.",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK,
    )

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "camphor_validation_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def write_geometry_table() -> None:
    lines = [
        "% Generated by scripts/generate_camphor_material.py.",
        "\\begin{center}",
        "\\refstepcounter{table}\\label{tab:si-camphor-geometry-statistics}",
        "\\parbox{0.94\\linewidth}{\\small\\raggedright Table~\\thetable: Camphor accepted no-Kraitchman model. Signed shifts are final SE minus BDPCS3 initial values; propagated errors are standard errors from the final covariance.\\par}",
        "\\smallskip",
        "\\footnotesize",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Parameter set & Count & Mean shift & RMS shift & Max $|$shift$|$ & Mean/max error \\\\",
        "\\midrule",
    ]
    specs = [
        ("bond", "Bonds / m\\AA{}"),
        ("angle", "Angles / deg"),
        ("dihedral", "Dihedrals / deg"),
    ]
    for kind, label in specs:
        shifts, errors = geometry_by_kind(kind)
        lines.append(
            f"{label} & {len(shifts)} & {fixed(mean(shifts), 3)} & {fixed(rms(shifts), 3)} & "
            f"{fixed(max(abs(v) for v in shifts), 3)} & {fixed(mean(errors), 3)}/{fixed(max(errors), 3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (GENERATED / "camphor_geometry_statistics.tex").write_text("\n".join(lines) + "\n")


def write_stability_table() -> None:
    rows = read_csv("camphor_stability_summary.csv")
    by_variant = {row["variant"]: row for row in rows}
    selected = [
        ("ref_no_kra_full_validation", "No-Kraitchman BDPCS3 predicates", "retained"),
        ("coord_cartesian_symmetry", "Symmetry-Cartesian replay", "same model"),
        ("all_sigmas_x3", "All predicate errors x3", "stable"),
        ("bdpcs3_plus_kraitchman_partial_broad", "BDPCS3 + broad Kraitchman", "not retained"),
        ("bdpcs3_plus_kraitchman_partial_tight", "BDPCS3 + tight Kraitchman", "rejected"),
        ("no_h_predicates", "No H predicates", "rejected"),
        ("kraitchman_partial_autostabilized", "Partial Kraitchman model", "rejected"),
    ]
    lines = [
        "% Generated by scripts/generate_camphor_material.py.",
        "\\begin{center}",
        "\\refstepcounter{table}\\label{tab:si-camphor-stability-controls}",
        "\\parbox{0.94\\linewidth}{\\small\\raggedright Table~\\thetable: Camphor stability controls. Geometry columns report maximum absolute final-minus-BDPCS3 deviations in the corresponding run.\\par}",
        "\\smallskip",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Model variant & RMS / MHz & Rank & Cond. & Max bond / m\\AA{} & Max angle / deg & Outcome \\\\",
        "\\midrule",
    ]
    for key, label, outcome in selected:
        row = by_variant[key]
        lines.append(
            f"{label} & {fnum(row['rotational_rms_MHz'], 4)} & {row['rank']} & {fnum(row['condition_number'], 1)} & "
            f"{fnum(row['bond_delta_initial_max_mA'], 3)} & {fnum(row['angle_delta_initial_max_deg'], 3)} & {outcome} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{center}"]
    (GENERATED / "camphor_stability_controls.tex").write_text("\n".join(lines) + "\n")


def max_float(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") not in ("", "nan")]
    return max(values) if values else float("nan")


def write_validation_table() -> None:
    rows = [row for row in read_csv("camphor_final_validation_runs.csv") if row["status"] == "ok"]
    groups: list[tuple[str, list[dict[str, str]], str]] = []
    groups.append(("Coordinate-model replay", [row for row in rows if row["check"] == "coordinate_model"], "basis independent"))
    groups.append(("Huber robust loss", [row for row in rows if row["check"] == "robust_loss"], "stable"))
    groups.append(("Predicate sigma scans", [row for row in rows if row["check"] == "predicate_sigma_scan"], "stable"))
    groups.append(("Random multistarts", [row for row in rows if row["check"] == "multistart"], "stable"))
    heavy_leave = [
        row
        for row in rows
        if row["check"] == "leave_predicate_group_out"
        and "_H_" not in row["label"]
        and "_CH_" not in row["label"]
    ]
    h_leave = [
        row
        for row in rows
        if row["check"] == "leave_predicate_group_out"
        and ("_H_" in row["label"] or "_CH_" in row["label"])
    ]
    groups.append(("Leave non-H predicate groups out", heavy_leave, "moderate drift"))
    groups.append(("Leave H predicate groups out", h_leave, "failure mode"))

    lines = [
        "% Generated by scripts/generate_camphor_material.py.",
        "\\begin{center}",
        "\\refstepcounter{table}\\label{tab:si-camphor-final-validation}",
        "\\parbox{0.94\\linewidth}{\\small\\raggedright Table~\\thetable: Camphor final-validation summary. RMSD and maximum atomic shifts are computed after alignment to the accepted no-Kraitchman reference geometry.\\par}",
        "\\smallskip",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Check & Runs & Max RMS / MHz & Max geom. RMSD / \\AA{} & Max atom shift / \\AA{} & Interpretation \\\\",
        "\\midrule",
    ]
    for label, group, interpretation in groups:
        lines.append(
            f"{label} & {len(group)} & {fnum(max_float(group, 'rotational_rms_MHz'), 4)} & "
            f"{fnum(max_float(group, 'geometry_rmsd_angstrom'), 4)} & "
            f"{fnum(max_float(group, 'max_atom_shift_angstrom'), 4)} & {interpretation} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{center}"]
    (GENERATED / "camphor_final_validation.tex").write_text("\n".join(lines) + "\n")


def write_kraitchman_table() -> None:
    rows = read_csv("kraitchman_predicate_vs_bdpcs3.csv")
    lines = [
        "% Generated by scripts/generate_camphor_material.py.",
        "\\begin{center}",
        "\\refstepcounter{table}\\label{tab:si-camphor-kraitchman}",
        "\\parbox{0.94\\linewidth}{\\small\\raggedright Table~\\thetable: Kraitchman-derived trial predicates for camphor compared with BDPCS3 reference parameters. Full CSV: \\texttt{data/camphor/kraitchman\\_predicate\\_vs\\_bdpcs3.csv}.\\par}",
        "\\smallskip",
        "\\footnotesize",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Set & Count & RMS diff. & Max $|$diff.$|$ & Largest example \\\\",
        "\\midrule",
    ]
    for kind, label in [("R", "Distances / m\\AA{}"), ("A", "Angles / deg")]:
        subset = [row for row in rows if row["kind"] == kind]
        diffs = [float(row["difference"]) for row in subset]
        largest = max(subset, key=lambda row: abs(float(row["difference"])))
        lines.append(
            f"{label} & {len(subset)} & {fnum(rms(diffs), 3)} & {fnum(max(abs(v) for v in diffs), 3)} & "
            f"{largest['label']} ({fnum(largest['difference'], 3)}) \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (GENERATED / "camphor_kraitchman_comparison.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    GENERATED.mkdir(exist_ok=True)
    make_figure()
    write_geometry_table()
    write_stability_table()
    write_validation_table()
    write_kraitchman_table()


if __name__ == "__main__":
    main()
