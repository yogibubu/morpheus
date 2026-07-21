#!/usr/bin/env python3
"""Generate the inspectable-SONIC ledger used in the MORPHEUS manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XYZ = ROOT / "data" / "camphor" / "camphor_se_no_kraitchman.xyz"
OUTPUTS = (ROOT / "figures", ROOT / "overleaf_submission" / "figures")

NAVY = "#17324d"
TEAL = "#168b91"
AMBER = "#d99527"
VIOLET = "#7656a5"
GREY = "#667784"
LIGHT = "#f6f8f9"
ATOM_COLORS = {"H": "#dfe5e8", "C": "#2f3d46", "O": "#d85849"}
ATOM_SIZES = {"H": 13, "C": 42, "O": 52}
RADII = {"H": 0.31, "C": 0.76, "O": 0.66}


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(value) for value in fields[1:4]])
    return symbols, np.asarray(coords, dtype=float)


def infer_bonds(symbols: list[str], coords: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            cutoff = 1.23 * (RADII[symbols[i]] + RADII[symbols[j]])
            if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
                bonds.append((i, j))
    return bonds


def project(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    rz, rx, ry = np.deg2rad((-35.0, 62.0, 12.0))
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rotated = centered @ (rot_z @ rot_x @ rot_y).T
    xy = rotated[:, :2]
    return xy / max(float(np.ptp(xy, axis=0).max()), 1.0e-8)


def draw_camphor(ax, symbols: list[str], xy: np.ndarray, bonds: list[tuple[int, int]]) -> None:
    for i, j in bonds:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#8a969d", lw=1.0, zorder=1)
    for symbol in ("H", "C", "O"):
        indices = [i for i, value in enumerate(symbols) if value == symbol]
        ax.scatter(
            xy[indices, 0],
            xy[indices, 1],
            s=ATOM_SIZES[symbol],
            c=ATOM_COLORS[symbol],
            edgecolors="#aab2b7" if symbol == "H" else "white",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_xlim(-0.63, 0.63)
    ax.set_ylim(-0.58, 0.58)
    ax.set_aspect("equal")
    ax.axis("off")


def stretch_overlay(ax, xy: np.ndarray) -> None:
    for i, j in ((0, 1), (3, 4)):
        vector = xy[j] - xy[i]
        unit = vector / max(float(np.linalg.norm(vector)), 1.0e-8)
        midpoint = 0.5 * (xy[i] + xy[j])
        ax.add_patch(FancyArrowPatch(midpoint, midpoint - 0.20 * unit, arrowstyle="-|>", color=TEAL,
                                     mutation_scale=9, lw=1.6, zorder=5))
        ax.add_patch(FancyArrowPatch(midpoint, midpoint + 0.20 * unit, arrowstyle="-|>", color=TEAL,
                                     mutation_scale=9, lw=1.6, zorder=5))


def bend_overlay(ax, xy: np.ndarray) -> None:
    center = xy[3]
    a = xy[2] - center
    b = xy[4] - center
    theta_a = float(np.degrees(np.arctan2(a[1], a[0])))
    theta_b = float(np.degrees(np.arctan2(b[1], b[0])))
    if theta_b < theta_a:
        theta_b += 360.0
    if theta_b - theta_a > 180.0:
        theta_a, theta_b = theta_b - 360.0, theta_a
    ax.add_patch(Arc(center, 0.40, 0.40, theta1=theta_a, theta2=theta_b, color=AMBER, lw=2.0, zorder=5))
    for point in (xy[2], xy[4]):
        direction = point - center
        normal = np.array([-direction[1], direction[0]])
        normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
        ax.add_patch(FancyArrowPatch(point, point + 0.14 * normal, arrowstyle="-|>", color=AMBER,
                                     mutation_scale=9, lw=1.5, zorder=5))


def puckering_overlay(ax, xy: np.ndarray) -> None:
    for order, index in enumerate((0, 2, 4, 6)):
        direction = 0.17 if order % 2 == 0 else -0.17
        ax.add_patch(FancyArrowPatch(xy[index], xy[index] + np.array([0.0, direction]), arrowstyle="-|>",
                                     color=VIOLET, mutation_scale=9, lw=1.6, zorder=5))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    symbols, coords = read_xyz(XYZ)
    bonds = infer_bonds(symbols, coords)
    xy = project(coords)

    fig = plt.figure(figsize=(7.25, 4.55), facecolor="white")
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    canvas.text(0.035, 0.945, "Inspectable SONIC variables in a MORPHEUS model", color=NAVY,
                fontsize=13.0, fontweight="bold", va="center")
    canvas.text(0.036, 0.895, "The fitted space remains traceable from each named variable to its molecular motion and model role.",
                color=GREY, fontsize=7.9, va="center")

    headers = ((0.045, "HUMAN-READABLE RECORD"), (0.405, "CARTESIAN FINGERPRINT"), (0.735, "REFINEMENT ROLE"))
    for x, label in headers:
        canvas.text(x, 0.835, label, color=GREY, fontsize=7.0, fontweight="bold", va="center")

    rows = (
        ("Stre-SALC", r"$q_1=2^{-1/2}(R_a+R_b)$", "stretch family  •  length", "symmetry-allowed", "candidate active variable", TEAL, stretch_overlay),
        ("Bend-SALC", r"$q_2=\sum_i c_i\,\theta_i$", "bend family  •  angle", "orbit-complete", "linked or constrained", AMBER, bend_overlay),
        ("RPck", r"$q_3=\sum_i d_i\,U_i$", "protected ring family  •  rad", "rank protected", "retained cage deformation", VIOLET, puckering_overlay),
    )
    y_positions = (0.655, 0.425, 0.195)
    for index, ((name, formula, metadata, badge, role, accent, overlay), y) in enumerate(zip(rows, y_positions, strict=True)):
        canvas.add_patch(FancyBboxPatch((0.035, y - 0.095), 0.930, 0.185,
                                       boxstyle="round,pad=0.006,rounding_size=0.012",
                                       facecolor=LIGHT if index % 2 == 0 else "white",
                                       edgecolor="#dfe5e8", linewidth=0.75))
        canvas.add_patch(FancyBboxPatch((0.035, y - 0.095), 0.008, 0.185,
                                       boxstyle="round,pad=0,rounding_size=0.004",
                                       facecolor=accent, edgecolor=accent, linewidth=0))
        canvas.text(0.058, y + 0.038, name, color=accent, fontsize=9.3, fontweight="bold", va="center")
        canvas.text(0.058, y - 0.005, formula, color=NAVY, fontsize=8.5, va="center")
        canvas.text(0.058, y - 0.052, metadata, color=GREY, fontsize=6.8, va="center")

        molecule_ax = fig.add_axes((0.415, y - 0.083, 0.235, 0.165))
        draw_camphor(molecule_ax, symbols, xy, bonds)
        overlay(molecule_ax, xy)
        canvas.text(0.670, y, r"$-h\quad 0\quad +h$", color=GREY, fontsize=7.0, ha="center", va="center")

        canvas.text(0.755, y + 0.026, badge.upper(), color=accent, fontsize=6.5, fontweight="bold", va="center",
                    bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": accent, "linewidth": 0.7})
        canvas.text(0.755, y - 0.027, role, color=NAVY, fontsize=7.5, fontweight="bold", va="center")

    canvas.text(0.50, 0.035,
                "Stored expansion  •  family / symmetry / unit  •  analytic Wilson row  •  protection and provenance  •  optional motion trajectory",
                color=NAVY, fontsize=7.0, ha="center", va="center")

    for directory in OUTPUTS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "sonic_inspectability.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(directory / "sonic_inspectability.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
