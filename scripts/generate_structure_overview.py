#!/usr/bin/env python3
"""Generate the manuscript structure overview figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

ATOM_COLORS = {
    "H": "#D9D9D9",
    "C": "#2F3437",
    "N": "#2B6CB0",
    "O": "#D64A3A",
}

ATOM_SIZES = {"H": 22, "C": 72, "N": 82, "O": 82}


@dataclass
class Panel:
    title: str
    smiles: str | None = None
    xyz: Path | None = None
    mode: str = "2d"
    note: str | None = None


PANELS = [
    Panel("Glycolaldehyde", "O=CCO"),
    Panel("Cyclopentadiene", "C1C=CC=C1"),
    Panel("Nitrobenzene", "O=[N+]([O-])c1ccccc1"),
    Panel("p-EBN", "C#Cc1ccc(C#N)cc1"),
    Panel("Azulene", "c1ccc2cccc2cc1"),
    Panel("Glycine I/II", "NCC(=O)O"),
    Panel("Maleic anhydride", "O=C1OC(=O)C=C1"),
    Panel("Phthalic anhydride", "O=C1OC(=O)c2ccccc12"),
    Panel("Succinic anhydride", "O=C1OC(=O)CC1"),
    Panel("Norbornane parent", "C1CC2CCC1C2", mode="3d", note="structural parent"),
    Panel(
        "Norcamphor",
        "O=C1C2CCC(C2)C1",
        xyz=Path("/Users/vincenzobarone/MATRIX/working/semiexp/norcamphor_table3_kraitchman_cartesian/semiexp_geometry.xyz"),
        mode="3d",
        note="bridged ketone",
    ),
    Panel(
        "Camphor",
        "CC1(C)C2CCC1(C)C(=O)C2",
        xyz=ROOT / "data" / "camphor" / "camphor_se_no_kraitchman.xyz",
        mode="3d",
        note="methylated test",
    ),
    Panel("Testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", mode="3d"),
]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    start = 2 if lines and lines[0].strip().isdigit() else 0
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def mol_to_3d(smiles: str) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260703
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.asarray([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return symbols, coords, bonds


def infer_bonds(symbols: list[str], coords: np.ndarray) -> list[tuple[int, int]]:
    radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
    bonds: list[tuple[int, int]] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            cutoff = 1.23 * (radii.get(symbols[i], 0.75) + radii.get(symbols[j], 0.75))
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


def project(coords: np.ndarray) -> np.ndarray:
    coords = coords - coords.mean(axis=0)
    # Fixed rotations chosen to show the bridge and methyl substituents clearly.
    rz = np.deg2rad(-35)
    rx = np.deg2rad(62)
    ry = np.deg2rad(12)
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return coords @ (rot_z @ rot_x @ rot_y).T


def draw_2d(ax, panel: Panel) -> None:
    mol = Chem.MolFromSmiles(panel.smiles)
    Chem.rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(520, 360), kekulize=True, wedgeBonds=True)
    ax.imshow(img)
    ax.axis("off")


def draw_3d(ax, panel: Panel) -> None:
    if panel.xyz and panel.xyz.exists():
        symbols, coords = read_xyz(panel.xyz)
        bonds = infer_bonds(symbols, coords)
    else:
        symbols, coords, bonds = mol_to_3d(panel.smiles)
    coords = project(coords)
    depth = coords[:, 2]
    for i, j in sorted(bonds, key=lambda ij: (depth[ij[0]] + depth[ij[1]]) / 2):
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="#8A9197",
            lw=1.35,
            solid_capstyle="round",
            zorder=1 + (depth[i] + depth[j]) / 2,
        )
    order = np.argsort(depth)
    for i in order:
        symbol = symbols[i]
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=ATOM_SIZES.get(symbol, 65),
            color=ATOM_COLORS.get(symbol, "#777777"),
            edgecolor="white",
            linewidth=0.55,
            zorder=10 + depth[i],
        )
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) * 0.62
    center = coords[:, :2].mean(axis=0)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.25, 7.35))
    grid = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.16)
    axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(16)]
    for ax in axes:
        ax.axis("off")

    for idx, panel in enumerate(PANELS):
        ax = axes[idx]
        if panel.mode == "3d":
            draw_3d(ax, panel)
        else:
            draw_2d(ax, panel)
        ax.text(0.5, 1.02, panel.title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, weight="bold")
        if panel.note:
            ax.text(0.5, -0.035, panel.note, transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="#4A5561")

    # Leave the last three cells for a compact legend and family cue.
    legend_ax = axes[-3]
    legend_ax.axis("off")
    legend_ax.text(0.02, 0.78, "3D bridged series", transform=legend_ax.transAxes, fontsize=8.2, weight="bold", color="#263238")
    legend_ax.text(0.02, 0.58, "norbornane to norcamphor to camphor", transform=legend_ax.transAxes, fontsize=7.2, color="#263238")
    legend_ax.text(0.02, 0.35, "Camphor adds methyl substitution to the norbornane ketone framework.", transform=legend_ax.transAxes, fontsize=6.8, color="#4A5561", wrap=True)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "studied_structures_3d.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "studied_structures_3d.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
