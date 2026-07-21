#!/usr/bin/env python3
"""Build a self-contained JCP submission folder without overwriting a prior one."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def local_inputs(fls_name: str) -> set[Path]:
    inputs: set[Path] = set()
    for line in (ROOT / fls_name).read_text(encoding="utf-8").splitlines():
        if not line.startswith("INPUT ./"):
            continue
        relative = Path(line.removeprefix("INPUT ./"))
        if relative.suffix.lower() in {".tex", ".bib", ".bbl", ".pdf", ".png", ".jpg", ".jpeg"}:
            inputs.add(relative)
    return inputs


def copy_relative(relative: Path, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")

    upload = target / "UPLOAD_FILES"
    source = target / "LATEX_SOURCE"
    upload.mkdir(parents=True)
    source.mkdir(parents=True)

    uploads = {
        "main_jcp.pdf": "01_MORPHEUS_main_manuscript.pdf",
        "supporting_information.pdf": "02_MORPHEUS_supplementary_material.pdf",
        "cover_letter_jcp.pdf": "03_MORPHEUS_cover_letter.pdf",
        "JCP_TABLE_FIGURE_DESCRIPTIONS.txt": "04_MORPHEUS_alt_text.txt",
    }
    for origin, final_name in uploads.items():
        shutil.copy2(ROOT / origin, upload / final_name)

    required = {
        Path("main_jcp.tex"),
        Path("supporting_information.tex"),
        Path("cover_letter_jcp.tex"),
        Path("references.bib"),
        Path("main_jcp.bbl"),
        Path("JCP_TABLE_FIGURE_DESCRIPTIONS.txt"),
    }
    for fls in ("main_jcp.fls", "supporting_information.fls", "cover_letter_jcp.fls"):
        required.update(local_inputs(fls))
    for relative in sorted(required):
        copy_relative(relative, source)

    source_readme = source / "README_BUILD.txt"
    source_readme.write_text(
        "Build commands (TeX Live with REVTeX 4.2):\n"
        "latexmk -pdf -interaction=nonstopmode -halt-on-error main_jcp.tex\n"
        "latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex\n"
        "latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter_jcp.tex\n",
        encoding="utf-8",
    )

    archive_base = upload / "05_MORPHEUS_LaTeX_source"
    shutil.make_archive(str(archive_base), "zip", root_dir=source)

    readme = target / "README_UPLOAD.txt"
    readme.write_text(
        "MORPHEUS — JCP submission package\n\n"
        "Initial submission:\n"
        "1. Upload 01_MORPHEUS_main_manuscript.pdf as the manuscript.\n"
        "2. Upload 02_MORPHEUS_supplementary_material.pdf as Supplementary Material.\n"
        "3. Upload 03_MORPHEUS_cover_letter.pdf as the cover letter.\n"
        "4. Keep 04_MORPHEUS_alt_text.txt ready for the accessibility field/file.\n"
        "5. Upload 05_MORPHEUS_LaTeX_source.zip only if the portal requests source files.\n\n"
        "AIP asks for a single compiled manuscript PDF at initial submission and a separate\n"
        "Supplementary Material PDF. The manuscript PDF already contains all figures.\n"
        "The LATEX_SOURCE directory is a readable copy of the source archive.\n",
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "JCP_SUBMISSION_CHECKLIST.md", target / "JCP_SUBMISSION_CHECKLIST.md")

    print(target)


if __name__ == "__main__":
    main()
