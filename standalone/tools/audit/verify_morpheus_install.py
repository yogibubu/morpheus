#!/usr/bin/env python3
"""Install MORPHEUS into a fresh venv and run the standalone smoke fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


EXPECTED_WHEEL_MODULES = {
    "__init__.py",
    "_version.py",
    "cli.py",
    "cli_commands.py",
    "cli_parser.py",
    "cli_support.py",
    "fit.py",
    "validation.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    release = args.release.expanduser().resolve()
    wheelhouse = release / "wheels"
    _verify_manifest(release)
    _verify_wheel_surface(wheelhouse)

    with tempfile.TemporaryDirectory(prefix="morpheus-clean-install-") as scratch_text:
        scratch = Path(scratch_text)
        environment = scratch / "venv"
        subprocess.run([args.python, "-m", "venv", str(environment)], check=True)
        bindir = environment / ("Scripts" if sys.platform == "win32" else "bin")
        python = bindir / ("python.exe" if sys.platform == "win32" else "python")
        morpheus = bindir / ("morpheus.exe" if sys.platform == "win32" else "morpheus")
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--find-links",
                str(wheelhouse),
                "matrix-morpheus",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "from importlib.util import find_spec;"
                    "assert find_spec('matrix_cli') is None;"
                    "import matrix_morpheus.cli_commands"
                ),
            ],
            check=True,
            cwd=scratch,
        )
        doctor = json.loads(
            subprocess.check_output(
                [str(morpheus), "doctor", "--json"],
                text=True,
                cwd=scratch,
            )
        )
        if doctor["status"] != "PASS" or not doctor["standalone_dispatch"]:
            raise RuntimeError(f"MORPHEUS doctor failed standalone dispatch: {doctor}")
        for module in ("matrix_oracle", "matrix_smith", "matrix_trinity"):
            if not doctor["required_modules"].get(module):
                raise RuntimeError(f"MORPHEUS doctor did not validate {module}")
        examples = scratch / "examples"
        subprocess.run([str(morpheus), "examples", str(examples)], check=True, cwd=scratch)
        output = scratch / "run"
        xyzin = scratch / "water.xyzin"
        subprocess.run(
            [
                str(morpheus),
                "fit",
                "--xyz",
                str(examples / "water" / "parent.xyz"),
                "--observations",
                str(examples / "water" / "isotopologues.toml"),
                "--xyzin",
                str(xyzin),
                "--outdir",
                str(output),
                "--coordinate-model",
                "gic",
                "--max-iter",
                "2",
            ],
            check=True,
            cwd=scratch,
        )
        manifest = json.loads((output / "semiexp_manifest.json").read_text(encoding="utf-8"))
        if not str(manifest["backend"]["backtransform"]).startswith("LINK hybrid typed SONIC"):
            raise RuntimeError("clean MORPHEUS fit did not delegate realization to LINK")
        required = (
            "semiexp_geometry.xyz",
            "semiexp_report.html",
            "semiexp_report.txt",
            "semiexp_results.tex",
            "semiexp_results.pdf",
            "semiexp_manifest.json",
        )
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"clean MORPHEUS fit missed outputs: {missing}")
        summary = {
            "schema": "matrix.morpheus.clean_install.v1",
            "python": subprocess.check_output([str(python), "--version"], text=True).strip(),
            "morpheus_version": subprocess.check_output(
                [str(morpheus), "--version"], text=True
            ).strip(),
            "coordinate_model": manifest["parameters"]["coordinate_model"],
            "backtransform": manifest["backend"]["backtransform"],
            "doctor": doctor,
            "matrix_cli_installed": False,
            "status": "PASS",
        }
        path = release / "morpheus-clean-install.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
    return 0


def _verify_manifest(release: Path) -> None:
    payload = json.loads(
        (release / "morpheus-release-manifest.json").read_text(encoding="utf-8")
    )
    for record in payload["artifacts"]:
        path = release / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"release checksum mismatch: {path}")


def _verify_wheel_surface(wheelhouse: Path) -> None:
    wheels = sorted(wheelhouse.glob("matrix_morpheus-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one matrix-morpheus wheel, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
        names = {Path(name).name for name in members if name.startswith("matrix_morpheus/")}
    missing = EXPECTED_WHEEL_MODULES - names
    if missing:
        raise RuntimeError(f"MORPHEUS wheel surface is incomplete: {sorted(missing)}")
    if "matrix_morpheus/data/examples/water/parent.xyz" not in members:
        raise RuntimeError("MORPHEUS wheel does not contain the standalone example")


if __name__ == "__main__":
    raise SystemExit(main())
