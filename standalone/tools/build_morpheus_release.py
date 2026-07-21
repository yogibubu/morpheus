#!/usr/bin/env python3
"""Build a self-contained MORPHEUS wheelhouse and publication example kit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


PROJECTS = (
    "matrix-core",
    "matrix-chem",
    "matrix-link",
    "matrix-fragments",
    "matrix-qm",
    "matrix-rovib",
    "matrix-engines",
    "matrix-gaussian",
    "matrix-smith",
    "matrix-oracle",
    "matrix-trinity",
    "matrix-morpheus",
)
PINNED_MATRIX_REVISION = "187ea913261fc8a20b4260b10175b0f37d74c87d"
RELEASE_INPUTS = (
    *(f"packages/{name}" for name in PROJECTS),
    "docs/MORPHEUS_QUICKSTART.md",
    "docs/manuals/morpheus_manual.pdf",
    "docs/releases/MORPHEUS_0.1.0rc6.md",
    "CITATION.cff",
    "LICENSE",
    "tools/build_morpheus_release.py",
    "tools/verify_morpheus_install.py",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="User-selected release directory")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build a diagnostic artifact and record modified release inputs",
    )
    args = parser.parse_args()
    dirty_paths = _dirty_release_inputs(root)
    if dirty_paths and not args.allow_dirty:
        raise RuntimeError(
            "MORPHEUS release inputs are not committed; commit them or use "
            f"--allow-dirty for a diagnostic build: {', '.join(dirty_paths)}"
        )
    missing = _missing_internal_dependencies(root, PROJECTS)
    if missing:
        raise RuntimeError("MORPHEUS wheelhouse is incomplete: " + ", ".join(missing))

    output = args.output.expanduser().resolve()
    wheelhouse = output / "wheels"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    for name in PROJECTS:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(wheelhouse),
                str(root / "packages" / name),
            ],
            check=True,
        )

    shutil.copytree(
        root / "packages/matrix-morpheus/examples/semiexp",
        output / "examples",
        dirs_exist_ok=True,
    )
    shutil.copyfile(root / "docs/MORPHEUS_QUICKSTART.md", output / "MORPHEUS_QUICKSTART.md")
    shutil.copyfile(root / "docs/manuals/morpheus_manual.pdf", output / "MORPHEUS_MANUAL.pdf")
    shutil.copyfile(
        root / "docs/releases/MORPHEUS_0.1.0rc6.md", output / "RELEASE_NOTES.md"
    )
    shutil.copyfile(root / "CITATION.cff", output / "CITATION.cff")
    shutil.copyfile(root / "LICENSE", output / "LICENSE")

    projects = list(PROJECTS)
    version = _project_version(root / "packages/matrix-morpheus")
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "matrix.morpheus.release.v1",
        "morpheus_version": version,
        "matrix_revision": _revision(root),
        "matrix_dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "python": sys.version.split()[0],
        "components": projects,
        "component_versions": {
            name: _project_version(root / "packages" / name) for name in projects
        },
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
        "install": (
            f"python -m pip install --find-links {wheelhouse} "
            f"matrix-morpheus=={version}"
        ),
    }
    manifest_path = output / "morpheus-release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


def _project_version(project: Path) -> str:
    return str(
        tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
    )


def _missing_internal_dependencies(root: Path, projects: tuple[str, ...]) -> list[str]:
    included = set(projects)
    missing: set[str] = set()
    for name in projects:
        metadata = tomllib.loads(
            (root / "packages" / name / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        for requirement in metadata.get("dependencies", ()):
            dependency = _requirement_name(str(requirement))
            if (
                dependency.startswith("matrix-")
                and (root / "packages" / dependency).is_dir()
                and dependency not in included
            ):
                missing.add(f"{name} -> {dependency}")
    return sorted(missing)


def _requirement_name(requirement: str) -> str:
    boundary = len(requirement)
    for marker in "[<>=!~; ":
        position = requirement.find(marker)
        if position >= 0:
            boundary = min(boundary, position)
    return requirement[:boundary].strip().lower().replace("_", "-")


def _revision(root: Path) -> str:
    return PINNED_MATRIX_REVISION


def _dirty_release_inputs(root: Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *RELEASE_INPUTS],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # GitHub source archives and copied standalone snapshots have no .git
        # directory; provenance is carried by PINNED_MATRIX_REVISION instead.
        return []
    return sorted(line[3:].strip() for line in output.splitlines() if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
