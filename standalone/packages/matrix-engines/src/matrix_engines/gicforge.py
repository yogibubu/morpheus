"""Discovery helpers for MATRIX/SMITH GICForge Fortran backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LEGACY_GICFORGE_FILES = (
    "dina25.f",
    "coord.f",
    "mkprim.f",
    "mkcyc.f",
    "mksalc.f",
    "locsvd.f",
    "gicprune.f",
    "gic_type_symmetry.f",
    "pcsgeo.f",
    "entors.f",
    "symang.f",
    "tools1.f",
    "tools2.f",
    "symm.f",
)


@dataclass(frozen=True)
class FortranBackendLayout:
    root: Path
    legacy_source_dir: Path
    matrix_kernel: Path
    legacy_compile_script: Path
    legacy_executable: Path


def gicforge_fortran_layout(repo_root: Path | None = None) -> FortranBackendLayout:
    root = _repo_root(repo_root)
    backend_root = root / "engines" / "fortran" / "gicforge"
    return FortranBackendLayout(
        root=backend_root,
        legacy_source_dir=backend_root / "legacy_merlino",
        matrix_kernel=backend_root / "frag_tric_bmat.f",
        legacy_compile_script=backend_root / "compile_legacy",
        legacy_executable=backend_root / "build" / "gicforge_legacy",
    )


def legacy_gicforge_source_paths(repo_root: Path | None = None) -> tuple[Path, ...]:
    layout = gicforge_fortran_layout(repo_root)
    return tuple(layout.legacy_source_dir / name for name in LEGACY_GICFORGE_FILES)


def validate_legacy_gicforge_sources(repo_root: Path | None = None) -> tuple[Path, ...]:
    missing = tuple(path for path in legacy_gicforge_source_paths(repo_root) if not path.is_file())
    if missing:
        return missing
    layout = gicforge_fortran_layout(repo_root)
    required = (
        layout.matrix_kernel,
        layout.root / "local_equiv.f",
        layout.legacy_compile_script,
        layout.legacy_source_dir / "MANIFEST.md",
        layout.root / "include" / "bdpcs3_hbond_params.inc",
    )
    return tuple(path for path in required if not path.is_file())


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "engines").is_dir():
            return candidate
    raise RuntimeError("cannot locate MATRIX repository root")
