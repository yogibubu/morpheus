from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from matrix_core.paths import repo_root


@dataclass(frozen=True)
class BackendSpec:
    """Description of a compiled Fortran backend."""

    name: str
    executable: str
    source_dir: str
    build_script: str = "compile_MAC"
    aliases: tuple[str, ...] = ()

    def candidates(self, root: Path) -> list[Path]:
        names = (self.executable, *self.aliases)
        return [root / "bin" / name for name in names]

    def build_command(self, root: Path) -> list[str]:
        return [str(root / self.source_dir / self.build_script)]


@dataclass(frozen=True)
class SourceBackendSpec:
    """Description of a Fortran source backend that is linked by another driver."""

    name: str
    source_dir: str
    primary_source: str
    compile_check: str = "compile_check"

    def source_path(self, root: Path) -> Path:
        return root / self.source_dir / self.primary_source

    def check_command(self, root: Path) -> list[str]:
        return [str(root / self.source_dir / self.compile_check)]


BACKENDS: dict[str, BackendSpec] = {
    "gicforge": BackendSpec(
        name="gicforge",
        executable="gicforge.x",
        aliases=("prova.x",),
        source_dir="fortran/gicforge",
    ),
    "dvr": BackendSpec(
        name="dvr",
        executable="path_dvr.x",
        source_dir="fortran/dvr",
    ),
}


SOURCE_BACKENDS: dict[str, SourceBackendSpec] = {
    "vpt2_vci": SourceBackendSpec(
        name="vpt2_vci",
        source_dir="fortran/vpt2_vci",
        primary_source="vci_core.f",
    ),
    "semiexp": SourceBackendSpec(
        name="semiexp",
        source_dir="fortran/semiexp",
        primary_source="semiexp_core.f",
    ),
}


def backend_executable(name: str, root: Path | None = None) -> str:
    if name not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise KeyError(f"Unknown Fortran backend {name!r}; known backends: {known}")
    return BACKENDS[name].executable


def resolve_backend(name: str, root: Path | None = None) -> Path:
    """Resolve a backend executable from `bin/` first, then from PATH."""
    if name not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise KeyError(f"Unknown Fortran backend {name!r}; known backends: {known}")
    spec = BACKENDS[name]
    root = Path(root) if root is not None else repo_root()

    for candidate in spec.candidates(root):
        if candidate.exists():
            return candidate

    for candidate in _oracle_engine_candidates(name, root):
        if candidate.exists():
            return candidate

    for exe_name in (spec.executable, *spec.aliases):
        found = shutil.which(exe_name)
        if found:
            return Path(found)

    candidates = ", ".join(
        str(path) for path in (*spec.candidates(root), *_oracle_engine_candidates(name, root))
    )
    raise FileNotFoundError(
        f"Fortran backend {name!r} not found. Tried {candidates} and PATH."
    )


def resolve_source_backend(name: str, root: Path | None = None) -> Path:
    """Resolve a source-only Fortran backend primary source."""
    if name not in SOURCE_BACKENDS:
        known = ", ".join(sorted(SOURCE_BACKENDS))
        raise KeyError(
            f"Unknown Fortran source backend {name!r}; known backends: {known}"
        )
    root = Path(root) if root is not None else repo_root()
    path = SOURCE_BACKENDS[name].source_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Fortran source backend {name!r} not found at {path}")
    return path


def _oracle_engine_candidates(name: str, root: Path) -> tuple[Path, ...]:
    if name == "gicforge":
        return (root / "engines" / "fortran" / "gicforge" / "build" / "gicforge_legacy",)
    return ()
