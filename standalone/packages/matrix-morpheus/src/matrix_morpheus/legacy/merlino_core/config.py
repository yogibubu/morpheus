from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_CONFIG_NAME = "merlino.toml"


@dataclass(frozen=True)
class MerlinoConfig:
    """Central Merlino runtime configuration."""

    gaussian_executable: str = "gdv"
    gaussian_nproc: int = 8
    gaussian_memory: str = "32GB"
    default_backend: str = "auto"
    backend_preferences: dict[str, str] = field(default_factory=dict)
    project_root: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_root"] = None if self.project_root is None else str(self.project_root)
        return data

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MerlinoConfig":
        gaussian = dict(data.get("gaussian", {}))
        backend = dict(data.get("backend", {}))
        project = dict(data.get("project", {}))
        return cls(
            gaussian_executable=str(gaussian.get("executable", data.get("gaussian_executable", "gdv"))),
            gaussian_nproc=int(gaussian.get("nproc", data.get("gaussian_nproc", 8))),
            gaussian_memory=str(gaussian.get("memory", data.get("gaussian_memory", "32GB"))),
            default_backend=str(backend.get("default", data.get("default_backend", "auto"))),
            backend_preferences=dict(backend.get("preferences", data.get("backend_preferences", {}))),
            project_root=Path(project["root"]) if project.get("root") else None,
        )


def load_config(path: Path | None = None, *, workdir: Path | None = None) -> MerlinoConfig:
    """Load `merlino.toml`; missing config returns defaults."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if workdir is not None:
        candidates.extend([Path(workdir) / DEFAULT_CONFIG_NAME, Path(workdir) / ".merlino" / DEFAULT_CONFIG_NAME])
    candidates.append(Path(DEFAULT_CONFIG_NAME))

    for candidate in candidates:
        if candidate.exists():
            data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            return MerlinoConfig.from_mapping(data)
    return MerlinoConfig()


def write_default_config(path: Path, *, overwrite: bool = False) -> Path:
    """Write a small editable TOML configuration file."""
    target = Path(path)
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "[gaussian]",
                'executable = "gdv"',
                "nproc = 8",
                'memory = "32GB"',
                "",
                "[backend]",
                'default = "auto"',
                "",
                "[backend.preferences]",
                'gicforge = "fortran"',
                'dvr = "auto"',
                'vpt2_vci = "python"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target
