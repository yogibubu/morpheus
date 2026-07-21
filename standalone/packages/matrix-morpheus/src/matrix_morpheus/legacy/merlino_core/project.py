from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


PROJECT_STATE_DIR = ".merlino"
PROJECT_STATE_FILE = "project.json"


@dataclass
class ProjectState:
    """Minimal persistent Merlino4 project state."""

    workdir: Path
    name: str = "Merlino project"
    active_workflow: str = "molecule"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_dir(self) -> Path:
        return self.workdir / PROJECT_STATE_DIR

    @property
    def state_path(self) -> Path:
        return self.state_dir / PROJECT_STATE_FILE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workdir"] = str(self.workdir)
        return data

    def save(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.state_path

    @classmethod
    def load(cls, workdir: Path) -> "ProjectState":
        workdir = Path(workdir)
        path = workdir / PROJECT_STATE_DIR / PROJECT_STATE_FILE
        if not path.exists():
            return cls(workdir=workdir)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            workdir=Path(data.get("workdir", workdir)),
            name=str(data.get("name", "Merlino project")),
            active_workflow=str(data.get("active_workflow", "molecule")),
            metadata=dict(data.get("metadata", {})),
        )


def ensure_project_state(workdir: Path) -> ProjectState:
    """Load or create the project state for a work directory."""
    state = ProjectState.load(workdir)
    state.workdir.mkdir(parents=True, exist_ok=True)
    if not state.state_path.exists():
        state.save()
    return state
