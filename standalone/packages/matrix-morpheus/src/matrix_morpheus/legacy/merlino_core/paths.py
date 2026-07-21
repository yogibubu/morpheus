from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Locate the Merlino repository root by walking upward to `.git`."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate Merlino repository root from {current}")
