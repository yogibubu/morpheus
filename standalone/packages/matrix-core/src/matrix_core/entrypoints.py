"""Resolve installed MATRIX commands with an explicit source-tree fallback."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def matrix_command() -> tuple[str, ...]:
    executable = shutil.which("matrix")
    if executable is not None:
        return (executable,)
    for parent in Path(__file__).resolve().parents:
        development_launcher = parent / "tools" / "matrix_run.py"
        if (parent / "VERSION").is_file() and development_launcher.is_file():
            return (sys.executable, str(development_launcher))
    return (sys.executable, "-m", "matrix_cli")


__all__ = ["matrix_command"]
