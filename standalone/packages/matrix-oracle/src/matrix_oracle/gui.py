"""Standalone launcher for the canonical MATRIX ORACLE window."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle-gui",
        description="Open the canonical MATRIX ORACLE desktop window",
    )
    parser.add_argument(
        "xyzin",
        nargs="?",
        type=Path,
        help="Existing ORACLE enriched XYZ project",
    )
    parser.add_argument("--workdir", type=Path, help="Workspace for runs and reports")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen window and exit immediately",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from matrix_gui.app import run as run_matrix_gui
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(("matrix_gui", "PySide6")):
            raise SystemExit(
                "The ORACLE desktop requires the GUI extra: "
                "`python -m pip install matrix-oracle[qt]`."
            ) from exc
        raise
    forwarded = ["ORACLE"]
    if args.xyzin is not None:
        forwarded.append(str(args.xyzin))
    if args.workdir is not None:
        forwarded.extend(("--workdir", str(args.workdir)))
    if args.smoke_test:
        forwarded.append("--smoke-test")
    return run_matrix_gui(forwarded)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


__all__ = ["build_parser", "main", "run"]
