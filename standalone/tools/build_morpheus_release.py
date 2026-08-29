#!/usr/bin/env python3
"""Compatibility launcher for the canonical MORPHEUS release builder."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "build" / "build_morpheus_release.py"),
    run_name="__main__",
)
