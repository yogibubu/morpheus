#!/usr/bin/env python3
"""Compatibility launcher for the canonical MORPHEUS install verifier."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "audit" / "verify_morpheus_install.py"),
    run_name="__main__",
)
