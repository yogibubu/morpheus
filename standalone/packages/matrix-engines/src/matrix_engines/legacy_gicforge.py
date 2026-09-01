"""Compatibility imports for the vendored GICForge reference harness."""

from .gicforge_reference import (
    GICForgeReferenceRun,
    LegacyGICForgeRun,
    gicforge_reference_executable,
    legacy_gicforge_executable,
    read_gicforge_reference_run,
    read_legacy_gicforge_run,
    run_gicforge_reference,
    run_legacy_gicforge,
)

__all__ = [
    "LegacyGICForgeRun",
    "GICForgeReferenceRun",
    "legacy_gicforge_executable",
    "gicforge_reference_executable",
    "read_legacy_gicforge_run",
    "read_gicforge_reference_run",
    "run_legacy_gicforge",
    "run_gicforge_reference",
]
