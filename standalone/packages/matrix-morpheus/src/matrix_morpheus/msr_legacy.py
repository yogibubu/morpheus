"""Compatibility imports for historical MSR input consumers.

The implementation lives in :mod:`matrix_morpheus.msr_import`; this module
is retained only so existing scripts can continue reading legacy MSR files.
"""

from .msr_import import (
    MSRControls,
    MSRInput,
    MSR_INPUT_SUFFIXES,
    MSR_LEGACY_SUFFIXES,
    MSRLegacyControls,
    MSRLegacyInput,
    is_msr_legacy_file,
    read_msr_legacy_geometry,
    read_msr_legacy_input,
    read_msr_legacy_observations,
)

__all__ = [
    "MSR_LEGACY_SUFFIXES",
    "MSR_INPUT_SUFFIXES",
    "MSRControls",
    "MSRInput",
    "MSRLegacyControls",
    "MSRLegacyInput",
    "is_msr_legacy_file",
    "read_msr_legacy_geometry",
    "read_msr_legacy_input",
    "read_msr_legacy_observations",
]
