"""Core infrastructure for Merlino4.

This package is intentionally small at the start of the refactor. It provides
stable helpers that older Merlino3 modules can adopt incrementally.
"""

from .config import MerlinoConfig, load_config, write_default_config
from .errors import BackendError, InputError, MerlinoError, ParseError, ScientificValidationError
from .manifest import RunManifest, build_run_manifest, file_checksums, sha256_file, write_manifest
from .isotopologues import (
    XYZIN_ISOTOPOLOGUES_SCHEMA,
    XyzinIsotopologueRecord,
    XyzinIsotopologueValidationIssue,
    format_xyzin_isotopologue_issues,
    format_substitutions,
    has_xyzin_isotopologues,
    mass_number,
    merge_xyzin_isotopologue_records,
    parse_substitutions,
    parse_xyzin_isotopologue_records,
    read_xyzin_isotopologue_records,
    validate_xyzin_isotopologue_file,
    validate_xyzin_isotopologue_records,
    write_xyzin_isotopologue_records,
    xyzin_isotopologue_validation_errors,
    xyzin_isotopologue_section_lines,
)
from .numerics import RankCondition, damped_normal_step, limit_step, objective, rank_condition
from .paths import repo_root
from .project import ProjectState, ensure_project_state
from .workspace import WorkspaceLayout, ensure_workspace, slugify
from .xyzin_sections import (
    has_section,
    read_sectioned_lines,
    remove_section_from_lines,
    replace_section,
    replace_section_in_lines,
    section_content,
    write_sectioned_lines,
)
from .xyzin_geometry import XyzinGeometry, read_xyzin_geometry, replace_xyzin_geometry

__all__ = [
    "BackendError",
    "InputError",
    "MerlinoConfig",
    "MerlinoError",
    "ParseError",
    "ProjectState",
    "RankCondition",
    "RunManifest",
    "ScientificValidationError",
    "WorkspaceLayout",
    "XYZIN_ISOTOPOLOGUES_SCHEMA",
    "XyzinIsotopologueRecord",
    "XyzinIsotopologueValidationIssue",
    "XyzinGeometry",
    "build_run_manifest",
    "damped_normal_step",
    "ensure_project_state",
    "ensure_workspace",
    "file_checksums",
    "format_xyzin_isotopologue_issues",
    "format_substitutions",
    "limit_step",
    "load_config",
    "mass_number",
    "merge_xyzin_isotopologue_records",
    "objective",
    "parse_substitutions",
    "parse_xyzin_isotopologue_records",
    "rank_condition",
    "has_section",
    "has_xyzin_isotopologues",
    "read_sectioned_lines",
    "read_xyzin_geometry",
    "read_xyzin_isotopologue_records",
    "remove_section_from_lines",
    "replace_section",
    "replace_section_in_lines",
    "replace_xyzin_geometry",
    "repo_root",
    "section_content",
    "sha256_file",
    "slugify",
    "validate_xyzin_isotopologue_file",
    "validate_xyzin_isotopologue_records",
    "write_sectioned_lines",
    "write_default_config",
    "write_manifest",
    "write_xyzin_isotopologue_records",
    "xyzin_isotopologue_validation_errors",
    "xyzin_isotopologue_section_lines",
]
