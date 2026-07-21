"""Public ORACLE perception API with lazily loaded optional GUI clients."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from matrix_chem import (
    AccuracyLadderPlan,
    BackTransformationResult,
    Primitive,
    PrimitiveCoordinateContract,
    PrimitiveTarget,
    RefinementLayer,
    ValenceLevel,
    apply_accuracy_ladder_plan,
    backtransform_primitive_targets,
    build_accuracy_ladder_plan,
    build_l1_refinement_targets,
    build_primitive_contract,
    build_primitives,
    core_valence_bond_shift,
    primitive_b_matrix,
    read_primitive_contract,
    target_values_from_plan,
    validate_primitive_contract,
)

from ._version import __version__
from .api import (
    ORACLE_BATCH_SCHEMA,
    ORACLE_REPORT_SCHEMA,
    SUPPORTED_INPUT_FORMATS,
    OracleAnalysis,
    OracleAnalysisRequest,
    analyze_structure,
    analyze_structures,
    oracle_human_report_lines,
    oracle_version,
    write_oracle_analysis_reports,
)
from .config import (
    OracleConfig,
    OraclePaths,
    OracleSymmetryConfig,
    load_oracle_config,
    oracle_config_template,
    write_oracle_config_template,
)
from .scope import (
    DOWNSTREAM_OWNERSHIP,
    ORACLE_EXCLUDED_CAPABILITIES,
    ORACLE_OWNED_CAPABILITIES,
    ORACLE_SCOPE_SCHEMA,
    oracle_scope_contract,
)


def _exports(module: str, names: str) -> dict[str, str]:
    return {name: module for name in names.split()}


_LAZY_EXPORTS = {
    **_exports(
        "atom_classes",
        "SYNTHON_ATOM_CLASS_SCHEMA SynthonAtomClass SynthonAtomClassResult SynthonAtomClassThresholds classify_synthon_atoms",
    ),
    **_exports(
        "refinement",
        "ORACLE_REFINEMENT_SCHEMA OracleGeometryRefinement refine_l1_geometry",
    ),
}

_PUBLIC_LAZY_EXPORTS = dict(_LAZY_EXPORTS)

_PUBLIC_EXPORTS = (
    "AccuracyLadderPlan",
    "BackTransformationResult",
    "ORACLE_REPORT_SCHEMA",
    "ORACLE_BATCH_SCHEMA",
    "SUPPORTED_INPUT_FORMATS",
    "OracleAnalysis",
    "OracleAnalysisRequest",
    "OracleConfig",
    "OraclePaths",
    "OracleSymmetryConfig",
    "Primitive",
    "PrimitiveCoordinateContract",
    "PrimitiveTarget",
    "RefinementLayer",
    "ValenceLevel",
    "DOWNSTREAM_OWNERSHIP",
    "ORACLE_EXCLUDED_CAPABILITIES",
    "ORACLE_OWNED_CAPABILITIES",
    "ORACLE_SCOPE_SCHEMA",
    "analyze_structure",
    "analyze_structures",
    "backtransform_primitive_targets",
    "apply_accuracy_ladder_plan",
    "build_accuracy_ladder_plan",
    "build_l1_refinement_targets",
    "build_primitive_contract",
    "build_primitives",
    "core_valence_bond_shift",
    "load_oracle_config",
    "oracle_config_template",
    "oracle_version",
    "oracle_human_report_lines",
    "oracle_scope_contract",
    "primitive_b_matrix",
    "read_primitive_contract",
    "target_values_from_plan",
    "validate_primitive_contract",
    "write_oracle_config_template",
    "write_oracle_analysis_reports",
    "__version__",
)

__all__ = [*_PUBLIC_EXPORTS, *sorted(_PUBLIC_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
