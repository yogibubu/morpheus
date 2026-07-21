from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from matrix_core import (
    key_value_section_lines,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)

from .class_advisor import PrimitiveClassSpec, SynthonPrimitiveClassSpec, parse_primitive_class_spec
from .predicates import predicate_sigmas_for_reference_level


ORACLE_XYZ_MORPHEUS_SCHEMA = "oracle.xyz.morpheus.v1"


@dataclass(frozen=True)
class MorpheusSection:
    status: str = "complete"
    source_kind: str = "semiexp"
    source_path: Path | None = None
    xyzin_path: Path | None = None
    run_dir: Path | None = None
    manifest_path: Path | None = None
    text_report_path: Path | None = None
    html_report_path: Path | None = None
    latex_tables_path: Path | None = None
    geometry_path: Path | None = None
    parameters_path: Path | None = None
    residuals_path: Path | None = None
    rotational_constants_path: Path | None = None
    diagnostics_path: Path | None = None
    backend: str = "python"
    coordinate_model: str = "gic"
    observable: str = "moments"
    components: tuple[str, ...] = ()
    rms_MHz: float = 0.0
    rotational_rms_MHz: float = 0.0
    rotational_mean_square_MHz2: float = 0.0
    iterations: int = 0
    stationary_point: str = ""
    convergence: str = ""
    rank: int = 0
    condition_number: float = 0.0
    isotopologue_count: int = 0
    parameter_count: int = 0
    active_parameter_count: int = 0
    warning_count: int = 0
    schema: str = ORACLE_XYZ_MORPHEUS_SCHEMA

    def __post_init__(self) -> None:
        for attr in (
            "source_path",
            "xyzin_path",
            "run_dir",
            "manifest_path",
            "text_report_path",
            "html_report_path",
            "latex_tables_path",
            "geometry_path",
            "parameters_path",
            "residuals_path",
            "rotational_constants_path",
            "diagnostics_path",
        ):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        object.__setattr__(self, "components", tuple(self.components))


@dataclass(frozen=True)
class InitialGeometryPredicateSpec:
    distance_sigma_angstrom: float = 0.003
    angle_sigma_degree: float = 0.3
    dihedral_sigma_degree: float = 0.5
    scope: tuple[str, ...] = (
        "heavy_bonds",
        "oh_bonds",
        "heavy_angles",
        "coh_angles",
        "xh_angles",
        "ring_torsions",
    )
    enabled: bool = False
    reference_level: str = "medium"


@dataclass(frozen=True)
class MorpheusInputConfig:
    coordinate_model: str | None = None
    observable: str | None = None
    components: str | None = None
    fixed_parameters: tuple[str, ...] = ()
    initial_geometry_predicates: InitialGeometryPredicateSpec = InitialGeometryPredicateSpec()
    primitive_classes: tuple[PrimitiveClassSpec, ...] = ()
    synthon_primitive_classes: SynthonPrimitiveClassSpec = SynthonPrimitiveClassSpec()
    primitive_class_min: float | None = None
    primitive_class_cross_max: float | None = None
    primitive_class_budget: str | None = None


def morpheus_section_from_result(
    result,
    *,
    xyzin_path: Path | str | None,
    outdir: Path | str,
    backend: str = "python",
    source_kind: str = "semiexp",
    source_path: Path | str | None = None,
    html_report_path: Path | str | None = None,
    latex_tables_path: Path | str | None = None,
    status: str = "complete",
) -> MorpheusSection:
    run_dir = Path(outdir)
    diagnostics = result.diagnostics
    rot_diffs = tuple(row.difference_MHz for row in result.rotational_constants)
    rotational_mse = sum(diff * diff for diff in rot_diffs) / len(rot_diffs) if rot_diffs else 0.0
    rotational_rms = rotational_mse**0.5
    return MorpheusSection(
        status=status,
        source_kind=source_kind,
        source_path=None if source_path is None else Path(source_path),
        xyzin_path=None if xyzin_path is None else Path(xyzin_path),
        run_dir=run_dir,
        manifest_path=result.manifest,
        text_report_path=run_dir / "semiexp_report.txt",
        html_report_path=None if html_report_path is None else Path(html_report_path),
        latex_tables_path=None if latex_tables_path is None else Path(latex_tables_path),
        geometry_path=run_dir / "semiexp_geometry.xyz",
        parameters_path=run_dir / "semiexp_parameters.csv",
        residuals_path=run_dir / "semiexp_residuals.csv",
        rotational_constants_path=run_dir / "semiexp_rotational_constants.csv",
        diagnostics_path=run_dir / "semiexp_diagnostics.csv",
        backend=backend,
        coordinate_model=diagnostics.coordinate_model,
        observable=diagnostics.observable,
        components=diagnostics.components,
        rms_MHz=float(result.rms_MHz),
        rotational_rms_MHz=float(rotational_rms),
        rotational_mean_square_MHz2=float(rotational_mse),
        iterations=int(result.iterations),
        stationary_point=result.stationary_point,
        convergence=diagnostics.convergence_reason,
        rank=int(diagnostics.rank),
        condition_number=float(diagnostics.condition_number),
        isotopologue_count=len(result.rotational_constants),
        parameter_count=len(result.parameters),
        active_parameter_count=sum(1 for parameter in result.parameters if parameter.active),
        warning_count=_manifest_warning_count(result.manifest),
    )


def morpheus_section_lines(section: MorpheusSection) -> list[str]:
    return key_value_section_lines(
        ORACLE_XYZ_MORPHEUS_SCHEMA,
        {
            "STATUS": section.status,
            "SOURCE_KIND": section.source_kind,
            "SOURCE_PATH": section.source_path,
            "XYZIN": section.xyzin_path,
            "RUN_DIR": section.run_dir,
            "MANIFEST": section.manifest_path,
            "TEXT_REPORT": section.text_report_path,
            "HTML_REPORT": section.html_report_path,
            "LATEX_TABLES": section.latex_tables_path,
            "GEOMETRY": section.geometry_path,
            "PARAMETERS": section.parameters_path,
            "RESIDUALS": section.residuals_path,
            "ROTATIONAL_CONSTANTS": section.rotational_constants_path,
            "DIAGNOSTICS": section.diagnostics_path,
            "BACKEND": section.backend,
            "COORDINATE_MODEL": section.coordinate_model,
            "OBSERVABLE": section.observable,
            "COMPONENTS": ",".join(section.components),
            "RMS_MHZ": _format_float(section.rms_MHz),
            "ROTATIONAL_RMS_MHZ": _format_float(section.rotational_rms_MHz),
            "ROTATIONAL_MEAN_SQUARE_MHZ2": _format_float(section.rotational_mean_square_MHz2),
            "ITERATIONS": section.iterations,
            "STATIONARY_POINT": section.stationary_point,
            "CONVERGENCE": section.convergence,
            "RANK": section.rank,
            "CONDITION_NUMBER": _format_float(section.condition_number),
            "ISOTOPOLOGUE_COUNT": section.isotopologue_count,
            "PARAMETER_COUNT": section.parameter_count,
            "ACTIVE_PARAMETER_COUNT": section.active_parameter_count,
            "WARNING_COUNT": section.warning_count,
        },
        key_order=(
            "STATUS",
            "SOURCE_KIND",
            "SOURCE_PATH",
            "XYZIN",
            "RUN_DIR",
            "MANIFEST",
            "TEXT_REPORT",
            "HTML_REPORT",
            "LATEX_TABLES",
            "GEOMETRY",
            "PARAMETERS",
            "RESIDUALS",
            "ROTATIONAL_CONSTANTS",
            "DIAGNOSTICS",
            "BACKEND",
            "COORDINATE_MODEL",
            "OBSERVABLE",
            "COMPONENTS",
            "RMS_MHZ",
            "ROTATIONAL_RMS_MHZ",
            "ROTATIONAL_MEAN_SQUARE_MHZ2",
            "ITERATIONS",
            "STATIONARY_POINT",
            "CONVERGENCE",
            "RANK",
            "CONDITION_NUMBER",
            "ISOTOPOLOGUE_COUNT",
            "PARAMETER_COUNT",
            "ACTIVE_PARAMETER_COUNT",
            "WARNING_COUNT",
        ),
    )


def parse_morpheus_section(lines: list[str] | tuple[str, ...]) -> MorpheusSection:
    values = parse_key_value_section(lines)
    schema = values.get("SCHEMA", ORACLE_XYZ_MORPHEUS_SCHEMA)
    if schema != ORACLE_XYZ_MORPHEUS_SCHEMA:
        raise ValueError(f"unsupported MORPHEUS schema: {schema}")
    return MorpheusSection(
        status=values.get("STATUS", "complete"),
        source_kind=values.get("SOURCE_KIND", "semiexp"),
        source_path=_optional_path(values.get("SOURCE_PATH")),
        xyzin_path=_optional_path(values.get("XYZIN")),
        run_dir=_optional_path(values.get("RUN_DIR")),
        manifest_path=_optional_path(values.get("MANIFEST")),
        text_report_path=_optional_path(values.get("TEXT_REPORT")),
        html_report_path=_optional_path(values.get("HTML_REPORT")),
        latex_tables_path=_optional_path(values.get("LATEX_TABLES")),
        geometry_path=_optional_path(values.get("GEOMETRY")),
        parameters_path=_optional_path(values.get("PARAMETERS")),
        residuals_path=_optional_path(values.get("RESIDUALS")),
        rotational_constants_path=_optional_path(values.get("ROTATIONAL_CONSTANTS")),
        diagnostics_path=_optional_path(values.get("DIAGNOSTICS")),
        backend=values.get("BACKEND", "python"),
        coordinate_model=values.get("COORDINATE_MODEL", "gic"),
        observable=values.get("OBSERVABLE", "moments"),
        components=_text_tuple(values.get("COMPONENTS", "")),
        rms_MHz=_float_value(values.get("RMS_MHZ"), 0.0),
        rotational_rms_MHz=_float_value(values.get("ROTATIONAL_RMS_MHZ"), 0.0),
        rotational_mean_square_MHz2=_float_value(
            values.get("ROTATIONAL_MEAN_SQUARE_MHZ2"),
            0.0,
        ),
        iterations=_int_value(values.get("ITERATIONS"), 0),
        stationary_point=values.get("STATIONARY_POINT", ""),
        convergence=values.get("CONVERGENCE", ""),
        rank=_int_value(values.get("RANK"), 0),
        condition_number=_float_value(values.get("CONDITION_NUMBER"), 0.0),
        isotopologue_count=_int_value(values.get("ISOTOPOLOGUE_COUNT"), 0),
        parameter_count=_int_value(values.get("PARAMETER_COUNT"), 0),
        active_parameter_count=_int_value(values.get("ACTIVE_PARAMETER_COUNT"), 0),
        warning_count=_int_value(values.get("WARNING_COUNT"), 0),
        schema=schema,
    )


def read_morpheus_section(path: Path | str) -> MorpheusSection:
    content = section_content(read_sectioned_lines(Path(path)), "MORPHEUS")
    if not content:
        raise ValueError("missing #MORPHEUS section")
    return parse_morpheus_section(content)


def read_morpheus_input_config(path: Path | str) -> MorpheusInputConfig | None:
    content = section_content(read_sectioned_lines(Path(path)), "MORPHEUS")
    if not content:
        return None
    return parse_morpheus_input_config(content)


def parse_morpheus_input_config(lines: list[str] | tuple[str, ...]) -> MorpheusInputConfig:
    values = parse_key_value_section(lines)
    fixed = _multi_value_tuple(values, ("FIXED_PARAMETER", "FIXED_PARAMETERS", "FIXED"))
    coordinate_model = values.get("FIT_COORDINATES") or values.get("COORDINATE_MODEL")
    predicates = _initial_geometry_predicate_spec(values)
    primitive_specs, synthon_specs = _primitive_class_specs(values)
    return MorpheusInputConfig(
        coordinate_model=_blank_to_none(coordinate_model),
        observable=_blank_to_none(values.get("OBSERVABLE")),
        components=_normalize_input_components(values.get("COMPONENTS")),
        fixed_parameters=fixed,
        initial_geometry_predicates=predicates,
        primitive_classes=primitive_specs,
        synthon_primitive_classes=synthon_specs,
        primitive_class_min=_optional_float(values.get("PRIMITIVE_CLASS_MIN")),
        primitive_class_cross_max=_optional_float(values.get("PRIMITIVE_CLASS_CROSS_MAX")),
        primitive_class_budget=_blank_to_none(values.get("PRIMITIVE_CLASS_BUDGET")),
    )


def write_morpheus_section(path: Path | str, section: MorpheusSection) -> None:
    replace_section(Path(path), "MORPHEUS", morpheus_section_lines(section))


def write_morpheus_section_from_result(
    path: Path | str,
    result,
    *,
    outdir: Path | str,
    backend: str = "python",
    source_kind: str = "semiexp",
    source_path: Path | str | None = None,
    html_report_path: Path | str | None = None,
    latex_tables_path: Path | str | None = None,
    status: str = "complete",
) -> MorpheusSection:
    section = morpheus_section_from_result(
        result,
        xyzin_path=path,
        outdir=outdir,
        backend=backend,
        source_kind=source_kind,
        source_path=source_path,
        html_report_path=html_report_path,
        latex_tables_path=latex_tables_path,
        status=status,
    )
    write_morpheus_section(path, section)
    return section


def _manifest_warning_count(path: Path | None) -> int:
    if path is None or not Path(path).is_file():
        return 0
    try:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    parameters = data.get("parameters", {})
    try:
        return int(parameters.get("n_warnings", 0))
    except (TypeError, ValueError):
        return 0


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def _text_tuple(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item for item in raw.split(",") if item)


def _multi_value_tuple(values: dict[str, str], keys: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for key in keys:
        raw = values.get(key)
        if not raw:
            continue
        for item in raw.replace(";", ",").split(","):
            text = item.strip()
            if text and text not in result:
                result.append(text)
    return tuple(result)


def _primitive_class_entries(values: dict[str, str], keys: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for key in keys:
        raw = values.get(key)
        if not raw:
            continue
        for item in raw.split(";"):
            text = item.strip()
            if text and text not in result:
                result.append(text)
    return tuple(result)


def _blank_to_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _normalize_input_components(raw: str | None) -> str | None:
    text = _blank_to_none(raw)
    if text is None:
        return None
    low = text.lower().replace(" ", "")
    if low in {"a,b,c", "abc", "ia,ib,ic", "iaibic"}:
        return "ABC"
    if low in {"a,b", "ab", "ia,ib", "iaib"}:
        return "AB"
    if low in {"a,c", "ac", "ia,ic", "iaic"}:
        return "AC"
    if low in {"b,c", "bc", "ib,ic", "ibic"}:
        return "BC"
    return text


def _initial_geometry_predicate_spec(values: dict[str, str]) -> InitialGeometryPredicateSpec:
    raw = values.get("PREDICATES") or values.get("PREDICATE")
    if not raw:
        return InitialGeometryPredicateSpec(enabled=False)
    tokens = shlex.split(raw)
    if not tokens:
        return InitialGeometryPredicateSpec(enabled=False)
    mode = tokens[0].strip().upper()
    if mode in {"NONE", "OFF", "FALSE", "0"}:
        return InitialGeometryPredicateSpec(enabled=False)
    if mode != "INITIAL_GEOMETRY":
        raise ValueError(
            "#MORPHEUS PREDICATES currently supports INITIAL_GEOMETRY, NONE or OFF"
        )
    assignments = _assignment_dict(tokens[1:])
    reference_level = (
        assignments.get("REFERENCE_LEVEL")
        or assignments.get("LEVEL")
        or values.get("PREDICATE_REFERENCE_LEVEL")
        or "medium"
    )
    default_distance, default_angle, default_dihedral = predicate_sigmas_for_reference_level(
        reference_level
    )
    scope_raw = values.get("PREDICATE_SCOPE") or assignments.get("SCOPE", "")
    scope = tuple(item.strip().lower() for item in scope_raw.split(",") if item.strip())
    return InitialGeometryPredicateSpec(
        distance_sigma_angstrom=_float_or_auto(
            assignments.get("DISTANCE_SIGMA"), default_distance
        ),
        angle_sigma_degree=_float_or_auto(assignments.get("ANGLE_SIGMA"), default_angle),
        dihedral_sigma_degree=_float_or_auto(
            assignments.get("DIHEDRAL_SIGMA"), default_dihedral
        ),
        scope=scope or InitialGeometryPredicateSpec().scope,
        enabled=True,
        reference_level=reference_level,
    )


def _primitive_class_specs(
    values: dict[str, str],
) -> tuple[tuple[PrimitiveClassSpec, ...], SynthonPrimitiveClassSpec]:
    raw_classes = _primitive_class_entries(values, ("PRIMITIVE_CLASS", "PRIMITIVE_CLASSES"))
    explicit = tuple(
        parse_primitive_class_spec(item)
        for item in raw_classes
        if item.strip().upper() not in {"AUTO", "AUTO_SYNTHON", "SYNTHON"}
    )
    mode = (values.get("PRIMITIVE_CLASSES") or values.get("PRIMITIVE_CLASS") or "").upper()
    advisor_raw = values.get("PRIMITIVE_CLASS_ADVISOR", "")
    advisor_tokens = shlex.split(advisor_raw)
    advisor_mode = advisor_tokens[0].upper() if advisor_tokens else ""
    assignments = _assignment_dict(advisor_tokens[1:])
    auto_enabled = any(token in mode for token in ("AUTO", "SYNTHON")) or advisor_mode in {
        "AUTO",
        "AUTO_SYNTHON",
        "SYNTHON",
    }
    include_raw = (
        values.get("PRIMITIVE_CLASS_INCLUDE")
        or assignments.get("INCLUDE")
        or "bonds,angles"
    ).lower()
    include = {item.strip() for item in include_raw.split(",") if item.strip()}
    min_group_size = int(
        _float_or_auto(
            values.get("PRIMITIVE_CLASS_MIN_GROUP_SIZE")
            or assignments.get("MIN_GROUP_SIZE"),
            2.0,
        )
    )
    bo_bins_raw = values.get("PRIMITIVE_CLASS_BOND_ORDER_BINS") or assignments.get(
        "BOND_ORDER_BINS"
    )
    synthon = SynthonPrimitiveClassSpec(
        enabled=auto_enabled,
        level=(
            values.get("SYNTHON_LEVEL")
            or values.get("PRIMITIVE_CLASS_SYNTHON_LEVEL")
            or assignments.get("LEVEL")
            or "auto"
        ).strip().lower(),
        include_bonds=not include or "bonds" in include or "bond" in include,
        include_angles=not include or "angles" in include or "angle" in include,
        min_group_size=min_group_size,
        bond_order_bins=_truthy(bo_bins_raw, True),
    )
    return explicit, synthon


def _assignment_dict(tokens: list[str] | tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def _optional_float(raw: str | None) -> float | None:
    text = _blank_to_none(raw)
    return None if text is None else float(text)


def _float_or_auto(raw: str | None, default: float) -> float:
    text = _blank_to_none(raw)
    if text is None or text.upper() == "AUTO":
        return float(default)
    return float(text)


def _truthy(raw: str | None, default: bool) -> bool:
    text = _blank_to_none(raw)
    if text is None:
        return default
    return text.lower() in {"1", "true", "yes", "on", "y"}


def _float_value(raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    return float(raw.replace("D", "E").replace("d", "e"))


def _int_value(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    return int(float(raw.replace("D", "E").replace("d", "e")))


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"
