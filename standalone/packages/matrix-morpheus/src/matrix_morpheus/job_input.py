from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

from .contracts import (
    DEFAULT_SEMIEXP_OBSERVABLE,
    DEFAULT_SEMIEXP_ROBUST_LOSS,
    DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS,
    HYDROGEN_PARAMETER_CONSTRAINT,
    IsotopologueObservation,
    ParameterClassConstraint,
    QMParameterPredicate,
)
from .geometry_input import SemiexperimentalGeometryInput, _modredundant_fixed_patterns
from .io import observations_from_mapping


SEMIEXP_JOB_SCHEMA = "oracle.semiexp.job.v1"
LEGACY_SEMIEXP_JOB_SCHEMA = "merlino.semiexp.job.v1"
SUPPORTED_SEMIEXP_JOB_SCHEMAS = (SEMIEXP_JOB_SCHEMA, LEGACY_SEMIEXP_JOB_SCHEMA)


@dataclass(frozen=True)
class SemiexperimentalJobInput:
    path: Path
    title: str
    geometry: SemiexperimentalGeometryInput
    observations: Path | None = None
    observations_inline: tuple[IsotopologueObservation, ...] = ()
    backend: str = "python"
    observable: str = DEFAULT_SEMIEXP_OBSERVABLE
    rotational_components: str = DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS
    coordinate_model: str = "gic"
    max_iter: int | None = None
    step: float = 1.0e-4
    damping: float = 1.0e-8
    max_step: float = 0.25
    prune_condition: float = 0.0
    robust_loss: str = DEFAULT_SEMIEXP_ROBUST_LOSS
    robust_scale: float = 0.0
    leave_one_out: bool = False
    checkpoint: Path | None = None
    restart: Path | None = None
    fixed_parameters: tuple[str, ...] = ()
    qm_predicates: tuple[QMParameterPredicate, ...] = ()
    parameter_classes: tuple[ParameterClassConstraint, ...] = ()


def is_semiexperimental_job_file(path: Path) -> bool:
    target = Path(path)
    suffixes = [item.lower() for item in target.suffixes]
    if (
        suffixes[-2:] in ([".mse", ".toml"], [".semiexp", ".toml"])
        or target.suffix.lower() == ".mfit"
    ):
        return True
    if target.suffix.lower() != ".toml":
        return False
    try:
        data = _read_toml(target)
    except Exception:
        return False
    return data.get("schema") in SUPPORTED_SEMIEXP_JOB_SCHEMAS


def read_semiexperimental_job(path: Path) -> SemiexperimentalJobInput:
    target = Path(path)
    data = _read_toml(target)
    schema = data.get("schema")
    if schema not in SUPPORTED_SEMIEXP_JOB_SCHEMAS:
        supported = ", ".join(repr(item) for item in SUPPORTED_SEMIEXP_JOB_SCHEMAS)
        raise ValueError(f"Semiexperimental job must declare one of: {supported}")
    title = str(data.get("title") or target.stem)
    geometry = _geometry_from_job_mapping(target, data, title=title)
    files = _mapping(data.get("files", {}), "files")
    observations = files.get("observations")
    observations_path = _resolve_relative(target, Path(str(observations))) if observations else None
    observations_inline = observations_from_mapping(data) if data.get("isotopologues") else ()
    fit = _mapping(data.get("fit", {}), "fit")
    constraints = _mapping(data.get("constraints", {}), "constraints")
    fixed_parameters = _fixed_parameters_from_mapping(constraints)
    return SemiexperimentalJobInput(
        path=target,
        title=title,
        geometry=geometry,
        observations=observations_path,
        observations_inline=observations_inline,
        backend=str(fit.get("backend", "python")),
        observable=str(fit.get("observable", DEFAULT_SEMIEXP_OBSERVABLE)),
        rotational_components=str(
            fit.get("rotational_components", DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS)
        ),
        coordinate_model=str(fit.get("coordinate_model", "gic")),
        max_iter=_optional_int(fit.get("max_iter")),
        step=float(fit.get("step", 1.0e-4)),
        damping=float(fit.get("damping", 1.0e-8)),
        max_step=float(fit.get("max_step", 0.25)),
        prune_condition=float(fit.get("prune_condition", 0.0)),
        robust_loss=str(fit.get("robust_loss", DEFAULT_SEMIEXP_ROBUST_LOSS)),
        robust_scale=float(fit.get("robust_scale", 0.0)),
        leave_one_out=bool(fit.get("leave_one_out", False)),
        checkpoint=_optional_path(target, fit.get("checkpoint")),
        restart=_optional_path(target, fit.get("restart")),
        fixed_parameters=fixed_parameters,
        qm_predicates=_qm_predicates_from_mapping(data.get("qm_predicates", ())),
        parameter_classes=_parameter_classes_from_mapping(data.get("parameter_classes", ())),
    )


def read_semiexperimental_job_geometry(path: Path) -> SemiexperimentalGeometryInput:
    return read_semiexperimental_job(path).geometry


def _read_toml(path: Path) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _geometry_from_job_mapping(
    path: Path, data: dict, *, title: str
) -> SemiexperimentalGeometryInput:
    geometry_value = data.get("geometry")
    if geometry_value is None:
        files = _mapping(data.get("files", {}), "files")
        geometry_file = files.get("geometry")
        if geometry_file:
            from .geometry_input import read_geometry_input

            parsed = read_geometry_input(_resolve_relative(path, Path(str(geometry_file))))
            return SemiexperimentalGeometryInput(
                parsed.atoms,
                np.asarray(parsed.coordinates_angstrom, dtype=float),
                title or parsed.comment,
                parsed.fixed_parameters,
                parsed.source_format,
            )
        raise ValueError("Semiexperimental job needs a [geometry] table or files.geometry")
    geometry = _mapping(geometry_value, "geometry")
    units = str(geometry.get("units", "angstrom")).strip().lower()
    if units not in {"angstrom", "ang", "a"}:
        raise ValueError("Semiexperimental job geometry units must be angstrom")
    atoms, coords = _cartesian_atoms_from_mapping(geometry)
    constraints = _mapping(data.get("constraints", {}), "constraints")
    fixed = _fixed_parameters_from_mapping(constraints)
    return SemiexperimentalGeometryInput(
        tuple(atoms),
        np.asarray(coords, dtype=float),
        title,
        fixed,
        "matrix_morpheus_job",
    )


def _cartesian_atoms_from_mapping(geometry: dict) -> tuple[list[str], list[list[float]]]:
    rows = geometry.get("atoms", geometry.get("cartesian"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Semiexperimental job geometry needs a non-empty atoms/cartesian list")
    atoms: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        if isinstance(row, dict):
            symbol = str(row["atom"])
            xyz = [float(row["x"]), float(row["y"]), float(row["z"])]
        elif isinstance(row, (list, tuple)) and len(row) >= 4:
            symbol = str(row[0])
            xyz = [float(row[1]), float(row[2]), float(row[3])]
        else:
            raise ValueError("Each semiexp job geometry row must be [atom, x, y, z] or a mapping")
        if not symbol.strip():
            raise ValueError("Semiexp job atom symbol cannot be empty")
        atoms.append(symbol.strip())
        coords.append(xyz)
    return atoms, coords


def _fixed_parameters_from_mapping(constraints: dict) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    modredundant = constraints.get("modredundant", ())
    if isinstance(modredundant, str):
        modredundant = [line for line in modredundant.splitlines() if line.strip()]
    fixed_gic = _constraint_items(
        constraints.get("fixed_gic_patterns", ()),
        constraints.get("fixed", ()),
        constraints.get("gic_constraints", ()),
        constraints.get("expression_constraints", ()),
        constraints.get("fixed_expressions", ()),
        constraints.get("gic_definitions", ()),
        constraints.get("coordinate_definitions", ()),
        constraints.get("definitions", ()),
    )
    automatic = []
    if bool(constraints.get("fix_hydrogen_parameters", False)):
        automatic.append(HYDROGEN_PARAMETER_CONSTRAINT)
    for item in (
        *_modredundant_fixed_patterns(tuple(modredundant or ())),
        *(fixed_gic or ()),
        *automatic,
    ):
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _constraint_items(*values: object) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        if not value:
            continue
        if isinstance(value, str):
            items.extend(
                item.strip() for item in _split_top_level(value, separators=",;") if item.strip()
            )
            continue
        try:
            iterator = iter(value)  # type: ignore[arg-type]
        except TypeError:
            items.append(str(value).strip())
            continue
        items.extend(str(item).strip() for item in iterator if str(item).strip())
    return tuple(items)


def _split_top_level(raw: str, *, separators: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    for char in str(raw):
        if char == "(":
            round_depth += 1
        elif char == ")" and round_depth > 0:
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        if char in separators and round_depth == 0 and square_depth == 0 and brace_depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _qm_predicates_from_mapping(items) -> tuple[QMParameterPredicate, ...]:
    if not items:
        return ()
    if not isinstance(items, list):
        raise ValueError("qm_predicates must be a list")
    predicates: list[QMParameterPredicate] = []
    for item in items:
        data = _mapping(item, "qm_predicates entry")
        pattern = str(data.get("pattern", data.get("label_pattern", ""))).strip()
        predicates.append(
            QMParameterPredicate(
                pattern,
                float(data["value"]),
                float(data["sigma"]),
                source=str(data.get("source", "qm")),
            )
        )
    return tuple(predicates)


def _parameter_classes_from_mapping(items) -> tuple[ParameterClassConstraint, ...]:
    if not items:
        return ()
    if not isinstance(items, list):
        raise ValueError("parameter_classes must be a list")
    classes: list[ParameterClassConstraint] = []
    for item in items:
        data = _mapping(item, "parameter_classes entry")
        patterns = data.get("patterns", ())
        if isinstance(patterns, str):
            patterns = [part.strip() for part in patterns.split("|") if part.strip()]
        classes.append(
            ParameterClassConstraint(
                str(data["name"]),
                tuple(str(pattern).strip() for pattern in patterns if str(pattern).strip()),
                str(data.get("mode", "shared")),
            )
        )
    return tuple(classes)


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Semiexperimental job {name} must be a table")
    return value


def _resolve_relative(base: Path, target: Path) -> Path:
    return target if target.is_absolute() else base.parent / target


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_path(base: Path, value) -> Path | None:
    if value is None or value == "":
        return None
    return _resolve_relative(base, Path(str(value)))
