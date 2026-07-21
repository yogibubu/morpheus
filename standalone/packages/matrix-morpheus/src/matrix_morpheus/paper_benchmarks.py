from __future__ import annotations

import ast
import csv
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


PAPER_BENCHMARK_SCHEMA = "oracle.semiexp.paper_regression.v1"
LEGACY_PAPER_BENCHMARK_SCHEMA = "merlino.semiexp.paper_regression.v1"
SUPPORTED_PAPER_BENCHMARK_SCHEMAS = (PAPER_BENCHMARK_SCHEMA, LEGACY_PAPER_BENCHMARK_SCHEMA)
DEFAULT_SNAPSHOT = Path("benchmarks/semiexp_msr/golden/semiexp_paper_regression.json")
DEFAULT_OUTPUT_DIR = Path("benchmarks/semiexp_msr/generated")
CASE_ORDER = (
    "glycolaldehyde",
    "cyclopentadiene",
    "nitrobenzene",
    "p-EBN",
    "azulene",
    "norcamphor",
    "glycine_I",
    "glycine_II",
)
PAIR_ORDER = ("AB", "AC", "BC")

SYSTEM_LABELS = {
    "glycolaldehyde": "Glycolaldehyde",
    "glycine_I": "Glycine I",
    "glycine_II": "Glycine II",
    "cyclopentadiene": "Cyclopentadiene",
    "nitrobenzene": "Nitrobenzene",
    "p-EBN": "p-EBN",
    "azulene": "Azulene",
    "norcamphor": "Norcamphor",
}

POINT_GROUP_LATEX = {
    "C1": r"C$_1$",
    "Cs": r"C$_s$",
    "C2v": r"C$_{2v}$",
}

MODEL_LABELS = {
    "gic": "GIC",
    "cartesian_symmetry": "Cart.",
}

PAIR_MOMENTS = {
    "AB": r"$(I_a,I_b)$",
    "AC": r"$(I_a,I_c)$",
    "BC": r"$(I_b,I_c)$",
}

_CASE_REQUIRED_FIELDS = {
    "point_group": str,
    "coordinate_model": str,
    "rotational_pair": str,
    "components": list,
    "isotopologues": int,
    "reported_constants": int,
    "final_gics": int,
    "totally_symmetric_gics": int,
    "effective_parameters": int,
    "rank": int,
    "primitive_constraints": int,
    "accepted_steps": int,
    "rejected_steps": int,
    "condition_number": (int, float),
    "rotational_rms_MHz": (int, float),
    "rotational_max_MHz": (int, float),
    "minimum_hessian_eigenvalue": (int, float),
    "stationary_point": str,
    "run_dir": str,
}

_PAIR_REQUIRED_FIELDS = {
    "rank": int,
    "condition_number": (int, float),
    "rotational_rms_MHz": (int, float),
    "rotational_max_MHz": (int, float),
    "run_dir": str,
}


class SnapshotValidationError(ValueError):
    """Raised when a paper benchmark snapshot is not portable or complete."""


def repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "packages").is_dir():
            return candidate
    return here.parents[1]


def load_paper_benchmark_snapshot(path: Path | None = None) -> dict[str, Any]:
    snapshot_path = _resolve_repo_path(path or DEFAULT_SNAPSHOT)
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    validate_paper_benchmark_snapshot(data, source=snapshot_path)
    return data


def validate_paper_benchmark_snapshot(
    snapshot: dict[str, Any], *, source: Path | None = None
) -> None:
    label = str(source) if source is not None else "snapshot"
    if snapshot.get("schema") not in SUPPORTED_PAPER_BENCHMARK_SCHEMAS:
        raise SnapshotValidationError(f"{label}: unsupported schema {snapshot.get('schema')!r}")
    cases = snapshot.get("cases")
    if not isinstance(cases, dict):
        raise SnapshotValidationError(f"{label}: cases must be an object")
    if tuple(cases) != CASE_ORDER:
        raise SnapshotValidationError(f"{label}: cases must be ordered as {CASE_ORDER}")
    for name in CASE_ORDER:
        _validate_case(name, cases[name], label)
    planar = snapshot.get("planar_pair_diagnostics")
    if not isinstance(planar, dict):
        raise SnapshotValidationError(f"{label}: planar_pair_diagnostics must be an object")
    if set(planar) != {"nitrobenzene", "azulene"}:
        raise SnapshotValidationError(
            f"{label}: planar diagnostics must cover nitrobenzene and azulene"
        )
    for system, diagnostics in planar.items():
        _validate_planar_diagnostics(system, diagnostics, label)


def refresh_snapshot_from_outputs(snapshot: dict[str, Any]) -> dict[str, Any]:
    validate_paper_benchmark_snapshot(snapshot)
    refreshed = deepcopy(snapshot)
    for name in CASE_ORDER:
        case = refreshed.get("cases", {}).get(name)
        if not case:
            continue
        run_dir = _resolve_repo_path(Path(case["run_dir"]))
        if run_dir.is_dir():
            _refresh_case_from_run_dir(case, run_dir)
    for system, diagnostics in refreshed.get("planar_pair_diagnostics", {}).items():
        pairs = diagnostics.get("pairs", {})
        for pair in PAIR_ORDER:
            pair_data = pairs.get(pair)
            if not pair_data:
                continue
            run_dir_value = pair_data.get("run_dir") or _default_pair_run_dir(system, pair)
            if run_dir_value is None:
                continue
            run_dir = _resolve_repo_path(Path(run_dir_value))
            if run_dir.is_dir():
                _refresh_pair_from_run_dir(pair_data, run_dir)
        if pairs:
            diagnostics["selected"] = min(
                pairs,
                key=lambda key: (
                    float(pairs[key].get("rotational_rms_MHz", math.inf)),
                    float(pairs[key].get("condition_number", math.inf)),
                ),
            )
    validate_paper_benchmark_snapshot(refreshed)
    return refreshed


def write_snapshot(snapshot: dict[str, Any], path: Path | None = None) -> Path:
    validate_paper_benchmark_snapshot(snapshot)
    target = _resolve_repo_path(path or DEFAULT_SNAPSHOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def write_paper_benchmark_artifacts(
    snapshot: dict[str, Any], outdir: Path | None = None
) -> dict[str, Path]:
    validate_paper_benchmark_snapshot(snapshot)
    target_dir = _resolve_repo_path(outdir or DEFAULT_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = target_dir / "paper_benchmark_summary.csv"
    planar_csv = target_dir / "paper_planar_pair_diagnostics.csv"
    summary_tex = target_dir / "benchmark_summary.tex"
    planar_tex = target_dir / "planar_pair_diagnostics.tex"
    _write_summary_csv(snapshot, summary_csv)
    _write_planar_csv(snapshot, planar_csv)
    summary_tex.write_text(_benchmark_summary_latex(snapshot), encoding="utf-8")
    planar_tex.write_text(_planar_pair_latex(snapshot), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "planar_csv": planar_csv,
        "summary_tex": summary_tex,
        "planar_tex": planar_tex,
    }


def generate_paper_benchmark_artifacts(
    *,
    snapshot_path: Path | None = None,
    outdir: Path | None = None,
    refresh_from_outputs: bool = True,
    update_snapshot: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    snapshot = load_paper_benchmark_snapshot(snapshot_path)
    if refresh_from_outputs:
        snapshot = refresh_snapshot_from_outputs(snapshot)
    if update_snapshot:
        write_snapshot(snapshot, snapshot_path)
    return snapshot, write_paper_benchmark_artifacts(snapshot, outdir)


def validate_paper_run_outputs(snapshot: dict[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    """Validate available run directories backing the paper-regression snapshot.

    The golden snapshot is intentionally portable: archived or regenerated run
    directories may live outside a clean checkout.  This helper therefore
    returns missing cases separately from incomplete ones, and treats any
    present run directory as an end-to-end reproducibility contract.
    """

    data = snapshot if snapshot is not None else load_paper_benchmark_snapshot()
    validate_paper_benchmark_snapshot(data)
    missing: list[str] = []
    incomplete: list[str] = []
    complete: list[str] = []
    required = (
        "semiexp_manifest.json",
        "semiexp_diagnostics.csv",
        "semiexp_rotational_constants.csv",
        "semiexp_parameters.csv",
        "semiexp_report.html",
        "semiexp_weight_diagnostics.csv",
    )
    for name in CASE_ORDER:
        run_dir = _resolve_repo_path(Path(data["cases"][name]["run_dir"]))
        if not run_dir.is_dir():
            missing.append(name)
            continue
        missing_files = [item for item in required if not (run_dir / item).is_file()]
        manifest = _read_json(run_dir / "semiexp_manifest.json")
        outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
        if "weight_diagnostics" not in outputs:
            missing_files.append("manifest.outputs.weight_diagnostics")
        if missing_files:
            incomplete.append(f"{name}:{','.join(missing_files)}")
        else:
            complete.append(name)
    return {
        "complete": tuple(complete),
        "missing": tuple(missing),
        "incomplete": tuple(incomplete),
    }


def _refresh_case_from_run_dir(case: dict[str, Any], run_dir: Path) -> None:
    diagnostics = _read_key_value_csv(run_dir / "semiexp_diagnostics.csv")
    manifest = _read_json(run_dir / "semiexp_manifest.json")
    parameters = manifest.get("parameters", {}) if isinstance(manifest, dict) else {}
    case["coordinate_model"] = str(
        diagnostics.get("coordinate_model", case.get("coordinate_model", "gic"))
    )
    components = _parse_components(diagnostics.get("components")) or tuple(
        case.get("components", ())
    )
    if components:
        case["components"] = list(components)
        case["rotational_pair"] = _components_to_pair(components)
    if "rank" in diagnostics:
        case["rank"] = int(float(diagnostics["rank"]))
    if "n_optimized_parameters" in diagnostics:
        case["effective_parameters"] = int(float(diagnostics["n_optimized_parameters"]))
    if "accepted_steps" in diagnostics:
        case["accepted_steps"] = int(float(diagnostics["accepted_steps"]))
    if "rejected_steps" in diagnostics:
        case["rejected_steps"] = int(float(diagnostics["rejected_steps"]))
    if "condition_number" in diagnostics:
        case["condition_number"] = float(diagnostics["condition_number"])
    if "n_gic_parameters" in parameters:
        case["final_gics"] = int(parameters["n_gic_parameters"])
    if "n_active_gic_parameters" in parameters:
        case["totally_symmetric_gics"] = int(parameters["n_active_gic_parameters"])
    if "n_effective_parameters" in parameters:
        case["effective_parameters"] = int(parameters["n_effective_parameters"])
    if "n_isotopologues" in parameters:
        case["isotopologues"] = int(parameters["n_isotopologues"])
    if "fixed_parameters" in parameters:
        case["primitive_constraints"] = len(
            _unique_primitive_constraints(parameters["fixed_parameters"])
        )
    if "stationary_point" in parameters:
        case["stationary_point"] = str(parameters["stationary_point"])
    n_constants, rms, max_abs = _rotational_stats(run_dir / "semiexp_rotational_constants.csv")
    if n_constants:
        case["reported_constants"] = n_constants
        case["rotational_rms_MHz"] = rms
        case["rotational_max_MHz"] = max_abs
    minimum_eigenvalue = _minimum_eigenvalue(run_dir / "semiexp_hessian_eigenvalues.csv")
    if minimum_eigenvalue is not None:
        case["minimum_hessian_eigenvalue"] = minimum_eigenvalue


def _refresh_pair_from_run_dir(pair_data: dict[str, Any], run_dir: Path) -> None:
    diagnostics = _read_key_value_csv(run_dir / "semiexp_diagnostics.csv")
    if "rank" in diagnostics:
        pair_data["rank"] = int(float(diagnostics["rank"]))
    if "condition_number" in diagnostics:
        pair_data["condition_number"] = float(diagnostics["condition_number"])
    _n_constants, rms, max_abs = _rotational_stats(run_dir / "semiexp_rotational_constants.csv")
    if _n_constants:
        pair_data["rotational_rms_MHz"] = rms
        pair_data["rotational_max_MHz"] = max_abs


def _write_summary_csv(snapshot: dict[str, Any], path: Path) -> None:
    columns = (
        "system",
        "point_group",
        "coordinate_model",
        "rotational_pair",
        "isotopologues",
        "reported_constants",
        "final_gics",
        "totally_symmetric_gics",
        "effective_parameters",
        "rank",
        "primitive_constraints",
        "accepted_steps",
        "rejected_steps",
        "condition_number",
        "rotational_rms_MHz",
        "rotational_max_MHz",
        "minimum_hessian_eigenvalue",
        "stationary_point",
        "run_dir",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name in CASE_ORDER:
            row = {"system": name}
            row.update(snapshot["cases"][name])
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_planar_csv(snapshot: dict[str, Any], path: Path) -> None:
    columns = (
        "system",
        "selected",
        "pair",
        "rank",
        "condition_number",
        "rotational_rms_MHz",
        "rotational_max_MHz",
        "run_dir",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for system, diagnostics in snapshot.get("planar_pair_diagnostics", {}).items():
            selected = diagnostics.get("selected", "")
            for pair in PAIR_ORDER:
                values = diagnostics["pairs"][pair]
                writer.writerow(
                    {
                        "system": system,
                        "selected": selected,
                        "pair": pair,
                        "rank": values["rank"],
                        "condition_number": values["condition_number"],
                        "rotational_rms_MHz": values["rotational_rms_MHz"],
                        "rotational_max_MHz": values["rotational_max_MHz"],
                        "run_dir": values["run_dir"],
                    }
                )


def _benchmark_summary_latex(snapshot: dict[str, Any]) -> str:
    rows = []
    for name in CASE_ORDER:
        case = snapshot["cases"][name]
        rows.append(
            " & ".join(
                (
                    SYSTEM_LABELS[name],
                    POINT_GROUP_LATEX.get(case["point_group"], case["point_group"]),
                    MODEL_LABELS.get(case["coordinate_model"], case["coordinate_model"]),
                    case["rotational_pair"],
                    str(case["isotopologues"]),
                    f"{case['final_gics']}/{case['totally_symmetric_gics']}",
                    f"{case['effective_parameters']}/{case['rank']}",
                    _constraint_label(name, int(case["primitive_constraints"])),
                    _steps_label(case),
                    f"{case['rotational_rms_MHz']:.4f}/{case['rotational_max_MHz']:.4f}",
                )
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""% Generated by `python -m matrix semiexp-benchmark --paper`.
\begin{{table*}}
\caption{{One-page audit of the representative \texttt{{MORPHEUS}}
semiexperimental refinements. ``Model'' is the working-coordinate model used
for the main row. ``Fit'' gives the independent rotational components entering
the normal equations; for moment-based fits, $AB$ denotes $(I_a,I_b)$. ``GIC/A''
gives the algorithmically generated non-redundant coordinate count and the
totally symmetric active count before constraint projection. ``Eff./rank''
gives the independent variables after primitive-constraint projection and the
final numerical rank. Rotational residuals are RMS/maximum absolute values in
MHz over all reported diagnostic constants.}}
\label{{tab:benchmark-summary}}
\scriptsize
\setlength{{\tabcolsep}}{{2.0pt}}
\begin{{tabular}}{{@{{}}llclrcclcl@{{}}}}
\toprule
System & PG & Model & Fit & Iso. & GIC/A & Eff./rank & Constraints & Steps & RMS/max \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _planar_pair_latex(snapshot: dict[str, Any]) -> str:
    rows = []
    for system, diagnostics in snapshot.get("planar_pair_diagnostics", {}).items():
        for pair in PAIR_ORDER:
            values = diagnostics["pairs"][pair]
            marker = r"$^\ast$" if pair == diagnostics.get("selected") else ""
            rows.append(
                " & ".join(
                    (
                        SYSTEM_LABELS.get(system, system),
                        pair + marker,
                        PAIR_MOMENTS[pair],
                        str(values["rank"]),
                        f"{values['condition_number']:.1f}",
                        f"{values['rotational_rms_MHz']:.4f}/{values['rotational_max_MHz']:.4f}",
                    )
                )
                + r" \\"
            )
    body = "\n".join(rows)
    return rf"""% Generated by `python -m matrix semiexp-benchmark --paper`.
\begin{{table*}}
\caption{{Planar-component diagnostics for the planar benchmarks. Only two
principal moments are independent for a planar molecule; the selected pair
marked by an asterisk is used in the normal equations, while all three
rotational constants are recomputed after convergence. The residuals are
RMS/maximum absolute MHz values over the complete diagnostic $A$, $B$, and $C$
constant set.}}
\label{{tab:planar-pair-diagnostics}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lclrrl@{{}}}}
\toprule
System & Pair & Fitted moments & Rank & Cond. & Diagnostic RMS/max \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _read_key_value_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["key"]: row["value"] for row in csv.DictReader(handle)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rotational_stats(path: Path) -> tuple[int, float, float]:
    if not path.is_file():
        return 0, 0.0, 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        residuals = [float(row["difference_MHz"]) for row in csv.DictReader(handle)]
    if not residuals:
        return 0, 0.0, 0.0
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return len(residuals), rms, max(abs(value) for value in residuals)


def _minimum_eigenvalue(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return min(float(row["eigenvalue"]) for row in rows)


def _validate_case(name: str, case: Any, label: str) -> None:
    if not isinstance(case, dict):
        raise SnapshotValidationError(f"{label}: case {name} must be an object")
    _require_fields(case, _CASE_REQUIRED_FIELDS, f"{label}: case {name}")
    _validate_relative_run_dir(case["run_dir"], f"{label}: case {name}.run_dir")
    if case["coordinate_model"] not in MODEL_LABELS:
        raise SnapshotValidationError(
            f"{label}: case {name} has unknown coordinate_model {case['coordinate_model']!r}"
        )
    if case["rotational_pair"] not in {"ABC", *PAIR_ORDER}:
        raise SnapshotValidationError(
            f"{label}: case {name} has invalid rotational_pair {case['rotational_pair']!r}"
        )
    if tuple(case["components"]) != _pair_to_components(case["rotational_pair"]):
        raise SnapshotValidationError(
            f"{label}: case {name} components do not match rotational_pair"
        )
    if case["reported_constants"] != case["isotopologues"] * 3:
        raise SnapshotValidationError(
            f"{label}: case {name} reported_constants must equal 3*isotopologues"
        )
    if case["rank"] > case["effective_parameters"]:
        raise SnapshotValidationError(f"{label}: case {name} rank exceeds effective_parameters")
    if case["totally_symmetric_gics"] > case["final_gics"]:
        raise SnapshotValidationError(
            f"{label}: case {name} active GIC count exceeds final GIC count"
        )
    for key in (
        "isotopologues",
        "reported_constants",
        "final_gics",
        "totally_symmetric_gics",
        "effective_parameters",
        "rank",
        "primitive_constraints",
        "accepted_steps",
        "rejected_steps",
    ):
        if case[key] < 0:
            raise SnapshotValidationError(f"{label}: case {name}.{key} must be non-negative")
    for key in (
        "condition_number",
        "rotational_rms_MHz",
        "rotational_max_MHz",
        "minimum_hessian_eigenvalue",
    ):
        _validate_finite_nonnegative(case[key], f"{label}: case {name}.{key}")
    if case["rotational_max_MHz"] < case["rotational_rms_MHz"]:
        raise SnapshotValidationError(f"{label}: case {name} max residual is smaller than RMS")


def _validate_planar_diagnostics(system: str, diagnostics: Any, label: str) -> None:
    if not isinstance(diagnostics, dict):
        raise SnapshotValidationError(f"{label}: planar diagnostics for {system} must be an object")
    selected = diagnostics.get("selected")
    if selected not in PAIR_ORDER:
        raise SnapshotValidationError(
            f"{label}: planar diagnostics for {system} has invalid selected pair"
        )
    pairs = diagnostics.get("pairs")
    if not isinstance(pairs, dict) or tuple(pairs) != PAIR_ORDER:
        raise SnapshotValidationError(
            f"{label}: planar diagnostics for {system} must contain ordered AB/AC/BC pairs"
        )
    for pair in PAIR_ORDER:
        pair_data = pairs[pair]
        _require_fields(pair_data, _PAIR_REQUIRED_FIELDS, f"{label}: planar {system}.{pair}")
        _validate_relative_run_dir(pair_data["run_dir"], f"{label}: planar {system}.{pair}.run_dir")
        for key in ("rank", "condition_number", "rotational_rms_MHz", "rotational_max_MHz"):
            _validate_finite_nonnegative(pair_data[key], f"{label}: planar {system}.{pair}.{key}")
        if pair_data["rotational_max_MHz"] < pair_data["rotational_rms_MHz"]:
            raise SnapshotValidationError(
                f"{label}: planar {system}.{pair} max residual is smaller than RMS"
            )
    selected_rms = pairs[selected]["rotational_rms_MHz"]
    best_rms = min(pair["rotational_rms_MHz"] for pair in pairs.values())
    if not math.isclose(selected_rms, best_rms, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise SnapshotValidationError(
            f"{label}: planar {system} selected pair is not the lowest-RMS pair"
        )


def _require_fields(data: Any, fields: dict[str, Any], label: str) -> None:
    if not isinstance(data, dict):
        raise SnapshotValidationError(f"{label}: expected object")
    missing = [key for key in fields if key not in data]
    if missing:
        raise SnapshotValidationError(f"{label}: missing fields {', '.join(missing)}")
    for key, expected_type in fields.items():
        if not isinstance(data[key], expected_type) or isinstance(data[key], bool):
            raise SnapshotValidationError(f"{label}: {key} has invalid type")


def _validate_relative_run_dir(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or value.startswith("~") or ".." in path.parts:
        raise SnapshotValidationError(f"{label}: run_dir must be a repository-relative path")


def _validate_finite_nonnegative(value: int | float, label: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise SnapshotValidationError(f"{label}: value must be finite and non-negative")


def _pair_to_components(pair: str) -> tuple[str, ...]:
    if pair == "ABC":
        return ("Ia", "Ib", "Ic")
    return tuple(f"I{axis.lower()}" for axis in pair)


def _parse_components(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    value = ast.literal_eval(raw)
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _components_to_pair(components: tuple[str, ...]) -> str:
    normalized = tuple(component.replace("I", "").lower() for component in components)
    if normalized == ("a", "b", "c"):
        return "ABC"
    if normalized == ("a", "b"):
        return "AB"
    if normalized == ("a", "c"):
        return "AC"
    if normalized == ("b", "c"):
        return "BC"
    return "".join(component.replace("I", "").upper() for component in components)


def _constraint_label(system: str, count: int) -> str:
    if count == 0:
        return "none"
    if system in {"glycine_I", "glycine_II"}:
        return f"{count} function"
    if system == "p-EBN":
        return f"{count} frozen"
    if system == "norcamphor":
        return f"{count} H + 58 pred."
    return f"{count} primitive"


def _steps_label(case: dict[str, Any]) -> str:
    stationary = str(case.get("stationary_point", ""))
    if stationary == "minimum":
        suffix = "min"
    elif stationary in {"positive_fitted_directions", "flat_or_rank_deficient"}:
        suffix = "pos."
    else:
        suffix = stationary or "-"
    return f"{case['accepted_steps']}/{case['rejected_steps']}; {suffix}"


def _default_pair_run_dir(system: str, pair: str) -> str | None:
    if system == "nitrobenzene":
        return f"working/semiexp/nitrobenzene/run_pair_{pair}"
    if system == "azulene":
        return f"working/semiexp/azulene/run_msr_like_planar_{pair}"
    return None


def _unique_primitive_constraints(parameters: Any) -> set[str]:
    if not isinstance(parameters, list):
        return set()
    return {
        _canonical_primitive_constraint(str(parameter))
        for parameter in parameters
        if _looks_like_hard_constraint_record(str(parameter))
    }


def _looks_like_hard_constraint_record(parameter: str) -> bool:
    text = parameter.strip()
    if not text:
        return False
    low = text.lower()
    if "inactive" in low or "remove" in low or "active" in low:
        return False
    if re.search(r"\b(value|frozen|freeze|fixed)\b", low):
        return True
    if re.search(r"\(\s*f\s*(?:[,)]|$)", low):
        return True
    if "=" in text:
        return False
    return bool(re.match(r"^(r|a|d|u|l|bond|angle|dihedral|out_of_plane|linear)", low))


def _canonical_primitive_constraint(parameter: str) -> str:
    match = re.fullmatch(r"([A-Za-z_]+)\(([^()]*)\)", parameter.strip())
    if not match:
        return parameter.strip().lower()
    name = match.group(1).lower()
    atoms = tuple(int(part.strip()) for part in match.group(2).split(",") if part.strip())
    if name == "bond" and len(atoms) == 2:
        atoms = tuple(sorted(atoms))
    elif name in {"angle", "dihedral"}:
        atoms = min(atoms, atoms[::-1])
    return f"{name}({','.join(str(atom) for atom in atoms)})"


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repository_root() / path
