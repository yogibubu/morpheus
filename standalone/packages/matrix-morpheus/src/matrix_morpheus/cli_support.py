"""Command support owned by the standalone MORPHEUS package.

The MATRIX suite imports these helpers as a compatibility surface, so the
standalone and aggregate CLIs execute one implementation.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")
UNHANDLED = object()

def _parse_fixed_parameters(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _split_top_level(raw, separators=",;") if part.strip())

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

def _parse_qm_predicates(items: list[str], predicate_type: type) -> tuple:
    predicates = []
    for item in items:
        parts = item.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError("--qm-predicate must be label_pattern:value:sigma[:source]")
        source = parts[3] if len(parts) == 4 else "qm"
        predicates.append(predicate_type(parts[0], float(parts[1]), float(parts[2]), source=source))
    return tuple(predicates)

def _parse_parameter_classes(items: list[str], class_type: type) -> tuple:
    constraints = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError("--parameter-class must be name:shared|fixed:pattern[|pattern...]")
        patterns = tuple(part.strip() for part in parts[2].split("|") if part.strip())
        constraints.append(class_type(parts[0].strip(), patterns, parts[1].strip()))
    return tuple(constraints)

def _primitive_class_budget(
    raw: str,
    *,
    observations: tuple,
    rotational_components: str,
) -> int | None:
    text = str(raw or "auto").strip().lower()
    if text == "all":
        return None
    if text == "auto":
        component_count = len(_semiexp_components_for_budget(rotational_components))
        return max(1, len(observations) * component_count)
    value = int(text)
    if value < 0:
        raise ValueError("--primitive-class-budget must be auto, all, or a non-negative integer")
    return value

def _semiexp_components_for_budget(rotational_components: str) -> tuple[str, ...]:
    text = str(rotational_components or "auto").upper()
    if text in {"AB", "AC", "BC"}:
        return tuple(text)
    return ("A", "B", "C")

def _semiexp_synthon_auto_score(
    result,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    xy_sigma_limit = 2.0e-3
    ch_sigma_limit = 5.0e-3
    heavy_angle_sigma_limit = 0.2
    max_xy_bond_sigma = 0.0
    max_xh_bond_sigma = 0.0
    max_ch_bond_sigma = 0.0
    max_heavy_angle_sigma = 0.0
    bond_violations = 0
    angle_violations = 0
    for parameter in result.geometry_parameters:
        symbols = tuple(getattr(parameter, "atom_symbols", ()) or ())
        if parameter.kind == "bond":
            sigma = float(parameter.sigma_angstrom or 0.0)
            if "H" not in symbols:
                max_xy_bond_sigma = max(max_xy_bond_sigma, sigma)
                if sigma > xy_sigma_limit:
                    bond_violations += 1
            elif "H" in symbols:
                if set(symbols) == {"C", "H"}:
                    limit = ch_sigma_limit
                    max_ch_bond_sigma = max(max_ch_bond_sigma, sigma)
                else:
                    limit = xy_sigma_limit
                    max_xh_bond_sigma = max(max_xh_bond_sigma, sigma)
                if sigma > limit:
                    bond_violations += 1
        elif parameter.kind == "angle":
            sigma = float(parameter.sigma_degree or 0.0)
            if "H" not in symbols:
                max_heavy_angle_sigma = max(max_heavy_angle_sigma, sigma)
                if sigma > heavy_angle_sigma_limit:
                    angle_violations += 1
    diagnostics = result.diagnostics
    rank_defect = max(0, int(diagnostics.n_optimized_parameters) - int(diagnostics.rank))
    condition = float(diagnostics.condition_number)
    if not math.isfinite(condition):
        condition = 1.0e99
    threshold_penalty = (
        max(0.0, max_xy_bond_sigma - xy_sigma_limit) / xy_sigma_limit
        + max(0.0, max_xh_bond_sigma - xy_sigma_limit) / xy_sigma_limit
        + max(0.0, max_ch_bond_sigma - ch_sigma_limit) / ch_sigma_limit
        + max(0.0, max_heavy_angle_sigma - heavy_angle_sigma_limit) / heavy_angle_sigma_limit
    )
    violations = bond_violations + angle_violations
    return (
        float(violations),
        float(rank_defect),
        float(threshold_penalty),
        float(max_xy_bond_sigma),
        float(max_xh_bond_sigma),
        float(max_ch_bond_sigma),
        float(max_heavy_angle_sigma),
        float(condition),
        -int(diagnostics.n_optimized_parameters),
    )

def _merge_unique(left: tuple[_T, ...], right: tuple[_T, ...]) -> tuple[_T, ...]:
    result: list[_T] = []
    for item in (*left, *right):
        if item not in result:
            result.append(item)
    return tuple(result)

def _job_default(value: _T, default: _T, job_value: _T | None) -> _T:
    if job_value is not None and value == default:
        return job_value
    return value

def _sensitivity_min_fit_count(raw: str) -> int | None:
    text = str(raw or "auto").strip().lower()
    if text in {"auto", ""}:
        return None
    if text in {"none", "off", "threshold", "threshold-only"}:
        return 0
    value = int(text)
    if value < 0:
        raise ValueError("--sensitivity-min-fit must be auto, none, or a non-negative integer")
    return value

def _sensitivity_safe_apply_gate(
    *,
    base_request,
    candidate_request,
    fit_semiexperimental_geometry,
    outdir: Path,
    max_iter: int | None,
    step: float,
    damping: float,
    max_step: float,
    prune_condition: float,
    rot_rel_tol: float,
    rot_abs_tol: float,
    condition_factor: float,
    max_bond_delta: float,
    max_angle_delta: float,
) -> dict[str, object]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        base = fit_semiexperimental_geometry(
            base_request,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            outdir=root / "chemical_model",
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "chemical_model_invalid",
            "base_error": f"{type(exc).__name__}: {exc}",
            "action": (
                "Add chemical predicates, parameter classes, or fixed coordinates "
                "until the base MORPHEUS model is publishable; rerun the sensitivity "
                "advisor only as a conservative tuning step."
            ),
        }
    try:
        candidate = fit_semiexperimental_geometry(
            candidate_request,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            outdir=root / "advisor_model",
        )
    except Exception as exc:
        return {
            "accepted": False,
            "reason": "candidate_preflight_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    base_rot = _semiexp_rotational_rms(base)
    candidate_rot = _semiexp_rotational_rms(candidate)
    max_dr, max_da = _semiexp_geometry_delta(base, candidate)
    base_condition = float(base.diagnostics.condition_number)
    candidate_condition = float(candidate.diagnostics.condition_number)
    if not math.isfinite(base_condition):
        base_condition = 1.0e99
    if not math.isfinite(candidate_condition):
        candidate_condition = 1.0e99
    reasons: list[str] = []
    rot_limit = base_rot * (1.0 + float(rot_rel_tol)) + float(rot_abs_tol)
    if candidate_rot > rot_limit:
        reasons.append("rotational_rms_worse")
    if int(candidate.diagnostics.rank) < int(base.diagnostics.rank):
        reasons.append("rank_lower")
    if candidate_condition > max(base_condition * float(condition_factor), base_condition):
        reasons.append("condition_worse")
    if max_dr > float(max_bond_delta):
        reasons.append("geometry_bond_drift")
    if max_da > float(max_angle_delta):
        reasons.append("geometry_angle_drift")
    return {
        "accepted": not reasons,
        "reason": "accepted" if not reasons else ",".join(reasons),
        "base_rotational_rms_MHz": base_rot,
        "candidate_rotational_rms_MHz": candidate_rot,
        "rotational_rms_limit_MHz": rot_limit,
        "base_rank": int(base.diagnostics.rank),
        "candidate_rank": int(candidate.diagnostics.rank),
        "base_condition_number": base_condition,
        "candidate_condition_number": candidate_condition,
        "max_bond_delta_A": max_dr,
        "max_angle_delta_deg": max_da,
        "max_bond_delta_limit_A": float(max_bond_delta),
        "max_angle_delta_limit_deg": float(max_angle_delta),
    }

def _semiexp_rotational_rms(result) -> float:
    diffs = [float(row.difference_MHz) for row in result.rotational_constants]
    return math.sqrt(sum(diff * diff for diff in diffs) / len(diffs)) if diffs else 0.0

def _semiexp_geometry_delta(base, candidate) -> tuple[float, float]:
    base_rows = {(row.kind, row.label): row for row in getattr(base, "geometry_parameters", ())}
    max_bond = 0.0
    max_angle = 0.0
    for row in getattr(candidate, "geometry_parameters", ()):
        base_row = base_rows.get((row.kind, row.label))
        if base_row is None:
            continue
        if row.value_angstrom is not None and base_row.value_angstrom is not None:
            max_bond = max(
                max_bond, abs(float(row.value_angstrom) - float(base_row.value_angstrom))
            )
        if row.value_degree is not None and base_row.value_degree is not None:
            delta = (float(row.value_degree) - float(base_row.value_degree) + 180.0) % 360.0 - 180.0
            max_angle = max(max_angle, abs(delta))
    return max_bond, max_angle

def _semiexp_aligned_displacements(result) -> tuple[float, float]:
    """Return rigid-body-aligned maximum and RMS atom displacements in Angstrom."""

    import numpy as np

    initial = np.asarray(result.initial_coordinates_angstrom, dtype=float)
    final = np.asarray(result.final_coordinates_angstrom, dtype=float)
    if initial.shape != final.shape or initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("MORPHEUS returned incompatible initial/final Cartesian geometries")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
        raise ValueError("MORPHEUS returned non-finite Cartesian coordinates")
    initial_centered = initial - np.mean(initial, axis=0)
    final_centered = final - np.mean(final, axis=0)
    left, _, right_t = np.linalg.svd(final_centered.T @ initial_centered)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = final_centered @ rotation
    displacement = np.linalg.norm(aligned - initial_centered, axis=1)
    return float(np.max(displacement, initial=0.0)), float(
        np.sqrt(np.mean(displacement * displacement)) if displacement.size else 0.0
    )

def _semiexp_fit_comparison_contract(
    *,
    free_result,
    constrained_result,
    displacement_limit: float,
    regularization_predicates: tuple[object, ...],
    regularization_scale: float,
    excluded_rotational_constants: tuple[str, ...],
    advisor_rows: tuple[object, ...] = (),
) -> dict[str, object]:
    """Serialize the scientific distinction between free and constrained fits."""

    if displacement_limit <= 0.0 or not math.isfinite(displacement_limit):
        raise ValueError("A finite positive displacement limit is required for fit comparison")

    def summary(result) -> dict[str, object]:
        max_displacement, rms_displacement = _semiexp_aligned_displacements(result)
        full_rank = result.diagnostics.rank == result.diagnostics.n_optimized_parameters
        well_conditioned = math.isfinite(result.diagnostics.condition_number) and (
            result.diagnostics.condition_number <= 1.0e8
        )
        return {
            "rotational_rms_MHz": _semiexp_rotational_rms(result),
            "max_atom_displacement_A": max_displacement,
            "rms_atom_displacement_A": rms_displacement,
            "within_displacement_limit": max_displacement <= displacement_limit,
            "rank": result.diagnostics.rank,
            "n_optimized_parameters": result.diagnostics.n_optimized_parameters,
            "condition_number": result.diagnostics.condition_number,
            "stationary_point": result.stationary_point,
            "full_rank": full_rank,
            "well_conditioned": well_conditioned,
        }

    advisor_by_id = {
        str(row.label).split()[0]: row
        for row in advisor_rows
        if getattr(row, "predicate_sigma", 0.0) > 0.0
    }

    def predicate_record(predicate) -> dict[str, object]:
        row = advisor_by_id.get(predicate.label_pattern)
        full_label = str(row.label) if row is not None else predicate.label_pattern
        lower = full_label.lower()
        unit = "angstrom" if "str" in lower or "bond" in lower else "radian"
        return {
            "label": predicate.label_pattern,
            "definition": full_label,
            "chemical_role": str(getattr(row, "chemical_role", "soft")),
            "center": float(predicate.value),
            "sigma": float(predicate.sigma),
            "unit": unit,
            "source": predicate.source,
        }

    return {
        "schema": "matrix.morpheus.fit_comparison.v1",
        "displacement_limit_A": displacement_limit,
        "observation_policy": "explicit_exclusion_only",
        "excluded_rotational_constants": list(excluded_rotational_constants),
        "free_fit": summary(free_result),
        "constrained_fit": summary(constrained_result),
        "constraint_model": {
            "kind": "gaussian_priors_on_sensitivity_selected_soft_sonic_coordinates",
            "center": "input_sonic_values",
            "scale": regularization_scale,
            "count": len(regularization_predicates),
            "predicates": [predicate_record(predicate) for predicate in regularization_predicates],
        },
    }

def _write_sensitivity_gate_summary(path: Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _prune_semiexp_delivery_artifacts(
    outdir: Path,
    keep_names: set[str],
    *,
    extra_outputs: dict[str, Path] | None = None,
) -> tuple[str, ...]:
    """Reduce a reliable run directory to its coauthor-facing delivery files."""

    import shutil

    root = Path(outdir).resolve()
    if not root.is_dir():
        raise ValueError(f"MORPHEUS output directory does not exist: {root}")
    retained = set(keep_names)
    for child in root.iterdir():
        if child.name in retained:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    manifest_path = root / "semiexp_manifest.json"
    if manifest_path.is_file():
        from matrix_core.manifest import sha256_file

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_names = {
            "geometry": root / "semiexp_geometry.xyz",
            "html_report": root / "semiexp_report.html",
            "latex_standalone": root / "semiexp_results.tex",
            "latex_pdf": root / "semiexp_results.pdf",
            "geometry_safety": root / "semiexp_geometry_safety.json",
        }
        output_names.update(extra_outputs or {})
        data["outputs"] = {name: str(path) for name, path in output_names.items() if path.is_file()}
        data["output_sha256"] = {
            name: sha256_file(path) for name, path in output_names.items() if path.is_file()
        }
        data.setdefault("parameters", {})["delivery_cleanup"] = "reliable_result_minimal"
        data["parameters"]["delivery_files"] = sorted(
            child.name for child in root.iterdir() if child.is_file()
        )
        manifest_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return tuple(sorted(child.name for child in root.iterdir()))

def _compile_semiexperimental_latex(path: Path) -> Path:
    """Compile the standalone coauthor report and require a valid PDF artifact."""

    import shutil

    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"Standalone MORPHEUS LaTeX source does not exist: {source}")
    pdflatex = shutil.which("pdflatex")
    tectonic = shutil.which("tectonic")
    if pdflatex is None and tectonic is None:
        raise RuntimeError(
            "MORPHEUS cannot complete the reliable delivery: neither pdflatex nor "
            "tectonic is installed"
        )
    pdf = source.with_suffix(".pdf")
    commands = []
    if pdflatex is not None:
        commands.append(
            (
                "pdflatex",
                (
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(source.parent),
                    str(source),
                ),
            )
        )
    if tectonic is not None:
        commands.append(
            (
                "tectonic",
                (tectonic, "--outdir", str(source.parent), str(source)),
            )
        )
    failures = []
    for label, command in commands:
        pdf.unlink(missing_ok=True)
        completed = subprocess.run(
            command,
            cwd=source.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode == 0 and pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-"):
            print(f"morpheus_latex_pdf: {pdf}")
            return pdf
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        failures.append(f"{label}: {tail or 'no diagnostic output'}")
    raise RuntimeError("MORPHEUS standalone LaTeX compilation failed:\n" + "\n".join(failures))

def _append_manifest_output(manifest_path: Path, name: str, path: Path) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("outputs", {})[name] = str(path)
    if path.is_file():
        from matrix_core.manifest import sha256_file

        data.setdefault("output_sha256", {})[name] = sha256_file(path)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _ensemble_output_paths(outdir: Path) -> dict[str, Path]:
    root = Path(outdir)
    return {
        "text_report": root / "ensemble_class_corrections.txt",
        "class_corrections_csv": root / "ensemble_class_corrections.csv",
        "class_report_csv": root / "ensemble_class_report.csv",
        "molecule_blocks_csv": root / "ensemble_molecule_blocks.csv",
        "scientific_manifest": root / "ensemble_manifest.json",
        "covariance_csv": root / "ensemble_covariance.csv",
        "correlation_csv": root / "ensemble_correlation.csv",
    }
