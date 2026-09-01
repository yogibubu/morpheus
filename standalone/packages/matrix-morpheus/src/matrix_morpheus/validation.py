from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np

from matrix_chem.geometry_io import write_xyz

from .contracts import (
    HYDROGEN_PARAMETER_CONSTRAINT,
    QMParameterPredicate,
    SemiexperimentalFitRequest,
)
from .fit import fit_semiexperimental_geometry
from .models import SemiexperimentalFitResult


FINAL_VALIDATION_SCHEMA = "oracle.semiexp.final_validation.v1"


@dataclass(frozen=True)
class SemiexperimentalFinalValidationOptions:
    coordinate_check: bool = True
    huber_check: bool = True
    predicate_scan_scales: tuple[float, ...] = (0.5, 2.0)
    leave_predicate_groups: bool = True
    max_predicate_groups: int = 12
    multistart: int = 0
    multistart_sigma_angstrom: float = 0.001
    random_seed: int = 20260703
    geometry_rmsd_warning_angstrom: float = 0.001
    max_atom_shift_warning_angstrom: float = 0.003


@dataclass(frozen=True)
class SemiexperimentalPredicateAuditRow:
    index: int
    source: str
    label_pattern: str
    primitive_kind: str
    value: float
    sigma: float
    unit: str
    matches: int
    involves_hydrogen: bool
    role: str


@dataclass(frozen=True)
class SemiexperimentalValidationRunRow:
    label: str
    check: str
    status: str
    rotational_rms_MHz: float
    weighted_rms: float
    rank: int
    condition_number: float
    convergence_reason: str
    stationary_point: str
    geometry_rmsd_angstrom: float
    max_atom_shift_angstrom: float
    robust_downweighted_observations: int
    notes: str = ""
    outdir: str = ""
    error: str = ""


@dataclass(frozen=True)
class SemiexperimentalFinalValidationIssue:
    severity: str
    code: str
    message: str
    context: str = ""


@dataclass(frozen=True)
class SemiexperimentalFinalValidationResult:
    outdir: Path
    summary: dict[str, object]
    predicate_audit: tuple[SemiexperimentalPredicateAuditRow, ...]
    runs: tuple[SemiexperimentalValidationRunRow, ...]
    issues: tuple[SemiexperimentalFinalValidationIssue, ...]

    @property
    def summary_path(self) -> Path:
        return self.outdir / "semiexp_final_validation.json"

    @property
    def runs_path(self) -> Path:
        return self.outdir / "semiexp_final_validation_runs.csv"

    @property
    def predicate_audit_path(self) -> Path:
        return self.outdir / "semiexp_predicate_audit.csv"

    @property
    def issues_path(self) -> Path:
        return self.outdir / "semiexp_final_validation_issues.csv"


def run_semiexperimental_final_validation(
    request: SemiexperimentalFitRequest,
    result: SemiexperimentalFitResult,
    outdir: Path,
    *,
    options: SemiexperimentalFinalValidationOptions | None = None,
    max_iter: int | None = None,
    step: float = 1.0e-4,
    damping: float = 1.0e-8,
    max_step: float = 0.25,
    prune_condition: float = 0.0,
) -> SemiexperimentalFinalValidationResult:
    """Run publication-level robustness checks for a completed SEfit."""

    opts = options or SemiexperimentalFinalValidationOptions()
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    predicate_audit = audit_semiexperimental_predicates(request, result)
    runs: list[SemiexperimentalValidationRunRow] = [
        _run_row("reference", "reference", result, result)
    ]

    if opts.coordinate_check:
        target_model = (
            "cartesian_symmetry" if request.coordinate_model == "gic" else "gic"
        )
        runs.append(
            _fit_validation_variant(
                "coordinate_model",
                f"coordinate_{target_model}",
                replace(request, coordinate_model=target_model, leave_one_out=False),
                result,
                root / "coordinate_model",
                max_iter=max_iter,
                step=step,
                damping=damping,
                max_step=max_step,
                prune_condition=prune_condition,
            )
        )

    if opts.huber_check and request.robust_loss != "huber":
        runs.append(
            _fit_validation_variant(
                "robust_loss",
                "huber",
                replace(request, robust_loss="huber", leave_one_out=False),
                result,
                root / "robust_huber",
                max_iter=max_iter,
                step=step,
                damping=damping,
                max_step=max_step,
                prune_condition=prune_condition,
            )
        )

    predicate_groups = _predicate_groups(request.qm_predicates)
    selected_groups = tuple(predicate_groups)[: max(0, int(opts.max_predicate_groups))]
    for group in selected_groups:
        if opts.leave_predicate_groups:
            filtered = tuple(
                predicate
                for predicate in request.qm_predicates
                if _predicate_group_key(predicate) != group
            )
            runs.append(
                _fit_validation_variant(
                    "leave_predicate_group_out",
                    f"leave_out_{_safe_label(group)}",
                    replace(request, qm_predicates=filtered, leave_one_out=False),
                    result,
                    root / "leave_predicate_group_out" / _safe_label(group),
                    max_iter=max_iter,
                    step=step,
                    damping=damping,
                    max_step=max_step,
                    prune_condition=prune_condition,
                )
            )
        for scale in opts.predicate_scan_scales:
            if scale <= 0.0 or abs(float(scale) - 1.0) < 1.0e-12:
                continue
            scanned = tuple(
                replace(predicate, sigma=float(predicate.sigma) * float(scale))
                if _predicate_group_key(predicate) == group
                else predicate
                for predicate in request.qm_predicates
            )
            runs.append(
                _fit_validation_variant(
                    "predicate_sigma_scan",
                    f"{_safe_label(group)}_x{float(scale):.6g}",
                    replace(request, qm_predicates=scanned, leave_one_out=False),
                    result,
                    root / "predicate_sigma_scan" / f"{_safe_label(group)}_x{float(scale):.6g}",
                    max_iter=max_iter,
                    step=step,
                    damping=damping,
                    max_step=max_step,
                    prune_condition=prune_condition,
                )
            )

    if opts.multistart > 0:
        rng = np.random.default_rng(int(opts.random_seed))
        for index in range(int(opts.multistart)):
            start_dir = root / "multistart" / f"start_{index + 1:03d}"
            start_dir.mkdir(parents=True, exist_ok=True)
            perturbed = np.asarray(result.initial_coordinates_angstrom, dtype=float) + rng.normal(
                0.0,
                float(opts.multistart_sigma_angstrom),
                size=np.asarray(result.initial_coordinates_angstrom).shape,
            )
            start_xyz = start_dir / "initial_geometry.xyz"
            write_xyz(
                start_xyz,
                result.atoms,
                perturbed,
                comment=(
                    "MATRIX/MORPHEUS final-validation multistart geometry; "
                    f"sigma={opts.multistart_sigma_angstrom:g} A"
                ),
            )
            runs.append(
                _fit_validation_variant(
                    "multistart",
                    f"start_{index + 1:03d}",
                    replace(request, initial_geometry=start_xyz, leave_one_out=False),
                    result,
                    start_dir / "fit",
                    max_iter=max_iter,
                    step=step,
                    damping=damping,
                    max_step=max_step,
                    prune_condition=prune_condition,
                )
            )

    summary = _validation_summary(request, result, opts, predicate_audit, tuple(runs))
    issues = _validation_issues(request, result, opts, predicate_audit, tuple(runs), summary)
    validation = SemiexperimentalFinalValidationResult(
        root,
        summary,
        predicate_audit,
        tuple(runs),
        tuple(issues),
    )
    _write_final_validation_artifacts(validation)
    return validation


def audit_semiexperimental_predicates(
    request: SemiexperimentalFitRequest,
    result: SemiexperimentalFitResult,
) -> tuple[SemiexperimentalPredicateAuditRow, ...]:
    atoms = tuple(str(atom).strip().capitalize() for atom in result.atoms)
    labels = tuple(result.gic_labels or ())
    rows: list[SemiexperimentalPredicateAuditRow] = []
    for index, predicate in enumerate(request.qm_predicates, start=1):
        primitive_kind, atom_indices = _predicate_primitive_signature(predicate.label_pattern)
        unit = "angstrom" if primitive_kind == "distance" else "degree"
        if primitive_kind == "gic":
            unit = "native"
        matches = 1 if atom_indices else _count_label_matches(predicate.label_pattern, labels)
        involves_h = any(1 <= atom <= len(atoms) and atoms[atom - 1] == "H" for atom in atom_indices)
        rows.append(
            SemiexperimentalPredicateAuditRow(
                index=index,
                source=str(predicate.source),
                label_pattern=str(predicate.label_pattern),
                primitive_kind=primitive_kind,
                value=float(predicate.value),
                sigma=float(predicate.sigma),
                unit=unit,
                matches=int(matches),
                involves_hydrogen=bool(involves_h),
                role=_predicate_role(predicate, primitive_kind, involves_h),
            )
        )
    return tuple(rows)


def final_validation_runs_csv(rows: tuple[SemiexperimentalValidationRunRow, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "label",
            "check",
            "status",
            "rotational_rms_MHz",
            "weighted_rms",
            "rank",
            "condition_number",
            "convergence_reason",
            "stationary_point",
            "geometry_rmsd_angstrom",
            "max_atom_shift_angstrom",
            "robust_downweighted_observations",
            "notes",
            "outdir",
            "error",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.label,
                row.check,
                row.status,
                f"{row.rotational_rms_MHz:.12g}",
                f"{row.weighted_rms:.12g}",
                row.rank,
                f"{row.condition_number:.12g}",
                row.convergence_reason,
                row.stationary_point,
                f"{row.geometry_rmsd_angstrom:.12g}",
                f"{row.max_atom_shift_angstrom:.12g}",
                row.robust_downweighted_observations,
                row.notes,
                row.outdir,
                row.error,
            ]
        )
    return stream.getvalue()


def predicate_audit_csv(rows: tuple[SemiexperimentalPredicateAuditRow, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "index",
            "source",
            "label_pattern",
            "primitive_kind",
            "value",
            "sigma",
            "unit",
            "matches",
            "involves_hydrogen",
            "role",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.index,
                row.source,
                row.label_pattern,
                row.primitive_kind,
                f"{row.value:.12g}",
                f"{row.sigma:.12g}",
                row.unit,
                row.matches,
                int(row.involves_hydrogen),
                row.role,
            ]
        )
    return stream.getvalue()


def final_validation_issues_csv(rows: tuple[SemiexperimentalFinalValidationIssue, ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["severity", "code", "message", "context"])
    for row in rows:
        writer.writerow([row.severity, row.code, row.message, row.context])
    return stream.getvalue()


def _fit_validation_variant(
    check: str,
    label: str,
    request: SemiexperimentalFitRequest,
    reference: SemiexperimentalFitResult,
    outdir: Path,
    *,
    max_iter: int | None,
    step: float,
    damping: float,
    max_step: float,
    prune_condition: float,
) -> SemiexperimentalValidationRunRow:
    try:
        result = fit_semiexperimental_geometry(
            request,
            max_iter=max_iter,
            step=step,
            damping=damping,
            max_step=max_step,
            prune_condition=prune_condition,
            outdir=outdir,
        )
    except Exception as exc:
        return SemiexperimentalValidationRunRow(
            label=label,
            check=check,
            status="error",
            rotational_rms_MHz=float("nan"),
            weighted_rms=float("nan"),
            rank=0,
            condition_number=float("nan"),
            convergence_reason="error",
            stationary_point="not_checked",
            geometry_rmsd_angstrom=float("nan"),
            max_atom_shift_angstrom=float("nan"),
            robust_downweighted_observations=0,
            outdir=str(outdir),
            error=f"{type(exc).__name__}: {exc}",
        )
    return _run_row(label, check, result, reference, outdir=outdir)


def _run_row(
    label: str,
    check: str,
    result: SemiexperimentalFitResult,
    reference: SemiexperimentalFitResult,
    *,
    outdir: Path | None = None,
) -> SemiexperimentalValidationRunRow:
    rmsd, max_shift = _geometry_delta(
        result.final_coordinates_angstrom,
        reference.final_coordinates_angstrom,
    )
    return SemiexperimentalValidationRunRow(
        label=label,
        check=check,
        status="ok",
        rotational_rms_MHz=_rotational_rms_MHz(result),
        weighted_rms=float(result.diagnostics.weighted_rms),
        rank=int(result.diagnostics.rank),
        condition_number=float(result.diagnostics.condition_number),
        convergence_reason=str(result.diagnostics.convergence_reason),
        stationary_point=str(result.stationary_point),
        geometry_rmsd_angstrom=rmsd,
        max_atom_shift_angstrom=max_shift,
        robust_downweighted_observations=int(result.diagnostics.robust_downweighted_observations),
        outdir="" if outdir is None else str(outdir),
    )


def _validation_summary(
    request: SemiexperimentalFitRequest,
    result: SemiexperimentalFitResult,
    options: SemiexperimentalFinalValidationOptions,
    predicate_audit: tuple[SemiexperimentalPredicateAuditRow, ...],
    runs: tuple[SemiexperimentalValidationRunRow, ...],
) -> dict[str, object]:
    exp_diag = _weight_summary(result.weight_diagnostics, "experimental")
    pred_diag = _weight_summary(result.weight_diagnostics, "predicate")
    loo = _leave_one_out_summary(result)
    structural = _structural_uncertainty_summary(result)
    successful = tuple(row for row in runs if row.status == "ok" and row.label != "reference")
    return {
        "schema": FINAL_VALIDATION_SCHEMA,
        "reproducibility": _reproducibility_payload(request, result, options),
        "model": {
            "coordinate_model": request.coordinate_model,
            "observable": result.diagnostics.observable,
            "components": result.diagnostics.components,
            "rank": int(result.diagnostics.rank),
            "n_optimized_parameters": int(result.diagnostics.n_optimized_parameters),
            "condition_number": float(result.diagnostics.condition_number),
            "stationary_point": result.stationary_point,
            "convergence_reason": result.diagnostics.convergence_reason,
            "rotational_rms_MHz": _rotational_rms_MHz(result),
            "weighted_rms": float(result.diagnostics.weighted_rms),
            "reduced_chi_square": float(result.diagnostics.reduced_chi_square),
            "minimum_hessian_eigenvalue": _min_finite(result.hessian_eigenvalues),
            "max_abs_correlation": _max_offdiag_abs(result.correlation),
        },
        "predicate_audit": {
            "n_predicates": len(predicate_audit),
            "n_hydrogen_predicates": sum(1 for row in predicate_audit if row.involves_hydrogen),
            "n_kraitchman_predicates": sum(1 for row in predicate_audit if row.role == "kraitchman"),
            "n_zero_match_predicates": sum(1 for row in predicate_audit if row.matches == 0),
            "hydrogen_blocked": HYDROGEN_PARAMETER_CONSTRAINT in request.fixed_parameters,
            "fixed_parameters": request.fixed_parameters,
        },
        "influence": {
            "experimental": exp_diag,
            "predicate": pred_diag,
        },
        "leave_one_out": loo,
        "structural_uncertainty": structural,
        "robustness_runs": {
            "requested": asdict(options),
            "n_runs": len(runs),
            "n_successful_stress_runs": len(successful),
            "n_failed_stress_runs": sum(1 for row in runs if row.status != "ok"),
            "max_geometry_rmsd_angstrom": _max_value(
                row.geometry_rmsd_angstrom for row in successful
            ),
            "max_atom_shift_angstrom": _max_value(
                row.max_atom_shift_angstrom for row in successful
            ),
            "max_rotational_rms_MHz": _max_value(row.rotational_rms_MHz for row in successful),
            "max_condition_number": _max_value(row.condition_number for row in successful),
        },
    }


def _validation_issues(
    request: SemiexperimentalFitRequest,
    result: SemiexperimentalFitResult,
    options: SemiexperimentalFinalValidationOptions,
    predicate_audit: tuple[SemiexperimentalPredicateAuditRow, ...],
    runs: tuple[SemiexperimentalValidationRunRow, ...],
    summary: dict[str, object],
) -> tuple[SemiexperimentalFinalValidationIssue, ...]:
    issues: list[SemiexperimentalFinalValidationIssue] = []

    def add(severity: str, code: str, message: str, context: str = "") -> None:
        issues.append(SemiexperimentalFinalValidationIssue(severity, code, message, context))

    if HYDROGEN_PARAMETER_CONSTRAINT in request.fixed_parameters:
        add(
            "warning",
            "hydrogen_parameters_blocked",
            "Hydrogen coordinates are fixed rather than controlled by soft predicates.",
            HYDROGEN_PARAMETER_CONSTRAINT,
        )
    if any(row.matches == 0 for row in predicate_audit):
        add(
            "warning",
            "predicate_without_match",
            "At least one non-primitive predicate pattern matches no final GIC label.",
        )
    if result.diagnostics.rank < result.diagnostics.n_optimized_parameters:
        add(
            "error",
            "rank_deficient",
            "The final linearized least-squares model is rank deficient.",
            f"rank={result.diagnostics.rank};parameters={result.diagnostics.n_optimized_parameters}",
        )
    condition = float(result.diagnostics.condition_number)
    if not np.isfinite(condition) or condition > 1.0e8:
        add(
            "warning",
            "ill_conditioned",
            "The final weighted Jacobian is ill-conditioned.",
            f"condition_number={condition:.6g}",
        )
    if result.stationary_point != "minimum":
        add(
            "warning",
            "not_strict_minimum",
            "The least-squares Hessian does not classify the solution as a strict minimum.",
            result.stationary_point,
        )
    for row in runs:
        if row.status != "ok":
            add(
                "warning",
                "validation_refit_failed",
                "A robustness refit failed.",
                f"{row.check}:{row.label}:{row.error}",
            )
            continue
        if row.label == "reference":
            continue
        if row.convergence_reason == "max_iter":
            add(
                "warning",
                "validation_refit_not_converged",
                "A robustness refit reached the iteration limit.",
                f"{row.check}:{row.label};condition_number={row.condition_number:.6g}",
            )
        if row.stationary_point != "minimum":
            add(
                "warning",
                "validation_refit_not_minimum",
                "A robustness refit did not end at a strict minimum.",
                f"{row.check}:{row.label};stationary_point={row.stationary_point}",
            )
        if row.rank < result.diagnostics.rank:
            add(
                "warning",
                "validation_refit_rank_loss",
                "A robustness refit has lower numerical rank than the reference fit.",
                f"{row.check}:{row.label};rank={row.rank};reference_rank={result.diagnostics.rank}",
            )
        if (
            np.isfinite(row.condition_number)
            and np.isfinite(float(result.diagnostics.condition_number))
            and row.condition_number > 10.0 * float(result.diagnostics.condition_number)
        ):
            add(
                "info",
                "validation_refit_condition_increase",
                "A robustness refit is substantially less well conditioned than the reference fit.",
                f"{row.check}:{row.label};condition_number={row.condition_number:.6g}",
            )
        if row.geometry_rmsd_angstrom > options.geometry_rmsd_warning_angstrom:
            add(
                "info",
                "stress_geometry_shift",
                "A robustness refit changed the final geometry beyond the RMSD reporting threshold.",
                f"{row.check}:{row.label};rmsd_A={row.geometry_rmsd_angstrom:.6g}",
            )
        if row.max_atom_shift_angstrom > options.max_atom_shift_warning_angstrom:
            add(
                "info",
                "stress_atom_shift",
                "A robustness refit moved at least one atom beyond the displacement reporting threshold.",
                f"{row.check}:{row.label};max_atom_shift_A={row.max_atom_shift_angstrom:.6g}",
            )
    return tuple(issues)


def _write_final_validation_artifacts(validation: SemiexperimentalFinalValidationResult) -> None:
    validation.summary_path.write_text(
        json.dumps(validation.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation.runs_path.write_text(final_validation_runs_csv(validation.runs), encoding="utf-8")
    validation.predicate_audit_path.write_text(
        predicate_audit_csv(validation.predicate_audit),
        encoding="utf-8",
    )
    validation.issues_path.write_text(
        final_validation_issues_csv(validation.issues),
        encoding="utf-8",
    )


def _predicate_groups(
    predicates: tuple[QMParameterPredicate, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_predicate_group_key(predicate) for predicate in predicates))


def _predicate_group_key(predicate: QMParameterPredicate) -> str:
    primitive_kind, _atoms = _predicate_primitive_signature(predicate.label_pattern)
    return f"{predicate.source}:{primitive_kind}"


def _predicate_primitive_signature(label_pattern: str) -> tuple[str, tuple[int, ...]]:
    text = str(label_pattern).strip()
    head = text.split("(", 1)[0].strip().upper()
    kind = {
        "R": "distance",
        "B": "distance",
        "A": "angle",
        "V": "angle",
        "D": "dihedral",
        "T": "dihedral",
        "U": "out_of_plane",
        "O": "out_of_plane",
        "L": "linear_bend",
    }.get(head, "gic")
    if "(" not in text or ")" not in text:
        return kind, ()
    body = text[text.find("(") + 1 : text.find(")")]
    atoms: list[int] = []
    for token in body.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            atoms.append(value)
    return kind, tuple(atoms)


def _count_label_matches(pattern: str, labels: tuple[str, ...]) -> int:
    low = str(pattern).lower()
    return sum(1 for label in labels if low in label.lower())


def _predicate_role(
    predicate: QMParameterPredicate,
    primitive_kind: str,
    involves_hydrogen: bool,
) -> str:
    source = str(predicate.source).lower()
    if "kraitchman" in source:
        return "kraitchman"
    if involves_hydrogen:
        return "hydrogen_frame"
    if primitive_kind == "gic":
        return "gic_prior"
    return "structural_prior"


def _rotational_rms_MHz(result: SemiexperimentalFitResult) -> float:
    diffs = np.asarray([row.difference_MHz for row in result.rotational_constants], dtype=float)
    return float(np.sqrt(np.mean(diffs * diffs))) if diffs.size else 0.0


def _geometry_delta(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.shape != ref.shape or cand.size == 0:
        return float("nan"), float("nan")
    atom_shifts = np.linalg.norm(cand - ref, axis=1)
    return float(np.sqrt(np.mean(atom_shifts * atom_shifts))), float(np.max(atom_shifts))


def _weight_summary(rows: tuple[object, ...], kind: str) -> dict[str, object]:
    subset = tuple(row for row in rows if getattr(row, "kind", "") == kind)
    if not subset:
        return {
            "rows": 0,
            "max_leverage": 0.0,
            "max_abs_studentized_residual": 0.0,
            "max_cooks_distance": 0.0,
            "max_chi_square_contribution": 0.0,
            "weight_fraction": 0.0,
        }
    return {
        "rows": len(subset),
        "max_leverage": _max_value(getattr(row, "leverage", 0.0) for row in subset),
        "max_abs_studentized_residual": _max_value(
            abs(float(getattr(row, "studentized_residual", 0.0))) for row in subset
        ),
        "max_cooks_distance": _max_value(getattr(row, "cooks_distance", 0.0) for row in subset),
        "max_chi_square_contribution": _max_value(
            getattr(row, "chi_square_contribution", 0.0) for row in subset
        ),
        "weight_fraction": float(
            sum(float(getattr(row, "total_weight_fraction", 0.0)) for row in subset)
        ),
    }


def _leave_one_out_summary(result: SemiexperimentalFitResult) -> dict[str, object]:
    rows = result.leave_one_out
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "max_omitted_rotational_rms_MHz": _max_value(
            row.omitted_rotational_rms_MHz for row in rows
        ),
        "max_omitted_rotational_max_abs_MHz": _max_value(
            row.omitted_rotational_max_abs_MHz for row in rows
        ),
        "max_cartesian_rms_shift_angstrom": _max_value(row.cartesian_rms_shift_angstrom for row in rows),
        "max_cartesian_max_shift_angstrom": _max_value(row.cartesian_max_shift_angstrom for row in rows),
        "min_rank": int(min(row.rank for row in rows)),
        "max_condition_number": _max_value(row.condition_number for row in rows),
    }


def _structural_uncertainty_summary(result: SemiexperimentalFitResult) -> dict[str, object]:
    buckets: dict[str, list[float]] = {
        "heavy_bond_sigma_A": [],
        "ch_bond_sigma_A": [],
        "xh_bond_sigma_A": [],
        "heavy_angle_sigma_deg": [],
        "h_angle_sigma_deg": [],
    }
    for row in result.geometry_parameters:
        symbols = tuple(getattr(row, "atom_symbols", ()) or ())
        if row.kind == "bond" and row.sigma_angstrom is not None:
            if "H" not in symbols:
                buckets["heavy_bond_sigma_A"].append(float(row.sigma_angstrom))
            elif set(symbols) == {"C", "H"}:
                buckets["ch_bond_sigma_A"].append(float(row.sigma_angstrom))
            else:
                buckets["xh_bond_sigma_A"].append(float(row.sigma_angstrom))
        if row.kind == "angle" and row.sigma_degree is not None:
            if "H" in symbols:
                buckets["h_angle_sigma_deg"].append(float(row.sigma_degree))
            else:
                buckets["heavy_angle_sigma_deg"].append(float(row.sigma_degree))
    return {
        name: {
            "count": len(values),
            "max": _max_value(values),
            "mean": float(np.mean(values)) if values else 0.0,
        }
        for name, values in buckets.items()
    }


def _reproducibility_payload(
    request: SemiexperimentalFitRequest,
    result: SemiexperimentalFitResult,
    options: SemiexperimentalFinalValidationOptions,
) -> dict[str, object]:
    input_path = Path(request.initial_geometry)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "initial_geometry": str(input_path),
        "initial_geometry_sha256": _sha256_file(input_path),
        "result_manifest": "" if result.manifest is None else str(result.manifest),
        "result_manifest_sha256": _sha256_file(result.manifest) if result.manifest else "",
        "request": {
            "fixed_parameters": request.fixed_parameters,
            "observable": request.observable,
            "rotational_components": request.rotational_components,
            "coordinate_model": request.coordinate_model,
            "robust_loss": request.robust_loss,
            "robust_scale": request.robust_scale,
            "leave_one_out": request.leave_one_out,
            "n_observations": len(request.observations),
            "observation_labels": tuple(obs.label for obs in request.observations),
            "n_qm_predicates": len(request.qm_predicates),
            "n_parameter_classes": len(request.parameter_classes),
        },
        "predicate_sha256": _sha256_text(
            json.dumps(
                [
                    {
                        "label_pattern": predicate.label_pattern,
                        "value": predicate.value,
                        "sigma": predicate.sigma,
                        "source": predicate.source,
                    }
                    for predicate in request.qm_predicates
                ],
                sort_keys=True,
            )
        ),
        "validation_options": asdict(options),
    }


def _sha256_file(path: Path | str | None) -> str:
    if path is None:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _min_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else 0.0


def _max_offdiag_abs(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    mask = ~np.eye(arr.shape[0], arr.shape[1], dtype=bool)
    values = np.abs(arr[mask])
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else 0.0


def _max_value(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _safe_label(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "group"


__all__ = [
    "FINAL_VALIDATION_SCHEMA",
    "SemiexperimentalFinalValidationOptions",
    "SemiexperimentalFinalValidationResult",
    "SemiexperimentalPredicateAuditRow",
    "SemiexperimentalFinalValidationIssue",
    "SemiexperimentalValidationRunRow",
    "audit_semiexperimental_predicates",
    "final_validation_issues_csv",
    "final_validation_runs_csv",
    "predicate_audit_csv",
    "run_semiexperimental_final_validation",
]
