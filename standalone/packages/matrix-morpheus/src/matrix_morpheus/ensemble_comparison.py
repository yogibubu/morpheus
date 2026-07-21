from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json

import numpy as np

from .ensemble import (
    EnsembleClassCorrection,
    EnsembleClassCorrectionFit,
    EnsembleMolecule,
    _ensemble_molecule_design,
    _weighted_rms,
    fit_ensemble_class_corrections,
    read_ensemble_job,
    write_ensemble_class_correction_outputs,
)


def run_ensemble_prior_comparison(
    job_path: Path,
    outdir: Path,
    *,
    soft_prior_sigma: float = 1.0e-3,
) -> dict[str, EnsembleClassCorrectionFit]:
    """Run no-prior, soft-prior and hard-constraint variants for one ensemble job."""

    job = read_ensemble_job(Path(job_path))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "no_prior": _classes_without_priors(job.classes),
        "soft_prior": _classes_with_soft_angular_priors(job.classes, soft_prior_sigma),
        "hard_constraint": _classes_with_angular_hard_constraints(job.classes),
    }
    results: dict[str, EnsembleClassCorrectionFit] = {}
    for name, classes in variants.items():
        result = fit_ensemble_class_corrections(
            job.molecules,
            classes,
            step=job.step,
            rcond=job.rcond,
        )
        results[name] = result
        write_ensemble_class_correction_outputs(out / name, result)
    (out / "ensemble_prior_comparison.csv").write_text(_comparison_csv(results), encoding="utf-8")
    (out / "ensemble_prior_comparison.json").write_text(_comparison_json(results), encoding="utf-8")
    run_ensemble_prior_scan(job_path, out / "prior_scan")
    run_ensemble_leave_one_molecule_out(
        job_path, out / "leave_one_molecule_out", soft_prior_sigma=soft_prior_sigma
    )
    return results


def run_ensemble_prior_scan(
    job_path: Path,
    outdir: Path,
    *,
    sigmas: tuple[float, ...] = (
        1.0e-5,
        3.0e-5,
        1.0e-4,
        3.0e-4,
        1.0e-3,
        3.0e-3,
        1.0e-2,
        3.0e-2,
        1.0e-1,
    ),
) -> list[dict[str, float]]:
    job = read_ensemble_job(Path(job_path))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    for sigma in sigmas:
        result = fit_ensemble_class_corrections(
            job.molecules,
            _classes_with_soft_angular_priors(job.classes, sigma),
            step=job.step,
            rcond=job.rcond,
        )
        rows.append(
            {
                "prior_sigma": float(sigma),
                "rank": float(result.rank),
                "accepted": float(result.acceptance.accepted),
                "acceptance_status": result.acceptance.status,
                "scaled_condition_number": float(result.condition_number),
                "weighted_rms_before": float(result.weighted_rms_before),
                "weighted_rms_after": float(result.weighted_rms_after),
            }
        )
    (out / "prior_sigma_scan.csv").write_text(_scan_csv(rows), encoding="utf-8")
    (out / "prior_sigma_scan.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def run_ensemble_leave_one_molecule_out(
    job_path: Path,
    outdir: Path,
    *,
    soft_prior_sigma: float = 1.0e-3,
) -> list[dict[str, float | str]]:
    job = read_ensemble_job(Path(job_path))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    classes = _classes_with_soft_angular_priors(job.classes, soft_prior_sigma)
    rows: list[dict[str, float | str]] = []
    for heldout in job.molecules:
        train = tuple(molecule for molecule in job.molecules if molecule.name != heldout.name)
        if not train:
            continue
        result = fit_ensemble_class_corrections(train, classes, step=job.step, rcond=job.rcond)
        before, after = _evaluate_fit_on_molecule(heldout, result, step=job.step)
        transferability = after / before if before > 0.0 else float("inf")
        rows.append(
            {
                "heldout": heldout.name,
                "train_molecules": float(len(train)),
                "rank": float(result.rank),
                "accepted": float(result.acceptance.accepted),
                "acceptance_status": result.acceptance.status,
                "scaled_condition_number": float(result.condition_number),
                "weighted_rms_before": before,
                "weighted_rms_predicted": after,
                "delta_weighted_rms": after - before,
                "transferability_score": transferability,
            }
        )
    (out / "leave_one_molecule_out.csv").write_text(_loo_csv(rows), encoding="utf-8")
    (out / "leave_one_molecule_out.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def run_ensemble_synthon_threshold_scan(
    job_path: Path,
    outdir: Path,
    *,
    thresholds: tuple[float, ...] = (0.015, 0.025, 0.035, 0.05, 0.075),
) -> list[dict[str, float | str]]:
    """Scan the continuous synthon atom-typing threshold for a Zeff-based job."""

    job = read_ensemble_job(Path(job_path))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        classes = tuple(
            replace(item, synthon_threshold=float(threshold))
            if item.synthon_zeff and item.synthon_threshold is not None
            else item
            for item in job.classes
        )
        try:
            result = fit_ensemble_class_corrections(
                job.molecules,
                classes,
                step=job.step,
                rcond=job.rcond,
            )
        except Exception as exc:
            rows.append(
                {
                    "synthon_threshold": float(threshold),
                    "status": "failed",
                    "rank": 0.0,
                    "accepted": 0.0,
                    "acceptance_status": "failed",
                    "scaled_condition_number": float("nan"),
                    "weighted_rms_before": float("nan"),
                    "weighted_rms_after": float("nan"),
                    "min_matched_coordinates": 0.0,
                    "error": str(exc),
                }
            )
            continue
        matched = [
            sum(block.matched_counts.get(item.name, 0) for block in result.molecule_blocks)
            for item in result.classes
        ]
        rows.append(
            {
                "synthon_threshold": float(threshold),
                "status": "ok",
                "rank": float(result.rank),
                "accepted": float(result.acceptance.accepted),
                "acceptance_status": result.acceptance.status,
                "scaled_condition_number": float(result.condition_number),
                "weighted_rms_before": float(result.weighted_rms_before),
                "weighted_rms_after": float(result.weighted_rms_after),
                "min_matched_coordinates": float(min(matched) if matched else 0),
                "error": "",
            }
        )
    (out / "synthon_threshold_scan.csv").write_text(
        _synthon_threshold_scan_csv(rows), encoding="utf-8"
    )
    (out / "synthon_threshold_scan.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def write_ensemble_jpcl_artifacts(
    job_path: Path,
    paper_dir: Path,
    *,
    outdir: Path | None = None,
    soft_prior_sigma: float = 1.0e-3,
) -> dict[str, Path]:
    """Run the anhydride ensemble workflow and refresh JPCL CSV/TeX fragments."""

    paper = Path(paper_dir)
    generated = paper / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    analysis = Path(outdir) if outdir is not None else paper / "analysis"
    run_ensemble_prior_comparison(job_path, analysis, soft_prior_sigma=soft_prior_sigma)

    artifacts: dict[str, Path] = {}
    comparison = _read_csv_dicts(analysis / "ensemble_prior_comparison.csv")
    scan = _read_csv_dicts(analysis / "prior_scan" / "prior_sigma_scan.csv")
    loo = _read_csv_dicts(analysis / "leave_one_molecule_out" / "leave_one_molecule_out.csv")
    support = _read_csv_dicts(analysis / "soft_prior" / "ensemble_class_report.csv")
    classes = _read_csv_dicts(analysis / "soft_prior" / "ensemble_class_corrections.csv")

    artifacts["model_scheme"] = _write(generated / "model_scheme.tex", _model_scheme_tex())
    artifacts["prior_comparison"] = _write(
        generated / "anhydrides_prior_comparison.tex", _comparison_table_tex(comparison)
    )
    artifacts["classes_priors"] = _write(
        generated / "anhydrides_classes_priors.tex", _classes_table_tex(classes)
    )
    artifacts["class_support"] = _write(
        generated / "class_support_report.tex", _support_table_tex(support)
    )
    artifacts["condition_wrms_plot"] = _write(
        generated / "condition_wrms_plot.tex", _condition_wrms_plot_tex(comparison)
    )
    artifacts["prior_sigma_scan_plot"] = _write(
        generated / "prior_sigma_scan_plot.tex", _prior_scan_plot_tex(scan, "weighted_rms_after")
    )
    artifacts["prior_sigma_condition_plot"] = _write(
        generated / "prior_sigma_condition_plot.tex",
        _prior_scan_plot_tex(scan, "scaled_condition_number", log_y=True),
    )
    artifacts["leave_one_molecule_out"] = _write(
        generated / "leave_one_molecule_out.tex", _loo_table_tex(loo)
    )
    return artifacts


def suggest_ensemble_class_priors(
    classes: tuple[EnsembleClassCorrection, ...],
    *,
    bend_sigma: float = 1.0e-3,
    torsion_sigma: float = 2.0e-3,
    oop_sigma: float = 1.0e-3,
) -> tuple[EnsembleClassCorrection, ...]:
    result: list[EnsembleClassCorrection] = []
    for item in classes:
        if item.prior_value is not None or item.prior_sigma is not None:
            result.append(item)
            continue
        kind = item.kind.strip().lower().replace("-", "_")
        sigma = None
        if kind in {"bend", "angle", "a"}:
            sigma = bend_sigma
        elif kind in {"torsion", "dihedral", "d"}:
            sigma = torsion_sigma
        elif kind in {"out_of_plane", "oop", "u"}:
            sigma = oop_sigma
        result.append(
            item if sigma is None else replace(item, prior_value=0.0, prior_sigma=float(sigma))
        )
    return tuple(result)


def _classes_without_priors(
    classes: tuple[EnsembleClassCorrection, ...],
) -> tuple[EnsembleClassCorrection, ...]:
    return tuple(replace(item, prior_value=None, prior_sigma=None) for item in classes)


def _classes_with_soft_angular_priors(
    classes: tuple[EnsembleClassCorrection, ...],
    sigma: float,
) -> tuple[EnsembleClassCorrection, ...]:
    return suggest_ensemble_class_priors(
        tuple(replace(item, prior_value=None, prior_sigma=None) for item in classes),
        bend_sigma=sigma,
        torsion_sigma=2.0 * sigma,
        oop_sigma=sigma,
    )


def _classes_with_angular_hard_constraints(
    classes: tuple[EnsembleClassCorrection, ...],
) -> tuple[EnsembleClassCorrection, ...]:
    return tuple(
        replace(item, prior_value=None, prior_sigma=None)
        for item in classes
        if not _is_angular_class(item)
    )


def _is_angular_class(item: EnsembleClassCorrection) -> bool:
    return item.kind.strip().lower().replace("-", "_") in {
        "bend",
        "angle",
        "a",
        "torsion",
        "dihedral",
        "d",
        "out_of_plane",
        "oop",
        "u",
    }


def _comparison_csv(results: dict[str, EnsembleClassCorrectionFit]) -> str:
    lines = [
        "variant,n_classes,rank,acceptance_status,accepted,scaled_condition_number,weighted_rms_before,weighted_rms_after"
    ]
    for name, result in results.items():
        lines.append(
            f"{name},{len(result.classes)},{result.rank},{result.acceptance.status},{int(result.acceptance.accepted)},"
            f"{result.condition_number:.12g},"
            f"{result.weighted_rms_before:.12g},{result.weighted_rms_after:.12g}"
        )
    return "\n".join(lines) + "\n"


def _comparison_json(results: dict[str, EnsembleClassCorrectionFit]) -> str:
    payload = {
        name: {
            "n_classes": len(result.classes),
            "rank": result.rank,
            "acceptance_status": result.acceptance.status,
            "accepted": result.acceptance.accepted,
            "scaled_condition_number": result.condition_number,
            "weighted_rms_before": result.weighted_rms_before,
            "weighted_rms_after": result.weighted_rms_after,
            "classes": {
                item.name: {
                    "correction": result.corrections[item.name],
                    "sigma": result.sigma[item.name],
                    "prior_value": item.prior_value,
                    "prior_sigma": item.prior_sigma,
                    "prior_residual": result.prior_residual_after.get(item.name),
                }
                for item in result.classes
            },
        }
        for name, result in results.items()
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _evaluate_fit_on_molecule(
    molecule: EnsembleMolecule,
    result: EnsembleClassCorrectionFit,
    *,
    step: float,
) -> tuple[float, float]:
    solution = np.asarray([result.corrections[item.name] for item in result.classes], dtype=float)
    with TemporaryDirectory(prefix="matrix_ensemble_loo_") as tmp:
        block = _ensemble_molecule_design(molecule, result.classes, step=step, root=Path(tmp))
    after = block.residual - block.design @ solution
    return _weighted_rms(block.residual, block.measurement.weights), _weighted_rms(
        after, block.measurement.weights
    )


def _scan_csv(rows: list[dict[str, float]]) -> str:
    lines = [
        "prior_sigma,rank,acceptance_status,accepted,scaled_condition_number,weighted_rms_before,weighted_rms_after"
    ]
    for row in rows:
        lines.append(
            f"{row['prior_sigma']:.12g},{int(row['rank'])},{row.get('acceptance_status', '')},"
            f"{int(row.get('accepted', 0.0))},{row['scaled_condition_number']:.12g},"
            f"{row['weighted_rms_before']:.12g},{row['weighted_rms_after']:.12g}"
        )
    return "\n".join(lines) + "\n"


def _loo_csv(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "heldout,train_molecules,rank,scaled_condition_number,weighted_rms_before,weighted_rms_predicted,delta_weighted_rms,transferability_score"
    ]
    for row in rows:
        lines.append(
            f"{row['heldout']},{int(row['train_molecules'])},{int(row['rank'])},{float(row['scaled_condition_number']):.12g},"
            f"{float(row['weighted_rms_before']):.12g},{float(row['weighted_rms_predicted']):.12g},"
            f"{float(row['delta_weighted_rms']):.12g},{float(row['transferability_score']):.12g}"
        )
    return "\n".join(lines) + "\n"


def _synthon_threshold_scan_csv(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "synthon_threshold,status,rank,acceptance_status,accepted,scaled_condition_number,"
        "weighted_rms_before,weighted_rms_after,min_matched_coordinates,error"
    ]
    for row in rows:
        lines.append(
            f"{float(row['synthon_threshold']):.12g},{row['status']},{int(float(row['rank']))},"
            f"{row.get('acceptance_status', '')},{int(float(row.get('accepted', 0.0)))},"
            f"{float(row['scaled_condition_number']):.12g},{float(row['weighted_rms_before']):.12g},"
            f"{float(row['weighted_rms_after']):.12g},{int(float(row['min_matched_coordinates']))},"
            f"{json.dumps(str(row['error']))}"
        )
    return "\n".join(lines) + "\n"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _tex(text: object) -> str:
    return str(text).replace("_", r"\_")


def _format_float(value: object, digits: int = 5) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "--"
    if number != 0.0 and (abs(number) >= 1.0e4 or abs(number) <= 1.0e-3):
        exponent = int(np.floor(np.log10(abs(number))))
        mantissa = number / (10.0**exponent)
        return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _format_prior(row: dict[str, str]) -> str:
    value = row.get("prior_value", "")
    sigma = row.get("prior_sigma", "")
    if not value or not sigma:
        return "none"
    return rf"${float(value):.1f} \pm {_format_math_number(float(sigma))}$"


def _format_math_number(number: float) -> str:
    if number != 0.0 and (abs(number) >= 1.0e4 or abs(number) <= 1.0e-3):
        exponent = int(np.floor(np.log10(abs(number))))
        mantissa = number / (10.0**exponent)
        return rf"{mantissa:.2f}\times 10^{{{exponent}}}"
    return f"{number:.3g}"


def _selector(row: dict[str, str]) -> str:
    atoms = row.get("atom_symbols", "").replace("|", "--")
    value_min = row.get("value_min", "")
    value_max = row.get("value_max", "")
    extra = []
    if value_min:
        extra.append(rf"$r\geq{_format_float(value_min, 2)}$")
    if value_max:
        extra.append(rf"$r\leq{_format_float(value_max, 2)}$")
    return _tex(atoms) + (", " + ", ".join(extra) if extra else "")


def _classes_table_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Class & Coordinate kind & Selector & Prior \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"{_tex(row['class'])} & {_tex(row.get('kind', ''))} & {_selector(row)} & {_format_prior(row)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _support_table_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Class & Kind & Matched coordinates & Molecules & $|\mathrm{prior\ residual}|/\sigma_{\mathrm{prior}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        ratio = row.get("posterior_over_prior_sigma", "")
        lines.append(
            rf"{_tex(row['class'])} & {_tex(row.get('kind', ''))} & {int(float(row['matched_coordinates']))} & "
            rf"{int(float(row['molecule_count']))} & {('--' if not ratio else _format_float(ratio, 3))} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _comparison_table_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Variant & Status & Classes & Rank & Cond. & WRMS$_0$ & WRMS \\",
        r"\midrule",
    ]
    for row in rows:
        variant = row["variant"].replace("_", " ").title()
        lines.append(
            rf"{variant} & {_tex(row.get('acceptance_status', ''))} & "
            rf"{int(row['n_classes'])} & {int(row['rank'])} & {_format_float(row['scaled_condition_number'])} & "
            rf"{_format_float(row['weighted_rms_before'])} & {_format_float(row['weighted_rms_after'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _loo_table_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Held-out & Train & Cond. & WRMS$_0$ & WRMS$_\mathrm{pred}$ & Score \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["heldout"]).replace("_", " ").title()
        lines.append(
            rf"{_tex(label)} & "
            rf"{int(row['train_molecules'])} & {_format_float(row['scaled_condition_number'], 2)} & "
            rf"{_format_float(row['weighted_rms_before'])} & {_format_float(row['weighted_rms_predicted'])} & "
            rf"{_format_float(row['transferability_score'], 2)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _condition_wrms_plot_tex(rows: list[dict[str, str]]) -> str:
    marks = {"no_prior": "*", "soft_prior": "square*", "hard_constraint": "triangle*"}
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.82\linewidth,",
        r"  height=5.2cm,",
        r"  xmode=log,",
        r"  xlabel={Scaled condition number},",
        r"  ylabel={Weighted RMS after fit},",
        r"  grid=both,",
        r"  legend style={draw=none,fill=none,at={(0.03,0.03)},anchor=south west},",
        r"]",
    ]
    for row in rows:
        variant = row["variant"]
        label = variant.replace("_", " ").title()
        lines.append(
            rf"\addplot+[only marks,mark={marks.get(variant, '*')},mark size=2.6pt] coordinates "
            rf"{{({float(row['scaled_condition_number']):.12g},{float(row['weighted_rms_after']):.12g})}};"
        )
        lines.append(rf"\addlegendentry{{{label}}}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", ""])
    return "\n".join(lines)


def _prior_scan_plot_tex(rows: list[dict[str, str]], field: str, *, log_y: bool = False) -> str:
    ylabel = (
        "Scaled condition number"
        if field == "scaled_condition_number"
        else "Weighted RMS after fit"
    )
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.82\linewidth,",
        r"  height=5.2cm,",
        r"  xmode=log,",
        *([r"  ymode=log,"] if log_y else []),
        r"  xlabel={Bend prior $\sigma$},",
        rf"  ylabel={{{ylabel}}},",
        r"  grid=both,",
        r"]",
        r"\addplot+[mark=*,mark size=2pt] coordinates {",
    ]
    for row in rows:
        lines.append(rf"  ({float(row['prior_sigma']):.12g},{float(row[field]):.12g})")
    lines.extend([r"};", r"\end{axis}", r"\end{tikzpicture}", ""])
    return "\n".join(lines)


def _model_scheme_tex() -> str:
    return r"""\begin{tikzpicture}[x=1cm,y=1cm,>=Latex]
\tikzstyle{box}=[draw,rounded corners=3pt,align=center,minimum height=0.75cm]
\node[box,fill=blue!8,minimum width=2.6cm] (m1) at (0,2.4) {Molecule 1\\topology, symmetry};
\node[box,fill=blue!8,minimum width=2.6cm] (m2) at (0,1.2) {Molecule 2\\topology, symmetry};
\node[box,fill=blue!8,minimum width=2.6cm] (m3) at (0,0.0) {Molecule $n$\\topology, symmetry};
\node[box,fill=green!10,minimum width=2.9cm] (coords) at (3.4,1.2) {non-redundant\\GIC spaces};
\node[box,fill=yellow!16,minimum width=2.9cm] (classes) at (6.8,1.2) {shared chemical\\class map $T_{m,kc}$};
\node[box,fill=orange!15,minimum width=2.6cm] (fit) at (10.0,1.2) {global fit\\$\Delta_c$};
\draw[->] (m1.east) -- (coords.west);
\draw[->] (m2.east) -- (coords.west);
\draw[->] (m3.east) -- (coords.west);
\draw[->] (coords.east) -- (classes.west);
\draw[->] (classes.east) -- (fit.west);
\node[align=center,font=\small] at (6.8,0.1) {soft priors enter as\\Gaussian observations};
\node[align=center,font=\small] at (10.0,0.1) {rank, covariance,\\condition number};
\end{tikzpicture}
"""


__all__ = [
    "run_ensemble_prior_comparison",
    "run_ensemble_prior_scan",
    "run_ensemble_leave_one_molecule_out",
    "run_ensemble_synthon_threshold_scan",
    "suggest_ensemble_class_priors",
    "write_ensemble_jpcl_artifacts",
]
