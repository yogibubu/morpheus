"""Lightweight MORPHEUS argument parsers shared by both command suites."""

from __future__ import annotations

from pathlib import Path

def add_commands(sub, *, root: Path) -> None:
    semiexp = sub.add_parser(
        "semiexp",
        help="Fit semiexperimental equilibrium geometry with MORPHEUS",
    )
    semiexp.add_argument(
        "--job",
        type=Path,
        help="MATRIX/Merlino semiexperimental job file or legacy MSR file",
    )
    semiexp.add_argument(
        "--xyz",
        "--geometry",
        dest="xyz",
        type=Path,
        help="Initial parent Cartesian geometry in XYZ or Gaussian .com/.gjf format",
    )
    semiexp.add_argument(
        "--observations",
        type=Path,
        help="CSV/JSON/TOML with isotopologue B0 constants and corrections, or legacy MSR file",
    )
    semiexp.add_argument(
        "--xyzin",
        type=Path,
        help="Canonical MATRIX xyzin container to create/update before SEfit",
    )
    semiexp.add_argument("--no-write-section", action="store_true", help="Do not update #MORPHEUS")
    semiexp.add_argument(
        "--r0-preflight",
        action="store_true",
        help="Fit raw B0 constants without DeltaBvib and report identifiability diagnostics",
    )
    semiexp.add_argument(
        "--include-r0-report",
        action="store_true",
        help=(
            "Run or retain the diagnostic r0 fit and include input/r0/rs/reSE in the "
            "final PIC report"
        ),
    )
    semiexp.add_argument("--outdir", type=Path, required=True)
    semiexp.add_argument("--backend", choices=("python", "fortran77"), default="python")
    semiexp.add_argument(
        "--fixed",
        default="",
        help="Comma/semicolon-separated fixed GIC patterns or Gaussian-style constraints",
    )
    semiexp.add_argument("--fix-hydrogens", action="store_true")
    semiexp.add_argument(
        "--no-auto-stabilize",
        action="store_true",
        help=(
            "Disable MORPHEUS automatic stabilization. By default, an "
            "underdetermined free GIC fit is stabilized by blocking X-H "
            "coordinates before the fit is attempted."
        ),
    )
    semiexp.add_argument("--max-iter", type=int, default=None)
    semiexp.add_argument("--step", type=float, default=1.0e-4)
    semiexp.add_argument("--damping", type=float, default=1.0e-8)
    semiexp.add_argument("--max-step", type=float, default=0.25)
    semiexp.add_argument(
        "--max-atom-displacement",
        type=float,
        default=None,
        help=(
            "Reject a fitted geometry when the largest aligned atom displacement "
            "from the starting structure exceeds this value in Angstrom."
        ),
    )
    semiexp.add_argument(
        "--keep-all-artifacts",
        action="store_true",
        help="Keep intermediate diagnostics even after the reliability checks pass.",
    )
    semiexp.add_argument("--prune-condition", type=float, default=0.0)
    semiexp.add_argument(
        "--robust-loss",
        choices=("none", "huber", "soft_l1", "cauchy"),
        default="none",
    )
    semiexp.add_argument("--robust-scale", type=float, default=0.0)
    semiexp.add_argument("--leave-one-out", action="store_true")
    semiexp.add_argument(
        "--final-validation",
        action="store_true",
        help=(
            "Run post-fit robustness, precision and reproducibility checks and "
            "write semiexp_final_validation artifacts."
        ),
    )
    semiexp.add_argument("--validation-no-coordinate-check", action="store_true")
    semiexp.add_argument("--validation-no-huber-check", action="store_true")
    semiexp.add_argument("--validation-no-predicate-scan", action="store_true")
    semiexp.add_argument("--validation-no-leave-predicate-groups", action="store_true")
    semiexp.add_argument(
        "--validation-sigma-scale",
        type=float,
        action="append",
        default=[],
        help="Predicate sigma scale for final validation scans; repeatable.",
    )
    semiexp.add_argument("--validation-max-predicate-groups", type=int, default=12)
    semiexp.add_argument("--validation-multistart", type=int, default=0)
    semiexp.add_argument("--validation-multistart-sigma", type=float, default=0.001)
    semiexp.add_argument("--validation-random-seed", type=int, default=20260703)
    semiexp.add_argument("--checkpoint", type=Path, default=None)
    semiexp.add_argument("--restart", type=Path, default=None)
    semiexp.add_argument(
        "--observable",
        choices=("moments", "rotational_constants", "auto"),
        default="moments",
    )
    semiexp.add_argument(
        "--coordinate-model",
        choices=("gic", "cartesian_symmetry"),
        default="gic",
    )
    semiexp.add_argument(
        "--rotational-components",
        choices=("auto", "ABC", "AB", "AC", "BC"),
        default="auto",
    )
    semiexp.add_argument(
        "--qm-predicate",
        action="append",
        default=[],
        help="QM prior as label_pattern:value:sigma[:source]; can be repeated",
    )
    semiexp.add_argument(
        "--kraitchman-predicates",
        action="store_true",
        help="Add distance/angle predicates derived from single-substitution Kraitchman coordinates",
    )
    semiexp.add_argument(
        "--kraitchman-distance-sigma",
        type=float,
        default=0.01,
        help="Distance sigma in Angstrom for Kraitchman-derived predicates",
    )
    semiexp.add_argument(
        "--kraitchman-angle-sigma",
        type=float,
        default=1.0,
        help="Angle sigma in degrees for Kraitchman-derived predicates",
    )
    semiexp.add_argument(
        "--kraitchman-partial-predicates",
        action="store_true",
        help=(
            "Also create Kraitchman predicates for primitives containing only some "
            "Kraitchman-seeded atoms; conservative default requires all atoms seeded"
        ),
    )
    semiexp.add_argument(
        "--sensitivity-advisor",
        action="store_true",
        help=(
            "Rank symmetry-adapted non-redundant GICs by weighted effect on "
            "rotational constants and write tuning suggestions for the current "
            "chemical model."
        ),
    )
    semiexp.add_argument(
        "--apply-sensitivity-advisor",
        action="store_true",
        help=(
            "Apply sensitivity-advisor predicates/fixed patterns to the fit. "
            "Without this flag the advisor is diagnostic only. The chemical "
            "model must already be valid; the advisor is only a conservative "
            "tuning layer."
        ),
    )
    semiexp.add_argument(
        "--force-sensitivity-advisor",
        action="store_true",
        help="Apply sensitivity-advisor suggestions without the safety gate.",
    )
    semiexp.add_argument("--sensitivity-gate-rot-rel-tol", type=float, default=0.02)
    semiexp.add_argument("--sensitivity-gate-rot-abs-tol", type=float, default=1.0e-3)
    semiexp.add_argument("--sensitivity-gate-condition-factor", type=float, default=10.0)
    semiexp.add_argument("--sensitivity-gate-max-bond-delta", type=float, default=0.01)
    semiexp.add_argument("--sensitivity-gate-max-angle-delta", type=float, default=1.0)
    semiexp.add_argument("--sensitivity-fit-threshold", type=float, default=0.15)
    semiexp.add_argument("--sensitivity-fixed-threshold", type=float, default=1.0e-6)
    semiexp.add_argument(
        "--sensitivity-min-fit",
        default="auto",
        help=(
            "Minimum number of sensitivity-ranked GICs to keep free: auto, none, "
            "or an integer. Auto keeps enough coordinates when many isotopologues "
            "are available."
        ),
    )
    semiexp.add_argument("--sensitivity-distance-sigma", type=float, default=0.003)
    semiexp.add_argument("--sensitivity-angle-sigma", type=float, default=0.3)
    semiexp.add_argument("--sensitivity-torsion-sigma", type=float, default=0.5)
    semiexp.add_argument(
        "--sensitivity-soft-predicate-scale",
        type=float,
        default=1.0,
        help="Scale predicates for non-selected soft/intermolecular GICs.",
    )
    semiexp.add_argument(
        "--sensitivity-null-predicate-scale",
        type=float,
        default=1.0,
        help="Additional scale for nearly null non-selected GIC predicates.",
    )
    semiexp.add_argument(
        "--sensitivity-fit-regularization-scale",
        type=float,
        default=0.0,
        help=(
            "Weak predicate scale for selected soft/intermolecular GICs; "
            "use 0 to leave them fully unregularized."
        ),
    )
    semiexp.add_argument(
        "--exclude-rotational-constant",
        action="append",
        default=[],
        metavar="LABEL:COMPONENT",
        help=(
            "Explicitly exclude one measured A, B or C rotational constant. "
            "Repeatable; exclusions are recorded in the fit-comparison contract."
        ),
    )
    semiexp.add_argument(
        "--compare-free-fit",
        action="store_true",
        help=(
            "Also run the otherwise identical fit without regularization of the "
            "sensitivity-selected soft SONIC coordinates and report both results."
        ),
    )
    semiexp.add_argument(
        "--parameter-class",
        action="append",
        default=[],
        help="Class constraint as name:shared|fixed:pattern[|pattern...]; can be repeated",
    )
    semiexp.add_argument(
        "--primitive-class",
        action="append",
        default=[],
        help=(
            "Primitive-defined class as name:primitive[|primitive...]. MORPHEUS maps "
            "the primitives onto disjoint GIC classes using coefficient thresholds."
        ),
    )
    semiexp.add_argument(
        "--primitive-class-min",
        type=float,
        default=0.70,
        help="Minimum GIC coefficient fraction required to assign a primitive class",
    )
    semiexp.add_argument(
        "--primitive-class-cross-max",
        type=float,
        default=0.20,
        help="Maximum competing class fraction allowed for an unambiguous assignment",
    )
    semiexp.add_argument(
        "--primitive-class-budget",
        default="auto",
        help="Maximum number of primitive-derived classes: auto, all, or an integer",
    )
    semiexp_ensemble = sub.add_parser(
        "semiexp-ensemble",
        help="Fit shared class corrections across multiple semiexperimental molecule jobs",
    )
    semiexp_ensemble.add_argument("--job", type=Path, required=True)
    semiexp_ensemble.add_argument("--outdir", type=Path, required=True)
    semiexp_ensemble_paper = sub.add_parser(
        "semiexp-ensemble-paper",
        help="Run ensemble paper comparisons and write JPCL-ready artifacts",
    )
    semiexp_ensemble_paper.add_argument("--job", type=Path, required=True)
    semiexp_ensemble_paper.add_argument("--paper-dir", type=Path, required=True)
    semiexp_ensemble_paper.add_argument("--outdir", type=Path)
    semiexp_ensemble_paper.add_argument("--soft-prior-sigma", type=float, default=1.0e-3)
    semiexp_ensemble_prior_scan = sub.add_parser(
        "semiexp-ensemble-prior-scan",
        help="Scan ensemble soft-prior sigma values",
    )
    semiexp_ensemble_prior_scan.add_argument("--job", type=Path, required=True)
    semiexp_ensemble_prior_scan.add_argument("--outdir", type=Path, required=True)
    semiexp_ensemble_prior_scan.add_argument("--sigma", type=float, action="append", default=[])
    semiexp_ensemble_synthon_scan = sub.add_parser(
        "semiexp-ensemble-synthon-scan",
        help="Scan Zeff synthon thresholds for an ensemble job",
    )
    semiexp_ensemble_synthon_scan.add_argument("--job", type=Path, required=True)
    semiexp_ensemble_synthon_scan.add_argument("--outdir", type=Path, required=True)
    semiexp_ensemble_synthon_scan.add_argument(
        "--threshold", type=float, action="append", default=[]
    )
    semiexp_benchmark = sub.add_parser(
        "semiexp-benchmark",
        help="Generate MORPHEUS benchmark and paper tables from a regression snapshot",
    )
    semiexp_benchmark.add_argument("--snapshot", type=Path)
    semiexp_benchmark.add_argument("--outdir", type=Path)
    semiexp_benchmark.add_argument("--no-refresh", action="store_true")
    semiexp_benchmark.add_argument("--update-snapshot", action="store_true")
