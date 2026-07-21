from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TypeVar


_T = TypeVar("_T")
MATRIX_PROGRESS_PREFIX = "MATRIX_PROGRESS "


def _topology_worker_argument(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 'auto' or a non-negative integer") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("worker count cannot be negative")
    return count


def _print_matrix_progress(current: int, total: int, detail: str) -> None:
    print(
        MATRIX_PROGRESS_PREFIX
        + json.dumps(
            {"current": int(current), "total": int(total), "detail": str(detail)},
            sort_keys=True,
        ),
        flush=True,
    )


def find_repo_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("MATRIX_HOME") or os.environ.get("ORACLE_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()

    search_from = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (search_from, *search_from.parents):
        if (candidate / "packages").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def add_repo_packages_to_path(repo_root: Path | None = None) -> None:
    root = find_repo_root() if repo_root is None else Path(repo_root)
    packages = root / "packages"
    if not packages.is_dir():
        return
    for src in sorted(packages.glob("*/src")):
        text = str(src)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_parser(
    *,
    repo_root: Path | None = None,
    prog: str = "matrix",
) -> argparse.ArgumentParser:
    root = find_repo_root() if repo_root is None else Path(repo_root)
    parser = argparse.ArgumentParser(prog=prog, description="MATRIX workflow CLI")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="Create a MATRIX project workspace")
    init.add_argument("workdir", type=Path)

    provenance = sub.add_parser(
        "provenance",
        help="Verify a The ONE project provenance hash chain",
    )
    provenance.add_argument("project", type=Path)
    provenance.add_argument("--json", action="store_true")

    oracle = sub.add_parser(
        "oracle",
        add_help=False,
        help="Run ORACLE topology, symmetry and continuous perception",
    )
    oracle.add_argument("oracle_args", nargs=argparse.REMAINDER)

    apoc = sub.add_parser(
        "apoc",
        add_help=False,
        help="Run APOC electronic population analysis",
    )
    apoc.add_argument("apoc_args", nargs=argparse.REMAINDER)

    gui = sub.add_parser(
        "gui",
        help="Launch the MATRIX desktop or one tool window",
        description="Launch the MATRIX desktop or one tool window",
    )
    gui.add_argument("tool", nargs="?", help="ORACLE, SMITH, LINK, MORPHEUS or TRINITY")
    gui.add_argument("xyzin", nargs="?", type=Path, help="Existing MATRIX enriched XYZ project")
    gui.add_argument("--workdir", type=Path, help="Workspace for runs, logs and reports")
    gui.add_argument("--the-one", action="store_true", help="Force the guided The ONE entry point")
    gui.add_argument("--smoke-test", action="store_true", help="Create the shell and exit")

    validate = sub.add_parser("validate", help="Validate an enriched XYZ after preprocessing")
    validate.add_argument("xyzin", type=Path)
    validate.add_argument("--require-fragments", action="store_true")

    topology = sub.add_parser("topology", help="Inspect frozen MATRIX topology sections")
    topology_sub = topology.add_subparsers(dest="topology_command")
    topology_report = topology_sub.add_parser("report", help="Write a readable topology report")
    topology_report.add_argument("xyzin", type=Path)
    topology_report.add_argument("output", type=Path, nargs="?")
    topology_snapshot = topology_sub.add_parser(
        "snapshot",
        help="Write a compact topology golden snapshot",
    )
    topology_snapshot.add_argument("xyzin", type=Path)
    topology_snapshot.add_argument("output", type=Path)

    contracts = sub.add_parser("contracts", help="List standalone xyzin tool contracts")
    contracts.add_argument("--tool", help="Show one tool contract by key or planned name")
    contracts.add_argument("--framework", action="store_true", help="Show the planned MATRIX name")
    contracts.add_argument(
        "--check-xyzin",
        type=Path,
        help="Check whether an xyzin contains the required sections for selected contracts",
    )
    contracts.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format",
    )
    contracts.add_argument("--no-gui", action="store_true", help="Omit the GUI orchestrator")

    help_cmd = sub.add_parser(
        "help",
        aliases=("manuals",),
        help="Show online help and manual links for MATRIX tools",
    )
    help_cmd.add_argument("tool", nargs="?", help="Tool key, planned name or compatibility alias")
    help_cmd.add_argument("--xyzin", type=Path, help="Show readiness against a MATRIX xyzin file")
    help_cmd.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format",
    )
    help_cmd.add_argument("--no-gui", action="store_true", help="Omit the GUI orchestrator")

    properties = sub.add_parser("properties", help="Inspect normalized QM properties")
    properties_sub = properties.add_subparsers(dest="properties_command")
    properties_summary = properties_sub.add_parser("summary", help="Summarize #PROPERTIES")
    properties_summary.add_argument("xyzin", type=Path)
    properties_summary.add_argument("--name", help="Filter by normalized property name")
    properties_summary.add_argument("--atom", type=int, help="Filter by one-based atom index")
    properties_summary.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    properties_compare = properties_sub.add_parser(
        "compare",
        help="Compare one normalized property across xyzin files",
    )
    properties_compare.add_argument("reference", type=Path)
    properties_compare.add_argument("candidates", nargs="+", type=Path)
    properties_compare.add_argument("--name", required=True, help="Normalized property name")
    properties_compare.add_argument("--atom", type=int, help="Filter by one-based atom index")
    properties_compare.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index after filtering when multiple records match",
    )
    properties_compare.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance in the property unit",
    )
    properties_compare.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance")
    properties_compare.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )

    qm = sub.add_parser("qm", help="Remote QM job orchestration")
    qm_sub = qm.add_subparsers(dest="qm_command")
    remote_submit = qm_sub.add_parser("remote-submit", help="Submit a QM input on a remote host")
    remote_submit.add_argument("input", type=Path)
    remote_submit.add_argument(
        "--engine",
        choices=("gdv32", "g16", "molpro", "orca", "mrcc", "cfour", "xtb", "pyscf"),
        required=True,
    )
    remote_submit.add_argument("--host", default=os.environ.get("MATRIX_REMOTE_HOST", "oracle"))
    remote_submit.add_argument(
        "--remote-root", default=os.environ.get("MATRIX_REMOTE_ROOT", "~/matrix")
    )
    remote_submit.add_argument("--extra-arg", action="append", default=[])
    remote_submit.add_argument("--ssh", default="ssh")
    remote_submit.add_argument("--scp", default="scp")
    remote_monitor = qm_sub.add_parser(
        "remote-run-monitor",
        help="Submit a background QM job, poll it, and fetch its native output",
    )
    remote_monitor.add_argument("input", type=Path)
    remote_monitor.add_argument(
        "--engine",
        choices=("gdv32", "g16", "molpro", "orca", "mrcc", "cfour", "xtb", "pyscf"),
        required=True,
    )
    remote_monitor.add_argument("--host", default=os.environ.get("MATRIX_REMOTE_HOST", "oracle"))
    remote_monitor.add_argument(
        "--remote-root", default=os.environ.get("MATRIX_REMOTE_ROOT", "~/matrix")
    )
    remote_monitor.add_argument("--dest", type=Path, default=Path("remote_qm_runs"))
    remote_monitor.add_argument(
        "--promote",
        choices=(
            "none",
            "auto",
            "molpro",
            "gaussian-log-hessian",
            "gaussian-rovib",
            "gaussian-electronic",
            "gaussian-fchk",
            "orca",
            "xtb-hessian",
            "pyscf-hessian",
        ),
        default="none",
    )
    remote_monitor.add_argument("--xyzin", type=Path)
    remote_monitor.add_argument("--poll-seconds", type=float, default=60.0)
    remote_monitor.add_argument(
        "--max-wait-seconds",
        type=float,
        default=0.0,
        help="Stop monitoring after this many seconds; zero waits indefinitely",
    )
    remote_monitor.add_argument("--extra-arg", action="append", default=[])
    remote_monitor.add_argument("--ssh", default="ssh")
    remote_monitor.add_argument("--scp", default="scp")
    remote_status = qm_sub.add_parser("remote-status", help="List remote MATRIX QM jobs")
    remote_status.add_argument("--host", default=os.environ.get("MATRIX_REMOTE_HOST", "oracle"))
    remote_status.add_argument(
        "--remote-root", default=os.environ.get("MATRIX_REMOTE_ROOT", "~/matrix")
    )
    remote_status.add_argument("--ssh", default="ssh")
    remote_fetch = qm_sub.add_parser(
        "remote-fetch",
        help="Fetch a remote MATRIX QM job output and optionally promote it",
    )
    remote_fetch.add_argument("job")
    remote_fetch.add_argument("--host", default=os.environ.get("MATRIX_REMOTE_HOST", "oracle"))
    remote_fetch.add_argument(
        "--remote-root", default=os.environ.get("MATRIX_REMOTE_ROOT", "~/matrix")
    )
    remote_fetch.add_argument("--dest", type=Path, default=Path("remote_qm_runs"))
    remote_fetch.add_argument(
        "--promote",
        choices=(
            "none",
            "auto",
            "molpro",
            "gaussian-log-hessian",
            "gaussian-rovib",
            "gaussian-electronic",
            "gaussian-fchk",
            "orca",
            "xtb-hessian",
            "pyscf-hessian",
        ),
        default="none",
    )
    remote_fetch.add_argument("--xyzin", type=Path)
    remote_fetch.add_argument("--ssh", default="ssh")
    remote_fetch.add_argument("--scp", default="scp")

    link = sub.add_parser(
        "link",
        help="LINK coordinate realization and external-driver workflows",
    )
    link_sub = link.add_subparsers(dest="link_command")
    link_preprocess = link_sub.add_parser("preprocess", help="Import a source into xyzin")
    _add_preprocess_arguments(link_preprocess)
    link_driver_run = link_sub.add_parser(
        "driver-run",
        help="Run LINK with an external program selecting the next SONIC point",
    )
    _add_external_driver_arguments(link_driver_run)
    link_mock_sentinel = link_sub.add_parser(
        "mock-sentinel",
        help="Run the deterministic non-genetic SENTINEL protocol mock",
    )
    _add_mock_sentinel_arguments(link_mock_sentinel)

    babel = sub.add_parser("babel", help="Compatibility alias for ORACLE preprocessing")
    babel_sub = babel.add_subparsers(dest="babel_command")
    preprocess = babel_sub.add_parser("preprocess", help="Import a source into xyzin")
    _add_preprocess_arguments(preprocess)

    gaussian = sub.add_parser("gaussian", help="Gaussian adapter and job utilities")
    gaussian_sub = gaussian.add_subparsers(dest="gaussian_command")
    gaussian_summary = gaussian_sub.add_parser("summary", help="Summarize a Gaussian log/out")
    gaussian_summary.add_argument("log", type=Path)
    gaussian_status = gaussian_sub.add_parser("status", help="Inspect Gaussian job state")
    gaussian_status.add_argument("workdir", type=Path)
    gaussian_run = gaussian_sub.add_parser("run", help="Run Gaussian from a work directory")
    gaussian_run.add_argument("workdir", type=Path)
    gaussian_run.add_argument("--executable")
    gaussian_run.add_argument("--input", type=Path)
    gaussian_run.add_argument("--background", action="store_true")
    gaussian_run.add_argument("--timeout", type=float)
    gaussian_formchk = gaussian_sub.add_parser("formchk", help="Run formchk on a checkpoint file")
    gaussian_formchk.add_argument("chk", type=Path)
    gaussian_formchk.add_argument("fchk", type=Path, nargs="?")
    gaussian_formchk.add_argument("--executable", default=None)
    gaussian_formchk.add_argument("--timeout", type=float)
    gaussian_fchk = gaussian_sub.add_parser("fchk-summary", help="Summarize FCHK/QFF blocks")
    gaussian_fchk.add_argument("fchk", type=Path)
    gaussian_semidiagonal_input = gaussian_sub.add_parser(
        "semidiagonal-input",
        help="Write a Gaussian two-link vibrot/numerical-gradient input for F3ijj",
    )
    gaussian_semidiagonal_input.add_argument("output", type=Path)
    gaussian_semidiagonal_input.add_argument("--oldchk", required=True)
    gaussian_semidiagonal_input.add_argument("--route", default="#p B3LYP/6-31G(d)")
    gaussian_semidiagonal_input.add_argument("--harmonic-chk")
    gaussian_semidiagonal_input.add_argument("--cubic-chk")
    gaussian_semidiagonal_input.add_argument("--symmetry", default="sym=com")
    gaussian_semidiagonal_input.add_argument("--xyzin", type=Path)
    gaussian_semidiagonal_input.add_argument("--manifest", type=Path)
    gaussian_semidiagonal_input.add_argument("--no-manifest", action="store_true")
    gaussian_semidiagonal_summary = gaussian_sub.add_parser(
        "semidiagonal-summary",
        help="Summarize Gaussian semidiagonal cubic rovibrational output",
    )
    gaussian_semidiagonal_summary.add_argument("log", type=Path)
    gaussian_semidiagonal_summary.add_argument("--json", action="store_true")
    gaussian_semidiagonal_promote = gaussian_sub.add_parser(
        "semidiagonal-promote",
        help="Summarize and promote Gaussian semidiagonal rovibrational output into xyzin",
    )
    gaussian_semidiagonal_promote.add_argument("log", type=Path)
    gaussian_semidiagonal_promote.add_argument("xyzin", type=Path)
    gaussian_semidiagonal_promote.add_argument("--json", action="store_true")
    gaussian_promote_fchk = gaussian_sub.add_parser(
        "promote-fchk",
        help="Promote Gaussian FCHK Hessian/normal-mode/QFF data into a MATRIX xyzin",
    )
    gaussian_promote_fchk.add_argument("fchk", type=Path)
    gaussian_promote_fchk.add_argument("xyzin", type=Path)
    gaussian_promote_fchk.add_argument("--no-cartesian-hessian", action="store_true")
    gaussian_promote_fchk.add_argument("--no-normal-modes", action="store_true")
    gaussian_promote_fchk.add_argument("--no-qff", action="store_true")
    gaussian_promote_fchk.add_argument("--no-electronic", action="store_true")
    gaussian_promote_fchk.add_argument("--no-orbitals", action="store_true")
    gaussian_promote_log_hessian = gaussian_sub.add_parser(
        "promote-log-hessian",
        help="Promote a printed Gaussian log Cartesian Hessian into a MATRIX xyzin",
    )
    gaussian_promote_log_hessian.add_argument("log", type=Path)
    gaussian_promote_log_hessian.add_argument("xyzin", type=Path)
    gaussian_promote_log_hessian.add_argument("--no-normal-modes", action="store_true")
    gaussian_promote_electronic = gaussian_sub.add_parser(
        "promote-electronic",
        help="Promote Gaussian electronic states/transitions into a MATRIX xyzin",
    )
    gaussian_promote_electronic.add_argument("log", type=Path)
    gaussian_promote_electronic.add_argument("xyzin", type=Path)
    gaussian_promote_electronic.add_argument("--no-electronic", action="store_true")
    gaussian_promote_electronic.add_argument("--no-transitions", action="store_true")
    gaussian_promote_electronic.add_argument(
        "--orbital-file",
        type=Path,
        action="append",
        default=[],
        help="Register an external Molden/Cube/FCHK orbital or density file in #ORBITALS",
    )
    gaussian_promote_rovib = gaussian_sub.add_parser(
        "promote-rovib",
        help="Promote Gaussian rovibrational log data into a MATRIX xyzin",
    )
    gaussian_promote_rovib.add_argument("log", type=Path)
    gaussian_promote_rovib.add_argument("xyzin", type=Path)
    gaussian_promote_rovib.add_argument("--no-vibrational", action="store_true")
    gaussian_promote_rovib.add_argument("--no-rotational", action="store_true")
    gaussian_promote_rovib.add_argument("--no-deltabvib", action="store_true")
    gaussian_promote_rovib.add_argument("--no-semidiagonal-cubic", action="store_true")
    gaussian_promote_rovib.add_argument("--no-invert-imaginary", action="store_true")
    gaussian_promote_rovib.add_argument(
        "--exclude-mode",
        type=int,
        action="append",
        default=[],
        help="Exclude a normal-mode index from alpha-derived DeltaBvib",
    )
    gaussian_promote_quadrupole = gaussian_sub.add_parser(
        "promote-quadrupole",
        help="Promote Gaussian nuclear quadrupole coupling constants into #PROPERTIES",
    )
    gaussian_promote_quadrupole.add_argument("log", type=Path)
    gaussian_promote_quadrupole.add_argument("xyzin", type=Path)

    molpro = sub.add_parser("molpro", help="Molpro output adapter utilities")
    molpro_sub = molpro.add_subparsers(dest="molpro_command")
    molpro_status = molpro_sub.add_parser("status", help="Inspect Molpro job state")
    molpro_status.add_argument("workdir", type=Path)
    molpro_status.add_argument("--input", type=Path)
    molpro_status.add_argument("--output", type=Path)
    molpro_run = molpro_sub.add_parser("run", help="Run Molpro from a work directory")
    molpro_run.add_argument("workdir", type=Path)
    molpro_run.add_argument("--executable")
    molpro_run.add_argument("--input", type=Path)
    molpro_run.add_argument("--output", type=Path)
    molpro_run.add_argument("--background", action="store_true")
    molpro_run.add_argument("--timeout", type=float)
    molpro_run.add_argument("--extra-arg", action="append", default=[])
    molpro_summary = molpro_sub.add_parser("summary", help="Summarize a Molpro output")
    molpro_summary.add_argument("output", type=Path)
    molpro_promote = molpro_sub.add_parser(
        "promote",
        help="Preprocess a Molpro output into a MATRIX xyzin",
    )
    molpro_promote.add_argument("output", type=Path)
    molpro_promote.add_argument("xyzin", type=Path)
    molpro_promote.add_argument("--symmetry-distance", type=float, default=1.0e-3)
    molpro_promote.add_argument("--symmetry-inertia", type=float, default=1.0e-3)
    molpro_promote.add_argument("--max-rotation-order", type=int, default=6)
    molpro_promote_quadrupole = molpro_sub.add_parser(
        "promote-quadrupole",
        help="Convert Molpro EFG tensors into quadrupole #PROPERTIES records",
    )
    molpro_promote_quadrupole.add_argument("output", type=Path)
    molpro_promote_quadrupole.add_argument("xyzin", type=Path)
    molpro_promote_quadrupole.add_argument("--atom", type=int, help="One-based EFG nucleus")
    molpro_promote_quadrupole.add_argument(
        "--isotope",
        help="Isotope label such as 14N; default uses MATRIX isotope table for the atom",
    )
    molpro_molden = molpro_sub.add_parser(
        "molden",
        help="Register a Molpro-produced Molden orbital file in #ORBITALS",
    )
    molpro_molden.add_argument("output", type=Path)
    molpro_molden.add_argument("xyzin", type=Path)
    molpro_molden.add_argument("--molden", type=Path, help="Molden file produced by Molpro")

    orca = sub.add_parser("orca", help="ORCA job utilities")
    orca_sub = orca.add_subparsers(dest="orca_command")
    orca_status = orca_sub.add_parser("status", help="Inspect ORCA job state")
    orca_status.add_argument("workdir", type=Path)
    orca_status.add_argument("--input", type=Path)
    orca_status.add_argument("--output", type=Path)
    orca_run = orca_sub.add_parser("run", help="Run ORCA from a work directory")
    orca_run.add_argument("workdir", type=Path)
    orca_run.add_argument("--executable")
    orca_run.add_argument("--input", type=Path)
    orca_run.add_argument("--output", type=Path)
    orca_run.add_argument("--background", action="store_true")
    orca_run.add_argument("--timeout", type=float)
    orca_run.add_argument("--extra-arg", action="append", default=[])
    orca_summary = orca_sub.add_parser("summary", help="Summarize an ORCA output")
    orca_summary.add_argument("output", type=Path)
    orca_promote = orca_sub.add_parser(
        "promote",
        help="Preprocess an ORCA output and optional Hessian into a MATRIX xyzin",
    )
    orca_promote.add_argument("output", type=Path)
    orca_promote.add_argument("xyzin", type=Path)
    orca_promote.add_argument("--symmetry-distance", type=float, default=1.0e-3)
    orca_promote.add_argument("--symmetry-inertia", type=float, default=1.0e-3)
    orca_promote.add_argument("--max-rotation-order", type=int, default=6)
    orca_promote_quadrupole = orca_sub.add_parser(
        "promote-quadrupole",
        help="Promote ORCA quadrupole/EFG data into #PROPERTIES",
    )
    orca_promote_quadrupole.add_argument("output", type=Path)
    orca_promote_quadrupole.add_argument("xyzin", type=Path)
    orca_molden = orca_sub.add_parser(
        "molden",
        help="Convert ORCA GBW to Molden with orca_2mkl and register #ORBITALS",
    )
    orca_molden.add_argument("gbw", type=Path)
    orca_molden.add_argument("xyzin", type=Path)
    orca_molden.add_argument("--output", type=Path, help="Destination Molden file")
    orca_molden.add_argument("--executable", default="orca_2mkl")
    orca_molden.add_argument("--timeout", type=float)

    mrcc = sub.add_parser("mrcc", help="MRCC output adapter utilities")
    mrcc_sub = mrcc.add_subparsers(dest="mrcc_command")
    mrcc_status = mrcc_sub.add_parser("status", help="Inspect MRCC job state")
    mrcc_status.add_argument("workdir", type=Path)
    mrcc_status.add_argument("--input", type=Path)
    mrcc_status.add_argument("--output", type=Path)
    mrcc_run = mrcc_sub.add_parser("run", help="Run MRCC from a work directory")
    mrcc_run.add_argument("workdir", type=Path)
    mrcc_run.add_argument("--executable")
    mrcc_run.add_argument("--input", type=Path)
    mrcc_run.add_argument("--output", type=Path)
    mrcc_run.add_argument("--background", action="store_true")
    mrcc_run.add_argument("--timeout", type=float)
    mrcc_run.add_argument("--extra-arg", action="append", default=[])
    mrcc_summary = mrcc_sub.add_parser("summary", help="Summarize an MRCC output")
    mrcc_summary.add_argument("output", type=Path)
    mrcc_promote = mrcc_sub.add_parser(
        "promote",
        help="Preprocess an MRCC output into a MATRIX xyzin",
    )
    mrcc_promote.add_argument("output", type=Path)
    mrcc_promote.add_argument("xyzin", type=Path)
    mrcc_promote.add_argument("--symmetry-distance", type=float, default=1.0e-3)
    mrcc_promote.add_argument("--symmetry-inertia", type=float, default=1.0e-3)
    mrcc_promote.add_argument("--max-rotation-order", type=int, default=6)

    cfour = sub.add_parser("cfour", help="CFOUR job utilities")
    cfour_sub = cfour.add_subparsers(dest="cfour_command")
    cfour_status = cfour_sub.add_parser("status", help="Inspect CFOUR job state")
    cfour_status.add_argument("workdir", type=Path)
    cfour_status.add_argument("--input", type=Path)
    cfour_status.add_argument("--output", type=Path)
    cfour_run = cfour_sub.add_parser("run", help="Run CFOUR from a work directory")
    cfour_run.add_argument("workdir", type=Path)
    cfour_run.add_argument("--executable")
    cfour_run.add_argument("--input", type=Path)
    cfour_run.add_argument("--output", type=Path)
    cfour_run.add_argument("--background", action="store_true")
    cfour_run.add_argument("--timeout", type=float)
    cfour_run.add_argument("--extra-arg", action="append", default=[])

    xtb = sub.add_parser("xtb", help="xTB job and output adapter utilities")
    xtb_sub = xtb.add_subparsers(dest="xtb_command")
    xtb_status = xtb_sub.add_parser("status", help="Inspect xTB job state")
    xtb_status.add_argument("workdir", type=Path)
    xtb_status.add_argument("--input", type=Path)
    xtb_status.add_argument("--output", type=Path)
    xtb_run = xtb_sub.add_parser("run", help="Run xTB from a work directory")
    xtb_run.add_argument("workdir", type=Path)
    xtb_run.add_argument("--executable")
    xtb_run.add_argument("--input", type=Path)
    xtb_run.add_argument("--output", type=Path)
    xtb_run.add_argument("--background", action="store_true")
    xtb_run.add_argument("--timeout", type=float)
    xtb_run.add_argument("--extra-arg", action="append", default=[])
    xtb_summary = xtb_sub.add_parser("summary", help="Summarize an xTB output")
    xtb_summary.add_argument("output", type=Path)
    xtb_summary.add_argument("--geometry", type=Path)

    pyscf = sub.add_parser("pyscf", help="PySCF job and structured-output utilities")
    pyscf_sub = pyscf.add_subparsers(dest="pyscf_command")
    pyscf_status = pyscf_sub.add_parser("status", help="Inspect PySCF job state")
    pyscf_status.add_argument("workdir", type=Path)
    pyscf_status.add_argument("--input", type=Path)
    pyscf_status.add_argument("--output", type=Path)
    pyscf_run = pyscf_sub.add_parser("run", help="Run a PySCF script from a work directory")
    pyscf_run.add_argument("workdir", type=Path)
    pyscf_run.add_argument("--executable")
    pyscf_run.add_argument("--input", type=Path)
    pyscf_run.add_argument("--output", type=Path)
    pyscf_run.add_argument("--background", action="store_true")
    pyscf_run.add_argument("--timeout", type=float)
    pyscf_run.add_argument("--extra-arg", action="append", default=[])
    pyscf_summary = pyscf_sub.add_parser("summary", help="Summarize a MATRIX PySCF output")
    pyscf_summary.add_argument("output", type=Path)

    et = sub.add_parser("et", help="eT job utilities")
    et_sub = et.add_subparsers(dest="et_command")
    et_status = et_sub.add_parser("status", help="Inspect eT job state")
    et_status.add_argument("workdir", type=Path)
    et_status.add_argument("--input", type=Path)
    et_status.add_argument("--output", type=Path)
    et_run = et_sub.add_parser("run", help="Run eT from a work directory")
    et_run.add_argument("workdir", type=Path)
    et_run.add_argument("--executable")
    et_run.add_argument("--input", type=Path)
    et_run.add_argument("--output", type=Path)
    et_run.add_argument("--background", action="store_true")
    et_run.add_argument("--timeout", type=float)
    et_run.add_argument("--extra-arg", action="append", default=[])

    lcb25 = sub.add_parser("lcb25", help="Manage the local MATRIX LCB25 geometry cache")
    lcb25_sub = lcb25.add_subparsers(dest="lcb25_command")
    fetch = lcb25_sub.add_parser("fetch", help="Download/extract LCB25 geometries once")
    fetch.add_argument("--root", type=Path, default=root / "data" / "lcb25")
    fetch.add_argument("--dataset", action="append", help="PCS2, SE or HPCS2; repeatable")
    fetch.add_argument("--force", action="store_true")

    fragments = sub.add_parser("fragments", help="Manage topology-backed fragment workflows")
    fragments_sub = fragments.add_subparsers(dest="fragments_command")
    plan = fragments_sub.add_parser("plan", help="Write the initial #FRAGMENTS section")
    plan.add_argument("xyzin", type=Path)
    fragments_build = fragments_sub.add_parser("build", help="Build concrete #FRAGMENTS")
    fragments_build.add_argument("xyzin", type=Path)
    fragments_state = fragments_sub.add_parser(
        "set-state",
        help="Set fragment charge and multiplicity in the shared #FRAGMENTS contract",
    )
    fragments_state.add_argument("xyzin", type=Path)
    fragments_state.add_argument(
        "state",
        nargs="+",
        metavar="ID:CHARGE:MULTIPLICITY",
        help="Electronic state, for example F001:0:1",
    )
    centers = fragments_sub.add_parser(
        "centers",
        help="Build virtual bond/ring interaction centers for GICForge",
    )
    centers.add_argument("xyzin", type=Path)

    rovib = sub.add_parser("rovib", help="Standalone rovibrational xyzin utilities")
    rovib_sub = rovib.add_subparsers(dest="rovib_command")
    rovib_summary = rovib_sub.add_parser("summarize", help="Summarize rovib sections")
    rovib_summary.add_argument("xyzin", type=Path)
    rovib_rotational = rovib_sub.add_parser(
        "rotational", help="Compute and store the rotational state from equilibrium geometry"
    )
    rovib_rotational.add_argument("xyzin", type=Path)
    rovib_rotational.add_argument("--out", type=Path, help="Rotational analysis report")
    rovib_rotational.add_argument("--no-report", action="store_true")
    rovib_rotational.add_argument("--no-vibrational-analysis", action="store_true")
    rovib_rotational.add_argument("--coriolis-threshold-cm1", type=float, default=1.0)
    rovib_vibin = rovib_sub.add_parser("vibin", help="Build Merlino-compatible vibin from FCHK")
    rovib_vibin.add_argument("xyzin", type=Path)
    rovib_vibin.add_argument("--fchk", type=Path, required=True)
    rovib_vibin.add_argument("--workdir", type=Path)
    rovib_vibin.add_argument("--no-project-tr", action="store_true")
    rovib_vibin.add_argument("--no-update-vibrational", action="store_true")
    rovib_coriolis = rovib_sub.add_parser("coriolis", help="Compute sparse Coriolis terms")
    rovib_coriolis.add_argument("xyzin", type=Path)
    rovib_coriolis.add_argument("--vibin", type=Path)
    rovib_coriolis.add_argument("--threshold-cm1", type=float, default=1.0)
    rovib_coriolis.add_argument("--all-pairs", action="store_true")
    rovib_coriolis.add_argument("--append-vibin", action="store_true")
    rovib_coriolis.add_argument("--out", type=Path)
    rovib_qcent = rovib_sub.add_parser("qcent", help="Compute quartic centrifugal distortion")
    rovib_qcent.add_argument("xyzin", type=Path)
    rovib_qcent.add_argument("--vibin", type=Path)
    rovib_qcent.add_argument("--append-vibin", action="store_true")
    rovib_qcent.add_argument("--out", type=Path)
    rovib_one_mode = rovib_sub.add_parser(
        "one-mode", help="Gaussian-aligned dI/dQ, I(Q) and 1/I(Q) diagnostic"
    )
    rovib_one_mode.add_argument("--log", type=Path, required=True)
    rovib_one_mode.add_argument("--fchk", type=Path, required=True)
    rovib_one_mode.add_argument("--mode", type=int, required=True)
    rovib_one_mode.add_argument("--qmax", type=float, default=1.0)
    rovib_one_mode.add_argument("--nq", type=int, default=101)
    rovib_one_mode.add_argument("--axis", choices=("A", "B", "C"))
    rovib_one_mode.add_argument("--quartic-cm1", type=float, default=0.0)
    rovib_one_mode.add_argument("--basis-size", type=int, default=32)
    rovib_one_mode.add_argument("--out", type=Path)
    rovib_external = rovib_sub.add_parser(
        "import-external", help="Import CeDiTT/alpha-resonance JSON or CeDiTT CSV"
    )
    rovib_external.add_argument("xyzin", type=Path)
    rovib_external.add_argument("payload", type=Path)
    rovib_wmsrot = rovib_sub.add_parser(
        "wmsrot-input",
        help="Export a WMS-Rot browser input file from normalized xyzin sections",
    )
    rovib_wmsrot.add_argument("xyzin", type=Path)
    rovib_wmsrot.add_argument("--out", type=Path)
    rovib_wmsrot.add_argument("--j-min", type=int, default=0)
    rovib_wmsrot.add_argument("--j-max", type=int, default=30)
    rovib_wmsrot.add_argument("--auto-estimate-j-range", action="store_true")
    rovib_wmsrot.add_argument("--reduction", choices=("A", "S"))
    rovib_wmsrot_run = rovib_sub.add_parser(
        "wmsrot-run",
        help="Run the vendored WMS-Rot Hamiltonian engine on normalized xyzin data",
    )
    rovib_wmsrot_run.add_argument("xyzin", type=Path)
    rovib_wmsrot_run.add_argument("--out", type=Path, required=True)
    rovib_wmsrot_run.add_argument("--plot", type=Path)
    rovib_wmsrot_run.add_argument("--table", type=Path, help="Write a LaTeX transition table")
    rovib_wmsrot_run.add_argument("--fwhm-mhz", type=float, default=0.0)
    rovib_wmsrot_run.add_argument("--no-write-section", action="store_true")
    rovib_wmsrot_run.add_argument("--j-min", type=int, default=0)
    rovib_wmsrot_run.add_argument("--j-max", type=int, default=30)
    rovib_wmsrot_run.add_argument("--intensity-cut", type=float, default=1.0e-20)
    rovib_wmsrot_run.add_argument("--reduction", choices=("A", "S"))
    rovib_wmsrot_run.add_argument("--no-a-type", action="store_true")
    rovib_wmsrot_run.add_argument("--no-b-type", action="store_true")
    rovib_wmsrot_run.add_argument("--no-c-type", action="store_true")
    rovib_vib_spectrum = rovib_sub.add_parser(
        "vib-spectrum",
        help="Build a broadened IR/Raman/VCD/ROA spectrum from #VIBRATIONAL",
    )
    rovib_vib_spectrum.add_argument("xyzin", type=Path)
    rovib_vib_spectrum.add_argument(
        "--observable",
        choices=("IR", "RAMAN", "VCD", "ROA"),
        default="IR",
    )
    rovib_vib_spectrum.add_argument(
        "--source",
        choices=("harmonic", "anharmonic", "hybrid"),
        default="harmonic",
    )
    rovib_vib_spectrum.add_argument(
        "--level2-xyzin",
        type=Path,
        help="Level-2 xyzin used for hybrid harmonic(level1)+anharmonic correction(level2)",
    )
    rovib_vib_spectrum.add_argument("--csv", type=Path, required=True)
    rovib_vib_spectrum.add_argument("--plot", type=Path)
    rovib_vib_spectrum.add_argument("--peaks", type=Path)
    rovib_vib_spectrum.add_argument("--mode-match-csv", type=Path)
    rovib_vib_spectrum.add_argument("--min-mode-overlap", type=float, default=0.70)
    rovib_vib_spectrum.add_argument("--fwhm-cm1", type=float, default=10.0)
    rovib_vib_spectrum.add_argument("--step-cm1", type=float, default=1.0)
    rovib_vib_spectrum.add_argument(
        "--lineshape",
        choices=("gaussian", "lorentzian"),
        default="gaussian",
    )
    rovib_vib_spectrum.add_argument("--no-normalize", action="store_true")
    rovib_vib_compare = rovib_sub.add_parser(
        "vib-compare",
        help="Compare two vibrational spectra with mirror plotting for IR/Raman",
    )
    rovib_vib_compare.add_argument("xyzin", type=Path)
    rovib_vib_compare.add_argument(
        "second_xyzin",
        type=Path,
        nargs="?",
        help="Optional second xyzin file; defaults to the first file",
    )
    rovib_vib_compare.add_argument(
        "--observable",
        choices=("IR", "RAMAN", "VCD", "ROA"),
        default="IR",
    )
    rovib_vib_compare.add_argument(
        "--first-source",
        choices=("harmonic", "anharmonic", "hybrid"),
        default="harmonic",
    )
    rovib_vib_compare.add_argument(
        "--second-source",
        choices=("harmonic", "anharmonic", "hybrid"),
        default="anharmonic",
    )
    rovib_vib_compare.add_argument("--csv", type=Path, required=True)
    rovib_vib_compare.add_argument("--plot", type=Path)
    rovib_vib_compare.add_argument("--mode-match-csv", type=Path)
    rovib_vib_compare.add_argument("--min-mode-overlap", type=float, default=0.70)
    rovib_vib_compare.add_argument("--fwhm-cm1", type=float, default=10.0)
    rovib_vib_compare.add_argument("--step-cm1", type=float, default=1.0)
    rovib_vib_compare.add_argument(
        "--lineshape",
        choices=("gaussian", "lorentzian"),
        default="gaussian",
    )
    rovib_vib_compare.add_argument("--no-normalize", action="store_true")
    rovib_vib_compare.add_argument(
        "--no-mirror-second",
        action="store_true",
        help="Do not mirror the second spectrum even for IR/Raman",
    )
    rovib_nist_ir = rovib_sub.add_parser(
        "nist-ir",
        help="Download a NIST gas-phase IR JCAMP spectrum and convert it to CSV",
    )
    rovib_nist_ir.add_argument(
        "identifier",
        help="NIST ID, CAS registry number or molecule name",
    )
    rovib_nist_ir.add_argument("--out", type=Path, required=True)
    rovib_nist_ir.add_argument("--index", type=int, default=1)
    rovib_nist_ir.add_argument("--timeout", type=float, default=20.0)
    rovib_dos = rovib_sub.add_parser("dos", help="Build direct vibrational DOS from #VIBRATIONAL")
    rovib_dos.add_argument("xyzin", type=Path)
    rovib_dos.add_argument("--vmax", type=int, default=6)
    rovib_dos.add_argument("--emax", type=float, default=8000.0)
    rovib_dos.add_argument("--emin", type=float, default=0.0)
    rovib_dos.add_argument("--bin-cm1", type=float, default=50.0)
    rovib_dos.add_argument("--ncap", type=float, default=10.0)
    rovib_dos.add_argument("--temperature", type=float)
    rovib_dos.add_argument("--out", type=Path)
    rovib_dos.add_argument("--q-out", type=Path)
    rovib_dos.add_argument("--number-out", type=Path)
    rovib_dos.add_argument("--cache", type=Path)
    rovib_dos.add_argument("--no-cache", action="store_true")
    rovib_dos_rovib = rovib_sub.add_parser(
        "dos-rovib", help="Convolve vibrational and rotational DOS"
    )
    rovib_dos_rovib.add_argument("xyzin", type=Path)
    rovib_dos_rovib.add_argument("--vib-dos", type=Path)
    rovib_dos_rovib.add_argument("--out", type=Path)
    rovib_dos_rovib.add_argument("--rot-out", type=Path)
    rovib_dos_rovib.add_argument("--q-out", type=Path)
    rovib_dos_rovib.add_argument("--number-out", type=Path)
    rovib_dos_rovib.add_argument("--emax-rot", type=float)
    rovib_dos_rovib.add_argument("--jmax", type=int)

    thermo = sub.add_parser("thermo", help="Run thermochemistry from a MATRIX xyzin")
    thermo.add_argument("xyzin", type=Path)
    thermo.add_argument("--out", type=Path, help="Write the readable thermo report here")
    thermo.add_argument("--no-report", action="store_true", help="Do not write thermo.report")
    thermo.add_argument("--no-write-section", action="store_true", help="Do not update #THERMO")
    thermo.add_argument("--cutoff-cm1", type=float, default=10.0)
    thermo.add_argument("--keep-low-positive", action="store_true")

    kinetics = sub.add_parser(
        "kinetics", help="Canonical TST and microcanonical RRKM kinetics"
    )
    kinetics_sub = kinetics.add_subparsers(dest="kinetics_command")
    kinetics_single = kinetics_sub.add_parser(
        "single", help="Run one reaction channel using reactant and TS DOS"
    )
    kinetics_single.add_argument("reactant_xyzin", type=Path)
    kinetics_single.add_argument("transition_state_xyzin", type=Path)
    kinetics_single.add_argument("--reactant-dos", type=Path, required=True)
    kinetics_single.add_argument("--ts-dos", type=Path, required=True)
    kinetics_single.add_argument("--barrier-cm1", type=float, required=True)
    kinetics_single.add_argument(
        "--barrier-reference", choices=("zero_point",), default="zero_point"
    )
    kinetics_single.add_argument("--temperature", type=float)
    kinetics_single.add_argument("--network-id", default="network-1")
    kinetics_single.add_argument("--reaction-id", default="R1")
    kinetics_single.add_argument("--reactant-id", default="R")
    kinetics_single.add_argument("--ts-id", default="TS1")
    kinetics_single.add_argument("--product-id", default="P")
    kinetics_single.add_argument("--path-degeneracy", type=float, default=1.0)
    kinetics_single.add_argument("--tunneling", choices=("none", "wigner"), default="none")
    kinetics_single.add_argument("--imaginary-frequency-cm1", type=float)
    kinetics_single.add_argument("--rrkm-out", type=Path)
    kinetics_single.add_argument("--manifest", type=Path)
    kinetics_single.add_argument("--report", type=Path)
    kinetics_single.add_argument("--no-write-section", action="store_true")
    kinetics_collision = kinetics_sub.add_parser(
        "collision", help="Hard-sphere bimolecular collision rate"
    )
    kinetics_collision.add_argument("--mass-a-amu", type=float, required=True)
    kinetics_collision.add_argument("--mass-b-amu", type=float, required=True)
    kinetics_collision.add_argument("--radius-a-angstrom", type=float, required=True)
    kinetics_collision.add_argument("--radius-b-angstrom", type=float, required=True)
    kinetics_collision.add_argument("--temperature", type=float, required=True)
    kinetics_collision.add_argument(
        "--convention", choices=("merlino-rates.f", "standard"), default="merlino-rates.f"
    )
    kinetics_gorin = kinetics_sub.add_parser(
        "gorin", help="Legacy Merlino -C6/r6 Gorin capture rate"
    )
    kinetics_gorin.add_argument("--mass-a-amu", type=float, required=True)
    kinetics_gorin.add_argument("--mass-b-amu", type=float, required=True)
    kinetics_gorin.add_argument("--c6", type=float, required=True)
    kinetics_gorin.add_argument("--temperature", type=float, required=True)
    kinetics_nad = kinetics_sub.add_parser(
        "nonadiabatic", help="Spin-forbidden Landau-Zener TST rate"
    )
    kinetics_nad.add_argument("--reduced-mass-amu", type=float, required=True)
    kinetics_nad.add_argument("--spin-orbit-cm1", type=float, required=True)
    kinetics_nad.add_argument("--gradient-difference-ha-angstrom", type=float, required=True)
    kinetics_nad.add_argument("--crossing-energy-cm1", type=float, required=True)
    kinetics_nad.add_argument("--q-mecp", type=float, required=True)
    kinetics_nad.add_argument("--q-reactants", type=float, required=True)
    kinetics_nad.add_argument("--temperature", type=float, required=True)

    gf = sub.add_parser("gf", help="Run TRINITY harmonic analysis from a Cartesian Hessian")
    gf.add_argument("--fchk", type=Path)
    gf.add_argument(
        "--hessian-engine",
        choices=(
            "xyzin", "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf"
        ),
        help="Electronic-structure Hessian adapter used with --hessian-file",
    )
    gf.add_argument(
        "--hessian-file", type=Path, help="Cartesian Hessian source for --hessian-engine"
    )
    gf.add_argument("--hessian-cfour-grd", type=Path)
    gf.add_argument("--hessian-cfour-output", type=Path)
    gf.add_argument(
        "--hessian-xtb-geometry",
        type=Path,
        help="XYZ geometry matching the native xTB Hessian (default: sibling xtbopt.xyz)",
    )
    gf.add_argument(
        "--hessian-xtb-spectrum",
        type=Path,
        help="Native xTB vibspectrum companion used for frequency comparison",
    )
    gf.add_argument(
        "--hessian-xtb-output",
        type=Path,
        help="xTB output used to record the exact version and GFN Hamiltonian",
    )
    gf.add_argument(
        "--hessian-pyscf-input",
        type=Path,
        help="PySCF input script included in Hessian provenance by SHA-256",
    )
    gf.add_argument(
        "--write-hessian-section",
        action="store_true",
        help="Promote --hessian-engine/--hessian-file data into #CARTESIAN_HESSIAN before GF",
    )
    gf.add_argument("--xyzin", type=Path, help="Frozen MATRIX xyzin with a BUILT #GIC section")
    gf.add_argument("--out", type=Path, help="Write the TRINITY harmonic report")
    gf.add_argument("--csv-dir", type=Path, help="Write TRINITY harmonic CSV tables")
    gf.add_argument("--scale-file", type=Path)
    gf.add_argument("--scale", action="append", default=[])
    gf.add_argument(
        "--scale-class",
        action="append",
        default=[],
        help="Apply one Pulay factor to a named GIC class, name:factor:pattern|pattern",
    )
    gf.add_argument(
        "--scale-preview",
        "--dry-run-scaling",
        action="store_true",
        help="Print Pulay scaling assignments from #GIC and exit without running GF",
    )
    gf.add_argument("--local", action="store_true", help="Apply local force-field filtering")
    gf.add_argument("--symmetry-blocks", action="store_true", help="Solve separated irrep blocks")
    gf.add_argument(
        "--force-threshold", type=float, help="Zero internal force constants below threshold"
    )
    gf.add_argument(
        "--large-amplitude-frequency-cutoff-cm",
        type=float,
        default=250.0,
        help="Mark large-amplitude candidates active only below this local GF frequency",
    )
    gf.add_argument(
        "--large-amplitude-frequency-model",
        choices=("projected", "full", "diagonal-g", "separate"),
        default="projected",
        help=(
            "Large-amplitude block frequency model; projected uses "
            "the constrained metric inverse((G^-1)_SS)"
        ),
    )
    gf.add_argument(
        "--cv-correction",
        action="store_true",
        help="Apply the CV_radial Gaussian bond-length correction before building B/G",
    )
    gf.add_argument(
        "--cv-correction-sigma-scale",
        type=float,
        default=1.3,
        help="Scale factor multiplying covalent-radius sums in the CV_radial Gaussian",
    )
    gf.add_argument(
        "--cv-correction-threshold",
        type=float,
        default=0.9,
        help="Minimum CV_radial Gaussian weight required to correct a bond primitive",
    )
    gf.add_argument("--no-write-section", action="store_true", help="Do not update #GF_PED")
    gf.add_argument(
        "--subtract-electrostatic",
        action="store_true",
        help="Subtract CM5/synthon electrostatic Hessian terms before GF",
    )
    gf.add_argument(
        "--subtract-uff-vdw",
        action="store_true",
        help="Subtract UFF van der Waals Hessian terms before GF",
    )
    gf.add_argument(
        "--nonbonded-14-scale",
        type=float,
        default=0.5,
        help="Scale factor for 1-4 electrostatic and UFF-vdW terms",
    )

    vpt2_vci = sub.add_parser("vpt2-vci", help="Run VPT2/VCI from normalized MATRIX QFF data")
    vpt2_source = vpt2_vci.add_mutually_exclusive_group(required=True)
    vpt2_source.add_argument("--xyzin", type=Path, help="MATRIX xyzin containing #QFF")
    vpt2_source.add_argument("--fchk", type=Path, help="Gaussian FCHK adapter input")
    vpt2_source.add_argument("--qff-file", type=Path, help="Indexed QFF text adapter input")
    vpt2_source.add_argument(
        "--zion-force-field",
        type=Path,
        help="Compiled ARCHITECT/ZION field to project onto a TRINITY normal-mode QFF",
    )
    vpt2_source.add_argument(
        "--collect", type=Path, help="Collect post-run VPT2/VCI outputs into #VPT2_VCI"
    )
    vpt2_vci.add_argument("--max-quanta", type=int, default=2)
    vpt2_vci.add_argument("--roots", type=int, default=10)
    vpt2_vci.add_argument("--vci-method", choices=("dense", "davidson"), default="dense")
    vpt2_vci.add_argument("--run-dir", type=Path, help="Write report, CSV tables and manifest here")
    vpt2_vci.add_argument("--out", type=Path, help="Write the readable VPT2/VCI report")
    vpt2_vci.add_argument("--csv-dir", type=Path, help="Write VPT2/VCI CSV tables")
    vpt2_vci.add_argument(
        "--no-write", action="store_true", help="With --collect, do not update #VPT2_VCI"
    )
    vpt2_vci.add_argument(
        "--zion-xyzin", type=Path, help="MATRIX xyzin defining the frozen local SONIC basis"
    )
    vpt2_vci.add_argument("--zion-fit-amplitude", type=float, default=0.03)
    vpt2_vci.add_argument(
        "--zion-fit-pairs", type=int, default=0, help="Paired training points; 0 selects by rank"
    )
    vpt2_vci.add_argument("--zion-holdout-pairs", type=int, default=8)
    vpt2_vci.add_argument("--zion-fit-seed", type=int, default=20260716)
    vpt2_vci.add_argument("--zion-fit-workers", type=int, default=0)
    vpt2_vci.add_argument(
        "--zion-qff-out", type=Path, help="Write the fitted one-based indexed QFF"
    )
    vpt2_vci.add_argument(
        "--zion-fit-json", type=Path, help="Write coefficients, modes and fit diagnostics as JSON"
    )
    vpt2_vci.add_argument(
        "--zion-fit-only", action="store_true", help="Create and validate the QFF without VPT2/VCI"
    )

    hybrid_vibrations = sub.add_parser(
        "hybrid-vibrations",
        help="Route large-amplitude path modes to DVR/VCI and ordinary modes to VPT2/GVPT2",
    )
    hybrid_vibrations.add_argument("--fchk", type=Path, required=True)
    hybrid_vibrations.add_argument(
        "--path-pair",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("LOWER", "UPPER"),
        help="Flanking geometries defining one aligned central path tangent; repeat for a block",
    )
    hybrid_vibrations.add_argument("--anharmonic-log", type=Path)
    hybrid_vibrations.add_argument("--dvr-levels", type=Path)
    hybrid_vibrations.add_argument(
        "--variational-transition",
        action="append",
        default=[],
        metavar="LOWER:UPPER[:LABEL]",
        help="Transition read from the DVR/VCI level table; defaults to 0:1",
    )
    hybrid_vibrations.add_argument("--gaussian-input", type=Path)
    hybrid_vibrations.add_argument("--checkpoint", help="Checkpoint used by the Gaussian input")
    hybrid_vibrations.add_argument(
        "--route",
        default="",
        help="Electronic model/state route without Freq, Geom or Guess keywords",
    )
    hybrid_vibrations.add_argument("--processors", type=int)
    hybrid_vibrations.add_argument("--memory")
    hybrid_vibrations.add_argument("--minimum-principal-overlap", type=float, default=0.90)
    hybrid_vibrations.add_argument("--minimum-projection-gap", type=float, default=0.10)
    hybrid_vibrations.add_argument("--maximum-active-projection", type=float, default=0.20)
    hybrid_vibrations.add_argument("--minimum-tangent-singular-value", type=float, default=1.0e-6)
    hybrid_vibrations.add_argument("--maximum-mode-orthogonality-error", type=float, default=1.0e-5)
    hybrid_vibrations.add_argument("--report", type=Path)
    hybrid_vibrations.add_argument("--csv", type=Path)

    dvr = sub.add_parser("dvr", help="Prepare scan/path DVR workflows")
    dvr_sub = dvr.add_subparsers(dest="dvr_command")
    dvr_prepare = dvr_sub.add_parser(
        "prepare", help="Prepare DVR manifest from a Gaussian scan log"
    )
    dvr_prepare.add_argument("log", type=Path)
    dvr_prepare.add_argument("--outdir", type=Path, required=True)
    dvr_prepare.add_argument("--figdir", type=Path)
    dvr_prepare.add_argument("--prefix", default="puckering_dvr")
    dvr_prepare.add_argument("--boundary", default="periodic")
    dvr_prepare.add_argument(
        "--solver",
        choices=("fourier", "sinc-dvr", "fortran-sinc-dvr", "fortran-gaussian"),
        default="fourier",
    )
    dvr_prepare.add_argument("--no-rotconst", action="store_true")
    dvr_prepare.add_argument("--no-cremer-pople", action="store_true")
    dvr_prepare.add_argument("--check-only", action="store_true")
    dvr_prepare.add_argument("--xyzin", type=Path, help="Update this MATRIX xyzin with #DVR")
    dvr_run = dvr_sub.add_parser("run", help="Run DVR directly and update #DVR outputs")
    dvr_run.add_argument("log", type=Path, nargs="?")
    dvr_run.add_argument("--outdir", type=Path)
    dvr_run.add_argument("--figdir", type=Path)
    dvr_run.add_argument("--prefix", default="puckering_dvr")
    dvr_run.add_argument("--boundary", default="periodic")
    dvr_run.add_argument(
        "--solver",
        choices=("fourier", "sinc-dvr", "fortran-sinc-dvr", "fortran-gaussian"),
        default="fourier",
    )
    dvr_run.add_argument("--no-rotconst", action="store_true")
    dvr_run.add_argument("--no-cremer-pople", action="store_true")
    dvr_run.add_argument("--check-only", action="store_true")
    dvr_run.add_argument(
        "--xyzin",
        type=Path,
        help="Read #DVR when LOG is omitted; otherwise update this MATRIX xyzin",
    )
    dvr_run.add_argument("--timeout", type=float)
    dvr_collect = dvr_sub.add_parser("collect", help="Collect post-run DVR outputs into #DVR")
    dvr_collect.add_argument("xyzin", type=Path)
    dvr_collect.add_argument(
        "--no-write",
        action="store_true",
        help="Read and summarize outputs without updating #DVR",
    )

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

    trinity = sub.add_parser(
        "trinity",
        help="Prepare LINK external energy/gradient geometry optimization state",
    )
    trinity_sub = trinity.add_subparsers(dest="trinity_command")
    trinity_prepare = trinity_sub.add_parser("prepare", help="Write a prepared #TRINITY section")
    trinity_prepare.add_argument("xyzin", type=Path)
    trinity_prepare.add_argument("--run-dir", type=Path, required=True)
    trinity_prepare.add_argument("--engine-command", required=True)
    trinity_prepare.add_argument("--coordinate-model", choices=("gic", "cartesian"), default="gic")
    trinity_prepare.add_argument("--active-space", default="total_symmetric")
    trinity_prepare.add_argument("--max-steps", type=int, default=50)
    trinity_prepare.add_argument("--trust-radius", type=float, default=0.2)
    trinity_prepare.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    trinity_prepare.add_argument("--step-tolerance", type=float, default=1.0e-5)
    trinity_prepare.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    trinity_prepare.add_argument("--energy-unit", default="hartree")
    trinity_prepare.add_argument("--gradient-unit", default="hartree/bohr")
    trinity_prepare.add_argument("--external-protocol", default="xyz-energy-gradient-v1")
    trinity_status = trinity_sub.add_parser("status", help="Summarize #TRINITY state")
    trinity_status.add_argument("xyzin", type=Path)
    trinity_scan_prepare = trinity_sub.add_parser(
        "scan-prepare",
        help="Prepare LINK displaced geometries for external point calculations",
    )
    trinity_scan_prepare.add_argument("xyzin", type=Path)
    trinity_scan_prepare.add_argument("--run-dir", type=Path, required=True)
    trinity_scan_prepare.add_argument("--engine-command", default="")
    trinity_scan_prepare.add_argument(
        "--coordinate-kind",
        choices=("sonic", "normal-mode", "cartesian"),
        default="sonic",
    )
    trinity_scan_prepare.add_argument(
        "--coordinate",
        required=True,
        help="GIC label/name/index, normal-mode index, or comma-separated Cartesian vector",
    )
    trinity_scan_prepare.add_argument("--step", type=float, default=0.01)
    trinity_scan_prepare.add_argument("--points-each-side", type=int, default=1)
    trinity_scan_prepare.add_argument(
        "--retained-group",
        default="C1",
        help="Minimum point group retained by every PES point; default C1",
    )
    trinity_scan_prepare.add_argument(
        "--displacement",
        type=float,
        action="append",
        default=[],
        help="Explicit displacement value; repeat to override symmetric step grid",
    )
    trinity_scan_prepare.add_argument(
        "--external-protocol",
        default="xyz-energy-gradient-json-v1",
    )
    trinity_scan_run = trinity_sub.add_parser(
        "scan-run",
        help="Run a prepared LINK external scan command and collect JSON point results",
    )
    trinity_scan_run.add_argument("xyzin", type=Path)
    trinity_scan_run.add_argument("--run-dir", type=Path, required=True)
    trinity_scan_run.add_argument("--engine-command", default="")
    trinity_scan_run.add_argument(
        "--backend",
        choices=(
            "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf", "et", "architect"
        ),
        help="Use a built-in MATRIX QM or ARCHITECT point adapter instead of --engine-command",
    )
    trinity_scan_run.add_argument("--route", default="")
    trinity_scan_run.add_argument("--method", default="")
    trinity_scan_run.add_argument("--basis", default="")
    trinity_scan_run.add_argument("--charge", type=int, default=0)
    trinity_scan_run.add_argument("--multiplicity", type=int, default=1)
    trinity_scan_run.add_argument("--electronic-state", type=int, default=0)
    trinity_scan_run.add_argument("--excited-states", type=int)
    trinity_scan_run.add_argument(
        "--state-spin", choices=("singlet", "triplet"), default="singlet"
    )
    trinity_scan_run.add_argument("--freeze-core", action="store_true")
    trinity_scan_run.add_argument(
        "--gradient-mode",
        choices=("analytic", "numerical", "cartesian-numerical"),
        default="analytic",
    )
    trinity_scan_run.add_argument(
        "--numerical-gradient-step-bohr", type=float, default=1.0e-3
    )
    trinity_scan_run.add_argument(
        "--numerical-gradient-stencil", choices=("central", "forward"), default="central"
    )
    trinity_scan_run.add_argument("--backend-workers", type=int, default=1)
    trinity_scan_run.add_argument("--backend-memory-gb", type=int)
    trinity_scan_run.add_argument("--executable")
    trinity_scan_run.add_argument(
        "--force-field", type=Path, help="ARCHITECT force-field JSON for --backend architect"
    )
    trinity_scan_run.add_argument("--extra-arg", action="append", default=[])
    trinity_scan_run.add_argument(
        "--coordinate-kind",
        choices=("sonic", "normal-mode", "cartesian"),
        default="sonic",
    )
    trinity_scan_run.add_argument("--coordinate", required=True)
    trinity_scan_run.add_argument("--step", type=float, default=0.01)
    trinity_scan_run.add_argument("--points-each-side", type=int, default=1)
    trinity_scan_run.add_argument(
        "--retained-group",
        default="C1",
        help="Minimum point group retained by every PES point; default C1",
    )
    trinity_scan_run.add_argument("--timeout", type=float)
    trinity_driver_run = trinity_sub.add_parser(
        "driver-run",
        help="Run LINK with an external program selecting the next SONIC point",
    )
    _add_external_driver_arguments(trinity_driver_run)
    trinity_optimize = trinity_sub.add_parser(
        "optimize",
        help="Run an optimization from a Gaussian-like Cartesian or SMILES input",
    )
    trinity_optimize.add_argument("input", type=Path)
    trinity_optimize.add_argument("--run-dir", type=Path, required=True)
    trinity_optimize_run = trinity_sub.add_parser(
        "optimize-run",
        help="Run the LINK information-efficient geometry optimizer",
    )
    trinity_optimize_run.add_argument("xyzin", type=Path)
    trinity_optimize_run.add_argument("--run-dir", type=Path, required=True)
    trinity_optimize_run.add_argument(
        "--optimized-xyzin",
        type=Path,
        help="Write the final LINK geometry to a new enriched XYZ project",
    )
    trinity_optimize_run.add_argument(
        "--background", action="store_true", help="Detach the complete LINK run and return its PID"
    )
    trinity_optimize_run.add_argument("--engine-command", default="")
    trinity_optimize_run.add_argument(
        "--backend",
        choices=(
            "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf", "et", "architect"
        ),
        help="Use a built-in MATRIX QM or ARCHITECT point adapter instead of --engine-command",
    )
    trinity_optimize_run.add_argument("--route", default="")
    trinity_optimize_run.add_argument("--method", default="")
    trinity_optimize_run.add_argument("--basis", default="")
    trinity_optimize_run.add_argument("--charge", type=int, default=0)
    trinity_optimize_run.add_argument("--multiplicity", type=int, default=1)
    trinity_optimize_run.add_argument("--electronic-state", type=int, default=0)
    trinity_optimize_run.add_argument("--excited-states", type=int)
    trinity_optimize_run.add_argument(
        "--state-spin", choices=("singlet", "triplet"), default="singlet"
    )
    trinity_optimize_run.add_argument("--freeze-core", action="store_true")
    trinity_optimize_run.add_argument("--executable")
    trinity_optimize_run.add_argument(
        "--force-field", type=Path, help="ARCHITECT force-field JSON for --backend architect"
    )
    trinity_optimize_run.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional backend keyword/argument; repeat as needed",
    )
    trinity_optimize_run.add_argument(
        "--gradient-mode",
        choices=("analytic", "numerical", "cartesian-numerical"),
        default="analytic",
        help=(
            "analytic: backend gradient; numerical: adaptive parallel LINK/SONIC energy "
            "differences; cartesian-numerical: diagnostic backend 3N energy differences"
        ),
    )
    trinity_optimize_run.add_argument(
        "--numerical-gradient-step-bohr", type=float, default=1.0e-3
    )
    trinity_optimize_run.add_argument(
        "--numerical-gradient-stencil", choices=("central", "forward"), default="central"
    )
    trinity_optimize_run.add_argument("--backend-workers", type=int, default=1)
    trinity_optimize_run.add_argument("--backend-memory-gb", type=int)
    trinity_optimize_run.add_argument(
        "--coordinate-kind",
        choices=("cartesian", "sonic"),
        default="cartesian",
    )
    trinity_optimize_run.add_argument(
        "--coordinate",
        action="append",
        default=[],
        help="SONIC coordinate label/name/index; repeat for a reduced active space",
    )
    trinity_optimize_run.add_argument(
        "--variables",
        type=Path,
        help="JSON active-variable contract; non-SONIC variables are projected onto SONIC",
    )
    trinity_optimize_run.add_argument("--max-steps", type=int, default=50)
    trinity_optimize_run.add_argument(
        "--convergence",
        choices=("normal", "tight"),
        default="normal",
        help="LINK convergence preset; explicit tolerance options override the preset",
    )
    trinity_optimize_run.add_argument("--trust-radius", type=float, default=0.2)
    trinity_optimize_run.add_argument("--max-trust-radius", type=float, default=0.3)
    trinity_optimize_run.add_argument("--gradient-tolerance", type=float, default=4.5e-4)
    trinity_optimize_run.add_argument("--step-tolerance", type=float, default=1.8e-3)
    trinity_optimize_run.add_argument("--energy-tolerance", type=float, default=1.0e-6)
    trinity_optimize_run.add_argument("--max-force-tolerance", type=float)
    trinity_optimize_run.add_argument("--rms-force-tolerance", type=float)
    trinity_optimize_run.add_argument("--max-displacement-tolerance", type=float)
    trinity_optimize_run.add_argument("--rms-displacement-tolerance", type=float)
    trinity_optimize_run.add_argument("--fd-step", type=float, default=0.01)
    trinity_optimize_run.add_argument("--fd-hard-characteristic-scale", type=float, default=0.05)
    trinity_optimize_run.add_argument("--fd-soft-characteristic-scale", type=float, default=0.20)
    trinity_optimize_run.add_argument("--fd-min-step", type=float, default=1.0e-4)
    trinity_optimize_run.add_argument("--fd-max-step", type=float, default=0.05)
    trinity_optimize_run.add_argument("--energy-noise", type=float, default=1.0e-8)
    trinity_optimize_run.add_argument(
        "--auto-energy-noise-samples",
        type=int,
        default=0,
        help="Repeat the initial point this many times to estimate the energy noise floor for FD steps",
    )
    trinity_optimize_run.add_argument("--adaptive-fd-mode", action="store_true")
    trinity_optimize_run.add_argument("--fd-central-gradient-factor", type=float, default=5.0)
    trinity_optimize_run.add_argument("--selective-fd-refresh", action="store_true")
    trinity_optimize_run.add_argument("--fd-refresh-interval", type=int, default=3)
    trinity_optimize_run.add_argument("--fd-gradient-change-tolerance", type=float, default=1.0e-4)
    trinity_optimize_run.add_argument("--selective-min-refresh-fraction", type=float, default=0.25)
    trinity_optimize_run.add_argument("--selective-coupling-threshold", type=float, default=0.05)
    trinity_optimize_run.add_argument("--selective-fallback-rejections", type=int, default=1)
    trinity_optimize_run.add_argument(
        "--selective-fallback-gradient-growth", type=float, default=1.5
    )
    trinity_optimize_run.add_argument("--surrogate-max-samples", type=int, default=12)
    trinity_optimize_run.add_argument("--fd-parallel-workers", type=int, default=1)
    trinity_optimize_run.add_argument("--hessian-coupling-threshold", type=float, default=1.0e-8)
    trinity_optimize_run.add_argument("--sparse-hessian-updates", action="store_true")
    trinity_optimize_run.add_argument("--min-hessian-eigenvalue", type=float, default=1.0e-4)
    trinity_optimize_run.add_argument("--max-hessian-condition", type=float, default=1.0e8)
    trinity_optimize_run.add_argument("--fragment-radial-curvature", type=float)
    trinity_optimize_run.add_argument("--fragment-tangential-curvature", type=float)
    trinity_optimize_run.add_argument("--fragment-rotation-curvature", type=float)
    trinity_optimize_run.add_argument(
        "--coordinate-schedule",
        choices=("auto", "joint", "inter-intra-joint", "inter-intra-micro"),
        default="auto",
    )
    trinity_optimize_run.add_argument("--coordinate-phase-max-steps", type=int, default=8)
    trinity_optimize_run.add_argument("--coordinate-phase-gradient-factor", type=float, default=3.0)
    trinity_optimize_run.add_argument(
        "--backtransform-continuation-step",
        type=float,
        default=0.12,
        help="Maximum ring/inversion/mixed-soft continuation increment in radians",
    )
    trinity_optimize_run.add_argument("--backtransform-max-substeps", type=int, default=32)
    trinity_optimize_run.add_argument("--max-coordinate-step", type=float, default=0.25)
    trinity_optimize_run.add_argument("--line-search-reductions", type=int, default=6)
    trinity_optimize_run.add_argument("--energy-increase-tolerance", type=float)
    trinity_optimize_run.add_argument(
        "--hessian-update", choices=("auto", "bfgs", "sr1", "bofill"), default="auto"
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-model",
        choices=("auto", "berny", "almloef"),
        default="auto",
        help="Primitive model Hessian used when no explicit Hessian is supplied",
    )
    trinity_optimize_run.add_argument(
        "--enable-gdiis",
        action="store_true",
        help="Enable experimental safeguarded GDIIS substitution (disabled by default)",
    )
    trinity_optimize_run.add_argument("--coordinate-drift-warning", type=float, default=0.25)
    trinity_optimize_run.add_argument(
        "--core-valence-exponential",
        "--cv-exponential",
        dest="core_valence_exponential",
        action="store_true",
        help=(
            "Add the ORACLE CV_radial exponential energy, analytic gradient and "
            "Hessian to every LINK backend evaluation"
        ),
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian",
        type=Path,
        help="Optimizer-coordinate Hessian JSON used to seed the quasi-Newton model",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-engine",
        choices=(
            "xyzin", "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf"
        ),
        help="Electronic-structure format used by --initial-hessian-file",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-file",
        type=Path,
        help="Cartesian Hessian file/output read through --initial-hessian-engine",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-gradient-file",
        type=Path,
        help=(
            "Cartesian gradient in hartree/bohr (.json, .npy or text); when supplied, "
            "LINK asks ARCHITECT for B-prime and applies the exact off-equilibrium "
            "Hessian transformation"
        ),
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-b-prime-workers",
        type=int,
        default=0,
        help="Workers used by ARCHITECT's sparse analytic B-prime builder (0=automatic)",
    )
    trinity_optimize_run.add_argument("--initial-hessian-cfour-grd", type=Path)
    trinity_optimize_run.add_argument("--initial-hessian-cfour-output", type=Path)
    trinity_optimize_run.add_argument(
        "--initial-hessian-gaussian",
        type=Path,
        help=(
            "Deprecated compatibility alias for "
            "--initial-hessian-engine gaussian --initial-hessian-file"
        ),
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-seed-backend",
        choices=("gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour"),
        help="Run a low-level frequency/Hessian job before optimization and use it as seed",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-seed-command",
        default="",
        help=(
            "External Hessian seed command. Placeholders: {xyzin}, {workdir}, "
            "{hessian}, {output}. Use with --initial-hessian-seed-backend and "
            "--initial-hessian-file when a backend-specific input is needed."
        ),
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-seed-route",
        default="",
        help="Route/options for built-in low-level Hessian seed jobs",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-seed-method",
        default="",
        help="Method for built-in low-level Hessian seed jobs when route is not used",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-seed-basis",
        default="",
        help="Basis for built-in low-level Hessian seed jobs when route is not used",
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-gaussian-route",
        help=(
            "Deprecated compatibility alias for "
            "--initial-hessian-seed-backend gaussian --initial-hessian-seed-route"
        ),
    )
    trinity_optimize_run.add_argument(
        "--initial-hessian-run-dir",
        type=Path,
        help="Run directory for generated Hessian seed jobs (default: <run-dir>/hessian_seed)",
    )
    trinity_optimize_run.add_argument("--symmetry-reduction", action="store_true")
    trinity_optimize_run.add_argument("--one-sided", action="store_true")
    trinity_optimize_run.add_argument("--no-analytic-gradient", action="store_true")
    trinity_optimize_run.add_argument("--cache-tolerance", type=float, default=1.0e-10)
    trinity_optimize_run.add_argument(
        "--resume",
        action="store_true",
        help="Reuse optimizer_cache.jsonl from the run directory when present",
    )
    trinity_optimize_run.add_argument("--timeout", type=float)

    trinity_optimize_gf = trinity_sub.add_parser(
        "optimize-gf",
        help="Run geometry optimization, write the optimized xyzin, then launch GF/SONIC",
    )
    trinity_optimize_gf.add_argument("xyzin", type=Path)
    trinity_optimize_gf.add_argument("--run-dir", type=Path, required=True)
    trinity_optimize_gf.add_argument("--optimized-xyzin", type=Path)
    trinity_optimize_gf.add_argument("--engine-command", default="")
    trinity_optimize_gf.add_argument(
        "--backend",
        choices=(
            "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf", "et", "architect"
        ),
        help="Use a built-in MATRIX QM or ARCHITECT point adapter instead of --engine-command",
    )
    trinity_optimize_gf.add_argument("--route", default="")
    trinity_optimize_gf.add_argument("--method", default="")
    trinity_optimize_gf.add_argument("--basis", default="")
    trinity_optimize_gf.add_argument("--charge", type=int, default=0)
    trinity_optimize_gf.add_argument("--multiplicity", type=int, default=1)
    trinity_optimize_gf.add_argument("--executable")
    trinity_optimize_gf.add_argument(
        "--force-field", type=Path, help="ARCHITECT force-field JSON for --backend architect"
    )
    trinity_optimize_gf.add_argument(
        "--coordinate-kind",
        choices=("cartesian", "sonic"),
        default="cartesian",
    )
    trinity_optimize_gf.add_argument("--coordinate", action="append", default=[])
    trinity_optimize_gf.add_argument("--max-steps", type=int, default=50)
    trinity_optimize_gf.add_argument("--trust-radius", type=float, default=0.2)
    trinity_optimize_gf.add_argument("--max-trust-radius", type=float, default=0.5)
    trinity_optimize_gf.add_argument("--gradient-tolerance", type=float, default=4.5e-4)
    trinity_optimize_gf.add_argument("--step-tolerance", type=float, default=1.8e-3)
    trinity_optimize_gf.add_argument("--energy-tolerance", type=float, default=1.0e-6)
    trinity_optimize_gf.add_argument("--fd-step", type=float, default=0.01)
    trinity_optimize_gf.add_argument("--fd-min-step", type=float, default=1.0e-4)
    trinity_optimize_gf.add_argument("--fd-max-step", type=float, default=0.05)
    trinity_optimize_gf.add_argument("--energy-noise", type=float, default=1.0e-8)
    trinity_optimize_gf.add_argument("--adaptive-fd-mode", action="store_true")
    trinity_optimize_gf.add_argument("--selective-fd-refresh", action="store_true")
    trinity_optimize_gf.add_argument("--fd-refresh-interval", type=int, default=3)
    trinity_optimize_gf.add_argument("--fd-gradient-change-tolerance", type=float, default=1.0e-4)
    trinity_optimize_gf.add_argument("--selective-min-refresh-fraction", type=float, default=0.25)
    trinity_optimize_gf.add_argument("--selective-coupling-threshold", type=float, default=0.05)
    trinity_optimize_gf.add_argument("--selective-fallback-rejections", type=int, default=1)
    trinity_optimize_gf.add_argument(
        "--selective-fallback-gradient-growth", type=float, default=1.5
    )
    trinity_optimize_gf.add_argument("--surrogate-max-samples", type=int, default=12)
    trinity_optimize_gf.add_argument("--fd-parallel-workers", type=int, default=1)
    trinity_optimize_gf.add_argument("--hessian-coupling-threshold", type=float, default=1.0e-8)
    trinity_optimize_gf.add_argument("--sparse-hessian-updates", action="store_true")
    trinity_optimize_gf.add_argument("--min-hessian-eigenvalue", type=float, default=1.0e-4)
    trinity_optimize_gf.add_argument("--max-hessian-condition", type=float, default=1.0e8)
    trinity_optimize_gf.add_argument("--fragment-radial-curvature", type=float)
    trinity_optimize_gf.add_argument("--fragment-tangential-curvature", type=float)
    trinity_optimize_gf.add_argument("--fragment-rotation-curvature", type=float)
    trinity_optimize_gf.add_argument(
        "--coordinate-schedule",
        choices=("auto", "joint", "inter-intra-joint", "inter-intra-micro"),
        default="auto",
    )
    trinity_optimize_gf.add_argument("--coordinate-phase-max-steps", type=int, default=8)
    trinity_optimize_gf.add_argument("--coordinate-phase-gradient-factor", type=float, default=3.0)
    trinity_optimize_gf.add_argument(
        "--backtransform-continuation-step",
        type=float,
        default=0.12,
        help="Maximum ring/inversion/mixed-soft continuation increment in radians",
    )
    trinity_optimize_gf.add_argument("--backtransform-max-substeps", type=int, default=32)
    trinity_optimize_gf.add_argument("--max-coordinate-step", type=float, default=0.25)
    trinity_optimize_gf.add_argument("--line-search-reductions", type=int, default=6)
    trinity_optimize_gf.add_argument("--energy-increase-tolerance", type=float)
    trinity_optimize_gf.add_argument(
        "--hessian-update", choices=("auto", "bfgs", "sr1", "bofill"), default="auto"
    )
    trinity_optimize_gf.add_argument("--coordinate-drift-warning", type=float, default=0.25)
    trinity_optimize_gf.add_argument("--no-analytic-gradient", action="store_true")
    trinity_optimize_gf.add_argument("--cache-tolerance", type=float, default=1.0e-10)
    trinity_optimize_gf.add_argument("--initial-hessian", type=Path)
    trinity_optimize_gf.add_argument(
        "--initial-hessian-engine",
        choices=(
            "xyzin", "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf"
        ),
    )
    trinity_optimize_gf.add_argument("--initial-hessian-file", type=Path)
    trinity_optimize_gf.add_argument("--initial-hessian-cfour-grd", type=Path)
    trinity_optimize_gf.add_argument("--initial-hessian-cfour-output", type=Path)
    trinity_optimize_gf.add_argument(
        "--initial-hessian-seed-backend",
        choices=("gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour"),
    )
    trinity_optimize_gf.add_argument("--initial-hessian-seed-command", default="")
    trinity_optimize_gf.add_argument("--initial-hessian-seed-route", default="")
    trinity_optimize_gf.add_argument("--initial-hessian-seed-method", default="")
    trinity_optimize_gf.add_argument("--initial-hessian-seed-basis", default="")
    trinity_optimize_gf.add_argument("--initial-hessian-run-dir", type=Path)
    trinity_optimize_gf.add_argument("--gf-fchk", type=Path)
    trinity_optimize_gf.add_argument(
        "--gf-hessian-engine",
        choices=(
            "xyzin", "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf"
        ),
    )
    trinity_optimize_gf.add_argument("--gf-hessian-file", type=Path)
    trinity_optimize_gf.add_argument("--gf-hessian-cfour-grd", type=Path)
    trinity_optimize_gf.add_argument("--gf-hessian-cfour-output", type=Path)
    trinity_optimize_gf.add_argument("--gf-out", type=Path)
    trinity_optimize_gf.add_argument("--gf-csv-dir", type=Path)
    trinity_optimize_gf.add_argument("--gf-symmetry-blocks", action="store_true")
    trinity_optimize_gf.add_argument("--gf-write-hessian-section", action="store_true")
    trinity_optimize_gf.add_argument("--no-gf-section", action="store_true")
    trinity_optimize_gf.add_argument("--timeout", type=float)

    trinity_benchmark_optimizer = trinity_sub.add_parser(
        "benchmark-optimizer",
        help="Run compact LINK optimizer validation benchmarks",
    )
    trinity_benchmark_optimizer.add_argument("--run-dir", type=Path, required=True)
    trinity_benchmark_optimizer.add_argument("--max-steps", type=int, default=20)
    trinity_benchmark_optimizer.add_argument("--include-sonic", action="store_true")

    trinity_optimize_chain = trinity_sub.add_parser(
        "optimize-chain",
        help="Run a multilevel geometry optimization chain from an RDKit SMILES geometry",
    )
    trinity_optimize_chain.add_argument("--smiles", required=True)
    trinity_optimize_chain.add_argument("--run-dir", type=Path, required=True)
    trinity_optimize_chain.add_argument(
        "--background",
        action="store_true",
        help="Detach the complete multilevel chain and return its PID",
    )
    trinity_optimize_chain.add_argument("--title", default="")
    trinity_optimize_chain.add_argument("--charge", type=int)
    trinity_optimize_chain.add_argument("--multiplicity", type=int)
    trinity_optimize_chain.add_argument("--random-seed", type=int, default=61453)
    trinity_optimize_chain.add_argument(
        "--level",
        action="append",
        default=[],
        help=(
            "JSON object defining one level, e.g. "
            '\'{"name":"hf-sto3g","backend":"gaussian","route":"#p HF/STO-3G Force NoSymm"}\''
        ),
    )
    trinity_optimize_chain.add_argument(
        "--level-file",
        type=Path,
        help="JSON file containing either a list of level objects or {'levels': [...]}",
    )
    trinity_ir = trinity_sub.add_parser(
        "ir-from-fchk",
        help=(
            "Fit an ARCHITECT dipole surface from FCHK derivatives and calculate "
            "TRINITY harmonic IR intensities"
        ),
    )
    trinity_ir.add_argument("fchk", type=Path)
    trinity_ir.add_argument("--output", type=Path, required=True)
    trinity_ir.add_argument(
        "--surface-output",
        type=Path,
        help="also write the standalone fitted ARCHITECT property surface",
    )
    trinity_ir.add_argument("--displacement", type=float, default=0.02)
    trinity_ir.add_argument("--symmetry-tolerance", type=float, default=1.0e-10)
    trinity_report = trinity_sub.add_parser(
        "report",
        help="Write human-readable and JSON reports from a multilevel optimization manifest",
    )
    trinity_report.add_argument("manifest", type=Path)
    trinity_report.add_argument("--out", type=Path, default=Path("optimization_report.txt"))
    trinity_report.add_argument("--json-out", type=Path, default=Path("optimization_report.json"))
    trinity_report.add_argument("--title", default="")
    trinity_report.add_argument("--charge", type=int)
    trinity_report.add_argument("--multiplicity", type=int)
    trinity_report.add_argument("--initial-geometry-source", default="")

    reference_search = sub.add_parser(
        "multistructure-reference-search",
        help="Search the local semiexperimental geometry reference library",
    )
    reference_search.add_argument("--query-xyz", type=Path, required=True)
    reference_search.add_argument("--library-root", type=Path)
    reference_search.add_argument("--outdir", type=Path, required=True)
    reference_search.add_argument("--top-k", type=int, default=10)
    reference_search.add_argument("--ring-weight", type=float, default=0.25)
    reference_search.add_argument("--no-ring-comparison", action="store_true")

    reference_build = sub.add_parser(
        "multistructure-build-reference-geometry",
        help="Build a reference-assisted geometry from local semiexperimental fragments",
    )
    reference_build.add_argument("--query-xyz", type=Path, required=True)
    reference_build.add_argument("--library-root", type=Path)
    reference_build.add_argument("--outdir", type=Path, required=True)
    reference_build.add_argument("--top-library-matches", type=int, default=25)
    reference_build.add_argument("--max-fragment-matches", type=int, default=8)
    reference_build.add_argument("--min-fragment-support", type=int, default=1)
    reference_build.add_argument("--zeff-threshold", type=float, default=0.08)
    reference_build.add_argument("--apply-kind", action="append", default=[])
    reference_build.add_argument("--ring-weight", type=float, default=0.25)
    reference_build.add_argument("--no-ring-comparison", action="store_true")

    gicforge = sub.add_parser(
        "gicforge",
        aliases=("smith", "SMITH"),
        help="Plan or build SMITH SONIC coordinate sections",
    )
    gicforge_sub = gicforge.add_subparsers(dest="gicforge_command")
    gic_plan = gicforge_sub.add_parser("plan", help="Write planned #GIC/#SYCART sections")
    gic_plan.add_argument("xyzin", type=Path)
    gic_plan.add_argument("--symmetrize", action="store_true")
    gic_plan.add_argument("--sycart", action="store_true")
    gic_plan.add_argument(
        "--fragment-mode",
        choices=("special-coordinates", "pseudo-bonds", "none"),
        default="special-coordinates",
        help="How SMITH should handle prebuilt molecular fragments",
    )
    _add_xh_stretch_arguments(gic_plan, default_policy=None)
    gic_build = gicforge_sub.add_parser("build", help="Build frozen #GIC/#SYCART sections")
    gic_build.add_argument("xyzin", type=Path)
    gic_build.add_argument("--symmetrize", action="store_true")
    gic_build.add_argument("--sycart", action="store_true")
    gic_build.add_argument(
        "--symmetry-group", help="Use a reduced point group such as C1, Cs, Ci or C2"
    )
    gic_build.add_argument(
        "--local-salc",
        action="store_true",
        help="Build deterministic center/ring local-pseudosymmetry SALCs (A1 first)",
    )
    gic_build.add_argument("--local-zeff-tolerance", type=float, default=5.0e-4)
    gic_build.add_argument("--local-distance-tolerance", type=float, default=1.0e-3)
    gic_build.add_argument("--local-template-rms-threshold", type=float, default=0.12)
    gic_build.add_argument("--local-template-margin", type=float, default=0.02)
    gic_build.add_argument("--local-angle-class-tolerance", type=float, default=0.02)
    gic_build.add_argument(
        "--fragment-mode",
        choices=("special-coordinates", "pseudo-bonds", "none"),
        help="Override the planned fragment handling mode",
    )
    _add_xh_stretch_arguments(gic_build, default_policy=None)
    gic_build.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not write the default human-readable SONIC report and Cartesian motions",
    )
    gic_build.add_argument("--diagnostics-dir", type=Path)
    standalone = gicforge_sub.add_parser(
        "standalone",
        help="Build from one SMITH extended-XYZ input containing geometry and directives",
    )
    standalone.add_argument("input", type=Path)
    standalone.add_argument("output", type=Path, nargs="?")
    standalone.add_argument("--no-diagnostics", action="store_true")
    standalone.add_argument("--diagnostics-dir", type=Path)
    bmatrix = gicforge_sub.add_parser("bmatrix", help="Evaluate the frozen GIC B matrix")
    bmatrix.add_argument("xyzin", type=Path)
    bmatrix.add_argument("output", type=Path, nargs="?")
    report = gicforge_sub.add_parser("report", help="Write a readable frozen-GIC report")
    report.add_argument("xyzin", type=Path)
    report.add_argument("output", type=Path, nargs="?")
    motions = gicforge_sub.add_parser(
        "motions",
        help="Write per-SONIC -Delta/reference/+Delta Cartesian trajectories and vectors",
    )
    motions.add_argument("xyzin", type=Path)
    motions.add_argument("output_directory", type=Path, nargs="?")
    motions.add_argument("--distance-step", type=float, default=0.05, metavar="ANGSTROM")
    motions.add_argument(
        "--angle-step-degrees",
        type=float,
        default=5.0,
        metavar="DEGREES",
    )
    motions.add_argument(
        "--ring-step-degrees",
        type=float,
        default=math.degrees(0.10),
        metavar="DEGREES",
    )
    motions.add_argument(
        "--metric",
        choices=("euclidean", "mass-weighted"),
        default="euclidean",
        help="Cartesian minimum-norm metric used for the frozen-B right inverse",
    )
    motions.add_argument(
        "--mass-source",
        choices=("auto", "average", "default-isotope", "hessian", "isotopologue"),
        default="auto",
        help="Mass provenance for the mass-weighted right inverse",
    )
    motions.add_argument(
        "--isotopologue-label",
        help="Label in #ISOTOPOLOGUES (used with isotopologue or auto mass source)",
    )
    motions.add_argument(
        "--max-atom-displacement",
        type=float,
        default=0.10,
        metavar="ANGSTROM",
        help="Safety cap for the generated mathematical displacement",
    )
    motions.add_argument(
        "--topology-workers",
        type=_topology_worker_argument,
        default="auto",
        metavar="AUTO|N",
        help=(
            "Processes for independent ORACLE topology checks; auto adapts to the task, "
            "0 uses all available processors (default: auto)"
        ),
    )
    motions.add_argument(
        "--no-topology-cache",
        action="store_true",
        help="Ignore and do not update the restartable ORACLE-perception cache",
    )
    view_motions = gicforge_sub.add_parser(
        "view",
        help="Open the interactive SONIC and normal-mode motion viewer",
    )
    view_motions.add_argument("xyzin", type=Path)
    view_motions.add_argument(
        "--source",
        choices=("sonic", "normal-cartesian", "normal-sonic"),
        default="sonic",
    )
    view_motions.add_argument("--sonic-hessian", type=Path)
    salc_snapshot = gicforge_sub.add_parser(
        "salc-snapshot",
        help="Write a compact SALC coefficient snapshot for a frozen #GIC section",
    )
    salc_snapshot.add_argument("xyzin", type=Path)
    salc_snapshot.add_argument("output", type=Path)
    corpus = gicforge_sub.add_parser(
        "corpus",
        help="List or summarize the demanding GIC regression corpus",
    )
    corpus.add_argument("--root", type=Path, help="Override the GIC corpus root directory")
    corpus.add_argument(
        "--suffix",
        action="append",
        help="Filter by suffix, for example .inp or fchk",
    )
    corpus.add_argument("--limit", type=int, help="Limit listed records")
    corpus.add_argument(
        "--format",
        choices=("summary", "paths", "json"),
        default="summary",
        help="Output format",
    )
    corpus_audit = gicforge_sub.add_parser(
        "corpus-audit",
        help="Audit geometry imports for the GIC regression corpus",
    )
    corpus_audit.add_argument("--root", type=Path, help="Override the GIC corpus root directory")
    corpus_audit.add_argument(
        "--suffix",
        action="append",
        help="Filter by suffix; defaults to geometry inputs",
    )
    corpus_audit.add_argument("--limit", type=int, help="Limit audited or listed records")
    corpus_audit.add_argument(
        "--format",
        choices=("summary", "failures", "json"),
        default="summary",
        help="Output format",
    )
    corpus_audit.add_argument(
        "--status",
        choices=("all", "pass", "fail"),
        default="all",
        help="Entry status filter for JSON output",
    )
    fortran_audit = gicforge_sub.add_parser(
        "fortran-audit",
        help="Compare MATRIX/SMITH GIC/B rows against the vendored Merlino Fortran backend",
    )
    fortran_audit.add_argument("--root", type=Path, help="Override the GIC corpus root directory")
    fortran_audit.add_argument("--workdir", type=Path, help="Keep audit work directories here")
    fortran_audit.add_argument(
        "--molecule",
        action="append",
        help="Corpus-relative molecule path to audit; repeatable",
    )
    fortran_audit.add_argument("--limit", type=int)
    fortran_audit.add_argument("--tolerance", type=float, default=2.0e-8)
    fortran_audit.add_argument(
        "--format",
        choices=("summary", "cases", "failures", "json"),
        default="summary",
    )
    gaussian_input = gicforge_sub.add_parser(
        "gaussian-input",
        help="Write Gaussian input from validated #GIC state",
    )
    gaussian_input.add_argument("xyzin", type=Path)
    gaussian_input.add_argument("output", type=Path)
    gaussian_input.add_argument("--route", default="#p hf/sto-3g opt=readallgic")
    gaussian_input.add_argument("--title")
    gaussian_input.add_argument("--charge", type=int)
    gaussian_input.add_argument("--multiplicity", type=int)
    gaussian_input.add_argument(
        "--g16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the commercial Gaussian 16 compatibility export profile (default: enabled)",
    )

    return parser


def main(
    argv: list[str] | None = None,
    *,
    repo_root: Path | None = None,
    prog: str = "matrix",
) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args and command_args[0] == "oracle":
        from matrix_oracle.cli import matrix_main as oracle_matrix_main

        return oracle_matrix_main(command_args[1:])
    if command_args and command_args[0] == "apoc":
        from matrix_apoc.cli import matrix_main as apoc_matrix_main

        return apoc_matrix_main(command_args[1:])
    root = find_repo_root() if repo_root is None else Path(repo_root)
    add_repo_packages_to_path(root)
    parser = build_parser(repo_root=root, prog=prog)
    args = parser.parse_args(command_args)
    if (
        args.command == "trinity"
        and args.trinity_command in {"optimize-run", "optimize-chain"}
        and args.background
    ):
        child_argv = list(sys.argv[1:] if argv is None else argv)
        child_argv.remove("--background")
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "matrix_background.log"
        status_path = run_dir / "matrix_background.json"
        env = os.environ.copy()
        package_paths = [str(path) for path in sorted((root / "packages").glob("*/src"))]
        env["PYTHONPATH"] = os.pathsep.join(
            package_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        command = [
            sys.executable,
            "-c",
            "from matrix_core.cli import matrix_main; raise SystemExit(matrix_main())",
            *child_argv,
        ]
        with log_path.open("ab") as stream:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        status_path.write_text(
            json.dumps(
                {
                    "schema": "matrix.background_job.v1",
                    "pid": process.pid,
                    "command": command,
                    "log": str(log_path),
                    "run_dir": str(run_dir),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"background_pid: {process.pid}")
        print(f"background_log: {log_path}")
        print(f"background_status: {status_path}")
        return 0
    if args.command == "init":
        from matrix_core.workspace import ensure_workspace

        ensure_workspace(args.workdir)
        print(f"Created MATRIX workspace: {args.workdir}")
        return 0
    if args.command == "provenance":
        from matrix_core.provenance import ProvenanceLedger, provenance_path

        project = args.project.expanduser().resolve()
        ledger_path = provenance_path(project)
        if not ledger_path.is_file():
            parser.error(f"provenance ledger not found: {ledger_path}")
        verification = ProvenanceLedger(project).verify()
        payload = {
            "valid": verification.valid,
            "event_count": verification.event_count,
            "last_hash": verification.last_hash,
            "errors": list(verification.errors),
            "ledger": str(ledger_path),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif verification.valid:
            print(
                f"Valid The ONE provenance: {verification.event_count} events; "
                f"last hash {verification.last_hash}"
            )
        else:
            print("Invalid The ONE provenance:")
            for error in verification.errors:
                print(f"- {error}")
        return 0 if verification.valid else 2
    if args.command == "oracle":
        from matrix_oracle.cli import matrix_main as oracle_matrix_main

        return oracle_matrix_main(args.oracle_args)
    if args.command == "apoc":
        from matrix_apoc.cli import matrix_main as apoc_matrix_main

        return apoc_matrix_main(args.apoc_args)
    if args.command == "gui":
        from matrix_gui.app import run as run_matrix_gui

        gui_argv: list[str] = []
        if args.tool:
            gui_argv.append(args.tool)
        if args.xyzin is not None:
            gui_argv.append(str(args.xyzin))
        if args.workdir is not None:
            gui_argv.extend(("--workdir", str(args.workdir)))
        if args.the_one:
            gui_argv.append("--the-one")
        if args.smoke_test:
            gui_argv.append("--smoke-test")
        return run_matrix_gui(gui_argv)
    if args.command == "validate":
        from matrix_chem import write_validation_section

        result = write_validation_section(args.xyzin, require_fragments=args.require_fragments)
        print(f"Validated MATRIX molecule: {args.xyzin} ({result.status})")
        return 0
    if args.command == "topology" and args.topology_command == "report":
        from matrix_chem import topology_report_lines, write_topology_report

        if args.output is None:
            print("\n".join(topology_report_lines(args.xyzin)))
            return 0
        output = write_topology_report(args.xyzin, args.output)
        print(f"Wrote MATRIX topology report: {output}")
        return 0
    if args.command == "topology" and args.topology_command == "snapshot":
        from matrix_chem import write_topology_snapshot

        output = write_topology_snapshot(args.output, args.xyzin)
        print(f"Wrote MATRIX topology snapshot: {output}")
        return 0
    if args.command == "contracts":
        from matrix_core import (
            PLANNED_FRAMEWORK_EXPANSION,
            PLANNED_FRAMEWORK_NAME,
            tool_contract,
            tool_contract_lines,
            tool_contract_markdown_table,
            tool_contract_readinesses,
            tool_contracts,
            tool_contracts_json,
            tool_readiness_json,
            tool_readiness_lines,
            tool_readiness_markdown_table,
        )

        if args.framework:
            if args.format == "json":
                print(
                    json.dumps(
                        {
                            "planned_name": PLANNED_FRAMEWORK_NAME,
                            "expanded_name": PLANNED_FRAMEWORK_EXPANSION,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.format == "markdown":
                print("| Planned name | Expanded name |")
                print("| --- | --- |")
                print(f"| {PLANNED_FRAMEWORK_NAME} | {PLANNED_FRAMEWORK_EXPANSION} |")
            else:
                print(f"framework: {PLANNED_FRAMEWORK_NAME}")
                print(f"  expanded_name: {PLANNED_FRAMEWORK_EXPANSION}")
            return 0

        rows = (
            (tool_contract(args.tool),)
            if args.tool
            else tool_contracts(include_gui=not args.no_gui)
        )
        if args.check_xyzin is not None:
            readinesses = tool_contract_readinesses(args.check_xyzin, rows)
            if args.format == "json":
                print(tool_readiness_json(readinesses))
            elif args.format == "markdown":
                print(tool_readiness_markdown_table(readinesses))
            else:
                print("\n".join(tool_readiness_lines(readinesses)))
            return 0 if all(readiness.ready for readiness in readinesses) else 2
        if args.format == "json":
            print(tool_contracts_json(rows))
        elif args.format == "markdown":
            print(tool_contract_markdown_table(rows))
        else:
            print("\n".join(tool_contract_lines(rows)))
        return 0
    if args.command in {"help", "manuals"}:
        from matrix_core import online_help_json, online_help_lines, online_help_markdown

        if args.format == "json":
            print(
                online_help_json(
                    args.tool,
                    xyzin=args.xyzin,
                    include_gui=not args.no_gui,
                )
            )
        elif args.format == "markdown":
            print(
                online_help_markdown(
                    args.tool,
                    xyzin=args.xyzin,
                    include_gui=not args.no_gui,
                )
            )
        else:
            print(
                "\n".join(
                    online_help_lines(
                        args.tool,
                        xyzin=args.xyzin,
                        include_gui=not args.no_gui,
                    )
                )
            )
        return 0
    if args.command == "properties" and args.properties_command == "summary":
        from matrix_qm import (
            filtered_property_records,
            properties_summary_lines,
            property_record_to_dict,
            read_properties_section,
        )

        section = read_properties_section(args.xyzin)
        if args.format == "json":
            records = filtered_property_records(section, name=args.name, atom=args.atom)
            print(
                json.dumps(
                    {
                        "xyzin": str(args.xyzin),
                        "schema": section.schema,
                        "count": len(records),
                        "records": [property_record_to_dict(record) for record in records],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                "\n".join(
                    properties_summary_lines(
                        section,
                        name=args.name,
                        atom=args.atom,
                    )
                )
            )
        return 0
    if args.command == "properties" and args.properties_command == "compare":
        from matrix_qm import (
            compare_property_records,
            filtered_property_records,
            property_comparison_lines,
            property_comparison_to_dict,
            read_properties_section,
        )

        reference_section = read_properties_section(args.reference)
        reference_records = filtered_property_records(
            reference_section,
            name=args.name,
            atom=args.atom,
        )
        if args.index < 0 or args.index >= len(reference_records):
            print(
                f"No reference property at index {args.index} after filtering {args.reference}",
                file=sys.stderr,
            )
            return 2
        reference = reference_records[args.index]
        comparisons = []
        for candidate_path in args.candidates:
            candidate_section = read_properties_section(candidate_path)
            candidate_records = filtered_property_records(
                candidate_section,
                name=args.name,
                atom=args.atom,
            )
            if args.index < 0 or args.index >= len(candidate_records):
                print(
                    f"No candidate property at index {args.index} after filtering {candidate_path}",
                    file=sys.stderr,
                )
                return 2
            comparisons.append(
                compare_property_records(
                    reference,
                    candidate_records[args.index],
                    reference_label=str(args.reference),
                    candidate_label=str(candidate_path),
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
        comparison_tuple = tuple(comparisons)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "reference": str(args.reference),
                        "count": len(comparison_tuple),
                        "comparisons": [
                            property_comparison_to_dict(comparison)
                            for comparison in comparison_tuple
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print("\n".join(property_comparison_lines(comparison_tuple)))
        return 0 if all(comparison.compatible for comparison in comparison_tuple) else 1
    if args.command == "qm" and args.qm_command == "remote-submit":
        from matrix_engines import remote_qm_submit

        result = remote_qm_submit(
            args.input,
            engine=args.engine,
            host=args.host,
            remote_root=args.remote_root,
            extra_args=tuple(args.extra_arg),
            ssh_executable=args.ssh,
            scp_executable=args.scp,
        )
        print(f"host: {result.host}")
        print(f"engine: {result.engine}")
        print(f"job: {result.job}")
        print(f"remote_input: {result.remote_input}")
        print(f"workdir: {result.workdir}")
        print(f"log: {result.log}")
        if result.native_output:
            print(f"native_output: {result.native_output}")
        return 0
    if args.command == "qm" and args.qm_command == "remote-status":
        from matrix_engines import remote_qm_status

        print(
            remote_qm_status(
                host=args.host,
                remote_root=args.remote_root,
                ssh_executable=args.ssh,
            ),
            end="",
        )
        return 0
    if args.command == "qm" and args.qm_command == "remote-run-monitor":
        from matrix_engines import (
            RemoteQMError,
            remote_qm_fetch,
            remote_qm_job_status,
            remote_qm_output_completed_normally,
            remote_qm_submit,
        )

        if args.poll_seconds <= 0:
            raise ValueError("--poll-seconds must be positive")
        result = remote_qm_submit(
            args.input,
            engine=args.engine,
            host=args.host,
            remote_root=args.remote_root,
            extra_args=tuple(args.extra_arg),
            ssh_executable=args.ssh,
            scp_executable=args.scp,
        )
        print(f"submitted: {result.job}", flush=True)
        print(f"host: {result.host}", flush=True)
        print(f"engine: {result.engine}", flush=True)
        print(f"remote_output: {result.native_output}", flush=True)
        started = time.monotonic()
        last_state = ""
        while True:
            if args.max_wait_seconds > 0 and time.monotonic() - started > args.max_wait_seconds:
                raise RemoteQMError(
                    f"monitor timeout for {result.job}; the remote background job was not cancelled"
                )
            try:
                status = remote_qm_job_status(
                    result.job,
                    host=args.host,
                    remote_root=args.remote_root,
                    ssh_executable=args.ssh,
                )
            except RemoteQMError as exc:
                print(f"status unavailable; will retry: {exc}", flush=True)
                time.sleep(args.poll_seconds)
                continue
            if status.state != last_state:
                print(f"status: {status.state} pid={status.pid}", flush=True)
                last_state = status.state
            if status.state == "FINISHED":
                break
            time.sleep(args.poll_seconds)
        fetched = remote_qm_fetch(
            result.job,
            host=args.host,
            destination=args.dest,
            remote_root=args.remote_root,
            promote=args.promote,
            xyzin=args.xyzin,
            ssh_executable=args.ssh,
            scp_executable=args.scp,
        )
        print(f"fetched: {fetched.output_path}", flush=True)
        if not remote_qm_output_completed_normally(fetched.output_path, result.engine):
            print(
                f"error: {result.engine} ended without its normal-termination marker; "
                f"inspect {fetched.output_path}",
                flush=True,
            )
            return 2
        print(f"completed normally: {result.job}", flush=True)
        return 0
    if args.command == "qm" and args.qm_command == "remote-fetch":
        from matrix_engines import remote_fetch_cli_hint, remote_qm_fetch

        result = remote_qm_fetch(
            args.job,
            host=args.host,
            destination=args.dest,
            remote_root=args.remote_root,
            promote=args.promote,
            xyzin=args.xyzin,
            ssh_executable=args.ssh,
            scp_executable=args.scp,
        )
        print(f"host: {result.host}")
        print(f"job: {result.job}")
        print(f"engine: {result.engine}")
        print(f"destination: {result.destination}")
        print(f"native_output: {result.output_path}")
        print(f"metadata: {result.metadata_path}")
        print(f"manifest: {result.manifest_path}")
        for role, path in sorted(result.artifacts.items()):
            print(f"artifact_{role}: {path}")
        if result.promotion is not None:
            print(f"promotion_mode: {result.promotion.mode}")
            print(f"promotion_status: {result.promotion.status}")
            print(f"promotion_message: {result.promotion.message}")
        print(f"message: {remote_fetch_cli_hint(result)}")
        return 0
    if _is_link_command(args):
        from matrix_chem import SymmetryThresholds, preprocess_to_enriched_xyz

        result = preprocess_to_enriched_xyz(
            args.source,
            args.output,
            source_kind=args.source_kind,
            symmetry_thresholds=SymmetryThresholds(
                distance_angstrom=args.symmetry_distance,
                inertia_relative=args.symmetry_inertia,
                max_rotation_order=args.max_rotation_order,
            ),
        )
        if args.validate:
            from matrix_chem import write_validation_section

            validation = write_validation_section(args.output)
            print(f"Validated MATRIX molecule: {args.output} ({validation.status})")
        print(
            "Preprocessed ORACLE-import molecule: "
            f"{result.path} ({result.geometry.natoms} atoms, "
            f"PG={result.point_group}, bonds={result.topology_bond_count}, "
            f"rings={result.ring_count})"
        )
        return 0
    if args.command == "gaussian" and args.gaussian_command == "summary":
        from matrix_gaussian import summarize_gaussian_log

        summary = summarize_gaussian_log(args.log)
        print(f"path: {summary.path}")
        print(f"normal_termination: {int(summary.normal_termination)}")
        print(f"scf_count: {len(summary.scf_energies_hartree)}")
        if summary.scf_energies_hartree:
            print(f"last_scf_hartree: {summary.scf_energies_hartree[-1]:.12g}")
        print(f"standard_orientations: {summary.standard_orientation_count}")
        print(f"input_orientations: {summary.input_orientation_count}")
        print(f"frequencies: {len(summary.frequencies_cm)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "status":
        from matrix_gaussian import gaussian_job_status

        status = gaussian_job_status(args.workdir)
        print(f"status: {status.status}")
        print(f"workdir: {status.workdir}")
        print(f"log: {status.log_path}")
        if status.input_path is not None:
            print(f"input: {status.input_path}")
        if status.pid is not None:
            print(f"pid: {status.pid}")
        print(f"normal_termination: {int(status.normal_termination)}")
        print(f"error_termination: {int(status.error_termination)}")
        print(f"message: {status.message}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "run":
        from matrix_gaussian import run_gaussian_job

        result = run_gaussian_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            background=args.background,
            timeout=args.timeout,
        )
        print(f"gaussian_input: {result.input_path}")
        print(f"gaussian_log: {result.log_path}")
        if result.pid is not None:
            print(f"pid: {result.pid}")
        if result.exit_code is not None:
            print(f"exit_code: {result.exit_code}")
        if result.success is not None:
            print(f"success: {int(result.success)}")
        print(f"message: {result.message}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "formchk":
        from matrix_gaussian import FORMCHK_EXECUTABLE, formchk_checkpoint

        output = formchk_checkpoint(
            args.chk,
            args.fchk,
            executable=args.executable or FORMCHK_EXECUTABLE,
            timeout=args.timeout,
        )
        print(f"Wrote formatted checkpoint: {output}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "fchk-summary":
        from matrix_gaussian import read_gaussian_fchk_qff

        data = read_gaussian_fchk_qff(args.fchk)
        print(f"path: {args.fchk}")
        print(f"atoms: {len(data.atomic_numbers)}")
        print(f"hessian_lower: {len(data.cartesian_hessian_lower)}")
        print(f"harmonic_frequencies: {len(data.harmonic_frequencies_cm)}")
        print(f"anharmonic_frequencies: {len(data.anharmonic_frequencies_cm)}")
        print(f"anharmonic_e2_values: {len(data.anharmonic_e2)}")
        print(f"normal_mode_values: {len(data.normal_modes)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "semidiagonal-input":
        from matrix_gaussian import write_semidiagonal_cubic_vibrot_gaussian_input

        output = write_semidiagonal_cubic_vibrot_gaussian_input(
            args.output,
            oldchk=args.oldchk,
            route=args.route,
            harmonic_chk=args.harmonic_chk,
            cubic_chk=args.cubic_chk,
            symmetry=args.symmetry,
            xyzin=args.xyzin,
            manifest_path=args.manifest,
            write_manifest=not args.no_manifest,
        )
        print(f"Wrote Gaussian semidiagonal vibrot input: {output}")
        if not args.no_manifest:
            manifest = args.manifest or output.with_suffix(output.suffix + ".manifest.json")
            print(f"manifest: {manifest}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "semidiagonal-summary":
        from matrix_gaussian import compute_deltabvib_from_semidiagonal_cubic_data
        from matrix_gaussian import read_gaussian_semidiagonal_cubic_rovib
        from matrix_gaussian import semidiagonal_cubic_cm

        data = read_gaussian_semidiagonal_cubic_rovib(args.log)
        delta = compute_deltabvib_from_semidiagonal_cubic_data(data)
        cubic_count = len(semidiagonal_cubic_cm(data))
        summary = _semidiagonal_summary_dict(args.log, data, delta, cubic_count)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            _print_semidiagonal_summary(summary)
        return 0
    if args.command == "gaussian" and args.gaussian_command == "semidiagonal-promote":
        from matrix_gaussian import compute_deltabvib_from_semidiagonal_cubic_data
        from matrix_gaussian import promote_gaussian_rovib_to_xyzin
        from matrix_gaussian import read_gaussian_semidiagonal_cubic_rovib
        from matrix_gaussian import semidiagonal_cubic_cm

        data = read_gaussian_semidiagonal_cubic_rovib(args.log)
        delta = compute_deltabvib_from_semidiagonal_cubic_data(data)
        cubic_count = len(semidiagonal_cubic_cm(data))
        promotion = promote_gaussian_rovib_to_xyzin(args.log, args.xyzin)
        summary = _semidiagonal_summary_dict(args.log, data, delta, cubic_count)
        summary.update(
            {
                "xyzin": str(args.xyzin),
                "wrote_vibrational": promotion.wrote_vibrational,
                "wrote_rotational": promotion.wrote_rotational,
                "wrote_deltabvib": promotion.wrote_deltabvib,
                "wrote_semidiagonal_cubic": promotion.wrote_semidiagonal_cubic,
            }
        )
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            _print_semidiagonal_summary(summary)
            print(f"xyzin: {args.xyzin}")
            print(f"wrote_vibrational: {int(promotion.wrote_vibrational)}")
            print(f"wrote_rotational: {int(promotion.wrote_rotational)}")
            print(f"wrote_deltabvib: {int(promotion.wrote_deltabvib)}")
            print(f"wrote_semidiagonal_cubic: {int(promotion.wrote_semidiagonal_cubic)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "promote-fchk":
        from matrix_gaussian import promote_gaussian_fchk_to_xyzin

        result = promote_gaussian_fchk_to_xyzin(
            args.fchk,
            args.xyzin,
            write_cartesian_hessian=not args.no_cartesian_hessian,
            write_normal_modes=not args.no_normal_modes,
            write_qff=not args.no_qff,
            write_electronic=not args.no_electronic,
            write_orbitals=not args.no_orbitals,
        )
        print(f"Promoted Gaussian FCHK data: {result.fchk_path} -> {result.xyzin}")
        print(f"wrote_cartesian_hessian: {int(result.wrote_cartesian_hessian)}")
        print(f"wrote_normal_modes: {int(result.wrote_normal_modes)}")
        print(f"wrote_qff: {int(result.wrote_qff)}")
        print(f"wrote_electronic: {int(result.wrote_electronic)}")
        print(f"wrote_orbitals: {int(result.wrote_orbitals)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "promote-log-hessian":
        from matrix_gaussian import promote_gaussian_log_hessian_to_xyzin

        result = promote_gaussian_log_hessian_to_xyzin(
            args.log,
            args.xyzin,
            write_normal_modes=not args.no_normal_modes,
        )
        print(f"Promoted Gaussian log Hessian: {result.log_path} -> {result.xyzin}")
        print(f"wrote_cartesian_hessian: {int(result.wrote_cartesian_hessian)}")
        print(f"wrote_normal_modes: {int(result.wrote_normal_modes)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "promote-electronic":
        from matrix_gaussian import promote_gaussian_electronic_log_to_xyzin

        result = promote_gaussian_electronic_log_to_xyzin(
            args.log,
            args.xyzin,
            write_electronic=not args.no_electronic,
            write_transitions=not args.no_transitions,
            orbital_files=tuple(args.orbital_file),
        )
        print(f"Promoted Gaussian electronic data: {result.log_path} -> {result.xyzin}")
        print(f"wrote_electronic: {int(result.wrote_electronic)}")
        print(f"wrote_transitions: {int(result.wrote_transitions)}")
        print(f"wrote_orbitals: {int(result.wrote_orbitals)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "promote-rovib":
        from matrix_gaussian import promote_gaussian_rovib_to_xyzin

        result = promote_gaussian_rovib_to_xyzin(
            args.log,
            args.xyzin,
            write_vibrational=not args.no_vibrational,
            write_rotational=not args.no_rotational,
            write_deltabvib=not args.no_deltabvib,
            write_semidiagonal_cubic=not args.no_semidiagonal_cubic,
            invert_imaginary_modes=not args.no_invert_imaginary,
            exclude_modes=tuple(args.exclude_mode),
        )
        print(f"Promoted Gaussian rovib data: {result.log_path} -> {result.xyzin}")
        print(f"wrote_vibrational: {int(result.wrote_vibrational)}")
        print(f"wrote_rotational: {int(result.wrote_rotational)}")
        print(f"wrote_deltabvib: {int(result.wrote_deltabvib)}")
        print(f"wrote_semidiagonal_cubic: {int(result.wrote_semidiagonal_cubic)}")
        return 0
    if args.command == "gaussian" and args.gaussian_command == "promote-quadrupole":
        from matrix_gaussian import promote_gaussian_quadrupole_properties_to_xyzin

        result = promote_gaussian_quadrupole_properties_to_xyzin(args.log, args.xyzin)
        print(f"Promoted Gaussian quadrupole properties: {result.log_path} -> {result.xyzin}")
        print(f"wrote_properties: {int(result.wrote_properties)}")
        print(f"property_count: {result.property_count}")
        return 0
    if args.command == "molpro" and args.molpro_command == "status":
        from matrix_molpro import molpro_job_status

        status = molpro_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "molpro" and args.molpro_command == "run":
        from matrix_molpro import run_molpro_job

        result = run_molpro_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "molpro" and args.molpro_command == "summary":
        from matrix_molpro import summarize_molpro_output

        summary = summarize_molpro_output(args.output)
        print(f"path: {summary.path}")
        print(f"atoms: {summary.geometry.natoms}")
        print(f"charge: {summary.charge}")
        print(f"multiplicity: {summary.multiplicity}")
        print(f"atomic_coordinate_blocks: {summary.atomic_coordinate_blocks}")
        return 0
    if args.command == "molpro" and args.molpro_command == "promote":
        from matrix_molpro import promote_molpro_output_to_xyzin

        result = promote_molpro_output_to_xyzin(
            args.output,
            args.xyzin,
            symmetry_distance=args.symmetry_distance,
            symmetry_inertia=args.symmetry_inertia,
            max_rotation_order=args.max_rotation_order,
        )
        print(
            "Promoted Molpro output: "
            f"{args.output} -> {result.path} ({result.geometry.natoms} atoms, "
            f"PG={result.point_group}, bonds={result.topology_bond_count}, "
            f"rings={result.ring_count})"
        )
        return 0
    if args.command == "molpro" and args.molpro_command == "promote-quadrupole":
        from matrix_molpro import promote_molpro_quadrupole_properties_to_xyzin

        result = promote_molpro_quadrupole_properties_to_xyzin(
            args.output,
            args.xyzin,
            atom=args.atom,
            isotope=args.isotope or "",
        )
        print(f"Promoted Molpro quadrupole properties: {result.output_path} -> {result.xyzin}")
        print(f"wrote_properties: {int(result.wrote_properties)}")
        print(f"property_count: {result.property_count}")
        return 0
    if args.command == "molpro" and args.molpro_command == "molden":
        from matrix_molpro import promote_molpro_molden_to_xyzin

        result = promote_molpro_molden_to_xyzin(
            args.output,
            args.xyzin,
            molden=args.molden,
        )
        print(f"Registered Molpro Molden file: {result.molden_path} -> {result.xyzin}")
        print(f"wrote_orbitals: {int(result.wrote_orbitals)}")
        return 0
    if args.command == "orca" and args.orca_command == "status":
        from matrix_orca import orca_job_status

        status = orca_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "orca" and args.orca_command == "run":
        from matrix_orca import run_orca_job

        result = run_orca_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "orca" and args.orca_command == "summary":
        from matrix_orca import summarize_orca_output

        summary = summarize_orca_output(args.output)
        print(f"path: {summary.path}")
        print(f"atoms: {summary.geometry.natoms}")
        print(f"charge: {summary.charge}")
        print(f"multiplicity: {summary.multiplicity}")
        if summary.final_energy_hartree is not None:
            print(f"final_energy_hartree: {summary.final_energy_hartree:.12g}")
        print(f"frequencies: {len(summary.frequencies_cm)}")
        print(f"cartesian_hessian: {int(summary.cartesian_hessian is not None)}")
        print(f"cartesian_coordinate_blocks: {summary.cartesian_coordinate_blocks}")
        print(f"normal_termination: {int(summary.normal_termination)}")
        return 0
    if args.command == "orca" and args.orca_command == "promote":
        from matrix_orca import promote_orca_output_to_xyzin

        result = promote_orca_output_to_xyzin(
            args.output,
            args.xyzin,
            symmetry_distance=args.symmetry_distance,
            symmetry_inertia=args.symmetry_inertia,
            max_rotation_order=args.max_rotation_order,
        )
        print(f"Promoted ORCA output: {result.output_path} -> {result.xyzin}")
        print(f"wrote_geometry: {int(result.wrote_geometry)}")
        print(f"wrote_cartesian_hessian: {int(result.wrote_cartesian_hessian)}")
        return 0
    if args.command == "orca" and args.orca_command == "promote-quadrupole":
        from matrix_orca import promote_orca_quadrupole_properties_to_xyzin

        result = promote_orca_quadrupole_properties_to_xyzin(args.output, args.xyzin)
        print(f"Promoted ORCA quadrupole properties: {result.output_path} -> {result.xyzin}")
        print(f"wrote_properties: {int(result.wrote_properties)}")
        print(f"property_count: {result.property_count}")
        return 0
    if args.command == "orca" and args.orca_command == "molden":
        from matrix_orca import promote_orca_molden_to_xyzin

        result = promote_orca_molden_to_xyzin(
            args.gbw,
            args.xyzin,
            output=args.output,
            executable=args.executable,
            timeout=args.timeout,
        )
        print(f"Converted ORCA GBW to Molden: {result.gbw_path} -> {result.molden_path}")
        print(f"Registered ORCA Molden file: {result.molden_path} -> {result.xyzin}")
        print(f"wrote_orbitals: {int(result.wrote_orbitals)}")
        return 0
    if args.command == "mrcc" and args.mrcc_command == "status":
        from matrix_mrcc import mrcc_job_status

        status = mrcc_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "mrcc" and args.mrcc_command == "run":
        from matrix_mrcc import run_mrcc_job

        result = run_mrcc_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "mrcc" and args.mrcc_command == "summary":
        from matrix_mrcc import summarize_mrcc_output

        summary = summarize_mrcc_output(args.output)
        print(f"path: {summary.path}")
        print(f"atoms: {summary.geometry.natoms}")
        print(f"charge: {summary.charge}")
        print(f"multiplicity: {summary.multiplicity}")
        print(f"cartesian_coordinate_blocks: {summary.cartesian_coordinate_blocks}")
        print(f"frequencies: {len(summary.frequencies_cm)}")
        print(f"cartesian_hessian: {int(summary.cartesian_hessian is not None)}")
        return 0
    if args.command == "mrcc" and args.mrcc_command == "promote":
        from matrix_mrcc import promote_mrcc_output_to_xyzin

        result = promote_mrcc_output_to_xyzin(
            args.output,
            args.xyzin,
            symmetry_distance=args.symmetry_distance,
            symmetry_inertia=args.symmetry_inertia,
            max_rotation_order=args.max_rotation_order,
        )
        print(
            "Promoted MRCC output: "
            f"{args.output} -> {result.path} ({result.geometry.natoms} atoms, "
            f"PG={result.point_group}, bonds={result.topology_bond_count}, "
            f"rings={result.ring_count})"
        )
        return 0
    if args.command == "cfour" and args.cfour_command == "status":
        from matrix_cfour import cfour_job_status

        status = cfour_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "cfour" and args.cfour_command == "run":
        from matrix_cfour import run_cfour_job

        result = run_cfour_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "xtb" and args.xtb_command == "status":
        from matrix_xtb import xtb_job_status

        status = xtb_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "xtb" and args.xtb_command == "run":
        from matrix_xtb import run_xtb_job

        result = run_xtb_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "xtb" and args.xtb_command == "summary":
        from matrix_xtb import summarize_xtb_output

        summary = summarize_xtb_output(args.output, geometry=args.geometry)
        print(f"path: {summary.path}")
        print(f"atoms: {summary.geometry.natoms}")
        print(f"charge: {summary.charge}")
        print(f"multiplicity: {summary.multiplicity}")
        if summary.final_energy_hartree is not None:
            print(f"final_energy_hartree: {summary.final_energy_hartree:.12g}")
        if summary.gradient_norm_hartree_per_bohr is not None:
            print(f"gradient_norm_hartree_per_bohr: {summary.gradient_norm_hartree_per_bohr:.12g}")
        if summary.homo_lumo_gap_ev is not None:
            print(f"homo_lumo_gap_ev: {summary.homo_lumo_gap_ev:.12g}")
        print(f"normal_termination: {int(summary.normal_termination)}")
        return 0
    if args.command == "pyscf" and args.pyscf_command == "status":
        from matrix_pyscf import pyscf_job_status

        status = pyscf_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "pyscf" and args.pyscf_command == "run":
        from matrix_pyscf import run_pyscf_job

        result = run_pyscf_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "pyscf" and args.pyscf_command == "summary":
        from matrix_pyscf import summarize_pyscf_output

        summary = summarize_pyscf_output(args.output)
        print(f"path: {summary.path}")
        print(f"atoms: {summary.geometry.natoms if summary.geometry is not None else 0}")
        print(f"charge: {summary.charge}")
        print(f"multiplicity: {summary.multiplicity}")
        print(f"method: {summary.method}")
        print(f"basis: {summary.basis}")
        if summary.final_energy_hartree is not None:
            print(f"final_energy_hartree: {summary.final_energy_hartree:.12g}")
        print(f"gradient: {int(summary.gradient_hartree_per_bohr is not None)}")
        print(f"converged: {int(summary.converged)}")
        print(f"normal_termination: {int(summary.normal_termination)}")
        return 0
    if args.command == "et" and args.et_command == "status":
        from matrix_et import et_job_status

        status = et_job_status(
            args.workdir,
            input_path=args.input,
            output_path=args.output,
        )
        _print_external_qm_status(status)
        return 0
    if args.command == "et" and args.et_command == "run":
        from matrix_et import run_et_job

        result = run_et_job(
            args.workdir,
            executable=args.executable,
            input_path=args.input,
            output_path=args.output,
            background=args.background,
            timeout=args.timeout,
            extra_args=tuple(args.extra_arg),
        )
        _print_external_qm_run_result(result)
        return 0
    if args.command == "lcb25" and args.lcb25_command == "fetch":
        from matrix_link import sync_lcb25_library

        manifest = sync_lcb25_library(args.root, datasets=args.dataset, force=args.force)
        print(f"Synced LCB25 library: {manifest}")
        return 0
    if args.command == "fragments" and args.fragments_command == "plan":
        from matrix_fragments import write_fragment_plan_section

        write_fragment_plan_section(args.xyzin)
        print(f"Planned MATRIX fragment workflow: {args.xyzin}")
        return 0
    if args.command == "fragments" and args.fragments_command == "build":
        from matrix_fragments import write_fragment_build_section

        definition = write_fragment_build_section(args.xyzin)
        print(
            "Built MATRIX fragments: "
            f"{args.xyzin} (fragments={len(definition.fragments)}, "
            f"reference={definition.reference_fragment})"
        )
        return 0
    if args.command == "fragments" and args.fragments_command == "set-state":
        from matrix_fragments import write_fragment_electronic_states

        states: dict[str, tuple[int, int]] = {}
        for value in args.state:
            fields = value.split(":")
            if len(fields) != 3 or not fields[0]:
                raise ValueError(
                    f"invalid fragment state {value!r}; expected ID:CHARGE:MULTIPLICITY"
                )
            identifier, charge, multiplicity = fields
            if identifier in states:
                raise ValueError(f"duplicate fragment state: {identifier}")
            try:
                states[identifier] = (int(charge), int(multiplicity))
            except ValueError as exc:
                raise ValueError(f"invalid fragment state integers: {value!r}") from exc
        definition = write_fragment_electronic_states(args.xyzin, states)
        print(
            "Updated MATRIX fragment electronic states: "
            f"{args.xyzin} (fragments={len(definition.fragments)})"
        )
        return 0
    if args.command == "fragments" and args.fragments_command == "centers":
        from matrix_fragments import write_interaction_center_section

        definition = write_interaction_center_section(args.xyzin)
        print(
            "Built MATRIX interaction centers: "
            f"{args.xyzin} (centers={len(definition.centers)}, "
            f"interactions={len(definition.interactions)})"
        )
        return 0
    if args.command == "rovib" and args.rovib_command == "summarize":
        from matrix_rovib import rovib_summary_lines, summarize_xyzin

        print("\n".join(rovib_summary_lines(summarize_xyzin(args.xyzin))))
        return 0
    if args.command == "rovib" and args.rovib_command == "rotational":
        from matrix_rovib import analyze_rotational_state

        result = analyze_rotational_state(
            args.xyzin,
            report=not args.no_report or args.out is not None,
            report_path=args.out,
            include_vibrational_analysis=not args.no_vibrational_analysis,
            coriolis_threshold_cm1=args.coriolis_threshold_cm1,
        )
        print(
            "Computed rotational state: "
            f"A/B/C(e)={result.equilibrium_MHz[0]:.8g}/"
            f"{result.equilibrium_MHz[1]:.8g}/{result.equilibrium_MHz[2]:.8g} MHz, "
            f"rotor={result.rotational.rotor_type}, sigma={result.rotational.symmetry_number}"
        )
        if result.report is not None:
            print(f"Wrote rotational report: {result.report}")
        for warning in result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        return 0
    if args.command == "rovib" and args.rovib_command == "vibin":
        from matrix_rovib import vibin_from_xyzin_fchk

        result = vibin_from_xyzin_fchk(
            args.xyzin,
            args.fchk,
            workdir=args.workdir,
            project_TR=not args.no_project_tr,
            update_vibrational_section=not args.no_update_vibrational,
        )
        print(
            "Built rovib vibin: "
            f"{result.vibin} (nvib={result.data.nvib}, "
            f"imag_like={result.n_imag_like})"
        )
        return 0
    if args.command == "rovib" and args.rovib_command == "coriolis":
        from matrix_rovib import (
            append_coriolis_to_vibin,
            compute_coriolis_from_xyzin,
            coriolis_report_lines,
        )

        result = compute_coriolis_from_xyzin(
            args.xyzin,
            vibin=args.vibin,
            Geff_thr_cm1=args.threshold_cm1,
            only_upper=not args.all_pairs,
        )
        text = "\n".join(coriolis_report_lines(result))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote Coriolis report: {args.out}")
        else:
            print(text)
        if args.append_vibin:
            vibin_path = args.vibin or (args.xyzin.parent / "vibin")
            append_coriolis_to_vibin(vibin_path, result)
            print(f"Appended Coriolis block: {vibin_path}")
        return 0
    if args.command == "rovib" and args.rovib_command == "qcent":
        from matrix_rovib import append_qcent_to_vibin, compute_qcent_from_xyzin, qcent_report_lines

        result = compute_qcent_from_xyzin(args.xyzin, vibin=args.vibin)
        text = "\n".join(qcent_report_lines(result))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote QCENT report: {args.out}")
        else:
            print(text)
        if args.append_vibin:
            vibin_path = args.vibin or (args.xyzin.parent / "vibin")
            append_qcent_to_vibin(vibin_path, result)
            print(f"Appended QCENT block: {vibin_path}")
        return 0
    if args.command == "rovib" and args.rovib_command == "one-mode":
        from matrix_gaussian import probe_gaussian_one_mode, write_gaussian_one_mode_json

        result = probe_gaussian_one_mode(
            args.log, args.fchk, mode=args.mode, qmax=args.qmax, nq=args.nq,
            axis=args.axis, quartic_cm1=args.quartic_cm1, basis_size=args.basis_size,
        )
        if args.out is not None:
            write_gaussian_one_mode_json(args.out, result)
            print(f"Wrote Gaussian one-mode diagnostic: {args.out}")
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "rovib" and args.rovib_command == "import-external":
        from matrix_rovib import promote_external_rovib_payload

        result = promote_external_rovib_payload(args.payload, args.xyzin)
        print(
            f"Imported external rovib payload ({result.source}): {args.xyzin}; "
            f"quartic={len(result.quartic_distortion_MHz)}, "
            f"sextic={len(result.sextic_distortion_MHz)}, alpha={len(result.alpha_rows_MHz)}"
        )
        return 0
    if args.command == "rovib" and args.rovib_command == "wmsrot-input":
        from matrix_rovib import (
            WMSRotInputOptions,
            wmsrot_input_text_from_xyzin,
            write_wmsrot_input,
        )

        options = WMSRotInputOptions(
            j_min=args.j_min,
            j_max=args.j_max,
            auto_estimate_j_range=args.auto_estimate_j_range,
            reduction=args.reduction,
        )
        if args.out is not None:
            out = write_wmsrot_input(args.xyzin, args.out, options=options)
            print(f"Wrote WMS-Rot input: {out}")
        else:
            print(wmsrot_input_text_from_xyzin(args.xyzin, options=options), end="")
        return 0
    if args.command == "rovib" and args.rovib_command == "wmsrot-run":
        from matrix_rovib import (
            WMSRotEngineUnavailable,
            WMSRotSimulationOptions,
            write_wmsrot_spectrum_outputs,
        )

        options = WMSRotSimulationOptions(
            j_min=args.j_min,
            j_max=args.j_max,
            intensity_cut=args.intensity_cut,
            reduction=args.reduction,
            a_type=not args.no_a_type,
            b_type=not args.no_b_type,
            c_type=not args.no_c_type,
        )
        try:
            result = write_wmsrot_spectrum_outputs(
                args.xyzin,
                args.out,
                plot_path=args.plot,
                table_path=args.table,
                fwhm_MHz=args.fwhm_mhz,
                options=options,
                write_section=not args.no_write_section,
            )
        except WMSRotEngineUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Wrote WMS-Rot spectrum line list: {result.csv_path} "
            f"(transitions={len(result.transitions)})"
        )
        if result.plot_path is not None:
            print(f"Wrote rotational spectrum plot: {result.plot_path}")
        if result.table_path is not None:
            print(f"Wrote rotational spectrum table: {result.table_path}")
        if not args.no_write_section:
            print(f"Updated #ROTATIONAL_SPECTRUM: {args.xyzin}")
        return 0
    if args.command == "rovib" and args.rovib_command == "vib-spectrum":
        from matrix_rovib import VibrationalSpectrumOptions, write_vibrational_spectrum_outputs

        if args.source == "hybrid" and args.level2_xyzin is None:
            print("rovib vib-spectrum --source hybrid requires --level2-xyzin", file=sys.stderr)
            return 2
        options = VibrationalSpectrumOptions(
            fwhm_cm1=args.fwhm_cm1,
            step_cm1=args.step_cm1,
            lineshape=args.lineshape,
            normalize=not args.no_normalize,
        )
        spectrum = write_vibrational_spectrum_outputs(
            args.xyzin,
            csv_path=args.csv,
            plot_path=args.plot,
            peaks_path=args.peaks,
            level2_xyzin=args.level2_xyzin,
            mode_match_csv_path=args.mode_match_csv,
            observable=args.observable,
            source=args.source,
            min_mode_overlap=args.min_mode_overlap,
            options=options,
        )
        outputs = [f"CSV: {args.csv}"]
        if args.plot is not None:
            outputs.append(f"plot: {args.plot}")
        if args.peaks is not None:
            outputs.append(f"peaks: {args.peaks}")
        if args.mode_match_csv is not None:
            outputs.append(f"mode matches: {args.mode_match_csv}")
        print(
            f"Wrote {spectrum.source} {spectrum.observable} spectrum "
            f"({len(spectrum.peaks)} peaks, {len(spectrum.x_cm1)} points): " + ", ".join(outputs)
        )
        return 0
    if args.command == "rovib" and args.rovib_command == "vib-compare":
        from matrix_rovib import (
            VibrationalSpectrumOptions,
            write_vibrational_spectrum_comparison_outputs,
        )

        if "hybrid" in {args.first_source, args.second_source} and args.second_xyzin is None:
            print(
                "rovib vib-compare with source=hybrid requires a second xyzin file", file=sys.stderr
            )
            return 2
        options = VibrationalSpectrumOptions(
            fwhm_cm1=args.fwhm_cm1,
            step_cm1=args.step_cm1,
            lineshape=args.lineshape,
            normalize=not args.no_normalize,
        )
        comparison = write_vibrational_spectrum_comparison_outputs(
            args.xyzin,
            csv_path=args.csv,
            plot_path=args.plot,
            second_xyzin=args.second_xyzin,
            observable=args.observable,
            first_source=args.first_source,
            second_source=args.second_source,
            options=options,
            mirror_second=False if args.no_mirror_second else None,
            min_mode_overlap=args.min_mode_overlap,
            mode_match_csv_path=args.mode_match_csv,
        )
        outputs = [f"CSV: {args.csv}"]
        if args.plot is not None:
            outputs.append(f"plot: {args.plot}")
        if args.mode_match_csv is not None:
            outputs.append(f"mode matches: {args.mode_match_csv}")
        mirror = "mirrored" if comparison.mirror_second else "not mirrored"
        second_file = args.second_xyzin if args.second_xyzin is not None else args.xyzin
        print(
            f"Wrote {args.observable} comparison "
            f"({args.xyzin} {args.first_source} vs {second_file} {args.second_source}, "
            f"second {mirror}): " + ", ".join(outputs)
        )
        return 0
    if args.command == "rovib" and args.rovib_command == "nist-ir":
        from matrix_rovib import fetch_nist_ir_gas_phase_csv

        result = fetch_nist_ir_gas_phase_csv(
            args.identifier,
            args.out,
            index=args.index,
            timeout=args.timeout,
        )
        if result.status != "downloaded":
            print(result.message, file=sys.stderr)
            return 3
        print(f"{result.message}: {result.csv_path}")
        return 0
    if args.command == "rovib" and args.rovib_command == "dos":
        from matrix_rovib import direct_vibrational_dos_from_xyzin

        out = args.out or (args.xyzin.parent / "dos_vib.dat")
        result = direct_vibrational_dos_from_xyzin(
            args.xyzin,
            vmax=args.vmax,
            emax_cm1=args.emax,
            emin_cm1=args.emin,
            bin_cm1=args.bin_cm1,
            ncap=args.ncap,
            t_k=args.temperature,
            out=out,
            q_out=args.q_out,
            number_out=args.number_out,
            cache_path=(
                None
                if args.no_cache
                else (args.cache or args.xyzin.parent / "dos_cache.json")
            ),
        )
        print(
            f"Wrote vibrational DOS: {result.path} (bins={len(result.bins_logg)}, "
            f"Q_vib={result.Q_vib:.8g}, TS={int(result.is_transition_state)}, "
            f"cache_hit={int(result.cache_hit)})"
        )
        if result.q_path is not None:
            print(f"Wrote vibrational Q(T): {result.q_path}")
        if result.number_path is not None:
            print(f"Wrote cumulative vibrational states: {result.number_path}")
        return 0
    if args.command == "rovib" and args.rovib_command == "dos-rovib":
        from matrix_rovib import rovib_pipeline

        result = rovib_pipeline(
            args.xyzin,
            vib_dos=args.vib_dos,
            out=args.out,
            rot_out=args.rot_out,
            q_out=args.q_out,
            number_out=args.number_out,
            emax_rot=args.emax_rot,
            jmax=args.jmax,
        )
        print(f"Wrote rovibrational DOS: {result.dos_rovib}")
        if result.q_path is not None:
            print(f"Wrote rovib Q(T): {result.q_path} (Q={result.Q_rovib:.8g})")
        if result.number_path is not None:
            print(f"Wrote cumulative rovibrational states: {result.number_path}")
        return 0
    if args.command == "thermo":
        from matrix_thermo import run_thermo_on_xyzin

        result = run_thermo_on_xyzin(
            args.xyzin,
            report=not args.no_report or args.out is not None,
            report_path=args.out,
            write_section=not args.no_write_section,
            cutoff_cm1=args.cutoff_cm1,
            keep_low_positive=args.keep_low_positive,
        )
        total = result.total
        q_text = "" if total is None or total.Q_dimless is None else f", Q={total.Q_dimless:.8g}"
        print(f"Ran MATRIX Thermo: {args.xyzin}{q_text}")
        if args.out is not None:
            print(f"Wrote thermo report: {args.out}")
        elif not args.no_report:
            print(f"Wrote thermo report: {args.xyzin.parent / 'thermo.report'}")
        if not args.no_write_section:
            print(f"Updated #THERMO: {args.xyzin}")
        return 0
    if args.command == "kinetics" and args.kinetics_command == "single":
        from matrix_kinetics import run_single_reaction

        stem = args.reaction_id.strip() or "R1"
        outdir = args.reactant_xyzin.parent
        result = run_single_reaction(
            args.reactant_xyzin,
            args.transition_state_xyzin,
            reactant_dos=args.reactant_dos,
            transition_state_dos=args.ts_dos,
            barrier_cm1=args.barrier_cm1,
            barrier_reference=args.barrier_reference,
            temperature_K=args.temperature,
            network_id=args.network_id,
            reaction_id=stem,
            reactant_id=args.reactant_id,
            transition_state_id=args.ts_id,
            product_id=args.product_id,
            path_degeneracy=args.path_degeneracy,
            tunneling_model=args.tunneling,
            imaginary_frequency_cm1=args.imaginary_frequency_cm1,
            rrkm_csv=args.rrkm_out or (outdir / f"{stem}.rrkm.csv"),
            manifest=args.manifest or (outdir / f"{args.network_id}.kinetics.json"),
            report=args.report or (outdir / f"{stem}.kinetics.report"),
            write_section=not args.no_write_section,
        )
        print(
            f"TST/RRKM {result.reaction.reaction_id}: "
            f"T={result.temperature_K:.6g} K, k_TST={result.k_tst_s1:.12e} s^-1, "
            f"RRKM points={len(result.rrkm)}"
        )
        print(f"Wrote RRKM table: {result.rrkm_csv}")
        print(f"Wrote kinetics network: {result.manifest_path}")
        print(f"Wrote kinetics report: {result.report_path}")
        if not args.no_write_section:
            print(f"Updated #KINETICS: {args.reactant_xyzin}")
        return 0
    if args.command == "kinetics" and args.kinetics_command in {
        "collision", "gorin", "nonadiabatic"
    }:
        from dataclasses import asdict
        from matrix_kinetics import (
            gorin_capture_rate,
            hard_sphere_collision_rate,
            nonadiabatic_tst_rate,
        )

        if args.kinetics_command == "collision":
            result = hard_sphere_collision_rate(
                args.mass_a_amu, args.mass_b_amu,
                args.radius_a_angstrom, args.radius_b_angstrom,
                args.temperature, convention=args.convention,
            )
        elif args.kinetics_command == "gorin":
            result = gorin_capture_rate(
                args.mass_a_amu, args.mass_b_amu, args.c6, args.temperature
            )
        else:
            result = nonadiabatic_tst_rate(
                args.reduced_mass_amu, args.spin_orbit_cm1,
                args.gradient_difference_ha_angstrom, args.crossing_energy_cm1,
                args.q_mecp, args.q_reactants, args.temperature,
            )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    if args.command == "gf":
        from matrix_gf import (
            format_gf_scaling_preview,
            gf_scaling_preview_from_xyzin,
            run_gf_report_from_fchk,
            run_xyzin_gf_report_from_engine_hessian,
            run_xyzin_gf_report_from_hessian_input,
            run_xyzin_gf_report_from_fchk,
            run_xyzin_gf_report_from_xyzin,
            write_csv_tables,
            write_gf_ped_section_from_report,
        )

        if args.scale_preview:
            if args.xyzin is None:
                raise ValueError("gf --scale-preview needs --xyzin with a frozen #GIC section")
            preview = gf_scaling_preview_from_xyzin(
                args.xyzin,
                scale_path=args.scale_file,
                scale_records=tuple(args.scale),
                scale_class_records=tuple(args.scale_class),
            )
            print(format_gf_scaling_preview(preview))
            return 0

        if args.xyzin is None:
            if args.fchk is None or args.hessian_file is not None:
                raise ValueError("gf needs --fchk alone, or --xyzin with a Hessian source")
            report = run_gf_report_from_fchk(args.fchk)
            prefix = "gf"
            source_kind = "fchk"
            source_path = args.fchk
        elif args.hessian_file is not None:
            if args.fchk is not None:
                raise ValueError("use either --fchk or --hessian-file for gf, not both")
            if not args.hessian_engine:
                raise ValueError("gf --hessian-file needs --hessian-engine")
            if args.write_hessian_section:
                from matrix_qm import (
                    cartesian_hessian_section_from_hessian_input,
                    hessian_input_from_engine,
                    write_cartesian_hessian_section_atomic,
                )

                hessian_input = hessian_input_from_engine(
                    args.hessian_engine,
                    args.hessian_file,
                    grd=args.hessian_cfour_grd,
                    output=(
                        args.hessian_xtb_output
                        if args.hessian_engine == "xtb"
                        else args.hessian_cfour_output
                    ),
                    geometry=args.hessian_xtb_geometry,
                    spectrum=args.hessian_xtb_spectrum,
                    input_path=args.hessian_pyscf_input,
                )
                report = run_xyzin_gf_report_from_hessian_input(
                    hessian_input,
                    args.hessian_file,
                    args.xyzin,
                    hessian_source=f"{args.hessian_engine}-hessian {args.hessian_file}",
                    scale_path=args.scale_file,
                    scale_records=tuple(args.scale),
                    scale_class_records=tuple(args.scale_class),
                    local=args.local,
                    force_threshold=args.force_threshold,
                    block_by_irrep=args.symmetry_blocks,
                    subtract_electrostatic=args.subtract_electrostatic,
                    subtract_uff_vdw=args.subtract_uff_vdw,
                    nonbonded_14_scale=args.nonbonded_14_scale,
                    large_amplitude_frequency_cutoff_cm=(
                        args.large_amplitude_frequency_cutoff_cm
                    ),
                    large_amplitude_frequency_model=args.large_amplitude_frequency_model,
                    cv_correction=args.cv_correction,
                    cv_correction_sigma_scale=args.cv_correction_sigma_scale,
                    cv_correction_threshold=args.cv_correction_threshold,
                )
                write_cartesian_hessian_section_atomic(
                    args.xyzin,
                    cartesian_hessian_section_from_hessian_input(
                        hessian_input,
                        source=f"{args.hessian_engine}-hessian {args.hessian_file}",
                    ),
                )
            else:
                report = run_xyzin_gf_report_from_engine_hessian(
                    args.hessian_engine,
                    args.hessian_file,
                    args.xyzin,
                    cfour_grd=args.hessian_cfour_grd,
                    cfour_output=args.hessian_cfour_output,
                    xtb_geometry=args.hessian_xtb_geometry,
                    xtb_spectrum=args.hessian_xtb_spectrum,
                    xtb_output=args.hessian_xtb_output,
                    pyscf_input=args.hessian_pyscf_input,
                    scale_path=args.scale_file,
                    scale_records=tuple(args.scale),
                    scale_class_records=tuple(args.scale_class),
                    local=args.local,
                    force_threshold=args.force_threshold,
                    block_by_irrep=args.symmetry_blocks,
                    subtract_electrostatic=args.subtract_electrostatic,
                    subtract_uff_vdw=args.subtract_uff_vdw,
                    nonbonded_14_scale=args.nonbonded_14_scale,
                    large_amplitude_frequency_cutoff_cm=(
                        args.large_amplitude_frequency_cutoff_cm
                    ),
                    large_amplitude_frequency_model=args.large_amplitude_frequency_model,
                    cv_correction=args.cv_correction,
                    cv_correction_sigma_scale=args.cv_correction_sigma_scale,
                    cv_correction_threshold=args.cv_correction_threshold,
                )
            prefix = "gic_gf"
            source_kind = str(args.hessian_engine)
            source_path = args.hessian_file
        elif args.fchk is None:
            report = run_xyzin_gf_report_from_xyzin(
                args.xyzin,
                scale_path=args.scale_file,
                scale_records=tuple(args.scale),
                scale_class_records=tuple(args.scale_class),
                local=args.local,
                force_threshold=args.force_threshold,
                block_by_irrep=args.symmetry_blocks,
                subtract_electrostatic=args.subtract_electrostatic,
                subtract_uff_vdw=args.subtract_uff_vdw,
                nonbonded_14_scale=args.nonbonded_14_scale,
                large_amplitude_frequency_cutoff_cm=args.large_amplitude_frequency_cutoff_cm,
                large_amplitude_frequency_model=args.large_amplitude_frequency_model,
                cv_correction=args.cv_correction,
                cv_correction_sigma_scale=args.cv_correction_sigma_scale,
                cv_correction_threshold=args.cv_correction_threshold,
            )
            prefix = "gic_gf"
            source_kind = "xyzin"
            source_path = args.xyzin
        else:
            report = run_xyzin_gf_report_from_fchk(
                args.fchk,
                args.xyzin,
                scale_path=args.scale_file,
                scale_records=tuple(args.scale),
                scale_class_records=tuple(args.scale_class),
                local=args.local,
                force_threshold=args.force_threshold,
                block_by_irrep=args.symmetry_blocks,
                subtract_electrostatic=args.subtract_electrostatic,
                subtract_uff_vdw=args.subtract_uff_vdw,
                nonbonded_14_scale=args.nonbonded_14_scale,
                large_amplitude_frequency_cutoff_cm=args.large_amplitude_frequency_cutoff_cm,
                large_amplitude_frequency_model=args.large_amplitude_frequency_model,
                cv_correction=args.cv_correction,
                cv_correction_sigma_scale=args.cv_correction_sigma_scale,
                cv_correction_threshold=args.cv_correction_threshold,
            )
            prefix = "gic_gf"
            source_kind = "fchk"
            source_path = args.fchk
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report.text + "\n", encoding="utf-8")
            print(f"Wrote TRINITY harmonic report: {args.out}")
        else:
            print(report.text)
        if args.csv_dir is not None:
            written = write_csv_tables(report, args.csv_dir, prefix=prefix)
            print(f"Wrote TRINITY harmonic CSV tables: {len(written)} files in {args.csv_dir}")
        if args.xyzin is not None and not args.no_write_section:
            write_gf_ped_section_from_report(
                args.xyzin,
                report,
                source_kind=source_kind,
                source_path=source_path,
                report_path=args.out,
                csv_dir=args.csv_dir,
            )
            print(f"Updated #GF_PED: {args.xyzin}")
        return 0
    if args.command == "vpt2-vci":
        from matrix_vpt2_vci import (
            NormalModeQFFFitConfig,
            VCIOptions,
            collect_vpt2_vci_outputs_from_xyzin,
            refresh_vpt2_vci_section,
            load_force_field,
            run_vpt2_vci_report,
            vpt2_vci_output_summary_lines,
            vpt2_vci_section_from_run,
            write_csv_tables,
            write_vpt2_vci_manifest,
            write_vpt2_vci_section,
            fit_normal_mode_qff_from_zion,
            write_indexed_qff,
            write_normal_mode_qff_result,
        )

        if args.collect is not None:
            snapshot = (
                collect_vpt2_vci_outputs_from_xyzin(args.collect)
                if args.no_write
                else refresh_vpt2_vci_section(args.collect)
            )
            print("\n".join(vpt2_vci_output_summary_lines(snapshot)))
            if not args.no_write:
                print(f"Updated #VPT2_VCI: {args.collect}")
            return 0

        zion_fit = None
        if args.zion_force_field is not None:
            if args.zion_xyzin is None:
                raise ValueError("--zion-force-field requires --zion-xyzin")
            zion_fit = fit_normal_mode_qff_from_zion(
                args.zion_xyzin,
                args.zion_force_field,
                config=NormalModeQFFFitConfig(
                    amplitude=args.zion_fit_amplitude,
                    training_pairs=args.zion_fit_pairs,
                    holdout_pairs=args.zion_holdout_pairs,
                    seed=args.zion_fit_seed,
                    workers=args.zion_fit_workers,
                ),
            )
            qff = zion_fit.force_field
            qff_output = args.zion_qff_out
            fit_json = args.zion_fit_json
            if args.run_dir is not None:
                qff_output = qff_output or (args.run_dir / "zion_normal_mode.qff")
                fit_json = fit_json or (args.run_dir / "zion_normal_mode_fit.json")
            if qff_output is not None:
                print(f"QFF: {write_indexed_qff(zion_fit, qff_output)}")
            if fit_json is not None:
                print(f"fit: {write_normal_mode_qff_result(zion_fit, fit_json)}")
            print(json.dumps(zion_fit.diagnostics.to_dict(), indent=2, sort_keys=True))
            if args.zion_fit_only:
                return 0
        else:
            if args.zion_fit_only:
                raise ValueError("--zion-fit-only requires --zion-force-field")
            qff = load_force_field(
                fchk_path=args.fchk, qff_path=args.qff_file, xyzin_path=args.xyzin
            )
        report = run_vpt2_vci_report(
            qff,
            max_quanta=args.max_quanta,
            roots=args.roots,
            options=VCIOptions(),
            vci_method=args.vci_method,
        )
        report_path = args.out
        csv_dir = args.csv_dir
        run_dir = args.run_dir
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_path or (run_dir / "vpt2_vci.report")
            csv_dir = csv_dir or run_dir
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report.text + "\n", encoding="utf-8")
            print(f"Wrote VPT2/VCI report: {report_path}")
        else:
            print(report.text)
        written_csv: dict[str, Path] = {}
        if csv_dir is not None:
            written_csv = write_csv_tables(report, csv_dir)
            print(f"Wrote VPT2/VCI CSV tables: {len(written_csv)} files in {csv_dir}")
        outputs = {}
        if report_path is not None:
            outputs["report"] = report_path
        outputs.update(
            {
                key.removesuffix(".csv").replace("mode_contributions", "mode_contributions"): path
                for key, path in written_csv.items()
            }
        )
        manifest_path = None
        source_kind, source_path = _vpt2_vci_source(args)
        if run_dir is not None:
            manifest_path = write_vpt2_vci_manifest(
                run_dir=run_dir,
                inputs={source_kind: source_path} if source_path is not None else {},
                outputs=outputs,
                max_quanta=args.max_quanta,
                roots=args.roots,
                vci_method=args.vci_method,
                source_kind=source_kind,
                status="complete" if "comparison" in outputs and "report" in outputs else "partial",
            )
            outputs["manifest"] = manifest_path
            print(f"manifest: {manifest_path}")
        if args.xyzin is not None and args.xyzin.exists() and outputs:
            status = "complete" if "comparison" in outputs and "report" in outputs else "partial"
            write_vpt2_vci_section(
                args.xyzin,
                vpt2_vci_section_from_run(
                    source_kind=source_kind,
                    source_path=source_path,
                    run_dir=run_dir,
                    report_path=report_path,
                    csv_dir=csv_dir,
                    manifest_path=manifest_path,
                    max_quanta=args.max_quanta,
                    roots=args.roots,
                    vci_method=args.vci_method,
                    outputs=outputs,
                    status=status,
                ),
            )
            print(f"Updated #VPT2_VCI: {args.xyzin}")
        return 0
    if args.command == "hybrid-vibrations":
        import csv

        import numpy as np

        from matrix_chem import read_geometry
        from matrix_core import read_xyzin_geometry
        from matrix_gaussian import (
            read_gaussian_fchk,
            read_gaussian_fchk_geometry,
            read_gaussian_fundamentals,
            write_gaussian_select_anharmonic_input,
        )
        from matrix_vpt2_vci import (
            PathModeSelectionSettings,
            VariationalBand,
            assemble_hybrid_spectrum,
            centered_path_tangent,
            select_path_mode_block,
        )

        fchk = read_gaussian_fchk(args.fchk)
        reference = read_gaussian_fchk_geometry(args.fchk)
        harmonic = np.asarray(fchk.harmonic_frequencies_cm, dtype=float)
        dimension = 3 * reference.natoms
        if harmonic.size < 1 or fchk.normal_modes.size != harmonic.size * dimension:
            raise ValueError(
                "the FCHK must contain a complete harmonic frequency and normal-mode block"
            )
        modes = np.asarray(fchk.normal_modes, dtype=float).reshape((len(harmonic), dimension))
        tangents: list[np.ndarray] = []
        path_pairs: list[dict[str, str]] = []
        for lower_path, upper_path in args.path_pair:
            lower = (
                read_xyzin_geometry(lower_path)
                if lower_path.suffix.casefold() == ".xyzin"
                else read_geometry(lower_path)
            )
            upper = (
                read_xyzin_geometry(upper_path)
                if upper_path.suffix.casefold() == ".xyzin"
                else read_geometry(upper_path)
            )
            if lower.atoms != reference.atoms or upper.atoms != reference.atoms:
                raise ValueError(
                    "every path geometry must preserve the FCHK atom count, identity and order"
                )
            tangents.append(
                centered_path_tangent(
                    lower.coordinates_angstrom,
                    upper.coordinates_angstrom,
                    reference.coordinates_angstrom,
                    fchk.masses_amu,
                )
            )
            path_pairs.append({"lower": str(lower_path), "upper": str(upper_path)})
        selection = select_path_mode_block(
            modes,
            np.asarray(tangents, dtype=float),
            fchk.masses_amu,
            settings=PathModeSelectionSettings(
                minimum_principal_overlap=args.minimum_principal_overlap,
                minimum_projection_gap=args.minimum_projection_gap,
                maximum_active_projection=args.maximum_active_projection,
                minimum_tangent_singular_value=args.minimum_tangent_singular_value,
                maximum_mode_orthogonality_error=args.maximum_mode_orthogonality_error,
            ),
        )

        generated_input = None
        if args.gaussian_input is not None:
            if not args.checkpoint or not args.route:
                raise ValueError(
                    "--gaussian-input requires both --checkpoint and --route"
                )
            generated_input = write_gaussian_select_anharmonic_input(
                args.gaussian_input,
                checkpoint=args.checkpoint,
                route=args.route,
                active_mode_indices=selection.active_mode_indices,
                mode_count=len(harmonic),
                processors=args.processors,
                memory=args.memory,
            )

        perturbative: dict[int, float] = {}
        fundamental_rows = ()
        if args.anharmonic_log is not None:
            fundamental_rows = read_gaussian_fundamentals(args.anharmonic_log, harmonic)
            path_modes = set(selection.path_mode_indices)
            active_modes = set(selection.active_mode_indices)
            wrongly_active = [
                row.internal_mode_index
                for row in fundamental_rows
                if row.internal_mode_index in path_modes and row.status == "active"
            ]
            wrongly_inactive = [
                row.internal_mode_index
                for row in fundamental_rows
                if row.internal_mode_index in active_modes and row.status != "active"
            ]
            if wrongly_active or wrongly_inactive:
                raise ValueError(
                    "Gaussian selective-anharmonic status is inconsistent with the accepted "
                    f"partition (path active={wrongly_active}, ordinary inactive={wrongly_inactive})"
                )
            perturbative = {
                row.internal_mode_index: row.fundamental_cm
                for row in fundamental_rows
                if row.internal_mode_index in active_modes
            }

        variational_bands: list[VariationalBand] = []
        if args.dvr_levels is not None:
            with args.dvr_levels.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "state" not in rows[0]:
                raise ValueError("DVR/VCI level CSV must contain a state column")
            energy_key = next(
                (
                    key
                    for key in ("energy_above_ground_cm-1", "energy_cm-1", "energy_cm")
                    if key in rows[0]
                ),
                None,
            )
            if energy_key is None:
                raise ValueError(
                    "DVR/VCI level CSV must contain energy_above_ground_cm-1, "
                    "energy_cm-1 or energy_cm"
                )
            levels = {int(row["state"]): float(row[energy_key]) for row in rows}
            transition_specs = args.variational_transition or ["0:1:path fundamental"]
            for specification in transition_specs:
                fields = specification.split(":", 2)
                if len(fields) < 2:
                    raise ValueError(
                        "--variational-transition must have LOWER:UPPER[:LABEL] form"
                    )
                lower_state, upper_state = int(fields[0]), int(fields[1])
                if lower_state not in levels or upper_state not in levels:
                    raise ValueError(
                        f"DVR/VCI levels do not contain transition {lower_state}:{upper_state}"
                    )
                label = fields[2].strip() if len(fields) == 3 else f"{lower_state}->{upper_state}"
                variational_bands.append(
                    VariationalBand(
                        label=label,
                        transition_cm=levels[upper_state] - levels[lower_state],
                        lower_state=lower_state,
                        upper_state=upper_state,
                    )
                )

        spectrum = None
        if fundamental_rows and variational_bands:
            spectrum = assemble_hybrid_spectrum(
                harmonic, perturbative, selection, variational_bands
            )
        payload: dict[str, object] = {
            "schema": "matrix.trinity.hybrid-vibrations.v1",
            "status": "complete" if spectrum is not None else "partitioned",
            "inputs": {
                "fchk": str(args.fchk),
                "path_pairs": path_pairs,
                "anharmonic_log": str(args.anharmonic_log) if args.anharmonic_log else None,
                "dvr_levels": str(args.dvr_levels) if args.dvr_levels else None,
            },
            "selection": {
                "path_mode_indices": list(selection.path_mode_indices),
                "active_mode_indices": list(selection.active_mode_indices),
                "projection_scores": selection.projection_scores.tolist(),
                "principal_overlaps": selection.principal_overlaps.tolist(),
                "projection_gap": selection.projection_gap,
                "maximum_active_projection": selection.maximum_active_projection,
                "mode_orthogonality_error": selection.mode_orthogonality_error,
                "tangent_rank": selection.tangent_rank,
                "criteria": {
                    "minimum_principal_overlap": selection.settings.minimum_principal_overlap,
                    "minimum_projection_gap": selection.settings.minimum_projection_gap,
                    "maximum_active_projection": selection.settings.maximum_active_projection,
                    "minimum_tangent_singular_value": selection.settings.minimum_tangent_singular_value,
                    "maximum_mode_orthogonality_error": selection.settings.maximum_mode_orthogonality_error,
                },
            },
            "generated_gaussian_input": str(generated_input) if generated_input else None,
            "ordinary_bands": (
                [
                    {
                        "mode_index": band.mode_index,
                        "harmonic_cm-1": band.harmonic_cm,
                        "fundamental_cm-1": band.fundamental_cm,
                        "treatment": band.treatment,
                        "path_projection": band.path_projection,
                    }
                    for band in spectrum.ordinary_modes
                ]
                if spectrum is not None
                else []
            ),
            "variational_bands": [
                {
                    "label": band.label,
                    "lower_state": band.lower_state,
                    "upper_state": band.upper_state,
                    "transition_cm-1": band.transition_cm,
                }
                for band in variational_bands
            ],
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        if args.csv is not None and spectrum is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    ("band", "mode_index", "lower_state", "upper_state", "value_cm-1", "treatment")
                )
                for band in spectrum.ordinary_modes:
                    writer.writerow(
                        (f"mode {band.mode_index}", band.mode_index, "", "", band.fundamental_cm, band.treatment)
                    )
                for band in spectrum.variational_bands:
                    writer.writerow(
                        (band.label, "", band.lower_state, band.upper_state, band.transition_cm, "variational")
                    )
        print(rendered)
        return 0
    if args.command == "dvr" and args.dvr_command == "prepare":
        import shlex

        from matrix_dvr import (
            DVRRequest,
            build_fortran_bridge_args,
            build_fortran_shell_command,
            build_path_analysis_args,
            dvr_section_from_request,
            is_fortran_solver,
            resolve_dvr_executable,
            write_dvr_manifest,
            write_dvr_section,
        )

        request = DVRRequest(
            repo_root=root,
            log_path=args.log,
            outdir=args.outdir,
            figdir=args.figdir or (args.outdir / "figures"),
            prefix=args.prefix,
            boundary=args.boundary,
            solver=args.solver,
            compute_rotconst=not args.no_rotconst,
            label_cremer_pople=not args.no_cremer_pople,
            check_only=args.check_only,
        )
        python_args = build_path_analysis_args(request)
        command = f"{shlex.quote(request.python_executable)} {shlex.join(python_args)}"
        manifest_args = python_args
        if is_fortran_solver(request.solver):
            bridge_args = build_fortran_bridge_args(request, resolve_dvr_executable(root))
            command = build_fortran_shell_command(request, python_args, bridge_args)
            manifest_args = [command]
        manifest = write_dvr_manifest(request, manifest_args)
        if args.xyzin is not None:
            write_dvr_section(
                args.xyzin,
                dvr_section_from_request(request, manifest_path=manifest),
            )
            print(f"Updated #DVR: {args.xyzin}")
        print(f"manifest: {manifest}")
        print(f"command: {command}")
        return 0
    if args.command == "dvr" and args.dvr_command == "run":
        from matrix_dvr import (
            DVRRequest,
            dvr_output_summary_lines,
            dvr_request_from_section,
            dvr_section_from_request,
            read_dvr_section,
            refresh_dvr_section,
            run_dvr_request,
            write_dvr_section,
        )

        if args.log is None:
            if args.xyzin is None:
                raise ValueError("dvr run needs LOG --outdir, or --xyzin containing #DVR")
            request = dvr_request_from_section(read_dvr_section(args.xyzin), repo_root=root)
        else:
            if args.outdir is None:
                raise ValueError("dvr run with LOG needs --outdir")
            request = DVRRequest(
                repo_root=root,
                log_path=args.log,
                outdir=args.outdir,
                figdir=args.figdir or (args.outdir / "figures"),
                prefix=args.prefix,
                boundary=args.boundary,
                solver=args.solver,
                compute_rotconst=not args.no_rotconst,
                label_cremer_pople=not args.no_cremer_pople,
                check_only=args.check_only,
            )

        result = run_dvr_request(request, timeout=args.timeout)
        print(f"manifest: {result.manifest_path}")
        print(f"status: {result.status}")
        if args.xyzin is not None:
            write_dvr_section(
                args.xyzin,
                dvr_section_from_request(
                    request,
                    manifest_path=result.manifest_path,
                    status=result.status,
                ),
            )
            snapshot = refresh_dvr_section(args.xyzin)
            print("\n".join(dvr_output_summary_lines(snapshot)))
            print(f"Updated #DVR: {args.xyzin}")
        return 0
    if args.command == "dvr" and args.dvr_command == "collect":
        from matrix_dvr import (
            collect_dvr_outputs_from_xyzin,
            dvr_output_summary_lines,
            refresh_dvr_section,
        )

        snapshot = (
            collect_dvr_outputs_from_xyzin(args.xyzin)
            if args.no_write
            else refresh_dvr_section(args.xyzin)
        )
        print("\n".join(dvr_output_summary_lines(snapshot)))
        if not args.no_write:
            print(f"Updated #DVR: {args.xyzin}")
        return 0
    if args.command == "semiexp":
        from matrix_morpheus import (
            DEFAULT_SEMIEXP_OBSERVABLE,
            DEFAULT_SEMIEXP_ROBUST_LOSS,
            DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS,
            HYDROGEN_PARAMETER_CONSTRAINT,
            ParameterClassConstraint,
            QMParameterPredicate,
            SemiexperimentalFinalValidationOptions,
            SemiexperimentalFitRequest,
            SYNTHON_CLASS_LEVELS,
            advise_semiexperimental_gic_sensitivity,
            derive_primitive_class_plan,
            fit_semiexperimental_geometry,
            fit_ground_state_r0_geometry,
            initial_geometry_predicates,
            is_msr_legacy_file,
            kraitchman_seed_predicates,
            parse_primitive_class_spec,
            primitive_class_decision_lines,
            prepare_semiexperimental_xyzin,
            preview_semiexperimental_gics,
            read_geometry_input,
            read_morpheus_input_config,
            read_msr_legacy_input,
            read_observations,
            read_semiexperimental_job,
            run_semiexperimental_final_validation,
            semiexperimental_latex_tables,
            synthon_primitive_class_specs,
            write_morpheus_section_from_result,
            write_semiexperimental_html_report,
            write_semiexperimental_standalone_latex,
        )

        legacy_msr_job = bool(args.job and is_msr_legacy_file(args.job))
        legacy_input = read_msr_legacy_input(args.job) if legacy_msr_job else None
        job = None if legacy_msr_job or not args.job else read_semiexperimental_job(args.job)
        xyzin_config = read_morpheus_input_config(args.xyzin) if args.xyzin is not None else None
        geometry_path = args.xyz or (job.path if job is not None else None) or args.xyzin
        observations_inline = job.observations_inline if job is not None else ()
        observations_path = (
            args.observations
            or (None if observations_inline else (job.observations if job is not None else None))
            or args.xyzin
        )
        if legacy_msr_job:
            geometry_path = args.xyz or args.job
            observations_path = args.observations or args.job
            observations_inline = ()
        if geometry_path is None:
            raise ValueError("semiexp needs --geometry, --job or --xyzin")
        if observations_path is None and not observations_inline:
            raise ValueError(
                "semiexp needs --observations, inline [[isotopologues]], "
                "a [files].observations entry in --job, or --xyzin"
            )
        preprocess = prepare_semiexperimental_xyzin(
            Path(geometry_path),
            observations_source=Path(observations_path) if observations_path is not None else None,
            observations_inline=observations_inline,
            xyzin_path=args.xyzin,
            workdir=args.outdir,
        )
        geometry_path = preprocess.xyzin
        observations = read_observations(preprocess.xyzin)
        print(f"semiexp_xyzin: {preprocess.xyzin}")
        if preprocess.created_or_updated_geometry:
            print("semiexp_xyzin_geometry: updated")
        if preprocess.updated_isotopologues:
            print("semiexp_xyzin_isotopologues: updated")

        fixed = _merge_unique(
            preprocess.source_fixed_parameters, job.fixed_parameters if job else ()
        )
        if xyzin_config is not None:
            fixed = _merge_unique(fixed, xyzin_config.fixed_parameters)
        fixed = _merge_unique(fixed, _parse_fixed_parameters(args.fixed))
        if args.fix_hydrogens:
            fixed = _merge_unique(fixed, (HYDROGEN_PARAMETER_CONSTRAINT,))
        xyzin_observable = xyzin_config.observable if xyzin_config is not None else None
        observable = _job_default(
            args.observable,
            DEFAULT_SEMIEXP_OBSERVABLE,
            job.observable if job else xyzin_observable,
        )
        xyzin_coordinate_model = xyzin_config.coordinate_model if xyzin_config is not None else None
        coordinate_model = _job_default(
            args.coordinate_model,
            "gic",
            job.coordinate_model if job else xyzin_coordinate_model,
        )
        xyzin_rotational_components = xyzin_config.components if xyzin_config is not None else None
        rotational_components = _job_default(
            args.rotational_components,
            DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS,
            job.rotational_components if job else xyzin_rotational_components,
        )
        qm_predicates = _merge_unique(
            job.qm_predicates if job else (),
            _parse_qm_predicates(args.qm_predicate, QMParameterPredicate),
        )
        if (
            xyzin_config is not None
            and xyzin_config.initial_geometry_predicates.enabled
            and not args.qm_predicate
        ):
            geometry_input_for_predicates = read_geometry_input(Path(geometry_path))
            spec = xyzin_config.initial_geometry_predicates
            generated_predicates = initial_geometry_predicates(
                tuple(geometry_input_for_predicates.atoms),
                geometry_input_for_predicates.coordinates_angstrom,
                distance_sigma_angstrom=spec.distance_sigma_angstrom,
                angle_sigma_degree=spec.angle_sigma_degree,
                dihedral_sigma_degree=spec.dihedral_sigma_degree,
                scope=spec.scope,
            )
            qm_predicates = _merge_unique(qm_predicates, generated_predicates)
            print(
                "xyzin_initial_geometry_predicates: "
                f"count={len(generated_predicates)} "
                f"sigma_R={spec.distance_sigma_angstrom:g} "
                f"sigma_A={spec.angle_sigma_degree:g} "
                f"sigma_D={spec.dihedral_sigma_degree:g}"
            )
        if args.kraitchman_predicates:
            geometry_input_for_kraitchman = read_geometry_input(Path(geometry_path))
            kraitchman_predicates = kraitchman_seed_predicates(
                tuple(geometry_input_for_kraitchman.atoms),
                geometry_input_for_kraitchman.coordinates_angstrom,
                observations,
                sigma_distance_angstrom=args.kraitchman_distance_sigma,
                sigma_angle_degree=args.kraitchman_angle_sigma,
                require_all_atoms_seeded=not args.kraitchman_partial_predicates,
            )
            qm_predicates = _merge_unique(qm_predicates, kraitchman_predicates)
            print(
                "kraitchman_predicates: "
                f"count={len(kraitchman_predicates)} "
                f"sigma_R={args.kraitchman_distance_sigma:g} "
                f"sigma_A={args.kraitchman_angle_sigma:g}"
            )
        parameter_classes = _merge_unique(
            job.parameter_classes if job else (),
            _parse_parameter_classes(args.parameter_class, ParameterClassConstraint),
        )
        primitive_classes = tuple(
            parse_primitive_class_spec(item) for item in getattr(args, "primitive_class", ())
        )
        if xyzin_config is not None:
            primitive_classes = _merge_unique(primitive_classes, xyzin_config.primitive_classes)
            synthon_spec = xyzin_config.synthon_primitive_classes
            if synthon_spec.enabled:
                synthon_budget = _primitive_class_budget(
                    xyzin_config.primitive_class_budget or args.primitive_class_budget,
                    observations=observations,
                    rotational_components=rotational_components,
                )
                geometry_input_for_classes = read_geometry_input(Path(geometry_path))
                if synthon_spec.level == "auto":
                    preview_for_auto = preview_semiexperimental_gics(
                        Path(geometry_path),
                        observations,
                    )
                    primitive_class_min_for_auto = (
                        xyzin_config.primitive_class_min
                        if xyzin_config.primitive_class_min is not None
                        else args.primitive_class_min
                    )
                    primitive_class_cross_for_auto = (
                        xyzin_config.primitive_class_cross_max
                        if xyzin_config.primitive_class_cross_max is not None
                        else args.primitive_class_cross_max
                    )
                    candidate_records = []
                    for candidate_level in ("coarse", "medium", "fine"):
                        candidate_generated = synthon_primitive_class_specs(
                            tuple(geometry_input_for_classes.atoms),
                            geometry_input_for_classes.coordinates_angstrom,
                            level=candidate_level,
                            include_bonds=synthon_spec.include_bonds,
                            include_angles=synthon_spec.include_angles,
                            min_group_size=synthon_spec.min_group_size,
                            bond_order_bins=synthon_spec.bond_order_bins,
                        )
                        candidate_classes = _merge_unique(
                            primitive_classes,
                            candidate_generated,
                        )
                        candidate_plan = derive_primitive_class_plan(
                            preview_for_auto.gic_labels,
                            candidate_classes,
                            min_fraction=primitive_class_min_for_auto,
                            cross_fraction_max=primitive_class_cross_for_auto,
                            max_classes=synthon_budget,
                        )
                        candidate_fixed = _merge_unique(fixed, candidate_plan.fixed_patterns)
                        candidate_parameter_classes = _merge_unique(
                            parameter_classes,
                            candidate_plan.parameter_classes,
                        )
                        candidate_request = SemiexperimentalFitRequest(
                            initial_geometry=geometry_path,
                            observations=observations,
                            fixed_parameters=candidate_fixed,
                            observable=observable,
                            rotational_components=rotational_components,
                            qm_predicates=qm_predicates,
                            parameter_classes=candidate_parameter_classes,
                            coordinate_model=coordinate_model,
                            robust_loss=(
                                job.robust_loss
                                if job and args.robust_loss == DEFAULT_SEMIEXP_ROBUST_LOSS
                                else args.robust_loss
                            ),
                            robust_scale=args.robust_scale,
                            leave_one_out=False,
                            excluded_rotational_constants=tuple(
                                args.exclude_rotational_constant
                            ),
                        )
                        candidate_outdir = (
                            args.outdir / "_synthon_auto_candidates" / candidate_level
                        )
                        candidate_result = fit_semiexperimental_geometry(
                            candidate_request,
                            max_iter=(
                                args.max_iter
                                if args.max_iter is not None
                                else (job.max_iter if job else None)
                            ),
                            step=(
                                args.step
                                if args.step != 1.0e-4
                                else (job.step if job and job.step is not None else 1.0e-4)
                            ),
                            damping=(
                                args.damping
                                if args.damping != 1.0e-8
                                else (job.damping if job and job.damping is not None else 1.0e-8)
                            ),
                            max_step=(
                                args.max_step
                                if args.max_step != 0.25
                                else (job.max_step if job and job.max_step is not None else 0.25)
                            ),
                            prune_condition=(
                                args.prune_condition
                                if args.prune_condition != 0.0
                                else (
                                    job.prune_condition
                                    if job and job.prune_condition is not None
                                    else 0.0
                                )
                            ),
                            outdir=candidate_outdir,
                        )
                        score = _semiexp_synthon_auto_score(candidate_result)
                        candidate_records.append(
                            (
                                score,
                                candidate_level,
                                candidate_generated,
                                candidate_plan,
                                candidate_result,
                            )
                        )
                        print(
                            "synthon_auto_candidate: "
                            f"level={candidate_level} "
                            f"classes={len(candidate_plan.parameter_classes)} "
                            f"active={candidate_result.diagnostics.n_optimized_parameters} "
                            f"rank={candidate_result.diagnostics.rank} "
                            f"cond={candidate_result.diagnostics.condition_number:.6g} "
                            f"max_sigma_XY={score[3]:.6g} "
                            f"max_sigma_XH={score[4]:.6g} "
                            f"max_sigma_CH={score[5]:.6g} "
                            f"max_sigma_A={score[6]:.6g} "
                            f"violations={score[0]}"
                        )
                    selected = min(candidate_records, key=lambda item: item[0])
                    synthon_level = selected[1]
                    generated_classes = selected[2]
                    print(
                        "synthon_auto_selected: "
                        f"level={synthon_level} "
                        f"classes={len(selected[3].parameter_classes)} "
                        f"score={selected[0]}"
                    )
                else:
                    synthon_level = synthon_spec.level
                    if synthon_level not in SYNTHON_CLASS_LEVELS:
                        raise ValueError(f"Unknown SYNTHON_LEVEL: {synthon_level}")
                    generated_classes = synthon_primitive_class_specs(
                        tuple(geometry_input_for_classes.atoms),
                        geometry_input_for_classes.coordinates_angstrom,
                        level=synthon_level,
                        include_bonds=synthon_spec.include_bonds,
                        include_angles=synthon_spec.include_angles,
                        min_group_size=synthon_spec.min_group_size,
                        bond_order_bins=synthon_spec.bond_order_bins,
                    )
                primitive_classes = _merge_unique(primitive_classes, generated_classes)
                print(
                    "xyzin_synthon_primitive_classes: "
                    f"count={len(generated_classes)} "
                    f"level={synthon_level} "
                    f"include_bonds={synthon_spec.include_bonds} "
                    f"include_angles={synthon_spec.include_angles}"
                )
        backend = _job_default(args.backend, "python", job.backend if job else None)
        max_iter = args.max_iter if args.max_iter is not None else (job.max_iter if job else None)
        step = _job_default(args.step, 1.0e-4, job.step if job else None)
        damping = _job_default(args.damping, 1.0e-8, job.damping if job else None)
        max_step = _job_default(args.max_step, 0.25, job.max_step if job else None)
        prune_condition = _job_default(
            args.prune_condition,
            0.0,
            job.prune_condition if job else None,
        )
        robust_loss = _job_default(
            args.robust_loss,
            DEFAULT_SEMIEXP_ROBUST_LOSS,
            job.robust_loss if job else None,
        )
        robust_scale = _job_default(args.robust_scale, 0.0, job.robust_scale if job else None)
        leave_one_out = bool(args.leave_one_out or (job.leave_one_out if job else False))
        checkpoint = (
            args.checkpoint if args.checkpoint is not None else (job.checkpoint if job else None)
        )
        restart = args.restart if args.restart is not None else (job.restart if job else None)
        legacy_robust_profile = bool(
            legacy_input
            and legacy_input.controls.condition_active
            and not args.no_auto_stabilize
        )
        if legacy_input and legacy_input.controls.outlier_active:
            if args.robust_loss == DEFAULT_SEMIEXP_ROBUST_LOSS:
                robust_loss = "huber"
            if args.robust_scale == 0.0:
                robust_scale = 0.1
        if legacy_robust_profile:
            if args.damping == 1.0e-8:
                damping = 1.0e-3
            if args.max_step == 0.25:
                max_step = 5.0e-3
            if args.max_iter is None:
                max_iter = 500
            print(
                "morpheus_legacy_automatic_profile: "
                f"outlier={legacy_input.controls.outlier or 'inactive'} "
                f"condition={legacy_input.controls.condition or 'inactive'} "
                f"propagation={legacy_input.controls.propagation or 'default'} "
                f"robust_loss={robust_loss} robust_scale={robust_scale:g} "
                f"damping={damping:g} max_step={max_step:g}"
            )
        if primitive_classes:
            if coordinate_model != "gic":
                raise ValueError("--primitive-class is only supported with --coordinate-model gic")
            primitive_class_budget_raw = args.primitive_class_budget
            if (
                xyzin_config is not None
                and xyzin_config.primitive_class_budget is not None
                and args.primitive_class_budget == "auto"
            ):
                primitive_class_budget_raw = xyzin_config.primitive_class_budget
            primitive_class_min = (
                xyzin_config.primitive_class_min
                if xyzin_config is not None
                and xyzin_config.primitive_class_min is not None
                and args.primitive_class_min == 0.70
                else args.primitive_class_min
            )
            primitive_class_cross_max = (
                xyzin_config.primitive_class_cross_max
                if xyzin_config is not None
                and xyzin_config.primitive_class_cross_max is not None
                and args.primitive_class_cross_max == 0.20
                else args.primitive_class_cross_max
            )
            class_budget = _primitive_class_budget(
                primitive_class_budget_raw,
                observations=observations,
                rotational_components=rotational_components,
            )
            preview = preview_semiexperimental_gics(Path(geometry_path), observations)
            class_plan = derive_primitive_class_plan(
                preview.gic_labels,
                primitive_classes,
                min_fraction=primitive_class_min,
                cross_fraction_max=primitive_class_cross_max,
                max_classes=class_budget,
            )
            fixed = _merge_unique(fixed, class_plan.fixed_patterns)
            parameter_classes = _merge_unique(parameter_classes, class_plan.parameter_classes)
            print(
                "primitive_class_plan: "
                f"classes={len(class_plan.parameter_classes)} "
                f"fixed={len(class_plan.fixed_patterns)} "
                f"rejected={len(class_plan.rejected_labels)}"
            )
            for item in class_plan.parameter_classes:
                print(
                    f"primitive_class: {item.name} "
                    f"gics={len(item.patterns)} patterns={'|'.join(item.patterns)}"
                )
            for line in primitive_class_decision_lines(class_plan):
                print(line)
        sensitivity_advisor_enabled = bool(args.sensitivity_advisor or legacy_robust_profile)
        sensitivity_apply_enabled = bool(args.apply_sensitivity_advisor or legacy_robust_profile)
        sensitivity_force_enabled = bool(args.force_sensitivity_advisor or legacy_robust_profile)
        sensitivity_fit_threshold = (
            1.1
            if legacy_robust_profile and args.sensitivity_fit_threshold == 0.15
            else args.sensitivity_fit_threshold
        )
        sensitivity_min_fit = (
            "none"
            if legacy_robust_profile and args.sensitivity_min_fit == "auto"
            else args.sensitivity_min_fit
        )
        advisor = None
        if sensitivity_advisor_enabled:
            if coordinate_model != "gic":
                raise ValueError(
                    "--sensitivity-advisor is only supported with --coordinate-model gic"
                )
            advisor_request = SemiexperimentalFitRequest(
                initial_geometry=geometry_path,
                observations=observations,
                fixed_parameters=fixed,
                observable=observable,
                rotational_components=rotational_components,
                qm_predicates=qm_predicates,
                parameter_classes=parameter_classes,
                coordinate_model=coordinate_model,
                robust_loss=robust_loss,
                robust_scale=robust_scale,
                leave_one_out=leave_one_out,
                excluded_rotational_constants=tuple(args.exclude_rotational_constant),
            )
            advisor = advise_semiexperimental_gic_sensitivity(
                advisor_request,
                step=step,
                fit_relative_threshold=sensitivity_fit_threshold,
                fixed_relative_threshold=args.sensitivity_fixed_threshold,
                min_fit_count=_sensitivity_min_fit_count(sensitivity_min_fit),
                distance_sigma_angstrom=args.sensitivity_distance_sigma,
                angle_sigma_degree=args.sensitivity_angle_sigma,
                torsion_sigma_degree=args.sensitivity_torsion_sigma,
                soft_predicate_scale=args.sensitivity_soft_predicate_scale,
                null_predicate_scale=args.sensitivity_null_predicate_scale,
                fit_regularization_scale=args.sensitivity_fit_regularization_scale,
            )
            args.outdir.mkdir(parents=True, exist_ok=True)
            advisor_path = args.outdir / "semiexp_sensitivity_advisor.csv"
            advisor_path.write_text(advisor.csv, encoding="utf-8")
            advisor_applied = False
            if sensitivity_apply_enabled:
                candidate_fixed = _merge_unique(fixed, advisor.fixed_patterns)
                candidate_qm_predicates = _merge_unique(qm_predicates, advisor.predicates)
                if sensitivity_force_enabled:
                    fixed = candidate_fixed
                    qm_predicates = candidate_qm_predicates
                    advisor_applied = True
                    _write_sensitivity_gate_summary(
                        args.outdir / "semiexp_sensitivity_gate.json",
                        {"accepted": True, "reason": "forced"},
                    )
                else:
                    gate = _sensitivity_safe_apply_gate(
                        base_request=advisor_request,
                        candidate_request=SemiexperimentalFitRequest(
                            initial_geometry=geometry_path,
                            observations=observations,
                            fixed_parameters=candidate_fixed,
                            observable=observable,
                            rotational_components=rotational_components,
                            qm_predicates=candidate_qm_predicates,
                            parameter_classes=parameter_classes,
                            coordinate_model=coordinate_model,
                            robust_loss=robust_loss,
                            robust_scale=robust_scale,
                            leave_one_out=leave_one_out,
                            excluded_rotational_constants=tuple(
                                args.exclude_rotational_constant
                            ),
                        ),
                        fit_semiexperimental_geometry=fit_semiexperimental_geometry,
                        outdir=args.outdir / "_sensitivity_gate",
                        max_iter=max_iter,
                        step=step,
                        damping=damping,
                        max_step=max_step,
                        prune_condition=prune_condition,
                        rot_rel_tol=args.sensitivity_gate_rot_rel_tol,
                        rot_abs_tol=args.sensitivity_gate_rot_abs_tol,
                        condition_factor=args.sensitivity_gate_condition_factor,
                        max_bond_delta=args.sensitivity_gate_max_bond_delta,
                        max_angle_delta=args.sensitivity_gate_max_angle_delta,
                    )
                    _write_sensitivity_gate_summary(
                        args.outdir / "semiexp_sensitivity_gate.json",
                        gate,
                    )
                    if gate["accepted"]:
                        fixed = candidate_fixed
                        qm_predicates = candidate_qm_predicates
                        advisor_applied = True
            print(
                "morpheus_sensitivity_advisor: "
                f"fit={advisor.fit_count} "
                f"predicate={advisor.predicate_count} "
                f"fixed={advisor.fixed_count} "
                f"applied={advisor_applied} "
                f"csv={advisor_path}"
            )
        if (
            coordinate_model == "gic"
            and not args.no_auto_stabilize
            and not qm_predicates
            and not parameter_classes
        ):
            preview = preview_semiexperimental_gics(Path(geometry_path), observations)
            if preview.suggested_classes:
                parameter_classes = _merge_unique(parameter_classes, preview.suggested_classes)
                print(f"morpheus_auto_advisor: parameter_classes={len(preview.suggested_classes)}")
                for item in preview.suggested_classes:
                    print(
                        f"morpheus_auto_class: {item.name} "
                        f"mode={item.mode} patterns={'|'.join(item.patterns)}"
                    )
            row_budget = len(observations) * len(
                _semiexp_components_for_budget(rotational_components)
            )
            if len(preview.gic_labels) > row_budget and HYDROGEN_PARAMETER_CONSTRAINT not in fixed:
                fixed = _merge_unique(fixed, (HYDROGEN_PARAMETER_CONSTRAINT,))
                print(
                    "morpheus_auto_stabilize: "
                    f"free_gic_parameters={len(preview.gic_labels)} "
                    f"fit_rows={row_budget}; action=fix_hydrogens"
                )
        request = SemiexperimentalFitRequest(
            initial_geometry=geometry_path,
            observations=observations,
            fixed_parameters=fixed,
            observable=observable,
            rotational_components=rotational_components,
            qm_predicates=qm_predicates,
            parameter_classes=parameter_classes,
            coordinate_model=coordinate_model,
            robust_loss=robust_loss,
            robust_scale=robust_scale,
            leave_one_out=leave_one_out,
            excluded_rotational_constants=tuple(args.exclude_rotational_constant),
        )
        fit_options = {
            "max_iter": max_iter,
            "step": step,
            "damping": damping,
            "max_step": max_step,
            "prune_condition": prune_condition,
            "checkpoint": checkpoint,
            "restart": restart,
            "outdir": args.outdir,
        }
        free_request = None
        regularization_predicates = tuple(
            predicate
            for predicate in request.qm_predicates
            if predicate.source == "morpheus_sensitivity_advisor_fit_regularization"
        )
        if args.compare_free_fit:
            if args.r0_preflight:
                raise ValueError("--compare-free-fit is incompatible with --r0-preflight")
            from dataclasses import replace as dataclass_replace

            free_request = (
                dataclass_replace(
                    request,
                    qm_predicates=tuple(
                        predicate
                        for predicate in request.qm_predicates
                        if predicate.source
                        != "morpheus_sensitivity_advisor_fit_regularization"
                    ),
                )
                if regularization_predicates
                else request
            )
        r0_report_result = None
        if args.r0_preflight:
            from dataclasses import replace as dataclass_replace

            preflight = fit_ground_state_r0_geometry(request, **fit_options)
            result = preflight.fit
            request = dataclass_replace(request, observations=preflight.observations)
            print("morpheus_fit_kind: R0_PRELIMINARY")
            for warning in preflight.warnings:
                print(f"morpheus_r0_warning: {warning}")
        else:
            if args.include_r0_report:
                r0_options = dict(fit_options)
                r0_options["checkpoint"] = None
                r0_options["restart"] = None
                r0_options["outdir"] = args.outdir / "_r0_report"
                r0_preflight = fit_ground_state_r0_geometry(request, **r0_options)
                r0_report_result = r0_preflight.fit
                print("morpheus_structural_path: INPUT -> R0 -> RS(KRAITCHMAN) -> RE(SE)")
                for warning in r0_preflight.warnings:
                    print(f"morpheus_r0_warning: {warning}")
            result = fit_semiexperimental_geometry(request, **fit_options)
        free_result = None
        if free_request is not None:
            if free_request is request:
                free_result = result
            else:
                free_options = dict(fit_options)
                free_options["checkpoint"] = None
                free_options["restart"] = None
                free_options["outdir"] = args.outdir / "_free_fit_comparison"
                free_result = fit_semiexperimental_geometry(free_request, **free_options)
        displacement_limit = args.max_atom_displacement
        if displacement_limit is None and legacy_robust_profile:
            displacement_limit = 3.0e-3
        safety: dict[str, object] | None = None
        if displacement_limit is not None:
            if displacement_limit <= 0.0:
                raise ValueError("--max-atom-displacement must be positive")
            max_displacement, rms_displacement = _semiexp_aligned_displacements(result)
            full_rank = result.diagnostics.rank == result.diagnostics.n_optimized_parameters
            well_conditioned = math.isfinite(result.diagnostics.condition_number) and (
                result.diagnostics.condition_number <= 1.0e8
            )
            reliable = bool(
                max_displacement <= float(displacement_limit)
                and result.stationary_point == "minimum"
                and full_rank
                and well_conditioned
            )
            safety = {
                "accepted": max_displacement <= float(displacement_limit),
                "reliable": reliable,
                "max_atom_displacement_A": max_displacement,
                "rms_atom_displacement_A": rms_displacement,
                "limit_A": float(displacement_limit),
                "stationary_point": result.stationary_point,
                "full_rank": full_rank,
                "condition_number": result.diagnostics.condition_number,
                "condition_limit": 1.0e8,
            }
            _write_sensitivity_gate_summary(
                args.outdir / "semiexp_geometry_safety.json",
                safety,
            )
            print(
                "morpheus_geometry_safety: "
                f"max_displacement_A={max_displacement:.9g} "
                f"rms_displacement_A={rms_displacement:.9g} "
                f"limit_A={float(displacement_limit):.9g} "
                f"accepted={safety['accepted']}"
            )
            if not safety["accepted"]:
                raise ValueError(
                    "MORPHEUS rejected the fitted structure: maximum aligned atom "
                    f"displacement {max_displacement:.6g} A exceeds the "
                    f"{float(displacement_limit):.6g} A safety limit"
                )
        fit_comparison = None
        if free_result is not None:
            fit_comparison = _semiexp_fit_comparison_contract(
                free_result=free_result,
                constrained_result=result,
                displacement_limit=float(displacement_limit),
                regularization_predicates=regularization_predicates,
                regularization_scale=float(args.sensitivity_fit_regularization_scale),
                excluded_rotational_constants=tuple(args.exclude_rotational_constant),
                advisor_rows=tuple(advisor.rows) if advisor is not None else (),
            )
            comparison_path = args.outdir / "semiexp_fit_comparison.json"
            _write_sensitivity_gate_summary(comparison_path, fit_comparison)
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json", "fit_comparison", comparison_path
            )
        report_path = write_semiexperimental_html_report(
            args.outdir / "semiexp_report.html",
            result,
            request,
            r0_result=r0_report_result,
            fit_comparison=fit_comparison,
        )
        tables_path = write_semiexperimental_standalone_latex(
            args.outdir / "semiexp_results.tex",
            result,
            request=request,
            safety=safety,
            r0_result=r0_report_result,
            fit_comparison=fit_comparison,
        )
        latex_pdf_path = _compile_semiexperimental_latex(tables_path)
        _append_manifest_output(args.outdir / "semiexp_manifest.json", "html_report", report_path)
        _append_manifest_output(args.outdir / "semiexp_manifest.json", "latex_tables", tables_path)
        _append_manifest_output(
            args.outdir / "semiexp_manifest.json", "latex_pdf", latex_pdf_path
        )
        delivery_input_path: Path | None = None
        delivery_geometry_input_path: Path | None = None
        if legacy_msr_job and args.job is not None:
            import shutil

            delivery_input_path = args.outdir / Path(args.job).name
            if delivery_input_path.resolve() != Path(args.job).resolve():
                shutil.copy2(args.job, delivery_input_path)
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json", "input_msr", delivery_input_path
            )
            if args.xyz is not None:
                delivery_geometry_input_path = args.outdir / Path(args.xyz).name
                if delivery_geometry_input_path.resolve() != Path(args.xyz).resolve():
                    shutil.copy2(args.xyz, delivery_geometry_input_path)
                _append_manifest_output(
                    args.outdir / "semiexp_manifest.json",
                    "input_geometry",
                    delivery_geometry_input_path,
                )
        if args.final_validation:
            validation_scales = (
                tuple(float(item) for item in args.validation_sigma_scale)
                if args.validation_sigma_scale
                else (0.5, 2.0)
            )
            if args.validation_no_predicate_scan:
                validation_scales = ()
            validation = run_semiexperimental_final_validation(
                request,
                result,
                args.outdir / "semiexp_final_validation",
                options=SemiexperimentalFinalValidationOptions(
                    coordinate_check=not args.validation_no_coordinate_check,
                    huber_check=not args.validation_no_huber_check,
                    predicate_scan_scales=validation_scales,
                    leave_predicate_groups=not args.validation_no_leave_predicate_groups,
                    max_predicate_groups=args.validation_max_predicate_groups,
                    multistart=args.validation_multistart,
                    multistart_sigma_angstrom=args.validation_multistart_sigma,
                    random_seed=args.validation_random_seed,
                ),
                max_iter=max_iter,
                step=step,
                damping=damping,
                max_step=max_step,
                prune_condition=prune_condition,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_summary",
                validation.summary_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_runs",
                validation.runs_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "predicate_audit",
                validation.predicate_audit_path,
            )
            _append_manifest_output(
                args.outdir / "semiexp_manifest.json",
                "final_validation_issues",
                validation.issues_path,
            )
            print(f"final_validation: {validation.summary_path}")
            print(f"final_validation_runs: {validation.runs_path}")
            print(f"final_validation_issues: {len(validation.issues)}")
        if args.xyzin is not None and not args.no_write_section:
            write_morpheus_section_from_result(
                preprocess.xyzin,
                result,
                outdir=args.outdir,
                backend=backend,
                source_path=args.job or args.xyz or observations_path,
                html_report_path=report_path,
                latex_tables_path=tables_path,
            )
            print(f"updated_morpheus_section: {preprocess.xyzin}")
        if safety and safety.get("reliable") and not args.keep_all_artifacts:
            delivery_files = {
                "semiexp_geometry.xyz",
                "semiexp_report.html",
                "semiexp_results.tex",
                "semiexp_results.pdf",
                "semiexp_manifest.json",
                "semiexp_geometry_safety.json",
            }
            extra_outputs: dict[str, Path] = {}
            if fit_comparison is not None:
                delivery_files.add("semiexp_fit_comparison.json")
                extra_outputs["fit_comparison"] = args.outdir / "semiexp_fit_comparison.json"
            if delivery_input_path is not None:
                delivery_files.add(delivery_input_path.name)
                extra_outputs["input_msr"] = delivery_input_path
            if delivery_geometry_input_path is not None:
                delivery_files.add(delivery_geometry_input_path.name)
                extra_outputs["input_geometry"] = delivery_geometry_input_path
            retained = _prune_semiexp_delivery_artifacts(
                args.outdir,
                delivery_files,
                extra_outputs=extra_outputs,
            )
            print(f"morpheus_delivery_cleanup: retained={','.join(retained)}")
        print(f"manifest: {result.manifest}")
        print(f"report: {report_path}")
        rms_label = (
            "rms_MHz"
            if result.diagnostics.observable == "rotational_constants"
            else "rms_observable"
        )
        print(f"{rms_label}: {result.rms_MHz:.8g}")
        rot_diffs = [row.difference_MHz for row in result.rotational_constants]
        rotational_rms = (
            math.sqrt(sum(diff * diff for diff in rot_diffs) / len(rot_diffs)) if rot_diffs else 0.0
        )
        rotational_mse = (
            sum(diff * diff for diff in rot_diffs) / len(rot_diffs) if rot_diffs else 0.0
        )
        print(f"rotational_rms_MHz: {rotational_rms:.8g}")
        print(f"rotational_mean_square_MHz2: {rotational_mse:.8g}")
        print(f"rotational_mean_square_1e3_MHz2: {1000.0 * rotational_mse:.8g}")
        print(f"iterations: {result.iterations}")
        print(f"stationary_point: {result.stationary_point}")
        print(f"convergence: {result.diagnostics.convergence_reason}")
        print(f"rank: {result.diagnostics.rank}")
        print(f"condition_number: {result.diagnostics.condition_number:.8g}")
        print(f"observable: {result.diagnostics.observable}")
        print(f"components: {','.join(result.diagnostics.components)}")
        print(f"backend: {backend}")
        print(f"coordinate_model: {result.diagnostics.coordinate_model}")
        return 0
    if args.command == "semiexp-ensemble":
        from matrix_core.manifest import build_run_manifest
        from matrix_morpheus import fit_ensemble_job

        result = fit_ensemble_job(args.job, outdir=args.outdir)
        outputs = _ensemble_output_paths(args.outdir)
        build_run_manifest(
            workflow="semiexp_ensemble",
            status=result.acceptance.status,
            run_dir=args.outdir,
            inputs={"job": args.job},
            outputs=outputs,
            parameters={
                "classes": len(result.classes),
                "molecules": len(result.molecule_blocks),
                "rank": result.rank,
                "scaled_condition_number": result.condition_number,
                "weighted_rms_before": result.weighted_rms_before,
                "weighted_rms_after": result.weighted_rms_after,
                "accepted": result.acceptance.accepted,
            },
            backend={"solver": "python", "model": "linearized shared class corrections"},
            messages=list(result.acceptance.reasons) + list(result.acceptance.review_items),
        ).write(args.outdir / "run_manifest.json")
        print(f"report: {args.outdir / 'ensemble_class_corrections.txt'}")
        print(f"manifest: {args.outdir / 'run_manifest.json'}")
        print(f"classes: {len(result.classes)}")
        print(f"molecules: {len(result.molecule_blocks)}")
        print(f"rank: {result.rank}")
        print(f"scaled_condition_number: {result.condition_number:.8g}")
        print(f"acceptance_status: {result.acceptance.status}")
        if result.acceptance.reasons:
            print("acceptance_failures: " + " | ".join(result.acceptance.reasons))
        if result.acceptance.review_items:
            print("acceptance_review: " + " | ".join(result.acceptance.review_items))
        print(f"weighted_rms_before: {result.weighted_rms_before:.8g}")
        print(f"weighted_rms_after: {result.weighted_rms_after:.8g}")
        for item in result.classes:
            print(
                f"class:{item.name}: correction={result.corrections[item.name]:.10g} "
                f"sigma={result.sigma[item.name]:.4g}"
            )
        return 0
    if args.command == "semiexp-ensemble-paper":
        from matrix_morpheus import write_ensemble_jpcl_artifacts

        artifacts = write_ensemble_jpcl_artifacts(
            args.job,
            args.paper_dir,
            outdir=args.outdir,
            soft_prior_sigma=args.soft_prior_sigma,
        )
        print(f"paper_dir: {args.paper_dir}")
        if args.outdir is not None:
            print(f"analysis_dir: {args.outdir}")
        for name, path in sorted(artifacts.items()):
            print(f"{name}: {path}")
        return 0
    if args.command == "semiexp-ensemble-prior-scan":
        from matrix_morpheus import run_ensemble_prior_scan

        kwargs = {}
        if args.sigma:
            kwargs["sigmas"] = tuple(args.sigma)
        rows = run_ensemble_prior_scan(args.job, args.outdir, **kwargs)
        print(f"rows: {len(rows)}")
        print(f"csv: {args.outdir / 'prior_sigma_scan.csv'}")
        print(f"json: {args.outdir / 'prior_sigma_scan.json'}")
        return 0
    if args.command == "semiexp-ensemble-synthon-scan":
        from matrix_morpheus import run_ensemble_synthon_threshold_scan

        kwargs = {}
        if args.threshold:
            kwargs["thresholds"] = tuple(args.threshold)
        rows = run_ensemble_synthon_threshold_scan(args.job, args.outdir, **kwargs)
        print(f"rows: {len(rows)}")
        print(f"csv: {args.outdir / 'synthon_threshold_scan.csv'}")
        print(f"json: {args.outdir / 'synthon_threshold_scan.json'}")
        return 0
    if args.command == "semiexp-benchmark":
        from matrix_morpheus import generate_paper_benchmark_artifacts

        snapshot, artifacts = generate_paper_benchmark_artifacts(
            snapshot_path=args.snapshot,
            outdir=args.outdir,
            refresh_from_outputs=not args.no_refresh,
            update_snapshot=args.update_snapshot,
        )
        print(f"cases: {len(snapshot.get('cases', {}))}")
        print(f"planar_diagnostics: {len(snapshot.get('planar_pair_diagnostics', {}))}")
        for name, path in sorted(artifacts.items()):
            print(f"{name}: {path}")
        return 0
    if args.command == "trinity" and args.trinity_command == "prepare":
        from matrix_trinity import prepare_trinity_section

        section = prepare_trinity_section(
            args.xyzin,
            run_dir=args.run_dir,
            engine_command=args.engine_command,
            coordinate_model=args.coordinate_model,
            active_space=args.active_space,
            max_steps=args.max_steps,
            trust_radius=args.trust_radius,
            gradient_tolerance=args.gradient_tolerance,
            step_tolerance=args.step_tolerance,
            energy_tolerance=args.energy_tolerance,
            energy_unit=args.energy_unit,
            gradient_unit=args.gradient_unit,
            external_protocol=args.external_protocol,
        )
        print(f"Updated #TRINITY: {args.xyzin}")
        print(f"manifest: {section.manifest_path}")
        print(f"run_dir: {section.run_dir}")
        print(f"coordinate_model: {section.coordinate_model}")
        print(f"engine_command: {section.engine_command}")
        return 0
    if args.command == "trinity" and args.trinity_command == "status":
        from matrix_trinity import read_trinity_section, trinity_section_summary_lines

        print("\n".join(trinity_section_summary_lines(read_trinity_section(args.xyzin))))
        return 0
    if args.command == "trinity" and args.trinity_command == "scan-prepare":
        from matrix_trinity import (
            PESExplorationPolicy,
            prepare_coordinate_scan,
            symmetric_displacements,
            write_scan_manifest,
        )

        policy = PESExplorationPolicy(retained_group=args.retained_group)
        direction = _trinity_scan_direction(
            args.xyzin,
            args.coordinate_kind,
            args.coordinate,
            retained_group=policy.retained_group,
        )
        displacements = (
            tuple(float(value) for value in args.displacement)
            if args.displacement
            else symmetric_displacements(args.step, args.points_each_side)
        )
        points = prepare_coordinate_scan(
            args.xyzin,
            direction,
            displacements,
            run_dir=args.run_dir,
            exploration_policy=policy,
        )
        manifest = write_scan_manifest(
            args.run_dir / "link_scan_manifest.json",
            xyzin_path=args.xyzin,
            direction=direction,
            points=points,
            engine_command=args.engine_command,
            external_protocol=args.external_protocol,
            exploration_policy=policy,
        )
        print(f"scan points: {len(points)}")
        print(f"manifest: {manifest}")
        print(f"coordinate: {direction.kind} {direction.label}")
        print(f"retained_group: {policy.retained_group}")
        return 0
    if args.command == "trinity" and args.trinity_command == "scan-run":
        from matrix_trinity import (
            PESExplorationPolicy,
            QMScanBackend,
            finite_difference_derivatives,
            prepare_coordinate_scan,
            run_external_scan_points,
            run_qm_scan_points,
            symmetric_displacements,
            write_finite_difference_derivatives,
            write_point_results_jsonl,
            write_scan_manifest,
        )

        if not args.backend and not args.engine_command.strip():
            raise ValueError("trinity scan-run needs --backend or --engine-command")
        policy = PESExplorationPolicy(retained_group=args.retained_group)
        direction = _trinity_scan_direction(
            args.xyzin,
            args.coordinate_kind,
            args.coordinate,
            retained_group=policy.retained_group,
        )
        points = prepare_coordinate_scan(
            args.xyzin,
            direction,
            symmetric_displacements(args.step, args.points_each_side),
            run_dir=args.run_dir,
            exploration_policy=policy,
        )
        if args.backend:
            results = run_qm_scan_points(
                args.xyzin,
                points,
                QMScanBackend(
                    name=args.backend,
                    route=args.route,
                    method=args.method,
                    basis=args.basis,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                    electronic_state=args.electronic_state,
                    excited_states=args.excited_states,
                    state_spin=args.state_spin,
                    freeze_core=args.freeze_core,
                    executable=args.executable,
                    force_field=args.force_field,
                    timeout=args.timeout,
                    extra_args=tuple(args.extra_arg),
                    gradient_mode=args.gradient_mode,
                    numerical_gradient_step_bohr=args.numerical_gradient_step_bohr,
                    numerical_gradient_stencil=args.numerical_gradient_stencil,
                    processors=args.backend_workers,
                    memory_gb=args.backend_memory_gb,
                ),
                run_dir=args.run_dir,
                exploration_policy=policy,
            )
            engine_command = f"matrix-qm-backend:{args.backend}"
        else:
            results = run_external_scan_points(
                points,
                engine_command=args.engine_command,
                timeout=args.timeout,
            )
            engine_command = args.engine_command
        log_path = write_point_results_jsonl(args.run_dir / "link_point_results.jsonl", results)
        manifest = write_scan_manifest(
            args.run_dir / "link_scan_manifest.json",
            xyzin_path=args.xyzin,
            direction=direction,
            points=points,
            engine_command=engine_command,
            exploration_policy=policy,
        )
        completed = tuple(result for result in results if result.status == "completed")
        print(f"scan points: {len(points)}")
        print(f"completed: {len(completed)}")
        print(f"results: {log_path}")
        print(f"manifest: {manifest}")
        print(f"retained_group: {policy.retained_group}")
        if len(completed) >= 2:
            derivatives = finite_difference_derivatives(
                completed,
                coordinate_label=direction.label,
                max_order=min(4, len(completed) - 1),
            )
            derivatives_path = write_finite_difference_derivatives(
                args.run_dir / "link_finite_difference_derivatives.json",
                derivatives,
            )
            print(f"derivatives: {derivatives_path}")
            if derivatives.energy_derivatives_hartree:
                print(
                    "energy derivatives hartree: "
                    + " ".join(f"{value:.12g}" for value in derivatives.energy_derivatives_hartree)
                )
        return 0
    if (args.command == "link" and args.link_command == "driver-run") or (
        args.command == "trinity" and args.trinity_command == "driver-run"
    ):
        from matrix_trinity import (
            QMScanBackend,
            active_variable_contract_from_file,
            coordinate_model_from_xyzin,
            run_external_driver_loop,
        )

        backend = None
        if args.backend:
            backend = QMScanBackend(
                name=args.backend,
                route=args.route,
                method=args.method,
                basis=args.basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                force_field=args.force_field,
                timeout=args.timeout,
                extra_args=tuple(args.extra_arg),
            )
        calculator_backends = {}
        for record in args.calculator_profile:
            calculator_id, separator, backend_name = record.partition("=")
            calculator_id = calculator_id.strip()
            backend_name = backend_name.strip()
            if not separator or not calculator_id or not backend_name:
                raise ValueError("--calculator-profile must have the form ID=BACKEND")
            if calculator_id in calculator_backends or calculator_id == "link-default":
                raise ValueError(f"duplicate or reserved calculator profile: {calculator_id}")
            calculator_backends[calculator_id] = QMScanBackend(
                name=backend_name,
                route=args.route,
                method=args.method,
                basis=args.basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                timeout=args.timeout,
                extra_args=tuple(args.extra_arg),
                force_field=args.force_field,
            )
        calculator_commands = {}
        for record in args.calculator_command:
            calculator_id, separator, command = record.partition("=")
            calculator_id = calculator_id.strip()
            command = command.strip()
            if not separator or not calculator_id or not command:
                raise ValueError("--calculator-command must have the form ID=COMMAND")
            if (
                calculator_id in calculator_commands
                or calculator_id in calculator_backends
                or calculator_id == "link-default"
            ):
                raise ValueError(f"duplicate or reserved calculator profile: {calculator_id}")
            calculator_commands[calculator_id] = command
        if args.variables and args.coordinate:
            raise ValueError("use either --variables or --coordinate, not both")
        variable_contract = (
            active_variable_contract_from_file(
                args.xyzin,
                args.variables,
                retained_group=args.retained_group,
                pes_exploration=True,
            )
            if args.variables
            else None
        )
        model = (
            variable_contract.model
            if variable_contract is not None
            else coordinate_model_from_xyzin(
                args.xyzin,
                kind="sonic",
                coordinates=tuple(args.coordinate),
                pes_exploration=True,
                retained_group=args.retained_group,
            )
        )
        result = run_external_driver_loop(
            args.xyzin,
            run_dir=args.run_dir,
            driver_command=args.driver_command,
            coordinate_model=model,
            engine_command=args.engine_command,
            backend=backend,
            timeout=args.timeout,
            max_cycles=args.max_cycles,
            batch_workers=args.batch_workers,
            initial_evaluation_owner=args.initial_evaluation_owner,
            initial_properties=tuple(args.property or ("energy",)),
            active_variable_contract=variable_contract,
            calculator_backends=calculator_backends,
            calculator_engine_commands=calculator_commands,
            resume=args.resume,
            run_id=args.run_id,
            retained_group=args.retained_group,
        )
        print(f"status: {result.status}")
        print(f"cycles: {result.cycles}")
        print(f"points: {result.point_count}")
        print(f"completed: {result.completed_point_count}")
        print(f"summary: {result.summary_path}")
        print(f"trace: {result.trace_path}")
        print(f"retained_group: {args.retained_group}")
        return 0 if result.status in {"complete", "completed", "stop"} else 2
    if args.command == "link" and args.link_command == "mock-sentinel":
        from matrix_trinity.mock_sentinel import main as mock_sentinel_main

        command = [str(args.request), str(args.response), "--mode", args.mode]
        command.extend(("--batch-size", str(args.batch_size)))
        if args.driver_owned:
            command.append("--driver-owned")
        return mock_sentinel_main(command)
    if args.command == "trinity" and args.trinity_command == "optimize":
        from matrix_trinity import run_optimization_input

        result = run_optimization_input(args.input, run_dir=args.run_dir)
        print(f"status: {result.status}")
        print(f"iterations: {len(result.iterations)}")
        print(f"final energy hartree: {result.final_energy_hartree:.12f}")
        print(f"log: {(args.run_dir / 'optimization.log').resolve()}")
        return 0 if result.converged else 2
    if args.command == "trinity" and args.trinity_command == "optimize-run":
        from matrix_trinity import (
            QMScanBackend,
            OptimizerSettings,
            build_optimizer_hessian_seed,
            coordinate_model_from_xyzin,
            optimize_geometry,
            optimizer_hessian_from_engine_hessian,
            optimizer_hessian_from_gaussian_hessian,
            read_cartesian_gradient,
            read_optimizer_hessian,
        )

        if not args.backend and not args.engine_command.strip():
            raise ValueError("trinity optimize-run needs --backend or --engine-command")
        backend = None
        if args.backend:
            backend = QMScanBackend(
                name=args.backend,
                route=args.route,
                method=args.method,
                basis=args.basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                timeout=args.timeout,
                force_field=args.force_field,
                extra_args=tuple(args.extra_arg),
                gradient_mode=args.gradient_mode,
                numerical_gradient_step_bohr=args.numerical_gradient_step_bohr,
                numerical_gradient_stencil=args.numerical_gradient_stencil,
                processors=args.backend_workers,
                memory_gb=args.backend_memory_gb,
                electronic_state=args.electronic_state,
                excited_states=args.excited_states,
                state_spin=args.state_spin,
                freeze_core=args.freeze_core,
            )
        convergence = {
            "normal": (1.0e-6, 4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3),
            "tight": (1.0e-8, 1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5),
        }[args.convergence]
        settings = OptimizerSettings(
            max_steps=args.max_steps,
            trust_radius=args.trust_radius,
            max_trust_radius=args.max_trust_radius,
            gradient_tolerance=args.gradient_tolerance,
            step_tolerance=args.step_tolerance,
            energy_tolerance=(
                convergence[0]
                if args.convergence == "tight" and args.energy_tolerance == 1.0e-6
                else args.energy_tolerance
            ),
            max_force_tolerance=(
                convergence[1] if args.max_force_tolerance is None else args.max_force_tolerance
            ),
            rms_force_tolerance=(
                convergence[2] if args.rms_force_tolerance is None else args.rms_force_tolerance
            ),
            max_displacement_tolerance=(
                convergence[3]
                if args.max_displacement_tolerance is None
                else args.max_displacement_tolerance
            ),
            rms_displacement_tolerance=(
                convergence[4]
                if args.rms_displacement_tolerance is None
                else args.rms_displacement_tolerance
            ),
            fd_step=args.fd_step,
            fd_hard_characteristic_scale=args.fd_hard_characteristic_scale,
            fd_soft_characteristic_scale=args.fd_soft_characteristic_scale,
            fd_min_step=args.fd_min_step,
            fd_max_step=args.fd_max_step,
            energy_noise=args.energy_noise,
            energy_noise_samples=args.auto_energy_noise_samples,
            adaptive_fd_mode=args.adaptive_fd_mode,
            fd_central_gradient_factor=args.fd_central_gradient_factor,
            selective_fd_refresh=args.selective_fd_refresh,
            fd_refresh_interval=args.fd_refresh_interval,
            fd_gradient_change_tolerance=args.fd_gradient_change_tolerance,
            selective_min_refresh_fraction=args.selective_min_refresh_fraction,
            selective_coupling_threshold=args.selective_coupling_threshold,
            selective_fallback_rejections=args.selective_fallback_rejections,
            selective_fallback_gradient_growth=args.selective_fallback_gradient_growth,
            surrogate_max_samples=args.surrogate_max_samples,
            fd_parallel_workers=args.fd_parallel_workers,
            hessian_coupling_threshold=args.hessian_coupling_threshold,
            sparse_hessian_updates=args.sparse_hessian_updates,
            symmetry_reduction=args.symmetry_reduction,
            two_sided=not args.one_sided,
            prefer_analytic_gradient=not args.no_analytic_gradient,
            cache_tolerance=args.cache_tolerance,
            resume=args.resume,
            min_hessian_eigenvalue=args.min_hessian_eigenvalue,
            max_hessian_condition=args.max_hessian_condition,
            max_coordinate_step=args.max_coordinate_step,
            line_search_reductions=args.line_search_reductions,
            energy_increase_tolerance=args.energy_increase_tolerance,
            hessian_update=args.hessian_update,
            initial_hessian_model=args.initial_hessian_model,
            enable_gdiis=args.enable_gdiis,
            coordinate_drift_warning=args.coordinate_drift_warning,
            fragment_radial_curvature=args.fragment_radial_curvature,
            fragment_tangential_curvature=args.fragment_tangential_curvature,
            fragment_rotation_curvature=args.fragment_rotation_curvature,
            coordinate_schedule=args.coordinate_schedule,
            coordinate_phase_max_steps=args.coordinate_phase_max_steps,
            coordinate_phase_gradient_factor=args.coordinate_phase_gradient_factor,
            backtransform_continuation_step=args.backtransform_continuation_step,
            backtransform_max_substeps=args.backtransform_max_substeps,
            include_cv_exponential_field=args.core_valence_exponential,
        )
        if args.variables and args.coordinate:
            raise ValueError("use either --variables or --coordinate, not both")
        if args.variables:
            from matrix_trinity import active_variable_contract_from_file

            coordinate_model = active_variable_contract_from_file(args.xyzin, args.variables).model
        else:
            coordinate_model = coordinate_model_from_xyzin(
                args.xyzin,
                kind=args.coordinate_kind,
                coordinates=tuple(args.coordinate),
            )
        seed_job_requested = bool(
            args.initial_hessian_seed_backend or args.initial_hessian_seed_command
        )
        if args.initial_hessian_gradient_file is not None and not args.initial_hessian_file:
            raise ValueError(
                "--initial-hessian-gradient-file requires --initial-hessian-file"
            )
        seed_options = [
            bool(args.initial_hessian),
            bool(args.initial_hessian_file and not seed_job_requested),
            bool(args.initial_hessian_gaussian),
            seed_job_requested,
            bool(args.initial_hessian_gaussian_route),
        ]
        if sum(seed_options) > 1:
            raise ValueError(
                "use only one initial Hessian source: --initial-hessian, "
                "--initial-hessian-file, --initial-hessian-gaussian, "
                "--initial-hessian-seed-backend/--initial-hessian-seed-command, "
                "or --initial-hessian-gaussian-route"
            )
        initial_hessian = None
        initial_hessian_source = "metric-diagonal"
        if args.initial_hessian_gaussian:
            print(
                "warning: --initial-hessian-gaussian is deprecated; use "
                "--initial-hessian-engine gaussian --initial-hessian-file",
                file=sys.stderr,
            )
        if args.initial_hessian_gaussian_route:
            print(
                "warning: --initial-hessian-gaussian-route is deprecated; use "
                "--initial-hessian-seed-backend gaussian --initial-hessian-seed-route",
                file=sys.stderr,
            )
        if args.initial_hessian:
            initial_hessian = read_optimizer_hessian(
                args.initial_hessian,
                expected_labels=coordinate_model.labels,
            )
            initial_hessian_source = f"optimizer-hessian {args.initial_hessian}"
        elif args.initial_hessian_file:
            if not args.initial_hessian_engine:
                raise ValueError("--initial-hessian-file needs --initial-hessian-engine")
            imported_gradient = (
                None
                if args.initial_hessian_gradient_file is None
                else read_cartesian_gradient(args.initial_hessian_gradient_file)
            )
            initial_hessian = optimizer_hessian_from_engine_hessian(
                args.initial_hessian_engine,
                args.initial_hessian_file,
                coordinate_model,
                grd=args.initial_hessian_cfour_grd,
                output=args.initial_hessian_cfour_output,
                cartesian_gradient_hartree_per_bohr=imported_gradient,
                use_b_prime=imported_gradient is not None,
                b_prime_parallel_workers=args.initial_hessian_b_prime_workers,
            )
            initial_hessian_source = (
                f"{args.initial_hessian_engine}-hessian {args.initial_hessian_file}"
                + (" + architect-b-prime" if imported_gradient is not None else "")
            )
        elif args.initial_hessian_gaussian:
            initial_hessian = optimizer_hessian_from_gaussian_hessian(
                args.initial_hessian_gaussian,
                coordinate_model,
            )
            initial_hessian_source = f"gaussian-hessian {args.initial_hessian_gaussian}"
        elif args.initial_hessian_seed_backend or args.initial_hessian_seed_command:
            seed_engine = args.initial_hessian_seed_backend
            if not seed_engine:
                raise ValueError(
                    "--initial-hessian-seed-command needs --initial-hessian-seed-backend"
                )
            seed_dir = args.initial_hessian_run_dir or (args.run_dir / "hessian_seed")
            initial_hessian, seed_output = build_optimizer_hessian_seed(
                args.xyzin,
                coordinate_model,
                engine=seed_engine,
                run_dir=seed_dir,
                route=args.initial_hessian_seed_route,
                method=args.initial_hessian_seed_method,
                basis=args.initial_hessian_seed_basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                timeout=args.timeout,
                engine_command=args.initial_hessian_seed_command,
                hessian_path=args.initial_hessian_file,
            )
            initial_hessian_source = f"{seed_engine}-hessian-seed {seed_output}"
        elif args.initial_hessian_gaussian_route:
            seed_dir = args.initial_hessian_run_dir or (args.run_dir / "hessian_seed")
            initial_hessian, seed_log = build_optimizer_hessian_seed(
                args.xyzin,
                coordinate_model,
                engine="gaussian",
                run_dir=seed_dir,
                route=args.initial_hessian_gaussian_route,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                timeout=args.timeout,
            )
            initial_hessian_source = f"gaussian-hessian-job {seed_log}"
        result = optimize_geometry(
            args.xyzin,
            run_dir=args.run_dir,
            coordinate_model=coordinate_model,
            engine_command=args.engine_command,
            backend=backend,
            settings=settings,
            timeout=args.timeout,
            initial_hessian=initial_hessian,
            initial_hessian_source=initial_hessian_source,
        )
        if args.optimized_xyzin is not None:
            import shutil

            from matrix_core import replace_xyz_block
            from matrix_chem import MolecularGeometry, read_enriched_xyz

            optimized_xyzin = args.optimized_xyzin.expanduser().resolve()
            optimized_xyzin.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.xyzin, optimized_xyzin)
            geometry = read_enriched_xyz(args.xyzin)
            optimized_geometry = MolecularGeometry(
                atoms=geometry.atoms,
                coordinates_angstrom=result.final_coordinates_angstrom,
                comment=f"MATRIX LINK optimized geometry; source={args.xyzin}",
                source_format="link_optimized",
                source_path=optimized_xyzin,
                charge=geometry.charge,
                multiplicity=geometry.multiplicity,
            )
            replace_xyz_block(
                optimized_xyzin,
                optimized_geometry.xyz_lines(),
            )
            print(f"optimized_xyzin: {optimized_xyzin}")
        print(f"status: {result.status}")
        print(f"converged: {int(result.converged)}")
        print(f"final_energy_hartree: {result.final_energy_hartree:.12g}")
        print(f"optimization_steps: {len(result.iterations)}")
        print(f"gaussian_equivalent_steps: {len(result.iterations)}")
        print(f"qm_evaluations: {result.qm_evaluations}")
        print(f"energy_evaluations: {result.energy_evaluations}")
        print(f"gradient_evaluations: {result.gradient_evaluations}")
        print(f"hessian_evaluations: {result.hessian_evaluations}")
        print(f"fd_displacements: {result.fd_displacements}")
        print(f"cache_hits: {result.cache_hits}")
        print(f"cache: {result.cache_path}")
        print(f"trajectory: {result.trajectory_path}")
        print(f"trace: {result.trace_path}")
        print(f"summary: {result.summary_path}")
        print(f"final_hessian: {result.final_hessian_path}")
        return 0
    if args.command == "trinity" and args.trinity_command == "optimize-gf":
        import shutil

        from matrix_core.xyzin_geometry import replace_xyzin_geometry
        from matrix_gf import (
            run_xyzin_gf_report_from_engine_hessian,
            run_xyzin_gf_report_from_fchk,
            run_xyzin_gf_report_from_xyzin,
            write_csv_tables,
            write_gf_ped_section_from_report,
        )
        from matrix_trinity import (
            QMScanBackend,
            OptimizerSettings,
            build_optimizer_hessian_seed,
            coordinate_model_from_xyzin,
            optimize_geometry,
            optimizer_hessian_from_engine_hessian,
            read_optimizer_hessian,
        )

        if not args.backend and not args.engine_command.strip():
            raise ValueError("trinity optimize-gf needs --backend or --engine-command")
        backend = None
        if args.backend:
            backend = QMScanBackend(
                name=args.backend,
                route=args.route,
                method=args.method,
                basis=args.basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                timeout=args.timeout,
                force_field=args.force_field,
            )
        settings = OptimizerSettings(
            max_steps=args.max_steps,
            trust_radius=args.trust_radius,
            max_trust_radius=args.max_trust_radius,
            gradient_tolerance=args.gradient_tolerance,
            step_tolerance=args.step_tolerance,
            energy_tolerance=args.energy_tolerance,
            fd_step=args.fd_step,
            fd_min_step=args.fd_min_step,
            fd_max_step=args.fd_max_step,
            energy_noise=args.energy_noise,
            adaptive_fd_mode=args.adaptive_fd_mode,
            selective_fd_refresh=args.selective_fd_refresh,
            fd_refresh_interval=args.fd_refresh_interval,
            fd_gradient_change_tolerance=args.fd_gradient_change_tolerance,
            selective_min_refresh_fraction=args.selective_min_refresh_fraction,
            selective_coupling_threshold=args.selective_coupling_threshold,
            selective_fallback_rejections=args.selective_fallback_rejections,
            selective_fallback_gradient_growth=args.selective_fallback_gradient_growth,
            surrogate_max_samples=args.surrogate_max_samples,
            fd_parallel_workers=args.fd_parallel_workers,
            hessian_coupling_threshold=args.hessian_coupling_threshold,
            sparse_hessian_updates=args.sparse_hessian_updates,
            prefer_analytic_gradient=not args.no_analytic_gradient,
            cache_tolerance=args.cache_tolerance,
            min_hessian_eigenvalue=args.min_hessian_eigenvalue,
            max_hessian_condition=args.max_hessian_condition,
            max_coordinate_step=args.max_coordinate_step,
            line_search_reductions=args.line_search_reductions,
            energy_increase_tolerance=args.energy_increase_tolerance,
            hessian_update=args.hessian_update,
            coordinate_drift_warning=args.coordinate_drift_warning,
            fragment_radial_curvature=args.fragment_radial_curvature,
            fragment_tangential_curvature=args.fragment_tangential_curvature,
            fragment_rotation_curvature=args.fragment_rotation_curvature,
            coordinate_schedule=args.coordinate_schedule,
            coordinate_phase_max_steps=args.coordinate_phase_max_steps,
            coordinate_phase_gradient_factor=args.coordinate_phase_gradient_factor,
            backtransform_continuation_step=args.backtransform_continuation_step,
            backtransform_max_substeps=args.backtransform_max_substeps,
        )
        coordinate_model = coordinate_model_from_xyzin(
            args.xyzin,
            kind=args.coordinate_kind,
            coordinates=tuple(args.coordinate),
        )
        initial_hessian = None
        initial_hessian_source = "metric-diagonal"
        if args.initial_hessian:
            initial_hessian = read_optimizer_hessian(
                args.initial_hessian,
                expected_labels=coordinate_model.labels,
            )
            initial_hessian_source = f"optimizer-hessian {args.initial_hessian}"
        elif args.initial_hessian_file:
            if not args.initial_hessian_engine:
                raise ValueError("--initial-hessian-file needs --initial-hessian-engine")
            initial_hessian = optimizer_hessian_from_engine_hessian(
                args.initial_hessian_engine,
                args.initial_hessian_file,
                coordinate_model,
                grd=args.initial_hessian_cfour_grd,
                output=args.initial_hessian_cfour_output,
            )
            initial_hessian_source = (
                f"{args.initial_hessian_engine}-hessian {args.initial_hessian_file}"
            )
        elif args.initial_hessian_seed_backend or args.initial_hessian_seed_command:
            seed_engine = args.initial_hessian_seed_backend
            if not seed_engine:
                raise ValueError(
                    "--initial-hessian-seed-command needs --initial-hessian-seed-backend"
                )
            seed_dir = args.initial_hessian_run_dir or (args.run_dir / "hessian_seed")
            initial_hessian, seed_output = build_optimizer_hessian_seed(
                args.xyzin,
                coordinate_model,
                engine=seed_engine,
                run_dir=seed_dir,
                route=args.initial_hessian_seed_route,
                method=args.initial_hessian_seed_method,
                basis=args.initial_hessian_seed_basis,
                charge=args.charge,
                multiplicity=args.multiplicity,
                executable=args.executable,
                timeout=args.timeout,
                engine_command=args.initial_hessian_seed_command,
                hessian_path=args.initial_hessian_file,
            )
            initial_hessian_source = f"{seed_engine}-hessian-seed {seed_output}"
        result = optimize_geometry(
            args.xyzin,
            run_dir=args.run_dir,
            coordinate_model=coordinate_model,
            engine_command=args.engine_command,
            backend=backend,
            settings=settings,
            timeout=args.timeout,
            initial_hessian=initial_hessian,
            initial_hessian_source=initial_hessian_source,
        )
        optimized_xyzin = args.optimized_xyzin or (args.run_dir / "optimized.xyzin")
        optimized_xyzin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.xyzin, optimized_xyzin)
        replace_xyzin_geometry(
            optimized_xyzin,
            result.atoms,
            result.final_coordinates_angstrom,
            comment=f"MATRIX optimized geometry; source={args.xyzin}",
        )
        gf_source_kind = "xyzin"
        gf_source_path = optimized_xyzin
        if args.gf_fchk is not None and args.gf_hessian_file is not None:
            raise ValueError("trinity optimize-gf uses either --gf-fchk or --gf-hessian-file")
        if args.gf_fchk is not None:
            report = run_xyzin_gf_report_from_fchk(
                args.gf_fchk,
                optimized_xyzin,
                block_by_irrep=args.gf_symmetry_blocks,
            )
            gf_source_kind = "fchk"
            gf_source_path = args.gf_fchk
        elif args.gf_hessian_file is not None:
            if not args.gf_hessian_engine:
                raise ValueError("--gf-hessian-file needs --gf-hessian-engine")
            if args.gf_write_hessian_section:
                from matrix_qm import (
                    cartesian_hessian_section_from_hessian_input,
                    hessian_input_from_engine,
                    write_cartesian_hessian_section,
                )

                hessian_input = hessian_input_from_engine(
                    args.gf_hessian_engine,
                    args.gf_hessian_file,
                    grd=args.gf_hessian_cfour_grd,
                    output=args.gf_hessian_cfour_output,
                )
                write_cartesian_hessian_section(
                    optimized_xyzin,
                    cartesian_hessian_section_from_hessian_input(
                        hessian_input,
                        source=f"{args.gf_hessian_engine}-hessian {args.gf_hessian_file}",
                    ),
                )
            report = run_xyzin_gf_report_from_engine_hessian(
                args.gf_hessian_engine,
                args.gf_hessian_file,
                optimized_xyzin,
                cfour_grd=args.gf_hessian_cfour_grd,
                cfour_output=args.gf_hessian_cfour_output,
                block_by_irrep=args.gf_symmetry_blocks,
            )
            gf_source_kind = str(args.gf_hessian_engine)
            gf_source_path = args.gf_hessian_file
        else:
            report = run_xyzin_gf_report_from_xyzin(
                optimized_xyzin,
                block_by_irrep=args.gf_symmetry_blocks,
            )
        if args.gf_out is not None:
            args.gf_out.parent.mkdir(parents=True, exist_ok=True)
            args.gf_out.write_text(report.text + "\n", encoding="utf-8")
        if args.gf_csv_dir is not None:
            write_csv_tables(report, args.gf_csv_dir, prefix="gic_gf")
        if not args.no_gf_section:
            write_gf_ped_section_from_report(
                optimized_xyzin,
                report,
                source_kind=gf_source_kind,
                source_path=gf_source_path,
                report_path=args.gf_out,
                csv_dir=args.gf_csv_dir,
            )
        workflow_path = args.run_dir / "optimize_gf_workflow.json"
        geometry_check = None
        if getattr(report, "geometry_comparison", None) is not None:
            geometry_check = {
                "raw_rms_angstrom": report.geometry_comparison.raw_rms_angstrom,
                "raw_max_angstrom": report.geometry_comparison.raw_max_angstrom,
                "aligned_rms_angstrom": report.geometry_comparison.aligned_rms_angstrom,
                "aligned_max_angstrom": report.geometry_comparison.aligned_max_angstrom,
                "warning": report.geometry_comparison.warning,
            }
        workflow_path.write_text(
            json.dumps(
                {
                    "schema": "matrix.trinity.optimize_gf.workflow.v1",
                    "optimizer_summary": str(result.summary_path),
                    "optimizer_final_hessian": str(result.final_hessian_path),
                    "optimized_xyzin": str(optimized_xyzin),
                    "gf_hessian_source_kind": gf_source_kind,
                    "gf_hessian_source_path": str(gf_source_path),
                    "gf_coordinate_geometry": str(optimized_xyzin),
                    "gf_geometry_rule": "B and G are built from the optimized xyzin geometry; external Cartesian Hessian coordinates are reported by geometry_check.",
                    "geometry_check": geometry_check,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"optimizer_status: {result.status}")
        print(f"optimizer_converged: {int(result.converged)}")
        print(f"optimizer_summary: {result.summary_path}")
        print(f"optimizer_final_hessian: {result.final_hessian_path}")
        print(f"optimized_xyzin: {optimized_xyzin}")
        print(f"optimize_gf_workflow: {workflow_path}")
        if args.gf_out is not None:
            print(f"gf_report: {args.gf_out}")
        if args.gf_csv_dir is not None:
            print(f"gf_csv_dir: {args.gf_csv_dir}")
        return 0
    if args.command == "trinity" and args.trinity_command == "benchmark-optimizer":
        from matrix_trinity import run_optimizer_validation_benchmark

        report = run_optimizer_validation_benchmark(
            args.run_dir,
            max_steps=args.max_steps,
            include_sonic=args.include_sonic,
        )
        completed = sum(1 for item in report.runs if item.converged)
        print(f"benchmark_runs: {len(report.runs)}")
        print(f"converged: {completed}")
        print(f"json: {report.json_path}")
        print(f"markdown: {report.markdown_path}")
        return 0
    if args.command == "trinity" and args.trinity_command == "optimize-chain":
        from matrix_trinity import optimization_level_from_json, optimize_from_smiles_multilevel

        level_payloads = list(args.level)
        if args.level_file is not None:
            payload = json.loads(args.level_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("levels", [])
            if not isinstance(payload, list):
                raise ValueError("--level-file must contain a JSON list or {'levels': [...]}")
            level_payloads.extend(json.dumps(item) for item in payload)
        if not level_payloads:
            raise ValueError(
                "trinity optimize-chain needs at least one --level or --level-file entry"
            )
        result = optimize_from_smiles_multilevel(
            args.smiles,
            run_dir=args.run_dir,
            levels=tuple(optimization_level_from_json(item) for item in level_payloads),
            title=args.title,
            charge=args.charge,
            multiplicity=args.multiplicity,
            random_seed=args.random_seed,
        )
        print(f"initial_xyzin: {result.initial_xyzin}")
        print(f"final_xyzin: {result.final_xyzin}")
        print(f"manifest: {result.manifest_path}")
        print(f"report: {result.report_text_path}")
        print(f"report_json: {result.report_json_path}")
        for item in result.results:
            print(
                f"level: {item.summary_path.parent.name} status={item.status} "
                f"steps={len(item.iterations)} qm_evaluations={item.qm_evaluations}"
            )
        return 0
    if args.command == "trinity" and args.trinity_command == "ir-from-fchk":
        from matrix_trinity import dipole_surface_and_ir_from_gaussian_fchk

        result = dipole_surface_and_ir_from_gaussian_fchk(
            args.fchk,
            displacement=args.displacement,
            symmetry_tolerance=args.symmetry_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.surface_output is not None:
            args.surface_output.parent.mkdir(parents=True, exist_ok=True)
            args.surface_output.write_text(
                json.dumps(
                    result.property_surface.to_dict(), indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
        print(args.output)
        for label, frequency, intensity in zip(
            result.mode_labels,
            result.frequencies_cm1,
            result.intensities_km_mol,
            strict=True,
        ):
            print(f"{label} {frequency:12.4f} cm-1 {intensity:12.6f} km mol-1")
        return 0
    if args.command == "trinity" and args.trinity_command == "report":
        from matrix_trinity import write_optimization_report

        text_path, json_path = write_optimization_report(
            args.manifest,
            text_path=args.out,
            json_path=args.json_out,
            title=args.title,
            charge=args.charge,
            multiplicity=args.multiplicity,
            initial_geometry_source=args.initial_geometry_source,
        )
        print(f"report: {text_path}")
        print(f"report_json: {json_path}")
        return 0
    if args.command == "multistructure-reference-search":
        from matrix_morpheus import search_reference_library

        result = search_reference_library(
            args.query_xyz,
            library_root=args.library_root,
            top_k=args.top_k,
            include_ring_comparison=not args.no_ring_comparison,
            ring_weight=args.ring_weight,
            outdir=args.outdir,
        )
        print(f"reference_library: {result.library_root}")
        print(f"matches: {len(result.matches)}")
        print(f"skipped: {len(result.skipped)}")
        if result.matches:
            top = result.matches[0]
            print(f"top_match: {top.slug} similarity={top.similarity_combined:.8g}")
        print(f"outputs: {args.outdir}")
        return 0
    if args.command == "multistructure-build-reference-geometry":
        from matrix_morpheus import build_reference_assisted_geometry

        apply_kinds = tuple(args.apply_kind) if args.apply_kind else None
        kwargs = {}
        if apply_kinds is not None:
            kwargs["apply_kinds"] = apply_kinds
        result = build_reference_assisted_geometry(
            args.query_xyz,
            library_root=args.library_root,
            top_library_matches=args.top_library_matches,
            max_fragment_matches=args.max_fragment_matches,
            min_fragment_support=args.min_fragment_support,
            zeff_threshold=args.zeff_threshold,
            include_ring_comparison=not args.no_ring_comparison,
            ring_weight=args.ring_weight,
            outdir=args.outdir,
            **kwargs,
        )
        print(f"reference_library: {result.library_root}")
        print(f"targets: {len(result.targets)}")
        print(f"unmatched: {len(result.unmatched)}")
        print(f"iterations: {result.iterations}")
        print(f"rms_target_residual_final: {result.rms_target_residual_final:.8g}")
        print(f"outputs: {args.outdir}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "plan":
        from matrix_smith import write_gicforge_plan_sections

        plan_kwargs = {
            "symmetrize": args.symmetrize,
            "sycart": args.sycart,
            "fragment_mode": args.fragment_mode,
        }
        _add_xh_stretch_kwargs(args, plan_kwargs)
        write_gicforge_plan_sections(
            args.xyzin,
            **plan_kwargs,
        )
        print(f"Planned GICForge workflow: {args.xyzin}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "build":
        from matrix_smith import LocalSALCSettings, write_gicforge_build_sections

        build_kwargs = {
            "symmetrize": args.symmetrize,
            "sycart": args.sycart,
        }
        if args.local_salc:
            build_kwargs["local_salc"] = True
            build_kwargs["local_salc_settings"] = LocalSALCSettings(
                zeff_tolerance=args.local_zeff_tolerance,
                distance_tolerance_angstrom=args.local_distance_tolerance,
                template_rms_threshold=args.local_template_rms_threshold,
                template_min_margin=args.local_template_margin,
                angle_class_tolerance=args.local_angle_class_tolerance,
            )
        if args.fragment_mode:
            build_kwargs["fragment_mode"] = args.fragment_mode
        if args.symmetry_group:
            build_kwargs["symmetry_group"] = args.symmetry_group
        _add_xh_stretch_kwargs(args, build_kwargs)
        definition = write_gicforge_build_sections(
            args.xyzin,
            **build_kwargs,
        )
        print(
            "Built GICForge definition: "
            f"{args.xyzin} (GICs={len(definition.gics)}, rank={definition.rank})"
        )
        if not args.no_diagnostics and args.xyzin.is_file():
            from matrix_smith import write_sonic_diagnostics

            diagnostics = write_sonic_diagnostics(
                args.xyzin,
                args.diagnostics_dir,
                progress_callback=_print_matrix_progress,
            )
            print(f"Wrote SMITH SONIC diagnostics: {diagnostics.output_directory}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "standalone":
        from matrix_smith import write_smith_build_sections_from_input

        definition = write_smith_build_sections_from_input(args.input, args.output)
        output = args.output if args.output is not None else args.input.with_suffix(".xyzin")
        print(
            "Built SMITH/SONIC definition: "
            f"{output} (GICs={len(definition.gics)}, rank={definition.rank})"
        )
        if not args.no_diagnostics and Path(output).is_file():
            from matrix_smith import write_sonic_diagnostics

            diagnostics = write_sonic_diagnostics(
                output,
                args.diagnostics_dir,
                progress_callback=_print_matrix_progress,
            )
            print(f"Wrote SMITH SONIC diagnostics: {diagnostics.output_directory}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "bmatrix":
        from matrix_smith import (
            build_gic_b_matrix_from_xyzin,
            gic_b_matrix_lines,
            write_gic_b_matrix,
        )

        if args.output is None:
            matrix = build_gic_b_matrix_from_xyzin(args.xyzin)
            print("\n".join(gic_b_matrix_lines(matrix)))
            return 0
        matrix = write_gic_b_matrix(args.xyzin, args.output)
        print(
            "Wrote GIC B matrix: "
            f"{args.output} (rows={len(matrix.rows)}, "
            f"columns={len(matrix.cartesian_columns)})"
        )
        return 0
    if _is_smith_command(args) and args.gicforge_command == "report":
        from matrix_smith import gic_report_from_xyzin, write_gic_report

        if args.output is None:
            print("\n".join(gic_report_from_xyzin(args.xyzin)))
            return 0
        output = write_gic_report(args.xyzin, args.output)
        print(f"Wrote MATRIX SMITH/SONIC report: {output}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "motions":
        from matrix_smith import write_sonic_diagnostics

        diagnostics = write_sonic_diagnostics(
            args.xyzin,
            args.output_directory,
            distance_step_angstrom=args.distance_step,
            angle_step_radian=math.radians(args.angle_step_degrees),
            ring_puckering_step_radian=math.radians(args.ring_step_degrees),
            max_atom_displacement_angstrom=args.max_atom_displacement,
            cartesian_metric=args.metric,
            mass_source=args.mass_source,
            isotopologue_label=args.isotopologue_label,
            topology_workers=args.topology_workers,
            use_topology_cache=not args.no_topology_cache,
            progress_callback=_print_matrix_progress,
        )
        print(
            "Wrote SMITH SONIC Cartesian motions: "
            f"{diagnostics.output_directory} (coordinates={len(diagnostics.motions)})"
        )
        return 0
    if _is_smith_command(args) and args.gicforge_command == "view":
        try:
            from matrix_gui.motion_viewer import launch_motion_viewer
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("PySide6"):
                raise SystemExit(
                    "PySide6 is required by the SMITH motion viewer. Install the GUI extra."
                ) from exc
            raise
        return launch_motion_viewer(
            args.xyzin,
            source=args.source,
            sonic_hessian=args.sonic_hessian,
        )
    if _is_smith_command(args) and args.gicforge_command == "salc-snapshot":
        from matrix_smith import write_salc_snapshot_from_xyzin

        output = write_salc_snapshot_from_xyzin(args.xyzin, args.output)
        print(f"Wrote MATRIX SMITH SALC snapshot: {output}")
        return 0
    if _is_smith_command(args) and args.gicforge_command == "corpus":
        from matrix_smith import (
            default_gic_corpus_root,
            format_gic_corpus_paths,
            format_gic_corpus_summary,
            gic_corpus_records,
            summarize_gic_corpus,
        )

        corpus_root = args.root or default_gic_corpus_root(root)
        summary = summarize_gic_corpus(corpus_root, suffixes=args.suffix)
        if args.format == "json":
            payload = {
                "root": str(summary.root),
                "total_files": summary.total_files,
                "suffix_counts": summary.suffix_counts,
                "role_counts": summary.role_counts,
                "entries": gic_corpus_records(summary, limit=args.limit),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.format == "paths":
            print("\n".join(format_gic_corpus_paths(summary, limit=args.limit)))
            return 0
        print("\n".join(format_gic_corpus_summary(summary)))
        return 0
    if _is_smith_command(args) and args.gicforge_command == "corpus-audit":
        from matrix_smith import (
            audit_gic_corpus_geometry,
            default_gic_corpus_root,
            format_gic_corpus_geometry_audit_summary,
            format_gic_corpus_geometry_failures,
            gic_corpus_geometry_audit_records,
        )

        corpus_root = args.root or default_gic_corpus_root(root)
        audit = audit_gic_corpus_geometry(
            corpus_root,
            suffixes=args.suffix,
            limit=args.limit if args.format == "summary" else None,
        )
        if args.format == "json":
            payload = {
                "root": str(audit.root),
                "total_files": audit.total_files,
                "passed_files": audit.passed_files,
                "failed_files": audit.failed_files,
                "source_format_counts": audit.source_format_counts,
                "error_counts": audit.error_counts,
                "entries": gic_corpus_geometry_audit_records(
                    audit,
                    status=args.status,
                    limit=args.limit,
                ),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.format == "failures":
            print("\n".join(format_gic_corpus_geometry_failures(audit, limit=args.limit)))
            return 0
        print("\n".join(format_gic_corpus_geometry_audit_summary(audit)))
        return 0
    if _is_smith_command(args) and args.gicforge_command == "fortran-audit":
        from matrix_smith import (
            audit_gicforge_fortran_corpus,
            default_gic_corpus_root,
            format_gicforge_fortran_audit_cases,
            format_gicforge_fortran_audit_summary,
            gicforge_fortran_audit_records,
        )

        corpus_root = args.root or default_gic_corpus_root(root)
        audit = audit_gicforge_fortran_corpus(
            root=corpus_root,
            molecules=args.molecule,
            workdir=args.workdir,
            repo_root=root,
            limit=args.limit,
            tolerance=args.tolerance,
        )
        if args.format == "json":
            payload = {
                "root": str(audit.root),
                "workdir": None if audit.workdir is None else str(audit.workdir),
                "tolerance": audit.tolerance,
                "cases": len(audit.results),
                "passed": audit.passed,
                "failed": audit.failed,
                "errored": audit.errored,
                "skipped": audit.skipped,
                "max_row_space_residual": audit.max_row_space_residual,
                "results": gicforge_fortran_audit_records(audit),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.format == "failures":
            lines = format_gicforge_fortran_audit_cases(audit, status="fail")
            lines.extend(format_gicforge_fortran_audit_cases(audit, status="error"))
            print("\n".join(lines))
            return 0
        if args.format == "cases":
            print("\n".join(format_gicforge_fortran_audit_cases(audit)))
            return 0
        print("\n".join(format_gicforge_fortran_audit_summary(audit)))
        return 0
    if _is_smith_command(args) and args.gicforge_command == "gaussian-input":
        from matrix_smith import write_gicforge_gaussian_input

        output = write_gicforge_gaussian_input(
            args.xyzin,
            args.output,
            route=args.route,
            title=args.title,
            charge=args.charge,
            multiplicity=args.multiplicity,
            g16_compatibility=args.g16,
        )
        print(f"Wrote Gaussian input: {output}")
        return 0
    parser.print_help()
    return 0


def _semidiagonal_summary_dict(log_path, data, delta, cubic_count: int) -> dict[str, object]:
    return {
        "path": str(log_path),
        "natoms": data.natoms,
        "nvib": data.nvib,
        "linear": bool(data.linear),
        "beq_MHz": list(delta.beq_MHz),
        "deltabvib_MHz": list(delta.total_MHz),
        "deltabvib_harmonic_MHz": list(delta.harmonic_MHz),
        "deltabvib_coriolis_MHz": list(delta.coriolis_MHz),
        "deltabvib_anharmonic_MHz": list(delta.anharmonic_MHz),
        "semidiagonal_cubic_terms": int(cubic_count),
    }


def _print_semidiagonal_summary(summary: dict[str, object]) -> None:
    print(f"path: {summary['path']}")
    print(f"natoms: {summary['natoms']}")
    print(f"nvib: {summary['nvib']}")
    print(f"linear: {int(bool(summary['linear']))}")
    for key in (
        "beq_MHz",
        "deltabvib_MHz",
        "deltabvib_harmonic_MHz",
        "deltabvib_coriolis_MHz",
        "deltabvib_anharmonic_MHz",
    ):
        values = summary[key]
        if not isinstance(values, list):
            continue
        print(f"{key}: " + " ".join(f"{float(value):.8f}" for value in values))
    print(f"semidiagonal_cubic_terms: {summary['semidiagonal_cubic_terms']}")


def _print_external_qm_status(status) -> None:
    print(f"program: {status.program}")
    print(f"status: {status.status}")
    print(f"workdir: {status.workdir}")
    print(f"output: {status.output_path}")
    if status.input_path is not None:
        print(f"input: {status.input_path}")
    if status.pid is not None:
        print(f"pid: {status.pid}")
    if status.exit_code is not None:
        print(f"exit_code: {status.exit_code}")
    print(f"normal_termination: {int(status.normal_termination)}")
    print(f"error_termination: {int(status.error_termination)}")
    print(f"message: {status.message}")


def _print_external_qm_run_result(result) -> None:
    print(f"program: {result.program}")
    print(f"input: {result.input_path}")
    print(f"output: {result.output_path}")
    print(f"executable: {result.executable}")
    if result.pid is not None:
        print(f"pid: {result.pid}")
    if result.exit_code is not None:
        print(f"exit_code: {result.exit_code}")
    if result.success is not None:
        print(f"success: {int(result.success)}")
    print(f"message: {result.message}")


def matrix_main(argv: list[str] | None = None) -> int:
    """Console-script alias for the MATRIX framework CLI."""
    return main(argv, prog="matrix")


def oracle_main(argv: list[str] | None = None) -> int:
    """Compatibility console-script alias for the legacy ORACLE CLI."""
    return main(argv, prog="oracle")


def link_main(argv: list[str] | None = None) -> int:
    """Console-script entry point for LINK workflows."""
    command_args = sys.argv[1:] if argv is None else argv
    alias_parser = argparse.ArgumentParser(
        prog="link", description="LINK coordinate realization and external-driver workflows"
    )
    alias_sub = alias_parser.add_subparsers(dest="link_command", required=True)
    alias_preprocess = alias_sub.add_parser("preprocess", help="Import a source into xyzin")
    _add_preprocess_arguments(alias_preprocess)
    alias_driver_run = alias_sub.add_parser(
        "driver-run",
        help="Run LINK with an external program selecting the next SONIC point",
    )
    _add_external_driver_arguments(alias_driver_run)
    alias_parser.parse_args(command_args)
    return main(["link", *command_args], prog="matrix")


def smith_main(argv: list[str] | None = None) -> int:
    """Console-script alias for SMITH, the SONIC coordinate tool."""
    command_args = sys.argv[1:] if argv is None else argv
    return main(["smith", *command_args], prog="SMITH")


def _add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-kind",
        choices=(
            "auto",
            "smiles",
            "xyz",
            "enriched_xyz",
            "gaussian",
            "fchk",
            "mol",
            "sdf",
            "mol2",
            "molpro",
            "mrcc",
            "orca",
        ),
        default="auto",
    )
    parser.add_argument("--symmetry-distance", type=float, default=1.0e-3)
    parser.add_argument("--symmetry-inertia", type=float, default=1.0e-3)
    parser.add_argument("--max-rotation-order", type=int, default=6)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Write #VALIDATION after ORACLE import and perception",
    )


def _add_external_driver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("xyzin", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--driver-command",
        required=True,
        help="Command template using {request}, {response}, {run_dir} and {cycle}",
    )
    parser.add_argument("--engine-command", default="")
    parser.add_argument(
        "--backend",
        choices=(
            "gaussian", "g16", "gdv", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf", "et", "architect"
        ),
        help="Optional LINK-owned electronic-structure or ARCHITECT backend",
    )
    parser.add_argument("--route", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--basis", default="")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--executable")
    parser.add_argument(
        "--force-field", type=Path, help="ARCHITECT force-field JSON for --backend architect"
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--calculator-profile",
        action="append",
        default=[],
        metavar="ID=BACKEND",
        help=(
            "Advertise an authorized LINK calculator profile to SENTINEL; repeatable. "
            "Route/method/basis settings are inherited from this command"
        ),
    )
    parser.add_argument(
        "--calculator-command",
        action="append",
        default=[],
        metavar="ID=COMMAND",
        help=(
            "Advertise an authorized LINK/SMITH calculator command; repeatable. "
            "COMMAND may use the normal engine-command placeholders"
        ),
    )
    parser.add_argument(
        "--variables",
        type=Path,
        help="JSON active-variable contract shared by LINK and SENTINEL",
    )
    parser.add_argument(
        "--coordinate",
        action="append",
        default=[],
        help="SONIC label/name/index; repeat for a reduced active contract",
    )
    parser.add_argument(
        "--initial-evaluation-owner",
        choices=("link", "driver"),
        default="link",
    )
    parser.add_argument(
        "--property",
        choices=("energy", "gradient", "hessian"),
        action="append",
        default=[],
        help="Property required at the initial point; repeat as needed",
    )
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--batch-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the atomic LINK checkpoint in --run-dir",
    )
    parser.add_argument(
        "--run-id",
        help="Stable run identifier; generated automatically for a new run",
    )
    parser.add_argument(
        "--retained-group",
        default="C1",
        help="Minimum point group retained by the complete PES exploration; default C1",
    )


def _add_mock_sentinel_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument(
        "--mode",
        choices=("scan-1d", "scan-2d", "partial", "batch"),
        default="scan-1d",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--driver-owned", action="store_true")


def _add_xh_stretch_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_policy: str | None,
) -> None:
    parser.add_argument(
        "--xh-stretch-policy",
        choices=("symmetrize", "local-all", "local-selected"),
        default=default_policy,
        help="Control whether X-H stretches are symmetrized or kept as local modes",
    )
    parser.add_argument(
        "--local-xh-bond",
        action="append",
        default=[],
        metavar="I-J",
        help="One-based X-H bond to keep local; repeatable",
    )
    parser.add_argument(
        "--local-xh-class",
        choices=("XH", "XH2", "XH3"),
        action="append",
        default=[],
        help="Keep all X-H stretches of the selected local class unsymmetrized",
    )


def _add_xh_stretch_kwargs(args: argparse.Namespace, kwargs: dict[str, object]) -> None:
    local_bonds = _parse_local_xh_bonds(getattr(args, "local_xh_bond", ()))
    local_classes = tuple(getattr(args, "local_xh_class", ()) or ())
    policy = getattr(args, "xh_stretch_policy", None)
    if policy is None and (local_bonds or local_classes):
        policy = "local-selected"
    if policy is not None:
        kwargs["xh_stretch_policy"] = policy
    if local_bonds:
        kwargs["local_xh_bonds"] = local_bonds
    if local_classes:
        kwargs["local_xh_classes"] = local_classes


def _parse_local_xh_bonds(raw_bonds: list[str] | tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for raw in raw_bonds or ():
        for item in str(raw).replace(";", ",").split(","):
            text = item.strip()
            if not text:
                continue
            parts = re.split(r"[-:/]", text)
            if len(parts) != 2:
                raise SystemExit(f"invalid --local-xh-bond selector: {raw!r}")
            left, right = int(parts[0]), int(parts[1])
            pairs.append((left, right) if left <= right else (right, left))
    return tuple(dict.fromkeys(pairs))


def _is_link_command(args: argparse.Namespace) -> bool:
    return (args.command == "link" and args.link_command == "preprocess") or (
        args.command == "babel" and args.babel_command == "preprocess"
    )


def _is_smith_command(args: argparse.Namespace) -> bool:
    return args.command in {"gicforge", "smith", "SMITH"}


def _parse_fixed_parameters(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _split_top_level(raw, separators=",;") if part.strip())


def _vpt2_vci_source(args) -> tuple[str, Path | None]:
    if getattr(args, "zion_force_field", None) is not None:
        return "zion_force_field", args.zion_force_field
    if getattr(args, "xyzin", None) is not None:
        return "xyzin", args.xyzin
    if getattr(args, "fchk", None) is not None:
        return "fchk", args.fchk
    if getattr(args, "qff_file", None) is not None:
        return "qff_file", args.qff_file
    return "unknown", None


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
        data["outputs"] = {
            name: str(path) for name, path in output_names.items() if path.is_file()
        }
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
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError(
            "MORPHEUS cannot complete the reliable delivery: pdflatex is not installed"
        )
    completed = subprocess.run(
        (
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(source.parent),
            str(source),
        ),
        cwd=source.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    pdf = source.with_suffix(".pdf")
    if completed.returncode != 0 or not pdf.is_file():
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            "MORPHEUS standalone LaTeX compilation failed"
            + (f":\n{tail}" if tail else "")
        )
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("MORPHEUS LaTeX compiler produced an invalid PDF artifact")
    print(f"morpheus_latex_pdf: {pdf}")
    return pdf


def _trinity_scan_direction(
    xyzin: Path,
    kind: str,
    coordinate: str,
    *,
    retained_group: str = "C1",
):
    from matrix_trinity import (
        coordinate_direction_from_cartesian_vector,
        coordinate_direction_from_pes_exploration_gic,
        coordinate_direction_from_normal_mode,
    )

    if kind == "sonic":
        try:
            parsed: str | int = int(coordinate)
        except ValueError:
            parsed = coordinate
        return coordinate_direction_from_pes_exploration_gic(
            xyzin,
            parsed,
            retained_group=retained_group,
        )
    if kind == "normal-mode":
        return coordinate_direction_from_normal_mode(xyzin, int(coordinate))
    if kind == "cartesian":
        values = [float(item) for item in coordinate.replace(",", " ").split()]
        return coordinate_direction_from_cartesian_vector(values)
    raise ValueError(f"unsupported TRINITY scan coordinate kind: {kind}")


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
