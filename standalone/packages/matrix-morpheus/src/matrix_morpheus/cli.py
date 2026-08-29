"""Standalone command-line entry point for MORPHEUS."""

from __future__ import annotations

import argparse
from importlib import resources
from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import sys

from ._version import __version__


_MATRIX_COMMANDS = {
    "fit": "semiexp",
    "ensemble": "semiexp-ensemble",
    "ensemble-paper": "semiexp-ensemble-paper",
    "ensemble-scan": "semiexp-ensemble-prior-scan",
    "ensemble-synthon-scan": "semiexp-ensemble-synthon-scan",
    "benchmark": "semiexp-benchmark",
}


def main(argv: list[str] | None = None) -> int:
    """Run MORPHEUS without requiring a source checkout or MATRIX_HOME."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(f"MORPHEUS {__version__}")
        return 0
    if arguments[0] == "doctor":
        return _doctor(arguments[1:])
    if arguments[0] == "examples":
        return _examples(arguments[1:])
    if arguments[0] in {"anharm-deltavib", "cubic-deltavib"}:
        return _cubic_deltavib(arguments[1:])

    command = _MATRIX_COMMANDS.get(arguments[0])
    if command is None:
        command = "semiexp"
        forwarded = arguments
    else:
        forwarded = arguments[1:]
    return _workflow_main([command, *forwarded])


def _workflow_main(arguments: list[str]) -> int:
    """Parse and execute MORPHEUS workflows without the aggregate MATRIX CLI."""
    from .cli_commands import dispatch
    from .cli_parser import add_commands
    from .cli_support import UNHANDLED

    parser = argparse.ArgumentParser(
        prog="MORPHEUS",
        description="MORPHEUS semiexperimental refinement CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_commands(subparsers, root=Path.cwd())
    parsed = parser.parse_args(arguments)
    result = dispatch(parsed, parser, Path.cwd())
    if result is UNHANDLED:
        parser.error(f"unsupported MORPHEUS command: {parsed.command}")
    return int(result)


def _print_help() -> None:
    print(
        """usage: morpheus COMMAND [options]

MORPHEUS algorithmic model generation and semiexperimental refinement.

commands:
  fit                    fit one semiexperimental equilibrium structure
  ensemble               fit shared classes across several structures
  ensemble-paper         generate the ensemble analysis artifacts
  ensemble-scan          scan ensemble prior strengths
  ensemble-synthon-scan  scan continuous synthon class thresholds
  benchmark              regenerate checked publication benchmark tables
  anharm-deltavib        reuse a parent Gaussian Freq=Anharm field for isotopologues
  examples DIRECTORY     copy the bundled standalone example
  doctor [--json]        check the clean installation

For command-specific options use, for example, `morpheus fit --help`.
The `fit` command may be omitted: `morpheus --xyzin ... --outdir ...` is valid.
"""
    )


def _doctor(arguments: list[str]) -> int:
    unknown = [item for item in arguments if item != "--json"]
    if unknown:
        print(f"doctor: unexpected arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2
    modules = (
        "numpy",
        "matrix_core",
        "matrix_chem",
        "matrix_link",
        "matrix_smith",
        "matrix_gaussian",
        "matrix_oracle",
        "matrix_trinity",
    )
    checks = {name: find_spec(name) is not None for name in modules}
    try:
        from .cli_commands import dispatch as _dispatch  # noqa: F401
        from .cli_parser import add_commands as _add_commands  # noqa: F401
    except Exception:
        standalone_dispatch = False
    else:
        standalone_dispatch = True
    payload = {
        "schema": "matrix.morpheus.doctor.v1",
        "morpheus_version": __version__,
        "python": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 11),
        "required_modules": checks,
        "standalone_dispatch": standalone_dispatch,
    }
    payload["status"] = (
        "PASS"
        if payload["python_supported"] and all(checks.values()) and standalone_dispatch
        else "FAIL"
    )
    if "--json" in arguments:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"MORPHEUS {__version__}: {payload['status']}")
        print(f"Python {payload['python']}: {'PASS' if payload['python_supported'] else 'FAIL'}")
        for name, available in checks.items():
            print(f"required {name}: {'PASS' if available else 'FAIL'}")
        print(f"standalone command dispatch: {'PASS' if standalone_dispatch else 'FAIL'}")
    return 0 if payload["status"] == "PASS" else 1


def _cubic_deltavib(arguments: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="morpheus anharm-deltavib")
    parser.add_argument("gaussian_log", type=Path, nargs="?")
    parser.add_argument("--fchk", type=Path)
    parser.add_argument("--curvilinear-json", type=Path)
    parser.add_argument("--comparison-tolerance-mhz", type=float, default=5.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    from .cubic_corrections import (
        corrections_from_gaussian_anharmonic,
        corrections_from_gaussian_cartesian_cubic,
    )
    from .io import read_observations

    observations = read_observations(parsed.observations)
    if parsed.gaussian_log is None and parsed.curvilinear_json is None:
        parser.error("provide a Gaussian log, --curvilinear-json, or both")
    if parsed.fchk is not None and parsed.gaussian_log is None:
        parser.error("--fchk requires a Gaussian log")
    result = None
    if parsed.fchk is not None:
        result = corrections_from_gaussian_anharmonic(
            parsed.gaussian_log,
            parsed.fchk,
            observations,
            output_csv=None if parsed.curvilinear_json is not None else parsed.output,
        )
    elif parsed.gaussian_log is not None:
        result = corrections_from_gaussian_cartesian_cubic(
            parsed.gaussian_log,
            observations,
            output_csv=None if parsed.curvilinear_json is not None else parsed.output,
        )
    if parsed.curvilinear_json is not None:
        from matrix_trinity import read_curvilinear_deltabvib_results
        from .dual_deltavib import combine_isotopic_deltabvib_channels

        combined = combine_isotopic_deltabvib_channels(
            observations,
            read_curvilinear_deltabvib_results(parsed.curvilinear_json),
            cartesian=result,
            comparison_tolerance_MHz=parsed.comparison_tolerance_mhz,
            output_csv=parsed.output,
            report_json=parsed.report,
        )
        print(f"isotopologues: {len(combined.curvilinear)}")
        print("authoritative_channel: curvilinear-sonic")
        print(f"cartesian_validation: {'PASS' if combined.validation_passed else 'WARN'}")
        print(f"corrected_observations: {combined.output_csv}")
        if combined.report_json is not None:
            print(f"comparison_report: {combined.report_json}")
        return 0
    assert result is not None
    print(f"isotopologues: {len(result.corrections)}")
    print(f"corrected_observations: {result.output_csv}")
    return 0


def _examples(arguments: list[str]) -> int:
    force = "--force" in arguments
    paths = [Path(item) for item in arguments if item != "--force"]
    if len(paths) != 1:
        print("usage: morpheus examples DIRECTORY [--force]", file=sys.stderr)
        return 2
    destination = paths[0].expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        print(f"examples directory is not empty: {destination}; use --force", file=sys.stderr)
        return 2
    destination.mkdir(parents=True, exist_ok=True)
    source = resources.files("matrix_morpheus").joinpath("data", "examples")
    _copy_resources(source, destination)
    print(destination)
    return 0


def _copy_resources(source, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_resources(item, target)
        else:
            with item.open("rb") as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)


if __name__ == "__main__":
    raise SystemExit(main())
