"""Command-line interface for the standalone ORACLE perception package."""

from __future__ import annotations

import argparse
from importlib import resources
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import shutil
import sys

from .api import (
    ORACLE_BATCH_SCHEMA,
    SUPPORTED_INPUT_FORMATS,
    OracleAnalysisRequest,
    analyze_structure,
    analyze_structures,
    oracle_version,
    write_oracle_analysis_reports,
)
from .config import load_oracle_config, write_oracle_config_template


def build_parser(*, prog: str = "oracle") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="ORACLE topology, symmetry and continuous molecular perception",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {oracle_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a structure and write enriched xyzin")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("-o", "--output", type=Path, required=True)
    analyze.add_argument("--report", type=Path, help="Write the versioned JSON analysis report")
    analyze.add_argument(
        "--human-report",
        type=Path,
        help="Write the complete human-readable perception/PIC report",
    )
    analyze.add_argument("--snapshot", type=Path, help="Write a topology snapshot JSON")
    analyze.add_argument("--config", type=Path, help="TOML configuration file")
    analyze.add_argument(
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
    analyze.add_argument("--symmetry-distance", type=float)
    analyze.add_argument("--symmetry-inertia", type=float)
    analyze.add_argument("--max-rotation-order", type=int)
    analyze.add_argument("--no-validate", action="store_true")
    analyze.add_argument("--json", action="store_true", help="Print the summary as JSON")

    formats = sub.add_parser("formats", help="List supported input formats and extras")
    formats.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="Check the standalone ORACLE installation")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true")

    init_config = sub.add_parser("init-config", help="Write a portable configuration template")
    init_config.add_argument("output", type=Path)
    init_config.add_argument("--force", action="store_true")

    examples = sub.add_parser("examples", help="Copy the bundled publication examples")
    examples.add_argument("output", type=Path)
    examples.add_argument("--force", action="store_true")

    report = sub.add_parser(
        "report",
        help="Report an existing ORACLE state without reperceiving it",
    )
    report.add_argument("xyzin", type=Path)
    report.add_argument("--json-output", type=Path)
    report.add_argument("--human-output", type=Path)
    report.add_argument("--json", action="store_true", dest="print_json")

    batch = sub.add_parser(
        "batch",
        help="Run independent ORACLE perceptions concurrently from a JSON manifest",
    )
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--workers", type=int, default=0)
    batch.add_argument("--json", action="store_true", dest="print_json")

    refine = sub.add_parser(
        "refine-l1",
        help="Convert an ORACLE-enriched L1 geometry to a reanalyzed PL1 state",
    )
    refine.add_argument("source", type=Path)
    refine.add_argument("-o", "--output", type=Path, required=True)
    refine.add_argument("--no-core-valence", action="store_true")
    refine.add_argument("--no-conjugation", action="store_true")
    refine.add_argument("--no-hydrogen-bonds", action="store_true")
    refine.add_argument("--cv-weight-threshold", type=float, default=0.9)
    refine.add_argument("--tolerance", type=float, default=1.0e-8)
    refine.add_argument("--max-iterations", type=int, default=50)
    refine.add_argument("--json", action="store_true")

    gui = sub.add_parser("gui", help="Launch the optional ORACLE GUI")
    gui.add_argument("xyzin", nargs="?", type=Path)
    return parser


def main(argv: list[str] | None = None, *, prog: str = "oracle") -> int:
    args = build_parser(prog=prog).parse_args(argv)
    if args.command == "analyze":
        result = analyze_structure(
            args.source,
            args.output,
            source_kind=args.source_kind,
            config=args.config,
            symmetry_distance=args.symmetry_distance,
            symmetry_inertia=args.symmetry_inertia,
            max_rotation_order=args.max_rotation_order,
            report=args.report,
            human_report=args.human_report,
            topology_snapshot=args.snapshot,
            validate=not args.no_validate,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"output: {result.output}")
            print(f"status: {result.status}")
            print(f"atoms: {result.atom_count}")
            print(f"point_group: {result.point_group}")
            print(f"symmetry_operations: {result.symmetry_operation_count}")
            print(f"bonds: {result.bond_count}")
            print(f"rings: {result.ring_count}")
            print(f"aromatic_atoms: {result.aromatic_atom_count}")
            print(f"synthons: {result.synthon_count}")
            print(f"primitives: {result.primitive_count}")
            print(f"primitive_b_rank: {result.primitive_b_matrix_rank}")
            print(f"primitive_b_sha256: {result.primitive_b_matrix_sha256}")
            print(f"topology_sha256: {result.topology_sha256}")
            if result.report is not None:
                print(f"report: {result.report}")
            if result.human_report is not None:
                print(f"human_report: {result.human_report}")
            if result.topology_snapshot is not None:
                print(f"snapshot: {result.topology_snapshot}")
        return 0 if result.status in {"PASS", "WARN"} else 2
    if args.command == "formats":
        records = [
            {"kind": kind, "input": description, "availability": availability}
            for kind, description, availability in SUPPORTED_INPUT_FORMATS
        ]
        if args.json:
            print(json.dumps(records, indent=2))
        else:
            for record in records:
                print(f"{record['kind']:14s} {record['input']:38s} {record['availability']}")
        return 0
    if args.command == "doctor":
        return _doctor(args.config, as_json=args.json)
    if args.command == "init-config":
        path = write_oracle_config_template(args.output, overwrite=args.force)
        print(path)
        return 0
    if args.command == "examples":
        copied = _copy_examples(args.output, overwrite=args.force)
        for path in copied:
            print(path)
        return 0
    if args.command == "report":
        human_output = args.human_output
        if human_output is None and args.json_output is None:
            human_output = args.xyzin.with_name(f"{args.xyzin.stem}.oracle.txt")
        payload = write_oracle_analysis_reports(
            args.xyzin,
            json_output=args.json_output,
            human_output=human_output,
        )
        if args.print_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            if args.json_output is not None:
                print(args.json_output)
            if human_output is not None:
                print(human_output)
        return 0
    if args.command == "batch":
        requests = _batch_requests(args.manifest)
        results = analyze_structures(requests, workers=args.workers)
        payload = {
            "schema": ORACLE_BATCH_SCHEMA,
            "manifest": str(args.manifest.resolve()),
            "requested_workers": args.workers,
            "count": len(results),
            "results": [result.to_dict() for result in results],
        }
        if args.print_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for result in results:
                print(f"{result.status:5s} {result.output}")
        return 0 if all(result.status in {"PASS", "WARN"} for result in results) else 2
    if args.command == "refine-l1":
        from .refinement import refine_l1_geometry

        result = refine_l1_geometry(
            args.source,
            args.output,
            include_core_valence=not args.no_core_valence,
            include_conjugation=not args.no_conjugation,
            include_hydrogen_bonds=not args.no_hydrogen_bonds,
            cv_weight_threshold=args.cv_weight_threshold,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        payload = {
            "source": str(result.source),
            "output": str(result.output),
            "input_level": "L1",
            "output_level": "PL1",
            "target_count": result.target_count,
            "iterations": result.back_transformation.iterations,
            "maximum_residual": result.back_transformation.maximum_residual,
            "status": result.analysis.status,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    if args.command == "gui":
        from .gui import run as gui_run

        gui_args = [str(args.xyzin)] if args.xyzin is not None else []
        return gui_run(gui_args)
    raise AssertionError(f"unhandled ORACLE command: {args.command}")


def _batch_requests(path: Path) -> tuple[OracleAnalysisRequest, ...]:
    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != ORACLE_BATCH_SCHEMA:
        raise ValueError(f"ORACLE batch manifest must use {ORACLE_BATCH_SCHEMA}")
    records = payload.get("requests")
    if not isinstance(records, list) or not records:
        raise ValueError("ORACLE batch manifest requires a non-empty requests list")
    base = manifest.parent

    def selected(record: dict[str, object], name: str, *, required: bool = False) -> Path | None:
        value = str(record.get(name, "")).strip()
        if not value:
            if required:
                raise ValueError(f"ORACLE batch request is missing {name}")
            return None
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    return tuple(
        OracleAnalysisRequest(
            source=selected(record, "source", required=True),  # type: ignore[arg-type]
            output=selected(record, "output", required=True),  # type: ignore[arg-type]
            source_kind=str(record.get("source_kind", "auto")),
            report=selected(record, "report"),
            human_report=selected(record, "human_report"),
            topology_snapshot=selected(record, "topology_snapshot"),
            config=selected(record, "config"),
            validate=bool(record.get("validate", True)),
        )
        for record in records
        if isinstance(record, dict)
    )


def matrix_main(argv: list[str] | None = None) -> int:
    return main(argv, prog="matrix oracle")


def _doctor(config_path: Path | None, *, as_json: bool) -> int:
    config = load_oracle_config(config_path)
    required_modules = ("numpy", "matrix_core", "matrix_chem", "matrix_oracle")
    optional_modules = (
        "rdkit",
        "matrix_link",
        "matrix_gaussian",
        "matrix_molpro",
        "matrix_mrcc",
        "matrix_orca",
        "PySide6",
    )
    required = {name: find_spec(name) is not None for name in required_modules}
    optional = {name: find_spec(name) is not None for name in optional_modules}
    paths = {
        "data_dir": config.paths.data_dir,
        "cache_dir": config.paths.cache_dir,
        "work_dir": config.paths.work_dir,
    }
    payload = {
        "oracle_version": oracle_version(),
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "config": str(config.source) if config.source is not None else None,
        "required_modules": required,
        "optional_modules": optional,
        "paths": {
            name: {
                "value": str(path) if path is not None else None,
                "exists": path.is_dir() if path is not None else None,
            }
            for name, path in paths.items()
        },
        "symmetry": {
            "distance_angstrom": config.symmetry.distance_angstrom,
            "inertia_relative": config.symmetry.inertia_relative,
            "max_rotation_order": config.symmetry.max_rotation_order,
        },
    }
    ok = bool(payload["python_supported"]) and all(required.values())
    payload["status"] = "PASS" if ok else "FAIL"
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ORACLE {payload['oracle_version']}: {payload['status']}")
        print(f"Python {payload['python']}: {'PASS' if payload['python_supported'] else 'FAIL'}")
        for name, present in required.items():
            print(f"required {name}: {'PASS' if present else 'MISSING'}")
        for name, present in optional.items():
            print(f"optional {name}: {'AVAILABLE' if present else 'not installed'}")
        for name, path in paths.items():
            if path is None:
                print(f"path {name}: not configured")
            else:
                state = "exists" if path.is_dir() else "does not exist"
                print(f"path {name}: {path} ({state})")
    return 0 if ok else 1


def _copy_examples(output: Path, *, overwrite: bool) -> tuple[Path, ...]:
    target = Path(output).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    source = resources.files("matrix_oracle.data").joinpath("examples")
    copied: list[Path] = []
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite example: {destination}")
        with resources.as_file(item) as local:
            shutil.copyfile(local, destination)
        copied.append(destination)
    return tuple(sorted(copied))


if __name__ == "__main__":
    raise SystemExit(main())
