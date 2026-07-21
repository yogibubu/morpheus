#!/usr/bin/env python3
"""Build and run the complete two-SONIC water protocol example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
from matrix_smith import write_gicforge_build_sections
from matrix_trinity import active_variable_contract_from_file, run_external_driver_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repository = source.parents[4]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    xyzin = args.run_dir / "water.xyzin"
    preprocess_to_enriched_xyz(source / "water.xyz", xyzin)
    write_validation_section(xyzin)
    write_gicforge_build_sections(xyzin, symmetrize=False)
    contract = active_variable_contract_from_file(
        xyzin,
        source / "active_variables.json",
        pes_exploration=True,
    )
    sentinel = repository / "tools" / "link_sentinel_v1" / "mock_sentinel.py"
    evaluator = source / "analytic_pes.py"
    result = run_external_driver_loop(
        xyzin,
        run_dir=args.run_dir / "exchange",
        driver_command=(
            f"{sys.executable} {sentinel} {{request}} {{response}} --mode scan-2d"
        ),
        coordinate_model=contract.model,
        active_variable_contract=contract,
        engine_command=(
            f"{sys.executable} {evaluator} {{xyz}} {{result}} {{index}}"
        ),
        initial_properties=("energy", "gradient", "hessian"),
        batch_workers=4,
    )
    print(f"status={result.status} cycles={result.cycles} points={result.point_count}")
    return 0 if result.status == "complete" and result.point_count == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
