"""End-to-end ORACLE L1-to-PL1 Cartesian refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from matrix_chem import (
    AccuracyLadderPlan,
    BackTransformationResult,
    RefinementLayer,
    ValenceLevel,
    apply_accuracy_ladder_plan,
    build_accuracy_ladder_plan,
    build_topology_objects,
    read_enriched_xyz,
    read_primitive_contract,
)
from matrix_chem.geometry_io import write_xyz
from matrix_chem.topology.elements import atomic_number
from matrix_core import replace_section, sha256_file

from .api import OracleAnalysis, analyze_structure


ORACLE_REFINEMENT_SCHEMA = "oracle.xyz.accuracy_ladder_refinement.v1"


@dataclass(frozen=True)
class OracleGeometryRefinement:
    source: Path
    output: Path
    plan: AccuracyLadderPlan
    back_transformation: BackTransformationResult
    analysis: OracleAnalysis

    @property
    def target_count(self) -> int:
        return len(self.plan.targets)


def refine_l1_geometry(
    source: Path | str,
    output: Path | str,
    *,
    include_core_valence: bool = True,
    include_conjugation: bool = True,
    include_hydrogen_bonds: bool = True,
    cv_weight_threshold: float = 0.9,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
) -> OracleGeometryRefinement:
    """Convert an ORACLE-enriched L1 geometry into a reanalyzed PL1 state."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("L1 source and PL1 output must be different files")
    geometry = read_enriched_xyz(source_path)
    contract = read_primitive_contract(source_path)
    numbers = tuple(_required_atomic_number(symbol) for symbol in geometry.atoms)
    _continuous, _discrete, _rings, synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom, numbers
    )
    plan = build_accuracy_ladder_plan(
        contract.primitives,
        numbers,
        valence_level=ValenceLevel.L1,
        include_core_valence=include_core_valence,
        coordinates_angstrom=geometry.coordinates_angstrom,
        synthons=synthons,
        include_bl1_conjugation=include_conjugation,
        include_pl1_hydrogen_bonds=include_hydrogen_bonds,
        cv_weight_threshold=cv_weight_threshold,
    )
    back_transformation = apply_accuracy_ladder_plan(
        plan,
        contract.primitives,
        geometry.coordinates_angstrom,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not back_transformation.converged:
        raise RuntimeError(
            "ORACLE L1-to-PL1 back-transformation did not converge: "
            f"maximum residual {back_transformation.maximum_residual:.6g}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oracle-pl1-") as scratch_text:
        cartesian_source = Path(scratch_text) / "pl1.xyz"
        write_xyz(
            cartesian_source,
            geometry.atoms,
            back_transformation.coordinates_angstrom,
            comment=f"ORACLE PL1 refinement of {source_path.name}",
        )
        analysis = analyze_structure(cartesian_source, output_path, source_kind="xyz")
    replace_section(
        output_path,
        "ACCURACY_LADDER_REFINEMENT",
        _refinement_section_lines(
            source_path,
            plan,
            back_transformation,
            include_core_valence=include_core_valence,
            include_conjugation=include_conjugation,
            include_hydrogen_bonds=include_hydrogen_bonds,
            cv_weight_threshold=cv_weight_threshold,
        ),
    )
    return OracleGeometryRefinement(
        source=source_path,
        output=output_path,
        plan=plan,
        back_transformation=back_transformation,
        analysis=analysis,
    )


def _refinement_section_lines(
    source: Path,
    plan: AccuracyLadderPlan,
    result: BackTransformationResult,
    **settings,
) -> list[str]:
    counts = {
        layer: sum(target.layer is layer for target in plan.targets)
        for layer in RefinementLayer
    }
    return [
        f"SCHEMA {ORACLE_REFINEMENT_SCHEMA}",
        "OWNER ORACLE",
        "INPUT_LEVEL L1",
        "OUTPUT_LEVEL PL1",
        f"SOURCE {source}",
        f"SOURCE_SHA256 {sha256_file(source)}",
        f"CORE_VALENCE {str(bool(settings['include_core_valence'])).upper()}",
        f"BL1_CONJUGATION {str(bool(settings['include_conjugation'])).upper()}",
        f"PL1_HYDROGEN_BONDS {str(bool(settings['include_hydrogen_bonds'])).upper()}",
        "CV_AMPLITUDE_MODEL RADIUS_AWARE_PERIOD_LINE",
        f"CV_WEIGHT_THRESHOLD {float(settings['cv_weight_threshold']):.16g}",
        f"CV_TARGETS {counts[RefinementLayer.CORE_VALENCE]}",
        f"BL1_TARGETS {counts[RefinementLayer.BL1_CONJUGATION]}",
        f"PL1_TARGETS {counts[RefinementLayer.PL1_SELECTED_PAIR]}",
        f"ITERATIONS {result.iterations}",
        f"MAXIMUM_RESIDUAL {result.maximum_residual:.16g}",
        "STATUS PASS",
    ]


def _required_atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"L1-to-PL1 refinement does not support atom label {symbol!r}")
    return int(number)
