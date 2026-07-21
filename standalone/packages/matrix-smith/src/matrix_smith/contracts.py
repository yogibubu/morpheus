from __future__ import annotations

from pathlib import Path

from matrix_core import read_sectioned_lines, replace_section, section_content
from matrix_chem.topology.contracts import (
    MATRIX_XYZ_SYNTHONS_SCHEMA,
    MATRIX_XYZ_TOPOLOGY_SCHEMA,
    MATRIX_XYZ_VALIDATION_SCHEMA,
    SUPPORTED_VALIDATION_SCHEMAS,
    schema_line_supported,
)


ORACLE_XYZ_GIC_SCHEMA = "oracle.xyz.gic.v1"
ORACLE_XYZ_SYCART_SCHEMA = "oracle.xyz.sycart.v1"
REQUIRED_VALIDATION_SCHEMA = MATRIX_XYZ_VALIDATION_SCHEMA


class GICForgeContractError(ValueError):
    """Raised when GICForge cannot consume the enriched XYZ state."""


def validate_gicforge_prerequisites(path: Path) -> None:
    lines = read_sectioned_lines(Path(path))
    validation = section_content(lines, "VALIDATION")
    if not validation:
        raise GICForgeContractError("missing #VALIDATION section")
    expected = f"SCHEMA {REQUIRED_VALIDATION_SCHEMA}"
    if not schema_line_supported(validation[0], SUPPORTED_VALIDATION_SCHEMAS):
        raise GICForgeContractError(
            f"#VALIDATION must start with {expected!r}; found {validation[0]!r}"
        )
    status = _validation_status(validation)
    if status != "PASS":
        raise GICForgeContractError(f"#VALIDATION status must be PASS; found {status or 'UNKNOWN'}")


def gic_plan_section_lines(
    *,
    symmetrize: bool = False,
    improper_dihedrals: bool = False,
    fragment_mode: str = "SPECIAL_COORDINATES",
    xh_stretch_policy: str = "SYMMETRIZE",
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
) -> list[str]:
    mode = _normalized_fragment_mode(fragment_mode)
    policy = str(xh_stretch_policy or "SYMMETRIZE").strip().replace("-", "_").upper()
    bonds = (
        ",".join(f"{int(left)}-{int(right)}" for left, right in local_xh_bonds)
        if local_xh_bonds
        else "NONE"
    )
    classes = (
        ",".join(str(item).strip().upper() for item in local_xh_classes)
        if local_xh_classes
        else "NONE"
    )
    return [
        f"SCHEMA {ORACLE_XYZ_GIC_SCHEMA}",
        "STATUS PLANNED",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} "
        f"TOPOLOGY={MATRIX_XYZ_TOPOLOGY_SCHEMA} SYNTHONS={MATRIX_XYZ_SYNTHONS_SCHEMA} "
        "SYMMETRY=oracle.xyz.symmetry.v1",
        "INDEXING ATOMS=ONE_BASED",
        f"SYMMETRIZE {_bool_text(symmetrize)}",
        "OUT_OF_PLANE_MODE OUT_OF_PLANE",
        f"FRAGMENT_MODE {mode}",
        f"XH_STRETCH_POLICY {policy}",
        f"LOCAL_XH_BONDS {bonds}",
        f"LOCAL_XH_CLASSES {classes}",
        "BACKEND UNASSIGNED",
        "[FROZEN_GICS]",
        "PENDING GICFORGE_IMPLEMENTATION",
    ]


def sycart_plan_section_lines() -> list[str]:
    return [
        f"SCHEMA {ORACLE_XYZ_SYCART_SCHEMA}",
        "STATUS PLANNED",
        f"DEPENDENCIES VALIDATION={MATRIX_XYZ_VALIDATION_SCHEMA} GIC=oracle.xyz.gic.v1",
        "INDEXING ATOMS=ONE_BASED",
        "[SYCART]",
        "PENDING GICFORGE_IMPLEMENTATION",
    ]


def write_gicforge_plan_sections(
    path: Path,
    *,
    symmetrize: bool = False,
    sycart: bool = False,
    improper_dihedrals: bool = False,
    fragment_mode: str = "SPECIAL_COORDINATES",
    xh_stretch_policy: str = "SYMMETRIZE",
    local_xh_bonds: tuple[tuple[int, int], ...] = (),
    local_xh_classes: tuple[str, ...] = (),
) -> None:
    target = Path(path)
    validate_gicforge_prerequisites(target)
    replace_section(
        target,
        "GIC",
        gic_plan_section_lines(
            symmetrize=symmetrize,
            improper_dihedrals=improper_dihedrals,
            fragment_mode=fragment_mode,
            xh_stretch_policy=xh_stretch_policy,
            local_xh_bonds=local_xh_bonds,
            local_xh_classes=local_xh_classes,
        ),
    )
    if sycart:
        replace_section(target, "SYCART", sycart_plan_section_lines())


def write_gicforge_gaussian_input(
    path: Path,
    output: Path,
    *,
    route: str = "#p hf/sto-3g opt=readallgic",
    title: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    g16_compatibility: bool = True,
) -> Path:
    """Create Gaussian input for a validated GICForge molecule state."""
    validate_gicforge_prerequisites(Path(path))
    from matrix_gaussian import write_gicforge_gaussian_input as write_gaussian

    return write_gaussian(
        Path(path),
        Path(output),
        route=route,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
        g16_compatibility=g16_compatibility,
    )


def _validation_status(validation_lines: list[str]) -> str | None:
    for line in validation_lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "STATUS":
            return parts[1].upper()
    return None


def _bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _normalized_fragment_mode(value: str | None) -> str:
    text = (value or "SPECIAL_COORDINATES").strip().upper().replace("-", "_")
    aliases = {
        "PSEUDO": "PSEUDO_BONDS",
        "PSEUDOBONDS": "PSEUDO_BONDS",
        "HBOND": "PSEUDO_BONDS",
        "HBONDS": "PSEUDO_BONDS",
        "H_BONDS": "PSEUDO_BONDS",
        "SPECIAL": "SPECIAL_COORDINATES",
        "FRAGMENT_COORDINATES": "SPECIAL_COORDINATES",
        "FRAGMENT": "SPECIAL_COORDINATES",
        "NONE": "NONE",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"SPECIAL_COORDINATES", "PSEUDO_BONDS", "NONE"}:
        raise GICForgeContractError(f"unsupported fragment mode: {value}")
    return normalized
