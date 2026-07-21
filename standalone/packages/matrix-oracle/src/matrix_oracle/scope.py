"""Frozen publication boundary for ORACLE inside MATRIX."""

from __future__ import annotations

from typing import Any


ORACLE_SCOPE_SCHEMA = "matrix.oracle.scope.v1"

ORACLE_OWNED_CAPABILITIES = (
    "geometry_import_and_normalization",
    "continuous_and_discrete_topology",
    "fragments_pseudobonds_rings_and_aromaticity",
    "orientation_covariant_symmetry_and_thresholded_projection",
    "continuous_sigma_pi_pi_pi_bond_indices",
    "effective_atomic_numbers_synthons_and_atom_classes",
    "redundant_primitive_internal_coordinates_and_wilson_b_matrix",
    "primitive_space_structural_improvement_with_cv_conjugation_and_selected_pairs",
)

DOWNSTREAM_OWNERSHIP = {
    "SONIC construction from the primitive Wilson B matrix": "SMITH",
    "internal-to-Cartesian realization, geometry optimization and scans": "LINK",
    "semiexperimental refinement without coordinate back-transformation": "MORPHEUS",
    "B-prime, Hessian transformation, force-field construction and ZION": "ARCHITECT",
    "PES-point proposal through the LINK protocol": "SENTINEL",
    "multilevel derivative assembly and vibrational perturbation theory": "TRINITY",
}

ORACLE_EXCLUDED_CAPABILITIES = (
    "force_field_parameter_fitting",
    "hessian_coupling_selection",
    "zion_compilation_and_runtime",
    "global_or_local_geometry_optimization",
    "normal_mode_and_vibrational_property_analysis",
    "internal_to_cartesian_realization",
    "b_matrix_first_derivative",
)


def oracle_scope_contract() -> dict[str, Any]:
    """Return the machine-readable ownership boundary used by docs and reports."""

    return {
        "schema": ORACLE_SCOPE_SCHEMA,
        "owner": "ORACLE",
        "owned_capabilities": list(ORACLE_OWNED_CAPABILITIES),
        "excluded_capabilities": list(ORACLE_EXCLUDED_CAPABILITIES),
        "downstream_ownership": dict(DOWNSTREAM_OWNERSHIP),
        "scientific_model": {
            "observables_with_qm": "CM5 charges + Mayer bond orders",
            "observables_without_qm": (
                "electronegativity charges + Pauling bond orders"
            ),
            "availability_rule": (
                "the QM pair is used only when both complete vectors are available; "
                "otherwise the geometry-only pair is used"
            ),
            "synthon": "one canonical seven-component continuous descriptor",
            "cv_posterior": "radius-aware-period-line Gaussian",
            "descriptor_vdw_radii": "Merz-Kollman/Bondi",
            "uff_nonbonded_radii": "UFF only inside the UFF potential",
        },
        "compatibility_policy": (
            "legacy schemas are accepted only as migration input; every new ORACLE "
            "state emits one MATRIX schema and one scientific model, while capabilities "
            "owned by other tools are not exported by the standalone ORACLE public API"
        ),
    }
