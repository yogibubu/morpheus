"""Shared GICForge policy constants.

This module is the Python source of truth for the frozen GSNIC GIC contract.
Fortran kernels mirror these names and policies explicitly; downstream tools
should import these constants instead of hard-coding family names locally.
"""

from __future__ import annotations

from dataclasses import dataclass


COORDINATE_METHOD_NAME = "SONIC"
COORDINATE_METHOD_LONG_NAME = "Symmetry-Oriented Natural Internal Coordinates"
LEGACY_COORDINATE_METHOD_NAME = "SMITH"
GIC_BACKEND = "oracle-native-primitive.v1"
SYCART_BACKEND = "oracle-native-cartesian-nullspace.v1"
B_MATRIX_BACKEND = "oracle-native-analytic-bmatrix.v1"
RANK_METHOD = "analytic_b_matrix_mgs_greedy"
RANK_TOLERANCE = 1.0e-7
DIAGNOSTIC_FINITE_DIFFERENCE_STEP = 1.0e-5
# Gaussian/GDV DefRed and FCEstG classify an angle as linear when it lies
# within 15 degrees of pi.  All SONIC generators use this single threshold.
LINEAR_ANGLE_DEGREES = 165.0
PSEUDO_BOND_EFFECTIVE_ORDER = 0.25
PSEUDO_CYCLE_CLOSURE_VDW_SCALE = 1.05
ORDINARY_REDUCTION_CLASS = "ORDINARY"
SPECIAL_REDUCTION_CLASS = "SPECIAL_PROTECTED"
REDUCTION_POLICY = "SPECIAL_PROTECTED_FIRST_THEN_ORDINARY_ANALYTIC_RANK"
LOCAL_SYMMETRIZATION_METHOD = "LOCAL_BLOCK_SALC"
POINT_GROUP_PROJECTOR_METHOD = "POINT_GROUP_PROJECTOR"
SYMMETRIZATION_POLICY = "ORACLE_TYPE_BLOCK_SUM_AND_DIFFERENCE"
PROJECTOR_SYMMETRIZATION_POLICY = "HOMOGENEOUS_TYPE_BLOCK_POINT_GROUP_PROJECTOR"
SYMMETRY_OPERATION_TOLERANCE_ANGSTROM = 1.0e-3
SYMMETRY_OPERATION_NEAR_THRESHOLD_FRACTION = 0.25
SALC_PATH_OVERLAP_WARNING_THRESHOLD = 0.98
SALC_PATH_PIVOT_GAP_WARNING = 1.0e-6
FRAGMENT_MODE_SPECIAL_COORDINATES = "SPECIAL_COORDINATES"
FRAGMENT_MODE_PSEUDO_BONDS = "PSEUDO_BONDS"
FRAGMENT_MODE_NONE = "NONE"
XH_STRETCH_POLICY_SYMMETRIZE = "SYMMETRIZE"
XH_STRETCH_POLICY_LOCAL_ALL = "LOCAL_ALL"
XH_STRETCH_POLICY_LOCAL_SELECTED = "LOCAL_SELECTED"
XH_STRETCH_CLASS_XH = "XH"
XH_STRETCH_CLASS_XH2 = "XH2"
XH_STRETCH_CLASS_XH3 = "XH3"
XH_STRETCH_CLASSES = frozenset(
    {
        XH_STRETCH_CLASS_XH,
        XH_STRETCH_CLASS_XH2,
        XH_STRETCH_CLASS_XH3,
    }
)
XH_STRETCH_POLICIES = frozenset(
    {
        XH_STRETCH_POLICY_SYMMETRIZE,
        XH_STRETCH_POLICY_LOCAL_ALL,
        XH_STRETCH_POLICY_LOCAL_SELECTED,
    }
)
FRAGMENT_MODES = frozenset(
    {
        FRAGMENT_MODE_NONE,
        FRAGMENT_MODE_PSEUDO_BONDS,
        FRAGMENT_MODE_SPECIAL_COORDINATES,
    }
)


@dataclass(frozen=True)
class PrimitiveFamilyPolicy:
    family: str
    function: str
    prefix: str
    reduction_class: str
    symmetry_block: str


PRIMITIVE_FAMILY_POLICIES = (
    PrimitiveFamilyPolicy("STRETCH", "R", "Str", ORDINARY_REDUCTION_CLASS, "STRETCH"),
    PrimitiveFamilyPolicy(
        "LOCAL_XH_STRETCH",
        "R",
        "XHSt",
        ORDINARY_REDUCTION_CLASS,
        "LOCAL_XH_STRETCH",
    ),
    PrimitiveFamilyPolicy("BEND", "A", "Bend", ORDINARY_REDUCTION_CLASS, "BEND"),
    PrimitiveFamilyPolicy(
        "CYCLIC_BEND",
        "A",
        "CyBe",
        ORDINARY_REDUCTION_CLASS,
        "CYCLIC_BEND",
    ),
    PrimitiveFamilyPolicy(
        "SPIRO_BEND",
        "A",
        "Spir",
        ORDINARY_REDUCTION_CLASS,
        "SPIRO_BEND",
    ),
    PrimitiveFamilyPolicy(
        "LINEAR_BEND",
        "L",
        "LinB",
        ORDINARY_REDUCTION_CLASS,
        "LINEAR_BEND",
    ),
    PrimitiveFamilyPolicy("TORSION", "D", "Tors", ORDINARY_REDUCTION_CLASS, "TORSION"),
    PrimitiveFamilyPolicy(
        "PSEUDO_CYCLE_BEND",
        "RPCB",
        "PsAn",
        ORDINARY_REDUCTION_CLASS,
        "PSEUDO_CYCLE_BEND",
    ),
    PrimitiveFamilyPolicy(
        "PSEUDO_CYCLE_TORSION",
        "RPCK",
        "PsTo",
        ORDINARY_REDUCTION_CLASS,
        "PSEUDO_CYCLE_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "CYCLIC_TORSION",
        "D",
        "CyTo",
        ORDINARY_REDUCTION_CLASS,
        "CYCLIC_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "RING_PUCKER_COMPONENT",
        "RPCK",
        "RPck",
        ORDINARY_REDUCTION_CLASS,
        "RING_PUCKER_COMPONENT",
    ),
    PrimitiveFamilyPolicy(
        "CONDENSED_RING_TORSION",
        "D",
        "CoTo",
        ORDINARY_REDUCTION_CLASS,
        "CONDENSED_RING_TORSION",
    ),
    PrimitiveFamilyPolicy(
        "BUTTERFLY",
        "D",
        "BtFl",
        ORDINARY_REDUCTION_CLASS,
        "BUTTERFLY",
    ),
    PrimitiveFamilyPolicy(
        "OUT_OF_PLANE",
        "U",
        "OuPl",
        ORDINARY_REDUCTION_CLASS,
        "OUT_OF_PLANE",
    ),
    PrimitiveFamilyPolicy(
        "IMPROPER_DIHEDRAL",
        "IMPD",
        "ImpD",
        ORDINARY_REDUCTION_CLASS,
        "OUT_OF_PLANE",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_DISTANCE",
        "FC_DIST",
        "FCDi",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_DISTANCE",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_CENTER_ATOM_DISTANCE",
        "FCA_DIST",
        "FCAt",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_CENTER_ATOM",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_TRANSLATION",
        "FTRANS",
        "FTrn",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_TRANSLATION",
    ),
    PrimitiveFamilyPolicy(
        "FRAG_ORIENTATION",
        "FROT",
        "FRot",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_FRAGMENT_ORIENTATION",
    ),
    PrimitiveFamilyPolicy(
        "CENTER_ATOM_DISTANCE",
        "CENTER_ATOM_DIST",
        "CnAt",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_CENTER_ATOM",
    ),
    PrimitiveFamilyPolicy(
        "HBOND_DISTANCE",
        "R",
        "HBDi",
        SPECIAL_REDUCTION_CLASS,
        "SPECIAL_HBOND_DISTANCE",
    ),
)

PRIMITIVE_POLICY_BY_FAMILY = {policy.family: policy for policy in PRIMITIVE_FAMILY_POLICIES}
PRIMITIVE_POLICY_BY_FUNCTION = {policy.function: policy for policy in PRIMITIVE_FAMILY_POLICIES}
PRIMITIVE_FAMILY_ORDER = tuple(policy.family for policy in PRIMITIVE_FAMILY_POLICIES)
SPECIAL_PRIMITIVE_FAMILIES = frozenset(
    policy.family
    for policy in PRIMITIVE_FAMILY_POLICIES
    if policy.reduction_class == SPECIAL_REDUCTION_CLASS
)
SYMMETRY_BLOCK_BY_FAMILY = {
    policy.family: policy.symmetry_block for policy in PRIMITIVE_FAMILY_POLICIES
}


def primitive_reduction_class(family: str) -> str:
    policy = PRIMITIVE_POLICY_BY_FAMILY.get(family)
    if policy is None:
        return ORDINARY_REDUCTION_CLASS
    return policy.reduction_class


def primitive_prefix(family: str) -> str:
    return PRIMITIVE_POLICY_BY_FAMILY[family].prefix


def primitive_symmetry_block(family: str) -> str:
    return SYMMETRY_BLOCK_BY_FAMILY.get(family, family)
