"""SWITCH: internal SMILES and name-to-geometry services for MATRIX."""

from .geometry import (
    SWITCH_GEOMETRY_SCHEMA,
    build_cartesian_seed,
    complete_graph_hydrogens,
)
from .aromaticity import SWITCH_AROMATICITY_SCHEMA, perceive_aromaticity
from .depict import (
    SWITCH_DEPICTION_SCHEMA,
    MoleculeDepictionLayout,
    build_molecule_depiction_layout,
    render_molecule_png,
    render_molecule_svg,
)
from .model import (
    SWITCH_GRAPH_SCHEMA,
    SwitchAtom,
    SwitchBond,
    SwitchMolecularGraph,
    graph_from_topology,
)
from .names import NameResolution, NameResolutionError, resolve_name
from .matching import (
    CommonSubgraphMatch,
    find_substructure_matches,
    maximum_common_connected_subgraphs,
)
from .canonical import (
    SWITCH_CANONICAL_SCHEMA,
    canonical_atom_ranks,
    canonical_graph_key,
    canonical_smiles,
)
from .smarts import SMARTS_SUBSET_SCHEMA, SmartsSubsetError, parse_smarts
from .stereoisomers import (
    SWITCH_STEREO_ENUMERATION_SCHEMA,
    enumerate_stereoisomers,
)
from .backend_policy import (
    ALLOW_RDKIT_FALLBACK,
    STRICT_SWITCH,
    SWITCH_BACKEND_POLICY_SCHEMA,
    resolve_switch_backend_policy,
)
from .parser import (
    SmilesParseError,
    SwitchUnsupportedFeatureError,
    clear_smiles_parse_cache,
    parse_smiles,
)
from .rdkit_fallback import (
    RDKIT_FALLBACK_SCHEMA,
    RDKitFallbackUnavailableError,
    SwitchFallbackWarning,
    rdkit_fallback_available,
)
from .writer import write_smiles
from .workflow import name_to_cartesian, smiles_to_cartesian
from .validation import (
    HbondCandidate,
    GeometryFileValidation,
    SwitchSeedComparison,
    SwitchValidation,
    compare_switch_graphs,
    validate_switch_geometry,
    validate_geometry_file,
)

__all__ = [
    "SWITCH_GEOMETRY_SCHEMA",
    "SWITCH_AROMATICITY_SCHEMA",
    "SWITCH_DEPICTION_SCHEMA",
    "SWITCH_GRAPH_SCHEMA",
    "SWITCH_CANONICAL_SCHEMA",
    "SWITCH_STEREO_ENUMERATION_SCHEMA",
    "SMARTS_SUBSET_SCHEMA",
    "CommonSubgraphMatch",
    "MoleculeDepictionLayout",
    "NameResolution",
    "NameResolutionError",
    "SmilesParseError",
    "SwitchUnsupportedFeatureError",
    "RDKitFallbackUnavailableError",
    "SwitchFallbackWarning",
    "ALLOW_RDKIT_FALLBACK",
    "STRICT_SWITCH",
    "SWITCH_BACKEND_POLICY_SCHEMA",
    "RDKIT_FALLBACK_SCHEMA",
    "SmartsSubsetError",
    "SwitchAtom",
    "SwitchBond",
    "SwitchMolecularGraph",
    "build_cartesian_seed",
    "build_molecule_depiction_layout",
    "complete_graph_hydrogens",
    "canonical_atom_ranks",
    "canonical_graph_key",
    "canonical_smiles",
    "clear_smiles_parse_cache",
    "graph_from_topology",
    "find_substructure_matches",
    "enumerate_stereoisomers",
    "name_to_cartesian",
    "maximum_common_connected_subgraphs",
    "parse_smiles",
    "rdkit_fallback_available",
    "resolve_switch_backend_policy",
    "parse_smarts",
    "perceive_aromaticity",
    "resolve_name",
    "render_molecule_png",
    "render_molecule_svg",
    "smiles_to_cartesian",
    "write_smiles",
    "HbondCandidate",
    "GeometryFileValidation",
    "SwitchValidation",
    "validate_switch_geometry",
    "SwitchSeedComparison",
    "compare_switch_graphs",
    "validate_geometry_file",
]
