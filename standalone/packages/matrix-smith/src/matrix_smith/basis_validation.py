from __future__ import annotations

import numpy as np

from .contracts import GICForgeContractError
from .coordinate_diagnostics import normalized_sonic_condition_diagnostics
from .evaluation import build_gic_b_matrix
from .models import GICDefinition
from .policy import FRAGMENT_MODE_NONE, MAX_NORMALIZED_SONIC_CONDITION


def validate_frozen_sonic_basis(
    definition: GICDefinition,
    *,
    rank_tolerance: float,
) -> None:
    """Reject a frozen SONIC chart that is rank-deficient or unusably conditioned."""

    matrix = build_gic_b_matrix(
        definition,
        coordinates_angstrom=definition.reference_coordinates_angstrom,
    )
    diagnostics = normalized_sonic_condition_diagnostics(
        np.asarray(matrix.rows, dtype=float),
        tolerance=rank_tolerance,
        maximum_condition_number=MAX_NORMALIZED_SONIC_CONDITION,
    )
    evaluated_rank = int(diagnostics["rank"])
    if evaluated_rank < definition.target_rank:
        raise GICForgeContractError(
            "frozen SONIC Jacobian is rank deficient at its reference geometry: "
            f"need {definition.target_rank}, evaluated rank {evaluated_rank}, "
            f"status {diagnostics['status']}"
        )
    condition = float(diagnostics["condition_number"])
    condition_gate_required = (
        definition.fragment_mode != FRAGMENT_MODE_NONE
        or "SCIENTIFIC_PATH TRANSITION_STATE" in definition.semantic_diagnostics
    )
    if condition_gate_required and (
        not np.isfinite(condition) or condition > MAX_NORMALIZED_SONIC_CONDITION
    ):
        raise GICForgeContractError(
            "frozen SONIC Jacobian exceeds the normalized condition gate at its "
            f"reference geometry: condition {condition:.12g}, maximum "
            f"{MAX_NORMALIZED_SONIC_CONDITION:.12g}"
        )
