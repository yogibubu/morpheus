"""Global audits and construction for direct sums of typed ONIC blocks.

Individual coordinate builders retain ownership of their scientific kernels.
This module only verifies that their ordered Wilson rows span exactly one
whole-system internal Cartesian space and then freezes the composite contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem import MolecularSymmetry, cartesian_operation_matrix

from .bmatrix import SparseBMatrix
from .cartesian_blocks import symmetry_adapted_cartesian_basis
from .onic_blocks import (
    CompositeOnicDefinition,
    OnicCoordinateBlock,
    OnicGlobalAudit,
    onic_reference_fingerprint,
)


COMPOSITE_ONIC_ABSOLUTE_RANK_TOLERANCE = 1.0e-10
COMPOSITE_ONIC_RELATIVE_RANK_TOLERANCE = 1.0e-8
COMPOSITE_ONIC_ROW_SPACE_TOLERANCE = 5.0e-8


@dataclass(frozen=True)
class CompositeOnicJacobianAudit:
    """Numerical evidence underlying a serialized global ONIC audit."""

    global_audit: OnicGlobalAudit
    singular_values: tuple[float, ...]
    singular_value_cutoff: float
    row_space_residual: float
    rigid_mode_residual: float
    covariance_residual: float


def audit_composite_onic_jacobian(
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    blocks: Sequence[OnicCoordinateBlock],
    block_b_matrices: Sequence[SparseBMatrix],
    symmetry: MolecularSymmetry,
    site_anchor_atom_indices_one_based: Sequence[int] = (),
    rank_absolute_tolerance: float = COMPOSITE_ONIC_ABSOLUTE_RANK_TOLERANCE,
    rank_relative_tolerance: float = COMPOSITE_ONIC_RELATIVE_RANK_TOLERANCE,
    row_space_tolerance: float = COMPOSITE_ONIC_ROW_SPACE_TOLERANCE,
) -> CompositeOnicJacobianAudit:
    """Audit the ordered block Jacobian against the exact internal subspace.

    Row normalization is used only for the scale-independent rank and
    conditioning audit.  It does not alter any coordinate definition or the
    serialized block operators.
    """

    symbols = tuple(str(atom) for atom in atoms)
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if reference.shape != (len(symbols), 3) or not np.all(np.isfinite(reference)):
        raise ValueError("composite ONIC reference must be a finite natoms-by-3 array")
    frozen_blocks = tuple(blocks)
    matrices = tuple(block_b_matrices)
    if not frozen_blocks or len(frozen_blocks) != len(matrices):
        raise ValueError("composite ONIC audit needs one Jacobian per nonempty block list")
    absolute_tolerance = _positive_finite(rank_absolute_tolerance, "absolute rank tolerance")
    relative_tolerance = _positive_finite(rank_relative_tolerance, "relative rank tolerance")
    row_space_limit = _positive_finite(row_space_tolerance, "row-space tolerance")
    expected_labels: list[str] = []
    dense_rows: list[np.ndarray] = []
    for block, matrix in zip(frozen_blocks, matrices, strict=True):
        if matrix.column_count != reference.size:
            raise ValueError(
                f"block {block.identifier} Jacobian has {matrix.column_count} columns; "
                f"expected {reference.size}"
            )
        if matrix.row_count != block.target_rank:
            raise ValueError(
                f"block {block.identifier} Jacobian has {matrix.row_count} rows; "
                f"expected {block.target_rank}"
            )
        if matrix.row_labels and matrix.row_labels != block.coordinate_identifiers:
            raise ValueError(
                f"block {block.identifier} Jacobian row order contradicts its contract"
            )
        expected_labels.extend(block.coordinate_identifiers)
        dense_rows.append(matrix.to_dense())
    if len(set(expected_labels)) != len(expected_labels):
        raise ValueError("composite ONIC coordinate identifiers are not globally unique")

    jacobian = np.vstack(dense_rows)
    row_norms = np.linalg.norm(jacobian, axis=1)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= absolute_tolerance):
        raise ValueError("composite ONIC Jacobian contains a zero or non-finite row")
    normalized = jacobian / row_norms[:, None]
    _left, singular_values, right_transpose = np.linalg.svd(normalized, full_matrices=False)
    cutoff = max(absolute_tolerance, relative_tolerance * float(singular_values[0]))
    evaluated_rank = int(np.count_nonzero(singular_values > cutoff))
    target_rank = sum(block.target_rank for block in frozen_blocks)

    complete_basis = symmetry_adapted_cartesian_basis(
        symbols,
        reference,
        symmetry=symmetry,
        site_anchor_atom_indices_one_based=site_anchor_atom_indices_one_based,
        require_site_anchor=bool(site_anchor_atom_indices_one_based),
        rank_absolute_tolerance=absolute_tolerance,
        rank_relative_tolerance=relative_tolerance,
    )
    if target_rank != complete_basis.target_rank:
        raise ValueError(
            "composite ONIC declared rank does not equal the whole-system internal rank: "
            f"declared={target_rank}, required={complete_basis.target_rank}"
        )
    active_right = right_transpose[:evaluated_rank]
    row_projector = active_right.T @ active_right
    expected_projector = complete_basis.cartesian_from_q @ complete_basis.cartesian_from_q.T
    row_space_residual = float(np.linalg.norm(row_projector - expected_projector, ord=2))
    rigid_projector = np.eye(reference.size) - expected_projector
    rigid_mode_residual = float(np.linalg.norm(normalized @ rigid_projector, ord=2))
    covariance_residual = max(
        (
            float(
                np.linalg.norm(
                    operation @ row_projector @ operation.T - row_projector,
                    ord=2,
                )
            )
            for operation in (
                cartesian_operation_matrix(
                    np.asarray(item.rotation, dtype=float),
                    tuple(atom - 1 for atom in item.permutation),
                    natoms=len(symbols),
                )
                for item in symmetry.operations
            )
        ),
        default=0.0,
    )
    maximum_residual = max(row_space_residual, rigid_mode_residual, covariance_residual)
    status = (
        "PASS" if evaluated_rank == target_rank and maximum_residual <= row_space_limit else "FAIL"
    )
    condition_number = (
        float(singular_values[0] / singular_values[evaluated_rank - 1])
        if evaluated_rank
        else float("inf")
    )
    global_audit = OnicGlobalAudit(
        status=status,
        cartesian_dimension=reference.size,
        external_mode_count=complete_basis.external_mode_count,
        target_rank=target_rank,
        evaluated_rank=evaluated_rank,
        nullity=reference.size - evaluated_rank,
        covariance_residual=covariance_residual,
        condition_number=condition_number,
        messages=(
            "JACOBIAN_BLOCK_ORDER=" + ",".join(block.identifier for block in frozen_blocks),
            "RANK_AUDIT=ROW_NORMALIZED_SVD",
            f"SINGULAR_VALUE_CUTOFF={cutoff:.12g}",
            f"ROW_SPACE_RESIDUAL={row_space_residual:.12g}",
            f"RIGID_MODE_RESIDUAL={rigid_mode_residual:.12g}",
            "DIRECT_SUM_DOES_NOT_ASSERT_CARTESIAN_ROW_ORTHOGONALITY",
        ),
    )
    return CompositeOnicJacobianAudit(
        global_audit=global_audit,
        singular_values=tuple(float(value) for value in singular_values),
        singular_value_cutoff=cutoff,
        row_space_residual=row_space_residual,
        rigid_mode_residual=rigid_mode_residual,
        covariance_residual=covariance_residual,
    )


def build_audited_composite_onic_definition(
    atoms: Sequence[str],
    reference_coordinates_angstrom: np.ndarray | Sequence[Sequence[float]],
    *,
    blocks: Sequence[OnicCoordinateBlock],
    block_b_matrices: Sequence[SparseBMatrix],
    symmetry: MolecularSymmetry,
    site_anchor_atom_indices_one_based: Sequence[int] = (),
    orientation: str = "SONIC",
    provenance: Sequence[str] = (),
) -> CompositeOnicDefinition:
    """Return a frozen composite only after the global Jacobian audit passes."""

    symbols = tuple(str(atom) for atom in atoms)
    reference_array = np.asarray(reference_coordinates_angstrom, dtype=float)
    frozen_blocks = tuple(blocks)
    audit = audit_composite_onic_jacobian(
        symbols,
        reference_array,
        blocks=frozen_blocks,
        block_b_matrices=block_b_matrices,
        symmetry=symmetry,
        site_anchor_atom_indices_one_based=site_anchor_atom_indices_one_based,
    )
    if audit.global_audit.status != "PASS":
        raise ValueError(
            "composite ONIC Jacobian failed its global rank/null-space/covariance audit: "
            f"rank={audit.global_audit.evaluated_rank}/{audit.global_audit.target_rank}, "
            f"row-space={audit.row_space_residual:.3e}, "
            f"rigid={audit.rigid_mode_residual:.3e}, "
            f"covariance={audit.covariance_residual:.3e}"
        )
    atom_indices = tuple(range(1, len(symbols) + 1))
    reference = tuple(tuple(float(value) for value in row) for row in reference_array)
    return CompositeOnicDefinition(
        orientation=orientation,
        atom_indices_one_based=atom_indices,
        reference_coordinates_angstrom=reference,
        reference_fingerprint_sha256=onic_reference_fingerprint(atom_indices, reference),
        blocks=frozen_blocks,
        global_audit=audit.global_audit,
        provenance=tuple(str(item) for item in provenance),
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result
