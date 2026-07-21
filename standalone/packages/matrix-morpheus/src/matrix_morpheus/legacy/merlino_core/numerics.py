from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankCondition:
    rank: int
    condition_number: float
    singular_values: np.ndarray


def objective(weighted_residual: np.ndarray) -> float:
    return 0.5 * float(weighted_residual @ weighted_residual)


def limit_step(step: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(step))
    if max_norm <= 0.0 or norm <= max_norm:
        return step
    return step * (max_norm / norm)


def damped_normal_step(weighted_jacobian: np.ndarray, weighted_residual: np.ndarray, damping: float) -> np.ndarray:
    lhs = weighted_jacobian.T @ weighted_jacobian + float(damping) * np.eye(weighted_jacobian.shape[1])
    rhs = weighted_jacobian.T @ weighted_residual
    return np.linalg.solve(lhs, rhs)


def rank_condition(matrix: np.ndarray) -> RankCondition:
    if matrix.size == 0:
        return RankCondition(rank=0, condition_number=float("inf"), singular_values=np.array(()))
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(matrix.shape) * np.finfo(float).eps * (float(singular[0]) if singular.size else 0.0)
    rank = int(np.sum(singular > threshold))
    if singular.size and singular[-1] > threshold:
        condition = float(singular[0] / singular[-1])
    else:
        condition = float("inf")
    return RankCondition(rank=rank, condition_number=condition, singular_values=singular)
