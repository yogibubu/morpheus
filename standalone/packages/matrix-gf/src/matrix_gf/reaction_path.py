"""SONIC Wilson-GF tools for relaxed Qim-type reaction paths.

The path direction is represented in the frozen, non-redundant SONIC tangent
space.  At a non-stationary path point the direction is removed with the
Wilson kinetic metric before the transverse Hessian is diagonalized.  This is
the coordinate-space counterpart of constructing curvilinear normal modes
orthogonal to a relaxed reaction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .harmonic import HESSIAN_EIGENVALUE_TO_CM


@dataclass(frozen=True)
class SonicQimDirection:
    """Unstable saddle-point direction normalized in the Wilson metric."""

    eigenvalue: float
    frequency_cm: float
    sonic_direction: np.ndarray
    constraint_covector: np.ndarray
    negative_mode_count: int


@dataclass(frozen=True)
class SonicTransverseModes:
    """Path-orthogonal SONIC modes at a stationary or non-stationary point."""

    path_tangent: np.ndarray
    transverse_basis: np.ndarray
    projected_force_constants: np.ndarray
    eigenvalues: np.ndarray
    frequencies_cm: np.ndarray
    sonic_modes: np.ndarray
    gradient_transverse_fraction: float | None
    maximum_path_hessian_coupling: float


def curvilinear_sonic_force_constants(
    cartesian_hessian: np.ndarray,
    cartesian_gradient: np.ndarray,
    coordinates: np.ndarray,
    evaluate_b_matrix: Callable[[np.ndarray], np.ndarray],
    *,
    masses: np.ndarray | None = None,
    internal_step: float = 1.0e-4,
    stationary_gradient_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Transform a Cartesian Hessian at a non-stationary point to SONIC space.

    The derivative of the contravariant back-transform is retained by central
    finite differences of the internal gradient.  This is the gradient-times-
    coordinate-curvature term that distinguishes curvilinear and rectilinear
    generalized normal modes away from stationary points.  When the Cartesian
    gradient vanishes, the routine returns the ordinary Wilson transformation
    exactly, so the two definitions coincide at a stationary point.

    Coordinates, gradients, Hessians, and the Wilson B matrix must use one
    consistent Cartesian length unit.  ``masses`` may contain one value per
    atom or one value per Cartesian component.
    """

    coords = np.asarray(coordinates, dtype=float)
    shape = coords.shape
    flat = coords.reshape(-1)
    gradient = np.asarray(cartesian_gradient, dtype=float).reshape(-1)
    hessian = np.asarray(cartesian_hessian, dtype=float)
    if gradient.shape != flat.shape or hessian.shape != (flat.size, flat.size):
        raise ValueError("Cartesian coordinates, gradient, and Hessian have incompatible sizes")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise ValueError("Cartesian derivatives must be finite")
    step = float(internal_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("internal finite-difference step must be positive")
    inverse_mass = _cartesian_inverse_mass(masses, flat.size)
    b_matrix = _validated_b_matrix(evaluate_b_matrix(coords), flat.size)
    backtransform = _wilson_backtransform(b_matrix, inverse_mass)
    ordinary = backtransform.T @ hessian @ backtransform
    ordinary = 0.5 * (ordinary + ordinary.T)
    if float(np.linalg.norm(gradient)) <= stationary_gradient_tolerance:
        return ordinary

    internal_gradient = backtransform.T @ gradient
    result = np.empty_like(ordinary)
    for column in range(b_matrix.shape[0]):
        displacement = step * backtransform[:, column]
        plus_coords = (flat + displacement).reshape(shape)
        minus_coords = (flat - displacement).reshape(shape)
        plus_b = _validated_b_matrix(evaluate_b_matrix(plus_coords), flat.size)
        minus_b = _validated_b_matrix(evaluate_b_matrix(minus_coords), flat.size)
        if plus_b.shape != b_matrix.shape or minus_b.shape != b_matrix.shape:
            raise ValueError("SONIC B-matrix dimension changed along curvilinear displacement")
        plus_backtransform = _wilson_backtransform(plus_b, inverse_mass)
        minus_backtransform = _wilson_backtransform(minus_b, inverse_mass)
        plus_gradient = gradient + hessian @ displacement
        minus_gradient = gradient - hessian @ displacement
        plus_internal_gradient = plus_backtransform.T @ plus_gradient
        minus_internal_gradient = minus_backtransform.T @ minus_gradient
        result[:, column] = (
            plus_internal_gradient - minus_internal_gradient
        ) / (2.0 * step)
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)):
        raise ValueError("curvilinear SONIC Hessian contains non-finite values")
    # Retain the variable to make the differentiated quantity explicit and to
    # guard against a future accidental replacement by B H B^T.
    del internal_gradient
    return result


def sonic_qim_direction(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    *,
    require_first_order_saddle: bool = True,
) -> SonicQimDirection:
    """Return the rectilinear imaginary-mode direction in SONIC coordinates.

    The returned vector ``l`` obeys ``l.T @ inv(G) @ l = 1``.  Consequently a
    scalar displacement along it is a mass-scaled Wilson normal coordinate,
    as required by the Bowman Qim construction.
    """

    f_mat, g_mat, g_half, _g_inverse_half = _validated_fg(force_constants, g_matrix)
    symmetric = g_half @ f_mat @ g_half
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    negative_count = int(np.count_nonzero(eigenvalues < 0.0))
    if negative_count == 0:
        raise ValueError("Qim construction requires an imaginary saddle-point mode")
    if require_first_order_saddle and negative_count != 1:
        raise ValueError(
            f"Qim construction requires a first-order saddle, found {negative_count} negative modes"
        )
    eigenvalue = float(eigenvalues[0])
    direction = g_half @ eigenvectors[:, 0]
    direction = _canonical_phase(direction)
    constraint = np.linalg.solve(g_mat, direction)
    frequency = -float(np.sqrt(abs(eigenvalue)) * HESSIAN_EIGENVALUE_TO_CM)
    return SonicQimDirection(
        eigenvalue=eigenvalue,
        frequency_cm=frequency,
        sonic_direction=direction,
        constraint_covector=constraint,
        negative_mode_count=negative_count,
    )


def sonic_transverse_modes(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    path_tangent: np.ndarray,
    *,
    gradient: np.ndarray | None = None,
) -> SonicTransverseModes:
    """Diagonalize a curvilinear SONIC Hessian in the path-transverse subspace.

    ``force_constants`` must include the gradient-dependent coordinate-
    curvature term at a non-stationary point, as returned by
    :func:`curvilinear_sonic_force_constants`.  ``gradient`` is the covariant
    derivative ``dE/dq``.  Its transverse
    fraction is zero on a perfectly relaxed path and therefore supplies a
    direct convergence diagnostic at non-stationary points.
    """

    f_mat, g_mat, g_half, g_inverse_half = _validated_fg(force_constants, g_matrix)
    tangent = np.asarray(path_tangent, dtype=float).reshape(-1)
    if tangent.shape != (f_mat.shape[0],) or not np.all(np.isfinite(tangent)):
        raise ValueError("path tangent must be a finite vector in the SONIC basis")
    metric_tangent = g_inverse_half @ tangent
    norm = float(np.linalg.norm(metric_tangent))
    if norm <= 1.0e-14:
        raise ValueError("path tangent has zero Wilson-metric norm")
    metric_tangent /= norm
    tangent = g_half @ metric_tangent
    tangent = _canonical_phase(tangent)
    metric_tangent = g_inverse_half @ tangent

    # Complete QR gives an orthonormal complement in mass-scaled SONIC space.
    q_matrix, _ = np.linalg.qr(metric_tangent[:, None], mode="complete")
    transverse_orthogonal = q_matrix[:, 1:]
    transverse_basis = g_half @ transverse_orthogonal
    projected = transverse_basis.T @ f_mat @ transverse_basis
    projected = 0.5 * (projected + projected.T)
    eigenvalues, vectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    modes = transverse_basis @ vectors
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies *= HESSIAN_EIGENVALUE_TO_CM

    tangent_curvature = float(tangent @ f_mat @ tangent)
    cross = tangent @ f_mat @ modes
    transverse_curvatures = np.diag(modes.T @ f_mat @ modes)
    denominators = np.sqrt(np.abs(tangent_curvature * transverse_curvatures))
    normalized_cross = np.divide(
        np.abs(cross),
        denominators,
        out=np.zeros_like(cross),
        where=denominators > 1.0e-14,
    )

    transverse_fraction: float | None = None
    if gradient is not None:
        covariant_gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if covariant_gradient.shape != tangent.shape or not np.all(
            np.isfinite(covariant_gradient)
        ):
            raise ValueError("gradient must be a finite covariant SONIC vector")
        orthogonal_gradient = g_half @ covariant_gradient
        total = float(np.linalg.norm(orthogonal_gradient))
        transverse = transverse_orthogonal.T @ orthogonal_gradient
        transverse_fraction = float(np.linalg.norm(transverse) / max(total, 1.0e-30))

    return SonicTransverseModes(
        path_tangent=tangent,
        transverse_basis=transverse_basis,
        projected_force_constants=projected,
        eigenvalues=eigenvalues,
        frequencies_cm=frequencies,
        sonic_modes=modes,
        gradient_transverse_fraction=transverse_fraction,
        maximum_path_hessian_coupling=(
            float(np.max(normalized_cross)) if normalized_cross.size else 0.0
        ),
    )


def _validated_fg(
    force_constants: np.ndarray, g_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_mat = np.asarray(force_constants, dtype=float)
    g_mat = np.asarray(g_matrix, dtype=float)
    if f_mat.ndim != 2 or f_mat.shape[0] != f_mat.shape[1] or f_mat.shape != g_mat.shape:
        raise ValueError("F and G must be square matrices with the same shape")
    if not np.all(np.isfinite(f_mat)) or not np.all(np.isfinite(g_mat)):
        raise ValueError("F and G must contain only finite values")
    f_mat = 0.5 * (f_mat + f_mat.T)
    g_mat = 0.5 * (g_mat + g_mat.T)
    values, vectors = np.linalg.eigh(g_mat)
    if np.any(values <= 0.0):
        raise ValueError("G matrix must be positive definite")
    g_half = (vectors * np.sqrt(values)) @ vectors.T
    g_inverse_half = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    return f_mat, g_mat, g_half, g_inverse_half


def _canonical_phase(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _cartesian_inverse_mass(masses: np.ndarray | None, size: int) -> np.ndarray:
    if masses is None:
        return np.ones(size, dtype=float)
    values = np.asarray(masses, dtype=float).reshape(-1)
    if values.size * 3 == size:
        values = np.repeat(values, 3)
    if values.size != size or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("masses must be positive and match atoms or Cartesian components")
    return 1.0 / values


def _validated_b_matrix(b_matrix: np.ndarray, cartesian_size: int) -> np.ndarray:
    matrix = np.asarray(b_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != cartesian_size:
        raise ValueError("SONIC B matrix has incompatible Cartesian dimension")
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) != matrix.shape[0]:
        raise ValueError("SONIC B matrix must be finite and have independent rows")
    return matrix


def _wilson_backtransform(b_matrix: np.ndarray, inverse_mass: np.ndarray) -> np.ndarray:
    metric = (b_matrix * inverse_mass[None, :]) @ b_matrix.T
    return (inverse_mass[:, None] * b_matrix.T) @ np.linalg.inv(metric)
