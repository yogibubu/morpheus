"""Shared numerical kernels and compute-backend selection for MATRIX."""

from .compute import (
    COMPUTE_BACKEND_SCHEMA,
    DEFAULT_GPU_MIN_BATCH,
    ComputeBackend,
    available_compute_backends,
    compute_backend_report,
    coreml_compute_unit,
    gpu_validation_enabled,
    neural_engine_validation_enabled,
    resolve_compute_backend,
    resolve_neural_inference_backend,
)
from .diagonalizer import (
    DiagonalizerBackend,
    EighResult,
    available_diagonalizer_backends,
    best_diagonalizer_backend,
    diagonalize_hermitian,
    eigh_arrays,
    eigvalsh_array,
)
from .native import (
    DEFAULT_NATIVE_MIN_WORK,
    NATIVE_BACKEND_SCHEMA,
    NativeBackend,
    resolve_native_backend,
)
from .rank_revealing import (
    RankRevealingRowSelection,
    select_rank_revealing_rows,
)
from .linear_algebra import (
    SingularSpectrum,
    normalized_matrix_condition,
    numerical_matrix_rank,
    singular_spectrum,
    spectrum_rank,
)

__all__ = [
    "COMPUTE_BACKEND_SCHEMA",
    "DEFAULT_GPU_MIN_BATCH",
    "DEFAULT_NATIVE_MIN_WORK",
    "ComputeBackend",
    "DiagonalizerBackend",
    "EighResult",
    "NATIVE_BACKEND_SCHEMA",
    "NativeBackend",
    "RankRevealingRowSelection",
    "SingularSpectrum",
    "available_compute_backends",
    "available_diagonalizer_backends",
    "best_diagonalizer_backend",
    "compute_backend_report",
    "coreml_compute_unit",
    "diagonalize_hermitian",
    "eigh_arrays",
    "eigvalsh_array",
    "gpu_validation_enabled",
    "neural_engine_validation_enabled",
    "resolve_compute_backend",
    "resolve_neural_inference_backend",
    "resolve_native_backend",
    "normalized_matrix_condition",
    "numerical_matrix_rank",
    "select_rank_revealing_rows",
    "singular_spectrum",
    "spectrum_rank",
]
