"""Fortran backend discovery and execution contracts for Merlino4."""

from .backends import (
    BackendSpec,
    SourceBackendSpec,
    backend_executable,
    resolve_backend,
    resolve_source_backend,
)

__all__ = [
    "BackendSpec",
    "SourceBackendSpec",
    "backend_executable",
    "resolve_backend",
    "resolve_source_backend",
]
