"""Compatibility facade for LINK point realization and evaluation services."""

from matrix_link import scan as _scan
from matrix_link.scan import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_scan, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_scan)))
