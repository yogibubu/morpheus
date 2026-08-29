"""Compatibility facade for the optimizer now owned by :mod:`matrix_link`."""

from matrix_link import optimizer as _optimizer
from matrix_link.optimizer import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_optimizer, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_optimizer)))
