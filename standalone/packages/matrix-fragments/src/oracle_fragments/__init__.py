"""Compatibility alias for :mod:`matrix_fragments`."""

from importlib import import_module
import sys

_module = import_module("matrix_fragments")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
