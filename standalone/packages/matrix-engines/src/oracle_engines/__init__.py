"""Compatibility alias for :mod:`matrix_engines`."""

from importlib import import_module
import sys

_module = import_module("matrix_engines")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
