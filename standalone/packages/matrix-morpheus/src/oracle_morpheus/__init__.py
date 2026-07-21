"""Compatibility alias for :mod:`matrix_morpheus`."""

from importlib import import_module
import sys

_module = import_module("matrix_morpheus")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
