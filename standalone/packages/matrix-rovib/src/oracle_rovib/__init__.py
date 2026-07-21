"""Compatibility alias for :mod:`matrix_rovib`."""

from importlib import import_module
import sys

_module = import_module("matrix_rovib")
globals().update(_module.__dict__)
sys.modules[__name__] = _module
