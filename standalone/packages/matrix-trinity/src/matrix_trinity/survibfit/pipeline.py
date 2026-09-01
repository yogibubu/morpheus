"""Compatibility facade for the SMITH-owned SONIC topology kernel."""

from matrix_smith.survibfit import pipeline as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)
