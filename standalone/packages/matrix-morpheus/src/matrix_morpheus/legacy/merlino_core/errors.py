from __future__ import annotations


class MerlinoError(RuntimeError):
    """Base class for user-facing Merlino errors."""


class InputError(MerlinoError):
    """Invalid, missing or inconsistent user input."""


class BackendError(MerlinoError):
    """External backend failure or unavailable executable."""


class ParseError(MerlinoError):
    """Input/output parsing failed."""


class ScientificValidationError(MerlinoError):
    """Scientific contract or consistency check failed."""
