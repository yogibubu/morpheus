from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SEMIEXP_OBSERVABLE = "moments"
DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS = "auto"
DEFAULT_SEMIEXP_ROBUST_LOSS = "none"
HYDROGEN_PARAMETER_CONSTRAINT = "@hydrogen_parameters"


@dataclass(frozen=True)
class RotationalConstants:
    """Rotational constants in MHz."""

    A_MHz: float
    B_MHz: float
    C_MHz: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.A_MHz, self.B_MHz, self.C_MHz)


@dataclass(frozen=True)
class VibrationalCorrection:
    """QM vibrational correction Delta B_vib in MHz."""

    delta_A_MHz: float = 0.0
    delta_B_MHz: float = 0.0
    delta_C_MHz: float = 0.0
    source: str = "unspecified"
    convention: str = "subtract"

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.delta_A_MHz, self.delta_B_MHz, self.delta_C_MHz)

    @property
    def sign(self) -> float:
        return _correction_sign(self.convention)


@dataclass(frozen=True)
class ElectronicCorrection:
    """Electronic correction Delta B_elec in MHz."""

    delta_A_MHz: float = 0.0
    delta_B_MHz: float = 0.0
    delta_C_MHz: float = 0.0
    source: str = "unspecified"
    convention: str = "subtract"

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.delta_A_MHz, self.delta_B_MHz, self.delta_C_MHz)

    @property
    def sign(self) -> float:
        return _correction_sign(self.convention)


@dataclass(frozen=True)
class CorrectedRotationalConstants:
    """Experimental constants corrected to semiexperimental equilibrium values."""

    observed: RotationalConstants
    correction: VibrationalCorrection
    electronic_correction: ElectronicCorrection = field(default_factory=ElectronicCorrection)

    @property
    def equilibrium(self) -> RotationalConstants:
        a, b, c = self.observed.as_tuple()
        da, db, dc = self.correction.as_tuple()
        ea, eb, ec = self.electronic_correction.as_tuple()
        vib_sign = self.correction.sign
        elec_sign = self.electronic_correction.sign
        return RotationalConstants(
            a + vib_sign * da + elec_sign * ea,
            b + vib_sign * db + elec_sign * eb,
            c + vib_sign * dc + elec_sign * ec,
        )


@dataclass(frozen=True)
class IsotopologueObservation:
    label: str
    constants: RotationalConstants
    substitutions: dict[int, int] = field(default_factory=dict)
    correction: VibrationalCorrection = field(default_factory=VibrationalCorrection)
    electronic_correction: ElectronicCorrection = field(default_factory=ElectronicCorrection)
    weights: RotationalConstants | None = None

    @property
    def corrected(self) -> RotationalConstants:
        return CorrectedRotationalConstants(
            self.constants, self.correction, self.electronic_correction
        ).equilibrium


@dataclass(frozen=True)
class QMParameterPredicate:
    """Weighted QM prior/predicate for a generated GIC parameter."""

    label_pattern: str
    value: float
    sigma: float
    source: str = "qm"

    @property
    def weight(self) -> float:
        if self.sigma <= 0.0:
            raise ValueError("QM predicate sigma must be positive")
        return 1.0 / (self.sigma * self.sigma)


@dataclass(frozen=True)
class ParameterClassConstraint:
    """Constraint applied to a class of generated GIC parameters.

    `shared` means all matched GICs receive one common least-squares
    correction. `fixed` keeps the whole class blocked.
    """

    name: str
    patterns: tuple[str, ...]
    mode: str = "shared"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Parameter class name cannot be empty")
        if self.mode not in {"shared", "fixed"}:
            raise ValueError("Parameter class mode must be shared or fixed")
        if not self.patterns or any(not pattern.strip() for pattern in self.patterns):
            raise ValueError("Parameter class patterns cannot be empty")


@dataclass(frozen=True)
class SemiexperimentalFitRequest:
    initial_geometry: Path
    observations: tuple[IsotopologueObservation, ...]
    fixed_parameters: tuple[str, ...] = ()
    observable: str = DEFAULT_SEMIEXP_OBSERVABLE
    rotational_components: str = DEFAULT_SEMIEXP_ROTATIONAL_COMPONENTS
    qm_predicates: tuple[QMParameterPredicate, ...] = ()
    parameter_classes: tuple[ParameterClassConstraint, ...] = ()
    coordinate_model: str = "gic"
    robust_loss: str = DEFAULT_SEMIEXP_ROBUST_LOSS
    robust_scale: float = 0.0
    leave_one_out: bool = False
    excluded_rotational_constants: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.observations:
            raise ValueError("Semiexperimental fit needs at least one isotopologue")
        labels = [item.label for item in self.observations]
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate isotopologue labels are not allowed")
        if self.observable not in {"moments", "rotational_constants", "auto"}:
            raise ValueError("observable must be moments, rotational_constants or auto")
        if self.rotational_components not in {"auto", "ABC", "AB", "AC", "BC"}:
            raise ValueError("rotational_components must be auto, ABC, AB, AC or BC")
        if self.coordinate_model not in {"gic", "cartesian_symmetry"}:
            raise ValueError("coordinate_model must be gic or cartesian_symmetry")
        if self.robust_loss not in {"none", "huber", "soft_l1", "cauchy"}:
            raise ValueError("robust_loss must be none, huber, soft_l1 or cauchy")
        if self.robust_scale < 0.0:
            raise ValueError("robust_scale must be non-negative")
        known_labels = set(labels)
        for exclusion in self.excluded_rotational_constants:
            label, separator, component = str(exclusion).rpartition(":")
            if not separator or label not in known_labels or component.upper() not in {"A", "B", "C"}:
                raise ValueError(
                    "Excluded rotational constants must use an existing "
                    "isotopologue label and A, B or C: LABEL:COMPONENT"
                )
        for predicate in self.qm_predicates:
            if not predicate.label_pattern.strip():
                raise ValueError("QM predicate label pattern cannot be empty")
            if predicate.sigma <= 0.0:
                raise ValueError("QM predicate sigma must be positive")
        for parameter_class in self.parameter_classes:
            parameter_class.validate()


def _correction_sign(convention: str) -> float:
    text = str(convention or "subtract").strip().lower().replace("-", "_")
    if text in {"subtract", "subtractive", "observed_minus_delta", "b0_minus_delta"}:
        return -1.0
    if text in {"add", "additive", "observed_plus_delta", "b0_plus_delta", "msr"}:
        return 1.0
    raise ValueError(f"Unknown semiexperimental correction convention: {convention}")
