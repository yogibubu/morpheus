"""Analytic radial potential forms used by the resident ZAFF runtime.

The conversion contract preserves the well depth ``epsilon`` and minimum
position ``r_min``.  Exp-PE additionally preserves the source curvature,
whereas the double-exponential form can preserve both the quadratic and cubic
derivatives at the minimum.  Energy and distance units are deliberately
generic, provided they are used consistently for the derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RadialDerivatives:
    """Energy and first three radial derivatives at one distance."""

    energy: float
    first: float
    second: float
    third: float


@dataclass(frozen=True)
class MinimumDerivatives:
    """Observable description of a radial well at its minimum."""

    epsilon: float
    r_min: float
    second: float
    third: float | None = None

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.second) or self.second <= 0.0:
            raise ValueError("radial curvature at the minimum must be positive")
        if self.third is not None and not isfinite(self.third):
            raise ValueError("radial cubic derivative must be finite")

    @property
    def dimensionless_second(self) -> float:
        return self.second * self.r_min**2 / self.epsilon

    @property
    def dimensionless_third(self) -> float | None:
        if self.third is None:
            return None
        return self.third * self.r_min**3 / self.epsilon


@dataclass(frozen=True)
class ExpPEPotential:
    """Undamped Yang--Sun--Deng Exp-PE modified-Morse pair potential."""

    epsilon: float
    r_min: float
    alpha: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.alpha) or self.alpha <= 4.0:
            raise ValueError("Exp-PE alpha must exceed 4 for a stable minimum")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        x = r / self.r_min
        alpha = self.alpha
        exp_full = _exp(alpha * (1.0 - x))
        exp_half = _exp(0.5 * alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        polynomial_1 = 4.0 * x**3 - 4.0 * x
        polynomial_2 = 12.0 * x**2 - 4.0
        polynomial_3 = 24.0 * x
        dimensionless_0 = exp_full - polynomial * exp_half
        dimensionless_1 = -alpha * exp_full - exp_half * (polynomial_1 - 0.5 * alpha * polynomial)
        dimensionless_2 = alpha**2 * exp_full + exp_half * (
            -polynomial_2 + alpha * polynomial_1 - 0.25 * alpha**2 * polynomial
        )
        dimensionless_3 = -(alpha**3) * exp_full + exp_half * (
            -polynomial_3
            + 1.5 * alpha * polynomial_2
            - 0.75 * alpha**2 * polynomial_1
            + 0.125 * alpha**3 * polynomial
        )
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            dimensionless_0,
            dimensionless_1,
            dimensionless_2,
            dimensionless_3,
        )

    def energy(self, distance: float) -> float:
        """Evaluate the radial energy without derivative construction."""

        r = _validate_distance(distance)
        x = r / self.r_min
        exp_full = _exp(self.alpha * (1.0 - x))
        exp_half = _exp(0.5 * self.alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        return float(self.epsilon * (exp_full - polynomial * exp_half))

    @property
    def minimum(self) -> MinimumDerivatives:
        values = self.derivatives(self.r_min)
        return MinimumDerivatives(self.epsilon, self.r_min, values.second, values.third)


@dataclass(frozen=True)
class DampedExpPEPotential:
    """Runtime Exp-PE including the short-range rational damping factor.

    ``epsilon`` and ``r_min`` are fitted scale parameters; unlike the undamped
    class, callers must not interpret them directly as the realized well depth
    and minimum position.
    """

    epsilon: float
    r_min: float
    alpha: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.alpha) or self.alpha <= 4.0:
            raise ValueError("damped Exp-PE alpha must exceed 4")

    def energy(self, distance: float) -> float:
        r = _validate_distance(distance)
        x = r / self.r_min
        exp_full = _exp(self.alpha * (1.0 - x))
        exp_half = _exp(0.5 * self.alpha * (1.0 - x))
        polynomial = x**4 - 2.0 * x**2 + 3.0
        return float(self.epsilon * (exp_full - polynomial * exp_half) / (1.0 + (0.72 / x) ** 8))


@dataclass(frozen=True)
class MorsePotential:
    """Shifted standard Morse pair potential with minimum energy ``-epsilon``."""

    epsilon: float
    r_min: float
    beta: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("Morse beta must be positive")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        displacement = r / self.r_min - 1.0
        first_exp = _exp(-self.beta * displacement)
        second_exp = first_exp * first_exp
        b = self.beta
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            second_exp - 2.0 * first_exp,
            -2.0 * b * second_exp + 2.0 * b * first_exp,
            4.0 * b**2 * second_exp - 2.0 * b**2 * first_exp,
            -8.0 * b**3 * second_exp + 2.0 * b**3 * first_exp,
        )


@dataclass(frozen=True)
class InversePowerStretchPotential:
    """Generalized Kratzer/Thakkar bond potential.

    The energy is ``depth * (1 - (r0 / r)**exponent)**2``.  It is zero at
    ``r0``, approaches ``depth`` upon dissociation and has a repulsive
    inverse-power wall for a positive exponent.  Unlike the simple Kratzer
    special case (``exponent=1``), the exponent can be obtained directly from
    the quadratic and cubic QM derivatives at the reference distance.
    """

    depth: float
    r0: float
    exponent: float

    def __post_init__(self) -> None:
        _validate_well(self.depth, self.r0)
        if not isfinite(self.exponent) or self.exponent <= 0.0:
            raise ValueError("inverse-power stretching exponent must be positive")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        p = float(self.exponent)
        inverse_power = (float(self.r0) / r) ** p
        shape = 1.0 - inverse_power
        shape_1 = p * inverse_power / r
        shape_2 = -p * (p + 1.0) * inverse_power / r**2
        shape_3 = p * (p + 1.0) * (p + 2.0) * inverse_power / r**3
        depth = float(self.depth)
        return RadialDerivatives(
            depth * shape**2,
            2.0 * depth * shape * shape_1,
            2.0 * depth * (shape_1**2 + shape * shape_2),
            2.0 * depth * (3.0 * shape_1 * shape_2 + shape * shape_3),
        )


@dataclass(frozen=True)
class SPFStretchPotential:
    """Simons--Parr--Finlan expansion in ``y = 1 - r0/r``.

    ``coefficients[n-2]`` multiplies ``y**n``.  Starting the expansion at
    second order keeps the supplied reference distance stationary.  A cubic
    truncation is a local model; production use must add a stabilizing even
    term or pass the global-shape audit performed by the ARCHITECT compiler.
    """

    r0: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isfinite(float(self.r0)) or float(self.r0) <= 0.0:
            raise ValueError("SPF reference distance must be positive")
        coefficients = tuple(float(value) for value in self.coefficients)
        if not coefficients or not all(isfinite(value) for value in coefficients):
            raise ValueError("SPF coefficients must be a nonempty finite sequence")
        object.__setattr__(self, "coefficients", coefficients)

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        r0 = float(self.r0)
        y = 1.0 - r0 / r
        y_1 = r0 / r**2
        y_2 = -2.0 * r0 / r**3
        y_3 = 6.0 * r0 / r**4
        polynomial = [0.0, 0.0, 0.0, 0.0]
        for power, coefficient in enumerate(self.coefficients, start=2):
            polynomial[0] += coefficient * y**power
            polynomial[1] += coefficient * power * y ** (power - 1)
            if power >= 2:
                polynomial[2] += coefficient * power * (power - 1) * y ** (power - 2)
            if power >= 3:
                polynomial[3] += coefficient * power * (power - 1) * (power - 2) * y ** (power - 3)
        return RadialDerivatives(
            polynomial[0],
            polynomial[1] * y_1,
            polynomial[2] * y_1**2 + polynomial[1] * y_2,
            (polynomial[3] * y_1**3 + 3.0 * polynomial[2] * y_1 * y_2 + polynomial[1] * y_3),
        )


@dataclass(frozen=True)
class DoubleExponentialPotential:
    """Two-exponential well with independently matchable second and third derivatives."""

    epsilon: float
    r_min: float
    repulsive_exponent: float
    attractive_exponent: float

    def __post_init__(self) -> None:
        _validate_well(self.epsilon, self.r_min)
        if not (
            isfinite(self.repulsive_exponent)
            and isfinite(self.attractive_exponent)
            and self.repulsive_exponent > self.attractive_exponent > 0.0
        ):
            raise ValueError("double-exponential exponents must satisfy p > q > 0")

    def derivatives(self, distance: float) -> RadialDerivatives:
        r = _validate_distance(distance)
        displacement = r / self.r_min - 1.0
        p = self.repulsive_exponent
        q = self.attractive_exponent
        denominator = p - q
        repulsive = _exp(-p * displacement)
        attractive = _exp(-q * displacement)
        return _scale_derivatives(
            self.epsilon,
            self.r_min,
            (q * repulsive - p * attractive) / denominator,
            p * q * (attractive - repulsive) / denominator,
            p * q * (p * repulsive - q * attractive) / denominator,
            p * q * (q**2 * attractive - p**2 * repulsive) / denominator,
        )


def mie_minimum_derivatives(
    epsilon: float,
    r_min: float,
    repulsive_power: float,
    attractive_power: float,
) -> MinimumDerivatives:
    """Return exact minimum derivatives for a normalized Mie ``m-n`` well.

    The potential is ``epsilon * [n*x^-m - m*x^-n] / (m-n)``, where
    ``x = r/r_min``.  UFF is the ``m=12, n=6`` special case.
    """
    _validate_well(epsilon, r_min)
    m = float(repulsive_power)
    n = float(attractive_power)
    if not (isfinite(m) and isfinite(n) and m > n > 0.0):
        raise ValueError("Mie powers must satisfy m > n > 0")
    dimensionless_second = m * n
    dimensionless_third = -m * n * (m + n + 3.0)
    return MinimumDerivatives(
        float(epsilon),
        float(r_min),
        float(epsilon) * dimensionless_second / float(r_min) ** 2,
        float(epsilon) * dimensionless_third / float(r_min) ** 3,
    )


def exppe_from_minimum(target: MinimumDerivatives) -> ExpPEPotential:
    """Construct Exp-PE preserving ``epsilon``, ``r_min`` and curvature."""
    alpha = sqrt(2.0 * (target.dimensionless_second + 8.0))
    return ExpPEPotential(target.epsilon, target.r_min, alpha)


def damped_exppe_from_minimum(target: MinimumDerivatives) -> DampedExpPEPotential:
    """Fit parameters of the damped runtime Exp-PE to source minimum data.

    The returned ``r_min`` is the Exp-PE radial scale. The minimum of the
    *damped* evaluated potential is exactly the target distance within the
    numerical fit tolerance.
    """

    initial = exppe_from_minimum(target)
    reference = float(target.r_min)
    # A 1e-4 scale makes the five-point curvature residual cancellation-bound
    # on some x86_64 libm/BLAS combinations, which can stop least_squares at a
    # false platform-specific minimum.  The 1e-3 scale retains ample local
    # accuracy while producing stable residual Jacobians on x86_64 and ARM64.
    step = max(1.0e-7, 1.0e-3 * reference)

    def values(parameters: np.ndarray, distance: float) -> float:
        epsilon = np.exp(parameters[0])
        radial_scale = np.exp(parameters[1])
        alpha = 4.0 + np.exp(parameters[2])
        x = distance / radial_scale
        undamped = np.exp(alpha * (1.0 - x)) - (x**4 - 2.0 * x**2 + 3.0) * np.exp(
            0.5 * alpha * (1.0 - x)
        )
        return float(epsilon * undamped / (1.0 + (0.72 / x) ** 8))

    def residual(parameters: np.ndarray) -> np.ndarray:
        fm2 = values(parameters, reference - 2.0 * step)
        fm1 = values(parameters, reference - step)
        f0 = values(parameters, reference)
        fp1 = values(parameters, reference + step)
        fp2 = values(parameters, reference + 2.0 * step)
        first = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * step)
        second = (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * step**2)
        return np.asarray(
            (
                (f0 + target.epsilon) / target.epsilon,
                first * reference / target.epsilon,
                (second - target.second) * reference**2 / target.epsilon,
            )
        )

    fit = least_squares(
        residual,
        np.asarray(
            (
                np.log(initial.epsilon),
                np.log(initial.r_min),
                np.log(initial.alpha - 4.0),
            )
        ),
        bounds=(
            np.asarray(
                (
                    np.log(target.epsilon * 0.05),
                    np.log(target.r_min * 0.5),
                    np.log(0.05),
                )
            ),
            np.asarray(
                (
                    np.log(target.epsilon * 20.0),
                    np.log(target.r_min * 1.5),
                    np.log(100.0),
                )
            ),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=2000,
    )
    if not fit.success or float(np.max(np.abs(residual(fit.x)))) > 2.0e-5:
        raise RuntimeError("damped Exp-PE minimum-preserving fit failed")
    return DampedExpPEPotential(
        epsilon=float(np.exp(fit.x[0])),
        r_min=float(np.exp(fit.x[1])),
        alpha=float(4.0 + np.exp(fit.x[2])),
    )


def morse_from_minimum(target: MinimumDerivatives) -> MorsePotential:
    """Construct standard Morse preserving ``epsilon``, ``r_min`` and curvature."""
    beta = sqrt(0.5 * target.dimensionless_second)
    return MorsePotential(target.epsilon, target.r_min, beta)


def inverse_power_stretch_from_local_derivatives(
    r0: float,
    curvature: float,
    cubic: float,
) -> InversePowerStretchPotential:
    """Construct the positive inverse-power stretch matching ``k2`` and ``k3``.

    For ``V = D [1 - (r0/r)^p]^2``, the local derivatives satisfy
    ``k2 = 2 D p^2/r0^2`` and ``k3 = -6 D p^2(p+1)/r0^3``.  A non-positive
    inferred exponent signals that this two-parameter global form is
    incompatible with the supplied local derivatives; callers should then use
    a Morse or a higher-order SPF expansion rather than clipping the exponent.
    """

    reference = _validate_distance(r0)
    k2 = float(curvature)
    k3 = float(cubic)
    if not isfinite(k2) or k2 <= 0.0 or not isfinite(k3):
        raise ValueError("inverse-power construction requires finite k2 > 0 and finite k3")
    exponent = -1.0 - k3 * reference / (3.0 * k2)
    if exponent <= 0.0:
        raise ValueError(
            "local derivatives imply a non-positive inverse-power exponent; "
            "use Morse or a stabilized SPF expansion"
        )
    depth = k2 * reference**2 / (2.0 * exponent**2)
    return InversePowerStretchPotential(depth, reference, exponent)


def spf_stretch_from_local_derivatives(
    r0: float,
    curvature: float,
    cubic: float,
    *,
    higher_coefficients: tuple[float, ...] = (),
) -> SPFStretchPotential:
    """Build an SPF series whose quadratic and cubic derivatives are exact.

    Extra coefficients start at fourth order and therefore leave ``k2`` and
    ``k3`` unchanged.  They can be obtained from higher derivatives or chosen
    by the global stability audit.
    """

    reference = _validate_distance(r0)
    k2 = float(curvature)
    k3 = float(cubic)
    extras = tuple(float(value) for value in higher_coefficients)
    if not isfinite(k2) or k2 <= 0.0 or not isfinite(k3):
        raise ValueError("SPF construction requires finite k2 > 0 and finite k3")
    if not all(isfinite(value) for value in extras):
        raise ValueError("higher SPF coefficients must be finite")
    c2 = 0.5 * k2 * reference**2
    c3 = k2 * reference**2 + k3 * reference**3 / 6.0
    return SPFStretchPotential(reference, (c2, c3) + extras)


def morse_from_zero_crossing(
    epsilon: float,
    r_min: float,
    zero_crossing: float,
) -> MorsePotential:
    """Apply the Abraham--Stolevik minimum/zero-crossing Morse mapping."""
    _validate_well(epsilon, r_min)
    r_zero = _validate_distance(zero_crossing)
    if r_zero >= r_min:
        raise ValueError("Morse zero crossing must lie below the minimum distance")
    beta = log(2.0) / (1.0 - r_zero / r_min)
    return MorsePotential(float(epsilon), float(r_min), beta)


def double_exponential_from_minimum(target: MinimumDerivatives) -> DoubleExponentialPotential:
    """Construct a double exponential preserving derivatives through third order."""
    dimensionless_third = target.dimensionless_third
    if dimensionless_third is None:
        raise ValueError("double-exponential conversion requires the cubic derivative")
    product = target.dimensionless_second
    total = -dimensionless_third / product
    discriminant = total**2 - 4.0 * product
    if total <= 0.0 or discriminant < -1.0e-12:
        raise ValueError("target derivatives do not define two positive real exponents")
    root = sqrt(max(0.0, discriminant))
    repulsive = 0.5 * (total + root)
    attractive = 0.5 * (total - root)
    return DoubleExponentialPotential(target.epsilon, target.r_min, repulsive, attractive)


def uff_to_exppe(epsilon: float, r_min: float) -> ExpPEPotential:
    """Convert a UFF 12-6 pair while preserving its minimum Hessian."""
    return exppe_from_minimum(mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0))


def uff_to_double_exponential(epsilon: float, r_min: float) -> DoubleExponentialPotential:
    """Convert a UFF 12-6 pair through the cubic derivative at the minimum."""
    return double_exponential_from_minimum(mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0))


def uff_pair_to_exppe(atomic_number_i: int, atomic_number_j: int) -> ExpPEPotential:
    """Convert the mixed MATRIX/UFF pair for two elements to Exp-PE.

    The returned potential uses hartree and bohr, matching the analytic
    non-bonded derivative engine in :mod:`matrix_chem`.
    """
    from matrix_chem import uff_pair_parameters

    r_min, epsilon = uff_pair_parameters(atomic_number_i, atomic_number_j)
    return uff_to_exppe(epsilon, r_min)


def uff_pair_to_double_exponential(
    atomic_number_i: int,
    atomic_number_j: int,
) -> DoubleExponentialPotential:
    """Convert the mixed MATRIX/UFF pair through cubic order at its minimum."""
    from matrix_chem import uff_pair_parameters

    r_min, epsilon = uff_pair_parameters(atomic_number_i, atomic_number_j)
    return uff_to_double_exponential(epsilon, r_min)


def _scale_derivatives(
    epsilon: float,
    r_min: float,
    energy: float,
    first: float,
    second: float,
    third: float,
) -> RadialDerivatives:
    return RadialDerivatives(
        float(epsilon) * energy,
        float(epsilon) * first / float(r_min),
        float(epsilon) * second / float(r_min) ** 2,
        float(epsilon) * third / float(r_min) ** 3,
    )


def _validate_well(epsilon: float, r_min: float) -> None:
    if not isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("radial well depth must be positive")
    if not isfinite(float(r_min)) or float(r_min) <= 0.0:
        raise ValueError("radial minimum distance must be positive")


def _validate_distance(distance: float) -> float:
    value = float(distance)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("radial distance must be positive")
    return value


def _exp(value: float) -> float:
    # Kept local so every model follows the same scalar evaluation path.
    return exp(value)
