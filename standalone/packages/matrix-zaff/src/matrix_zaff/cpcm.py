"""Analytic spectral CPCM for spherical and ellipsoidal confinement."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any, Literal, Mapping

import numpy as np
from numba import njit


ZAFF_CPCM_SCHEMA = "matrix.zaff.cpcm_reaction_field.v1"


@njit(cache=True)
def _packed_modal_values(
    coordinates: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    semiaxes: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    term_counts: np.ndarray,
) -> np.ndarray:
    """Evaluate every compiled Cartesian mode without derivative allocations."""

    count = coordinates.shape[0]
    mode_count = coefficients.shape[0]
    maximum_power = 0
    for mode in range(mode_count):
        for term in range(term_counts[mode]):
            for axis in range(3):
                maximum_power = max(maximum_power, exponents[mode, term, axis])

    values = np.empty((count, mode_count), dtype=np.float64)
    powers = np.empty((3, maximum_power + 1), dtype=np.float64)
    for site in range(count):
        relative0 = coordinates[site, 0] - center[0]
        relative1 = coordinates[site, 1] - center[1]
        relative2 = coordinates[site, 2] - center[2]
        scaled0 = (
            relative0 * rotation[0, 0]
            + relative1 * rotation[1, 0]
            + relative2 * rotation[2, 0]
        ) / semiaxes[0]
        scaled1 = (
            relative0 * rotation[0, 1]
            + relative1 * rotation[1, 1]
            + relative2 * rotation[2, 1]
        ) / semiaxes[1]
        scaled2 = (
            relative0 * rotation[0, 2]
            + relative1 * rotation[1, 2]
            + relative2 * rotation[2, 2]
        ) / semiaxes[2]
        for axis in range(3):
            powers[axis, 0] = 1.0
        for degree in range(1, maximum_power + 1):
            powers[0, degree] = powers[0, degree - 1] * scaled0
            powers[1, degree] = powers[1, degree - 1] * scaled1
            powers[2, degree] = powers[2, degree - 1] * scaled2
        for mode in range(mode_count):
            value = 0.0
            for term in range(term_counts[mode]):
                value += (
                    coefficients[mode, term]
                    * powers[0, exponents[mode, term, 0]]
                    * powers[1, exponents[mode, term, 1]]
                    * powers[2, exponents[mode, term, 2]]
                )
            values[site, mode] = value
    return values


@njit(cache=True)
def _packed_modal_values_gradients(
    coordinates: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    semiaxes: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    term_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate compiled modes and analytic gradients without Hessian work."""

    count = coordinates.shape[0]
    mode_count = coefficients.shape[0]
    maximum_power = 0
    for mode in range(mode_count):
        for term in range(term_counts[mode]):
            for axis in range(3):
                maximum_power = max(maximum_power, exponents[mode, term, axis])
    values = np.empty((count, mode_count), dtype=np.float64)
    gradients = np.empty((count, mode_count, 3), dtype=np.float64)
    powers = np.empty((3, maximum_power + 1), dtype=np.float64)
    for site in range(count):
        relative0 = coordinates[site, 0] - center[0]
        relative1 = coordinates[site, 1] - center[1]
        relative2 = coordinates[site, 2] - center[2]
        scaled = np.empty(3, dtype=np.float64)
        for axis in range(3):
            scaled[axis] = (
                relative0 * rotation[0, axis]
                + relative1 * rotation[1, axis]
                + relative2 * rotation[2, axis]
            ) / semiaxes[axis]
            powers[axis, 0] = 1.0
        for degree in range(1, maximum_power + 1):
            for axis in range(3):
                powers[axis, degree] = powers[axis, degree - 1] * scaled[axis]
        for mode in range(mode_count):
            value = 0.0
            local_gradient = np.zeros(3, dtype=np.float64)
            for term in range(term_counts[mode]):
                coefficient = coefficients[mode, term]
                exponent0 = exponents[mode, term, 0]
                exponent1 = exponents[mode, term, 1]
                exponent2 = exponents[mode, term, 2]
                value += (
                    coefficient
                    * powers[0, exponent0]
                    * powers[1, exponent1]
                    * powers[2, exponent2]
                )
                if exponent0:
                    local_gradient[0] += (
                        coefficient
                        * exponent0
                        * powers[0, exponent0 - 1]
                        * powers[1, exponent1]
                        * powers[2, exponent2]
                        / semiaxes[0]
                    )
                if exponent1:
                    local_gradient[1] += (
                        coefficient
                        * exponent1
                        * powers[0, exponent0]
                        * powers[1, exponent1 - 1]
                        * powers[2, exponent2]
                        / semiaxes[1]
                    )
                if exponent2:
                    local_gradient[2] += (
                        coefficient
                        * exponent2
                        * powers[0, exponent0]
                        * powers[1, exponent1]
                        * powers[2, exponent2 - 1]
                        / semiaxes[2]
                    )
            values[site, mode] = value
            for cartesian in range(3):
                gradients[site, mode, cartesian] = (
                    local_gradient[0] * rotation[cartesian, 0]
                    + local_gradient[1] * rotation[cartesian, 1]
                    + local_gradient[2] * rotation[cartesian, 2]
                )
    return values, gradients


@dataclass(frozen=True)
class CPCMConfinement:
    """Frozen spherical or triaxial-ellipsoidal conductor boundary."""

    center_bohr: np.ndarray
    semiaxes_bohr: np.ndarray
    rotation: np.ndarray
    dielectric: float = math.inf
    maximum_degree: int = 6
    polar_order: int = 12
    azimuthal_order: int = 24

    def __post_init__(self) -> None:
        center = np.asarray(self.center_bohr, dtype=float).reshape(3)
        axes = np.asarray(self.semiaxes_bohr, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        if (
            np.any(~np.isfinite(center))
            or np.any(~np.isfinite(axes))
            or np.any(axes <= 0.0)
        ):
            raise ValueError("CPCM center and semiaxes must be finite and positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12):
            raise ValueError("CPCM ellipsoid rotation must be orthogonal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("CPCM ellipsoid rotation must be right handed")
        dielectric = float(self.dielectric)
        if not (math.isinf(dielectric) or dielectric > 1.0):
            raise ValueError("CPCM dielectric must exceed one or be infinite")
        if not 0 <= int(self.maximum_degree) <= 12:
            raise ValueError("CPCM maximum degree must lie between zero and twelve")
        order = np.argsort(-axes)
        axes = axes[order]
        rotation = rotation[:, order]
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
        object.__setattr__(self, "center_bohr", center)
        object.__setattr__(self, "semiaxes_bohr", axes)
        object.__setattr__(self, "rotation", rotation)

    @property
    def shape(self) -> str:
        relative_spread = float(np.ptp(self.semiaxes_bohr) / np.max(self.semiaxes_bohr))
        return "SPHERE" if relative_spread <= 1.0e-10 else "TRIAXIAL_ELLIPSOID"

    @property
    def dielectric_factor(self) -> float:
        if math.isinf(float(self.dielectric)):
            return 1.0
        return (float(self.dielectric) - 1.0) / float(self.dielectric)


@dataclass(frozen=True)
class CPCMResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    spectral_coefficients: np.ndarray
    boundary_residual_rms_hartree_per_e: float
    backend: str
    schema: str = ZAFF_CPCM_SCHEMA


@dataclass(frozen=True)
class CPCMSecondOrderResult:
    """CPCM energy, gradient, and full analytic Cartesian Hessian."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    backend: str
    schema: str = ZAFF_CPCM_SCHEMA


@dataclass(frozen=True)
class _PolynomialMode:
    degree: int
    order: int
    exponents: np.ndarray
    coefficients: np.ndarray
    reaction_weight: float


@dataclass(frozen=True)
class CPCMReactionField:
    """Separated CPCM operator in spherical or ellipsoidal harmonics.

    The ellipsoidal implementation uses Lamé functions of the first and
    second kinds and their normalization constants.  Once compiled, each
    interior solid harmonic is represented by its exact Cartesian polynomial,
    so energy, gradient, and Hessian-vector products contain no coordinate
    transformations and no numerical differentiation.
    """

    confinement: CPCMConfinement
    modes: tuple[_PolynomialMode, ...]
    modal_backend: str

    @cached_property
    def _packed_polynomials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a dense immutable representation for the compiled value kernel."""

        maximum_terms = max(len(mode.coefficients) for mode in self.modes)
        exponents = np.zeros((len(self.modes), maximum_terms, 3), dtype=np.int64)
        coefficients = np.zeros((len(self.modes), maximum_terms), dtype=float)
        term_counts = np.empty(len(self.modes), dtype=np.int64)
        for index, mode in enumerate(self.modes):
            count = len(mode.coefficients)
            exponents[index, :count] = mode.exponents
            coefficients[index, :count] = mode.coefficients
            term_counts[index] = count
        exponents.setflags(write=False)
        coefficients.setflags(write=False)
        term_counts.setflags(write=False)
        return exponents, coefficients, term_counts

    @classmethod
    def compile(cls, confinement: CPCMConfinement) -> "CPCMReactionField":
        if confinement.shape == "SPHERE":
            modes = _compile_spherical_modes(confinement)
            backend = "SPHERICAL_HARMONICS"
        else:
            sorted_axes = np.sort(confinement.semiaxes_bohr)[::-1]
            relative_gaps = np.diff(sorted_axes) / sorted_axes[0]
            if np.any(np.abs(relative_gaps) <= 1.0e-8):
                # The spheroidal limit is evaluated with a controlled,
                # machine-resolvable eccentricity.  This avoids the singular
                # triaxial parametrization while retaining the requested
                # a=b != c boundary; convergence is quadratic in the
                # perturbation and is checked by the regression tests.
                if abs(sorted_axes[0] - sorted_axes[1]) / sorted_axes[0] <= 1.0e-8:
                    modes = _compile_spheroidal_modes(confinement)
                    backend = "SPHEROIDAL_LAME_HARMONICS"
                else:
                    raise ValueError("unsupported axisymmetric CPCM ordering")
            else:
                modes = _compile_ellipsoidal_modes(confinement)
                backend = "ELLIPSOIDAL_LAME_HARMONICS"
        return cls(confinement=confinement, modes=modes, modal_backend=backend)

    def evaluate(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        backend: Literal["auto", "direct", "fmm"] = "auto",
        fmm_precision: float = 1.0e-10,
        fmm_minimum_sources: int = 256,
    ) -> CPCMResult:
        """Evaluate variational CPCM energy and analytic Cartesian gradient."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        values, gradients, _ = self._mode_derivatives(xyz, second=False)
        moments = values.T @ q
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        factor = self.confinement.dielectric_factor
        energy = 0.5 * factor * float(np.dot(weights, moments**2))
        gradient = factor * q[:, None] * np.einsum(
            "m,m,nmd->nd", weights, moments, gradients
        )
        reaction_coefficients = factor * weights * moments
        residual = self._boundary_residual(
            xyz,
            q,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sources=fmm_minimum_sources,
        )
        return CPCMResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            spectral_coefficients=reaction_coefficients,
            boundary_residual_rms_hartree_per_e=residual,
            backend=self.modal_backend,
        )

    def hessian_vector_product(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the exact analytic CPCM Hessian for a frozen cavity."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        direction = np.asarray(vector_bohr, dtype=float)
        if direction.size == xyz.size:
            direction = direction.reshape(xyz.shape)
        if direction.shape != xyz.shape or np.any(~np.isfinite(direction)):
            raise ValueError("CPCM Hessian-vector dimensions are inconsistent")
        values, gradients, hessians = self._mode_derivatives(xyz, second=True)
        moments = values.T @ q
        moment_dot = np.einsum("nmd,nd,n->m", gradients, direction, q)
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        first = np.einsum("m,m,nmd->nd", weights, moment_dot, gradients)
        second = np.einsum(
            "m,m,nmde,ne->nd", weights, moments, hessians, direction
        )
        factor = self.confinement.dielectric_factor
        return (factor * q[:, None] * (first + second)).reshape(-1)

    def reaction_potential(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        targets_bohr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the truncated CPCM reaction potential at interior targets."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        targets = xyz if targets_bohr is None else np.asarray(targets_bohr, dtype=float)
        self._validated_targets(targets)
        source_values, _, _ = self._mode_derivatives(xyz, second=False)
        target_values, _, _ = self._mode_derivatives(targets, second=False)
        moments = source_values.T @ q
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return self.confinement.dielectric_factor * target_values @ (weights * moments)

    def kernel_product(
        self, coordinates_bohr: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """Apply the finite-dielectric reaction kernel to site charges."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        values, _, _ = self._mode_derivatives(xyz, second=False)
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return self.confinement.dielectric_factor * values @ (weights * (values.T @ q))

    def kernel_directional_product(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Differentiate ``K(R) q`` along a Cartesian site displacement."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        direction = np.asarray(vector_bohr, dtype=float)
        if direction.size == xyz.size:
            direction = direction.reshape(xyz.shape)
        if direction.shape != xyz.shape:
            raise ValueError("CPCM kernel directional vector has the wrong shape")
        values, gradients, _ = self._mode_derivatives(xyz, second=False)
        value_dot = np.einsum("nmd,nd->nm", gradients, direction)
        moments = values.T @ q
        moment_dot = value_dot.T @ q
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return self.confinement.dielectric_factor * (
            value_dot @ (weights * moments)
            + values @ (weights * moment_dot)
        )

    def reaction_energy_gradient(
        self, coordinates_bohr: np.ndarray, charges: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return reaction energy and gradient without a boundary-residual audit."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        values, gradients, _ = self._mode_derivatives(xyz, second=False)
        moments = values.T @ q
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        factor = self.confinement.dielectric_factor
        energy = 0.5 * factor * float(np.dot(weights, moments**2))
        gradient = factor * q[:, None] * np.einsum(
            "m,m,nmd->nd", weights, moments, gradients
        )
        return energy, gradient

    def reaction_energy(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
    ) -> float:
        """Return only the variational reaction energy."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        moments = self.modal_values(xyz).T @ q
        return self.reaction_energy_from_modal_moments(moments)

    def reaction_energy_gradient_hessian(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
    ) -> CPCMSecondOrderResult:
        """Return E/G/H analytically for stationary-point characterization."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        values, gradients, hessians = self._mode_derivatives(xyz, second=True)
        moments = values.T @ q
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        factor = self.confinement.dielectric_factor
        energy = 0.5 * factor * float(np.dot(weights, moments**2))
        gradient = factor * q[:, None] * np.einsum(
            "m,m,nma->na",
            weights,
            moments,
            gradients,
        )
        outer = factor * np.einsum(
            "i,j,m,ima,jmb->iajb",
            q,
            q,
            weights,
            gradients,
            gradients,
        )
        diagonal = factor * q[:, None, None] * np.einsum(
            "m,m,imab->iab",
            weights,
            moments,
            hessians,
        )
        for site in range(len(xyz)):
            outer[site, :, site, :] += diagonal[site]
        return CPCMSecondOrderResult(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient.reshape(-1),
            hessian_hartree_per_bohr2=outer.reshape(3 * len(xyz), 3 * len(xyz)),
            backend=self.modal_backend,
        )

    def modal_values(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        """Return compiled interior-mode values for incremental algorithms.

        This public view lets rigid-body Monte Carlo update the CPCM moments
        for one moved molecule in O(M) work without reevaluating every site.
        """

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        self._validated_targets(coordinates)
        exponents, coefficients, term_counts = self._packed_polynomials
        return _packed_modal_values(
            coordinates,
            self.confinement.center_bohr,
            self.confinement.rotation,
            self.confinement.semiaxes_bohr,
            exponents,
            coefficients,
            term_counts,
        )

    def modal_moments(
        self, coordinates_bohr: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """Return charge-weighted modal moments for a complete configuration."""

        xyz, q = self._validated_system(coordinates_bohr, charges)
        return self.modal_values(xyz).T @ q

    def reaction_energy_from_modal_moments(
        self, moments: np.ndarray
    ) -> float:
        """Evaluate CPCM energy from previously accumulated modal moments."""

        values = np.asarray(moments, dtype=float).reshape(-1)
        if values.shape != (len(self.modes),) or np.any(~np.isfinite(values)):
            raise ValueError("CPCM modal moments have the wrong dimensions")
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return (
            0.5
            * self.confinement.dielectric_factor
            * float(np.dot(weights, values**2))
        )

    def reaction_gradient_from_modal_moments(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        moments: np.ndarray,
    ) -> np.ndarray:
        """Return analytic target gradients for externally accumulated moments."""

        xyz = np.asarray(coordinates_bohr, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        accumulated = np.asarray(moments, dtype=float).reshape(-1)
        self._validated_targets(xyz)
        if (
            len(q) != len(xyz)
            or np.any(~np.isfinite(q))
            or accumulated.shape != (len(self.modes),)
            or np.any(~np.isfinite(accumulated))
        ):
            raise ValueError("CPCM modal-gradient dimensions are inconsistent")
        _values, gradients, _ = self._mode_derivatives(xyz, second=False)
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return (
            self.confinement.dielectric_factor
            * q[:, None]
            * np.einsum("m,m,nmd->nd", weights, accumulated, gradients)
        )

    def reaction_batch_energies(
        self, coordinates_bohr: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """Vectorized fixed-charge CPCM energies for a geometry population."""

        geometries = np.asarray(coordinates_bohr, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        if (
            geometries.ndim != 3
            or geometries.shape[1:] != (len(q), 3)
            or np.any(~np.isfinite(geometries))
        ):
            raise ValueError("CPCM batch must have shape (ngeometry, nsite, 3)")
        flattened = geometries.reshape(-1, 3)
        self._validated_targets(flattened)
        values, _, _ = self._mode_derivatives(flattened, second=False)
        values = values.reshape(len(geometries), len(q), len(self.modes))
        moments = np.einsum("bnm,n->bm", values, q)
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return (
            0.5
            * self.confinement.dielectric_factor
            * np.einsum("m,bm,bm->b", weights, moments, moments)
        )

    def charge_flow_curvature(
        self,
        coordinates_bohr: np.ndarray,
        left_site: int,
        right_site: int,
        scale: float,
    ) -> float:
        """Return ``b.T K b`` for one two-site charge-flow column."""

        targets = np.asarray(coordinates_bohr, dtype=float)
        self._validated_targets(targets)
        if not (
            0 <= int(left_site) < len(targets)
            and 0 <= int(right_site) < len(targets)
            and int(left_site) != int(right_site)
        ):
            raise ValueError("CPCM charge-flow endpoints are inconsistent")
        values, _, _ = self._mode_derivatives(targets, second=False)
        difference = values[int(left_site)] - values[int(right_site)]
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        return (
            self.confinement.dielectric_factor
            * float(scale) ** 2
            * float(np.dot(weights, difference**2))
        )

    def _mode_derivatives(
        self, coordinates_bohr: np.ndarray, *, second: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        if not second:
            exponents, coefficients, term_counts = self._packed_polynomials
            values, gradients = _packed_modal_values_gradients(
                coordinates,
                self.confinement.center_bohr,
                self.confinement.rotation,
                self.confinement.semiaxes_bohr,
                exponents,
                coefficients,
                term_counts,
            )
            return values, gradients, np.empty((0, 0, 0, 0), dtype=float)
        local = (
            coordinates - self.confinement.center_bohr
        ) @ self.confinement.rotation
        scaled = local / self.confinement.semiaxes_bohr
        count = len(scaled)
        values = np.empty((count, len(self.modes)), dtype=float)
        gradients = np.empty((count, len(self.modes), 3), dtype=float)
        hessians = np.empty((count, len(self.modes), 3, 3), dtype=float)
        inverse_axes = 1.0 / self.confinement.semiaxes_bohr
        for index, mode in enumerate(self.modes):
            value, grad_scaled, hess_scaled = _evaluate_polynomial(
                scaled, mode.exponents, mode.coefficients, second=second
            )
            values[:, index] = value
            grad_local = grad_scaled * inverse_axes
            gradients[:, index, :] = grad_local @ self.confinement.rotation.T
            if second:
                scale = inverse_axes[:, None] * inverse_axes[None, :]
                local_hessian = hess_scaled * scale[None, :, :]
                hessians[:, index, :, :] = np.einsum(
                    "ab,nbc,dc->nad",
                    self.confinement.rotation,
                    local_hessian,
                    self.confinement.rotation,
                )
            else:
                hessians[:, index, :, :] = 0.0
        return values, gradients, hessians

    def _validated_system(
        self, coordinates_bohr: np.ndarray, charges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        q = np.asarray(charges, dtype=float).reshape(-1)
        if (
            xyz.shape != (len(q), 3)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(q))
        ):
            raise ValueError("CPCM coordinates and charges are inconsistent")
        self._validated_targets(xyz)
        return xyz, q

    def _validated_targets(self, targets_bohr: np.ndarray) -> None:
        targets = np.asarray(targets_bohr, dtype=float)
        if targets.ndim != 2 or targets.shape[1:] != (3,):
            raise ValueError("CPCM targets must have shape (n, 3)")
        local = (targets - self.confinement.center_bohr) @ self.confinement.rotation
        scaled = local / self.confinement.semiaxes_bohr
        if np.any(np.sum(scaled**2, axis=1) >= 1.0 - 1.0e-10):
            raise ValueError("every CPCM charge site must lie strictly inside confinement")

    def _boundary_residual(
        self,
        coordinates_bohr: np.ndarray,
        charges: np.ndarray,
        *,
        backend: Literal["auto", "direct", "fmm"],
        fmm_precision: float,
        fmm_minimum_sources: int,
    ) -> float:
        from .nonbonded import laplace_potential_gradient_targets

        unit = _fibonacci_sphere(max(128, 4 * len(self.modes)))
        local = unit * self.confinement.semiaxes_bohr
        boundary = self.confinement.center_bohr + local @ self.confinement.rotation.T
        solute = laplace_potential_gradient_targets(
            coordinates_bohr,
            charges,
            boundary,
            backend=backend,
            fmm_precision=fmm_precision,
            fmm_minimum_sources=fmm_minimum_sources,
        ).potential
        source_values, _, _ = self._mode_derivatives(coordinates_bohr, second=False)
        boundary_values, _, _ = self._mode_derivatives(boundary, second=False)
        moments = source_values.T @ charges
        weights = np.asarray([mode.reaction_weight for mode in self.modes])
        induced = boundary_values @ (weights * moments)
        return float(np.sqrt(np.mean((solute + induced) ** 2)))


def cpcm_confinement_to_record(
    confinement: CPCMConfinement,
) -> dict[str, Any]:
    """Serialize a frozen CPCM cavity for a ZAFF model."""

    return {
        "schema": ZAFF_CPCM_SCHEMA,
        "boundary_contract": "FROZEN_CONFINEMENT",
        "center_bohr": confinement.center_bohr.tolist(),
        "semiaxes_bohr": confinement.semiaxes_bohr.tolist(),
        "rotation": confinement.rotation.tolist(),
        "dielectric": (
            "INFINITY"
            if math.isinf(float(confinement.dielectric))
            else float(confinement.dielectric)
        ),
        "maximum_degree": int(confinement.maximum_degree),
        "shape": confinement.shape,
        "basis": (
            "REAL_SPHERICAL_HARMONICS"
            if confinement.shape == "SPHERE"
            else "ELLIPSOIDAL_LAME_HARMONICS"
        ),
        "derivative_contract": "ANALYTIC_E_G_HVP",
    }


def cpcm_confinement_from_record(record: Mapping[str, Any]) -> CPCMConfinement:
    """Deserialize and validate a frozen ZAFF CPCM cavity."""

    if str(record.get("schema", "")) != ZAFF_CPCM_SCHEMA:
        raise ValueError("unsupported ZAFF CPCM record")
    dielectric_value = record.get("dielectric", "INFINITY")
    dielectric = (
        math.inf
        if str(dielectric_value).upper() == "INFINITY"
        else float(dielectric_value)
    )
    return CPCMConfinement(
        center_bohr=np.asarray(record["center_bohr"], dtype=float),
        semiaxes_bohr=np.asarray(record["semiaxes_bohr"], dtype=float),
        rotation=np.asarray(record["rotation"], dtype=float),
        dielectric=dielectric,
        maximum_degree=int(record.get("maximum_degree", 6)),
    )


def attach_cpcm_reaction_field(
    model: Mapping[str, Any], confinement: CPCMConfinement
) -> dict[str, Any]:
    """Return a ZAFF seed-model record carrying a validated CPCM cavity."""

    if model.get("interfacial_pcm_reaction_field") is not None:
        raise ValueError(
            "homogeneous CPCM and interfacial PCM are alternative reaction fields"
        )
    payload = dict(model)
    payload["cpcm_reaction_field"] = cpcm_confinement_to_record(confinement)
    return payload


def _compile_spherical_modes(
    confinement: CPCMConfinement,
) -> tuple[_PolynomialMode, ...]:
    radius = float(confinement.semiaxes_bohr[0])
    rng = np.random.default_rng(871233)
    modes: list[_PolynomialMode] = []
    for degree in range(confinement.maximum_degree + 1):
        exponents = _monomial_exponents(degree, homogeneous=True)
        samples = _sample_unit_ball(rng, max(48, 5 * len(exponents)))
        validation = _sample_unit_ball(rng, max(24, 2 * len(exponents)))
        for order in range(-degree, degree + 1):
            fit_values = _solid_spherical_harmonic(samples, degree, order, radius)
            coefficients = _fit_polynomial(samples, exponents, fit_values)
            _validate_polynomial_fit(
                validation,
                exponents,
                coefficients,
                _solid_spherical_harmonic(validation, degree, order, radius),
            )
            weight = -4.0 * math.pi / (
                (2 * degree + 1) * radius ** (2 * degree + 1)
            )
            modes.append(
                _PolynomialMode(degree, order, exponents, coefficients, weight)
            )
    return tuple(modes)


def _compile_ellipsoidal_modes(
    confinement: CPCMConfinement,
) -> tuple[_PolynomialMode, ...]:
    from scipy.special import ellip_harm, ellip_harm_2, ellip_normal

    axes = np.asarray(confinement.semiaxes_bohr, dtype=float)
    a, b, c = (float(value) for value in axes)
    h2 = a * a - b * b
    k2 = a * a - c * c
    rng = np.random.default_rng(991827)
    modes: list[_PolynomialMode] = []
    for degree in range(confinement.maximum_degree + 1):
        for order in range(1, 2 * degree + 2):
            family = _lame_family(degree, order)
            exponents = _ellipsoidal_monomial_exponents(degree, family)
            samples = _sample_unit_ball(rng, max(64, 8 * len(exponents)))
            validation = _sample_unit_ball(rng, max(32, 4 * len(exponents)))
            physical = samples * axes
            validation_physical = validation * axes
            fit_values = _solid_ellipsoidal_harmonic(
                physical, h2, k2, degree, order, family
            )
            coefficients = _fit_polynomial(samples, exponents, fit_values)
            _validate_polynomial_fit(
                validation,
                exponents,
                coefficients,
                _solid_ellipsoidal_harmonic(
                    validation_physical, h2, k2, degree, order, family
                ),
                relative_tolerance=2.0e-7,
            )
            first = float(ellip_harm(h2, k2, degree, order, a))
            second = float(ellip_harm_2(h2, k2, degree, order, a))
            normalization = float(ellip_normal(h2, k2, degree, order))
            if (
                not np.isfinite(first)
                or not np.isfinite(second)
                or not np.isfinite(normalization)
                or abs(first) <= np.finfo(float).tiny
                or normalization <= 0.0
            ):
                raise ValueError(
                    f"unstable Lamé normalization for degree {degree}, order {order}"
                )
            weight = (
                -4.0
                * math.pi
                / (2 * degree + 1)
                * second
                / (normalization * first)
            )
            modes.append(
                _PolynomialMode(degree, order, exponents, coefficients, weight)
            )
    return tuple(modes)


def _compile_spheroidal_modes(
    confinement: CPCMConfinement,
) -> tuple[_PolynomialMode, ...]:
    """Compile the prolate/oblate spheroidal limit of the Lamé kernel.

    SciPy's triaxial Lamé functions are singular when the two equatorial
    semiaxes coincide.  Evaluating them at a fixed relative eccentricity of
    1e-7 gives the analytic spheroidal limit to the precision required by the
    CPCM spectral truncation, while preserving the physical semiaxes in the
    returned confinement and in all coordinate/gradient kernels.
    """

    axes = np.asarray(confinement.semiaxes_bohr, dtype=float).copy()
    axes[1] = axes[0] * (1.0 - 1.0e-7)
    regularized = CPCMConfinement(
        center_bohr=confinement.center_bohr,
        semiaxes_bohr=axes,
        rotation=confinement.rotation,
        dielectric=confinement.dielectric,
        maximum_degree=confinement.maximum_degree,
        polar_order=confinement.polar_order,
        azimuthal_order=confinement.azimuthal_order,
    )
    return _compile_ellipsoidal_modes(regularized)


def _solid_spherical_harmonic(
    scaled_coordinates: np.ndarray,
    degree: int,
    order: int,
    radius: float,
) -> np.ndarray:
    from scipy.special import sph_harm_y

    coordinates = np.asarray(scaled_coordinates, dtype=float) * float(radius)
    radial = np.linalg.norm(coordinates, axis=1)
    safe = np.where(radial > 0.0, radial, 1.0)
    theta = np.arccos(np.clip(coordinates[:, 2] / safe, -1.0, 1.0))
    phi = np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2.0 * np.pi)
    if order < 0:
        harmonic = (
            math.sqrt(2.0)
            * (-1) ** order
            * sph_harm_y(degree, -order, theta, phi).imag
        )
    elif order == 0:
        harmonic = sph_harm_y(degree, 0, theta, phi).real
    else:
        harmonic = (
            math.sqrt(2.0)
            * (-1) ** order
            * sph_harm_y(degree, order, theta, phi).real
        )
    result = radial**degree * np.asarray(harmonic, dtype=float)
    if degree > 0:
        result = np.where(radial > 0.0, result, 0.0)
    return result


def _solid_ellipsoidal_harmonic(
    local_coordinates: np.ndarray,
    h2: float,
    k2: float,
    degree: int,
    order: int,
    family: str,
) -> np.ndarray:
    from scipy.special import ellip_harm

    coordinates = np.asarray(local_coordinates, dtype=float)
    ellipsoidal = _ellipsoidal_coordinate_magnitudes(coordinates, h2, k2)
    value = np.ones(len(coordinates), dtype=float)
    for column in range(3):
        value *= np.asarray(
            ellip_harm(h2, k2, degree, order, ellipsoidal[:, column]),
            dtype=float,
        )
    x_negative = coordinates[:, 0] < 0.0
    y_negative = coordinates[:, 1] < 0.0
    z_negative = coordinates[:, 2] < 0.0
    odd = degree % 2 == 1
    if family == "K":
        sign = np.where(x_negative, -1.0, 1.0) if odd else 1.0
    elif family == "L":
        sign = np.where(y_negative, -1.0, 1.0)
        if not odd:
            sign = sign * np.where(x_negative, -1.0, 1.0)
    elif family == "M":
        sign = np.where(z_negative, -1.0, 1.0)
        if not odd:
            sign = sign * np.where(x_negative, -1.0, 1.0)
    else:
        sign = np.where(y_negative ^ z_negative, -1.0, 1.0)
        if odd:
            sign = sign * np.where(x_negative, -1.0, 1.0)
    return value * sign


def _ellipsoidal_coordinate_magnitudes(
    local_coordinates: np.ndarray, h2: float, k2: float
) -> np.ndarray:
    result = np.empty((len(local_coordinates), 3), dtype=float)
    for index, coordinate in enumerate(np.asarray(local_coordinates, dtype=float)):
        x2, y2, z2 = coordinate**2
        coefficients = np.asarray(
            [
                1.0,
                -(h2 + k2 + x2 + y2 + z2),
                h2 * k2 + x2 * (h2 + k2) + y2 * k2 + z2 * h2,
                -x2 * h2 * k2,
            ]
        )
        roots = np.roots(coefficients)
        if np.max(np.abs(roots.imag)) > 1.0e-7 * max(1.0, k2):
            raise ValueError("failed to obtain real ellipsoidal coordinates")
        squared = np.sort(np.maximum(roots.real, 0.0))
        result[index] = np.sqrt(squared[::-1])
    return result


def _lame_family(degree: int, order: int) -> str:
    r = degree // 2
    if order <= r + 1:
        return "K"
    if order <= degree + 1:
        return "L"
    if order <= 2 * degree - r + 1:
        return "M"
    return "N"


def _ellipsoidal_monomial_exponents(degree: int, family: str) -> np.ndarray:
    odd = degree % 2
    parity = {
        "K": (odd, 0, 0),
        "L": (1 - odd, 1, 0),
        "M": (1 - odd, 0, 1),
        "N": (odd, 1, 1),
    }[family]
    exponents = [
        exponent
        for exponent in _monomial_exponents(degree, homogeneous=False)
        if all(exponent[axis] % 2 == parity[axis] for axis in range(3))
        and (sum(exponent) - degree) % 2 == 0
    ]
    if not exponents:
        raise ValueError("empty Cartesian polynomial basis for Lamé mode")
    return np.asarray(exponents, dtype=int)


def _monomial_exponents(degree: int, *, homogeneous: bool) -> np.ndarray:
    exponents = []
    minimum = degree if homogeneous else 0
    for total in range(minimum, degree + 1):
        if not homogeneous and (total - degree) % 2:
            continue
        for x_power in range(total + 1):
            for y_power in range(total - x_power + 1):
                exponents.append((x_power, y_power, total - x_power - y_power))
    return np.asarray(exponents, dtype=int)


def _sample_unit_ball(
    rng: np.random.Generator, count: int
) -> np.ndarray:
    direction = rng.normal(size=(int(count), 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = rng.uniform(0.15**3, 0.82**3, size=int(count)) ** (1.0 / 3.0)
    return direction * radius[:, None]


def _fit_polynomial(
    coordinates: np.ndarray, exponents: np.ndarray, values: np.ndarray
) -> np.ndarray:
    design = _polynomial_design(coordinates, exponents)
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=1.0e-13)
    if rank != len(exponents):
        raise ValueError("rank-deficient Cartesian representation of spectral mode")
    return coefficients


def _validate_polynomial_fit(
    coordinates: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    reference: np.ndarray,
    *,
    relative_tolerance: float = 2.0e-10,
) -> None:
    predicted = _polynomial_design(coordinates, exponents) @ coefficients
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    error = float(np.max(np.abs(predicted - reference)) / scale)
    if not np.isfinite(error) or error > relative_tolerance:
        raise ValueError(
            f"Cartesian spectral polynomial failed validation (relative error {error:.3e})"
        )


def _polynomial_design(
    coordinates: np.ndarray, exponents: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=float)
    return np.prod(
        xyz[:, None, :] ** np.asarray(exponents, dtype=int)[None, :, :],
        axis=2,
    )


def _evaluate_polynomial(
    coordinates: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    *,
    second: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(coordinates, dtype=float)
    powers = np.asarray(exponents, dtype=int)
    values = _polynomial_design(xyz, powers) @ coefficients
    gradient = np.zeros((len(xyz), 3), dtype=float)
    hessian = np.zeros((len(xyz), 3, 3), dtype=float)
    for term, coefficient in zip(powers, coefficients, strict=True):
        for axis in range(3):
            if term[axis] == 0:
                continue
            reduced = term.copy()
            reduced[axis] -= 1
            gradient[:, axis] += (
                coefficient
                * term[axis]
                * np.prod(xyz**reduced[None, :], axis=1)
            )
            if not second:
                continue
            for other in range(3):
                prefactor = term[axis] * (term[other] - (axis == other))
                if prefactor == 0:
                    continue
                twice_reduced = reduced.copy()
                twice_reduced[other] -= 1
                hessian[:, axis, other] += (
                    coefficient
                    * prefactor
                    * np.prod(xyz**twice_reduced[None, :], axis=1)
                )
    return values, gradient, hessian


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / int(count)
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


__all__ = [
    "ZAFF_CPCM_SCHEMA",
    "CPCMConfinement",
    "CPCMReactionField",
    "CPCMResult",
    "CPCMSecondOrderResult",
    "attach_cpcm_reaction_field",
    "cpcm_confinement_from_record",
    "cpcm_confinement_to_record",
]
