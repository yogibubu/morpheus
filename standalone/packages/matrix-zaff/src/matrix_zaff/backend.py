"""Resident common-backend adapter for compiled ZAFF artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from matrix_engines import (
    DerivativeOrder,
    PotentialCapabilities,
    PotentialEvaluation,
    PotentialSystem,
)

from .artifact import ZaffArtifact, load_zaff_artifact
from .seed_runtime import (
    ZaffSeedDynamicsRuntime,
    evaluate_zaff_seed_energy,
    evaluate_zaff_seed_model,
)
from .sonic_runtime import ZaffSonicRuntime


BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class ZaffBackend:
    """Prepare an in-process ZAFF session from an immutable artifact."""

    name: str = "zaff"

    def prepare(
        self,
        system: PotentialSystem,
        *,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> "ZaffHarmonicSession | ZaffSeedSession | ZaffSonicSession":
        if model is None or not str(model).strip():
            raise ValueError("ZAFF backend requires a compiled artifact path")
        settings = dict(options or {})
        allowed = {"zoom_level", "xyzin", "b_derivative_workers"}
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(f"unsupported ZAFF backend options: {', '.join(unknown)}")
        artifact = load_zaff_artifact(model)
        if artifact.atoms != system.atoms:
            raise ValueError("potential system atoms differ from the ZAFF artifact")
        seed_model = artifact.metadata.get("diagonal_seed_model")
        if artifact.anharmonic_model_present and settings.get("zoom_level") is not None:
            raise ValueError("compiled SONIC artifacts do not accept a harmonic zoom level")
        expected_charge = int(artifact.metadata.get("charge", system.charge))
        expected_multiplicity = int(
            artifact.metadata.get("multiplicity", system.multiplicity)
        )
        if expected_charge != system.charge or expected_multiplicity != system.multiplicity:
            raise ValueError("potential system state differs from the ZAFF artifact")
        if artifact.anharmonic_model_present:
            if settings.get("xyzin") is None:
                raise ValueError("compiled SONIC ZAFF artifacts require the frozen XYZIN")
            return ZaffSonicSession(
                system=system,
                artifact=artifact,
                xyzin=str(settings["xyzin"]),
                b_derivative_workers=int(settings.get("b_derivative_workers", 0)),
            )
        if seed_model is not None:
            return ZaffSeedSession(
                system=system,
                artifact=artifact,
                model=dict(seed_model),
                zoom_level=(
                    None
                    if settings.get("zoom_level") is None
                    else str(settings["zoom_level"])
                ),
            )
        return ZaffHarmonicSession(
            system=system,
            artifact=artifact,
            zoom_level=(
                None
                if settings.get("zoom_level") is None
                else str(settings["zoom_level"])
            ),
        )


@dataclass(frozen=True)
class ZaffHarmonicSession:
    """Prepared Cartesian harmonic ZAFF runtime with immutable resident arrays."""

    system: PotentialSystem
    artifact: ZaffArtifact
    zoom_level: str | None = None
    backend_name: str = "zaff"

    def __post_init__(self) -> None:
        self.artifact.hessian(self.zoom_level)

    @property
    def model_identifier(self) -> str:
        return self.artifact.sha256 or "in-memory-zaff-artifact"

    @property
    def capabilities(self) -> PotentialCapabilities:
        return PotentialCapabilities(
            maximum_derivative_order=DerivativeOrder.HESSIAN,
            batch=True,
            resident=True,
            persistent_neighbor_list=False,
            persistent_fmm=False,
            polarization=False,
            reaction_field=False,
            thread_safe=True,
            devices=("cpu",),
            metadata={
                "runtime": "compiled-cartesian-harmonic",
                "zoom_level": self._selected_zoom,
            },
        )

    @property
    def _selected_zoom(self) -> str:
        return self.artifact.active_zoom_level if self.zoom_level is None else self.zoom_level

    def energy(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        displacement = self._displacement(coordinates_angstrom)
        hessian = self.artifact.hessian(self.zoom_level)
        harmonic_gradient = hessian @ displacement
        energy = (
            self.artifact.energy_reference_hartree
            + float(self.artifact.gradient_reference_hartree_per_bohr @ displacement)
            + 0.5 * float(displacement @ harmonic_gradient)
        )
        return self._evaluation(energy, DerivativeOrder.ENERGY)

    def energy_gradient(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        displacement = self._displacement(coordinates_angstrom)
        hessian = self.artifact.hessian(self.zoom_level)
        harmonic_gradient = hessian @ displacement
        gradient = self.artifact.gradient_reference_hartree_per_bohr + harmonic_gradient
        energy = (
            self.artifact.energy_reference_hartree
            + float(self.artifact.gradient_reference_hartree_per_bohr @ displacement)
            + 0.5 * float(displacement @ harmonic_gradient)
        )
        return self._evaluation(
            energy,
            DerivativeOrder.GRADIENT,
            gradient=gradient,
        )

    def energy_gradient_hessian(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> PotentialEvaluation:
        displacement = self._displacement(coordinates_angstrom)
        hessian = self.artifact.hessian(self.zoom_level)
        harmonic_gradient = hessian @ displacement
        gradient = self.artifact.gradient_reference_hartree_per_bohr + harmonic_gradient
        energy = (
            self.artifact.energy_reference_hartree
            + float(self.artifact.gradient_reference_hartree_per_bohr @ displacement)
            + 0.5 * float(displacement @ harmonic_gradient)
        )
        return self._evaluation(
            energy,
            DerivativeOrder.HESSIAN,
            gradient=gradient,
            hessian=hessian,
        )

    def evaluate_batch(
        self,
        geometries_angstrom: Sequence[np.ndarray],
        *,
        derivative_order: DerivativeOrder = DerivativeOrder.ENERGY,
    ) -> tuple[PotentialEvaluation, ...]:
        order = DerivativeOrder(derivative_order)
        geometries = _geometry_batch(geometries_angstrom, self.system.natoms)
        if len(geometries) == 0:
            return ()
        displacement = (
            geometries - self.artifact.reference_coordinates_angstrom[None, :, :]
        ).reshape(len(geometries), -1) / BOHR_TO_ANGSTROM
        hessian = self.artifact.hessian(self.zoom_level)
        harmonic_gradients = displacement @ hessian.T
        energies = (
            self.artifact.energy_reference_hartree
            + displacement @ self.artifact.gradient_reference_hartree_per_bohr
            + 0.5 * np.einsum(
                "ij,ij->i",
                displacement,
                harmonic_gradients,
            )
        )
        gradients = (
            None
            if order is DerivativeOrder.ENERGY
            else harmonic_gradients
            + self.artifact.gradient_reference_hartree_per_bohr[None, :]
        )
        return tuple(
            self._evaluation(
                float(energy),
                order,
                gradient=None if gradients is None else gradients[index],
                hessian=hessian if order is DerivativeOrder.HESSIAN else None,
                batch=True,
            )
            for index, energy in enumerate(energies)
        )

    def _displacement(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (self.system.natoms, 3):
            raise ValueError("ZAFF geometry must have shape (natoms, 3)")
        if np.any(~np.isfinite(coordinates)):
            raise ValueError("ZAFF geometry contains non-finite values")
        return (
            coordinates - self.artifact.reference_coordinates_angstrom
        ).reshape(-1) / BOHR_TO_ANGSTROM

    def _evaluation(
        self,
        energy: float,
        order: DerivativeOrder,
        *,
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        batch: bool = False,
    ) -> PotentialEvaluation:
        return PotentialEvaluation(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=hessian,
            derivative_order=order,
            backend=self.backend_name,
            model=self.model_identifier,
            execution={
                "backend": "numpy",
                "device": "cpu",
                "precision": "float64",
                "resident": True,
                "artifact_loaded_once": True,
                "batch": batch,
                "zoom_level": self._selected_zoom,
            },
        ).validate_for_system(self.system)


class ZaffSeedSession:
    """Prepared transferable ZAFF session with persistent acceleration state."""

    backend_name = "zaff"

    def __init__(
        self,
        *,
        system: PotentialSystem,
        artifact: ZaffArtifact,
        model: Mapping[str, Any],
        zoom_level: str | None = None,
    ) -> None:
        self.system = system
        self.artifact = artifact
        self.model = dict(model)
        self.zoom_level = zoom_level
        self.runtime = ZaffSeedDynamicsRuntime(
            self.model,
            zoom_level=zoom_level,
        )

    @property
    def model_identifier(self) -> str:
        return self.artifact.sha256 or "in-memory-zaff-seed"

    @property
    def capabilities(self) -> PotentialCapabilities:
        return PotentialCapabilities(
            maximum_derivative_order=DerivativeOrder.HESSIAN,
            batch=True,
            resident=True,
            persistent_neighbor_list=True,
            persistent_fmm=self.runtime.electrostatics.persistent_fmm,
            polarization=self.zoom_level is None,
            reaction_field=(
                self.model.get("cpcm_reaction_field") is not None
                or self.model.get("interfacial_pcm_reaction_field") is not None
            ),
            thread_safe=False,
            devices=("cpu",),
            metadata={"runtime": "transferable-diagonal-seed"},
        )

    def energy(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        energy, execution = evaluate_zaff_seed_energy(
            self.model,
            coordinates_angstrom,
            zoom_level=self.zoom_level,
            neighbor_list=self.runtime.neighbor_list,
            electrostatic_operator=self.runtime.electrostatics,
        )
        return self._evaluation(
            energy,
            DerivativeOrder.ENERGY,
            execution=execution,
        )

    def energy_gradient(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        result = self.runtime.evaluate(coordinates_angstrom)
        return self._evaluation(
            result.energy_hartree,
            DerivativeOrder.GRADIENT,
            gradient=result.gradient_hartree_per_bohr,
            execution=result.execution,
        )

    def energy_gradient_hessian(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> PotentialEvaluation:
        result = evaluate_zaff_seed_model(
            self.model,
            coordinates_angstrom,
            hessian=True,
            zoom_level=self.zoom_level,
        )
        return self._evaluation(
            result.energy_hartree,
            DerivativeOrder.HESSIAN,
            gradient=result.gradient_hartree_per_bohr,
            hessian=result.hessian_hartree_per_bohr2,
            execution=result.execution,
        )

    def evaluate_batch(
        self,
        geometries_angstrom: Sequence[np.ndarray],
        *,
        derivative_order: DerivativeOrder = DerivativeOrder.ENERGY,
    ) -> tuple[PotentialEvaluation, ...]:
        order = DerivativeOrder(derivative_order)
        geometries = _geometry_batch(geometries_angstrom, self.system.natoms)
        evaluator = (
            self.energy
            if order is DerivativeOrder.ENERGY
            else self.energy_gradient
            if order is DerivativeOrder.GRADIENT
            else self.energy_gradient_hessian
        )
        return tuple(evaluator(geometry) for geometry in geometries)

    def _evaluation(
        self,
        energy: float,
        order: DerivativeOrder,
        *,
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        execution: Mapping[str, Any],
    ) -> PotentialEvaluation:
        return PotentialEvaluation(
            energy_hartree=float(energy),
            gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=hessian,
            derivative_order=order,
            backend=self.backend_name,
            model=self.model_identifier,
            execution={
                **dict(execution),
                "resident": True,
                "artifact_loaded_once": True,
                "derivative_order": order.name,
            },
        ).validate_for_system(self.system)


class ZaffSonicSession:
    """Prepared physical/Taylor SONIC session backed by the SMITH chart engine."""

    backend_name = "zaff"

    def __init__(
        self,
        *,
        system: PotentialSystem,
        artifact: ZaffArtifact,
        xyzin: str,
        b_derivative_workers: int = 0,
    ) -> None:
        self.system = system
        self.artifact = artifact
        self.xyzin = xyzin
        self.b_derivative_workers = int(b_derivative_workers)
        if self.b_derivative_workers < 0:
            raise ValueError("SONIC B-derivative worker count cannot be negative")
        self.runtime = ZaffSonicRuntime(artifact, xyzin)

    @property
    def model_identifier(self) -> str:
        return self.artifact.sha256 or "in-memory-zaff-sonic"

    @property
    def capabilities(self) -> PotentialCapabilities:
        return PotentialCapabilities(
            maximum_derivative_order=DerivativeOrder.HESSIAN,
            batch=True,
            resident=True,
            persistent_neighbor_list=False,
            persistent_fmm=False,
            polarization=False,
            reaction_field=False,
            thread_safe=True,
            devices=("cpu",),
            metadata={"runtime": "compiled-local-sonic"},
        )

    def energy(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        return self._evaluation(
            self.runtime.energy(coordinates_angstrom),
            DerivativeOrder.ENERGY,
            execution={"coordinate_transform_backend": "SMITH_SONIC_VALUES_ONLY"},
        )

    def energy_gradient(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation:
        result = self.runtime.evaluate(coordinates_angstrom)
        return self._evaluation(
            result.energy_hartree,
            DerivativeOrder.GRADIENT,
            gradient=result.gradient_hartree_per_bohr,
            execution=result.execution,
        )

    def energy_gradient_hessian(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> PotentialEvaluation:
        result = self.runtime.evaluate(
            coordinates_angstrom,
            include_hessian=True,
            b_derivative_workers=self.b_derivative_workers,
        )
        return self._evaluation(
            result.energy_hartree,
            DerivativeOrder.HESSIAN,
            gradient=result.gradient_hartree_per_bohr,
            hessian=result.hessian_hartree_per_bohr2,
            execution=result.execution,
        )

    def evaluate_batch(
        self,
        geometries_angstrom: Sequence[np.ndarray],
        *,
        derivative_order: DerivativeOrder = DerivativeOrder.ENERGY,
    ) -> tuple[PotentialEvaluation, ...]:
        order = DerivativeOrder(derivative_order)
        geometries = _geometry_batch(geometries_angstrom, self.system.natoms)
        evaluator = (
            self.energy
            if order is DerivativeOrder.ENERGY
            else self.energy_gradient
            if order is DerivativeOrder.GRADIENT
            else self.energy_gradient_hessian
        )
        return tuple(evaluator(geometry) for geometry in geometries)

    def _evaluation(
        self,
        energy: float,
        order: DerivativeOrder,
        *,
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        execution: Mapping[str, Any],
    ) -> PotentialEvaluation:
        return PotentialEvaluation(
            energy_hartree=float(energy),
            gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=hessian,
            derivative_order=order,
            backend=self.backend_name,
            model=self.model_identifier,
            execution={
                **dict(execution),
                "resident": True,
                "artifact_loaded_once": True,
                "derivative_order": order.name,
            },
        ).validate_for_system(self.system)


def _geometry_batch(
    values: Sequence[np.ndarray],
    natoms: int,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        geometries = np.asarray(values, dtype=float)
    else:
        items = tuple(np.asarray(item, dtype=float) for item in values)
        if not items:
            return np.empty((0, natoms, 3), dtype=float)
        geometries = np.stack(items)
    if geometries.shape[1:] != (natoms, 3) or geometries.ndim != 3:
        raise ValueError("ZAFF batch must have shape (npoints, natoms, 3)")
    if np.any(~np.isfinite(geometries)):
        raise ValueError("ZAFF batch contains non-finite values")
    return geometries


__all__ = [
    "ZaffBackend",
    "ZaffHarmonicSession",
    "ZaffSeedSession",
    "ZaffSonicSession",
]
