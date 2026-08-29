"""Resident GAFF2/CM5 solvent recipes and Morse--Gaussian GLOB boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .compatibility import normalize_legacy_zaff_payload

from .confinement import (
    EllipsoidalVdwConfinement,
    VdwConfinementResult,
    VdwConfinementSecondOrderResult,
)
from .tip3p_fb import ANGSTROM_TO_BOHR


GLOB_SOLVENT_LIBRARY_SCHEMA = "matrix.zaff.glob_solvent_library.v1"
GLOB_UVDW_TYPE = "GLOB-CENTER-OF-MASS"


@dataclass(frozen=True)
class Gaff2Cm5SolventForceField:
    """A rigid-solvent GAFF2 nonbonded model carrying explicit CM5 charges.

    The same atom types and connectivity are construction inputs for the
    resident ARCHITECT GAFF2 compiler when a flexible solvent is requested.
    """

    name: str
    formula: str
    canonical_smiles: str
    atomic_numbers: np.ndarray
    masses_dalton: np.ndarray
    bonds: tuple[tuple[int, int, float], ...]
    gaff2_atom_types: tuple[str, ...]
    cm5_charges_e: np.ndarray
    vdw_rmin_over_2_angstrom: np.ndarray
    vdw_epsilon_kcal_per_mol: np.ndarray
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        numbers = np.asarray(self.atomic_numbers, dtype=int)
        masses = np.asarray(self.masses_dalton, dtype=float)
        charges = np.asarray(self.cm5_charges_e, dtype=float)
        radii = np.asarray(self.vdw_rmin_over_2_angstrom, dtype=float)
        epsilon = np.asarray(self.vdw_epsilon_kcal_per_mol, dtype=float)
        count = len(numbers)
        if (
            count == 0
            or any(len(value) != count for value in (masses, charges, radii, epsilon))
            or len(self.gaff2_atom_types) != count
            or np.any(numbers <= 0)
            or np.any(masses <= 0.0)
            or np.any(radii < 0.0)
            or np.any(epsilon < 0.0)
            or any(np.any(~np.isfinite(value)) for value in (masses, charges, radii, epsilon))
        ):
            raise ValueError("GAFF2/CM5 solvent arrays are inconsistent")
        if not np.isclose(np.sum(charges), 0.0, atol=1.0e-8):
            raise ValueError("the CM5 charges of a neutral solvent must sum to zero")
        for left, right, order in self.bonds:
            if not 0 <= left < right < count or not float(order) > 0.0:
                raise ValueError("invalid solvent connectivity")
        for value in (numbers, masses, charges, radii, epsilon):
            value.setflags(write=False)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "masses_dalton", masses)
        object.__setattr__(self, "cm5_charges_e", charges)
        object.__setattr__(self, "vdw_rmin_over_2_angstrom", radii)
        object.__setattr__(self, "vdw_epsilon_kcal_per_mol", epsilon)


@dataclass(frozen=True)
class GlobSolventModel:
    """Auditable construction recipe for one organic GLOB solvent."""

    name: str
    aliases: tuple[str, ...]
    record: Mapping[str, Any]
    provenance: Mapping[str, str]

    @property
    def site_count(self) -> int:
        return len(self.record["atomic_numbers"])

    @property
    def masses_dalton(self) -> np.ndarray:
        return np.asarray(self.record["masses_dalton"], dtype=float)

    @property
    def reference_coordinates_angstrom(self) -> np.ndarray:
        coordinates = np.asarray(
            self.record["reference_coordinates_angstrom"], dtype=float
        )
        coordinates.setflags(write=False)
        return coordinates

    def with_cm5_charges(
        self, charges_e: Sequence[float] | None = None
    ) -> Gaff2Cm5SolventForceField:
        """Bind supplied CM5 charges, or the resident L0/APOC reference."""

        return Gaff2Cm5SolventForceField(
            name=self.name,
            formula=str(self.record["formula"]),
            canonical_smiles=str(self.record["canonical_smiles"]),
            atomic_numbers=np.asarray(self.record["atomic_numbers"], dtype=int),
            masses_dalton=self.masses_dalton,
            bonds=tuple(
                (int(left), int(right), float(order))
                for left, right, order in self.record["bonds"]
            ),
            gaff2_atom_types=tuple(self.record["gaff2_atom_types"]),
            cm5_charges_e=np.asarray(
                self.record["cm5_charges_e"]
                if charges_e is None
                else charges_e,
                dtype=float,
            ),
            vdw_rmin_over_2_angstrom=np.asarray(
                self.record["gaff2_vdw_rmin_over_2_angstrom"], dtype=float
            ),
            vdw_epsilon_kcal_per_mol=np.asarray(
                self.record["gaff2_vdw_epsilon_kcal_per_mol"], dtype=float
            ),
            provenance={
                **self.provenance,
                "force_field": "MATRIX resident GAFF2",
                "charges": "molecule-specific CM5",
                "cm5_level": str(self.record["cm5_provenance"]["level"]),
                "cm5_population_analyzer": str(
                    self.record["cm5_provenance"]["population_analyzer"]
                ),
            },
        )


@dataclass(frozen=True)
class MolecularUvdwConfinement:
    """Apply one analytic GLOB potential to molecular centers of mass."""

    solvent: GlobSolventModel
    field: EllipsoidalVdwConfinement

    @property
    def center_weights(self) -> np.ndarray:
        masses = self.solvent.masses_dalton
        return masses / np.sum(masses)

    def molecular_centers_bohr(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        xyz = self._validated_coordinates(coordinates_bohr)
        return np.einsum("s,msk->mk", self.center_weights, xyz)

    def energy(self, coordinates_bohr: np.ndarray) -> float:
        """Energy-only path for one MC configuration."""

        centers = self.molecular_centers_bohr(coordinates_bohr)
        return self.field.energy_single_type(centers, GLOB_UVDW_TYPE)

    def evaluate(self, coordinates_bohr: np.ndarray) -> VdwConfinementResult:
        """Energy and exact Cartesian gradient for MD or optimization."""

        xyz = self._validated_coordinates(coordinates_bohr)
        centers = np.einsum("s,msk->mk", self.center_weights, xyz)
        result = self.field.evaluate_single_type(centers, GLOB_UVDW_TYPE)
        center_gradient = result.gradient_hartree_per_bohr.reshape(-1, 3)
        gradient = (
            center_gradient[:, None, :] * self.center_weights[None, :, None]
        )
        return VdwConfinementResult(result.energy_hartree, gradient.reshape(-1))

    def evaluate_with_hessian(
        self, coordinates_bohr: np.ndarray
    ) -> VdwConfinementSecondOrderResult:
        """Energy, gradient, and exact Cartesian Hessian."""

        xyz = self._validated_coordinates(coordinates_bohr)
        molecule_count, site_count, _ = xyz.shape
        centers = np.einsum("s,msk->mk", self.center_weights, xyz)
        center_result = self.field.evaluate_single_type_with_hessian(
            centers, GLOB_UVDW_TYPE
        )
        transform = np.zeros(
            (3 * molecule_count, 3 * molecule_count * site_count), dtype=float
        )
        for molecule in range(molecule_count):
            for site, weight in enumerate(self.center_weights):
                for axis in range(3):
                    transform[
                        3 * molecule + axis,
                        3 * (molecule * site_count + site) + axis,
                    ] = weight
        gradient = transform.T @ center_result.gradient_hartree_per_bohr
        hessian = transform.T @ center_result.hessian_hartree_per_bohr2 @ transform
        return VdwConfinementSecondOrderResult(
            center_result.energy_hartree, gradient, hessian
        )

    def batch_energies(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        """Vectorized energy-only path for GA or MC populations."""

        populations = np.asarray(coordinates_bohr, dtype=float)
        if (
            populations.ndim != 4
            or populations.shape[2:] != (self.solvent.site_count, 3)
            or np.any(~np.isfinite(populations))
        ):
            raise ValueError(
                "solvent population must have shape "
                "(nconfiguration, nmolecule, nsite, 3)"
            )
        centers = np.einsum(
            "s,pmsk->pmk", self.center_weights, populations
        )
        return self.field.batch_energies(
            centers, (GLOB_UVDW_TYPE,) * centers.shape[1]
        )

    def _validated_coordinates(self, coordinates_bohr: np.ndarray) -> np.ndarray:
        xyz = np.asarray(coordinates_bohr, dtype=float)
        if (
            xyz.ndim != 3
            or xyz.shape[1:] != (self.solvent.site_count, 3)
            or np.any(~np.isfinite(xyz))
        ):
            raise ValueError(
                "solvent coordinates must have shape (nmolecule, nsite, 3)"
            )
        return xyz


def _library_record() -> dict[str, Any]:
    path = files("matrix_zaff").joinpath("data/glob_solvents_2015.json")
    record = normalize_legacy_zaff_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if record.get("schema") != GLOB_SOLVENT_LIBRARY_SCHEMA:
        raise ValueError("unsupported resident GLOB solvent library")
    return record


def available_glob_solvents() -> tuple[str, ...]:
    """Return the canonical organic-solvent names; water is intentionally absent."""

    return tuple(sorted(_library_record()["solvents"]))


def glob_solvent_model(name: str) -> GlobSolventModel:
    """Resolve a canonical name or case-insensitive alias."""

    library = _library_record()
    requested = str(name).strip().casefold()
    for canonical, raw in library["solvents"].items():
        aliases = tuple(str(value) for value in raw["aliases"])
        if requested in {canonical.casefold(), *(value.casefold() for value in aliases)}:
            return GlobSolventModel(
                name=canonical,
                aliases=aliases,
                record=raw,
                provenance={
                    str(key): str(value)
                    for key, value in library["source"].items()
                },
            )
    raise KeyError(f"unknown resident GLOB solvent: {name}")


def glob_source_uvdw_kj_per_mol(
    name: str,
    distance_from_wall_angstrom: np.ndarray | float,
    *,
    compact: bool = True,
) -> np.ndarray:
    """Evaluate the published polynomial or its compact-C2 fit target."""

    uvdw = glob_solvent_model(name).record["uvdw"]
    distance = np.asarray(distance_from_wall_angstrom, dtype=float)
    if np.any(distance < 0.0):
        raise ValueError("distance from the wall cannot be negative")
    cutoff = float(uvdw["compact_cutoff_angstrom"])
    polynomial = np.polynomial.polynomial.polyval(
        distance,
        np.asarray(uvdw["source_polynomial_coefficients_ascending"], dtype=float),
    )
    if not compact:
        return np.where(distance <= cutoff, polynomial, 0.0)
    x = np.minimum(distance / cutoff, 1.0)
    switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return np.where(distance < cutoff, switch * polynomial, 0.0)


def glob_uvdw_confinement(
    name: str,
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
) -> MolecularUvdwConfinement:
    """Build the fitted Morse--Gaussian confinement on a sphere or ellipsoid."""

    solvent = glob_solvent_model(name)
    uvdw = solvent.record["uvdw"]
    field = EllipsoidalVdwConfinement(
        center_bohr=np.asarray(center_bohr, dtype=float),
        semiaxes_bohr=np.asarray(semiaxes_bohr, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        morse_parameters={
            GLOB_UVDW_TYPE: tuple(
                float(value)
                for value in uvdw["morse_parameters_atomic_units"]
            )
        },
        gaussian_terms={
            GLOB_UVDW_TYPE: tuple(
                tuple(float(value) for value in term)
                for term in uvdw["gaussian_terms_atomic_units"]
            )
        },
        layer_depth_bohr=(
            float(uvdw["compact_cutoff_angstrom"]) * ANGSTROM_TO_BOHR
        ),
    )
    return MolecularUvdwConfinement(solvent=solvent, field=field)


__all__ = [
    "GLOB_SOLVENT_LIBRARY_SCHEMA",
    "GLOB_UVDW_TYPE",
    "Gaff2Cm5SolventForceField",
    "GlobSolventModel",
    "MolecularUvdwConfinement",
    "available_glob_solvents",
    "glob_solvent_model",
    "glob_source_uvdw_kj_per_mol",
    "glob_uvdw_confinement",
]
