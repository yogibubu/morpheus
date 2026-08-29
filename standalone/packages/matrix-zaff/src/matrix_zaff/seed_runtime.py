"""Resident execution of transferable diagonal ZAFF seed fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .compatibility import normalize_legacy_zaff_payload

from matrix_chem import (
    HydrogenBondPairList,
    Primitive,
    eval_primitive,
    prepare_hydrogen_bond_recognition,
    primitive_b_matrix,
)
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.elements import atomic_number

from .basic_primitive_derivative import analytic_basic_primitive_hessian_vector
from .bonded import (
    BondOrderDampedAnglePotential,
    BondOrderDampedTorsionPotential,
    BondOrderRadialFactor,
    UFFFourierTerm,
)
from .charge_response import (
    ChargeResponseSite,
    PersistentAllPairSplitChargeResponse,
    PersistentSplitChargeResponse,
    SplitChargeChannel,
)
from .confinement import EllipsoidalVdwConfinement, vdw_confinement_from_record
from .cpcm import CPCMReactionField, cpcm_confinement_from_record
from .interfacial_pcm import (
    InterfacialPCMReactionField,
    interfacial_pcm_reaction_field_from_record,
)
from .local_charge_electrostatics import evaluate_local_charge_electrostatics
from .native_kernels import (
    damped_exppe_energy as _native_damped_exppe_energy,
    damped_exppe_energy_gradient as _native_damped_exppe_energy_gradient,
    damped_exppe_hessian_vector as _native_damped_exppe_hessian_vector,
    local_valence_energy as _native_local_valence_energy,
    local_valence_energy_gradient as _native_local_valence_energy_gradient,
    local_valence_hessian_vector as _native_local_valence_hessian_vector,
    morse_bond_energy as _native_morse_bond_energy,
    morse_bond_energy_gradient as _native_morse_bond_energy_gradient,
    morse_bond_hessian_vector as _native_morse_bond_hessian_vector,
    native_zaff_backend,
)
from .nonbonded import (
    BOHR_TO_ANGSTROM,
    MMRuntimePolicy,
    PersistentGaussianElectrostaticOperator,
    PersistentVerletNeighborList,
    build_nonbonded_neighbor_list,
    electrostatic_energy_gradient,
    electrostatic_hessian_vector_product,
    select_mm_runtime_policy,
)
from .radial import exppe_from_minimum, mie_minimum_derivatives

ZAFF_SEED_MODEL_SCHEMA = "matrix.zaff.diagonal_seed.v2"
ZAFF_SEED_PRIOR_SCHEMA = "matrix.zaff.full_fit_priors.v1"
ZAFF_FAST_RIGID_TORSIONAL_ZOOM = "FAST_RIGID_TORSIONAL"
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631
WaterChargeResponseEvaluator = Callable[..., Any]


class ZaffSeedCoverageError(ValueError):
    """The transferable library cannot parameterize the requested molecule."""


@dataclass(frozen=True)
class ZaffSeedEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None
    execution: Mapping[str, Any]


class ZaffSeedDynamicsRuntime:
    """Persistent E+G runtime for repeated ZAFF seed evaluations."""

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        zoom_level: str | None = None,
        neighbor_skin_angstrom: float = 1.0,
        charge_response_model: str = "variational_qeq_sqe",
        water_charge_response_evaluator: WaterChargeResponseEvaluator | None = None,
    ) -> None:
        self.model = model
        self.fast = _fast_zoom_selected(zoom_level)
        self.zoom_level = zoom_level
        response_model = str(charge_response_model).strip().casefold()
        if response_model not in {
            "variational_qeq_sqe",
            "water_hbond_surrogate",
        }:
            raise ValueError("unsupported ZAFF charge-response runtime model")
        if self.fast and response_model != "variational_qeq_sqe":
            raise ValueError(
                "rigid-torsional zoom and water response surrogate are "
                "independent contracts and cannot be combined"
            )
        self.charge_response_model = response_model
        self.water_charge_response_evaluator = water_charge_response_evaluator
        self.evaluation_count = 0
        nonbonded = dict(model["nonbonded"])
        self.policy = select_mm_runtime_policy(
            np.zeros((len(model["atoms"]), 3)),
            cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
            neighbor_list_minimum_atoms=int(
                nonbonded["neighbor_list_minimum_atoms"]
            ),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            fmm_precision=float(nonbonded["fmm_precision"]),
            materialize_neighbor_pairs=False,
        )
        self.neighbor_list = PersistentVerletNeighborList(
            cutoff_bohr=(
                float(nonbonded["short_range_cutoff_angstrom"])
                / BOHR_TO_ANGSTROM
            ),
            skin_bohr=float(neighbor_skin_angstrom) / BOHR_TO_ANGSTROM,
        )
        atom_parameters = tuple(model["nonbonded_atoms"])
        self.electrostatics = PersistentGaussianElectrostaticOperator(
            np.asarray(
                [float(item["atomic_charge"]) for item in atom_parameters]
            ),
            tuple(int(atomic_number(symbol) or 0) for symbol in model["atoms"]),
            backend="auto",
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        )
        self.charge_response = _persistent_charge_response_from_model(
            model,
            backend="auto",
            reaction_field=_model_reaction_field(model),
        )
        self.hbond_pair_list = None
        if response_model == "water_hbond_surrogate":
            numbers = tuple(
                int(atomic_number(symbol) or 0) for symbol in model["atoms"]
            )
            bonds = tuple(
                tuple(int(value) for value in item["atoms"])
                for item in model["bonds"]
            )
            self.hbond_pair_list = prepare_hydrogen_bond_recognition(
                numbers,
                bonds,
                selector_threshold=1.0e-8,
                minimum_angle_degrees=90.0,
            ).new_pair_list(skin_angstrom=float(neighbor_skin_angstrom))
        self.morse_bond_arrays = _seed_morse_bond_arrays(model)
        self.local_valence_arrays = _seed_local_valence_arrays(model)

    def evaluate(self, coordinates_angstrom: np.ndarray) -> ZaffSeedEvaluation:
        energy, gradient, execution = _seed_energy_gradient(
            self.model,
            coordinates_angstrom,
            fast=self.fast,
            charge_response_model=self.charge_response_model,
            water_charge_response_evaluator=self.water_charge_response_evaluator,
            hbond_pair_list=self.hbond_pair_list,
            neighbor_list=self.neighbor_list,
            electrostatic_operator=self.electrostatics,
            charge_response_runtime=self.charge_response,
            runtime_policy=self.policy,
            morse_bond_arrays=self.morse_bond_arrays,
            local_valence_arrays=self.local_valence_arrays,
        )
        self.evaluation_count += 1
        execution = {
            **execution,
            "persistent_neighbor_list": True,
            "neighbor_list_rebuild_count": self.neighbor_list.rebuild_count,
            "neighbor_list_reuse_count": self.neighbor_list.reuse_count,
            "persistent_electrostatic_operator": True,
            "persistent_fmm": self.electrostatics.persistent_fmm,
            "hbond_pair_list_rebuild_count": (
                0
                if self.hbond_pair_list is None
                else self.hbond_pair_list.rebuild_count
            ),
            "electrostatic_evaluation_count": self.electrostatics.evaluation_count,
        }
        return ZaffSeedEvaluation(energy, gradient, None, execution)

    def hessian_vector_product(
        self,
        coordinates_angstrom: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the analytic Hessian with all reusable runtime state resident."""

        return _seed_hessian_vector_product(
            self.model,
            coordinates_angstrom,
            vector_bohr,
            fast=self.fast,
            neighbor_list=self.neighbor_list,
            electrostatic_operator=self.electrostatics,
            charge_response_runtime=self.charge_response,
            runtime_policy=self.policy,
            morse_bond_arrays=self.morse_bond_arrays,
            local_valence_arrays=self.local_valence_arrays,
        )


@lru_cache(maxsize=16)
def _compile_cpcm_record(payload: str) -> CPCMReactionField:
    return CPCMReactionField.compile(
        cpcm_confinement_from_record(normalize_legacy_zaff_payload(json.loads(payload)))
    )


def _model_cpcm_field(model: Mapping[str, Any]) -> CPCMReactionField | None:
    raw = model.get("cpcm_reaction_field")
    if raw is None:
        return None
    return _compile_cpcm_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )


@lru_cache(maxsize=16)
def _compile_interfacial_pcm_record(
    payload: str,
) -> InterfacialPCMReactionField:
    return interfacial_pcm_reaction_field_from_record(
        normalize_legacy_zaff_payload(json.loads(payload))
    )


def _model_interfacial_pcm_field(
    model: Mapping[str, Any],
) -> InterfacialPCMReactionField | None:
    raw = model.get("interfacial_pcm_reaction_field")
    if raw is None:
        return None
    return _compile_interfacial_pcm_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )


def _model_reaction_field(
    model: Mapping[str, Any],
) -> CPCMReactionField | InterfacialPCMReactionField | None:
    cpcm = _model_cpcm_field(model)
    interfacial = _model_interfacial_pcm_field(model)
    if cpcm is not None and interfacial is not None:
        raise ValueError(
            "ZAFF model cannot combine homogeneous CPCM and interfacial PCM"
        )
    return cpcm if cpcm is not None else interfacial


def _persistent_charge_response_from_model(
    model: Mapping[str, Any],
    *,
    backend: str,
    reaction_field: CPCMReactionField | InterfacialPCMReactionField | None,
) -> PersistentSplitChargeResponse | PersistentAllPairSplitChargeResponse:
    response_record = dict(model["charge_response"])
    sites = tuple(
        ChargeResponseSite(
            float(item["reference_charge"]),
            float(item["gaussian_width_bohr"]),
            tuple(
                (int(parent), float(weight))
                for parent, weight in item["atom_weights"]
            ),
            str(item["label"]),
        )
        for item in response_record["sites"]
    )
    nonbonded = dict(model["nonbonded"])
    generator = response_record.get("channel_generator")
    if generator is not None:
        generated = dict(generator)
        return PersistentAllPairSplitChargeResponse(
            sites,
            np.asarray(generated["response_lengths"], dtype=float),
            np.asarray(generated["rmin_half_angstrom"], dtype=float),
            np.asarray(
                response_record["reference_geometry_angstrom"], dtype=float
            )
            / BOHR_TO_ANGSTROM,
            tolerance=1.0e-11,
            backend=backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
            reaction_field=reaction_field,
            switch_minimum_angstrom=float(
                generated.get("switch_minimum_angstrom", 4.0)
            ),
            switch_scale=float(generated.get("switch_scale", 1.25)),
            switch_exponent=int(generated.get("switch_exponent", 6)),
            pair_chunk_size=int(generated.get("pair_chunk_size", 65536)),
        )
    channels = tuple(
        SplitChargeChannel(
            int(item["left_site"]),
            int(item["right_site"]),
            float(item["hardness_hartree"]),
            float(item["switch_radius_bohr"]),
            int(item["switch_exponent"]),
            float(item["reference_bias_hartree"]),
        )
        for item in response_record["channels"]
    )
    return PersistentSplitChargeResponse(
        sites,
        channels,
        tolerance=1.0e-11,
        backend=backend,
        fmm_precision=float(nonbonded["fmm_precision"]),
        fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
        reaction_field=reaction_field,
    )


@lru_cache(maxsize=16)
def _compile_vdw_confinement_record(payload: str) -> EllipsoidalVdwConfinement:
    return vdw_confinement_from_record(
        normalize_legacy_zaff_payload(json.loads(payload))
    )


def _model_vdw_confinement(
    model: Mapping[str, Any],
    *,
    cpcm_field: CPCMReactionField | None = None,
) -> EllipsoidalVdwConfinement | None:
    raw = model.get("vdw_confinement")
    if raw is None:
        return None
    calibration = model.get("vdw_calibration")
    if calibration is not None and str(
        dict(calibration).get("status", "")
    ) != "FROZEN_FOR_PRODUCTION":
        raise ValueError("ZAFF runtime refuses a non-frozen UvdW calibration")
    confinement = _compile_vdw_confinement_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )
    if cpcm_field is not None:
        cpcm = cpcm_field.confinement
        if not (
            np.allclose(
                confinement.center_bohr, cpcm.center_bohr, rtol=0.0, atol=1.0e-10
            )
            and np.allclose(
                confinement.semiaxes_bohr,
                cpcm.semiaxes_bohr,
                rtol=0.0,
                atol=1.0e-10,
            )
            and np.allclose(
                confinement.rotation, cpcm.rotation, rtol=0.0, atol=1.0e-10
            )
        ):
            raise ValueError("CPCM and UvdW must share one confinement ellipsoid")
    return confinement



def evaluate_zaff_seed_model(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    hessian: bool = False,
    zoom_level: str | None = None,
) -> ZaffSeedEvaluation:
    """Evaluate transferable ZAFF with analytic energy, gradient and Hessian."""

    fast = _fast_zoom_selected(zoom_level)
    energy, gradient, execution = _seed_energy_gradient(
        model, coordinates_angstrom, fast=fast
    )
    matrix = None
    if hessian:
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        ncart = xyz.size
        matrix = np.zeros((ncart, ncart), dtype=float)
        interfacial_field = _model_interfacial_pcm_field(model)
        for column in range(ncart):
            direction = np.zeros(ncart)
            direction[column] = 1.0
            matrix[:, column] = _seed_hessian_vector_product(
                model,
                xyz,
                direction,
                fast=fast,
                include_fixed_reaction_field=interfacial_field is None,
            )
        if interfacial_field is not None:
            charges = np.asarray(
                [
                    float(item["atomic_charge"])
                    for item in model["nonbonded_atoms"]
                ]
            )
            coordinates_bohr = xyz / BOHR_TO_ANGSTROM
            batch_size = 32
            for start in range(0, ncart, batch_size):
                stop = min(ncart, start + batch_size)
                directions = np.zeros((stop - start, ncart), dtype=float)
                directions[np.arange(stop - start), np.arange(start, stop)] = 1.0
                products = interfacial_field.hessian_vector_products(
                    coordinates_bohr,
                    charges,
                    directions,
                )
                matrix[:, start:stop] += products.T
            execution = {
                **dict(execution),
                "interfacial_hessian_batch_size": batch_size,
            }
        matrix = 0.5 * (matrix + matrix.T)
    return ZaffSeedEvaluation(energy, gradient, matrix, execution)


def evaluate_zaff_fast_rigid_torsional_batch_energies(
    model: Mapping[str, Any],
    geometries_angstrom: Sequence[np.ndarray] | np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Vectorized energy-only hot loop for rigid-torsional population ranking."""

    geometries = np.asarray(geometries_angstrom, dtype=float)
    natoms = len(model["atoms"])
    if (
        geometries.ndim != 3
        or geometries.shape[1:] != (natoms, 3)
        or np.any(~np.isfinite(geometries))
    ):
        raise ValueError("fast ZAFF batch must have shape (ncandidate, natoms, 3)")
    block_size = max(1, int(chunk_size))
    if len(geometries) == 0:
        return np.zeros(0)
    pair_left, pair_right = np.triu_indices(natoms, k=1)
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    widths = np.asarray(
        [
            float(covalent_radius(int(atomic_number(symbol) or 0)) or 0.75)
            / BOHR_TO_ANGSTROM
            for symbol in model["atoms"]
        ]
    )
    pair_charge = charges[pair_left] * charges[pair_right]
    pair_beta = 1.0 / np.sqrt(
        2.0 * (widths[pair_left] ** 2 + widths[pair_right] ** 2)
    )
    epsilon = np.sqrt(
        np.asarray(
            [
                float(atom_parameters[index]["epsilon_hartree"])
                for index in pair_left
            ]
        )
        * np.asarray(
            [
                float(atom_parameters[index]["epsilon_hartree"])
                for index in pair_right
            ]
        )
    )
    r_min = np.asarray(
        [
            float(atom_parameters[left]["rmin_half_angstrom"])
            + float(atom_parameters[right]["rmin_half_angstrom"])
            for left, right in zip(pair_left, pair_right, strict=True)
        ]
    )
    alpha = np.asarray(
        [
            exppe_from_minimum(
                mie_minimum_derivatives(float(depth), float(radius), 12.0, 6.0)
            ).alpha
            for depth, radius in zip(epsilon, r_min, strict=True)
        ]
    )
    try:
        from scipy.special import erf as vector_erf
    except ImportError:  # pragma: no cover - MATRIX-SMITH normally supplies SciPy.
        vector_erf = np.vectorize(math.erf, otypes=[float])

    energies = np.zeros(len(geometries))
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    for start in range(0, len(geometries), block_size):
        stop = min(start + block_size, len(geometries))
        block = geometries[start:stop]
        delta = block[:, pair_left, :] - block[:, pair_right, :]
        distance_angstrom = np.linalg.norm(delta, axis=2)
        if np.any(distance_angstrom <= 1.0e-12):
            raise ValueError("coincident atoms in fast ZAFF batch")
        distance_bohr = distance_angstrom / BOHR_TO_ANGSTROM
        electrostatic = np.sum(
            pair_charge[None, :]
            * vector_erf(pair_beta[None, :] * distance_bohr)
            / distance_bohr,
            axis=1,
        )
        scaled = distance_angstrom / r_min[None, :]
        exp_full = np.exp(alpha[None, :] * (1.0 - scaled))
        exp_half = np.exp(0.5 * alpha[None, :] * (1.0 - scaled))
        polynomial = scaled**4 - 2.0 * scaled**2 + 3.0
        radial = epsilon[None, :] * (exp_full - polynomial * exp_half)
        ratio = (0.72 * r_min[None, :] / distance_angstrom) ** 8
        collective = np.sum(radial / (1.0 + ratio), axis=1)
        torsional = np.zeros(stop - start)
        for raw in model["torsions"]:
            i, j, k, ell = (int(value) for value in raw["atoms"])
            b1 = block[:, i, :] - block[:, j, :]
            b2 = block[:, k, :] - block[:, j, :]
            b3 = block[:, ell, :] - block[:, k, :]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            b2_unit = b2 / np.linalg.norm(b2, axis=1)[:, None]
            phi = np.arctan2(
                np.einsum("bi,bi->b", np.cross(n1, n2), b2_unit),
                np.einsum("bi,bi->b", n1, n2),
            )
            for term in raw["terms"]:
                torsional += float(term["amplitude_hartree"]) * (
                    1.0
                    + np.cos(
                        int(term["periodicity"]) * phi
                        - float(term["phase_radian"])
                    )
                )
        reaction = (
            np.zeros(stop - start)
            if reaction_field is None
            else reaction_field.reaction_batch_energies(
                block / BOHR_TO_ANGSTROM, charges
            )
        )
        vdw = (
            np.zeros(stop - start)
            if vdw_confinement is None
            else vdw_confinement.batch_energies(
                block / BOHR_TO_ANGSTROM, model["atoms"]
            )
        )
        energies[start:stop] = (
            electrostatic + collective + torsional + reaction + vdw
        )
    return energies


def evaluate_zaff_seed_energy(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    zoom_level: str | None = None,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: PersistentSplitChargeResponse | None = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a general seed field without constructing any derivatives."""

    if model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("unsupported ZAFF seed model schema")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(model["atoms"]), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("ZAFF seed evaluation geometry has wrong shape")
    fast = _fast_zoom_selected(zoom_level)
    numerical = native_zaff_backend(len(xyz))
    energy = 0.0
    if not fast:
        bond_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        if numerical.accelerated and len(bond_arrays[0]):
            bond_energy, _bond_count = _native_morse_bond_energy(
                xyz,
                *bond_arrays,
            )
            energy += bond_energy
        else:
            for raw in model["bonds"]:
                distance = eval_primitive(
                    Primitive("bond", tuple(int(value) for value in raw["atoms"])),
                    xyz,
                )
                exponent = math.exp(
                    -float(raw["alpha_per_angstrom"])
                    * (distance - float(raw["r0_angstrom"]))
                )
                energy += float(raw["d_e_hartree"]) * (1.0 - exponent) ** 2
        valence_arrays = (
            local_valence_arrays
            if local_valence_arrays is not None
            else _seed_local_valence_arrays(model)
        )
        if numerical.accelerated and (
            len(valence_arrays[0]) or len(valence_arrays[2])
        ):
            valence_energy, _valence_count = _native_local_valence_energy(
                xyz, *valence_arrays
            )
            energy += valence_energy
        for raw in (() if numerical.accelerated else model["angles"]):
            atoms3 = tuple(int(value) for value in raw["atoms"])
            primitives = (
                Primitive("bond", (atoms3[0], atoms3[1])),
                Primitive("bond", (atoms3[1], atoms3[2])),
                Primitive("angle", atoms3),
            )
            factors = tuple(
                BondOrderRadialFactor(
                    float(item["reference_distance_angstrom"]),
                    float(item["covalent_radius_sum_angstrom"]),
                )
                for item in raw["radial_factors"]
            )
            energy += BondOrderDampedAnglePotential(
                2.0 * float(raw["k_hartree_per_radian2"]),
                float(raw["theta0_radian"]),
                factors[0],
                factors[1],
            ).energy(*(eval_primitive(item, xyz) for item in primitives))
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        phi = eval_primitive(Primitive("dihedral", atoms4), xyz)
        terms = tuple(
            UFFFourierTerm(
                float(item["amplitude_hartree"]),
                int(item["periodicity"]),
                float(item["phase_radian"]),
            )
            for item in raw["terms"]
        )
        if fast:
            energy += sum(
                term.amplitude
                * (1.0 + math.cos(term.periodicity * phi - term.phase))
                for term in terms
            )
        else:
            bonds = (
                Primitive("bond", (atoms4[0], atoms4[1])),
                Primitive("bond", (atoms4[1], atoms4[2])),
                Primitive("bond", (atoms4[2], atoms4[3])),
            )
            factors = tuple(
                BondOrderRadialFactor(
                    float(item["reference_distance_angstrom"]),
                    float(item["covalent_radius_sum_angstrom"]),
                )
                for item in raw["radial_factors"]
            )
            energy += BondOrderDampedTorsionPotential(
                factors[0], factors[1], factors[2], terms
            ).energy(
                *(eval_primitive(item, xyz) for item in bonds),
                phi,
            )

    nonbonded = dict(model["nonbonded"])
    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
    )
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    operator = electrostatic_operator
    if operator is None:
        operator = PersistentGaussianElectrostaticOperator(
            charges,
            tuple(int(atomic_number(symbol) or 0) for symbol in model["atoms"]),
            backend=policy.electrostatic_backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        )
    energy += operator.energy(coordinates_bohr)
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    if reaction_field is not None:
        energy += float(
            reaction_field.reaction_batch_energies(
                coordinates_bohr[None, :, :], charges
            )[0]
        )
    response = None
    if not fast:
        response_runtime = charge_response_runtime or (
            _persistent_charge_response_from_model(
                model,
                backend=policy.electrostatic_backend,
                reaction_field=reaction_field,
            )
        )
        response = response_runtime.solve(
            coordinates_bohr,
            compute_gradient=False,
        )
        energy += response.energy_correction_hartree
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        energy += float(
            vdw_confinement.batch_energies(
                coordinates_bohr[None, :, :], model["atoms"]
            )[0]
        )
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
    )
    if numerical.accelerated:
        radial_energy, _pair_count = _native_damped_exppe_energy(
            xyz,
            np.asarray(
                [float(item["epsilon_hartree"]) for item in atom_parameters]
            ),
            np.asarray(
                [float(item["rmin_half_angstrom"]) for item in atom_parameters]
            ),
            pairs,
            float(nonbonded["short_range_cutoff_angstrom"]),
        )
        energy += radial_energy
    else:
        for left, right in pairs:
            left_parameter = atom_parameters[left]
            right_parameter = atom_parameters[right]
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            distance = float(np.linalg.norm(xyz[left] - xyz[right]))
            radial = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            ).energy(distance)
            energy += radial / (1.0 + (0.72 * r_min / distance) ** 8)
    return float(energy), {
        "backend": "zaff_seed_energy_only",
        "mm_runtime": policy.to_dict(),
        "electrostatic_backend": operator.selected_backend,
        "cpcm_backend": (
            "DISABLED" if cpcm_field is None else cpcm_field.modal_backend
        ),
        "reaction_field_backend": (
            "DISABLED"
            if reaction_field is None
            else reaction_field.modal_backend
        ),
        "vdw_confinement_backend": (
            "DISABLED"
            if vdw_confinement is None
            else "ELLIPSOIDAL_HOMOTHETIC_POLYNOMIAL"
        ),
        "polarization_iterations": 0 if response is None else response.iterations,
        "coordinate_transform_backend": "NONE_ENERGY_ONLY",
    }


def _seed_energy_gradient(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    fast: bool = False,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: PersistentSplitChargeResponse | None = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
    charge_response_model: str = "variational_qeq_sqe",
    water_charge_response_evaluator: WaterChargeResponseEvaluator | None = None,
    hbond_pair_list: HydrogenBondPairList | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("unsupported ZAFF seed model schema")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(model["atoms"]), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("ZAFF seed evaluation geometry has wrong shape")
    energy = 0.0
    gradient_angstrom = np.zeros(xyz.size, dtype=float)
    primitives: list[Primitive] = []
    coordinate_gradients: list[float] = []
    numerical = native_zaff_backend(len(xyz))
    if not fast:
        bond_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        if numerical.accelerated and len(bond_arrays[0]):
            bond_energy, bond_gradient, _bond_count = (
                _native_morse_bond_energy_gradient(
                    xyz,
                    *bond_arrays,
                )
            )
            energy += bond_energy
            gradient_angstrom += (
                bond_gradient.reshape(-1) / BOHR_TO_ANGSTROM
            )
        else:
            for raw in model["bonds"]:
                primitive = Primitive(
                    "bond", tuple(int(value) for value in raw["atoms"])
                )
                distance = eval_primitive(primitive, xyz)
                exponent = math.exp(
                    -float(raw["alpha_per_angstrom"])
                    * (distance - float(raw["r0_angstrom"]))
                )
                depth = float(raw["d_e_hartree"])
                alpha = float(raw["alpha_per_angstrom"])
                energy += depth * (1.0 - exponent) ** 2
                coordinate_gradients.append(
                    2.0 * depth * alpha * exponent * (1.0 - exponent)
                )
                primitives.append(primitive)
    valence_arrays = (
        local_valence_arrays
        if local_valence_arrays is not None
        else _seed_local_valence_arrays(model)
    )
    if (
        not fast
        and numerical.accelerated
        and (len(valence_arrays[0]) or len(valence_arrays[2]))
    ):
        valence_energy, valence_gradient, _valence_count = (
            _native_local_valence_energy_gradient(xyz, *valence_arrays)
        )
        energy += valence_energy
        gradient_angstrom += valence_gradient.reshape(-1) / BOHR_TO_ANGSTROM
    for raw in (
        ()
        if fast or numerical.accelerated
        else model["angles"]
    ):
        atoms3 = tuple(int(value) for value in raw["atoms"])
        primitive = Primitive("angle", atoms3)
        left_bond = Primitive("bond", (atoms3[0], atoms3[1]))
        right_bond = Primitive("bond", (atoms3[1], atoms3[2]))
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        potential = BondOrderDampedAnglePotential(
            2.0 * float(raw["k_hartree_per_radian2"]),
            float(raw["theta0_radian"]),
            factors[0],
            factors[1],
        )
        internal = potential.derivatives(
            eval_primitive(left_bond, xyz),
            eval_primitive(right_bond, xyz),
            eval_primitive(primitive, xyz),
        )
        energy += internal.energy
        primitives.extend((left_bond, right_bond, primitive))
        coordinate_gradients.extend(float(value) for value in internal.gradient)
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        primitive = Primitive("dihedral", atoms4)
        torsion_bonds = (
            Primitive("bond", (atoms4[0], atoms4[1])),
            Primitive("bond", (atoms4[1], atoms4[2])),
            Primitive("bond", (atoms4[2], atoms4[3])),
        )
        terms = tuple(
            UFFFourierTerm(
                float(item["amplitude_hartree"]),
                int(item["periodicity"]),
                float(item["phase_radian"]),
            )
            for item in raw["terms"]
        )
        if fast:
            phi = eval_primitive(primitive, xyz)
            value = first = 0.0
            for term in terms:
                argument = term.periodicity * phi - term.phase
                value += term.amplitude * (1.0 + math.cos(argument))
                first -= (
                    term.amplitude * term.periodicity * math.sin(argument)
                )
            energy += value
            primitives.append(primitive)
            coordinate_gradients.append(first)
            continue
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedTorsionPotential(
            factors[0], factors[1], factors[2], terms
        ).derivatives(
            *(eval_primitive(bond, xyz) for bond in torsion_bonds),
            eval_primitive(primitive, xyz),
        )
        energy += internal.energy
        primitives.extend((*torsion_bonds, primitive))
        coordinate_gradients.extend(float(value) for value in internal.gradient)
    if primitives:
        gradient_angstrom += primitive_b_matrix(primitives, xyz).T @ np.asarray(
            coordinate_gradients
        )

    nonbonded = dict(model["nonbonded"])
    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
    )
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    gradient_bohr = gradient_angstrom * BOHR_TO_ANGSTROM
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    response_model = str(charge_response_model).strip().casefold()
    local_response = None
    if response_model == "water_hbond_surrogate":
        if water_charge_response_evaluator is None:
            raise RuntimeError(
                "water_hbond_surrogate requires an explicit perception callback"
            )
        waters = _water_groups_from_seed_model(model)
        local_response = water_charge_response_evaluator(
            xyz / BOHR_TO_ANGSTROM,
            waters,
            reference_charges_e=charges,
            recognition=hbond_pair_list,
        )
        widths = np.asarray(
            [
                float(item["gaussian_width_bohr"])
                for item in model["charge_response"]["sites"]
            ]
        )
        local_electrostatic = evaluate_local_charge_electrostatics(
            xyz / BOHR_TO_ANGSTROM,
            widths,
            local_response,
            backend=policy.electrostatic_backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
            reaction_field=reaction_field,
        )
        energy += local_electrostatic.energy_hartree
        gradient_bohr += local_electrostatic.gradient_hartree_per_bohr
        electrostatic_backend = local_electrostatic.backend
        pair_correction_count = 0
    elif response_model == "variational_qeq_sqe":
        electrostatic = (
            electrostatic_energy_gradient(
                xyz / BOHR_TO_ANGSTROM,
                charges,
                (),
                atomic_numbers=tuple(
                    int(atomic_number(symbol) or 0) for symbol in model["atoms"]
                ),
                electrostatic_model="gaussian_erf_all_pairs",
                backend="auto",
                fmm_precision=float(nonbonded["fmm_precision"]),
                fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            )
            if electrostatic_operator is None
            else electrostatic_operator.evaluate(xyz / BOHR_TO_ANGSTROM)
        )
        energy += electrostatic.energy_hartree
        gradient_bohr += electrostatic.gradient_hartree_per_bohr.reshape(-1)
        electrostatic_backend = electrostatic.backend
        pair_correction_count = electrostatic.pair_correction_count
        if reaction_field is not None:
            reaction_energy, reaction_gradient = (
                reaction_field.reaction_energy_gradient(
                    xyz / BOHR_TO_ANGSTROM, charges
                )
            )
            energy += reaction_energy
            gradient_bohr += reaction_gradient.reshape(-1)
    else:
        raise ValueError("unsupported ZAFF charge-response runtime model")
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        vdw_result = vdw_confinement.evaluate(
            xyz / BOHR_TO_ANGSTROM, model["atoms"]
        )
        energy += vdw_result.energy_hartree
        gradient_bohr += vdw_result.gradient_hartree_per_bohr
    response = None
    if not fast and response_model == "variational_qeq_sqe":
        response_runtime = charge_response_runtime or (
            _persistent_charge_response_from_model(
                model,
                backend=policy.electrostatic_backend,
                reaction_field=reaction_field,
            )
        )
        response = response_runtime.solve(xyz / BOHR_TO_ANGSTROM)
        energy += response.energy_correction_hartree
        gradient_bohr += response.gradient_correction_hartree_per_bohr

    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
    )
    if numerical.accelerated:
        radial_energy, radial_gradient, _pair_count = (
            _native_damped_exppe_energy_gradient(
                xyz,
                np.asarray(
                    [float(item["epsilon_hartree"]) for item in atom_parameters]
                ),
                np.asarray(
                    [
                        float(item["rmin_half_angstrom"])
                        for item in atom_parameters
                    ]
                ),
                pairs,
                float(nonbonded["short_range_cutoff_angstrom"]),
            )
        )
        energy += radial_energy
        gradient_bohr += radial_gradient.reshape(-1)
    else:
        for left, right in pairs:
            left_parameter, right_parameter = (
                atom_parameters[left],
                atom_parameters[right],
            )
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            potential = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            )
            delta = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(delta))
            radial = potential.derivatives(distance)
            ratio = (0.72 * r_min / distance) ** 8
            switch = 1.0 / (1.0 + ratio)
            switch_derivative = 8.0 * switch * (1.0 - switch) / distance
            energy += switch * radial.energy
            pair_gradient_bohr = (
                switch * radial.first + switch_derivative * radial.energy
            ) * BOHR_TO_ANGSTROM * delta / distance
            gradient_bohr[3 * left : 3 * left + 3] += pair_gradient_bohr
            gradient_bohr[3 * right : 3 * right + 3] -= pair_gradient_bohr
    fast_terms = (
        (
            "electrostatics",
            "exppe_nonbonded",
            "torsions",
            "reaction_field",
            "vdw_confinement",
        )
        if fast
        else (
            "covalent_bonds",
            "angles",
            "torsions",
            "electrostatics",
            "charge_response_polarization",
            "reaction_field",
            "exppe_nonbonded",
            "vdw_confinement",
        )
    )
    omitted_terms = (
        ("covalent_bonds", "angles", "charge_response_polarization")
        if fast
        else ()
    )
    return float(energy), gradient_bohr, {
        "backend": (
            "zaff_fast_rigid_torsional"
            if fast
            else "zaff_diagonal_seed"
        ),
        "mm_runtime": policy.to_dict(),
        "electrostatic_backend": electrostatic_backend,
        "polarization_backend": (
            "DISABLED_BY_FAST_RIGID_TORSIONAL_CONTRACT"
            if fast
            else (
                "MATRIX_SHARED_LOCAL_HBOND_RESPONSE"
                if local_response is not None
                else response.backend
            )
        ),
        "cpcm_backend": (
            "DISABLED"
            if cpcm_field is None
            else cpcm_field.modal_backend
        ),
        "reaction_field_backend": (
            "DISABLED"
            if reaction_field is None
            else reaction_field.modal_backend
        ),
        "vdw_confinement_backend": (
            "DISABLED"
            if vdw_confinement is None
            else "ELLIPSOIDAL_HOMOTHETIC_POLYNOMIAL"
        ),
        "polarization_iterations": 0 if response is None else response.iterations,
        "polarization_residual_norm": (
            0.0 if response is None else response.residual_norm
        ),
        "coordinate_transform_backend": (
            "NONE_DIRECT_CARTESIAN_DIHEDRALS"
            if fast
            else "MATRIX_PRIMITIVE_ANALYTIC"
        ),
        "pair_correction_count": pair_correction_count,
        "included_terms": fast_terms,
        "omitted_terms": omitted_terms,
        "gradient_backend": "ANALYTIC_CARTESIAN",
    }


def _water_groups_from_seed_model(
    model: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    """Return O-H-H groups without assuming contiguous atom ordering."""

    atoms = tuple(str(value).strip().upper() for value in model["atoms"])
    if any(atom not in {"O", "H"} for atom in atoms):
        raise ValueError(
            "the closed-form water response currently requires an all-water system"
        )
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(atoms))}
    for item in model["bonds"]:
        left, right = (int(value) for value in item["atoms"])
        adjacency[left].append(right)
        adjacency[right].append(left)
    waters: list[tuple[int, int, int]] = []
    assigned: set[int] = set()
    for oxygen, atom in enumerate(atoms):
        if atom != "O":
            continue
        hydrogens = sorted(
            neighbor
            for neighbor in adjacency[oxygen]
            if atoms[neighbor] == "H"
        )
        if len(hydrogens) != 2:
            raise ValueError("each surrogate water oxygen must have exactly two hydrogens")
        group = (oxygen, hydrogens[0], hydrogens[1])
        if any(index in assigned for index in group):
            raise ValueError("surrogate water groups overlap")
        assigned.update(group)
        waters.append(group)
    if len(assigned) != len(atoms):
        raise ValueError("every atom must belong to one surrogate water group")
    return tuple(waters)


def _seed_hessian_vector_product(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    vector_bohr: np.ndarray,
    *,
    fast: bool = False,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: PersistentSplitChargeResponse | None = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
    include_fixed_reaction_field: bool = True,
) -> np.ndarray:
    """Apply the exact analytic seed Hessian without numerical differentiation."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    direction = np.asarray(vector_bohr, dtype=float).reshape(-1)
    if xyz.shape != (len(model["atoms"]), 3) or direction.size != xyz.size:
        raise ValueError("ZAFF seed Hessian-vector dimensions are inconsistent")
    product = np.zeros(xyz.size)
    numerical = native_zaff_backend(len(xyz))
    if not fast:
        bond_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        if numerical.accelerated and len(bond_arrays[0]):
            bond_product, _bond_count = _native_morse_bond_hessian_vector(
                xyz,
                *bond_arrays,
                direction.reshape(xyz.shape),
            )
            product += bond_product.reshape(-1)
        else:
            for raw in model["bonds"]:
                primitive = Primitive(
                    "bond", tuple(int(value) for value in raw["atoms"])
                )
                distance = eval_primitive(primitive, xyz)
                alpha = float(raw["alpha_per_angstrom"])
                depth = float(raw["d_e_hartree"])
                exponent = math.exp(
                    -alpha * (distance - float(raw["r0_angstrom"]))
                )
                gradient_internal = np.asarray(
                    [2.0 * depth * alpha * exponent * (1.0 - exponent)]
                )
                hessian_internal = np.asarray(
                    [
                        [
                            2.0
                            * depth
                            * alpha**2
                            * (2.0 * exponent**2 - exponent)
                        ]
                    ]
                )
                product += _internal_term_hessian_vector_product(
                    (primitive,),
                    gradient_internal,
                    hessian_internal,
                    xyz,
                    direction,
                )
    valence_arrays = (
        local_valence_arrays
        if local_valence_arrays is not None
        else _seed_local_valence_arrays(model)
    )
    if (
        not fast
        and numerical.accelerated
        and (len(valence_arrays[0]) or len(valence_arrays[2]))
    ):
        valence_product, _valence_count = _native_local_valence_hessian_vector(
            xyz, *valence_arrays, direction.reshape(xyz.shape)
        )
        product += valence_product.reshape(-1)
    for raw in (
        ()
        if fast or numerical.accelerated
        else model["angles"]
    ):
        atoms3 = tuple(int(value) for value in raw["atoms"])
        primitives = (
            Primitive("bond", (atoms3[0], atoms3[1])),
            Primitive("bond", (atoms3[1], atoms3[2])),
            Primitive("angle", atoms3),
        )
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedAnglePotential(
            2.0 * float(raw["k_hartree_per_radian2"]),
            float(raw["theta0_radian"]),
            factors[0],
            factors[1],
        ).derivatives(*(eval_primitive(item, xyz) for item in primitives))
        product += _internal_term_hessian_vector_product(
            primitives,
            internal.gradient,
            internal.hessian,
            xyz,
            direction,
        )
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        if fast:
            primitive = Primitive("dihedral", atoms4)
            phi = eval_primitive(primitive, xyz)
            first = second = 0.0
            for item in raw["terms"]:
                amplitude = float(item["amplitude_hartree"])
                periodicity = int(item["periodicity"])
                argument = periodicity * phi - float(item["phase_radian"])
                first -= amplitude * periodicity * math.sin(argument)
                second -= amplitude * periodicity**2 * math.cos(argument)
            product += _internal_term_hessian_vector_product(
                (primitive,),
                np.asarray([first]),
                np.asarray([[second]]),
                xyz,
                direction,
            )
            continue
        primitives = (
            Primitive("bond", (atoms4[0], atoms4[1])),
            Primitive("bond", (atoms4[1], atoms4[2])),
            Primitive("bond", (atoms4[2], atoms4[3])),
            Primitive("dihedral", atoms4),
        )
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedTorsionPotential(
            factors[0],
            factors[1],
            factors[2],
            tuple(
                UFFFourierTerm(
                    float(item["amplitude_hartree"]),
                    int(item["periodicity"]),
                    float(item["phase_radian"]),
                )
                for item in raw["terms"]
            ),
        ).derivatives(*(eval_primitive(item, xyz) for item in primitives))
        product += _internal_term_hessian_vector_product(
            primitives,
            internal.gradient,
            internal.hessian,
            xyz,
            direction,
        )

    nonbonded = dict(model["nonbonded"])
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    if electrostatic_operator is None:
        product += electrostatic_hessian_vector_product(
            coordinates_bohr,
            charges,
            direction,
            (),
            atomic_numbers=tuple(
                int(atomic_number(symbol) or 0) for symbol in model["atoms"]
            ),
            electrostatic_model="gaussian_erf_all_pairs",
            backend="auto",
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        )
    else:
        product += electrostatic_operator.hessian_vector_product(
            coordinates_bohr,
            direction,
        )
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    if reaction_field is not None and include_fixed_reaction_field:
        product += reaction_field.hessian_vector_product(
            coordinates_bohr, charges, direction
        )
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        product += vdw_confinement.hessian_vector_product(
            coordinates_bohr, model["atoms"], direction
        )
    if not fast:
        if charge_response_runtime is None:
            response_runtime = _persistent_charge_response_from_model(
                model,
                backend="auto",
                reaction_field=reaction_field,
            )
            product += response_runtime.hessian_vector_product(
                coordinates_bohr,
                direction,
            )
        else:
            product += charge_response_runtime.hessian_vector_product(
                coordinates_bohr,
                direction,
            )

    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
    )
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
    )
    direction_xyz = direction.reshape(xyz.shape)
    if numerical.accelerated:
        radial_product, _pair_count = _native_damped_exppe_hessian_vector(
            xyz,
            np.asarray(
                [float(item["epsilon_hartree"]) for item in atom_parameters]
            ),
            np.asarray(
                [float(item["rmin_half_angstrom"]) for item in atom_parameters]
            ),
            pairs,
            float(nonbonded["short_range_cutoff_angstrom"]),
            direction_xyz,
        )
        product += radial_product.reshape(-1)
    else:
        for left, right in pairs:
            left_parameter, right_parameter = (
                atom_parameters[left],
                atom_parameters[right],
            )
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            radial = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            ).derivatives(float(np.linalg.norm(xyz[left] - xyz[right])))
            delta = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(delta))
            unit = delta / distance
            ratio = (0.72 * r_min / distance) ** 8
            switch = 1.0 / (1.0 + ratio)
            switch_first = 8.0 * switch * (1.0 - switch) / distance
            switch_second = (
                8.0
                * switch
                * (1.0 - switch)
                * (8.0 * (1.0 - 2.0 * switch) - 1.0)
                / distance**2
            )
            first = switch * radial.first + switch_first * radial.energy
            second = (
                switch * radial.second
                + 2.0 * switch_first * radial.first
                + switch_second * radial.energy
            )
            block = (second - first / distance) * np.outer(unit, unit)
            block += (first / distance) * np.eye(3)
            pair_direction = direction_xyz[left] - direction_xyz[right]
            pair_product = BOHR_TO_ANGSTROM**2 * (block @ pair_direction)
            product[3 * left : 3 * left + 3] += pair_product
            product[3 * right : 3 * right + 3] -= pair_product
    return product


def _seed_nonbonded_pairs(
    coordinates_angstrom: np.ndarray,
    coordinates_bohr: np.ndarray,
    short_range_backend: str,
    cutoff_angstrom: float,
    neighbor_list: PersistentVerletNeighborList | None,
    *,
    native: bool,
) -> tuple[tuple[int, int], ...] | np.ndarray:
    if short_range_backend == "spatial_neighbor_list":
        if neighbor_list is not None:
            if native:
                return neighbor_list.native_candidates(coordinates_bohr)
            return neighbor_list.pairs(coordinates_bohr)
        pairs = build_nonbonded_neighbor_list(
            coordinates_bohr,
            cutoff_bohr=float(cutoff_angstrom) / BOHR_TO_ANGSTROM,
        )
    else:
        left, right = np.triu_indices(len(coordinates_angstrom), k=1)
        if native:
            return np.column_stack((left, right)).astype(np.intp, copy=False)
        return tuple(zip(left.tolist(), right.tolist(), strict=True))
    if native:
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    return pairs


def _seed_morse_bond_arrays(
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = tuple(model["bonds"])
    arrays = (
        np.asarray(
            [tuple(int(value) for value in raw["atoms"]) for raw in records],
            dtype=np.intp,
        ).reshape(-1, 2),
        np.asarray(
            [float(raw["d_e_hartree"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["alpha_per_angstrom"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["r0_angstrom"]) for raw in records],
            dtype=float,
        ),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _seed_local_valence_arrays(
    model: Mapping[str, Any],
) -> tuple[np.ndarray, ...]:
    """Pack angle/torsion records once for the portable native runtime."""

    angle_records = tuple(model["angles"])
    torsion_records = tuple(model["torsions"])
    angle_atoms = np.asarray(
        [tuple(int(value) for value in raw["atoms"]) for raw in angle_records],
        dtype=np.intp,
    ).reshape(-1, 3)
    angle_parameters = np.asarray(
        [
            (
                2.0 * float(raw["k_hartree_per_radian2"]),
                float(raw["theta0_radian"]),
                float(raw["radial_factors"][0]["reference_distance_angstrom"]),
                float(raw["radial_factors"][0]["covalent_radius_sum_angstrom"]),
                float(raw["radial_factors"][1]["reference_distance_angstrom"]),
                float(raw["radial_factors"][1]["covalent_radius_sum_angstrom"]),
            )
            for raw in angle_records
        ],
        dtype=float,
    ).reshape(-1, 6)
    torsion_atoms = np.asarray(
        [tuple(int(value) for value in raw["atoms"]) for raw in torsion_records],
        dtype=np.intp,
    ).reshape(-1, 4)
    torsion_parameters = np.asarray(
        [
            tuple(
                value
                for factor in raw["radial_factors"]
                for value in (
                    float(factor["reference_distance_angstrom"]),
                    float(factor["covalent_radius_sum_angstrom"]),
                )
            )
            for raw in torsion_records
        ],
        dtype=float,
    ).reshape(-1, 6)
    offsets = np.zeros(len(torsion_records) + 1, dtype=np.intp)
    for index, raw in enumerate(torsion_records):
        offsets[index + 1] = offsets[index] + len(raw["terms"])
    terms = np.asarray(
        [
            (
                float(term["amplitude_hartree"]),
                float(term["periodicity"]),
                float(term["phase_radian"]),
            )
            for raw in torsion_records
            for term in raw["terms"]
        ],
        dtype=float,
    ).reshape(-1, 3)
    arrays = (
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        offsets,
        terms,
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _fast_zoom_selected(zoom_level: str | None) -> bool:
    if zoom_level is None:
        return False
    normalized = str(zoom_level).strip().upper().replace("-", "_")
    if normalized != ZAFF_FAST_RIGID_TORSIONAL_ZOOM:
        raise ValueError(f"unsupported transferable ZAFF zoom level: {zoom_level}")
    return True


def _internal_term_hessian_vector_product(
    primitives: Sequence[Primitive],
    gradient_internal: np.ndarray,
    hessian_internal: np.ndarray,
    coordinates_angstrom: np.ndarray,
    vector_bohr: np.ndarray,
) -> np.ndarray:
    """Apply ``B.T F B + sum(g_i B'_i)`` with analytic primitive B-prime."""

    b_matrix = primitive_b_matrix(primitives, coordinates_angstrom)
    direction_angstrom = (
        BOHR_TO_ANGSTROM * np.asarray(vector_bohr, dtype=float).reshape(-1)
    )
    product_angstrom = b_matrix.T @ (
        np.asarray(hessian_internal, dtype=float) @ (b_matrix @ direction_angstrom)
    )
    function_map = {"bond": "R", "angle": "A", "dihedral": "D"}
    for primitive, coefficient in zip(
        primitives, np.asarray(gradient_internal, dtype=float), strict=True
    ):
        proxy = SimpleNamespace(
            function=function_map[primitive.kind],
            atoms=tuple(int(atom) + 1 for atom in primitive.atoms),
            ref_atoms=(),
            frame_atoms=(),
            ref_frame_atoms=(),
            identifier=f"SEED_{primitive.kind}_{primitive.atoms}",
        )
        product_angstrom += float(
            coefficient
        ) * analytic_basic_primitive_hessian_vector(
            proxy,
            coordinates_angstrom,
            direction_angstrom,
        )
    return BOHR_TO_ANGSTROM * product_angstrom



__all__ = [
    "ZAFF_FAST_RIGID_TORSIONAL_ZOOM",
    "ZAFF_SEED_MODEL_SCHEMA",
    "ZaffSeedCoverageError",
    "ZaffSeedDynamicsRuntime",
    "ZaffSeedEvaluation",
    "evaluate_zaff_fast_rigid_torsional_batch_energies",
    "evaluate_zaff_seed_energy",
    "evaluate_zaff_seed_model",
]
