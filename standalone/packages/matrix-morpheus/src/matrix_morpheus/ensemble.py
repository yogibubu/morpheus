from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import tomllib

import numpy as np

from matrix_chem.topology.pipeline import build_topology_objects
from matrix_core.numerics import RankCondition, rank_condition

from .contracts import SemiexperimentalFitRequest
from .fit import (
    MeasurementModel,
    _active_mask,
    _atomic_number,
    _build_measurement_model,
    _combined_fixed_parameters,
    _fixed_primitives_from_patterns,
    _gic_fixed_patterns,
    _gic_model,
    _gic_values,
    _gicforge_a1_mask,
    _jacobian_constants_wrt_gics,
    _make_gicforge_backend,
    _measurement_vector,
    _request_with_auto_resolved_isotopologues,
    _symmetry_expanded_fixed_primitives,
    _validate_observations,
)
from .geometry_input import read_geometry_input
from .io import read_observations
from .job_input import read_semiexperimental_job
from .msr_legacy import is_msr_legacy_file, read_msr_legacy_input


ENSEMBLE_JOB_SCHEMA = "oracle.semiexp.ensemble.v1"
LEGACY_ENSEMBLE_JOB_SCHEMA = "merlino.semiexp.ensemble.v1"
SUPPORTED_ENSEMBLE_JOB_SCHEMAS = (ENSEMBLE_JOB_SCHEMA, LEGACY_ENSEMBLE_JOB_SCHEMA)


@dataclass(frozen=True)
class EnsembleMolecule:
    """One molecule in a multi-molecule class-correction fit."""

    name: str
    request: SemiexperimentalFitRequest


@dataclass(frozen=True)
class EnsembleClassCorrection:
    """A global correction shared across matched GICs in all molecules.

    The correction is applied to the generated coordinate displacement relative
    to each molecule's computational reference geometry. Matching uses the same
    substring policy as ordinary SEfit parameter classes.
    """

    name: str
    patterns: tuple[str, ...] = ()
    kind: str = ""
    atom_symbols: tuple[str, ...] = ()
    value_min: float | None = None
    value_max: float | None = None
    synthon_signatures: tuple[str, ...] = ()
    synthon_zeff: tuple[float, ...] = ()
    synthon_threshold: float | None = None
    prior_value: float | None = None
    prior_sigma: float | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Ensemble class correction name cannot be empty")
        if not self.patterns and not self.kind and not self.atom_symbols:
            raise ValueError(
                f"Ensemble class correction {self.name!r} needs patterns, kind, atom_symbols, or a combination"
            )
        if any(not str(pattern).strip() for pattern in self.patterns):
            raise ValueError(f"Ensemble class correction {self.name!r} contains an empty pattern")
        normalized_kind = _normalize_kind(self.kind)
        if self.kind and not normalized_kind:
            raise ValueError(f"Unknown ensemble class coordinate kind: {self.kind}")
        if self.atom_symbols and len(self.atom_symbols) < 2 and normalized_kind != "out_of_plane":
            raise ValueError(
                f"Ensemble class correction {self.name!r} needs at least two atom symbols"
            )
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min > self.value_max
        ):
            raise ValueError(
                f"Ensemble class correction {self.name!r} has value_min larger than value_max"
            )
        if (
            self.value_min is not None or self.value_max is not None
        ) and normalized_kind != "stretch":
            raise ValueError(
                f"Ensemble class correction {self.name!r} value ranges are currently supported for stretches"
            )
        if self.synthon_threshold is not None and self.synthon_threshold < 0.0:
            raise ValueError(
                f"Ensemble class correction {self.name!r} synthon_threshold must be non-negative"
            )
        if self.synthon_zeff and self.synthon_threshold is None:
            raise ValueError(
                f"Ensemble class correction {self.name!r} needs synthon_threshold with synthon_zeff"
            )
        if (self.prior_value is None) != (self.prior_sigma is None):
            raise ValueError(
                f"Ensemble class correction {self.name!r} needs both prior_value and prior_sigma"
            )
        if self.prior_sigma is not None and self.prior_sigma <= 0.0:
            raise ValueError(
                f"Ensemble class correction {self.name!r} prior_sigma must be positive"
            )


@dataclass(frozen=True)
class EnsembleMoleculeBlock:
    molecule: str
    labels: tuple[str, ...]
    matched_counts: dict[str, int]
    n_rows: int
    n_active_gics: int
    residual_before: float
    residual_after: float
    rank_contribution: int


@dataclass(frozen=True)
class EnsembleNumericalDiagnostics:
    rank: int
    n_columns: int
    n_rows: int
    residual_degrees_of_freedom: int
    condition_number: float
    singular_values: tuple[float, ...]
    column_norms: tuple[float, ...]
    warnings: tuple[str, ...] = ()
    high_correlation_pairs: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class EnsembleAcceptancePolicy:
    require_full_rank: bool = True
    max_condition_number: float = 1.0e8
    min_residual_degrees_of_freedom: int = 1
    min_molecule_support: int = 2
    high_correlation_review_threshold: float = 0.98
    high_correlation_reject_threshold: float = 0.9999

    def validate(self) -> None:
        if self.max_condition_number <= 0.0 or not np.isfinite(self.max_condition_number):
            raise ValueError("Ensemble acceptance max_condition_number must be positive and finite")
        if self.min_residual_degrees_of_freedom < 0:
            raise ValueError(
                "Ensemble acceptance min_residual_degrees_of_freedom must be non-negative"
            )
        if self.min_molecule_support < 1:
            raise ValueError("Ensemble acceptance min_molecule_support must be at least one")
        if not 0.0 <= self.high_correlation_review_threshold <= 1.0:
            raise ValueError(
                "Ensemble acceptance high_correlation_review_threshold must be between 0 and 1"
            )
        if not 0.0 <= self.high_correlation_reject_threshold <= 1.0:
            raise ValueError(
                "Ensemble acceptance high_correlation_reject_threshold must be between 0 and 1"
            )
        if self.high_correlation_reject_threshold < self.high_correlation_review_threshold:
            raise ValueError(
                "Ensemble acceptance reject correlation threshold must be >= review threshold"
            )


@dataclass(frozen=True)
class EnsembleAcceptanceDecision:
    status: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    review_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnsembleClassCorrectionFit:
    classes: tuple[EnsembleClassCorrection, ...]
    corrections: dict[str, float]
    sigma: dict[str, float]
    covariance: np.ndarray
    correlation: np.ndarray
    residual_before: np.ndarray
    residual_after: np.ndarray
    prior_residual_after: dict[str, float]
    weighted_rms_before: float
    weighted_rms_after: float
    rank: int
    condition_number: float
    molecule_blocks: tuple[EnsembleMoleculeBlock, ...]
    acceptance_policy: EnsembleAcceptancePolicy = field(default_factory=EnsembleAcceptancePolicy)
    diagnostics: EnsembleNumericalDiagnostics = field(
        default_factory=lambda: EnsembleNumericalDiagnostics(0, 0, 0, 0, float("inf"), (), ())
    )
    acceptance: EnsembleAcceptanceDecision = field(
        default_factory=lambda: EnsembleAcceptanceDecision(
            "rejected", False, ("not evaluated",), ()
        )
    )

    @property
    def text(self) -> str:
        lines = [
            "Multi-molecule class-correction SE refinement",
            f"molecules: {len(self.molecule_blocks)}",
            f"classes: {len(self.classes)}",
            f"rank: {self.rank}",
            f"residual degrees of freedom: {self.diagnostics.residual_degrees_of_freedom}",
            f"scaled condition number: {self.condition_number:.8g}",
            f"weighted RMS before: {self.weighted_rms_before:.8g}",
            f"weighted RMS after: {self.weighted_rms_after:.8g}",
            f"acceptance: {self.acceptance.status}",
        ]
        if self.acceptance.reasons:
            lines.extend(["", "Acceptance failures:"])
            for reason in self.acceptance.reasons:
                lines.append(f"  {reason}")
        if self.acceptance.review_items:
            lines.extend(["", "Acceptance review items:"])
            for item in self.acceptance.review_items:
                lines.append(f"  {item}")
        if self.diagnostics.warnings:
            lines.extend(["", "Numerical diagnostics:"])
            for warning in self.diagnostics.warnings:
                lines.append(f"  warning: {warning}")
        if self.diagnostics.high_correlation_pairs:
            lines.extend(["", "Highly correlated class corrections:"])
            for left, right, value in self.diagnostics.high_correlation_pairs:
                lines.append(f"  {left} / {right}: {value: .6f}")
        if self.prior_residual_after:
            lines.extend(["", "Soft prior residuals:"])
            for name, value in self.prior_residual_after.items():
                lines.append(f"  {name}: {value: .10g}")
        lines.extend(["", "Class corrections:"])
        for item in self.classes:
            lines.append(
                f"  {item.name}: {self.corrections[item.name]: .10g} "
                f"+/- {self.sigma[item.name]:.4g}"
            )
        lines.extend(["", "Molecule blocks:"])
        for block in self.molecule_blocks:
            counts = ", ".join(f"{name}={count}" for name, count in block.matched_counts.items())
            lines.append(
                f"  {block.molecule}: rows={block.n_rows} active_gics={block.n_active_gics} "
                f"rank={block.rank_contribution} rms {block.residual_before:.6g}->{block.residual_after:.6g} "
                f"matches=[{counts}]"
            )
        return "\n".join(lines)


def fit_ensemble_class_corrections(
    molecules: tuple[EnsembleMolecule, ...] | list[EnsembleMolecule],
    classes: tuple[EnsembleClassCorrection, ...] | list[EnsembleClassCorrection],
    *,
    step: float = 1.0e-4,
    rcond: float = 1.0e-10,
    acceptance_policy: EnsembleAcceptancePolicy | None = None,
    outdir: Path | None = None,
) -> EnsembleClassCorrectionFit:
    """Fit transferable coordinate corrections from several molecules.

    This is a linearized ensemble layer around the computational reference
    geometries.  Each molecule keeps its own topology, symmetry, isotopologue
    table and coordinate model.  The global variables are class corrections
    shared across matched totally symmetric GICs:

    ``q_m = q_m(QC) + T_m Delta``

    where ``Delta`` is common to all molecules for a given class.  The returned
    fit is intended as the robust first stage for large parent-only or
    isotope-poor homologous series; ordinary single-molecule SEfit remains the
    nonlinear engine for fully determined cases.
    """

    molecule_items = tuple(molecules)
    class_items = tuple(classes)
    if not molecule_items:
        raise ValueError("Ensemble fit needs at least one molecule")
    if not class_items:
        raise ValueError("Ensemble fit needs at least one shared class correction")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Ensemble finite-difference step must be positive and finite")
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("Ensemble rcond must be non-negative and finite")
    policy = acceptance_policy or EnsembleAcceptancePolicy()
    policy.validate()
    for item in class_items:
        item.validate()
    class_names = tuple(item.name for item in class_items)
    if len(set(class_names)) != len(class_names):
        raise ValueError("Ensemble class correction names must be unique")

    row_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    weights_blocks: list[np.ndarray] = []
    molecule_blocks: list[EnsembleMoleculeBlock] = []
    total_matches = {item.name: 0 for item in class_items}

    with TemporaryDirectory(prefix="matrix_ensemble_gicforge_") as tmp:
        root = Path(tmp)
        for molecule in molecule_items:
            block = _ensemble_molecule_design(molecule, class_items, step=step, root=root)
            for name, count in block.matched_counts.items():
                total_matches[name] += int(count)
            row_blocks.append(block.design)
            weights_blocks.append(block.measurement.weights)
            residual_blocks.append(block.residual)
            molecule_blocks.append(
                EnsembleMoleculeBlock(
                    molecule=molecule.name,
                    labels=block.labels,
                    matched_counts=block.matched_counts,
                    n_rows=len(block.residual),
                    n_active_gics=int(np.count_nonzero(block.active_mask)),
                    residual_before=_weighted_rms(block.residual, block.measurement.weights),
                    residual_after=0.0,
                    rank_contribution=rank_condition(
                        block.design * np.sqrt(block.measurement.weights)[:, None]
                    ).rank,
                )
            )

    design = np.vstack(row_blocks)
    residual = np.concatenate(residual_blocks)
    weights = np.concatenate(weights_blocks)
    missing = [name for name, count in total_matches.items() if count == 0]
    if missing:
        raise ValueError(
            "Ensemble class corrections did not match any active coordinate: " + ", ".join(missing)
        )
    prior_design, prior_residual, prior_weights, prior_names = _prior_equations(class_items)
    fit_design = np.vstack([design, prior_design]) if len(prior_residual) else design
    fit_residual = np.concatenate([residual, prior_residual]) if len(prior_residual) else residual
    fit_weights = np.concatenate([weights, prior_weights]) if len(prior_residual) else weights
    weighted_design = fit_design * np.sqrt(fit_weights)[:, None]
    weighted_residual = fit_residual * np.sqrt(fit_weights)
    if not np.any(np.abs(weighted_design) > 0.0):
        raise ValueError(
            "No ensemble class correction matched an active coordinate with non-zero sensitivity"
        )

    solution, covariance, conditioning, column_norms = _solve_scaled_weighted_lstsq(
        weighted_design,
        weighted_residual,
        class_names,
        rcond=rcond,
    )
    residual_after = residual - design @ solution
    prior_after = (
        prior_residual - prior_design @ solution
        if len(prior_residual)
        else np.zeros(0, dtype=float)
    )
    sigma_values = (
        np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if covariance.size
        else np.zeros(len(class_items))
    )
    correlation = _correlation_from_covariance(covariance)
    corrections = {name: float(value) for name, value in zip(class_names, solution)}
    sigma = {name: float(value) for name, value in zip(class_names, sigma_values)}

    updated_blocks: list[EnsembleMoleculeBlock] = []
    offset = 0
    for block in molecule_blocks:
        n = block.n_rows
        after = residual_after[offset : offset + n]
        w = weights[offset : offset + n]
        updated_blocks.append(
            EnsembleMoleculeBlock(
                molecule=block.molecule,
                labels=block.labels,
                matched_counts=block.matched_counts,
                n_rows=block.n_rows,
                n_active_gics=block.n_active_gics,
                residual_before=block.residual_before,
                residual_after=_weighted_rms(after, w),
                rank_contribution=block.rank_contribution,
            )
        )
        offset += n

    diagnostics = _ensemble_numerical_diagnostics(
        class_items,
        updated_blocks,
        conditioning,
        column_norms,
        correlation,
        n_rows=int(fit_design.shape[0]),
        n_columns=int(fit_design.shape[1]),
        high_correlation_threshold=policy.high_correlation_review_threshold,
    )
    acceptance = _evaluate_ensemble_acceptance(class_items, updated_blocks, diagnostics, policy)

    result = EnsembleClassCorrectionFit(
        classes=class_items,
        corrections=corrections,
        sigma=sigma,
        covariance=covariance,
        correlation=correlation,
        residual_before=residual,
        residual_after=residual_after,
        prior_residual_after={name: float(value) for name, value in zip(prior_names, prior_after)},
        weighted_rms_before=_weighted_rms(residual, weights),
        weighted_rms_after=_weighted_rms(residual_after, weights),
        rank=conditioning.rank,
        condition_number=conditioning.condition_number,
        molecule_blocks=tuple(updated_blocks),
        acceptance_policy=policy,
        diagnostics=diagnostics,
        acceptance=acceptance,
    )
    if outdir is not None:
        write_ensemble_class_correction_outputs(Path(outdir), result)
    return result


@dataclass(frozen=True)
class EnsembleJobInput:
    path: Path
    title: str
    molecules: tuple[EnsembleMolecule, ...]
    classes: tuple[EnsembleClassCorrection, ...]
    step: float = 1.0e-4
    rcond: float = 1.0e-10
    acceptance_policy: EnsembleAcceptancePolicy = field(default_factory=EnsembleAcceptancePolicy)


def read_ensemble_job(path: Path) -> EnsembleJobInput:
    target = Path(path)
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    if data.get("schema") not in SUPPORTED_ENSEMBLE_JOB_SCHEMAS:
        supported = ", ".join(repr(item) for item in SUPPORTED_ENSEMBLE_JOB_SCHEMAS)
        raise ValueError(f"Ensemble job must declare one of: {supported}")
    title = str(data.get("title") or target.stem)
    fit = _mapping(data.get("fit", {}), "fit")
    acceptance = _mapping(data.get("acceptance", {}), "acceptance")
    molecules = tuple(_ensemble_molecules_from_mapping(target, data.get("molecules", ())))
    classes = tuple(_ensemble_classes_from_mapping(data.get("classes", ())))
    step = float(fit.get("step", 1.0e-4))
    rcond = float(fit.get("rcond", 1.0e-10))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Ensemble job fit.step must be positive and finite")
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("Ensemble job fit.rcond must be non-negative and finite")
    return EnsembleJobInput(
        path=target,
        title=title,
        molecules=molecules,
        classes=classes,
        step=step,
        rcond=rcond,
        acceptance_policy=_acceptance_policy_from_mapping(acceptance),
    )


def fit_ensemble_job(path: Path, *, outdir: Path | None = None) -> EnsembleClassCorrectionFit:
    job = read_ensemble_job(path)
    return fit_ensemble_class_corrections(
        job.molecules,
        job.classes,
        step=job.step,
        rcond=job.rcond,
        acceptance_policy=job.acceptance_policy,
        outdir=outdir,
    )


@dataclass(frozen=True)
class _MoleculeDesign:
    design: np.ndarray
    residual: np.ndarray
    measurement: MeasurementModel
    labels: tuple[str, ...]
    active_mask: np.ndarray
    matched_counts: dict[str, int]


def _ensemble_molecule_design(
    molecule: EnsembleMolecule,
    classes: tuple[EnsembleClassCorrection, ...],
    *,
    step: float,
    root: Path,
) -> _MoleculeDesign:
    request = molecule.request
    request.validate()
    geometry_input = read_geometry_input(Path(request.initial_geometry))
    atoms = tuple(geometry_input.atoms)
    coords = np.asarray(geometry_input.coordinates_angstrom, dtype=float)
    request, _warnings = _request_with_auto_resolved_isotopologues(request, atoms, coords)
    _validate_observations(request.observations, len(atoms))
    z_numbers = np.array([_atomic_number(symbol) for symbol in atoms], dtype=int)
    synthon_signatures = _synthon_signatures(coords, z_numbers)
    synthon_zeff = _synthon_zeff(coords, z_numbers)
    backend = _make_gicforge_backend(atoms, root / _safe_name(molecule.name))
    prims, u_matrix, labels = _gic_model(coords, z_numbers, request, backend)
    fixed_parameters = _combined_fixed_parameters(
        request.fixed_parameters, geometry_input.fixed_parameters
    )
    fixed_gic_patterns = _gic_fixed_patterns(fixed_parameters)
    fixed_primitives = _symmetry_expanded_fixed_primitives(
        atoms,
        coords,
        prims,
        _fixed_primitives_from_patterns(fixed_parameters),
    )
    active = _active_mask(
        labels, fixed_gic_patterns, request.parameter_classes
    ) & _gicforge_a1_mask(labels)
    q_values = _gic_values(prims, u_matrix, coords)
    measurement = _build_measurement_model(request, atoms, coords, prims, u_matrix, labels)
    calculated = _measurement_vector(atoms, coords, request, q_values, labels, measurement)
    residual = measurement.observed - calculated
    jac_active = _jacobian_constants_wrt_gics(
        list(atoms),
        coords,
        request,
        prims,
        u_matrix,
        active,
        labels,
        measurement,
        step=step,
    )
    transform, matched_counts = _ensemble_class_transform(
        atoms, coords, labels, active, classes, synthon_signatures, synthon_zeff
    )
    design = jac_active @ transform
    if fixed_primitives:
        # Hard primitive constraints stay local to the molecule.  They remove
        # directions in ordinary SEfit; the ensemble layer currently reports
        # them but does not mix them into global class corrections.
        pass
    return _MoleculeDesign(design, residual, measurement, labels, active, matched_counts)


def _ensemble_class_transform(
    atoms: tuple[str, ...],
    coords: np.ndarray,
    labels: tuple[str, ...],
    active_mask: np.ndarray,
    classes: tuple[EnsembleClassCorrection, ...],
    synthon_signatures: tuple[str, ...] = (),
    synthon_zeff: tuple[float, ...] = (),
) -> tuple[np.ndarray, dict[str, int]]:
    active_indices = np.where(active_mask)[0]
    transform = np.zeros((len(active_indices), len(classes)), dtype=float)
    counts = {item.name: 0 for item in classes}
    kinds = {item.name: set() for item in classes}
    for row, idx in enumerate(active_indices):
        label = labels[int(idx)]
        for col, item in enumerate(classes):
            projection = _class_projection(
                item,
                label,
                atoms,
                coords=coords,
                synthon_signatures=synthon_signatures,
                synthon_zeff=synthon_zeff,
            )
            if abs(projection) > 0.0:
                transform[row, col] = projection
                counts[item.name] += 1
                kinds[item.name].add(_coordinate_kind(label))
    mixed = {name: sorted(value) for name, value in kinds.items() if len(value) > 1}
    if mixed:
        details = "; ".join(f"{name}: {','.join(value)}" for name, value in mixed.items())
        raise ValueError(f"Ensemble class corrections cannot mix coordinate types ({details})")
    return transform, counts


def _class_matches(item: EnsembleClassCorrection, label: str, atoms: tuple[str, ...]) -> bool:
    return abs(_class_projection(item, label, atoms)) > 0.0


def _class_projection(
    item: EnsembleClassCorrection,
    label: str,
    atoms: tuple[str, ...],
    *,
    coords: np.ndarray | None = None,
    synthon_signatures: tuple[str, ...] = (),
    synthon_zeff: tuple[float, ...] = (),
) -> float:
    low = label.lower()
    if item.patterns and not any(str(pattern).lower() in low for pattern in item.patterns):
        return 0.0
    wanted_kind = _normalize_kind(item.kind)
    terms = _primitive_terms(label)
    if wanted_kind:
        matching_kind = [term for term in terms if term[0] == wanted_kind]
        if not matching_kind:
            return 0.0
        if item.atom_symbols:
            wanted_atoms = _class_atom_signature(wanted_kind, item.atom_symbols)
            matching_kind = [
                term
                for term in matching_kind
                if _primitive_signature(term[0], term[1], atoms) == wanted_atoms
                and _term_value_in_range(item, term, coords)
                and _term_synthon_matches(item, term, synthon_signatures)
                and _term_synthon_zeff_matches(item, term, synthon_zeff)
            ]
            if not matching_kind:
                return 0.0
        else:
            matching_kind = [
                term
                for term in matching_kind
                if _term_value_in_range(item, term, coords)
                and _term_synthon_matches(item, term, synthon_signatures)
                and _term_synthon_zeff_matches(item, term, synthon_zeff)
            ]
            if not matching_kind:
                return 0.0
        return float(sum(term[2] for term in matching_kind))
    if item.atom_symbols:
        wanted_atoms_by_kind = {
            kind: _class_atom_signature(kind, item.atom_symbols)
            for kind in {term[0] for term in terms}
        }
        matching_atoms = [
            term
            for term in terms
            if _primitive_signature(term[0], term[1], atoms) == wanted_atoms_by_kind[term[0]]
            and _term_value_in_range(item, term, coords)
            and _term_synthon_matches(item, term, synthon_signatures)
            and _term_synthon_zeff_matches(item, term, synthon_zeff)
        ]
        return float(sum(term[2] for term in matching_atoms)) if matching_atoms else 0.0
    return 1.0 if item.patterns else 0.0


def _coordinate_kind(label: str) -> str:
    text = str(label).lower()
    if "str" in text or "r(" in text:
        return "stretch"
    if "lin" in text or "linear" in text:
        return "linear_bend"
    if "ang" in text or "a(" in text:
        return "bend"
    if "oop" in text or "out" in text or "oop(" in text:
        return "out_of_plane"
    if "dih" in text or "tor" in text or "d(" in text:
        return "torsion"
    return "unknown"


def _normalize_kind(kind: str) -> str:
    text = str(kind or "").strip().lower().replace("-", "_")
    aliases = {
        "": "",
        "r": "stretch",
        "bond": "stretch",
        "stretch": "stretch",
        "a": "bend",
        "angle": "bend",
        "bend": "bend",
        "l": "linear_bend",
        "linear": "linear_bend",
        "linear_bend": "linear_bend",
        "d": "torsion",
        "dihedral": "torsion",
        "torsion": "torsion",
        "u": "out_of_plane",
        "oop": "out_of_plane",
        "out_of_plane": "out_of_plane",
    }
    return aliases.get(text, "")


def _primitive_tokens(label: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((kind, indices) for kind, indices, _coeff in _primitive_terms(label))


def _primitive_terms(label: str) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    text = str(label)
    bracket = re.search(r"\[(.*)\]", text)
    source = bracket.group(1) if bracket else text
    terms: list[tuple[str, tuple[int, ...], float]] = []
    kind_map = {
        "R": "stretch",
        "A": "bend",
        "L": "linear_bend",
        "D": "torsion",
        "U": "out_of_plane",
        "O": "out_of_plane",
    }
    pattern = re.compile(
        r"([+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*\*\s*([RALDUO])\(\s*([0-9,\s]+?)\s*\)"
    )
    for match in pattern.finditer(source):
        kind = kind_map.get(match.group(2).upper())
        if not kind:
            continue
        indices = tuple(int(item) for item in re.findall(r"\d+", match.group(3)))
        if not indices:
            continue
        coeff = float(match.group(1).replace(" ", ""))
        terms.append((kind, indices, coeff))
    if terms:
        return tuple(terms)
    tokens: list[tuple[str, tuple[int, ...]]] = []
    for match in re.finditer(r"\b([RALDUO])\(\s*([0-9,\s]+?)\s*\)", str(label)):
        kind = kind_map.get(match.group(1).upper())
        if not kind:
            continue
        indices = tuple(int(item) for item in re.findall(r"\d+", match.group(2)))
        if indices:
            tokens.append((kind, indices))
    return tuple((kind, indices, 1.0) for kind, indices in tokens)


def _primitive_symbols(indices: tuple[int, ...], atoms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(_primitive_ordered_symbols(indices, atoms)))


def _term_value_in_range(
    item: EnsembleClassCorrection,
    term: tuple[str, tuple[int, ...], float],
    coords: np.ndarray | None,
) -> bool:
    if item.value_min is None and item.value_max is None:
        return True
    value = _primitive_value(term[0], term[1], coords)
    if value is None:
        return False
    if item.value_min is not None and value < item.value_min:
        return False
    if item.value_max is not None and value > item.value_max:
        return False
    return True


def _primitive_value(
    kind: str, indices: tuple[int, ...], coords: np.ndarray | None
) -> float | None:
    if coords is None or kind != "stretch" or len(indices) != 2:
        return None
    i, j = indices[0] - 1, indices[1] - 1
    if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
        return None
    return float(
        np.linalg.norm(np.asarray(coords[i], dtype=float) - np.asarray(coords[j], dtype=float))
    )


def _term_synthon_matches(
    item: EnsembleClassCorrection,
    term: tuple[str, tuple[int, ...], float],
    synthon_signatures: tuple[str, ...],
) -> bool:
    if not item.synthon_signatures:
        return True
    if not synthon_signatures:
        return False
    wanted = {str(value).strip() for value in item.synthon_signatures if str(value).strip()}
    if not wanted:
        return True
    for atom_index in _synthon_relevant_atoms(term[0], term[1]):
        if (
            1 <= atom_index <= len(synthon_signatures)
            and synthon_signatures[atom_index - 1] in wanted
        ):
            return True
    return False


def _synthon_relevant_atoms(kind: str, indices: tuple[int, ...]) -> tuple[int, ...]:
    if kind == "bend" and len(indices) == 3:
        return (indices[1],)
    if kind == "torsion" and len(indices) == 4:
        return (indices[1], indices[2])
    if kind == "out_of_plane" and len(indices) >= 1:
        return (indices[0],)
    return indices


def _term_synthon_zeff_matches(
    item: EnsembleClassCorrection,
    term: tuple[str, tuple[int, ...], float],
    synthon_zeff: tuple[float, ...],
) -> bool:
    if not item.synthon_zeff:
        return True
    if item.synthon_threshold is None or not synthon_zeff:
        return False
    observed = _primitive_synthon_zeff_signature(term[0], term[1], synthon_zeff)
    wanted = _class_synthon_zeff_signature(term[0], item.synthon_zeff)
    if not observed or len(observed) != len(wanted):
        return False
    return all(abs(float(a) - float(b)) <= item.synthon_threshold for a, b in zip(observed, wanted))


def _primitive_synthon_zeff_signature(
    kind: str,
    indices: tuple[int, ...],
    synthon_zeff: tuple[float, ...],
) -> tuple[float, ...]:
    values = []
    for index in indices:
        if index < 1 or index > len(synthon_zeff):
            return ()
        values.append(float(synthon_zeff[index - 1]))
    if kind == "bend" and len(values) == 3:
        ends = sorted((values[0], values[2]))
        return (ends[0], values[1], ends[1])
    if kind == "torsion" and len(values) == 4:
        return tuple(sorted((values[1], values[2])))
    if kind == "out_of_plane" and values:
        return (values[0],)
    if kind == "stretch":
        return tuple(sorted(values))
    return tuple(values)


def _class_synthon_zeff_signature(kind: str, values: tuple[float, ...]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if kind == "bend" and len(normalized) == 3:
        ends = sorted((normalized[0], normalized[2]))
        return (ends[0], normalized[1], ends[1])
    if kind == "torsion" and len(normalized) == 2:
        return tuple(sorted(normalized))
    if kind == "torsion" and len(normalized) == 4:
        return tuple(sorted((normalized[1], normalized[2])))
    if kind == "out_of_plane" and normalized:
        return (normalized[0],)
    if kind == "stretch":
        return tuple(sorted(normalized))
    return normalized


def _synthon_signatures(coords: np.ndarray, z_numbers: np.ndarray) -> tuple[str, ...]:
    try:
        _continuous, _graph, _ringset, synthons, _aromaticity = build_topology_objects(
            coords, z_numbers
        )
        return tuple(str(synthons.canonical_signature_str(i)) for i in range(len(z_numbers)))
    except Exception:
        return ()


def _synthon_zeff(coords: np.ndarray, z_numbers: np.ndarray) -> tuple[float, ...]:
    try:
        _continuous, _graph, _ringset, synthons, _aromaticity = build_topology_objects(
            coords, z_numbers
        )
        return tuple(float(synthons.Zeff(i)) for i in range(len(z_numbers)))
    except Exception:
        return ()


def _primitive_signature(
    kind: str, indices: tuple[int, ...], atoms: tuple[str, ...]
) -> tuple[str, ...]:
    symbols = _primitive_ordered_symbols(indices, atoms)
    if not symbols:
        return ()
    if kind == "bend" and len(symbols) == 3:
        ends = sorted((symbols[0], symbols[2]))
        return (ends[0], symbols[1], ends[1])
    if kind == "torsion" and len(symbols) == 4:
        return tuple(sorted((symbols[1], symbols[2])))
    if kind == "out_of_plane" and len(symbols) == 4:
        return (symbols[0],)
    if kind == "stretch":
        return tuple(sorted(symbols))
    return tuple(symbols)


def _class_atom_signature(kind: str, symbols: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_normalize_symbol(symbol) for symbol in symbols)
    if kind == "bend" and len(normalized) == 3:
        ends = sorted((normalized[0], normalized[2]))
        return (ends[0], normalized[1], ends[1])
    if kind == "torsion" and len(normalized) == 2:
        return tuple(sorted(normalized))
    if kind == "torsion" and len(normalized) == 4:
        return tuple(sorted((normalized[1], normalized[2])))
    if kind == "out_of_plane" and len(normalized) == 1:
        return normalized
    if kind == "out_of_plane" and len(normalized) == 4:
        return (normalized[0],)
    if kind == "stretch":
        return tuple(sorted(normalized))
    return normalized


def _primitive_ordered_symbols(indices: tuple[int, ...], atoms: tuple[str, ...]) -> tuple[str, ...]:
    symbols = []
    for index in indices:
        if index < 1 or index > len(atoms):
            return ()
        symbols.append(_normalize_symbol(atoms[index - 1]))
    return tuple(symbols)


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    return text[:1].upper() + text[1:].lower() if text else ""


def _ensemble_molecules_from_mapping(path: Path, items: object) -> list[EnsembleMolecule]:
    if not isinstance(items, list) or not items:
        raise ValueError("Ensemble job needs a non-empty [[molecules]] list")
    molecules: list[EnsembleMolecule] = []
    for item in items:
        data = _mapping(item, "molecules entry")
        name = str(data.get("name") or data.get("label") or "").strip()
        job_path = data.get("job")
        if not job_path:
            raise ValueError("Each ensemble molecule needs a job path")
        resolved_job = _resolve_relative(path, Path(str(job_path)))
        if is_msr_legacy_file(resolved_job):
            legacy = read_msr_legacy_input(resolved_job)
            if not legacy.observations:
                raise ValueError(
                    f"Ensemble molecule {name or resolved_job.stem!r} has no observations"
                )
            request = SemiexperimentalFitRequest(
                initial_geometry=legacy.path,
                observations=tuple(legacy.observations),
                fixed_parameters=legacy.geometry.fixed_parameters,
                coordinate_model="gic",
            )
            molecules.append(EnsembleMolecule(name or resolved_job.stem, request))
            continue
        job = read_semiexperimental_job(resolved_job)
        if job.observations_inline:
            observations = job.observations_inline
        elif job.observations is not None:
            observations = read_observations(job.observations)
        else:
            observations = ()
        if not observations:
            raise ValueError(f"Ensemble molecule {name or job.title!r} has no observations")
        request = SemiexperimentalFitRequest(
            initial_geometry=job.path,
            observations=tuple(observations),
            fixed_parameters=job.fixed_parameters,
            observable=job.observable,
            rotational_components=job.rotational_components,
            qm_predicates=job.qm_predicates,
            parameter_classes=job.parameter_classes,
            coordinate_model=job.coordinate_model,
            robust_loss=job.robust_loss,
            robust_scale=job.robust_scale,
            leave_one_out=job.leave_one_out,
        )
        molecules.append(EnsembleMolecule(name or job.title, request))
    return molecules


def _ensemble_classes_from_mapping(items: object) -> list[EnsembleClassCorrection]:
    if not isinstance(items, list) or not items:
        raise ValueError("Ensemble job needs a non-empty [[classes]] list")
    classes: list[EnsembleClassCorrection] = []
    for item in items:
        data = _mapping(item, "classes entry")
        patterns = data.get("patterns", ())
        if isinstance(patterns, str):
            patterns = [part.strip() for part in re.split(r"[|,;]", patterns) if part.strip()]
        atoms = data.get("atoms", data.get("atom_symbols", ()))
        if isinstance(atoms, str):
            atoms = [part.strip() for part in re.split(r"[|,;]", atoms) if part.strip()]
        classes.append(
            EnsembleClassCorrection(
                name=str(data["name"]),
                patterns=tuple(str(pattern).strip() for pattern in patterns),
                kind=str(data.get("kind", "")),
                atom_symbols=tuple(str(atom).strip() for atom in atoms),
                value_min=_optional_float(data.get("value_min", data.get("min_value"))),
                value_max=_optional_float(data.get("value_max", data.get("max_value"))),
                synthon_signatures=_string_tuple(
                    data.get("synthon_signatures", data.get("synthons", data.get("synthon")))
                ),
                synthon_zeff=_float_tuple(data.get("synthon_zeff", data.get("zeff"))),
                synthon_threshold=_optional_float(
                    data.get("synthon_threshold", data.get("zeff_threshold"))
                ),
                prior_value=_optional_float(data.get("prior_value", data.get("prior"))),
                prior_sigma=_optional_float(data.get("prior_sigma", data.get("sigma"))),
            )
        )
    return classes


def _acceptance_policy_from_mapping(data: dict) -> EnsembleAcceptancePolicy:
    policy = EnsembleAcceptancePolicy(
        require_full_rank=_optional_bool(data.get("require_full_rank"), True),
        max_condition_number=float(data.get("max_condition_number", 1.0e8)),
        min_residual_degrees_of_freedom=int(data.get("min_residual_degrees_of_freedom", 1)),
        min_molecule_support=int(data.get("min_molecule_support", 2)),
        high_correlation_review_threshold=float(
            data.get("high_correlation_review_threshold", 0.98)
        ),
        high_correlation_reject_threshold=float(
            data.get("high_correlation_reject_threshold", 0.9999)
        ),
    )
    policy.validate()
    return policy


def _prior_equations(
    classes: tuple[EnsembleClassCorrection, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    residuals: list[float] = []
    weights: list[float] = []
    names: list[str] = []
    for col, item in enumerate(classes):
        if item.prior_value is None or item.prior_sigma is None:
            continue
        row = np.zeros(len(classes), dtype=float)
        row[col] = 1.0
        rows.append(row)
        residuals.append(float(item.prior_value))
        weights.append(1.0 / (float(item.prior_sigma) ** 2))
        names.append(item.name)
    if not rows:
        return (
            np.zeros((0, len(classes)), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            (),
        )
    return (
        np.vstack(rows),
        np.asarray(residuals, dtype=float),
        np.asarray(weights, dtype=float),
        tuple(names),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in re.split(r"[|,;]", value) if part.strip())
    try:
        return tuple(str(part).strip() for part in value if str(part).strip())
    except TypeError:
        return ()


def _float_tuple(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in re.split(r"[|,;]", value) if part.strip())
    try:
        return tuple(float(part) for part in value)
    except TypeError:
        return ()


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Ensemble job {name} must be a table")
    return value


def _resolve_relative(base_file: Path, target: Path) -> Path:
    return target if target.is_absolute() else base_file.parent / target


def _solve_scaled_weighted_lstsq(
    weighted_design: np.ndarray,
    weighted_residual: np.ndarray,
    class_names: tuple[str, ...],
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    column_norms = np.linalg.norm(weighted_design, axis=0)
    zero = [name for name, norm in zip(class_names, column_norms) if norm <= 0.0]
    if zero:
        raise ValueError(
            "Ensemble class corrections have zero weighted sensitivity: " + ", ".join(zero)
        )
    scaled_design = weighted_design / column_norms[None, :]
    u, s, vt = np.linalg.svd(scaled_design, full_matrices=False)
    cutoff = _svd_cutoff(s, scaled_design.shape, rcond=rcond)
    keep = s > cutoff
    if not np.any(keep):
        raise ValueError(
            "Ensemble design matrix has no numerically significant singular directions"
        )
    scaled_solution = vt[keep, :].T @ ((u[:, keep].T @ weighted_residual) / s[keep])
    solution = scaled_solution / column_norms
    scaled_covariance = _covariance_from_svd(s, vt, keep)
    covariance = scaled_covariance / np.outer(column_norms, column_norms)
    return (
        solution,
        covariance,
        _rank_condition_from_svd(s, scaled_design.shape, rcond=rcond),
        column_norms,
    )


def _covariance_from_weighted_design(weighted_design: np.ndarray, *, rcond: float) -> np.ndarray:
    if weighted_design.size == 0:
        return np.zeros((0, 0), dtype=float)
    _u, s, vt = np.linalg.svd(weighted_design, full_matrices=False)
    if not len(s):
        return np.zeros((weighted_design.shape[1], weighted_design.shape[1]), dtype=float)
    cutoff = _svd_cutoff(s, weighted_design.shape, rcond=rcond)
    inv_s2 = np.array(
        [1.0 / (value * value) if value > cutoff else 0.0 for value in s], dtype=float
    )
    return (vt.T * inv_s2) @ vt


def _covariance_from_svd(singular: np.ndarray, vt: np.ndarray, keep: np.ndarray) -> np.ndarray:
    inv_s2 = np.zeros_like(singular, dtype=float)
    inv_s2[keep] = 1.0 / (singular[keep] * singular[keep])
    return (vt.T * inv_s2) @ vt


def _svd_cutoff(singular: np.ndarray, shape: tuple[int, int], *, rcond: float) -> float:
    if singular.size == 0:
        return 0.0
    relative = max(float(rcond), max(shape) * np.finfo(float).eps)
    return relative * float(singular[0])


def _rank_condition_from_svd(
    singular: np.ndarray, shape: tuple[int, int], *, rcond: float
) -> RankCondition:
    if singular.size == 0:
        return RankCondition(rank=0, condition_number=float("inf"), singular_values=np.array(()))
    cutoff = _svd_cutoff(singular, shape, rcond=rcond)
    rank = int(np.sum(singular > cutoff))
    if rank == len(singular) and singular[-1] > cutoff:
        condition = float(singular[0] / singular[-1])
    else:
        condition = float("inf")
    return RankCondition(
        rank=rank, condition_number=condition, singular_values=np.asarray(singular, dtype=float)
    )


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    if covariance.size == 0:
        return covariance
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denom = np.outer(sigma, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0.0)
    return corr


def _ensemble_numerical_diagnostics(
    classes: tuple[EnsembleClassCorrection, ...],
    blocks: list[EnsembleMoleculeBlock],
    conditioning: object,
    column_norms: np.ndarray,
    correlation: np.ndarray,
    *,
    n_rows: int,
    n_columns: int,
    high_correlation_threshold: float = 0.98,
) -> EnsembleNumericalDiagnostics:
    warnings: list[str] = []
    if conditioning.rank < n_columns:
        warnings.append(f"rank deficient global class model ({conditioning.rank}/{n_columns})")
    if not np.isfinite(conditioning.condition_number):
        warnings.append("singular scaled design matrix")
    elif conditioning.condition_number > 1.0e8:
        warnings.append(
            f"ill-conditioned scaled design matrix ({conditioning.condition_number:.3g})"
        )
    if n_rows <= conditioning.rank:
        warnings.append("no residual degrees of freedom after numerical rank determination")

    molecule_total = len(blocks)
    for item in classes:
        matched = sum(block.matched_counts.get(item.name, 0) for block in blocks)
        molecule_count = sum(1 for block in blocks if block.matched_counts.get(item.name, 0) > 0)
        if matched == 0:
            warnings.append(f"class {item.name} has no matched active coordinates")
        elif molecule_total > 1 and molecule_count < 2:
            warnings.append(f"class {item.name} is supported by only {molecule_count} molecule")

    high_pairs = _high_correlation_pairs(classes, correlation, threshold=high_correlation_threshold)
    if high_pairs:
        warnings.append(f"{len(high_pairs)} class-correction pair(s) have |correlation| >= 0.98")

    return EnsembleNumericalDiagnostics(
        rank=int(conditioning.rank),
        n_columns=int(n_columns),
        n_rows=int(n_rows),
        residual_degrees_of_freedom=max(int(n_rows) - int(conditioning.rank), 0),
        condition_number=float(conditioning.condition_number),
        singular_values=tuple(float(value) for value in conditioning.singular_values),
        column_norms=tuple(float(value) for value in column_norms),
        warnings=tuple(warnings),
        high_correlation_pairs=high_pairs,
    )


def _evaluate_ensemble_acceptance(
    classes: tuple[EnsembleClassCorrection, ...],
    blocks: list[EnsembleMoleculeBlock],
    diagnostics: EnsembleNumericalDiagnostics,
    policy: EnsembleAcceptancePolicy,
) -> EnsembleAcceptanceDecision:
    failures: list[str] = []
    review: list[str] = []
    if policy.require_full_rank and diagnostics.rank < diagnostics.n_columns:
        failures.append(
            f"global class model is rank deficient ({diagnostics.rank}/{diagnostics.n_columns})"
        )
    if not np.isfinite(diagnostics.condition_number):
        failures.append("scaled design matrix is singular")
    elif diagnostics.condition_number > policy.max_condition_number:
        failures.append(
            f"scaled condition number {diagnostics.condition_number:.6g} exceeds {policy.max_condition_number:.6g}"
        )
    if diagnostics.residual_degrees_of_freedom < policy.min_residual_degrees_of_freedom:
        failures.append(
            "residual degrees of freedom "
            f"{diagnostics.residual_degrees_of_freedom} below {policy.min_residual_degrees_of_freedom}"
        )

    molecule_total = len(blocks)
    required_support = min(policy.min_molecule_support, molecule_total)
    for item in classes:
        molecule_count = sum(1 for block in blocks if block.matched_counts.get(item.name, 0) > 0)
        if molecule_count < required_support:
            failures.append(
                f"class {item.name} is supported by {molecule_count} molecule(s), below {required_support}"
            )

    for left, right, value in diagnostics.high_correlation_pairs:
        message = f"class corrections {left}/{right} have correlation {value:.6f}"
        if abs(value) >= policy.high_correlation_reject_threshold:
            failures.append(message)
        else:
            review.append(message)

    if failures:
        return EnsembleAcceptanceDecision("rejected", False, tuple(failures), tuple(review))
    if review:
        return EnsembleAcceptanceDecision("review", True, (), tuple(review))
    return EnsembleAcceptanceDecision("accepted", True, (), ())


def _high_correlation_pairs(
    classes: tuple[EnsembleClassCorrection, ...],
    correlation: np.ndarray,
    *,
    threshold: float = 0.98,
) -> tuple[tuple[str, str, float], ...]:
    if correlation.size == 0:
        return ()
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            value = float(correlation[i, j])
            if abs(value) >= threshold:
                pairs.append((classes[i].name, classes[j].name, value))
    return tuple(sorted(pairs, key=lambda item: abs(item[2]), reverse=True))


def _weighted_rms(residual: np.ndarray, weights: np.ndarray) -> float:
    if residual.size == 0:
        return 0.0
    weighted = np.asarray(residual, dtype=float) * np.sqrt(np.asarray(weights, dtype=float))
    return float(np.sqrt(np.mean(weighted * weighted)))


def _safe_name(name: str) -> str:
    text = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(name).strip())
    return text or "molecule"


def write_ensemble_class_correction_outputs(
    outdir: Path, result: EnsembleClassCorrectionFit
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ensemble_class_corrections.txt").write_text(result.text + "\n", encoding="utf-8")
    (outdir / "ensemble_class_corrections.csv").write_text(_class_csv(result), encoding="utf-8")
    (outdir / "ensemble_class_report.csv").write_text(_class_report_csv(result), encoding="utf-8")
    (outdir / "ensemble_molecule_blocks.csv").write_text(_blocks_csv(result), encoding="utf-8")
    (outdir / "ensemble_manifest.json").write_text(_manifest_json(result), encoding="utf-8")
    np.savetxt(outdir / "ensemble_covariance.csv", result.covariance, delimiter=",")
    np.savetxt(outdir / "ensemble_correlation.csv", result.correlation, delimiter=",")


def _class_csv(result: EnsembleClassCorrectionFit) -> str:
    lines = [
        "class,kind,atom_symbols,patterns,value_min,value_max,synthon_signatures,synthon_zeff,"
        "synthon_threshold,prior_value,prior_sigma,prior_residual,correction,sigma,acceptance_status"
    ]
    for item in result.classes:
        atoms = "|".join(item.atom_symbols)
        patterns = "|".join(item.patterns)
        prior_value = "" if item.prior_value is None else f"{item.prior_value:.12g}"
        prior_sigma = "" if item.prior_sigma is None else f"{item.prior_sigma:.12g}"
        prior_residual = result.prior_residual_after.get(item.name)
        prior_residual_text = "" if prior_residual is None else f"{prior_residual:.12g}"
        lines.append(
            f"{item.name},{_normalize_kind(item.kind)},{atoms},{patterns},"
            f"{'' if item.value_min is None else f'{item.value_min:.12g}'},"
            f"{'' if item.value_max is None else f'{item.value_max:.12g}'},"
            f"{'|'.join(item.synthon_signatures)},"
            f"{'|'.join(f'{value:.12g}' for value in item.synthon_zeff)},"
            f"{'' if item.synthon_threshold is None else f'{item.synthon_threshold:.12g}'},"
            f"{prior_value},{prior_sigma},{prior_residual_text},"
            f"{result.corrections[item.name]:.12g},{result.sigma[item.name]:.12g},{result.acceptance.status}"
        )
    return "\n".join(lines) + "\n"


def _blocks_csv(result: EnsembleClassCorrectionFit) -> str:
    lines = [
        "molecule,n_rows,n_active_gics,rank_contribution,weighted_rms_before,weighted_rms_after,matched_counts"
    ]
    for block in result.molecule_blocks:
        counts = ";".join(f"{name}:{count}" for name, count in block.matched_counts.items())
        lines.append(
            f"{block.molecule},{block.n_rows},{block.n_active_gics},{block.rank_contribution},"
            f"{block.residual_before:.12g},{block.residual_after:.12g},{counts}"
        )
    return "\n".join(lines) + "\n"


def _class_report_csv(result: EnsembleClassCorrectionFit) -> str:
    lines = [
        "class,kind,atom_symbols,value_min,value_max,synthon_signatures,synthon_zeff,synthon_threshold,"
        "prior_value,prior_sigma,prior_residual,matched_coordinates,molecule_count,column_norm,"
        "correction,sigma,posterior_over_prior_sigma,acceptance_status"
    ]
    column_norms = result.diagnostics.column_norms or tuple(0.0 for _ in result.classes)
    for index, item in enumerate(result.classes):
        matched = sum(block.matched_counts.get(item.name, 0) for block in result.molecule_blocks)
        molecule_count = sum(
            1 for block in result.molecule_blocks if block.matched_counts.get(item.name, 0) > 0
        )
        prior_residual = result.prior_residual_after.get(item.name)
        ratio = ""
        if item.prior_sigma is not None and prior_residual is not None:
            ratio = f"{abs(prior_residual) / item.prior_sigma:.12g}"
        lines.append(
            f"{item.name},{_normalize_kind(item.kind)},{'|'.join(item.atom_symbols)},"
            f"{'' if item.value_min is None else f'{item.value_min:.12g}'},"
            f"{'' if item.value_max is None else f'{item.value_max:.12g}'},"
            f"{'|'.join(item.synthon_signatures)},"
            f"{'|'.join(f'{value:.12g}' for value in item.synthon_zeff)},"
            f"{'' if item.synthon_threshold is None else f'{item.synthon_threshold:.12g}'},"
            f"{'' if item.prior_value is None else f'{item.prior_value:.12g}'},"
            f"{'' if item.prior_sigma is None else f'{item.prior_sigma:.12g}'},"
            f"{'' if prior_residual is None else f'{prior_residual:.12g}'},"
            f"{matched},{molecule_count},{column_norms[index]:.12g},"
            f"{result.corrections[item.name]:.12g},{result.sigma[item.name]:.12g},{ratio},{result.acceptance.status}"
        )
    return "\n".join(lines) + "\n"


def _manifest_json(result: EnsembleClassCorrectionFit) -> str:
    payload = {
        "schema": "oracle.semiexp.ensemble.result.v1",
        "model": {
            "equation": "q_SE(m,k) = q_QC(m,k) + Delta[class(m,k)]",
            "linearization": "single weighted linearized correction around each molecule computational reference geometry",
            "class_projection": {
                "stretches": "unordered atom pair",
                "bends": "central atom preserved; terminal atoms canonicalized",
                "torsions": "unordered central-atom pair",
                "out_of_plane": "central atom of U(center,a,b,c)",
            },
            "coordinate_types_are_not_mixed": True,
        },
        "molecules": len(result.molecule_blocks),
        "classes": [
            {
                "name": item.name,
                "kind": _normalize_kind(item.kind),
                "atom_symbols": list(item.atom_symbols),
                "patterns": list(item.patterns),
                "value_min": item.value_min,
                "value_max": item.value_max,
                "synthon_signatures": list(item.synthon_signatures),
                "synthon_zeff": list(item.synthon_zeff),
                "synthon_threshold": item.synthon_threshold,
                "prior_value": item.prior_value,
                "prior_sigma": item.prior_sigma,
                "prior_residual": result.prior_residual_after.get(item.name),
                "correction": result.corrections[item.name],
                "sigma": result.sigma[item.name],
            }
            for item in result.classes
        ],
        "rank": result.rank,
        "scaled_condition_number": result.condition_number,
        "acceptance": {
            "status": result.acceptance.status,
            "accepted": result.acceptance.accepted,
            "reasons": list(result.acceptance.reasons),
            "review_items": list(result.acceptance.review_items),
            "policy": {
                "require_full_rank": result.acceptance_policy.require_full_rank,
                "max_condition_number": result.acceptance_policy.max_condition_number,
                "min_residual_degrees_of_freedom": result.acceptance_policy.min_residual_degrees_of_freedom,
                "min_molecule_support": result.acceptance_policy.min_molecule_support,
                "high_correlation_review_threshold": result.acceptance_policy.high_correlation_review_threshold,
                "high_correlation_reject_threshold": result.acceptance_policy.high_correlation_reject_threshold,
            },
        },
        "numerical_diagnostics": {
            "rank": result.diagnostics.rank,
            "n_columns": result.diagnostics.n_columns,
            "n_rows": result.diagnostics.n_rows,
            "residual_degrees_of_freedom": result.diagnostics.residual_degrees_of_freedom,
            "scaled_condition_number": result.diagnostics.condition_number,
            "singular_values": list(result.diagnostics.singular_values),
            "column_norms": list(result.diagnostics.column_norms),
            "warnings": list(result.diagnostics.warnings),
            "high_correlation_pairs": [
                {"left": left, "right": right, "correlation": value}
                for left, right, value in result.diagnostics.high_correlation_pairs
            ],
        },
        "weighted_rms_before": result.weighted_rms_before,
        "weighted_rms_after": result.weighted_rms_after,
        "molecule_blocks": [
            {
                "molecule": block.molecule,
                "n_rows": block.n_rows,
                "n_active_gics": block.n_active_gics,
                "matched_counts": block.matched_counts,
                "rank_contribution": block.rank_contribution,
                "weighted_rms_before": block.residual_before,
                "weighted_rms_after": block.residual_after,
            }
            for block in result.molecule_blocks
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


__all__ = [
    "ENSEMBLE_JOB_SCHEMA",
    "EnsembleAcceptanceDecision",
    "EnsembleAcceptancePolicy",
    "EnsembleClassCorrection",
    "EnsembleClassCorrectionFit",
    "EnsembleJobInput",
    "EnsembleMolecule",
    "EnsembleMoleculeBlock",
    "EnsembleNumericalDiagnostics",
    "fit_ensemble_job",
    "fit_ensemble_class_corrections",
    "read_ensemble_job",
    "write_ensemble_class_correction_outputs",
]
