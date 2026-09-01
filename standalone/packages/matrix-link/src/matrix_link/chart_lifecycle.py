"""Accepted-state lifecycle for native LINK optimization charts.

ORACLE owns every chemical decision and SMITH owns every coordinate build.
This module only validates and applies their typed results at accepted LINK
states.  It deliberately contains no atom-, molecule-, contact- or
stationary-point classification rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from matrix_numerics import singular_spectrum

from .coordinate_domain import near_linear_ordinary_angle_ids


LINK_CHART_LIFECYCLE_SCHEMA = "matrix.link.chart_lifecycle.v1"
CHART_ACTION_KEEP = "KEEP"
CHART_ACTION_DEFER = "DEFER"
CHART_ACTION_REBUILD = "REBUILD"
CHART_ACTION_INVALID = "INVALID"
CHART_TASK_MINIMUM = "MINIMUM"
CHART_TASK_TRANSITION_STATE = "TRANSITION_STATE"


class ChartLifecycleError(RuntimeError):
    """A typed ORACLE/SMITH chart transition failed a LINK invariant."""


class ChartCandidateUnavailable(ChartLifecycleError):
    """A proposed chart cannot satisfy the frozen rank/conditioning gates."""


@dataclass(frozen=True)
class ChartIdentity:
    task_regime: str
    atom_order_sha256: str
    oracle_contract_sha256: str
    oracle_topology_sha256: str
    smith_definition_sha256: str
    target_rank: int

    def __post_init__(self) -> None:
        regime = str(self.task_regime).strip().upper()
        if regime not in {CHART_TASK_MINIMUM, CHART_TASK_TRANSITION_STATE}:
            raise ValueError("chart task regime must be MINIMUM or TRANSITION_STATE")
        object.__setattr__(self, "task_regime", regime)
        for name in (
            "atom_order_sha256",
            "oracle_contract_sha256",
            "oracle_topology_sha256",
            "smith_definition_sha256",
        ):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, value)
        if int(self.target_rank) < 0:
            raise ValueError("chart target rank must be non-negative")
        object.__setattr__(self, "target_rank", int(self.target_rank))


@dataclass(frozen=True)
class OracleChartDecision:
    """Typed ORACLE disposition consumed mechanically by LINK."""

    action: str
    task_regime: str
    reason: str
    oracle_contract_sha256: str
    oracle_topology_sha256: str
    persistence_certificate: str

    def __post_init__(self) -> None:
        action = str(self.action).strip().upper()
        if action not in {
            CHART_ACTION_KEEP,
            CHART_ACTION_DEFER,
            CHART_ACTION_REBUILD,
            CHART_ACTION_INVALID,
        }:
            raise ValueError("unsupported ORACLE chart lifecycle action")
        regime = str(self.task_regime).strip().upper()
        if regime not in {CHART_TASK_MINIMUM, CHART_TASK_TRANSITION_STATE}:
            raise ValueError("ORACLE chart decision has an unsupported task regime")
        if not str(self.reason).strip():
            raise ValueError("ORACLE chart decision requires an auditable reason")
        if action == CHART_ACTION_REBUILD and not str(self.persistence_certificate).strip():
            raise ValueError("ORACLE rebuild decisions require a persistence certificate")
        for name in ("oracle_contract_sha256", "oracle_topology_sha256"):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "task_regime", regime)


@dataclass(frozen=True)
class ChartCandidate:
    """One SMITH-built chart at the accepted Cartesian geometry."""

    identity: ChartIdentity
    coordinate_model: object
    reference_coordinates_angstrom: np.ndarray
    b_matrix: np.ndarray
    source_xyzin_path: Path | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        b_matrix = np.asarray(self.b_matrix, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
            raise ValueError("chart reference coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("chart reference coordinates contain non-finite values")
        if b_matrix.shape != (self.identity.target_rank, coordinates.size):
            raise ValueError("chart Wilson matrix shape contradicts its target rank")
        if not np.all(np.isfinite(b_matrix)):
            raise ValueError("chart Wilson matrix contains non-finite values")
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates.copy())
        object.__setattr__(self, "b_matrix", b_matrix.copy())
        if self.source_xyzin_path is not None:
            object.__setattr__(self, "source_xyzin_path", Path(self.source_xyzin_path))


@dataclass(frozen=True)
class ChartSnapshot:
    epoch: int
    candidate: ChartCandidate

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise ValueError("chart epoch must be non-negative")
        object.__setattr__(self, "epoch", int(self.epoch))


@dataclass(frozen=True)
class ChartLifecycleResult:
    status: str
    reason: str
    decision: OracleChartDecision
    previous: ChartSnapshot
    current: ChartSnapshot
    previous_chart_valid_for_hessian_transport: bool = True

    @property
    def changed(self) -> bool:
        return self.current.epoch != self.previous.epoch

    @property
    def coordinate_changed(self) -> bool:
        return self.current.candidate.identity.smith_definition_sha256 != (
            self.previous.candidate.identity.smith_definition_sha256
        )

    @property
    def requires_commit(self) -> bool:
        return self.status in {"REBUILT", "REVALIDATED"}


class OracleChartAssessor(Protocol):
    def assess_accepted_geometry(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> OracleChartDecision: ...

    def commit_rebuild(self, decision: OracleChartDecision) -> None: ...


class SmithChartRebuilder(Protocol):
    def rebuild_chart(
        self,
        decision: OracleChartDecision,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> ChartCandidate: ...


class ChartLifecycleController:
    """Apply typed ORACLE/SMITH results at accepted geometries only."""

    def __init__(
        self,
        snapshot: ChartSnapshot,
        *,
        oracle: OracleChartAssessor,
        smith: SmithChartRebuilder,
        rank_absolute_tolerance: float = 1.0e-8,
        maximum_condition_number: float = 1.0e8,
        geometry_tolerance_angstrom: float = 1.0e-12,
    ) -> None:
        if rank_absolute_tolerance <= 0.0:
            raise ValueError("chart rank tolerance must be positive")
        if maximum_condition_number <= 1.0:
            raise ValueError("chart maximum condition number must exceed one")
        if geometry_tolerance_angstrom < 0.0:
            raise ValueError("chart geometry tolerance must be non-negative")
        self.snapshot = snapshot
        self.oracle = oracle
        self.smith = smith
        self.rank_absolute_tolerance = float(rank_absolute_tolerance)
        self.maximum_condition_number = float(maximum_condition_number)
        self.geometry_tolerance_angstrom = float(geometry_tolerance_angstrom)

    def validate_initial_chart(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        coordinate_model: object,
        *,
        task_regime: str,
    ) -> None:
        candidate = self.snapshot.candidate
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        self._validate_accepted_frame(self.snapshot, tuple(atoms), coordinates)
        if self.snapshot.epoch != 0:
            raise ChartLifecycleError("dynamic chart resume requires a persisted epoch manifest")
        if candidate.coordinate_model is not coordinate_model:
            raise ChartLifecycleError("initial lifecycle chart is not the optimizer chart object")
        if candidate.identity.target_rank != len(getattr(coordinate_model, "labels", ())):
            raise ChartLifecycleError("initial lifecycle chart rank contradicts optimizer model")
        if candidate.identity.task_regime != str(task_regime).strip().upper():
            raise ChartLifecycleError("initial lifecycle regime contradicts optimizer task")
        if not np.allclose(
            candidate.reference_coordinates_angstrom,
            coordinates,
            rtol=0.0,
            atol=self.geometry_tolerance_angstrom,
        ):
            raise ChartLifecycleError("initial lifecycle geometry is not the optimizer reference")

    def validate_proposed_geometry(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
    ) -> tuple[bool, str]:
        """Check a prospective TS geometry before LINK accepts its step.

        This is deliberately a fixed-chart numerical check only.  It does not
        ask ORACLE for a lifecycle decision and therefore has no chemical
        policy or side effects.  A prospective point that leaves the current
        chart domain is rejected by LINK before it becomes an accepted state;
        the normal trust-region recovery can then reduce the Cartesian step.
        """

        previous = self.snapshot
        identity = previous.candidate.identity
        decision = OracleChartDecision(
            action=CHART_ACTION_KEEP,
            task_regime=identity.task_regime,
            reason="LINK_PRE_ACCEPT_FIXED_CHART_VALIDATION",
            oracle_contract_sha256=identity.oracle_contract_sha256,
            oracle_topology_sha256=identity.oracle_topology_sha256,
            persistence_certificate="",
        )
        _candidate, reason, valid = self._revalidate_fixed_ts_chart(
            previous,
            decision,
            np.asarray(coordinates_angstrom, dtype=float),
        )
        return bool(valid), reason

    def evaluate_accepted_geometry(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
    ) -> ChartLifecycleResult:
        atom_order = tuple(str(atom) for atom in atoms)
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        previous = self.snapshot
        self._validate_accepted_frame(previous, atom_order, coordinates)
        decision = self.oracle.assess_accepted_geometry(
            previous,
            atom_order,
            coordinates.copy(),
        )
        if decision.task_regime != previous.candidate.identity.task_regime:
            raise ChartLifecycleError("ORACLE changed task regime within one optimization")
        if decision.action == CHART_ACTION_INVALID:
            raise ChartLifecycleError(f"ORACLE rejected accepted geometry: {decision.reason}")
        if decision.action in {CHART_ACTION_KEEP, CHART_ACTION_DEFER}:
            (
                _fixed_candidate,
                fixed_reason,
                previous_chart_valid_for_hessian_transport,
            ) = self._revalidate_fixed_ts_chart(previous, decision, coordinates)
            if not previous_chart_valid_for_hessian_transport:
                escalation = getattr(
                    self.oracle,
                    "require_rebuild_for_invalid_chart",
                    None,
                )
                if not callable(escalation):
                    raise ChartLifecycleError(
                        "invalid fixed chart cannot be deferred without an ORACLE "
                        "rebuild authorization"
                    )
                decision = escalation(decision, fixed_reason)
            else:
                if decision.action == CHART_ACTION_KEEP and (
                    decision.oracle_contract_sha256
                    != previous.candidate.identity.oracle_contract_sha256
                    or decision.oracle_topology_sha256
                    != previous.candidate.identity.oracle_topology_sha256
                ):
                    raise ChartLifecycleError("ORACLE KEEP decision changed frozen identity")
                return ChartLifecycleResult(
                    status=decision.action,
                    reason=decision.reason,
                    decision=decision,
                    previous=previous,
                    current=previous,
                )
        (
            fixed_candidate,
            fixed_reason,
            previous_chart_valid_for_hessian_transport,
        ) = self._revalidate_fixed_ts_chart(
            previous,
            decision,
            coordinates,
        )
        if fixed_candidate is not None:
            return ChartLifecycleResult(
                status="REVALIDATED",
                reason=f"{decision.reason}; {fixed_reason}",
                decision=decision,
                previous=previous,
                current=ChartSnapshot(epoch=previous.epoch, candidate=fixed_candidate),
                previous_chart_valid_for_hessian_transport=True,
            )
        try:
            candidate = self.smith.rebuild_chart(
                decision,
                previous,
                atom_order,
                coordinates.copy(),
            )
            self._validate_rebuild(previous, decision, candidate, atom_order, coordinates)
        except ChartCandidateUnavailable as exc:
            if not previous_chart_valid_for_hessian_transport:
                raise ChartLifecycleError(
                    "SMITH cannot replace a numerically invalid fixed chart"
                ) from exc
            return ChartLifecycleResult(
                status="REBUILD_DEFERRED",
                reason=f"{decision.reason}; {fixed_reason}; {exc}",
                decision=decision,
                previous=previous,
                current=previous,
                previous_chart_valid_for_hessian_transport=(
                    previous_chart_valid_for_hessian_transport
                ),
            )
        coordinate_changed = candidate.identity.smith_definition_sha256 != (
            previous.candidate.identity.smith_definition_sha256
        )
        if not coordinate_changed:
            return ChartLifecycleResult(
                status="REVALIDATED",
                reason=f"{decision.reason}; {fixed_reason}",
                decision=decision,
                previous=previous,
                current=ChartSnapshot(epoch=previous.epoch, candidate=candidate),
                previous_chart_valid_for_hessian_transport=(
                    previous_chart_valid_for_hessian_transport
                ),
            )
        current = ChartSnapshot(epoch=previous.epoch + 1, candidate=candidate)
        return ChartLifecycleResult(
            status="REBUILT",
            reason=f"{decision.reason}; {fixed_reason}",
            decision=decision,
            previous=previous,
            current=current,
            previous_chart_valid_for_hessian_transport=(
                previous_chart_valid_for_hessian_transport
            ),
        )

    def _revalidate_fixed_ts_chart(
        self,
        previous: ChartSnapshot,
        decision: OracleChartDecision,
        coordinates: np.ndarray,
    ) -> tuple[ChartCandidate | None, str, bool]:
        """Keep a TS chart unless its current Wilson matrix is invalid."""

        if decision.task_regime != CHART_TASK_TRANSITION_STATE:
            return None, "NON_TS_DYNAMIC_CHART_POLICY", True
        model = previous.candidate.coordinate_model
        definition = getattr(model, "sonic_definition", None)
        if str(getattr(model, "kind", "")).strip().casefold() != "sonic" or definition is None:
            return None, "TS_FIXED_CHART_UNAVAILABLE_NON_SONIC_MODEL", False
        invalid_angles = near_linear_ordinary_angle_ids(definition, coordinates)
        if invalid_angles:
            return (
                None,
                "TS_FIXED_CHART_ORDINARY_ANGLE_DOMAIN_INVALID="
                + ",".join(invalid_angles),
                False,
            )
        from matrix_smith import build_gic_b_matrix

        try:
            b_matrix = np.asarray(
                build_gic_b_matrix(
                    definition,
                    coordinates_angstrom=coordinates,
                ).rows,
                dtype=float,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            return None, f"TS_FIXED_CHART_DOMAIN_INVALID={type(exc).__name__}", False
        target_rank = previous.candidate.identity.target_rank
        expected_shape = (target_rank, coordinates.size)
        if b_matrix.shape != expected_shape:
            return (
                None,
                "TS_FIXED_CHART_SHAPE_INVALID="
                f"{b_matrix.shape!r}!={expected_shape!r}",
                False,
            )
        row_norms = np.linalg.norm(b_matrix, axis=1)
        if np.any(~np.isfinite(row_norms)) or np.any(
            row_norms <= self.rank_absolute_tolerance
        ):
            return None, "TS_FIXED_CHART_ZERO_OR_NONFINITE_ROW", False
        normalized = b_matrix / row_norms[:, None]
        spectrum = singular_spectrum(
            normalized,
            absolute_tolerance=self.rank_absolute_tolerance,
        )
        if spectrum.rank != target_rank:
            return (
                None,
                f"TS_FIXED_CHART_RANK_INVALID={spectrum.rank}/{target_rank}",
                False,
            )
        if not np.isfinite(spectrum.condition_number) or (
            spectrum.condition_number > self.maximum_condition_number
        ):
            return (
                None,
                "TS_FIXED_CHART_CONDITION_INVALID="
                f"{spectrum.condition_number:.12g}>{self.maximum_condition_number:.12g}",
                False,
            )
        identity = ChartIdentity(
            task_regime=decision.task_regime,
            atom_order_sha256=previous.candidate.identity.atom_order_sha256,
            oracle_contract_sha256=decision.oracle_contract_sha256,
            oracle_topology_sha256=decision.oracle_topology_sha256,
            smith_definition_sha256=(
                previous.candidate.identity.smith_definition_sha256
            ),
            target_rank=target_rank,
        )
        candidate = ChartCandidate(
            identity=identity,
            coordinate_model=model,
            reference_coordinates_angstrom=coordinates,
            b_matrix=b_matrix,
            source_xyzin_path=previous.candidate.source_xyzin_path,
        )
        return (
            candidate,
            "TS_FIXED_CHART_NUMERICALLY_VALID "
            f"RANK={spectrum.rank}/{target_rank} "
            f"NORMALIZED_CONDITION={spectrum.condition_number:.12g}",
            True,
        )

    def commit_transition(self, result: ChartLifecycleResult) -> None:
        """Commit ORACLE and LINK state after optimizer transport succeeds."""

        if not result.requires_commit:
            raise ChartLifecycleError("a non-rebuild chart result cannot be committed")
        if result.previous is not self.snapshot:
            raise ChartLifecycleError("chart transition result is stale")
        self.oracle.commit_rebuild(result.decision)
        self.snapshot = result.current

    def _validate_accepted_frame(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates: np.ndarray,
    ) -> None:
        if coordinates.shape != snapshot.candidate.reference_coordinates_angstrom.shape:
            raise ChartLifecycleError("accepted geometry changed the chart atom frame")
        if not np.all(np.isfinite(coordinates)):
            raise ChartLifecycleError("accepted geometry contains non-finite values")
        if atom_order_sha256(atoms) != snapshot.candidate.identity.atom_order_sha256:
            raise ChartLifecycleError("accepted geometry changed atom order")

    def _validate_rebuild(
        self,
        previous: ChartSnapshot,
        decision: OracleChartDecision,
        candidate: ChartCandidate,
        atoms: tuple[str, ...],
        coordinates: np.ndarray,
    ) -> None:
        identity = candidate.identity
        if identity.task_regime != decision.task_regime:
            raise ChartLifecycleError("SMITH chart changed the ORACLE task regime")
        if identity.atom_order_sha256 != atom_order_sha256(atoms):
            raise ChartLifecycleError("SMITH chart changed atom order")
        if identity.oracle_contract_sha256 != decision.oracle_contract_sha256:
            raise ChartLifecycleError("SMITH chart does not consume the assessed ORACLE contract")
        if identity.oracle_topology_sha256 != decision.oracle_topology_sha256:
            raise ChartLifecycleError("SMITH chart does not consume the assessed ORACLE topology")
        if identity.target_rank != previous.candidate.identity.target_rank:
            raise ChartLifecycleError("SMITH chart changed the vibrational target rank")
        if not np.allclose(
            candidate.reference_coordinates_angstrom,
            coordinates,
            rtol=0.0,
            atol=self.geometry_tolerance_angstrom,
        ):
            raise ChartLifecycleError("SMITH changed the accepted post-ORACLE geometry")
        spectrum = singular_spectrum(
            candidate.b_matrix,
            absolute_tolerance=self.rank_absolute_tolerance,
        )
        if spectrum.rank != identity.target_rank:
            raise ChartLifecycleError("SMITH rebuilt a rank-deficient chart")
        if not np.isfinite(spectrum.condition_number) or (
            spectrum.condition_number > self.maximum_condition_number
        ):
            raise ChartLifecycleError("SMITH rebuilt an ill-conditioned chart")


def atom_order_sha256(atoms: Sequence[str]) -> str:
    payload = "\0".join(str(atom).strip() for atom in atoms).encode("utf-8")
    return sha256(payload).hexdigest()


def chart_lifecycle_result_to_json(result: ChartLifecycleResult) -> dict[str, object]:
    """Serialize one transition decision without backend or chemistry inference."""

    return {
        "schema": LINK_CHART_LIFECYCLE_SCHEMA,
        "status": result.status,
        "reason": result.reason,
        "action": result.decision.action,
        "task_regime": result.decision.task_regime,
        "previous_epoch": result.previous.epoch,
        "current_epoch": result.current.epoch,
        "oracle_contract_sha256": result.decision.oracle_contract_sha256,
        "oracle_topology_sha256": result.decision.oracle_topology_sha256,
        "persistence_certificate": result.decision.persistence_certificate,
        "previous_smith_definition_sha256": (
            result.previous.candidate.identity.smith_definition_sha256
        ),
        "current_smith_definition_sha256": (
            result.current.candidate.identity.smith_definition_sha256
        ),
        "target_rank": result.current.candidate.identity.target_rank,
        "changed": result.changed,
        "coordinate_changed": result.coordinate_changed,
        "previous_chart_valid_for_hessian_transport": (
            result.previous_chart_valid_for_hessian_transport
        ),
    }


class OraclePerceptionChartAdapter:
    """Translate ORACLE's native assessment without adding LINK policy."""

    def __init__(self, assessor: object) -> None:
        from matrix_oracle import OptimizationChartAssessor

        if not isinstance(assessor, OptimizationChartAssessor):
            raise TypeError("assessor must be an ORACLE OptimizationChartAssessor")
        self.assessor = assessor
        self._pending: dict[str, object] = {}
        self._latest_assessment: object | None = None

    def assess_accepted_geometry(
        self,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> OracleChartDecision:
        from matrix_chem.topology.elements import atomic_number

        numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
        if numbers != self.assessor.atomic_numbers:
            raise ChartLifecycleError("accepted atom identities contradict the ORACLE assessor")
        assessment = self.assessor.assess_accepted_geometry(coordinates_angstrom)
        self._latest_assessment = assessment
        decision = OracleChartDecision(
            action=assessment.action,
            task_regime=assessment.task_regime,
            reason=assessment.reason,
            oracle_contract_sha256=assessment.contract_sha256,
            oracle_topology_sha256=assessment.topology_sha256,
            persistence_certificate=assessment.persistence_certificate,
        )
        self._pending.clear()
        if decision.action == CHART_ACTION_REBUILD:
            self._pending[decision.persistence_certificate] = assessment
        return decision

    def commit_rebuild(self, decision: OracleChartDecision) -> None:
        assessment = self._pending.pop(decision.persistence_certificate, None)
        if assessment is None:
            raise ChartLifecycleError("ORACLE rebuild decision has no pending native assessment")
        self.assessor.commit_rebuild(assessment)

    def require_rebuild_for_invalid_chart(
        self,
        decision: OracleChartDecision,
        chart_domain_reason: str,
    ) -> OracleChartDecision:
        """Ask ORACLE to sign the current state for immediate chart recovery."""

        assessment = self._latest_assessment
        if assessment is None or (
            assessment.task_regime != decision.task_regime
            or assessment.contract_sha256 != decision.oracle_contract_sha256
            or assessment.topology_sha256 != decision.oracle_topology_sha256
        ):
            raise ChartLifecycleError("invalid-chart escalation has no matching ORACLE assessment")
        assessment = self.assessor.require_rebuild_for_invalid_chart(
            assessment,
            chart_domain_reason=chart_domain_reason,
        )
        rebuilt = OracleChartDecision(
            action=assessment.action,
            task_regime=assessment.task_regime,
            reason=assessment.reason,
            oracle_contract_sha256=assessment.contract_sha256,
            oracle_topology_sha256=assessment.topology_sha256,
            persistence_certificate=assessment.persistence_certificate,
        )
        self._pending.clear()
        self._pending[rebuilt.persistence_certificate] = assessment
        return rebuilt


class SmithXyzinChartRebuilder:
    """Build a new chart from a fresh ORACLE artifact using canonical SMITH."""

    def __init__(
        self,
        artifact_provider: Callable[
            [OracleChartDecision, tuple[str, ...], np.ndarray], Path | str
        ],
        *,
        symmetrize: bool = False,
    ) -> None:
        if not callable(artifact_provider):
            raise TypeError("artifact_provider must be callable")
        self.artifact_provider = artifact_provider
        self.symmetrize = bool(symmetrize)

    def rebuild_chart(
        self,
        decision: OracleChartDecision,
        snapshot: ChartSnapshot,
        atoms: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
    ) -> ChartCandidate:
        from matrix_core.sectioned_xyz import read_sectioned_lines, section_content
        from matrix_smith import (
            GICForgeContractError,
            GICForgeRankDeficiencyError,
            build_gic_b_matrix,
            sonic_definition_identity_sha256,
            write_gicforge_build_sections,
        )

        artifact = Path(self.artifact_provider(decision, atoms, coordinates_angstrom.copy()))
        if section_content(read_sectioned_lines(artifact), "GIC"):
            raise ChartLifecycleError(
                "ORACLE chart artifact contains a stale SMITH GIC section"
            )
        try:
            definition = write_gicforge_build_sections(
                artifact,
                symmetrize=self.symmetrize,
                fragment_context=(
                    "minimum"
                    if decision.task_regime == CHART_TASK_MINIMUM
                    else "transition_state"
                ),
            )
        except GICForgeRankDeficiencyError as exc:
            raise ChartCandidateUnavailable(
                "SMITH exact-rank candidate unavailable "
                f"(target={exc.target_rank}, rank={exc.selected_rank}, "
                f"candidates={exc.candidate_count})"
            ) from exc
        except GICForgeContractError as exc:
            raise ChartCandidateUnavailable(
                f"SMITH rejected the candidate chart ({exc})"
            ) from exc
        b_matrix = np.asarray(
            build_gic_b_matrix(
                definition,
                coordinates_angstrom=coordinates_angstrom,
            ).rows,
            dtype=float,
        )
        from .internal_coordinates import cartesian_from_internal_jacobian
        from .optimizer import OptimizerCoordinateModel
        from .scan import ANGSTROM_TO_BOHR

        directions = cartesian_from_internal_jacobian(b_matrix, rcond=1.0e-8).T
        labels = tuple(gic.name or gic.identifier for gic in definition.gics)
        model = OptimizerCoordinateModel(
            kind="sonic",
            labels=labels,
            directions_angstrom=directions,
            metric_diagonal=np.maximum(
                np.sum((directions * ANGSTROM_TO_BOHR) ** 2, axis=1),
                1.0e-12,
            ),
            sonic_labels=labels,
            sonic_definition=definition,
        )
        identity = ChartIdentity(
            task_regime=decision.task_regime,
            atom_order_sha256=atom_order_sha256(atoms),
            oracle_contract_sha256=decision.oracle_contract_sha256,
            oracle_topology_sha256=decision.oracle_topology_sha256,
            smith_definition_sha256=sonic_definition_identity_sha256(definition),
            target_rank=len(labels),
        )
        return ChartCandidate(
            identity=identity,
            coordinate_model=model,
            reference_coordinates_angstrom=coordinates_angstrom,
            b_matrix=b_matrix,
            source_xyzin_path=artifact,
        )


def chart_lifecycle_controller_from_xyzin(
    source_xyzin: Path | str,
    *,
    run_dir: Path | str,
    coordinate_model: object,
    stationary_point: str,
) -> ChartLifecycleController:
    """Compose the canonical ORACLE--SMITH lifecycle for one LINK SONIC run."""

    from matrix_chem import read_enriched_xyz
    from matrix_chem.topology.elements import atomic_number
    from matrix_oracle import (
        OptimizationChartAssessor,
        materialize_optimization_chart_artifact,
    )
    from matrix_smith import (
        build_gic_b_matrix,
        sonic_definition_identity_sha256,
    )

    source = Path(source_xyzin).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"chart lifecycle source does not exist: {source}")
    kind = str(getattr(coordinate_model, "kind", "")).strip().casefold()
    definition = getattr(coordinate_model, "sonic_definition", None)
    if kind != "sonic" or definition is None:
        raise ChartLifecycleError("chart lifecycle requires the canonical SONIC model")
    labels = tuple(str(value) for value in getattr(coordinate_model, "labels", ()))
    full_labels = tuple(gic.name or gic.identifier for gic in definition.gics)
    if labels != full_labels or getattr(coordinate_model, "rank_reduced_labels", ()):
        raise ChartLifecycleError(
            "chart lifecycle requires the complete exact-rank SONIC definition"
        )
    if int(definition.target_rank) != len(labels):
        raise ChartLifecycleError("SONIC definition does not have its exact target rank")
    normalized_stationary_point = str(stationary_point).strip().casefold().replace("-", "_")
    if normalized_stationary_point == "minimum":
        regime = CHART_TASK_MINIMUM
    elif normalized_stationary_point == "transition_state":
        regime = CHART_TASK_TRANSITION_STATE
    else:
        raise ChartLifecycleError(
            "chart lifecycle requires an explicit minimum or transition-state task"
        )

    geometry = read_enriched_xyz(source)
    atoms = tuple(geometry.atoms)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    numbers = tuple(int(atomic_number(atom) or 0) for atom in atoms)
    if any(number <= 0 for number in numbers):
        raise ChartLifecycleError("chart lifecycle source has an unsupported atomic label")
    assessor = OptimizationChartAssessor(
        numbers,
        coordinates,
        task_regime=regime,
    )
    reference = assessor.reference_identity
    b_matrix = np.asarray(
        build_gic_b_matrix(
            definition,
            coordinates_angstrom=coordinates,
        ).rows,
        dtype=float,
    )
    spectrum = singular_spectrum(b_matrix, absolute_tolerance=1.0e-8)
    if spectrum.rank != len(labels):
        raise ChartLifecycleError("initial SONIC lifecycle chart is rank deficient")
    if not np.isfinite(spectrum.condition_number) or spectrum.condition_number > 1.0e8:
        raise ChartLifecycleError("initial SONIC lifecycle chart is ill-conditioned")
    initial = ChartCandidate(
        identity=ChartIdentity(
            task_regime=regime,
            atom_order_sha256=atom_order_sha256(atoms),
            oracle_contract_sha256=reference.contract_sha256,
            oracle_topology_sha256=reference.topology_sha256,
            smith_definition_sha256=sonic_definition_identity_sha256(definition),
            target_rank=len(labels),
        ),
        coordinate_model=coordinate_model,
        reference_coordinates_angstrom=coordinates,
        b_matrix=b_matrix,
        source_xyzin_path=source,
    )
    root = Path(run_dir).expanduser().resolve() / "chart_epochs"
    root.mkdir(parents=True, exist_ok=True)
    sequence = 0

    def artifact_provider(
        decision: OracleChartDecision,
        accepted_atoms: tuple[str, ...],
        accepted_coordinates: np.ndarray,
    ) -> Path:
        nonlocal sequence
        sequence += 1
        target = root / f"oracle_candidate_{sequence:06d}.xyzin"
        return materialize_optimization_chart_artifact(
            source,
            target,
            accepted_atoms,
            accepted_coordinates,
            task_regime=decision.task_regime,
            expected_contract_sha256=decision.oracle_contract_sha256,
        )

    return ChartLifecycleController(
        ChartSnapshot(epoch=0, candidate=initial),
        oracle=OraclePerceptionChartAdapter(assessor),
        smith=SmithXyzinChartRebuilder(
            artifact_provider,
            symmetrize=bool(definition.symmetrize),
        ),
    )


__all__ = [
    "CHART_ACTION_DEFER",
    "CHART_ACTION_INVALID",
    "CHART_ACTION_KEEP",
    "CHART_ACTION_REBUILD",
    "CHART_TASK_MINIMUM",
    "CHART_TASK_TRANSITION_STATE",
    "LINK_CHART_LIFECYCLE_SCHEMA",
    "ChartCandidate",
    "ChartIdentity",
    "ChartLifecycleController",
    "ChartLifecycleError",
    "ChartLifecycleResult",
    "ChartSnapshot",
    "OracleChartAssessor",
    "OracleChartDecision",
    "OraclePerceptionChartAdapter",
    "SmithXyzinChartRebuilder",
    "SmithChartRebuilder",
    "atom_order_sha256",
    "chart_lifecycle_controller_from_xyzin",
    "chart_lifecycle_result_to_json",
]
