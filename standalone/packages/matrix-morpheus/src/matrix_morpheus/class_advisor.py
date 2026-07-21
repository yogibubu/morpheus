from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_chem.topology.pipeline import build_topology_objects

from .contracts import ParameterClassConstraint


@dataclass(frozen=True)
class PrimitiveClassSpec:
    """User-facing chemical class defined by primitive-coordinate patterns."""

    name: str
    patterns: tuple[str, ...]

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Primitive class name cannot be empty")
        if not self.patterns or any(not pattern.strip() for pattern in self.patterns):
            raise ValueError("Primitive class patterns cannot be empty")


@dataclass(frozen=True)
class DerivedPrimitiveClassPlan:
    """Disjoint GIC classes and blocked complement derived from primitive classes."""

    fixed_patterns: tuple[str, ...]
    parameter_classes: tuple[ParameterClassConstraint, ...]
    rejected_labels: tuple[str, ...]
    budget: int
    class_support: tuple[tuple[str, int, float], ...] = ()
    ambiguous_labels: tuple[str, ...] = ()
    budget_limited_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynthonPrimitiveClassSpec:
    """Options for topology/synthon-derived primitive classes."""

    enabled: bool = False
    level: str = "auto"
    include_bonds: bool = True
    include_angles: bool = True
    min_group_size: int = 2
    bond_order_bins: bool = True


@dataclass(frozen=True)
class SynthonClassThresholds:
    """Quantization thresholds for continuous synthon descriptors."""

    covalency_step: float
    delocalization_step: float
    strain_step: float
    bond_order_step: float
    include_synthon_signature: bool = True


SYNTHON_CLASS_LEVELS = {
    "coarse": SynthonClassThresholds(
        covalency_step=0.50,
        delocalization_step=0.50,
        strain_step=0.50,
        bond_order_step=0.50,
        include_synthon_signature=False,
    ),
    "medium": SynthonClassThresholds(
        covalency_step=0.20,
        delocalization_step=0.20,
        strain_step=0.20,
        bond_order_step=0.25,
    ),
    "fine": SynthonClassThresholds(
        covalency_step=0.10,
        delocalization_step=0.10,
        strain_step=0.10,
        bond_order_step=0.10,
    ),
}


def synthon_level_for_budget(class_budget: int | None) -> str:
    """Choose a synthon class refinement level from the experimental data budget."""

    if class_budget is None:
        return "fine"
    if class_budget <= 5:
        return "coarse"
    if class_budget <= 12:
        return "medium"
    return "fine"


def parse_primitive_class_spec(raw: str) -> PrimitiveClassSpec:
    """Parse `name:primitive[|primitive...]` into a primitive class specification."""

    parts = str(raw).split(":", 1)
    if len(parts) != 2:
        raise ValueError("--primitive-class must be name:primitive[|primitive...]")
    spec = PrimitiveClassSpec(
        name=parts[0].strip(),
        patterns=tuple(part.strip() for part in parts[1].split("|") if part.strip()),
    )
    spec.validate()
    return spec


def synthon_primitive_class_specs(
    atoms: tuple[str, ...] | list[str],
    coords_angstrom,
    *,
    level: str = "medium",
    include_bonds: bool = True,
    include_angles: bool = True,
    min_group_size: int = 2,
    bond_order_bins: bool = True,
) -> tuple[PrimitiveClassSpec, ...]:
    """Generate primitive classes from MATRIX topology/synthon descriptors.

    The classes are intentionally conservative: atom types include element,
    rounded synthon signature and local bonding descriptors; bond classes can
    additionally be split by the continuous bond-order bin.  MORPHEUS still
    maps these primitive classes onto the actual reduced GIC labels with
    coefficient thresholds and the available experimental data budget.
    """

    if min_group_size < 1:
        raise ValueError("min_group_size must be at least one")
    thresholds = _synthon_thresholds(level)
    symbols = tuple(str(atom) for atom in atoms)
    coords = np.asarray(coords_angstrom, dtype=float)
    z_numbers = np.array([atomic_number(symbol) or 0 for symbol in symbols], dtype=int)
    _continuous, graph, _ringset, synthons, _aromaticity = build_topology_objects(coords, z_numbers)
    atom_classes = tuple(
        _synthon_atom_class(symbols, synthons, idx, thresholds) for idx in range(len(symbols))
    )
    groups: dict[str, list[str]] = {}

    if include_bonds:
        for i, j in sorted(tuple(sorted((int(a), int(b)))) for a, b in graph.bonds):
            left, right = sorted((atom_classes[i], atom_classes[j]))
            bo_tag = ""
            if bond_order_bins:
                bo_tag = "_" + _bond_order_bin(
                    float(synthons.bond_order(i, j)),
                    thresholds.bond_order_step,
                )
            name = _safe_class_name(f"{left}_{right}{bo_tag}_stretches")
            groups.setdefault(name, []).append(f"R({i + 1},{j + 1})")

    if include_angles:
        adjacency = tuple(
            tuple(sorted(int(item) for item in graph.adjacency[index]))
            for index in range(len(symbols))
        )
        for center, neighbors in enumerate(adjacency):
            for pos, left in enumerate(neighbors):
                for right in neighbors[pos + 1 :]:
                    ends = sorted((atom_classes[left], atom_classes[right]))
                    name = _safe_class_name(
                        f"{ends[0]}_{atom_classes[center]}_{ends[1]}_bends"
                    )
                    groups.setdefault(name, []).append(
                        f"A({left + 1},{center + 1},{right + 1})"
                    )

    return tuple(
        PrimitiveClassSpec(name, tuple(dict.fromkeys(patterns)))
        for name, patterns in sorted(groups.items())
        if len(tuple(dict.fromkeys(patterns))) >= min_group_size
    )


def primitive_class_decision_lines(plan: DerivedPrimitiveClassPlan) -> tuple[str, ...]:
    """Human-readable advisor decisions for CLI/report diagnostics."""

    lines: list[str] = []
    selected = {pattern for item in plan.parameter_classes for pattern in item.patterns}
    support = {name: (count, score) for name, count, score in plan.class_support}
    for item in plan.parameter_classes:
        count, score = support.get(item.name, (len(item.patterns), 0.0))
        lines.append(
            "advisor_decision: "
            f"class={item.name} status=accepted gics={len(item.patterns)} "
            f"support_count={count} support_score={score:.6g} "
            f"reason=dominant_primitives_within_data_budget"
        )
    for name in plan.budget_limited_classes:
        count, score = support.get(name, (0, 0.0))
        lines.append(
            "advisor_decision: "
            f"class={name} status=rejected support_count={count} "
            f"support_score={score:.6g} reason=data_budget"
        )
    for label in plan.ambiguous_labels:
        state = "accepted" if label in selected else "fixed"
        lines.append(
            "advisor_decision: "
            f"gic={label} status={state} reason=ambiguous_cross_class_coefficients"
        )
    for label in plan.rejected_labels:
        lines.append(
            "advisor_decision: "
            f"gic={label} status=fixed reason=below_assignment_threshold"
        )
    for label in plan.fixed_patterns:
        if label not in selected and label not in set(plan.rejected_labels):
            lines.append(
                "advisor_decision: "
                f"gic={label} status=fixed reason=outside_selected_classes"
            )
    return tuple(dict.fromkeys(lines))


def derive_primitive_class_plan(
    gic_labels: tuple[str, ...],
    primitive_classes: tuple[PrimitiveClassSpec, ...],
    *,
    min_fraction: float = 0.70,
    cross_fraction_max: float = 0.20,
    max_classes: int | None = None,
) -> DerivedPrimitiveClassPlan:
    """Map primitive-class definitions onto a disjoint reduced GIC class model.

    Classes are interpreted as a priority cascade: broader classes should be
    declared first and more specific classes later. A GIC is assigned to the
    last class whose primitive coefficient reaches `min_fraction`; earlier
    classes are used only when all later classes stay below
    `cross_fraction_max`. Unsupported GICs are frozen. If `max_classes` is set,
    only the best-supported classes are kept.
    """

    if not primitive_classes:
        return DerivedPrimitiveClassPlan((), (), (), 0)
    if min_fraction <= 0.0:
        raise ValueError("min_fraction must be positive")
    if cross_fraction_max < 0.0:
        raise ValueError("cross_fraction_max must be non-negative")
    for spec in primitive_classes:
        spec.validate()

    class_patterns = {
        spec.name: tuple(_canonical_primitive(pattern) for pattern in spec.patterns)
        for spec in primitive_classes
    }
    assignments: dict[str, list[tuple[str, float]]] = {spec.name: [] for spec in primitive_classes}
    rejected: list[str] = []
    fixed: list[str] = []
    ambiguous: list[str] = []

    for label in gic_labels:
        gid = _gic_id(label)
        coeffs = _primitive_coefficients(label)
        scores = {
            spec.name: max(
                (coeffs.get(pattern, 0.0) for pattern in class_patterns[spec.name]),
                default=0.0,
            )
            for spec in primitive_classes
        }
        positive_scores = sorted((score for score in scores.values() if score > 0.0), reverse=True)
        if len(positive_scores) > 1 and positive_scores[1] >= cross_fraction_max:
            ambiguous.append(gid)
        assigned_name = ""
        assigned_score = 0.0
        for index, spec in enumerate(primitive_classes):
            score = scores[spec.name]
            later_scores = [scores[item.name] for item in primitive_classes[index + 1 :]]
            if score >= min_fraction and all(item <= cross_fraction_max for item in later_scores):
                assigned_name = spec.name
                assigned_score = score
        if assigned_name:
            assignments[assigned_name].append((gid, assigned_score))
        else:
            fixed.append(gid)
            if any(score > 0.0 for score in scores.values()):
                rejected.append(gid)

    class_order = sorted(
        primitive_classes,
        key=lambda spec: (
            -len(assignments[spec.name]),
            -sum(score for _gid, score in assignments[spec.name]),
            spec.name,
        ),
    )
    if max_classes is not None:
        if max_classes < 0:
            raise ValueError("max_classes must be non-negative")
        class_order = class_order[:max_classes]
    kept_names = {spec.name for spec in class_order if assignments[spec.name]}
    budget_limited = tuple(
        spec.name
        for spec in primitive_classes
        if assignments[spec.name] and spec.name not in kept_names
    )

    constraints: list[ParameterClassConstraint] = []
    for spec in class_order:
        labels = tuple(gid for gid, _score in assignments[spec.name])
        if labels:
            constraints.append(ParameterClassConstraint(spec.name, labels, "shared"))

    for name, labels in assignments.items():
        if name not in kept_names:
            fixed.extend(gid for gid, _score in labels)

    selected = {pattern for constraint in constraints for pattern in constraint.patterns}
    fixed.extend(
        _gic_id(label)
        for label in gic_labels
        if _gic_id(label) not in selected and _gic_id(label) not in fixed
    )
    return DerivedPrimitiveClassPlan(
        fixed_patterns=tuple(dict.fromkeys(fixed)),
        parameter_classes=tuple(constraints),
        rejected_labels=tuple(dict.fromkeys(rejected)),
        budget=len(constraints) if max_classes is None else max_classes,
        class_support=tuple(
            (
                spec.name,
                len(assignments[spec.name]),
                float(sum(score for _gid, score in assignments[spec.name])),
            )
            for spec in primitive_classes
        ),
        ambiguous_labels=tuple(dict.fromkeys(ambiguous)),
        budget_limited_classes=budget_limited,
    )


def _gic_id(label: str) -> str:
    return str(label).split(None, 1)[0]


def _canonical_primitive(pattern: str) -> str:
    text = re.sub(r"\s+", "", str(pattern))
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\(([^)]*)\)", text)
    if match is None:
        return text
    kind = match.group(1).upper()
    atoms = tuple(int(item) for item in match.group(2).split(",") if item)
    return _canonical_internal(kind, atoms)


def _canonical_internal(kind: str, atoms: tuple[int, ...]) -> str:
    if kind == "R" and len(atoms) == 2:
        atoms = tuple(sorted(atoms))
    elif kind in {"A", "B"} and len(atoms) == 3:
        atoms = min(atoms, tuple(reversed(atoms)))
    elif kind in {"D", "T"} and len(atoms) == 4:
        atoms = min(atoms, tuple(reversed(atoms)))
    args = ",".join(str(item) for item in atoms)
    return f"{kind}({args})"


def _primitive_coefficients(label: str) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for value, primitive in re.findall(
        r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\*\s*"
        r"([A-Za-z][A-Za-z0-9_]*\(\s*\d+(?:\s*,\s*\d+)*\s*\))",
        str(label),
    ):
        key = _canonical_primitive(primitive)
        coeffs[key] = max(coeffs.get(key, 0.0), abs(float(value)))
    return coeffs


def _synthon_thresholds(level: str) -> SynthonClassThresholds:
    key = str(level or "medium").strip().lower()
    if key == "auto":
        key = "medium"
    if key not in SYNTHON_CLASS_LEVELS:
        known = ", ".join((*sorted(SYNTHON_CLASS_LEVELS), "auto"))
        raise ValueError(f"Unknown synthon class level {level!r}; known: {known}")
    return SYNTHON_CLASS_LEVELS[key]


def _synthon_atom_class(
    symbols: tuple[str, ...],
    synthons,
    idx: int,
    thresholds: SynthonClassThresholds,
) -> str:
    z = int(getattr(synthons, "Z", [0])[idx])
    symbol = atomic_symbol(z) if z else symbols[idx]
    parts = [symbol]
    if thresholds.include_synthon_signature:
        parts.append(str(synthons.canonical_signature_str(idx)).replace("-", "_"))
    parts.extend(
        (
            "c" + _quantized_tag(float(synthons.covalency(idx)), thresholds.covalency_step),
            "d" + _quantized_tag(
                float(synthons.delocalization(idx)),
                thresholds.delocalization_step,
            ),
            "s" + _quantized_tag(float(synthons.strain(idx)), thresholds.strain_step),
            "pi" + _quantized_tag(float(synthons.pi_index(idx)), thresholds.bond_order_step),
            "pipi"
            + _quantized_tag(float(synthons.pi_pi_index(idx)), thresholds.bond_order_step),
        )
    )
    return _safe_class_name("_".join(parts))


def _bond_order_bin(value: float, step: float) -> str:
    return "bo" + _quantized_tag(value, step)


def _quantized_tag(value: float, step: float) -> str:
    if step <= 0.0:
        raise ValueError("Synthon quantization step must be positive")
    bucket = int(round(float(value) / float(step)))
    text = str(bucket).replace("-", "m")
    return text


def _safe_class_name(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw))
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = f"C_{text}"
    return text or "class"
