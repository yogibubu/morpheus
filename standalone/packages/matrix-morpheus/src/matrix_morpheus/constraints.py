"""Constraint parsing, evaluation and projection for MORPHEUS fits."""

from __future__ import annotations

import ast
import re

import numpy as np

from matrix_link import cartesian_from_internal_jacobian
from matrix_smith.survibfit.pipeline import b_matrix_analytic
from matrix_smith.survibfit.primitives import Primitive, eval_primitives

from .models import (
    GICExpressionConstraint,
    GICExpressionDefinition,
    PrimitiveLinearConstraint,
)


_GAUSSIAN_FREEZE_OPTIONS = {"f", "freeze", "frozen", "fixed"}
_GAUSSIAN_NONCONSTRAINT_OPTIONS = {
    "a",
    "active",
    "activate",
    "add",
    "d",
    "diff",
    "r",
    "remove",
    "inactive",
    "k",
    "kill",
    "removeall",
    "printonly",
    "modify",
    "unfreeze",
    "unfrozen",
}


def _gic_values(prims: object, u_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    return u_matrix.T @ eval_primitives(prims, coords)


def _is_linear_constraint_pattern(item: str) -> bool:
    return str(item).strip().lower().startswith("linear(")


def _primitives_from_fixed_pattern(pattern: str) -> tuple[Primitive, ...]:
    frozen_primitive = _primitives_from_gaussian_current_freeze(pattern)
    if frozen_primitive:
        return frozen_primitive
    text = str(pattern).strip().lower()
    if _top_level_value_marker(text) is not None:
        return ()
    if _first_top_level_equals(text) is not None:
        return ()
    match = re.match(
        r"^(r|b|bond|stretch|a|angle|bend|d|dihedral|torsion|u|out_of_plane|l|linear|linear_bend)\(([^)]*)",
        text,
    )
    if not match:
        return ()
    kind, args_text = match.groups()
    kind = {
        "r": "bond",
        "b": "bond",
        "stretch": "bond",
        "a": "angle",
        "bend": "angle",
        "d": "dihedral",
        "torsion": "dihedral",
        "u": "out_of_plane",
        "l": "linear_bend",
        "linear": "linear_bend",
    }.get(kind, kind)
    args = [part.strip() for part in re.split(r"[,;]", args_text) if part.strip()]
    values: list[int] = []
    mode: int | None = None
    for arg in args:
        if arg.startswith("mode="):
            try:
                mode = int(arg.split("=", 1)[1])
            except ValueError:
                return ()
            continue
        try:
            values.append(int(arg))
        except ValueError:
            return ()
    if kind == "bond" and len(values) >= 2:
        atoms = tuple(value - 1 for value in values[:2])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("bond", atoms),)
    if kind == "angle" and len(values) >= 3:
        atoms = tuple(value - 1 for value in values[:3])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("angle", atoms),)
    if kind == "dihedral" and len(values) >= 4:
        atoms = tuple(value - 1 for value in values[:4])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("dihedral", atoms),)
    if kind == "out_of_plane" and len(values) >= 4:
        atoms = tuple(value - 1 for value in values[:4])
        if any(atom < 0 for atom in atoms):
            return ()
        return (Primitive("out_of_plane", atoms),)
    if kind == "linear_bend" and len(values) >= 3:
        atoms = tuple(value - 1 for value in values[:3])
        if any(atom < 0 for atom in atoms):
            return ()
        if len(values) >= 5:
            mode = values[4]
        elif len(values) == 4 and values[3] in {-1, -2}:
            mode = values[3]
        if mode in {-1, -2}:
            return (Primitive("linear_bend", atoms, mode=mode),)
        return (
            Primitive("linear_bend", atoms, mode=-1),
            Primitive("linear_bend", atoms, mode=-2),
        )
    return ()


def _parse_linear_constraint_pattern(pattern: str) -> PrimitiveLinearConstraint | None:
    text = str(pattern).strip()
    if not _is_linear_constraint_pattern(text):
        return None
    if not text.endswith(")"):
        raise ValueError(f"Invalid linear primitive constraint: {pattern}")
    body = text[text.find("(") + 1 : -1].strip()
    if "=" not in body:
        raise ValueError(f"Linear primitive constraint needs '=': {pattern}")
    expr_text, target_text = body.rsplit("=", 1)
    terms = _parse_linear_constraint_terms(expr_text)
    if not terms:
        raise ValueError(f"Linear primitive constraint has no primitive terms: {pattern}")
    primitives = tuple(item[1] for item in terms)
    coefficients = tuple(item[0] for item in terms)
    angular = any(
        primitive.kind in {"angle", "dihedral", "out_of_plane", "linear_bend"}
        for primitive in primitives
    )
    if angular and any(primitive.kind == "bond" for primitive in primitives):
        raise ValueError(
            f"Linear primitive constraint cannot mix bond and angular primitives: {pattern}"
        )
    target = _parse_linear_constraint_target(target_text, angular=angular)
    return PrimitiveLinearConstraint(text, primitives, coefficients, target, angular)


def _parse_linear_constraint_terms(expr_text: str) -> list[tuple[float, Primitive]]:
    expr = re.sub(r"\s+", "", expr_text)
    if not expr:
        return []
    term_re = re.compile(
        r"(?P<sign>[+-]?)"
        r"(?:(?P<coeff>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*)?"
        r"(?P<kind>bond|angle|dihedral|out_of_plane|linear_bend)"
        r"\((?P<args>[^)]*)\)"
    )
    terms: list[tuple[float, Primitive]] = []
    pos = 0
    for match in term_re.finditer(expr):
        if match.start() != pos:
            raise ValueError(f"Invalid linear primitive expression near {expr[pos:]!r}")
        sign = -1.0 if match.group("sign") == "-" else 1.0
        coeff = float(match.group("coeff")) if match.group("coeff") else 1.0
        primitive_text = f"{match.group('kind')}({match.group('args')})"
        primitives = _primitives_from_fixed_pattern(primitive_text)
        if len(primitives) != 1:
            raise ValueError(
                f"Linear primitive terms must resolve to one primitive: {primitive_text}"
            )
        terms.append((sign * coeff, primitives[0]))
        pos = match.end()
    if pos != len(expr):
        raise ValueError(f"Invalid linear primitive expression near {expr[pos:]!r}")
    return terms


def _parse_linear_constraint_target(target_text: str, *, angular: bool) -> float:
    text = str(target_text).strip().lower()
    if not text:
        raise ValueError("Linear primitive constraint target cannot be empty")
    unit = ""
    if text.endswith("deg"):
        unit = "deg"
        text = text[:-3].strip()
    elif text.endswith("rad"):
        unit = "rad"
        text = text[:-3].strip()
    try:
        value = float(text.replace("d", "e").replace("D", "E"))
    except ValueError as exc:
        raise ValueError(f"Invalid linear primitive constraint target: {target_text}") from exc
    if angular and unit != "rad":
        return float(np.deg2rad(value))
    if not angular and unit in {"deg", "rad"}:
        raise ValueError("Bond linear constraints cannot use angular units")
    return value


def _parse_gic_expression_constraint_pattern(
    pattern: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    text = str(pattern).strip()
    if _is_linear_constraint_pattern(text):
        return None
    if _primitives_from_gaussian_current_freeze(text):
        return None
    wrapper = re.match(
        r"^(gic|constraint|freeze|fixed)\((.*)\)$", text, flags=re.IGNORECASE | re.DOTALL
    )
    if wrapper:
        return _parse_gic_expression_constraint_body(
            wrapper.group(2).strip(), text, definitions=definitions
        )
    named = _parse_gaussian_named_expression(text, definitions=definitions)
    if named is not None:
        return named
    expression_options = _parse_gaussian_expression_options(text, definitions=definitions)
    if expression_options is not None:
        return expression_options
    if _legacy_expression_target_split(text) is not None:
        return _parse_gic_expression_constraint_body(text, text, definitions=definitions)
    return None


def _parse_gic_expression_definition_pattern(pattern: str) -> GICExpressionDefinition | None:
    text = str(pattern).strip()
    if _is_linear_constraint_pattern(text):
        return None
    return _parse_gaussian_named_definition(text)


def _primitives_from_gaussian_current_freeze(pattern: str) -> tuple[Primitive, ...]:
    text = str(pattern).strip()
    if not text or _top_level_value_marker(text) is not None:
        return ()
    parsed = _parse_gaussian_named_expression(text) or _parse_gaussian_expression_options(text)
    if parsed is None or parsed.target is not None:
        return ()
    return _simple_primitives_from_gic_expression(parsed.expression)


def _simple_primitives_from_gic_expression(expression: str) -> tuple[Primitive, ...]:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return ()
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        return ()
    try:
        primitive = _primitive_from_gic_expression_call(
            tree.body.func.id,
            tree.body.args,
            tree.body.keywords,
            np.zeros((10000, 3), dtype=float),
            {},
        )
    except ValueError:
        return ()
    if primitive.kind == "linear_bend" and primitive.mode not in {-1, -2}:
        return (
            Primitive("linear_bend", primitive.atoms, mode=-1),
            Primitive("linear_bend", primitive.atoms, mode=-2),
        )
    return (primitive,)


def _parse_gic_expression_constraint_body(
    body: str,
    name: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint:
    named = _parse_gaussian_named_expression(body, definitions=definitions)
    if named is not None:
        return named
    expression_options = _parse_gaussian_expression_options(body, definitions=definitions)
    if expression_options is not None:
        return expression_options
    value_split = _split_value_option_from_expression(body, definitions=definitions)
    if value_split is not None:
        expression, target = value_split
    else:
        split_at = _legacy_expression_target_split(body)
        if split_at is None:
            expression = body
            target = None
        else:
            expression = body[:split_at].strip()
            target = _parse_expression_constraint_target(
                body[split_at + 1 :],
                angular_default=_gic_expression_uses_angular_default_units(
                    expression, definitions=definitions
                ),
            )
    expression = _strip_outer_square_brackets(expression)
    if not expression:
        raise ValueError(f"GIC expression constraint has no expression: {name}")
    _validate_gic_expression(expression)
    return GICExpressionConstraint(name=name, expression=expression, target=target)


def _split_value_option_from_expression(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> tuple[str, float] | None:
    marker = _top_level_value_marker(text)
    if marker is None:
        return None
    expression = text[:marker].strip(" \t,;")
    target, has_constraint = _parse_gaussian_constraint_options(
        text[marker:],
        angular_default=_gic_expression_uses_angular_default_units(
            expression, definitions=definitions
        ),
    )
    if not has_constraint or target is None:
        raise ValueError(f"Gaussian Value= constraint needs a numeric target: {text}")
    return expression, target


def _top_level_value_marker(text: str) -> int | None:
    lower = str(text).lower()
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "(":
            round_depth += 1
        elif char == ")" and round_depth > 0:
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
        if (
            round_depth == 0
            and square_depth == 0
            and brace_depth == 0
            and lower.startswith("value", idx)
        ):
            before_ok = idx == 0 or not (lower[idx - 1].isalnum() or lower[idx - 1] == "_")
            after = idx + len("value")
            probe = after
            while probe < len(text) and text[probe].isspace():
                probe += 1
            if before_ok and probe < len(text) and text[probe] == "=":
                return idx
        idx += 1
    return None


def _parse_gaussian_constraint_options(
    rest: str, *, angular_default: bool = False
) -> tuple[float | None, bool]:
    text = str(rest).strip()
    if not text:
        return None, False
    value_re = re.compile(
        r"(?i)\bvalue\s*=\s*"
        r"(?P<target>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][+-]?\d+)?(?:\s*(?:deg|rad))?)"
    )
    match = value_re.search(text)
    target: float | None = None
    cleaned = text
    if match:
        target = _parse_expression_constraint_target(
            match.group("target"), angular_default=angular_default
        )
        cleaned = text[: match.start()] + text[match.end() :]
    elif text.startswith("="):
        target = _parse_expression_constraint_target(text[1:], angular_default=angular_default)
        cleaned = ""
    has_constraint = target is not None
    saw_freeze = False
    saw_nonconstraint_action = False
    cleaned = cleaned.replace(",", " ").replace(";", " ").strip()
    leftovers: list[str] = []
    for token in cleaned.split():
        low = token.lower()
        option_name = low.split("=", 1)[0]
        if option_name in _GAUSSIAN_FREEZE_OPTIONS:
            saw_freeze = True
            has_constraint = True
            continue
        if option_name in _GAUSSIAN_NONCONSTRAINT_OPTIONS:
            saw_nonconstraint_action = True
            continue
        if option_name in {"fc", "forceconstant", "stepsize", "nsteps", "min", "max"}:
            continue
        leftovers.append(token)
    if leftovers:
        raise ValueError(f"Unsupported Gaussian GIC constraint option(s): {rest}")
    if saw_nonconstraint_action and not saw_freeze:
        has_constraint = False
    return target, has_constraint


def _parse_gaussian_expression_options(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    raw = str(text).strip()
    option_at = _first_top_level_gaussian_option(raw)
    if option_at is None:
        return None
    expression = _strip_outer_square_brackets(raw[:option_at].strip(" \t,;"))
    if not expression:
        return None
    try:
        target, has_constraint = _parse_gaussian_constraint_options(
            raw[option_at:],
            angular_default=_gic_expression_uses_angular_default_units(
                expression, definitions=definitions
            ),
        )
    except ValueError:
        return None
    if not has_constraint:
        return None
    _validate_gic_expression(expression)
    return GICExpressionConstraint(name=raw, expression=expression, target=target)


def _parse_gaussian_named_expression(
    text: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> GICExpressionConstraint | None:
    raw = str(text).strip()
    if not raw or "=" not in raw:
        return None
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return None
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return None
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
    except ValueError:
        return None
    expression = expression.strip()
    if not expression:
        raise ValueError(f"GIC expression constraint has no expression: {text}")
    _validate_gic_expression(expression)
    angular_default = _gic_expression_uses_angular_default_units(
        expression, definitions=definitions
    )
    target = None
    has_constraint = False
    option = (name_match.group("option") or "").strip()
    if option:
        try:
            target, has_constraint = _parse_gaussian_constraint_options(
                option, angular_default=angular_default
            )
        except ValueError:
            return None
    rest = rest.strip()
    if rest:
        try:
            rest_target, rest_has_constraint = _parse_gaussian_constraint_options(
                rest,
                angular_default=angular_default,
            )
        except ValueError:
            return None
        if rest_target is not None:
            target = rest_target
        has_constraint = has_constraint or rest_has_constraint
    if not has_constraint:
        return None
    return GICExpressionConstraint(
        name=name_match.group("name"), expression=expression, target=target
    )


def _parse_gaussian_named_definition(text: str) -> GICExpressionDefinition | None:
    raw = str(text).strip()
    if not raw or "=" not in raw:
        return None
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return None
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return None
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
    except ValueError:
        return None
    expression = _strip_outer_square_brackets(expression.strip())
    if not expression:
        return None
    try:
        _validate_gic_expression(expression)
        option = (name_match.group("option") or "").strip()
        angular_default = _gic_expression_uses_angular_default_units(expression)
        if option:
            _parse_gaussian_constraint_options(option, angular_default=angular_default)
        if rest.strip():
            _parse_gaussian_constraint_options(rest, angular_default=angular_default)
    except ValueError:
        return None
    return GICExpressionDefinition(name=name_match.group("name"), expression=expression)


def _is_gaussian_gic_definition_record(text: str) -> bool:
    raw = str(text).strip()
    eq_at = _first_top_level_equals(raw)
    if eq_at is None:
        return False
    left = raw[:eq_at].strip()
    right = raw[eq_at + 1 :].strip()
    name_match = re.match(
        r"^(?P<name>[A-Za-z_][A-Za-z0-9_'\"]*)(?:\((?P<option>.*)\))?$",
        left,
        flags=re.IGNORECASE,
    )
    if not name_match:
        return False
    try:
        expression, rest = _split_gaussian_named_expression_rhs(right)
        _validate_gic_expression(expression)
        option = (name_match.group("option") or "").strip()
        angular_default = _gic_expression_uses_angular_default_units(expression)
        if option:
            _parse_gaussian_constraint_options(option, angular_default=angular_default)
        if rest.strip():
            _parse_gaussian_constraint_options(rest, angular_default=angular_default)
    except ValueError:
        return False
    return True


def _split_gaussian_named_expression_rhs(text: str) -> tuple[str, str]:
    right = str(text).strip()
    if not right:
        raise ValueError("Gaussian GIC expression is empty")
    if right.startswith("["):
        return _extract_outer_square_brackets(right)
    option_at = _first_top_level_gaussian_option(right)
    if option_at is None:
        return right, ""
    return right[:option_at].strip(), right[option_at:].strip()


def _legacy_expression_target_split(text: str) -> int | None:
    split_at = _last_top_level_equals(text)
    if split_at is None:
        return None
    target_text = str(text)[split_at + 1 :]
    try:
        _parse_expression_constraint_target(target_text)
    except ValueError:
        return None
    return split_at


def _first_top_level_gaussian_option(text: str) -> int | None:
    for token, start, _end in _top_level_token_spans(text):
        if _is_gaussian_option_token(token):
            return start
    return None


def _top_level_token_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    token_start: int | None = None
    for idx, char in enumerate(str(text)):
        top = depth_round == 0 and depth_square == 0 and depth_brace == 0
        if top and (char.isspace() or char in ",;"):
            if token_start is not None:
                spans.append((text[token_start:idx], token_start, idx))
                token_start = None
        else:
            if token_start is None:
                token_start = idx
        if char == "(":
            depth_round += 1
        elif char == ")" and depth_round > 0:
            depth_round -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]" and depth_square > 0:
            depth_square -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}" and depth_brace > 0:
            depth_brace -= 1
    if token_start is not None:
        spans.append((text[token_start:], token_start, len(text)))
    return spans


def _is_gaussian_option_token(token: str) -> bool:
    low = str(token).strip().lower()
    if not low:
        return False
    name = low.split("=", 1)[0]
    return name in (
        _GAUSSIAN_FREEZE_OPTIONS
        | _GAUSSIAN_NONCONSTRAINT_OPTIONS
        | {"value", "fc", "forceconstant", "stepsize", "nsteps", "min", "max"}
    )


def _first_top_level_equals(text: str) -> int | None:
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "=" and depth_round == 0 and depth_square == 0 and depth_brace == 0:
            return idx
    return None


def _last_top_level_equals(text: str) -> int | None:
    positions = []
    depth_round = 0
    depth_square = 0
    depth_brace = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "=" and depth_round == 0 and depth_square == 0 and depth_brace == 0:
            positions.append(idx)
    return positions[-1] if positions else None


def _extract_outer_square_brackets(text: str) -> tuple[str, str]:
    if not text.startswith("["):
        raise ValueError(f"Expected '[' in GIC expression: {text}")
    depth = 0
    for idx, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[1:idx], text[idx + 1 :]
    raise ValueError(f"Unclosed '[' in GIC expression: {text}")


def _strip_outer_square_brackets(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("["):
        expression, rest = _extract_outer_square_brackets(stripped)
        if rest.strip():
            return stripped
        return expression.strip()
    return stripped


def _parse_expression_constraint_target(
    target_text: str, *, angular_default: bool = False
) -> float:
    text = str(target_text).strip()
    if not text:
        raise ValueError("GIC expression constraint target cannot be empty")
    lower = text.lower()
    unit = ""
    if lower.endswith("deg"):
        unit = "deg"
        text = text[:-3].strip()
    elif lower.endswith("rad"):
        unit = "rad"
        text = text[:-3].strip()
    try:
        value = float(text.replace("d", "e").replace("D", "E"))
    except ValueError as exc:
        raise ValueError(f"Invalid GIC expression constraint target: {target_text}") from exc
    if unit == "deg" or (angular_default and unit == ""):
        return float(np.deg2rad(value))
    return value


def _validate_gic_expression(expression: str) -> None:
    _parse_gic_expression_ast(expression)


def _parse_gic_expression_ast(expression: str) -> ast.Expression:
    normalized = _normalize_gaussian_gic_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid GIC expression syntax: {expression}") from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Expression
            | ast.Load
            | ast.BinOp
            | ast.UnaryOp
            | ast.Call
            | ast.Name
            | ast.Constant,
        ):
            continue
        if isinstance(node, ast.operator | ast.unaryop | ast.keyword):
            continue
        raise ValueError(f"Unsupported syntax in GIC expression: {expression}")
    return tree


def _normalize_gaussian_gic_expression(expression: str) -> str:
    """Normalize Gaussian grouping/call delimiters to Python AST syntax."""
    normalized = str(expression).strip().replace("^", "**")
    return normalized.translate(str.maketrans({"[": "(", "]": ")", "{": "(", "}": ")"}))


def _gic_expression_uses_angular_default_units(
    expression: str,
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> bool:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return False
    definition_map = _gic_expression_definition_map(definitions)
    visiting: set[str] = set()

    def name_is_angular(name: str) -> bool:
        upper = name.upper()
        if upper in visiting:
            return False
        definition = definition_map.get(name) or definition_map.get(upper)
        if definition is None:
            return False
        visiting.add(upper)
        try:
            return expression_is_angular(definition.body)
        finally:
            visiting.remove(upper)

    def expression_is_angular(root: ast.AST) -> bool:
        has_angular = False
        call_function_nodes = {
            id(node.func)
            for node in ast.walk(root)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                if id(node) in call_function_nodes:
                    continue
                if node.id in {"pi", "PI"}:
                    continue
                if name_is_angular(node.id):
                    has_angular = True
                    continue
                return False
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                return False
            kind = node.func.id.lower()
            if kind in {
                "a",
                "angle",
                "bend",
                "d",
                "dihedral",
                "torsion",
                "u",
                "out_of_plane",
                "l",
                "linear",
                "linear_bend",
            }:
                has_angular = True
                continue
            if kind in {"r", "b", "bond", "stretch"}:
                return False
            if kind in {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "arcsin",
                "arccos",
                "arctan",
                "sqrt",
                "exp",
                "log",
                "abs",
                "min",
                "max",
            }:
                return False
            return False
        return has_angular

    return expression_is_angular(tree.body)


def _gic_expression_uses_gic_names(expression: str) -> bool:
    try:
        tree = _parse_gic_expression_ast(expression)
    except ValueError:
        return bool(re.search(r"\bGIC\d+\b", expression, flags=re.IGNORECASE))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and re.match(r"^GIC\d+$", node.id, flags=re.IGNORECASE):
            return True
    return False


def _gic_expression_constraint_targets(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    if not constraints:
        return np.zeros(0, dtype=float)
    current = _gic_expression_constraint_values(
        constraints, coords, prims, u_matrix, labels, definitions=definitions
    )
    targets = [
        float(constraint.target) if constraint.target is not None else float(current[idx])
        for idx, constraint in enumerate(constraints)
    ]
    return np.asarray(targets, dtype=float)


def _gic_expression_constraint_values(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    values = [
        _evaluate_gic_expression(
            constraint.expression, coords, prims, u_matrix, labels, definitions=definitions
        )
        for constraint in constraints
    ]
    return np.asarray(values, dtype=float)


def _gic_expression_constraint_residuals(
    constraints: tuple[GICExpressionConstraint, ...],
    targets: np.ndarray,
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    target = np.asarray(targets, dtype=float)
    if target.size != len(constraints):
        raise ValueError("GIC expression target size does not match constraints")
    return target - _gic_expression_constraint_values(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def _gic_expression_constraint_b_matrix(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    step: float = 1.0e-5,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    if not constraints:
        return np.zeros((0, np.asarray(coords, dtype=float).size), dtype=float)
    base = np.asarray(coords, dtype=float)
    flat = base.reshape(-1)
    rows = np.zeros((len(constraints), flat.size), dtype=float)
    for idx in range(flat.size):
        delta = step * max(1.0, abs(float(flat[idx])))
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += delta
        minus[idx] -= delta
        values_plus = _gic_expression_constraint_values(
            constraints,
            plus.reshape(base.shape),
            prims,
            u_matrix,
            labels,
            definitions=definitions,
        )
        values_minus = _gic_expression_constraint_values(
            constraints,
            minus.reshape(base.shape),
            prims,
            u_matrix,
            labels,
            definitions=definitions,
        )
        rows[:, idx] = (values_plus - values_minus) / (2.0 * delta)
    return rows


def _evaluate_gic_expression(
    expression: str,
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> float:
    tree = _parse_gic_expression_ast(expression)
    q_values = _gic_values(prims, u_matrix, coords) if len(labels) else np.zeros(0, dtype=float)
    symbols = _gic_expression_symbol_values(labels, q_values)
    value = _evaluate_gic_expression_node(
        tree.body,
        np.asarray(coords, dtype=float),
        symbols,
        _gic_expression_definition_map(definitions),
    )
    if not np.isfinite(value):
        raise ValueError(f"Non-finite GIC expression value: {expression}")
    return float(value)


def _gic_expression_symbol_values(
    labels: tuple[str, ...], q_values: np.ndarray
) -> dict[str, float]:
    symbols: dict[str, float] = {"pi": float(np.pi), "PI": float(np.pi)}
    for idx, value in enumerate(np.asarray(q_values, dtype=float), start=1):
        default = f"GIC{idx:03d}"
        for key in {default, f"GIC{idx}"}:
            symbols[key] = float(value)
            symbols[key.upper()] = float(value)
        label = labels[idx - 1] if idx - 1 < len(labels) else ""
        label_match = re.match(r"\s*(GIC\d+)\b", label)
        if label_match:
            key = label_match.group(1)
            symbols[key] = float(value)
            symbols[key.upper()] = float(value)
        name_match = re.search(r"\bGICForge\s+([A-Za-z0-9_'\"]+)", label)
        if name_match:
            raw_name = name_match.group(1)
            for key in {raw_name, _safe_gic_symbol(raw_name)}:
                if key and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                    symbols[key] = float(value)
                    symbols[key.upper()] = float(value)
    return symbols


def _safe_gic_symbol(name: str) -> str:
    safe = re.sub(r"\W+", "_", str(name).strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe and safe[0].isdigit():
        safe = f"GIC_{safe}"
    return safe


def _gic_expression_definition_map(
    definitions: tuple[GICExpressionDefinition, ...],
) -> dict[str, ast.Expression]:
    mapped: dict[str, ast.Expression] = {}
    for definition in definitions:
        tree = _parse_gic_expression_ast(definition.expression)
        for key in {definition.name, definition.name.upper(), _safe_gic_symbol(definition.name)}:
            if key:
                mapped[key] = tree
                mapped[key.upper()] = tree
    return mapped


def _evaluate_gic_expression_node(
    node: ast.AST,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("GIC expressions only support numeric constants")
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return float(symbols[node.id])
        upper = node.id.upper()
        if upper in symbols:
            return float(symbols[upper])
        definition = (definitions or {}).get(node.id) or (definitions or {}).get(upper)
        if definition is not None:
            key = upper
            if key in stack:
                cycle = " -> ".join((*stack, key))
                raise ValueError(f"Cyclic GIC expression definition: {cycle}")
            value = _evaluate_gic_expression_node(
                definition.body,
                coords,
                symbols,
                definitions,
                (*stack, key),
            )
            symbols[node.id] = float(value)
            symbols[upper] = float(value)
            return float(value)
        raise ValueError(f"Unknown GIC expression symbol: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_gic_expression_node(node.operand, coords, symbols, definitions, stack)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Unsupported unary operator in GIC expression")
    if isinstance(node, ast.BinOp):
        left = _evaluate_gic_expression_node(node.left, coords, symbols, definitions, stack)
        right = _evaluate_gic_expression_node(node.right, coords, symbols, definitions, stack)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator in GIC expression")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("GIC expression functions must be named functions")
        name = node.func.id
        return _evaluate_gic_expression_call(
            name, node.args, node.keywords, coords, symbols, definitions, stack
        )
    raise ValueError("Unsupported syntax in GIC expression")


def _evaluate_gic_expression_call(
    name: str,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> float:
    lowered = name.lower()
    unary_functions = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "arcsin": "asin",
        "acos": "acos",
        "arccos": "acos",
        "atan": "atan",
        "arctan": "atan",
        "sqrt": "sqrt",
        "exp": "exp",
        "log": "log",
    }
    if lowered in unary_functions:
        if keywords or len(args) != 1:
            raise ValueError(f"Function {name} expects one positional argument")
        value = _evaluate_gic_expression_node(args[0], coords, symbols, definitions, stack)
        return float(getattr(np, unary_functions[lowered])(value))
    if lowered == "abs":
        if keywords or len(args) != 1:
            raise ValueError("Function abs expects one positional argument")
        return abs(_evaluate_gic_expression_node(args[0], coords, symbols, definitions, stack))
    if lowered in {"min", "max"}:
        if keywords or len(args) < 1:
            raise ValueError(f"Function {name} expects positional arguments")
        values = [
            _evaluate_gic_expression_node(arg, coords, symbols, definitions, stack) for arg in args
        ]
        return float(min(values) if lowered == "min" else max(values))
    if lowered in {"x", "y", "z"}:
        if keywords or len(args) != 1:
            raise ValueError(f"Cartesian function {name} expects one atom index")
        atom = _gic_expression_atom_index(args[0], coords, symbols, definitions, stack)
        axis = {"x": 0, "y": 1, "z": 2}[lowered]
        return float(coords[atom, axis])
    if lowered in {"cart", "cartesian"}:
        if keywords or len(args) != 2:
            raise ValueError(f"Cartesian function {name} expects atom index and axis")
        atom = _gic_expression_atom_index(args[0], coords, symbols, definitions, stack)
        axis = _gic_expression_cartesian_axis(args[1], coords, symbols, definitions, stack)
        return float(coords[atom, axis])
    if lowered == "dotdiff":
        if keywords or len(args) != 4:
            raise ValueError("DotDiff expects four atom indices")
        i, j, k, l = [
            _gic_expression_atom_index(arg, coords, symbols, definitions, stack) for arg in args
        ]
        return float(np.dot(coords[i] - coords[j], coords[k] - coords[l]))
    primitive = _primitive_from_gic_expression_call(
        name, args, keywords, coords, symbols, definitions, stack
    )
    return float(eval_primitives([primitive], coords)[0])


def _gic_expression_atom_index(
    node: ast.expr,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> int:
    value = _evaluate_gic_expression_node(node, coords, symbols, definitions, stack)
    index = int(round(value))
    if abs(value - index) > 1.0e-10 or index < 1 or index > len(coords):
        raise ValueError(f"Invalid atom index in GIC expression: {value}")
    return index - 1


def _gic_expression_cartesian_axis(
    node: ast.expr,
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> int:
    if isinstance(node, ast.Name):
        axis_name = node.id.lower()
        if axis_name in {"x", "y", "z"}:
            return {"x": 0, "y": 1, "z": 2}[axis_name]
    value = _evaluate_gic_expression_node(node, coords, symbols, definitions, stack)
    axis = int(round(value))
    if abs(value - axis) > 1.0e-10:
        raise ValueError(f"Invalid Cartesian axis in GIC expression: {value}")
    if axis in {-1, 1}:
        return 0
    if axis in {-2, 2}:
        return 1
    if axis in {-3, 3}:
        return 2
    raise ValueError(f"Invalid Cartesian axis in GIC expression: {value}")


def _primitive_from_gic_expression_call(
    name: str,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    coords: np.ndarray,
    symbols: dict[str, float],
    definitions: dict[str, ast.Expression] | None = None,
    stack: tuple[str, ...] = (),
) -> Primitive:
    lowered = name.lower()
    numeric_args = [
        _evaluate_gic_expression_node(arg, coords, symbols, definitions, stack) for arg in args
    ]
    int_args = [int(round(value)) for value in numeric_args]
    if any(abs(value - round(value)) > 1.0e-10 for value in numeric_args):
        raise ValueError(f"Primitive function {name} needs integer atom indices")

    def atoms(count: int) -> tuple[int, ...]:
        if len(int_args) < count:
            raise ValueError(f"Primitive function {name} needs {count} atom indices")
        result = tuple(value - 1 for value in int_args[:count])
        if any(atom < 0 for atom in result):
            raise ValueError(f"Primitive function {name} has invalid atom index")
        return result

    if lowered in {"r", "b", "bond", "stretch"}:
        if keywords or len(int_args) != 2:
            raise ValueError(f"Primitive function {name} expects two atoms")
        return Primitive("bond", atoms(2))
    if lowered in {"a", "angle", "bend"}:
        if keywords or len(int_args) != 3:
            raise ValueError(f"Primitive function {name} expects three atoms")
        return Primitive("angle", atoms(3))
    if lowered in {"d", "dihedral", "torsion"}:
        if keywords or len(int_args) != 4:
            raise ValueError(f"Primitive function {name} expects four atoms")
        return Primitive("dihedral", atoms(4))
    if lowered in {"u", "out_of_plane"}:
        if keywords or len(int_args) != 4:
            raise ValueError(f"Primitive function {name} expects four atoms")
        return Primitive("out_of_plane", atoms(4))
    if lowered in {"l", "linear", "linear_bend"}:
        mode = None
        for keyword in keywords:
            if keyword.arg != "mode":
                raise ValueError(f"Unsupported keyword for {name}: {keyword.arg}")
            mode = int(
                round(
                    _evaluate_gic_expression_node(
                        keyword.value, coords, symbols, definitions, stack
                    )
                )
            )
        if len(int_args) == 5:
            mode = int_args[4]
        elif len(int_args) == 4:
            mode = int_args[3]
        elif len(int_args) != 3:
            raise ValueError(f"Primitive function {name} expects three atoms and a mode")
        if mode not in {-1, -2}:
            raise ValueError(f"Linear-bend function {name} needs mode -1 or -2")
        return Primitive("linear_bend", atoms(3), mode=mode)
    raise ValueError(f"Unsupported GIC expression function: {name}")


def _primitive_constraint_key(primitive: Primitive) -> tuple[str, tuple[int, ...], int]:
    atoms = primitive.atoms
    if primitive.kind == "bond" and len(atoms) == 2:
        atoms = tuple(sorted(atoms))
    elif primitive.kind in {"angle", "dihedral"}:
        reverse = tuple(reversed(atoms))
        atoms = min(atoms, reverse)
    elif primitive.kind == "out_of_plane" and len(atoms) == 4:
        atoms = (atoms[0], *tuple(sorted(atoms[1:])))
    elif primitive.kind == "linear_bend" and len(atoms) == 3:
        atoms = (atoms[1], *tuple(sorted((atoms[0], atoms[2]))))
    return (primitive.kind, tuple(atoms), primitive.mode)


def _fixed_primitive_targets(
    fixed_primitives: tuple[Primitive, ...], coords: np.ndarray
) -> np.ndarray:
    if not fixed_primitives:
        return np.zeros(0, dtype=float)
    return np.asarray(
        eval_primitives(list(fixed_primitives), np.asarray(coords, dtype=float)), dtype=float
    )


def _project_fixed_primitives(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    target_values: np.ndarray,
    *,
    tolerance: float = 1.0e-11,
    max_iter: int = 10,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...] = (),
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    """Project a candidate geometry back onto fixed primitive values."""
    if not fixed_primitives and not linear_constraints and not expression_constraints:
        return np.asarray(coords, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if target.size != len(fixed_primitives):
        raise ValueError("Fixed-primitive target size does not match constraints")
    expression_target_values = (
        np.asarray(expression_targets, dtype=float)
        if expression_targets is not None
        else _gic_expression_constraint_targets(
            expression_constraints,
            coords,
            prims,
            np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
            labels,
            definitions=expression_definitions,
        )
    )
    projected = np.asarray(coords, dtype=float).copy()
    projection_max_iter = max(max_iter, 20) if expression_constraints else max_iter
    for _ in range(projection_max_iter):
        residual = _combined_primitive_constraint_residual(
            projected,
            fixed_primitives,
            target,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_target_values,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        if not np.all(np.isfinite(residual)):
            raise ValueError("Non-finite fixed-primitive residual")
        if float(np.max(np.abs(residual))) <= tolerance:
            return projected
        b_fixed = _combined_primitive_constraint_b_matrix(
            projected,
            fixed_primitives,
            linear_constraints,
            expression_constraints=expression_constraints,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        dx = cartesian_from_internal_jacobian(b_fixed, rcond=1.0e-10) @ residual
        if not np.all(np.isfinite(dx)):
            raise ValueError("Non-finite fixed-primitive projection step")
        projected = projected + dx.reshape(projected.shape)
    residual = _combined_primitive_constraint_residual(
        projected,
        fixed_primitives,
        target,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_target_values,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=expression_definitions,
    )
    if float(np.max(np.abs(residual))) > 100.0 * tolerance:
        raise ValueError("Fixed-primitive projection did not converge")
    return projected


def _primitive_constraint_residual(
    fixed_primitives: tuple[Primitive, ...],
    current: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    for idx, primitive in enumerate(fixed_primitives):
        if primitive.kind in {"dihedral", "out_of_plane"}:
            residual[idx] = (residual[idx] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def _combined_primitive_constraint_residual(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if fixed_primitives:
        current = np.asarray(eval_primitives(list(fixed_primitives), coords), dtype=float)
        parts.append(_primitive_constraint_residual(fixed_primitives, current, fixed_targets))
    if linear_constraints:
        parts.append(_linear_constraint_residuals(linear_constraints, coords))
    if expression_constraints:
        if expression_targets is None:
            raise ValueError("GIC expression constraints need target values")
        parts.append(
            _gic_expression_constraint_residuals(
                expression_constraints,
                expression_targets,
                coords,
                prims,
                np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
                labels,
                definitions=expression_definitions,
            )
        )
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def _combined_primitive_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    rows: list[np.ndarray] = []
    if fixed_primitives:
        rows.append(b_matrix_analytic(list(fixed_primitives), coords))
    if linear_constraints:
        rows.append(_linear_constraint_b_matrix(linear_constraints, coords))
    if expression_constraints:
        rows.append(
            _gic_expression_constraint_b_matrix(
                expression_constraints,
                coords,
                prims,
                np.asarray(u_matrix if u_matrix is not None else np.zeros((0, 0)), dtype=float),
                labels,
                definitions=expression_definitions,
            )
        )
    if not rows:
        return np.zeros((0, np.asarray(coords).size), dtype=float)
    return np.vstack(rows)


def _finite_difference_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    expression_definitions: tuple[GICExpressionDefinition, ...] = (),
    step: float = 1.0e-6,
) -> np.ndarray:
    """Finite-difference derivative of combined constraint values.

    `_combined_primitive_constraint_b_matrix` returns derivatives of the
    constrained values.  The residual is target minus current value, therefore
    the finite-difference residual derivative has the opposite sign.
    """
    base = np.asarray(coords, dtype=float)
    flat = base.reshape(-1)
    residual0 = _combined_primitive_constraint_residual(
        base,
        fixed_primitives,
        fixed_targets,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=expression_definitions,
    )
    rows = np.zeros((residual0.size, flat.size), dtype=float)
    if residual0.size == 0:
        return rows
    for idx in range(flat.size):
        delta = float(step) * max(1.0, abs(float(flat[idx])))
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += delta
        minus[idx] -= delta
        residual_plus = _combined_primitive_constraint_residual(
            plus.reshape(base.shape),
            fixed_primitives,
            fixed_targets,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        residual_minus = _combined_primitive_constraint_residual(
            minus.reshape(base.shape),
            fixed_primitives,
            fixed_targets,
            linear_constraints,
            expression_constraints=expression_constraints,
            expression_targets=expression_targets,
            prims=prims,
            u_matrix=u_matrix,
            labels=labels,
            expression_definitions=expression_definitions,
        )
        rows[:, idx] = -(residual_plus - residual_minus) / (2.0 * delta)
    return rows


def _linear_constraint_values(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    values = []
    for constraint in constraints:
        primitive_values = np.asarray(
            eval_primitives(list(constraint.primitives), coords), dtype=float
        )
        coeffs = np.asarray(constraint.coefficients, dtype=float)
        values.append(float(coeffs @ primitive_values))
    return np.asarray(values, dtype=float)


def _linear_constraint_residuals(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    current = _linear_constraint_values(constraints, coords)
    target = np.asarray([constraint.target for constraint in constraints], dtype=float)
    residual = target - current
    for idx, constraint in enumerate(constraints):
        if constraint.angular:
            residual[idx] = (residual[idx] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def _linear_constraint_b_matrix(
    constraints: tuple[PrimitiveLinearConstraint, ...],
    coords: np.ndarray,
) -> np.ndarray:
    rows = []
    for constraint in constraints:
        primitive_b = b_matrix_analytic(list(constraint.primitives), coords)
        coeffs = np.asarray(constraint.coefficients, dtype=float)
        rows.append(coeffs @ primitive_b)
    if not rows:
        return np.zeros((0, np.asarray(coords).size), dtype=float)
    return np.vstack(rows)


def _fixed_primitives_from_patterns(fixed: tuple[str, ...]) -> tuple[Primitive, ...]:
    primitives: list[Primitive] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()
    for item in fixed:
        for primitive in _primitives_from_fixed_pattern(item):
            key = _primitive_constraint_key(primitive)
            if key in seen:
                continue
            primitives.append(primitive)
            seen.add(key)
    return tuple(primitives)


def _linear_primitive_constraints_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[PrimitiveLinearConstraint, ...]:
    constraints: list[PrimitiveLinearConstraint] = []
    for item in fixed:
        parsed = _parse_linear_constraint_pattern(item)
        if parsed is not None:
            constraints.append(parsed)
    return tuple(constraints)


def _gic_expression_constraints_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[GICExpressionConstraint, ...]:
    constraints: list[GICExpressionConstraint] = []
    definitions = _gic_expression_definitions_from_patterns(fixed)
    for item in fixed:
        parsed = _parse_gic_expression_constraint_pattern(item, definitions=definitions)
        if parsed is not None:
            constraints.append(parsed)
    return tuple(constraints)


def _gic_expression_definitions_from_patterns(
    fixed: tuple[str, ...],
) -> tuple[GICExpressionDefinition, ...]:
    definitions: list[GICExpressionDefinition] = []
    seen: set[str] = set()
    for item in fixed:
        parsed = _parse_gic_expression_definition_pattern(item)
        if parsed is None:
            continue
        key = parsed.name.lower()
        if key in seen:
            definitions = [
                definition for definition in definitions if definition.name.lower() != key
            ]
        definitions.append(parsed)
        seen.add(key)
    return tuple(definitions)


def parse_gaussian_style_constraints(
    records: tuple[str, ...] | list[str],
) -> tuple[
    tuple[Primitive, ...],
    tuple[PrimitiveLinearConstraint, ...],
    tuple[GICExpressionConstraint, ...],
    tuple[GICExpressionDefinition, ...],
]:
    """Parse MATRIX/Gaussian-style primitive, linear and expression constraints."""
    items = tuple(str(item) for item in records)
    definitions = _gic_expression_definitions_from_patterns(items)
    return (
        _fixed_primitives_from_patterns(items),
        _linear_primitive_constraints_from_patterns(items),
        _gic_expression_constraints_from_patterns(items),
        definitions,
    )


def gic_expression_constraint_values(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _gic_expression_constraint_values(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def gic_expression_constraint_targets(
    constraints: tuple[GICExpressionConstraint, ...],
    coords: np.ndarray,
    prims: object,
    u_matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _gic_expression_constraint_targets(
        constraints,
        coords,
        prims,
        u_matrix,
        labels,
        definitions=definitions,
    )


def combined_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    definitions: tuple[GICExpressionDefinition, ...] = (),
) -> np.ndarray:
    return _combined_primitive_constraint_b_matrix(
        coords,
        fixed_primitives,
        linear_constraints,
        expression_constraints=expression_constraints,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=definitions,
    )


def finite_difference_constraint_b_matrix(
    coords: np.ndarray,
    fixed_primitives: tuple[Primitive, ...],
    fixed_targets: np.ndarray,
    linear_constraints: tuple[PrimitiveLinearConstraint, ...],
    *,
    expression_constraints: tuple[GICExpressionConstraint, ...] = (),
    expression_targets: np.ndarray | None = None,
    prims: object = (),
    u_matrix: np.ndarray | None = None,
    labels: tuple[str, ...] = (),
    definitions: tuple[GICExpressionDefinition, ...] = (),
    step: float = 1.0e-6,
) -> np.ndarray:
    return _finite_difference_constraint_b_matrix(
        coords,
        fixed_primitives,
        fixed_targets,
        linear_constraints,
        expression_constraints=expression_constraints,
        expression_targets=expression_targets,
        prims=prims,
        u_matrix=u_matrix,
        labels=labels,
        expression_definitions=definitions,
        step=step,
    )


__all__ = [
    "GICExpressionConstraint",
    "GICExpressionDefinition",
    "PrimitiveLinearConstraint",
    "combined_constraint_b_matrix",
    "finite_difference_constraint_b_matrix",
    "gic_expression_constraint_targets",
    "gic_expression_constraint_values",
    "parse_gaussian_style_constraints",
]
