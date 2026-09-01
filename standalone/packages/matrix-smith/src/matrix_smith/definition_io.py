"""Shared section and symmetry readers for frozen SMITH definitions."""

from __future__ import annotations

from matrix_core import section_content

from .contracts import GICForgeContractError
from .models import GICPointGroupOperation


def _point_group(lines: list[str]) -> str:
    for line in section_content(lines, "SYMMETRY"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "POINT_GROUP":
            return parts[1]
    return "UNKNOWN"


def _apply_symmetry_group_limit(
    point_group: str,
    operations: tuple[GICPointGroupOperation, ...],
    symmetry_group: str | None,
) -> tuple[str, tuple[GICPointGroupOperation, ...]]:
    target = (symmetry_group or "").strip()
    if not target:
        return point_group, operations
    if point_group.upper() == target.upper():
        return point_group, operations
    if not operations:
        if target.upper() == "C1":
            return "C1", ()
        raise GICForgeContractError(
            f"cannot reduce {point_group} to {target}: no stored symmetry operations"
        )
    identity = _identity_symmetry_operation(operations)
    target_key = target.upper()
    if target_key == "C1":
        return "C1", (identity,)
    keep = [identity]
    if target_key == "CS":
        match = next(
            (operation for operation in operations if _is_sigma_operation(operation)), None
        )
    elif target_key == "CI":
        match = next((operation for operation in operations if operation.label == "i"), None)
    elif target_key == "C2":
        match = next(
            (operation for operation in operations if operation.label.startswith("C2")), None
        )
    else:
        raise GICForgeContractError(
            f"unsupported reduced symmetry group {target!r}; "
            "supported limits are C1, Cs, Ci, C2, or the full point group"
        )
    if match is None:
        raise GICForgeContractError(
            f"cannot reduce {point_group} to {target}: no compatible stored operation"
        )
    keep.append(match)
    return target, tuple(keep)


def _identity_symmetry_operation(
    operations: tuple[GICPointGroupOperation, ...],
) -> GICPointGroupOperation:
    match = next((operation for operation in operations if operation.label == "E"), None)
    if match is not None:
        return match
    natoms = len(operations[0].permutation) if operations else 0
    return GICPointGroupOperation(
        label="E",
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        permutation=tuple(range(1, natoms + 1)),
    )


def _is_sigma_operation(operation: GICPointGroupOperation) -> bool:
    return operation.label.startswith("sigma")


def _symmetry_operations(lines: list[str]) -> tuple[GICPointGroupOperation, ...]:
    symmetry = section_content(lines, "SYMMETRY")
    if not symmetry:
        return ()
    operation_lines = _subsection(symmetry, "OPERATIONS")
    if not operation_lines:
        return ()
    operations: list[GICPointGroupOperation] = []
    for line in operation_lines:
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        parts = text.split()
        fields = _key_values(parts[1:])
        try:
            matrix_values = _parse_float_list(fields["MATRIX"])
            if len(matrix_values) != 9:
                raise ValueError("operation matrix must have 9 values")
            permutation = _parse_atom_list(fields["PERMUTATION"])
            operations.append(
                GICPointGroupOperation(
                    label=fields["LABEL"],
                    rotation=tuple(
                        tuple(float(value) for value in matrix_values[start : start + 3])
                        for start in (0, 3, 6)
                    ),
                    permutation=permutation,
                )
            )
        except KeyError as exc:
            raise GICForgeContractError(f"invalid #SYMMETRY operation line: {line}") from exc
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid #SYMMETRY operation numeric field: {line}"
            ) from exc
    return tuple(operations)


def _subsection(section_lines: list[str], name: str) -> list[str]:
    header = f"[{name.upper()}]"
    start = None
    for idx, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for idx in range(start, len(section_lines)):
        text = section_lines[idx].strip()
        if text.startswith("[") and text.endswith("]"):
            end = idx
            break
    return list(section_lines[start:end])


def _key_values(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.upper()] = value
    return fields


def _parse_atom_list(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    try:
        return tuple(int(item) for item in text.split(",") if item)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid atom list: {text}") from exc


def _parse_float_list(text: str) -> tuple[float, ...]:
    if not text:
        return ()
    try:
        return tuple(float(item) for item in text.split(",") if item)
    except ValueError as exc:
        raise GICForgeContractError(f"invalid float list: {text}") from exc
