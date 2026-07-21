from __future__ import annotations

import math
from pathlib import Path
import re
from matrix_chem import read_enriched_xyz
from matrix_core import build_run_manifest
from matrix_core import read_sectioned_lines, section_content


ORACLE_GAUSSIAN_GIC_INPUT_SCHEMA = "oracle.gaussian.gic_input.v1"
REQUIRED_GIC_SCHEMA = "oracle.xyz.gic.v1"
DEFAULT_GIC_ROUTE = "#p hf/sto-3g opt=readallgic"
DEFAULT_SEMIDIAGONAL_ROUTE = "#p B3LYP/6-31G(d)"
DEFAULT_POINT_ROUTE = "#p HF/STO-3G Force"
_GEOM_PAREN_RE = re.compile(r"\bgeom\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE)
_GEOM_VALUE_RE = re.compile(
    r"\bgeom\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_OUTPUT_RE = re.compile(
    r"\boutput\s*=\s*(?:\([^)]*\)|[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_POP_PAREN_RE = re.compile(
    r"\bpop(?:ulation)?\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE
)
_POP_VALUE_RE = re.compile(
    r"\bpop(?:ulation)?\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    flags=re.IGNORECASE,
)
_MAYER_IOP_RE = re.compile(r"\biop\s*\(\s*6\s*/\s*80\s*=\s*1\s*\)", flags=re.IGNORECASE)
_OPT_PAREN_RE = re.compile(r"\bopt\s*=\s*\((?P<body>[^)]*)\)", flags=re.IGNORECASE)
_OPT_VALUE_RE = re.compile(r"\bopt\s*=\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)", flags=re.IGNORECASE)
_OPT_BARE_RE = re.compile(r"\bopt\b(?!\s*=)", flags=re.IGNORECASE)
_READALLGIC_OPTION_RE = re.compile(r"\breadallgic\b", flags=re.IGNORECASE)
_GICALLSYM_RE = re.compile(r"\bgic(?:all)?symm?\b", flags=re.IGNORECASE)
_GIC_VALUE_RE = re.compile(r"\bvalue\s*=", flags=re.IGNORECASE)
_GIC_LABEL_OPTION_RE = re.compile(r"\((?:Frozen|Inactive|Fl[A-Za-z]+|Value\s*=[^)]*)\)", flags=re.IGNORECASE)
_GIC_FUNCTION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)\s*\(([^()]*)\)")
_GIC_DIHEDRAL_RE = re.compile(r"\bD\s*\([^()]*\)", flags=re.IGNORECASE)
_GIC_OUT_OF_PLANE_RE = re.compile(r"\bU\s*\(([^()]*)\)", flags=re.IGNORECASE)
_GIC_NONACTIVE_RE = re.compile(r"\b(?:frozen|freeze|inactive|remove|printonly)\b", flags=re.IGNORECASE)
_GIC_INACTIVE_RE = re.compile(r"\b(?:inactive|remove|printonly)\b", flags=re.IGNORECASE)
ANGSTROM_TO_BOHR = 1.8897261246257702


class GaussianWriteError(ValueError):
    """Raised when ORACLE state cannot be exported to Gaussian input."""


def write_gaussian_point_input(
    output: Path,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    route: str = DEFAULT_POINT_ROUTE,
    title: str = "MATRIX LINK point",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
    ensure_force: bool = True,
    connectivity_bonds: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]] | None = None,
) -> Path:
    """Write a Cartesian Gaussian single-point/force input."""

    import numpy as np

    target = Path(output)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GaussianWriteError("point coordinates must have shape natoms x 3")
    if len(atoms) != coords.shape[0]:
        raise GaussianWriteError("atom count does not match coordinate array")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        _point_route_with_connectivity(
            _normalize_point_route(route, ensure_force=ensure_force),
            connectivity_bonds is not None,
        ),
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
    ]
    for atom, xyz in zip(atoms, coords, strict=True):
        lines.append(
            f"{atom:<3s} {float(xyz[0]): .10f} {float(xyz[1]): .10f} {float(xyz[2]): .10f}"
        )
    if connectivity_bonds is not None:
        lines.append("")
        lines.extend(_gaussian_connectivity_lines(len(atoms), connectivity_bonds))
    lines.append("")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _point_route_with_connectivity(route: str, enabled: bool) -> str:
    if not enabled or re.search(r"\bgeom\s*=", route, flags=re.IGNORECASE):
        return route
    return f"{route} Geom=Connectivity"


def _gaussian_connectivity_lines(
    natoms: int,
    bonds: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]],
) -> list[str]:
    neighbors: dict[int, list[tuple[int, float]]] = {index: [] for index in range(1, natoms + 1)}
    for left_zero, right_zero, order in bonds:
        left, right = int(left_zero) + 1, int(right_zero) + 1
        if left not in neighbors or right not in neighbors or left == right:
            raise GaussianWriteError("connectivity bond indices are invalid")
        neighbors[min(left, right)].append((max(left, right), float(order)))
    lines = []
    for atom in range(1, natoms + 1):
        suffix = " ".join(
            f"{other} {order:.6g}" for other, order in sorted(neighbors[atom])
        )
        lines.append(f"{atom} {suffix}".rstrip())
    return lines


def write_gaussian_oniom_point_input(
    output: Path,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    high_atoms: tuple[int, ...] | list[int],
    atom_types: tuple[str, ...] | list[str] = (),
    route: str,
    title: str = "MATRIX LINK ONIOM point",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
    ensure_force: bool = True,
) -> Path:
    """Write a two-layer Cartesian ONIOM force point.

    high_atoms uses zero-based MATRIX indices. Atom types default to a
    conservative UFF mapping for common organic and microsolvation elements.
    """

    import numpy as np

    target = Path(output)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3):
        raise GaussianWriteError("ONIOM point coordinates must have shape natoms x 3")
    high = {int(index) for index in high_atoms}
    if not high or min(high) < 0 or max(high) >= len(atoms):
        raise GaussianWriteError("ONIOM high-layer atom indices are invalid")
    types = tuple(atom_types) if atom_types else tuple(_default_uff_type(atom) for atom in atoms)
    if len(types) != len(atoms):
        raise GaussianWriteError("ONIOM atom type count must match atoms")
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        _normalize_point_route(route, ensure_force=ensure_force),
        "",
        title,
        "",
        f"{int(charge)} {int(multiplicity)}",
    ]
    for index, (atom, atom_type, xyz) in enumerate(zip(atoms, types, coords, strict=True)):
        layer = "H" if index in high else "L"
        lines.append(
            f"{atom}-{atom_type}-0.0 {float(xyz[0]): .10f} {float(xyz[1]): .10f} "
            f"{float(xyz[2]): .10f} {layer}"
        )
    lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _default_uff_type(atom: str) -> str:
    symbol = str(atom).strip().capitalize()
    mapping = {
        "H": "H_",
        "C": "C_3",
        "N": "N_3",
        "O": "O_3",
        "F": "F_",
        "P": "P_3+3",
        "S": "S_3+2",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I_",
    }
    try:
        return mapping[symbol]
    except KeyError as exc:
        raise GaussianWriteError(
            f"no default UFF atom type for {atom}; provide oniom_atom_types"
        ) from exc


def ensure_gaussian_output_pickett(route: str) -> str:
    """Return a Gaussian route that explicitly asks for Pickett output."""
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        text = f"# {text}"
    if _OUTPUT_RE.search(text):
        text = _OUTPUT_RE.sub("output=pickett", text, count=1)
    else:
        text = f"{text} output=pickett"
    return _collapse_route(text)


def _ensure_oracle_population_contract(route: str) -> str:
    """Request CM5 charges and Mayer bond orders in ORACLE Gaussian jobs."""

    text = route.strip()
    if match := _POP_PAREN_RE.search(text):
        options = [item.strip() for item in match.group("body").split(",") if item.strip()]
        if not any(option.lower() == "hirshfeld" for option in options):
            options.append("Hirshfeld")
            text = _POP_PAREN_RE.sub(f"Pop=({','.join(options)})", text, count=1)
    elif match := _POP_VALUE_RE.search(text):
        value = match.group("value")
        if value.lower() != "hirshfeld":
            text = _POP_VALUE_RE.sub(f"Pop=({value},Hirshfeld)", text, count=1)
    else:
        text = f"{text} Pop=Hirshfeld"
    if not _MAYER_IOP_RE.search(text):
        text = f"{text} IOp(6/80=1)"
    return _collapse_route(text)


def write_gicforge_gaussian_input(
    enriched_xyz: Path,
    output: Path,
    *,
    route: str = DEFAULT_GIC_ROUTE,
    title: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    link0: tuple[str, ...] = (),
    total_symmetric_only: bool = False,
    freeze_non_total: bool = True,
    g16_compatibility: bool = True,
) -> Path:
    """Write a Gaussian input file from an enriched XYZ carrying #GIC."""
    source = Path(enriched_xyz)
    _require_gic_section(source)
    geometry = read_enriched_xyz(source)
    job_charge = (
        charge if charge is not None else geometry.charge if geometry.charge is not None else 0
    )
    job_multiplicity = (
        multiplicity
        if multiplicity is not None
        else geometry.multiplicity
        if geometry.multiplicity is not None
        else 1
    )
    lines = [
        *[item.strip() for item in link0 if item.strip()],
        _ensure_oracle_population_contract(_normalize_route(route)),
        "",
        title or geometry.comment or source.stem,
        "",
        f"{job_charge} {job_multiplicity}",
    ]
    for atom, (x, y, z) in zip(geometry.atoms, geometry.coordinates_angstrom):
        lines.append(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}")
    lines.append("")
    gic_lines = _gaussian_gic_lines(
        source,
        total_symmetric_only=total_symmetric_only,
        freeze_non_total=freeze_non_total,
    )
    if gic_lines:
        if g16_compatibility:
            gic_lines = _gaussian_g16_compatible_active_dihedral_lines(gic_lines)
        lines.extend(_gaussian_gic_lines_with_values(gic_lines, geometry.coordinates_angstrom))
        lines.append("")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_semidiagonal_cubic_vibrot_gaussian_input(
    output: Path,
    *,
    oldchk: str,
    route: str = DEFAULT_SEMIDIAGONAL_ROUTE,
    harmonic_chk: str | None = None,
    cubic_chk: str | None = None,
    symmetry: str = "sym=com",
    xyzin: Path | str | None = None,
    manifest_path: Path | str | None = None,
    write_manifest: bool = True,
) -> Path:
    """Write the two-link Gaussian vibrot/numerical-gradient job for F3ijj.

    The input assumes `oldchk` already contains the optimized geometry and basis.
    Gaussian then performs a harmonic `freq=vibrot` step followed by
    `freq=(numer,readharm,vibrot)`, which differentiates analytical gradients
    and prints the semidiagonal cubic constants used by MATRIX.
    """
    if not oldchk.strip():
        raise GaussianWriteError("oldchk cannot be empty")
    route_body = _route_body(route)
    harmonic = harmonic_chk or _replace_checkpoint_suffix(oldchk, ".harm.chk")
    cubic = cubic_chk or _replace_checkpoint_suffix(oldchk, ".cubic.chk")
    sym = symmetry.strip()
    options = " ".join(item for item in (route_body, sym, "geom=allcheck", "freq=vibrot") if item)
    cubic_options = " ".join(
        item for item in (route_body, sym, "geom=allcheck", "freq=(numer,readharm,vibrot)") if item
    )
    lines = [
        f"%oldchk={oldchk}",
        f"%chk={harmonic}",
        f"#p {options}",
        "",
        "",
        "--Link1--",
        f"%oldchk={harmonic}",
        f"%chk={cubic}",
        f"#p {cubic_options}",
        "",
        "",
    ]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    if write_manifest:
        inputs = {"oldchk": Path(oldchk)}
        if xyzin is not None:
            inputs["xyzin"] = Path(xyzin)
        manifest = build_run_manifest(
            workflow="gaussian-semidiagonal-cubic-vibrot-input",
            status="prepared",
            run_dir=target.parent,
            inputs=inputs,
            outputs={"gaussian_input": target},
            parameters={
                "route": route,
                "route_body": route_body,
                "oldchk": oldchk,
                "harmonic_chk": harmonic,
                "cubic_chk": cubic,
                "symmetry": sym,
                "keywords": ("freq=vibrot", "freq=(numer,readharm,vibrot)"),
            },
            backend={"name": "Gaussian"},
        )
        manifest.write(
            Path(manifest_path)
            if manifest_path is not None
            else target.with_suffix(target.suffix + ".manifest.json")
        )
    return target


def _require_gic_section(path: Path) -> None:
    lines = read_sectioned_lines(Path(path))
    gic = section_content(lines, "GIC")
    if not gic:
        raise GaussianWriteError("missing #GIC section")
    expected = f"SCHEMA {REQUIRED_GIC_SCHEMA}"
    if gic[0].strip() != expected:
        raise GaussianWriteError(f"#GIC must start with {expected!r}; found {gic[0]!r}")


def _gaussian_gic_lines(
    path: Path,
    *,
    total_symmetric_only: bool = False,
    freeze_non_total: bool = True,
) -> list[str]:
    try:
        from matrix_smith import gaussian_gic_lines_from_xyzin
    except ImportError:
        return []
    return gaussian_gic_lines_from_xyzin(
        Path(path),
        total_symmetric_only=total_symmetric_only,
        freeze_non_total=freeze_non_total,
    )


def _gaussian_gic_lines_with_values(
    gic_lines: list[str],
    coordinates_angstrom: object,
) -> list[str]:
    coords = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in coordinates_angstrom
    ]
    values: dict[str, float] = {}
    deferred_labels: set[str] = set()
    out: list[str] = []
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            out.append(line)
            continue
        label, expression = parsed
        base_label = _gaussian_base_label(label)
        if _GIC_VALUE_RE.search(label):
            out.append(line)
            continue
        if _GIC_NONACTIVE_RE.search(label):
            deferred_labels.add(base_label)
            out.append(line)
            continue
        # Fragment declarations define Gaussian GIC objects rather than scalar
        # coordinates.  They must remain literal (for example
        # ``F001=Fragment(1-5)``) and must not receive a Value option.
        if re.fullmatch(r"\s*Fragment\s*\([^()]*\)\s*", expression, flags=re.IGNORECASE):
            out.append(line)
            continue
        # Gaussian's five-argument L(i,j,k,m,n) is an oriented linear-bend
        # component, not the ordinary three-atom angle.  Let Gaussian evaluate
        # its own exact convention instead of manufacturing an initial value.
        if re.search(r"\bL\s*\(", expression, flags=re.IGNORECASE):
            out.append(line)
            continue
        identifiers = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression))
        if identifiers & deferred_labels:
            # Gaussian evaluates fragment/center helper variables declared as
            # Inactive.  Their numerical convention belongs to Gaussian, so
            # dependent active rows must likewise be left without Value.
            out.append(line)
            continue
        try:
            value = _evaluate_gaussian_gic_expression(expression, coords, values)
        except (ArithmeticError, ValueError, KeyError, SyntaxError, NameError) as exc:
            raise GaussianWriteError(
                f"cannot evaluate initial value for Gaussian GIC {base_label}: {expression}"
            ) from exc
        values[base_label] = value
        out.append(_gaussian_line_with_inline_value(label, expression, value))
    return out


def _gaussian_g16_compatible_active_dihedral_lines(gic_lines: list[str]) -> list[str]:
    """Translate native out-of-plane rows and expand composite torsions for G16.

    ORACLE and SMITH always store U(center,n1,n2,n3).  Gaussian 16 receives the
    equivalent improper D(n1,center,n3,n2) only in the generated input file.
    Gaussian 16 also does not apply periodic wrapping to the components of an
    active generic GIC containing multiple D(...) terms.  Composite rows are
    therefore expanded into simple periodic coordinates.  Post-processing reads
    the optimized Cartesians and projects them back onto the unchanged native
    SONIC contract.
    """

    out: list[str] = []
    active_dihedrals: set[str] = set()
    component_index = 1
    for line in gic_lines:
        parsed = _gaussian_definition(line)
        if parsed is None:
            out.append(line)
            continue
        label, expression = parsed
        expression = _gaussian_g16_out_of_plane_to_improper(expression)
        line = f"{label} = {expression}"
        direct_dihedral = re.fullmatch(r"\s*D\s*\([^()]*\)\s*", expression, flags=re.IGNORECASE)
        if direct_dihedral and not _GIC_NONACTIVE_RE.search(label):
            active_dihedrals.add(_canonical_gaussian_expression(direct_dihedral.group(0)))
            out.append(line)
            continue
        dihedrals = _GIC_DIHEDRAL_RE.findall(expression)
        if len(dihedrals) > 1:
            nonactive = _GIC_NONACTIVE_RE.search(label) is not None
            option = (
                "Frozen"
                if re.search(r"\bfrozen\b", label, flags=re.IGNORECASE)
                else "Inactive"
                if nonactive
                else None
            )
            local_dihedrals: set[str] = set()
            for dihedral in dihedrals:
                canonical = _canonical_gaussian_expression(dihedral)
                if canonical in local_dihedrals or (
                    not nonactive and canonical in active_dihedrals
                ):
                    continue
                local_dihedrals.add(canonical)
                if not nonactive:
                    active_dihedrals.add(canonical)
                component_label = f"V{component_index:04d}"
                if option is not None:
                    component_label = f"{component_label}({option})"
                out.append(f"{component_label} = {canonical}")
                component_index += 1
        else:
            out.append(line)
    return out


def _gaussian_g16_out_of_plane_to_improper(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        atoms = [item.strip() for item in match.group(1).split(",")]
        if len(atoms) != 4 or any(not re.fullmatch(r"[1-9][0-9]*", atom) for atom in atoms):
            raise GaussianWriteError(
                "G16 out-of-plane translation requires U(center,n1,n2,n3) atom indices"
            )
        center, n1, n2, n3 = atoms
        return f"D({n1},{center},{n3},{n2})"

    return _GIC_OUT_OF_PLANE_RE.sub(replace, expression)


def _canonical_gaussian_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression.strip())


def _gaussian_label_with_option(label: str, option: str) -> str:
    text = label.strip()
    if "(" in text and text.endswith(")"):
        return f"{text[:-1]},{option})"
    return f"{text}({option})"


def _gaussian_definition(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    label, expression = line.split("=", 1)
    label = label.strip()
    expression = expression.strip()
    if not label or not expression:
        return None
    return label, expression


def _gaussian_base_label(label: str) -> str:
    return _GIC_LABEL_OPTION_RE.sub("", label.strip()).strip()


def _gaussian_line_with_inline_value(label: str, expression: str, value: float) -> str:
    label_text = label.strip()
    # Gaussian 16 does not accept Value together with Frozen/Inactive in the
    # option list.  Non-active coordinates are evaluated from the Cartesian
    # geometry by Gaussian, so their explicit initial value is unnecessary.
    if _GIC_NONACTIVE_RE.search(label_text):
        return f"{label_text} = {expression.strip()}"
    if "(" in label_text and label_text.endswith(")"):
        label_text = f"{label_text[:-1]},Value={_gaussian_float(value)})"
    else:
        label_text = f"{label_text}(Value={_gaussian_float(value)})"
    return f"{label_text} = {expression.strip()}"


def _gaussian_float(value: float) -> str:
    text = f"{value:.12g}"
    if "." not in text and "e" not in text.lower():
        text = f"{text}.0"
    return text


def _evaluate_gaussian_gic_expression(
    expression: str,
    coords: list[tuple[float, float, float]],
    values: dict[str, float],
) -> float:
    direct = re.fullmatch(r"\s*([RADUXYZ])\s*\(([^()]*)\)\s*", expression, flags=re.IGNORECASE)
    if direct:
        args = [item.strip() for item in direct.group(2).split(",") if item.strip()]
        return _evaluate_direct_geometry_function(direct.group(1).upper(), args, coords)
    text = expression.replace("^", "**")
    previous = None
    while previous != text:
        previous = text
        text = _GIC_FUNCTION_RE.sub(
            lambda match: _evaluate_gaussian_gic_function(match, coords, values),
            text,
        )
    env = {
        **values,
        "sqrt": math.sqrt,
        "Sqrt": math.sqrt,
        "SQRT": math.sqrt,
        "exp": math.exp,
        "Exp": math.exp,
        "EXP": math.exp,
        "sin": math.sin,
        "Sin": math.sin,
        "SIN": math.sin,
        "cos": math.cos,
        "Cos": math.cos,
        "COS": math.cos,
        "tan": math.tan,
        "Tan": math.tan,
        "TAN": math.tan,
        "acos": math.acos,
        "Acos": math.acos,
        "ARCCOS": math.acos,
        "arccos": math.acos,
        "atan2": math.atan2,
        "Atan2": math.atan2,
        "ATAN2": math.atan2,
    }
    return float(eval(text, {"__builtins__": {}}, env))


def _evaluate_direct_geometry_function(
    name: str,
    args: list[str],
    coords: list[tuple[float, float, float]],
) -> float:
    value = _evaluate_geometry_function(name, args, coords)
    if name in {"A", "D", "U"}:
        return math.degrees(value)
    if name in {"R", "X", "Y", "Z"}:
        return value / ANGSTROM_TO_BOHR
    return value


def _evaluate_gaussian_gic_function(
    match: re.Match[str],
    coords: list[tuple[float, float, float]],
    values: dict[str, float],
) -> str:
    name = match.group(1)
    args = [item.strip() for item in match.group(2).split(",") if item.strip()]
    upper = name.upper()
    if upper in {"R", "A", "D", "U", "X", "Y", "Z"}:
        value = _evaluate_geometry_function(upper, args, coords)
        return f"({value:.17g})"
    if upper in {"SQRT", "EXP", "SIN", "COS", "TAN", "ACOS", "ARCCOS", "ATAN2"}:
        return match.group(0)
    if name in values:
        return f"({values[name]:.17g})"
    return match.group(0)


def _evaluate_geometry_function(
    name: str,
    args: list[str],
    coords: list[tuple[float, float, float]],
) -> float:
    if name in {"X", "Y", "Z"}:
        if len(args) != 1:
            raise ValueError(f"{name} needs one atom index")
        atom = _atom_index(args[0], coords)
        axis = {"X": 0, "Y": 1, "Z": 2}[name]
        return coords[atom][axis] * ANGSTROM_TO_BOHR
    if name == "R":
        if len(args) != 2:
            raise ValueError("R needs two atom indices")
        first, second = (_atom_index(arg, coords) for arg in args)
        return _distance(coords[first], coords[second]) * ANGSTROM_TO_BOHR
    if name == "A":
        if len(args) != 3:
            raise ValueError("A needs three atom indices")
        first, center, third = (_atom_index(arg, coords) for arg in args)
        return _angle(coords[first], coords[center], coords[third])
    if name == "D":
        if len(args) != 4:
            raise ValueError("D needs four atom indices")
        i, j, k, l = (_atom_index(arg, coords) for arg in args)
        return _dihedral(coords[i], coords[j], coords[k], coords[l])
    if name == "U":
        if len(args) != 4:
            raise ValueError("U needs four atom indices")
        center, first, second, third = (_atom_index(arg, coords) for arg in args)
        return _out_of_plane(coords[center], coords[first], coords[second], coords[third])
    raise ValueError(f"unsupported geometry function {name}")


def _atom_index(text: str, coords: list[tuple[float, float, float]]) -> int:
    index = int(text) - 1
    if index < 0 or index >= len(coords):
        raise ValueError(f"atom index out of range: {text}")
    return index


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _angle(
    first: tuple[float, float, float],
    center: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    u = tuple(a - b for a, b in zip(first, center))
    v = tuple(a - b for a, b in zip(third, center))
    norm = _norm(u) * _norm(v)
    if norm <= 1.0e-14:
        raise FloatingPointError("angle is singular")
    cosine = max(-1.0, min(1.0, _dot(u, v) / norm))
    return math.acos(cosine)


def _dihedral(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    b0 = tuple(a - b for a, b in zip(p0, p1))
    b1 = tuple(a - b for a, b in zip(p2, p1))
    b2 = tuple(a - b for a, b in zip(p3, p2))
    b1_unit = _unit(b1)
    v = tuple(a - _dot(b0, b1_unit) * b for a, b in zip(b0, b1_unit))
    w = tuple(a - _dot(b2, b1_unit) * b for a, b in zip(b2, b1_unit))
    x_value = _dot(v, w)
    y_value = _dot(_cross(b1_unit, v), w)
    return math.atan2(y_value, x_value)


def _out_of_plane(
    center: tuple[float, float, float],
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    u = _unit(tuple(a - b for a, b in zip(first, center)))
    v = _unit(tuple(a - b for a, b in zip(second, center)))
    w = _unit(tuple(a - b for a, b in zip(third, center)))
    normal = _cross(v, w)
    normal_norm = _norm(normal)
    if normal_norm <= 1.0e-14:
        raise FloatingPointError("out-of-plane coordinate is singular")
    sine = max(-1.0, min(1.0, _dot(u, normal) / normal_norm))
    return math.asin(sine)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 1.0e-14:
        raise FloatingPointError("zero-length vector")
    return tuple(value / norm for value in vector)


def _normalize_route(route: str) -> str:
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        text = f"# {text}"
    if _GICALLSYM_RE.search(text):
        raise GaussianWriteError(
            "MATRIX exports already symmetrized GICs; do not request Gaussian GICAllSym"
        )
    if _route_has_opt(text):
        text = _strip_readallgic_geom(text)
        return ensure_gaussian_output_pickett(_ensure_readallgic_opt(text))
    return ensure_gaussian_output_pickett(_ensure_readallgic_geom(text))


def _normalize_point_route(route: str, *, ensure_force: bool = True) -> str:
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        text = f"# {text}"
    if ensure_force and not re.search(r"\bforce\b", text, flags=re.IGNORECASE):
        text = f"{text} Force"
    return _collapse_route(text)


def _route_body(route: str) -> str:
    text = route.strip()
    if not text:
        raise GaussianWriteError("Gaussian route cannot be empty")
    if not text.startswith("#"):
        return text
    return re.sub(r"^#\s*[A-Za-z]?\s*", "", text).strip()


def _replace_checkpoint_suffix(path: str, suffix: str) -> str:
    text = path.strip()
    lower = text.lower()
    for ext in (".chk", ".fchk", ".fch"):
        if lower.endswith(ext):
            return text[: -len(ext)] + suffix
    return text + suffix


def _route_has_opt(route: str) -> bool:
    return bool(
        _OPT_PAREN_RE.search(route) or _OPT_VALUE_RE.search(route) or _OPT_BARE_RE.search(route)
    )


def _collapse_route(route: str) -> str:
    return " ".join(route.split())


def _strip_readallgic_geom(route: str) -> str:
    text = _GEOM_PAREN_RE.sub(_remove_readallgic_geom_parenthesized, route)
    return _GEOM_VALUE_RE.sub(_remove_readallgic_geom_value, text)


def _remove_readallgic_geom_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    kept = [option for option in options if not _READALLGIC_OPTION_RE.fullmatch(option)]
    if len(kept) == len(options):
        return match.group(0)
    if not kept:
        return ""
    return f"geom=({','.join(kept)})"


def _remove_readallgic_geom_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if value.lower() in {"gic", "readgic", "readallgic"}:
        return ""
    return match.group(0)


def _ensure_readallgic_opt(route: str) -> str:
    if _OPT_PAREN_RE.search(route):
        return _OPT_PAREN_RE.sub(_readallgic_opt_parenthesized, route, count=1)
    if match := _OPT_VALUE_RE.search(route):
        value = match.group("value")
        replacement = (
            "opt=readallgic"
            if value.lower() in {"gic", "readgic", "readallgic"}
            else f"opt=(readallgic,{value})"
        )
        return _OPT_VALUE_RE.sub(replacement, route, count=1)
    if _OPT_BARE_RE.search(route):
        return _OPT_BARE_RE.sub("opt=readallgic", route, count=1)
    return f"{route} opt=readallgic"


def _readallgic_opt_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    if not any(_READALLGIC_OPTION_RE.fullmatch(option) for option in options):
        options.insert(0, "readallgic")
    return f"opt=({','.join(options)})"


def _ensure_readallgic_geom(route: str) -> str:
    if _GEOM_PAREN_RE.search(route):
        return _GEOM_PAREN_RE.sub(_readallgic_geom_parenthesized, route, count=1)
    if match := _GEOM_VALUE_RE.search(route):
        value = match.group("value")
        replacement = (
            "geom=readallgic"
            if value.lower() in {"gic", "readgic", "readallgic"}
            else f"geom=(readallgic,{value})"
        )
        return _GEOM_VALUE_RE.sub(replacement, route, count=1)
    return f"{route} geom=readallgic"


def _readallgic_geom_parenthesized(match: re.Match[str]) -> str:
    options = [item.strip() for item in match.group("body").split(",") if item.strip()]
    if not any(_READALLGIC_OPTION_RE.fullmatch(option) for option in options):
        options.insert(0, "readallgic")
    return f"geom=({','.join(options)})"
