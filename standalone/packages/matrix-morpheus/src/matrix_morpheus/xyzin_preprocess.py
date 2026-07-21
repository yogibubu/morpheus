from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matrix_core.xyzin_geometry import replace_xyzin_geometry

from .contracts import IsotopologueObservation
from .geometry_input import read_geometry_input
from .io import read_observations
from .xyzin_observations import (
    has_xyzin_isotopologues,
    read_xyzin_isotopologues,
    write_xyzin_isotopologues,
)


@dataclass(frozen=True)
class SemiexperimentalXyzinPreprocessResult:
    xyzin: Path
    source_fixed_parameters: tuple[str, ...]
    observations: tuple[IsotopologueObservation, ...]
    created_or_updated_geometry: bool
    updated_isotopologues: bool


def prepare_semiexperimental_xyzin(
    geometry_source: Path,
    *,
    observations_source: Path | None = None,
    observations_inline: tuple[IsotopologueObservation, ...] = (),
    xyzin_path: Path | None = None,
    workdir: Path | None = None,
) -> SemiexperimentalXyzinPreprocessResult:
    geometry_source = Path(geometry_source)
    xyzin = Path(xyzin_path) if xyzin_path is not None else _xyzin_target(geometry_source, workdir)
    created_or_updated_geometry = False
    source_fixed: tuple[str, ...] = ()

    if geometry_source.resolve() != xyzin.resolve():
        geometry = read_geometry_input(geometry_source)
        replace_xyzin_geometry(
            xyzin,
            geometry.atoms,
            geometry.coordinates_angstrom,
            comment=geometry.comment or geometry_source.stem,
        )
        source_fixed = geometry.fixed_parameters
        created_or_updated_geometry = True
    else:
        source_fixed = read_geometry_input(xyzin).fixed_parameters

    updated_isotopologues = False
    if observations_inline:
        observations = observations_inline
        write_xyzin_isotopologues(xyzin, observations)
        updated_isotopologues = True
    elif observations_source is not None:
        observations_source = Path(observations_source)
        if observations_source.exists() and observations_source.resolve() == xyzin.resolve():
            observations = read_xyzin_isotopologues(xyzin)
        else:
            observations = read_observations(observations_source)
            write_xyzin_isotopologues(xyzin, observations)
            updated_isotopologues = True
    else:
        observations = read_xyzin_isotopologues(xyzin) if has_xyzin_isotopologues(xyzin) else ()

    return SemiexperimentalXyzinPreprocessResult(
        xyzin=xyzin,
        source_fixed_parameters=source_fixed,
        observations=observations,
        created_or_updated_geometry=created_or_updated_geometry,
        updated_isotopologues=updated_isotopologues,
    )


def _xyzin_target(geometry_source: Path, workdir: Path | None) -> Path:
    if geometry_source.name.lower() == "xyzin" or geometry_source.suffix.lower() == ".xyzin":
        return geometry_source
    candidates: list[Path] = []
    if workdir is not None:
        candidates.extend((Path(workdir) / "xyzin", Path(workdir) / "working" / "xyzin"))
    candidates.extend(
        (Path.cwd() / "xyzin", geometry_source.parent / "xyzin", Path.cwd() / "working" / "xyzin")
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate
    if workdir is not None:
        return Path(workdir) / "xyzin"
    return geometry_source.parent / "xyzin"
