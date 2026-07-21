"""Persistent local and remote capability profile used by The ONE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Mapping


ENVIRONMENT_SCHEMA = "matrix.environment.v1"
DEFAULT_CONFIG_PATH = Path("~/.config/matrix/environment.json").expanduser()

PROGRAM_DEFINITIONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("gdv", "Gaussian development version (GDV)", "qm", ("matrix-gdv", "gdv-run")),
    ("g16", "Gaussian 16", "qm", ("g16",)),
    ("orca", "ORCA", "qm", ("orca",)),
    ("molpro", "Molpro", "qm", ("molpro",)),
    ("mrcc", "MRCC", "qm", ("dmrcc",)),
    ("cfour", "CFOUR", "qm", ("xcfour",)),
    ("xtb", "xTB", "qm", ("xtb",)),
    ("pyscf", "PySCF launcher", "qm", ("pyscf-python",)),
    ("psi4", "Psi4", "qm", ("psi4",)),
    ("et", "eT electronic-structure program", "qm", ("eT_launch.py", "eT")),
    ("avogadro2", "Avogadro 2", "viewer", ("avogadro2",)),
    ("avogadro1", "Avogadro 1", "viewer", ("avogadro",)),
    ("molden", "Molden", "viewer", ("molden",)),
    ("vmd", "VMD", "viewer", ("vmd",)),
)

PROGRAM_ENVIRONMENT = {
    "gdv": "MATRIX_GDV_EXE",
    "g16": "MATRIX_G16_EXE",
    "orca": "MATRIX_ORCA_EXE",
    "molpro": "MATRIX_MOLPRO_EXE",
    "mrcc": "MATRIX_MRCC_EXE",
    "cfour": "MATRIX_CFOUR_EXE",
    "xtb": "MATRIX_XTB_EXE",
    "pyscf": "MATRIX_PYSCF_EXE",
    "psi4": "MATRIX_PSI4_EXE",
    "et": "MATRIX_ET_EXE",
    "avogadro2": "MATRIX_AVOGADRO2_EXE",
    "avogadro1": "MATRIX_AVOGADRO1_EXE",
    "molden": "MATRIX_MOLDEN_EXE",
    "vmd": "MATRIX_VMD_EXE",
}

_MACOS_APPLICATIONS = {
    "avogadro2": Path("/Applications/Avogadro2.app"),
    "avogadro1": Path("/Applications/Avogadro.app"),
    "vmd": Path("/Applications/VMD.app"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ExternalProgram:
    key: str
    label: str
    category: str
    executable: str = ""
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.category not in {"qm", "viewer"}:
            raise ValueError(f"unsupported external-program category: {self.category}")

    @property
    def available(self) -> bool:
        return bool(resolve_program_path(self.executable))


@dataclass(frozen=True)
class MachineLimits:
    processors: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    neural_engine_count: int = 0
    max_concurrent_jobs: int = 1

    def __post_init__(self) -> None:
        if self.processors < 1:
            raise ValueError("processors must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")
        if self.neural_engine_count < 0:
            raise ValueError("neural_engine_count cannot be negative")
        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")


@dataclass(frozen=True)
class RemoteMachine:
    name: str
    host: str
    remote_root: str = "~/matrix"
    enabled: bool = True
    limits: MachineLimits = field(default_factory=MachineLimits)
    engines: tuple[str, ...] = ()
    requires_vpn: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("remote machine name is required")
        if not self.host.strip():
            raise ValueError("remote machine SSH host is required")
        engines = {item.strip().casefold() for item in self.engines if item.strip()}
        object.__setattr__(self, "engines", tuple(sorted(engines)))


@dataclass(frozen=True)
class MatrixEnvironment:
    matrix_root: str
    projects_root: str
    programs: tuple[ExternalProgram, ...]
    local_limits: MachineLimits
    remote_machines: tuple[RemoteMachine, ...] = ()
    schema: str = ENVIRONMENT_SCHEMA
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_SCHEMA:
            raise ValueError(f"unsupported MATRIX environment schema: {self.schema}")
        if not self.matrix_root.strip() or not self.projects_root.strip():
            raise ValueError("MATRIX and project roots are required")
        program_keys = [program.key for program in self.programs]
        if len(program_keys) != len(set(program_keys)):
            raise ValueError("external program keys must be unique")
        machine_names = [machine.name for machine in self.remote_machines]
        if len(machine_names) != len(set(machine_names)):
            raise ValueError("remote machine names must be unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatrixEnvironment":
        raw_programs = data.get("programs", ())
        raw_machines = data.get("remote_machines", ())
        raw_local = data.get("local_limits", {})
        if not isinstance(raw_programs, (list, tuple)):
            raise ValueError("programs must be an array")
        if not isinstance(raw_machines, (list, tuple)):
            raise ValueError("remote_machines must be an array")
        if not isinstance(raw_local, Mapping):
            raise ValueError("local_limits must be an object")
        stored_programs = tuple(ExternalProgram(**dict(item)) for item in raw_programs)
        stored_by_key = {program.key: program for program in stored_programs}
        programs = tuple(
            stored_by_key.pop(detected.key, detected)
            for detected in detect_external_programs()
        ) + tuple(stored_by_key.values())
        machines: list[RemoteMachine] = []
        for raw in raw_machines:
            item = dict(raw)
            limits = item.get("limits", {})
            if not isinstance(limits, Mapping):
                raise ValueError("remote machine limits must be an object")
            item["limits"] = MachineLimits(**dict(limits))
            item["engines"] = tuple(item.get("engines", ()))
            machines.append(RemoteMachine(**item))
        return cls(
            matrix_root=str(data["matrix_root"]),
            projects_root=str(data["projects_root"]),
            programs=programs,
            local_limits=MachineLimits(**dict(raw_local)),
            remote_machines=tuple(machines),
            schema=str(data.get("schema", "")),
            created_utc=str(data.get("created_utc", "")),
            updated_utc=str(data.get("updated_utc", "")),
        )

    def program(self, key: str) -> ExternalProgram | None:
        normalized = key.strip().casefold()
        return next((item for item in self.programs if item.key == normalized), None)


def resolve_program_path(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    expanded = Path(text).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    return shutil.which(text) or ""


def detect_external_programs() -> tuple[ExternalProgram, ...]:
    programs: list[ExternalProgram] = []
    for key, label, category, candidates in PROGRAM_DEFINITIONS:
        detected = ""
        if sys.platform == "darwin" and key in _MACOS_APPLICATIONS:
            application = _MACOS_APPLICATIONS[key]
            if application.exists():
                detected = str(application)
        if not detected:
            for candidate in candidates:
                detected = shutil.which(candidate) or ""
                if detected:
                    break
        programs.append(ExternalProgram(key, label, category, detected, bool(detected)))
    return tuple(programs)


def detect_local_limits() -> MachineLimits:
    processors = os.cpu_count() or 1
    memory_bytes = 0
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(("sysctl", "-n", "hw.memsize"), text=True)
            memory_bytes = int(output.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            memory_bytes = 0
    elif Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    memory_gb = round(memory_bytes / (1024**3), 1) if memory_bytes else 1.0
    return MachineLimits(
        processors=processors,
        memory_gb=max(1.0, memory_gb),
        neural_engine_count=_detect_apple_neural_engine_count(),
        max_concurrent_jobs=max(1, min(4, processors // 2)),
    )


def _detect_apple_neural_engine_count() -> int:
    """Return the number of Apple Neural Engine devices, not their core count."""

    if sys.platform != "darwin" or platform.machine().casefold() not in {"arm64", "aarch64"}:
        return 0
    try:
        output = subprocess.check_output(("sysctl", "-n", "hw.optional.arm64"), text=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    return 1 if output.strip() == "1" else 0


def default_environment(
    *, matrix_root: Path | str | None = None, projects_root: Path | str | None = None
) -> MatrixEnvironment:
    root = Path(matrix_root or os.environ.get("MATRIX_HOME") or Path.cwd()).expanduser().resolve()
    projects = Path(
        projects_root
        or os.environ.get("MATRIX_PROJECTS_DIR")
        or (Path.home() / "Documents" / "MATRIX-projects")
    ).expanduser().resolve()
    return MatrixEnvironment(
        matrix_root=str(root),
        projects_root=str(projects),
        programs=detect_external_programs(),
        local_limits=detect_local_limits(),
    )


def load_environment(
    path: Path | str = DEFAULT_CONFIG_PATH, *, missing_ok: bool = False
) -> MatrixEnvironment:
    target = Path(path).expanduser()
    if missing_ok and not target.is_file():
        return default_environment()
    return MatrixEnvironment.from_dict(json.loads(target.read_text(encoding="utf-8")))


def write_environment(
    environment: MatrixEnvironment, path: Path | str = DEFAULT_CONFIG_PATH
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = environment.to_dict()
    data["updated_utc"] = _utc_now()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def environment_exports(
    environment: MatrixEnvironment, *, config_path: Path | str = DEFAULT_CONFIG_PATH
) -> dict[str, str]:
    exports = {
        "MATRIX_CONFIG": str(Path(config_path).expanduser().resolve()),
        "MATRIX_HOME": str(Path(environment.matrix_root).expanduser().resolve()),
        "MATRIX_PROJECTS_DIR": str(Path(environment.projects_root).expanduser().resolve()),
        "MATRIX_LOCAL_PROCESSORS": str(environment.local_limits.processors),
        "MATRIX_LOCAL_MEMORY_GB": str(environment.local_limits.memory_gb),
        "MATRIX_LOCAL_GPU_COUNT": str(environment.local_limits.gpu_count),
        "MATRIX_LOCAL_NEURAL_ENGINE_COUNT": str(
            environment.local_limits.neural_engine_count
        ),
        "MATRIX_LOCAL_MAX_JOBS": str(environment.local_limits.max_concurrent_jobs),
    }
    for program in environment.programs:
        variable = PROGRAM_ENVIRONMENT.get(program.key)
        resolved = resolve_program_path(program.executable) if program.enabled else ""
        if variable and resolved:
            exports[variable] = resolved
    gdv = exports.get("MATRIX_GDV_EXE")
    g16 = exports.get("MATRIX_G16_EXE")
    if gdv or g16:
        exports["ORACLE_GAUSSIAN_EXE"] = gdv or g16 or ""
        exports["MATRIX_GAUSSIAN_BACKEND"] = "gdv" if gdv else "g16"
    enabled_remotes = [machine for machine in environment.remote_machines if machine.enabled]
    if enabled_remotes:
        primary = enabled_remotes[0]
        exports.update(
            {
                "MATRIX_REMOTE_HOST": primary.host,
                "MATRIX_REMOTE_ROOT": primary.remote_root,
                "MATRIX_REMOTE_PROCESSORS": str(primary.limits.processors),
                "MATRIX_REMOTE_MEMORY_GB": str(primary.limits.memory_gb),
                "MATRIX_REMOTE_GPU_COUNT": str(primary.limits.gpu_count),
                "MATRIX_REMOTE_NEURAL_ENGINE_COUNT": str(
                    primary.limits.neural_engine_count
                ),
                "MATRIX_REMOTE_MAX_JOBS": str(primary.limits.max_concurrent_jobs),
                "MATRIX_REMOTE_REQUIRES_VPN": "1" if primary.requires_vpn else "0",
            }
        )
    return exports
