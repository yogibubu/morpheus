from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time


GAUSSIAN_EXECUTABLE = "gdv"
FORMCHK_EXECUTABLE = "formchk"
NORMAL_TERMINATION_MARKER = "Normal termination of Gaussian"
ERROR_TERMINATION_MARKER = "Error termination"
LOG_CANDIDATES = ("gauin.log", "gauout.log")
PID_FILE = "gaussian.pid"


class GaussianInputError(RuntimeError):
    """Raised when no runnable Gaussian input can be found."""


@dataclass(frozen=True)
class GaussianJobStatus:
    workdir: Path
    input_path: Path | None
    log_path: Path
    status: str
    normal_termination: bool
    error_termination: bool
    pid: int | None = None
    exit_code: int | None = None
    message: str = ""


@dataclass(frozen=True)
class GaussianRunResult:
    workdir: Path
    input_path: Path
    log_path: Path
    executable: str
    pid: int | None
    exit_code: int | None
    success: bool | None
    message: str


def ensure_gjf_input(
    workdir: Path,
    *,
    input_name: str = "gauin",
    gjf_name: str = "gauin.gjf",
) -> Path:
    """Return a Gaussian input path, copying `gauin` to `gauin.gjf` if needed."""
    workdir = Path(workdir).resolve()
    gauin_gjf = workdir / gjf_name
    gauin_raw = workdir / input_name
    if not gauin_gjf.exists() and gauin_raw.exists():
        gauin_gjf.write_text(
            gauin_raw.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
    if gauin_gjf.exists():
        return gauin_gjf
    if gauin_raw.exists():
        return gauin_raw
    raise GaussianInputError(f"{gjf_name} or {input_name} not found")


def select_latest_log(workdir: Path, *, candidates: tuple[str, ...] = LOG_CANDIDATES) -> Path:
    """Select the most likely active Gaussian log in a work directory."""
    workdir = Path(workdir).resolve()
    existing: list[tuple[float, int, Path]] = []
    for name in candidates:
        path = workdir / name
        if path.exists():
            try:
                stat = path.stat()
            except OSError:
                continue
            existing.append((stat.st_mtime, stat.st_size, path))
    if not existing:
        return workdir / candidates[0]
    existing.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return existing[0][2]


def gaussian_completed_normally(log_path: Path) -> bool:
    if not Path(log_path).exists():
        return False
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return NORMAL_TERMINATION_MARKER in text


def gaussian_has_error_termination(log_path: Path) -> bool:
    if not Path(log_path).exists():
        return False
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return ERROR_TERMINATION_MARKER in text


def gaussian_completion_message(workdir: Path, exit_code: int) -> tuple[bool, str]:
    """Return `(success, message)` for a finished Gaussian process."""
    log_path = select_latest_log(workdir)
    if gaussian_completed_normally(log_path):
        return True, "Gaussian completed successfully"
    if exit_code == 0 and log_path.exists() and not gaussian_has_error_termination(log_path):
        return True, f"Gaussian completed (check {log_path.name})"
    return False, f"Gaussian finished with errors (exit_code={exit_code}; see {log_path.name})"


def gaussian_job_status(workdir: Path) -> GaussianJobStatus:
    workdir = Path(workdir)
    input_path = _optional_input_path(workdir)
    log_path = select_latest_log(workdir)
    pid = _read_pid(workdir / PID_FILE)
    running = pid is not None and _pid_is_running(pid)
    normal = gaussian_completed_normally(log_path)
    error = gaussian_has_error_termination(log_path)
    if normal:
        status = "completed"
        message = "Gaussian normal termination detected"
    elif running:
        status = "running"
        message = f"Gaussian process appears to be running (pid={pid})"
    elif error:
        status = "failed"
        message = "Gaussian error termination detected"
    elif log_path.exists():
        status = "unknown"
        message = f"Gaussian log exists without termination marker: {log_path.name}"
    elif input_path is not None:
        status = "ready"
        message = f"Gaussian input found: {input_path.name}"
    else:
        status = "missing"
        message = "No Gaussian input or log found"
    return GaussianJobStatus(
        workdir=workdir,
        input_path=input_path,
        log_path=log_path,
        status=status,
        normal_termination=normal,
        error_termination=error,
        pid=pid,
        message=message,
    )


def run_gaussian_job(
    workdir: Path,
    *,
    executable: str | None = None,
    input_path: Path | None = None,
    background: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> GaussianRunResult:
    """Run Gaussian from an ORACLE work directory.

    This is the non-GUI backend equivalent of Merlino's QProcess launcher.
    """
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    executable = executable or os.environ.get("ORACLE_GAUSSIAN_EXE") or GAUSSIAN_EXECUTABLE
    gauin = Path(input_path) if input_path is not None else ensure_gjf_input(workdir)
    if not gauin.is_absolute():
        gauin = workdir / gauin
    if not gauin.exists():
        raise GaussianInputError(f"Gaussian input not found: {gauin}")
    cmd = [executable, str(gauin)]
    log_path = select_latest_log(workdir)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    if background:
        process = subprocess.Popen(cmd, cwd=workdir, env=run_env)
        (workdir / PID_FILE).write_text(str(process.pid) + "\n", encoding="utf-8")
        return GaussianRunResult(
            workdir=workdir,
            input_path=gauin,
            log_path=log_path,
            executable=executable,
            pid=process.pid,
            exit_code=None,
            success=None,
            message=f"Gaussian started in background (pid={process.pid})",
        )
    completed = subprocess.run(cmd, cwd=workdir, env=run_env, timeout=timeout, check=False)
    success, message = gaussian_completion_message(workdir, int(completed.returncode))
    _write_finished_pid_file(workdir / PID_FILE, int(completed.returncode))
    return GaussianRunResult(
        workdir=workdir,
        input_path=gauin,
        log_path=select_latest_log(workdir),
        executable=executable,
        pid=None,
        exit_code=int(completed.returncode),
        success=success,
        message=message,
    )


def formchk_checkpoint(
    chk_path: Path,
    fchk_path: Path | None = None,
    *,
    executable: str = FORMCHK_EXECUTABLE,
    timeout: float | None = None,
) -> Path:
    chk = Path(chk_path)
    if fchk_path is None:
        fchk = chk.with_suffix(".fchk")
    else:
        fchk = Path(fchk_path)
    fchk.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([executable, str(chk), str(fchk)], cwd=chk.parent, check=True, timeout=timeout)
    return fchk


def _optional_input_path(workdir: Path) -> Path | None:
    for name in ("gauin.gjf", "gauin"):
        path = workdir / name
        if path.exists():
            return path
    return None


def _read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").splitlines()[0].strip()
        return int(raw)
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_finished_pid_file(path: Path, exit_code: int) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(f"finished exit_code={exit_code} at {stamp}\n", encoding="utf-8")
