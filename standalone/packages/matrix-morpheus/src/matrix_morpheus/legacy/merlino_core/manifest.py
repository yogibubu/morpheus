from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ORACLE_MANIFEST_SCHEMA = "oracle.run.v1"


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, data: Mapping[str, Any]) -> Path:
    """Write a deterministic JSON manifest and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


@dataclass(frozen=True)
class RunManifest:
    """Canonical manifest for an ORACLE workflow run."""

    workflow: str
    status: str
    run_dir: Path
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    input_sha256: dict[str, str] = field(default_factory=dict)
    output_sha256: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    schema_version: str = ORACLE_MANIFEST_SCHEMA
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        return data

    def write(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self.run_dir / "manifest.json"
        return write_manifest(target, self.to_dict())


def file_checksums(paths: Mapping[str, Path]) -> dict[str, str]:
    """Return checksums for existing regular files."""
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
        if Path(path).is_file()
    }


def build_run_manifest(
    *,
    workflow: str,
    status: str,
    run_dir: Path,
    inputs: Mapping[str, Path] | None = None,
    outputs: Mapping[str, Path] | None = None,
    parameters: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    messages: list[str] | None = None,
) -> RunManifest:
    input_paths = dict(inputs or {})
    output_paths = dict(outputs or {})
    return RunManifest(
        workflow=workflow,
        status=status,
        run_dir=Path(run_dir),
        inputs={name: str(path) for name, path in sorted(input_paths.items())},
        outputs={name: str(path) for name, path in sorted(output_paths.items())},
        input_sha256=file_checksums(input_paths),
        output_sha256=file_checksums(output_paths),
        parameters=dict(parameters or {}),
        backend=dict(backend or {}),
        messages=list(messages or []),
    )
