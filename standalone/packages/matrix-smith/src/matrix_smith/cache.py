"""SMITH JSON cache with optional scientific-state invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matrix_core import atomic_json_write, scientific_cache_envelope, unwrap_scientific_cache


def smith_cache_key(*parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()


def cache_file(cache_dir: str | Path, key: str, suffix: str = ".json") -> Path:
    target = Path(cache_dir) / "smith" / key[:2] / (key + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_json(
    cache_dir: str | Path,
    key: str,
    *,
    expected_state_sha256: str | None = None,
) -> dict[str, object] | None:
    target = cache_file(cache_dir, key)
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if expected_state_sha256 is None:
        return payload
    return unwrap_scientific_cache(
        payload,
        expected_state_sha256=expected_state_sha256,
    )


def store_json(
    cache_dir: str | Path,
    key: str,
    payload: dict[str, object],
    *,
    state_sha256: str | None = None,
) -> Path:
    target = cache_file(cache_dir, key)
    stored = (
        payload
        if state_sha256 is None
        else scientific_cache_envelope(payload, state_sha256=state_sha256)
    )
    atomic_json_write(target, stored, allow_nan=False)
    return target
