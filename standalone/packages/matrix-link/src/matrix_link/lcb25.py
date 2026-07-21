from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urljoin
from urllib.request import urlretrieve
import zipfile


LCB25_BASE_URL = "https://www.skies-village.it/webtools/Databases/LCB25/"
LCB25_DATASETS = ("PCS2", "SE", "HPCS2")
LCB25_CACHE_SCHEMA = "oracle.lcb25.cache.v1"


@dataclass(frozen=True)
class LCB25Dataset:
    label: str
    url: str
    archive_name: str


def lcb25_dataset_url(label: str) -> str:
    normalized = str(label).strip().upper()
    if normalized not in LCB25_DATASETS:
        raise ValueError(f"unsupported LCB25 dataset {label!r}; expected one of {LCB25_DATASETS}")
    return urljoin(LCB25_BASE_URL, f"{normalized}.zip")


def lcb25_download_plan() -> tuple[LCB25Dataset, ...]:
    return tuple(
        LCB25Dataset(label=label, url=lcb25_dataset_url(label), archive_name=f"{label}.zip")
        for label in LCB25_DATASETS
    )


def download_lcb25_dataset(label: str, target_dir: Path) -> Path:
    """Download one LCB25 ZIP archive to `target_dir`.

    Network use is explicit; tests cover URL planning without downloading.
    """
    dataset = LCB25Dataset(
        label=str(label).strip().upper(),
        url=lcb25_dataset_url(label),
        archive_name=f"{str(label).strip().upper()}.zip",
    )
    outdir = Path(target_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    archive = outdir / dataset.archive_name
    urlretrieve(dataset.url, archive)
    return archive


def sync_lcb25_library(
    target_root: Path,
    *,
    datasets: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
) -> Path:
    """Download, extract and manifest the local ORACLE-managed LCB25 cache."""
    root = Path(target_root)
    archive_dir = root / "archives"
    xyz_root = root / "xyz"
    archive_dir.mkdir(parents=True, exist_ok=True)
    xyz_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for label in _normalize_dataset_labels(datasets):
        archive = archive_dir / f"{label}.zip"
        if force or not archive.exists():
            archive = download_lcb25_dataset(label, archive_dir)

        extracted_dir = xyz_root / label
        existing_xyz = tuple(
            sorted(path for path in extracted_dir.rglob("*.xyz") if path.is_file())
        )
        if force and extracted_dir.exists():
            shutil.rmtree(extracted_dir)
            existing_xyz = ()
        xyz_files = existing_xyz or extract_lcb25_archive(archive, extracted_dir)
        entries.append(
            {
                "label": label,
                "url": lcb25_dataset_url(label),
                "archive": str(archive.relative_to(root)),
                "archive_sha256": _sha256_file(archive),
                "extracted_dir": str(extracted_dir.relative_to(root)),
                "xyz_count": len(xyz_files),
                "xyz_files": [str(path.relative_to(root)) for path in xyz_files],
            }
        )

    manifest = {
        "schema": LCB25_CACHE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": LCB25_BASE_URL,
        "datasets": entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def extract_lcb25_archive(archive: Path, target_dir: Path) -> tuple[Path, ...]:
    """Extract an LCB25 archive and return extracted XYZ files."""
    archive = Path(archive)
    outdir = Path(target_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (outdir / member.filename).resolve()
            if outdir.resolve() not in destination.parents and destination != outdir.resolve():
                raise ValueError(f"unsafe path in LCB25 archive: {member.filename}")
        zf.extractall(outdir)
    return tuple(sorted(path for path in outdir.rglob("*.xyz") if path.is_file()))


def _normalize_dataset_labels(datasets: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    labels = (
        LCB25_DATASETS
        if datasets is None
        else tuple(str(item).strip().upper() for item in datasets)
    )
    for label in labels:
        if label not in LCB25_DATASETS:
            raise ValueError(
                f"unsupported LCB25 dataset {label!r}; expected one of {LCB25_DATASETS}"
            )
    return tuple(labels)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
