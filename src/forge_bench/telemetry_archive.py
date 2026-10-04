from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .telemetry_contracts import (
    ArchiveState,
    CaptureProfile,
    TELEMETRY_SCHEMA_VERSION,
)

ARCHIVE_FORMAT = "forge-bench-telemetry-archive"
ARCHIVE_MANIFEST_SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ArchiveFileRecord:
    relative_path: str
    sha256: str
    size_bytes: int
    record_count: int | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _validate_archive_relative_path(self.relative_path)
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("record_count must be non-negative when supplied")


@dataclass(frozen=True)
class ArchiveManifest:
    experiment_id: str
    state: ArchiveState
    capture_profile: CaptureProfile
    created_at_utc: str
    sealed_at_utc: str | None = None
    raw_files: tuple[ArchiveFileRecord, ...] = ()
    raw_bytes: int = 0
    notes: tuple[str, ...] = ()
    archive_format: str = ARCHIVE_FORMAT
    manifest_schema_version: int = ARCHIVE_MANIFEST_SCHEMA_VERSION
    telemetry_schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if self.raw_bytes < 0:
            raise ValueError("raw_bytes must be non-negative")
        paths = [item.relative_path for item in self.raw_files]
        if len(paths) != len(set(paths)):
            raise ValueError("raw manifest paths must be unique")


@dataclass(frozen=True)
class TelemetryArchivePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def raw_cells(self) -> Path:
        return self.raw / "cells"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def share(self) -> Path:
        return self.root / "share"

    @property
    def manifest(self) -> Path:
        return self.root / "archive_manifest.json"

    def cell_raw(self, forge_cell_id: str) -> Path:
        _validate_component(forge_cell_id, "forge_cell_id")
        return self.raw_cells / forge_cell_id


def initialize_archive(
    root: str | Path,
    *,
    experiment_id: str,
    capture_profile: CaptureProfile = CaptureProfile.MAXIMAL,
) -> TelemetryArchivePaths:
    paths = TelemetryArchivePaths(Path(root))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.raw_cells.mkdir(parents=True, exist_ok=True)
    paths.derived.mkdir(parents=True, exist_ok=True)
    paths.share.mkdir(parents=True, exist_ok=True)

    if paths.manifest.exists():
        existing = read_manifest(paths.manifest)
        if existing.experiment_id != experiment_id:
            raise ValueError(
                "archive already belongs to experiment "
                f"{existing.experiment_id!r}, not {experiment_id!r}"
            )
        return paths

    manifest = ArchiveManifest(
        experiment_id=experiment_id,
        state=ArchiveState.OPEN,
        capture_profile=capture_profile,
        created_at_utc=_utc_now(),
    )
    _write_manifest_atomic(paths.manifest, manifest)
    return paths


def ensure_cell_layout(
    paths: TelemetryArchivePaths,
    forge_cell_id: str,
) -> Path:
    cell = paths.cell_raw(forge_cell_id)
    cell.mkdir(parents=True, exist_ok=True)
    return cell


def seal_archive(
    paths: TelemetryArchivePaths,
    *,
    state: ArchiveState,
    notes: Iterable[str] = (),
) -> ArchiveManifest:
    if state == ArchiveState.OPEN:
        raise ValueError("seal_archive state must be partial or complete")

    existing = read_manifest(paths.manifest)
    raw_files = tuple(_raw_file_records(paths))
    manifest = ArchiveManifest(
        experiment_id=existing.experiment_id,
        state=state,
        capture_profile=existing.capture_profile,
        created_at_utc=existing.created_at_utc,
        sealed_at_utc=_utc_now(),
        raw_files=raw_files,
        raw_bytes=sum(item.size_bytes for item in raw_files),
        notes=tuple(str(note) for note in notes),
    )
    _write_manifest_atomic(paths.manifest, manifest)
    return manifest


def verify_archive(paths: TelemetryArchivePaths) -> list[str]:
    manifest = read_manifest(paths.manifest)
    errors: list[str] = []

    if manifest.archive_format != ARCHIVE_FORMAT:
        errors.append(
            f"archive_format is {manifest.archive_format!r}, expected {ARCHIVE_FORMAT!r}"
        )
    if manifest.manifest_schema_version != ARCHIVE_MANIFEST_SCHEMA_VERSION:
        errors.append(
            "unsupported archive manifest schema "
            f"{manifest.manifest_schema_version}"
        )

    if manifest.state == ArchiveState.OPEN:
        return errors

    actual = {item.relative_path: item for item in _raw_file_records(paths)}
    expected = {item.relative_path: item for item in manifest.raw_files}

    for missing in sorted(set(expected) - set(actual)):
        errors.append(f"missing raw file: {missing}")
    for unexpected in sorted(set(actual) - set(expected)):
        errors.append(f"unexpected raw file after seal: {unexpected}")

    for relative_path in sorted(set(expected) & set(actual)):
        wanted = expected[relative_path]
        found = actual[relative_path]
        if wanted.sha256 != found.sha256:
            errors.append(f"sha256 mismatch: {relative_path}")
        if wanted.size_bytes != found.size_bytes:
            errors.append(f"size mismatch: {relative_path}")
        if (
            wanted.record_count is not None
            and found.record_count != wanted.record_count
        ):
            errors.append(f"record-count mismatch: {relative_path}")

    if manifest.raw_bytes != sum(item.size_bytes for item in actual.values()):
        errors.append("raw byte total does not match manifest")
    return errors


def read_manifest(path: str | Path) -> ArchiveManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_files = tuple(
        ArchiveFileRecord(**item) for item in payload.get("raw_files", [])
    )
    return ArchiveManifest(
        experiment_id=str(payload["experiment_id"]),
        state=ArchiveState(payload["state"]),
        capture_profile=CaptureProfile(payload["capture_profile"]),
        created_at_utc=str(payload["created_at_utc"]),
        sealed_at_utc=payload.get("sealed_at_utc"),
        raw_files=raw_files,
        raw_bytes=int(payload.get("raw_bytes", 0)),
        notes=tuple(str(note) for note in payload.get("notes", [])),
        archive_format=str(payload.get("archive_format", "")),
        manifest_schema_version=int(payload.get("manifest_schema_version", 0)),
        telemetry_schema_version=int(payload.get("telemetry_schema_version", 0)),
    )


def _raw_file_records(paths: TelemetryArchivePaths) -> list[ArchiveFileRecord]:
    records: list[ArchiveFileRecord] = []
    if not paths.raw.exists():
        return records
    for path in sorted(item for item in paths.raw.rglob("*") if item.is_file()):
        relative_path = path.relative_to(paths.root).as_posix()
        records.append(
            ArchiveFileRecord(
                relative_path=relative_path,
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
                record_count=_record_count(path),
                media_type=_media_type(path),
            )
        )
    return records


def _record_count(path: Path) -> int | None:
    if path.suffix not in {".jsonl", ".ndjson"}:
        return None
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".ndjson": "application/x-ndjson",
        ".parquet": "application/vnd.apache.parquet",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }.get(suffix)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest_atomic(path: Path, manifest: ArchiveManifest) -> None:
    payload = _manifest_to_dict(manifest)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_to_dict(manifest: ArchiveManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["state"] = manifest.state.value
    payload["capture_profile"] = manifest.capture_profile.value
    payload["raw_files"] = [asdict(item) for item in manifest.raw_files]
    return payload


def _validate_component(value: str, label: str) -> None:
    if not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_' or '-'"
        )


def _validate_archive_relative_path(value: str) -> None:
    normalized = value.replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in normalized.split("/")
        or not normalized.startswith("raw/")
    ):
        raise ValueError("archive file record must refer to a raw/ relative path")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
