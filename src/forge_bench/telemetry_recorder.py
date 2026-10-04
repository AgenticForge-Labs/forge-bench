from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .telemetry_archive import (
    TelemetryArchivePaths,
    assert_raw_writable,
    ensure_cell_layout,
    initialize_archive,
    read_manifest,
    seal_archive,
)
from .telemetry_contracts import (
    ArchiveState,
    CaptureProfile,
    ClockOrigin,
    TelemetryCapabilities,
    TelemetryContext,
    TelemetryEvent,
    TelemetryRecord,
    TimePoint,
    record_to_dict,
)

TELEMETRY_JOURNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RecoveryFragment:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RecorderIssue:
    operation: str
    exception_type: str
    message: str
    wall_time_unix_ns: int


class AppendOnlyJsonlJournal:
    """Durable append-only JSONL spool with monotonically increasing envelopes.

    Each successful append is flushed and fsynced by default. If a previous
    process died after writing only part of the final line, reopening preserves
    those trailing bytes under raw evidence/recovery before truncating the live
    journal back to its last complete newline.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        durable: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.durable = bool(durable)
        self._lock = threading.Lock()
        self.recovery_fragment = _recover_trailing_fragment(self.path)
        self._next_sequence = _validate_and_next_sequence(self.path)
        self._fd = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        self._closed = False

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def append(
        self,
        record: TelemetryRecord | TelemetryCapabilities,
    ) -> int:
        payload = {
            "journal_schema_version": TELEMETRY_JOURNAL_SCHEMA_VERSION,
            "sequence": self._next_sequence,
            "persisted_wall_time_unix_ns": time.time_ns(),
            "record": record_to_dict(record),
        }
        line = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry journal is closed")
            sequence = self._next_sequence
            _write_all(self._fd, line)
            if self.durable:
                os.fsync(self._fd)
            self._next_sequence += 1
            return sequence

    def sync(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry journal is closed")
            os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.durable:
                os.fsync(self._fd)
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> "AppendOnlyJsonlJournal":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close()
        return False


class CellTelemetryRecorder:
    def __init__(
        self,
        *,
        parent: "ExperimentTelemetryRecorder",
        context: TelemetryContext,
        cell_origin: ClockOrigin,
        journal: AppendOnlyJsonlJournal,
    ) -> None:
        self.parent = parent
        self.context = context
        self.cell_origin = cell_origin
        self.journal = journal
        self._closed = False

    def event(
        self,
        name: str,
        *,
        phase: str | None = None,
        severity: str = "info",
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        if self._closed:
            self.parent._issue(
                "cell.event",
                RuntimeError("cell telemetry recorder is closed"),
            )
            return False
        record = TelemetryEvent(
            context=self.context,
            time=TimePoint.capture(
                self.parent.experiment_origin,
                cell_origin=self.cell_origin,
            ),
            source="forge_bench",
            name=name,
            phase=phase,
            severity=severity,
            attributes=attributes or {},
        )
        return self.parent._append_safely(
            self.journal,
            record,
            operation=f"cell.event:{name}",
        )

    def capabilities(self, record: TelemetryCapabilities) -> bool:
        if record.context != self.context:
            self.parent._issue(
                "cell.capabilities",
                ValueError("capability context does not match cell context"),
            )
            return False
        return self.parent._append_safely(
            self.journal,
            record,
            operation="cell.capabilities",
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.journal.close()
        except Exception as exc:  # telemetry must not invalidate benchmark work
            self.parent._issue("cell.close", exc)
        self._closed = True


class ExperimentTelemetryRecorder:
    """Crash-resistant Forge lifecycle recorder over the raw telemetry archive."""

    def __init__(
        self,
        root: str | Path,
        *,
        experiment_id: str,
        capture_profile: CaptureProfile = CaptureProfile.MAXIMAL,
        durable: bool = True,
    ) -> None:
        root_path = Path(root)
        manifest_preexisted = (root_path / "archive_manifest.json").exists()
        self.paths = initialize_archive(
            root_path,
            experiment_id=experiment_id,
            capture_profile=capture_profile,
        )
        assert_raw_writable(self.paths)
        if manifest_preexisted:
            raise RuntimeError(
                "existing open telemetry archives are not resumed automatically; "
                "seal the interrupted archive partial and start a new experiment "
                "so monotonic elapsed clocks cannot silently reset"
            )
        self.experiment_id = experiment_id
        self.capture_profile = capture_profile
        self.durable = bool(durable)
        self.experiment_origin = ClockOrigin.capture()
        self.context = TelemetryContext(experiment_id=experiment_id)
        self.issues: list[RecorderIssue] = []
        self._cells: dict[str, CellTelemetryRecorder] = {}
        self._closed = False
        self._sealed_state: ArchiveState | None = None
        self._experiment_journal = AppendOnlyJsonlJournal(
            self.paths.raw / "experiment-events.jsonl",
            durable=self.durable,
        )

    @property
    def sealed_state(self) -> ArchiveState | None:
        return self._sealed_state

    @property
    def degraded(self) -> bool:
        return bool(self.issues)

    def event(
        self,
        name: str,
        *,
        phase: str | None = None,
        severity: str = "info",
        attributes: dict[str, Any] | None = None,
    ) -> bool:
        record = TelemetryEvent(
            context=self.context,
            time=TimePoint.capture(self.experiment_origin),
            source="forge_bench",
            name=name,
            phase=phase,
            severity=severity,
            attributes=attributes or {},
        )
        return self._append_safely(
            self._experiment_journal,
            record,
            operation=f"experiment.event:{name}",
        )

    def cell(
        self,
        intent: dict[str, Any],
        *,
        environment: str,
    ) -> CellTelemetryRecorder | None:
        cell_id = str(intent.get("forge_cell_id") or "").strip()
        if not cell_id:
            self._issue("cell.create", ValueError("intent is missing forge_cell_id"))
            return None
        existing = self._cells.get(cell_id)
        if existing is not None:
            return existing

        try:
            forge = dict(intent.get("forge") or {})
            context = TelemetryContext(
                experiment_id=self.experiment_id,
                forge_cell_id=cell_id,
                run_index=_optional_int(intent.get("forge_run_index")),
                repeat=_optional_int(forge.get("repeat")),
                agent=_optional_str(forge.get("agent")),
                agent_version=_optional_str(forge.get("agent_version")),
                model=_optional_str(forge.get("model")),
                task=_optional_str(forge.get("instance_id")),
                environment=str(environment),
                factors=_factor_payload(forge),
            )
            cell_dir = ensure_cell_layout(self.paths, cell_id)
            journal = AppendOnlyJsonlJournal(
                cell_dir / "events.jsonl",
                durable=self.durable,
            )
            recorder = CellTelemetryRecorder(
                parent=self,
                context=context,
                cell_origin=ClockOrigin.capture(),
                journal=journal,
            )
            self._cells[cell_id] = recorder
            return recorder
        except Exception as exc:
            self._issue(f"cell.create:{cell_id}", exc)
            return None

    def close_cell(self, forge_cell_id: str) -> None:
        recorder = self._cells.get(forge_cell_id)
        if recorder is not None:
            recorder.close()

    def complete(self, *, notes: Iterable[str] = ()) -> None:
        self.event(
            "experiment.completed",
            phase="experiment",
            attributes={"telemetry_degraded": self.degraded},
        )
        self._seal(
            ArchiveState.PARTIAL if self.degraded else ArchiveState.COMPLETE,
            notes=notes,
        )

    def interrupt(
        self,
        exc: BaseException,
        *,
        notes: Iterable[str] = (),
    ) -> None:
        self.event(
            "experiment.interrupted",
            phase="experiment",
            severity="error",
            attributes={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        combined = list(notes)
        combined.append(
            f"execution interrupted by {type(exc).__name__}: {exc}"
        )
        self._seal(ArchiveState.PARTIAL, notes=combined)

    def _seal(
        self,
        state: ArchiveState,
        *,
        notes: Iterable[str] = (),
    ) -> None:
        if self._closed:
            return
        for recorder in list(self._cells.values()):
            recorder.close()
        try:
            self._experiment_journal.close()
        except Exception as exc:
            self._issue("experiment.close", exc)

        issue_notes = [
            f"telemetry issue during {issue.operation}: "
            f"{issue.exception_type}: {issue.message}"
            for issue in self.issues
        ]
        try:
            manifest = seal_archive(
                self.paths,
                state=ArchiveState.PARTIAL if self.issues else state,
                notes=tuple(notes) + tuple(issue_notes),
            )
            self._sealed_state = manifest.state
        except Exception as exc:
            self._issue("archive.seal", exc)
        self._closed = True

    def _append_safely(
        self,
        journal: AppendOnlyJsonlJournal,
        record: TelemetryRecord | TelemetryCapabilities,
        *,
        operation: str,
    ) -> bool:
        try:
            journal.append(record)
            return True
        except Exception as exc:
            self._issue(operation, exc)
            return False

    def issue(self, operation: str, exc: BaseException) -> None:
        self._issue(operation, exc)

    def _issue(self, operation: str, exc: BaseException) -> None:
        self.issues.append(
            RecorderIssue(
                operation=operation,
                exception_type=type(exc).__name__,
                message=str(exc),
                wall_time_unix_ns=time.time_ns(),
            )
        )

    def __enter__(self) -> "ExperimentTelemetryRecorder":
        self.event(
            "experiment.started",
            phase="experiment",
            attributes={"capture_profile": self.capture_profile.value},
        )
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is None:
            self.complete()
        else:
            self.interrupt(exc)
        return False


def recover_interrupted_archive(
    root: str | Path,
    *,
    reason: str,
) -> ArchiveState:
    """Recover torn JSONL tails and seal an abandoned open archive partial.

    This intentionally does not resume the experiment: monotonic elapsed clocks
    belong to the original process and cannot be reconstructed after a hard
    process/system failure.
    """
    paths = TelemetryArchivePaths(Path(root))
    manifest = read_manifest(paths.manifest)
    if manifest.state != ArchiveState.OPEN:
        raise RuntimeError(
            f"telemetry archive is already sealed as {manifest.state.value}"
        )
    for path in sorted(paths.raw.rglob("*.jsonl")):
        journal = AppendOnlyJsonlJournal(path)
        journal.close()
    sealed = seal_archive(
        paths,
        state=ArchiveState.PARTIAL,
        notes=(f"recovered interrupted archive: {reason}",),
    )
    return sealed.state


def _factor_payload(forge: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "agent_key",
        "model_key",
        "arm",
        "max_turns",
        "budget_warning_ratio",
        "api_provider",
        "upstream_provider",
        "reasoning",
        "repeat_seed",
        "block_position",
    )
    return {
        field: forge[field]
        for field in fields
        if field in forge
    }


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _validate_and_next_sequence(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 1
    expected = 1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid telemetry JSONL at {path}:{line_number}"
                ) from exc
            if int(payload.get("journal_schema_version", 0)) != TELEMETRY_JOURNAL_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported telemetry journal schema at {path}:{line_number}"
                )
            sequence = int(payload.get("sequence", 0))
            if sequence != expected:
                raise ValueError(
                    f"telemetry sequence mismatch at {path}:{line_number}: "
                    f"expected {expected}, found {sequence}"
                )
            if not isinstance(payload.get("record"), dict):
                raise ValueError(
                    f"telemetry journal record missing at {path}:{line_number}"
                )
            expected += 1
    return expected


def _recover_trailing_fragment(path: Path) -> RecoveryFragment | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return None
        last_newline = _find_last_newline(handle)
        boundary = 0 if last_newline is None else last_newline + 1
        handle.seek(boundary)
        fragment = handle.read()

    if not fragment:
        return None

    digest = hashlib.sha256(fragment).hexdigest()
    recovery_dir = path.parent / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    recovery_path = recovery_dir / (
        f"{path.name}.trailing-{digest[:12]}.bin"
    )
    if not recovery_path.exists():
        recovery_path.write_bytes(fragment)
        _fsync_path(recovery_path)

    with path.open("r+b") as handle:
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())

    return RecoveryFragment(
        relative_path=recovery_path.name,
        sha256=digest,
        size_bytes=len(fragment),
    )


def _find_last_newline(handle) -> int | None:
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    position = end
    chunk_size = 64 * 1024
    while position > 0:
        start = max(0, position - chunk_size)
        handle.seek(start)
        chunk = handle.read(position - start)
        index = chunk.rfind(b"\n")
        if index >= 0:
            return start + index
        position = start
    return None


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(fd, view[written:])
        if count <= 0:
            raise OSError("short write to telemetry journal")
        written += count


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
