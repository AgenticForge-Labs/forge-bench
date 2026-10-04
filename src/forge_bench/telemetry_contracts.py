from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

TELEMETRY_SCHEMA_VERSION = 1


class CaptureProfile(str, Enum):
    MAXIMAL = "maximal"
    STANDARD = "standard"
    MINIMAL = "minimal"


class AvailabilityStatus(str, Enum):
    CAPTURED = "captured"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class ArchiveState(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ClockOrigin:
    wall_time_unix_ns: int
    monotonic_ns: int

    @classmethod
    def capture(cls) -> "ClockOrigin":
        return cls(
            wall_time_unix_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
        )


@dataclass(frozen=True)
class TimePoint:
    wall_time_unix_ns: int
    monotonic_ns: int
    experiment_elapsed_ns: int
    cell_elapsed_ns: int | None = None

    @classmethod
    def capture(
        cls,
        experiment_origin: ClockOrigin,
        *,
        cell_origin: ClockOrigin | None = None,
    ) -> "TimePoint":
        monotonic_ns = time.monotonic_ns()
        wall_time_unix_ns = time.time_ns()
        experiment_elapsed_ns = monotonic_ns - experiment_origin.monotonic_ns
        cell_elapsed_ns = (
            None
            if cell_origin is None
            else monotonic_ns - cell_origin.monotonic_ns
        )
        if experiment_elapsed_ns < 0:
            raise ValueError("experiment monotonic clock moved backwards")
        if cell_elapsed_ns is not None and cell_elapsed_ns < 0:
            raise ValueError("cell monotonic clock moved backwards")
        return cls(
            wall_time_unix_ns=wall_time_unix_ns,
            monotonic_ns=monotonic_ns,
            experiment_elapsed_ns=experiment_elapsed_ns,
            cell_elapsed_ns=cell_elapsed_ns,
        )


@dataclass(frozen=True)
class TelemetryContext:
    experiment_id: str
    forge_cell_id: str | None = None
    run_index: int | None = None
    repeat: int | None = None
    agent: str | None = None
    agent_version: str | None = None
    model: str | None = None
    task: str | None = None
    environment: str | None = None
    factors: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if self.forge_cell_id is not None and not self.forge_cell_id.strip():
            raise ValueError("forge_cell_id must be non-empty when supplied")
        if self.run_index is not None and self.run_index < 1:
            raise ValueError("run_index must be >= 1 when supplied")
        if self.repeat is not None and self.repeat < 1:
            raise ValueError("repeat must be >= 1 when supplied")
        _validate_json_mapping(self.factors)


@dataclass(frozen=True)
class TelemetryCapability:
    name: str
    status: AvailabilityStatus
    source: str
    reason: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("capability name must be non-empty")
        if not self.source.strip():
            raise ValueError("capability source must be non-empty")
        if self.status != AvailabilityStatus.CAPTURED and not self.reason:
            raise ValueError(
                "non-captured telemetry capabilities must include a reason"
            )


@dataclass(frozen=True)
class TelemetryCapabilities:
    context: TelemetryContext
    profile: CaptureProfile
    capabilities: tuple[TelemetryCapability, ...]
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        names = [capability.name for capability in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("telemetry capability names must be unique")


@dataclass(frozen=True)
class TelemetryEvent:
    context: TelemetryContext
    time: TimePoint
    source: str
    name: str
    phase: str | None = None
    severity: str = "info"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common_record(
            source=self.source,
            name=self.name,
            attributes=self.attributes,
        )


@dataclass(frozen=True)
class MetricSample:
    context: TelemetryContext
    time: TimePoint
    source: str
    metric: str
    value: float
    unit: str
    kind: str = "gauge"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_common_record(
            source=self.source,
            name=self.metric,
            attributes=self.attributes,
        )
        if not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")
        if not self.unit.strip():
            raise ValueError("metric unit must be non-empty")
        if self.kind not in {"gauge", "counter"}:
            raise ValueError("metric kind must be 'gauge' or 'counter'")


@dataclass(frozen=True)
class ProcessSample:
    context: TelemetryContext
    time: TimePoint
    source: str
    pid: int
    parent_pid: int | None = None
    process_start_unix_ns: int | None = None
    status: str | None = None
    thread_count: int | None = None
    cpu_percent: float | None = None
    cpu_user_seconds: float | None = None
    cpu_system_seconds: float | None = None
    rss_bytes: int | None = None
    vms_bytes: int | None = None
    read_bytes: int | None = None
    write_bytes: int | None = None
    voluntary_context_switches: int | None = None
    involuntary_context_switches: int | None = None
    minor_page_faults: int | None = None
    major_page_faults: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.pid < 1:
            raise ValueError("pid must be >= 1")
        if self.parent_pid is not None and self.parent_pid < 1:
            raise ValueError("parent_pid must be >= 1 when supplied")
        _validate_nonnegative_fields(self)
        _validate_json_mapping(self.attributes)


@dataclass(frozen=True)
class SystemSample:
    context: TelemetryContext
    time: TimePoint
    source: str
    cpu_total_percent: float | None = None
    cpu_per_core_percent: tuple[float, ...] = ()
    load_1m: float | None = None
    load_5m: float | None = None
    load_15m: float | None = None
    memory_total_bytes: int | None = None
    memory_available_bytes: int | None = None
    memory_used_bytes: int | None = None
    swap_used_bytes: int | None = None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    network_rx_bytes: int | None = None
    network_tx_bytes: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        _validate_nonnegative_fields(self)
        if any(value < 0 for value in self.cpu_per_core_percent):
            raise ValueError("per-core CPU percentages must be non-negative")
        _validate_json_mapping(self.attributes)


@dataclass(frozen=True)
class GpuSample:
    context: TelemetryContext
    time: TimePoint
    source: str
    device_index: int
    device_uuid: str | None = None
    device_name: str | None = None
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    graphics_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    process_memory_bytes: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.device_index < 0:
            raise ValueError("device_index must be >= 0")
        _validate_nonnegative_fields(self)
        _validate_json_mapping(self.attributes)


@dataclass(frozen=True)
class ArtifactReference:
    context: TelemetryContext
    logical_name: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str | None = None
    role: str | None = None
    private: bool = True
    schema_version: int = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.logical_name.strip():
            raise ValueError("logical_name must be non-empty")
        _validate_relative_path(self.relative_path)
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


TelemetryRecord = (
    TelemetryEvent
    | MetricSample
    | ProcessSample
    | SystemSample
    | GpuSample
    | ArtifactReference
)


def record_to_dict(record: TelemetryRecord | TelemetryCapabilities) -> dict[str, Any]:
    payload = asdict(record)
    payload["record_type"] = _record_type(record)
    return _json_normalize(payload)


def record_to_json(record: TelemetryRecord | TelemetryCapabilities) -> str:
    return json.dumps(
        record_to_dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_type(record: object) -> str:
    names = {
        TelemetryEvent: "event",
        MetricSample: "metric",
        ProcessSample: "process",
        SystemSample: "system",
        GpuSample: "gpu",
        ArtifactReference: "artifact",
        TelemetryCapabilities: "capabilities",
    }
    record_type = names.get(type(record))
    if record_type is None:
        raise TypeError(f"unsupported telemetry record type: {type(record)!r}")
    return record_type


def _validate_common_record(
    *,
    source: str,
    name: str,
    attributes: Mapping[str, Any],
) -> None:
    if not source.strip():
        raise ValueError("source must be non-empty")
    if not name.strip():
        raise ValueError("name must be non-empty")
    _validate_json_mapping(attributes)


def _validate_json_mapping(value: Mapping[str, Any]) -> None:
    try:
        json.dumps(_json_normalize(dict(value)), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("telemetry attributes must be JSON-safe") from exc


def _validate_nonnegative_fields(record: object) -> None:
    for name, value in vars(record).items():
        if name in {
            "schema_version",
            "device_index",
            "pid",
            "parent_pid",
            "context",
            "time",
            "source",
            "status",
            "device_uuid",
            "device_name",
            "attributes",
        }:
            continue
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _validate_relative_path(value: str) -> None:
    if not value.strip():
        raise ValueError("relative_path must be non-empty")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if normalized.startswith("/") or ".." in parts:
        raise ValueError("relative_path must stay within the archive")


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    return value
