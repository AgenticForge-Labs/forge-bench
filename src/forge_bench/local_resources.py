from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from .telemetry_contracts import (
    AvailabilityStatus,
    CaptureProfile,
    ProcessSample,
    SystemSample,
    TelemetryCapabilities,
    TelemetryCapability,
    TimePoint,
)
from .telemetry_recorder import AppendOnlyJsonlJournal, ExperimentTelemetryRecorder


@dataclass(frozen=True)
class LocalResourceSamplerStats:
    sample_cycles: int
    system_samples: int
    process_samples: int
    max_processes_per_cycle: int
    root_pid: int
    interval_seconds: float
    sync_every_cycles: int


class LocalProcessResourceSampler:
    """Sample local host CPU/RAM and the Forge/Harbor process tree.

    This sampler intentionally does not claim visibility into Dockerized agent
    processes. Docker/cgroup attribution belongs to the next container telemetry
    layer. Process identity uses PID plus process creation time so PID reuse does
    not merge longitudinal trajectories.
    """

    def __init__(
        self,
        recorder: ExperimentTelemetryRecorder,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 1.0,
        sync_every_cycles: int = 5,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if sync_every_cycles < 1:
            raise ValueError("sync_every_cycles must be >= 1")

        self.recorder = recorder
        self.root_pid = int(root_pid or os.getpid())
        self.interval_seconds = float(interval_seconds)
        self.sync_every_cycles = int(sync_every_cycles)
        self.profile = recorder.capture_profile
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._sample_cycles = 0
        self._system_samples = 0
        self._process_samples = 0
        self._max_processes = 0
        self._previous_process_cpu: dict[tuple[int, int], tuple[int, float]] = {}

        self._system_journal = AppendOnlyJsonlJournal(
            recorder.paths.raw / "system.jsonl",
            durable=False,
        )
        self._process_journal = AppendOnlyJsonlJournal(
            recorder.paths.raw / "processes.jsonl",
            durable=False,
        )
        self._capabilities_journal = AppendOnlyJsonlJournal(
            recorder.paths.raw / "resource-capabilities.jsonl",
            durable=True,
        )
        self._capabilities_journal.append(self._capabilities())
        self._capabilities_journal.close()

        # Prime psutil's non-blocking CPU percentage state before the first
        # timed sample. Cumulative CPU counters are also stored independently.
        psutil.cpu_percent(interval=None, percpu=True)

    @property
    def stats(self) -> LocalResourceSamplerStats:
        return LocalResourceSamplerStats(
            sample_cycles=self._sample_cycles,
            system_samples=self._system_samples,
            process_samples=self._process_samples,
            max_processes_per_cycle=self._max_processes,
            root_pid=self.root_pid,
            interval_seconds=self.interval_seconds,
            sync_every_cycles=self.sync_every_cycles,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("local resource sampler is closed")
        if self._thread is not None:
            raise RuntimeError("local resource sampler has already been started")

        self.recorder.event(
            "telemetry.local_resources.started",
            phase="telemetry",
            attributes={
                "root_pid": self.root_pid,
                "interval_seconds": self.interval_seconds,
                "scope": "local_host_and_forge_process_tree",
                "profile": self.profile.value,
            },
        )
        self._thread = threading.Thread(
            target=self._run,
            name="forge-bench-local-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> LocalResourceSamplerStats:
        if self._closed:
            return self.stats
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))
            if self._thread.is_alive():
                self.recorder.issue(
                    "local_resources.stop",
                    RuntimeError("resource sampler thread did not stop promptly"),
                )

        try:
            self._system_journal.sync()
            self._process_journal.sync()
        except Exception as exc:
            self.recorder.issue("local_resources.sync", exc)

        for name, journal in (
            ("system", self._system_journal),
            ("process", self._process_journal),
        ):
            try:
                journal.close()
            except Exception as exc:
                self.recorder.issue(f"local_resources.close:{name}", exc)

        self._closed = True
        stats = self.stats
        self.recorder.event(
            "telemetry.local_resources.stopped",
            phase="telemetry",
            attributes={
                "sample_cycles": stats.sample_cycles,
                "system_samples": stats.system_samples,
                "process_samples": stats.process_samples,
                "max_processes_per_cycle": stats.max_processes_per_cycle,
            },
        )
        return stats

    def sample_once(self) -> tuple[SystemSample, list[ProcessSample]]:
        if self._closed:
            raise RuntimeError("local resource sampler is closed")

        point = TimePoint.capture(self.recorder.experiment_origin)
        system_sample = self._system_sample(point)
        process_samples = self._process_samples_for_tree(point)

        self._system_journal.append(system_sample)
        for sample in process_samples:
            self._process_journal.append(sample)

        self._sample_cycles += 1
        self._system_samples += 1
        self._process_samples += len(process_samples)
        self._max_processes = max(self._max_processes, len(process_samples))

        if self._sample_cycles % self.sync_every_cycles == 0:
            self._system_journal.sync()
            self._process_journal.sync()

        return system_sample, process_samples

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
                self.recorder.issue("local_resources.root_process", exc)
                break
            except Exception as exc:
                # Telemetry is evidence, not an execution dependency.
                self.recorder.issue("local_resources.sample", exc)
                self.recorder.event(
                    "telemetry.local_resources.error",
                    phase="telemetry",
                    severity="error",
                    attributes={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                break

            if self._stop.wait(self.interval_seconds):
                break

    def _system_sample(self, point: TimePoint) -> SystemSample:
        per_core = tuple(float(value) for value in psutil.cpu_percent(
            interval=None,
            percpu=True,
        ))
        cpu_total = (
            sum(per_core) / len(per_core)
            if per_core
            else None
        )
        cpu_times = psutil.cpu_times()
        virtual_memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load = _load_average()
        frequencies = _cpu_frequencies()

        attributes: dict[str, Any] = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_pid": os.getpid(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        }
        if frequencies:
            attributes["cpu_per_core_frequency_mhz"] = frequencies

        return SystemSample(
            context=self.recorder.context,
            time=point,
            source="psutil",
            cpu_total_percent=cpu_total,
            cpu_per_core_percent=per_core,
            cpu_user_seconds=_optional_float(getattr(cpu_times, "user", None)),
            cpu_system_seconds=_optional_float(getattr(cpu_times, "system", None)),
            cpu_idle_seconds=_optional_float(getattr(cpu_times, "idle", None)),
            cpu_iowait_seconds=_optional_float(getattr(cpu_times, "iowait", None)),
            cpu_frequency_mhz=(
                sum(frequencies) / len(frequencies)
                if frequencies
                else None
            ),
            load_1m=load[0],
            load_5m=load[1],
            load_15m=load[2],
            memory_total_bytes=int(virtual_memory.total),
            memory_available_bytes=int(virtual_memory.available),
            memory_used_bytes=int(virtual_memory.used),
            swap_used_bytes=int(swap.used),
            attributes=attributes,
        )

    def _process_samples_for_tree(self, point: TimePoint) -> list[ProcessSample]:
        root = psutil.Process(self.root_pid)
        processes = [root]
        try:
            processes.extend(root.children(recursive=True))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            pass

        unique: dict[tuple[int, int], psutil.Process] = {}
        for process in processes:
            try:
                create_ns = int(process.create_time() * 1_000_000_000)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            unique[(process.pid, create_ns)] = process

        samples: list[ProcessSample] = []
        live_keys: set[tuple[int, int]] = set()
        for key in sorted(unique):
            process = unique[key]
            sample = self._process_sample(process, key, point)
            if sample is not None:
                samples.append(sample)
                live_keys.add(key)

        # Prevent unbounded state growth after short-lived child processes exit.
        self._previous_process_cpu = {
            key: value
            for key, value in self._previous_process_cpu.items()
            if key in live_keys
        }
        return samples

    def _process_sample(
        self,
        process: psutil.Process,
        key: tuple[int, int],
        point: TimePoint,
    ) -> ProcessSample | None:
        denied: list[str] = []
        try:
            with process.oneshot():
                parent_pid = _process_value(process, "ppid", denied)
                status = _process_value(process, "status", denied)
                thread_count = _process_value(process, "num_threads", denied)
                cpu_times = _process_value(process, "cpu_times", denied)
                memory = _process_value(process, "memory_info", denied)
                full_memory = (
                    _process_value(process, "memory_full_info", denied)
                    if self.profile == CaptureProfile.MAXIMAL
                    else None
                )
                io = _process_value(process, "io_counters", denied)
                switches = _process_value(process, "num_ctx_switches", denied)
                name = _process_value(process, "name", denied)
                exe = _process_value(process, "exe", denied)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None

        user_seconds = _optional_float(getattr(cpu_times, "user", None))
        system_seconds = _optional_float(getattr(cpu_times, "system", None))
        cpu_total_seconds = (
            None
            if user_seconds is None or system_seconds is None
            else user_seconds + system_seconds
        )
        cpu_percent = self._process_cpu_percent(
            key,
            point.monotonic_ns,
            cpu_total_seconds,
        )
        minor_faults, major_faults = _linux_page_faults(process.pid)

        attributes: dict[str, Any] = {
            "is_root": process.pid == self.root_pid,
        }
        if name not in (None, ""):
            attributes["name"] = str(name)
        if exe not in (None, ""):
            attributes["exe"] = str(exe)
        if denied:
            attributes["unavailable_fields"] = sorted(set(denied))
        if minor_faults is not None or major_faults is not None:
            attributes["page_fault_source"] = "procfs"

        return ProcessSample(
            context=self.recorder.context,
            time=point,
            source="psutil",
            pid=process.pid,
            parent_pid=_optional_int(parent_pid),
            process_start_unix_ns=key[1],
            status=None if status in (None, "") else str(status),
            thread_count=_optional_int(thread_count),
            cpu_percent=cpu_percent,
            cpu_user_seconds=user_seconds,
            cpu_system_seconds=system_seconds,
            rss_bytes=_attr_int(memory, "rss"),
            vms_bytes=_attr_int(memory, "vms"),
            uss_bytes=_attr_int(full_memory, "uss"),
            pss_bytes=_attr_int(full_memory, "pss"),
            swap_bytes=_attr_int(full_memory, "swap"),
            read_bytes=_attr_int(io, "read_bytes"),
            write_bytes=_attr_int(io, "write_bytes"),
            voluntary_context_switches=_attr_int(switches, "voluntary"),
            involuntary_context_switches=_attr_int(switches, "involuntary"),
            minor_page_faults=minor_faults,
            major_page_faults=major_faults,
            attributes=attributes,
        )

    def _process_cpu_percent(
        self,
        key: tuple[int, int],
        monotonic_ns: int,
        cpu_total_seconds: float | None,
    ) -> float | None:
        if cpu_total_seconds is None:
            return None
        previous = self._previous_process_cpu.get(key)
        self._previous_process_cpu[key] = (monotonic_ns, cpu_total_seconds)
        if previous is None:
            return None
        previous_ns, previous_cpu = previous
        elapsed_seconds = (monotonic_ns - previous_ns) / 1_000_000_000
        if elapsed_seconds <= 0:
            return None
        return max(0.0, (cpu_total_seconds - previous_cpu) / elapsed_seconds * 100.0)

    def _capabilities(self) -> TelemetryCapabilities:
        root = psutil.Process(self.root_pid)
        capabilities = [
            TelemetryCapability(
                name="host.cpu",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="percent+seconds",
            ),
            TelemetryCapability(
                name="host.memory",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="By",
            ),
            TelemetryCapability(
                name="process.tree",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
            ),
            TelemetryCapability(
                name="process.cpu",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="percent+seconds",
            ),
            TelemetryCapability(
                name="process.memory",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="By",
            ),
            TelemetryCapability(
                name="process.io",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="By",
            ),
            TelemetryCapability(
                name="process.context_switches",
                status=AvailabilityStatus.CAPTURED,
                source="psutil",
                unit="{switch}",
            ),
            _full_memory_capability(root),
            _page_fault_capability(root.pid),
            _frequency_capability(),
            TelemetryCapability(
                name="container.workload",
                status=AvailabilityStatus.NOT_APPLICABLE,
                source="forge_bench",
                reason=(
                    "container/cgroup attribution is intentionally deferred to "
                    "the container telemetry layer; host process ancestry must "
                    "not be mislabeled as Docker workload telemetry"
                ),
            ),
            TelemetryCapability(
                name="host.disk",
                status=AvailabilityStatus.NOT_APPLICABLE,
                source="forge_bench",
                reason="disk/filesystem telemetry is deferred to the next layer",
            ),
            TelemetryCapability(
                name="host.network",
                status=AvailabilityStatus.NOT_APPLICABLE,
                source="forge_bench",
                reason="network telemetry is deferred to the next layer",
            ),
            TelemetryCapability(
                name="gpu",
                status=AvailabilityStatus.NOT_APPLICABLE,
                source="forge_bench",
                reason="GPU telemetry is implemented in a dedicated later layer",
            ),
        ]
        return TelemetryCapabilities(
            context=self.recorder.context,
            profile=self.profile,
            capabilities=tuple(capabilities),
        )


def record_local_resources_not_applicable(
    recorder: ExperimentTelemetryRecorder,
    *,
    reason: str,
) -> None:
    """Record that local host/process samples would not describe the trial."""
    journal = AppendOnlyJsonlJournal(
        recorder.paths.raw / "resource-capabilities.jsonl",
        durable=True,
    )
    try:
        journal.append(
            TelemetryCapabilities(
                context=recorder.context,
                profile=recorder.capture_profile,
                capabilities=tuple(
                    TelemetryCapability(
                        name=name,
                        status=AvailabilityStatus.NOT_APPLICABLE,
                        source="forge_bench",
                        reason=reason,
                    )
                    for name in (
                        "host.cpu",
                        "host.memory",
                        "process.tree",
                        "process.cpu",
                        "process.memory",
                    )
                ),
            )
        )
    finally:
        journal.close()


def _process_value(
    process: psutil.Process,
    method: str,
    denied: list[str],
) -> Any:
    try:
        return getattr(process, method)()
    except (psutil.AccessDenied, PermissionError):
        denied.append(method)
        return None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        raise


def _load_average() -> tuple[float | None, float | None, float | None]:
    try:
        values = psutil.getloadavg()
    except (AttributeError, OSError):
        return None, None, None
    return tuple(float(value) for value in values)


def _cpu_frequencies() -> tuple[float, ...]:
    try:
        values = psutil.cpu_freq(percpu=True)
    except (AttributeError, OSError, NotImplementedError):
        return ()
    if not values:
        return ()
    return tuple(
        float(item.current)
        for item in values
        if getattr(item, "current", None) is not None
        and float(item.current) >= 0
    )


def _full_memory_capability(root: psutil.Process) -> TelemetryCapability:
    if not hasattr(root, "memory_full_info"):
        return TelemetryCapability(
            name="process.memory_full",
            status=AvailabilityStatus.UNAVAILABLE,
            source="psutil",
            reason="psutil does not expose memory_full_info on this platform",
            unit="By",
        )
    try:
        full = root.memory_full_info()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        return TelemetryCapability(
            name="process.memory_full",
            status=AvailabilityStatus.UNAVAILABLE,
            source="psutil",
            reason=f"{type(exc).__name__}: {exc}",
            unit="By",
        )
    if not any(hasattr(full, name) for name in ("uss", "pss", "swap")):
        return TelemetryCapability(
            name="process.memory_full",
            status=AvailabilityStatus.UNAVAILABLE,
            source="psutil",
            reason="USS/PSS/swap fields are not exposed on this platform",
            unit="By",
        )
    return TelemetryCapability(
        name="process.memory_full",
        status=AvailabilityStatus.CAPTURED,
        source="psutil",
        unit="By",
    )


def _page_fault_capability(pid: int) -> TelemetryCapability:
    minor, major = _linux_page_faults(pid)
    if minor is None and major is None:
        return TelemetryCapability(
            name="process.page_faults",
            status=AvailabilityStatus.UNAVAILABLE,
            source="procfs",
            reason="/proc process fault counters are unavailable",
            unit="{fault}",
        )
    return TelemetryCapability(
        name="process.page_faults",
        status=AvailabilityStatus.CAPTURED,
        source="procfs",
        unit="{fault}",
    )


def _frequency_capability() -> TelemetryCapability:
    if _cpu_frequencies():
        return TelemetryCapability(
            name="host.cpu_frequency",
            status=AvailabilityStatus.CAPTURED,
            source="psutil",
            unit="MHz",
        )
    return TelemetryCapability(
        name="host.cpu_frequency",
        status=AvailabilityStatus.UNAVAILABLE,
        source="psutil",
        reason="CPU frequency is not exposed on this platform",
        unit="MHz",
    )


def _linux_page_faults(pid: int) -> tuple[int | None, int | None]:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None, None

    close = text.rfind(")")
    if close < 0:
        return None, None
    fields = text[close + 2 :].split()
    if len(fields) <= 9:
        return None, None
    try:
        # fields[0] is proc stat field 3 (state); minflt is field 10 and
        # majflt is field 12.
        return int(fields[7]), int(fields[9])
    except (TypeError, ValueError):
        return None, None


def _attr_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    field = getattr(value, name, None)
    if field is None:
        return None
    try:
        number = int(field)
    except (TypeError, ValueError):
        return None
    return max(0, number)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
