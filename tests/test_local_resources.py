from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from forge_bench.harbor_cli import _resource_sampling_interval
from forge_bench.local_resources import (
    LocalProcessResourceSampler,
    record_local_resources_not_applicable,
)
from forge_bench.telemetry_archive import read_manifest, verify_archive
from forge_bench.telemetry_contracts import (
    ArchiveState,
    AvailabilityStatus,
    CaptureProfile,
)
from forge_bench.telemetry_recorder import ExperimentTelemetryRecorder


def _journal_records(path: Path) -> list[dict]:
    return [
        json.loads(line)["record"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class LocalProcessResourceSamplerTests(unittest.TestCase):
    def test_sample_once_captures_host_and_forge_process_tree(self):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import time; "
                    "payload = bytearray(4 * 1024 * 1024); "
                    "time.sleep(5)"
                ),
            ]
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "telemetry"
                recorder = ExperimentTelemetryRecorder(
                    root,
                    experiment_id="experiment-1",
                    capture_profile=CaptureProfile.MAXIMAL,
                )
                with recorder:
                    sampler = LocalProcessResourceSampler(
                        recorder,
                        interval_seconds=0.05,
                        sync_every_cycles=1,
                    )
                    system_one, processes_one = sampler.sample_once()
                    time.sleep(0.03)
                    system_two, processes_two = sampler.sample_once()
                    stats = sampler.stop()

                self.assertGreater(system_one.memory_total_bytes or 0, 0)
                self.assertGreaterEqual(system_one.cpu_user_seconds or 0.0, 0.0)
                self.assertEqual(system_one.cpu_per_core_percent, ())
                self.assertGreater(len(system_two.cpu_per_core_percent), 0)
                self.assertIsNotNone(system_two.cpu_total_percent)

                pids = {sample.pid for sample in processes_one}
                self.assertIn(os.getpid(), pids)
                self.assertIn(child.pid, pids)
                self.assertTrue(
                    all((sample.rss_bytes or 0) >= 0 for sample in processes_one)
                )

                second_root = next(
                    sample for sample in processes_two if sample.pid == os.getpid()
                )
                self.assertIsNotNone(second_root.cpu_percent)
                self.assertGreaterEqual(second_root.cpu_percent or 0.0, 0.0)

                self.assertEqual(stats.sample_cycles, 2)
                self.assertEqual(stats.system_samples, 2)
                self.assertGreaterEqual(stats.process_samples, 4)

                system_records = _journal_records(root / "raw" / "system.jsonl")
                process_records = _journal_records(root / "raw" / "processes.jsonl")
                self.assertEqual(len(system_records), 2)
                self.assertEqual(
                    {row["record_type"] for row in system_records},
                    {"system"},
                )
                self.assertEqual(
                    {row["record_type"] for row in process_records},
                    {"process"},
                )
                self.assertIn(
                    child.pid,
                    {row["pid"] for row in process_records},
                )

                manifest = read_manifest(root / "archive_manifest.json")
                self.assertEqual(manifest.state, ArchiveState.COMPLETE)
                self.assertEqual(verify_archive(recorder.paths), [])
        finally:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)

    def test_background_sampling_runs_multiple_cycles_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                sampler = LocalProcessResourceSampler(
                    recorder,
                    interval_seconds=0.03,
                    sync_every_cycles=1,
                )
                sampler.start()
                time.sleep(0.11)
                stats = sampler.stop()

            self.assertGreaterEqual(stats.sample_cycles, 2)
            self.assertGreaterEqual(stats.system_samples, 2)
            self.assertGreaterEqual(stats.process_samples, 2)
            self.assertEqual(verify_archive(recorder.paths), [])

    def test_capabilities_make_container_gap_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                sampler = LocalProcessResourceSampler(recorder)
                sampler.stop()

            records = _journal_records(
                root / "raw" / "resource-capabilities.jsonl"
            )
            self.assertEqual(len(records), 1)
            capabilities = {
                item["name"]: item
                for item in records[0]["capabilities"]
            }
            self.assertEqual(
                capabilities["host.cpu"]["status"],
                AvailabilityStatus.CAPTURED.value,
            )
            self.assertEqual(
                capabilities["host.memory"]["status"],
                AvailabilityStatus.CAPTURED.value,
            )
            self.assertEqual(
                capabilities["container.workload"]["status"],
                AvailabilityStatus.NOT_APPLICABLE.value,
            )
            self.assertIn(
                "container/cgroup attribution",
                capabilities["container.workload"]["reason"],
            )

    def test_standard_profile_avoids_expensive_full_memory_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
                capture_profile=CaptureProfile.STANDARD,
            )
            with recorder:
                sampler = LocalProcessResourceSampler(recorder)
                _system, processes = sampler.sample_once()
                sampler.stop()

            self.assertTrue(processes)
            self.assertTrue(all(sample.uss_bytes is None for sample in processes))
            capabilities = _journal_records(
                root / "raw" / "resource-capabilities.jsonl"
            )[0]["capabilities"]
            full_memory = next(
                item for item in capabilities if item["name"] == "process.memory_full"
            )
            self.assertEqual(
                full_memory["status"],
                AvailabilityStatus.NOT_APPLICABLE.value,
            )

    def test_sampler_failure_degrades_telemetry_without_raising_into_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                sampler = LocalProcessResourceSampler(
                    recorder,
                    interval_seconds=0.01,
                )
                with patch.object(
                    sampler,
                    "sample_once",
                    side_effect=OSError("synthetic sampler failure"),
                ):
                    sampler.start()
                    time.sleep(0.04)
                    sampler.stop()

            self.assertTrue(recorder.degraded)
            manifest = read_manifest(root / "archive_manifest.json")
            self.assertEqual(manifest.state, ArchiveState.PARTIAL)
            self.assertTrue(
                any("synthetic sampler failure" in note for note in manifest.notes)
            )

    def test_remote_environment_can_record_local_sampling_not_applicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                record_local_resources_not_applicable(
                    recorder,
                    reason="remote Modal sandbox owns resource execution",
                )

            record = _journal_records(
                root / "raw" / "resource-capabilities.jsonl"
            )[0]
            self.assertTrue(
                all(
                    item["status"] == AvailabilityStatus.NOT_APPLICABLE.value
                    for item in record["capabilities"]
                )
            )
            self.assertTrue(
                all(
                    "remote Modal" in item["reason"]
                    for item in record["capabilities"]
                )
            )


class CaptureProfileIntervalTests(unittest.TestCase):
    def test_default_intervals_follow_capture_profile(self):
        self.assertEqual(
            _resource_sampling_interval(
                CaptureProfile.MAXIMAL,
                requested_interval=None,
            ),
            1.0,
        )
        self.assertEqual(
            _resource_sampling_interval(
                CaptureProfile.STANDARD,
                requested_interval=None,
            ),
            2.0,
        )
        self.assertIsNone(
            _resource_sampling_interval(
                CaptureProfile.MINIMAL,
                requested_interval=None,
            )
        )

    def test_explicit_interval_is_validated(self):
        self.assertEqual(
            _resource_sampling_interval(
                CaptureProfile.MAXIMAL,
                requested_interval=0.25,
            ),
            0.25,
        )
        with self.assertRaisesRegex(SystemExit, "must be > 0"):
            _resource_sampling_interval(
                CaptureProfile.MAXIMAL,
                requested_interval=0,
            )
        with self.assertRaisesRegex(SystemExit, "minimal"):
            _resource_sampling_interval(
                CaptureProfile.MINIMAL,
                requested_interval=0.5,
            )


if __name__ == "__main__":
    unittest.main()
