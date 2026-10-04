from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from forge_bench.telemetry_archive import (
    ensure_cell_layout,
    initialize_archive,
    read_manifest,
    seal_archive,
    verify_archive,
)
from forge_bench.telemetry_contracts import (
    ArchiveState,
    ArtifactReference,
    AvailabilityStatus,
    CaptureProfile,
    ClockOrigin,
    MetricSample,
    TelemetryCapabilities,
    TelemetryCapability,
    TelemetryContext,
    TelemetryEvent,
    TimePoint,
    record_to_dict,
    record_to_json,
)


class TelemetryContractTests(unittest.TestCase):
    def _context(self) -> TelemetryContext:
        return TelemetryContext(
            experiment_id="experiment-1",
            forge_cell_id="forge-0001-abc",
            run_index=1,
            repeat=1,
            agent="hermes",
            model="provider/model",
            task="task-a",
            environment="docker",
            factors={"treatment": "baseline", "max_turns": 100},
        )

    def test_event_and_metric_are_stable_json_records(self):
        origin = ClockOrigin.capture()
        time.sleep(0.001)
        point = TimePoint.capture(origin)
        event = TelemetryEvent(
            context=self._context(),
            time=point,
            source="forge",
            name="cell.started",
            attributes={"attempt": 1, "labels": ["a", "b"]},
        )
        metric = MetricSample(
            context=self._context(),
            time=point,
            source="forge",
            metric="forge.tokens.total",
            value=1234,
            unit="{token}",
            kind="counter",
        )

        event_payload = record_to_dict(event)
        metric_payload = json.loads(record_to_json(metric))
        self.assertEqual(event_payload["record_type"], "event")
        self.assertEqual(event_payload["schema_version"], 1)
        self.assertEqual(
            event_payload["context"]["forge_cell_id"],
            "forge-0001-abc",
        )
        self.assertGreaterEqual(event_payload["time"]["experiment_elapsed_ns"], 0)
        self.assertEqual(metric_payload["record_type"], "metric")
        self.assertEqual(metric_payload["kind"], "counter")
        self.assertEqual(metric_payload["value"], 1234)

    def test_capability_matrix_requires_explicit_unavailable_reason(self):
        capabilities = TelemetryCapabilities(
            context=self._context(),
            profile=CaptureProfile.MAXIMAL,
            capabilities=(
                TelemetryCapability(
                    name="process.cpu",
                    status=AvailabilityStatus.CAPTURED,
                    source="procfs",
                    unit="%",
                ),
                TelemetryCapability(
                    name="gpu.power",
                    status=AvailabilityStatus.UNAVAILABLE,
                    source="nvml",
                    reason="no NVIDIA device",
                    unit="W",
                ),
            ),
        )
        payload = record_to_dict(capabilities)
        self.assertEqual(payload["record_type"], "capabilities")
        self.assertEqual(payload["profile"], "maximal")
        with self.assertRaisesRegex(ValueError, "must include a reason"):
            TelemetryCapability(
                name="gpu.temperature",
                status=AvailabilityStatus.UNAVAILABLE,
                source="nvml",
            )

    def test_context_preserves_arbitrary_experimental_factors(self):
        payload = record_to_dict(
            TelemetryCapabilities(
                context=self._context(),
                profile=CaptureProfile.MAXIMAL,
                capabilities=(),
            )
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["context"]["factors"],
            {"treatment": "baseline", "max_turns": 100},
        )

    def test_artifact_reference_rejects_archive_escape(self):
        with self.assertRaisesRegex(ValueError, "within the archive"):
            ArtifactReference(
                context=self._context(),
                logical_name="secret",
                relative_path="../outside.txt",
                sha256="0" * 64,
                size_bytes=1,
            )

    def test_metric_rejects_non_finite_values(self):
        origin = ClockOrigin.capture()
        with self.assertRaisesRegex(ValueError, "finite"):
            MetricSample(
                context=self._context(),
                time=TimePoint.capture(origin),
                source="test",
                metric="bad",
                value=float("nan"),
                unit="1",
            )


class TelemetryArchiveTests(unittest.TestCase):
    def test_archive_layout_and_raw_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            paths = initialize_archive(
                root,
                experiment_id="experiment-1",
                capture_profile=CaptureProfile.MAXIMAL,
            )
            self.assertTrue(paths.raw_cells.is_dir())
            self.assertTrue(paths.derived.is_dir())
            self.assertTrue(paths.share.is_dir())

            open_manifest = read_manifest(paths.manifest)
            self.assertEqual(open_manifest.state, ArchiveState.OPEN)
            self.assertEqual(open_manifest.capture_profile, CaptureProfile.MAXIMAL)

            cell = ensure_cell_layout(paths, "forge-0001-abc")
            events = cell / "events.jsonl"
            events.write_text(
                '{"record_type":"event","name":"start"}\n'
                '{"record_type":"event","name":"finish"}\n',
                encoding="utf-8",
            )
            (cell / "opaque.bin").write_bytes(b"raw-evidence")

            manifest = seal_archive(
                paths,
                state=ArchiveState.COMPLETE,
                notes=("synthetic test archive",),
            )
            self.assertEqual(manifest.state, ArchiveState.COMPLETE)
            self.assertEqual(len(manifest.raw_files), 2)
            counts = {
                item.relative_path: item.record_count for item in manifest.raw_files
            }
            self.assertEqual(
                counts["raw/cells/forge-0001-abc/events.jsonl"],
                2,
            )
            self.assertEqual(verify_archive(paths), [])

    def test_raw_mutation_after_seal_is_detected_without_derived_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = initialize_archive(
                Path(tmp) / "experiment",
                experiment_id="experiment-1",
            )
            cell = ensure_cell_layout(paths, "forge-0001-abc")
            raw = cell / "events.jsonl"
            raw.write_text('{"event":"one"}\n', encoding="utf-8")
            seal_archive(paths, state=ArchiveState.COMPLETE)
            self.assertEqual(verify_archive(paths), [])

            # Derived/share outputs are rebuildable and intentionally excluded
            # from the immutable raw-evidence integrity root.
            (paths.derived / "summary.csv").write_text("a,b\n1,2\n")
            (paths.share / "report.txt").write_text("shareable")
            self.assertEqual(verify_archive(paths), [])

            raw.write_text('{"event":"changed"}\n', encoding="utf-8")
            errors = verify_archive(paths)
            self.assertTrue(any("sha256 mismatch" in error for error in errors))

    def test_sealed_archive_is_write_closed_through_archive_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = initialize_archive(
                Path(tmp) / "experiment",
                experiment_id="experiment-1",
            )
            ensure_cell_layout(paths, "forge-0001-abc")
            seal_archive(paths, state=ArchiveState.COMPLETE)
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                ensure_cell_layout(paths, "forge-0002-def")
            with self.assertRaisesRegex(RuntimeError, "already sealed"):
                seal_archive(paths, state=ArchiveState.COMPLETE)

    def test_existing_archive_rejects_capture_profile_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            initialize_archive(
                root,
                experiment_id="experiment-1",
                capture_profile=CaptureProfile.MAXIMAL,
            )
            with self.assertRaisesRegex(ValueError, "capture profile"):
                initialize_archive(
                    root,
                    experiment_id="experiment-1",
                    capture_profile=CaptureProfile.MINIMAL,
                )

    def test_partial_archive_is_a_valid_sealed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = initialize_archive(
                Path(tmp) / "experiment",
                experiment_id="experiment-1",
            )
            ensure_cell_layout(paths, "forge-0001-abc")
            manifest = seal_archive(
                paths,
                state=ArchiveState.PARTIAL,
                notes=("trial interrupted",),
            )
            self.assertEqual(manifest.state, ArchiveState.PARTIAL)
            self.assertEqual(verify_archive(paths), [])

    def test_cell_ids_cannot_escape_raw_cells_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = initialize_archive(
                Path(tmp) / "experiment",
                experiment_id="experiment-1",
            )
            with self.assertRaisesRegex(ValueError, "forge_cell_id"):
                ensure_cell_layout(paths, "../outside")


if __name__ == "__main__":
    unittest.main()
