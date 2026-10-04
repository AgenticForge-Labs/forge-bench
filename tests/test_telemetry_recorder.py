from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from forge_bench.harbor_runtime import run_materialized_trials
from forge_bench.telemetry_archive import read_manifest, verify_archive
from forge_bench.telemetry_contracts import ArchiveState, CaptureProfile, TelemetryContext, TelemetryEvent, TimePoint, ClockOrigin
from forge_bench.telemetry_recorder import (
    AppendOnlyJsonlJournal,
    ExperimentTelemetryRecorder,
    recover_interrupted_archive,
)


def _event(name: str) -> TelemetryEvent:
    origin = ClockOrigin.capture()
    return TelemetryEvent(
        context=TelemetryContext(experiment_id="experiment-1"),
        time=TimePoint.capture(origin),
        source="test",
        name=name,
    )


class AppendOnlyJournalTests(unittest.TestCase):
    def test_sequence_survives_clean_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with AppendOnlyJsonlJournal(path) as journal:
                self.assertEqual(journal.append(_event("one")), 1)
                self.assertEqual(journal.append(_event("two")), 2)

            with AppendOnlyJsonlJournal(path) as journal:
                self.assertEqual(journal.next_sequence, 3)
                self.assertEqual(journal.append(_event("three")), 3)

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [1, 2, 3])
            self.assertEqual(
                [row["record"]["name"] for row in rows],
                ["one", "two", "three"],
            )

    def test_torn_tail_is_preserved_then_removed_from_live_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with AppendOnlyJsonlJournal(path) as journal:
                journal.append(_event("one"))

            with path.open("ab") as handle:
                handle.write(b'{"incomplete":')
                handle.flush()

            with AppendOnlyJsonlJournal(path) as journal:
                fragment = journal.recovery_fragment
                self.assertIsNotNone(fragment)
                self.assertEqual(fragment.size_bytes, len(b'{"incomplete":'))
                self.assertEqual(journal.next_sequence, 2)
                journal.append(_event("two"))

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [1, 2])
            recovery_files = list((path.parent / "recovery").glob("*.bin"))
            self.assertEqual(len(recovery_files), 1)
            self.assertEqual(recovery_files[0].read_bytes(), b'{"incomplete":')

    def test_invalid_complete_line_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"not":"a journal envelope"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "journal schema"):
                AppendOnlyJsonlJournal(path)


class ExperimentRecorderTests(unittest.TestCase):
    def test_successful_context_seals_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
                capture_profile=CaptureProfile.MAXIMAL,
            )
            with recorder:
                cell = recorder.cell(
                    {
                        "forge_cell_id": "forge-0001-abc",
                        "forge_run_index": 1,
                        "forge": {
                            "repeat": 1,
                            "agent": "hermes",
                            "model": "provider/model",
                            "instance_id": "task-a",
                            "arm": "baseline",
                            "max_turns": 100,
                        },
                    },
                    environment="docker",
                )
                self.assertIsNotNone(cell)
                cell.event("cell.synthetic", phase="test")

            manifest = read_manifest(root / "archive_manifest.json")
            self.assertEqual(manifest.state, ArchiveState.COMPLETE)
            self.assertEqual(verify_archive(recorder.paths), [])
            self.assertTrue(
                (root / "raw" / "experiment-events.jsonl").is_file()
            )
            self.assertTrue(
                (root / "raw" / "cells" / "forge-0001-abc" / "events.jsonl").is_file()
            )

    def test_exception_seals_partial_archive_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with recorder:
                    raise RuntimeError("boom")

            manifest = read_manifest(root / "archive_manifest.json")
            self.assertEqual(manifest.state, ArchiveState.PARTIAL)
            self.assertTrue(
                any("RuntimeError: boom" in note for note in manifest.notes)
            )
            self.assertEqual(verify_archive(recorder.paths), [])

    def test_recorder_issue_degrades_archive_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                with patch.object(
                    recorder._experiment_journal,
                    "append",
                    side_effect=OSError("disk trouble"),
                ):
                    self.assertFalse(recorder.event("will.fail"))

            self.assertTrue(recorder.degraded)
            manifest = read_manifest(root / "archive_manifest.json")
            self.assertEqual(manifest.state, ArchiveState.PARTIAL)
            self.assertTrue(any("disk trouble" in note for note in manifest.notes))

    def test_existing_open_archive_is_not_resumed_with_reset_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            recorder.event("experiment.synthetic")
            recorder._experiment_journal.close()
            with self.assertRaisesRegex(RuntimeError, "not resumed automatically"):
                ExperimentTelemetryRecorder(
                    root,
                    experiment_id="experiment-1",
                )

            state = recover_interrupted_archive(
                root,
                reason="synthetic hard crash",
            )
            self.assertEqual(state, ArchiveState.PARTIAL)
            manifest = read_manifest(root / "archive_manifest.json")
            self.assertTrue(
                any("synthetic hard crash" in note for note in manifest.notes)
            )
            self.assertEqual(verify_archive(recorder.paths), [])

    def test_sealed_archive_cannot_be_reopened_for_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            with ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            ):
                pass
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                ExperimentTelemetryRecorder(
                    root,
                    experiment_id="experiment-1",
                )


class HarborLifecycleRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_harbor_cells_emit_ordered_lifecycle_journals(self):
        class FakeTrial:
            def __init__(self, config):
                self.config = config
                self.id = f"trial-{config.index}"

            @classmethod
            async def create(cls, config):
                return cls(config)

            async def run(self):
                return SimpleNamespace(
                    id=self.id,
                    exception_info=None,
                    index=self.config.index,
                )

        materialized = [
            (
                {
                    "forge_cell_id": f"forge-{index:04d}-test",
                    "forge_run_index": index,
                    "forge": {
                        "repeat": 1,
                        "agent": "pi",
                        "model": "provider/model",
                        "instance_id": f"task-{index}",
                        "arm": "baseline",
                    },
                },
                SimpleNamespace(
                    index=index,
                    environment=SimpleNamespace(
                        type=SimpleNamespace(value="docker")
                    ),
                ),
            )
            for index in (1, 2)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with recorder:
                with patch(
                    "forge_bench.harbor_runtime.normalize_harbor_trial_result",
                    side_effect=lambda result, intent: SimpleNamespace(
                        valid=True,
                        resolved=result.index == 1,
                    ),
                ):
                    harbor, forge = await run_materialized_trials(
                        materialized,
                        n_concurrent=2,
                        trial_class=FakeTrial,
                        telemetry=recorder,
                    )

            self.assertEqual([result.index for result in harbor], [1, 2])
            self.assertEqual([result.resolved for result in forge], [True, False])
            self.assertEqual(recorder.sealed_state, ArchiveState.COMPLETE)

            expected = [
                "cell.queued",
                "cell.started",
                "harbor.trial.create.started",
                "harbor.trial.create.completed",
                "harbor.trial.run.started",
                "harbor.trial.run.completed",
                "forge.result.normalize.started",
                "forge.result.normalize.completed",
                "cell.completed",
            ]
            for index in (1, 2):
                path = (
                    root
                    / "raw"
                    / "cells"
                    / f"forge-{index:04d}-test"
                    / "events.jsonl"
                )
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(
                    [row["record"]["name"] for row in rows],
                    expected,
                )
                self.assertEqual(
                    [row["sequence"] for row in rows],
                    list(range(1, len(expected) + 1)),
                )

    async def test_sibling_trial_finishes_before_failure_is_propagated(self):
        completed: list[int] = []

        class MixedTrial:
            def __init__(self, config):
                self.config = config
                self.id = f"trial-{config.index}"

            @classmethod
            async def create(cls, config):
                return cls(config)

            async def run(self):
                if self.config.index == 1:
                    await asyncio.sleep(0.01)
                    raise RuntimeError("first failed")
                await asyncio.sleep(0.03)
                completed.append(self.config.index)
                return SimpleNamespace(
                    id=self.id,
                    exception_info=None,
                    index=self.config.index,
                )

        materialized = [
            (
                {
                    "forge_cell_id": f"forge-{index:04d}-mixed",
                    "forge_run_index": index,
                    "forge": {
                        "repeat": 1,
                        "agent": "pi",
                        "model": "provider/model",
                        "instance_id": f"task-{index}",
                        "arm": "baseline",
                    },
                },
                SimpleNamespace(
                    index=index,
                    environment=SimpleNamespace(
                        type=SimpleNamespace(value="docker")
                    ),
                ),
            )
            for index in (1, 2)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExperimentTelemetryRecorder(
                Path(tmp) / "telemetry",
                experiment_id="experiment-1",
            )
            with patch(
                "forge_bench.harbor_runtime.normalize_harbor_trial_result",
                side_effect=lambda result, intent: SimpleNamespace(
                    valid=True,
                    resolved=True,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "first failed"):
                    with recorder:
                        await run_materialized_trials(
                            materialized,
                            n_concurrent=2,
                            trial_class=MixedTrial,
                            telemetry=recorder,
                        )

        self.assertEqual(completed, [2])

    async def test_trial_exception_is_recorded_before_propagation(self):
        class BrokenTrial:
            id = "trial-broken"

            @classmethod
            async def create(cls, _config):
                return cls()

            async def run(self):
                raise RuntimeError("agent exploded")

        intent = {
            "forge_cell_id": "forge-0001-broken",
            "forge_run_index": 1,
            "forge": {
                "repeat": 1,
                "agent": "pi",
                "model": "provider/model",
                "instance_id": "task-a",
                "arm": "baseline",
            },
        }
        config = SimpleNamespace(
            environment=SimpleNamespace(type=SimpleNamespace(value="docker"))
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "telemetry"
            recorder = ExperimentTelemetryRecorder(
                root,
                experiment_id="experiment-1",
            )
            with self.assertRaisesRegex(RuntimeError, "agent exploded"):
                with recorder:
                    await run_materialized_trials(
                        [(intent, config)],
                        trial_class=BrokenTrial,
                        telemetry=recorder,
                    )

            manifest = read_manifest(root / "archive_manifest.json")
            self.assertEqual(manifest.state, ArchiveState.PARTIAL)
            rows = [
                json.loads(line)
                for line in (
                    root
                    / "raw"
                    / "cells"
                    / "forge-0001-broken"
                    / "events.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["record"]["name"], "cell.failed")
            self.assertEqual(
                rows[-1]["record"]["attributes"]["exception_type"],
                "RuntimeError",
            )


if __name__ == "__main__":
    unittest.main()
