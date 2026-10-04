from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forge_bench.designs import ModelSpec
from forge_bench.harbor_plan import compile_harbor_plan, write_harbor_plan
from forge_bench.planning import build_run_plan


class HarborPlanCompilerTests(unittest.TestCase):
    def _plan(self):
        models = [
            ModelSpec(
                key="m1",
                label="Model One",
                model="provider/model-one",
                upstream_provider="relace",
            ),
            ModelSpec(
                key="m2",
                label="Model Two",
                model="provider/model-two",
                upstream_provider="relace",
            ),
        ]
        return build_run_plan(
            ["baseline", "ponytail"],
            ["task-a", "task-b"],
            2,
            260920,
            models=models,
            max_turns_levels=(50, 100),
            budget_warning_ratio_levels=(None, 0.75),
        )[0]

    def test_projection_preserves_every_randomized_cell_in_order(self):
        run_plan = self._plan()
        projection = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )

        self.assertEqual(projection["trial_count"], len(run_plan))
        self.assertEqual(
            projection["harbor_dataset"],
            "swe-bench/swe-bench-verified",
        )
        for source, trial in zip(run_plan, projection["trials"], strict=True):
            self.assertEqual(trial["forge_run_index"], source["run_index"])
            self.assertEqual(trial["forge"], source)
            self.assertEqual(
                trial["dataset"]["task_names"],
                [source["instance_id"]],
            )
            self.assertEqual(trial["agent"]["model_name"], source["model"])
            self.assertEqual(
                trial["agent"]["kwargs"]["treatment"],
                source["arm"],
            )
            self.assertEqual(
                trial["agent"]["kwargs"]["max_turns"],
                source["max_turns"],
            )
            self.assertEqual(
                trial["agent"]["kwargs"]["budget_warning_ratio"],
                source["budget_warning_ratio"],
            )

    def test_projection_is_deterministic_and_has_unique_cell_ids(self):
        run_plan = self._plan()
        first = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )
        second = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )
        self.assertEqual(first, second)
        ids = [trial["forge_cell_id"] for trial in first["trials"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_projection_rejects_reordered_run_indices(self):
        run_plan = self._plan()
        run_plan[0], run_plan[1] = run_plan[1], run_plan[0]
        with self.assertRaisesRegex(ValueError, "order must match run_index"):
            compile_harbor_plan(
                run_plan,
                forge_dataset="SWE-bench/SWE-bench_Verified",
            )

    def test_unmapped_dataset_is_recorded_without_breaking_forge_plan(self):
        projection = compile_harbor_plan(
            self._plan(),
            forge_dataset="SWE-bench/SWE-bench_Lite",
        )
        self.assertFalse(projection["dataset_mapped"])
        self.assertIsNone(projection["harbor_dataset"])
        self.assertTrue(
            all(trial["dataset"]["name"] is None for trial in projection["trials"])
        )

    def test_write_harbor_plan_round_trips_json(self):
        run_plan = self._plan()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "harbor_plan.json"
            expected = write_harbor_plan(
                path,
                run_plan,
                forge_dataset="SWE-bench/SWE-bench_Verified",
            )
            self.assertEqual(json.loads(path.read_text()), expected)


if __name__ == "__main__":
    unittest.main()
