from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forge_bench.designs import AgentSpec, ModelSpec, load_design
from forge_bench.harbor_plan import compile_harbor_plan, write_harbor_plan
from forge_bench.planning import build_run_plan


class HarborPlanCompilerTests(unittest.TestCase):
    def _models(self):
        return [
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

    def _generic_plan(self):
        agents = [
            AgentSpec(key="hermes", label="Hermes", agent="hermes", version="v2026.9.14"),
            AgentSpec(key="pi", label="Pi", agent="pi"),
        ]
        return build_run_plan(
            ["baseline"],
            ["task-a", "task-b"],
            2,
            260920,
            models=self._models(),
            agents=agents,
            max_turns_levels=(100,),
            budget_warning_ratio_levels=(None,),
        )[0]

    def _hermes_control_plan(self):
        return build_run_plan(
            ["baseline", "ponytail"],
            ["task-a"],
            1,
            260920,
            models=self._models()[:1],
            agents=[AgentSpec(key="hermes", label="Hermes", agent="hermes")],
            max_turns_levels=(50, 100),
            budget_warning_ratio_levels=(None, 0.75),
        )[0]

    def test_projection_crosses_agent_and_model_as_independent_factors(self):
        run_plan = self._generic_plan()
        projection = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )

        self.assertEqual(projection["schema_version"], 2)
        self.assertEqual(projection["trial_count"], 16)
        self.assertTrue(projection["executable"])
        self.assertTrue(projection["all_cells_supported"])
        self.assertEqual(projection["harbor_api_baseline"], "0.23.0")
        self.assertEqual(
            {(row["forge"]["agent"], row["forge"]["model"]) for row in projection["trials"]},
            {
                ("hermes", "provider/model-one"),
                ("hermes", "provider/model-two"),
                ("pi", "provider/model-one"),
                ("pi", "provider/model-two"),
            },
        )
        for source, trial in zip(run_plan, projection["trials"], strict=True):
            self.assertEqual(trial["forge_run_index"], source["run_index"])
            self.assertEqual(trial["forge"], source)
            self.assertEqual(trial["agent"]["mode"], "harbor-native")
            self.assertEqual(trial["agent"]["name"], source["agent"])
            self.assertEqual(trial["agent"]["model_name"], source["model"])
            self.assertEqual(
                trial["agent"]["control_semantics"]["max_turns"],
                "agent-default",
            )
            self.assertEqual(
                trial["dataset"]["task_names"],
                [source["instance_id"]],
            )

    def test_hermes_specific_controls_stay_on_compatibility_path(self):
        run_plan = self._hermes_control_plan()
        projection = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )
        self.assertTrue(projection["executable"])
        for source, trial in zip(run_plan, projection["trials"], strict=True):
            self.assertEqual(trial["agent"]["mode"], "forge-hermes-compat")
            self.assertEqual(
                trial["agent"]["import_path"],
                "forge_bench.harbor_hermes:ForgeBenchHermes",
            )
            self.assertEqual(trial["agent"]["kwargs"]["treatment"], source["arm"])
            self.assertEqual(
                trial["agent"]["kwargs"]["max_turns"],
                source["max_turns"],
            )
            self.assertEqual(
                trial["agent"]["kwargs"]["budget_warning_ratio"],
                source["budget_warning_ratio"],
            )

    def test_non_hermes_treatment_or_budget_controls_fail_closed_at_plan_time(self):
        run_plan = build_run_plan(
            ["baseline", "ponytail"],
            ["task-a"],
            1,
            260920,
            models=self._models()[:1],
            agents=[AgentSpec(key="pi", label="Pi", agent="pi")],
            max_turns_levels=(50, 100),
            budget_warning_ratio_levels=(None,),
        )[0]
        projection = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
        )
        self.assertFalse(projection["executable"])
        self.assertFalse(projection["all_cells_supported"])
        self.assertTrue(projection["unsupported_reasons"])
        self.assertTrue(
            all(not trial["supported"] for trial in projection["trials"])
        )

    def test_projection_is_deterministic_and_has_unique_cell_ids(self):
        run_plan = self._generic_plan()
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
        run_plan = self._generic_plan()
        run_plan[0], run_plan[1] = run_plan[1], run_plan[0]
        with self.assertRaisesRegex(ValueError, "order must match run_index"):
            compile_harbor_plan(
                run_plan,
                forge_dataset="SWE-bench/SWE-bench_Verified",
            )

    def test_unmapped_dataset_is_recorded_without_breaking_forge_plan(self):
        projection = compile_harbor_plan(
            self._generic_plan(),
            forge_dataset="SWE-bench/SWE-bench_Lite",
        )
        self.assertFalse(projection["dataset_mapped"])
        self.assertFalse(projection["executable"])
        self.assertIsNone(projection["harbor_dataset"])
        self.assertTrue(
            all(trial["dataset"]["name"] is None for trial in projection["trials"])
        )

    def test_design_loader_allows_non_openrouter_provider_for_harbor_planning(self):
        text = """version: 1
name: provider-neutral
description: Harbor planning should not require the legacy Hermes provider.
selection:
  dataset: verified
  split: test
  difficulty: medium
  instance_ids: [django__django-13516]
provider:
  api: openai
  upstream: openai
  require_same_upstream: true
agents:
  - codex
models:
  - key: gpt
    label: GPT
    model: openai/gpt-test
treatments: [baseline]
randomization:
  mode: full-factorial-within-block
  blocks: 1
  seed: 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "design.yaml"
            path.write_text(text, encoding="utf-8")
            design = load_design(path)
        self.assertEqual(design.agents[0].agent, "codex")
        self.assertEqual(design.models[0].api_provider, "openai")
        self.assertEqual(design.models[0].model, "openai/gpt-test")

    def test_write_harbor_plan_round_trips_json(self):
        run_plan = self._generic_plan()
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
