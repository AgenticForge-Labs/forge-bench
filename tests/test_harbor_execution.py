from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from forge_bench.designs import AgentSpec, ModelSpec
from forge_bench.harbor_execution import HarborExecutionConfig, execution_from_plan
from forge_bench.harbor_plan import compile_harbor_plan
from forge_bench.harbor_runtime import run_materialized_trials
from forge_bench.planning import build_run_plan


class HarborExecutionConfigTests(unittest.TestCase):
    def test_docker_and_modal_validate_with_resource_overrides(self):
        for environment in ("docker", "modal"):
            config = HarborExecutionConfig(
                environment=environment,
                n_concurrent=4,
                cpus=2,
                memory_mb=4096,
                storage_mb=8192,
                gpus=1,
            ).validated()
            self.assertEqual(config.environment, environment)
            self.assertEqual(config.n_concurrent, 4)
            self.assertEqual(
                config.environment_kwargs(),
                {
                    "override_cpus": 2,
                    "override_memory_mb": 4096,
                    "override_storage_mb": 8192,
                    "override_gpus": 1,
                },
            )

    def test_invalid_environment_or_resource_fails_before_execution(self):
        with self.assertRaisesRegex(ValueError, "supports environment"):
            HarborExecutionConfig(environment="openshell").validated()
        with self.assertRaisesRegex(ValueError, "n_concurrent"):
            HarborExecutionConfig(n_concurrent=0).validated()
        with self.assertRaisesRegex(ValueError, "gpus"):
            HarborExecutionConfig(gpus=0).validated()

    def test_environment_override_does_not_change_scientific_cell_identity(self):
        run_plan = build_run_plan(
            ["baseline"],
            ["task-a"],
            1,
            260920,
            models=[
                ModelSpec(
                    key="m1",
                    label="Model One",
                    model="provider/model-one",
                    upstream_provider="relace",
                )
            ],
            agents=[AgentSpec(key="pi", label="Pi", agent="pi")],
            max_turns_levels=(100,),
            budget_warning_ratio_levels=(None,),
        )[0]
        docker = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
            environment="docker",
        )
        modal = compile_harbor_plan(
            run_plan,
            forge_dataset="SWE-bench/SWE-bench_Verified",
            environment="modal",
        )
        self.assertEqual(
            [row["forge_cell_id"] for row in docker["trials"]],
            [row["forge_cell_id"] for row in modal["trials"]],
        )
        self.assertEqual(docker["run_plan_sha256"], modal["run_plan_sha256"])
        self.assertNotEqual(docker["environment"], modal["environment"])

    def test_execution_from_plan_can_override_environment_without_mutating_plan(self):
        plan = {"environment": "docker", "run_plan_sha256": "abc"}
        config = execution_from_plan(
            plan,
            environment="modal",
            n_concurrent=8,
            cpus=4,
        )
        self.assertEqual(config.environment, "modal")
        self.assertEqual(config.n_concurrent, 8)
        self.assertEqual(config.cpus, 4)
        self.assertEqual(plan["environment"], "docker")


class HarborConcurrentRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_concurrency_preserves_result_order(self):
        class FakeTrial:
            active = 0
            max_active = 0

            def __init__(self, config):
                self.config = config

            @classmethod
            async def create(cls, config):
                return cls(config)

            async def run(self):
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
                try:
                    await asyncio.sleep(0.01 * (4 - self.config.index))
                    return self.config.index
                finally:
                    type(self).active -= 1

        materialized = [
            (
                {"forge_run_index": index},
                SimpleNamespace(index=index),
            )
            for index in (1, 2, 3)
        ]
        with patch(
            "forge_bench.harbor_runtime.normalize_harbor_trial_result",
            side_effect=lambda result, intent: intent["forge_run_index"],
        ):
            harbor_results, forge_results = await run_materialized_trials(
                materialized,
                n_concurrent=2,
                trial_class=FakeTrial,
            )

        self.assertEqual(harbor_results, [1, 2, 3])
        self.assertEqual(forge_results, [1, 2, 3])
        self.assertEqual(FakeTrial.max_active, 2)


if __name__ == "__main__":
    unittest.main()
