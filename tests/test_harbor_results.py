from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

from forge_bench.harbor_results import normalize_harbor_trial_result


def _intent(*, compatibility: bool = False) -> dict:
    agent = {
        "mode": "forge-hermes-compat" if compatibility else "harbor-native",
        "model_name": "deepseek/deepseek-v4-flash-0731",
        "control_semantics": {
            "treatment": "forge-controlled" if compatibility else "baseline",
            "max_turns": "forge-controlled" if compatibility else "agent-default",
            "budget_warning_ratio": (
                "forge-controlled" if compatibility else "agent-default"
            ),
        },
    }
    if compatibility:
        agent["import_path"] = "forge_bench.harbor_hermes:ForgeBenchHermes"
    else:
        agent["name"] = "pi"
    return {
        "forge_cell_id": "forge-0001-test",
        "agent": agent,
        "forge": {
            "agent": "hermes" if compatibility else "pi",
            "arm": "ponytail" if compatibility else "baseline",
            "instance_id": "django__django-13516",
            "repeat": 1,
            "run_index": 1,
            "model": "deepseek/deepseek-v4-flash-0731",
            "api_provider": "openrouter",
            "upstream_provider": "relace",
            "reasoning": "none",
            "max_turns": 50 if compatibility else 100,
            "budget_warning_ratio": 0.75 if compatibility else None,
        },
    }


def _timing(seconds: float):
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SimpleNamespace(started_at=started, finished_at=started + timedelta(seconds=seconds))


def _trial_result(
    *,
    reward=1.0,
    metadata=None,
    model="deepseek/deepseek-v4-flash-0731",
    agent="pi",
    agent_version="1.2.3",
    exception=None,
):
    base_metadata = {}
    if metadata:
        base_metadata.update(metadata)
    verifier_result = (
        None if reward is None else SimpleNamespace(rewards={"reward": reward})
    )
    return SimpleNamespace(
        id=UUID("00000000-0000-0000-0000-000000000123"),
        agent_result=SimpleNamespace(
            n_input_tokens=1000,
            n_cache_tokens=700,
            n_output_tokens=120,
            cost_usd=0.012,
            metadata=base_metadata,
        ),
        verifier_result=verifier_result,
        exception_info=exception,
        agent_info=SimpleNamespace(
            name=agent,
            version=agent_version,
            model_info=SimpleNamespace(name=model, provider="openrouter"),
        ),
        agent_execution=_timing(15.0),
        verifier=_timing(3.5),
        config=SimpleNamespace(
            task=SimpleNamespace(source="swe-bench/swe-bench-verified"),
            environment=SimpleNamespace(type=SimpleNamespace(value="docker")),
        ),
        trial_uri="file:///tmp/trial",
    )


class HarborResultNormalizationTests(unittest.TestCase):
    def test_native_agent_uses_harbor_metrics_and_records_provenance(self):
        result = normalize_harbor_trial_result(_trial_result(), _intent())
        self.assertTrue(result.valid)
        self.assertTrue(result.resolved)
        self.assertEqual(result.agent, "pi")
        self.assertEqual(result.agent_version, "1.2.3")
        self.assertEqual(result.environment, "docker")
        self.assertEqual(
            result.harbor_trial_id,
            "00000000-0000-0000-0000-000000000123",
        )
        self.assertEqual(result.input_tokens, 300)
        self.assertEqual(result.cache_read_tokens, 700)
        self.assertEqual(result.output_tokens, 120)
        self.assertEqual(result.total_tokens, 1120)
        self.assertEqual(result.cost_usd, 0.012)
        self.assertEqual(result.cost_source, "harbor_agent_context")
        self.assertIsNone(result.max_turns)
        self.assertIsNone(result.budget_warning_ratio)

    def test_native_agent_requires_verifier_reward(self):
        result = normalize_harbor_trial_result(
            _trial_result(reward=None),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertFalse(result.evaluation_completed)
        self.assertIn("did not produce a reward", result.error)

    def test_native_agent_mismatch_invalidates_observation(self):
        result = normalize_harbor_trial_result(
            _trial_result(agent="opencode"),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertIn("wrong agent", result.error)

    def test_model_mismatch_invalidates_scientific_observation(self):
        result = normalize_harbor_trial_result(
            _trial_result(model="deepseek/wrong-model"),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertIn("wrong model", result.error)

    def test_exact_hermes_metadata_wins_on_compatibility_path(self):
        metadata = {
            "forge_api_provider": "openrouter",
            "forge_upstream_provider": "relace",
            "forge_reasoning": "none",
            "forge_patch_nonempty": True,
            "forge_files_changed": 2,
            "forge_diff_lines": 9,
            "forge_api_calls": 7,
            "forge_tool_calls": 11,
            "forge_session_id": "session-1",
            "forge_trace_exported": True,
            "forge_uncached_input_tokens": 111,
            "forge_cache_read_tokens": 222,
            "forge_cache_write_tokens": 33,
            "forge_output_tokens": 44,
            "forge_reasoning_tokens": 5,
            "forge_total_tokens": 410,
            "forge_estimated_cost_usd": 0.02,
            "forge_actual_cost_usd": 0.015,
            "forge_cost_source": "openrouter",
        }
        result = normalize_harbor_trial_result(
            _trial_result(
                metadata=metadata,
                agent="forge-bench-hermes",
                agent_version="v2026.9.14",
            ),
            _intent(compatibility=True),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.agent, "hermes")
        self.assertEqual(result.input_tokens, 111)
        self.assertEqual(result.cache_read_tokens, 222)
        self.assertEqual(result.cache_write_tokens, 33)
        self.assertEqual(result.output_tokens, 44)
        self.assertEqual(result.reasoning_tokens, 5)
        self.assertEqual(result.total_tokens, 410)
        self.assertEqual(result.estimated_cost_usd, 0.02)
        self.assertEqual(result.actual_cost_usd, 0.015)
        self.assertEqual(result.cost_usd, 0.015)
        self.assertEqual(result.cost_source, "provider_actual")
        self.assertEqual(result.max_turns, 50)
        self.assertEqual(result.budget_warning_ratio, 0.75)
        self.assertEqual(result.files_changed, 2)
        self.assertEqual(result.diff_lines, 9)
        self.assertEqual(result.api_calls, 7)
        self.assertEqual(result.tool_calls, 11)
        self.assertEqual(result.session_id, "session-1")
        self.assertTrue(result.trace_exported)

    def test_patch_capture_failure_invalidates_compatibility_observation(self):
        metadata = {
            "forge_api_provider": "openrouter",
            "forge_upstream_provider": "relace",
            "forge_reasoning": "none",
            "forge_patch_nonempty": True,
            "forge_patch_capture_error": "git failure",
        }
        result = normalize_harbor_trial_result(
            _trial_result(
                metadata=metadata,
                agent="forge-bench-hermes",
            ),
            _intent(compatibility=True),
        )
        self.assertFalse(result.valid)
        self.assertIn("patch capture", result.error)

    def test_empty_patch_compatibility_cell_can_remain_valid_without_reward(self):
        metadata = {
            "forge_api_provider": "openrouter",
            "forge_upstream_provider": "relace",
            "forge_reasoning": "none",
            "forge_patch_nonempty": False,
            "forge_files_changed": 0,
            "forge_diff_lines": 0,
        }
        result = normalize_harbor_trial_result(
            _trial_result(
                reward=None,
                metadata=metadata,
                agent="forge-bench-hermes",
            ),
            _intent(compatibility=True),
        )
        self.assertTrue(result.valid)
        self.assertFalse(result.resolved)


if __name__ == "__main__":
    unittest.main()
