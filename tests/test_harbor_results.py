from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from forge_bench.harbor_results import normalize_harbor_trial_result


def _intent() -> dict:
    return {
        "forge_cell_id": "forge-0001-test",
        "forge": {
            "arm": "ponytail",
            "instance_id": "django__django-13516",
            "repeat": 1,
            "run_index": 1,
            "model": "deepseek/deepseek-v4-flash-0731",
            "api_provider": "openrouter",
            "upstream_provider": "relace",
            "reasoning": "none",
            "max_turns": 50,
            "budget_warning_ratio": 0.75,
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
    exception=None,
):
    base_metadata = {
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
    }
    if metadata:
        base_metadata.update(metadata)
    verifier_result = (
        None
        if reward is None
        else SimpleNamespace(rewards={"reward": reward})
    )
    return SimpleNamespace(
        agent_result=SimpleNamespace(
            n_input_tokens=1000,
            n_cache_tokens=700,
            n_output_tokens=120,
            cost_usd=0.012,
            metadata=base_metadata,
        ),
        verifier_result=verifier_result,
        exception_info=exception,
        agent_info=SimpleNamespace(model_info=SimpleNamespace(name=model)),
        agent_execution=_timing(15.0),
        verifier=_timing(3.5),
        config=SimpleNamespace(task=SimpleNamespace(source="swe-bench/swe-bench-verified")),
        trial_uri="file:///tmp/trial",
    )


class HarborResultNormalizationTests(unittest.TestCase):
    def test_normalizes_harbor_metrics_without_double_counting_cache(self):
        result = normalize_harbor_trial_result(_trial_result(), _intent())
        self.assertTrue(result.valid)
        self.assertTrue(result.resolved)
        self.assertEqual(result.input_tokens, 300)
        self.assertEqual(result.cache_read_tokens, 700)
        self.assertEqual(result.output_tokens, 120)
        self.assertEqual(result.total_tokens, 1120)
        self.assertEqual(result.api_calls, 7)
        self.assertEqual(result.tool_calls, 11)
        self.assertEqual(result.files_changed, 2)
        self.assertEqual(result.diff_lines, 9)
        self.assertEqual(result.wall_seconds, 15.0)
        self.assertEqual(result.evaluation_seconds, 3.5)
        self.assertEqual(result.cost_usd, 0.012)
        self.assertEqual(result.session_id, "session-1")
        self.assertTrue(result.trace_exported)

    def test_patch_capture_failure_invalidates_scientific_observation(self):
        result = normalize_harbor_trial_result(
            _trial_result(metadata={"forge_patch_capture_error": "git failure"}),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertIn("patch capture", result.error)

    def test_model_mismatch_invalidates_scientific_observation(self):
        result = normalize_harbor_trial_result(
            _trial_result(model="deepseek/wrong-model"),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertIn("wrong model", result.error)

    def test_missing_verifier_reward_invalidates_nonempty_patch(self):
        result = normalize_harbor_trial_result(
            _trial_result(reward=None),
            _intent(),
        )
        self.assertFalse(result.valid)
        self.assertFalse(result.evaluation_completed)
        self.assertIn("did not produce a reward", result.error)

    def test_empty_patch_can_remain_valid_without_verifier_reward(self):
        result = normalize_harbor_trial_result(
            _trial_result(
                reward=None,
                metadata={
                    "forge_patch_nonempty": False,
                    "forge_files_changed": 0,
                    "forge_diff_lines": 0,
                },
            ),
            _intent(),
        )
        self.assertTrue(result.valid)
        self.assertFalse(result.resolved)


if __name__ == "__main__":
    unittest.main()
