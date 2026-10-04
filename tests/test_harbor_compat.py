from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

HARBOR_AVAILABLE = importlib.util.find_spec("harbor") is not None


@unittest.skipUnless(HARBOR_AVAILABLE, "Harbor is an optional migration dependency")
class HarborCompatibilityTests(unittest.TestCase):
    def test_native_harbor_agent_materializes_by_name_and_model(self):
        from harbor.models.trial.config import TaskConfig

        from forge_bench.harbor_runtime import build_harbor_trial_config

        intent = {
            "forge_cell_id": "forge-0001-test",
            "supported": True,
            "agent": {
                "mode": "harbor-native",
                "name": "pi",
                "model_name": "provider/model-one",
                "kwargs": {"version": "0.0-test"},
                "supported": True,
                "control_semantics": {
                    "treatment": "baseline",
                    "max_turns": "agent-default",
                    "budget_warning_ratio": "agent-default",
                },
            },
            "forge": {
                "agent": "pi",
                "arm": "baseline",
                "instance_id": "django__django-13516",
                "repeat": 1,
                "run_index": 1,
                "model": "provider/model-one",
                "api_provider": "openrouter",
                "upstream_provider": "relace",
                "reasoning": "none",
                "max_turns": 100,
                "budget_warning_ratio": None,
            },
        }
        task = TaskConfig(name="swe-bench/django__django-13516", ref="latest")
        with tempfile.TemporaryDirectory() as tmp:
            config = build_harbor_trial_config(
                intent,
                task,
                trials_dir=Path(tmp),
                environment="docker",
            )

        self.assertEqual(config.agent.name, "pi")
        self.assertIsNone(config.agent.import_path)
        self.assertEqual(config.agent.model_name, "provider/model-one")
        self.assertEqual(config.agent.kwargs["version"], "0.0-test")
        self.assertFalse(
            any("caveman skill" in text for text in config.extra_instructions)
        )
        self.assertEqual(str(config.environment.type.value), "docker")

    def test_forge_hermes_compatibility_path_preserves_exact_controls(self):
        from harbor.models.trial.config import TaskConfig

        from forge_bench.harbor_runtime import build_harbor_trial_config

        intent = {
            "forge_cell_id": "forge-0001-test",
            "supported": True,
            "agent": {
                "mode": "forge-hermes-compat",
                "import_path": "forge_bench.harbor_hermes:ForgeBenchHermes",
                "model_name": "deepseek/deepseek-v4-flash-0731",
                "kwargs": {
                    "version": "v2026.9.14",
                    "treatment": "caveman",
                    "api_provider": "openrouter",
                    "upstream_provider": "relace",
                    "reasoning": "none",
                    "max_turns": 50,
                    "budget_warning_ratio": 0.75,
                    "toolsets": "hermes-cli",
                },
                "supported": True,
                "control_semantics": {
                    "treatment": "forge-controlled",
                    "max_turns": "forge-controlled",
                    "budget_warning_ratio": "forge-controlled",
                },
            },
            "forge": {
                "agent": "hermes",
                "arm": "caveman",
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
        task = TaskConfig(name="swe-bench/django__django-13516", ref="latest")
        with tempfile.TemporaryDirectory() as tmp:
            config = build_harbor_trial_config(
                intent,
                task,
                trials_dir=Path(tmp),
                environment="docker",
            )

        self.assertEqual(
            config.agent.import_path,
            "forge_bench.harbor_hermes:ForgeBenchHermes",
        )
        self.assertIsNone(config.agent.name)
        self.assertEqual(config.agent.kwargs["max_turns"], 50)
        self.assertEqual(config.agent.kwargs["budget_warning_ratio"], 0.75)
        self.assertTrue(
            any("caveman skill" in text for text in config.extra_instructions)
        )

    def test_forge_hermes_config_preserves_experimental_factors(self):
        from forge_bench.harbor_hermes import ForgeBenchHermes

        with tempfile.TemporaryDirectory() as tmp:
            agent = ForgeBenchHermes(
                logs_dir=Path(tmp),
                model_name="deepseek/deepseek-v4-flash-0731",
                treatment="ponytail",
                api_provider="openrouter",
                upstream_provider="relace",
                reasoning="none",
                budget_warning_ratio=0.75,
                max_turns=50,
                toolsets="hermes-cli",
                version="v2026.9.14",
            )
            config = yaml.safe_load(
                agent._build_config_yaml(
                    "deepseek/deepseek-v4-flash-0731",
                    agent.options.max_turns,
                )
            )

        self.assertEqual(config["agent"]["max_turns"], 50)
        self.assertEqual(config["agent"]["reasoning_effort"], "none")
        self.assertEqual(config["agent"]["budget_warning_ratio"], 0.75)
        self.assertEqual(config["provider_routing"]["only"], ["relace"])
        self.assertEqual(
            config["provider_routing"]["models"]["deepseek/deepseek-v4-flash-0731"]["only"],
            ["relace"],
        )
        self.assertFalse(config["compression"]["enabled"])
        self.assertEqual(config["plugins"]["enabled"], ["ponytail"])
        self.assertFalse(config["memory"]["memory_enabled"])

    def test_unknown_api_provider_is_rejected_by_forge_hermes_only(self):
        from pydantic import ValidationError

        from forge_bench.harbor_hermes import ForgeBenchHermesOptions

        with self.assertRaises(ValidationError):
            ForgeBenchHermesOptions(api_provider="anthropic")


if __name__ == "__main__":
    unittest.main()
