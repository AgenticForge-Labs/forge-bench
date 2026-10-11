"""Regression tests run by Forge Bench's standard unittest discovery."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forge_bench.provider_attribution import analyze_events, load_events, main


def completed(generation: str, model: str = "deepseek/v4.1") -> dict:
    return {
        "source": "agenticforge-ai-runtime",
        "event_type": "ai.execution.completed",
        "subject_id": generation,
        "trace_id": "benchmark-run-7",
        "payload": {
            "provider": "openrouter", "request_id": generation, "model": model,
            "execution_kind": "text", "cost_usd": 0.001, "latency_seconds": 0.25,
        },
    }


def observed(generation: str, serving: str | None = "DigitalOcean") -> dict:
    return {
        "source": "agenticforge-ai-runtime",
        "event_type": "ai.execution.provider_observed",
        "subject_id": generation,
        "trace_id": "benchmark-run-7",
        "payload": {
            "generation_id": generation,
            "requested_model": "deepseek/v4.1",
            "verification_status": "verified" if serving else "unverified",
            "serving_provider": serving,
            "verification_error": "http_404" if not serving else None,
            "preferred_providers": ["digitalocean", "inference-net"],
            "allowed_providers": ["digitalocean", "inference-net"],
            "preferred_provider_selected": serving == "DigitalOcean" if serving else None,
            "secondary_provider_selected": serving == "Inference.net" if serving else None,
            "fallback_used": None,
            "total_cost_usd": 0.0006 if serving else None,
            "latency_ms": 140 if serving else None,
            "reasoning_tokens": 20 if serving else None,
        },
    }


def failed() -> dict:
    return {
        "source": "agenticforge-ai-runtime",
        "event_type": "ai.execution.failed",
        "trace_id": "benchmark-run-8",
        "payload": {
            "api_provider": "openrouter",
            "model": "deepseek/v4.1",
            "execution_kind": "text",
            "error_type": "OpenRouterError",
            "requested_provider_controls": {"only": ["inference-net"]},
        },
    }


class ProviderAttributionTests(unittest.TestCase):
    def test_verified_join_uses_actual_billing_and_latency(self) -> None:
        report = analyze_events([
            completed("gen-1"), observed("gen-1", "DigitalOcean"),
            completed("gen-2"), observed("gen-2", "Inference.net"),
            failed(), observed("gen-1", "DigitalOcean"),
        ])
        self.assertEqual(report["counts"], {
            "completed": 2, "verified": 2, "unverified": 0,
            "failed": 1, "unlinked_observations": 0,
        })
        digital = next(x for x in report["per_provider"] if x["serving_provider"] == "DigitalOcean")
        self.assertAlmostEqual(digital["mean_cost_usd"], 0.0006)
        self.assertEqual(digital["mean_latency_ms"], 140)
        self.assertIsNone(digital["reliability_rate"])
        self.assertIsNone(report["generations"][0]["fallback_used"])
        self.assertTrue(report["generations"][1]["secondary_provider_selected"])
        self.assertAlmostEqual(report["model_route_reliability"][0]["completion_rate"], 2 / 3)
        self.assertIsNone(report["failures"][0]["serving_provider"])

    def test_unverified_never_inherits_requested_provider(self) -> None:
        report = analyze_events([
            completed("gen-404"), observed("gen-404", None),
            completed("gen-unknown"),
            observed("gen-orphan", "DigitalOcean"),
        ])
        rows = {row["generation_id"]: row for row in report["generations"]}
        self.assertIsNone(rows["gen-404"]["serving_provider"])
        self.assertEqual(rows["gen-404"]["verification_status"], "unverified")
        self.assertEqual(rows["gen-404"]["cost_source"], "runtime_execution")
        self.assertIsNone(rows["gen-unknown"]["serving_provider"])
        self.assertTrue(rows["gen-orphan"]["observed_without_execution"])
        self.assertEqual(report["counts"]["unlinked_observations"], 1)
        self.assertEqual(report["counts"]["unverified"], 2)

    def test_verified_replaces_earlier_unverified(self) -> None:
        result = analyze_events([
            completed("gen-1"), observed("gen-1", None), observed("gen-1"),
        ])
        self.assertEqual(result["generations"][0]["verification_status"], "verified")

    def test_wrong_source_and_fal_excluded(self) -> None:
        wrong = {**completed("gen-x"), "source": "forge-worlds"}
        fal = completed("fal-1")
        fal["payload"]["provider"] = "fal"
        result = analyze_events([
            wrong, fal, {"source": "unknown", "event_type": "ai.execution.failed"},
        ])
        self.assertEqual(result["counts"]["completed"], 0)

    def test_loader_and_cli_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "events.jsonl"
            with source.open("w", encoding="utf-8") as file:
                for row in (completed("gen-1"), observed("gen-1"), failed()):
                    file.write(json.dumps({"data": row}) + "\n")
            self.assertEqual(len(load_events(source)), 3)
            destination = tmp / "report"
            with patch.object(
                sys, "argv", ["forge-bench-provider-report", str(source), "--output-dir", str(destination)]
            ):
                main()
            report = json.loads((destination / "provider-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["verified"], 1)
            csv_text = (destination / "provider-generations.csv").read_text(encoding="utf-8")
            self.assertIn("DigitalOcean", csv_text)
            self.assertIn("fallback_used", csv_text)
            self.assertTrue((destination / "provider-summary.csv").is_file())

    def test_loader_rejects_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"payload":{"model":"x"}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected an af_events"):
                load_events(path)
            path.write_text('{bad}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_events(path)


if __name__ == "__main__":
    unittest.main()
