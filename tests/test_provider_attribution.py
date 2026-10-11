"""Platform-provider event import must not turn requested providers into facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def test_verified_provider_join_retains_actual_billing_and_latency() -> None:
    rows = [
        completed("gen-1"), observed("gen-1", "DigitalOcean"),
        completed("gen-2"), observed("gen-2", "Inference.net"),
        failed(),
        # duplicate evidence is counted only once per generation
        observed("gen-1", "DigitalOcean"),
    ]
    report = analyze_events(rows)
    assert report["counts"] == {
        "completed": 2, "verified": 2, "unverified": 0,
        "failed": 1, "unlinked_observations": 0,
    }
    digital = next(x for x in report["per_provider"] if x["serving_provider"] == "DigitalOcean")
    assert digital["mean_cost_usd"] == pytest.approx(0.0006)
    assert digital["mean_latency_ms"] == 140
    assert digital["reliability_rate"] is None
    assert report["generations"][0]["fallback_used"] is None
    assert report["generations"][1]["secondary_provider_selected"] is True
    assert report["model_route_reliability"][0]["completion_rate"] == pytest.approx(2 / 3)
    assert report["failures"][0]["serving_provider"] is None


def test_unverified_never_inherits_requested_provider_identity() -> None:
    report = analyze_events([
        completed("gen-404"), observed("gen-404", None),
        completed("gen-unknown"),
        observed("gen-orphan", "DigitalOcean"),
    ])
    rows = {row["generation_id"]: row for row in report["generations"]}
    assert rows["gen-404"]["serving_provider"] is None
    assert rows["gen-404"]["verification_status"] == "unverified"
    assert rows["gen-404"]["cost_source"] == "runtime_execution"
    assert rows["gen-unknown"]["serving_provider"] is None
    assert rows["gen-orphan"]["observed_without_execution"] is True
    assert report["counts"]["unlinked_observations"] == 1
    assert report["counts"]["unverified"] == 2


def test_verified_observation_replaces_earlier_unverified_duplicate() -> None:
    result = analyze_events([completed("gen-1"), observed("gen-1", None), observed("gen-1")])
    assert result["generations"][0]["verification_status"] == "verified"


def test_wrong_source_and_fal_events_not_counted_as_openrouter() -> None:
    wrong = {**completed("gen-x"), "source": "forge-worlds"}
    fal = completed("fal-1")
    fal["payload"]["provider"] = "fal"
    report = analyze_events([wrong, fal, {"source": "unknown", "event_type": "ai.execution.failed"}])
    assert report["counts"]["completed"] == 0


def test_loader_and_cli_export_do_not_copy_prompt_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = tmp_path / "events.jsonl"
    evidence = [completed("gen-1"), observed("gen-1"), failed()]
    with events.open("w", encoding="utf-8") as file:
        for value in evidence:
            file.write(json.dumps({"data": value}) + "\n")
    assert len(load_events(events)) == 3

    out = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["forge-bench-provider-report", str(events), "--output-dir", str(out)]
    )
    main()
    payload = json.loads((out / "provider-report.json").read_text(encoding="utf-8"))
    assert payload["counts"]["verified"] == 1
    csv_text = (out / "provider-generations.csv").read_text(encoding="utf-8")
    assert "DigitalOcean" in csv_text
    assert "fallback_used" in csv_text
    assert (out / "provider-summary.csv").exists()


def test_loader_rejects_malformed_event_row(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"payload":{"model":"x"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected an af_events"):
        load_events(path)
    path.write_text('{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_events(path)
