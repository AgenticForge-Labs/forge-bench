"""Reproducible, read-only analysis of Platform's OpenRouter observation events.

This imports exported `af_events` JSONL. Platform remains the sole producer of
serving-provider truth; Forge Bench does not guess it from configured routing.
No provider credentials or live Directus access are required.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def load_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if isinstance(value, dict) and isinstance(value.get("data"), dict):
                value = value["data"]
            if not isinstance(value, dict) or not isinstance(value.get("event_type"), str):
                raise ValueError(f"{path}:{number}: expected an af_events JSON object")
            result.append(value)
    return result


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _observation_for_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "ai.execution.provider_observed":
            continue
        data = _payload(event)
        generation = _text(data.get("generation_id"))
        if not generation:
            continue
        current = matched.get(generation)
        if current is None or (
            data.get("verification_status") == "verified"
            and _payload(current).get("verification_status") != "verified"
        ):
            matched[generation] = event
    return matched


def _provider_row(
    generation: str, execution: dict[str, Any] | None, observation: dict[str, Any] | None
) -> dict[str, Any]:
    ex = _payload(execution or {})
    obs = _payload(observation or {})
    verified = (
        obs.get("verification_status") == "verified"
        and isinstance(obs.get("serving_provider"), str)
        and bool(obs["serving_provider"].strip())
    )
    cost = _number(obs.get("total_cost_usd")) if verified else None
    cost_source = "openrouter_generation" if cost is not None else "unavailable"
    if cost is None:
        cost = _number(ex.get("cost_usd"))
        if cost is not None:
            cost_source = "runtime_execution"
    latency = _number(obs.get("latency_ms")) if verified else None
    latency_source = "openrouter_generation" if latency is not None else "unavailable"
    if latency is None:
        elapsed = _number(ex.get("latency_seconds"))
        if elapsed is not None:
            latency = elapsed * 1000.0
            latency_source = "runtime_execution"
    return {
        "generation_id": generation,
        "model": obs.get("requested_model") or ex.get("model"),
        "execution_kind": ex.get("execution_kind") or obs.get("execution_kind"),
        "trace_id": (execution or observation or {}).get("trace_id"),
        "verification_status": "verified" if verified else "unverified",
        "verification_error": obs.get("verification_error"),
        "serving_provider": obs["serving_provider"] if verified else None,
        "preferred_providers": obs.get("preferred_providers") or [],
        "allowed_providers": obs.get("allowed_providers") or [],
        "preferred_provider_selected": obs.get("preferred_provider_selected") if verified else None,
        "secondary_provider_selected": obs.get("secondary_provider_selected") if verified else None,
        "fallback_used": obs.get("fallback_used") if verified else None,
        "cost_usd": cost,
        "cost_source": cost_source,
        "latency_ms": latency,
        "latency_source": latency_source,
        "prompt_tokens": obs.get("prompt_tokens") if verified else None,
        "completion_tokens": obs.get("completion_tokens") if verified else None,
        "cached_tokens": obs.get("cached_tokens") if verified else None,
        "reasoning_tokens": obs.get("reasoning_tokens") if verified else None,
        "observed_without_execution": execution is None,
    }


def analyze_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Join by actual generation ID, never by configured model/provider preference."""
    source = [
        event for event in events
        if event.get("source") == "agenticforge-ai-runtime"
    ]
    observations = _observation_for_id(source)
    completions: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for event in source:
        kind = event.get("event_type")
        body = _payload(event)
        if kind == "ai.execution.completed" and body.get("provider") == "openrouter":
            generation = _text(body.get("request_id"))
            if generation:
                completions.setdefault(generation, event)
        elif kind == "ai.execution.failed" and body.get("api_provider") == "openrouter":
            failures.append({
                "model": body.get("model"),
                "execution_kind": body.get("execution_kind"),
                "error_type": body.get("error_type"),
                "requested_provider_controls": body.get("requested_provider_controls") or {},
                "trace_id": event.get("trace_id"),
                "serving_provider": None,
            })

    rows = [
        _provider_row(key, completions.get(key), observations.get(key))
        for key in sorted(set(completions) | set(observations))
    ]
    per_provider: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["verification_status"] == "verified":
            groups[(str(row["model"]), str(row["serving_provider"]))].append(row)
    for (model, provider), members in sorted(groups.items()):
        costs = [x["cost_usd"] for x in members if x["cost_usd"] is not None]
        latencies = [x["latency_ms"] for x in members if x["latency_ms"] is not None]
        secondary = [x["secondary_provider_selected"] for x in members
                     if x["secondary_provider_selected"] is not None]
        per_provider.append({
            "model": model,
            "serving_provider": provider,
            "verified_generations": len(members),
            "mean_cost_usd": mean(costs) if costs else None,
            "total_cost_usd": sum(costs) if costs else None,
            "mean_latency_ms": mean(latencies) if latencies else None,
            "secondary_selection_count": sum(x is True for x in secondary),
            "secondary_selection_fraction": (
                sum(x is True for x in secondary) / len(secondary) if secondary else None
            ),
            "reliability_rate": None,  # Failures lack a verified serving provider.
        })
    failures_by_model = dict(sorted(Counter(str(x["model"]) for x in failures).items()))
    completed_by_model = dict(sorted(Counter(
        str(_payload(x).get("model")) for x in completions.values()
    ).items()))
    model_route_reliability = []
    for model in sorted(set(failures_by_model) | set(completed_by_model)):
        succeeded = completed_by_model.get(model, 0)
        failed = failures_by_model.get(model, 0)
        model_route_reliability.append({
            "model": model,
            "completed": succeeded,
            "failed": failed,
            "completion_rate": succeeded / (succeeded + failed) if succeeded + failed else None,
            "caveat": "Across observed Runtime calls, not attributed to upstream providers",
        })
    return {
        "schema_version": 1,
        "source": "agenticforge-platform.af_events",
        "counts": {
            "completed": len(completions),
            "verified": sum(r["verification_status"] == "verified" for r in rows),
            "unverified": sum(r["verification_status"] != "verified" for r in rows),
            "failed": len(failures),
            "unlinked_observations": sum(r["observed_without_execution"] for r in rows),
        },
        "generations": rows,
        "failures": failures,
        "per_provider": per_provider,
        "model_route_reliability": model_route_reliability,
        "limits": [
            "Actual serving providers are included only when OpenRouter metadata verified them",
            "Secondary endpoint selection does not prove fallback",
            "Upstream-provider reliability cannot be calculated from unattributed failed requests",
            "Coverage depends on configured Platform evidence publishing and completed metadata lookups",
            "Data are observational, not randomized provider performance comparisons",
        ],
    }


def _csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Keep provider routing as JSON rather than Python repr for portability.
            cleaned = {
                name: (json.dumps(row[name]) if isinstance(row.get(name), (dict, list)) else row.get(name))
                for name in columns
            }
            writer.writerow(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported Platform AI provider-attribution events")
    parser.add_argument("events_jsonl", type=Path, help="JSONL export of Platform af_events")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_events(load_events(args.events_jsonl))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "provider-report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generation_columns = [
        "generation_id", "model", "execution_kind", "trace_id",
        "verification_status", "verification_error", "serving_provider",
        "preferred_providers", "allowed_providers", "preferred_provider_selected",
        "secondary_provider_selected", "fallback_used", "cost_usd", "cost_source",
        "latency_ms", "latency_source", "prompt_tokens", "completion_tokens",
        "cached_tokens", "reasoning_tokens", "observed_without_execution",
    ]
    _csv_rows(args.output_dir / "provider-generations.csv", result["generations"], generation_columns)
    _csv_rows(
        args.output_dir / "provider-summary.csv", result["per_provider"],
        ["model", "serving_provider", "verified_generations", "mean_cost_usd",
         "total_cost_usd", "mean_latency_ms", "secondary_selection_count",
         "secondary_selection_fraction", "reliability_rate"],
    )
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
