from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import yaml

TRACE_PLUGIN_NAME = "forge-bench-trace"
TRACE_EVENT_FILE = "forge-bench-events.jsonl"

TRACE_EVENTS = (
    "on_session_start",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "on_skill_lifecycle",
    "subagent_start",
    "subagent_stop",
    "agent_loop_stopped",
    "on_session_end",
    "on_session_finalize",
)

PLUGIN_YAML = """name: forge-bench-trace
version: "1.0.0"
description: Low-overhead Forge Bench observer for timing and audit traces.
"""

PLUGIN_INIT = r'''"""Forge Bench native Hermes observer.

This plugin is intentionally observer-only.  It never returns a hook directive,
changes model context, changes tool arguments, or exposes an additional tool.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_OUT = Path(__file__).resolve().parents[2] / "forge-bench-events.jsonl"
_LOCK = threading.Lock()

_FIELDS = {
    "on_session_start": (
        "session_id", "task_id", "platform", "model", "provider",
    ),
    "pre_llm_call": (
        "session_id", "task_id", "model", "provider", "iteration",
        "is_first_turn",
    ),
    "post_llm_call": (
        "session_id", "task_id", "model", "provider", "iteration",
    ),
    "pre_api_request": (
        "session_id", "task_id", "api_request_id", "model", "provider",
        "base_url", "api_mode", "api_call_count", "message_count",
        "tool_count", "approx_input_tokens", "request_char_count",
        "max_tokens", "retry_count",
    ),
    "post_api_request": (
        "session_id", "task_id", "api_request_id", "model", "provider",
        "base_url", "api_mode", "api_call_count", "api_duration",
        "started_at", "ended_at", "first_chunk_at", "finish_reason",
        "message_count", "response_model", "usage",
        "assistant_content_chars", "assistant_tool_call_count",
        "retry_count",
    ),
    "api_request_error": (
        "session_id", "task_id", "api_request_id", "model", "provider",
        "status_code", "retry_count", "max_retries", "retryable",
        "reason", "error_type", "error_code",
    ),
    "pre_tool_call": (
        "session_id", "task_id", "tool_call_id", "tool_name",
    ),
    "post_tool_call": (
        "session_id", "task_id", "tool_call_id", "tool_name",
        "duration_ms",
    ),
    "on_skill_lifecycle": (
        "session_id", "task_id", "action", "skill_name", "provenance",
        "use_count", "reused", "reuse_after_patch",
    ),
    "subagent_start": (
        "session_id", "task_id", "parent_session_id", "child_session_id",
        "child_role",
    ),
    "subagent_stop": (
        "session_id", "task_id", "parent_session_id", "child_session_id",
        "child_role", "child_status", "duration_ms",
    ),
    "agent_loop_stopped": (
        "session_id", "task_id", "session_key", "platform", "reason",
        "invalidation_reason",
    ),
    "on_session_end": (
        "session_id", "task_id", "turn_id", "completed", "failed",
        "interrupted", "turn_exit_reason", "model", "platform",
    ),
    "on_session_finalize": (
        "session_id", "task_id",
    ),
}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _record(event, kwargs):
    fields = _FIELDS.get(event, tuple(kwargs))
    record = {
        "event": event,
        "captured_at_unix": time.time(),
        "captured_at_monotonic_ns": time.monotonic_ns(),
    }
    for key in fields:
        if key in kwargs:
            record[key] = _jsonable(kwargs[key])
    # Preserve the names of additive Hermes fields without serializing huge
    # request/response/result payloads.  Full messages/tool payloads are kept
    # separately through Hermes' native session export and state.db snapshot.
    record["available_fields"] = sorted(str(key) for key in kwargs)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        with _OUT.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _callback(event):
    def observe(**kwargs):
        _record(event, kwargs)
        return None
    return observe


def register(ctx):
    for event in (
        "on_session_start",
        "pre_llm_call",
        "post_llm_call",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "on_skill_lifecycle",
        "subagent_start",
        "subagent_stop",
        "agent_loop_stopped",
        "on_session_end",
        "on_session_finalize",
    ):
        ctx.register_hook(event, _callback(event))
'''


def install_trace_plugin(profile: Path) -> None:
    """Install and enable the observer inside one isolated Hermes profile."""
    plugin = profile / "plugins" / TRACE_PLUGIN_NAME
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(PLUGIN_YAML, encoding="utf-8")
    (plugin / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")

    config_path = profile / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    plugins = cfg.setdefault("plugins", {})
    enabled = list(plugins.get("enabled") or [])
    if TRACE_PLUGIN_NAME not in enabled:
        enabled.append(TRACE_PLUGIN_NAME)
    plugins["enabled"] = enabled
    plugins.setdefault("disabled", [])
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def trace_plugin_enabled(profile: Path) -> bool:
    try:
        cfg = yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return TRACE_PLUGIN_NAME in list((cfg.get("plugins") or {}).get("enabled") or [])


def load_events(profile: Path) -> list[dict[str, Any]]:
    path = profile / TRACE_EVENT_FILE
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_events(events: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    api_events = [row for row in events if row.get("event") == "post_api_request"]
    tool_events = [row for row in events if row.get("event") == "post_tool_call"]
    skill_events = [row for row in events if row.get("event") == "on_skill_lifecycle"]

    api_durations = [
        value
        for row in api_events
        if (value := _finite_number(row.get("api_duration"))) is not None
        and value >= 0
    ]
    tool_durations = [
        value / 1000.0
        for row in tool_events
        if (value := _finite_number(row.get("duration_ms"))) is not None
        and value >= 0
    ]
    terminal_durations = [
        value / 1000.0
        for row in tool_events
        if str(row.get("tool_name") or "") in {"terminal", "process"}
        and (value := _finite_number(row.get("duration_ms"))) is not None
        and value >= 0
    ]

    ttft: list[float] = []
    for row in api_events:
        started = _finite_number(row.get("started_at"))
        first = _finite_number(row.get("first_chunk_at"))
        if started is not None and first is not None and first >= started:
            ttft.append(first - started)

    api_total = float(sum(api_durations))
    tool_total = float(sum(tool_durations))
    residual = float(wall_seconds) - api_total - tool_total
    skill_names = sorted({
        str(row.get("skill_name"))
        for row in skill_events
        if row.get("skill_name")
    })

    by_tool: dict[str, dict[str, Any]] = {}
    for row in tool_events:
        name = str(row.get("tool_name") or "unknown")
        entry = by_tool.setdefault(name, {"calls": 0, "seconds": 0.0})
        entry["calls"] += 1
        duration = _finite_number(row.get("duration_ms"))
        if duration is not None and duration >= 0:
            entry["seconds"] += duration / 1000.0

    return {
        "observer": "native Hermes plugin hook",
        "wall_seconds": float(wall_seconds),
        "event_count": len(events),
        "api_post_event_count": len(api_events),
        "tool_post_event_count": len(tool_events),
        "skill_lifecycle_event_count": len(skill_events),
        "skill_names_observed": skill_names,
        "caveman_skill_observed": any("caveman" in name.lower() for name in skill_names),
        "api_wait_seconds": api_total if api_durations else None,
        "tool_execution_seconds": tool_total if tool_durations else None,
        "terminal_execution_seconds": (
            float(sum(terminal_durations)) if terminal_durations else None
        ),
        "unattributed_wall_seconds": max(0.0, residual)
        if api_durations or tool_durations
        else None,
        "overlap_excess_seconds": max(0.0, -residual)
        if api_durations or tool_durations
        else None,
        "api_duration_mean_seconds": (
            statistics.mean(api_durations) if api_durations else None
        ),
        "api_duration_median_seconds": (
            statistics.median(api_durations) if api_durations else None
        ),
        "api_duration_p95_seconds": _percentile(api_durations, 0.95),
        "ttft_mean_seconds": statistics.mean(ttft) if ttft else None,
        "ttft_median_seconds": statistics.median(ttft) if ttft else None,
        "tool_time_by_name": by_tool,
        "note": (
            "API and tool durations come from Hermes lifecycle payloads. "
            "Unattributed wall time is wall - API durations - tool durations; "
            "it includes orchestration, hook overhead, local Python work, and "
            "any uninstrumented waits. Negative raw residual is reported as "
            "overlap_excess_seconds rather than forcing components to add."
        ),
    }


def finalize_event_artifacts(
    profile: Path,
    run_dir: Path,
    wall_seconds: float,
) -> dict[str, Any]:
    """Copy/split raw observer events and write a per-run timing summary."""
    events = load_events(profile)
    source = profile / TRACE_EVENT_FILE
    if source.is_file():
        shutil.copy2(source, run_dir / "hermes-events.jsonl")

    groups = {
        "api-events.jsonl": {"pre_api_request", "post_api_request", "api_request_error"},
        "tool-events.jsonl": {"pre_tool_call", "post_tool_call"},
        "lifecycle-events.jsonl": set(TRACE_EVENTS)
        - {
            "pre_api_request", "post_api_request", "api_request_error",
            "pre_tool_call", "post_tool_call",
        },
    }
    for filename, names in groups.items():
        rows = [row for row in events if row.get("event") in names]
        if rows:
            (run_dir / filename).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

    summary = summarize_events(events, wall_seconds)
    (run_dir / "timing-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
