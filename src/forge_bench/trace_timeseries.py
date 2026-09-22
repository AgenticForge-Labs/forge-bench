from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import LABEL, Result, T975


TOKEN_POOLS = (
    ("cache_read_tokens", "Cache-read input"),
    ("input_tokens", "Other input"),
    ("output_tokens", "Output"),
    ("reasoning_tokens", "Reasoning"),
)
POOL_COLORS = {
    "cache_read_tokens": "#8b5cf6",
    "input_tokens": "#0ea5e9",
    "output_tokens": "#f59e0b",
    "reasoning_tokens": "#ef4444",
}
ARM_COLORS = {
    "baseline": "#64748b",
    "caveman": "#0891b2",
    "ponytail": "#7c3aed",
    "caveman_ponytail": "#db2777",
    "lean_tools": "#16a34a",
    "all_three": "#ea580c",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _time(row: dict[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = _number(row.get(field))
        if value > 0:
            return value
    return None


def _run_events(result: Result) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run = Path(result.run_dir)
    api = [row for row in _read_jsonl(run / "api-events.jsonl") if row.get("event") == "post_api_request"]
    tools = [row for row in _read_jsonl(run / "tool-events.jsonl") if row.get("event") == "post_tool_call"]
    api.sort(key=lambda row: _time(row, "ended_at", "captured_at_unix") or 0)
    tools.sort(key=lambda row: _time(row, "captured_at_unix") or 0)
    return api, tools


def _pool_values(event: dict[str, Any]) -> dict[str, float]:
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    return {
        "cache_read_tokens": _number(usage.get("cache_read_tokens", usage.get("cached_tokens"))),
        "input_tokens": _number(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _number(usage.get("output_tokens", usage.get("completion_tokens"))),
        "reasoning_tokens": _number(usage.get("reasoning_tokens")),
    }


def _timeline(result: Result) -> dict[str, Any] | None:
    api, tools = _run_events(result)
    if not api and not tools:
        return None
    starts = [t for row in api if (t := _time(row, "started_at")) is not None]
    ends = [t for row in api if (t := _time(row, "ended_at", "captured_at_unix")) is not None]
    tool_times = [t for row in tools if (t := _time(row, "captured_at_unix")) is not None]
    start = min(starts + ends + tool_times)
    end = max(ends + tool_times)
    span = max(1e-6, end - start)

    events: list[tuple[float, str, dict[str, Any]]] = []
    for row in api:
        timestamp = _time(row, "ended_at", "captured_at_unix")
        if timestamp is not None:
            events.append((timestamp, "api", row))
    for row in tools:
        timestamp = _time(row, "captured_at_unix")
        if timestamp is not None:
            events.append((timestamp, "tool", row))
    events.sort(key=lambda item: item[0])

    totals = {pool: 0.0 for pool, _ in TOKEN_POOLS}
    tool_count = 0
    tool_seconds = 0.0
    snapshots = [{"progress": 0.0, **totals, "tool_calls": 0.0, "tool_seconds": 0.0}]
    for timestamp, kind, row in events:
        progress = min(100.0, max(0.0, 100.0 * (timestamp - start) / span))
        if kind == "api":
            values = _pool_values(row)
            for pool in totals:
                totals[pool] += values[pool]
        else:
            tool_count += 1
            tool_seconds += max(0.0, _number(row.get("duration_ms")) / 1000.0)
        snapshots.append({
            "progress": progress,
            **totals,
            "tool_calls": float(tool_count),
            "tool_seconds": tool_seconds,
        })

    return {
        "result": result,
        "api": api,
        "tools": tools,
        "start": start,
        "span": span,
        "snapshots": snapshots,
        "total_tokens": sum(totals.values()),
    }


def _sample(snapshots: list[dict[str, float]], progress: float) -> dict[str, float]:
    selected = snapshots[0]
    for row in snapshots:
        if row["progress"] > progress:
            break
        selected = row
    return selected


def _ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    center = statistics.mean(values)
    if len(values) < 2:
        return center, center, center
    half = T975.get(len(values) - 1, 1.959964) * statistics.stdev(values) / math.sqrt(len(values))
    return center, max(0.0, center - half), center + half


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: Any, output: Path, stem: str) -> str:
    path = output / f"{stem}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_example_timelines(output: Path, timelines: list[dict[str, Any]], descriptors: list[dict[str, str]]) -> str | None:
    if not timelines:
        return None
    examples: list[dict[str, Any]] = []
    for model in descriptors:
        candidates = [item for item in timelines if item["result"].model == model["model"]]
        if candidates:
            examples.append(max(candidates, key=lambda item: item["total_tokens"]))
    if not examples:
        return None
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 4.3 * len(examples)), squeeze=False)
    for row_index, item in enumerate(examples):
        result = item["result"]
        ax = axes[row_index][0]
        x_api = [100 * ((_time(event, "ended_at", "captured_at_unix") or item["start"]) - item["start"]) / item["span"] for event in item["api"]]
        cumulative = {pool: [] for pool, _ in TOKEN_POOLS}
        totals = {pool: 0.0 for pool, _ in TOKEN_POOLS}
        for event in item["api"]:
            values = _pool_values(event)
            for pool in totals:
                totals[pool] += values[pool]
                cumulative[pool].append(totals[pool])
        bottoms = np.zeros(len(x_api))
        for pool, label in TOKEN_POOLS:
            heights = np.asarray(cumulative[pool])
            ax.fill_between(x_api, bottoms, bottoms + heights, step="post", alpha=0.72, color=POOL_COLORS[pool], label=label)
            bottoms = bottoms + heights
        run_status = "valid" if result.valid else "invalid"
        ax.set_title(f"{result.model} — {LABEL.get(result.arm, result.arm)} — {result.task} ({run_status} result)")
        ax.set_xlabel("Elapsed run (%)")
        ax.set_ylabel("Cumulative tokens")
        ax.legend(fontsize=8, loc="upper left")

        tool_ax = axes[row_index][1]
        tool_x = [100 * ((_time(event, "captured_at_unix") or item["start"]) - item["start"]) / item["span"] for event in item["tools"]]
        tool_ax.scatter(tool_x, np.arange(1, len(tool_x) + 1), s=16, alpha=0.75, color=ARM_COLORS.get(result.arm, "#334155"))
        tool_ax.set_title(f"Computer tool calls ({len(tool_x)})")
        tool_ax.set_xlabel("Elapsed run (%)")
        tool_ax.set_ylabel("Cumulative tool-call index")
        tool_ax.set_xlim(0, 100)
    fig.suptitle("Highest observed trace-token run per model (result validity shown)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, output, "trace_example_timeline")


def _trajectory_rows(timelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for item in timelines:
        result = item["result"]
        for progress in range(0, 101, 5):
            snap = _sample(item["snapshots"], float(progress))
            raw.append({
                "model": result.model,
                "arm": result.arm,
                "task": result.task,
                "repeat": result.repeat,
                "run_index": result.run_index,
                "result_valid": result.valid,
                "result_completed": result.completed,
                "progress_percent": progress,
                **{pool: snap[pool] for pool, _ in TOKEN_POOLS},
                "tool_calls": snap["tool_calls"],
                "tool_seconds": snap["tool_seconds"],
            })
    return raw


def _plot_trajectories(output: Path, raw: list[dict[str, Any]], descriptors: list[dict[str, str]]) -> list[str]:
    if not raw:
        return []
    arms = list(dict.fromkeys(str(row["arm"]) for row in raw))
    models = [d for d in descriptors if any(row["model"] == d["model"] for row in raw)]
    metrics = [(pool, label) for pool, label in TOKEN_POOLS] + [("tool_calls", "Cumulative tool calls"), ("tool_seconds", "Cumulative tool execution seconds")]

    # Average repeats within task first; tasks then contribute equally to each curve.
    by_task: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_task[(str(row["model"]), str(row["arm"]), str(row["task"]), int(row["progress_percent"]))].append(row)
    task_rows: list[dict[str, Any]] = []
    for (model, arm, task, progress), rows in by_task.items():
        task_rows.append({
            "model": model, "arm": arm, "task": task, "progress_percent": progress,
            "valid_repeats": sum(bool(row["result_valid"]) for row in rows),
            "trace_repeats": len(rows),
            **{metric: statistics.mean(float(row[metric]) for row in rows) for metric, _ in metrics},
        })
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[(str(row["model"]), str(row["arm"]), int(row["progress_percent"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (model, arm, progress), rows in grouped.items():
        entry: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "progress_percent": progress,
            "n_tasks_with_trace": len(rows),
            "n_tasks_with_valid_result": sum(int(row["valid_repeats"] > 0) for row in rows),
        }
        for metric, _ in metrics:
            center, low, high = _ci([float(row[metric]) for row in rows])
            entry[f"mean_{metric}"] = center
            entry[f"ci95_low_{metric}"] = low
            entry[f"ci95_high_{metric}"] = high
        summary.append(entry)

    _write_csv(output / "trace_trajectory_run_progress.csv", raw)
    _write_csv(output / "trace_trajectory_task_means.csv", task_rows)
    _write_csv(output / "trace_trajectory_summary.csv", summary)

    fig, axes = plt.subplots(len(metrics), len(models), figsize=(6.7 * max(1, len(models)), 2.8 * len(metrics)), squeeze=False, sharex=True)
    for col, descriptor in enumerate(models):
        for row_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index][col]
            for arm in arms:
                rows = sorted([r for r in summary if r["model"] == descriptor["model"] and r["arm"] == arm], key=lambda r: r["progress_percent"])
                if not rows:
                    continue
                x = [int(r["progress_percent"]) for r in rows]
                y = [float(r[f"mean_{metric}"]) for r in rows]
                color = ARM_COLORS.get(arm, "#334155")
                ax.plot(x, y, color=color, lw=2, label=LABEL.get(arm, arm))
                low = [float(r[f"ci95_low_{metric}"]) for r in rows]
                high = [float(r[f"ci95_high_{metric}"]) for r in rows]
                ax.fill_between(x, low, high, color=color, alpha=0.13)
            ax.set_title(f"{descriptor['label']} — {title}" if row_index == 0 else title, fontsize=10)
            ax.set_ylabel("Seconds" if metric == "tool_seconds" else ("Tool calls" if metric == "tool_calls" else "Tokens"))
            ax.grid(alpha=0.2)
            if row_index == 0:
                ax.legend(fontsize=8, ncol=2)
            if row_index == len(metrics) - 1:
                ax.set_xlabel("Normalized elapsed run progress (%)")
    fig.suptitle("Token pools and computer activity over the run", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return [_save(fig, output, "trace_token_tool_trajectories")]


def _request_work_rows(timelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in timelines:
        result = item["result"]
        api, tools = item["api"], item["tools"]
        for index, event in enumerate(api):
            end = _time(event, "ended_at", "captured_at_unix")
            next_start = _time(api[index + 1], "started_at") if index + 1 < len(api) else None
            if end is None:
                continue
            stop = next_start or item["start"] + item["span"]
            following = [row for row in tools if end <= (_time(row, "captured_at_unix") or 0) <= stop]
            values = _pool_values(event)
            rows.append({
                "model": result.model,
                "arm": result.arm,
                "task": result.task,
                "run_index": result.run_index,
                "result_valid": result.valid,
                "api_request_index": index + 1,
                "response_output_tokens": values["output_tokens"],
                "response_reasoning_tokens": values["reasoning_tokens"],
                "following_tool_calls": len(following),
                "following_tool_seconds": sum(_number(row.get("duration_ms")) for row in following) / 1000.0,
                "following_tool_names": ";".join(str(row.get("tool_name") or "") for row in following),
            })
    return rows


def _plot_request_work(output: Path, rows: list[dict[str, Any]], descriptors: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    models = [d for d in descriptors if any(row["model"] == d["model"] for row in rows)]
    fig, axes = plt.subplots(len(models), 2, figsize=(6.4 * 2, 4.1 * len(models)), squeeze=False)
    measures = (
        ("following_tool_calls", "Tool calls before next model request"),
        ("following_tool_seconds", "Tool execution seconds before next model request"),
    )
    for row_index, descriptor in enumerate(models):
        subset = [row for row in rows if row["model"] == descriptor["model"]]
        for col, (measure, xlabel) in enumerate(measures):
            ax = axes[row_index][col]
            for arm in dict.fromkeys(str(row["arm"]) for row in subset):
                group = [row for row in subset if row["arm"] == arm]
                for valid, marker in ((True, "o"), (False, "x")):
                    status_group = [row for row in group if bool(row["result_valid"]) is valid]
                    if status_group:
                        ax.scatter(
                            [row[measure] for row in status_group],
                            [row["response_output_tokens"] for row in status_group],
                            label=LABEL.get(arm, arm) if valid or not any(r["result_valid"] for r in group) else None,
                            marker=marker, alpha=0.56, s=22,
                            color=ARM_COLORS.get(arm, "#334155"),
                        )
            ax.set_title(descriptor["label"] if col == 0 else f"{descriptor['label']} — tool time")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Tokens generated in this response")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Per-response generation and subsequent computer activity (x = invalid result)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output, "trace_response_vs_tool_work")


def write_trace_timeseries(
    output: Path,
    results: list[Result],
    descriptors: list[dict[str, str]],
) -> list[str]:
    """Write trace-level token-pool and computer-work time series when observer traces exist."""
    timelines = [item for result in results if (item := _timeline(result)) is not None]
    if not timelines:
        return []
    generated: list[str] = []
    example = _plot_example_timelines(output, timelines, descriptors)
    if example:
        generated.append(example)
    trajectory = _plot_trajectories(output, _trajectory_rows(timelines), descriptors)
    generated.extend(trajectory)
    request_rows = _request_work_rows(timelines)
    _write_csv(output / "trace_response_tool_work.csv", request_rows)
    scatter = _plot_request_work(output, request_rows, descriptors)
    if scatter:
        generated.append(scatter)
    return generated
