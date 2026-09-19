from __future__ import annotations

import csv
import html
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from .config import LABEL, METRICS, T975, Result


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else math.nan


def ci95(values: Iterable[float]) -> tuple[float, float, float, int]:
    vals = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    n = len(vals)
    if not vals:
        return math.nan, math.nan, math.nan, 0

    center = statistics.mean(vals)
    if n < 2:
        return center, center, center, n

    sem = statistics.stdev(vals) / math.sqrt(n)
    critical = T975.get(n - 1, 1.959964)
    half = critical * sem
    return center, max(0.0, center - half), center + half, n


def task_summary(
    results: list[Result],
    arms: list[str],
    tasks: list[str],
) -> list[dict[str, Any]]:
    """Average repeats within a task so tasks, not repeated calls, define n."""
    rows: list[dict[str, Any]] = []

    for arm in arms:
        for task in tasks:
            raw = [
                result
                for result in results
                if result.arm == arm and result.task == task
            ]
            good = [result for result in raw if result.valid]
            costs = [
                result.cost_usd
                for result in good
                if result.cost_usd is not None
            ]

            rows.append(
                {
                    "arm": arm,
                    "label": LABEL[arm],
                    "task": task,
                    "runs": len(raw),
                    "valid_runs": len(good),
                    "all_runs_valid": bool(raw) and len(good) == len(raw),
                    "total_tokens": mean(
                        result.total_tokens for result in good
                    ),
                    "input_tokens": mean(
                        result.input_tokens for result in good
                    ),
                    "output_tokens": mean(
                        result.output_tokens for result in good
                    ),
                    "reasoning_tokens": mean(
                        result.reasoning_tokens for result in good
                    ),
                    "cache_read_tokens": mean(
                        result.cache_read_tokens for result in good
                    ),
                    "cost_usd": mean(costs),
                    "wall_seconds": mean(
                        result.wall_seconds for result in good
                    ),
                    "api_calls": mean(
                        result.api_calls for result in good
                    ),
                    "diff_lines": mean(
                        result.diff_lines for result in good
                    ),
                }
            )

    return rows


def aggregate_summary(
    task_rows: list[dict[str, Any]],
    results: list[Result],
    arms: list[str],
    expected_tasks: list[str],
) -> list[dict[str, Any]]:
    """Calculate treatment means and t CIs across task-level means."""
    out: list[dict[str, Any]] = []

    for arm in arms:
        rows = [
            row
            for row in task_rows
            if row["arm"] == arm and row["valid_runs"] > 0
        ]
        raw = [result for result in results if result.arm == arm]

        entry: dict[str, Any] = {
            "arm": arm,
            "label": LABEL[arm],
            "runs": len(raw),
            "valid_runs": sum(result.valid for result in raw),
            "run_valid_rate": (
                100 * sum(result.valid for result in raw) / len(raw)
                if raw
                else math.nan
            ),
            "tasks_expected": len(expected_tasks),
            "tasks_valid": len(rows),
            "all_tasks_valid": len(rows) == len(expected_tasks),
        }

        for metric in METRICS:
            center, low, high, n = ci95(
                row[metric] for row in rows
            )
            entry[f"mean_{metric}"] = center
            entry[f"ci95_low_{metric}"] = low
            entry[f"ci95_high_{metric}"] = high
            entry[f"n_tasks_{metric}"] = n

        entry["actual_cost_runs"] = sum(
            result.cost_source == "openrouter_actual"
            for result in raw
            if result.valid
        )
        out.append(entry)

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float, kind: str) -> str:
    if not math.isfinite(value):
        return "n/a"
    if kind == "usd":
        return "$" + f"{value:.4f}"
    if kind == "tokens":
        return (
            f"{value / 1000:.1f}k"
            if abs(value) >= 1000
            else f"{value:.0f}"
        )
    if kind == "seconds":
        return f"{value:.1f}s"
    return f"{value:.1f}"


def plot_metric(
    output: Path,
    summary: list[dict[str, Any]],
    metric: str,
) -> None:
    title, kind = METRICS[metric]
    labels = [row["label"] for row in summary]
    values = [float(row[f"mean_{metric}"]) for row in summary]
    lows = [float(row[f"ci95_low_{metric}"]) for row in summary]
    highs = [float(row[f"ci95_high_{metric}"]) for row in summary]

    lower = [
        max(0.0, value - low)
        if math.isfinite(value) and math.isfinite(low)
        else 0.0
        for value, low in zip(values, lows)
    ]
    upper = [
        max(0.0, high - value)
        if math.isfinite(value) and math.isfinite(high)
        else 0.0
        for value, high in zip(values, highs)
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = list(range(len(labels)))
    bars = ax.bar(x, values, yerr=[lower, upper], capsize=7)
    ax.set_title(f"{title} by Hermes treatment")
    ax.set_xticks(x, labels)
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_value(value, kind),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.text(
        0.5,
        0.015,
        "Mean across three QuixBugs tasks; error bars are 95% Student-t CIs across task means.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / f"{metric}.png", dpi=180)
    fig.savefig(output / f"{metric}.svg")
    plt.close(fig)


def plot_validity(
    output: Path,
    summary: list[dict[str, Any]],
) -> None:
    labels = [row["label"] for row in summary]
    values = [float(row["run_valid_rate"]) for row in summary]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = list(range(len(labels)))
    bars = ax.bar(x, values)
    ax.set_title("Valid repair rate by Hermes treatment")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Valid runs (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output / "valid_repair_rate.png", dpi=180)
    fig.savefig(output / "valid_repair_rate.svg")
    plt.close(fig)


def write_html_report(
    output: Path,
    summary: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    metric_cards = "".join(
        (
            f'<div class="card"><img src="{metric}.png" '
            f'alt="{html.escape(METRICS[metric][0])}"></div>'
        )
        for metric in METRICS
    )
    metric_cards += (
        '<div class="card"><img src="valid_repair_rate.png" '
        'alt="Valid repair rate"></div>'
    )

    table_rows: list[str] = []
    for row in summary:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['tasks_valid']}/{row['tasks_expected']}</td>"
            f"<td>{format_value(row['mean_total_tokens'], 'tokens')} "
            f"[{format_value(row['ci95_low_total_tokens'], 'tokens')}, "
            f"{format_value(row['ci95_high_total_tokens'], 'tokens')}]</td>"
            f"<td>{format_value(row['mean_cost_usd'], 'usd')} "
            f"[{format_value(row['ci95_low_cost_usd'], 'usd')}, "
            f"{format_value(row['ci95_high_cost_usd'], 'usd')}]</td>"
            f"<td>{format_value(row['mean_wall_seconds'], 'seconds')} "
            f"[{format_value(row['ci95_low_wall_seconds'], 'seconds')}, "
            f"{format_value(row['ci95_high_wall_seconds'], 'seconds')}]</td>"
            f"<td>{row['run_valid_rate']:.0f}%</td>"
            "</tr>"
        )

    body = f'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forge Bench report</title>
<style>
body{{font-family:Inter,system-ui,Arial,sans-serif;background:#f8fafc;color:#111827;max-width:1180px;margin:auto;padding:36px 24px 72px}}
h1{{font-size:34px;margin-bottom:6px}}
p{{color:#64748b;line-height:1.5}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}}
.card{{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:8px;box-shadow:0 3px 12px #0f172a0a}}
.card img{{width:100%;display:block}}
table{{width:100%;border-collapse:collapse;background:white;border:1px solid #e5e7eb}}
th,td{{padding:11px;border-bottom:1px solid #e5e7eb;text-align:right;font-size:13px}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f1f5f9;color:#475569}}
.note{{background:#eef2ff;border:1px solid #c7d2fe;border-radius:12px;padding:13px 15px;color:#3730a3}}
</style>
<h1>Forge Bench</h1>
<p><code>{html.escape(meta['model'])}</code> via OpenRouter, pinned to
<code>{html.escape(meta['upstream_provider'])}</code>. Randomization seed:
<code>{meta['seed']}</code>.</p>
<p class="note">Primary bars are means across task-level means, not pooled agent calls.
Error bars are two-sided 95% Student-t confidence intervals across the three
QuixBugs tasks. With n=3 tasks these intervals are deliberately wide.</p>
<div class="grid">{metric_cards}</div>
<h2>Across-task summary</h2>
<table>
<thead><tr>
<th>Treatment</th>
<th>Valid tasks</th>
<th>Total tokens (95% CI)</th>
<th>Cost (95% CI)</th>
<th>Time (95% CI)</th>
<th>Valid runs</th>
</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<p>Raw evidence: <code>runs.csv</code>, <code>runs.json</code>,
<code>task_summary.csv</code>, <code>summary.csv</code>,
<code>run_plan.csv</code>, and per-run directories under <code>runs/</code>.</p>
'''
    (output / "report.html").write_text(body, encoding="utf-8")


def write_reports(
    output: Path,
    results: list[Result],
    arms: list[str],
    tasks: list[str],
    meta: dict[str, Any],
) -> None:
    raw = [
        asdict(result)
        for result in sorted(results, key=lambda item: item.run_index)
    ]
    per_task = task_summary(results, arms, tasks)
    summary = aggregate_summary(per_task, results, arms, tasks)

    write_csv(output / "runs.csv", raw)
    write_csv(output / "task_summary.csv", per_task)
    write_csv(output / "summary.csv", summary)
    (output / "runs.json").write_text(
        json.dumps(raw, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    for metric in METRICS:
        plot_metric(output, summary, metric)
    plot_validity(output, summary)
    write_html_report(output, summary, meta)
