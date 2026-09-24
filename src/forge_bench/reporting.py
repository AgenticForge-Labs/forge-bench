from __future__ import annotations

import csv
import html
import json
import math
import statistics
import shutil
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from .advanced_analysis import treatment_colors, write_advanced_analysis
from .config import LABEL, METRICS, T975, Result, budget_condition_label, condition_order_key
from .multi_model_analysis import write_multi_model_analysis


def _ordered_arms(arms: Iterable[str]) -> list[str]:
    values = list(dict.fromkeys(str(arm) for arm in arms))
    return sorted(values, key=condition_order_key)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else math.nan


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    try:
        n = float(numerator) if numerator is not None else math.nan
        d = float(denominator) if denominator is not None else math.nan
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(n) or not math.isfinite(d) or d <= 0:
        return math.nan
    return n / d


def finite_mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(valid) if valid else math.nan


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


def ci95_signed(values: Iterable[float]) -> tuple[float, float, float, int]:
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
    return center, center - half, center + half, n


def task_summary(
    results: list[Result],
    arms: list[str],
    tasks: list[str],
) -> list[dict[str, Any]]:
    """Average repeats within a task so tasks, not repeated calls, define n."""
    rows: list[dict[str, Any]] = []

    for arm in _ordered_arms(arms):
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
                    "label": (
                        f"{budget_condition_label(arm)} · ratio null"
                        if len({result.budget_warning_ratio for result in results}) > 1
                        and raw and raw[0].budget_warning_ratio is None
                        else budget_condition_label(arm)
                    ),
                    "base_arm": (raw[0].base_arm or arm) if raw else arm,
                    "budget_warning_ratio": raw[0].budget_warning_ratio if raw else None,
                    "budget_warning_ratio_label": (
                        "null" if not raw or raw[0].budget_warning_ratio is None
                        else str(raw[0].budget_warning_ratio)
                    ),
                    "task": task,
                    "runs": len(raw),
                    "valid_runs": len(good),
                    "all_runs_valid": bool(raw) and len(good) == len(raw),
                    "resolved_runs": sum(result.resolved for result in good),
                    "resolve_rate": (
                        100 * sum(result.resolved for result in good) / len(good)
                        if good else math.nan
                    ),
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
                    "tool_calls": mean(
                        result.tool_calls
                        for result in good
                        if result.tool_calls is not None
                    ),
                    # Per-run ratios are averaged within task so one unusually
                    # large run cannot dominate by simple ratio-of-totals.
                    "tokens_per_second": finite_mean(
                        safe_ratio(result.total_tokens, result.wall_seconds)
                        for result in good
                    ),
                    "seconds_per_api_call": finite_mean(
                        safe_ratio(result.wall_seconds, result.api_calls)
                        for result in good
                    ),
                    "tokens_per_api_call": finite_mean(
                        safe_ratio(result.total_tokens, result.api_calls)
                        for result in good
                    ),
                    "cost_per_api_call": finite_mean(
                        safe_ratio(result.cost_usd, result.api_calls)
                        for result in good
                    ),
                    "cost_per_second": finite_mean(
                        safe_ratio(result.cost_usd, result.wall_seconds)
                        for result in good
                    ),
                    "seconds_per_tool_call": finite_mean(
                        safe_ratio(result.wall_seconds, result.tool_calls)
                        for result in good
                        if result.tool_calls is not None
                    ),
                    "tool_calls_per_api_call": finite_mean(
                        safe_ratio(result.tool_calls, result.api_calls)
                        for result in good
                        if result.tool_calls is not None
                    ),
                    "api_wait_seconds": finite_mean(
                        result.api_wait_seconds
                        for result in good
                        if result.api_wait_seconds is not None
                    ),
                    "tool_execution_seconds": finite_mean(
                        result.tool_execution_seconds
                        for result in good
                        if result.tool_execution_seconds is not None
                    ),
                    "terminal_execution_seconds": finite_mean(
                        result.terminal_execution_seconds
                        for result in good
                        if result.terminal_execution_seconds is not None
                    ),
                    "unattributed_wall_seconds": finite_mean(
                        result.unattributed_wall_seconds
                        for result in good
                        if result.unattributed_wall_seconds is not None
                    ),
                    "api_duration_mean_seconds": finite_mean(
                        result.api_duration_mean_seconds
                        for result in good
                        if result.api_duration_mean_seconds is not None
                    ),
                    "api_duration_p95_seconds": finite_mean(
                        result.api_duration_p95_seconds
                        for result in good
                        if result.api_duration_p95_seconds is not None
                    ),
                    "ttft_mean_seconds": finite_mean(
                        result.ttft_mean_seconds
                        for result in good
                        if result.ttft_mean_seconds is not None
                    ),
                    "api_wait_fraction": finite_mean(
                        safe_ratio(result.api_wait_seconds, result.wall_seconds)
                        for result in good
                        if result.api_wait_seconds is not None
                    ),
                    "tool_execution_fraction": finite_mean(
                        safe_ratio(result.tool_execution_seconds, result.wall_seconds)
                        for result in good
                        if result.tool_execution_seconds is not None
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

    for arm in _ordered_arms(arms):
        rows = [
            row
            for row in task_rows
            if row["arm"] == arm and row["valid_runs"] > 0
        ]
        raw = [result for result in results if result.arm == arm]

        entry: dict[str, Any] = {
            "arm": arm,
            "label": (
                f"{budget_condition_label(arm)} · ratio null"
                if len({result.budget_warning_ratio for result in results}) > 1
                and raw and raw[0].budget_warning_ratio is None
                else budget_condition_label(arm)
            ),
            "base_arm": (raw[0].base_arm or arm) if raw else arm,
            "budget_warning_ratio": raw[0].budget_warning_ratio if raw else None,
            "budget_warning_ratio_label": (
                "null" if not raw or raw[0].budget_warning_ratio is None
                else str(raw[0].budget_warning_ratio)
            ),
            "runs": len(raw),
            "valid_runs": sum(result.valid for result in raw),
            "run_valid_rate": (
                100 * sum(result.valid for result in raw) / len(raw)
                if raw
                else math.nan
            ),
            "run_resolve_rate": (
                100
                * sum(result.resolved for result in raw if result.valid)
                / sum(result.valid for result in raw)
                if any(result.valid for result in raw)
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


def _plot_theme(theme: str) -> dict[str, str]:
    if theme == "dark":
        return {
            "figure": "#0b1020",
            "axes": "#111827",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "grid": "#64748b",
            "edge": "#94a3b8",
        }
    return {
        "figure": "#ffffff",
        "axes": "#ffffff",
        "text": "#111827",
        "muted": "#475569",
        "grid": "#94a3b8",
        "edge": "#475569",
    }


def _style_axes(fig: Any, ax: Any, theme: str) -> None:
    colors = _plot_theme(theme)
    fig.patch.set_facecolor(colors["figure"])
    ax.set_facecolor(colors["axes"])
    ax.tick_params(colors=colors["text"], labelsize=12, width=1.4)
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["edge"])
        spine.set_linewidth(1.4)
    ax.grid(axis="y", alpha=0.30, linewidth=1.15, color=colors["grid"])
    ax.set_axisbelow(True)


def _save_plot(fig: Any, output: Path, stem: str, theme: str) -> None:
    png = output / f"{stem}.{theme}.png"
    svg = output / f"{stem}.{theme}.svg"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if theme == "light":
        shutil.copyfile(png, output / f"{stem}.png")
        shutil.copyfile(svg, output / f"{stem}.svg")


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

    for theme in ("light", "dark"):
        fig, ax = plt.subplots(figsize=(max(10.2, 1.75 * len(labels)), 6.8))
        _style_axes(fig, ax, theme)
        x = list(range(len(labels)))
        arm_colors = treatment_colors(
            [str(row["arm"]) for row in summary],
            theme,
        )
        if values and math.isfinite(values[0]):
            ax.axhline(
                values[0], color=_plot_theme(theme)["muted"], alpha=0.48,
                linestyle=(0, (4, 4)), linewidth=1.2, zorder=0,
            )
        bars = ax.bar(
            x,
            values,
            color=[arm_colors[str(row["arm"])] for row in summary],
            yerr=[lower, upper],
            capsize=8,
            error_kw={"elinewidth": 2.2, "capthick": 2.0, "ecolor": _plot_theme(theme)["text"]},
            linewidth=1.15,
            edgecolor=_plot_theme(theme)["edge"],
        )
        ax.set_title(
            f"{title} by Hermes treatment",
            fontsize=19,
            fontweight="semibold",
            pad=14,
        )
        ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=12)
        ax.set_ylabel(title, fontsize=14)
        ax.grid(axis="y", alpha=0.30, linewidth=1.15)

        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    format_value(value, kind),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="semibold",
                    color=_plot_theme(theme)["text"],
                )

        fig.text(
            0.5,
            0.018,
            "Mean across selected SWE-bench tasks; error bars are 95% Student-t CIs across task means.",
            ha="center",
            fontsize=11,
            color=_plot_theme(theme)["muted"],
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save_plot(fig, output, metric, theme)


def plot_validity(
    output: Path,
    summary: list[dict[str, Any]],
) -> None:
    labels = [row["label"] for row in summary]
    values = [float(row["run_resolve_rate"]) for row in summary]

    for theme in ("light", "dark"):
        fig, ax = plt.subplots(figsize=(max(10.2, 1.75 * len(labels)), 6.8))
        _style_axes(fig, ax, theme)
        x = list(range(len(labels)))
        arm_colors = treatment_colors(
            [str(row["arm"]) for row in summary],
            theme,
        )
        if values and math.isfinite(values[0]):
            ax.axhline(
                values[0], color=_plot_theme(theme)["muted"], alpha=0.48,
                linestyle=(0, (4, 4)), linewidth=1.2, zorder=0,
            )
        bars = ax.bar(
            x,
            values,
            color=[arm_colors[str(row["arm"])] for row in summary],
            linewidth=1.15,
            edgecolor=_plot_theme(theme)["edge"],
        )
        ax.set_title(
            "SWE-bench resolve rate by Hermes treatment",
            fontsize=19,
            fontweight="semibold",
            pad=14,
        )
        ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=12)
        ax.set_ylabel("Resolved valid runs (%)", fontsize=14)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.30, linewidth=1.15)

        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="semibold",
                    color=_plot_theme(theme)["text"],
                )

        fig.tight_layout()
        _save_plot(fig, output, "resolve_rate", theme)


def _theme_image(stem: str, alt: str, theme: str) -> str:
    return (
        f'<img src="{stem}.{theme}.svg" alt="{html.escape(alt)}">'
    )


def write_html_report(
    output: Path,
    summary: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    analysis_mode: str = "basic",
) -> None:
    visible_metrics = [
        metric
        for metric in METRICS
        if metric != "reasoning_tokens"
        or any(abs(float(row.get(f"mean_{metric}", 0.0))) > 1e-12 for row in summary)
    ]

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
            f"<td>{row['run_resolve_rate']:.0f}%</td>"
            f"<td>{row['run_valid_rate']:.0f}%</td>"
            "</tr>"
        )

    advanced_specs = [
        ("advanced_cost_time", "Cost-time efficiency frontier"),
        ("advanced_time_budget", "Direct wall-time budget by treatment"),
        ("advanced_api_latency", "API latency distribution by treatment"),
        ("advanced_time_vs_api_calls", "Wall time versus API-call count"),
        ("advanced_tokens_per_second", "Token throughput by treatment"),
        ("advanced_seconds_per_api_call", "Time per API call by treatment"),
        ("advanced_tokens_per_api_call", "Tokens per API call by treatment"),
        ("advanced_cost_per_api_call", "Cost per API call by treatment"),
        ("advanced_tool_calls_per_api_call", "Tool calls per API call"),
        ("advanced_seconds_per_tool_call", "Wall time per tool call"),
        ("advanced_token_effects", "Task-normalized token effects"),
        ("advanced_pca_biplot", "PCA biplot of efficiency profiles"),
        ("advanced_pca_scree", "PCA scree plot"),
        ("advanced_pca_loadings", "PC1 and PC2 feature loadings"),
        ("advanced_clusters", "Exploratory PCA-space clusters"),
        ("advanced_correlations", "Efficiency correlation matrix"),
    ]

    n_tasks = max((int(row["tasks_expected"]) for row in summary), default=0)
    for theme in ("light", "dark"):
        metric_cards = "".join(
            f'<div class="card">{_theme_image(metric, METRICS[metric][0], theme)}</div>'
            for metric in visible_metrics
        )
        metric_cards += (
            '<div class="card">'
            + _theme_image("resolve_rate", "SWE-bench resolve rate", theme)
            + '</div>'
        )

        advanced_cards = ""
        if analysis_mode == "advanced":
            advanced_cards = "".join(
                f'<div class="card">{_theme_image(stem, alt, theme)}</div>'
                for stem, alt in advanced_specs
                if (output / f"{stem}.{theme}.svg").exists()
            )

        advanced_section = ""
        if advanced_cards:
            advanced_section = (
                "<h2>Advanced analysis</h2>"
                "<p>Exploratory task-level analyses include direct API/tool/orchestration time "
                "budgets when native Hermes trace events are available, cost/time tradeoffs, "
                "token throughput, time and tokens per API call, tool-call intensity when available, paired "
                "baseline-normalized effects, task-fixed-effect regressions, Caveman×Ponytail "
                "factorial regression when available, PCA, deterministic clustering, and "
                "correlations. Legacy runs do not contain exact API-wait or tool-execution "
                "durations, so ratio diagnostics use the saved wall time and call counts.</p>"
                f'<div class="grid">{advanced_cards}</div>'
            )

        if theme == "dark":
            palette = {
                "bg": "#090e1a",
                "fg": "#f8fafc",
                "muted": "#cbd5e1",
                "card": "#111827",
                "border": "#334155",
                "head": "#1e293b",
                "note": "#172554",
                "note_border": "#1d4ed8",
                "note_fg": "#dbeafe",
                "shadow": "#0008",
            }
        else:
            palette = {
                "bg": "#f8fafc",
                "fg": "#111827",
                "muted": "#64748b",
                "card": "#ffffff",
                "border": "#e5e7eb",
                "head": "#f1f5f9",
                "note": "#eef2ff",
                "note_border": "#c7d2fe",
                "note_fg": "#3730a3",
                "shadow": "#0f172a0a",
            }

        body = f'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forge Bench report ({theme})</title>
<style>
:root{{--bg:{palette["bg"]};--fg:{palette["fg"]};--muted:{palette["muted"]};--card:{palette["card"]};--border:{palette["border"]};--head:{palette["head"]};--note:{palette["note"]};--note-border:{palette["note_border"]};--note-fg:{palette["note_fg"]};--shadow:{palette["shadow"]}}}
body{{font-family:Inter,system-ui,Arial,sans-serif;background:var(--bg);color:var(--fg);max-width:1280px;margin:auto;padding:36px 24px 72px}}
h1{{font-size:36px;margin-bottom:6px}} h2{{font-size:25px;margin-top:34px}}
p{{color:var(--muted);line-height:1.55;font-size:15px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:8px;box-shadow:0 3px 12px var(--shadow)}}
.card img{{width:100%;display:block}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border)}}
th,td{{padding:12px;border-bottom:1px solid var(--border);text-align:right;font-size:14px}}
th:first-child,td:first-child{{text-align:left}} th{{background:var(--head);color:var(--muted)}}
.note{{background:var(--note);border:1px solid var(--note-border);border-radius:12px;padding:14px 16px;color:var(--note-fg)}}
code{{font-size:.92em}}
</style>
<h1>Forge Bench</h1>
<p><code>{html.escape(str(meta.get('model', 'unknown')))}</code> via OpenRouter, pinned to
<code>{html.escape(str(meta.get('upstream_provider', 'unknown')))}</code>. Randomization seed:
<code>{html.escape(str(meta.get('seed', 'n/a')))}</code>. Analysis mode:
<code>{html.escape(analysis_mode)}</code>. Report theme:
<code>{theme}</code>.</p>
<p class="note">Primary efficiency bars are means across task-level means, not pooled agent calls.
Unresolved tasks remain in token, cost, and time averages when the agent run itself is usable.
Error bars are two-sided 95% Student-t confidence intervals across the selected SWE-bench tasks.
Current task count: {n_tasks}.</p>
<div class="grid">{metric_cards}</div>
{advanced_section}
<h2>Across-task summary</h2>
<table>
<thead><tr>
<th>Treatment</th><th>Valid tasks</th><th>Total tokens (95% CI)</th>
<th>Cost (95% CI)</th><th>Time (95% CI)</th><th>Resolved runs</th><th>Usable runs</th>
</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
{(
    '<p>This is a regenerated analysis. Original per-run evidence remains in '
    f'<code>{html.escape(str(meta["reanalyzed_from"]))}</code>. This folder contains '
    'a snapshot of the run records plus regenerated summaries, figures, and report.</p>'
    if meta.get("reanalyzed_from")
    else '<p>Raw evidence: <code>runs.csv</code>, <code>runs.json</code>, '
    '<code>task_summary.csv</code>, <code>summary.csv</code>, '
    '<code>run_plan.csv</code>, and per-run directories under <code>runs/</code>.</p>'
)}
<p>Advanced mode also writes analysis tables prefixed with <code>advanced_</code>.</p>
'''
        (output / f"report-{theme}.html").write_text(body, encoding="utf-8")

    # Compatibility alias for existing tooling: report.html is the light report.
    shutil.copyfile(output / "report-light.html", output / "report.html")


def write_reports(
    output: Path,
    results: list[Result],
    arms: list[str],
    tasks: list[str],
    meta: dict[str, Any],
    *,
    analysis_mode: str = "basic",
) -> None:
    raw = [
        asdict(result)
        for result in sorted(results, key=lambda item: item.run_index)
    ]
    per_task = task_summary(results, arms, tasks)
    summary = aggregate_summary(per_task, results, arms, tasks)
    meta["analysis_mode"] = analysis_mode
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
        if metric == "reasoning_tokens" and not any(
            abs(float(row.get("mean_reasoning_tokens", 0.0))) > 1e-12
            for row in summary
        ):
            continue
        plot_metric(output, summary, metric)
    plot_validity(output, summary)

    if analysis_mode == "advanced":
        write_advanced_analysis(output, per_task)

    write_html_report(
        output,
        summary,
        meta,
        analysis_mode=analysis_mode,
    )



def _slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return "-".join(part for part in text.split("-") if part) or "model"


def _model_descriptors(
    results: list[Result],
    meta: dict[str, Any],
) -> list[dict[str, str]]:
    configured = meta.get("models")
    configured_order: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    if isinstance(configured, list):
        for row in configured:
            if not isinstance(row, dict) or not row.get("model"):
                continue
            model_id = str(row["model"])
            descriptor = {
                "key": str(row.get("key") or _slug(model_id.rsplit("/", 1)[-1])),
                "label": str(row.get("label") or model_id),
                "model": model_id,
                "upstream_provider": str(row.get("upstream_provider") or ""),
            }
            configured_order.append(descriptor)
            by_id[model_id] = descriptor

    observed_order: list[str] = []
    observed_upstream: dict[str, str] = {}
    for result in sorted(results, key=lambda item: item.run_index):
        if result.model not in observed_order:
            observed_order.append(result.model)
        observed_upstream.setdefault(result.model, result.upstream_provider)

    observed = set(observed_order)
    descriptors = [
        descriptor
        for descriptor in configured_order
        if descriptor["model"] in observed
    ]
    configured_ids = {descriptor["model"] for descriptor in descriptors}
    for model_id in observed_order:
        if model_id in configured_ids:
            continue
        descriptors.append(
            by_id.get(
                model_id,
                {
                    "key": _slug(model_id.rsplit("/", 1)[-1]),
                    "label": model_id,
                    "model": model_id,
                    "upstream_provider": observed_upstream.get(model_id, ""),
                },
            )
        )
    return descriptors


def model_pairwise_effects(
    model_task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Paired task-level B-A effects for two-model experiment designs."""
    if len(descriptors) != 2:
        return []
    a, b = descriptors
    out: list[dict[str, Any]] = []
    metrics = ("total_tokens", "cost_usd", "wall_seconds", "api_calls")
    arms = _ordered_arms(row["arm"] for row in model_task_rows)
    for arm in arms:
        a_rows = {
            str(row["task"]): row
            for row in model_task_rows
            if row.get("model") == a["model"]
            and row.get("arm") == arm
            and row.get("valid_runs", 0) > 0
        }
        b_rows = {
            str(row["task"]): row
            for row in model_task_rows
            if row.get("model") == b["model"]
            and row.get("arm") == arm
            and row.get("valid_runs", 0) > 0
        }
        common = sorted(set(a_rows) & set(b_rows))
        for metric in metrics:
            deltas: list[float] = []
            pct: list[float] = []
            a_values: list[float] = []
            b_values: list[float] = []
            for task in common:
                av = float(a_rows[task].get(metric, math.nan))
                bv = float(b_rows[task].get(metric, math.nan))
                if not math.isfinite(av) or not math.isfinite(bv):
                    continue
                a_values.append(av)
                b_values.append(bv)
                deltas.append(bv - av)
                if av != 0:
                    pct.append(100.0 * (bv - av) / av)
            center, low, high, n = ci95_signed(deltas)
            out.append(
                {
                    "arm": arm,
                    "label": LABEL.get(arm, arm),
                    "metric": metric,
                    "model_a": a["model"],
                    "model_a_label": a["label"],
                    "model_b": b["model"],
                    "model_b_label": b["label"],
                    "n_paired_tasks": n,
                    "mean_model_a": mean(a_values),
                    "mean_model_b": mean(b_values),
                    "mean_delta_b_minus_a": center,
                    "ci95_low_delta": low,
                    "ci95_high_delta": high,
                    "mean_percent_change_b_vs_a": mean(pct),
                }
            )
    return out


def _write_multi_model_index(
    output: Path,
    descriptors: list[dict[str, str]],
    comparison: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    table_rows = []
    for row in comparison:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{format_value(float(row['mean_total_tokens']), 'tokens')}</td>"
            f"<td>{format_value(float(row['mean_cost_usd']), 'usd')}</td>"
            f"<td>{format_value(float(row['mean_wall_seconds']), 'seconds')}</td>"
            f"<td>{float(row['run_resolve_rate']):.0f}%</td>"
            "</tr>"
        )
    model_links = " ".join(
        f'<a href="models/{html.escape(model["key"])}/report-{{theme}}.html">'
        f'{html.escape(model["label"])}</a>'
        for model in descriptors
    )

    primary_stems = [
        ("model_treatment_total_tokens.vertical", "Total tokens"),
        ("model_treatment_cost_usd.vertical", "Cost"),
        ("model_treatment_wall_seconds.vertical", "Wall time"),
        ("model_treatment_api_calls.vertical", "API calls"),
        ("model_treatment_resolve_rate.vertical", "Resolve rate"),
    ]
    horizontal_stems = [
        ("model_treatment_total_tokens.horizontal", "Total tokens — horizontal"),
        ("model_treatment_cost_usd.horizontal", "Cost — horizontal"),
        ("model_treatment_wall_seconds.horizontal", "Wall time — horizontal"),
        ("model_treatment_api_calls.horizontal", "API calls — horizontal"),
        ("model_treatment_resolve_rate.horizontal", "Resolve rate — horizontal"),
    ]
    effect_stems = [
        ("model_treatment_interaction_total_tokens", "Token interaction"),
        ("model_treatment_interaction_cost_usd", "Cost interaction"),
        ("model_treatment_interaction_wall_seconds", "Wall-time interaction"),
        ("model_treatment_interaction_api_calls", "API-call interaction"),
        ("model_harness_effect_total_tokens", "Harness effect on tokens"),
        ("model_effect_total_tokens", "V4.1 vs V4 token effect"),
        ("model_treatment_cost_time", "Combined cost-time frontier"),
        ("model_treatment_time_budget", "Direct time budget"),
    ]
    multivariate_stems = [
        ("model_treatment_multivariate_pca", "Multivariate agent-behavior PCA"),
        ("model_treatment_multivariate_pca_loadings", "PCA feature loadings"),
        ("workflow_state_transitions", "Workflow transition matrices"),
    ]
    trajectory_stems = [
        ("trajectory_cumulative_tokens", "Cumulative token trajectory"),
        ("trajectory_context_tokens", "Context growth"),
        ("trajectory_api_wait_seconds", "Cumulative API wait"),
        ("trajectory_tool_execution_seconds", "Cumulative tool execution"),
        ("trajectory_api_calls", "API-call trajectory"),
        ("trajectory_tool_calls", "Tool-call trajectory"),
        ("workflow_first_edit_progress", "First edit"),
        ("workflow_first_execute_progress", "First execution"),
        ("workflow_consecutive_same_state_fraction", "Repeated-state fraction"),
        ("workflow_edit_to_execute_transitions", "Edit-to-execute transitions"),
    ]
    trace_specs = [
        ("trace_example_timeline", "Token and tool timeline for high-usage runs"),
        ("trace_token_tool_trajectories", "Token pools and computer activity over progress"),
        ("trace_response_vs_tool_work", "Per-response tokens and following tool work"),
    ]
    for theme in ("light", "dark"):
        if theme == "dark":
            bg, fg, muted, card, border = "#090e1a", "#f8fafc", "#cbd5e1", "#111827", "#334155"
        else:
            bg, fg, muted, card, border = "#f8fafc", "#111827", "#64748b", "#ffffff", "#e5e7eb"
        links = model_links.replace("{theme}", theme)

        def cards(specs: list[tuple[str, str]]) -> str:
            return "".join(
                f'<div class="card"><h3>{html.escape(title)}</h3>'
                f'<img src="{stem}.{theme}.svg" alt="{html.escape(title)}"></div>'
                for stem, title in specs
                if (output / f"{stem}.{theme}.svg").exists()
            )

        primary_cards = cards(primary_stems)
        horizontal_cards = cards(horizontal_stems)
        effect_cards = cards(effect_stems)
        multivariate_cards = cards(multivariate_stems)
        trajectory_cards = cards(trajectory_stems)
        trace_cards = "".join(
            f'<div class="card"><h3>{html.escape(title)}</h3>'
            f'<img src="{stem}.png" alt="{html.escape(title)}"></div>'
            for stem, title in trace_specs
            if (output / f"{stem}.png").exists()
        )
        body = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forge Bench multi-model report ({theme})</title>
<style>
body{{font-family:Inter,system-ui,Arial,sans-serif;background:{bg};color:{fg};max-width:1180px;margin:auto;padding:36px 24px 72px}}
p{{color:{muted};line-height:1.55}} a{{color:inherit;font-weight:650}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px;margin:18px 0 28px}}
.card{{background:{card};border:1px solid {border};border-radius:14px;padding:10px}}
.card h3{{font-size:15px;margin:6px 8px 2px}} .card img{{width:100%;display:block}}
table{{width:100%;border-collapse:collapse;background:{card};border:1px solid {border}}}
th,td{{padding:12px;border-bottom:1px solid {border};text-align:right}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
code{{font-size:.92em}}
</style>
<h1>Forge Bench multi-model experiment</h1>
<p>Design <code>{html.escape(str(meta.get("design_name", "unnamed")))}</code>.
One execution plan is randomized across model × treatment × task cells; model-specific
reports keep treatment inference separated by model.</p>
<p>Model reports: {links}</p>
<p>The primary inferential unit is the task. If multiple randomized blocks are
run, repeated stochastic observations are averaged within task × model ×
treatment before across-task confidence intervals and factorial analyses.</p>

<h2>Primary model × treatment comparison</h2>
<p>Each model is a true subpanel of the same figure and shares the same axis
scale. Treatment colors are identical across models. Open circles are the
independent task means.</p>
<div class="grid">{primary_cards}</div>

<h2>Horizontal versions</h2>
<div class="grid">{horizontal_cards}</div>

<h2>Factorial effects and timing</h2>
<p>The primary regression is <code>log(outcome) ~ task + model * treatment</code>.
A secondary decomposition treats Caveman and Ponytail as separate binary
factors and fits <code>model * caveman * ponytail</code>.</p>
<div class="grid">{effect_cards}</div>

<h2>Multivariate workflow phenotype</h2>
<p>PCA uses task-level agent behavior rather than individual API/tool events as
replicates. Treatment is encoded by color and model by marker; numbered tags
identify the larger model × treatment centers, with a separate loading chart
for PC1 and PC2. State-transition probabilities are normalized within task
before across-task averaging.</p>
<div class="grid">{multivariate_cards}</div>

<h2>Within-run trajectories</h2>
<p>Trace events are aligned to 0–100% normalized elapsed run progress.
Repeated blocks are averaged within task first; trajectory summaries are then
averaged across tasks. Individual turns are observations along a run, not
independent replicates.</p>
<div class="grid">{trajectory_cards}</div>

<h2>Token pools versus computer work</h2>
<p>Observer traces show cache-read input, other input, output, and reasoning
tokens alongside tool calls and execution time. The response/tool scatter links
each model response to the tool calls before its next request; it is descriptive,
not a causal estimate. Trace-observed activity is retained even when a run's
overall benchmark result is invalid; the trace CSVs mark result validity so
those observations can be inspected separately.</p>
<div class="grid">{trace_cards}</div>

<h2>Across-task summary</h2>
<table><thead><tr><th>Model</th><th>Treatment</th><th>Tokens</th><th>Cost</th><th>Time</th><th>Resolve</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p>Analysis tables include <code>model_pairwise_effects.csv</code>,
<code>model_harness_effects.csv</code>,
<code>model_treatment_regression_2x4.csv</code>,
<code>model_caveman_ponytail_regression_2x2x2.csv</code>,
the <code>multivariate_*</code> tables, and the <code>trajectory_*</code> /
<code>workflow_*</code>, and <code>trace_*</code> tables. Raw execution evidence remains in
<code>runs/</code>, <code>runs.json</code>, and <code>run_plan.csv</code>.</p>
"""
        (output / f"report-{theme}.html").write_text(body, encoding="utf-8")
    shutil.copyfile(output / "report-light.html", output / "report.html")


def write_experiment_reports(
    output: Path,
    results: list[Result],
    arms: list[str],
    tasks: list[str],
    meta: dict[str, Any],
    *,
    analysis_mode: str = "basic",
) -> None:
    """Write normal reports for one model or separated reports for multi-model designs."""
    descriptors = _model_descriptors(results, meta)
    if len(descriptors) <= 1:
        write_reports(
            output,
            results,
            arms,
            tasks,
            meta,
            analysis_mode=analysis_mode,
        )
        return

    raw = [
        asdict(result)
        for result in sorted(results, key=lambda item: item.run_index)
    ]
    write_csv(output / "runs.csv", raw)
    (output / "runs.json").write_text(
        json.dumps(raw, indent=2) + "\n",
        encoding="utf-8",
    )
    meta["analysis_mode"] = analysis_mode
    (output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    combined_task_rows: list[dict[str, Any]] = []
    combined_summary: list[dict[str, Any]] = []
    for descriptor in descriptors:
        model_results = [
            result for result in results if result.model == descriptor["model"]
        ]
        if not model_results:
            continue
        model_output = output / "models" / descriptor["key"]
        model_output.mkdir(parents=True, exist_ok=True)
        model_meta = dict(meta)
        model_meta["model"] = descriptor["model"]
        model_meta["model_key"] = descriptor["key"]
        model_meta["model_label"] = descriptor["label"]
        model_meta["upstream_provider"] = descriptor["upstream_provider"]
        write_reports(
            model_output,
            model_results,
            arms,
            tasks,
            model_meta,
            analysis_mode=analysis_mode,
        )
        per_task = task_summary(model_results, arms, tasks)
        summary = aggregate_summary(per_task, model_results, arms, tasks)
        for row in per_task:
            combined_task_rows.append(
                {
                    "model_key": descriptor["key"],
                    "model_label": descriptor["label"],
                    "model": descriptor["model"],
                    **row,
                }
            )
        for row in summary:
            combined_summary.append(
                {
                    "model_key": descriptor["key"],
                    "model_label": descriptor["label"],
                    "model": descriptor["model"],
                    **row,
                }
            )

    write_csv(output / "model_task_summary.csv", combined_task_rows)
    write_csv(output / "model_treatment_summary.csv", combined_summary)

    # Root-level multi-model analysis is primary. Per-model reports above are
    # preserved as drill-down analyses.
    write_multi_model_analysis(
        output,
        results,
        combined_task_rows,
        combined_summary,
        descriptors,
    )
    _write_multi_model_index(output, descriptors, combined_summary, meta)


def create_reanalysis_output_dir(
    source: Path,
    *,
    analysis_mode: str,
) -> Path:
    """Create a fresh timestamped reanalysis directory inside a benchmark run."""
    root = source / "reanalysis"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = root / f"{stamp}-{analysis_mode}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base.name}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def reanalyze_output(
    source: Path,
    *,
    analysis_mode: str = "advanced",
) -> Path:
    source = source.expanduser().resolve()
    runs_file = source / "runs.json"
    if not runs_file.is_file():
        raise FileNotFoundError(f"Missing Forge Bench runs.json: {runs_file}")

    payload = json.loads(runs_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("runs.json must contain a list of run records")

    result_fields = {field.name for field in fields(Result)}
    defaults = {
        "evaluation_seconds": 0.0,
        "error": "",
        "base_arm": None,
        "budget_warning_ratio": None,
        "tool_calls": None,
        "api_wait_seconds": None,
        "tool_execution_seconds": None,
        "terminal_execution_seconds": None,
        "unattributed_wall_seconds": None,
        "api_duration_mean_seconds": None,
        "api_duration_p95_seconds": None,
        "ttft_mean_seconds": None,
        "timing_event_count": 0,
        "skill_lifecycle_event_count": 0,
        "trace_exported": False,
    }
    results: list[Result] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        normalized = dict(defaults)
        normalized.update({key: value for key, value in row.items() if key in result_fields})
        missing = [
            name for name in result_fields
            if name not in normalized and name != "error"
        ]
        if missing:
            raise ValueError(
                "runs.json predates required Forge Bench fields: " + ", ".join(sorted(missing))
            )
        results.append(Result(**normalized))

    if not results:
        raise ValueError("runs.json contains no usable run records")

    metadata_file = source / "metadata.json"
    meta = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.is_file()
        else {}
    )
    arms = _ordered_arms(result.arm for result in results)
    tasks = list(dict.fromkeys(result.task for result in results))
    meta = dict(meta)
    meta.setdefault("model", results[0].model)
    meta.setdefault("upstream_provider", results[0].upstream_provider)
    meta.setdefault("seed", "unknown")
    meta["reanalyzed"] = True
    meta["reanalyzed_from"] = str(source)
    meta["reanalyzed_at"] = datetime.now(timezone.utc).isoformat()

    output = create_reanalysis_output_dir(
        source,
        analysis_mode=analysis_mode,
    )
    write_experiment_reports(
        output,
        results,
        arms,
        tasks,
        meta,
        analysis_mode=analysis_mode,
    )
    return output
