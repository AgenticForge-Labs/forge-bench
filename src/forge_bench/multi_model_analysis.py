from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .advanced_analysis import TREATMENT_ORDER, treatment_colors
from .config import LABEL, METRICS, T975, Result


PRIMARY_METRICS = ("total_tokens", "cost_usd", "wall_seconds", "api_calls")
MULTIVARIATE_FEATURES = (
    "total_tokens",
    "wall_seconds",
    "api_calls",
    "tool_calls",
    "api_wait_fraction",
    "tool_execution_fraction",
    "diff_lines",
)
PROGRESS_GRID = tuple(range(0, 101, 5))
STATE_ORDER = ("reason", "inspect", "search", "edit", "execute", "skill", "other")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if _finite(value)]
    return float(np.mean(vals)) if vals else math.nan


def _ci95(values: Iterable[float], *, signed: bool = True) -> tuple[float, float, float, int]:
    vals = [float(value) for value in values if _finite(value)]
    if not vals:
        return math.nan, math.nan, math.nan, 0
    center = float(np.mean(vals))
    if len(vals) == 1:
        return center, center, center, 1
    sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    crit = T975.get(len(vals) - 1, 1.959964)
    low, high = center - crit * sem, center + crit * sem
    if not signed:
        low = max(0.0, low)
    return center, low, high, len(vals)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_arms(rows: Iterable[dict[str, Any]]) -> list[str]:
    values = list(dict.fromkeys(str(row["arm"]) for row in rows))
    return [arm for arm in TREATMENT_ORDER if arm in values] + sorted(
        arm for arm in values if arm not in TREATMENT_ORDER
    )


def _theme(theme: str) -> dict[str, str]:
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
    colors = _theme(theme)
    fig.patch.set_facecolor(colors["figure"])
    ax.set_facecolor(colors["axes"])
    ax.tick_params(colors=colors["text"], labelsize=11)
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["edge"])
    ax.grid(alpha=0.28, linewidth=1.0, color=colors["grid"])
    ax.set_axisbelow(True)


def _save(fig: Any, output: Path, stem: str, theme: str) -> None:
    fig.savefig(
        output / f"{stem}.{theme}.png",
        dpi=240,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        output / f"{stem}.{theme}.svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _metric_title(metric: str) -> tuple[str, str]:
    if metric in METRICS:
        return METRICS[metric]
    extra = {
        "tool_calls": ("Tool calls", "count"),
        "api_wait_seconds": ("API wait", "seconds"),
        "tool_execution_seconds": ("Tool execution", "seconds"),
        "unattributed_wall_seconds": ("Unattributed wall time", "seconds"),
        "api_wait_fraction": ("API wait fraction", "fraction"),
        "tool_execution_fraction": ("Tool execution fraction", "fraction"),
    }
    return extra.get(metric, (metric.replace("_", " ").title(), "raw"))


def _format(value: float, kind: str) -> str:
    if not math.isfinite(value):
        return "n/a"
    if kind == "usd":
        return "$" + f"{value:.4f}"
    if kind == "tokens":
        return f"{value / 1000:.1f}k" if abs(value) >= 1000 else f"{value:.0f}"
    if kind == "seconds":
        return f"{value:.1f}s"
    if kind == "fraction":
        return f"{100 * value:.0f}%"
    return f"{value:.1f}"


def _descriptor_order(descriptors: list[dict[str, str]]) -> dict[str, int]:
    return {str(row["model"]): index for index, row in enumerate(descriptors)}


def plot_faceted_metric(
    output: Path,
    task_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
    metric: str,
) -> None:
    """True shared-scale model panels in both vertical- and horizontal-bar styles."""
    title, kind = _metric_title(metric)
    arms = _ordered_arms(summary_rows)
    if not arms or len(descriptors) < 2:
        return
    values = [
        float(row.get(f"mean_{metric}", math.nan))
        for row in summary_rows
        if _finite(row.get(f"mean_{metric}"))
    ]
    highs = [
        float(row.get(f"ci95_high_{metric}", math.nan))
        for row in summary_rows
        if _finite(row.get(f"ci95_high_{metric}"))
    ]
    if not values:
        return
    axis_max = max(values + highs) * 1.12 if max(values + highs) > 0 else 1.0

    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)

        fig, axes = plt.subplots(
            1, len(descriptors), figsize=(7.0 * len(descriptors), 7.0),
            sharey=True, squeeze=False,
        )
        for model_i, (ax, descriptor) in enumerate(zip(axes[0], descriptors)):
            _style_axes(fig, ax, theme)
            model = descriptor["model"]
            model_summary = {
                str(row["arm"]): row
                for row in summary_rows if row.get("model") == model
            }
            x = np.arange(len(arms))
            means = [
                float(model_summary.get(arm, {}).get(f"mean_{metric}", math.nan))
                for arm in arms
            ]
            low = [
                float(model_summary.get(arm, {}).get(f"ci95_low_{metric}", math.nan))
                for arm in arms
            ]
            high = [
                float(model_summary.get(arm, {}).get(f"ci95_high_{metric}", math.nan))
                for arm in arms
            ]
            lower = [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(means, low)]
            upper = [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(means, high)]
            bars = ax.bar(
                x, means, color=[colors[arm] for arm in arms],
                yerr=[lower, upper], capsize=6, edgecolor=palette["edge"],
                linewidth=1.0,
                error_kw={"ecolor": palette["text"], "elinewidth": 1.7},
                zorder=2,
            )
            task_order = sorted({
                str(row["task"]) for row in task_rows if row.get("model") == model
            })
            offsets = np.linspace(-0.10, 0.10, max(1, len(task_order)))
            offset_by_task = {task: float(offsets[i]) for i, task in enumerate(task_order)}
            for arm_i, arm in enumerate(arms):
                points = [
                    row for row in task_rows
                    if row.get("model") == model and row.get("arm") == arm
                    and row.get("valid_runs", 0) > 0 and _finite(row.get(metric))
                ]
                ax.scatter(
                    [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                    [float(row[metric]) for row in points],
                    s=32, facecolors="none", edgecolors=palette["text"],
                    linewidths=1.0, zorder=4,
                )
            for bar, value in zip(bars, means):
                if _finite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, float(value),
                        _format(float(value), kind), ha="center", va="bottom",
                        fontsize=9.5, color=palette["text"],
                    )
            ax.set_title(str(descriptor["label"]), fontsize=16, fontweight="bold")
            ax.set_xticks(x, [LABEL.get(arm, arm) for arm in arms], rotation=18, ha="right")
            ax.set_ylim(0, axis_max)
            if model_i == 0:
                ax.set_ylabel(title, fontsize=13)
        fig.suptitle(
            f"{title}: model × treatment", fontsize=20, fontweight="bold",
            color=palette["text"],
        )
        fig.text(
            0.5, 0.01,
            "Bars are means across independent task means; open circles are task means; error bars are 95% Student-t CIs across tasks. Panels share one y scale.",
            ha="center", fontsize=10.5, color=palette["muted"],
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        _save(fig, output, f"model_treatment_{metric}.vertical", theme)

        fig, axes = plt.subplots(
            len(descriptors), 1, figsize=(11.0, 4.3 * len(descriptors)),
            sharex=True, squeeze=False,
        )
        for model_i, (ax, descriptor) in enumerate(zip([row[0] for row in axes], descriptors)):
            _style_axes(fig, ax, theme)
            model = descriptor["model"]
            model_summary = {
                str(row["arm"]): row
                for row in summary_rows if row.get("model") == model
            }
            y = np.arange(len(arms))
            means = [float(model_summary.get(arm, {}).get(f"mean_{metric}", math.nan)) for arm in arms]
            low = [float(model_summary.get(arm, {}).get(f"ci95_low_{metric}", math.nan)) for arm in arms]
            high = [float(model_summary.get(arm, {}).get(f"ci95_high_{metric}", math.nan)) for arm in arms]
            lower = [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(means, low)]
            upper = [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(means, high)]
            bars = ax.barh(
                y, means, color=[colors[arm] for arm in arms],
                xerr=[lower, upper], capsize=5, edgecolor=palette["edge"],
                linewidth=1.0,
                error_kw={"ecolor": palette["text"], "elinewidth": 1.7},
                zorder=2,
            )
            task_order = sorted({
                str(row["task"]) for row in task_rows if row.get("model") == model
            })
            offsets = np.linspace(-0.10, 0.10, max(1, len(task_order)))
            offset_by_task = {task: float(offsets[i]) for i, task in enumerate(task_order)}
            for arm_i, arm in enumerate(arms):
                points = [
                    row for row in task_rows
                    if row.get("model") == model and row.get("arm") == arm
                    and row.get("valid_runs", 0) > 0 and _finite(row.get(metric))
                ]
                ax.scatter(
                    [float(row[metric]) for row in points],
                    [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                    s=32, facecolors="none", edgecolors=palette["text"],
                    linewidths=1.0, zorder=4,
                )
            for bar, value in zip(bars, means):
                if _finite(value):
                    ax.text(
                        float(value), bar.get_y() + bar.get_height() / 2,
                        " " + _format(float(value), kind), ha="left", va="center",
                        fontsize=9.5, color=palette["text"],
                    )
            ax.set_title(str(descriptor["label"]), fontsize=16, fontweight="bold")
            ax.set_yticks(y, [LABEL.get(arm, arm) for arm in arms])
            ax.set_xlim(0, axis_max)
            if model_i == len(descriptors) - 1:
                ax.set_xlabel(title, fontsize=13)
        fig.suptitle(
            f"{title}: model × treatment", fontsize=20, fontweight="bold",
            color=palette["text"],
        )
        fig.text(
            0.5, 0.01, "Same data as the vertical version. Panels share one x scale.",
            ha="center", fontsize=10.5, color=palette["muted"],
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        _save(fig, output, f"model_treatment_{metric}.horizontal", theme)


def plot_faceted_resolve_rate(
    output: Path,
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    """Plot task-level resolve means/CIs so repeated runs never inflate n."""
    arms = _ordered_arms(task_rows)
    condition: dict[tuple[str, str], dict[str, Any]] = {}
    for descriptor in descriptors:
        for arm in arms:
            points = [
                row for row in task_rows
                if row.get("model") == descriptor["model"]
                and row.get("arm") == arm
                and row.get("valid_runs", 0) > 0
                and _finite(row.get("resolve_rate"))
            ]
            center, low, high, n = _ci95(
                (row["resolve_rate"] for row in points),
                signed=False,
            )
            condition[(descriptor["model"], arm)] = {
                "mean": center, "low": low, "high": high, "n": n, "points": points,
            }

    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        for orientation in ("vertical", "horizontal"):
            if orientation == "vertical":
                fig, axes = plt.subplots(
                    1, len(descriptors), figsize=(7 * len(descriptors), 6.5),
                    sharey=True, squeeze=False,
                )
                axes_list = list(axes[0])
            else:
                fig, axes = plt.subplots(
                    len(descriptors), 1, figsize=(10.5, 4 * len(descriptors)),
                    sharex=True, squeeze=False,
                )
                axes_list = [row[0] for row in axes]

            for index, (ax, descriptor) in enumerate(zip(axes_list, descriptors)):
                _style_axes(fig, ax, theme)
                values = [float(condition[(descriptor["model"], arm)]["mean"]) for arm in arms]
                lows = [float(condition[(descriptor["model"], arm)]["low"]) for arm in arms]
                highs = [float(condition[(descriptor["model"], arm)]["high"]) for arm in arms]
                lower = [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(values, lows)]
                upper = [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(values, highs)]
                pos = np.arange(len(arms))
                task_order = sorted({
                    str(row["task"])
                    for row in task_rows if row.get("model") == descriptor["model"]
                })
                offsets = np.linspace(-0.10, 0.10, max(1, len(task_order)))
                offset_by_task = {task: float(offsets[i]) for i, task in enumerate(task_order)}

                if orientation == "vertical":
                    ax.bar(
                        pos, values, color=[colors[a] for a in arms],
                        yerr=[lower, upper], capsize=6,
                        edgecolor=palette["edge"],
                        error_kw={"ecolor": palette["text"], "elinewidth": 1.5},
                    )
                    for arm_i, arm in enumerate(arms):
                        points = condition[(descriptor["model"], arm)]["points"]
                        ax.scatter(
                            [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                            [float(row["resolve_rate"]) for row in points],
                            s=32, facecolors="none", edgecolors=palette["text"],
                            linewidths=1.0, zorder=4,
                        )
                    ax.set_xticks(pos, [LABEL.get(a, a) for a in arms], rotation=18, ha="right")
                    ax.set_ylim(0, 105)
                    if index == 0:
                        ax.set_ylabel("Task resolve rate (%)")
                else:
                    ax.barh(
                        pos, values, color=[colors[a] for a in arms],
                        xerr=[lower, upper], capsize=5,
                        edgecolor=palette["edge"],
                        error_kw={"ecolor": palette["text"], "elinewidth": 1.5},
                    )
                    for arm_i, arm in enumerate(arms):
                        points = condition[(descriptor["model"], arm)]["points"]
                        ax.scatter(
                            [float(row["resolve_rate"]) for row in points],
                            [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                            s=32, facecolors="none", edgecolors=palette["text"],
                            linewidths=1.0, zorder=4,
                        )
                    ax.set_yticks(pos, [LABEL.get(a, a) for a in arms])
                    ax.set_xlim(0, 105)
                    if index == len(descriptors) - 1:
                        ax.set_xlabel("Task resolve rate (%)")
                ax.set_title(str(descriptor["label"]), fontsize=15, fontweight="bold")

            fig.suptitle(
                "SWE-bench resolve rate: model × treatment",
                fontsize=19, fontweight="bold", color=palette["text"],
            )
            fig.text(
                0.5, 0.01,
                "Bars and 95% CIs are across independent task-level resolve rates; repeated blocks are averaged within task first.",
                ha="center", fontsize=9.8, color=palette["muted"],
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))
            _save(fig, output, f"model_treatment_resolve_rate.{orientation}", theme)

def model_pairwise_effects(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(descriptors) != 2:
        return []
    a, b = descriptors
    rows: list[dict[str, Any]] = []
    for arm in _ordered_arms(task_rows):
        for metric in PRIMARY_METRICS:
            a_by_task = {
                str(row["task"]): row
                for row in task_rows
                if row.get("model") == a["model"] and row.get("arm") == arm
                and row.get("valid_runs", 0) > 0 and _finite(row.get(metric))
            }
            b_by_task = {
                str(row["task"]): row
                for row in task_rows
                if row.get("model") == b["model"] and row.get("arm") == arm
                and row.get("valid_runs", 0) > 0 and _finite(row.get(metric))
            }
            deltas: list[float] = []
            percent: list[float] = []
            tasks: list[str] = []
            for task in sorted(set(a_by_task) & set(b_by_task)):
                av = float(a_by_task[task][metric])
                bv = float(b_by_task[task][metric])
                deltas.append(bv - av)
                if av != 0:
                    percent.append(100.0 * (bv / av - 1.0))
                tasks.append(task)
            center, low, high, n = _ci95(deltas, signed=True)
            pcenter, plow, phigh, _ = _ci95(percent, signed=True)
            rows.append(
                {
                    "arm": arm,
                    "label": LABEL.get(arm, arm),
                    "metric": metric,
                    "model_a": a["model"],
                    "model_a_label": a["label"],
                    "model_b": b["model"],
                    "model_b_label": b["label"],
                    "n_paired_tasks": n,
                    "mean_delta_b_minus_a": center,
                    "ci95_low_delta": low,
                    "ci95_high_delta": high,
                    "mean_percent_change_b_vs_a": pcenter,
                    "ci95_low_percent_change": plow,
                    "ci95_high_percent_change": phigh,
                    "tasks": ";".join(tasks),
                }
            )
    return rows


def harness_effects_within_model(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arms = [arm for arm in _ordered_arms(task_rows) if arm != "baseline"]
    for descriptor in descriptors:
        model = descriptor["model"]
        baseline = {
            str(row["task"]): row
            for row in task_rows
            if row.get("model") == model and row.get("arm") == "baseline"
            and row.get("valid_runs", 0) > 0
        }
        for arm in arms:
            for metric in PRIMARY_METRICS:
                effects: list[float] = []
                for row in task_rows:
                    if (
                        row.get("model") != model or row.get("arm") != arm
                        or row.get("valid_runs", 0) <= 0 or not _finite(row.get(metric))
                    ):
                        continue
                    base = baseline.get(str(row["task"]))
                    if not base or not _finite(base.get(metric)) or float(base[metric]) == 0:
                        continue
                    effects.append(100.0 * (float(row[metric]) / float(base[metric]) - 1.0))
                center, low, high, n = _ci95(effects, signed=True)
                rows.append(
                    {
                        "model": model,
                        "model_label": descriptor["label"],
                        "arm": arm,
                        "label": LABEL.get(arm, arm),
                        "metric": metric,
                        "n_tasks": n,
                        "mean_percent_change_vs_own_baseline": center,
                        "ci95_low_percent_change": low,
                        "ci95_high_percent_change": high,
                    }
                )
    return rows


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    beta = np.linalg.pinv(X) @ y
    residual = y - X @ beta
    rank = int(np.linalg.matrix_rank(X))
    df = max(1, len(y) - rank)
    sigma2 = float(residual @ residual) / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    return beta, se, df


def factorial_2x4_regression(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(descriptors) != 2:
        return []
    rows = [row for row in task_rows if row.get("valid_runs", 0) > 0]
    tasks = sorted({str(row["task"]) for row in rows})
    arms = _ordered_arms(rows)
    if "baseline" not in arms or len(tasks) < 2:
        return []
    comparison_arms = [arm for arm in arms if arm != "baseline"]
    model_b = descriptors[1]["model"]
    out: list[dict[str, Any]] = []
    names = (
        ["intercept"]
        + [f"task:{task}" for task in tasks[1:]]
        + ["model_b"]
        + [f"arm:{arm}" for arm in comparison_arms]
        + [f"model_b:arm:{arm}" for arm in comparison_arms]
    )
    for metric in PRIMARY_METRICS:
        usable = [row for row in rows if _finite(row.get(metric)) and float(row[metric]) > 0]
        if len(usable) <= len(names):
            continue
        X: list[list[float]] = []
        y: list[float] = []
        for row in usable:
            mb = 1.0 if row.get("model") == model_b else 0.0
            arm = str(row["arm"])
            arm_flags = [1.0 if arm == candidate else 0.0 for candidate in comparison_arms]
            X.append(
                [1.0]
                + [1.0 if str(row["task"]) == task else 0.0 for task in tasks[1:]]
                + [mb] + arm_flags + [mb * value for value in arm_flags]
            )
            y.append(math.log(float(row[metric])))
        beta, se, df = _fit_ols(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        crit = T975.get(df, 1.959964)
        for term in ["model_b", *[f"arm:{a}" for a in comparison_arms], *[f"model_b:arm:{a}" for a in comparison_arms]]:
            index = names.index(term)
            b = float(beta[index])
            s = float(se[index])
            out.append(
                {
                    "formula": "log(metric) ~ task fixed effects + model * treatment",
                    "metric": metric,
                    "term": term,
                    "n_observations": len(y),
                    "n_tasks": len(tasks),
                    "df_residual": df,
                    "beta_log": b,
                    "se_log": s,
                    "estimated_percent_change": 100.0 * (math.exp(b) - 1.0),
                    "ci95_low_percent_change": 100.0 * (math.exp(b - crit * s) - 1.0),
                    "ci95_high_percent_change": 100.0 * (math.exp(b + crit * s) - 1.0),
                }
            )
    return out


def factorial_2x2x2_regression(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(descriptors) != 2:
        return []
    required = {"baseline", "caveman", "ponytail", "caveman_ponytail"}
    rows = [
        row for row in task_rows
        if row.get("valid_runs", 0) > 0 and row.get("arm") in required
    ]
    if not required.issubset({str(row["arm"]) for row in rows}):
        return []
    tasks = sorted({str(row["task"]) for row in rows})
    model_b = descriptors[1]["model"]
    terms = (
        "model", "caveman", "ponytail", "model:caveman", "model:ponytail",
        "caveman:ponytail", "model:caveman:ponytail",
    )
    names = ["intercept"] + [f"task:{task}" for task in tasks[1:]] + list(terms)
    out: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        usable = [row for row in rows if _finite(row.get(metric)) and float(row[metric]) > 0]
        if len(usable) <= len(names):
            continue
        X: list[list[float]] = []
        y: list[float] = []
        for row in usable:
            arm = str(row["arm"])
            m = 1.0 if row.get("model") == model_b else 0.0
            c = 1.0 if arm in {"caveman", "caveman_ponytail"} else 0.0
            p = 1.0 if arm in {"ponytail", "caveman_ponytail"} else 0.0
            X.append(
                [1.0]
                + [1.0 if str(row["task"]) == task else 0.0 for task in tasks[1:]]
                + [m, c, p, m * c, m * p, c * p, m * c * p]
            )
            y.append(math.log(float(row[metric])))
        beta, se, df = _fit_ols(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        crit = T975.get(df, 1.959964)
        for term in terms:
            index = names.index(term)
            b = float(beta[index])
            s = float(se[index])
            out.append(
                {
                    "formula": "log(metric) ~ task fixed effects + model * caveman * ponytail",
                    "metric": metric,
                    "term": term,
                    "n_observations": len(y),
                    "n_tasks": len(tasks),
                    "df_residual": df,
                    "beta_log": b,
                    "se_log": s,
                    "estimated_percent_change": 100.0 * (math.exp(b) - 1.0),
                    "ci95_low_percent_change": 100.0 * (math.exp(b - crit * s) - 1.0),
                    "ci95_high_percent_change": 100.0 * (math.exp(b + crit * s) - 1.0),
                }
            )
    return out


def plot_interaction(
    output: Path,
    summary_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
    metric: str,
) -> None:
    title, _ = _metric_title(metric)
    arms = _ordered_arms(summary_rows)
    if len(descriptors) != 2 or not arms:
        return
    for theme in ("light", "dark"):
        palette = _theme(theme)
        colors = treatment_colors(arms, theme)
        fig, ax = plt.subplots(figsize=(10.8, 6.8))
        _style_axes(fig, ax, theme)
        x = np.arange(len(arms))
        markers = ("o", "s")
        linestyles = ("-", "--")
        for model_i, descriptor in enumerate(descriptors):
            model_rows = {
                str(row["arm"]): row
                for row in summary_rows if row.get("model") == descriptor["model"]
            }
            values = [float(model_rows.get(arm, {}).get(f"mean_{metric}", math.nan)) for arm in arms]
            line_color = palette["text"] if model_i == 0 else palette["muted"]
            ax.plot(
                x, values, marker=markers[model_i], linestyle=linestyles[model_i],
                linewidth=2.4, markersize=9, color=line_color,
                label=str(descriptor["label"]), zorder=2,
            )
            for xi, arm, value in zip(x, arms, values):
                if _finite(value):
                    ax.scatter(
                        [xi], [value], s=95, marker=markers[model_i],
                        color=colors[arm], edgecolors=palette["text"],
                        linewidths=1.0, zorder=4,
                    )
        ax.set_xticks(x, [LABEL.get(arm, arm) for arm in arms])
        ax.set_ylabel(title)
        ax.set_title(f"{title}: model × treatment interaction", fontsize=18, fontweight="bold")
        ax.legend(title="Model", frameon=True)
        fig.tight_layout()
        _save(fig, output, f"model_treatment_interaction_{metric}", theme)


def plot_harness_effects(
    output: Path,
    effects: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
    metric: str,
) -> None:
    rows = [
        row for row in effects
        if row.get("metric") == metric and _finite(row.get("mean_percent_change_vs_own_baseline"))
    ]
    if not rows:
        return
    arms = [arm for arm in _ordered_arms(rows) if arm != "baseline"]
    extent = max(
        [abs(float(row["ci95_low_percent_change"])) for row in rows if _finite(row.get("ci95_low_percent_change"))]
        + [abs(float(row["ci95_high_percent_change"])) for row in rows if _finite(row.get("ci95_high_percent_change"))]
        + [1.0]
    ) * 1.15
    title, _ = _metric_title(metric)
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, axes = plt.subplots(
            1, len(descriptors), figsize=(7 * len(descriptors), 6.2),
            sharey=True, squeeze=False,
        )
        for ax, descriptor in zip(axes[0], descriptors):
            _style_axes(fig, ax, theme)
            subset = {
                str(row["arm"]): row for row in rows if row.get("model") == descriptor["model"]
            }
            x = np.arange(len(arms))
            means = [float(subset.get(arm, {}).get("mean_percent_change_vs_own_baseline", math.nan)) for arm in arms]
            low = [float(subset.get(arm, {}).get("ci95_low_percent_change", math.nan)) for arm in arms]
            high = [float(subset.get(arm, {}).get("ci95_high_percent_change", math.nan)) for arm in arms]
            ax.bar(
                x, means, color=[colors[a] for a in arms],
                yerr=[
                    [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(means, low)],
                    [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(means, high)],
                ],
                capsize=6, edgecolor=palette["edge"],
                error_kw={"ecolor": palette["text"], "elinewidth": 1.6},
            )
            ax.axhline(0, color=palette["text"], linewidth=1.0, alpha=0.7)
            ax.set_ylim(-extent, extent)
            ax.set_xticks(x, [LABEL.get(a, a) for a in arms], rotation=18, ha="right")
            ax.set_title(str(descriptor["label"]), fontsize=15, fontweight="bold")
        axes[0][0].set_ylabel("Percent change vs own baseline")
        fig.suptitle(f"Harness effect on {title}", fontsize=19, fontweight="bold", color=palette["text"])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, output, f"model_harness_effect_{metric}", theme)


def plot_model_effects(
    output: Path,
    pairwise: list[dict[str, Any]],
    metric: str,
) -> None:
    rows = [
        row for row in pairwise
        if row.get("metric") == metric and _finite(row.get("mean_percent_change_b_vs_a"))
    ]
    if not rows:
        return
    arms = _ordered_arms(rows)
    title, _ = _metric_title(metric)
    extent = max(
        [abs(float(row["ci95_low_percent_change"])) for row in rows if _finite(row.get("ci95_low_percent_change"))]
        + [abs(float(row["ci95_high_percent_change"])) for row in rows if _finite(row.get("ci95_high_percent_change"))]
        + [1.0]
    ) * 1.15
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, ax = plt.subplots(figsize=(10.5, 6.4))
        _style_axes(fig, ax, theme)
        by_arm = {str(row["arm"]): row for row in rows}
        x = np.arange(len(arms))
        means = [float(by_arm[a]["mean_percent_change_b_vs_a"]) for a in arms]
        low = [float(by_arm[a]["ci95_low_percent_change"]) for a in arms]
        high = [float(by_arm[a]["ci95_high_percent_change"]) for a in arms]
        ax.bar(
            x, means, color=[colors[a] for a in arms],
            yerr=[
                [max(0.0, m - l) for m, l in zip(means, low)],
                [max(0.0, h - m) for m, h in zip(means, high)],
            ],
            capsize=6, edgecolor=palette["edge"],
            error_kw={"ecolor": palette["text"], "elinewidth": 1.6},
        )
        ax.axhline(0, color=palette["text"], linewidth=1.0, alpha=0.7)
        ax.set_ylim(-extent, extent)
        ax.set_xticks(x, [LABEL.get(a, a) for a in arms])
        ax.set_ylabel("V4.1 percent change vs V4")
        ax.set_title(
            f"Model effect on {title}, paired within task and treatment",
            fontsize=18, fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, output, f"model_effect_{metric}", theme)


def combined_cost_time_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in summary_rows
        if _finite(row.get("mean_cost_usd")) and _finite(row.get("mean_wall_seconds"))
    ]
    for row in rows:
        cost = float(row["mean_cost_usd"])
        wall = float(row["mean_wall_seconds"])
        row["pareto_efficient"] = not any(
            float(other["mean_cost_usd"]) <= cost
            and float(other["mean_wall_seconds"]) <= wall
            and (
                float(other["mean_cost_usd"]) < cost
                or float(other["mean_wall_seconds"]) < wall
            )
            for other in rows if other is not row
        )
    return rows


def plot_combined_cost_time(
    output: Path,
    rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    if not rows:
        return
    arms = _ordered_arms(rows)
    markers = ["o", "s", "^", "D"]
    model_index = _descriptor_order(descriptors)
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        _style_axes(fig, ax, theme)
        for row in rows:
            mindex = model_index.get(str(row["model"]), 0)
            ax.scatter(
                float(row["mean_wall_seconds"]), float(row["mean_cost_usd"]),
                s=155 if row.get("pareto_efficient") else 95,
                marker=markers[mindex % len(markers)],
                color=colors[str(row["arm"])], edgecolors=palette["text"],
                linewidths=1.2 if row.get("pareto_efficient") else 0.8, zorder=3,
            )
            ax.annotate(
                f"{row['model_label']}\\n{LABEL.get(str(row['arm']), str(row['arm']))}",
                (float(row["mean_wall_seconds"]), float(row["mean_cost_usd"])),
                xytext=(7, 6), textcoords="offset points",
                fontsize=8.5, color=palette["text"],
            )
        ax.set_xlabel("Mean wall time (seconds)")
        ax.set_ylabel("Mean cost (USD)")
        ax.set_title("Combined model × treatment cost-time frontier", fontsize=18, fontweight="bold")
        fig.tight_layout()
        _save(fig, output, "model_treatment_cost_time", theme)


def _feature_matrix(task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    usable_features = [
        feature
        for feature in MULTIVARIATE_FEATURES
        if any(_finite(row.get(feature)) for row in task_rows)
    ]
    rows = [
        row for row in task_rows
        if row.get("valid_runs", 0) > 0
        and all(_finite(row.get(feature)) for feature in usable_features)
    ]
    if len(rows) < 4 or len(usable_features) < 2:
        return [], [], np.empty((0, 0))
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for feature in usable_features:
            value = float(row[feature])
            values.append(value if feature.endswith("_fraction") else math.log1p(max(0.0, value)))
        matrix.append(values)
    data = np.asarray(matrix, dtype=float)
    std = data.std(axis=0, ddof=1)
    keep = std > 1e-12
    data = data[:, keep]
    features = [feature for feature, flag in zip(usable_features, keep) if flag]
    if data.shape[1] < 2:
        return [], [], np.empty((0, 0))
    z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    return rows, features, z


def multivariate_analysis(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, features, z = _feature_matrix(task_rows)
    if z.size == 0:
        return [], [], [], [], []
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    explained = singular ** 2
    explained = explained / explained.sum()
    score_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score_rows.append(
            {
                "model": row["model"], "model_key": row["model_key"],
                "model_label": row["model_label"], "arm": row["arm"],
                "label": row["label"], "task": row["task"],
                "pc1": float(scores[index, 0]),
                "pc2": float(scores[index, 1]) if scores.shape[1] > 1 else 0.0,
            }
        )
    loading_rows = [
        {
            "feature": feature,
            "pc1_loading": float(vt[0, i]),
            "pc2_loading": float(vt[1, i]) if vt.shape[0] > 1 else 0.0,
        }
        for i, feature in enumerate(features)
    ]
    centroid_rows: list[dict[str, Any]] = []
    centroid_vectors: dict[tuple[str, str], np.ndarray] = {}
    model_order = _descriptor_order(descriptors)
    for descriptor in descriptors:
        for arm in _ordered_arms(rows):
            indices = [
                i for i, row in enumerate(rows)
                if row.get("model") == descriptor["model"] and row.get("arm") == arm
            ]
            if not indices:
                continue
            centroid = z[indices].mean(axis=0)
            centroid_vectors[(descriptor["model"], arm)] = centroid
            centroid_rows.append(
                {
                    "model": descriptor["model"], "model_key": descriptor["key"],
                    "model_label": descriptor["label"], "arm": arm,
                    "label": LABEL.get(arm, arm), "n_tasks": len(indices),
                    "pc1": float(scores[indices, 0].mean()),
                    "pc2": float(scores[indices, 1].mean()) if scores.shape[1] > 1 else 0.0,
                    "distance_from_global_centroid": float(np.linalg.norm(centroid)),
                    "model_order": model_order.get(descriptor["model"], 0),
                }
            )
    similarity_rows: list[dict[str, Any]] = []
    if len(descriptors) == 2:
        a, b = descriptors
        a_base = centroid_vectors.get((a["model"], "baseline"))
        b_base = centroid_vectors.get((b["model"], "baseline"))
        if a_base is not None and b_base is not None:
            model_shift = b_base - a_base
            model_norm = float(np.linalg.norm(model_shift))
            for descriptor in descriptors:
                base = centroid_vectors.get((descriptor["model"], "baseline"))
                if base is None:
                    continue
                for arm in [arm for arm in _ordered_arms(rows) if arm != "baseline"]:
                    target = centroid_vectors.get((descriptor["model"], arm))
                    if target is None:
                        continue
                    harness_shift = target - base
                    harness_norm = float(np.linalg.norm(harness_shift))
                    cosine = (
                        float(np.dot(model_shift, harness_shift) / (model_norm * harness_norm))
                        if model_norm > 0 and harness_norm > 0 else math.nan
                    )
                    similarity_rows.append(
                        {
                            "reference_vector": f"{a['label']} baseline -> {b['label']} baseline",
                            "model": descriptor["model"], "model_label": descriptor["label"],
                            "arm": arm, "label": LABEL.get(arm, arm),
                            "model_shift_norm": model_norm,
                            "harness_shift_norm": harness_norm,
                            "cosine_similarity_to_model_shift": cosine,
                            "angle_degrees": (
                                math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                                if _finite(cosine) else math.nan
                            ),
                        }
                    )
    return score_rows, loading_rows, [float(x) for x in explained], centroid_rows, similarity_rows



def multivariate_factorial_coefficients(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Fit one task-blocked 2×4 design to the standardized multivariate outcomes.

    Coefficients are descriptive standardized effect vectors. With only five
    independent tasks, this intentionally avoids presenting multivariate
    significance tests as if the 40 condition rows were independent replicates.
    """
    rows, features, z = _feature_matrix(task_rows)
    if z.size == 0 or len(descriptors) != 2:
        return []
    tasks = sorted({str(row["task"]) for row in rows})
    arms = _ordered_arms(rows)
    if "baseline" not in arms or len(tasks) < 2:
        return []
    comparison_arms = [arm for arm in arms if arm != "baseline"]
    model_b = descriptors[1]["model"]
    names = (
        ["intercept"]
        + [f"task:{task}" for task in tasks[1:]]
        + ["model"]
        + [f"treatment:{arm}" for arm in comparison_arms]
        + [f"model:treatment:{arm}" for arm in comparison_arms]
    )
    X: list[list[float]] = []
    for row in rows:
        model_flag = 1.0 if row.get("model") == model_b else 0.0
        arm = str(row["arm"])
        treatment_flags = [
            1.0 if arm == candidate else 0.0
            for candidate in comparison_arms
        ]
        X.append(
            [1.0]
            + [1.0 if str(row["task"]) == task else 0.0 for task in tasks[1:]]
            + [model_flag]
            + treatment_flags
            + [model_flag * value for value in treatment_flags]
        )
    matrix = np.asarray(X, dtype=float)
    coefficients = np.linalg.pinv(matrix) @ z
    report_terms = (
        ["model"]
        + [f"treatment:{arm}" for arm in comparison_arms]
        + [f"model:treatment:{arm}" for arm in comparison_arms]
    )
    out: list[dict[str, Any]] = []
    for term in report_terms:
        term_index = names.index(term)
        vector = coefficients[term_index]
        norm = float(np.linalg.norm(vector))
        for feature, beta in zip(features, vector):
            out.append(
                {
                    "formula": "standardized multivariate outcome ~ task fixed effects + model * treatment",
                    "term": term,
                    "feature": feature,
                    "standardized_beta": float(beta),
                    "term_vector_norm": norm,
                    "n_tasks": len(tasks),
                    "n_observations": len(rows),
                }
            )
    return out


def plot_multivariate_factorial_heatmap(
    output: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    terms = list(dict.fromkeys(str(row["term"]) for row in rows))
    features = list(dict.fromkeys(str(row["feature"]) for row in rows))
    lookup = {
        (str(row["term"]), str(row["feature"])): float(row["standardized_beta"])
        for row in rows if _finite(row.get("standardized_beta"))
    }
    matrix = np.asarray(
        [[lookup.get((term, feature), math.nan) for feature in features] for term in terms],
        dtype=float,
    )
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    limit = max(float(np.max(np.abs(finite))), 1e-6)
    for theme in ("light", "dark"):
        palette = _theme(theme)
        fig, ax = plt.subplots(
            figsize=(max(9.0, 1.25 * len(features)), max(5.5, 0.7 * len(terms)))
        )
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
        )
        fig.patch.set_facecolor(palette["figure"])
        ax.set_facecolor(palette["axes"])
        ax.tick_params(colors=palette["text"], labelsize=9.5)
        ax.set_xticks(range(len(features)), features, rotation=35, ha="right")
        ax.set_yticks(range(len(terms)), terms)
        ax.set_title(
            "Task-blocked multivariate model × treatment effect vectors",
            fontsize=18, fontweight="bold", color=palette["text"],
        )
        for spine in ax.spines.values():
            spine.set_color(palette["edge"])
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
        colorbar.set_label("Standardized coefficient", color=palette["text"])
        colorbar.ax.tick_params(colors=palette["text"])
        fig.text(
            0.5,
            0.01,
            "Each column is a standardized behavior feature; rows are effects from one common task-blocked 2×4 design.",
            ha="center",
            fontsize=9.5,
            color=palette["muted"],
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, output, "multivariate_factorial_effects", theme)


def multivariate_distances(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Task-paired distances in the same standardized feature space used for PCA."""
    rows, features, z = _feature_matrix(task_rows)
    if z.size == 0:
        return [], [], [], []
    vector_by_key = {
        (str(row["model"]), str(row["arm"]), str(row["task"])): z[index]
        for index, row in enumerate(rows)
    }
    arms = _ordered_arms(rows)

    harness_task: list[dict[str, Any]] = []
    for descriptor in descriptors:
        model = descriptor["model"]
        tasks = sorted({
            str(row["task"]) for row in rows if row.get("model") == model
        })
        for task in tasks:
            baseline = vector_by_key.get((model, "baseline", task))
            if baseline is None:
                continue
            for arm in [value for value in arms if value != "baseline"]:
                target = vector_by_key.get((model, arm, task))
                if target is None:
                    continue
                delta = target - baseline
                harness_task.append(
                    {
                        "model": model,
                        "model_key": descriptor["key"],
                        "model_label": descriptor["label"],
                        "arm": arm,
                        "label": LABEL.get(arm, arm),
                        "task": task,
                        "standardized_distance_from_own_baseline": float(np.linalg.norm(delta)),
                        **{
                            f"delta_{feature}": float(value)
                            for feature, value in zip(features, delta)
                        },
                    }
                )

    harness_summary: list[dict[str, Any]] = []
    for descriptor in descriptors:
        for arm in [value for value in arms if value != "baseline"]:
            subset = [
                row for row in harness_task
                if row["model"] == descriptor["model"] and row["arm"] == arm
            ]
            center, low, high, n = _ci95(
                (row["standardized_distance_from_own_baseline"] for row in subset),
                signed=False,
            )
            harness_summary.append(
                {
                    "model": descriptor["model"],
                    "model_key": descriptor["key"],
                    "model_label": descriptor["label"],
                    "arm": arm,
                    "label": LABEL.get(arm, arm),
                    "mean_standardized_distance": center,
                    "ci95_low_distance": low,
                    "ci95_high_distance": high,
                    "n_tasks": n,
                }
            )

    model_task: list[dict[str, Any]] = []
    if len(descriptors) == 2:
        a, b = descriptors
        tasks = sorted({
            str(row["task"]) for row in rows
            if row.get("model") in {a["model"], b["model"]}
        })
        for arm in arms:
            for task in tasks:
                av = vector_by_key.get((a["model"], arm, task))
                bv = vector_by_key.get((b["model"], arm, task))
                if av is None or bv is None:
                    continue
                delta = bv - av
                model_task.append(
                    {
                        "model_a": a["model"],
                        "model_a_label": a["label"],
                        "model_b": b["model"],
                        "model_b_label": b["label"],
                        "arm": arm,
                        "label": LABEL.get(arm, arm),
                        "task": task,
                        "standardized_model_distance": float(np.linalg.norm(delta)),
                        **{
                            f"delta_{feature}": float(value)
                            for feature, value in zip(features, delta)
                        },
                    }
                )

    model_summary: list[dict[str, Any]] = []
    for arm in arms:
        subset = [row for row in model_task if row["arm"] == arm]
        center, low, high, n = _ci95(
            (row["standardized_model_distance"] for row in subset),
            signed=False,
        )
        model_summary.append(
            {
                "arm": arm,
                "label": LABEL.get(arm, arm),
                "mean_standardized_model_distance": center,
                "ci95_low_distance": low,
                "ci95_high_distance": high,
                "n_tasks": n,
            }
        )
    return harness_task, harness_summary, model_task, model_summary


def plot_multivariate_harness_distance(
    output: Path,
    task_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    if not summary_rows:
        return
    arms = [arm for arm in _ordered_arms(summary_rows) if arm != "baseline"]
    values = [
        float(row["mean_standardized_distance"])
        for row in summary_rows if _finite(row.get("mean_standardized_distance"))
    ]
    highs = [
        float(row["ci95_high_distance"])
        for row in summary_rows if _finite(row.get("ci95_high_distance"))
    ]
    if not values:
        return
    ymax = max(values + highs) * 1.12
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, axes = plt.subplots(
            1, len(descriptors), figsize=(7 * len(descriptors), 6.2),
            sharey=True, squeeze=False,
        )
        for index, (ax, descriptor) in enumerate(zip(axes[0], descriptors)):
            _style_axes(fig, ax, theme)
            by_arm = {
                str(row["arm"]): row
                for row in summary_rows if row.get("model") == descriptor["model"]
            }
            x = np.arange(len(arms))
            means = [
                float(by_arm.get(arm, {}).get("mean_standardized_distance", math.nan))
                for arm in arms
            ]
            lows = [
                float(by_arm.get(arm, {}).get("ci95_low_distance", math.nan))
                for arm in arms
            ]
            high = [
                float(by_arm.get(arm, {}).get("ci95_high_distance", math.nan))
                for arm in arms
            ]
            ax.bar(
                x, means, color=[colors[arm] for arm in arms],
                yerr=[
                    [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(means, lows)],
                    [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(means, high)],
                ],
                capsize=5, edgecolor=palette["edge"],
                error_kw={"ecolor": palette["text"], "elinewidth": 1.5},
            )
            task_order = sorted({
                str(row["task"]) for row in task_rows if row.get("model") == descriptor["model"]
            })
            offsets = np.linspace(-0.10, 0.10, max(1, len(task_order)))
            offset_by_task = {task: float(offsets[i]) for i, task in enumerate(task_order)}
            for arm_i, arm in enumerate(arms):
                points = [
                    row for row in task_rows
                    if row.get("model") == descriptor["model"] and row.get("arm") == arm
                ]
                ax.scatter(
                    [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                    [float(row["standardized_distance_from_own_baseline"]) for row in points],
                    s=30, facecolors="none", edgecolors=palette["text"], linewidths=0.9,
                )
            ax.set_xticks(x, [LABEL.get(arm, arm) for arm in arms], rotation=18, ha="right")
            ax.set_ylim(0, ymax)
            ax.set_title(str(descriptor["label"]), fontsize=15, fontweight="bold")
            if index == 0:
                ax.set_ylabel("Standardized multivariate distance")
        fig.suptitle(
            "Multivariate harness distance from each model's baseline",
            fontsize=18, fontweight="bold", color=palette["text"],
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, output, "multivariate_harness_distance", theme)


def plot_multivariate_pca(
    output: Path,
    scores: list[dict[str, Any]],
    loadings: list[dict[str, Any]],
    explained: list[float],
    centroids: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    if not scores:
        return
    arms = _ordered_arms(scores)
    markers = ["o", "s", "^", "D"]
    model_index = _descriptor_order(descriptors)
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, ax = plt.subplots(figsize=(11.0, 8.0))
        _style_axes(fig, ax, theme)
        for row in scores:
            index = model_index.get(str(row["model"]), 0)
            ax.scatter(
                float(row["pc1"]), float(row["pc2"]), s=65, alpha=0.55,
                marker=markers[index % len(markers)], color=colors[str(row["arm"])],
                edgecolors=palette["edge"], linewidths=0.7, zorder=2,
            )
        for row in centroids:
            index = model_index.get(str(row["model"]), 0)
            ax.scatter(
                float(row["pc1"]), float(row["pc2"]), s=220,
                marker=markers[index % len(markers)], color=colors[str(row["arm"])],
                edgecolors=palette["text"], linewidths=2.0, zorder=5,
            )
            ax.annotate(
                f"{row['model_label']} / {LABEL.get(str(row['arm']), str(row['arm']))}",
                (float(row["pc1"]), float(row["pc2"])),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8.5, color=palette["text"],
            )
        x_extent = max([abs(float(row["pc1"])) for row in scores] + [1.0])
        y_extent = max([abs(float(row["pc2"])) for row in scores] + [1.0])
        loading_extent = max(
            [abs(float(row["pc1_loading"])) for row in loadings]
            + [abs(float(row["pc2_loading"])) for row in loadings] + [1e-9]
        )
        scale = 0.62 * min(x_extent, y_extent) / loading_extent
        for row in loadings:
            x = float(row["pc1_loading"]) * scale
            y = float(row["pc2_loading"]) * scale
            ax.annotate(
                "", xy=(x, y), xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": palette["text"]},
            )
            ax.text(
                x * 1.07, y * 1.07, str(row["feature"]),
                fontsize=9, color=palette["text"], ha="center",
            )
        ax.axhline(0, color=palette["muted"], linewidth=1.0, alpha=0.5)
        ax.axvline(0, color=palette["muted"], linewidth=1.0, alpha=0.5)
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)" if len(explained) > 1 else "PC2")
        ax.set_title(
            "Multivariate agent-behavior PCA: model × treatment × task",
            fontsize=18, fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, output, "model_treatment_multivariate_pca", theme)


def _load_events(run_dir: str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "hermes-events.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("captured_at_monotonic_ns") or 0),
            float(row.get("captured_at_unix") or 0),
        ),
    )


def _usage_number(usage: Any, names: tuple[str, ...]) -> float:
    if not isinstance(usage, dict):
        return 0.0
    for name in names:
        if _finite(usage.get(name)):
            return float(usage[name])
    for value in usage.values():
        if isinstance(value, dict):
            found = _usage_number(value, names)
            if found:
                return found
    return 0.0


def _event_time(row: dict[str, Any]) -> float:
    if _finite(row.get("captured_at_monotonic_ns")):
        return float(row["captured_at_monotonic_ns"]) / 1e9
    return float(row.get("captured_at_unix") or 0.0)


def _tool_state(name: str) -> str:
    lower = name.lower()
    if "skill" in lower:
        return "skill"
    if any(word in lower for word in ("edit", "patch", "write", "replace", "create_file")):
        return "edit"
    if any(word in lower for word in ("grep", "search", "find", "glob", "rg")):
        return "search"
    if any(word in lower for word in ("read", "view", "cat", "list", "tree", "ls")):
        return "inspect"
    if any(word in lower for word in ("terminal", "process", "bash", "shell", "python", "exec", "test")):
        return "execute"
    return "other"


def trajectory_rows(
    results: list[Result],
    descriptors: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model_lookup = {row["model"]: row for row in descriptors}
    raw_progress: list[dict[str, Any]] = []
    run_workflow: list[dict[str, Any]] = []
    run_transitions: list[dict[str, Any]] = []

    for result in results:
        if not result.valid:
            continue
        events = _load_events(result.run_dir)
        if not events:
            continue
        start = _event_time(events[0])
        end = _event_time(events[-1])
        span = max(1e-9, end - start)
        api_wait = tool_time = cumulative_tokens = context_tokens = 0.0
        api_calls = tool_calls = 0
        states: list[tuple[float, str]] = []
        event_snapshots: list[dict[str, float]] = [{
            "progress": 0.0,
            "api_wait_seconds": 0.0,
            "tool_execution_seconds": 0.0,
            "api_calls": 0.0,
            "tool_calls": 0.0,
            "cumulative_tokens": 0.0,
            "context_tokens": 0.0,
        }]
        for event in events:
            progress = max(0.0, min(100.0, 100.0 * (_event_time(event) - start) / span))
            kind = str(event.get("event") or "")
            if kind == "pre_api_request" and _finite(event.get("approx_input_tokens")):
                context_tokens = float(event["approx_input_tokens"])
            elif kind == "post_api_request":
                api_calls += 1
                if _finite(event.get("api_duration")):
                    api_wait += max(0.0, float(event["api_duration"]))
                usage = event.get("usage")
                cumulative_tokens += (
                    _usage_number(usage, ("input_tokens", "prompt_tokens"))
                    + _usage_number(usage, ("output_tokens", "completion_tokens"))
                    + _usage_number(usage, ("reasoning_tokens",))
                    + _usage_number(usage, ("cache_read_tokens", "cached_tokens"))
                )
                states.append((progress, "reason"))
            elif kind == "post_tool_call":
                tool_calls += 1
                if _finite(event.get("duration_ms")):
                    tool_time += max(0.0, float(event["duration_ms"]) / 1000.0)
                states.append((progress, _tool_state(str(event.get("tool_name") or ""))))
            event_snapshots.append({
                "progress": progress,
                "api_wait_seconds": api_wait,
                "tool_execution_seconds": tool_time,
                "api_calls": float(api_calls),
                "tool_calls": float(tool_calls),
                "cumulative_tokens": cumulative_tokens,
                "context_tokens": context_tokens,
            })

        descriptor = model_lookup.get(
            result.model,
            {"key": result.model.rsplit("/", 1)[-1], "label": result.model},
        )
        for grid in PROGRESS_GRID:
            candidates = [row for row in event_snapshots if row["progress"] <= grid]
            snap = candidates[-1] if candidates else event_snapshots[0]
            raw_progress.append({
                "model": result.model,
                "model_key": descriptor["key"],
                "model_label": descriptor["label"],
                "arm": result.arm,
                "label": LABEL.get(result.arm, result.arm),
                "task": result.task,
                "repeat": result.repeat,
                "run_index": result.run_index,
                "progress": grid,
                **{key: value for key, value in snap.items() if key != "progress"},
            })

        state_names = [state for _, state in states]
        first_edit = next((progress for progress, state in states if state == "edit"), math.nan)
        first_execute = next((progress for progress, state in states if state == "execute"), math.nan)
        repeated = sum(a == b for a, b in zip(state_names, state_names[1:]))
        inspect_search_before_edit = sum(
            state in {"inspect", "search"}
            for progress, state in states
            if not _finite(first_edit) or progress < first_edit
        )
        late = sum(progress >= 75.0 for progress, _ in states)
        edit_execute = sum(
            a == "edit" and b == "execute"
            for a, b in zip(state_names, state_names[1:])
        )
        run_workflow.append({
            "model": result.model,
            "model_key": descriptor["key"],
            "model_label": descriptor["label"],
            "arm": result.arm,
            "label": LABEL.get(result.arm, result.arm),
            "task": result.task,
            "repeat": result.repeat,
            "run_index": result.run_index,
            "state_events": len(states),
            "first_edit_progress": first_edit,
            "first_execute_progress": first_execute,
            "inspect_search_before_first_edit": inspect_search_before_edit,
            "consecutive_same_state_fraction": repeated / max(1, len(state_names) - 1),
            "edit_to_execute_transitions": edit_execute,
            "late_state_fraction": late / max(1, len(states)),
        })
        transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        source_totals: dict[str, int] = defaultdict(int)
        for source, target in zip(state_names, state_names[1:]):
            transition_counts[(source, target)] += 1
            source_totals[source] += 1
        for (source, target), count in transition_counts.items():
            run_transitions.append({
                "model": result.model,
                "model_key": descriptor["key"],
                "model_label": descriptor["label"],
                "arm": result.arm,
                "label": LABEL.get(result.arm, result.arm),
                "task": result.task,
                "repeat": result.repeat,
                "run_index": result.run_index,
                "source_state": source,
                "target_state": target,
                "count": count,
                "probability": count / max(1, source_totals[source]),
            })

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_progress:
        grouped[(str(row["model"]), str(row["arm"]), str(row["task"]), int(row["progress"]))].append(row)
    task_progress: list[dict[str, Any]] = []
    for (model, arm, task, progress), rows in grouped.items():
        first = rows[0]
        task_progress.append({
            "model": model,
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "arm": arm,
            "label": first["label"],
            "task": task,
            "progress": progress,
            "repeats": len(rows),
            **{
                metric: _mean(row[metric] for row in rows)
                for metric in (
                    "api_wait_seconds", "tool_execution_seconds", "api_calls",
                    "tool_calls", "cumulative_tokens", "context_tokens",
                )
            },
        })

    summary: list[dict[str, Any]] = []
    grouped_summary: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in task_progress:
        grouped_summary[(str(row["model"]), str(row["arm"]), int(row["progress"]))].append(row)
    for (model, arm, progress), rows in grouped_summary.items():
        first = rows[0]
        entry: dict[str, Any] = {
            "model": model,
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "arm": arm,
            "label": first["label"],
            "progress": progress,
            "n_tasks": len(rows),
        }
        for metric in (
            "api_wait_seconds", "tool_execution_seconds", "api_calls",
            "tool_calls", "cumulative_tokens", "context_tokens",
        ):
            center, low, high, n = _ci95((row[metric] for row in rows), signed=False)
            entry[f"mean_{metric}"] = center
            entry[f"ci95_low_{metric}"] = low
            entry[f"ci95_high_{metric}"] = high
            entry[f"n_tasks_{metric}"] = n
        summary.append(entry)

    return raw_progress, task_progress, summary, run_workflow, run_transitions


def plot_trajectory_panels(
    output: Path,
    summary: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
    metric: str,
) -> None:
    usable = [row for row in summary if _finite(row.get(f"mean_{metric}"))]
    if not usable:
        return
    arms = _ordered_arms(usable)
    markers = ["o", "s", "^", "D"]
    title_map = {
        "api_wait_seconds": "Cumulative API wait",
        "tool_execution_seconds": "Cumulative tool execution",
        "api_calls": "Cumulative API calls",
        "tool_calls": "Cumulative tool calls",
        "cumulative_tokens": "Cumulative token consumption",
        "context_tokens": "Approximate context size",
    }
    ylabel = title_map.get(metric, metric.replace("_", " ").title())
    model_index = _descriptor_order(descriptors)
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        ncols = 2
        nrows = int(math.ceil(len(arms) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(13.0, 4.7 * nrows),
            sharex=True, sharey=True, squeeze=False,
        )
        axes_flat = list(axes.flat)
        for ax, arm in zip(axes_flat, arms):
            _style_axes(fig, ax, theme)
            for descriptor in descriptors:
                rows = sorted(
                    [
                        row for row in usable
                        if row.get("arm") == arm and row.get("model") == descriptor["model"]
                    ],
                    key=lambda row: int(row["progress"]),
                )
                if not rows:
                    continue
                index = model_index.get(descriptor["model"], 0)
                ax.plot(
                    [int(row["progress"]) for row in rows],
                    [float(row[f"mean_{metric}"]) for row in rows],
                    marker=markers[index % len(markers)], markevery=2,
                    linewidth=2.2, linestyle="-" if index == 0 else "--",
                    color=colors[arm], label=str(descriptor["label"]),
                )
            ax.set_title(LABEL.get(arm, arm), fontsize=14, fontweight="bold")
            ax.set_xlabel("Normalized elapsed run progress (%)")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=True, fontsize=9)
        for ax in axes_flat[len(arms):]:
            ax.axis("off")
        fig.suptitle(
            f"{ylabel} trajectory by model and treatment",
            fontsize=19, fontweight="bold", color=palette["text"],
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _save(fig, output, f"trajectory_{metric}", theme)



def timing_budget_summary(
    task_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    components = ("api_wait_seconds", "tool_execution_seconds", "unattributed_wall_seconds")
    rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        for arm in _ordered_arms(task_rows):
            subset = [
                row for row in task_rows
                if row.get("model") == descriptor["model"]
                and row.get("arm") == arm
                and row.get("valid_runs", 0) > 0
            ]
            if not subset:
                continue
            entry: dict[str, Any] = {
                "model": descriptor["model"],
                "model_key": descriptor["key"],
                "model_label": descriptor["label"],
                "arm": arm,
                "label": LABEL.get(arm, arm),
                "n_tasks": len(subset),
            }
            for component in components:
                vals = [float(row[component]) for row in subset if _finite(row.get(component))]
                entry[f"mean_{component}"] = _mean(vals)
                entry[f"n_tasks_{component}"] = len(vals)
            rows.append(entry)
    return rows


def plot_timing_budget_facets(
    output: Path,
    rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    if not rows:
        return
    arms = _ordered_arms(rows)
    components = (
        ("api_wait_seconds", "API wait"),
        ("tool_execution_seconds", "Tool execution"),
        ("unattributed_wall_seconds", "Unattributed"),
    )
    all_totals = []
    for row in rows:
        total = sum(
            float(row.get(f"mean_{key}", 0.0))
            for key, _ in components
            if _finite(row.get(f"mean_{key}"))
        )
        all_totals.append(total)
    if not all_totals or max(all_totals) <= 0:
        return
    ymax = max(all_totals) * 1.12
    for theme in ("light", "dark"):
        palette = _theme(theme)
        component_colors = (
            "#38BDF8" if theme == "dark" else "#0284C7",
            "#C084FC" if theme == "dark" else "#9333EA",
            "#FBBF24" if theme == "dark" else "#D97706",
        )
        fig, axes = plt.subplots(
            1, len(descriptors), figsize=(7 * len(descriptors), 6.6),
            sharey=True, squeeze=False,
        )
        for index, (ax, descriptor) in enumerate(zip(axes[0], descriptors)):
            _style_axes(fig, ax, theme)
            by_arm = {
                str(row["arm"]): row
                for row in rows if row.get("model") == descriptor["model"]
            }
            x = np.arange(len(arms))
            bottom = np.zeros(len(arms))
            for (key, label), color in zip(components, component_colors):
                values = np.asarray([
                    float(by_arm.get(arm, {}).get(f"mean_{key}", 0.0))
                    if _finite(by_arm.get(arm, {}).get(f"mean_{key}"))
                    else 0.0
                    for arm in arms
                ])
                ax.bar(
                    x, values, bottom=bottom, label=label,
                    color=color, edgecolor=palette["edge"], linewidth=0.8,
                )
                bottom += values
            ax.set_title(str(descriptor["label"]), fontsize=15, fontweight="bold")
            ax.set_xticks(x, [LABEL.get(arm, arm) for arm in arms], rotation=18, ha="right")
            ax.set_ylim(0, ymax)
            if index == 0:
                ax.set_ylabel("Seconds")
            ax.legend(frameon=True, fontsize=9)
        fig.suptitle(
            "Direct wall-time decomposition by model and treatment",
            fontsize=19, fontweight="bold", color=palette["text"],
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, output, "model_treatment_time_budget", theme)


def workflow_summaries(
    workflow_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = (
        "state_events",
        "first_edit_progress",
        "first_execute_progress",
        "inspect_search_before_first_edit",
        "consecutive_same_state_fraction",
        "edit_to_execute_transitions",
        "late_state_fraction",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in workflow_runs:
        grouped[(str(row["model"]), str(row["arm"]), str(row["task"]))].append(row)
    task_rows: list[dict[str, Any]] = []
    for (model, arm, task), rows in grouped.items():
        first = rows[0]
        entry: dict[str, Any] = {
            "model": model,
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "arm": arm,
            "label": first["label"],
            "task": task,
            "valid_runs": len(rows),
        }
        for metric in metrics:
            entry[metric] = _mean(row.get(metric) for row in rows)
        task_rows.append(entry)

    summary: list[dict[str, Any]] = []
    grouped_summary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped_summary[(str(row["model"]), str(row["arm"]))].append(row)
    for (model, arm), rows in grouped_summary.items():
        first = rows[0]
        entry: dict[str, Any] = {
            "model": model,
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "arm": arm,
            "label": first["label"],
            "tasks_valid": len(rows),
        }
        for metric in metrics:
            center, low, high, n = _ci95((row.get(metric) for row in rows), signed=True)
            entry[f"mean_{metric}"] = center
            entry[f"ci95_low_{metric}"] = low
            entry[f"ci95_high_{metric}"] = high
            entry[f"n_tasks_{metric}"] = n
        summary.append(entry)
    return task_rows, summary


def plot_workflow_metric_facets(
    output: Path,
    task_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
    metric: str,
) -> None:
    values = [
        float(row.get(f"mean_{metric}", math.nan))
        for row in summary_rows if _finite(row.get(f"mean_{metric}"))
    ]
    if not values:
        return
    arms = _ordered_arms(summary_rows)
    titles = {
        "first_edit_progress": "First edit progress",
        "first_execute_progress": "First execution progress",
        "consecutive_same_state_fraction": "Repeated-state fraction",
        "edit_to_execute_transitions": "Edit → execute transitions",
        "late_state_fraction": "Late workflow fraction",
        "inspect_search_before_first_edit": "Inspect/search events before first edit",
    }
    title = titles.get(metric, metric.replace("_", " ").title())
    ymax = max(values + [
        float(row.get(f"ci95_high_{metric}", math.nan))
        for row in summary_rows if _finite(row.get(f"ci95_high_{metric}"))
    ]) * 1.12
    if ymax <= 0:
        ymax = 1.0
    for theme in ("light", "dark"):
        colors = treatment_colors(arms, theme)
        palette = _theme(theme)
        fig, axes = plt.subplots(
            1, len(descriptors), figsize=(7 * len(descriptors), 6.2),
            sharey=True, squeeze=False,
        )
        for index, (ax, descriptor) in enumerate(zip(axes[0], descriptors)):
            _style_axes(fig, ax, theme)
            by_arm = {
                str(row["arm"]): row
                for row in summary_rows if row.get("model") == descriptor["model"]
            }
            x = np.arange(len(arms))
            means = [float(by_arm.get(a, {}).get(f"mean_{metric}", math.nan)) for a in arms]
            low = [float(by_arm.get(a, {}).get(f"ci95_low_{metric}", math.nan)) for a in arms]
            high = [float(by_arm.get(a, {}).get(f"ci95_high_{metric}", math.nan)) for a in arms]
            ax.bar(
                x, means, color=[colors[a] for a in arms],
                yerr=[
                    [max(0.0, m - l) if _finite(m) and _finite(l) else 0.0 for m, l in zip(means, low)],
                    [max(0.0, h - m) if _finite(m) and _finite(h) else 0.0 for m, h in zip(means, high)],
                ],
                capsize=5, edgecolor=palette["edge"],
                error_kw={"ecolor": palette["text"], "elinewidth": 1.5},
            )
            task_order = sorted({
                str(row["task"]) for row in task_rows if row.get("model") == descriptor["model"]
            })
            offsets = np.linspace(-0.10, 0.10, max(1, len(task_order)))
            offset_by_task = {task: float(offsets[i]) for i, task in enumerate(task_order)}
            for arm_i, arm in enumerate(arms):
                points = [
                    row for row in task_rows
                    if row.get("model") == descriptor["model"]
                    and row.get("arm") == arm and _finite(row.get(metric))
                ]
                ax.scatter(
                    [arm_i + offset_by_task.get(str(row["task"]), 0.0) for row in points],
                    [float(row[metric]) for row in points],
                    s=30, facecolors="none", edgecolors=palette["text"], linewidths=0.9,
                )
            ax.set_xticks(x, [LABEL.get(a, a) for a in arms], rotation=18, ha="right")
            ax.set_ylim(0, ymax)
            ax.set_title(str(descriptor["label"]), fontsize=15, fontweight="bold")
            if index == 0:
                ax.set_ylabel(title)
        fig.suptitle(
            f"{title} by model and treatment",
            fontsize=19, fontweight="bold", color=palette["text"],
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, output, f"workflow_{metric}", theme)


def transition_summaries(
    transitions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Pool repeated runs within task first, then normalize source rows.
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    first_by_condition: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in transitions:
        key3 = (str(row["model"]), str(row["arm"]), str(row["task"]))
        first_by_condition.setdefault(key3, row)
        counts[(*key3, str(row["source_state"]), str(row["target_state"]))] += int(row["count"])

    task_rows: list[dict[str, Any]] = []
    for (model, arm, task), first in first_by_condition.items():
        for source in STATE_ORDER:
            source_total = sum(
                counts.get((model, arm, task, source, target), 0)
                for target in STATE_ORDER
            )
            if source_total <= 0:
                continue
            for target in STATE_ORDER:
                count = counts.get((model, arm, task, source, target), 0)
                task_rows.append({
                    "model": model,
                    "model_key": first["model_key"],
                    "model_label": first["model_label"],
                    "arm": arm,
                    "label": first["label"],
                    "task": task,
                    "source_state": source,
                    "target_state": target,
                    "count": count,
                    "probability": count / source_total,
                })

    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[(
            str(row["model"]), str(row["arm"]),
            str(row["source_state"]), str(row["target_state"]),
        )].append(row)
    for (model, arm, source, target), rows in grouped.items():
        first = rows[0]
        center, low, high, n = _ci95((row["probability"] for row in rows), signed=False)
        summary.append({
            "model": model,
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "arm": arm,
            "label": first["label"],
            "source_state": source,
            "target_state": target,
            "mean_probability": center,
            "ci95_low_probability": low,
            "ci95_high_probability": high,
            "n_tasks": n,
        })
    return task_rows, summary


def plot_transition_heatmaps(
    output: Path,
    summary: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> None:
    if not summary:
        return
    arms = _ordered_arms(summary)
    lookup = {
        (str(row["model"]), str(row["arm"]), str(row["source_state"]), str(row["target_state"])): float(row["mean_probability"])
        for row in summary if _finite(row.get("mean_probability"))
    }
    for theme in ("light", "dark"):
        palette = _theme(theme)
        fig, axes = plt.subplots(
            len(descriptors), len(arms),
            figsize=(4.1 * len(arms), 3.8 * len(descriptors)),
            sharex=True, sharey=True, squeeze=False,
        )
        image = None
        for model_i, descriptor in enumerate(descriptors):
            for arm_i, arm in enumerate(arms):
                ax = axes[model_i][arm_i]
                matrix = np.asarray([
                    [
                        lookup.get((descriptor["model"], arm, source, target), math.nan)
                        for target in STATE_ORDER
                    ]
                    for source in STATE_ORDER
                ], dtype=float)
                masked = np.ma.masked_invalid(matrix)
                image = ax.imshow(masked, vmin=0.0, vmax=1.0, aspect="auto")
                ax.set_facecolor(palette["axes"])
                ax.tick_params(colors=palette["text"], labelsize=8)
                if model_i == len(descriptors) - 1:
                    ax.set_xticks(range(len(STATE_ORDER)), STATE_ORDER, rotation=45, ha="right")
                else:
                    ax.set_xticks(range(len(STATE_ORDER)), [])
                if arm_i == 0:
                    ax.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
                    ax.set_ylabel(str(descriptor["label"]), color=palette["text"], fontsize=10)
                else:
                    ax.set_yticks(range(len(STATE_ORDER)), [])
                if model_i == 0:
                    ax.set_title(LABEL.get(arm, arm), color=palette["text"], fontsize=10, fontweight="bold")
        fig.patch.set_facecolor(palette["figure"])
        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72)
            cbar.set_label("Mean transition probability", color=palette["text"])
            cbar.ax.tick_params(colors=palette["text"])
        fig.suptitle(
            "Workflow state-transition matrices",
            fontsize=18, fontweight="bold", color=palette["text"],
        )
        fig.text(0.5, 0.01, "Rows = source state; columns = next state. Probabilities are normalized within task before across-task averaging.", ha="center", fontsize=9.5, color=palette["muted"])
        fig.subplots_adjust(left=0.08, right=0.91, top=0.88, bottom=0.14, wspace=0.12, hspace=0.18)
        _save(fig, output, "workflow_state_transitions", theme)


def write_multi_model_analysis(
    output: Path,
    results: list[Result],
    task_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    descriptors: list[dict[str, str]],
) -> list[str]:
    """Primary cross-model analysis. Tasks remain the independent units."""
    generated: list[str] = []
    for metric in METRICS:
        if metric == "reasoning_tokens" and not any(
            _finite(row.get(f"mean_{metric}")) and abs(float(row[f"mean_{metric}"])) > 1e-12
            for row in summary_rows
        ):
            continue
        plot_faceted_metric(output, task_rows, summary_rows, descriptors, metric)
        generated.extend([
            f"model_treatment_{metric}.vertical.light.png",
            f"model_treatment_{metric}.horizontal.light.png",
        ])
    plot_faceted_resolve_rate(output, task_rows, descriptors)

    pairwise = model_pairwise_effects(task_rows, descriptors)
    harness = harness_effects_within_model(task_rows, descriptors)
    regression_2x4 = factorial_2x4_regression(task_rows, descriptors)
    regression_2x2x2 = factorial_2x2x2_regression(task_rows, descriptors)
    pareto = combined_cost_time_rows(summary_rows)
    _write_csv(output / "model_pairwise_effects.csv", pairwise)
    _write_csv(output / "model_harness_effects.csv", harness)
    _write_csv(output / "model_treatment_regression_2x4.csv", regression_2x4)
    _write_csv(output / "model_caveman_ponytail_regression_2x2x2.csv", regression_2x2x2)
    _write_csv(output / "model_treatment_cost_time.csv", pareto)

    for metric in PRIMARY_METRICS:
        plot_interaction(output, summary_rows, descriptors, metric)
        plot_harness_effects(output, harness, descriptors, metric)
        plot_model_effects(output, pairwise, metric)
    plot_combined_cost_time(output, pareto, descriptors)

    scores, loadings, explained, centroids, similarities = multivariate_analysis(task_rows, descriptors)
    _write_csv(output / "multivariate_pca_scores.csv", scores)
    _write_csv(output / "multivariate_pca_loadings.csv", loadings)
    _write_csv(output / "multivariate_pca_centroids.csv", centroids)
    _write_csv(output / "multivariate_vector_similarity.csv", similarities)
    if explained:
        _write_csv(
            output / "multivariate_pca_explained_variance.csv",
            [
                {"component": index + 1, "explained_variance_fraction": value}
                for index, value in enumerate(explained)
            ],
        )
    plot_multivariate_pca(output, scores, loadings, explained, centroids, descriptors)

    timing_rows = timing_budget_summary(task_rows, descriptors)
    _write_csv(output / "model_treatment_time_budget.csv", timing_rows)
    plot_timing_budget_facets(output, timing_rows, descriptors)

    raw_progress, task_progress, trajectory_summary, workflow_runs, transitions = trajectory_rows(
        results, descriptors
    )
    _write_csv(output / "trajectory_run_progress.csv", raw_progress)
    _write_csv(output / "trajectory_task_progress.csv", task_progress)
    _write_csv(output / "trajectory_summary.csv", trajectory_summary)
    _write_csv(output / "workflow_run_metrics.csv", workflow_runs)

    workflow_task, workflow_summary = workflow_summaries(workflow_runs)
    _write_csv(output / "workflow_task_metrics.csv", workflow_task)
    _write_csv(output / "workflow_summary.csv", workflow_summary)
    for metric in (
        "first_edit_progress",
        "first_execute_progress",
        "inspect_search_before_first_edit",
        "consecutive_same_state_fraction",
        "edit_to_execute_transitions",
        "late_state_fraction",
    ):
        plot_workflow_metric_facets(output, workflow_task, workflow_summary, descriptors, metric)

    transition_task, transition_summary = transition_summaries(transitions)
    _write_csv(output / "workflow_state_transitions_run.csv", transitions)
    _write_csv(output / "workflow_state_transitions_task.csv", transition_task)
    _write_csv(output / "workflow_state_transitions.csv", transition_summary)
    plot_transition_heatmaps(output, transition_summary, descriptors)

    for metric in (
        "cumulative_tokens", "context_tokens", "api_wait_seconds",
        "tool_execution_seconds", "api_calls", "tool_calls",
    ):
        plot_trajectory_panels(output, trajectory_summary, descriptors, metric)
    return generated
