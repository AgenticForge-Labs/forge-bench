from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .config import Result, T975


METRICS = (
    "total_tokens",
    "cost_usd",
    "wall_seconds",
    "api_calls",
    "main_api_calls",
    "tool_calls",
    "iterations_used",
    "resolved",
)


def _finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[Any]) -> float:
    vals = [float(value) for value in values if _finite(value)]
    return statistics.mean(vals) if vals else math.nan


def _ci95(values: Iterable[Any]) -> tuple[float, float, float, int]:
    vals = [float(value) for value in values if _finite(value)]
    if not vals:
        return math.nan, math.nan, math.nan, 0
    center = statistics.mean(vals)
    if len(vals) < 2:
        return center, center, center, len(vals)
    sem = statistics.stdev(vals) / math.sqrt(len(vals))
    critical = T975.get(len(vals) - 1, 1.959964)
    half = critical * sem
    return center, center - half, center + half, len(vals)


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


def _warning_key(value: float | None) -> str:
    return "off" if value is None else f"{float(value):.6g}"


def _condition_key(result: Result) -> tuple[str, str, int | None, float | None, str]:
    return (
        result.model,
        result.arm,
        result.max_turns,
        result.budget_warning_ratio,
        result.task,
    )


def factor_levels(results: list[Result]) -> tuple[list[int], list[float | None]]:
    turns = sorted({
        int(result.max_turns)
        for result in results
        if result.max_turns is not None
    })
    warnings = list(dict.fromkeys(result.budget_warning_ratio for result in results))
    warnings.sort(key=lambda value: (-1.0 if value is None else float(value)))
    return turns, warnings


def factorial_active(results: list[Result]) -> bool:
    turns, warnings = factor_levels(results)
    return len(turns) > 1 or len(warnings) > 1


def task_condition_summary(results: list[Result]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None, float | None, str], list[Result]] = defaultdict(list)
    for result in results:
        groups[_condition_key(result)].append(result)

    rows: list[dict[str, Any]] = []
    for (model, arm, max_turns, warning, task), raw in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] if item[0][2] is not None else -1,
            -1.0 if item[0][3] is None else float(item[0][3]),
            item[0][4],
        ),
    ):
        good = [result for result in raw if result.valid]
        row: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "max_turns": max_turns,
            "budget_warning_ratio": warning,
            "budget_warning_label": _warning_key(warning),
            "task": task,
            "runs": len(raw),
            "valid_runs": len(good),
            "resolved_runs": sum(result.resolved for result in good),
        }
        for metric in METRICS:
            if metric == "resolved":
                row[metric] = _mean(1.0 if result.resolved else 0.0 for result in good)
            else:
                row[metric] = _mean(getattr(result, metric, None) for result in good)
        rows.append(row)
    return rows


def condition_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        groups[
            (
                str(row["model"]),
                str(row["arm"]),
                row.get("max_turns"),
                row.get("budget_warning_ratio"),
            )
        ].append(row)

    out: list[dict[str, Any]] = []
    for (model, arm, max_turns, warning), rows in groups.items():
        item: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "max_turns": max_turns,
            "budget_warning_ratio": warning,
            "budget_warning_label": _warning_key(warning),
            "tasks": len(rows),
            "valid_tasks": sum(int(row.get("valid_runs", 0) > 0) for row in rows),
        }
        for metric in METRICS:
            center, low, high, n = _ci95(
                row.get(metric) for row in rows if int(row.get("valid_runs", 0)) > 0
            )
            item[f"mean_{metric}"] = center
            item[f"ci95_low_{metric}"] = low
            item[f"ci95_high_{metric}"] = high
            item[f"n_tasks_{metric}"] = n
        out.append(item)
    return sorted(
        out,
        key=lambda row: (
            str(row["model"]),
            str(row["arm"]),
            int(row["max_turns"]) if row["max_turns"] is not None else -1,
            -1.0 if row["budget_warning_ratio"] is None else float(row["budget_warning_ratio"]),
        ),
    )


def _paired_contrast(
    by_condition: dict[tuple[str, str, int | None, float | None], dict[str, dict[str, Any]]],
    *,
    model: str,
    arm: str,
    a_turns: int | None,
    a_warning: float | None,
    b_turns: int | None,
    b_warning: float | None,
    factor: str,
    level_a: str,
    level_b: str,
) -> list[dict[str, Any]]:
    a = by_condition.get((model, arm, a_turns, a_warning), {})
    b = by_condition.get((model, arm, b_turns, b_warning), {})
    common = sorted(set(a) & set(b))
    out: list[dict[str, Any]] = []
    for metric in METRICS:
        deltas = [
            float(b[task][metric]) - float(a[task][metric])
            for task in common
            if _finite(a[task].get(metric)) and _finite(b[task].get(metric))
        ]
        center, low, high, n = _ci95(deltas)
        out.append(
            {
                "effect": factor,
                "model": model,
                "arm": arm,
                "metric": metric,
                "level_a": level_a,
                "level_b": level_b,
                "a_max_turns": a_turns,
                "a_budget_warning_ratio": a_warning,
                "b_max_turns": b_turns,
                "b_budget_warning_ratio": b_warning,
                "n_paired_tasks": n,
                "mean_delta_b_minus_a": center,
                "ci95_low_delta": low,
                "ci95_high_delta": high,
            }
        )
    return out


def paired_effects(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[
        tuple[str, str, int | None, float | None],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)
    models: set[str] = set()
    arms: set[str] = set()
    turns: set[int] = set()
    warnings: list[float | None] = []
    for row in task_rows:
        if int(row.get("valid_runs", 0)) <= 0:
            continue
        model = str(row["model"])
        arm = str(row["arm"])
        max_turns = row.get("max_turns")
        warning = row.get("budget_warning_ratio")
        by_condition[(model, arm, max_turns, warning)][str(row["task"])] = row
        models.add(model)
        arms.add(arm)
        if max_turns is not None:
            turns.add(int(max_turns))
        if warning not in warnings:
            warnings.append(warning)

    turn_levels = sorted(turns)
    warning_levels = sorted(
        warnings,
        key=lambda value: (-1.0 if value is None else float(value)),
    )
    rows: list[dict[str, Any]] = []

    for model in sorted(models):
        for arm in sorted(arms):
            for max_turns in turn_levels:
                for a_warning, b_warning in combinations(warning_levels, 2):
                    rows.extend(
                        _paired_contrast(
                            by_condition,
                            model=model,
                            arm=arm,
                            a_turns=max_turns,
                            a_warning=a_warning,
                            b_turns=max_turns,
                            b_warning=b_warning,
                            factor="budget_warning_ratio",
                            level_a=_warning_key(a_warning),
                            level_b=_warning_key(b_warning),
                        )
                    )

            for warning in warning_levels:
                for a_turns, b_turns in combinations(turn_levels, 2):
                    rows.extend(
                        _paired_contrast(
                            by_condition,
                            model=model,
                            arm=arm,
                            a_turns=a_turns,
                            a_warning=warning,
                            b_turns=b_turns,
                            b_warning=warning,
                            factor="max_turns",
                            level_a=str(a_turns),
                            level_b=str(b_turns),
                        )
                    )

            if len(turn_levels) >= 2 and len(warning_levels) >= 2:
                baseline_warning = None if None in warning_levels else warning_levels[0]
                for a_turns, b_turns in combinations(turn_levels, 2):
                    for warning in warning_levels:
                        if warning == baseline_warning:
                            continue
                        low_base = by_condition.get((model, arm, a_turns, baseline_warning), {})
                        low_warn = by_condition.get((model, arm, a_turns, warning), {})
                        high_base = by_condition.get((model, arm, b_turns, baseline_warning), {})
                        high_warn = by_condition.get((model, arm, b_turns, warning), {})
                        common = sorted(
                            set(low_base) & set(low_warn) & set(high_base) & set(high_warn)
                        )
                        for metric in METRICS:
                            values = []
                            for task in common:
                                quartet = (
                                    low_base[task].get(metric),
                                    low_warn[task].get(metric),
                                    high_base[task].get(metric),
                                    high_warn[task].get(metric),
                                )
                                if not all(_finite(value) for value in quartet):
                                    continue
                                lb, lw, hb, hw = (float(value) for value in quartet)
                                values.append((hw - hb) - (lw - lb))
                            center, low, high, n = _ci95(values)
                            rows.append(
                                {
                                    "effect": "max_turns_x_budget_warning_ratio",
                                    "model": model,
                                    "arm": arm,
                                    "metric": metric,
                                    "level_a": f"turns={a_turns}, reminder={_warning_key(warning)} vs off",
                                    "level_b": f"turns={b_turns}, reminder={_warning_key(warning)} vs off",
                                    "a_max_turns": a_turns,
                                    "a_budget_warning_ratio": warning,
                                    "b_max_turns": b_turns,
                                    "b_budget_warning_ratio": warning,
                                    "n_paired_tasks": n,
                                    "mean_delta_b_minus_a": center,
                                    "ci95_low_delta": low,
                                    "ci95_high_delta": high,
                                }
                            )
    return rows


def write_factorial_analysis(output: Path, results: list[Result]) -> bool:
    """Write factor-aware tables. Returns True when a factor has >1 level."""
    if not results:
        return False
    task_rows = task_condition_summary(results)
    _write_csv(output / "factorial_task_summary.csv", task_rows)
    _write_csv(output / "factorial_condition_summary.csv", condition_summary(task_rows))
    _write_csv(output / "factorial_paired_effects.csv", paired_effects(task_rows))
    return factorial_active(results)
