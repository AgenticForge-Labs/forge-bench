from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import LABEL, budget_condition_label


def find_latest_run(root: Path = Path("benchmark-results"), *, prefer_incomplete: bool = True) -> Path:
    """Find the newest Forge Bench run without requiring its timestamped name."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root not found: {root}")

    candidates: list[tuple[float, Path, bool]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if not (path / "metadata.json").is_file() and not (path / "run_plan.csv").is_file():
            continue
        expected = expected_runs(path)
        completed = len(load_partial_results(path))
        incomplete = expected > 0 and completed < expected
        mtimes = [
            p.stat().st_mtime
            for p in (path / "metadata.json", path / "run_plan.csv", path / "runs.partial.json")
            if p.exists()
        ]
        stamp = max(mtimes) if mtimes else path.stat().st_mtime
        candidates.append((stamp, path, incomplete))

    if not candidates:
        raise FileNotFoundError(f"No Forge Bench runs found under {root}")

    if prefer_incomplete:
        active = [row for row in candidates if row[2]]
        if active:
            candidates = active
    return max(candidates, key=lambda row: row[0])[1]


def read_run_plan(run_dir: Path) -> list[dict[str, str]]:
    plan = run_dir / "run_plan.csv"
    if not plan.is_file():
        return []
    with plan.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_runs(run_dir: Path) -> int:
    return len(read_run_plan(run_dir))


def planned_factors(run_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Return treatment order, model order, and display labels from the run plan."""
    plan = read_run_plan(run_dir)
    treatments: list[str] = []
    models: list[str] = []
    model_labels: dict[str, str] = {}
    for row in plan:
        arm = str(row.get("arm") or "")
        model = str(row.get("model") or "")
        if arm and arm not in treatments:
            treatments.append(arm)
        if model and model not in models:
            models.append(model)
        if model:
            model_labels.setdefault(model, str(row.get("model_label") or model.rsplit("/", 1)[-1]))
    return treatments, models, model_labels


def load_partial_results(run_dir: Path) -> list[dict[str, Any]]:
    """Read the best available partial result stream for a running experiment."""
    for filename in ("runs.partial.json", "runs.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    runs = run_dir / "runs"
    if runs.is_dir():
        for result_path in sorted(runs.glob("*/result.json")):
            try:
                row = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("run_index") or 0))
    return rows


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def classify_result(row: dict[str, Any]) -> str:
    if not bool(row.get("valid")):
        return "invalid"
    if bool(row.get("resolved")):
        return "resolved"
    if not bool(row.get("patch_nonempty")):
        return "no_patch"
    if bool(row.get("evaluation_completed")):
        return "evaluated_unresolved"
    return "unresolved_other"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = load_partial_results(run_dir)
    expected = expected_runs(run_dir)
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[classify_result(row)] += 1

    valid = [row for row in rows if bool(row.get("valid"))]
    costs = [float(row["cost_usd"]) for row in valid if _finite(row.get("cost_usd"))]
    times = [float(row["wall_seconds"]) for row in valid if _finite(row.get("wall_seconds"))]
    tokens = [float(row["total_tokens"]) for row in valid if _finite(row.get("total_tokens"))]

    return {
        "run_dir": str(run_dir.resolve()),
        "expected_runs": expected,
        "completed_runs": len(rows),
        "remaining_runs": max(0, expected - len(rows)) if expected else None,
        "completion_percent": (100.0 * len(rows) / expected) if expected else None,
        "valid_runs": len(valid),
        "resolved_runs": statuses["resolved"],
        "resolve_rate_percent": (100.0 * statuses["resolved"] / len(valid)) if valid else None,
        "status_counts": dict(statuses),
        "mean_cost_usd": mean(costs) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
        "mean_wall_seconds": mean(times) if times else None,
        "mean_total_tokens": mean(tokens) if tokens else None,
        "last_run_index": max((int(row.get("run_index") or 0) for row in rows), default=0),
    }


def _cell_means(
    rows: list[dict[str, Any]],
    treatments: list[str],
    models: list[str],
    metric: str,
) -> dict[tuple[str, str], tuple[float | None, int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("valid")) or not _finite(row.get(metric)):
            continue
        key = (str(row.get("model") or ""), str(row.get("arm") or ""))
        grouped[key].append(float(row[metric]))

    result: dict[tuple[str, str], tuple[float | None, int]] = {}
    for model in models:
        for arm in treatments:
            values = grouped.get((model, arm), [])
            result[(model, arm)] = (mean(values) if values else None, len(values))
    return result


def _plot_model_treatment_bars(
    ax: Any,
    rows: list[dict[str, Any]],
    treatments: list[str],
    models: list[str],
    model_labels: dict[str, str],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    if not treatments or not models:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No planned factors available", ha="center", va="center")
        return

    cells = _cell_means(rows, treatments, models, metric)
    x = np.arange(len(treatments), dtype=float)
    width = 0.8 / max(1, len(models))

    for model_i, model in enumerate(models):
        offset = (model_i - (len(models) - 1) / 2) * width
        heights = [
            cells[(model, arm)][0] if cells[(model, arm)][0] is not None else 0.0
            for arm in treatments
        ]
        bars = ax.bar(
            x + offset,
            heights,
            width=width * 0.92,
            label=model_labels.get(model, model.rsplit("/", 1)[-1]),
        )
        for bar, arm in zip(bars, treatments):
            value, n = cells[(model, arm)]
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x, [budget_condition_label(arm) for arm in treatments], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def make_dashboard(run_dir: Path, output: Path | None = None) -> Path:
    """Write one compact 2x2 PNG summarizing an in-progress benchmark."""
    rows = load_partial_results(run_dir)
    summary = summarize_run(run_dir)
    if not rows:
        raise RuntimeError(f"No completed results yet in {run_dir}")

    treatments, models, model_labels = planned_factors(run_dir)
    if not treatments:
        treatments = list(dict.fromkeys(str(row.get("arm") or "") for row in rows if row.get("arm")))
    if not models:
        models = list(dict.fromkeys(str(row.get("model") or "") for row in rows if row.get("model")))
        model_labels = {model: model.rsplit("/", 1)[-1] for model in models}

    output = output or (run_dir / "live-dashboard.png")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure_width = max(14.0, 9.0 + 1.4 * len(treatments))
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, 9))
    fig.suptitle(
        f"Forge Bench live status — {Path(run_dir).name}",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0][0]
    expected = int(summary["expected_runs"] or len(rows))
    completed = int(summary["completed_runs"])
    ax.barh(["Experiment"], [completed])
    ax.barh(["Experiment"], [max(0, expected - completed)], left=[completed], alpha=0.25)
    ax.set_xlim(0, max(1, expected))
    ax.set_xlabel("Runs")
    ax.set_title(f"Completion: {completed}/{expected} ({summary['completion_percent'] or 0:.1f}%)")
    ax.text(completed / 2 if completed else 0.1, 0, f"{completed} complete", ha="center", va="center")

    ax = axes[0][1]
    status_order = ["resolved", "evaluated_unresolved", "no_patch", "unresolved_other", "invalid"]
    status_counts = summary["status_counts"]
    labels = [name.replace("_", " ") for name in status_order if status_counts.get(name, 0)]
    values = [status_counts[name] for name in status_order if status_counts.get(name, 0)]
    if values:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Completed runs")
    ax.set_title("Outcome status")

    _plot_model_treatment_bars(
        axes[1][0],
        rows,
        treatments,
        models,
        model_labels,
        "cost_usd",
        "Mean cost by model × treatment",
        "Mean cost (USD)",
    )
    _plot_model_treatment_bars(
        axes[1][1],
        rows,
        treatments,
        models,
        model_labels,
        "wall_seconds",
        "Mean time by model × treatment",
        "Mean agent wall time (s)",
    )

    footer = (
        f"Valid={summary['valid_runs']}  Resolved={summary['resolved_runs']}  "
        f"Resolve={summary['resolve_rate_percent'] or 0:.1f}%  "
        f"Spend=${summary['total_cost_usd'] or 0:.4f}  "
        "Partial cell means are descriptive only; task coverage is still accruing."
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def print_status(run_dir: Path) -> None:
    summary = summarize_run(run_dir)
    print(f"run: {summary['run_dir']}")
    print(
        f"progress: {summary['completed_runs']}/{summary['expected_runs']} "
        f"({summary['completion_percent'] or 0:.1f}%)"
    )
    print(
        f"valid: {summary['valid_runs']}  resolved: {summary['resolved_runs']}  "
        f"resolve-rate: {summary['resolve_rate_percent'] or 0:.1f}%"
    )
    counts = summary["status_counts"]
    if counts:
        print("outcomes:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if summary["total_cost_usd"] is not None:
        print(
            f"cost: total=${summary['total_cost_usd']:.4f} "
            f"mean=${summary['mean_cost_usd']:.4f}"
        )
    if summary["mean_wall_seconds"] is not None:
        print(f"mean agent time: {summary['mean_wall_seconds']:.1f}s")
    if summary["mean_total_tokens"] is not None:
        print(f"mean tokens: {summary['mean_total_tokens']:,.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the newest Forge Bench run while it is still running.")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run", type=Path, help="Explicit run directory; otherwise auto-discover newest active run.")
    parser.add_argument("--latest-finished", action="store_true", help="Do not prefer an incomplete run during auto-discovery.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status JSON.")
    parser.add_argument("--dashboard", action="store_true", help="Generate a compact live-dashboard PNG.")
    parser.add_argument("--output", type=Path, help="Optional dashboard output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run.expanduser().resolve() if args.run else find_latest_run(
        args.root, prefer_incomplete=not args.latest_finished
    )
    summary = summarize_run(run_dir)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_status(run_dir)
    if args.dashboard:
        path = make_dashboard(run_dir, args.output)
        print(f"dashboard: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
