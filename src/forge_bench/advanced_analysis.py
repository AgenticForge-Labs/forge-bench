from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .config import LABEL, T975

ADVANCED_METRICS = ("total_tokens", "wall_seconds", "api_calls", "cost_usd")
PCA_FEATURES = ("total_tokens", "wall_seconds", "api_calls", "cost_usd", "diff_lines")
TREATMENT_ORDER = ("baseline", "caveman", "ponytail", "caveman_ponytail", "lean_tools", "all_three")


def _ordered_arms(arms: set[str] | list[str]) -> list[str]:
    values = list(dict.fromkeys(str(arm) for arm in arms))
    return [arm for arm in TREATMENT_ORDER if arm in values] + sorted(
        arm for arm in values if arm not in TREATMENT_ORDER
    )


def _treatment_colors(arms: list[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {arm: palette[index % len(palette)] for index, arm in enumerate(_ordered_arms(arms))}



def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    ax.tick_params(colors=colors["text"], labelsize=12, width=1.4)
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["edge"])
        spine.set_linewidth(1.4)
    ax.grid(alpha=0.30, linewidth=1.1, color=colors["grid"])
    ax.set_axisbelow(True)


def _save_themed(fig: Any, output: Path, stem: str, theme: str) -> None:
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


def paired_effects(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        row["task"]: row
        for row in task_rows
        if row.get("arm") == "baseline" and row.get("valid_runs", 0) > 0
    }
    rows: list[dict[str, Any]] = []
    arms = [arm for arm in _ordered_arms({str(row["arm"]) for row in task_rows}) if arm != "baseline"]
    for arm in arms:
        for metric in ADVANCED_METRICS:
            effects: list[float] = []
            used: list[str] = []
            for row in task_rows:
                if row.get("arm") != arm or row.get("valid_runs", 0) <= 0:
                    continue
                base = baseline.get(str(row["task"]))
                if not base:
                    continue
                if not (_finite(row.get(metric)) and _finite(base.get(metric))):
                    continue
                a = float(row[metric])
                b = float(base[metric])
                if a <= 0 or b <= 0:
                    continue
                effects.append(100.0 * (a / b - 1.0))
                used.append(str(row["task"]))
            if not effects:
                continue
            center = float(np.mean(effects))
            if len(effects) > 1:
                sem = float(np.std(effects, ddof=1) / math.sqrt(len(effects)))
                crit = T975.get(len(effects) - 1, 1.959964)
                low, high = center - crit * sem, center + crit * sem
            else:
                low = high = center
            rows.append(
                {
                    "arm": arm,
                    "label": LABEL.get(arm, arm),
                    "metric": metric,
                    "n_tasks": len(effects),
                    "mean_percent_change_vs_baseline": center,
                    "ci95_low_percent_change": low,
                    "ci95_high_percent_change": high,
                    "tasks": ";".join(used),
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


def treatment_regression(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in task_rows if row.get("valid_runs", 0) > 0]
    tasks = sorted({str(row["task"]) for row in rows})
    arms = _ordered_arms({str(row["arm"]) for row in rows})
    if "baseline" not in arms or len(tasks) < 2:
        return []
    comparison_arms = [arm for arm in arms if arm != "baseline"]
    out: list[dict[str, Any]] = []
    for metric in ADVANCED_METRICS:
        usable = [
            row for row in rows
            if _finite(row.get(metric)) and float(row[metric]) > 0
        ]
        if len(usable) <= len(tasks) + 1:
            continue
        X: list[list[float]] = []
        y: list[float] = []
        names = ["intercept"] + [f"task:{task}" for task in tasks[1:]] + [
            f"arm:{arm}" for arm in comparison_arms
        ]
        for row in usable:
            X.append(
                [1.0]
                + [1.0 if row["task"] == task else 0.0 for task in tasks[1:]]
                + [1.0 if row["arm"] == arm else 0.0 for arm in comparison_arms]
            )
            y.append(math.log(float(row[metric])))
        beta, se, df = _fit_ols(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        crit = T975.get(df, 1.959964)
        for arm in comparison_arms:
            idx = names.index(f"arm:{arm}")
            b, s = float(beta[idx]), float(se[idx])
            out.append(
                {
                    "model": "log(metric) ~ task fixed effects + treatment",
                    "metric": metric,
                    "term": arm,
                    "label": LABEL.get(arm, arm),
                    "n_observations": len(y),
                    "df_residual": df,
                    "beta_log": b,
                    "se_log": s,
                    "estimated_percent_change_vs_baseline": 100.0 * (math.exp(b) - 1.0),
                    "ci95_low_percent_change": 100.0 * (math.exp(b - crit * s) - 1.0),
                    "ci95_high_percent_change": 100.0 * (math.exp(b + crit * s) - 1.0),
                }
            )
    return out


def factorial_regression(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"baseline", "caveman", "ponytail", "caveman_ponytail"}
    rows = [
        row for row in task_rows
        if row.get("valid_runs", 0) > 0 and row.get("arm") in required
    ]
    if not required.issubset({str(row["arm"]) for row in rows}):
        return []
    tasks = sorted({str(row["task"]) for row in rows})
    out: list[dict[str, Any]] = []
    for metric in ADVANCED_METRICS:
        usable = [
            row for row in rows
            if _finite(row.get(metric)) and float(row[metric]) > 0
        ]
        if len(usable) <= len(tasks) + 3:
            continue
        X: list[list[float]] = []
        y: list[float] = []
        names = ["intercept"] + [f"task:{task}" for task in tasks[1:]] + [
            "caveman", "ponytail", "interaction"
        ]
        for row in usable:
            arm = str(row["arm"])
            cave = 1.0 if arm in {"caveman", "caveman_ponytail"} else 0.0
            pony = 1.0 if arm in {"ponytail", "caveman_ponytail"} else 0.0
            X.append(
                [1.0]
                + [1.0 if row["task"] == task else 0.0 for task in tasks[1:]]
                + [cave, pony, cave * pony]
            )
            y.append(math.log(float(row[metric])))
        beta, se, df = _fit_ols(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        crit = T975.get(df, 1.959964)
        for term in ("caveman", "ponytail", "interaction"):
            idx = names.index(term)
            b, s = float(beta[idx]), float(se[idx])
            out.append(
                {
                    "model": "log(metric) ~ task fixed effects + caveman * ponytail",
                    "metric": metric,
                    "term": term,
                    "n_observations": len(y),
                    "df_residual": df,
                    "beta_log": b,
                    "se_log": s,
                    "estimated_percent_change": 100.0 * (math.exp(b) - 1.0),
                    "ci95_low_percent_change": 100.0 * (math.exp(b - crit * s) - 1.0),
                    "ci95_high_percent_change": 100.0 * (math.exp(b + crit * s) - 1.0),
                }
            )
    return out


def _feature_matrix(task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    rows = [row for row in task_rows if row.get("valid_runs", 0) > 0]
    usable_features = [
        feature
        for feature in PCA_FEATURES
        if any(_finite(row.get(feature)) and float(row[feature]) > 0 for row in rows)
    ]
    complete = [
        row for row in rows
        if all(_finite(row.get(feature)) and float(row[feature]) >= 0 for feature in usable_features)
    ]
    if len(complete) < 3 or len(usable_features) < 2:
        return [], [], np.empty((0, 0))
    data = np.asarray(
        [
            [
                math.log1p(float(row[feature]))
                for feature in usable_features
            ]
            for row in complete
        ],
        dtype=float,
    )
    std = data.std(axis=0, ddof=1)
    keep = std > 1e-12
    data = data[:, keep]
    names = [name for name, flag in zip(usable_features, keep) if flag]
    if data.shape[1] < 2:
        return [], [], np.empty((0, 0))
    z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    return complete, names, z


def pca_analysis(task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    rows, features, z = _feature_matrix(task_rows)
    if z.size == 0:
        return [], [], []
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    variance = singular ** 2
    explained = variance / variance.sum()
    score_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score_rows.append(
            {
                "arm": row["arm"],
                "label": row.get("label", LABEL.get(str(row["arm"]), str(row["arm"]))),
                "task": row["task"],
                "pc1": float(scores[index, 0]),
                "pc2": float(scores[index, 1]) if scores.shape[1] > 1 else 0.0,
            }
        )
    loading_rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        loading_rows.append(
            {
                "feature": feature,
                "pc1_loading": float(vt[0, feature_index]),
                "pc2_loading": float(vt[1, feature_index]) if vt.shape[0] > 1 else 0.0,
            }
        )
    return score_rows, loading_rows, [float(value) for value in explained]


def _kmeans(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    centers = [z[0].copy()]
    while len(centers) < k:
        distance = np.min(
            np.stack([np.sum((z - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(z[int(np.argmax(distance))].copy())
    centers_arr = np.asarray(centers)
    labels = np.zeros(len(z), dtype=int)
    for _ in range(50):
        distances = np.stack(
            [np.sum((z - center) ** 2, axis=1) for center in centers_arr],
            axis=1,
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = z[labels == cluster]
            if len(members):
                centers_arr[cluster] = members.mean(axis=0)
    return labels, centers_arr


def cluster_analysis(task_rows: list[dict[str, Any]], pca_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, _, z = _feature_matrix(task_rows)
    if z.size == 0:
        return []
    k = min(3, max(2, len(z) // 5), len(z))
    labels, _ = _kmeans(z, k)
    scores_by_key = {
        (str(row["arm"]), str(row["task"])): row for row in pca_scores
    }
    out: list[dict[str, Any]] = []
    for row, cluster in zip(rows, labels):
        score = scores_by_key.get((str(row["arm"]), str(row["task"])), {})
        out.append(
            {
                "arm": row["arm"],
                "label": row.get("label", LABEL.get(str(row["arm"]), str(row["arm"]))),
                "task": row["task"],
                "cluster": int(cluster) + 1,
                "pc1": score.get("pc1", math.nan),
                "pc2": score.get("pc2", math.nan),
            }
        )
    return out


def correlation_analysis(task_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    rows, features, z = _feature_matrix(task_rows)
    if not rows:
        return [], np.empty((0, 0))
    return features, np.corrcoef(z, rowvar=False)


def _plot_pca(
    output: Path,
    scores: list[dict[str, Any]],
    loadings: list[dict[str, Any]],
    explained: list[float],
    theme: str,
) -> None:
    if not scores:
        return
    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    _style_axes(fig, ax, theme)
    arms = _ordered_arms({str(row["arm"]) for row in scores})
    colors = _treatment_colors(arms)

    for arm in arms:
        subset = [row for row in scores if row["arm"] == arm]
        ax.scatter(
            [row["pc1"] for row in subset],
            [row["pc2"] for row in subset],
            s=100,
            alpha=0.88,
            edgecolors=_theme(theme)["edge"],
            linewidths=1.0,
            color=colors[arm],
            label=LABEL.get(arm, arm),
            zorder=3,
        )

    x_extent = max(abs(float(row["pc1"])) for row in scores) or 1.0
    y_extent = max(abs(float(row["pc2"])) for row in scores) or 1.0
    loading_extent = max(
        [abs(float(row["pc1_loading"])) for row in loadings]
        + [abs(float(row["pc2_loading"])) for row in loadings]
        + [1e-9]
    )
    scale = 0.72 * min(x_extent, y_extent) / loading_extent
    arrow_color = _theme(theme)["text"]
    for row in loadings:
        x = float(row["pc1_loading"]) * scale
        y = float(row["pc2_loading"]) * scale
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": arrow_color},
            zorder=4,
        )
        ax.text(
            x * 1.08,
            y * 1.08,
            str(row["feature"]),
            fontsize=10.5,
            color=arrow_color,
            ha="center",
            va="center",
            fontweight="semibold",
        )

    ax.axhline(0, color=_theme(theme)["muted"], linewidth=1.0, alpha=0.55)
    ax.axvline(0, color=_theme(theme)["muted"], linewidth=1.0, alpha=0.55)
    pc1 = 100 * explained[0] if explained else 0.0
    pc2 = 100 * explained[1] if len(explained) > 1 else 0.0
    ax.set_title("PCA biplot of task-level efficiency profiles", fontsize=19, fontweight="semibold")
    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)", fontsize=14)
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)", fontsize=14)
    legend = ax.legend(title="Treatment", fontsize=11, title_fontsize=11, frameon=True)
    legend.get_frame().set_alpha(0.88)
    _save_themed(fig, output, "advanced_pca_biplot", theme)


def _plot_scree(output: Path, explained: list[float], theme: str) -> None:
    if not explained:
        return
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    _style_axes(fig, ax, theme)
    components = np.arange(1, len(explained) + 1)
    percent = np.asarray(explained) * 100.0
    ax.plot(components, percent, marker="o", linewidth=2.4, markersize=8)
    ax.bar(components, percent, alpha=0.32)
    ax.set_xticks(components)
    ax.set_xlabel("Principal component", fontsize=14)
    ax.set_ylabel("Explained variance (%)", fontsize=14)
    ax.set_title("PCA scree plot", fontsize=19, fontweight="semibold")
    for x, value in zip(components, percent):
        ax.text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=10.5, color=_theme(theme)["text"])
    _save_themed(fig, output, "advanced_pca_scree", theme)


def _plot_loadings(output: Path, loadings: list[dict[str, Any]], theme: str) -> None:
    if not loadings:
        return
    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    _style_axes(fig, ax, theme)
    labels = [str(row["feature"]) for row in loadings]
    y = np.arange(len(labels))
    height = 0.36
    ax.barh(y - height / 2, [float(row["pc1_loading"]) for row in loadings], height=height, label="PC1")
    ax.barh(y + height / 2, [float(row["pc2_loading"]) for row in loadings], height=height, label="PC2")
    ax.axvline(0, color=_theme(theme)["muted"], linewidth=1.2)
    ax.set_yticks(y, labels, fontsize=12)
    ax.set_xlabel("Loading", fontsize=14)
    ax.set_title("Feature loadings for PC1 and PC2", fontsize=19, fontweight="semibold")
    ax.legend(fontsize=11)
    _save_themed(fig, output, "advanced_pca_loadings", theme)


def _plot_clusters(output: Path, rows: list[dict[str, Any]], theme: str) -> None:
    if not rows or not all(_finite(row.get("pc1")) and _finite(row.get("pc2")) for row in rows):
        return
    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    _style_axes(fig, ax, theme)
    arms = _ordered_arms({str(row["arm"]) for row in rows})
    colors = _treatment_colors(arms)

    for arm in arms:
        subset = [row for row in rows if str(row["arm"]) == arm]
        ax.scatter(
            [float(row["pc1"]) for row in subset],
            [float(row["pc2"]) for row in subset],
            s=100,
            alpha=0.90,
            color=colors[arm],
            edgecolors=_theme(theme)["edge"],
            linewidths=1.0,
            label=LABEL.get(arm, arm),
            zorder=3,
        )

    for cluster in sorted({int(row["cluster"]) for row in rows}):
        subset = [row for row in rows if int(row["cluster"]) == cluster]
        xy = np.asarray([[float(row["pc1"]), float(row["pc2"])] for row in subset])
        center = xy.mean(axis=0)
        radius = max(float(np.linalg.norm(point - center)) for point in xy) if len(xy) else 0.0
        radius = max(radius * 1.18, 0.18)
        circle = Circle(
            center,
            radius,
            fill=False,
            linestyle="--",
            linewidth=2.0,
            edgecolor=_theme(theme)["muted"],
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1] + radius,
            f"Cluster {cluster}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=_theme(theme)["text"],
            fontweight="semibold",
        )

    ax.set_title("Exploratory clusters in PCA efficiency space", fontsize=19, fontweight="semibold")
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.legend(title="Treatment", fontsize=11, title_fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    _save_themed(fig, output, "advanced_clusters", theme)


def _plot_correlations(output: Path, features: list[str], corr: np.ndarray, theme: str) -> None:
    if not features:
        return
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    colors = _theme(theme)
    fig.patch.set_facecolor(colors["figure"])
    ax.set_facecolor(colors["axes"])
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(features)), features, fontsize=12)
    ax.tick_params(colors=colors["text"])
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_color(colors["text"])
    ax.set_title("Correlation of task-level efficiency measures", fontsize=19, fontweight="semibold", color=colors["text"])
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.ax.tick_params(labelsize=11, colors=colors["text"])
    _save_themed(fig, output, "advanced_correlations", theme)


def _plot_effects(output: Path, rows: list[dict[str, Any]], theme: str) -> None:
    token_rows = [row for row in rows if row["metric"] == "total_tokens"]
    if not token_rows:
        return
    token_rows.sort(key=lambda row: float(row["mean_percent_change_vs_baseline"]))
    fig, ax = plt.subplots(figsize=(10.5, max(5.8, 0.85 * len(token_rows) + 2.5)))
    _style_axes(fig, ax, theme)
    y = np.arange(len(token_rows))
    centers = np.asarray([float(row["mean_percent_change_vs_baseline"]) for row in token_rows])
    lower = centers - np.asarray([float(row["ci95_low_percent_change"]) for row in token_rows])
    upper = np.asarray([float(row["ci95_high_percent_change"]) for row in token_rows]) - centers
    ax.errorbar(
        centers,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=9,
        capsize=6,
        elinewidth=2.0,
        capthick=1.8,
    )
    ax.axvline(0, linewidth=1.6, linestyle="--", color=_theme(theme)["muted"])
    ax.set_yticks(y, [row["label"] for row in token_rows], fontsize=12)
    ax.set_xlabel("Paired change in total tokens vs baseline (%)", fontsize=14)
    ax.set_title("Task-normalized token effect", fontsize=19, fontweight="semibold")
    _save_themed(fig, output, "advanced_token_effects", theme)


def cost_time_analysis(task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    usable = [
        row for row in task_rows
        if row.get("valid_runs", 0) > 0
        and _finite(row.get("cost_usd"))
        and _finite(row.get("wall_seconds"))
    ]
    if not usable:
        return [], math.nan
    costs = np.asarray([float(row["cost_usd"]) for row in usable], dtype=float)
    times = np.asarray([float(row["wall_seconds"]) for row in usable], dtype=float)
    corr = float(np.corrcoef(costs, times)[0, 1]) if len(usable) > 1 else math.nan

    rows: list[dict[str, Any]] = []
    for arm in _ordered_arms({str(row["arm"]) for row in usable}):
        subset = [row for row in usable if str(row["arm"]) == arm]
        rows.append({
            "arm": arm,
            "label": LABEL.get(arm, arm),
            "mean_cost_usd": float(np.mean([float(row["cost_usd"]) for row in subset])),
            "mean_wall_seconds": float(np.mean([float(row["wall_seconds"]) for row in subset])),
            "n_tasks": len(subset),
            "task_level_cost_time_correlation": corr,
        })

    for row in rows:
        row["pareto_efficient"] = not any(
            other is not row
            and float(other["mean_cost_usd"]) <= float(row["mean_cost_usd"])
            and float(other["mean_wall_seconds"]) <= float(row["mean_wall_seconds"])
            and (
                float(other["mean_cost_usd"]) < float(row["mean_cost_usd"])
                or float(other["mean_wall_seconds"]) < float(row["mean_wall_seconds"])
            )
            for other in rows
        )
    return rows, corr


def _plot_cost_time(output: Path, rows: list[dict[str, Any]], corr: float, theme: str) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    _style_axes(fig, ax, theme)
    arms = _ordered_arms([str(row["arm"]) for row in rows])
    colors = _treatment_colors(arms)
    by_arm = {str(row["arm"]): row for row in rows}
    for arm in arms:
        row = by_arm[arm]
        ax.scatter(
            float(row["mean_cost_usd"]),
            float(row["mean_wall_seconds"]),
            s=150,
            color=colors[arm],
            edgecolors=_theme(theme)["edge"],
            linewidths=1.2,
            label=LABEL.get(arm, arm),
            zorder=3,
        )
        ax.annotate(
            LABEL.get(arm, arm),
            (float(row["mean_cost_usd"]), float(row["mean_wall_seconds"])),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10.5,
            color=_theme(theme)["text"],
        )

    frontier = sorted(
        [row for row in rows if bool(row["pareto_efficient"])],
        key=lambda row: float(row["mean_cost_usd"]),
    )
    if len(frontier) >= 2:
        ax.plot(
            [float(row["mean_cost_usd"]) for row in frontier],
            [float(row["mean_wall_seconds"]) for row in frontier],
            linestyle="--",
            linewidth=2.0,
            color=_theme(theme)["muted"],
            label="Pareto frontier",
            zorder=2,
        )
    ax.set_xlabel("Mean OpenRouter cost per task (USD)", fontsize=14)
    ax.set_ylabel("Mean wall-clock time per task (s)", fontsize=14)
    ax.set_title("Cost–time efficiency frontier", fontsize=19, fontweight="semibold")
    if math.isfinite(corr):
        ax.text(
            0.02,
            0.98,
            f"Task-level Pearson r = {corr:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color=_theme(theme)["text"],
        )
    ax.legend(fontsize=10.5)
    _save_themed(fig, output, "advanced_cost_time", theme)


def write_advanced_analysis(output: Path, task_rows: list[dict[str, Any]]) -> list[str]:
    generated: list[str] = []
    effects = paired_effects(task_rows)
    treatment = treatment_regression(task_rows)
    factorial = factorial_regression(task_rows)
    pca_scores, pca_loadings, explained = pca_analysis(task_rows)
    clusters = cluster_analysis(task_rows, pca_scores)
    corr_features, corr = correlation_analysis(task_rows)
    cost_time_rows, cost_time_corr = cost_time_analysis(task_rows)

    tables = {
        "advanced_paired_effects.csv": effects,
        "advanced_treatment_regression.csv": treatment,
        "advanced_factorial_regression.csv": factorial,
        "advanced_pca_scores.csv": pca_scores,
        "advanced_pca_loadings.csv": pca_loadings,
        "advanced_clusters.csv": clusters,
        "advanced_cost_time.csv": cost_time_rows,
    }
    for name, rows in tables.items():
        if rows:
            _write_csv(output / name, rows)
            generated.append(name)

    if explained:
        variance_rows = [
            {"component": index + 1, "explained_variance_percent": 100.0 * value}
            for index, value in enumerate(explained)
        ]
        _write_csv(output / "advanced_pca_variance.csv", variance_rows)
        generated.append("advanced_pca_variance.csv")

    if corr_features:
        corr_rows = [
            {"feature": feature, **{
                other: float(corr[i, j]) for j, other in enumerate(corr_features)
            }}
            for i, feature in enumerate(corr_features)
        ]
        _write_csv(output / "advanced_correlations.csv", corr_rows)
        generated.append("advanced_correlations.csv")

    for theme in ("light", "dark"):
        _plot_effects(output, effects, theme)
        _plot_pca(output, pca_scores, pca_loadings, explained, theme)
        _plot_scree(output, explained, theme)
        _plot_loadings(output, pca_loadings, theme)
        _plot_clusters(output, clusters, theme)
        _plot_correlations(output, corr_features, corr, theme)
        _plot_cost_time(output, cost_time_rows, cost_time_corr, theme)

    for stem in ("advanced_token_effects", "advanced_pca_biplot", "advanced_pca_scree", "advanced_pca_loadings", "advanced_clusters", "advanced_correlations", "advanced_cost_time"):
        if (output / f"{stem}.light.png").exists():
            generated.extend([
                f"{stem}.light.png", f"{stem}.dark.png",
                f"{stem}.light.svg", f"{stem}.dark.svg",
            ])
    return generated
