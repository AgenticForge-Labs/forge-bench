# AGENTS.md

## Purpose

This repository is Forge Bench, a reproducible benchmark harness for comparing AI coding-agent strategies on SWE-bench tasks. Preserve experimental reproducibility: do not silently change models, providers, task panels, randomization, treatment definitions, turn budgets, or analysis rules.

## Live-run inspection helpers

Agents should use the reusable live monitor instead of guessing timestamped run-directory names or manually parsing partial output.

After pulling the latest code and running `uv sync`, the main command is:

```bash
uv run forge-bench-live
```

With no arguments, it searches `benchmark-results/` and prefers the newest incomplete experiment. It reads `runs.partial.json` while a benchmark is active, falls back to final `runs.json`, and can also reconstruct progress from per-run `runs/*/result.json` files.

### Check current status

```bash
uv run forge-bench-live
```

This prints:
- auto-detected run directory;
- completed / expected cells and percent complete;
- valid and resolved counts;
- outcome breakdown including `resolved`, `evaluated_unresolved`, `no_patch`, and `invalid`;
- cumulative and mean cost;
- mean agent wall time;
- mean total tokens.

For machine-readable output:

```bash
uv run forge-bench-live --json
```

This is the preferred interface when another agent needs to reason about the current run programmatically.

### Generate one compact live figure

```bash
uv run forge-bench-live --dashboard
```

This writes:

```text
<detected-run>/live-dashboard.png
```

The 2x2 dashboard contains:
1. experiment completion;
2. current outcome counts;
3. mean cost by model × treatment;
4. mean agent wall time by model × treatment.

The treatment categories and model series are read from `run_plan.csv`; the dashboard is not hard-coded to four treatments or two models. Figure width expands with the number of planned treatments. Each completed model × treatment cell shows its current `n`.

The command prints the exact PNG path. If the surrounding agent/chat system supports file upload or attachment, attach that PNG directly. Do not regenerate charts manually unless the live helper is missing a required view.

You can choose an explicit output path:

```bash
uv run forge-bench-live --dashboard --output /tmp/forge-bench-live.png
```

### Inspect a specific run

```bash
uv run forge-bench-live \
  --run benchmark-results/deepseek-v4-v41-relace-YYYYMMDD-HHMMSS
```

By default auto-discovery prefers an incomplete run. To inspect the newest run regardless of completion:

```bash
uv run forge-bench-live --latest-finished
```

## Python API for agents

The same functionality is importable, so agents can build small analyses without shell parsing:

```python
from pathlib import Path
from forge_bench.live_status import (
    find_latest_run,
    load_partial_results,
    make_dashboard,
    planned_factors,
    summarize_run,
)

run = find_latest_run(Path("benchmark-results"))
summary = summarize_run(run)
rows = load_partial_results(run)
treatments, models, model_labels = planned_factors(run)
png = make_dashboard(run)

print(summary)
print(treatments, models)
print(png)
```

Prefer these helpers over duplicating run-discovery or partial-result parsing logic.

## Interpretation rules for partial runs

Live summaries are descriptive only. A randomized factorial experiment is incomplete until all planned cells in the block finish.

Do not collapse model and treatment together when looking for a treatment effect during a multi-model run. A treatment may help one model and hurt another; averaging across models can make a real model-specific effect appear to disappear. The live dashboard therefore displays model × treatment cells rather than separate model-only and treatment-only marginal means.

Do not treat partial model × treatment means as final treatment effects because the randomized execution order can leave temporarily unbalanced task coverage. The displayed `n` values are there to make that imbalance visible.

Do not infer that a treatment effect has vanished merely because one live metric is flat. Cost, wall time, tokens, API calls, and correctness can move differently. Use `uv run forge-bench-live --json` or the Python API for additional metrics if needed, and use the final Forge Bench report for inferential comparisons.

Do not discard unresolved valid runs from efficiency summaries. Forge Bench intentionally keeps valid unresolved runs as efficiency observations.

Distinguish outcome mechanisms:
- `resolved`: valid run and SWE-bench task resolved;
- `evaluated_unresolved`: non-empty patch was graded but did not resolve;
- `no_patch`: valid agent run produced no usable patch, so no SWE-bench evaluation was run;
- `unresolved_other`: valid unresolved run that does not fit the preceding categories;
- `invalid`: run failed Forge Bench validity checks.

For repeated experiments, do not treat stochastic repeats as additional independent tasks. Aggregate repeats within the complete experimental cell before across-task inference, matching the main Forge Bench reporting design. When iteration-budget factors are active, the complete cell is task × model × treatment × max_turns × budget_warning_ratio.

Do not collapse max_turns or budget_warning_ratio levels into stochastic repeats. They are randomized experimental factors. Use the factor-aware CSVs for their main effects and interactions; legacy treatment summaries are marginal across those factor levels.

## Normal experiment commands

Preview a YAML design without paid model calls:

```bash
uv run forge-bench \
  --design designs/deepseek-v4-v41-relace.yaml \
  --plan-only
```

Run it:

```bash
uv run forge-bench \
  --design designs/deepseek-v4-v41-relace.yaml
```

Do not override scientific design fields from the command line unless the user explicitly asks to change the experiment.
