# Forge Bench experiment designs

Experiment designs are versioned YAML files. They define the scientific factors
that should stay fixed together: task selection, model identities, OpenRouter
upstream routing, reasoning mode, Hermes turn budget, treatments, block count,
randomization seed, and analysis mode.

Runtime concerns such as Docker-vs-local Hermes, timeouts, output paths, and
whether SWE-bench grading is skipped remain CLI options.

## Run a design

Preview the exact randomized order without making model calls:

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

Forge Bench copies the source YAML to `design.yaml` in the result directory,
so the experiment remains auditable even if the repository design changes
later.

## Version 1 schema

```yaml
version: 1
name: example
description: Human-readable purpose.

selection:
  dataset: verified
  split: test
  difficulty: medium
  sample_size: 5
  # Optional. When present, these exact tasks are used.
  instance_ids:
    - django__django-13516

provider:
  api: openrouter
  upstream: relace
  # Reject the design if any model overrides the common upstream.
  require_same_upstream: true

reasoning: none

# Optional randomized agent-budget factors. Each list is crossed with every
# model, treatment, and selected task inside every randomized block.
factors:
  max_turns: [50, 100]
  budget_warning_ratio: [null, 0.75]

analysis_mode: advanced

models:
  - key: model-a
    label: Model A
    model: provider/model-a
  - key: model-b
    label: Model B
    model: provider/model-b
    # Optional per-model overrides are supported. A design with
    # require_same_upstream: true rejects a differing upstream.
    # upstream_provider: relace
    # reasoning: none

treatments:
  - baseline
  - caveman
  - ponytail
  - caveman_ponytail

randomization:
  mode: full-factorial-within-block
  blocks: 1
  seed: 260920
```

`factors.max_turns` accepts one or more positive integer levels.
`factors.budget_warning_ratio` accepts one or more levels, where `null` means
no model-visible iteration-budget reminder and a floating-point value strictly
between 0 and 1 means Hermes injects its one-time checkpoint reminder at that
fraction of the assigned iteration budget.

The historical root-level `max_turns` and `budget_warning_ratio` fields remain
supported as single-level shorthand, so existing experiment YAMLs are unchanged.
When a `factors` entry is present it takes precedence over the corresponding
root-level shorthand.

## Randomization

A block is the complete Cartesian product:

```text
models × treatments × max_turns × budget_warning_ratio × selected tasks
```

Every cell appears exactly once in each block. Forge Bench constructs the whole
block first and then performs one seeded shuffle over all cells. Model order,
treatment order, and task order are therefore interleaved rather than running
one model or treatment as a batch.

Additional blocks receive independently derived seeds. The master seed,
per-block seeds, global run index, and within-block position are all written to
`run_plan.csv` and `metadata.json`.

Repeated observations are averaged within model × treatment × max_turns ×
budget_warning_ratio × task before across-task confidence intervals are
calculated. This preserves task as the independent experimental unit while
keeping the two agent-budget factors separate.

## Analysis hierarchy

For multi-model designs the primary inferential unit is the selected task, not
an API call, tool event, or randomized run. If `blocks > 1`, Forge Bench first
averages repeats within each task × model × treatment × max_turns ×
budget_warning_ratio cell and then performs the across-task analysis.

When either agent-budget factor has more than one level, Forge Bench additionally
writes `factorial_task_summary.csv`, `factorial_condition_summary.csv`, and
`factorial_paired_effects.csv`. The paired-effects table contains task-paired
contrasts for reminder level, iteration budget, and their difference-in-
differences interaction. Legacy treatment plots remain marginal summaries across
factor levels and should not replace the factor-aware tables for inference.

The root report uses a task-blocked model × treatment analysis and shared-scale
model subpanels. Per-model reports preserve the original treatment analysis.
Multivariate and trajectory outputs preserve the same hierarchy: raw traces are
kept, repeated trajectories are averaged within task, and across-task summaries
are computed afterward.

## Multi-model reports

The root result directory keeps the single randomized experiment record:

- `runs.csv` / `runs.json`
- `run_plan.csv`
- `model_task_summary.csv`
- `model_treatment_summary.csv`
- `model_pairwise_effects.csv` for two-model designs
- `report-light.html` / `report-dark.html`

Each model also receives an independent treatment analysis under:

```text
models/<model-key>/
```

The root report also writes both vertical-bar and horizontal-bar shared-scale
model subpanels for the core metrics. The corresponding machine-readable files
include `model_treatment_regression_2x4.csv`,
`model_caveman_ponytail_regression_2x2x2.csv`,
`multivariate_*.csv`, `trajectory_*.csv`, and `workflow_*.csv`.


This prevents model identity from being accidentally treated as a repeated
observation of the same treatment. Advanced Caveman/Ponytail analyses therefore
remain within-model, while the root paired-effects table compares models on the
same task/treatment cells.
