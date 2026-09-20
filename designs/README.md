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
max_turns: 50
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

## Randomization

A block is the complete Cartesian product:

```text
models × treatments × selected tasks
```

Every cell appears exactly once in each block. Forge Bench constructs the whole
block first and then performs one seeded shuffle over all cells. Model order,
treatment order, and task order are therefore interleaved rather than running
one model or treatment as a batch.

Additional blocks receive independently derived seeds. The master seed,
per-block seeds, global run index, and within-block position are all written to
`run_plan.csv` and `metadata.json`.

Repeated observations are averaged within model × treatment × task before
across-task confidence intervals are calculated.

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

This prevents model identity from being accidentally treated as a repeated
observation of the same treatment. Advanced Caveman/Ponytail analyses therefore
remain within-model, while the root paired-effects table compares models on the
same task/treatment cells.
