# Forge Bench architecture

## Current authority

Forge Bench owns the **scientific benchmark design**:

- task selection and pinned task identity;
- model/provider/treatment factors;
- iteration-budget factors;
- randomized complete blocks and seeds;
- reproducibility metadata;
- result normalization for scientific analysis;
- statistical summaries, figures, and reports.

A runtime backend must consume the Forge plan. It must not silently resample,
recross, or redefine the experiment.

The authoritative cell ordering is written to `run_plan.csv`.

## Target execution architecture

The migration target is:

```text
Forge Bench
  experiment design / randomization / provenance / analysis
        |
        v
Harbor
  tasks / trials / agent execution / verification / retries / artifacts
        |
        +--> Docker (local/default)
        +--> Modal (optional burst cloud)
        +--> OpenShell (optional hardened environment, future adapter)

Operational lifecycle -> OpenTelemetry
Agent trajectory/evidence -> Harbor/ATIF-compatible artifacts
```

Harbor owns commodity execution and evaluation. Forge Bench does not need a
second container-orchestration or benchmark-grading system once the migration is
complete.

## Migration state

The migration is intentionally staged.

### Implemented in the Harbor-plan foundation

Forge Bench can project its already-randomized `run_plan.csv` cells into
`harbor_plan.json`. The projection:

- preserves one Harbor trial intent per Forge cell;
- preserves exact run order and block assignment;
- preserves model, provider, treatment, max-turns, reminder ratio, and task;
- assigns a stable content-derived `forge_cell_id`;
- maps the pinned Forge SWE-bench Verified source to Harbor's
  `swe-bench/swe-bench-verified` dataset;
- records the Harbor API baseline used when the projection was designed.

The projection is deliberately marked `executable: false`.

### Not implemented yet

The current `main` execution path still uses the legacy Forge Bench
Hermes/subprocess/Docker machinery. Follow-up PRs will add:

1. a Harbor Hermes agent adapter and Harbor-result -> Forge observation
   normalization;
2. real SWE-bench execution/grading through Harbor;
3. OpenTelemetry correlation and Modal selection through Harbor;
4. retirement of the legacy execution harness;
5. optional OpenShell support as a Harbor environment rather than a competing
   Forge runtime.

Until those PRs merge, `harbor_plan.json` is a checked migration contract and
provenance artifact, not a runnable benchmark configuration.

## Why Forge does not use Harbor's normal Cartesian expansion directly

A Forge design is randomized *before* execution. Each cell couples:

```text
block x model x treatment x max_turns x budget_warning_ratio x task
```

Harbor's normal JobConfig is excellent for generating Cartesian trials, but
letting it independently expand the experiment would lose Forge's exact
randomized cell identity/order. The migration therefore preserves the Forge plan
first and lets the Harbor adapter materialize those explicit cells.

This keeps the scientific unit and randomization under Forge Bench while Harbor
owns execution.

## OpenShell

OpenShell is optional. It may later provide a hardened Harbor execution
environment, but Forge Bench must not depend on it for its core design,
statistics, or Harbor integration.

## Modal

Modal is likewise an execution choice beneath Harbor. A scientifically identical
Forge design should be able to execute through Docker or Modal without changing
the Forge factors or analysis.

## Observability

OpenTelemetry is planned for operational spans and resource/runtime telemetry.
It is complementary to, not a replacement for, immutable benchmark artifacts
and agent trajectories. Stable Forge cell IDs and Harbor trial IDs will be the
join keys.
