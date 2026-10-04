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

### Implemented

Forge Bench projects its already-randomized `run_plan.csv` cells into
`harbor_plan.json`. The projection:

- preserves one Harbor trial intent per Forge cell;
- preserves exact run order and block assignment;
- preserves model, provider, treatment, max-turns, reminder ratio, and task;
- assigns a stable content-derived `forge_cell_id`;
- maps the pinned Forge SWE-bench Verified source to Harbor's
  `swe-bench/swe-bench-verified` dataset;
- pins the compatibility contract to Harbor 0.23.0 and Hermes v2026.9.14;
- is executable only when an explicit Harbor dataset mapping exists.

The opt-in `forge-bench-harbor` runner materializes those intents as explicit
Harbor `TrialConfig` objects in Forge's randomized order and executes them
sequentially while this boundary is being validated.

Forge does **not** maintain a competing Hermes integration. `ForgeBenchHermes`
subclasses Harbor's first-party Hermes adapter. Harbor owns Hermes installation,
task-environment execution, session export, ATIF conversion, and verification.
The Forge specialization owns only experimental treatment installation
(Caveman/Ponytail), pinned provider routing, turn/reminder factors, and patch
evidence needed to create a Forge observation.

Harbor `TrialResult` objects are normalized into the existing Forge `Result`
contract so the scientific reporting layer does not depend on which execution
backend produced the observation.

### Not implemented yet

The normal `forge-bench` command still uses the legacy Forge Bench
Hermes/subprocess/Docker machinery. Follow-up PRs will:

1. cut SWE-bench execution/grading over to Harbor as the normal runtime;
2. add OpenTelemetry correlation and Modal selection through Harbor;
3. retire the legacy execution harness;
4. optionally reintroduce useful OpenShell hardening as a Harbor environment
   rather than a competing Forge runtime.

The opt-in Harbor path is therefore real but is not yet the default Forge Bench
execution architecture.

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
