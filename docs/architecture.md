# Forge Bench architecture

## Current authority

Forge Bench owns the **scientific benchmark design**:

- task selection and pinned task identity;
- agent/model/provider/treatment factors;
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
`harbor_plan.json`. Agent and model are independent cell dimensions. Baseline
cells use Harbor's native installed-agent registry, so Forge can compare
different Harbor agents against different models without maintaining parallel
agent runtimes.

The projection:

- preserves exact run order and block assignment;
- preserves agent, model, treatment, task, and provenance identity;
- assigns a stable content-derived `forge_cell_id`;
- maps the pinned Forge SWE-bench Verified source to Harbor's
  `swe-bench/swe-bench-verified` dataset;
- pins the current compatibility contract to Harbor 0.23.0;
- records whether each cell uses Harbor-native control semantics or the legacy
  Forge Hermes compatibility path;
- marks unsupported agent/treatment/budget combinations non-executable before
  launch.

The opt-in `forge-bench-harbor` runner materializes explicit Harbor
`TrialConfig` objects and normalizes Harbor `TrialResult` objects into the
Forge `Result` contract. Normalized Harbor observations now retain agent,
agent version, environment, and Harbor trial identity in addition to model and
task outcome.

Existing Hermes Caveman/Ponytail, provider-route pinning, and Forge turn/reminder
budget behavior remain available through `ForgeBenchHermes`. That subclass is
not the general agent architecture; it is selected only when a Hermes experiment
needs those Forge-specific controls. No plugin behavior is ported to Pi,
OpenCode, Codex, or other agents in this stage.

For baseline heterogeneous-agent comparisons, iteration limits currently follow
the selected agent's own defaults. The plan labels those semantics
`agent-default`. Forge budget factors cannot be varied across heterogeneous
agents until a later capability-negotiation layer provides equivalent validated
controls.

### Local/Modal execution layer

The explicit-cell Harbor runner now supports both local Docker and Harbor's
native Modal environment with the same Forge plan. Environment, bounded
concurrency, and optional CPU/RAM/storage/GPU overrides are execution settings,
not scientific factors by default.

Forge does not use Harbor `JobConfig` for this layer. Harbor jobs normally
expand tasks × agents × attempts, which would regenerate the experiment after
Forge has already randomized it. Forge instead materializes the exact
`TrialConfig` list and runs those Harbor trials under a bounded asynchronous
semaphore. Harbor still owns each trial's environment, agent setup, verifier,
trajectory, and artifacts.

Environment changes do not participate in `forge_cell_id`. Each run writes an
execution manifest containing the source plan hash and execution settings, while
normalized results retain the actual environment and requested resource
overrides. This permits Docker/Modal reruns of the same scientific cells without
pretending they are a different experiment.

Software tests validate both environment configurations and Harbor 0.23 resource
fields without starting cloud resources. Real Modal execution additionally
requires `harbor[modal]==0.23.0` and normal Modal authentication; an
authenticated provider smoke is a separate validation gate.

### Not implemented yet

The normal `forge-bench` command still uses the legacy Forge
Hermes/subprocess/Docker machinery. Follow-up PRs will proceed linearly:

1. add agent-aware reporting over completed Harbor multi-agent experiments;
2. add a direct LLM/VLM trial kind for model-only experiments without a coding
   harness confound;
3. add optional self-hosted local/Modal model inference for open LLMs/VLMs,
   including explicit GPU type/runtime provenance;
4. add secure external capability attachments such as the bounded SO-ARM101
   broker without exposing hardware devices to agent containers;
5. add OpenTelemetry correlation and then retire the legacy execution harness
   after parity is demonstrated.

OpenShell remains optional hardening around a capability boundary or future
Harbor environment integration; it is not a second Forge execution architecture.

## Why Forge does not use Harbor's normal Cartesian expansion directly

A Forge design is randomized *before* execution. Each cell couples:

```text
block x agent x model x treatment x max_turns x budget_warning_ratio x task
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

## Telemetry and observability

Forge Bench now defines a versioned raw telemetry/archive contract independently
of any particular observability backend. Raw time-resolved evidence is the
scientific authority; derived time-series/features and public share packages are
rebuildable projections.

The archive preserves stable experiment/cell identity, wall-clock and monotonic
time, explicit telemetry-capability availability, and an integrity manifest over
sealed raw evidence. See `docs/telemetry.md`.

Concrete resource collectors are intentionally layered on top of this contract.
Planned sources include local process/CPU/RAM, container/disk/network, NVIDIA
GPU, model/tool/ATIF events, and equivalent available signals from Modal.

OpenTelemetry is still planned for distributed operational correlation and
optional export. It must not become the sole copy of scientific telemetry.
Stable Forge experiment/cell IDs and Harbor trial IDs will be the join keys.


OpenTelemetry is planned for operational spans and resource/runtime telemetry.
It is complementary to, not a replacement for, immutable benchmark artifacts
and agent trajectories. Stable Forge cell IDs and Harbor trial IDs will be the
join keys.
