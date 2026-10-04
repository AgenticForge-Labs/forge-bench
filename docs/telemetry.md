# Telemetry and raw archive contract

Forge Bench treats high-resolution execution evidence as a scientific data
product, not merely as runtime logging.

The governing rule is:

> Capture the richest practical raw evidence once, preserve it immutably, and
> rebuild temporal features, benchmark summaries, figures, and share packages
> from that archive.

## Authority boundaries

The telemetry archive complements existing authorities rather than replacing
them:

- Forge Bench owns experimental identity, randomized cells, normalized
  scientific observations, and analysis.
- Harbor owns trial execution, verification, agent trajectories/ATIF, and
  trial artifacts.
- The Forge telemetry archive owns the durable, time-correlated raw measurement
  record used for later temporal and multivariate analysis.
- OpenTelemetry is planned as a standards-compatible correlation/export layer.
  It is not the only copy of scientific telemetry and is not implemented by the
  archive-contract foundation.
- Derived tables and public/share reports are rebuildable products, not repair
  targets or primary evidence.

## Archive tiers

A telemetry archive has three deliberately different tiers:

```text
<experiment>/
├── archive_manifest.json
├── raw/
│   └── cells/
│       └── <forge_cell_id>/
├── derived/
└── share/
```

`raw/` is the immutable evidence tier once sealed. The manifest hashes every
raw file and records its byte size and, for JSONL/NDJSON, record count.

`derived/` is for rebuildable Parquet/time-series/features in later work.

`share/` is for compact CSVs, figures, reports, and privacy-aware exports.
Changes under `derived/` or `share/` do not invalidate raw-archive integrity.

An archive may be sealed as:

- `complete`: the expected run finished and raw evidence was finalized;
- `partial`: execution was interrupted or a channel was incomplete, but the
  available raw evidence is intentionally retained and hashed.

An open archive is still being written and therefore is not integrity-sealed.

## Versioned raw records

The initial telemetry schema defines stable contracts for:

- `TelemetryEvent`
- `MetricSample`
- `ProcessSample`
- `SystemSample`
- `GpuSample`
- `ArtifactReference`
- `TelemetryCapabilities`

Every time-varying record carries a `TelemetryContext` and a `TimePoint`.

The context keeps the scientific joins explicit:

```text
experiment_id
forge_cell_id
run_index
repeat
agent / agent_version
model
task
environment
factors             # arbitrary design factors preserved without schema churn
```

Resource and metric records also carry an explicit measurement `source` such
as `procfs`, `psutil`, `cgroup`, or `nvml`. This prevents values collected
through different mechanisms from being treated as interchangeable merely
because they share a metric name.

The clock representation stores both wall-clock and monotonic time:

```text
wall_time_unix_ns
monotonic_ns
experiment_elapsed_ns
cell_elapsed_ns
```

Wall time permits cross-process/distributed correlation. Monotonic elapsed time
is authoritative for duration on one machine and avoids wall-clock adjustment
errors.

## Missing telemetry is data

A missing measurement must never be silently represented as zero.

`TelemetryCapabilities` records whether each channel is:

- `captured`
- `unavailable`
- `not_applicable`
- `error`

Every non-captured capability includes a reason. This is especially important
when the same experiment is run locally and on Modal, where host-level visibility
can differ.

## Capture profiles

The schema reserves three explicit profiles:

- `maximal`: preferred research/archive mode; capture every practical channel.
- `standard`: lower-overhead general benchmark capture.
- `minimal`: essential lifecycle/provenance only.

This PR defines those meanings at the contract level. Sampling frequency and
which concrete collectors populate each profile are implemented in later PRs.

## Integrity

`archive_manifest.json` records the experiment identity, schema versions,
capture profile, archive state, raw file hashes, raw byte count, optional JSONL
record counts, and sealing notes.

`verify_archive()` detects:

- missing raw files;
- raw files added after sealing;
- SHA-256 changes;
- byte-size changes;
- JSONL record-count changes.

The integrity root intentionally excludes derived/share products so analytical
methods and presentation can evolve without mutating raw evidence.

## Durable lifecycle recorder

Harbor-backed execution now uses an append-only lifecycle recorder by default.
Each experiment writes `raw/experiment-events.jsonl`; each explicit Forge cell
writes its own `raw/cells/<forge_cell_id>/events.jsonl`. Per-cell journals avoid
a single contended event file when Harbor trials execute concurrently.

Each JSONL row is a journal envelope containing:

```text
journal_schema_version
sequence
persisted_wall_time_unix_ns
record
```

The record is one of the versioned telemetry contracts from this document.
Sequence numbers are strictly increasing within each journal and are validated
when a journal is reopened.

Lifecycle journals currently use **uncompressed JSONL with fsync after every
record**. That choice is deliberate: these are low-frequency control/lifecycle
events and durability is more important than compression. High-frequency CPU,
RAM, container, GPU, model, and tool streams will use their own buffering and
storage strategy in later PRs rather than weakening lifecycle durability.

If a previous process left a torn final JSONL line, reopening does not silently
discard it. The incomplete bytes are preserved under a raw `recovery/`
directory with their SHA-256 identity, then the live journal is truncated back
to the last complete newline before appending resumes. Invalid complete journal
rows fail closed rather than being rewritten.

Current lifecycle events include:

```text
experiment.started
experiment.completed / experiment.interrupted

cell.queued
cell.started
harbor.trial.create.started
harbor.trial.create.completed
harbor.trial.run.started
harbor.trial.run.completed
forge.result.normalize.started
forge.result.normalize.completed
cell.completed / cell.failed
```

These events describe boundaries Forge can observe directly. Environment setup,
agent subphases, verifier internals, model calls, tool calls, and artifact
collection are intentionally not fabricated from outside Harbor; those deeper
events belong to the later Harbor/ATIF/model-tool integration layer.

Recorder failures are **fail-open for benchmark execution**. Forge records a
telemetry issue when possible, continues the benchmark, and seals the telemetry
archive `partial` if the recorder was degraded. A Python exception or
cancellation during the benchmark also seals available evidence `partial`
before the original exception is re-raised.

Real `forge-bench-harbor` execution enables this lifecycle archive by default.
The CLI supports `--telemetry-dir`, `--experiment-id`,
`--telemetry-profile`, and `--no-telemetry`. Materialization-only runs do not
create a telemetry archive because no trial executes.

## Planned layers

This foundation deliberately does not collect resource telemetry yet. Dependent
work should proceed linearly:

1. local process/CPU/RAM sampling;
2. container/disk/network sampling;
3. NVIDIA GPU sampling;
4. OpenTelemetry correlation;
5. model/tool/ATIF temporal integration;
6. local OTel Collector profile;
7. Modal telemetry parity;
8. canonical Parquet temporal datasets;
9. temporal feature engine and refreshed multivariate/share outputs.

Old PR #13 contains useful trajectory-analysis concepts, but its data-capture
assumptions predate this archive. Those concepts should be salvaged later onto
the canonical temporal dataset rather than merging the stale branch directly.
