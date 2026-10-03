# OpenShell runtime foundation

Forge Bench can use NVIDIA OpenShell as an isolation backend for future benchmark
workloads without making OpenShell a hard Python dependency of the existing SWE-bench
harness.

The current foundation is intentionally task-agnostic. Existing Docker and local Hermes
runs are unchanged. The module in `src/forge_bench/openshell_runtime.py` owns only:

- generation of explicit OpenShell policy YAML;
- sandbox creation/deletion;
- upload/download;
- non-interactive command execution;
- effective-policy capture;
- OpenShell log capture; and
- explicit provider attachment at sandbox creation.

Robot task semantics, credentials, model-provider policy, and scoring belong to later
benchmark adapters.

## Design

```text
Forge Bench control process
    |
    +-- writes per-run OpenShell policy
    +-- creates sandbox
    +-- uploads only declared task inputs
    +-- execs the benchmark workload
    +-- downloads declared outputs/evidence
    +-- captures effective policy + OpenShell logs
    +-- deletes sandbox
```

The runtime uses the official `openshell` CLI rather than linking the OpenShell Python
SDK. This keeps existing Forge Bench installs reproducible and lets OpenShell remain an
optional execution dependency.

## Policy defaults

`OpenShellPolicy` defaults to:

- policy schema version 1;
- workdir included as writable;
- `/tmp` writable;
- Landlock as a hard requirement;
- workload user/group `sandbox`; and
- no outbound network access unless a caller supplies explicit `NetworkEndpoint` entries.

Provider names can be attached through `OpenShellSandboxSpec.providers`; OpenShell then
adds the provider-derived credential and network rules to the effective policy for newly
started sandbox processes. Forge Bench never copies provider secret values into the task
bundle.

Network endpoints require absolute executable paths. The current Forge Bench helper
intentionally supports only REST, WebSocket, and raw TCP endpoints. REST/WebSocket entries
require explicit method/path rules and default to `enforcement: enforce`; GraphQL, MCP, and
JSON-RPC are rejected until Forge Bench has protocol-specific rule types instead of silently
emitting invalid REST-shaped policy.

OpenShell's policy schema represents process identities as strings, including numeric IDs
(for example `"10000"`). `OpenShellPolicy` accepts integers for convenience but serializes
them to the schema-required string form.

The robot benchmark adapter will later use this to grant only its standalone broker client
access to the host-side robot broker, while the robot SDK, serial device, cameras, and
unrelated host files remain outside the sandbox.

## Lifecycle

The runtime creates a detached retained sandbox with a small idle canonical process, then
uploads inputs and executes managed commands using `openshell sandbox exec`. This follows
OpenShell's documented constraint that upload cannot yet be combined with a trailing main
command at sandbox creation.

For automated commands, `--no-login-shell` is used by default to avoid user shell startup
files changing output or side effects.

Every caller should use `delete()` in cleanup paths. OpenShell's delete request is
asynchronous, so the runtime does not treat "deletion accepted" as completion: it polls
`sandbox get --output json` until the sandbox is actually absent and fails on unexpected
probe errors or timeout.

## Validation

CI runs:

```bash
uv run python scripts/test_openshell_runtime.py
```

The contract test does not require an OpenShell installation. It verifies the generated
policy structure and exact CLI lifecycle commands with a fake command runner.

A real local OpenShell installation is still required before this runtime is promoted into
the default or selectable paid benchmark execution path. The CLI adapter has been checked
against the current OpenShell command surface (`sandbox create`, `upload`, `exec`,
`download`, `sandbox get --policy-only`, `logs`, and `sandbox delete`), but local
validation must still verify the installed CLI/gateway version, compute driver, sandbox image,
effective policy, network denials, upload/exec/download behavior, and cleanup.
