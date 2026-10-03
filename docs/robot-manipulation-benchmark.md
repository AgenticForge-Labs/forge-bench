# Physical robot manipulation benchmark

This runner measures an agent performing a real SO-ARM101 object-to-container task while
keeping reasoning isolated from the physical workstation.

The implementation composes existing owners rather than creating another robot stack:

```text
Hermes in OpenShell
    |
    | python3 robotctl.py ...
    v
OpenShell network policy
    |
    v
host.openshell.internal:8765
    |
    v
soarm101-broker
    |
    v
bounded soarm101 agent capabilities
    |
    v
SO-ARM101 Motion SDK + named cameras
    |
    v
physical robot
```

Forge Bench owns the experiment lifecycle, evidence collection, and scoring. The Motion SDK
owns calibration, motion validity, human authorization, robot safety, and camera acquisition.
OpenShell owns sandbox filesystem/process/network isolation.

## Dependencies

The current stacked development sequence is:

1. SO-ARM101 Motion SDK PR #76: bounded agent capability contract.
2. SO-ARM101 Motion SDK PR #77: HTTP broker + sandbox client.
3. Forge Bench PR #15: generic OpenShell runtime.
4. This physical benchmark adapter.

Do not describe this benchmark as available from either repository's `main` until those
dependencies are merged.

## Human authorization remains outside the benchmark

A human must inspect the physical scene, keep power/e-stop access available, and authorize the
bounded session before running the benchmark:

```bash
soarm101 agent arm --minutes 30
```

The benchmark runner never arms, disarms, or relaxes the robot. It starts the trusted broker,
checks `/v1/capabilities`, and refuses to create the sandbox when the human authority lease
is absent or expired.

The completion contract requires an `overhead` named camera. The `wrist` camera is useful
but optional for fine manipulation.

## OpenShell provider setup

The sandbox needs model-provider access and robot-broker access for different reasons.

Model-provider credentials should be supplied by an OpenShell provider attachment. Do not
copy an OpenRouter API key into the task directory. The provider profile must name the **real
executable** that Hermes uses to make its provider request. If Hermes is implemented by a
Python interpreter in your image, use the interpreter's real path rather than a wrapper or
symlink.

The runner passes the selected provider name to `openshell sandbox create --provider ...`.
Inspect the resulting effective policy before physical use:

```bash
openshell sandbox get <sandbox-name> --policy-only
```

The host-side independent final-image judge is separate from the sandbox and currently uses
`OPENROUTER_API_KEY` from the Forge Bench host environment. Use `--no-judge` for an
evidence-only dry run.

### Python robotctl limitation

The current broker client is a Python script. OpenShell sees the real Python executable as the
network caller, not `robotctl.py` itself. The benchmark policy therefore allows configured
Python interpreter paths to reach only the broker host/port and its explicit bounded REST
routes.

This means Python code in the sandbox could make the same bounded broker requests directly.
It still cannot gain unrestricted SDK/serial/camera/calibration access because the broker
does not expose those capabilities. A future native `robotctl` executable could make
client-executable identity itself an additional restriction.

## Run one task

From the Motion SDK checkout containing PR #77, install the current branch and make sure the
named cameras and calibration are configured. Then arm the physical robot.

From the Forge Bench checkout containing this PR:

```bash
uv sync

export OPENROUTER_API_KEY=...   # host-side independent judge only

uv run forge-bench-robot \
  --output benchmark-results/robot-object-to-container-001 \
  --robotctl ../soarm101-motion-sdk/src/soarm101_motion/robotctl.py \
  --provider openrouter \
  --model deepseek/deepseek-v4.1-flash-20260910
```

The default task and skill are:

```text
robot_tasks/object-to-container/TASK.md
robot_tasks/skills/soarm101-robot-camera/SKILL.md
```

The default sandbox image is `nousresearch/hermes-agent:latest`. For reproducible benchmark
runs, pin an immutable image digest after local validation instead of relying on `latest`.

If the Python interpreter in the image differs from the common defaults, repeat
`--robot-network-binary` with the actual path reported inside the sandbox, for example:

```bash
--robot-network-binary /usr/local/bin/python3.12
```

## What the agent receives

The runner uploads only:

- `TASK.md`
- `SKILL.md`
- `robotctl.py`

It does not upload the Motion SDK checkout, calibration files, workstation camera profile,
serial devices, camera devices, SSH credentials, Docker socket, or unrelated host files.

The agent is instructed to use Hermes `vision_analyze` on local files created by
`robotctl.py capture ...`. Hermes' vision tool accepts local image paths and on a
vision-capable main model sends the pixels back to that model as multimodal content.

## Completion and independent scoring

The agent must write `task-result.json`.

A claimed success is not sufficient. Forge Bench requires:

1. `task_status == "finished"`;
2. `evidence_camera == "overhead"`;
3. the cited downloaded image's SHA-256 equals the **latest overhead
   `capture_evidence` SHA** emitted by the trusted host broker; and
4. an independent host-side vision judge sees only that final overhead image and confirms
   that the object is clearly inside the container.

If any of the first three checks fail, the image is never sent to the semantic judge.

The independent judge uses the same strict visual definition as the task: rim contact,
outside overlap, beside-container placement, or an ambiguous/occluded view are not success.

With `--no-judge`, the first three checks still run but `score.success` remains null rather
than inventing a semantic success result.

## Evidence

Each run directory retains:

```text
inputs/
  TASK.md
  SKILL.md
  robotctl.py

openshell-policy.yaml
openshell-effective-policy.yaml
openshell-effective-policy-final.yaml
openshell-logs.txt

broker-events.jsonl
broker-stdout.txt
broker-stderr.txt
hermes-stdout.jsonl
hermes-stderr.txt

observations/
task-result.json
metadata.json
score.json
```

`metadata.json` records SHA-256 values for every supplied task/client artifact plus the
broker capabilities observed before the run.

`broker-events.jsonl` records bounded actions. Camera responses include a separate trusted
`capture_evidence` event with camera name, request ID, timestamp, and SHA-256 but not image
bytes.

`openshell-logs.txt` is retained so denied filesystem/network behavior and attempted escape
paths can be analyzed later rather than discarded.

## Physical validation order

Before an autonomous object move, validate the assembled path incrementally:

1. OpenShell create/upload/exec/download with no robot motion.
2. Broker `health`, `capabilities`, `state`, and image capture from inside OpenShell.
3. Known `agent_*` pose movement while supervised.
4. Small single-joint and tool/world jog requests.
5. Gripper open/close.
6. One complete object-to-container run.

The unresolved broader Cartesian-motion quality concerns in the Motion SDK remain relevant:
software authorization does not substitute for physical validation. Begin with conservative
motions and a clear workspace.
