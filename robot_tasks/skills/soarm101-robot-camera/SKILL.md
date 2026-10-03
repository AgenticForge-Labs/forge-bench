---
name: forge-bench-soarm101-robot-camera
description: Use the bounded SO-ARM101 broker client and overhead/wrist cameras for an object-to-container benchmark task.
---

# SO-ARM101 benchmark robot + camera skill

This benchmark runs inside an isolated sandbox. The physical robot and cameras are on the
trusted host. Use only the local `robotctl.py` client to request bounded actions.

Do not search for, import, or invoke an unrestricted robot SDK. Do not attempt to arm,
disarm, relax, calibrate, configure, or write servo registers. Human authorization is
established before the run. If the broker reports that authority is absent or expired,
report the task as not finished.

## Client commands

Use:

```bash
python3 robotctl.py capabilities
python3 robotctl.py state
python3 robotctl.py capture overhead
python3 robotctl.py capture wrist
python3 robotctl.py go-pose agent_start_overhead
python3 robotctl.py joint shoulder_pan --delta-deg 5
python3 robotctl.py jog --frame world --x-mm 0 --y-mm 5 --z-mm 0
python3 robotctl.py jog --frame tool --x-mm 0 --y-mm 0 --z-mm 2
python3 robotctl.py gripper open
python3 robotctl.py gripper close
python3 robotctl.py sleep
python3 robotctl.py stop
```

`robotctl.py` writes captures into the local `observations/` directory unless you provide
an explicit output path.

The broker is the authority for allowed actions. A rejected action is evidence that the
requested motion is not currently authorized or safe; do not bypass it.

## Cameras and vision

- `overhead` is the authoritative workspace/status/completion camera.
- `wrist` is the close robot-mounted camera for approach, grasp, edge clearance, and release.

After every capture that matters to your reasoning, call Hermes' `vision_analyze` tool on
the returned local image path. Do not infer scene state from the filename or from intended
robot motion.

Use fresh observations after consequential manipulation when the result is uncertain.

## Motion semantics

Discover current capabilities and state before motion.

Saved `agent_*` poses are coarse repositioning/viewpoint affordances. Prefer them for large
known repositioning and use bounded relative actions for corrections.

Single-joint adjustments change one named joint by a relative angle. Prefer small increments.

World-frame jog is for table/workspace directions. Read the `world_directions` block from
`robotctl.py capabilities`; do not guess raw XYZ signs. Multiply the reported
`model_delta_mm_per_physical_mm` direction vector by the requested physical distance.

Tool-frame jog is from the current gripper/TCP perspective. Tool X/Y/Z rotate with the
gripper and are useful for small approach/retract/lateral corrections.

The broker enforces per-command displacement/height limits and the underlying SDK safety
checks. Successful moves remain holding. `stop` is STOP/HOLD, not torque release.

## Task strategy

Use the overhead image to identify the target object and empty target container. Use the wrist
view when it materially improves fine manipulation. Reason iteratively:

1. observe;
2. choose one bounded action;
3. execute;
4. re-observe when needed;
5. stop if the required state is reached or safe progress is no longer possible.

Do not claim success from intended motion alone.

## Completion contract

The task is finished only when a **fresh overhead capture** clearly shows the target object
inside the target container.

"Inside" means the visible target object is contained within the container's interior
footprint/opening. It is not enough for the object to touch the rim, overlap the outside edge,
sit beside the container, or be too occluded/ambiguous to judge.

Your final overhead image must be the most recent overhead capture used for completion.
The benchmark independently checks its SHA against the trusted broker log and separately
judges the image on the host.

Before finishing, write exactly one JSON object to `task-result.json`.

Success:

```json
{
  "task_status": "finished",
  "evidence_camera": "overhead",
  "evidence_path": "observations/overhead-YYYYMMDD-HHMMSS.jpg"
}
```

Failure:

```json
{
  "task_status": "not_finished",
  "reason": "brief concrete reason",
  "last_evidence_path": "observations/relevant-image.jpg"
}
```

Never write `"task_status": "finished"` without a fresh overhead image that supports it.
