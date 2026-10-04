from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .harbor_execution import HarborExecutionConfig, execution_from_plan
from .harbor_results import normalize_harbor_trial_result
from .telemetry_recorder import CellTelemetryRecorder, ExperimentTelemetryRecorder

FORGE_SHARED_INSTRUCTIONS = """Forge Bench execution constraints:
- Work only from the supplied repository state and task instruction.
- Do not access the internet, external repositories, issue trackers, pull requests, patches, or prior benchmark sessions.
- Do not git fetch, add a network git remote, or delegate the task to another agent.
- Otherwise use the normal tools and workflow exposed by the selected Harbor agent.
- Make the smallest complete production-code fix that addresses the issue.
- Inspect relevant code before editing.
- Run relevant tests or targeted checks when the local environment permits.
- Do not modify tests merely to make them pass.
- Finish with a concise summary of the fix and checks performed.
"""

CAVEMAN_INSTRUCTION = (
    "Load and follow the installed caveman skill in full mode, "
    "without dropping commands or verification evidence."
)


def load_harbor_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "forge-bench-harbor-plan":
        raise ValueError("Not a Forge Bench Harbor plan")
    schema = int(payload.get("schema_version", 0))
    if schema != 2:
        if schema == 1:
            raise ValueError(
                "Harbor plan schema 1 predates the agent factor; regenerate the "
                "plan with the current 'forge-bench --plan-only' command"
            )
        raise ValueError(f"Unsupported Harbor plan schema: {schema!r}")
    return payload


def extra_instructions_for_intent(intent: dict[str, Any]) -> list[str]:
    instructions = [FORGE_SHARED_INSTRUCTIONS]
    agent_intent = dict(intent.get("agent") or {})
    if agent_intent.get("mode") != "forge-hermes-compat":
        return instructions

    treatment = str(intent["forge"]["arm"])
    if treatment in {"caveman", "caveman_ponytail", "all_three"}:
        instructions.append(CAVEMAN_INSTRUCTION)
    return instructions


def _harbor_imports():
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import DatasetConfig
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TrialConfig
        from harbor.trial.trial import Trial
    except ImportError as exc:
        raise RuntimeError(
            "Harbor execution requires harbor==0.23.0. "
            "Use an environment that installs the pinned Harbor version."
        ) from exc
    return AgentFactory, EnvironmentType, DatasetConfig, AgentConfig, EnvironmentConfig, TrialConfig, Trial


def build_harbor_trial_config(
    intent: dict[str, Any],
    task_config: Any,
    *,
    trials_dir: str | Path,
    environment: str = "docker",
    execution: HarborExecutionConfig | None = None,
) -> Any:
    (
        AgentFactory,
        EnvironmentType,
        _DatasetConfig,
        AgentConfig,
        EnvironmentConfig,
        TrialConfig,
        _Trial,
    ) = _harbor_imports()

    if not bool(intent.get("supported", True)):
        reason = str(
            (intent.get("agent") or {}).get("unsupported_reason")
            or "unsupported Forge/Harbor cell"
        )
        raise ValueError(reason)

    agent_intent = dict(intent.get("agent") or {})
    model_name = str(agent_intent.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("Harbor trial intent is missing agent.model_name")

    agent_kwargs = dict(agent_intent.get("kwargs") or {})
    agent_name = agent_intent.get("name")
    import_path = agent_intent.get("import_path")
    if bool(agent_name) == bool(import_path):
        raise ValueError(
            "Harbor trial intent must define exactly one of agent.name or agent.import_path"
        )

    resolved_execution = (
        execution.validated()
        if execution is not None
        else HarborExecutionConfig(environment=environment).validated()
    )
    try:
        environment_type = EnvironmentType(resolved_execution.environment)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported Harbor environment: {resolved_execution.environment!r}"
        ) from exc

    agent_config = AgentConfig(
        name=str(agent_name) if agent_name else None,
        import_path=str(import_path) if import_path else None,
        model_name=model_name,
        kwargs=agent_kwargs,
    )
    AgentFactory.run_preflight(agent_config)

    return TrialConfig(
        task=task_config,
        trial_name=str(intent["forge_cell_id"]),
        trials_dir=Path(trials_dir),
        agent=agent_config,
        environment=EnvironmentConfig(
            type=environment_type,
            **resolved_execution.environment_kwargs(),
        ),
        extra_instructions=extra_instructions_for_intent(intent),
    )


async def materialize_harbor_trial_configs(
    plan: dict[str, Any],
    *,
    trials_dir: str | Path,
    environment: str | None = None,
    execution: HarborExecutionConfig | None = None,
) -> list[tuple[dict[str, Any], Any]]:
    if not plan.get("executable"):
        reasons = list(plan.get("unsupported_reasons") or [])
        detail = "; ".join(str(reason) for reason in reasons)
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            "Harbor plan is not executable; check dataset mapping and supported "
            f"agent/treatment/budget combinations{suffix}"
        )

    (
        _AgentFactory,
        _EnvironmentType,
        DatasetConfig,
        _AgentConfig,
        _EnvironmentConfig,
        _TrialConfig,
        _Trial,
    ) = _harbor_imports()

    harbor_dataset = plan.get("harbor_dataset")
    if not isinstance(harbor_dataset, str) or not harbor_dataset:
        raise ValueError("Executable Harbor plan is missing harbor_dataset")

    intents = list(plan.get("trials") or [])
    unique_task_ids = list(
        dict.fromkeys(str(intent["forge"]["instance_id"]) for intent in intents)
    )
    resolved = await DatasetConfig(
        name=harbor_dataset,
        task_names=unique_task_ids,
    ).get_task_configs()

    task_by_instance: dict[str, Any] = {}
    for task_config in resolved:
        task_name = getattr(task_config, "name", None)
        task_path = getattr(task_config, "path", None)
        if isinstance(task_name, str) and task_name:
            short_name = task_name.rsplit("/", 1)[-1]
        elif task_path is not None:
            short_name = Path(task_path).name
        else:
            raise ValueError("Harbor returned a task config without name or path")
        task_by_instance[short_name] = task_config

    missing = [task for task in unique_task_ids if task not in task_by_instance]
    if missing:
        raise ValueError(
            "Harbor dataset did not resolve Forge task(s): " + ", ".join(missing)
        )

    resolved_execution = (
        execution.validated()
        if execution is not None
        else execution_from_plan(plan, environment=environment)
    )
    return [
        (
            intent,
            build_harbor_trial_config(
                intent,
                task_by_instance[str(intent["forge"]["instance_id"])],
                trials_dir=trials_dir,
                environment=resolved_execution.environment,
                execution=resolved_execution,
            ),
        )
        for intent in intents
    ]


async def run_materialized_trials(
    materialized: list[tuple[dict[str, Any], Any]],
    *,
    n_concurrent: int = 1,
    trial_class: Any | None = None,
    telemetry: ExperimentTelemetryRecorder | None = None,
) -> tuple[list[Any], list[Any]]:
    """Run explicit TrialConfigs concurrently without regenerating Forge cells.

    Result ordering always follows the authoritative randomized Forge plan even
    when multiple trials overlap in wall-clock execution.
    """
    if int(n_concurrent) < 1:
        raise ValueError("n_concurrent must be >= 1")

    if trial_class is None:
        *_, trial_class = _harbor_imports()

    semaphore = asyncio.Semaphore(int(n_concurrent))

    async def run_one(
        index: int,
        intent: dict[str, Any],
        config: Any,
    ) -> tuple[int, Any, Any]:
        cell = _telemetry_cell(telemetry, intent, config)
        queued_ns = time.monotonic_ns()
        _cell_event(
            cell,
            "cell.queued",
            phase="queue",
            attributes={"forge_run_index": intent.get("forge_run_index")},
        )
        try:
            async with semaphore:
                _cell_event(
                    cell,
                    "cell.started",
                    phase="execution",
                    attributes={
                        "queue_wait_ns": max(0, time.monotonic_ns() - queued_ns),
                    },
                )
                _cell_event(cell, "harbor.trial.create.started", phase="trial_create")
                trial = await trial_class.create(config)
                trial_id = _object_id(trial)
                _cell_event(
                    cell,
                    "harbor.trial.create.completed",
                    phase="trial_create",
                    attributes={"harbor_trial_id": trial_id},
                )

                _cell_event(
                    cell,
                    "harbor.trial.run.started",
                    phase="trial_run",
                    attributes={"harbor_trial_id": trial_id},
                )
                trial_result = await trial.run()
                result_trial_id = _object_id(trial_result) or trial_id
                exception_info = getattr(trial_result, "exception_info", None)
                _cell_event(
                    cell,
                    "harbor.trial.run.completed",
                    phase="trial_run",
                    severity="error" if exception_info is not None else "info",
                    attributes={
                        "harbor_trial_id": result_trial_id,
                        "reported_exception": exception_info is not None,
                    },
                )

                _cell_event(
                    cell,
                    "forge.result.normalize.started",
                    phase="normalize",
                    attributes={"harbor_trial_id": result_trial_id},
                )
                forge_result = normalize_harbor_trial_result(trial_result, intent)
                _cell_event(
                    cell,
                    "forge.result.normalize.completed",
                    phase="normalize",
                    attributes={
                        "harbor_trial_id": result_trial_id,
                        "valid": bool(forge_result.valid),
                        "resolved": bool(forge_result.resolved),
                    },
                )
                _cell_event(
                    cell,
                    "cell.completed",
                    phase="execution",
                    attributes={
                        "valid": bool(forge_result.valid),
                        "resolved": bool(forge_result.resolved),
                    },
                )
                return index, trial_result, forge_result
        except BaseException as exc:
            _cell_event(
                cell,
                "cell.failed",
                phase="execution",
                severity="error",
                attributes={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise
        finally:
            if cell is not None:
                cell.close()

    completed = await asyncio.gather(
        *(
            run_one(index, intent, config)
            for index, (intent, config) in enumerate(materialized)
        )
    )
    completed.sort(key=lambda item: item[0])
    return (
        [item[1] for item in completed],
        [item[2] for item in completed],
    )


async def run_harbor_plan(
    plan: dict[str, Any],
    *,
    trials_dir: str | Path,
    environment: str | None = None,
    execution: HarborExecutionConfig | None = None,
    telemetry: ExperimentTelemetryRecorder | None = None,
) -> tuple[list[Any], list[Any]]:
    """Execute the exact Forge cells through Harbor Docker or Modal.

    Forge owns the explicit randomized cell list. Harbor owns each individual
    trial's environment, agent, verification, trajectory, and artifacts.
    Concurrency changes execution overlap only; it never regenerates cell
    identities or changes the returned scientific order.
    """
    resolved_execution = (
        execution.validated()
        if execution is not None
        else execution_from_plan(plan, environment=environment)
    )
    materialized = await materialize_harbor_trial_configs(
        plan,
        trials_dir=trials_dir,
        execution=resolved_execution,
    )
    return await run_materialized_trials(
        materialized,
        n_concurrent=resolved_execution.n_concurrent,
        telemetry=telemetry,
    )



def _telemetry_cell(
    telemetry: ExperimentTelemetryRecorder | None,
    intent: dict[str, Any],
    config: Any,
) -> CellTelemetryRecorder | None:
    if telemetry is None:
        return None
    try:
        environment = getattr(getattr(config, "environment", None), "type", None)
        environment_value = str(getattr(environment, "value", environment) or "unknown")
        return telemetry.cell(intent, environment=environment_value)
    except Exception as exc:
        telemetry.issue("runtime.cell", exc)
        return None


def _cell_event(
    recorder: CellTelemetryRecorder | None,
    name: str,
    *,
    phase: str,
    severity: str = "info",
    attributes: dict[str, Any] | None = None,
) -> None:
    if recorder is None:
        return
    try:
        recorder.event(
            name,
            phase=phase,
            severity=severity,
            attributes=attributes,
        )
    except Exception as exc:
        recorder.parent.issue(f"runtime.event:{name}", exc)


def _object_id(value: Any) -> str | None:
    identifier = getattr(value, "id", None)
    if identifier in (None, ""):
        return None
    return str(identifier)
