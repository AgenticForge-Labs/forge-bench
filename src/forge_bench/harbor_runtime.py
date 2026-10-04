from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harbor_contracts import (
    FORGE_HERMES_IMPORT_PATH,
    PINNED_HERMES_VERSION,
    toolsets_for_treatment,
)
from .harbor_results import normalize_harbor_trial_result

FORGE_SHARED_INSTRUCTIONS = """Forge Bench execution constraints:
- Work only from the supplied repository state and task instruction.
- Do not access the internet, external repositories, issue trackers, pull requests, patches, or prior benchmark sessions.
- Do not git fetch, add a network git remote, or delegate the task to another agent.
- Otherwise use the normal local Hermes tools and workflow you consider useful.
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
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported Harbor plan schema: {payload.get('schema_version')!r}"
        )
    return payload


def extra_instructions_for_treatment(treatment: str) -> list[str]:
    instructions = [FORGE_SHARED_INSTRUCTIONS]
    if treatment in {"caveman", "caveman_ponytail", "all_three"}:
        instructions.append(CAVEMAN_INSTRUCTION)
    return instructions


def _harbor_imports():
    try:
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import DatasetConfig
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TrialConfig
        from harbor.trial.trial import Trial
    except ImportError as exc:
        raise RuntimeError(
            "Harbor execution requires harbor==0.23.0. "
            "Use an environment that installs the pinned Harbor version."
        ) from exc
    return EnvironmentType, DatasetConfig, AgentConfig, EnvironmentConfig, TrialConfig, Trial


def build_harbor_trial_config(
    intent: dict[str, Any],
    task_config: Any,
    *,
    trials_dir: str | Path,
    environment: str = "docker",
) -> Any:
    (
        EnvironmentType,
        _DatasetConfig,
        AgentConfig,
        EnvironmentConfig,
        TrialConfig,
        _Trial,
    ) = _harbor_imports()

    cell = dict(intent["forge"])
    treatment = str(cell["arm"])
    agent_kwargs = dict(intent.get("agent", {}).get("kwargs") or {})
    agent_kwargs.update(
        {
            "treatment": treatment,
            "api_provider": str(cell["api_provider"]),
            "upstream_provider": str(cell["upstream_provider"]),
            "reasoning": str(cell["reasoning"]),
            "max_turns": int(cell["max_turns"]),
            "budget_warning_ratio": cell["budget_warning_ratio"],
            "toolsets": toolsets_for_treatment(treatment),
            "version": PINNED_HERMES_VERSION,
        }
    )
    if treatment in {"ponytail", "caveman_ponytail", "all_three"}:
        agent_kwargs["extra_env"] = {"PONYTAIL_DEFAULT_MODE": "full"}

    try:
        environment_type = EnvironmentType(environment)
    except ValueError as exc:
        raise ValueError(f"Unsupported Harbor environment: {environment!r}") from exc

    return TrialConfig(
        task=task_config,
        trial_name=str(intent["forge_cell_id"]),
        trials_dir=Path(trials_dir),
        agent=AgentConfig(
            import_path=FORGE_HERMES_IMPORT_PATH,
            model_name=str(cell["model"]),
            kwargs=agent_kwargs,
        ),
        environment=EnvironmentConfig(type=environment_type),
        extra_instructions=extra_instructions_for_treatment(treatment),
    )


async def materialize_harbor_trial_configs(
    plan: dict[str, Any],
    *,
    trials_dir: str | Path,
    environment: str | None = None,
) -> list[tuple[dict[str, Any], Any]]:
    if not plan.get("executable"):
        raise ValueError(
            "Harbor plan is not executable; check dataset mapping and migration state"
        )

    (
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

    chosen_environment = str(environment or plan.get("environment") or "docker")
    return [
        (
            intent,
            build_harbor_trial_config(
                intent,
                task_by_instance[str(intent["forge"]["instance_id"])],
                trials_dir=trials_dir,
                environment=chosen_environment,
            ),
        )
        for intent in intents
    ]


async def run_harbor_plan(
    plan: dict[str, Any],
    *,
    trials_dir: str | Path,
    environment: str | None = None,
) -> tuple[list[Any], list[Any]]:
    """Execute explicit Forge cells through Harbor in randomized plan order.

    PR 2 is intentionally sequential. This preserves the established Forge run
    order while the execution boundary is validated. Controlled Harbor
    concurrency belongs in the later full execution migration.
    """
    *_, Trial = _harbor_imports()
    materialized = await materialize_harbor_trial_configs(
        plan,
        trials_dir=trials_dir,
        environment=environment,
    )

    harbor_results: list[Any] = []
    forge_results: list[Any] = []
    for intent, config in materialized:
        trial = await Trial.create(config)
        trial_result = await trial.run()
        harbor_results.append(trial_result)
        forge_results.append(normalize_harbor_trial_result(trial_result, intent))
    return harbor_results, forge_results
