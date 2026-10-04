from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import Result


def _seconds(timing: Any) -> float:
    if timing is None:
        return 0.0
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if not isinstance(started, datetime) or not isinstance(finished, datetime):
        return 0.0
    return max(0.0, (finished - started).total_seconds())


def _reward(trial_result: Any) -> float | None:
    verifier = getattr(trial_result, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None)
    if not isinstance(rewards, dict):
        return None
    value = rewards.get("reward")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def normalize_harbor_trial_result(
    trial_result: Any,
    forge_intent: dict[str, Any],
) -> Result:
    """Convert one Harbor TrialResult into the existing Forge observation model."""
    cell = dict(forge_intent["forge"])
    context = getattr(trial_result, "agent_result", None)
    metadata = dict(getattr(context, "metadata", None) or {})

    n_input = int(getattr(context, "n_input_tokens", 0) or 0)
    n_cache = int(getattr(context, "n_cache_tokens", 0) or 0)
    n_output = int(getattr(context, "n_output_tokens", 0) or 0)
    noncache_input = max(0, n_input - n_cache)

    reward = _reward(trial_result)
    evaluation_completed = reward is not None
    resolved = reward == 1.0

    exception = getattr(trial_result, "exception_info", None)
    completed = exception is None and context is not None
    turn_exit_reason = str(metadata.get("forge_turn_exit_reason") or "")
    budget_censored = turn_exit_reason.startswith("max_iterations_reached(")

    patch_nonempty = bool(metadata.get("forge_patch_nonempty", False))
    eval_ok = evaluation_completed or not patch_nonempty
    valid = (completed or budget_censored) and eval_ok

    errors: list[str] = []
    if exception is not None:
        errors.append(
            str(getattr(exception, "exception_type", "HarborError"))
            + ": "
            + str(getattr(exception, "exception_message", "trial failed"))
        )
    if budget_censored:
        errors.append("iteration budget reached")
    if patch_nonempty and not evaluation_completed:
        errors.append("Harbor verification did not produce a reward")
    if metadata.get("forge_patch_capture_error"):
        errors.append("patch capture: " + str(metadata["forge_patch_capture_error"]))

    cost = getattr(context, "cost_usd", None) if context is not None else None
    model_info = getattr(getattr(trial_result, "agent_info", None), "model_info", None)
    observed_model = str(getattr(model_info, "name", None) or cell["model"])

    repo = ""
    difficulty = ""
    config = getattr(trial_result, "config", None)
    task = getattr(config, "task", None)
    source = getattr(task, "source", None)
    if isinstance(source, str):
        repo = source

    run_dir = str(getattr(trial_result, "trial_uri", "") or "")

    return Result(
        arm=str(cell["arm"]),
        task=str(cell["instance_id"]),
        repeat=int(cell["repeat"]),
        run_index=int(cell["run_index"]),
        valid=valid,
        resolved=resolved,
        evaluation_completed=evaluation_completed,
        patch_nonempty=patch_nonempty,
        completed=completed,
        exit_code=0 if completed else 1,
        wall_seconds=_seconds(getattr(trial_result, "agent_execution", None)),
        evaluation_seconds=_seconds(getattr(trial_result, "verifier", None)),
        repo=repo,
        difficulty=difficulty,
        model=observed_model,
        api_provider=str(cell["api_provider"]),
        upstream_provider=str(cell["upstream_provider"]),
        session_id=str(metadata.get("forge_session_id") or ""),
        input_tokens=noncache_input,
        output_tokens=n_output,
        reasoning_tokens=int(metadata.get("forge_reasoning_tokens", 0) or 0),
        cache_read_tokens=n_cache,
        cache_write_tokens=int(metadata.get("forge_cache_write_tokens", 0) or 0),
        total_tokens=n_input + n_output,
        api_calls=int(metadata.get("forge_api_calls", 0) or 0),
        estimated_cost_usd=None,
        actual_cost_usd=float(cost) if cost is not None else None,
        cost_usd=float(cost) if cost is not None else None,
        cost_source="harbor_agent_context" if cost is not None else "unavailable",
        tool_calls=(
            int(metadata["forge_tool_calls"])
            if metadata.get("forge_tool_calls") is not None
            else None
        ),
        files_changed=int(metadata.get("forge_files_changed", 0) or 0),
        diff_lines=int(metadata.get("forge_diff_lines", 0) or 0),
        run_dir=run_dir,
        error="; ".join(errors),
        max_turns=int(cell["max_turns"]),
        budget_warning_ratio=cell["budget_warning_ratio"],
        main_api_calls=int(metadata.get("forge_api_calls", 0) or 0),
        auxiliary_api_calls=None,
        iterations_used=(
            int(metadata["forge_iterations_used"])
            if metadata.get("forge_iterations_used") is not None
            else None
        ),
        turn_exit_reason=turn_exit_reason,
        trace_exported=True,
    )
