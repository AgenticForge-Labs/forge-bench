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

    uncached_input = int(
        metadata.get("forge_uncached_input_tokens", max(0, n_input - n_cache)) or 0
    )
    cache_read = int(metadata.get("forge_cache_read_tokens", n_cache) or 0)
    cache_write = int(metadata.get("forge_cache_write_tokens", 0) or 0)
    output_tokens = int(metadata.get("forge_output_tokens", n_output) or 0)
    reasoning_tokens = int(metadata.get("forge_reasoning_tokens", 0) or 0)
    total_tokens = int(
        metadata.get(
            "forge_total_tokens",
            uncached_input + cache_read + cache_write + output_tokens,
        )
        or 0
    )

    reward = _reward(trial_result)
    evaluation_completed = reward is not None
    resolved = reward == 1.0

    exception = getattr(trial_result, "exception_info", None)
    completed = exception is None and context is not None
    turn_exit_reason = str(metadata.get("forge_turn_exit_reason") or "")
    budget_censored = turn_exit_reason.startswith("max_iterations_reached(")

    patch_nonempty = bool(metadata.get("forge_patch_nonempty", False))
    patch_capture_error = str(metadata.get("forge_patch_capture_error") or "")
    eval_ok = evaluation_completed or not patch_nonempty

    context_cost = getattr(context, "cost_usd", None) if context is not None else None
    estimated_cost = metadata.get("forge_estimated_cost_usd")
    actual_cost = metadata.get("forge_actual_cost_usd")
    estimated_cost = (
        float(estimated_cost) if estimated_cost not in (None, "") else None
    )
    actual_cost = float(actual_cost) if actual_cost not in (None, "") else None
    if actual_cost is not None and actual_cost > 0:
        cost = actual_cost
        cost_source = "openrouter_actual"
    elif estimated_cost is not None:
        cost = estimated_cost
        cost_source = str(metadata.get("forge_cost_source") or "hermes_estimate")
    elif context_cost is not None:
        cost = float(context_cost)
        cost_source = "harbor_agent_context"
    else:
        cost = None
        cost_source = "unavailable"
    model_info = getattr(getattr(trial_result, "agent_info", None), "model_info", None)
    observed_model = str(getattr(model_info, "name", None) or cell["model"])
    model_ok = observed_model == str(cell["model"])
    provider_ok = str(metadata.get("forge_api_provider") or cell["api_provider"]).lower() == str(
        cell["api_provider"]
    ).lower()
    upstream_ok = str(
        metadata.get("forge_upstream_provider") or cell["upstream_provider"]
    ) == str(cell["upstream_provider"])
    reasoning_ok = str(metadata.get("forge_reasoning") or cell["reasoning"]) == str(
        cell["reasoning"]
    )

    valid = (
        (completed or budget_censored)
        and eval_ok
        and not patch_capture_error
        and model_ok
        and provider_ok
        and upstream_ok
        and reasoning_ok
    )

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
    if patch_capture_error:
        errors.append("patch capture: " + patch_capture_error)
    if not model_ok:
        errors.append("wrong model: " + observed_model)
    if not provider_ok:
        errors.append(
            "wrong API provider: " + str(metadata.get("forge_api_provider") or "")
        )
    if not upstream_ok:
        errors.append(
            "wrong upstream provider: "
            + str(metadata.get("forge_upstream_provider") or "")
        )
    if not reasoning_ok:
        errors.append(
            "wrong reasoning mode: " + str(metadata.get("forge_reasoning") or "")
        )

    repo = str(cell.get("repo") or "")
    difficulty = str(cell.get("difficulty") or "")
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
        input_tokens=uncached_input,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        total_tokens=total_tokens,
        api_calls=int(metadata.get("forge_api_calls", 0) or 0),
        estimated_cost_usd=estimated_cost,
        actual_cost_usd=actual_cost,
        cost_usd=cost,
        cost_source=cost_source,
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
        trace_exported=bool(metadata.get("forge_trace_exported", False)),
    )
