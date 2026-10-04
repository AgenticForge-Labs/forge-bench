from __future__ import annotations

import random
from typing import Any

from .designs import AgentSpec, ModelSpec, default_design


def build_run_plan(
    arms: list[str],
    task_ids: list[str],
    repeats: int,
    master_seed: int,
    models: list[ModelSpec] | None = None,
    agents: list[AgentSpec] | None = None,
    max_turns_levels: tuple[int, ...] | list[int] | None = None,
    budget_warning_ratio_levels: tuple[float | None, ...] | list[float | None] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Build complete factorial blocks and shuffle each block once.

    The run plan is Forge Bench's scientific execution plan. Runtime backends
    may project these cells into their own configuration objects, but they must
    not resample, recross, or reorder the cells without recording a new design.

    Agent and model are independent randomized factors. Iteration budget and
    reminder ratio are explicit randomized factors, not
    global execution settings, so their main effects and interactions remain
    estimable independently of model, treatment, and task.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    seed_rng = random.Random(master_seed)
    repeat_seeds = [master_seed]
    while len(repeat_seeds) < repeats:
        candidate = seed_rng.randrange(1, 2**63)
        if candidate not in repeat_seeds:
            repeat_seeds.append(candidate)

    defaults = default_design()
    model_specs = models or list(defaults.models)
    agent_specs = agents or list(defaults.agents)
    turn_levels = list(max_turns_levels or defaults.max_turns_levels)
    warning_levels = list(
        budget_warning_ratio_levels or defaults.budget_warning_ratio_levels
    )

    plan: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = repeat_seeds[repeat - 1]
        block = [
            {
                "agent_key": agent_spec.key,
                "agent_label": agent_spec.label,
                "agent": agent_spec.agent,
                "agent_version": agent_spec.version,
                "agent_kwargs": dict(agent_spec.kwargs),
                "model_key": model_spec.key,
                "model_label": model_spec.label,
                "model": model_spec.model,
                "api_provider": model_spec.api_provider,
                "upstream_provider": model_spec.upstream_provider,
                "reasoning": model_spec.reasoning,
                "arm": arm,
                "max_turns": max_turns,
                "budget_warning_ratio": budget_warning_ratio,
                "instance_id": instance_id,
                "repeat": repeat,
                "repeat_seed": repeat_seed,
            }
            for agent_spec in agent_specs
            for model_spec in model_specs
            for instance_id in task_ids
            for arm in arms
            for max_turns in turn_levels
            for budget_warning_ratio in warning_levels
        ]
        random.Random(repeat_seed).shuffle(block)
        for block_position, item in enumerate(block, 1):
            item["block_position"] = block_position
            plan.append(item)

    for run_index, item in enumerate(plan, 1):
        item["run_index"] = run_index
    return plan, repeat_seeds
