from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .harbor_contracts import (
    FORGE_HERMES_IMPORT_PATH,
    PINNED_HARBOR_VERSION,
    PINNED_HERMES_VERSION,
    toolsets_for_treatment,
)

HARBOR_PLAN_SCHEMA_VERSION = 2

HARBOR_DATASET_BY_FORGE_DATASET = {
    "SWE-bench/SWE-bench_Verified": "swe-bench/swe-bench-verified",
}

_CELL_ID_FIELDS = (
    "run_index",
    "repeat",
    "repeat_seed",
    "block_position",
    "agent_key",
    "agent",
    "agent_version",
    "agent_kwargs",
    "model_key",
    "model",
    "api_provider",
    "upstream_provider",
    "reasoning",
    "arm",
    "max_turns",
    "budget_warning_ratio",
    "instance_id",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_plan_sha256(run_plan: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(run_plan)).encode("utf-8")).hexdigest()


def _cell_id(item: dict[str, Any]) -> str:
    identity = {field: item.get(field) for field in _CELL_ID_FIELDS}
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:12]
    return f"forge-{int(item['run_index']):04d}-{digest}"


def _budget_factors_active(run_plan: Sequence[dict[str, Any]]) -> bool:
    turn_levels = {item.get("max_turns") for item in run_plan}
    warning_levels = {item.get("budget_warning_ratio") for item in run_plan}
    return len(turn_levels) > 1 or len(warning_levels) > 1 or warning_levels != {None}


def _native_agent_intent(item: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(item.get("agent_kwargs") or {})
    version = item.get("agent_version")
    if version not in (None, ""):
        kwargs.setdefault("version", str(version))
    return {
        "mode": "harbor-native",
        "name": str(item["agent"]),
        "model_name": str(item["model"]),
        "kwargs": kwargs,
        "supported": True,
        "control_semantics": {
            "treatment": "baseline",
            "max_turns": "agent-default",
            "budget_warning_ratio": "agent-default",
        },
    }


def _forge_hermes_intent(item: dict[str, Any]) -> dict[str, Any]:
    treatment = str(item["arm"])
    kwargs = dict(item.get("agent_kwargs") or {})
    kwargs.update(
        {
            "treatment": treatment,
            "max_turns": int(item["max_turns"]),
            "budget_warning_ratio": item["budget_warning_ratio"],
            "api_provider": str(item["api_provider"]),
            "upstream_provider": str(item["upstream_provider"]),
            "reasoning": str(item["reasoning"]),
            "toolsets": toolsets_for_treatment(treatment),
            "version": str(item.get("agent_version") or PINNED_HERMES_VERSION),
        }
    )
    return {
        "mode": "forge-hermes-compat",
        "import_path": FORGE_HERMES_IMPORT_PATH,
        "model_name": str(item["model"]),
        "kwargs": kwargs,
        "supported": True,
        "control_semantics": {
            "treatment": "forge-controlled",
            "max_turns": "forge-controlled",
            "budget_warning_ratio": "forge-controlled",
        },
    }


def _unsupported_agent_intent(
    item: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "mode": "unsupported",
        "name": str(item["agent"]),
        "model_name": str(item["model"]),
        "kwargs": dict(item.get("agent_kwargs") or {}),
        "supported": False,
        "unsupported_reason": reason,
        "control_semantics": {
            "treatment": "unsupported",
            "max_turns": "unsupported",
            "budget_warning_ratio": "unsupported",
        },
    }


def compile_harbor_plan(
    run_plan: Sequence[dict[str, Any]],
    *,
    forge_dataset: str,
    environment: str = "docker",
) -> dict[str, Any]:
    """Project a Forge randomized plan into explicit Harbor trial intents.

    Forge randomizes scientific cells first. Harbor receives those exact cells;
    it must not independently regenerate the Cartesian experiment.

    Baseline cells use Harbor's native installed-agent registry, so agent and
    model are independent Forge factors. The existing Forge Hermes subclass is
    retained only for Hermes cells that require legacy Forge-specific treatment
    or budget controls. No plugin behavior is ported to other agents.
    """
    harbor_dataset = HARBOR_DATASET_BY_FORGE_DATASET.get(forge_dataset)
    dataset_mapped = harbor_dataset is not None
    budget_factors_active = _budget_factors_active(run_plan)

    trials: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    unsupported_reasons: list[str] = []

    for expected_index, raw_item in enumerate(run_plan, 1):
        item = dict(raw_item)
        missing = [field for field in _CELL_ID_FIELDS if field not in item]
        if missing:
            raise ValueError(
                f"Run-plan cell {expected_index} is missing required fields: "
                + ", ".join(missing)
            )
        if int(item["run_index"]) != expected_index:
            raise ValueError(
                "Run-plan order must match run_index exactly; "
                f"position {expected_index} contains run_index={item['run_index']!r}"
            )

        cell_id = _cell_id(item)
        if cell_id in seen_ids:
            raise ValueError(f"Duplicate Harbor cell identity generated: {cell_id}")
        seen_ids.add(cell_id)

        treatment = str(item["arm"])
        agent_name = str(item["agent"])
        needs_forge_controls = treatment != "baseline" or budget_factors_active
        if not needs_forge_controls:
            agent_intent = _native_agent_intent(item)
        elif agent_name == "hermes":
            agent_intent = _forge_hermes_intent(item)
        else:
            reason = (
                f"agent {agent_name!r} has no validated Forge implementation for "
                "non-baseline treatment or Forge budget factors; use baseline with "
                "agent-default budgeting or add a separately validated control adapter"
            )
            agent_intent = _unsupported_agent_intent(item, reason=reason)
            unsupported_reasons.append(reason)

        trials.append(
            {
                "forge_cell_id": cell_id,
                "forge_run_index": int(item["run_index"]),
                "trial_name": cell_id,
                "supported": bool(agent_intent["supported"]),
                "dataset": {
                    "name": harbor_dataset,
                    "task_names": [str(item["instance_id"])],
                },
                "agent": agent_intent,
                "environment": {"type": environment},
                "forge": item,
            }
        )

    all_supported = all(bool(trial["supported"]) for trial in trials)
    executable = dataset_mapped and all_supported
    if not dataset_mapped:
        execution_status = (
            "projection-only: Forge dataset has no explicit Harbor mapping yet"
        )
    elif not all_supported:
        execution_status = (
            "projection-only: one or more agent/treatment/budget cells are unsupported"
        )
    else:
        execution_status = (
            "ready for explicit-cell Harbor execution through forge-bench-harbor"
        )

    return {
        "schema_version": HARBOR_PLAN_SCHEMA_VERSION,
        "kind": "forge-bench-harbor-plan",
        "harbor_api_baseline": PINNED_HARBOR_VERSION,
        "forge_hermes_version": PINNED_HERMES_VERSION,
        "executable": executable,
        "execution_status": execution_status,
        "dataset_mapped": dataset_mapped,
        "all_cells_supported": all_supported,
        "unsupported_reasons": sorted(set(unsupported_reasons)),
        "forge_dataset": forge_dataset,
        "harbor_dataset": harbor_dataset,
        "environment": environment,
        "run_plan_sha256": run_plan_sha256(run_plan),
        "trial_count": len(trials),
        "trials": trials,
    }


def write_harbor_plan(
    path: str | Path,
    run_plan: Sequence[dict[str, Any]],
    *,
    forge_dataset: str,
    environment: str = "docker",
) -> dict[str, Any]:
    plan = compile_harbor_plan(
        run_plan,
        forge_dataset=forge_dataset,
        environment=environment,
    )
    Path(path).write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan
