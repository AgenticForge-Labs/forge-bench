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

HARBOR_PLAN_SCHEMA_VERSION = 1

HARBOR_DATASET_BY_FORGE_DATASET = {
    "SWE-bench/SWE-bench_Verified": "swe-bench/swe-bench-verified",
}

_CELL_ID_FIELDS = (
    "run_index",
    "repeat",
    "repeat_seed",
    "block_position",
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


def compile_harbor_plan(
    run_plan: Sequence[dict[str, Any]],
    *,
    forge_dataset: str,
    environment: str = "docker",
) -> dict[str, Any]:
    """Project a Forge randomized plan into Harbor-oriented trial intents.

    The projection preserves explicit Forge cells rather than asking Harbor to
    regenerate a Cartesian experiment. The Harbor runner materializes each
    intent as one TrialConfig in Forge's randomized order.

    Keeping this projection separate prevents Harbor's normal Cartesian job
    expansion from silently changing Forge Bench's already-randomized cells.
    """
    harbor_dataset = HARBOR_DATASET_BY_FORGE_DATASET.get(forge_dataset)
    dataset_mapped = harbor_dataset is not None

    trials: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
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

        trials.append(
            {
                "forge_cell_id": cell_id,
                "forge_run_index": int(item["run_index"]),
                "trial_name": cell_id,
                "dataset": {
                    "name": harbor_dataset,
                    "task_names": [str(item["instance_id"])],
                },
                "agent": {
                    "import_path": FORGE_HERMES_IMPORT_PATH,
                    "model_name": str(item["model"]),
                    "kwargs": {
                        "treatment": str(item["arm"]),
                        "max_turns": int(item["max_turns"]),
                        "budget_warning_ratio": item["budget_warning_ratio"],
                        "api_provider": str(item["api_provider"]),
                        "upstream_provider": str(item["upstream_provider"]),
                        "reasoning": str(item["reasoning"]),
                        "toolsets": toolsets_for_treatment(str(item["arm"])),
                        "version": PINNED_HERMES_VERSION,
                    },
                },
                "environment": {"type": environment},
                "forge": item,
            }
        )

    return {
        "schema_version": HARBOR_PLAN_SCHEMA_VERSION,
        "kind": "forge-bench-harbor-plan",
        "harbor_api_baseline": PINNED_HARBOR_VERSION,
        "hermes_version": PINNED_HERMES_VERSION,
        "executable": dataset_mapped,
        "execution_status": (
            "ready for explicit-cell Harbor execution through forge-bench-harbor"
            if dataset_mapped
            else "projection-only: Forge dataset has no explicit Harbor mapping yet"
        ),
        "dataset_mapped": dataset_mapped,
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
