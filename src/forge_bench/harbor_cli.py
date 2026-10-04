from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .harbor_execution import (
    SUPPORTED_HARBOR_ENVIRONMENTS,
    execution_from_plan,
    require_execution_dependencies,
)
from .harbor_runtime import (
    load_harbor_plan,
    materialize_harbor_trial_configs,
    run_harbor_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an explicit Forge Bench run plan through Harbor on local "
            "Docker or Modal without changing Forge cell identity."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment",
        choices=SUPPORTED_HARBOR_ENVIRONMENTS,
        default=None,
        help="Execution environment override. Defaults to the plan value (docker).",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Maximum explicit Forge cells to execute concurrently.",
    )
    parser.add_argument("--cpus", type=int, help="CPU override per Harbor trial.")
    parser.add_argument("--memory-mb", type=int, help="Memory override per Harbor trial.")
    parser.add_argument("--storage-mb", type=int, help="Storage override per Harbor trial.")
    parser.add_argument("--gpus", type=int, help="GPU count override per Harbor trial.")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help=(
            "Resolve Harbor TrialConfig objects and execution provenance, then "
            "exit without provisioning Docker/Modal or running agents."
        ),
    )
    return parser.parse_args()


def _execution_manifest(
    *,
    plan: dict,
    plan_path: Path,
    output: Path,
    execution,
    materialize_only: bool,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "forge-bench-harbor-execution",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path.resolve()),
        "plan_run_plan_sha256": plan.get("run_plan_sha256"),
        "trial_count": int(plan.get("trial_count", 0)),
        "scientific_cell_identity_changed": False,
        "mode": "materialize-only" if materialize_only else "execute",
        "execution": execution.as_dict(),
        "results": str(output),
    }


async def _main_async(args: argparse.Namespace) -> int:
    plan = load_harbor_plan(args.plan)
    execution = execution_from_plan(
        plan,
        environment=args.environment,
        n_concurrent=args.n_concurrent,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        gpus=args.gpus,
    )
    output = (
        args.output or args.trials_dir.parent / "forge-harbor-results.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(output.stem + "-execution.json")
    manifest_path.write_text(
        json.dumps(
            _execution_manifest(
                plan=plan,
                plan_path=args.plan,
                output=output,
                execution=execution,
                materialize_only=args.materialize_only,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.materialize_only:
        materialized = await materialize_harbor_trial_configs(
            plan,
            trials_dir=args.trials_dir,
            execution=execution,
        )
        payload = [
            {
                "forge_cell_id": intent["forge_cell_id"],
                "forge_run_index": intent["forge_run_index"],
                "trial_config": config.model_dump(mode="json"),
            }
            for intent, config in materialized
        ]
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Materialized {len(payload)} Harbor trial configs: {output}")
        print("Execution manifest:", manifest_path)
        print("No Docker/Modal environment or agent/model call was started.")
        return 0

    require_execution_dependencies(execution)
    _harbor, forge_results = await run_harbor_plan(
        plan,
        trials_dir=args.trials_dir,
        execution=execution,
    )
    output.write_text(
        json.dumps([asdict(result) for result in forge_results], indent=2) + "\n",
        encoding="utf-8",
    )
    usable = sum(result.valid for result in forge_results)
    solved = sum(result.resolved for result in forge_results if result.valid)
    print(
        f"Harbor {execution.environment} execution complete: "
        f"{usable}/{len(forge_results)} usable; {solved} resolved"
    )
    print("Concurrency:", execution.n_concurrent)
    print("Forge-normalized results:", output)
    print("Execution manifest:", manifest_path)
    return 0 if usable == len(forge_results) else 1


def main() -> int:
    return asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
