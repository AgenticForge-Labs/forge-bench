from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .harbor_runtime import (
    load_harbor_plan,
    materialize_harbor_trial_configs,
    run_harbor_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute an explicit Forge Bench run plan through Harbor."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--environment", default=None)
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Resolve Harbor TrialConfig objects and exit without running agents.",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    plan = load_harbor_plan(args.plan)
    output = (
        args.output or args.trials_dir.parent / "forge-harbor-results.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.materialize_only:
        materialized = await materialize_harbor_trial_configs(
            plan,
            trials_dir=args.trials_dir,
            environment=args.environment,
        )
        payload = [
            {
                "forge_cell_id": intent["forge_cell_id"],
                "trial_config": config.model_dump(mode="json"),
            }
            for intent, config in materialized
        ]
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Materialized {len(payload)} Harbor trial configs: {output}")
        print("No agent/model calls were made.")
        return 0

    _harbor, forge_results = await run_harbor_plan(
        plan,
        trials_dir=args.trials_dir,
        environment=args.environment,
    )
    output.write_text(
        json.dumps([asdict(result) for result in forge_results], indent=2) + "\n",
        encoding="utf-8",
    )
    usable = sum(result.valid for result in forge_results)
    solved = sum(result.resolved for result in forge_results if result.valid)
    print(
        f"Harbor execution complete: {usable}/{len(forge_results)} usable; "
        f"{solved} resolved"
    )
    print("Forge-normalized results:", output)
    return 0 if usable == len(forge_results) else 1


def main() -> int:
    return asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
