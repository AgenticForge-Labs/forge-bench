from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    ARMS,
    CAVE_SHA,
    DEFAULT_SEED,
    DEFAULT_TASKS,
    LABEL,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PONY_REPO,
    PONY_SHA,
    QUIX_SHA,
    QUIX_URL,
    Result,
)
from .harness import (
    get_quixbugs,
    install_arm,
    load_source_config,
    make_profile,
    run_one,
    source_home,
)
from .reporting import write_csv, write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Hermes token-saving strategies on pinned QuixBugs tasks."
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--hermes",
        default=shutil.which("hermes") or "hermes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/agenticforge/forge-bench").expanduser(),
    )
    return parser.parse_args()


def failed_result(
    arm: str,
    task: str,
    repeat: int,
    run_index: int,
    run_dir: Path,
    error: Exception,
) -> Result:
    return Result(
        arm=arm,
        task=task,
        repeat=repeat,
        run_index=run_index,
        valid=False,
        tests_passed=False,
        tests_untouched=False,
        completed=False,
        exit_code=2,
        wall_seconds=0.0,
        model=PINNED_MODEL,
        api_provider="openrouter",
        upstream_provider=PINNED_OPENROUTER_UPSTREAM,
        session_id="",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        total_tokens=0,
        api_calls=0,
        estimated_cost_usd=None,
        actual_cost_usd=None,
        cost_usd=None,
        cost_source="unavailable",
        tool_calls=None,
        files_changed=0,
        diff_lines=0,
        run_dir=str(run_dir),
        error=str(error),
    )


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not shutil.which(args.hermes) and not Path(args.hermes).exists():
        raise SystemExit(f"Hermes executable not found: {args.hermes}")
    if not shutil.which("git"):
        raise SystemExit("git is required")

    home = source_home()
    config = load_source_config(home)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output
        or Path("benchmark-results") / f"forge-bench-{stamp}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    arms = list(ARMS)

    qsrc = get_quixbugs(args.cache.resolve())

    plan = [
        {"arm": arm, "task": task, "repeat": repeat}
        for repeat in range(1, args.repeats + 1)
        for task in args.tasks
        for arm in arms
    ]
    random.Random(args.seed).shuffle(plan)
    for index, item in enumerate(plan, 1):
        item["run_index"] = index

    write_csv(output / "run_plan.csv", plan)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": PINNED_MODEL,
        "api_provider": "openrouter",
        "upstream_provider": PINNED_OPENROUTER_UPSTREAM,
        "quixbugs_repo": QUIX_URL,
        "quixbugs_commit": QUIX_SHA,
        "ponytail_repo": PONY_REPO,
        "ponytail_commit": PONY_SHA,
        "caveman_commit": CAVE_SHA,
        "tasks": args.tasks,
        "arms": arms,
        "repeats": args.repeats,
        "seed": args.seed,
        "randomization": "global shuffle across treatment x task x repeat runs",
        "confidence_interval": (
            "two-sided 95% Student-t across task means; "
            "repeats are averaged within task first"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Forge Bench")
    print("  model:", PINNED_MODEL)
    print("  OpenRouter upstream:", PINNED_OPENROUTER_UPSTREAM)
    print("  randomized runs:", len(plan), "seed=", args.seed)
    print("  output:", output)

    results: list[Result] = []

    with tempfile.TemporaryDirectory(prefix="forge-bench-") as tempdir:
        profiles: dict[str, Path] = {}

        for arm in arms:
            profile = Path(tempdir) / "profiles" / arm
            make_profile(home, config, profile)
            install_arm(args.hermes, profile, arm)
            profiles[arm] = profile

        for item in plan:
            arm = str(item["arm"])
            task = str(item["task"])
            repeat = int(item["repeat"])
            run_index = int(item["run_index"])

            print(
                f"[{run_index:02d}/{len(plan):02d}] "
                f"{LABEL[arm]} / {task} / r{repeat}"
            )

            try:
                result = run_one(
                    args.hermes,
                    profiles[arm],
                    arm,
                    task,
                    repeat,
                    run_index,
                    qsrc,
                    output,
                    args.timeout,
                )
            except Exception as exc:
                run_dir = (
                    output
                    / "runs"
                    / f"{run_index:02d}__{arm}__{task}__r{repeat}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "harness_error.txt").write_text(
                    str(exc) + "\n",
                    encoding="utf-8",
                )
                result = failed_result(
                    arm,
                    task,
                    repeat,
                    run_index,
                    run_dir,
                    exc,
                )

            results.append(result)

            # Preserve partial raw evidence if a long benchmark is interrupted.
            (output / "runs.partial.json").write_text(
                json.dumps(
                    [result.__dict__ for result in results],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cost = (
                "n/a"
                if result.cost_usd is None
                else "$" + f"{result.cost_usd:.4f}"
            )
            print(
                "   ",
                "VALID" if result.valid else "INVALID",
                f"tokens={result.total_tokens:,}",
                f"cost={cost}",
                f"time={result.wall_seconds:.1f}s",
                f"api_calls={result.api_calls}",
            )

    write_reports(output, results, arms, args.tasks, meta)

    partial = output / "runs.partial.json"
    if partial.exists():
        partial.unlink()

    valid = sum(result.valid for result in results)
    print(f"\nCompleted: {valid}/{len(results)} valid runs")
    print("Report:", output / "report.html")
    print("PNG figures:", output / "*.png")
    return 0 if valid == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
