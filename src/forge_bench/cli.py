from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    ARMS,
    CAVE_SHA,
    DEFAULT_DATASET,
    DEFAULT_DIFFICULTY,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_TOOLSETS,
    DEFAULT_SUITE,
    LABEL,
    LEAN_TOOLSETS,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PONY_REPO,
    PONY_SHA,
    Result,
)
from .harness import (
    install_arm,
    load_source_config,
    make_profile,
    run_one,
    sh,
    source_home,
)
from .reporting import write_csv, write_reports
from .swebench_backend import (
    EXPERIMENTS_SHA,
    build_candidates,
    candidate_rows,
    load_rows,
    normalize_difficulty,
    resolve_dataset_name,
    select_spread,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Hermes token-saving strategies on SWE-bench tasks."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Alias (verified, lite, full) or Hugging Face SWE-bench dataset id.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--difficulty",
        default="auto",
        help=(
            "Difficulty filter. Verified accepts easy/medium/hard/expert. "
            "auto = medium for Verified, unfiltered otherwise."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--smart-sample",
        action="store_true",
        help=(
            "Re-run the low/mid/high smart sampler instead of using the frozen "
            "initial three-task default suite."
        ),
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Explicit SWE-bench instance IDs; overrides default and smart sampling.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Save/print selected instances without model calls or Docker grading.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable historical Verified solve-rate signal during smart sampling.",
    )
    parser.add_argument(
        "--allow-same-repo",
        action="store_true",
        help="Allow multiple smart-sampled tasks from the same repository.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum seconds for each Hermes attempt.",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=int,
        default=3600,
        help="Maximum seconds for official SWE-bench grading of each patch.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Generate patches/efficiency data without official Docker grading.",
    )
    parser.add_argument("--hermes", default=shutil.which("hermes") or "hermes")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/agenticforge/forge-bench").expanduser(),
    )
    return parser.parse_args()


def failed_result(
    arm: str,
    instance: dict[str, Any],
    repeat: int,
    run_index: int,
    run_dir: Path,
    error: Exception,
) -> Result:
    return Result(
        arm=arm,
        task=str(instance["instance_id"]),
        repeat=repeat,
        run_index=run_index,
        valid=False,
        resolved=False,
        evaluation_completed=False,
        patch_nonempty=False,
        completed=False,
        exit_code=2,
        wall_seconds=0.0,
        evaluation_seconds=0.0,
        repo=str(instance["repo"]),
        difficulty=str(instance.get("difficulty") or ""),
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


def _difficulty_filter(
    args: argparse.Namespace,
    resolved_dataset: str,
) -> str | None:
    value = str(args.difficulty or "").strip()
    if value.lower() == "auto":
        return (
            DEFAULT_DIFFICULTY
            if resolved_dataset.endswith("SWE-bench_Verified")
            else None
        )
    if value.lower() in {"none", "all", "*", ""}:
        return None
    return value


def _selection_manifest_row(candidate: Any) -> dict[str, Any]:
    row = asdict(candidate)
    row.pop("problem_statement", None)
    return row


def _print_selection(selected: list[Any]) -> None:
    print("\nSelected SWE-bench instances")
    print("  tier   complexity  hist-solve  patch-lines  files  repository / instance")
    for candidate in selected:
        history = (
            "n/a"
            if candidate.historical_solve_rate is None
            else f"{100 * candidate.historical_solve_rate:5.1f}%"
        )
        print(
            f"  {candidate.selection_rank:5s}  "
            f"{candidate.complexity_score:10.3f}  "
            f"{history:>10s}  "
            f"{candidate.patch_changed_lines:11d}  "
            f"{candidate.patch_files:5d}  "
            f"{candidate.repo} / {candidate.instance_id}"
        )


def _preflight(args: argparse.Namespace) -> None:
    if args.selection_only:
        return
    if not shutil.which(args.hermes) and not Path(args.hermes).exists():
        raise SystemExit(f"Hermes executable not found: {args.hermes}")
    if not shutil.which("git"):
        raise SystemExit("git is required")

    if not args.skip_evaluation:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is required for official SWE-bench grading. "
                "Install/start Docker or use --skip-evaluation."
            )
        docker = sh(["docker", "info"], timeout=30)
        if docker.returncode:
            raise SystemExit(
                "Docker is installed but not available to this user. "
                "Start Docker/fix permissions or use --skip-evaluation.\n"
                + docker.stderr
            )


def _use_frozen_default(
    args: argparse.Namespace,
    resolved_dataset: str,
    difficulty: str | None,
) -> bool:
    return (
        not args.instance_ids
        and not args.smart_sample
        and resolved_dataset.endswith("SWE-bench_Verified")
        and normalize_difficulty(difficulty or "") == DEFAULT_DIFFICULTY
        and args.sample_size == len(DEFAULT_SUITE)
    )


def _frozen_candidates(
    row_by_id: dict[str, dict[str, Any]],
    cache_root: Path,
) -> tuple[list[Any], dict[str, Any]]:
    ids = [entry["instance_id"] for entry in DEFAULT_SUITE]
    missing = [instance_id for instance_id in ids if instance_id not in row_by_id]
    if missing:
        raise SystemExit(
            "Frozen default instance IDs are missing from the selected dataset: "
            + ", ".join(missing)
        )

    # Build normal Candidate objects from the current pinned dataset contents,
    # but do not rescan historical submissions. Then restore the selection
    # statistics recorded when the suite was frozen.
    candidates, _ = build_candidates(
        [row_by_id[instance_id] for instance_id in ids],
        difficulty=None,
        cache_root=cache_root,
        use_history=False,
    )
    by_id = {candidate.instance_id: candidate for candidate in candidates}

    selected = []
    for entry in DEFAULT_SUITE:
        candidate = by_id[entry["instance_id"]]
        candidate.selection_rank = str(entry["selection_rank"])
        candidate.selection_target = None
        candidate.patch_scope_percentile = float(entry["patch_scope_percentile"])
        candidate.complexity_score = candidate.patch_scope_percentile
        candidate.historical_solve_rate = float(entry["historical_solve_rate"])
        candidate.patch_changed_lines = int(entry["patch_changed_lines"])
        candidate.patch_files = int(entry["patch_files"])
        selected.append(candidate)

    meta = {
        "strategy": "frozen initial suite selected once by smart sampler",
        "frozen_on": "2026-09-19",
        "source_difficulty": "medium",
        "source_experiments_commit": EXPERIMENTS_SHA,
        "selection_band": "historical solve rate approximately 0.78-0.84",
        "note": (
            "Initial suite intentionally favors historically high-solve tasks "
            "while retaining variation in patch scope. Use --smart-sample to "
            "select a new spread from the current candidate pool."
        ),
    }
    return selected, meta


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1")

    resolved_dataset = resolve_dataset_name(args.dataset)
    difficulty = _difficulty_filter(args, resolved_dataset)
    cache_root = args.cache.expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output
        or Path("benchmark-results") / f"forge-bench-{stamp}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Forge Bench task selection")
    print("  dataset:", resolved_dataset)
    print("  split:", args.split)
    print("  difficulty:", difficulty or "unfiltered")

    rows = load_rows(resolved_dataset, args.split)
    row_by_id = {str(row["instance_id"]): row for row in rows}

    verified_dataset = resolved_dataset.endswith("SWE-bench_Verified")
    use_history = verified_dataset and not args.no_history

    if args.instance_ids:
        missing = [
            instance_id
            for instance_id in args.instance_ids
            if instance_id not in row_by_id
        ]
        if missing:
            raise SystemExit(
                "Unknown instance IDs for selected dataset: " + ", ".join(missing)
            )

        candidates, sampler_meta = build_candidates(
            [row_by_id[instance_id] for instance_id in args.instance_ids],
            difficulty=None,
            cache_root=cache_root,
            use_history=use_history,
        )
        by_id = {candidate.instance_id: candidate for candidate in candidates}
        selected = [by_id[instance_id] for instance_id in args.instance_ids]
        for index, candidate in enumerate(selected, 1):
            candidate.selection_rank = f"explicit-{index}"
        sampler_meta["strategy"] = "explicit instance ids"

    elif _use_frozen_default(args, resolved_dataset, difficulty):
        selected, sampler_meta = _frozen_candidates(row_by_id, cache_root)
        print("  selection: frozen initial suite")

    else:
        candidates, sampler_meta = build_candidates(
            rows,
            difficulty=difficulty,
            cache_root=cache_root,
            use_history=use_history,
        )
        selected = select_spread(
            candidates,
            count=args.sample_size,
            diverse_repos=not args.allow_same_repo,
        )
        sampler_meta["strategy"] = (
            "smart spread from 0.20 to 0.80 on composite complexity; "
            "unique repositories preferred"
        )

        pool_rows = candidate_rows(candidates)
        for row in pool_rows:
            row.pop("problem_statement", None)
        write_csv(output / "candidate_pool.csv", pool_rows)

    selected_rows = [_selection_manifest_row(candidate) for candidate in selected]
    write_csv(output / "selection.csv", selected_rows)
    (output / "selection.json").write_text(
        json.dumps(selected_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    _print_selection(selected)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": resolved_dataset,
        "split": args.split,
        "difficulty_filter": (
            normalize_difficulty(difficulty or "") if difficulty else None
        ),
        "selected_instance_ids": [candidate.instance_id for candidate in selected],
        "sampler": sampler_meta,
        "experiments_source_pin": EXPERIMENTS_SHA,
        "model": PINNED_MODEL,
        "api_provider": "openrouter",
        "upstream_provider": PINNED_OPENROUTER_UPSTREAM,
        "ponytail_repo": PONY_REPO,
        "ponytail_commit": PONY_SHA,
        "caveman_commit": CAVE_SHA,
        "arms": list(ARMS),
        "treatment_design": "baseline; each approach alone; all three together",
        "default_toolsets": DEFAULT_TOOLSETS,
        "lean_toolsets": LEAN_TOOLSETS,
        "repeats": args.repeats,
        "seed": args.seed,
        "randomization": (
            "global shuffle across treatment x selected instance x repeat"
        ),
        "confidence_interval": (
            "two-sided 95% Student-t across selected task means; "
            "repeats are averaged within task first"
        ),
        "official_evaluation": not args.skip_evaluation,
    }
    (output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.selection_only:
        print("\nSelection only; no model calls were made.")
        print("Manifest:", output / "selection.json")
        return 0

    _preflight(args)

    home = source_home()
    config = load_source_config(home)

    arms = list(ARMS)
    instances = [row_by_id[candidate.instance_id] for candidate in selected]
    task_ids = [str(instance["instance_id"]) for instance in instances]
    instance_by_id = {
        str(instance["instance_id"]): instance for instance in instances
    }

    plan = [
        {"arm": arm, "instance_id": instance_id, "repeat": repeat}
        for repeat in range(1, args.repeats + 1)
        for instance_id in task_ids
        for arm in arms
    ]
    random.Random(args.seed).shuffle(plan)
    for index, item in enumerate(plan, 1):
        item["run_index"] = index
    write_csv(output / "run_plan.csv", plan)

    print("\nForge Bench execution")
    print("  model:", PINNED_MODEL)
    print("  OpenRouter upstream:", PINNED_OPENROUTER_UPSTREAM)
    print("  randomized runs:", len(plan), "seed=", args.seed)
    print("  official grading:", not args.skip_evaluation)
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
            instance_id = str(item["instance_id"])
            instance = instance_by_id[instance_id]
            repeat = int(item["repeat"])
            run_index = int(item["run_index"])

            print(
                f"[{run_index:02d}/{len(plan):02d}] "
                f"{LABEL[arm]} / {instance_id} / r{repeat}"
            )

            try:
                result = run_one(
                    args.hermes,
                    profiles[arm],
                    instance,
                    resolved_dataset,
                    arm,
                    repeat,
                    run_index,
                    cache_root,
                    output,
                    args.timeout,
                    args.evaluation_timeout,
                    not args.skip_evaluation,
                )
            except Exception as exc:
                run_dir = (
                    output
                    / "runs"
                    / f"{run_index:02d}__{arm}__{instance_id}__r{repeat}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "harness_error.txt").write_text(
                    str(exc) + "\n",
                    encoding="utf-8",
                )
                result = failed_result(
                    arm, instance, repeat, run_index, run_dir, exc
                )

            results.append(result)
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
            status = (
                "RESOLVED"
                if result.resolved
                else "UNRESOLVED"
                if result.valid
                else "INVALID"
            )
            print(
                "   ",
                status,
                f"tokens={result.total_tokens:,}",
                f"cost={cost}",
                f"agent={result.wall_seconds:.1f}s",
                f"eval={result.evaluation_seconds:.1f}s",
                f"api_calls={result.api_calls}",
            )

    write_reports(output, results, arms, task_ids, meta)

    partial = output / "runs.partial.json"
    if partial.exists():
        partial.unlink()

    usable = sum(result.valid for result in results)
    solved = sum(result.resolved for result in results if result.valid)
    print(f"\nCompleted: {usable}/{len(results)} usable runs; {solved} resolved")
    print("Report:", output / "report.html")
    print("PNG figures:", output / "*.png")
    return 0 if usable == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
