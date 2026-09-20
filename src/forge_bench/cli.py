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
    DEFAULT_ANCHOR_IDS,
    DEFAULT_ARMS,
    DEFAULT_DATASET,
    DEFAULT_DIFFICULTY,
    DEFAULT_EXPECTED_IDS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_TOOLSETS,
    LABEL,
    LEAN_TOOLSETS,
    PINNED_MAX_TURNS,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PINNED_REASONING,
    PONY_REPO,
    PONY_SHA,
    Result,
)
from .harness import (
    install_arm,
    make_profile,
    prepare_workspace,
    run_one,
    sh,
    source_home,
)
from .reporting import write_csv, write_reports
from .swebench_backend import (
    EXPERIMENTS_SHA,
    build_candidates,
    candidate_rows,
    dataset_revision,
    load_rows,
    normalize_difficulty,
    resolve_dataset_name,
    select_anchor_neighborhood,
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
            "Use the broad low/mid/high complexity sampler instead of the "
            "default homogeneous anchor-neighborhood selector."
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
    parser.add_argument(
        "--hermes-runtime",
        choices=("docker", "local"),
        default="docker",
        help="Run each Hermes attempt in a fresh official Docker container (default) or use the local Hermes binary.",
    )
    parser.add_argument(
        "--hermes-image",
        default="nousresearch/hermes-agent:latest",
        help="Official Hermes Docker image reference. Pulled once, then its immutable image ID is used for every run.",
    )
    parser.add_argument("--hermes", default=shutil.which("hermes") or "hermes", help="Local Hermes binary; used only with --hermes-runtime local.")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARMS),
        default=list(DEFAULT_ARMS),
        help=(
            "Treatments to run. Default is the Caveman x Ponytail 2x2: "
            "baseline caveman ponytail caveman_ponytail. Lean variants remain "
            "available explicitly."
        ),
    )
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
    print("  tier      anchor-dist  hist-solve  patch-lines  files  repository / instance")
    for candidate in selected:
        history = (
            "n/a"
            if candidate.historical_solve_rate is None
            else f"{100 * candidate.historical_solve_rate:5.1f}%"
        )
        distance = (
            "n/a"
            if candidate.anchor_distance is None
            else f"{candidate.anchor_distance:.3f}"
        )
        print(
            f"  {candidate.selection_rank:8s}  "
            f"{distance:>11s}  "
            f"{history:>10s}  "
            f"{candidate.patch_changed_lines:11d}  "
            f"{candidate.patch_files:5d}  "
            f"{candidate.repo} / {candidate.instance_id}"
        )


def _preflight(args: argparse.Namespace) -> str | None:
    """Validate runtime prerequisites and return the immutable Hermes image ID."""
    if args.selection_only:
        return None
    if not shutil.which("git"):
        raise SystemExit("git is required")

    need_docker = args.hermes_runtime == "docker" or not args.skip_evaluation
    if need_docker:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is required for the selected Hermes runtime and/or official SWE-bench grading."
            )
        docker = sh(["docker", "info"], timeout=30)
        if docker.returncode:
            raise SystemExit(
                "Docker is installed but not available to this user.\n" + docker.stderr
            )

    image_id: str | None = None
    if args.hermes_runtime == "docker":
        pull = sh(["docker", "pull", args.hermes_image], timeout=900)
        if pull.returncode:
            raise SystemExit(
                f"Could not pull Hermes image {args.hermes_image}:\n{pull.stderr}"
            )
        inspect = sh(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.hermes_image],
            timeout=30,
        )
        if inspect.returncode or not inspect.stdout.strip():
            raise SystemExit(
                f"Could not resolve immutable image ID for {args.hermes_image}."
            )
        image_id = inspect.stdout.strip()

        probe_prefix = ["docker", "run", "--rm", "--pull=never", image_id]
        checks = [
            (
                [*probe_prefix, "--help"],
                ["--toolsets", "--reasoning", "--usage-file", "--ignore-rules"],
                "Hermes Docker one-shot CLI",
            ),
            (
                [*probe_prefix, "plugins", "install", "--help"],
                ["--ref", "--force", "--enable"],
                "Hermes Docker plugin installer",
            ),
            (
                [*probe_prefix, "skills", "install", "--help"],
                ["--name", "--force", "--yes"],
                "Hermes Docker skill installer",
            ),
        ]
    else:
        if not shutil.which(args.hermes) and not Path(args.hermes).exists():
            raise SystemExit(f"Hermes executable not found: {args.hermes}")
        checks = [
            (
                [args.hermes, "--help"],
                ["--toolsets", "--reasoning", "--usage-file", "--ignore-rules"],
                "Hermes one-shot CLI",
            ),
            (
                [args.hermes, "plugins", "install", "--help"],
                ["--ref", "--force", "--enable"],
                "Hermes plugin installer",
            ),
            (
                [args.hermes, "skills", "install", "--help"],
                ["--name", "--force", "--yes"],
                "Hermes skill installer",
            ),
        ]

    for argv, required, label in checks:
        probe = sh(argv, timeout=60)
        help_text = (probe.stdout or "") + "\n" + (probe.stderr or "")
        missing = [flag for flag in required if flag not in help_text]
        if probe.returncode not in (0, 1) or missing:
            detail = ", ".join(missing) if missing else f"exit {probe.returncode}"
            raise SystemExit(
                f"{label} is incompatible with Forge Bench ({detail}). "
                "Update Hermes/the Docker image before running paid calls."
            )

    return image_id


def _use_anchor_default(
    args: argparse.Namespace,
    resolved_dataset: str,
    difficulty: str | None,
) -> bool:
    return (
        not args.instance_ids
        and not args.smart_sample
        and resolved_dataset.endswith("SWE-bench_Verified")
        and normalize_difficulty(difficulty or "") == DEFAULT_DIFFICULTY
        and args.sample_size == DEFAULT_SAMPLE_SIZE
    )

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

    elif _use_anchor_default(args, resolved_dataset, difficulty):
        candidates, sampler_meta = build_candidates(
            rows,
            difficulty=difficulty,
            cache_root=cache_root,
            use_history=use_history,
        )
        selected = select_anchor_neighborhood(
            candidates,
            anchor_ids=DEFAULT_ANCHOR_IDS,
            count=args.sample_size,
        )
        selected_ids = tuple(candidate.instance_id for candidate in selected)
        if selected_ids != DEFAULT_EXPECTED_IDS:
            raise SystemExit(
                "Pinned homogeneous selector drifted from the validated five-task panel: "
                + ", ".join(selected_ids)
            )
        sampler_meta["strategy"] = (
            "homogeneous anchor neighborhood around django__django-13516 and "
            "pytest-dev__pytest-7571; similarity favored over broad coverage"
        )
        sampler_meta["anchors"] = list(DEFAULT_ANCHOR_IDS)
        sampler_meta["selection_goal"] = (
            "minimize expected between-task token/time variance for the "
            "Caveman x Ponytail treatment experiment"
        )
        print("  selection: homogeneous anchor neighborhood")

        pool_rows = candidate_rows(candidates)
        for row in pool_rows:
            row.pop("problem_statement", None)
        write_csv(output / "candidate_pool.csv", pool_rows)

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
        "dataset_revision": dataset_revision(resolved_dataset),
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
        "reasoning": PINNED_REASONING,
        "max_turns": PINNED_MAX_TURNS,
        "swebench_version": "4.1.0",
        "hermes_runtime": args.hermes_runtime,
        "hermes_image_ref": args.hermes_image if args.hermes_runtime == "docker" else None,
        "hermes_image_id": None,
        "ponytail_repo": PONY_REPO,
        "ponytail_commit": PONY_SHA,
        "caveman_commit": CAVE_SHA,
        "arms": list(args.arms),
        "available_arms": list(ARMS),
        "treatment_design": (
            "default Caveman x Ponytail 2x2 on normal hermes-cli tools; "
            "lean-tool variants remain optional"
        ),
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

    hermes_image = _preflight(args)
    if hermes_image:
        meta["hermes_image_id"] = hermes_image
        (output / "metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )

    home = source_home()

    arms = list(args.arms)
    instances = [row_by_id[candidate.instance_id] for candidate in selected]
    task_ids = [str(instance["instance_id"]) for instance in instances]
    instance_by_id = {
        str(instance["instance_id"]): instance for instance in instances
    }

    # Exercise repository materialization once before any paid model call. If
    # cache/workspace plumbing is broken, abort here instead of producing a
    # page of zero-token INVALID observations.
    with tempfile.TemporaryDirectory(prefix="forge-bench-workspace-smoke-") as smoke:
        smoke_instance = instances[0]
        prepare_workspace(
            cache_root,
            str(smoke_instance["repo"]),
            str(smoke_instance["base_commit"]),
            Path(smoke) / "workspace",
        )

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
    print("  reasoning:", PINNED_REASONING)
    print("  Hermes runtime:", args.hermes_runtime)
    if hermes_image:
        print("  Hermes image:", hermes_image)
    print("  randomized runs:", len(plan), "seed=", args.seed)
    print("  official grading:", not args.skip_evaluation)
    print("  output:", output)

    results: list[Result] = []

    with tempfile.TemporaryDirectory(prefix="forge-bench-") as tempdir:
        root = Path(tempdir)
        templates: dict[str, Path] = {}

        # Install treatment additions once per arm, then clone the resulting
        # profile for every individual run. This prevents state.db, memory,
        # Ponytail state, or any other session artifact from leaking between
        # tasks or repeats.
        for arm in arms:
            template = root / "templates" / arm
            make_profile(home, template)
            install_arm(
                args.hermes,
                template,
                arm,
                runtime=args.hermes_runtime,
                image=hermes_image,
            )
            for state_name in ("state.db", "state.db-shm", "state.db-wal"):
                state = template / state_name
                if state.exists():
                    state.unlink()
            templates[arm] = template

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

            run_profile = root / "run-profiles" / f"{run_index:02d}__{arm}"
            run_profile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(templates[arm], run_profile, symlinks=True)

            try:
                result = run_one(
                    args.hermes,
                    run_profile,
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
                    hermes_runtime=args.hermes_runtime,
                    hermes_image=hermes_image,
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
                print("    HARNESS ERROR:", str(exc).splitlines()[0])
                result = failed_result(
                    arm, instance, repeat, run_index, run_dir, exc
                )
            finally:
                shutil.rmtree(run_profile, ignore_errors=True)

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
