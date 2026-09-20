from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import webbrowser
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
from .designs import ModelSpec, default_design, load_design
from .harness import (
    install_arm,
    make_profile,
    prepare_workspace,
    run_one,
    sh,
    source_home,
)
from .reporting import reanalyze_output, write_csv, write_experiment_reports
from .trace_capture import install_trace_plugin

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
        "--design",
        type=Path,
        help=(
            "YAML experiment design. When supplied, it pins dataset selection, "
            "models, upstream provider(s), treatments, randomization blocks, "
            "seed, reasoning, max turns, and analysis mode. Runtime/output "
            "controls remain CLI options."
        ),
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of independently randomized repeat blocks. Repeats are "
            "averaged within task/treatment before across-task inference."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Master randomization seed. Each repeat block receives its own "
            "deterministically derived seed, recorded in metadata/run_plan.csv."
        ),
    )
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
        "--analysis-mode",
        choices=("basic", "advanced"),
        default="basic",
        help=(
            "basic = web-ready summary plots/report; advanced additionally runs "
            "paired effects, regressions, PCA, clustering, and correlations."
        ),
    )
    parser.add_argument(
        "--reanalyze",
        type=Path,
        help=(
            "Regenerate reports/figures from an existing Forge Bench output "
            "directory without model calls. Each invocation writes to a fresh "
            "timestamped subfolder under <run>/reanalysis/. Usually combine "
            "with --analysis-mode advanced."
        ),
    )
    parser.add_argument(
        "--open",
        dest="open_report",
        action="store_true",
        help="Open the generated report.html in the default browser when finished.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/agenticforge/forge-bench").expanduser(),
    )
    return parser.parse_args()


def open_report(path: Path) -> bool:
    """Open an HTML report in the user's default browser without failing the run."""
    try:
        return bool(webbrowser.open(path.expanduser().resolve().as_uri(), new=2))
    except Exception as exc:
        print(f"Warning: could not open report automatically: {exc}")
        return False


def failed_result(
    arm: str,
    instance: dict[str, Any],
    repeat: int,
    run_index: int,
    run_dir: Path,
    error: Exception,
    model_spec: ModelSpec | None = None,
) -> Result:
    model_spec = model_spec or default_design().models[0]
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
        model=model_spec.model,
        api_provider=model_spec.api_provider,
        upstream_provider=model_spec.upstream_provider,
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
            (
                [*probe_prefix, "sessions", "export", "--help"],
                ["--format", "--session-id"],
                "Hermes Docker session exporter",
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
            (
                [args.hermes, "sessions", "export", "--help"],
                ["--format", "--session-id"],
                "Hermes session exporter",
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

def build_run_plan(
    arms: list[str],
    task_ids: list[str],
    repeats: int,
    master_seed: int,
    models: list[ModelSpec] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Build complete model×treatment×task blocks and shuffle each block once."""
    seed_rng = random.Random(master_seed)
    repeat_seeds = [master_seed]
    while len(repeat_seeds) < repeats:
        candidate = seed_rng.randrange(1, 2**63)
        if candidate not in repeat_seeds:
            repeat_seeds.append(candidate)

    model_specs = models or list(default_design().models)
    plan: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = repeat_seeds[repeat - 1]
        block = [
            {
                "model_key": model_spec.key,
                "model_label": model_spec.label,
                "model": model_spec.model,
                "api_provider": model_spec.api_provider,
                "upstream_provider": model_spec.upstream_provider,
                "reasoning": model_spec.reasoning,
                "arm": arm,
                "instance_id": instance_id,
                "repeat": repeat,
                "repeat_seed": repeat_seed,
            }
            for model_spec in model_specs
            for instance_id in task_ids
            for arm in arms
        ]
        random.Random(repeat_seed).shuffle(block)
        for block_position, item in enumerate(block, 1):
            item["block_position"] = block_position
            plan.append(item)

    for run_index, item in enumerate(plan, 1):
        item["run_index"] = run_index
    return plan, repeat_seeds


def main() -> int:
    args = parse_args()

    design = load_design(args.design) if args.design is not None else default_design()
    if args.design is not None:
        args.dataset = design.dataset
        args.split = design.split
        args.difficulty = design.difficulty
        args.sample_size = design.sample_size
        args.instance_ids = list(design.instance_ids) if design.instance_ids else None
        args.smart_sample = False
        args.arms = list(design.arms)
        args.repeats = design.blocks
        args.seed = design.seed
        args.analysis_mode = design.analysis_mode

    if args.reanalyze is not None:
        source = args.reanalyze.expanduser().resolve()
        output = reanalyze_output(source, analysis_mode=args.analysis_mode)
        print("Reanalysis source:", source)
        print("Analysis mode:", args.analysis_mode)
        light_report = output / "report-light.html"
        dark_report = output / "report-dark.html"
        print("Reanalysis output:", output)
        print("Light report:", light_report)
        print("Dark report:", dark_report)
        if args.open_report and not open_report(light_report):
            print("Open manually with: xdg-open", light_report)
        return 0

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
    if args.design is not None:
        shutil.copy2(args.design.expanduser().resolve(), output / "design.yaml")

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
        "design_name": design.name,
        "design_source": design.source_path,
        "design_description": design.description,
        "model": design.models[0].model if len(design.models) == 1 else None,
        "models": [model.as_dict() for model in design.models],
        "api_provider": (
            design.models[0].api_provider
            if len({model.api_provider for model in design.models}) == 1
            else "mixed"
        ),
        "upstream_provider": (
            design.models[0].upstream_provider
            if len({model.upstream_provider for model in design.models}) == 1
            else "mixed"
        ),
        "reasoning": (
            design.models[0].reasoning
            if len({model.reasoning for model in design.models}) == 1
            else "mixed"
        ),
        "max_turns": design.max_turns,
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
        "repeat_seeds": [],
        "randomization": (
            "full-factorial randomized complete blocks: each block contains "
            "every model x treatment x selected-instance cell exactly once; "
            "all cells are shuffled together with the block's recorded repeat_seed"
        ),
        "confidence_interval": (
            "two-sided 95% Student-t across selected task means; "
            "repeats are averaged within model x treatment x task first"
        ),
        "trace_capture": {
            "enabled": True,
            "mechanism": "observer-only native Hermes plugin hooks",
            "artifacts": [
                "hermes-events.jsonl",
                "api-events.jsonl",
                "tool-events.jsonl",
                "lifecycle-events.jsonl",
                "timing-summary.json",
                "hermes-session.json",
                "hermes-session.jsonl",
                "hermes-session.trace.jsonl",
                "hermes-state.db",
                "treatment-evidence.json",
            ],
        },
        "official_evaluation": not args.skip_evaluation,
        "analysis_mode": args.analysis_mode,
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
    models = list(design.models)
    model_by_key = {model.key: model for model in models}
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

    plan, repeat_seeds = build_run_plan(
        arms,
        task_ids,
        args.repeats,
        args.seed,
        models=models,
    )
    meta["repeat_seeds"] = repeat_seeds
    (output / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "run_plan.csv", plan)

    print("\nForge Bench execution")
    print("  design:", design.name)
    print("  models:")
    for model_spec in models:
        print(
            "   ",
            model_spec.label,
            "=>",
            model_spec.model,
            f"via {model_spec.api_provider}/{model_spec.upstream_provider}",
            f"reasoning={model_spec.reasoning}",
        )
    print("  Hermes runtime:", args.hermes_runtime)
    if hermes_image:
        print("  Hermes image:", hermes_image)
    print("  randomized runs:", len(plan), "master seed=", args.seed)
    print("  repeat seeds:", ", ".join(str(seed) for seed in repeat_seeds))
    print("  official grading:", not args.skip_evaluation)
    print("  output:", output)

    results: list[Result] = []

    with tempfile.TemporaryDirectory(prefix="forge-bench-") as tempdir:
        root = Path(tempdir)
        templates: dict[tuple[str, str], Path] = {}

        # Build one pristine template per model x treatment so provider routing
        # is pinned before the run begins. Individual observations still receive
        # a fresh copy, preventing session/state leakage across randomized cells.
        for model_spec in models:
            for arm in arms:
                template = root / "templates" / model_spec.key / arm
                make_profile(
                    home,
                    template,
                    model=model_spec.model,
                    upstream_provider=model_spec.upstream_provider,
                    max_turns=design.max_turns,
                )
                install_arm(
                    args.hermes,
                    template,
                    arm,
                    runtime=args.hermes_runtime,
                    image=hermes_image,
                )
                # Treatment installers may edit plugins.enabled. Reassert the
                # observer after installation so every arm is instrumented equally.
                install_trace_plugin(template)
                for state_name in ("state.db", "state.db-shm", "state.db-wal"):
                    state = template / state_name
                    if state.exists():
                        state.unlink()
                # Admin-time plugin/skill installation must never leak observer
                # events or export artifacts into an experimental run profile.
                for artifact_name in (
                    "forge-bench-events.jsonl",
                    "benchmark-session.jsonl",
                    "benchmark-session.trace.jsonl",
                ):
                    artifact = template / artifact_name
                    if artifact.exists():
                        artifact.unlink()
                templates[(model_spec.key, arm)] = template

        for item in plan:
            model_key = str(item["model_key"])
            model_spec = model_by_key[model_key]
            arm = str(item["arm"])
            instance_id = str(item["instance_id"])
            instance = instance_by_id[instance_id]
            repeat = int(item["repeat"])
            run_index = int(item["run_index"])

            print(
                f"[{run_index:02d}/{len(plan):02d}] "
                f"{model_spec.label} / {LABEL[arm]} / {instance_id} / b{repeat}"
            )

            run_profile = root / "run-profiles" / f"{run_index:02d}__{model_key}__{arm}"
            run_profile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(templates[(model_key, arm)], run_profile, symlinks=True)

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
                    model=model_spec.model,
                    upstream_provider=model_spec.upstream_provider,
                    reasoning=model_spec.reasoning,
                    model_key=model_spec.key,
                )
            except Exception as exc:
                run_dir = (
                    output
                    / "runs"
                    / f"{run_index:02d}__{model_key}__{arm}__{instance_id}__r{repeat}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "harness_error.txt").write_text(
                    str(exc) + "\n",
                    encoding="utf-8",
                )
                print("    HARNESS ERROR:", str(exc).splitlines()[0])
                result = failed_result(
                    arm, instance, repeat, run_index, run_dir, exc, model_spec
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

    write_experiment_reports(
        output,
        results,
        arms,
        task_ids,
        meta,
        analysis_mode=args.analysis_mode,
    )

    partial = output / "runs.partial.json"
    if partial.exists():
        partial.unlink()

    usable = sum(result.valid for result in results)
    solved = sum(result.resolved for result in results if result.valid)
    print(f"\nCompleted: {usable}/{len(results)} usable runs; {solved} resolved")
    light_report = output / "report-light.html"
    dark_report = output / "report-dark.html"
    print("Light report:", light_report)
    print("Dark report:", dark_report)
    print("PNG figures:", output / "*.png")
    if args.open_report and not open_report(light_report):
        print("Open manually with: xdg-open", light_report)
    return 0 if usable == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
