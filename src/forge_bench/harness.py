from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ARMS,
    CAVE_URL,
    LEAN_TOOLSETS,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PONY_REPO,
    PONY_SHA,
    Result,
)


def sh(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    input_text: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode:
        raise RuntimeError(
            "command failed: "
            + " ".join(argv)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )
    return proc


def source_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser().resolve()


def load_source_config(home: Path) -> dict[str, Any]:
    path = home / "config.yaml"
    if not path.exists():
        raise SystemExit(f"Hermes config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Hermes config is not a mapping: {path}")
    return config


def make_profile(src: Path, config: dict[str, Any], dst: Path) -> None:
    """Clone credentials while making model/provider behavior deterministic."""
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(config))

    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    cfg["plugins"] = {**plugins, "enabled": [], "disabled": []}

    terminal = cfg.get("terminal") if isinstance(cfg.get("terminal"), dict) else {}
    cfg["terminal"] = {**terminal, "cwd": "."}

    for key in ("fallback_model", "fallback_providers", "smart_model_routing", "moa"):
        cfg.pop(key, None)

    cfg["provider_routing"] = {
        "only": [PINNED_OPENROUTER_UPSTREAM],
        "require_parameters": True,
        "models": {
            PINNED_MODEL: {
                "only": [PINNED_OPENROUTER_UPSTREAM],
                "require_parameters": True,
            }
        },
    }

    # Avoid auxiliary model calls changing the treatment totals. The selected
    # SWE-bench tasks are short enough that compression should not be necessary.
    cfg["compression"] = {"enabled": False}
    auxiliary = cfg.get("auxiliary") if isinstance(cfg.get("auxiliary"), dict) else {}
    auxiliary["title_generation"] = {
        "enabled": False,
        "model_upgrade_enabled": False,
        "provider": "auto",
        "model": "",
    }
    cfg["auxiliary"] = auxiliary

    (dst / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    for name in (".env", "auth.json"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def profile_env(profile: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(profile)
    env["HERMES_ENABLE_PROJECT_PLUGINS"] = "0"
    return env


def install_arm(hermes: str, profile: Path, arm: str) -> None:
    caveman, ponytail, _lean = ARMS.get(arm, (False, False, False))
    env = profile_env(profile)

    if ponytail:
        proc = sh(
            [hermes, "plugins", "install", PONY_REPO, "--ref", PONY_SHA, "--enable"],
            env=env,
            timeout=180,
            input_text="y\n" * 4,
        )
        if proc.returncode:
            raise RuntimeError("Ponytail install failed:\n" + proc.stdout + proc.stderr)

    if caveman:
        proc = sh(
            [
                hermes,
                "skills",
                "install",
                CAVE_URL,
                "--name",
                "caveman",
                "--force",
                "--yes",
            ],
            env=env,
            timeout=180,
        )
        if proc.returncode:
            raise RuntimeError("Caveman install failed:\n" + proc.stdout + proc.stderr)


def ensure_repo_cache(cache_root: Path, repo: str, base_commit: str) -> Path:
    """Create a base-commit-only bare cache.

    We intentionally do not clone repository heads. This prevents an agent from
    inspecting later commits or the solution PR through local git history.
    """
    bare = cache_root / "repos" / (repo.replace("/", "__") + ".git")
    if not bare.exists():
        bare.parent.mkdir(parents=True, exist_ok=True)
        init = sh(["git", "init", "--bare", str(bare)])
        if init.returncode:
            raise RuntimeError("git init failed: " + init.stderr)
        remote = sh(
            [
                "git",
                "--git-dir",
                str(bare),
                "remote",
                "add",
                "origin",
                f"https://github.com/{repo}.git",
            ]
        )
        if remote.returncode:
            raise RuntimeError("git remote add failed: " + remote.stderr)

    fetch = sh(
        [
            "git",
            "--git-dir",
            str(bare),
            "fetch",
            "--filter=blob:none",
            "--depth=1",
            "origin",
            base_commit,
        ],
        timeout=300,
    )
    if fetch.returncode:
        raise RuntimeError(
            f"Could not fetch {repo}@{base_commit}: " + fetch.stderr
        )

    return bare


def prepare_workspace(
    cache_root: Path,
    repo: str,
    base_commit: str,
    destination: Path,
) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    bare = ensure_repo_cache(cache_root, repo, base_commit)
    clone = sh(["git", "clone", "--shared", str(bare), str(destination)], timeout=300)
    if clone.returncode:
        raise RuntimeError("workspace clone failed: " + clone.stderr)

    checkout = sh(
        ["git", "checkout", "--detach", base_commit],
        cwd=destination,
        timeout=120,
    )
    if checkout.returncode:
        raise RuntimeError("base commit checkout failed: " + checkout.stderr)

    # Remove network remotes from the agent workspace. The issue statement and
    # base commit are the only task information the agent should receive.
    sh(["git", "remote", "remove", "origin"], cwd=destination)
    sh(["git", "reset", "--hard", base_commit], cwd=destination, check=True)
    sh(["git", "clean", "-fdx"], cwd=destination, check=True)


def benchmark_prompt(instance: dict[str, Any], arm: str) -> str:
    caveman, _ponytail, _lean = ARMS.get(arm, (False, False, False))
    treatment: list[str] = []

    if caveman:
        treatment.append(
            "Load and follow the installed caveman skill in full mode, "
            "without dropping commands or verification evidence."
        )

    issue = str(instance.get("problem_statement") or "").strip()
    return "\n".join(
        [
            "You are solving a SWE-bench software engineering task.",
            f"Repository: {instance['repo']}",
            f"Instance: {instance['instance_id']}",
            "",
            "Issue:",
            issue,
            "",
            "Constraints:",
            "- Work only from this repository state and the issue above.",
            "- Do not look up the issue, pull request, gold patch, or solution online.",
            "- Do not git fetch or add a network git remote.",
            "- Make the smallest complete production-code fix that addresses the issue.",
            "- Inspect relevant code before editing.",
            "- Run relevant tests or targeted checks when the local environment permits.",
            "- Do not modify tests merely to make them pass.",
            *[f"- {item}" for item in treatment],
            "",
            "Finish with a concise summary of the fix and checks performed.",
        ]
    )


def as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def session_db_values(profile: Path, session_id: str) -> dict[str, Any]:
    db = profile / "state.db"
    if not db.exists() or not session_id:
        return {}

    try:
        con = sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        columns = [row[1] for row in con.execute("pragma table_info(sessions)")]
        key = (
            "session_id"
            if "session_id" in columns
            else "id"
            if "id" in columns
            else ""
        )
        row = (
            con.execute(
                "select * from sessions where " + key + " = ?",
                (session_id,),
            ).fetchone()
            if key
            else None
        )
        con.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def diff_stats(workspace: Path) -> tuple[int, int, str]:
    patch = sh(["git", "diff", "--binary"], cwd=workspace).stdout
    numstat = sh(["git", "diff", "--numstat"], cwd=workspace).stdout.splitlines()
    changed = len(numstat)
    diff_lines = 0
    for line in numstat:
        fields = line.split("\t")[:2]
        diff_lines += sum(int(value) for value in fields if value.isdigit())
    return changed, diff_lines, patch


def evaluate_patch(
    run_dir: Path,
    dataset_name: str,
    instance_id: str,
    patch: str,
    *,
    timeout: int,
) -> tuple[bool, bool, float, str]:
    """Grade one patch with the official SWE-bench Docker harness."""
    if not patch.strip():
        return False, False, 0.0, "empty patch"

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    prediction = eval_dir / "prediction.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": "forge-bench",
                "model_patch": patch,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_id = "forge_bench_" + run_dir.name.replace("-", "_")
    argv = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(prediction),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--instance_ids",
        instance_id,
        "--cache_level",
        "instance",
    ]

    started = time.perf_counter()
    try:
        proc = sh(argv, cwd=eval_dir, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        (eval_dir / "stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
        (eval_dir / "stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
        return False, False, elapsed, "evaluation timeout"

    elapsed = time.perf_counter() - started
    (eval_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (eval_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    report = None
    report_root = eval_dir / "logs" / "evaluation" / run_id
    for path in sorted(report_root.rglob("results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "resolved_ids" in payload:
            report = payload
            break

    if report is None:
        return False, False, elapsed, "evaluation report not found"

    resolved = instance_id in set(report.get("resolved_ids") or [])
    return True, resolved, elapsed, ""


def run_one(
    hermes: str,
    profile: Path,
    instance: dict[str, Any],
    dataset_name: str,
    arm: str,
    repeat: int,
    run_index: int,
    cache_root: Path,
    output: Path,
    timeout: int,
    evaluation_timeout: int,
    evaluate: bool,
) -> Result:
    instance_id = str(instance["instance_id"])
    repo = str(instance["repo"])
    difficulty = str(instance.get("difficulty") or "")
    task_slug = instance_id.replace("/", "__")
    run_dir = output / "runs" / f"{run_index:02d}__{arm}__{task_slug}__r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"

    prepare_workspace(
        cache_root,
        repo,
        str(instance["base_commit"]),
        workspace,
    )

    prompt = benchmark_prompt(instance, arm)
    (run_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    usage_file = run_dir / "usage.json"

    env = profile_env(profile)
    _caveman, ponytail, lean = ARMS.get(arm, (False, False, False))
    if ponytail:
        env["PONYTAIL_DEFAULT_MODE"] = "full"

    argv = [
        hermes,
        "-z",
        prompt,
    ]
    if lean:
        argv.extend(["--toolsets", LEAN_TOOLSETS])
    argv.extend([
        "--provider",
        "openrouter",
        "--model",
        PINNED_MODEL,
        "--usage-file",
        str(usage_file),
    ])

    started = time.perf_counter()
    try:
        proc = sh(argv, cwd=workspace, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        (run_dir / "stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
        (run_dir / "stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
        return Result(
            arm=arm,
            task=instance_id,
            repeat=repeat,
            run_index=run_index,
            valid=False,
            resolved=False,
            evaluation_completed=False,
            patch_nonempty=False,
            completed=False,
            exit_code=124,
            wall_seconds=wall,
            evaluation_seconds=0.0,
            repo=repo,
            difficulty=difficulty,
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
            error="Hermes timeout",
        )

    wall = time.perf_counter() - started
    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    files_changed, diff_lines, patch = diff_stats(workspace)
    (run_dir / "model.patch").write_text(patch, encoding="utf-8")
    patch_nonempty = bool(patch.strip())

    evaluation_completed = False
    resolved = False
    evaluation_seconds = 0.0
    eval_error = ""

    if evaluate:
        (
            evaluation_completed,
            resolved,
            evaluation_seconds,
            eval_error,
        ) = evaluate_patch(
            run_dir,
            dataset_name,
            instance_id,
            patch,
            timeout=evaluation_timeout,
        )
    elif not patch_nonempty:
        eval_error = "evaluation skipped; empty patch"

    try:
        usage = (
            json.loads(usage_file.read_text(encoding="utf-8"))
            if usage_file.exists()
            else {}
        )
    except Exception:
        usage = {}

    grand = usage.get("total_including_auxiliary") or {}
    session_id = str(usage.get("session_id") or "")
    db = session_db_values(profile, session_id)

    estimated = as_float(grand.get("estimated_cost_usd"))
    if estimated is None:
        estimated = as_float(usage.get("estimated_cost_usd"))

    actual = as_float(grand.get("actual_cost_usd"))
    if actual is None:
        actual = as_float(usage.get("actual_cost_usd"))
    if actual is None:
        actual = as_float(db.get("actual_cost_usd"))

    cost = actual if actual is not None else estimated
    cost_source = (
        "openrouter_actual"
        if actual is not None
        else "hermes_estimate"
        if estimated is not None
        else "unavailable"
    )

    model = str(usage.get("model") or PINNED_MODEL)
    provider = str(usage.get("provider") or "openrouter")
    completed = (
        bool(usage.get("completed", proc.returncode == 0))
        and proc.returncode == 0
    )

    # "valid" means the agent run is usable for efficiency analysis. An
    # unresolved SWE-bench task is a legitimate outcome and stays in the data.
    eval_ok = (
        not evaluate
        or evaluation_completed
        or not patch_nonempty
    )
    valid = (
        completed
        and model == PINNED_MODEL
        and provider.lower() == "openrouter"
        and eval_ok
    )

    errors: list[str] = []
    if not completed:
        errors.append("Hermes incomplete")
    if model != PINNED_MODEL:
        errors.append("wrong model: " + model)
    if provider.lower() != "openrouter":
        errors.append("wrong API provider: " + provider)
    if eval_error:
        errors.append(eval_error)

    result = Result(
        arm=arm,
        task=instance_id,
        repeat=repeat,
        run_index=run_index,
        valid=valid,
        resolved=resolved,
        evaluation_completed=evaluation_completed,
        patch_nonempty=patch_nonempty,
        completed=completed,
        exit_code=proc.returncode,
        wall_seconds=wall,
        evaluation_seconds=evaluation_seconds,
        repo=repo,
        difficulty=difficulty,
        model=model,
        api_provider=provider,
        upstream_provider=PINNED_OPENROUTER_UPSTREAM,
        session_id=session_id,
        input_tokens=as_int(usage.get("input_tokens")),
        output_tokens=as_int(usage.get("output_tokens")),
        reasoning_tokens=as_int(usage.get("reasoning_tokens")),
        cache_read_tokens=as_int(usage.get("cache_read_tokens")),
        cache_write_tokens=as_int(usage.get("cache_write_tokens")),
        total_tokens=(
            as_int(grand.get("total_tokens"))
            or as_int(usage.get("total_tokens"))
        ),
        api_calls=(
            as_int(grand.get("api_calls"))
            or as_int(usage.get("api_calls"))
        ),
        estimated_cost_usd=estimated,
        actual_cost_usd=actual,
        cost_usd=cost,
        cost_source=cost_source,
        tool_calls=(
            as_int(db.get("tool_call_count"))
            if db.get("tool_call_count") is not None
            else None
        ),
        files_changed=files_changed,
        diff_lines=diff_lines,
        run_dir=str(run_dir),
        error="; ".join(errors),
    )
    (run_dir / "result.json").write_text(
        json.dumps(result.__dict__, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
