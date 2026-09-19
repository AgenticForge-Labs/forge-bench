from __future__ import annotations

import hashlib
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
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PONY_REPO,
    PONY_SHA,
    QUIX_SHA,
    QUIX_URL,
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


def hashfile(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Clone credentials while normalizing model/provider routing."""
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

    # These short one-shot jobs should never need compression. Disabling title
    # upgrades also keeps auxiliary LLM calls out of the treatment comparison.
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
    caveman, ponytail = ARMS.get(arm, (False, False))
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


def get_quixbugs(cache: Path) -> Path:
    dst = cache / ("QuixBugs-" + QUIX_SHA[:12])
    marker = dst / ".forge-bench-pin"
    if (
        dst.exists()
        and marker.exists()
        and marker.read_text(encoding="utf-8").strip() == QUIX_SHA
    ):
        return dst

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "--quiet", QUIX_URL, str(dst)], timeout=180, check=True)
    sh(
        ["git", "checkout", "--quiet", QUIX_SHA],
        cwd=dst,
        timeout=60,
        check=True,
    )
    marker.write_text(QUIX_SHA + "\n", encoding="utf-8")
    return dst


def prepare_workspace(source: Path, task: str, dst: Path) -> dict[str, str]:
    (dst / "python_programs").mkdir(parents=True)
    (dst / "json_testcases").mkdir()
    shutil.copy2(
        source / "python_programs" / f"{task}.py",
        dst / "python_programs" / f"{task}.py",
    )
    shutil.copy2(
        source / "json_testcases" / f"{task}.json",
        dst / "json_testcases" / f"{task}.json",
    )
    (dst / "python_programs" / "__init__.py").write_text("", encoding="utf-8")

    verifier = f'''import importlib, json
from pathlib import Path

TASK = {task!r}
func = getattr(importlib.import_module("python_programs." + TASK), TASK)
failures = []
count = 0
for count, line in enumerate((Path("json_testcases") / (TASK + ".json")).read_text().splitlines(), 1):
    args, expected = json.loads(line)
    try:
        actual = func(*args)
    except Exception as exc:
        failures.append((count, repr(exc), expected))
        continue
    if actual != expected:
        failures.append((count, actual, expected))

if failures:
    print(*failures[:5], sep="\\n")
    raise SystemExit(1)
print(f"PASS {{TASK}} ({{count}} vectors)")
'''
    (dst / "verify.py").write_text(verifier, encoding="utf-8")
    (dst / "BENCHMARK.md").write_text(
        f"QuixBugs repair benchmark. Source commit: {QUIX_SHA}. "
        "Correct implementations are absent.\n",
        encoding="utf-8",
    )

    protected = {
        "verify.py": hashfile(dst / "verify.py"),
        f"json_testcases/{task}.json": hashfile(
            dst / "json_testcases" / f"{task}.json"
        ),
    }

    if sh([sys.executable, "verify.py"], cwd=dst, timeout=20).returncode == 0:
        raise RuntimeError(f"{task} unexpectedly passes before repair")

    sh(["git", "init", "--quiet"], cwd=dst, check=True)
    sh(
        ["git", "config", "user.email", "benchmark@forge-bench.local"],
        cwd=dst,
        check=True,
    )
    sh(
        ["git", "config", "user.name", "Forge Bench"],
        cwd=dst,
        check=True,
    )
    sh(["git", "add", "."], cwd=dst, check=True)
    sh(
        ["git", "commit", "--quiet", "-m", "benchmark baseline"],
        cwd=dst,
        check=True,
    )
    return protected


def benchmark_prompt(task: str, arm: str) -> str:
    caveman, _ = ARMS.get(arm, (False, False))
    treatment: list[str] = []

    if caveman:
        treatment.append(
            "Load and follow the installed caveman skill in full mode, "
            "without dropping exact commands or test evidence."
        )

    return "\n".join(
        [
            f"Repair the QuixBugs Python task {task} in this repository.",
            f"Fix only python_programs/{task}.py unless absolutely necessary.",
            "Do not edit verify.py, json_testcases, or BENCHMARK.md.",
            "Do not fetch or look up a solution online. Do not delegate to a subagent.",
            "Make the smallest correct fix. Run python verify.py and finish only after it passes.",
            *treatment,
            "Final reply: state only what changed and whether python verify.py passed.",
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


def diff_stats(workspace: Path) -> tuple[int, int]:
    lines = sh(["git", "diff", "--numstat"], cwd=workspace).stdout.splitlines()
    changed = len(lines)
    diff_lines = 0
    for line in lines:
        fields = line.split("\t")[:2]
        diff_lines += sum(int(value) for value in fields if value.isdigit())
    return changed, diff_lines


def run_one(
    hermes: str,
    profile: Path,
    arm: str,
    task: str,
    repeat: int,
    run_index: int,
    qsrc: Path,
    output: Path,
    timeout: int,
) -> Result:
    run_dir = output / "runs" / f"{run_index:02d}__{arm}__{task}__r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    protected = prepare_workspace(qsrc, task, workspace)

    prompt = benchmark_prompt(task, arm)
    (run_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    usage_file = run_dir / "usage.json"

    env = profile_env(profile)
    if ARMS.get(arm, (False, False))[1]:
        env["PONYTAIL_DEFAULT_MODE"] = "full"

    argv = [
        hermes,
        "-z",
        prompt,
        "--provider",
        "openrouter",
        "--model",
        PINNED_MODEL,
        "--usage-file",
        str(usage_file),
    ]

    started = time.perf_counter()
    try:
        proc = sh(argv, cwd=workspace, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        (run_dir / "stdout.txt").write_text(
            str(exc.stdout or ""),
            encoding="utf-8",
        )
        (run_dir / "stderr.txt").write_text(
            str(exc.stderr or ""),
            encoding="utf-8",
        )
        return Result(
            arm,
            task,
            repeat,
            run_index,
            False,
            False,
            True,
            False,
            124,
            wall,
            PINNED_MODEL,
            "openrouter",
            PINNED_OPENROUTER_UPSTREAM,
            "",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            "unavailable",
            None,
            0,
            0,
            str(run_dir),
            "timeout",
        )

    wall = time.perf_counter() - started
    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    verify = sh([sys.executable, "verify.py"], cwd=workspace, timeout=20)
    (run_dir / "verify.txt").write_text(
        verify.stdout + verify.stderr,
        encoding="utf-8",
    )

    tests_passed = verify.returncode == 0
    tests_untouched = all(
        hashfile(workspace / rel) == expected
        for rel, expected in protected.items()
    )
    files_changed, diff_lines = diff_stats(workspace)

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

    valid = (
        tests_passed
        and tests_untouched
        and completed
        and model == PINNED_MODEL
        and provider.lower() == "openrouter"
    )

    errors: list[str] = []
    if not tests_passed:
        errors.append("verification failed")
    if not tests_untouched:
        errors.append("protected tests modified")
    if not completed:
        errors.append("Hermes incomplete")
    if model != PINNED_MODEL:
        errors.append("wrong model: " + model)
    if provider.lower() != "openrouter":
        errors.append("wrong API provider: " + provider)

    result = Result(
        arm=arm,
        task=task,
        repeat=repeat,
        run_index=run_index,
        valid=valid,
        tests_passed=tests_passed,
        tests_untouched=tests_untouched,
        completed=completed,
        exit_code=proc.returncode,
        wall_seconds=wall,
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
