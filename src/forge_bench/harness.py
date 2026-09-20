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

from .trace_capture import finalize_event_artifacts, install_trace_plugin, trace_plugin_enabled

from .config import (
    ARMS,
    CAVE_URL,
    DEFAULT_TOOLSETS,
    LEAN_TOOLSETS,
    PINNED_MAX_TURNS,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PINNED_REASONING,
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


def make_profile(src: Path, dst: Path) -> None:
    """Create a fresh, minimal Hermes home containing only benchmark settings and credentials."""
    dst.mkdir(parents=True, exist_ok=True)

    cfg: dict[str, Any] = {
        "_config_version": 45,
        "plugins": {"enabled": [], "disabled": []},
        "agent": {"max_turns": PINNED_MAX_TURNS},
        "provider_routing": {
            "only": [PINNED_OPENROUTER_UPSTREAM],
            "require_parameters": True,
            "models": {
                PINNED_MODEL: {
                    "only": [PINNED_OPENROUTER_UPSTREAM],
                    "require_parameters": True,
                }
            },
        },
        "compression": {"enabled": False},
        "auxiliary": {
            "title_generation": {
                "enabled": False,
                "model_upgrade_enabled": False,
                "provider": "auto",
                "model": "",
            }
        },
    }

    (dst / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    # Prevent the official Docker image from seeding the full bundled skill
    # catalog into each fresh benchmark profile. Caveman is installed explicitly
    # when its treatment is active.
    (dst / ".no-bundled-skills").write_text(
        "Forge Bench minimal profile\n",
        encoding="utf-8",
    )

    for name in (".env", "auth.json"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)

    install_trace_plugin(dst)


def profile_env(profile: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("HERMES_") or key.startswith("PONYTAIL_"):
            env.pop(key, None)

    xdg = profile / "xdg-config"
    xdg.mkdir(parents=True, exist_ok=True)
    env["HERMES_HOME"] = str(profile)
    env["HERMES_ENABLE_PROJECT_PLUGINS"] = "0"
    env["XDG_CONFIG_HOME"] = str(xdg)
    return env


def docker_hermes_argv(
    image: str,
    profile: Path,
    hermes_args: list[str],
    *,
    workspace: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Build an ephemeral official-Hermes Docker invocation."""
    uid = getattr(os, "getuid", lambda: 1000)()
    gid = getattr(os, "getgid", lambda: 1000)()

    argv = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "-e",
        f"HERMES_UID={uid}",
        "-e",
        f"HERMES_GID={gid}",
        "-e",
        "HERMES_ENABLE_PROJECT_PLUGINS=0",
        "-e",
        "XDG_CONFIG_HOME=/opt/data/xdg-config",
        "-v",
        f"{profile.resolve()}:/opt/data",
    ]

    # Support users who keep the OpenRouter key only in their shell rather than
    # ~/.hermes/.env. Docker's "-e NAME" form forwards the current value.
    if os.environ.get("OPENROUTER_API_KEY"):
        argv.extend(["-e", "OPENROUTER_API_KEY"])

    for key, value in (extra_env or {}).items():
        argv.extend(["-e", f"{key}={value}"])

    if workspace is not None:
        argv.extend(
            [
                "-v",
                f"{workspace.resolve()}:/workspace",
                "-w",
                "/workspace",
            ]
        )

    argv.append(image)
    argv.extend(hermes_args)
    return argv


def hermes_admin(
    hermes: str,
    profile: Path,
    hermes_args: list[str],
    *,
    runtime: str,
    image: str | None,
    timeout: int,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if runtime == "docker":
        if not image:
            raise RuntimeError("Docker Hermes image was not resolved")
        argv = docker_hermes_argv(image, profile, hermes_args)
        return sh(argv, timeout=timeout, input_text=input_text)

    return sh(
        [hermes, *hermes_args],
        env=profile_env(profile),
        timeout=timeout,
        input_text=input_text,
    )



def install_arm(
    hermes: str,
    profile: Path,
    arm: str,
    *,
    runtime: str = "local",
    image: str | None = None,
) -> None:
    caveman, ponytail, _lean = ARMS.get(arm, (False, False, False))

    if ponytail:
        proc = hermes_admin(
            hermes,
            profile,
            [
                "plugins",
                "install",
                PONY_REPO,
                "--ref",
                PONY_SHA,
                "--force",
                "--enable",
            ],
            runtime=runtime,
            image=image,
            timeout=240,
            input_text="y\n" * 4,
        )
        if proc.returncode:
            raise RuntimeError("Ponytail install failed:\n" + proc.stdout + proc.stderr)

    if caveman:
        proc = hermes_admin(
            hermes,
            profile,
            [
                "skills",
                "install",
                CAVE_URL,
                "--name",
                "caveman",
                "--force",
                "--yes",
            ],
            runtime=runtime,
            image=image,
            timeout=240,
        )
        if proc.returncode:
            raise RuntimeError("Caveman install failed:\n" + proc.stdout + proc.stderr)


def ensure_repo_cache(cache_root: Path, repo: str, base_commit: str) -> Path:
    """Create a base-commit-only bare cache.

    We intentionally do not clone repository heads. This prevents an agent from
    inspecting later commits or the solution PR through local git history.
    """
    bare = cache_root / "repos" / (repo.replace("/", "__") + ".git")

    # Self-heal caches created by older Forge Bench versions that used
    # --filter=blob:none. Those promisor repositories can fail when a local
    # workspace asks them for an object they never downloaded.
    if bare.exists():
        partial_checks = [
            ["git", "--git-dir", str(bare), "config", "--get", "remote.origin.promisor"],
            ["git", "--git-dir", str(bare), "config", "--get", "remote.origin.partialclonefilter"],
            ["git", "--git-dir", str(bare), "config", "--get", "extensions.partialClone"],
        ]
        if any(sh(argv).returncode == 0 for argv in partial_checks):
            shutil.rmtree(bare)

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
            "--no-tags",
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

    # Older Forge Bench revisions created this cache as a partial/promisor
    # repository with --filter=blob:none. Such a cache can be missing objects
    # and may attempt lazy network fetches when another local repository reads
    # from it. Clear those settings once a full commit has been fetched.
    sh(
        ["git", "--git-dir", str(bare), "config", "--unset-all", "remote.origin.promisor"],
    )
    sh(
        ["git", "--git-dir", str(bare), "config", "--unset-all", "remote.origin.partialclonefilter"],
    )
    sh(
        ["git", "--git-dir", str(bare), "config", "--unset-all", "extensions.partialClone"],
    )

    return bare


def prepare_workspace(
    cache_root: Path,
    repo: str,
    base_commit: str,
    destination: Path,
) -> None:
    """Materialize exactly one SWE-bench base commit into a self-contained repo."""
    if destination.exists():
        shutil.rmtree(destination)

    bare = ensure_repo_cache(cache_root, repo, base_commit)
    destination.mkdir(parents=True, exist_ok=True)

    init = sh(["git", "init", "-q"], cwd=destination, timeout=60)
    if init.returncode:
        raise RuntimeError("workspace git init failed: " + init.stderr)

    # Fetch the exact cached commit into a real local ref. The bare cache keeps
    # exact-SHA fetches only in FETCH_HEAD, so a normal clone can legitimately
    # omit them because no branch points at the object.
    fetch = sh(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "--no-tags",
            "--depth=1",
            str(bare),
            base_commit,
        ],
        cwd=destination,
        timeout=300,
    )
    if fetch.returncode:
        raise RuntimeError("workspace fetch from cache failed: " + fetch.stderr)

    checkout = sh(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=destination,
        timeout=120,
    )
    if checkout.returncode:
        raise RuntimeError("base commit checkout failed: " + checkout.stderr)

    actual = sh(["git", "rev-parse", "HEAD"], cwd=destination, check=True).stdout.strip()
    if actual != base_commit:
        raise RuntimeError(
            f"workspace commit mismatch: expected {base_commit}, got {actual}"
        )

    # No remote is configured in this workspace. Hermes receives only the
    # checked-out base commit and cannot inspect later repository history.
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
            "- Work only from this supplied repository state and issue statement.",
            "- Do not access the internet, external repositories, issue trackers, pull requests, patches, or prior benchmark sessions.",
            "- Do not git fetch, add a network git remote, or delegate the task to another agent.",
            "- Otherwise use the normal local Hermes tools and workflow you consider useful.",
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


def diff_stats(workspace: Path, base_commit: str) -> tuple[int, int, str]:
    """Capture the full working tree relative to base without touching its real index.

    A temporary index makes committed changes, staged changes, deletions, and
    untracked non-ignored files all appear in the model patch.
    """
    index_path = workspace.parent / ".forge-bench-index"
    if index_path.exists():
        index_path.unlink()

    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(index_path)

    try:
        sh(["git", "read-tree", base_commit], cwd=workspace, env=env, check=True)
        sh(["git", "add", "-A", "--", "."], cwd=workspace, env=env, check=True)

        patch_proc = sh(
            ["git", "diff", "--cached", "--binary", base_commit],
            cwd=workspace,
            env=env,
            check=True,
        )
        numstat_proc = sh(
            ["git", "diff", "--cached", "--numstat", base_commit],
            cwd=workspace,
            env=env,
            check=True,
        )
        patch = patch_proc.stdout
        numstat = numstat_proc.stdout.splitlines()
    finally:
        if index_path.exists():
            index_path.unlink()

    changed = len(numstat)
    diff_lines = 0
    for line in numstat:
        fields = line.split("\t")[:2]
        diff_lines += sum(int(value) for value in fields if value.isdigit())
    return changed, diff_lines, patch


def load_swebench_report(
    eval_dir: Path,
    model_name: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Read the run summary from SWE-bench 4.1 or newer report layouts."""
    report_paths = [
        eval_dir / f"{model_name}.{run_id}.json",
        eval_dir / "logs" / "evaluation" / run_id / "results.json",
    ]
    for path in report_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "resolved_ids" in payload:
            return payload
    return None


def evaluate_patch(
    run_dir: Path,
    instance: dict[str, Any],
    patch: str,
    *,
    timeout: int,
) -> tuple[bool, bool, float, str]:
    """Grade one patch with the official SWE-bench Docker harness."""
    if not patch.strip():
        return False, False, 0.0, "empty patch"

    instance_id = str(instance["instance_id"])
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Freeze grading to the exact task row already loaded by Forge Bench.
    # SWE-bench 4.1 accepts a local JSON dataset path, avoiding a second
    # mutable Hugging Face fetch during evaluation.
    dataset_file = eval_dir / "dataset.json"
    dataset_file.write_text(
        json.dumps([instance], indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    prediction = eval_dir / "prediction.jsonl"
    model_name = "forge-bench"
    prediction.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
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
        str(dataset_file),
        "--predictions_path",
        str(prediction),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--instance_ids",
        instance_id,
        # Forge Bench pins swebench 4.1.0. Keeping the instance image lets the
        # remaining treatment arms reuse it instead of rebuilding the same task.
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

    # SWE-bench 4.1 writes <model>.<run_id>.json in CWD. Newer releases
    # write logs/evaluation/<run_id>/results.json.
    report = load_swebench_report(eval_dir, model_name, run_id)

    if report is None:
        detail = f"evaluator exit {proc.returncode}"
        if proc.stderr.strip():
            detail += ": " + proc.stderr.strip().splitlines()[-1][:300]
        return False, False, elapsed, "evaluation report not found; " + detail

    error_ids = set(report.get("error_ids") or [])
    if instance_id in error_ids:
        return False, False, elapsed, "SWE-bench evaluator reported an infrastructure/test error"

    resolved_ids = set(report.get("resolved_ids") or [])
    unresolved_ids = set(report.get("unresolved_ids") or [])
    completed_ids = set(report.get("completed_ids") or [])

    completed = (
        instance_id in completed_ids
        or instance_id in resolved_ids
        or instance_id in unresolved_ids
    )
    if not completed:
        return False, False, elapsed, "SWE-bench evaluation did not complete for instance"

    return True, instance_id in resolved_ids, elapsed, ""


def _snapshot_state_db(profile: Path, run_dir: Path) -> bool:
    """Create a consistent post-run copy of Hermes state.db for audit/replay."""
    source = profile / "state.db"
    if not source.is_file():
        return False
    destination = run_dir / "hermes-state.db"
    try:
        src = sqlite3.connect("file:" + str(source) + "?mode=ro", uri=True)
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        return destination.is_file()
    except Exception as exc:
        with (run_dir / "trace-export-error.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"state.db snapshot failed: {exc}\n")
        return False


def _session_export_args(
    profile: Path,
    filename: str,
    *,
    runtime: str,
) -> tuple[Path, str]:
    host_path = profile / filename
    command_path = (
        f"/opt/data/{filename}"
        if runtime == "docker"
        else str(host_path)
    )
    return host_path, command_path


def _append_trace_error(run_dir: Path, message: str) -> None:
    with (run_dir / "trace-export-error.txt").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def export_session_artifacts(
    hermes: str,
    profile: Path,
    run_dir: Path,
    session_id: str,
    *,
    runtime: str,
    image: str | None,
) -> bool:
    """Export the native Hermes transcript + trace before the ephemeral profile is removed."""
    if not session_id:
        _append_trace_error(run_dir, "session export skipped: missing session_id")
        _snapshot_state_db(profile, run_dir)
        return False

    exported = True
    exports = (
        ("benchmark-session.jsonl", "jsonl", False),
        ("benchmark-session.trace.jsonl", "trace", True),
    )
    for temp_name, fmt, no_redact in exports:
        host_path, command_path = _session_export_args(
            profile,
            temp_name,
            runtime=runtime,
        )
        args = [
            "sessions",
            "export",
            command_path,
            "--format",
            fmt,
            "--session-id",
            session_id,
        ]
        if no_redact:
            args.append("--no-redact")
        try:
            proc = hermes_admin(
                hermes,
                profile,
                args,
                runtime=runtime,
                image=image,
                timeout=120,
            )
        except Exception as exc:
            exported = False
            _append_trace_error(
                run_dir,
                f"Hermes session {fmt} export raised: {exc}",
            )
            continue
        if proc.returncode or not host_path.is_file():
            exported = False
            detail = (proc.stderr or proc.stdout or "").strip()
            _append_trace_error(
                run_dir,
                f"Hermes session {fmt} export failed (exit {proc.returncode}): {detail[:1000]}",
            )
            continue

        target = (
            run_dir / "hermes-session.jsonl"
            if fmt == "jsonl"
            else run_dir / "hermes-session.trace.jsonl"
        )
        shutil.copy2(host_path, target)
        if fmt == "jsonl":
            try:
                rows = [
                    json.loads(line)
                    for line in host_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) == 1:
                    (run_dir / "hermes-session.json").write_text(
                        json.dumps(rows[0], indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                _append_trace_error(run_dir, f"pretty session export failed: {exc}")
        try:
            host_path.unlink()
        except OSError:
            pass

    _snapshot_state_db(profile, run_dir)

    # Preserve non-secret runtime configuration and installed-extension evidence.
    config = profile / "config.yaml"
    if config.is_file():
        shutil.copy2(config, run_dir / "hermes-config.yaml")
    manifest = {
        "trace_plugin_enabled": trace_plugin_enabled(profile),
        "plugins": sorted(
            path.name
            for path in (profile / "plugins").iterdir()
            if path.is_dir()
        ) if (profile / "plugins").is_dir() else [],
        "skill_files": sorted(
            str(path.relative_to(profile))
            for path in profile.rglob("SKILL.md")
        ),
        "session_id": session_id,
    }
    (run_dir / "hermes-profile-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return exported


def write_treatment_evidence(
    profile: Path,
    run_dir: Path,
    arm: str,
    timing: dict[str, Any],
) -> None:
    """Record assigned treatment plus observable installation/uptake evidence."""
    caveman_expected, ponytail_expected, lean_expected = ARMS.get(
        arm, (False, False, False)
    )
    try:
        cfg = yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    enabled = [
        str(value)
        for value in ((cfg.get("plugins") or {}).get("enabled") or [])
    ]
    plugin_dirs = (
        sorted(path.name for path in (profile / "plugins").iterdir() if path.is_dir())
        if (profile / "plugins").is_dir()
        else []
    )
    skill_files = sorted(str(path.relative_to(profile)) for path in profile.rglob("SKILL.md"))
    evidence = {
        "arm": arm,
        "assigned": {
            "caveman": caveman_expected,
            "ponytail": ponytail_expected,
            "lean_tools": lean_expected,
        },
        "installed": {
            "caveman_skill": any("caveman" in value.lower() for value in skill_files),
            "ponytail_plugin": any("ponytail" in value.lower() for value in [*enabled, *plugin_dirs]),
            "forge_bench_trace_plugin": trace_plugin_enabled(profile),
        },
        "plugins_enabled": enabled,
        "plugin_directories": plugin_dirs,
        "skill_files": skill_files,
        "observed": {
            "skill_lifecycle_event_count": int(timing.get("skill_lifecycle_event_count") or 0),
            "skill_names": timing.get("skill_names_observed") or [],
            "caveman_skill_lifecycle_seen": bool(timing.get("caveman_skill_observed")),
        },
        "interpretation": (
            "Treatment assignment remains the causal experimental variable. "
            "Observed skill lifecycle events are audit/compliance evidence and "
            "must not replace randomized assignment in treatment-effect estimates."
        ),
    }
    (run_dir / "treatment-evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def finalize_trace_artifacts(
    hermes: str,
    profile: Path,
    run_dir: Path,
    wall_seconds: float,
    session_id: str,
    *,
    runtime: str,
    image: str | None,
) -> tuple[dict[str, Any], bool]:
    """Best-effort trace finalization that can never invalidate a paid run."""
    try:
        timing = finalize_event_artifacts(profile, run_dir, wall_seconds)
    except Exception as exc:
        timing = {}
        _append_trace_error(run_dir, f"event finalization failed: {exc}")

    resolved_session_id = session_id
    if not resolved_session_id:
        candidates = timing.get("session_ids") if isinstance(timing, dict) else None
        if isinstance(candidates, list) and candidates:
            resolved_session_id = str(candidates[0])

    try:
        exported = export_session_artifacts(
            hermes,
            profile,
            run_dir,
            resolved_session_id,
            runtime=runtime,
            image=image,
        )
    except Exception as exc:
        exported = False
        _append_trace_error(run_dir, f"session artifact export failed: {exc}")
        _snapshot_state_db(profile, run_dir)
    return timing, exported


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
    hermes_runtime: str = "local",
    hermes_image: str | None = None,
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

    _caveman, ponytail, lean = ARMS.get(arm, (False, False, False))
    hermes_args = [
        "-z",
        prompt,
        "--ignore-rules",
        "--toolsets",
        LEAN_TOOLSETS if lean else DEFAULT_TOOLSETS,
        "--provider",
        "openrouter",
        "--model",
        PINNED_MODEL,
        "--reasoning",
        PINNED_REASONING,
    ]

    if hermes_runtime == "docker":
        if not hermes_image:
            raise RuntimeError("Docker Hermes image was not resolved")
        profile_usage = profile / "benchmark-usage.json"
        hermes_args.extend(["--usage-file", "/opt/data/benchmark-usage.json"])
        extra_env = {"PONYTAIL_DEFAULT_MODE": "full"} if ponytail else {}
        argv = docker_hermes_argv(
            hermes_image,
            profile,
            hermes_args,
            workspace=workspace,
            extra_env=extra_env,
        )
        run_cwd = None
        run_env = None
        usage_file = profile_usage
    else:
        usage_file = run_dir / "usage.json"
        hermes_args.extend(["--usage-file", str(usage_file)])
        argv = [hermes, *hermes_args]
        run_cwd = workspace
        run_env = profile_env(profile)
        if ponytail:
            run_env["PONYTAIL_DEFAULT_MODE"] = "full"

    started = time.perf_counter()
    try:
        proc = sh(argv, cwd=run_cwd, env=run_env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        (run_dir / "stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
        (run_dir / "stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
        if usage_file.exists() and usage_file != run_dir / "usage.json":
            shutil.copy2(usage_file, run_dir / "usage.json")
        try:
            timeout_usage = (
                json.loads(usage_file.read_text(encoding="utf-8"))
                if usage_file.exists()
                else {}
            )
        except Exception:
            timeout_usage = {}
        session_id = str(timeout_usage.get("session_id") or "")
        timing, trace_exported = finalize_trace_artifacts(
            hermes,
            profile,
            run_dir,
            wall,
            session_id,
            runtime=hermes_runtime,
            image=hermes_image,
        )
        write_treatment_evidence(profile, run_dir, arm, timing)
        result = Result(
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
            model=str(timeout_usage.get("model") or PINNED_MODEL),
            api_provider=str(timeout_usage.get("provider") or "openrouter"),
            upstream_provider=PINNED_OPENROUTER_UPSTREAM,
            session_id=session_id or (
                str(timing.get("session_ids", [""])[0])
                if timing.get("session_ids")
                else ""
            ),
            input_tokens=as_int(timeout_usage.get("input_tokens")),
            output_tokens=as_int(timeout_usage.get("output_tokens")),
            reasoning_tokens=as_int(timeout_usage.get("reasoning_tokens")),
            cache_read_tokens=as_int(timeout_usage.get("cache_read_tokens")),
            cache_write_tokens=as_int(timeout_usage.get("cache_write_tokens")),
            total_tokens=as_int(timeout_usage.get("total_tokens")),
            api_calls=as_int(timeout_usage.get("api_calls")),
            estimated_cost_usd=as_float(timeout_usage.get("estimated_cost_usd")),
            actual_cost_usd=as_float(timeout_usage.get("actual_cost_usd")),
            cost_usd=(
                as_float(timeout_usage.get("actual_cost_usd"))
                if as_float(timeout_usage.get("actual_cost_usd")) is not None
                else as_float(timeout_usage.get("estimated_cost_usd"))
            ),
            cost_source="partial_timeout_usage" if timeout_usage else "unavailable",
            tool_calls=None,
            files_changed=0,
            diff_lines=0,
            run_dir=str(run_dir),
            error="Hermes timeout",
            api_wait_seconds=as_float(timing.get("api_wait_seconds")),
            tool_execution_seconds=as_float(timing.get("tool_execution_seconds")),
            terminal_execution_seconds=as_float(timing.get("terminal_execution_seconds")),
            unattributed_wall_seconds=as_float(timing.get("unattributed_wall_seconds")),
            api_duration_mean_seconds=as_float(timing.get("api_duration_mean_seconds")),
            api_duration_p95_seconds=as_float(timing.get("api_duration_p95_seconds")),
            ttft_mean_seconds=as_float(timing.get("ttft_mean_seconds")),
            timing_event_count=as_int(timing.get("event_count")),
            skill_lifecycle_event_count=as_int(timing.get("skill_lifecycle_event_count")),
            trace_exported=trace_exported,
        )
        (run_dir / "result.json").write_text(
            json.dumps(result.__dict__, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    wall = time.perf_counter() - started
    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    if usage_file.exists() and usage_file != run_dir / "usage.json":
        shutil.copy2(usage_file, run_dir / "usage.json")

    files_changed, diff_lines, patch = diff_stats(
        workspace,
        str(instance["base_commit"]),
    )
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
            instance,
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

    timing, trace_exported = finalize_trace_artifacts(
        hermes,
        profile,
        run_dir,
        wall,
        session_id,
        runtime=hermes_runtime,
        image=hermes_image,
    )
    write_treatment_evidence(profile, run_dir, arm, timing)

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
    reasoning_tokens = as_int(usage.get("reasoning_tokens"))
    reasoning_ok = PINNED_REASONING != "none" or reasoning_tokens == 0
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
        and reasoning_ok
        and eval_ok
    )

    errors: list[str] = []
    if not completed:
        errors.append("Hermes incomplete")
    if model != PINNED_MODEL:
        errors.append("wrong model: " + model)
    if provider.lower() != "openrouter":
        errors.append("wrong API provider: " + provider)
    if not reasoning_ok:
        errors.append(f"reasoning was requested off but {reasoning_tokens} reasoning tokens were reported")
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
        reasoning_tokens=reasoning_tokens,
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
        api_wait_seconds=as_float(timing.get("api_wait_seconds")),
        tool_execution_seconds=as_float(timing.get("tool_execution_seconds")),
        terminal_execution_seconds=as_float(timing.get("terminal_execution_seconds")),
        unattributed_wall_seconds=as_float(timing.get("unattributed_wall_seconds")),
        api_duration_mean_seconds=as_float(timing.get("api_duration_mean_seconds")),
        api_duration_p95_seconds=as_float(timing.get("api_duration_p95_seconds")),
        ttft_mean_seconds=as_float(timing.get("ttft_mean_seconds")),
        timing_event_count=as_int(timing.get("event_count")),
        skill_lifecycle_event_count=as_int(timing.get("skill_lifecycle_event_count")),
        trace_exported=trace_exported,
    )
    (run_dir / "result.json").write_text(
        json.dumps(result.__dict__, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
