"""Physical robot benchmark orchestration for Forge Bench.

This runner composes two lower-level contracts:
- OpenShell owns sandbox isolation and provider attachment.
- The SO-ARM101 broker owns bounded robot/camera actions.

It intentionally does not import or execute the unrestricted robot SDK.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import yaml

from .openshell_runtime import (
    NetworkEndpoint,
    OpenShellPolicy,
    OpenShellRuntime,
    OpenShellSandboxSpec,
    RestRule,
)


DEFAULT_BROKER_HOST = "0.0.0.0"
DEFAULT_BROKER_CLIENT_HOST = "host.openshell.internal"
DEFAULT_BROKER_PORT = 8765
DEFAULT_SANDBOX_IMAGE = "agenticforge/forge-bench-hermes-openshell:local"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-20260910"
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4.1-flash-20260910"
DEFAULT_TOOLSETS = "hermes-cli"
BROKER_ROUTES = (
    RestRule("GET", "/v1/health"),
    RestRule("GET", "/v1/capabilities"),
    RestRule("GET", "/v1/state"),
    RestRule("POST", "/v1/capture"),
    RestRule("POST", "/v1/go-pose"),
    RestRule("POST", "/v1/joint"),
    RestRule("POST", "/v1/jog"),
    RestRule("POST", "/v1/gripper"),
    RestRule("POST", "/v1/sleep"),
    RestRule("POST", "/v1/stop"),
)


@dataclass(frozen=True)
class VisionJudgment:
    clear: bool
    object_inside_container: bool
    reason: str
    model: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class VisionJudge(Protocol):
    def judge(self, image_path: Path) -> VisionJudgment: ...


@dataclass(frozen=True)
class RobotBenchmarkScore:
    agent_status: str
    fresh_overhead_evidence: bool
    final_evidence_sha256: str | None
    latest_overhead_sha256: str | None
    semantic_judgment: VisionJudgment | None
    success: bool | None
    reason: str
    valid: bool = True

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.semantic_judgment is not None:
            payload["semantic_judgment"] = self.semantic_judgment.as_dict()
        return payload


@dataclass(frozen=True)
class RobotBenchmarkConfig:
    output: Path
    image: str
    provider: str
    model: str
    robotctl_source: Path
    task_source: Path
    skill_source: Path
    broker_command: str = "soarm101-broker"
    broker_port: int = DEFAULT_BROKER_PORT
    sandbox_name: str | None = None
    hermes_command: str = "hermes"
    hermes_provider: str = "openrouter"
    toolsets: str = DEFAULT_TOOLSETS
    max_turns: int = 100
    reasoning: str = "none"
    timeout: int = 1800
    robot_network_binaries: tuple[str, ...] = (
        "/opt/hermes/.venv/bin/python*",
        "/opt/hermes/tools/**/python*",
    )

    def validated(self) -> "RobotBenchmarkConfig":
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(
                f"robot benchmark output directory must be new or empty: {self.output}"
            )
        if not self.robotctl_source.is_file():
            raise FileNotFoundError(f"robotctl source not found: {self.robotctl_source}")
        if not self.task_source.is_file():
            raise FileNotFoundError(f"task file not found: {self.task_source}")
        if not self.skill_source.is_file():
            raise FileNotFoundError(f"skill file not found: {self.skill_source}")
        if not 1 <= int(self.broker_port) <= 65535:
            raise ValueError("broker port must be within 1..65535")
        if not self.robot_network_binaries:
            raise ValueError("at least one robot-network binary path is required")
        return self


class OpenRouterVisionJudge:
    """Independent host-side final-image judge.

    The judge sees only the final overhead image and completion definition. It
    does not receive the agent trace, claimed reasoning, or robot actions.
    """

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        self.model = model
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.opener = opener
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for automatic visual judging")

    def judge(self, image_path: Path) -> VisionJudgment:
        image = image_path.read_bytes()
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        image_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        schema = {
            "type": "object",
            "properties": {
                "clear": {"type": "boolean"},
                "object_inside_container": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["clear", "object_inside_container", "reason"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 180,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_completion_judgment",
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an independent visual evaluator for a robot manipulation "
                        "benchmark. Judge only the supplied final overhead image. A success "
                        "requires the visible target object to be clearly contained within "
                        "the target container's interior footprint/opening. Mark false if "
                        "the object is merely touching the rim, overlaps outside the edge, "
                        "is beside the container, or the view is too ambiguous/occluded."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Is the target object clearly inside the target container? "
                                "Return the schema exactly."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=180.0) as response:  # type: ignore[attr-defined]
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"visual judge HTTP {exc.code}: {detail}") from exc
        outer = json.loads(raw)
        content = outer["choices"][0]["message"]["content"]
        result = json.loads(content)
        return VisionJudgment(
            clear=bool(result["clear"]),
            object_inside_container=bool(result["object_inside_container"]),
            reason=str(result["reason"]),
            model=self.model,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def latest_overhead_capture_sha(events_path: Path) -> str | None:
    latest: tuple[float, str] | None = None
    for row in _read_jsonl(events_path):
        if row.get("action") != "capture_evidence" or not row.get("ok"):
            continue
        result = row.get("result")
        if not isinstance(result, dict) or result.get("name") != "overhead":
            continue
        sha = str(result.get("sha256") or "").strip()
        if not sha:
            continue
        timestamp = float(row.get("timestamp") or 0.0)
        if latest is None or timestamp >= latest[0]:
            latest = (timestamp, sha)
    return None if latest is None else latest[1]


def _resolve_evidence_path(run_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        # Sandbox paths are not host paths. Preserve the suffix under observations.
        parts = list(candidate.parts)
        if "observations" in parts:
            index = parts.index("observations")
            candidate = Path(*parts[index:])
        else:
            candidate = Path(candidate.name)
    if candidate.parts and candidate.parts[0] == "observations":
        return run_dir / candidate
    return run_dir / "observations" / candidate


def score_robot_completion(
    *,
    run_dir: Path,
    result_path: Path,
    broker_events_path: Path,
    judge: VisionJudge | None,
) -> RobotBenchmarkScore:
    if not result_path.is_file():
        return RobotBenchmarkScore(
            agent_status="missing",
            fresh_overhead_evidence=False,
            final_evidence_sha256=None,
            latest_overhead_sha256=latest_overhead_capture_sha(broker_events_path),
            semantic_judgment=None,
            success=False if judge is not None else None,
            reason="agent did not produce task-result.json",
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task result root must be a JSON object")
    status = str(payload.get("task_status") or "")
    latest_sha = latest_overhead_capture_sha(broker_events_path)
    if status != "finished":
        return RobotBenchmarkScore(
            agent_status=status or "not_finished",
            fresh_overhead_evidence=False,
            final_evidence_sha256=None,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=False,
            reason=str(payload.get("reason") or "agent did not claim completion"),
        )

    if payload.get("evidence_camera") != "overhead":
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=False,
            final_evidence_sha256=None,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=False,
            reason="finished result did not cite the overhead camera",
        )

    raw_path = str(payload.get("evidence_path") or "").strip()
    if not raw_path:
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=False,
            final_evidence_sha256=None,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=False,
            reason="finished result did not cite an evidence image path",
        )
    evidence_path = _resolve_evidence_path(run_dir, raw_path)
    if not evidence_path.is_file():
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=False,
            final_evidence_sha256=None,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=False,
            reason=f"cited evidence image was not downloaded: {evidence_path}",
        )

    evidence_sha = _sha256(evidence_path)
    fresh = bool(latest_sha and evidence_sha == latest_sha)
    if not fresh:
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=False,
            final_evidence_sha256=evidence_sha,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=False,
            reason="cited image is not the latest trusted overhead capture from the broker",
        )

    if judge is None:
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=True,
            final_evidence_sha256=evidence_sha,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=None,
            reason="fresh trusted evidence verified; semantic visual judgment not run",
        )

    try:
        judgment = judge.judge(evidence_path)
    except Exception as exc:
        return RobotBenchmarkScore(
            agent_status=status,
            fresh_overhead_evidence=True,
            final_evidence_sha256=evidence_sha,
            latest_overhead_sha256=latest_sha,
            semantic_judgment=None,
            success=None,
            reason=f"independent visual judge failed: {exc}",
            valid=False,
        )
    success = bool(judgment.clear and judgment.object_inside_container)
    return RobotBenchmarkScore(
        agent_status=status,
        fresh_overhead_evidence=True,
        final_evidence_sha256=evidence_sha,
        latest_overhead_sha256=latest_sha,
        semantic_judgment=judgment,
        success=success,
        reason=(
            "independent visual judge confirmed completion"
            if success
            else f"independent visual judge did not confirm completion: {judgment.reason}"
        ),
    )


def robot_broker_policy(
    *,
    port: int,
    python_binaries: Sequence[str],
) -> OpenShellPolicy:
    return OpenShellPolicy(
        read_only=("/opt/hermes",),
        read_write=("/tmp",),
        user=10000,
        group=10000,
        endpoints=(
            NetworkEndpoint(
                name="soarm101_robot_broker",
                host=DEFAULT_BROKER_CLIENT_HOST,
                port=int(port),
                binaries=tuple(str(path) for path in python_binaries),
                protocol="rest",
                enforcement="enforce",
                rules=BROKER_ROUTES,
            ),
        )
    )


class BrokerProcess:
    def __init__(
        self,
        *,
        command: str,
        port: int,
        token: str,
        events_path: Path,
    ) -> None:
        self.command = command
        self.port = int(port)
        self.token = token
        self.events_path = events_path
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env["SOARM101_BROKER_TOKEN"] = self.token
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                self.command,
                "--host",
                DEFAULT_BROKER_HOST,
                "--port",
                str(self.port),
                "--events",
                str(self.events_path),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                out, err = self.process.communicate(timeout=1)
                raise RuntimeError(
                    "robot broker exited during startup\n"
                    f"stdout:\n{out}\nstderr:\n{err}"
                )
            try:
                self.request("GET", "/v1/health")
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("robot broker did not become ready within 15 seconds")

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10.0) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict) or not body.get("ok"):
            raise RuntimeError(f"robot broker request failed: {body}")
        return body

    def require_armed(self) -> dict[str, object]:
        payload = self.request("GET", "/v1/capabilities")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("broker capabilities response is missing result")
        authority = result.get("authority")
        if not isinstance(authority, dict) or not authority.get("armed"):
            raise RuntimeError(
                "robot agent authority is not active; a human must run "
                "'soarm101 agent arm' before the benchmark"
            )
        cameras = result.get("cameras")
        configured = {
            str(name)
            for name in cameras
            if isinstance(name, str)
        } if isinstance(cameras, list) else set()
        if "overhead" not in configured:
            raise RuntimeError(
                "the physical benchmark requires a configured 'overhead' camera "
                "for authoritative completion evidence"
            )
        return result

    def stop(self) -> tuple[str, str]:
        if self.process is None:
            return "", ""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        stdout, stderr = self.process.communicate(timeout=1)
        return stdout or "", stderr or ""


def _copy_inputs(config: RobotBenchmarkConfig) -> dict[str, str]:
    config.output.mkdir(parents=True, exist_ok=True)
    inputs = config.output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    destinations = {
        "robotctl": inputs / "robotctl.py",
        "task": inputs / "TASK.md",
        "skill": inputs / "SKILL.md",
    }
    shutil.copy2(config.robotctl_source, destinations["robotctl"])
    shutil.copy2(config.task_source, destinations["task"])
    shutil.copy2(config.skill_source, destinations["skill"])

    hermes_home = inputs / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    hermes_config = {
        "_config_version": 45,
        "plugins": {"enabled": [], "disabled": []},
        "agent": {
            "max_turns": int(config.max_turns),
            "budget_warning_ratio": None,
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
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(hermes_config, sort_keys=False),
        encoding="utf-8",
    )
    (hermes_home / ".no-bundled-skills").write_text(
        "Forge Bench isolated physical robot profile\n",
        encoding="utf-8",
    )

    hashes = {
        key: _sha256(path)
        for key, path in destinations.items()
    }
    hashes["hermes_config"] = _sha256(hermes_home / "config.yaml")
    hashes["hermes_no_bundled_skills"] = _sha256(
        hermes_home / ".no-bundled-skills"
    )
    return hashes


def _hermes_prompt() -> str:
    return (
        "Read TASK.md and SKILL.md completely before acting. Follow SKILL.md as the "
        "robot/camera operating contract. Use vision_analyze on fresh captured image "
        "files before making visual claims. Work only through robotctl.py for robot and "
        "camera actions. Before finishing, write the exact completion JSON object required "
        "by SKILL.md to task-result.json in the current working directory. Do not claim "
        "success without fresh overhead evidence."
    )


def run_robot_benchmark(
    config: RobotBenchmarkConfig,
    *,
    runtime: OpenShellRuntime | None = None,
    judge: VisionJudge | None = None,
) -> RobotBenchmarkScore:
    config = config.validated()
    runtime = runtime or OpenShellRuntime()
    if not runtime.available():
        raise RuntimeError("OpenShell executable is not available")

    config.output.mkdir(parents=True, exist_ok=True)
    input_hashes = _copy_inputs(config)
    broker_events = config.output / "broker-events.jsonl"
    token = secrets.token_urlsafe(32)
    sandbox_name = config.sandbox_name or f"forge-bench-robot-{secrets.token_hex(4)}"
    broker = BrokerProcess(
        command=config.broker_command,
        port=config.broker_port,
        token=token,
        events_path=broker_events,
    )

    sandbox_created = False
    started_at = time.time()
    capabilities: dict[str, object] | None = None
    run_error: str | None = None
    hermes_exit_code: int | None = None
    try:
        broker.start()
        capabilities = broker.require_armed()

        policy = robot_broker_policy(
            port=config.broker_port,
            python_binaries=config.robot_network_binaries,
        )
        spec = OpenShellSandboxSpec(
            name=sandbox_name,
            image=config.image,
            policy=policy,
            providers=(config.provider,),
            labels={"forge-bench": "robot-manipulation"},
        )
        runtime.create(
            spec,
            policy_path=config.output / "openshell-policy.yaml",
        )
        sandbox_created = True

        prepare = runtime.exec(
            sandbox_name,
            [
                "/bin/mkdir",
                "-p",
                "/sandbox/.hermes",
                "/sandbox/.home",
                "/sandbox/.xdg-config",
            ],
            workdir="/sandbox",
            timeout=30,
        )
        if prepare.returncode != 0:
            raise RuntimeError(
                "could not prepare sandbox Hermes home: "
                + (prepare.stderr or prepare.stdout or f"exit {prepare.returncode}")
            )

        inputs = config.output / "inputs"
        for name in ("robotctl.py", "TASK.md", "SKILL.md"):
            runtime.upload(sandbox_name, inputs / name)
        runtime.upload(
            sandbox_name,
            inputs / "hermes-home" / "config.yaml",
            "/sandbox/.hermes/config.yaml",
        )
        runtime.upload(
            sandbox_name,
            inputs / "hermes-home" / ".no-bundled-skills",
            "/sandbox/.hermes/.no-bundled-skills",
        )

        runtime.effective_policy(
            sandbox_name,
            config.output / "openshell-effective-policy.yaml",
        )

        proc = runtime.exec(
            sandbox_name,
            [
                config.hermes_command,
                "chat",
                "--oneshot",
                "-q",
                _hermes_prompt(),
                "--format",
                "stream-json",
                "--max-turns",
                str(config.max_turns),
                "--ignore-rules",
                "--toolsets",
                config.toolsets,
                "--provider",
                config.hermes_provider,
                "--model",
                config.model,
                "--reasoning",
                config.reasoning,
            ],
            env={
                "SOARM101_BROKER_URL": (
                    f"http://{DEFAULT_BROKER_CLIENT_HOST}:{config.broker_port}"
                ),
                "SOARM101_BROKER_TOKEN": token,
                "HOME": "/sandbox/.home",
                "HERMES_HOME": "/sandbox/.hermes",
                "HERMES_WRITE_SAFE_ROOT": "/sandbox",
                "XDG_CONFIG_HOME": "/sandbox/.xdg-config",
                "HERMES_ENABLE_PROJECT_PLUGINS": "0",
            },
            timeout=config.timeout,
        )
        hermes_exit_code = int(proc.returncode)
        (config.output / "hermes-stdout.jsonl").write_text(
            proc.stdout or "",
            encoding="utf-8",
        )
        (config.output / "hermes-stderr.txt").write_text(
            proc.stderr or "",
            encoding="utf-8",
        )

        for source, destination in (
            ("task-result.json", config.output / "task-result.json"),
            ("observations", config.output),
        ):
            try:
                runtime.download(sandbox_name, source, destination)
            except Exception as exc:
                with (config.output / "download-errors.txt").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(f"{source}: {exc}\n")

        runtime.logs(
            sandbox_name,
            config.output / "openshell-logs.txt",
            level="info",
            since="2h",
        )
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        (config.output / "run-error.txt").write_text(
            run_error + "\n",
            encoding="utf-8",
        )
    finally:
        if sandbox_created:
            try:
                runtime.effective_policy(
                    sandbox_name,
                    config.output / "openshell-effective-policy-final.yaml",
                )
            except Exception:
                pass
            try:
                runtime.delete(sandbox_name)
            except Exception:
                pass
        broker_stdout, broker_stderr = broker.stop()
        (config.output / "broker-stdout.txt").write_text(
            broker_stdout,
            encoding="utf-8",
        )
        (config.output / "broker-stderr.txt").write_text(
            broker_stderr,
            encoding="utf-8",
        )

    metadata = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": time.time(),
        "model": config.model,
        "provider": config.provider,
        "hermes_provider": config.hermes_provider,
        "image": config.image,
        "sandbox_name": sandbox_name,
        "broker_port": config.broker_port,
        "input_sha256": input_hashes,
        "capabilities": capabilities,
        "hermes_exit_code": hermes_exit_code,
        "run_error": run_error,
    }
    (config.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    score = score_robot_completion(
        run_dir=config.output,
        result_path=config.output / "task-result.json",
        broker_events_path=broker_events,
        judge=judge,
    )
    if run_error is not None:
        score = replace(
            score,
            valid=False,
            success=False,
            reason=f"benchmark orchestration failed: {run_error}; {score.reason}",
        )
    (config.output / "score.json").write_text(
        json.dumps(score.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated SO-ARM101 manipulation benchmark."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_SANDBOX_IMAGE)
    parser.add_argument("--provider", default="hermes-openrouter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--robotctl", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("robot_tasks/object-to-container/TASK.md"),
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=Path("robot_tasks/skills/soarm101-robot-camera/SKILL.md"),
    )
    parser.add_argument("--broker-command", default="soarm101-broker")
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--sandbox-name")
    parser.add_argument("--hermes-command", default="hermes")
    parser.add_argument("--hermes-provider", default="openrouter")
    parser.add_argument("--toolsets", default=DEFAULT_TOOLSETS)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--robot-network-binary",
        action="append",
        dest="robot_network_binaries",
        help=(
            "Real Python interpreter path inside the sandbox allowed to call the robot "
            "broker; repeat as needed. Defaults cover the official Hermes image's Python runtime paths."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Verify trusted final evidence but skip independent visual semantic judging.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RobotBenchmarkConfig(
        output=args.output.expanduser().resolve(),
        image=args.image,
        provider=args.provider,
        model=args.model,
        robotctl_source=args.robotctl.expanduser().resolve(),
        task_source=args.task.expanduser().resolve(),
        skill_source=args.skill.expanduser().resolve(),
        broker_command=args.broker_command,
        broker_port=args.broker_port,
        sandbox_name=args.sandbox_name,
        hermes_command=args.hermes_command,
        hermes_provider=args.hermes_provider,
        toolsets=args.toolsets,
        max_turns=args.max_turns,
        reasoning=args.reasoning,
        timeout=args.timeout,
        robot_network_binaries=tuple(
            args.robot_network_binaries
            or RobotBenchmarkConfig.__dataclass_fields__["robot_network_binaries"].default
        ),
    )
    judge: VisionJudge | None = None
    if not args.no_judge:
        judge = OpenRouterVisionJudge(model=args.judge_model)
    score = run_robot_benchmark(config, judge=judge)
    print(json.dumps(score.as_dict(), indent=2))
    if not score.valid:
        return 3
    return 0 if score.success is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
