"""Generic OpenShell sandbox orchestration for Forge Bench.

This module is deliberately task-agnostic. It knows how to build a restrictive
OpenShell policy and manage a sandbox lifecycle, but it does not know about
SWE-bench, robots, or any benchmark-specific scoring.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RestRule:
    method: str
    path: str

    def as_policy(self) -> dict[str, object]:
        return {"allow": {"method": self.method.upper(), "path": self.path}}


@dataclass(frozen=True)
class NetworkEndpoint:
    name: str
    host: str
    port: int
    binaries: tuple[str, ...]
    protocol: str = "rest"
    enforcement: str = "enforce"
    rules: tuple[RestRule, ...] = ()

    def validated(self) -> "NetworkEndpoint":
        if not self.name.strip():
            raise ValueError("OpenShell network endpoint name cannot be empty")
        if not self.host.strip():
            raise ValueError("OpenShell network endpoint host cannot be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("OpenShell network endpoint port must be within 1..65535")
        if not self.binaries or any(not item.startswith("/") for item in self.binaries):
            raise ValueError("OpenShell network binaries must be non-empty absolute paths")
        if self.protocol not in {"rest", "websocket", "tcp"}:
            raise ValueError(
                "this Forge Bench policy helper supports only rest, websocket, or tcp; "
                "GraphQL/MCP/JSON-RPC require protocol-specific rule schemas"
            )
        if self.enforcement not in {"enforce", "audit"}:
            raise ValueError("OpenShell enforcement must be 'enforce' or 'audit'")
        if self.protocol == "tcp" and self.rules:
            raise ValueError("TCP endpoints cannot use request rules")
        if self.protocol in {"rest", "websocket"} and not self.rules:
            raise ValueError(f"{self.protocol} endpoints require explicit request rules")
        return self

    def as_policy(self) -> dict[str, object]:
        endpoint = self.validated()
        target: dict[str, object] = {
            "host": endpoint.host,
            "port": int(endpoint.port),
            "protocol": endpoint.protocol,
        }
        if endpoint.protocol != "tcp":
            target["enforcement"] = endpoint.enforcement
            target["rules"] = [rule.as_policy() for rule in endpoint.rules]
        return {
            "name": endpoint.name,
            "endpoints": [target],
            "binaries": [{"path": path} for path in endpoint.binaries],
        }


@dataclass(frozen=True)
class OpenShellPolicy:
    """Static filesystem/process policy plus explicit outbound network grants."""

    include_workdir: bool = True
    read_only: tuple[str, ...] = ()
    read_write: tuple[str, ...] = ("/tmp",)
    hard_require_landlock: bool = True
    user: int | str = "sandbox"
    group: int | str = "sandbox"
    endpoints: tuple[NetworkEndpoint, ...] = ()

    def as_dict(self) -> dict[str, object]:
        policy: dict[str, object] = {
            "version": 1,
            "filesystem_policy": {
                "include_workdir": bool(self.include_workdir),
                "read_only": list(self.read_only),
                "read_write": list(self.read_write),
            },
            "landlock": {
                "compatibility": (
                    "hard_requirement" if self.hard_require_landlock else "best_effort"
                )
            },
            "process": {
                # OpenShell's policy schema represents both named identities and
                # numeric UID/GID values as strings in YAML.
                "run_as_user": str(self.user),
                "run_as_group": str(self.group),
            },
        }
        if self.endpoints:
            policy["network_policies"] = {
                endpoint.name: endpoint.as_policy()
                for endpoint in self.endpoints
            }
        return policy

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.as_dict(), sort_keys=False),
            encoding="utf-8",
        )
        return path


@dataclass(frozen=True)
class OpenShellSandboxSpec:
    name: str
    image: str
    policy: OpenShellPolicy
    cpu: str | None = None
    memory: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    providers: tuple[str, ...] = ()

    def validated(self) -> "OpenShellSandboxSpec":
        if not self.name.strip():
            raise ValueError("OpenShell sandbox name cannot be empty")
        if not self.image.strip():
            raise ValueError("OpenShell sandbox image cannot be empty")
        return self


@dataclass
class OpenShellRuntime:
    """Small CLI-backed OpenShell lifecycle adapter.

    Forge Bench intentionally shells out to the official OpenShell CLI here.
    This keeps OpenShell optional and avoids adding its Python SDK as a hard
    dependency to existing benchmark environments.
    """

    executable: str = "openshell"
    runner: CommandRunner = subprocess.run

    def available(self) -> bool:
        return bool(shutil.which(self.executable) or Path(self.executable).exists())

    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 120,
        check: bool = True,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process_env = None if env is None else {**os.environ, **dict(env)}
        completed = self.runner(
            [self.executable, *argv],
            cwd=cwd,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if check and completed.returncode:
            raise RuntimeError(
                "OpenShell command failed: "
                + " ".join([self.executable, *argv])
                + "\nstdout:\n"
                + completed.stdout
                + "\nstderr:\n"
                + completed.stderr
            )
        return completed

    def create(
        self,
        spec: OpenShellSandboxSpec,
        *,
        policy_path: Path,
        timeout: int = 300,
    ) -> dict[str, object]:
        spec = spec.validated()
        spec.policy.write(policy_path)
        argv = [
            "sandbox",
            "create",
            "--name",
            spec.name,
            "--from",
            spec.image,
            "--policy",
            str(policy_path),
            "--detach",
            "--output",
            "json",
        ]
        if spec.cpu:
            argv.extend(["--cpu", spec.cpu])
        if spec.memory:
            argv.extend(["--memory", spec.memory])
        for key, value in sorted(spec.labels.items()):
            argv.extend(["--label", f"{key}={value}"])
        for provider in spec.providers:
            if not str(provider).strip():
                raise ValueError("OpenShell provider names cannot be empty")
            argv.extend(["--provider", str(provider).strip()])
        # Keep a scratch sandbox alive while Forge Bench uploads task files and
        # executes one or more managed commands.
        argv.extend(["--", "/bin/sh", "-lc", "while :; do sleep 3600; done"])
        completed = self._run(argv, timeout=timeout)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenShell sandbox create did not return valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("OpenShell sandbox create JSON root must be an object")
        return payload

    def upload(
        self,
        sandbox: str,
        source: Path,
        destination: str | None = None,
        *,
        timeout: int = 300,
    ) -> None:
        argv = ["sandbox", "upload", sandbox, str(source)]
        if destination is not None:
            argv.append(destination)
        self._run(argv, timeout=timeout)

    def exec(
        self,
        sandbox: str,
        command: Sequence[str],
        *,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
        no_login_shell: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = ["sandbox", "exec", "-n", sandbox, "--timeout", str(timeout)]
        if no_login_shell:
            argv.append("--no-login-shell")
        if workdir:
            argv.extend(["--workdir", workdir])
        for key, value in sorted((env or {}).items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.extend(["--", *command])
        return self._run(argv, timeout=timeout + 30, check=False)

    def download(
        self,
        sandbox: str,
        source: str,
        destination: Path,
        *,
        timeout: int = 300,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            ["sandbox", "download", sandbox, source, str(destination)],
            timeout=timeout,
        )

    def effective_policy(
        self,
        sandbox: str,
        destination: Path,
        *,
        timeout: int = 60,
    ) -> Path:
        completed = self._run(
            ["sandbox", "get", sandbox, "--policy-only"],
            timeout=timeout,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(completed.stdout, encoding="utf-8")
        return destination

    def logs(
        self,
        sandbox: str,
        destination: Path,
        *,
        level: str = "info",
        since: str = "1h",
        timeout: int = 60,
    ) -> Path:
        completed = self._run(
            ["logs", sandbox, "--level", level, "--since", since],
            timeout=timeout,
            check=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        return destination

    def delete(self, sandbox: str, *, timeout: int = 120) -> None:
        self._run(["sandbox", "delete", sandbox], timeout=timeout, check=False)
