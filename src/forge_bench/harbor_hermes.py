from __future__ import annotations

import json
import shlex
from typing import Any, override

import yaml
from pydantic import Field, field_validator

from harbor.agents.installed.hermes import Hermes, HermesOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .config import ARMS, CAVE_URL, PONY_REPO, PONY_SHA


class ForgeBenchHermesOptions(HermesOptions):
    treatment: str = Field(default="baseline")
    upstream_provider: str = Field(default="relace", min_length=1)
    reasoning: str = Field(default="none", min_length=1)
    budget_warning_ratio: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("treatment")
    @classmethod
    def validate_treatment(cls, value: str) -> str:
        if value not in ARMS:
            raise ValueError(f"unknown Forge Bench treatment: {value}")
        return value


class ForgeBenchHermes(Hermes):
    """Thin Forge specialization of Harbor's first-party Hermes adapter.

    Harbor owns installation, task-environment execution, session export, ATIF
    conversion and base token accounting. Forge only overlays experimental
    treatment/config factors and evidence required to map the trial back into a
    Forge Bench observation.
    """

    options_model = ForgeBenchHermesOptions
    options: ForgeBenchHermesOptions

    @staticmethod
    @override
    def name() -> str:
        return "forge-bench-hermes"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        caveman, ponytail, _lean = ARMS[self.options.treatment]
        env = {"HERMES_HOME": "/tmp/hermes"}

        if ponytail:
            command = (
                'export PATH="$HOME/.local/bin:$PATH"; '
                "printf 'y\\ny\\ny\\ny\\n' | "
                f"hermes plugins install {shlex.quote(PONY_REPO)} "
                f"--ref {shlex.quote(PONY_SHA)} --force --enable"
            )
            result = await self.exec_as_agent(
                environment,
                command=command,
                env=env,
                timeout_sec=240,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Ponytail installation failed: "
                    + (result.stderr or result.stdout or "unknown error")
                )

        if caveman:
            command = (
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"hermes skills install {shlex.quote(CAVE_URL)} "
                "--name caveman --force --yes"
            )
            result = await self.exec_as_agent(
                environment,
                command=command,
                env=env,
                timeout_sec=240,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Caveman installation failed: "
                    + (result.stderr or result.stdout or "unknown error")
                )

    @override
    def _build_config_yaml(self, model: str, max_turns: int | None = None) -> str:
        config = yaml.safe_load(super()._build_config_yaml(model, max_turns)) or {}

        agent = config.setdefault("agent", {})
        agent["reasoning_effort"] = self.options.reasoning
        agent["budget_warning_ratio"] = self.options.budget_warning_ratio

        config["provider_routing"] = {
            "only": [self.options.upstream_provider],
            "data_collection": "allow",
            "require_parameters": True,
            "models": {
                model: {
                    "only": [self.options.upstream_provider],
                    "require_parameters": True,
                }
            },
        }
        config["compression"] = {"enabled": False}
        config["auxiliary"] = {
            "title_generation": {
                "enabled": False,
                "model_upgrade_enabled": False,
                "provider": "auto",
                "model": "",
            }
        }

        _caveman, ponytail, _lean = ARMS[self.options.treatment]
        config["plugins"] = {
            "enabled": ["ponytail"] if ponytail else [],
            "disabled": [],
        }

        return yaml.safe_dump(config, sort_keys=False)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        base_commit = ""
        try:
            base = await self.exec_as_agent(
                environment,
                command="git rev-parse HEAD",
                timeout_sec=30,
            )
            if base.return_code == 0:
                base_commit = (base.stdout or "").strip()
        except Exception:
            base_commit = ""

        try:
            await super().run(instruction, environment, context)
        finally:
            metadata = dict(context.metadata or {})
            metadata.update(
                {
                    "forge_treatment": self.options.treatment,
                    "forge_upstream_provider": self.options.upstream_provider,
                    "forge_reasoning": self.options.reasoning,
                    "forge_max_turns": self.options.max_turns,
                    "forge_budget_warning_ratio": self.options.budget_warning_ratio,
                    "forge_base_commit": base_commit,
                }
            )
            if base_commit:
                try:
                    stats = await self._capture_patch(environment, base_commit)
                    metadata.update(stats)
                except Exception as exc:
                    metadata["forge_patch_capture_error"] = str(exc)
            else:
                metadata["forge_patch_capture_error"] = "could not resolve task base commit"
            context.metadata = metadata

    async def _capture_patch(
        self,
        environment: BaseEnvironment,
        base_commit: str,
    ) -> dict[str, Any]:
        quoted_base = shlex.quote(base_commit)
        command = (
            "set -e; "
            "tmp_index=$(mktemp); "
            "trap 'rm -f \"$tmp_index\"' EXIT; "
            f"GIT_INDEX_FILE=\"$tmp_index\" git read-tree {quoted_base}; "
            'GIT_INDEX_FILE="$tmp_index" git add -A -- .; '
            f'GIT_INDEX_FILE="$tmp_index" git diff --cached --binary {quoted_base} '
            "> /logs/agent/model.patch; "
            f'GIT_INDEX_FILE="$tmp_index" git diff --cached --numstat {quoted_base} '
            "> /logs/agent/forge-numstat.txt; "
            "cat /logs/agent/forge-numstat.txt"
        )
        result = await self.exec_as_agent(
            environment,
            command=command,
            timeout_sec=60,
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "git diff capture failed")

        files_changed = 0
        diff_lines = 0
        for line in (result.stdout or "").splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            files_changed += 1
            for raw in fields[:2]:
                if raw.isdigit():
                    diff_lines += int(raw)

        return {
            "forge_patch_nonempty": files_changed > 0,
            "forge_files_changed": files_changed,
            "forge_diff_lines": diff_lines,
        }

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        metadata = dict(context.metadata or {})

        session_path = self.logs_dir / "hermes-session.jsonl"
        if session_path.is_file():
            text = session_path.read_text(encoding="utf-8")
            metadata["forge_session_id"] = self._extract_native_session_id(text) or ""
            api_calls = 0
            tool_calls = 0
            for line in text.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = (
                    record.get("messages", [])
                    if isinstance(record, dict) and isinstance(record.get("messages"), list)
                    else [record]
                )
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if message.get("role") == "assistant":
                        if isinstance(message.get("usage"), dict) and message["usage"]:
                            api_calls += 1
                        calls = message.get("tool_calls")
                        if isinstance(calls, list):
                            tool_calls += len(calls)
            metadata["forge_api_calls"] = api_calls
            metadata["forge_tool_calls"] = tool_calls

        context.metadata = metadata
