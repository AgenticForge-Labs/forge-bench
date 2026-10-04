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
    api_provider: str = Field(default="openrouter")
    upstream_provider: str = Field(default="relace", min_length=1)
    reasoning: str = Field(default="none", min_length=1)
    budget_warning_ratio: float | None = Field(default=None, gt=0.0, lt=1.0)
    max_turns: int | None = Field(default=None, ge=1)

    @field_validator("treatment")
    @classmethod
    def validate_treatment(cls, value: str) -> str:
        if value not in ARMS:
            raise ValueError(f"unknown Forge Bench treatment: {value}")
        return value

    @field_validator("api_provider")
    @classmethod
    def validate_api_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "openrouter":
            raise ValueError("Forge Bench Harbor execution currently requires openrouter")
        return normalized


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
        # Harbor 0.23.0's Hermes adapter hard-codes 90 turns and exposes no
        # max_turns option. Override that here so Forge's randomized budget
        # remains authoritative. This stays compatible with the later Harbor
        # signature because Forge owns the final config value.
        config = yaml.safe_load(super()._build_config_yaml(model)) or {}

        effective_max_turns = (
            max_turns if max_turns is not None else self.options.max_turns
        )
        agent = config.setdefault("agent", {})
        if effective_max_turns is not None:
            agent["max_turns"] = int(effective_max_turns)
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
            try:
                await self._refresh_session_export(environment)
            except Exception:
                pass
            metadata = dict(context.metadata or {})
            metadata.update(
                {
                    "forge_treatment": self.options.treatment,
                    "forge_api_provider": self.options.api_provider,
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

    async def _refresh_session_export(self, environment: BaseEnvironment) -> None:
        """Re-export the primary oneshot session for pinned modern Hermes.

        Harbor 0.23.0 still asks for source=cli, while Hermes v2026.9.14 stores
        finite chat -q sessions as source=oneshot. Prefer oneshot and fall back
        to cli for compatibility with older Hermes releases.
        """
        command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "rm -f /logs/agent/hermes-session.jsonl; "
            "(hermes sessions export /logs/agent/hermes-session.jsonl "
            "--format jsonl --source oneshot "
            "|| hermes sessions export /logs/agent/hermes-session.jsonl "
            "--format jsonl --source cli) >/dev/null 2>&1 || true"
        )
        await self.exec_as_agent(
            environment,
            command=command,
            env={"HERMES_HOME": "/tmp/hermes"},
            timeout_sec=60,
        )

    @staticmethod
    def _session_records(text: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in (text or "").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            session = payload.get("session")
            if isinstance(session, dict):
                record = dict(session)
                if isinstance(payload.get("messages"), list):
                    record["messages"] = payload["messages"]
                records.append(record)
            else:
                records.append(payload)
        if records:
            return records
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            session = payload.get("session")
            return [dict(session)] if isinstance(session, dict) else [payload]
        return []

    @staticmethod
    def _usage_record(
        records: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        if session_id:
            for record in records:
                if str(record.get("id") or record.get("session_id") or "") == session_id:
                    return record
        return records[0] if records else {}

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        metadata = dict(context.metadata or {})

        session_path = self.logs_dir / "hermes-session.jsonl"
        if session_path.is_file():
            text = session_path.read_text(encoding="utf-8")
            metadata["forge_trace_exported"] = (self.logs_dir / "trajectory.json").is_file()
            session_id = self._extract_native_session_id(text) or ""
            metadata["forge_session_id"] = session_id

            records = self._session_records(text)
            usage = self._usage_record(records, session_id)

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

            uncached_input = as_int(usage.get("input_tokens"))
            output_tokens = as_int(usage.get("output_tokens"))
            cache_read = as_int(usage.get("cache_read_tokens"))
            cache_write = as_int(usage.get("cache_write_tokens"))
            reasoning_tokens = as_int(usage.get("reasoning_tokens"))
            api_calls = as_int(usage.get("api_call_count"))
            estimated_cost = as_float(usage.get("estimated_cost_usd"))
            actual_cost = as_float(usage.get("actual_cost_usd"))

            if any(
                key in usage
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            ):
                # Harbor's AgentContext input count is inclusive of cache.
                context.n_input_tokens = uncached_input + cache_read + cache_write
                context.n_cache_tokens = cache_read + cache_write
                context.n_output_tokens = output_tokens
                if actual_cost is not None and actual_cost > 0:
                    context.cost_usd = actual_cost
                elif estimated_cost is not None:
                    context.cost_usd = estimated_cost

                metadata.update(
                    {
                        "forge_uncached_input_tokens": uncached_input,
                        "forge_output_tokens": output_tokens,
                        "forge_cache_read_tokens": cache_read,
                        "forge_cache_write_tokens": cache_write,
                        "forge_reasoning_tokens": reasoning_tokens,
                        "forge_total_tokens": as_int(usage.get("total_tokens"))
                        or (
                            uncached_input
                            + output_tokens
                            + cache_read
                            + cache_write
                        ),
                        "forge_api_calls": api_calls,
                        "forge_estimated_cost_usd": estimated_cost,
                        "forge_actual_cost_usd": actual_cost,
                        "forge_cost_status": str(usage.get("cost_status") or ""),
                        "forge_cost_source": str(usage.get("cost_source") or ""),
                    }
                )

            tool_calls = 0
            fallback_api_calls = 0
            messages = usage.get("messages") if isinstance(usage.get("messages"), list) else []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "assistant":
                    if isinstance(message.get("usage"), dict) and message["usage"]:
                        fallback_api_calls += 1
                    calls = message.get("tool_calls")
                    if isinstance(calls, list):
                        tool_calls += len(calls)
            if not metadata.get("forge_api_calls"):
                metadata["forge_api_calls"] = fallback_api_calls
            metadata["forge_tool_calls"] = tool_calls

        context.metadata = metadata
