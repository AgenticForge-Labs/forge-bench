from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ARMS,
    DEFAULT_ARMS,
    DEFAULT_DATASET,
    DEFAULT_DIFFICULTY,
    DEFAULT_EXPECTED_IDS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    PINNED_MAX_TURNS,
    PINNED_MODEL,
    PINNED_OPENROUTER_UPSTREAM,
    PINNED_REASONING,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model: str
    api_provider: str = "openrouter"
    upstream_provider: str = PINNED_OPENROUTER_UPSTREAM
    reasoning: str = PINNED_REASONING

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentDesign:
    name: str
    description: str
    dataset: str
    split: str
    difficulty: str
    sample_size: int
    instance_ids: tuple[str, ...] | None
    arms: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    blocks: int
    seed: int
    max_turns: int
    budget_warning_ratio: float | None
    max_turns_levels: tuple[int, ...]
    budget_warning_ratio_levels: tuple[float | None, ...]
    analysis_mode: str
    require_same_upstream: bool
    source_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arms"] = list(self.arms)
        payload["models"] = [model.as_dict() for model in self.models]
        payload["instance_ids"] = list(self.instance_ids) if self.instance_ids else None
        return payload


def default_design() -> ExperimentDesign:
    return ExperimentDesign(
        name="default",
        description="Single-model Caveman x Ponytail 2x2 benchmark.",
        dataset=DEFAULT_DATASET,
        split="test",
        difficulty="auto",
        sample_size=DEFAULT_SAMPLE_SIZE,
        instance_ids=None,
        arms=tuple(DEFAULT_ARMS),
        models=(
            ModelSpec(
                key="deepseek-v4-flash-0731",
                label="DeepSeek V4 Flash 0731",
                model=PINNED_MODEL,
            ),
        ),
        blocks=1,
        seed=DEFAULT_SEED,
        max_turns=PINNED_MAX_TURNS,
        budget_warning_ratio=None,
        max_turns_levels=(PINNED_MAX_TURNS,),
        budget_warning_ratio_levels=(None,),
        analysis_mode="basic",
        require_same_upstream=True,
    )


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _clean_key(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in text):
        raise ValueError(f"{field} may contain only letters, numbers, '-' and '_'")
    return text


def load_design(path: Path) -> ExperimentDesign:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "design")

    version = int(root.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported design version: {version}")

    name = _clean_key(root.get("name"), "name")
    description = str(root.get("description") or "").strip()

    selection = _require_mapping(root.get("selection") or {}, "selection")
    dataset = str(selection.get("dataset") or DEFAULT_DATASET)
    split = str(selection.get("split") or "test")
    difficulty = str(selection.get("difficulty") or "auto")
    sample_size = int(selection.get("sample_size") or DEFAULT_SAMPLE_SIZE)
    if sample_size < 1:
        raise ValueError("selection.sample_size must be >= 1")
    instance_values = selection.get("instance_ids")
    instance_ids = None
    if instance_values is not None:
        if not isinstance(instance_values, list) or not instance_values:
            raise ValueError("selection.instance_ids must be a non-empty list")
        instance_ids = tuple(str(value).strip() for value in instance_values)
        if any(not value for value in instance_ids):
            raise ValueError("selection.instance_ids may not contain empty values")
        sample_size = len(instance_ids)

    treatments = root.get("treatments", list(DEFAULT_ARMS))
    if not isinstance(treatments, list) or not treatments:
        raise ValueError("treatments must be a non-empty list")
    arms = tuple(str(value) for value in treatments)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError("Unknown treatment(s): " + ", ".join(unknown))
    if len(set(arms)) != len(arms):
        raise ValueError("treatments must not contain duplicates")

    provider = _require_mapping(root.get("provider") or {}, "provider")
    api_provider = str(provider.get("api") or "openrouter").strip().lower()
    upstream = str(provider.get("upstream") or PINNED_OPENROUTER_UPSTREAM).strip()
    require_same_upstream = bool(provider.get("require_same_upstream", True))
    if api_provider != "openrouter":
        raise ValueError("Forge Bench experiment designs currently support api: openrouter only")
    if not upstream:
        raise ValueError("provider.upstream must be non-empty")

    model_values = root.get("models")
    if not isinstance(model_values, list) or not model_values:
        raise ValueError("models must be a non-empty list")

    models: list[ModelSpec] = []
    for index, item in enumerate(model_values):
        entry = _require_mapping(item, f"models[{index}]")
        model_id = str(entry.get("model") or "").strip()
        if not model_id:
            raise ValueError(f"models[{index}].model must be non-empty")
        key = _clean_key(entry.get("key") or model_id.rsplit("/", 1)[-1], f"models[{index}].key")
        label = str(entry.get("label") or key).strip()
        model_api = str(entry.get("api_provider") or api_provider).strip().lower()
        model_upstream = str(entry.get("upstream_provider") or upstream).strip()
        reasoning = str(entry.get("reasoning") or root.get("reasoning") or PINNED_REASONING).strip()
        if model_api != "openrouter":
            raise ValueError(f"models[{index}] uses unsupported api_provider {model_api!r}")
        models.append(
            ModelSpec(
                key=key,
                label=label,
                model=model_id,
                api_provider=model_api,
                upstream_provider=model_upstream,
                reasoning=reasoning,
            )
        )

    if len({model.key for model in models}) != len(models):
        raise ValueError("model keys must be unique")
    if require_same_upstream and len({model.upstream_provider for model in models}) != 1:
        raise ValueError("provider.require_same_upstream=true but model upstream providers differ")
    if require_same_upstream and len({model.api_provider for model in models}) != 1:
        raise ValueError("provider.require_same_upstream=true but model API providers differ")

    randomization = _require_mapping(root.get("randomization") or {}, "randomization")
    blocks = int(randomization.get("blocks", root.get("blocks", 1)))
    seed = int(randomization.get("seed", root.get("seed", DEFAULT_SEED)))
    if blocks < 1:
        raise ValueError("randomization.blocks must be >= 1")
    mode = str(randomization.get("mode") or "full-factorial-within-block")
    if mode != "full-factorial-within-block":
        raise ValueError(
            "randomization.mode must be 'full-factorial-within-block'; "
            "each block contains every model x treatment x task cell exactly once"
        )

    factors = _require_mapping(root.get("factors") or {}, "factors")

    max_turns_source = factors.get(
        "max_turns",
        [root.get("max_turns", PINNED_MAX_TURNS)],
    )
    if not isinstance(max_turns_source, list):
        max_turns_source = [max_turns_source]
    if not max_turns_source:
        raise ValueError("factors.max_turns must contain at least one level")
    max_turns_levels = tuple(int(value) for value in max_turns_source)
    if any(value < 1 for value in max_turns_levels):
        raise ValueError("all factors.max_turns levels must be >= 1")
    if len(set(max_turns_levels)) != len(max_turns_levels):
        raise ValueError("factors.max_turns must not contain duplicate levels")

    warning_source = factors.get(
        "budget_warning_ratio",
        [root.get("budget_warning_ratio")],
    )
    if not isinstance(warning_source, list):
        warning_source = [warning_source]
    if not warning_source:
        raise ValueError("factors.budget_warning_ratio must contain at least one level")
    budget_warning_ratio_levels = tuple(
        None if value is None else float(value)
        for value in warning_source
    )
    if any(
        value is not None and not (0.0 < value < 1.0)
        for value in budget_warning_ratio_levels
    ):
        raise ValueError(
            "all factors.budget_warning_ratio levels must be null or strictly between 0 and 1"
        )
    if len(set(budget_warning_ratio_levels)) != len(budget_warning_ratio_levels):
        raise ValueError("factors.budget_warning_ratio must not contain duplicate levels")

    # Keep the scalar attributes for backward compatibility with older callers.
    # Factorial execution uses the explicit level tuples below.
    max_turns = max_turns_levels[0]
    budget_warning_ratio = budget_warning_ratio_levels[0]

    analysis_mode = str(root.get("analysis_mode") or "advanced")
    if analysis_mode not in {"basic", "advanced"}:
        raise ValueError("analysis_mode must be basic or advanced")

    return ExperimentDesign(
        name=name,
        description=description,
        dataset=dataset,
        split=split,
        difficulty=difficulty,
        sample_size=sample_size,
        instance_ids=instance_ids,
        arms=arms,
        models=tuple(models),
        blocks=blocks,
        seed=seed,
        max_turns=max_turns,
        budget_warning_ratio=budget_warning_ratio,
        max_turns_levels=max_turns_levels,
        budget_warning_ratio_levels=budget_warning_ratio_levels,
        analysis_mode=analysis_mode,
        require_same_upstream=require_same_upstream,
        source_path=str(path),
    )


def design_model_map(design: ExperimentDesign) -> dict[str, ModelSpec]:
    return {model.key: model for model in design.models}
