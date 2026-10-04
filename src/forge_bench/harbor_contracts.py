from __future__ import annotations

from .config import ARMS, DEFAULT_TOOLSETS, LEAN_TOOLSETS

PINNED_HARBOR_VERSION = "0.23.0"
PINNED_HERMES_VERSION = "v2026.9.14"
FORGE_HERMES_IMPORT_PATH = "forge_bench.harbor_hermes:ForgeBenchHermes"


def toolsets_for_treatment(treatment: str) -> str:
    try:
        _caveman, _ponytail, lean = ARMS[treatment]
    except KeyError as exc:
        raise ValueError(f"Unknown Forge Bench treatment: {treatment!r}") from exc
    return LEAN_TOOLSETS if lean else DEFAULT_TOOLSETS
