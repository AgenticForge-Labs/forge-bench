from __future__ import annotations

from dataclasses import dataclass

PINNED_MODEL = "deepseek/deepseek-v4-flash-0731"
PINNED_OPENROUTER_UPSTREAM = "relace"
PINNED_REASONING = "none"
PINNED_MAX_TURNS = 50

PONY_REPO = "DietrichGebert/ponytail"
PONY_SHA = "e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156"

CAVE_SHA = "542442bab314973709f95b85b1ac0b3f6f5b5dc6"
CAVE_URL = (
    "https://raw.githubusercontent.com/JuliusBrussee/caveman/"
    + CAVE_SHA
    + "/skills/caveman/SKILL.md"
)

DEFAULT_DATASET = "verified"
DEFAULT_DIFFICULTY = "medium"
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_SEED = 260919
DEFAULT_HISTORICAL_SUBMISSIONS = 135

# Frozen initial suite, selected once by the smart sampler from SWE-bench
# Verified's official 15 min - 1 hour bucket on 2026-09-19.
DEFAULT_SUITE = [
    {
        "instance_id": "django__django-13516",
        "selection_rank": "low-1",
        "patch_scope_percentile": 0.285,
        "historical_solve_rate": 0.844,
        "patch_changed_lines": 4,
        "patch_files": 1,
    },
    {
        "instance_id": "pytest-dev__pytest-7571",
        "selection_rank": "low-2",
        "patch_scope_percentile": 0.362,
        "historical_solve_rate": 0.793,
        "patch_changed_lines": 4,
        "patch_files": 1,
    },
    {
        "instance_id": "sympy__sympy-20154",
        "selection_rank": "low-3",
        "patch_scope_percentile": 0.714,
        "historical_solve_rate": 0.785,
        "patch_changed_lines": 23,
        "patch_files": 1,
    },
]

DEFAULT_TOOLSETS = "hermes-cli"
LEAN_TOOLSETS = "file,terminal,skills,code_execution"

# Initial comparison: each token-saving approach alone, plus all together.
ARMS = {
    "baseline": (False, False, False),
    "lean_tools": (False, False, True),
    "caveman": (True, False, False),
    "ponytail": (False, True, False),
    "all_three": (True, True, True),
}

LABEL = {
    "baseline": "Baseline Hermes",
    "lean_tools": "Lean tools",
    "caveman": "Caveman",
    "ponytail": "Ponytail",
    "all_three": "Caveman + Ponytail + Lean",
}

METRICS = {
    "total_tokens": ("Total tokens", "tokens"),
    "input_tokens": ("Input tokens", "tokens"),
    "output_tokens": ("Output tokens", "tokens"),
    "reasoning_tokens": ("Reasoning tokens", "tokens"),
    "cache_read_tokens": ("Cache-read tokens", "tokens"),
    "cost_usd": ("OpenRouter cost", "usd"),
    "wall_seconds": ("Wall-clock time", "seconds"),
    "api_calls": ("API calls", "count"),
}

# Two-sided 95% Student-t critical values. n=3 tasks => df=2 => 4.30265.
T975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
    16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539,
    26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272,
}


@dataclass
class Result:
    arm: str
    task: str
    repeat: int
    run_index: int
    valid: bool
    resolved: bool
    evaluation_completed: bool
    patch_nonempty: bool
    completed: bool
    exit_code: int
    wall_seconds: float
    evaluation_seconds: float
    repo: str
    difficulty: str
    model: str
    api_provider: str
    upstream_provider: str
    session_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    api_calls: int
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    cost_usd: float | None
    cost_source: str
    tool_calls: int | None
    files_changed: int
    diff_lines: int
    run_dir: str
    error: str = ""
