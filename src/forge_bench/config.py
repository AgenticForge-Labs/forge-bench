from __future__ import annotations

from dataclasses import dataclass

PINNED_MODEL = "deepseek/deepseek-v4-flash-0731"
PINNED_OPENROUTER_UPSTREAM = "baidu"

QUIX_URL = "https://github.com/jkoppel/QuixBugs.git"
QUIX_SHA = "4257f44b0ff1181dedaedee6a447e133219fcebf"

PONY_REPO = "DietrichGebert/ponytail"
PONY_SHA = "e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156"

CAVE_SHA = "542442bab314973709f95b85b1ac0b3f6f5b5dc6"
CAVE_URL = (
    "https://raw.githubusercontent.com/JuliusBrussee/caveman/"
    + CAVE_SHA
    + "/skills/caveman/SKILL.md"
)

DEFAULT_TASKS = ["quicksort", "next_permutation", "shunting_yard"]
DEFAULT_SEED = 260919

ARMS = {
    "caveman": (True, False, False),
    "ponytail": (False, True, False),
    "execute_code": (False, False, True),
    "all_three": (True, True, True),
}

LABEL = {
    "baseline": "Baseline",
    "caveman": "Caveman",
    "ponytail": "Ponytail",
    "execute_code": "execute_code",
    "all_three": "All three",
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
    tests_passed: bool
    tests_untouched: bool
    completed: bool
    exit_code: int
    wall_seconds: float
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
