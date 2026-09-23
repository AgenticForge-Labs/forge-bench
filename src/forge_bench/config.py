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
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_SEED = 260919
DEFAULT_HISTORICAL_SUBMISSIONS = 135

# The second-stage benchmark anchors task selection on the two small,
# historically high-solve tasks whose baseline behavior we have already
# observed. Three additional tasks are chosen deterministically for similarity
# to these anchors from the pinned Verified medium pool.
DEFAULT_ANCHOR_IDS = (
    "django__django-13516",
    "pytest-dev__pytest-7571",
)

# Expected deterministic result of the pinned anchor-neighborhood selector.
# CI verifies this exact set so a future selector/data change cannot silently
# alter the second-stage experiment.
DEFAULT_EXPECTED_IDS = (
    "django__django-13516",
    "pytest-dev__pytest-7571",
    "django__django-15731",
    "django__django-16662",
    "django__django-7530",
)

DEFAULT_TOOLSETS = "hermes-cli"
LEAN_TOOLSETS = "file,terminal,skills,code_execution"

# Available treatments. Lean-tool variants remain available for follow-up
# experiments, but the default second-stage design is the clean Caveman x
# Ponytail 2x2 using normal Hermes tools in every arm.
ARMS = {
    "baseline": (False, False, False),
    "caveman": (True, False, False),
    "ponytail": (False, True, False),
    "caveman_ponytail": (True, True, False),
    "lean_tools": (False, False, True),
    "all_three": (True, True, True),
}

DEFAULT_ARMS = (
    "baseline",
    "caveman",
    "ponytail",
    "caveman_ponytail",
)

LABEL = {
    "baseline": "Baseline Hermes",
    "caveman": "Caveman",
    "ponytail": "Ponytail",
    "caveman_ponytail": "Caveman + Ponytail",
    "lean_tools": "Lean tools",
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

# Two-sided 95% Student-t critical values. Default n=5 tasks => df=4 => 2.77645.
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
    # Experimental factors are stored per run so factorial analyses never need
    # to infer the assigned condition from filenames or global metadata.
    max_turns: int | None = None
    budget_warning_ratio: float | None = None
    # Keep main-loop and auxiliary API calls separate. api_calls remains the
    # billable total including auxiliary calls for backward compatibility.
    main_api_calls: int | None = None
    auxiliary_api_calls: int | None = None
    iterations_used: int | None = None
    turn_exit_reason: str = ""
    # Direct timing decomposition from the Forge Bench native Hermes observer.
    # These stay optional so historical runs remain reanalyzable.
    api_wait_seconds: float | None = None
    tool_execution_seconds: float | None = None
    terminal_execution_seconds: float | None = None
    unattributed_wall_seconds: float | None = None
    api_duration_mean_seconds: float | None = None
    api_duration_p95_seconds: float | None = None
    ttft_mean_seconds: float | None = None
    timing_event_count: int = 0
    skill_lifecycle_event_count: int = 0
    trace_exported: bool = False
