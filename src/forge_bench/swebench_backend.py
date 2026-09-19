from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

EXPERIMENTS_REPO = "https://github.com/SWE-bench/experiments.git"
EXPERIMENTS_SHA = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"

DATASET_ALIASES = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "full": "SWE-bench/SWE-bench",
}

DATASET_REVISIONS = {
    # Current Verified data snapshot used to freeze the initial three tasks.
    "SWE-bench/SWE-bench_Verified": "78f471bf655a3137b2e8a75af1501690ec009ec3",
}

DIFFICULTY_ALIASES = {
    "easy": {"<15 min fix", "<15 min"},
    "medium": {"15 min - 1 hour", "15 min–1 hour", "15 min-1 hour"},
    "hard": {"1-4 hours", "1–4 hours"},
    "expert": {">4 hours", "4+ hours", "4 hours"},
}


@dataclass
class Candidate:
    instance_id: str
    repo: str
    base_commit: str
    difficulty: str
    problem_statement: str
    version: str
    patch_files: int
    patch_hunks: int
    patch_additions: int
    patch_deletions: int
    patch_changed_lines: int
    patch_chars: int
    fail_to_pass_count: int
    pass_to_pass_count: int
    historical_solve_rate: float | None
    historical_submissions: int
    patch_scope_percentile: float = 0.0
    history_hardness_percentile: float | None = None
    complexity_score: float = 0.0
    selection_target: float | None = None
    selection_rank: str = ""


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.lower(), name)


def normalize_difficulty(value: str) -> str:
    text = (value or "").strip()
    for alias, forms in DIFFICULTY_ALIASES.items():
        if text in forms:
            return alias
    return text


def dataset_revision(dataset_name: str) -> str | None:
    return DATASET_REVISIONS.get(resolve_dataset_name(dataset_name))


def load_rows(dataset_name: str, split: str = "test") -> list[dict[str, Any]]:
    resolved = resolve_dataset_name(dataset_name)
    revision = DATASET_REVISIONS.get(resolved)
    kwargs = {"revision": revision} if revision else {}
    ds = load_dataset(resolved, split=split, **kwargs)
    return [dict(row) for row in ds]


def patch_features(patch: str) -> dict[str, int]:
    files: set[str] = set()
    additions = 0
    deletions = 0
    hunks = 0

    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                files.add(path[2:] if path.startswith("b/") else path)
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {
        "patch_files": len(files),
        "patch_hunks": hunks,
        "patch_additions": additions,
        "patch_deletions": deletions,
        "patch_changed_lines": additions + deletions,
        "patch_chars": len(patch or ""),
    }


def _ensure_experiments_cache(cache_root: Path) -> tuple[Path, str]:
    root = cache_root / "swe-bench-experiments"
    marker = root / ".forge-bench-experiments-sha"

    if root.exists() and marker.exists() and marker.read_text().strip() == EXPERIMENTS_SHA:
        return root, EXPERIMENTS_SHA

    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    clone = _run(
        [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--no-checkout",
            EXPERIMENTS_REPO,
            str(root),
        ]
    )
    if clone.returncode:
        raise RuntimeError("Could not clone SWE-bench experiments: " + clone.stderr)

    sparse = _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=root)
    if sparse.returncode:
        raise RuntimeError("Could not initialize sparse checkout: " + sparse.stderr)

    pattern_file = root / ".git" / "info" / "sparse-checkout"
    pattern_file.write_text(
        "/evaluation/verified/*/results/results.json\n",
        encoding="utf-8",
    )

    fetch = _run(["git", "fetch", "--depth=1", "origin", EXPERIMENTS_SHA], cwd=root)
    if fetch.returncode:
        raise RuntimeError("Could not fetch pinned experiments commit: " + fetch.stderr)

    checkout = _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=root)
    if checkout.returncode:
        raise RuntimeError("Could not checkout pinned experiments commit: " + checkout.stderr)

    marker.write_text(EXPERIMENTS_SHA + "\n", encoding="utf-8")
    return root, EXPERIMENTS_SHA


def historical_solve_rates(
    cache_root: Path,
    instance_ids: set[str],
) -> tuple[dict[str, float], int, str]:
    """Return solve fraction across official Verified leaderboard submissions.

    Official Verified results list resolved IDs. For accepted Verified submissions,
    unresolved instances are the complement of the 500-instance benchmark. This
    function therefore uses every compatible results.json as one Bernoulli result
    per instance.
    """
    root, sha = _ensure_experiments_cache(cache_root)
    result_files = sorted(
        (root / "evaluation" / "verified").glob("*/results/results.json")
    )

    solved = {instance_id: 0 for instance_id in instance_ids}
    usable = 0

    for path in result_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        resolved = payload.get("resolved")
        if not isinstance(resolved, list):
            continue
        usable += 1
        for instance_id in set(resolved) & instance_ids:
            solved[instance_id] += 1

    if usable == 0:
        return {}, 0, sha

    return (
        {instance_id: count / usable for instance_id, count in solved.items()},
        usable,
        sha,
    )


def _percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    n = len(values)
    if n == 1:
        return [0.5]

    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and ordered[end][0] == ordered[pos][0]:
            end += 1
        midpoint = (pos + end - 1) / 2
        pct = midpoint / (n - 1)
        for _, original_index in ordered[pos:end]:
            ranks[original_index] = pct
        pos = end

    return ranks


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    difficulty: str | None,
    cache_root: Path,
    use_history: bool = True,
) -> tuple[list[Candidate], dict[str, Any]]:
    requested = normalize_difficulty(difficulty or "") if difficulty else None

    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_difficulty = normalize_difficulty(str(row.get("difficulty") or ""))
        if requested and row_difficulty != requested:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError(
            f"No SWE-bench rows match difficulty={difficulty!r}. "
            "The selected dataset may not provide difficulty annotations."
        )

    ids = {str(row["instance_id"]) for row in filtered}
    rates: dict[str, float] = {}
    submissions = 0
    experiments_sha = ""
    history_error = ""

    if use_history:
        try:
            rates, submissions, experiments_sha = historical_solve_rates(
                cache_root, ids
            )
        except Exception as exc:
            rates = {}
            submissions = 0
            experiments_sha = ""
            history_error = str(exc)

    candidates: list[Candidate] = []
    for row in filtered:
        features = patch_features(str(row.get("patch") or ""))
        f2p = row.get("FAIL_TO_PASS") or []
        p2p = row.get("PASS_TO_PASS") or []
        if isinstance(f2p, str):
            try:
                f2p = json.loads(f2p)
            except Exception:
                f2p = []
        if isinstance(p2p, str):
            try:
                p2p = json.loads(p2p)
            except Exception:
                p2p = []

        instance_id = str(row["instance_id"])
        candidates.append(
            Candidate(
                instance_id=instance_id,
                repo=str(row["repo"]),
                base_commit=str(row["base_commit"]),
                difficulty=normalize_difficulty(str(row.get("difficulty") or "")),
                problem_statement=str(row.get("problem_statement") or ""),
                version=str(row.get("version") or ""),
                fail_to_pass_count=len(f2p),
                pass_to_pass_count=len(p2p),
                historical_solve_rate=rates.get(instance_id),
                historical_submissions=submissions if rates else 0,
                **features,
            )
        )

    line_pct = _percentile_ranks(
        [float(candidate.patch_changed_lines) for candidate in candidates]
    )
    file_pct = _percentile_ranks(
        [float(candidate.patch_files) for candidate in candidates]
    )
    hunk_pct = _percentile_ranks(
        [float(candidate.patch_hunks) for candidate in candidates]
    )

    history_values = [
        1.0 - candidate.historical_solve_rate
        for candidate in candidates
        if candidate.historical_solve_rate is not None
    ]
    history_pct_values = _percentile_ranks(history_values)
    history_iter = iter(history_pct_values)

    for i, candidate in enumerate(candidates):
        candidate.patch_scope_percentile = (
            0.60 * line_pct[i] + 0.25 * file_pct[i] + 0.15 * hunk_pct[i]
        )
        if candidate.historical_solve_rate is not None:
            candidate.history_hardness_percentile = next(history_iter)
            candidate.complexity_score = (
                0.55 * candidate.patch_scope_percentile
                + 0.45 * candidate.history_hardness_percentile
            )
        else:
            candidate.complexity_score = candidate.patch_scope_percentile

    meta = {
        "candidate_count": len(candidates),
        "difficulty_filter": requested,
        "historical_submissions": submissions,
        "experiments_commit": experiments_sha,
        "history_error": history_error,
        "complexity_formula": (
            "0.55*patch_scope + 0.45*historical_hardness when history exists; "
            "patch_scope otherwise. patch_scope = "
            "0.60*changed_lines_percentile + 0.25*files_percentile + "
            "0.15*hunks_percentile."
        ),
    }
    return candidates, meta


def select_spread(
    candidates: list[Candidate],
    *,
    count: int = 3,
    diverse_repos: bool = True,
) -> list[Candidate]:
    if count < 1:
        raise ValueError("sample count must be at least 1")
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} candidates but only {len(candidates)} are available."
        )

    if count == 1:
        targets = [0.5]
    else:
        targets = [
            0.20 + index * (0.60 / (count - 1))
            for index in range(count)
        ]

    chosen: list[Candidate] = []
    used_ids: set[str] = set()
    used_repos: set[str] = set()

    labels = (
        ["low", "mid", "high"]
        if count == 3
        else [f"q{index + 1}" for index in range(count)]
    )

    for index, target in enumerate(targets):
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                abs(candidate.complexity_score - target),
                candidate.instance_id,
            ),
        )

        pick = None
        if diverse_repos:
            pick = next(
                (
                    candidate
                    for candidate in ranked
                    if candidate.instance_id not in used_ids
                    and candidate.repo not in used_repos
                ),
                None,
            )

        if pick is None:
            pick = next(
                candidate
                for candidate in ranked
                if candidate.instance_id not in used_ids
            )

        pick.selection_target = target
        pick.selection_rank = labels[index]
        chosen.append(pick)
        used_ids.add(pick.instance_id)
        used_repos.add(pick.repo)

    return chosen


def candidate_rows(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]
