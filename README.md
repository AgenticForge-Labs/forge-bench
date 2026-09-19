# Forge Bench

Forge Bench is a reproducible benchmark harness for comparing AI coding-agent strategies on correctness, token use, cost, latency, and tool behavior.

The default experiment is a full 2×2×2 factorial over three token-saving factors:

- Caveman: OFF / ON
- Ponytail: OFF / ON
- Lean toolset: OFF / ON

That produces eight Hermes treatments:

- Baseline Hermes
- Lean tools
- Caveman
- Caveman + Lean
- Ponytail
- Ponytail + Lean
- Caveman + Ponytail
- Caveman + Ponytail + Lean

When the lean factor is ON, Hermes is invoked with exactly `file,terminal,skills,code_execution`. When it is OFF, Hermes gets its normal CLI toolset. This lets Forge Bench measure the cost of advertising broad tools such as browser, web, memory, delegation, vision, computer use, cron jobs, and other irrelevant capabilities during coding work.

## Default SWE-bench experiment

Forge Bench now uses real SWE-bench repository tasks rather than one-line algorithm repairs.

The initial default suite is frozen to three SWE-bench Verified tasks, all from the official `15 min - 1 hour` (`medium`) difficulty bucket:

| Tier | Instance | Repository | Gold patch | Historical solve rate |
| --- | --- | --- | ---: | ---: |
| Low-1 | `django__django-13516` | django/django | 4 lines, 1 file | 84.4% |
| Low-2 | `pytest-dev__pytest-7571` | pytest-dev/pytest | 4 lines, 1 file, 3 hunks | 79.3% |
| Low-3 | `sympy__sympy-20154` | sympy/sympy | 23 lines, 1 file, 3 hunks | 78.5% |

These are intentionally all from the historically high-solve end of the official 15–60 minute bucket. The initial experiment is meant to compare token/cost efficiency on tasks that most capable agents can actually finish, while retaining some variation in patch scope. Freezing them makes repeated factorial comparisons directly comparable over time.

The default experiment is therefore 3 fixed tasks × 8 Hermes treatments = 24 randomized agent runs.

### Smart within-bucket sampling

Forge Bench deliberately does not pick the first three tasks or randomly sample three tasks and call them representative.

For every candidate in the requested difficulty bucket it calculates:

- gold-patch changed-line percentile
- gold-patch file-count percentile
- gold-patch hunk-count percentile
- historical solve fraction across compatible official SWE-bench Verified leaderboard submissions

The gold patch is used only by the benchmark controller to characterize task scope. It is never copied into the Hermes workspace or included in the prompt.

Patch scope is:

```text
0.60 * changed-lines percentile
+ 0.25 * files-touched percentile
+ 0.15 * diff-hunks percentile
```

When historical Verified results are available, composite complexity is:

```text
0.55 * patch-scope percentile
+ 0.45 * historical-hardness percentile
```

where historical hardness increases as historical solve rate falls.

When `--smart-sample` is requested, Forge Bench targets composite scores near 0.20, 0.50, and 0.80 for a three-task sample. This gives a low/middle/high spread while staying inside one official human difficulty category. Repository diversity is preferred so one codebase does not dominate the comparison.

The historical signal is derived from public `results/results.json` files in the official `SWE-bench/experiments` repository at a recorded source revision. The exact candidate pool, selected instances, formula, source commit, and selected task features are saved with every benchmark.

For datasets without official difficulty or compatible historical results, Forge Bench falls back gracefully to the available patch-scope features rather than inventing missing information.

## Reproducibility pins

The default experiment currently pins:

- model: `deepseek/deepseek-v4-flash-0731`
- API aggregator: OpenRouter
- upstream provider: `relace`
- SWE-bench experiments source: `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`
- Ponytail: `e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156`
- Caveman: `542442bab314973709f95b85b1ac0b3f6f5b5dc6`
- randomization seed: `260919`

Each treatment gets an isolated temporary Hermes home. Each agent run gets a fresh repository containing only the requested SWE-bench base commit; the workspace has no network Git remote and does not contain the solution PR.

Model fallbacks and smart model routing are disabled in benchmark profiles. OpenRouter routing is restricted to the pinned upstream provider.

## Install

Prerequisites:

- Python 3.12+
- `uv`
- `git`
- Hermes Agent available as `hermes`
- a working OpenRouter credential in the normal Hermes configuration
- Docker for official SWE-bench grading

The official SWE-bench evaluator is containerized. Its first use can require substantial downloads and disk space.

```bash
git clone https://github.com/AgenticForge-Labs/forge-bench.git
cd forge-bench
uv sync
```

## Inspect the sample first

This is the recommended first command. It makes no model calls and does not grade anything:

```bash
uv run forge-bench --selection-only
```

By default it prints the frozen three-task initial suite with:

- frozen low-range tier
- patch-scope percentile
- historical solve rate
- gold-patch changed-line count
- files touched
- repository and instance ID

It also saves `selection.json`, `selection.csv`, `candidate_pool.csv`, and `metadata.json`.

## Run the default benchmark

```bash
uv run forge-bench
```

The default is 3 selected tasks × 8 treatments = 24 randomized Hermes runs.

For repeated stochastic attempts:

```bash
uv run forge-bench --repeats 3
```

Repeats are averaged within task before across-task statistics are calculated.

## Other SWE-bench samples

Re-run the smart sampler within the default medium bucket:

```bash
uv run forge-bench --smart-sample --selection-only
```

Choose a different Verified difficulty bucket (these automatically use smart sampling because the frozen suite is medium-only):

```bash
uv run forge-bench --difficulty easy --selection-only
uv run forge-bench --difficulty hard --selection-only
uv run forge-bench --difficulty expert --selection-only
```

Take a larger spread sample:

```bash
uv run forge-bench --sample-size 5 --selection-only
```

Use SWE-bench Lite or the full text benchmark:

```bash
uv run forge-bench --dataset lite --difficulty all --sample-size 3 --selection-only
uv run forge-bench --dataset full --difficulty all --sample-size 3 --selection-only
```

An arbitrary Hugging Face dataset with standard SWE-bench fields can also be supplied as `--dataset owner/name`.

Run exact instances instead of sampling:

```bash
uv run forge-bench --instance-ids django__django-12345 sympy__sympy-12345
```

Disable the historical leaderboard signal and sample only from patch scope:

```bash
uv run forge-bench --no-history --selection-only
```

Generate Hermes patches without Docker grading:

```bash
uv run forge-bench --skip-evaluation
```

This is useful for plumbing tests, but it should not be used for correctness comparisons.

## Agent and evaluation separation

For each selected SWE-bench instance Forge Bench:

1. fetches only the exact `base_commit` into an isolated local repository cache;
2. creates a fresh workspace for the treatment;
3. removes the Git network remote;
4. gives Hermes only the issue statement, repository state, and treatment instructions;
5. records the resulting Git diff;
6. sends that diff to the official SWE-bench Docker evaluator.

The gold solution patch and test patch are never exposed to Hermes.

An unresolved task is still a valid efficiency observation. Its tokens, time, and cost remain in treatment averages. This avoids making an inefficient treatment look cheap merely because its expensive failures were discarded.

## Outputs

Each benchmark writes a timestamped directory under `benchmark-results/` containing:

- `selection.json` / `selection.csv` — exact selected tasks and their selection features
- `candidate_pool.csv` — all candidates considered by smart sampling
- `run_plan.csv` — exact randomized execution order
- `metadata.json` — dataset, source revisions, model/provider pins, formula, and settings
- `runs.csv` / `runs.json` — raw run-level results
- `task_summary.csv` — repeats averaged within treatment × task
- `summary.csv` — across-task treatment means and 95% confidence intervals
- `factorial_effects.csv` — marginal ON-vs-OFF effects for Caveman, Ponytail, and Lean tools
- `report.html`
- PNG and SVG figures for tokens, cost, agent wall time, API calls, SWE-bench resolve rate, and marginal token effects
- one evidence directory per run containing the prompt, Hermes stdout/stderr, usage JSON, generated patch, official evaluation output, and final workspace

## Confidence intervals

The selected SWE-bench tasks are treated as the independent experimental units.

If repeats are requested, Forge Bench first averages repeated runs within each treatment × task, then computes a two-sided 95% Student-t confidence interval across task means.

With only three selected tasks, the intervals are intentionally conservative and can be wide. The first experiment is meant to detect large, practically useful token/cost effects; a larger task sample should be used for precise effect-size estimates.
