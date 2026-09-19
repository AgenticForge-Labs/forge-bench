# Forge Bench

Forge Bench is a reproducible benchmark harness for comparing AI coding-agent strategies on correctness, token use, cost, latency, and tool behavior.

The default experiment compares four Hermes Agent treatments:

- Baseline Hermes
- Caveman only
- Ponytail only
- Caveman + Ponytail

Hermes' native tools, including `execute_code`, remain available normally in every treatment. Forge Bench does not force or suppress native tool selection; the treatments differ only by the added Caveman and/or Ponytail behavior.

## Default SWE-bench experiment

Forge Bench now uses real SWE-bench repository tasks rather than one-line algorithm repairs.

The default selection is:

- dataset: `SWE-bench/SWE-bench_Verified`
- official difficulty: `15 min - 1 hour` (`medium`)
- sample size: 3
- sampling: low / middle / high complexity within that difficulty bucket
- prefer three different repositories
- 4 Hermes treatments per selected task
- 12 randomized agent runs total

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

For a three-task sample, Forge Bench targets composite scores near 0.20, 0.50, and 0.80. This gives a low/middle/high spread while staying inside one official human difficulty category. Repository diversity is preferred so one codebase does not dominate the comparison.

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

It prints the three selected instances with:

- low / mid / high tier
- composite complexity
- historical solve rate
- gold-patch changed-line count
- files touched
- repository and instance ID

It also saves `selection.json`, `selection.csv`, `candidate_pool.csv`, and `metadata.json`.

## Run the default benchmark

```bash
uv run forge-bench
```

The default is 3 selected tasks × 4 treatments = 12 randomized Hermes runs.

For repeated stochastic attempts:

```bash
uv run forge-bench --repeats 3
```

Repeats are averaged within task before across-task statistics are calculated.

## Other SWE-bench samples

Choose a different Verified difficulty bucket:

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
- `report.html`
- PNG and SVG figures for tokens, cost, agent wall time, API calls, and SWE-bench resolve rate
- one evidence directory per run containing the prompt, Hermes stdout/stderr, usage JSON, generated patch, official evaluation output, and final workspace

## Confidence intervals

The selected SWE-bench tasks are treated as the independent experimental units.

If repeats are requested, Forge Bench first averages repeated runs within each treatment × task, then computes a two-sided 95% Student-t confidence interval across task means.

With only three selected tasks, the intervals are intentionally conservative and can be wide. The first experiment is meant to detect large, practically useful token/cost effects; a larger task sample should be used for precise effect-size estimates.
