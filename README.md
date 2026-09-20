# Forge Bench

Forge Bench is a reproducible benchmark harness for comparing AI coding-agent strategies on correctness, token use, cost, latency, and tool behavior.

The default experiment is now a clean 2x2 comparison of two behavioral
interventions while keeping the normal Hermes coding toolset constant:

- Baseline Hermes
- Caveman only
- Ponytail only
- Caveman + Ponytail

Lean tools remain available as an explicit optional treatment, along with the
legacy Caveman + Ponytail + Lean combination, but neither is part of the
default experiment. The first 15-run study showed that lean-tool pruning
reduced token use on average but increased wall-clock time and behaved
inconsistently across tasks, so it is being treated as a separate question
rather than a factor in the Caveman/Ponytail interaction test.

## Default SWE-bench experiment

The second-stage default uses **five deliberately homogeneous SWE-bench
Verified medium tasks**. The goal is low between-task variance in token use
and wall-clock time, not broad coverage of SWE-bench difficulty.

Two tasks are retained as anchors because the first experiment showed they are
small, historically high-solve tasks with useful baseline behavior. The pinned
selector then chose the three closest matches shown below:

| Role | Instance | Repository | Gold patch | Historical solve rate | Anchor distance |
| --- | --- | --- | ---: | ---: | ---: |
| Anchor 1 | `django__django-13516` | django/django | 4 lines, 1 file | 84.4% | 0.000 |
| Anchor 2 | `pytest-dev__pytest-7571` | pytest-dev/pytest | 4 lines, 1 file | 79.3% | 0.000 |
| Match 1 | `django__django-15731` | django/django | 4 lines, 1 file | 87.4% | 0.251 |
| Match 2 | `django__django-16662` | django/django | 5 lines, 1 file | 83.0% | 0.276 |
| Match 3 | `django__django-7530` | django/django | 2 lines, 1 file | 81.5% | 0.298 |

Forge Bench deterministically derives this panel from the pinned Verified
medium pool by similarity to the two anchors. CI also asserts the exact five
IDs so the experiment cannot silently drift. Eligible matches are
restricted to one-file fixes with 2–10 changed lines and a small number of
hunks. Among those, distance is computed from:

- gold-patch changed lines and hunks;
- gold-patch character size;
- FAIL_TO_PASS and PASS_TO_PASS counts;
- issue-statement length; and
- historical Verified solve rate when available.

Historical solve rate is kept within roughly 10 percentage points below the
anchor mean. Unlike the original smart sampler, repository diversity is **not**
forced for this default because similarity and variance reduction are the
primary design goals.

The exact selected five tasks and their anchor-distance values are written to
`selection.json`, `selection.csv`, and `candidate_pool.csv` before any
model calls. Because the dataset revision and historical-results source are
pinned, the selection is deterministic.

The default experiment is therefore **5 tasks × 4 treatments = 20 randomized
Hermes runs** per repeat.

### Optional broad within-bucket sampling

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

When `--smart-sample` is requested, Forge Bench uses the older broad-spread design, targeting composite scores from approximately 0.20 to 0.80 across the requested sample size. This is useful for generalization studies, but it is intentionally not the default for the low-variance Caveman/Ponytail experiment.

The historical signal is derived from public `results/results.json` files in the official `SWE-bench/experiments` repository at a recorded source revision. The exact candidate pool, selected instances, formula, source commit, and selected task features are saved with every benchmark.

For datasets without official difficulty or compatible historical results, Forge Bench falls back gracefully to the available patch-scope features rather than inventing missing information.

## Reproducibility pins

The default experiment currently pins:

- model: `deepseek/deepseek-v4-flash-0731`
- reasoning: `none` (runs reporting reasoning tokens are invalid)
- API aggregator: OpenRouter
- upstream provider: `relace`
- Hermes runtime: official `nousresearch/hermes-agent:latest` image, pulled once at benchmark start and resolved to its immutable image ID for all runs in that experiment
- Hermes maximum tool-loop iterations: 50
- SWE-bench evaluator: `swebench==4.1.0`
- SWE-bench Verified dataset revision: `78f471bf655a3137b2e8a75af1501690ec009ec3`
- SWE-bench experiments source: `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`
- Ponytail: `e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156`
- Caveman: `542442bab314973709f95b85b1ac0b3f6f5b5dc6`
- randomization seed: `260919`

Every individual observation starts from a fresh minimal Hermes home and a brand-new `docker run --rm` container. The profile contains only benchmark configuration, credentials, and the treatment being tested; user personalities, memories, hooks, project plugins, bundled skills, and prior sessions are not inherited. Treatment templates are installed once, then copied into a pristine per-run profile before the container starts.

Each agent run also gets a fresh self-contained repository at the requested SWE-bench base commit. The workspace has no network Git remote and does not contain the solution PR. Model fallbacks and compression are disabled, reasoning is explicitly off, and OpenRouter routing is restricted to the pinned upstream provider.

The exact selected SWE-bench row is written to a local JSON file after the agent finishes and passed to the official evaluator. This prevents grading from silently changing because a remote dataset branch moved.

## Install

Prerequisites:

- Python 3.12+
- `uv`
- `git`
- Docker
- a working OpenRouter credential in the normal Hermes `~/.hermes/.env` or `auth.json` (or `OPENROUTER_API_KEY` in the shell)

By default you do **not** need a host Hermes installation. Forge Bench pulls the official Hermes image, resolves the pulled image to an immutable Docker image ID, and starts a fresh container for every observation. The official SWE-bench evaluator also uses Docker, so the first run can require substantial image downloads and disk space.

A host Hermes installation remains available as an explicit fallback:

```bash
uv run forge-bench --hermes-runtime local
```

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

By default it prints the five-task homogeneous anchor neighborhood with:

- anchor/match role
- distance from the Django/pytest anchor profile
- historical solve rate
- gold-patch changed-line count
- files touched
- repository and instance ID

It saves `selection.json`, `selection.csv`, `candidate_pool.csv`, and `metadata.json` for the default anchored selection.

## Run the default benchmark

```bash
uv run forge-bench
```

The default is 5 selected tasks × 4 treatments = 20 randomized Hermes runs.

The four treatments all use the normal `hermes-cli` toolset. To explicitly
revisit lean-tool pruning, use for example:

```bash
uv run forge-bench --arms baseline lean_tools
```

Or include the legacy combined lean treatment:

```bash
uv run forge-bench --arms baseline caveman ponytail caveman_ponytail lean_tools all_three
```

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

Generate Hermes patches without SWE-bench grading:

```bash
uv run forge-bench --skip-evaluation
```

With the default Hermes runtime this still uses Docker for the fresh Hermes containers. To avoid Docker entirely for a local plumbing test, combine it with `--hermes-runtime local`. Skipped evaluation should not be used for correctness comparisons.

## Agent and evaluation separation

For each selected SWE-bench instance Forge Bench:

1. loads the task from the pinned SWE-bench dataset revision;
2. fetches only the exact `base_commit` into an isolated local repository cache;
3. creates a fresh self-contained workspace and removes its Git network remote;
4. copies the treatment template into a brand-new minimal Hermes home;
5. starts a new official Hermes Docker container with that profile and workspace mounted in;
6. gives Hermes only the issue statement, repository state, and treatment instructions, with reasoning disabled; the prompt blocks external solution sources and cross-run leakage but otherwise allows normal local Hermes workflow;
7. removes the Hermes container after the one-shot run;
8. captures the complete working tree relative to the base commit, including committed and untracked changes;
9. grades that patch with SWE-bench 4.1 against the exact frozen task row.

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
- PNG and SVG figures for tokens, cost, agent wall time, API calls, SWE-bench resolve rate
- one evidence directory per run containing the prompt, Hermes stdout/stderr, usage JSON, generated patch, official evaluation output, and final workspace

## Confidence intervals

The selected SWE-bench tasks are treated as the independent experimental units.

If repeats are requested, Forge Bench first averages repeated runs within each treatment × task, then computes a two-sided 95% Student-t confidence interval across task means.

With five selected tasks, the default second-stage experiment reduces the extreme
small-n uncertainty of the first three-task study, but the intervals remain
conservative. Repeats are still valuable because coding-agent trajectories are
stochastic; task homogeneity reduces between-task variance but does not remove
within-task run-to-run variation.
