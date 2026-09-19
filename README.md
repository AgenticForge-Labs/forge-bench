# Forge Bench

Forge Bench is a reproducible benchmark harness for comparing AI coding-agent strategies on correctness, token use, cost, latency, and tool behavior.

The first experiment compares four Hermes Agent treatments on three pinned QuixBugs program-repair tasks:

- Baseline Hermes
- Caveman only
- Ponytail only
- Caveman + Ponytail

Hermes' native tools, including `execute_code`, remain available normally in every treatment. Forge Bench does not force or suppress native tool selection; the treatments differ only by the added Caveman and/or Ponytail behavior.

## Reproducibility pins

The default experiment pins:

- Model: `deepseek/deepseek-v4-flash-0731`
- API aggregator: OpenRouter
- Upstream provider: `baidu` (Baidu Qianfan)
- QuixBugs commit: `4257f44b0ff1181dedaedee6a447e133219fcebf`
- Ponytail commit: `e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156`
- Caveman commit: `542442bab314973709f95b85b1ac0b3f6f5b5dc6`
- Randomization seed: `260919`

Forge Bench creates a fresh temporary Hermes home and a fresh Git workspace for each treatment. Model fallbacks and smart model routing are disabled for the benchmark profiles. OpenRouter provider routing is restricted to the pinned upstream provider.

The default upstream is Baidu Qianfan because the pinned DeepSeek endpoint is inexpensive, supports tools/tool choice/reasoning parameters, and was showing very high uptime when this benchmark was created. OpenRouter currently marks Baidu Qianfan as retaining prompts, so the default suite intentionally uses only public QuixBugs fixtures rather than private project code. Change the upstream before adapting Forge Bench to sensitive repositories.

## Run

Prerequisites:

- Python 3.12+
- `uv`
- `git`
- Hermes Agent installed and available as `hermes`
- a working OpenRouter credential in your normal Hermes configuration

Then:

```bash
git clone https://github.com/AgenticForge-Labs/forge-bench.git
cd forge-bench
uv sync
uv run forge-bench
```

The default run is 3 tasks × 4 treatments = 12 Hermes runs.

For repeated stochastic runs:

```bash
uv run forge-bench --repeats 3
```


## Outputs

Each run writes a timestamped directory under `benchmark-results/` containing:

- `report.html`
- `summary.csv` — treatment means and 95% confidence intervals across task means
- `task_summary.csv` — per-task treatment means, averaging repeats within task
- `runs.csv` and `runs.json` — raw run-level measurements
- `run_plan.csv` — the randomized execution order
- `metadata.json` — exact pins and benchmark configuration
- PNG and SVG figures for total tokens, input tokens, output tokens, cost, wall time, API calls, and valid repair rate
- one evidence directory per run with prompt, Hermes output, usage JSON, verifier output, and final workspace

### Confidence intervals

The primary comparison treats the three benchmark tasks as the independent units. If repeats are requested, Forge Bench first averages repeats within each task and treatment, then computes the treatment mean and a two-sided 95% Student-t confidence interval across task means.

With only three tasks, those intervals are intentionally conservative and can be wide. They are useful for this small experiment but should not be interpreted as a precise population estimate.

## Correctness guards

A run is counted as valid only if:

1. the original pinned QuixBugs vectors pass,
2. the verifier and test vectors are byte-identical to their pre-run hashes,
3. Hermes completes successfully,
4. the requested OpenRouter model route matches.

Correct QuixBugs implementations are never placed in the agent workspace.
