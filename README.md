# Forge Bench

Forge Bench is a basic, reproducible tool for comparing AI coding-agent harnesses, models, and providers. It uses experimental design and trace-level data science to understand long-running agent runs: how interventions change behavior, where tokens and time go, and which pain points are worth optimizing.

The project is intended to grow beyond this first study. Future work can add models, providers, tasks, harnesses, and broader execution environments such as Harbor. The goal is to test community assumptions with pinned configurations, raw traces, and analysis code that researchers can inspect, reproduce, and challenge.

## Why test token-saving tools?

People want coding agents to finish useful work with fewer tokens, lower cost, and less waiting. [Caveman](https://github.com/JuliusBrussee/caveman/tree/542442bab314973709f95b85b1ac0b3f6f5b5dc6/skills/caveman) asks agents to communicate more directly and avoid unnecessary response tokens. [Ponytail](https://github.com/DietrichGebert/ponytail/tree/e3ba2aa6f1e6f0bc4d69eb09c9f0d0a93af56156) encourages reuse, native features, and the smallest solution that meets the task.

Ponytail explicitly suggests using these approaches together. That creates a reasonable experimental hypothesis: if each intervention reduces work, combining them should deliver additive or compounding savings. Forge Bench was built to test assumptions like this instead of accepting them from a tool description or a single successful run.

## Featured study

This release combines two five-task SWE-bench Verified studies of a last-generation model and its current-generation successor under four harness conditions: Baseline Hermes, Caveman, Ponytail, and Caveman + Ponytail. The release bundle retains run records and raw traces with model, provider, treatment, task, and source-study identity.

> The short preview: the tools do not save work uniformly, their benefits do not reliably add, and the current-generation model is economical at baseline for a reason that is easy to miss from headline token prices.

The complete reproducible data package is in [studies/last-generation-current-generation-tool-interactions](studies/last-generation-current-generation-tool-interactions/).

## The first result: newer is cheaper at baseline; older wins with Ponytail

The current-generation V4.1 baseline is already cost-effective in this cache-heavy workload. Its non-cached tokens cost more, but cache reads are much cheaper, so its baseline cost is lower than the last-generation V4 baseline.

The unexpected result comes when the tools enter: Ponytail lowers V4 token use and tool calls, while V4.1 uses more of both with Ponytail. Caveman + Ponytail also does not show a reliable additive gain. The plot puts that interaction first.

![Model treatment interaction](studies/last-generation-current-generation-tool-interactions/combined_interactions.png)

This may reflect a model × tool interaction, provider routing, or differences in trace and task handling. V4 used Relace in both studies; V4.1 used Relace in the earlier study and DeepSeek in the later study. Community validation is requested: can others reproduce the pattern with the same tasks and pinned providers, and can trace inspection explain the model-specific behavior?

## Focused case study: an efficient last-generation configuration

The focused comparison asks how the current-generation baseline compares with the last-generation baseline and the last-generation model plus Ponytail.

- The current-generation baseline is already cost-effective. Its non-cached token prices are higher, but cache reads are much cheaper; this cache-heavy workload makes its baseline cost lower than the last-generation baseline.
- Adding Ponytail to the last-generation model goes further: it costs about **48% less** and takes about **32% less wall time** than the last-generation baseline while using substantially fewer tokens.

The current generation is effective on its own yet less compatible with this intervention. That interaction is more useful than a simple old-versus-new ranking.

![Focused configuration comparison](studies/last-generation-current-generation-tool-interactions/focused_configuration_comparison.png)

### Current-study PCA and raw trace trajectories

PCA uses the current study only, combining models and treatments. Color encodes treatment, point shape encodes model, and orange arrows show the biplot loading directions for standardized features. The trajectory figure is derived from the merged API, tool, and Hermes observer events.

![Shared PCA](studies/last-generation-current-generation-tool-interactions/combined_shared_pca.png)

![Combined raw trace trajectories](studies/last-generation-current-generation-tool-interactions/combined_trace_trajectories.png)

## Experimental design and pooled analysis

The two five-task studies are complementary randomized blocks. We merge their run records and raw traces while retaining study ID, provider, model, treatment, task, and source path for every row.

For interaction and focused comparisons, each usage outcome uses a log1p mixed-effects model with study and task random intercepts, then transforms estimates back to original units. The interaction figure omits intervals for readability; the accompanying CSV files retain estimates and uncertainty.

The model pools descriptive results but cannot separate the V4.1 provider change from the study change, and two study blocks are too few to estimate study-level variation precisely. That limitation is itself a reason to test provider × model and provider × tool interactions directly.

## Install and use Forge Bench

Forge Bench installs as a normal Python package and provides the `forge-bench` command:

```bash
git clone https://github.com/AgenticForge-Labs/forge-bench.git
cd forge-bench
uv sync
uv run forge-bench --check-credentials
```

Plan a design without making model calls, then run it into a named output directory:

```bash
uv run forge-bench --design designs/your-study.yaml --plan-only
uv run forge-bench --design designs/your-study.yaml --output benchmark-results/my-study
```

## Data and reproducibility

The [study package](studies/last-generation-current-generation-tool-interactions/) contains both pinned YAML designs, 80 joined run records, raw merged JSONL traces, source checksums, derived CSV tables, the analysis script, and only the five PNGs shown in this README. It excludes the original 2.6 GB execution workspace, Hermes state databases, evaluation logs, duplicate reports, SVG versions, and unused figures.

Rebuild the tables and figures from the published aggregate data:

```bash
.venv/bin/python studies/last-generation-current-generation-tool-interactions/analysis.py
```

The analysis is descriptive. There is one run per model × treatment × task cell in each study, and the upstream provider changed for V4.1 between studies.
