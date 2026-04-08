# LLM Actor/Distinguisher Experiments

This repository runs controlled actor/distinguisher experiments for large language models. The core question is: can one model imitate another model's conversational style well enough to fool a fresh instance of that target model?

The codebase is organized around a single modern experiment path:

- [run_experiment.py](run_experiment.py): run one actor/distinguisher pair for `N` trials under one condition
- [run_experiment_matrix.py](run_experiment_matrix.py): sweep many actor/distinguisher pairings
- [scripts/render_results_summary.py](scripts/render_results_summary.py): aggregate `results.csv` files into summary CSVs and PNGs
- [scripts/verify_model_slugs.py](scripts/verify_model_slugs.py): validate current OpenRouter model IDs against the live catalog

The older scripts [run_hf_experiment.py](run_hf_experiment.py) and [run_openrouter_single_stage_experiment.py](run_openrouter_single_stage_experiment.py) are legacy references. The main experiment workflow in this repo is the `run_experiment.py` / `run_experiment_matrix.py` path.

## Goal

The experiment compares two roles:

- `actor`: tries to imitate the target model
- `distinguisher`: interrogates the actor and decides whether it is talking to itself

The final scientific signal is the distinguisher's parsed `<answer>`:

- `1`: "I think I am talking to the same model"
- `0`: "I think I am talking to a different model"

The `success` field in result CSVs does not mean "the actor fooled the distinguisher." It only means the run completed cleanly enough to produce a valid parsed answer.

## Motivation

This setup is useful for:

- measuring how recognizable a model's conversational style is
- testing whether weaker or cheaper models can impersonate stronger ones
- comparing direct interrogation against a two-stage "query the specimen first" protocol
- studying which model families are easy or hard to distinguish
- generating reproducible transcripts with environment metadata for later analysis

## Experiment Types

The repo currently supports three experimental conditions from [turing_eval/conditions.py](turing_eval/conditions.py):

### 1. `no_query_no_explanation`

The distinguisher opens immediately. The actor receives an imitation prompt and must respond without any prior specimen interaction.

Use this when you want the cleanest direct imitation test.

### 2. `yes_query_no_explanation`

The actor first talks to a specimen instance of the target model, then the real distinguisher starts a fresh interrogation.

Use this when you want to test whether a short warm-up interaction improves mimicry.

### 3. `no_query_yes_explanation`

This is the same direct interrogation setup as condition 1, but the distinguisher is asked to include explicit reasoning in its final response.

Use this when you care about both the answer and the distinguisher's rationale.

## Routing Policy

The intended runtime policy in this repo is:

1. Use OpenRouter when the model is available there.
2. Use local Hugging Face inference as the fallback path.
3. If a model is not available on OpenRouter, run it through the HF backend on a GPU node.

That policy is implemented in the model registry at [turing_eval/models.py](turing_eval/models.py). Most models use `primary_backend="api"` with OpenRouter as the primary route and HF as the fallback route. Models that are not presently routed through OpenRouter in this repo use HF directly.

Important nuance: some registry nicknames intentionally point to the nearest currently routed OpenRouter slug rather than an exact family/version match. For example:

- `qwen2.5-7b` uses OpenRouter `qwen/qwen3-8b`, with exact HF fallback `Qwen/Qwen2.5-7B-Instruct`
- `qwen2.5-32b` uses OpenRouter `qwen/qwen3-32b`, with exact HF fallback `Qwen/Qwen2.5-32B-Instruct`
- `mistral-7b-v0.3` uses OpenRouter `mistralai/mistral-7b-instruct-v0.1`, with exact HF fallback `mistralai/Mistral-7B-Instruct-v0.3`

## Model Tiers

The repo can run the following model nicknames today. The tiers below are the recommended conceptual grouping for experiments.

### Tier 0: Frontier closed

Use these as the strongest "gold-standard" distinguishers or actors via OpenRouter.

- `gpt-5.4`
- `gemini-3-pro`
- `claude-opus-4.6`

Also registered:

- `claude-sonnet-4.6`

### Tier 1: Frontier open

These are the largest open-weight or open-ish frontier models in the current registry.

- `glm-5-reasoning`
- `deepseek-v3.2`
- `kimi-k2.5`
- `llama-3.3-70b`

Recommended policy:

- default to OpenRouter
- use HF only when you explicitly want a local run and the model has an HF fallback configured

### Tier 2: Mid

These fit well on serious lab GPUs and are good for controlled local runs.

- `qwen3-30b-a3b`
- `qwen2.5-32b`
- `exaone-4-32b`
- `gemma-3-27b`

Recommended policy:

- default to OpenRouter when available
- prefer HF on the cluster when you want reproducibility, exact sampling control, or high trial counts
- `exaone-4-32b` is currently configured as HF-local in this repo

### Tier 3: Weak baseline

These are the cheap baselines and easiest local models to run repeatedly.

- `gemma-3-12b`
- `qwen2.5-7b`
- `llama-3.1-8b`
- `phi-4`
- `mistral-7b-v0.3`

Note: the current registry exposes `llama-3.1-8b`, not `llama-3.2-8b`.

## Environment

This cluster uses Apptainer, not Docker.

For the API-only workflow:

- build with [scripts/build-container.sh](scripts/build-container.sh)
- run commands with [scripts/run-in-container.sh](scripts/run-in-container.sh)

For the HF workflow:

- build with [scripts/build-container-hf.sh](scripts/build-container-hf.sh)
- run commands with [scripts/run-in-container-hf.sh](scripts/run-in-container-hf.sh)

API keys are forwarded automatically by the container wrapper when present in your shell or `.env`.

Supported provider environment variables:

- `OPENROUTER_API_KEY`
- `DEEPSEEK_API_KEY`
- `MOONSHOT_API_KEY`
- `TOGETHER_API_KEY`
- `GROQ_API_KEY`
- `MISTRAL_API_KEY`
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated HF repos

## Quick Start

Build the default image:

```bash
./scripts/build-container.sh
```

Check the test suite:

```bash
./scripts/run-in-container.sh pytest tests/ -v
```

Validate OpenRouter slugs before a long API run:

```bash
./scripts/run-in-container.sh python -m scripts.verify_model_slugs
```

## Running Single-Pair Experiments

### Default behavior: let the registry choose

This uses the model registry's primary route, which is usually OpenRouter.

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor qwen2.5-7b \
  --distinguisher mistral-7b-v0.3 \
  --condition no_query_no_explanation \
  --num-trials 5 \
  --turn-limit 20 \
  --output-dir results
```

### Force API on both sides

Use this when you want to guarantee an OpenRouter-backed run.

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor deepseek-v3.2 \
  --actor-backend api \
  --distinguisher glm-5-reasoning \
  --distinguisher-backend api \
  --condition no_query_yes_explanation \
  --num-trials 3 \
  --turn-limit 25 \
  --output-dir results
```

### Force HF on one or both sides

Use the HF image and wrapper for local inference.

```bash
./scripts/run-in-container-hf.sh python run_experiment.py \
  --actor qwen2.5-7b \
  --actor-backend hf \
  --distinguisher mistral-7b-v0.3 \
  --distinguisher-backend hf \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --turn-limit 12 \
  --output-dir results
```

### Mixed API/HF runs

This is often the most useful comparison setup.

```bash
./scripts/run-in-container-hf.sh python run_experiment.py \
  --actor qwen2.5-7b \
  --actor-backend hf \
  --distinguisher claude-opus-4.6 \
  --distinguisher-backend api \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --turn-limit 12 \
  --output-dir results
```

### Condition B with specimen warm-up

This is the two-stage imitation condition.

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor gemma-3-12b \
  --distinguisher qwen2.5-7b \
  --condition yes_query_no_explanation \
  --num-trials 3 \
  --turn-limit 15 \
  --specimen-turn-limit 8 \
  --output-dir results
```

## Running Matrix Experiments

Sweep every actor/distinguisher pairing from a short list:

```bash
./scripts/run-in-container.sh python run_experiment_matrix.py \
  --models mistral-7b-v0.3 qwen2.5-7b claude-opus-4.6 \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --output-dir results/matrix_smoke
```

Include self-matches if you want the full square matrix:

```bash
./scripts/run-in-container.sh python run_experiment_matrix.py \
  --models gpt-5.4 gemini-3-pro claude-opus-4.6 \
  --condition no_query_yes_explanation \
  --num-trials 1 \
  --include-self-match \
  --output-dir results/frontier_matrix
```

## Running HF Jobs on a GPU Node

Do not expect meaningful HF inference from a login node. Use Slurm.

Build the HF image:

```bash
./scripts/build-container-hf.sh
```

Then submit a GPU batch job that uses the HF wrapper:

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=llm-eval-hf
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
cd /path/to/llm-eval

./scripts/run-in-container-hf.sh python run_experiment.py \
  --actor qwen2.5-7b \
  --actor-backend hf \
  --distinguisher mistral-7b-v0.3 \
  --distinguisher-backend api \
  --condition no_query_no_explanation \
  --num-trials 1 \
  --turn-limit 10 \
  --max-new-tokens 64 \
  --output-dir artifacts/hf-slurm-smoke \
  --run-name qwen_hf_actor_vs_mistral_api_smoke
EOF
```

Use this path when:

- the model is not available on OpenRouter
- you want exact local HF weights rather than the provider-routed API version
- you need higher throughput on repeated weak-baseline or mid-tier experiments

## Result Artifacts

Each run writes a directory containing:

- `results.csv`
- one `trial_XXX.json` file per trial

The transcript JSON includes:

- actor and distinguisher names
- raw conversation turns
- condition name
- parsed answer and optional reasoning
- environment metadata such as git commit and container SHA

## Aggregating Results

Render all discovered `results.csv` files into a summary CSV and PNG table:

```bash
./scripts/run-in-container.sh env MPLCONFIGDIR=/tmp/matplotlib \
  python scripts/render_results_summary.py \
  --input-root results \
  --output-dir artifacts/results-summary
```

To combine standard runs with tracked Slurm artifacts:

```bash
./scripts/run-in-container.sh env MPLCONFIGDIR=/tmp/matplotlib \
  python scripts/render_results_summary.py \
  --input-root results \
  --input-root artifacts/hf-slurm-smoke \
  --output-dir artifacts/results-summary
```

This writes:

- `artifacts/results-summary/summary.csv`
- `artifacts/results-summary/summary_table.png`
- one per-condition PNG heatmap

## Choosing Experiments

Good first runs:

- Tier 3 vs Tier 3 under `no_query_no_explanation`
- Tier 3 actor vs Tier 0 distinguisher under `no_query_no_explanation`
- Tier 3 actor vs Tier 3 distinguisher under `yes_query_no_explanation`

Good scaling runs:

- Tier 2 matrix under `no_query_no_explanation`
- Tier 0 vs Tier 1 cross-family comparisons
- API vs HF comparisons for the same nickname where the OpenRouter and HF routes differ

Good validation runs:

- exact HF fallback vs OpenRouter route for `qwen2.5-7b`
- exact HF fallback vs OpenRouter route for `mistral-7b-v0.3`
- OpenRouter slug validation before every large API sweep

## Practical Notes

- Start with `--num-trials 1` and a small `--turn-limit`.
- Use OpenRouter by default for convenience and reach.
- Use HF on a GPU node when OpenRouter does not have the model, or when local reproducibility matters more than convenience.
- Use the HF wrapper for any run with `--actor-backend hf` or `--distinguisher-backend hf`.
- Public HF repos do not require a token; gated repos do.
- For tracked summaries, keep PNG outputs under `artifacts/results-summary/` because the live `results/` tree may be ignored or symlinked outside the repo.
