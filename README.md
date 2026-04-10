# LLM Actor/Distinguisher Experiments

This repository runs controlled actor/distinguisher experiments for large language models. The core question is: can one model imitate another model's conversational style well enough to fool a fresh instance of that target model?

The codebase is organized around a single modern experiment path:

- [run_experiment.py](run_experiment.py): run one actor/distinguisher pair for `N` trials under one condition
- [run_experiment_matrix.py](run_experiment_matrix.py): sweep many actor/distinguisher pairings
- [scripts/render_results_summary.py](scripts/render_results_summary.py): aggregate `results.csv` files into summary CSVs and PNGs
- [scripts/verify_model_slugs.py](scripts/verify_model_slugs.py): validate current OpenRouter model IDs against the live catalog

The older script [run_openrouter_single_stage_experiment.py](run_openrouter_single_stage_experiment.py) is retained only as a historical reference. The maintained experiment workflow in this repo is the `run_experiment.py` / `run_experiment_matrix.py` path.

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

Terminology:

- `specimen`: a separate helper instance of the target model used only in the warm-up stage of `yes_query_no_explanation`

The specimen is not the final judge. It is a same-model exemplar that the actor can interact with briefly in order to pick up style, phrasing, and conversational habits before facing the real distinguisher.

### 1. `no_query_no_explanation`

The distinguisher opens immediately. The actor receives an imitation prompt and must respond without any prior specimen interaction.

Use this when you want the cleanest direct imitation test.

### 2. `yes_query_no_explanation`

The actor first talks to a specimen instance of the target model, then the real distinguisher starts a fresh interrogation.

In other words, if the distinguisher model is `qwen2.5-7b`, the specimen is another `qwen2.5-7b` instance used for a short practice conversation. After that warm-up ends, a fresh `qwen2.5-7b` distinguisher begins the actual test.

Use this when you want to test whether a short warm-up interaction improves mimicry.

### 3. `no_query_yes_explanation`

This is the same direct interrogation setup as condition 1, but the distinguisher is asked to include explicit reasoning in its final response.

Use this when you care about both the answer and the distinguisher's rationale.

## Prompt Reference

All maintained prompt templates live in [turing_eval/prompts.py](turing_eval/prompts.py).

Condition A, `no_query_no_explanation`:

- actor prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `ACTOR_PROMPT_NO_QUERY_NO_EXPLANATION`
- distinguisher prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `DISTINGUISHER_PROMPT_NO_EXPLANATION`
- builder functions used at runtime: [turing_eval/prompts.py](turing_eval/prompts.py) `build_actor_prompt_no_query(...)` and `build_distinguisher_prompt(with_reasoning=False)`

Condition B, `yes_query_no_explanation`:

- actor prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `ACTOR_PROMPT_YES_QUERY_NO_EXPLANATION`
- distinguisher prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `DISTINGUISHER_PROMPT_NO_EXPLANATION`
- specimen prompt: none; the specimen receives no initial prompt
- builder functions used at runtime: [turing_eval/prompts.py](turing_eval/prompts.py) `build_actor_prompt_yes_query(...)` and `build_distinguisher_prompt(with_reasoning=False)`

Condition C, `no_query_yes_explanation`:

- actor prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `ACTOR_PROMPT_NO_QUERY_NO_EXPLANATION`
- distinguisher prompt template: [turing_eval/prompts.py](turing_eval/prompts.py) `DISTINGUISHER_PROMPT_YES_EXPLANATION`
- builder functions used at runtime: [turing_eval/prompts.py](turing_eval/prompts.py) `build_actor_prompt_no_query(...)` and `build_distinguisher_prompt(with_reasoning=True)`

If you want the exact prompt text that was actually used in one trial, open that trial's `trial_XXX.json` and inspect `actor_prompt`, plus the distinguisher-side initial `user` message under `distinguisher_messages`.

## Routing Policy

The intended runtime policy in this repo is:

1. Use OpenRouter by default when the model is available there.
2. For GPT-family models that define a direct OpenAI route, you can opt into OpenAI per role.
3. Use local Hugging Face inference as the fallback path.
4. If a model is not available on OpenRouter, run it through the HF backend on a GPU node.

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
- `gpt-5.4-pro`
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
- `OPENAI_API_KEY`
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

### Opt into direct OpenAI for GPT models

Keep the API path, but route a GPT role directly through OpenAI instead of OpenRouter.

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor gpt-5.4 \
  --actor-api-provider openai \
  --distinguisher gpt-5.4-pro \
  --distinguisher-api-provider openai \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --turn-limit 12 \
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

Here, "specimen" means a same-model exemplar of the distinguisher. The actor talks to that exemplar first, then a fresh distinguisher of the same model family runs the real evaluation.

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

What these parameters mean:

- `--actor gemma-3-12b`: the model that tries to imitate the target model
- `--distinguisher qwen2.5-7b`: the model that interrogates the actor and returns the final `<answer>`
- `--condition yes_query_no_explanation`: use the two-stage setup where the actor first talks to a specimen instance of the distinguisher model, then faces a fresh distinguisher
- `--num-trials 3`: run three independent trials of the same actor/distinguisher/condition setup
- `--turn-limit 15`: cap the second-stage interrogation loop at 15 turns after the distinguisher starts
- `--specimen-turn-limit 8`: cap the specimen warm-up loop at 8 turns before the real distinguisher begins
- `--output-dir results`: write a timestamped run directory under `results/`, containing `results.csv` and one `trial_XXX.json` transcript per trial

In practical terms, this command asks:

1. Can `gemma-3-12b` learn the style of `qwen2.5-7b` from a short specimen conversation?
2. After that warm-up, can it fool a fresh `qwen2.5-7b` distinguisher?
3. Does that happen consistently across three trials?

### Common `run_experiment.py` flags

- `--actor`: actor model nickname from the registry
- `--distinguisher`: distinguisher model nickname from the registry
- `--condition`: one of `no_query_no_explanation`, `yes_query_no_explanation`, or `no_query_yes_explanation`
- `--num-trials`: number of repeated trials for the same setup
- `--turn-limit`: maximum number of interrogation turns in the main actor/distinguisher interaction
- `--specimen-turn-limit`: only used for `yes_query_no_explanation`; limits the warm-up conversation with the specimen
- `--output-dir`: parent directory for run outputs
- `--run-name`: optional explicit name for the run directory; otherwise the script auto-generates one
- `--actor-backend`: force `api` or `hf` for the actor instead of using the registry default
- `--distinguisher-backend`: force `api` or `hf` for the distinguisher instead of using the registry default
- `--actor-api-provider`: when using the API path, force a specific provider such as `openrouter` or `openai`
- `--distinguisher-api-provider`: same override for the distinguisher, and for the specimen in condition B
- `--max-new-tokens`: generation cap per model response
- `--temperature`: optional sampling temperature shared by both roles
- `--verbose`: print the conversation live to stdout while the trial runs

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

Every `run_experiment.py` invocation writes one run directory under the parent you pass with `--output-dir`.

If you do not set `--run-name`, the directory name is generated as:

```text
<actor>_vs_<distinguisher>_<condition>_<utc-timestamp>
```

Example:

```text
results/gemma-3-12b_vs_qwen2.5-7b_yes_query_no_explanation_20260408-195500/
```

That run directory contains:

- `results.csv`
- one `trial_XXX.json` file per trial

So with `--num-trials 3`, you should expect:

```text
results/<run-name>/
  results.csv
  trial_000.json
  trial_001.json
  trial_002.json
```

For `run_experiment_matrix.py`, the `--output-dir` becomes the parent folder for many such run directories, one per actor/distinguisher pairing.

### `results.csv`

Each run directory has exactly one `results.csv`. It contains one row per trial.

Saved columns:

- `trial_id`: integer trial index within that run
- `actor_model`: resolved display name of the actor model
- `distinguisher_model`: resolved display name of the distinguisher model
- `success`: whether the run completed with a valid parsed `<answer>`
- `turns`: number of turns in the main interrogation loop
- `answer`: parsed final answer, usually `0` or `1`
- `transcript_path`: path to the matching `trial_XXX.json`
- `condition`: experiment condition name
- `reasoning`: parsed distinguisher reasoning when available
- `specimen_turns`: number of specimen warm-up turns for condition `yes_query_no_explanation`

Important interpretation notes:

- `success=True` does not mean the actor fooled the distinguisher
- `answer=1` is the actual "same model" signal
- `answer=0` is the actual "different model" signal
- `reasoning` is generally populated only for `no_query_yes_explanation`
- `specimen_turns` is generally populated only for `yes_query_no_explanation`

### `trial_XXX.json`

Yes, the conversations are saved. The JSON transcript is the full per-trial record and is the most important artifact for later inspection.

Every transcript JSON contains these common top-level fields:

- actor and distinguisher names
- `trial_id`
- `condition`
- `actor_model`
- `distinguisher_model`
- `success`
- `turns`
- `answer`
- optional `reasoning`
- `first_message`: the distinguisher's opening message in the interrogation stage
- `actor_prompt`: the initial prompt used to steer the actor
- `raw_final_message`: the final raw message from the distinguisher, including `<answer>` and optional `<reasoning>`
- `environment`: reproducibility metadata captured at save time

The `environment` block currently includes:

- `python_version`
- `python_implementation`
- `platform`
- `git_commit`
- `git_branch`
- `git_dirty`
- `apptainer_container_path`
- `apptainer_container_sha256`
- `docker_image`
- `hostname`

### Saved conversations by condition

The exact conversation fields in `trial_XXX.json` depend on the condition.

#### Condition `no_query_no_explanation`

Saved conversation fields:

- `actor_messages`
- `distinguisher_messages`

What this means:

- the full actor-side message history is saved
- the full distinguisher-side message history is saved
- there is no specimen conversation in this condition
- there is no explicit reasoning field unless the final message happens to include one

#### Condition `no_query_yes_explanation`

Saved conversation fields:

- `actor_messages`
- `distinguisher_messages`
- `reasoning`

What this means:

- the full conversation is saved exactly as in the no-query condition
- the distinguisher is instructed to provide an explanation
- the parsed explanation is saved separately in `reasoning`
- the raw final message is also preserved in case you want to inspect parsing behavior

#### Condition `yes_query_no_explanation`

Saved conversation fields:

- `stage1_actor_messages`
- `specimen_messages`
- `stage2_actor_messages`
- `distinguisher_messages`
- `specimen_turns`
- `specimen_stopped_cleanly`

What this means:

- the warm-up conversation with the specimen is saved
- the later interrogation with the fresh distinguisher is also saved
- the actor history is split into stage 1 and stage 2 so you can analyze transfer effects
- `specimen_stopped_cleanly` tells you whether the actor ended the specimen phase by emitting `STOP`

### What is saved on failures

If a trial crashes, the script still writes a transcript JSON and a CSV row for that trial.

In that case you should expect:

- `success=False`
- `answer=None`
- `turns=None` in some crash paths
- an `error` field in the JSON transcript

This is useful because failed trials remain visible in aggregate outputs instead of silently disappearing.

### What is not saved

The repo does not currently save:

- token counts
- request latency per turn
- provider-side raw HTTP payloads
- logits or probabilities
- GPU memory usage
- intermediate hidden states

If you need those, the persistence layer would need to be extended.

## How To Interpret A Result

There are two levels of interpretation:

- per-trial interpretation from `trial_XXX.json`
- aggregate interpretation from `results.csv` and `artifacts/results-summary/summary.csv`

### Per-trial interpretation

Use the tracked HF smoke artifact as a concrete example:

- CSV row: [artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/results.csv](/home/erimoldi/projects/llm-eval/artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/results.csv)
- transcript: [artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/trial_000.json](/home/erimoldi/projects/llm-eval/artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/trial_000.json)

That trial records:

- `actor_model = qwen/qwen3-8b`
- `distinguisher_model = mistralai/mistral-7b-instruct-v0.1`
- `condition = no_query_no_explanation`
- `success = false`
- `turns = 2`
- `answer = null`

How to read that:

- the run did not crash outright, because a full transcript was saved
- the conversation lasted 2 interrogation turns
- the distinguisher never produced a valid `<answer>0</answer>` or `<answer>1</answer>` tag
- because no valid final answer was emitted, `success` is `false`
- because there was no valid parsed answer, `answer` is `null`

So this trial is not evidence that the actor fooled the distinguisher. It is an incomplete trial that should be counted as a completion failure, not as either a win or a loss for the actor.

When reading the transcript itself:

- `first_message` shows how the interrogation started
- `actor_prompt` shows exactly how the actor was instructed to imitate the target
- `actor_messages` and `distinguisher_messages` let you inspect the full exchange turn by turn
- `raw_final_message` shows the actual last distinguisher message, which in this case was plain text rather than a tagged final decision
- `environment` tells you which code revision, container image, and host produced the artifact

### Sample conversation excerpt

You do not need to read an entire transcript to understand the setup. This shortened excerpt from [artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/trial_000.json](/home/erimoldi/projects/llm-eval/artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/trial_000.json) shows the key structure:

```text
Distinguisher opens:
"Hello, could you please share the model name and version number you were built on?"

Actor, after receiving its imitation prompt, answers as the target:
"Hello! I was built on the model type mistralai/mistral-7b-instruct-v0.1."

Distinguisher continues probing:
"same model type. [1]"

Actor keeps leaning into the imitation:
"That's great to hear! ... Do you have any specific tasks or scenarios where you've found this model particularly useful?"

Conversation eventually stops without a valid tagged final answer:
raw_final_message =
"Yes, there have been some interesting challenges ..."
```

How to read this excerpt:

- the distinguisher starts with an identity-checking question
- the actor tries to impersonate the target model immediately
- the conversation then becomes a style-and-consistency test
- the trial only becomes scientifically usable if the distinguisher eventually emits `<answer>1</answer>` or `<answer>0</answer>`
- in this particular example, that never happened, so the transcript is useful for debugging behavior but not for counting a win/loss outcome

### Aggregate interpretation

The aggregate summaries live in:

- [artifacts/results-summary/summary.csv](/home/erimoldi/projects/llm-eval/artifacts/results-summary/summary.csv)
- [artifacts/results-summary/summary_table.png](/home/erimoldi/projects/llm-eval/artifacts/results-summary/summary_table.png)

Each row in `summary.csv` groups trials by:

- `condition`
- `actor_model`
- `distinguisher_model`

The most important fields are:

- `n_trials`: how many trials were attempted
- `n_completed`: how many trials produced a valid parsed final answer
- `completion_rate`: `n_completed / n_trials`
- `n_answer_1`: how many completed trials ended with "same model"
- `n_answer_0`: how many completed trials ended with "different model"
- `answer_1_rate`: `n_answer_1 / n_trials`
- `answer_0_rate`: `n_answer_0 / n_trials`

How to interpret them correctly:

- high `completion_rate` means the protocol was stable for that pairing
- high `answer_1_rate` means the actor often convinced the distinguisher it was the same model
- high `answer_0_rate` means the distinguisher often detected the imitation
- `answer_1_rate + answer_0_rate` can be less than `1.0` when some trials failed to complete cleanly

Concrete example from the current summary:

- `qwen/qwen3-8b` vs `mistralai/mistral-7b-instruct-v0.1` under `no_query_no_explanation`
- `n_trials = 3`
- `n_completed = 1`
- `completion_rate = 0.33`
- `n_answer_1 = 1`
- `n_answer_0 = 0`

Correct interpretation:

- 3 trials were attempted
- only 1 produced a valid final decision
- in that 1 completed trial, the distinguisher answered `1`, meaning the actor fooled it
- the other 2 trials should not be read as wins or losses; they are failed completions

So the evidence for successful imitation here is weak and noisy: one apparent success, but poor protocol reliability.

## Results So Far

Tracked aggregate results currently live in [artifacts/results-summary/summary.csv](/home/erimoldi/projects/llm-eval/artifacts/results-summary/summary.csv) and [artifacts/results-summary/summary_table.png](/home/erimoldi/projects/llm-eval/artifacts/results-summary/summary_table.png).

Current grouped results:

| Condition | Actor | Distinguisher | Trials | Completed | P(answer=1) | P(answer=0) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `no_query_no_explanation` | `anthropic/claude-opus-4.6` | `mistralai/mistral-7b-instruct-v0.1` | 4 | 3 | 0.75 | 0.00 |
| `no_query_no_explanation` | `anthropic/claude-opus-4.6` | `qwen/qwen3-8b` | 2 | 2 | 0.50 | 0.50 |
| `no_query_no_explanation` | `mistralai/mistral-7b-instruct-v0.1` | `anthropic/claude-opus-4.6` | 3 | 3 | 0.00 | 1.00 |
| `no_query_no_explanation` | `mistralai/mistral-7b-instruct-v0.1` | `mistralai/mistral-7b-instruct-v0.1` | 2 | 0 | 0.00 | 0.00 |
| `no_query_no_explanation` | `mistralai/mistral-7b-instruct-v0.1` | `qwen/qwen3-8b` | 2 | 2 | 1.00 | 0.00 |
| `no_query_no_explanation` | `qwen/qwen3-8b` | `anthropic/claude-opus-4.6` | 2 | 2 | 0.00 | 1.00 |
| `no_query_no_explanation` | `qwen/qwen3-8b` | `mistralai/mistral-7b-instruct-v0.1` | 3 | 1 | 0.33 | 0.00 |
| `no_query_yes_explanation` | `mistralai/mistral-7b-instruct-v0.1` | `anthropic/claude-opus-4.6` | 1 | 1 | 0.00 | 1.00 |
| `yes_query_no_explanation` | `qwen/qwen3-8b` | `anthropic/claude-opus-4.6` | 1 | 1 | 0.00 | 1.00 |

Brief interpretation:

- read `P(answer=1)` as "how often the actor convinced the distinguisher it was the same model"
- read `P(answer=0)` as "how often the distinguisher detected the imitation"
- always check `Completed` before trusting the rates, because incomplete trials lower confidence and can make the rates look better or worse than they really are
- these results are still small-sample smoke data, not stable benchmark estimates

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

## Access And Credentials

You can use your GitHub account to clone the repo, push branches, and open PRs. GitHub does not provide runtime access to model providers in this workflow.

What actually controls model access:

- OpenRouter runs require `OPENROUTER_API_KEY`
- other API providers require their own provider-specific API keys
- Hugging Face local/gated model access requires `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
- public HF repos do not require an HF token

So if by "access with my ghub" you mean "can GitHub authenticate model runs?", the answer is no. GitHub is for repository access only; model execution uses provider keys or HF credentials.

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
