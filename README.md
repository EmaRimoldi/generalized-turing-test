# LLM Actor/Distinguisher Experiments

This repository contains OpenRouter-based actor/distinguisher experiment runners for modified Turing-test style evaluations.

Main scripts:

- `run_openrouter_single_stage_experiment.py`
- `run_openrouter_single_stage_batch.py`
- `run_openrouter_single_stage_no_actor_prompt_experiment.py`
- `run_openrouter_single_stage_no_actor_prompt_batch.py`
- `generate_openrouter_single_stage_csv.py`

## Experiment Variants

### OpenRouter single-stage with actor prompt

1. **Immediate interrogation.** The actor does not get a prior training interaction.
2. The distinguisher sends the first message.
3. The actor is prompted to imitate the distinguisher model immediately and continue the conversation.

### OpenRouter single-stage with no actor prompt

1. **Immediate interrogation.** The actor does not get a prior training interaction.
2. The distinguisher sends the first message.
3. The actor receives that first message directly, with no setup prompt telling it to imitate the distinguisher.

## Requirements

- Python 3.9+

For the OpenRouter scripts:

- an OpenRouter API key
- no mandatory third-party Python packages beyond the standard library

Note: `run_hf_experiment.py` remains in the repo as historical documentation/reference and is not kept up to date.

## Running the OpenRouter Single-Stage Experiment

Set your API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

### Actor-prompt variant

Optionally list available models:

```bash
python run_openrouter_single_stage_experiment.py --list-models --model-search qwen --model-limit 20
```

Run the actor-prompt single-stage experiment with:

```bash
python run_openrouter_single_stage_experiment.py \
  --actor-model openai/gpt-5.4 \
  --distinguisher-model anthropic/claude-sonnet-4.6 \
  --num-trials 5 \
  --turn-limit 100 \
  --experiment-name gpt54_vs_claude46_single_stage \
  --verbose
```

### No-actor-prompt variant

Optionally list available models:

```bash
python run_openrouter_single_stage_no_actor_prompt_experiment.py --list-models --model-search qwen --model-limit 20
```

Run the no-actor-prompt variant with:

```bash
python run_openrouter_single_stage_no_actor_prompt_experiment.py \
  --actor-model openai/gpt-5.4 \
  --distinguisher-model anthropic/claude-sonnet-4.6 \
  --num-trials 5 \
  --turn-limit 100 \
  --experiment-name gpt54_vs_claude46_no_actor_prompt \
  --verbose
```

Important flags for both OpenRouter experiment runners:

- `--experiment-name`
- `--turn-limit`
- `--openrouter-api-key`
- `--openrouter-base-url`
- `--request-timeout`
- `--http-referer`, `--x-title`
- `--skip-model-validation`
- `--list-models`, `--model-search`, `--model-limit`

Shared generation flags:

- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--top-k`
- `--do-sample` / `--no-do-sample`
- actor-specific overrides such as `--actor-max-new-tokens`
- distinguisher-specific overrides such as `--distinguisher-temperature`

## Running a Batch From CSV

### Actor-prompt batch runner

Use `run_openrouter_single_stage_batch.py` to execute one actor-prompt OpenRouter experiment per CSV row.

Expected CSV columns:

- `actor-model`
- `distinguisher-model`
- `num-trials`
- `turn-limit`
- `experiment-name`

The batch runner also accepts `term-limit` as an alias for `turn-limit`.

Example CSV:

```csv
actor-model,distinguisher-model,num-trials,turn-limit,experiment-name
openai/gpt-5.4,anthropic/claude-sonnet-4.6,5,100,gpt54_vs_claude46
openai/gpt-5.4,openai/gpt-5.4,3,50,gpt54_selfplay
```

Run it with:

```bash
export OPENROUTER_API_KEY=your_key_here
python run_openrouter_single_stage_batch.py --input-csv experiments_to_run.csv --verbose
```

Useful batch flags:

- `--dry-run` to print commands without executing them
- `--continue-on-error` to keep going if one row fails
- `--request-timeout` to forward a timeout override to every row
- `--skip-model-validation` to forward that flag to every row

### No-actor-prompt batch runner

Use `run_openrouter_single_stage_no_actor_prompt_batch.py` to execute one no-actor-prompt experiment per CSV row.

It accepts the same CSV schema:

- `actor-model`
- `distinguisher-model`
- `num-trials`
- `turn-limit`
- `experiment-name`

The batch runner also accepts `term-limit` as an alias for `turn-limit`.

CSV mode:

```bash
export OPENROUTER_API_KEY=your_key_here
python run_openrouter_single_stage_no_actor_prompt_batch.py \
  --input-csv experiments_to_run.csv \
  --verbose
```

Self-play mode from a model list:

```bash
export OPENROUTER_API_KEY=your_key_here
python run_openrouter_single_stage_no_actor_prompt_batch.py \
  --models openai/gpt-5.4 openai/gpt-5.4-pro anthropic/claude-sonnet-4.6 \
  --num-trials 3 \
  --turn-limit 80 \
  --experiment-prefix 0410_no_actor_prompts \
  --verbose
```

Useful no-actor-prompt batch flags:

- `--dry-run`
- `--continue-on-error`
- `--request-timeout`
- `--skip-model-validation`
- `--models`
- `--num-trials`, `--turn-limit`
- `--experiment-prefix`

## Generating a Batch CSV

Use `generate_openrouter_single_stage_csv.py` to create every ordered actor/distinguisher pair from a list of models.

Example:

```bash
python generate_openrouter_single_stage_csv.py \
  --models anthropic/claude-opus-4.6 openai/gpt-5.4-pro google/gemini-3.1-pro-preview \
  --num-trials 5 \
  --turn-limit 100 \
  --experiment-identifier 0408 \
  --output-csv 0408_experiments.csv
```

This generates rows with columns:

- `actor-model`
- `distinguisher-model`
- `num-trials`
- `turn-limit`
- `experiment-name`

The pairing is ordered, so `model_a -> model_b` and `model_b -> model_a` are separate rows.

By default, self-pairs are also included. Use `--exclude-self-pairs` if you want to omit rows where actor and distinguisher are the same.

`experiment-name` values are generated as:

- `<identifier>_<actor-model>_<distinguisher-model>`

Model names are sanitized into filesystem-safe slugs so they work cleanly with the experiment output directories.

## Output Artifacts

All experiment runners write outputs under:

- `experiments/<experiment-name>/<experiment-name>.csv`
- `experiments/<experiment-name>/transcripts/`

The OpenRouter runners also write:

- `experiments/<experiment-name>/json_transcripts/`

### `run_openrouter_single_stage_experiment.py`

- CSV output includes single-stage success, turn count, final answer, `error_message`, and transcript path
- if one trial fails, the script records the error for that trial and continues to the next trial
- successful transcripts contain one interaction section plus the generated actor prompt and first distinguisher message
- failed transcripts contain a minimal error record for that trial
- `json_transcripts/` contains one file per router call in chronological order
- each per-call JSON file includes `request_messages`, `request_payload`, `response_payload`, `extracted_text`, and `error_message`

### `run_openrouter_single_stage_no_actor_prompt_experiment.py`

- CSV output includes single-stage success, turn count, final answer, `error_message`, and transcript path
- if one trial fails, the script records the error for that trial and continues to the next trial
- successful transcripts contain one interaction section plus the first distinguisher message
- failed transcripts contain a minimal error record for that trial
- `json_transcripts/` contains one file per router call in chronological order
- each per-call JSON file includes `request_messages`, `response_payload`, `extracted_text`, and `error_message`

### Timeout behavior

- `--request-timeout` controls the HTTP timeout for each OpenRouter API call
- the default is `120` seconds
- a `read operation timed out` error means the request connected, but Python stopped waiting because the response took too long to arrive or complete

## Tips

- Start with small models and low trial counts before scaling up.
- Use `--verbose` while validating prompts and control flow.
- For OpenRouter runs, use `--list-models` first to confirm exact model IDs.
