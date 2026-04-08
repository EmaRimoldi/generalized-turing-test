# LLM Actor/Distinguisher Experiments

This repository contains two experiment runners based on a modified Turing test:

- `run_hf_experiment.py` for the original two-stage experiment using local Hugging Face models
- `run_openrouter_single_stage_experiment.py` for a single-stage experiment using remote OpenRouter models

## Experiment Variants

### Hugging Face two-stage

1. **Stage 1 – Actor training.** The actor talks to a distinguisher instance to learn its style.
2. **Stage 2 – Distinguisher interrogation.** The same actor then talks to a fresh distinguisher instance that decides whether it is speaking to itself.

### OpenRouter single-stage

1. **Immediate interrogation.** The actor does not get a prior training interaction.
2. The distinguisher sends the first message.
3. The actor is prompted to imitate the distinguisher model immediately and continue the conversation.

## Requirements

- Python 3.9+

For `run_hf_experiment.py`:

- [PyTorch](https://pytorch.org/get-started/locally/) built for your CPU/GPU environment
- `transformers` ≥ 4.37
- `accelerate`
- Optional: `bitsandbytes` or other quantization/offloading libraries

For `run_openrouter_single_stage_experiment.py`:

- an OpenRouter API key
- no mandatory third-party Python packages beyond the standard library

Install the Hugging Face dependencies with:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
```

## Running the Hugging Face Experiment

Run the original two-stage local experiment with:

```bash
python run_hf_experiment.py \
  --actor-model meta-llama/Llama-2-7b-chat-hf \
  --distinguisher-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --num-trials 5 \
  --stage1-turn-limit 12 \
  --stage2-turn-limit 18 \
  --output-csv results/mixtral_vs_llama.csv \
  --transcript-dir transcripts \
  --actor-device cuda \
  --distinguisher-device cuda \
  --actor-dtype bfloat16 \
  --distinguisher-dtype bfloat16 \
  --verbose
```

Important flags:

- `--stage1-turn-limit`, `--stage2-turn-limit`
- `--actor-device`, `--distinguisher-device`
- `--actor-dtype`, `--distinguisher-dtype`
- `--revision`, `--cache-dir`, `--trust-remote-code`, `--no-fast-tokenizer`

## Running the OpenRouter Single-Stage Experiment

Set your API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Optionally list available models:

```bash
python run_openrouter_single_stage_experiment.py --list-models --model-search qwen --model-limit 20
```

Run the single-stage experiment with:

```bash
python run_openrouter_single_stage_experiment.py \
  --actor-model openai/gpt-5.4 \
  --distinguisher-model anthropic/claude-sonnet-4.6 \
  --num-trials 5 \
  --turn-limit 100 \
  --output-csv results/gpt54_vs_claude46_single_stage.csv \
  --transcript-dir transcripts \
  --verbose
```

Important flags:

- `--turn-limit`
- `--openrouter-api-key`
- `--openrouter-base-url`
- `--request-timeout`
- `--http-referer`, `--x-title`
- `--skip-model-validation`
- `--list-models`, `--model-search`, `--model-limit`

## Shared Generation Flags

Both scripts support:

- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--top-k`
- `--do-sample` / `--no-do-sample`
- actor-specific overrides such as `--actor-max-new-tokens`
- distinguisher-specific overrides such as `--distinguisher-temperature`

## Output Artifacts

### `run_hf_experiment.py`

- CSV output includes Stage 1 and Stage 2 success flags and turn counts, final answer, reasoning, and transcript path
- transcripts contain separate Stage 1 and Stage 2 sections

### `run_openrouter_single_stage_experiment.py`

- CSV output includes single-stage success, turn count, final answer, and transcript path
- transcripts contain one interaction section plus the generated actor prompt and first distinguisher message

## Tips

- Start with small models and low trial counts before scaling up.
- Use `--verbose` while validating prompts and control flow.
- For OpenRouter runs, use `--list-models` first to confirm exact model IDs.
- For local Hugging Face runs, `--actor-device auto --distinguisher-device auto` is usually the best default on GPU machines.
