# LLM Actor/Distinguisher Experiments

This repository contains a CLI utility for running a two-stage experiment to perturb LLM relative intelligence as defined by a modified Turing test.

1. **Stage 1 – Actor training.** An actor model converses with a distinguisher instance to learn the distinguisher’s style. The actor can keep asking questions until it emits `STOP 1` or the Stage‑1 turn limit is reached.
2. **Stage 2 – Distinguisher interrogation.** The same actor (memory intact) now talks to a *fresh* distinguisher instance that tries to determine if it is speaking to itself. Stage 2 begins when the distinguisher’s first message is prefixed with `START 2:` for the actor. The run ends when the distinguisher produces `<answer>0|1</answer>` plus 1‑3 sentences in `<reasoning></reasoning>`, or the Stage‑2 turn cap is hit.

Each trial records whether both stages completed successfully, the distinguisher’s final answer/reasoning, and paths to full chat transcripts for later inspection. This lets you batch experiments externally by simply calling the CLI repeatedly with different model pairings.

---

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/get-started/locally/) built for your CPU/GPU environment
- `transformers` ≥ 4.37 (older versions lack the `TextGenerationPipeline` chat features used here)
- `accelerate` (required because the script always sets a `device_map`; even `--*-device cpu` routes through Accelerate)
- Optional: `bitsandbytes` or any other libraries you rely on for quantization/offloading.

You can install the essentials via:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121   # pick the wheel that matches your setup
pip install transformers accelerate
```

If you plan to run large models, consider adding `bitsandbytes` or using Hugging Face’s `device_map="auto"` sharding on multi-GPU servers (Accelerate handles placement automatically once installed).

---

## Running an Experiment

All logic lives inside `run_hf_experiment.py`. Invoke it with the actor/distinguisher model IDs (either Hugging Face Hub IDs like `meta-llama/Llama-2-7b-chat-hf` or local directories) plus your preferred limits:

```bash
python run_hf_experiment.py \
  --actor-model meta-llama/Llama-2-7b-chat-hf \
  --distinguisher-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --num-trials 5 \
  --stage1-turn-limit 12 \
  --stage2-turn-limit 18 \
  --output-csv results/mixtral_vs_llama.csv \
  --transcript-dir transcripts \
  --actor-device cuda --distinguisher-device cuda \
  --actor-dtype bfloat16 --distinguisher-dtype bfloat16 \
  --verbose
```

Key behaviors:

- **Actor prompt** instills the mimicry objective; Stage 1 automatically starts with `START 1`, and Stage 2 uses `START 2:` as required.
- **Distinguisher prompt** instructs the model to decide between “self” vs “other” and produce `<answer>`/`<reasoning>` tags.
- **Turn limits** (`--stage1-turn-limit`, `--stage2-turn-limit`) cap the number of back-and-forth exchanges if the actor never emits `STOP 1` or the distinguisher never produces `<answer>…</answer>`.
- **Verbose mode** mirrors the entire conversation to stdout so you can watch learning/interrogation live.
- **Transcripts** for every trial are serialized to JSON with names like `transcripts/run_20240512-153055_trial_000.json`. The CSV row references the corresponding file so you can jump straight to the logs whenever something looks off.

---

## CLI Reference

| Flag | Purpose |
| --- | --- |
| `--actor-model`, `--distinguisher-model` | Hugging Face model ID or local path for each role. |
| `--num-trials` (`x`) | Number of repeated trials with the same configuration. |
| `--stage1-turn-limit` (`n`) | Max Stage‑1 exchanges (actor ↔ distinguisher trainer). |
| `--stage2-turn-limit` (`m`) | Max Stage‑2 exchanges (actor ↔ fresh distinguisher). |
| `--output-csv` | Target CSV with aggregated per-trial results. |
| `--transcript-dir` | Directory for JSON transcripts. |
| `--run-name` | Custom prefix for transcript filenames (defaults to timestamp). |
| `--verbose` | Stream both conversations while they run. |

### Device / precision overrides

- `--actor-device`, `--distinguisher-device`: Accept any `transformers` `device_map` string (`auto`, `cuda`, `cuda:1`, `cpu`, etc.).
- `--actor-dtype`, `--distinguisher-dtype`: Force `float16`, `bfloat16`, or `float32`.
- `--revision`, `--cache-dir`, `--trust-remote-code`, `--no-fast-tokenizer`: Passed directly into `AutoModelForCausalLM` / `AutoTokenizer`.

### Generation controls

Set defaults (apply to both agents) with:

```
--max-new-tokens 256 --temperature 0.7 --top-p 0.9 --top-k 50 --do-sample/--no-do-sample
```

Override per role via prefixed flags, e.g.:

```
--actor-max-new-tokens 128 --actor-no-do-sample
--distinguisher-temperature 0.3 --distinguisher-top-p 0.8
```

---

## Output Artifacts

1. **CSV summary (`--output-csv`)**
   - Columns: trial ID, actor/distinguisher IDs, Stage 1 & Stage 2 success flags + turn counts, final `<answer>` value, `<reasoning>` text, transcript path.
   - Use this for downstream statistics—e.g., compute how often the actor successfully fools the distinguisher.
2. **Transcripts (`--transcript-dir`)**
   - JSON files containing Stage 1 and Stage 2 message lists (exact role/content pairs), success indicators, and the distinguisher’s raw final emission.
   - These stay separate so you can keep CSVs lightweight but still audit any run.

---

## Tips

- **Small local tests:** Start with lightweight instruct models (`HuggingFaceTB/SmolLM2-1.7B-Instruct` etc.) on CPU to validate the flow before scaling up.
- **GPU servers:** Supply `--actor-device auto --distinguisher-device auto` to let Transformers shard models across available GPUs. Mix precision (`--*-dtype bfloat16`) to conserve memory.
- **Batching:** Use your scheduler (bash loop, Python script, SLURM job array) to fire off multiple CLI calls with different pairings and turn limits; every invocation can reuse the same transcript directory and just point to a different CSV.

With this setup you can efficiently explore how well different actors can mimic distinguishers across model families, contexts, and hardware targets. Happy experimenting!
