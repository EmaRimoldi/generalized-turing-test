# Reproducing experiments

The runner implements the paper's interactive protocols. The included data is
an earlier baseline snapshot; [data/README.md](data/README.md) describes its
coverage. Launching new runs and rebuilding the committed explorer are separate
workflows.

## Environment

Use Python 3.11 and install `requirements.txt` in a virtual environment, as shown
in the [README](README.md#run-experiments). Export `OPENROUTER_API_KEY` before
launching a hosted experiment. The CLI does not load `.env` automatically.

Alternatively, with Apptainer installed:

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY in .env.
./scripts/build-container.sh
./scripts/run-in-container.sh python run_experiment.py --help
```

The container definition is `apptainer/turing-eval.def`; the wrapper loads `.env`
and forwards provider keys. Both image definitions install `requirements.txt`;
the HF image additionally installs `requirements-hf.txt`. Build helpers try
fakeroot and unprivileged builds, retaining the previous image on failure.
If neither mode is available, `scripts/build-container-fallback.sh` prepares a
base image and a separate virtual environment; invoke its interpreter explicitly:

```bash
./scripts/run-in-container.sh /workspace/.venv-container/bin/python run_experiment.py --help
```

All paper model calls used OpenRouter. The optional
HF backend, alternate routes, and historical prompt variants remain available
for compatibility but are not the paper's default configuration.

## Paper baseline schedules

[`configs/paper_gtt.csv`](configs/paper_gtt.csv) and
[`configs/paper_gttq.csv`](configs/paper_gttq.csv) transcribe the grid and budgets
in [Appendix B.1](https://arxiv.org/html/2605.10851v2). They are newly
reconstructed schedules, not recovered original job manifests.

| Setting | GTT | GTTQ |
| --- | --- | --- |
| Condition | `no_query_no_explanation` | `yes_query_no_explanation` |
| Ordered pairs, including self-pairs | 81 | 81 |
| Trials per frontier-block cell | 40 | 25 |
| Trials per other cell | 25 | 10 |
| Total trials | 2,160 | 945 |
| Distinguisher turn cap | 40 | 40 |
| Specimen turn cap | — | 20, or early `STOP` |

The frontier block is Opus / GPT-5.4 / Gemini. The nine-model roster in these
files uses the registry nicknames `claude-opus-4.6`, `claude-sonnet-4.6`,
`gpt-5.4`, `gemini-3-pro`, `deepseek-v3.2`, `mistral-large-2512`,
`ministral-8b-2512`, `qwen3-32b`, and `grok-4.20`. Routes are defined in
[`turing_eval/models.py`](turing_eval/models.py).

```bash
python run_experiment_batch.py --csv configs/paper_gtt.csv --output-dir results/gtt
python run_experiment_batch.py --csv configs/paper_gttq.csv --output-dir results/gttq
```

These commands invoke paid APIs. Leave temperature and token caps unset to use
the served defaults. Runs refuse existing output directories. Batch schedules
require unique run names and are validated in full before any experiment starts.
Choose a new output root for a fresh run; these commands do not resume old runs.

The runner evaluates each specified pair; it does not randomly select a new
actor identity for every call. The balanced GTT probability is computed during
analysis by combining genuine-target and imitation branch estimates. Check how
self-pair data was collected before using it as the genuine-target baseline.

## Additional protocols

All conditions are implemented in [`turing_eval/conditions.py`](turing_eval/conditions.py),
with prompt text in [`turing_eval/prompts.py`](turing_eval/prompts.py).

| Purpose | Condition | Required extra arguments |
| --- | --- | --- |
| Fixed specimen-query budget | `yes_query_controlled_queries` | `--num-queries N` |
| Fixed interrogation budget | `no_query_controlled_turns` | `--num-turns N` |
| UTT: independent judge and nominated target | `no_query_trusted_distinguisher` | `--target-model MODEL` |
| Unprompted genuine-target control | `no_prompt_self_pair` | Same actor and distinguisher |

For example, Gemini imitates GPT-5.4 while Opus judges:

```bash
python run_experiment.py \
  --actor gemini-3-pro --distinguisher claude-opus-4.6 \
  --target-model gpt-5.4 --condition no_query_trusted_distinguisher \
  --num-trials 10 --turn-limit 40 --output-dir results/utt
```

For other variants, use `run_experiment.py --help`. `run_experiment_matrix.py`
sweeps a model list and omits self-pairs unless `--include-self-match` is set.
`run_experiment_batch.py` accepts explicit CSV schedules: one row per run with
`run_name,actor,distinguisher,num_trials,turn_limit,condition` and optional
`specimen_turn_limit,num_queries,num_turns,target_model` columns. Required
condition-specific budgets must be supplied.

The complete final raw data, original schedules for targeted experiments, and
paper-specific prompt-robustness analysis scripts are not included here. The
available runner variants should not be assumed to reproduce every appendix
experiment without its exact prompts and configuration.

## Outputs and scoring

Each run writes `results.csv` and `trial_XXX.json` beneath its run directory.
Role instructions are ordinary user messages. Transcripts retain the role
histories, prompts, routes, answers, budgets, and environment metadata.

- `answer=1`: the judge says “same model” (or the nominated target in UTT).
- `answer=0`: the judge says “different model.”
- `success` records runner completion; it is not actor or judge accuracy.
- `error=distinguisher_answered_in_opening` records an answer before interaction.
  Such tags require semantic inspection before being included as verdicts.

Provider failures are retried up to `--max-attempts` (default: three attempts per
requested trial). Failed attempts are stored separately from collected trials.
If the requested count cannot be collected, the runner saves the partial results
and returns a nonzero exit code. Batch and matrix launchers continue other runs,
report failures, and also exit unsuccessfully if any run is incomplete.

The shared protocol registry and option validation live in `turing_eval/cli.py`.
Batch and matrix launchers resolve the single-run script independently of the
working directory. Relative input/output paths still refer to the caller's
working directory. `scripts/run_batch_in_tmux.sh schedule.csv [output-dir]` is an
optional detached launcher using the local virtual environment and safe path
quoting; it requires tmux.

To inspect newly collected CSVs:

```bash
python scripts/render_results_summary.py \
  --input-root results/gtt --output-dir results/summary
```

This is a descriptive run summary, not the complete paper scoring pipeline.

## Rebuild the explorer offline

With Python 3.11 and Node.js 20:

```bash
npm --prefix visualizer/web ci
npm --prefix visualizer/web run build
python3 visualizer/scripts/serve_epsilon_visualizer.py
```

Open [localhost:8765](http://127.0.0.1:8765/). The build derives all 162 pairs from
the committed A/B grouped JSON files; model APIs and original Drive folders are
not needed. `tests/test_visualizer_build.py` checks the complete derived artifact
against its tracked snapshot. See the [visualizer guide](visualizer/README.md)
for aggregation policies and development commands.

## Historical trials and verification

Transcripts record Python/platform, Git commit/branch/dirty state, hostname, and
container metadata when available. Before checking out an old transcript's
commit, consult the [public history mapping](docs/history/README.md): public
commit identifiers differ from the original private history.

Reusing the recorded model, protocol, budgets, prompts, and code version allows
another execution, not a bit-exact replay. Hosted providers can change served
models and routing. Retain route metadata and record any deviations.

Offline validation:

```bash
python -m ruff check .
python -m pytest tests/ -q
npm --prefix visualizer/web test
npm --prefix visualizer/web run build
git diff --exit-code -- visualizer/web/public/epsilon_visualizer_data.json
```

The frontend build includes TypeScript checking. Python lint rules are defined
in `ruff.toml`, so local user settings do not alter the project's checks.
