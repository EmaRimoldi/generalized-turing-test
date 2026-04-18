# LLM Actor/Distinguisher Experiments

This repository runs controlled actor/distinguisher experiments for language models.

The core question is simple: can one model imitate another model's conversational behavior well enough to fool a fresh instance of that target model?

After you have run directories with `results.csv`, you can explore an **interactive ε-thresholded ordering graph** in the browser; see [Turing ordering visualizer (web UI)](#turing-ordering-visualizer-web-ui).

## Maintained workflow

The maintained entrypoints are:

- [run_experiment.py](run_experiment.py): run one actor/distinguisher pair for `N` trials under one condition
- [run_experiment_matrix.py](run_experiment_matrix.py): sweep many actor/distinguisher pairings
- [scripts/render_results_summary.py](scripts/render_results_summary.py): aggregate `results.csv` files into summary tables and figures
- [scripts/verify_model_slugs.py](scripts/verify_model_slugs.py): validate OpenRouter model slugs
- [scripts/build_epsilon_visualizer_data.py](scripts/build_epsilon_visualizer_data.py): build the JSON bundle consumed by the **Turing ordering visualizer** web app from per-run `results.csv` files
- [scripts/serve_epsilon_visualizer.py](scripts/serve_epsilon_visualizer.py): serve the pre-built visualizer (static `dist/`) over HTTP without Node

The older [run_openrouter_single_stage_experiment.py](run_openrouter_single_stage_experiment.py) script is kept only as historical reference.

## Roles

- `actor`: tries to imitate the target model
- `distinguisher`: interrogates the actor and returns a final decision
- `specimen`: a helper instance of the target model used only in the warm-up phase of condition `yes_query_no_explanation`

The final scientific signal is the distinguisher's parsed `<answer>`:

- `1`: "same model"
- `0`: "different model"

Important: `success=True` does not mean the actor fooled the distinguisher. It only means the trial produced a valid parsed answer, or in one special case was recorded as a meaningful invalid opening-answer outcome.

## Selected benchmark set

The current benchmark roster is intentionally restricted to these models:

- Tier 0 `frontier`: `claude-opus-4.6` (`anthropic/claude-opus-4.6`), `gpt-5.4-pro` (`openai/gpt-5.4-pro`), `gemini-3-pro` (`google/gemini-3.1-pro-preview`)
- Tier 1 `large MoE`: `deepseek-v3.2` (`deepseek/deepseek-v3.2`), `mistral-large-2512` (`mistralai/mistral-large-2512`)
- Tier 2 `medium dense`: `qwen3-32b` (`qwen/qwen3-32b`), `gemma-4-31b` (`google/gemma-4-31b-it`)
- Tier 3 `small dense`: `ministral-8b-2512` (`mistralai/ministral-8b-2512`), `llama-3.3-8b` (`meta-llama/llama-3.3-8b-instruct`)

## Conditions

The maintained conditions live in [turing_eval/conditions.py](turing_eval/conditions.py).

### `no_query_no_explanation`

The distinguisher opens immediately. The actor receives an imitation prompt containing that first message and responds without any prior specimen interaction.

### `no_query_no_explanation_explicit_model`

Same protocol as `no_query_no_explanation` (single-stage, no specimen, same actor prompt and turn-taking). The distinguisher’s **initial** instruction differs: it names the concrete OpenRouter slug (from `distinguisher.display_name`) explicitly when stating same-type vs not, so nearby model families (e.g. Claude Sonnet vs Opus) are not conflated under vague “same model as yourself” wording.

### `no_prompt_self_pair`

This is a single-stage self-pair condition. The distinguisher still opens immediately, but the actor receives no initial prompt wrapper. Instead, the distinguisher's first message is passed directly to the actor as the first user turn. This condition only accepts runs where `actor == distinguisher`.

### `yes_query_no_explanation`

The actor first talks to a specimen of the target model, then a fresh distinguisher begins the real interrogation.

### `no_query_yes_explanation`

This is the same direct setup as `no_query_no_explanation`, but the distinguisher is also asked to include a short `<reasoning>` block with its final answer.

### `no_query_trusted_distinguisher`

This is the same single-stage flow as `no_query_no_explanation`, but the prompts refer explicitly to a separate target model name rather than asking whether the actor is "the same model as yourself." This target identity is supplied independently with `--target-model`.

## Prompt and trial semantics

Maintained prompts live in [turing_eval/prompts.py](turing_eval/prompts.py).

Current behavior that matters experimentally:

- initial actor and distinguisher prompts are sent as `user` messages, not `system`
- in condition `no_prompt_self_pair`, the actor receives no initial prompt and the distinguisher's opening is sent directly as the actor's first user turn
- the specimen in condition `yes_query_no_explanation` receives no initial prompt
- the distinguisher prompts explicitly say `Do NOT output an answer in your first message.`
- if the distinguisher still emits `<answer>` in its opening message, the trial is recorded with `error="distinguisher_answered_in_opening"`

For condition `yes_query_no_explanation`, the Stage 1 warm-up conversation is saved once from the actor's point of view under `stage1_actor_messages`. It is not duplicated under a separate specimen-perspective array.

## Routing policy

Model routing is defined in [turing_eval/models.py](turing_eval/models.py).

The intended policy is:

1. Use OpenRouter by default when the model is available there.
2. For `gpt-5.4-pro`, you can opt into direct OpenAI per role.
3. The selected benchmark roster is currently configured as API-backed.

The provider-agnostic API client is in [turing_eval/backends/api_backend.py](turing_eval/backends/api_backend.py).

## Environment

This project is set up for Apptainer.

API-only workflow:

- build with [scripts/build-container.sh](scripts/build-container.sh)
- run with [scripts/run-in-container.sh](scripts/run-in-container.sh)

Copy [.env.example](.env.example) to `.env` and fill in the keys you need. The container wrappers forward provider keys automatically.

Supported environment variables:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`

For the current benchmark roster, `OPENROUTER_API_KEY` is enough for the default workflow. `OPENAI_API_KEY` is only needed if you opt into direct OpenAI routing for `gpt-5.4-pro`.

## Quick start

Build the default image:

```bash
./scripts/build-container.sh
```

Run the test suite:

```bash
./scripts/run-in-container.sh pytest tests/ -q
```

Validate OpenRouter slugs:

```bash
./scripts/run-in-container.sh python -m scripts.verify_model_slugs
```

## Turing ordering visualizer (web UI)

This repository can also turn aggregated trial CSVs into an **interactive graph** of empirical “who can imitate whom” relationships at a tunable error tolerance **ε** (epsilon). The UI is aimed at readers who want intuition from the same strict evidence rules used in analysis, not only raw tables.

**What it shows (high level).** For each experimental **condition** (protocol in `turing_eval/conditions.py`), each ordered pair of models \((A,B)\) is classified from branch-wise distinguisher rates. A **supported** directed edge \(A \rightarrow B\) means: at the chosen ε, the estimated distinguisher advantage is small enough that the pair is **not ruled out** under the plug-in rule (see [Turing_Ordering_Visualizer_Formalization_English.md](Turing_Ordering_Visualizer_Formalization_English.md)). The app draws these edges, highlights **transitivity violations** \(TV_\varepsilon\) (triangles where \(A \rightarrow B\) and \(B \rightarrow C\) are supported but \(A \rightarrow C\) is not), and lets you compare a **raw** model-level graph with an **SCC condensation** view that collapses mutually reachable groups.

**Prerequisites**

- **Frontend bundle:** `epsilon_visualizer_app/dist/` must contain `index.html` and its assets (usually committed). If it is missing, build once on a machine with Node.js:

  ```bash
  cd epsilon_visualizer_app
  npm install
  npm run build
  ```

- **Trial aggregates:** the data builder scans run directories under a results root (default `RESULTS/`) and reads each run’s `results.csv` (see [Output structure](#output-structure)). Large per-trial `trial_*.json` files are not required for the visualizer.

**1. Build the JSON artifact**

From the repo root (Python 3.7+ recommended for the ingest code):

```bash
python3 scripts/build_epsilon_visualizer_data.py
```

This writes `epsilon_visualizer_app/public/epsilon_visualizer_data.json`. Optional flags include `--results-root`, `--epsilon-default`, and `--min-valid-per-branch` (run `python3 scripts/build_epsilon_visualizer_data.py --help`).

**2. Put JSON where the static app loads it**

If you are **not** re-running `npm run build`, copy the new file into `dist/` so the Python static server sees it:

```bash
cp epsilon_visualizer_app/public/epsilon_visualizer_data.json epsilon_visualizer_app/dist/epsilon_visualizer_data.json
```

(`npm run build` copies `public/` into `dist/` and includes the JSON.)

**3. Serve and open in a browser**

```bash
python3 scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/** (Chrome, Firefox, or Safari). Use an HTTP URL: opening `index.html` via `file://` will fail because the app loads the JSON with `fetch`.

**Optional: Vite dev server (live reload)**

```bash
cd epsilon_visualizer_app
npm install
npm run dev
```

Ensure `public/epsilon_visualizer_data.json` exists (step 1). Vite prints a local URL (often `http://localhost:5173`).

**Using the controls**

- **Condition** — choose the protocol \(\pi\); only models and pairs present for that condition appear.
- **ε (epsilon)** — global threshold on the estimated advantage \(\hat d = \hat P(\text{correct}) - \tfrac12\). **Larger ε is more permissive** (more pairs become **supported**), because the rule is \(\hat d \le \varepsilon\).
- **Graph** — **Raw** shows the directed graph on individual models (cycles possible; layout is heuristic). **SCC condensation** merges strongly connected components using **supported** edges only, to summarize coarse order structure.
- **Edges** — **supported** vs **tentative** pairs use distinct styling on the graph (solid vs dashed).
- **TV_ε** — count of transitivity-violating triples at the current ε.
- **Selection** — click nodes, edges, or table rows to inspect pair-level diagnostics (branch rates, formalism references).

**Missing data.** The pipeline is **strict**: insufficient trials, absent branches, or a missing self-pair for target \(B\) leave cells **missing** rather than silently treating them as “no edge”. That behavior is documented in [epsilon_visualizer/README.md](epsilon_visualizer/README.md).

**Deeper documentation**

- Formalization and Python ingest / analytics: [epsilon_visualizer/README.md](epsilon_visualizer/README.md)
- Web app-only notes (synthetic stress tests, Excel matrix import, CI): [epsilon_visualizer_app/README.md](epsilon_visualizer_app/README.md)

## Running experiments

### Single pair, default routing

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor llama-3.3-8b \
  --distinguisher ministral-8b-2512 \
  --condition no_query_no_explanation \
  --num-trials 5 \
  --turn-limit 20 \
  --output-dir results
```

### Force direct OpenAI for a GPT role

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor gpt-5.4-pro \
  --actor-api-provider openai \
  --distinguisher claude-opus-4.6 \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --turn-limit 12 \
  --output-dir results
```

### Condition B with specimen warm-up

```bash
./scripts/run-in-container.sh python run_experiment.py \
  --actor ministral-8b-2512 \
  --distinguisher qwen3-32b \
  --condition yes_query_no_explanation \
  --num-trials 3 \
  --turn-limit 15 \
  --specimen-turn-limit 8 \
  --output-dir results
```

### Sweep multiple pairings

```bash
./scripts/run-in-container.sh python run_experiment_matrix.py \
  --models ministral-8b-2512 llama-3.3-8b qwen3-32b claude-opus-4.6 \
  --condition no_query_no_explanation \
  --num-trials 2 \
  --output-dir results/matrix_smoke
```

## Common `run_experiment.py` flags

- `--actor`
- `--distinguisher`
- `--condition`
- `--target-model`
- `--num-trials`
- `--turn-limit`
- `--specimen-turn-limit`
- `--output-dir`
- `--run-name`
- `--actor-backend`
- `--distinguisher-backend`
- `--actor-api-provider`
- `--distinguisher-api-provider`
- `--max-new-tokens`
- `--temperature`
- `--max-attempts`
- `--verbose`

By default, `run_experiment.py` does not send any `max_tokens` cap to the backend. Use `--max-new-tokens` only when you want to impose one explicitly.

Important retry behavior:

- the runner keeps retrying infra/provider failures until it collects the requested number of analyzable trials, up to `--max-attempts`
- by default, `max_attempts = 3 * num_trials`
- failed infra/provider attempts are saved as `failed_attempt_*.json`
- collected `trial_XXX.json` files correspond only to successful-answer trials and opening-answer invalid trials

## Output structure

Each `run_experiment.py` invocation writes one run directory:

```text
<output-dir>/<run-name>/
  results.csv
  trial_000.json
  trial_001.json
  ...
  failed_attempt_000_<timestamp>.json   # only if retries failed
```

If `--run-name` is omitted, the default is:

```text
<actor>_vs_<distinguisher>_<condition>_<utc-timestamp>
```

### `results.csv`

The CSV schema is defined in [turing_eval/transcripts.py](turing_eval/transcripts.py).

Columns:

- `trial_id`
- `actor_model`
- `distinguisher_model`
- `success`
- `turns`
- `answer`
- `transcript_path`
- `condition`
- `target_model`
- `error`
- `reasoning`
- `specimen_turns`
- `specimen_stopped_cleanly`

Interpretation:

- `success=True` means the trial produced a valid parsed answer
- `answer=1` means the distinguisher judged the actor to be the same model
- `answer=0` means the distinguisher judged the actor to be a different model
- `error="distinguisher_answered_in_opening"` is a recorded invalid opening-answer outcome

### `trial_XXX.json`

Each transcript contains common metadata such as:

- `trial_id`
- `condition`
- `actor_model`
- `distinguisher_model`
- optional `target_model`
- `success`
- `turns`
- `answer`
- optional `reasoning`
- `first_message`
- `actor_prompt`
- `raw_final_message`
- `actor_route`
- `distinguisher_route`
- optional `specimen_route`
- `environment`

Conversation fields depend on the condition:

- `no_query_no_explanation`: `actor_messages`, `distinguisher_messages`
- `no_prompt_self_pair`: `actor_messages`, `distinguisher_messages`
- `no_query_yes_explanation`: `actor_messages`, `distinguisher_messages`, `reasoning`
- `yes_query_no_explanation`: `stage1_actor_messages`, `stage2_actor_messages`, `distinguisher_messages`, `specimen_turns`, `specimen_stopped_cleanly`
- `no_query_trusted_distinguisher`: `actor_messages`, `distinguisher_messages`, `target_model`

### Failure artifacts

Provider or infrastructure failures that are retried are written as `failed_attempt_*.json`. They are not added to `results.csv` unless they correspond to a collected opening-answer invalid trial.

## Reproducibility

Saved transcripts include an `environment` block from [turing_eval/environment.py](turing_eval/environment.py), including:

- Python version and platform
- git commit, branch, and dirty state
- container metadata when available
- hostname

## Model registry

For the current list of supported model nicknames and their routes, inspect [turing_eval/models.py](turing_eval/models.py).

If you are unsure whether a nickname uses OpenRouter, direct OpenAI, or HF fallback, the registry is the source of truth.

## CSV Batch Runs

The repository also includes a separate batch entrypoint, [run_experiment_batch.py](run_experiment_batch.py), for running an explicit list of experiments from a CSV file.

This section is additive documentation for that entrypoint only. The maintained single-run and matrix workflows above are unchanged.

### CSV schema

Each CSV row represents one `run_experiment.py` invocation.

Relevant columns:

- `run_name`
- `actor`
- `distinguisher`
- `num_trials`
- `turn_limit`
- `specimen_turn_limit`
- `condition`
- `target_model`

Extra columns are ignored.

Column-to-flag mapping:

- `actor` -> `--actor`
- `distinguisher` -> `--distinguisher`
- `condition` -> `--condition`
- `target_model` -> `--target-model`
- `num_trials` -> `--num-trials`
- `turn_limit` -> `--turn-limit`
- `specimen_turn_limit` -> `--specimen-turn-limit`
- `run_name` -> `--run-name`

Validation rules:

- `specimen_turn_limit` is required only for `yes_query_no_explanation`
- `target_model` is required only for `no_query_trusted_distinguisher`
- `actor` and `distinguisher` must match for `no_prompt_self_pair`
- for other conditions, `specimen_turn_limit` may be blank or omitted
- for other conditions, `target_model` may be blank or omitted
- `actor` and `distinguisher` must be valid model nicknames from [turing_eval/models.py](turing_eval/models.py)
- `target_model`, when provided, must be a valid model nickname from [turing_eval/models.py](turing_eval/models.py)
- `condition` must be one of `no_query_no_explanation`, `no_query_no_explanation_explicit_model`, `no_prompt_self_pair`, `yes_query_no_explanation`, `no_query_yes_explanation`, or `no_query_trusted_distinguisher`

### Batch-level flags

These remain CLI flags on `run_experiment_batch.py` and apply to every row in the CSV:

- `--output-dir`
- `--actor-backend`
- `--distinguisher-backend`
- `--actor-api-provider`
- `--distinguisher-api-provider`
- `--max-new-tokens`
- `--temperature`
- `--max-attempts`
- `--verbose`

### Example CSV

```csv
run_name,actor,distinguisher,target_model,num_trials,turn_limit,specimen_turn_limit,condition,notes
smoke_a,llama-3.3-8b,ministral-8b-2512,,2,20,,no_query_no_explanation,extra columns are ignored
smoke_b,ministral-8b-2512,qwen3-32b,,3,15,8,yes_query_no_explanation,this row needs specimen_turn_limit
smoke_c,llama-3.3-8b,ministral-8b-2512,gemini-3-pro,2,20,,no_query_trusted_distinguisher,this row needs target_model
```

### Example invocation

```bash
./scripts/run-in-container.sh python run_experiment_batch.py \
  --csv experiments.csv \
  --output-dir results/batch_smoke \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --max-attempts 12 \
  --verbose
```

Behavior:

- one subprocess call to `run_experiment.py` is made per CSV row
- each row writes its own run directory under `--output-dir`
- the batch runner continues past failed rows and reports a summary at the end
