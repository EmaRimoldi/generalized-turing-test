# Turing ordering visualizer (analytics + data build)

This package implements the quantities and thresholding described in **`Turing_Ordering_Visualizer_Formalization_English.md`** (repository root). The interactive UI lives in **`../epsilon_visualizer_app/`**.

## Data model (short)

Per **protocol** π (here: CSV `condition` plus harness filters), and ordered pair \((A,B)\):

| Quantity | Role |
|----------|------|
| **Observed (after filters)** | Trial counts and branch-wise frequencies: `pSameRealB` (self-pair \(B,B\), fraction answer 1) and `pDiffImpostor` (pair \(A,B\), fraction answer 0), matching \(q_\pi^B(B)\) and \(r_\pi^B(A,B)\) in the formalization. |
| **Derived** | \(\hat P(\text{correct}) = \tfrac12 q + \tfrac12 r\), \(\hat d = \hat P - \tfrac12\), \(\hat\varepsilon^\* = \max(0,\hat d)\). |
| **Edge rule** | **Supported** \(A \to B\) at ε when \(\hat d = \hat P(\text{correct}) - \tfrac12 \le \varepsilon\) (and both branches meet `min_valid_per_branch`). See `thresholding.py`. |

The visualizer is strict-only: it joins genuine and imitation branches per the markdown. If those branches cannot be computed, the affected cells stay **missing** instead of falling back to a proxy.

**Missing data:** Absent runs, insufficient trials, or missing self-pair for \(B\) in strict mode produce **missing** cells — they are **not** treated as “no edge”.

## Input: `results.csv`

The repo may include **`RESULTS/*/results.csv`** (per-trial `*.json` under `RESULTS/` is gitignored as large). Trials are read from `results_root/*/results.csv` (see `discover_results_csvs`). Expected columns include:

- `actor_model`, `distinguisher_model`, `condition`
- `answer` ∈ {`0`,`1`}
- `success` (`true` / `false`)
- `error` (optional), `turns`, `specimen_turns`, `transcript_path` (optional)

Row validity and exclusions follow `IngestConfig` (`exclude_answered_in_opening`, `exclude_any_error`).

## Build the JSON artifact for the web UI

From the `llm-eval` repo root (with venv activated if you use one):

```bash
python3 scripts/build_epsilon_visualizer_data.py
```

By default this reads **`epsilon_visualizer_app/public/condition_a_model_pair_results.json`** (grouped Condition A JSON) and writes **`epsilon_visualizer_app/public/epsilon_visualizer_data.json`**. Copy that file into **`epsilon_visualizer_app/dist/`** after a rebuild, or run `npm run build` in `epsilon_visualizer_app/`, which bundles assets from `public/`.

**View the built app without Node:** from repo root, `python3 scripts/serve_epsilon_visualizer.py`, then open http://127.0.0.1:8765/

Pair rows include `dHat`, `epsilonStarHat`, branch diagnostics, and `formalismRef` pointing to the markdown document.

## Python analytics API

Importable helpers:

- **`ingest`**: load trials, compute `PairEvidence`, serialize artifact rows.
- **`transitivity`**: transitivity violations \(TV_\varepsilon\), SCC condensation heights, strict pairs.
- **`thresholding`**: plug-in \(d \le \varepsilon\) pair classification.

Example:

```bash
python -m pytest tests/test_transitivity.py
```

## How ε is interpreted

User-chosen **ε ≥ 0**: larger ε makes the **supported** relation **more permissive** (easier for \(A \succeq_{\varepsilon,\pi} B\)), because the rule is **distinguisher advantage small enough**: \(\hat d = \hat P(\text{correct}) - \tfrac12 \le \varepsilon\).

## Web UI: raw graph vs SCC buckets

- **Raw model graph:** Nodes are models; **supported** edges are **directed** as in \(G_{\varepsilon,\pi}\). Cycles may appear; layout uses **dagre** (layered), which is heuristic when the graph is not a DAG.
- **SCC condensation:** Strongly connected components are collapsed using **supported** edges only. Vertical placement uses **DAG height** on the condensation graph.

## Transitivity diagnostic

The UI reports **\(TV_\varepsilon\)**: distinct-model triples \((A,B,C)\) such that \(A \to B\) and \(B \to C\) are **supported** but \(A \to C\) is **not** supported (on the model-level adjacency).

## Grouped JSON vs per-trial CSV

The markdown describes an **ideal** two-branch experiment. The implementation uses **empirical** branch rates from your inputs. For **grouped JSON**, each cell supplies aggregate counts per directed pair; strict mode still requires valid self-pair and imitation counts where applicable. For **per-trial `results.csv`**, strict mode requires a **self-pair** run for target \(B\) to estimate the genuine branch; if that is missing, the strict cell stays **missing** rather than guessing.
