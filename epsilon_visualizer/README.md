# Turing ordering visualizer (analytics + data build)

This package implements the quantities and thresholding described in **`Turing_Ordering_Visualizer_Formalization_English.md`** (repository root). The interactive UI lives in **`../epsilon_visualizer_app/`**.

## Data model (short)

Per **protocol** π (here: CSV `condition` plus harness filters), and ordered pair \((A,B)\):

| Quantity | Role |
|----------|------|
| **Observed (after filters)** | Trial counts and branch-wise frequencies: `pSameRealB` (self-pair \(B,B\), fraction answer 1) and `pDiffImpostor` (pair \(A,B\), fraction answer 0), matching \(q_\pi^B(B)\) and \(r_\pi^B(A,B)\) in the formalization. |
| **Derived** | \(\hat P(\text{correct}) = \tfrac12 q + \tfrac12 r\), \(\hat d = \hat P - \tfrac12\), \(\hat\varepsilon^\* = \max(0,\hat d)\). |
| **Edge rule (conservative)** | **Supported** \(A \to B\) at ε when the **Wilson upper** bound on \(\hat P(\text{correct})\) is \(\le \tfrac12 + \varepsilon\) (and both branches meet `min_valid_per_branch`). See `thresholding.py`. |

**Strict vs surrogate:** Strict mode joins genuine and imitation branches per the markdown. Surrogate mode thresholds an imitation-only proxy and is labeled as such in the UI.

**Missing data:** Absent runs, insufficient trials, or missing self-pair for \(B\) in strict mode produce **missing** cells — they are **not** treated as “no edge”.

## Input: `results.csv`

Trials are read from `results_root/*/results.csv` (see `discover_results_csvs`). Expected columns include:

- `actor_model`, `distinguisher_model`, `condition`
- `answer` ∈ {`0`,`1`}
- `success` (`true` / `false`)
- `error` (optional), `turns`, `specimen_turns`, `transcript_path` (optional)

Row validity and exclusions follow `IngestConfig` (`exclude_answered_in_opening`, `exclude_any_error`).

## Build the JSON artifact for the web UI

From the `llm-eval` repo root (with venv activated if you use one):

```bash
python scripts/build_epsilon_visualizer_data.py
```

This writes **`epsilon_visualizer_app/public/epsilon_visualizer_data.json`**, copied into **`epsilon_visualizer_app/dist/`** when you run `npm run build`, and loaded by the UI via a URL relative to the page.

**View the built app without Node:** from repo root, `python3 scripts/serve_epsilon_visualizer.py`, then open http://127.0.0.1:8765/

Pair rows include `dHat`, `epsilonStarHat`, Wilson bounds, and `formalismRef` pointing to the markdown document.

## Python analytics API

Importable helpers:

- **`ingest`**: load trials, compute `PairEvidence`, serialize artifact rows.
- **`transitivity`**: transitivity violations \(TV_\varepsilon\), SCC condensation heights, strict pairs.
- **`stats` / `thresholding`**: Wilson intervals and pair classification.

Example:

```bash
python -m pytest tests/test_transitivity.py
```

## How ε is interpreted

User-chosen **ε ≥ 0**: larger ε makes the **supported** relation **more permissive** (easier for \(A \succeq_{\varepsilon,\pi} B\)), because the rule is **distinguisher advantage small enough**: conservative check \(\text{UB}(\hat P(\text{correct})) \le \tfrac12 + \varepsilon\), consistent with \(d \le \varepsilon\) in the formalization.

## Web UI: raw graph vs SCC buckets

- **Raw model graph:** Nodes are models; **supported** edges are **directed** as in \(G_{\varepsilon,\pi}\). Cycles may appear; layout uses **dagre** (layered), which is heuristic when the graph is not a DAG.
- **SCC condensation:** Strongly connected components are collapsed using **supported** edges only; **tentative** edges are drawn but do not merge clusters. Vertical placement uses **DAG height** on the condensation graph (after optional **transitive reduction** on that DAG only).

## Transitivity diagnostic

The UI reports **\(TV_\varepsilon\)**: triples \((A,B,C)\) such that \(A \to B\) and \(B \to C\) are **supported** but \(A \to C\) is **not** supported (on the model-level adjacency).

## Where theory meets your tables

The markdown describes an **ideal** two-branch experiment. The implementation uses **empirical** branch rates from your CSVs; strict mode requires a **self-pair** run for target \(B\) to estimate the “genuine \(B\)” branch. If that is missing, the strict cell stays **missing** rather than guessing.
