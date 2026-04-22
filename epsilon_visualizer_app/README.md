# Epsilon visualizer (web UI)

React + Vite UI for **Turing ordering** at a chosen threshold ε. Formal definitions live in **`../Turing_Ordering_Visualizer_Formalization_English.md`**.

## Data source (Condition A)

The shipped dataset is built only from:

- **`public/condition_a_model_pair_results.json`** — grouped model-pair aggregates (`model_pair_results_grouped` schema).

Do not rely on spreadsheet tables in this repo. After you edit that JSON (or replace it), regenerate the artifact the app loads:

```bash
# from repository root
python3 scripts/build_epsilon_visualizer_data.py
```

This writes **`public/epsilon_visualizer_data.json`** (and you should keep **`dist/epsilon_visualizer_data.json`** in sync for the static server—see below).

Optional flags:

- `--grouped-no-impute` — fail if any directed pair in `ordering.model_order` is missing from the JSON (no synthetic 5/5 fill).
- `--grouped-matrix-fallback PATH.csv` — only if you intentionally want extra measured cells merged in (not used by default).

## Run the static build (no Node required for serving)

If **`dist/`** is present:

```bash
cd /path/to/llm-eval
python3 scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/** . After rebuilding `epsilon_visualizer_data.json`, copy it into `dist/` or re-run `npm run build` so the served bundle stays current:

```bash
cp epsilon_visualizer_app/public/epsilon_visualizer_data.json epsilon_visualizer_app/dist/epsilon_visualizer_data.json
```

## Run with Vite (development)

```bash
cd epsilon_visualizer_app
npm install
npm run dev
```

Use the URL Vite prints (typically **http://localhost:5173**). The dev server reads JSON from **`public/`** via `fetch`.

## Per-trial results (optional)

To build the artifact from raw `results.csv` trees instead of the grouped JSON:

```bash
python3 scripts/build_epsilon_visualizer_data.py --results-root RESULTS
```

Theory and Python ingest modules: **`../epsilon_visualizer/README.md`**.
