# Turing ordering visualizer (epsilon graph)

This folder contains everything for the Condition **A** epsilon / transitivity visualizer: Python analytics, the Vite web UI, build/serve scripts, and the formalization document.

## Layout

| Path | Role |
|------|------|
| `docs/` | Formalization (`Turing_Ordering_Visualizer_Formalization_English.md`) and implementation notes |
| `python/epsilon_visualizer/` | Package: ingest, thresholding, condensation graph, transitivity |
| `web/` | Frontend (`npm ci`, `npm run build`); static output in `web/dist/` |
| `scripts/` | `build_epsilon_visualizer_data.py`, `serve_epsilon_visualizer.py`, `plot_grouped_json_pair_coverage.py` |

Imports remain `from epsilon_visualizer...` with `visualizer/python` on `PYTHONPATH` (see repo-root `pytest.ini`).

## Typical workflow (from repository root)

Grouped Condition **A** JSON lives at **`web/public/condition_a_model_pair_results.json`**. Rebuild the artifact the UI loads, sync `dist/`, and serve:

```bash
python3 visualizer/scripts/build_epsilon_visualizer_data.py
cp visualizer/web/public/epsilon_visualizer_data.json visualizer/web/dist/epsilon_visualizer_data.json
python3 visualizer/scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/** .

With Node available, after changing `public/` you can instead run `npm ci` and `npm run build` inside **`visualizer/web/`**, which bundles `public/` into `dist/`.

## CI

GitHub Pages deployment is configured in `.github/workflows/deploy-epsilon-visualizer.yml`.
