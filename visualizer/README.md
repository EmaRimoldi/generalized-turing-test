# Turing ordering visualizer (epsilon graph)

This folder contains the Condition **A+B** epsilon / transitivity visualizer:
Python analytics, the Vite web UI, build/serve scripts, and formalization notes.

## Layout and data ownership

| Path | Role |
|------|------|
| `docs/` | Mathematical working note; earlier design documents in `docs/archive/` |
| `python/epsilon_visualizer/` | Ingest, thresholding, condensation graph, and transitivity analytics |
| `web/public/condition_a_model_pair_results.json` | Tracked grouped input for A (`no_query_no_explanation`) |
| `web/public/condition_b_model_pair_results.json` | Tracked grouped input for B (`yes_query_no_explanation`) |
| `web/public/epsilon_visualizer_data.json` | Tracked snapshot of the combined artifact consumed by the UI |
| `web/public/exports/` | Ignored figure exports, generated on demand with `npm run export:triple-png` |
| `web/src/` | Frontend source |
| `web/dist/` | Ignored, generated static app; rebuilt locally and in deployment CI |
| `scripts/` | Data generation, plotting, and serving; older aggregation policies in `scripts/legacy/` |

Imports remain `from epsilon_visualizer...` with `visualizer/python` on
`PYTHONPATH` (see repository-root `pytest.ini`).

## Build and serve

Requirements: Python 3.11, Node.js 22.12+, and npm. From the repository root:

```bash
npm --prefix visualizer/web ci
npm --prefix visualizer/web run build
python3 visualizer/scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/**. `npm run build` runs the shared A+B generator
before Vite copies `public/` into `dist/`. A fresh clone needs this build before
using the Python static server; there is no prebuilt app in Git.

For frontend development:

```bash
npm --prefix visualizer/web run dev
```

Open **http://127.0.0.1:5173/**. The development command also regenerates A+B data
before starting Vite. Restart it after changing the grouped input files.

## Rebuild data only

The canonical command for the combined published artifact is:

```bash
npm --prefix visualizer/web run build:data
```

It invokes `scripts/build_multicondition_visualizer_data.py`, which reads both
tracked grouped JSON files and writes `web/public/epsilon_visualizer_data.json`.
The same command runs automatically before development and production builds.
Its current output contains both conditions and 162 directed pairs. Repository
input references are relative so regeneration is stable across checkouts.

When grouped inputs change, regenerate and commit the snapshot together with
them. Model calls and the original Drive directories are not needed for this
step. Original trial data and grouped inputs remain separate evidence; compiled
copies under `dist/` are disposable.

The grouped files contain an earlier 10-trial-per-pair snapshot. See the
[data inventory](../data/README.md) before interpreting them as paper results.
The PNG export utility preserves its historical model filtering, including
excluding GPT; its exports are exploratory views rather than canonical paper
figures. The browser and grouped inputs retain all nine models.

## Advanced and historical tools

- `build_epsilon_visualizer_data.py` supports single-condition builds and
  alternate CSV/trial inputs. It requires an explicit `--out` path;
  the published app uses the combined A+B generator above.
- `build_grouped_condition_json_from_results.py` creates grouped inputs from
  original trial JSON using the snapshot's historical policy. Review recovered
  opening answers and model routes before regenerating scientific inputs.
  It now reads verdicts only from generated assistant messages and rejects
  failed-trial answers. Historical Grok-2 relabeling requires the explicit
  `--legacy-model-aliases` option; inspect route metadata before using it.
- [`scripts/legacy/`](scripts/legacy/README.md) retains the earlier CSV-based
  aggregators, with different selection and alias policies. They require
  explicit original inputs and should write to separate outputs.
- `--grouped-no-impute` preserves missing cells and the declared model roster.
  Grouped counts must be consistent nonnegative integers; malformed or duplicate
  cells fail before fallback/imputation. Graphs and rankings exclude imputed
  evidence and respect the selected model set and minimum sample count. The
  strict A/B analysis rejects UTT inputs, which need separate target calibration.
- `plot_grouped_json_pair_coverage.py` inspects the grouped input coverage.
- `serve_epsilon_visualizer.py` serves the local production build without Node
  at serving time; Node is needed to build it first.

## Validation and deployment

Run the offline Python suite from the repository root:

```bash
python3 -m ruff check .
python3 -m pytest tests/ -q
npm --prefix visualizer/web test
npm --prefix visualizer/web run build
```

`tests/test_visualizer_build.py` regenerates the A+B artifact outside the
repository and compares it with the tracked snapshot, including all 162 pairs.
`.github/workflows/validate.yml` runs Python correctness checks and tests,
frontend evidence/geometry tests, TypeScript checking, and the production build. It also
checks that regeneration does not change the snapshot. Screen and image exports
share the graph geometry implementation.

GitHub Pages deployment is configured in
`.github/workflows/deploy-epsilon-visualizer.yml`. It builds from source with the
same npm command and uploads the resulting `web/dist/` artifact.

Automatic Pages deployment is opt-in: configure GitHub Pages to use GitHub
Actions, then set the repository variable `ENABLE_VISUALIZER_PAGES` to `true`.
The deployment workflow can also be run manually after Pages is configured.
Tests and frontend build validation run independently of this setting.
