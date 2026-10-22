# Turing ordering visualizer (epsilon graph)

This folder contains the Condition **A+B** epsilon / transitivity visualizer:
Python analytics, the Vite web UI, build/serve scripts, and formalization notes.

## Layout and data ownership

| Path | Role |
|------|------|
| `docs/` | Formalization and implementation notes |
| `python/epsilon_visualizer/` | Ingest, thresholding, condensation graph, and transitivity analytics |
| `web/public/condition_a_model_pair_results.json` | Tracked grouped input for A (`no_query_no_explanation`) |
| `web/public/condition_b_model_pair_results.json` | Tracked grouped input for B (`yes_query_no_explanation`) |
| `web/public/epsilon_visualizer_data.json` | Tracked snapshot of the combined artifact consumed by the UI |
| `web/public/exports/` | Tracked figure exports; retained independently of the compiled app |
| `web/src/` | Frontend source |
| `web/dist/` | Ignored, generated static app; rebuilt locally and in deployment CI |
| `scripts/` | Data generation, aggregation, plotting, and serving tools |

Imports remain `from epsilon_visualizer...` with `visualizer/python` on
`PYTHONPATH` (see repository-root `pytest.ini`).

## Build and serve

Requirements: Python 3.11, Node.js 20, and npm. From the repository root:

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

## Advanced and historical tools

- `build_epsilon_visualizer_data.py` supports single-condition builds and
  alternate CSV/trial inputs. Use `--out` to a separate path when exploring;
  the published app uses the combined A+B generator above.
- `aggregate_condition_a_grouped_json.py`,
  `aggregate_condition_b_grouped_json.py`, and
  `build_grouped_condition_json_from_results.py` create grouped inputs from
  original data. They retain their existing run-selection and filtering rules.
  Check the intended protocol before regenerating scientific inputs.
- `plot_grouped_json_pair_coverage.py` inspects the grouped input coverage.
- `serve_epsilon_visualizer.py` serves the local production build without Node
  at serving time; Node is needed to build it first.

## Validation and deployment

Run the offline Python suite from the repository root:

```bash
python3 -m pytest tests/ -q
```

`tests/test_visualizer_build.py` regenerates the A+B artifact outside the
repository and compares it with the tracked snapshot, including all 162 pairs.
`.github/workflows/validate.yml` runs the tests, builds the frontend, and checks
that regeneration does not change the snapshot.

GitHub Pages deployment is configured in
`.github/workflows/deploy-epsilon-visualizer.yml`. It builds from source with the
same npm command and uploads the resulting `web/dist/` artifact.

Automatic Pages deployment is opt-in: configure GitHub Pages to use GitHub
Actions, then set the repository variable `ENABLE_VISUALIZER_PAGES` to `true`.
The deployment workflow can also be run manually after Pages is configured.
Tests and frontend build validation run independently of this setting.
