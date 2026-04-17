# Epsilon visualizer (web UI)

React + Vite app for **Turing ordering** at a user-chosen threshold ε. Theory and CSV ingest are documented in **`../epsilon_visualizer/README.md`** and **`../Turing_Ordering_Visualizer_Formalization_English.md`**.

## Offline on your laptop (recommended)

No cluster, no `npm` required if **`dist/`** is already in the repo:

```bash
git clone <your-repo-url>
cd llm-eval
python3 scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/** in your browser (Chrome/Firefox/Safari). Use **Python 3.6+** (any recent `python3` on macOS/Linux is fine).

To **refresh the JSON** from checked-in **`RESULTS/*/results.csv`** after a `git pull`:

```bash
python3 scripts/build_epsilon_visualizer_data.py
cp epsilon_visualizer_app/public/epsilon_visualizer_data.json epsilon_visualizer_app/dist/epsilon_visualizer_data.json
```

Then reload the page (or restart the server). The build script needs a **Python 3.7+** environment if your ingest code uses modern typing (use a venv on your machine if the cluster had only 3.6).

## View without `npm` (cluster-friendly)

From **`llm-eval` repo root** (only Python stdlib):

```bash
python3 scripts/serve_epsilon_visualizer.py
```

Open **http://127.0.0.1:8765/**. This serves the pre-built **`dist/`** folder (JS + `epsilon_visualizer_data.json`).

If `dist/` is missing or stale, build once on a machine with Node: `npm install && npm run build` in this directory, then copy `dist/` back.

## Run with Vite (development)

1. Ensure **`public/epsilon_visualizer_data.json`** exists (see `../scripts/build_epsilon_visualizer_data.py`).
2. Install and start:

```bash
cd epsilon_visualizer_app
npm install
npm run dev
```

3. Open the URL Vite prints (typically `http://localhost:5173`).

Use the **ε** slider, **condition** selector (protocol π), and **Graph** dropdown (raw vs SCC condensation). **Transitive reduction** applies only to the **condensation DAG**, not the raw graph (which may contain cycles).

If the JSON fails to load, the UI shows a **message** with the `serve_epsilon_visualizer.py` command (opening `index.html` via `file://` usually breaks `fetch`).
