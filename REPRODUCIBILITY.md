# Reproducing Experiments

## Environment
The API experiment runner uses the Apptainer environment defined in
`apptainer/turing-eval.def`; Python dependencies are also listed in
`requirements.txt`. Start from the current default branch:

    git clone https://github.com/EmaRimoldi/generalized-turing-test.git
    cd generalized-turing-test
    cp .env.example .env
    # Fill in your real API keys in .env
    ./scripts/build-container.sh

All subsequent commands run via:

    ./scripts/run-in-container.sh <command>

## Replaying a past experiment
Every transcript JSON file contains an `environment` block with:
- Python version
- Git commit hash of the code that produced the transcript
- Apptainer container SHA-256
- Hostname

To replay a trial:

    git checkout <commit-from-transcript>
    ./scripts/build-container.sh
    ./scripts/run-in-container.sh python run_experiment.py \
      --actor <from-transcript> \
      --distinguisher <from-transcript> \
      --condition <from-transcript> \
      --num-trials 1 \
      --output-dir results/replay

Note that API-backed models are not bit-exactly reproducible because providers
may update model routing or quantization over time. The transcript's
`first_message` and full `distinguisher_messages` fields give you the exact
inputs; the variance you see on replay is itself a datum worth recording.

## Rebuilding the visualizer offline

The tracked grouped JSON files for Conditions A and B are the inputs to the
published visualizer. With Python 3.11 and Node.js 20 installed, run:

```bash
npm --prefix visualizer/web ci
npm --prefix visualizer/web run build
python3 visualizer/scripts/serve_epsilon_visualizer.py
```

After dependencies are installed, the data generation and build run locally;
they do not call model APIs or require the original Drive folders. The build
regenerates `visualizer/web/public/epsilon_visualizer_data.json` from both
conditions, including the current 162 directed pairs, and produces an ignored
`visualizer/web/dist/` directory. A regression test checks the complete artifact
against the tracked snapshot. CI also checks that the frontend build leaves
that snapshot unchanged.

This reproduces the visualizer from its grouped inputs. Regenerating those
grouped inputs from original trials is a separate step that requires the
original experiment data and its aggregation policy. See the
[visualizer guide](visualizer/README.md) for the data layout and advanced tools.
