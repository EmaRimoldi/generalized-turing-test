# Reproducing Experiments

## Environment
All code runs inside the pinned Apptainer image defined in
`apptainer/turing-eval.def` and `requirements.txt`. To rebuild the exact
environment from a fresh clone:

    git clone https://github.com/suahong/llm-eval.git
    cd llm-eval
    git checkout emanuele/mixed-backends-autonomous
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
