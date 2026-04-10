#!/usr/bin/env bash
# Run a command inside the turing-eval Apptainer image.
# Usage: ./scripts/run-in-container.sh <command> [args...]
# Example: ./scripts/run-in-container.sh pytest tests/ -v

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${REPO_ROOT}/apptainer/turing-eval.sif"

if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Image ${IMAGE} not found. Build it first with:" >&2
    echo "  ./scripts/build-container.sh" >&2
    exit 1
fi

# Load .env if present so API keys flow into the container.
ENV_ARGS=()
if [[ -f "${REPO_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# Forward all provider API keys that are set in the current shell.
for VAR in OPENAI_API_KEY OPENROUTER_API_KEY DEEPSEEK_API_KEY MOONSHOT_API_KEY TOGETHER_API_KEY GROQ_API_KEY MISTRAL_API_KEY; do
    if [[ -n "${!VAR:-}" ]]; then
        ENV_ARGS+=("--env" "${VAR}=${!VAR}")
    fi
done

# --cleanenv: start from an empty environment and only inherit what we explicitly pass.
# --pwd:     set the working directory inside the container.
# --bind:    ensure the repo is visible (this is usually automatic via $HOME, but be explicit).
exec apptainer exec \
    --cleanenv \
    --pwd "/workspace" \
    --bind "${REPO_ROOT}:/workspace" \
    "${ENV_ARGS[@]}" \
    "${IMAGE}" \
    "$@"
