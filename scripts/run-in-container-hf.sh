#!/usr/bin/env bash
# Run a command inside the HF-enabled Apptainer image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${REPO_ROOT}/apptainer/turing-eval-hf.sif"
CACHE_DIR="${HF_CACHE_DIR:-/tmp/${USER:-user}-llm-eval-hf-cache}"

if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Image ${IMAGE} not found. Build it first with:" >&2
    echo "  ./scripts/build-container-hf.sh" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}"

ENV_ARGS=(
    "--env" "HF_HOME=/hf-cache"
    "--env" "TRANSFORMERS_CACHE=/hf-cache"
)
GPU_ARGS=()

if [[ -f "${REPO_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

for VAR in OPENROUTER_API_KEY DEEPSEEK_API_KEY MOONSHOT_API_KEY TOGETHER_API_KEY GROQ_API_KEY MISTRAL_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
    if [[ -n "${!VAR:-}" ]]; then
        ENV_ARGS+=("--env" "${VAR}=${!VAR}")
    fi
done

if compgen -G "/dev/nvidia*" > /dev/null; then
    GPU_ARGS+=("--nv")
fi

exec apptainer exec \
    "${GPU_ARGS[@]}" \
    --cleanenv \
    --pwd "/workspace" \
    --bind "${REPO_ROOT}:/workspace" \
    --bind "${CACHE_DIR}:/hf-cache" \
    "${ENV_ARGS[@]}" \
    "${IMAGE}" \
    "$@"
