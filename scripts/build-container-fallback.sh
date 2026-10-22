#!/usr/bin/env bash
# Fallback: pull the base Python image and install deps into a host-side venv.
# Use this only if build-container.sh fails due to missing --fakeroot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIF_FILE="${REPO_ROOT}/apptainer/turing-eval.sif"
VENV_DIR="${REPO_ROOT}/.venv-container"

mkdir -p "${REPO_ROOT}/apptainer"

if [[ ! -f "${SIF_FILE}" ]]; then
    echo "Pulling base Python 3.11.9 image..."
    apptainer pull "${SIF_FILE}" docker://python:3.11.9-slim-bookworm
fi

echo "Creating host-side venv inside the container interpreter..."
apptainer exec "${SIF_FILE}" python -m venv "${VENV_DIR}"

echo "Installing pinned dependencies..."
apptainer exec "${SIF_FILE}" "${VENV_DIR}/bin/pip" install --upgrade pip==24.2
apptainer exec "${SIF_FILE}" "${VENV_DIR}/bin/pip" install -r "${REPO_ROOT}/requirements.txt"

echo "Fallback setup complete."
echo "Run with: ./scripts/run-in-container.sh /workspace/.venv-container/bin/python <script.py> [args...]"
