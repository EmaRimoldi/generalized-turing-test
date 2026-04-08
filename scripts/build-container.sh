#!/usr/bin/env bash
# Build the turing-eval Apptainer image.
# Tries --fakeroot first; falls back to a helpful error if not available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEF_FILE="${REPO_ROOT}/apptainer/turing-eval.def"
SIF_FILE="${REPO_ROOT}/apptainer/turing-eval.sif"

if [[ ! -f "${DEF_FILE}" ]]; then
    echo "ERROR: Definition file ${DEF_FILE} not found." >&2
    exit 1
fi

echo "Building ${SIF_FILE} from ${DEF_FILE}..."

# Try --fakeroot first (works on most modern HPC clusters).
if apptainer build --fakeroot "${SIF_FILE}" "${DEF_FILE}"; then
    echo "Build successful via --fakeroot."
    exit 0
fi

echo "WARNING: --fakeroot build failed. Trying without..." >&2
if apptainer build "${SIF_FILE}" "${DEF_FILE}"; then
    echo "Build successful without --fakeroot."
    exit 0
fi

cat >&2 <<ERR
ERROR: Both build attempts failed.

On this cluster, you may need to build the image differently:
  1. Ask the cluster admin to enable apptainer fakeroot for your user, OR
  2. Build on a machine where you have sudo and copy the .sif to the cluster, OR
  3. Use a pre-built base image without custom layers (pull approach):
       apptainer pull apptainer/turing-eval.sif docker://python:3.11.9-slim-bookworm
     Then install pip dependencies into a venv at runtime (slower startup).

See the Apptainer override document for details on the fallback venv workflow.
ERR
exit 1
