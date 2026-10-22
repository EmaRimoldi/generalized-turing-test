#!/usr/bin/env bash
# Build the API image, or pass --hf for the local Hugging Face image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME=turing-eval
if [[ ${1:-} == --hf && $# -eq 1 ]]; then
    IMAGE_NAME=turing-eval-hf
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--hf]" >&2
    exit 2
fi
IMAGE_STORE_DIR="${APPTAINER_IMAGE_DIR:-/tmp/llm-eval-apptainer}"
mkdir -p "${IMAGE_STORE_DIR}"
IMAGE_STORE_DIR="$(cd "${IMAGE_STORE_DIR}" && pwd)"
SIF_FILE="${IMAGE_STORE_DIR}/${IMAGE_NAME}.sif"
BUILD_FILE="${SIF_FILE}.building.$$"
trap 'rm -f "${BUILD_FILE}"' EXIT

for mode in fakeroot unprivileged; do
    set --
    if [[ ${mode} == fakeroot ]]; then set -- --fakeroot; fi
    echo "Building ${IMAGE_NAME} (${mode})..."
    if (cd "${REPO_ROOT}" && apptainer build "$@" "${BUILD_FILE}" "apptainer/${IMAGE_NAME}.def"); then
        mv -f "${BUILD_FILE}" "${SIF_FILE}"
        ln -sfn "${SIF_FILE}" "${REPO_ROOT}/apptainer/${IMAGE_NAME}.sif"
        echo "Built ${SIF_FILE}"
        exit 0
    else
        build_rc=$?
    fi
    if [[ ${build_rc} -eq 130 ]]; then exit 130; fi
    rm -f "${BUILD_FILE}"
done

echo "ERROR: Both Apptainer build attempts failed; the previous image was preserved." >&2
echo "See REPRODUCIBILITY.md for the local Python environment and fallback instructions." >&2
exit 1
