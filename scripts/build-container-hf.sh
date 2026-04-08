#!/usr/bin/env bash
# Build the HF-enabled Apptainer image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEF_FILE="${REPO_ROOT}/apptainer/turing-eval-hf.def"
IMAGE_LINK="${REPO_ROOT}/apptainer/turing-eval-hf.sif"
IMAGE_STORE_DIR="${APPTAINER_IMAGE_DIR:-/tmp/llm-eval-apptainer}"
SIF_FILE="${IMAGE_STORE_DIR}/turing-eval-hf.sif"

if [[ ! -f "${DEF_FILE}" ]]; then
    echo "ERROR: Definition file ${DEF_FILE} not found." >&2
    exit 1
fi

mkdir -p "${IMAGE_STORE_DIR}"
if [[ -e "${IMAGE_LINK}" && ! -L "${IMAGE_LINK}" ]]; then
    rm -f "${IMAGE_LINK}"
fi
ln -sfn "${SIF_FILE}" "${IMAGE_LINK}"
rm -f "${SIF_FILE}"

echo "Building ${SIF_FILE} from ${DEF_FILE}..."

if (
    cd "${REPO_ROOT}"
    apptainer build --fakeroot "${SIF_FILE}" "apptainer/turing-eval-hf.def"
); then
    echo "HF build successful via --fakeroot."
    exit 0
fi
build_rc=$?
if [[ ${build_rc} -eq 130 ]]; then
    echo "HF build interrupted." >&2
    exit 130
fi

echo "WARNING: HF --fakeroot build failed. Trying without..." >&2
if (
    cd "${REPO_ROOT}"
    apptainer build "${SIF_FILE}" "apptainer/turing-eval-hf.def"
); then
    echo "HF build successful without --fakeroot."
    exit 0
fi
build_rc=$?
if [[ ${build_rc} -eq 130 ]]; then
    echo "HF build interrupted." >&2
    exit 130
fi

echo "ERROR: HF image build failed." >&2
exit 1
