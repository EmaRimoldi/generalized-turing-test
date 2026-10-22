#!/usr/bin/env bash
# Run a CSV batch in tmux. Usage: run_batch_in_tmux.sh schedule.csv [output-dir]
# Install requirements.txt first; uses .venv/bin/python when available.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${TMUX_SESSION_NAME:-gtt-batch}"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 schedule.csv [output-dir]" >&2
    exit 2
fi
CSV_PATH="$1"
OUTPUT_DIR="${2:-results/batch}"
# Resolve caller-relative paths before the detached process changes directory.
CSV_PATH="$(cd "$(dirname "${CSV_PATH}")" && pwd)/$(basename "${CSV_PATH}")"
if [[ ! -f ${CSV_PATH} ]]; then echo "CSV not found: ${CSV_PATH}" >&2; exit 1; fi
if [[ ${OUTPUT_DIR} != /* ]]; then OUTPUT_DIR="${PWD}/${OUTPUT_DIR}"; fi
PYTHON_BIN="${PYTHON:-python3}"
if [[ -z ${PYTHON:-} && -x ${REPO_ROOT}/.venv/bin/python ]]; then PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"; fi
mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/batch-${SESSION}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session already exists: tmux attach -t ${SESSION}" >&2
    exit 1
fi
ENV_COMMAND=""
if [[ -f ${REPO_ROOT}/.env ]]; then
    printf -v ENV_COMMAND 'set -a && source %q && set +a && ' "${REPO_ROOT}/.env"
fi
# Quote every external value before passing the command through tmux's shell.
printf -v RUN_COMMAND 'cd %q && %s%q -u %q --csv %q --output-dir %q --verbose 2>&1 | tee -a %q' \
    "${REPO_ROOT}" "${ENV_COMMAND}" "${PYTHON_BIN}" "${REPO_ROOT}/run_experiment_batch.py" \
    "${CSV_PATH}" "${OUTPUT_DIR}" "${LOG_FILE}"
tmux new-session -d -s "${SESSION}" bash -o pipefail -c "${RUN_COMMAND}"
echo "Started ${SESSION}; attach with: tmux attach -t ${SESSION}"
echo "Log: ${LOG_FILE}"
