#!/usr/bin/env bash
# Avvia run_experiment_batch.py in una sessione tmux staccata: sopravvive a chiusura SSH / laptop.
#
# Uso (dalla root del repo):
#   ./scripts/run_batch_in_tmux.sh
#   ./scripts/run_batch_in_tmux.sh batch_configs/matrix_B_resume_no_gptpro.csv
#
# Variabili opzionali:
#   TMUX_SESSION_NAME=my-session  (default: llm-eval-batch)
#
# Rientrare nel log live:
#   tmux attach -t llm-eval-batch
# Staccare di nuovo: Ctrl+B poi D

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SESSION="${TMUX_SESSION_NAME:-llm-eval-batch}"
CSV_REL="${1:-batch_configs/matrix_B_resume_no_gptpro.csv}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/batch-${SESSION}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux: la sessione '${SESSION}' esiste già." >&2
  echo "  tmux attach -t ${SESSION}   # per vedere l'output" >&2
  echo "  tmux kill-session -t ${SESSION}   # solo se vuoi terminarla" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  echo "Manca ${REPO_ROOT}/.env (chiavi API)." >&2
  exit 1
fi

# Una sola riga per bash -lc (evita problemi di quoting con tmux).
tmux new-session -d -s "${SESSION}" \
  bash -lc "cd '${REPO_ROOT}' && set -a && . ./.env && set +a && export PYTHONUNBUFFERED=1 && exec uv run python run_experiment_batch.py --csv '${CSV_REL}' --output-dir RESULTS --verbose 2>&1 | tee -a '${LOG_FILE}'"

echo "Avviato tmux session: ${SESSION}"
echo "  Log file: ${LOG_FILE}"
echo "  Rientra con: tmux attach -t ${SESSION}"
echo "  Stacca con:  Ctrl+B poi D"
