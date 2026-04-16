import csv
import os
import re
from pathlib import Path

from turing_eval.models import MODEL_REGISTRY

CSV_PATH = "0414_experiment_2.csv"
JOB_DIR = Path("job_scripts")
LOG_DIR = Path("logs")
RESULTS_DIR = "results"

PARTITION = "ou_bcs_normal"
TIME = "12:00:00"
MEM = "4G"
CPUS = 1

CONDITION = "no_query_no_explanation"
NUM_TRIALS = 10
TURN_LIMIT = 40
SPECIMEN_TURN_LIMIT = 20

JOB_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def build_model_alias_map() -> dict[str, str]:
    """
    Maps every accepted model string to the canonical nickname:
    - nickname -> nickname
    - display_name -> nickname
    - primary_model_id -> nickname
    """
    alias_map: dict[str, str] = {}

    for nickname, spec in MODEL_REGISTRY.items():
        alias_map[nickname] = nickname

        if spec.display_name:
            alias_map[spec.display_name] = nickname

        if spec.primary_model_id:
            alias_map[spec.primary_model_id] = nickname

        if spec.fallback_model_id:
            alias_map[spec.fallback_model_id] = nickname

        for _, route_model_id in spec.alternate_api_routes.items():
            alias_map[route_model_id] = nickname

    return alias_map

MODEL_ALIAS_MAP = build_model_alias_map()

VALID_CONDITIONS = {
    "no_query_no_explanation",
    "yes_query_no_explanation",
    "no_query_yes_explanation",
    "no_query_trusted_distinguisher",
}


if CONDITION not in VALID_CONDITIONS:
    raise ValueError(f"Unsupported CONDITION {CONDITION!r}. Expected one of {sorted(VALID_CONDITIONS)}")

def normalize_model_name(raw_name: str) -> str:
    name = raw_name.strip()
    normalized = MODEL_ALIAS_MAP.get(name)
    if normalized is None:
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name {name!r}. "
            f"Expected a known nickname or display/model id. "
            f"Known nicknames: {known}"
        )
    return normalized

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        run_name = row["run_name"].strip()
        actor = normalize_model_name(row["actor"])
        distinguisher = normalize_model_name(row["distinguisher"])
        target_model = ""
        if CONDITION == "no_query_trusted_distinguisher":
            raw_target_model = row.get("target_model", "").strip()
            if not raw_target_model:
                raise ValueError(
                    f"Row for run_name={run_name!r} is missing target_model, "
                    "which is required for no_query_trusted_distinguisher"
                )
            target_model = normalize_model_name(raw_target_model)

        job_name = safe_name(run_name)
        script_path = JOB_DIR / f"{job_name}.sh"

        optional_lines: list[str] = []
        if CONDITION == "yes_query_no_explanation":
            optional_lines.append(f'  --specimen-turn-limit {SPECIMEN_TURN_LIMIT} \\\n')
        if CONDITION == "no_query_trusted_distinguisher":
            optional_lines.append(f'  --target-model "{target_model}" \\\n')
        optional_args = "".join(optional_lines)

        script_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {PARTITION}
#SBATCH -t {TIME}
#SBATCH -c {CPUS}
#SBATCH --mem={MEM}
#SBATCH -o {LOG_DIR}/{job_name}-%j.out

set -euo pipefail
module load apptainer/1.4.2

cd /orcd/data/tpoggio/001/om2/suahong/turing-eval/llm-eval

./scripts/run-in-container.sh python run_experiment.py \\
  --actor "{actor}" \\
  --distinguisher "{distinguisher}" \\
  --condition {CONDITION} \\
  --num-trials {NUM_TRIALS} \\
  --turn-limit {TURN_LIMIT} \\
{optional_args}  --run-name "{run_name}" \\
  --output-dir {RESULTS_DIR}
"""

        script_path.write_text(script_text, encoding="utf-8")
        os.chmod(script_path, 0o755)

print(f"Generated job scripts in {JOB_DIR}/")
