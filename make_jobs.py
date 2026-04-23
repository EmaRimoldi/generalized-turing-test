import argparse
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

DEFAULT_CONDITION = "no_query_no_explanation"
DEFAULT_NUM_TRIALS = 10
DEFAULT_TURN_LIMIT = 40
DEFAULT_SPECIMEN_TURN_LIMIT = 20


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
    "no_prompt_self_pair",
    "yes_query_no_explanation",
    "yes_query_trusted_distinguisher",
    "no_query_yes_explanation",
    "no_query_trusted_distinguisher",
    "50_50_trusted_distinguisher",
}


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


def clean_row_value(row: dict[str, str], column: str) -> str | None:
    value = row.get(column)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def row_value_or_default(row: dict[str, str], column: str, default: str | int) -> str:
    value = clean_row_value(row, column)
    return str(default) if value is None else value


def parse_row_int(row: dict[str, str], column: str, default: int, run_name: str) -> int:
    raw_value = row_value_or_default(row, column, default)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Row for run_name={run_name!r} has invalid integer for {column!r}: {raw_value!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Slurm job scripts from an experiment CSV."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="CSV file describing one job per row.")
    parser.add_argument("--job-dir", default=str(JOB_DIR), help="Directory for generated job scripts.")
    parser.add_argument("--log-dir", default=str(LOG_DIR), help="Directory for Slurm log files.")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Experiment output directory.")
    parser.add_argument("--partition", default=PARTITION)
    parser.add_argument("--time", default=TIME)
    parser.add_argument("--mem", default=MEM)
    parser.add_argument("--cpus", type=int, default=CPUS)
    parser.add_argument(
        "--default-condition",
        default=DEFAULT_CONDITION,
        choices=sorted(VALID_CONDITIONS),
        help="Fallback condition when a CSV row omits the condition column/value.",
    )
    parser.add_argument(
        "--default-num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Fallback trial count when a CSV row omits num_trials.",
    )
    parser.add_argument(
        "--default-turn-limit",
        type=int,
        default=DEFAULT_TURN_LIMIT,
        help="Fallback turn limit when a CSV row omits turn_limit.",
    )
    parser.add_argument(
        "--default-specimen-turn-limit",
        type=int,
        default=DEFAULT_SPECIMEN_TURN_LIMIT,
        help="Fallback specimen turn limit for yes-query conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    job_dir = Path(args.job_dir)
    log_dir = Path(args.log_dir)

    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")

        for row in reader:
            write_job_script(row, args, job_dir, log_dir)

    print(f"Generated job scripts in {job_dir}/")


def write_job_script(row: dict[str, str], args: argparse.Namespace, job_dir: Path, log_dir: Path) -> None:
    run_name = row["run_name"].strip()
    actor = normalize_model_name(row["actor"])
    distinguisher = normalize_model_name(row["distinguisher"])

    condition = row_value_or_default(row, "condition", args.default_condition)
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Row for run_name={run_name!r} has unsupported condition {condition!r}. "
            f"Expected one of {sorted(VALID_CONDITIONS)}"
        )
    num_trials = parse_row_int(row, "num_trials", args.default_num_trials, run_name)
    turn_limit = parse_row_int(row, "turn_limit", args.default_turn_limit, run_name)
    specimen_turn_limit = parse_row_int(
        row,
        "specimen_turn_limit",
        args.default_specimen_turn_limit,
        run_name,
    )

    if condition == "no_prompt_self_pair" and actor != distinguisher:
        raise ValueError(
            f"Row for run_name={run_name!r} must use the same actor and distinguisher "
            "for no_prompt_self_pair"
        )
    target_model = ""
    if condition in {
        "no_query_trusted_distinguisher",
        "50_50_trusted_distinguisher",
        "yes_query_trusted_distinguisher",
    }:
        raw_target_model = row.get("target_model", "").strip()
        if not raw_target_model:
            raise ValueError(
                f"Row for run_name={run_name!r} is missing target_model, "
                f"which is required for {condition}"
            )
        target_model = normalize_model_name(raw_target_model)

    job_name = safe_name(run_name)
    script_path = job_dir / f"{job_name}.sh"

    optional_lines: list[str] = []
    if condition in {"yes_query_no_explanation", "yes_query_trusted_distinguisher"}:
        optional_lines.append(f'  --specimen-turn-limit {specimen_turn_limit} \\\n')
    if condition in {
        "no_query_trusted_distinguisher",
        "50_50_trusted_distinguisher",
        "yes_query_trusted_distinguisher",
    }:
        optional_lines.append(f'  --target-model "{target_model}" \\\n')
    optional_args = "".join(optional_lines)

    script_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH -c {args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -o {log_dir}/{job_name}-%j.out

set -euo pipefail
module load apptainer/1.4.2

cd /orcd/data/tpoggio/001/om2/suahong/turing-eval/llm-eval

./scripts/run-in-container.sh python run_experiment.py \\
  --actor "{actor}" \\
  --distinguisher "{distinguisher}" \\
  --condition {condition} \\
  --num-trials {num_trials} \\
  --turn-limit {turn_limit} \\
{optional_args}  --run-name "{run_name}" \\
  --output-dir {args.results_dir}
"""

    script_path.write_text(script_text, encoding="utf-8")
    os.chmod(script_path, 0o755)


if __name__ == "__main__":
    main()
