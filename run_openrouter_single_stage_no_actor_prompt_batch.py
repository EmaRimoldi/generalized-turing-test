#!/usr/bin/env python3
"""
Run multiple OpenRouter single-stage no-actor-prompt experiments from a CSV definition file.

Expected columns:
    actor-model
    distinguisher-model
    num-trials
    turn-limit
    experiment-name

For convenience, "term-limit" is also accepted as an alias for "turn-limit".
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REQUIRED_COLUMNS = [
    "actor-model",
    "distinguisher-model",
    "num-trials",
    "experiment-name",
]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_openrouter_single_stage_no_actor_prompt_experiment.py once per row in a CSV file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", help="CSV file containing experiment definitions.")
    input_group.add_argument(
        "--models",
        nargs="+",
        help="List of model IDs to run in self-play mode (actor-model == distinguisher-model).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Used with --models. Number of trials to run for each listed model.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=40,
        help="Used with --models. Turn limit for each listed model.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Optional prefix for experiment names generated from --models.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later rows if one experiment fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Append --verbose to each generated experiment command.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Optional timeout override forwarded to each experiment command.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Forward --skip-model-validation to each experiment command.",
    )
    return parser.parse_args(argv)


def sanitize_model_for_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def normalize_row(raw_row: Dict[str, str], row_index: int) -> Dict[str, str]:
    row = {key.strip(): (value.strip() if value is not None else "") for key, value in raw_row.items() if key}

    missing = [column for column in REQUIRED_COLUMNS if not row.get(column)]
    if missing:
        raise ValueError(f"Row {row_index}: missing required columns/values: {', '.join(missing)}")

    turn_limit = row.get("turn-limit") or row.get("term-limit")
    if not turn_limit:
        raise ValueError(f"Row {row_index}: missing required column/value: turn-limit (or term-limit)")

    return {
        "actor-model": row["actor-model"],
        "distinguisher-model": row["distinguisher-model"],
        "num-trials": row["num-trials"],
        "turn-limit": turn_limit,
        "experiment-name": row["experiment-name"],
    }


def build_rows_from_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    assert args.models is not None
    rows: List[Dict[str, str]] = []
    for model_id in args.models:
        sanitized_model = sanitize_model_for_name(model_id)
        experiment_name = (
            f"{args.experiment_prefix}_{sanitized_model}" if args.experiment_prefix else sanitized_model
        )
        rows.append(
            {
                "actor-model": model_id,
                "distinguisher-model": model_id,
                "num-trials": str(args.num_trials),
                "turn-limit": str(args.turn_limit),
                "experiment-name": experiment_name,
            }
        )
    return rows


def build_command(row: Dict[str, str], args: argparse.Namespace) -> List[str]:
    script_path = Path(__file__).with_name("run_openrouter_single_stage_no_actor_prompt_experiment.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--actor-model",
        row["actor-model"],
        "--distinguisher-model",
        row["distinguisher-model"],
        "--num-trials",
        row["num-trials"],
        "--turn-limit",
        row["turn-limit"],
        "--experiment-name",
        row["experiment-name"],
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.request_timeout is not None:
        cmd.extend(["--request-timeout", str(args.request_timeout)])
    if args.skip_model_validation:
        cmd.append("--skip-model-validation")
    return cmd


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    failures = 0
    rows: List[Dict[str, str]]
    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        with input_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Input CSV is missing a header row.")
            rows = [normalize_row(raw_row, row_index) for row_index, raw_row in enumerate(reader, start=2)]
    else:
        rows = build_rows_from_models(args)

    for row_index, row in enumerate(rows, start=2):
        cmd = build_command(row, args)
        print(f"\n=== Row {row_index} | {row['experiment-name']} ===")
        print(" ".join(subprocess.list2cmdline([part]) for part in cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if completed.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                return completed.returncode

    if failures:
        print(f"\nCompleted with {failures} failed experiment(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
