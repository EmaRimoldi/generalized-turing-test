#!/usr/bin/env python3
"""
Run multiple OpenRouter single-stage experiments from a CSV definition file.

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
        description="Run run_openrouter_single_stage_experiment.py once per row in a CSV file."
    )
    parser.add_argument("--input-csv", required=True, help="CSV file containing experiment definitions.")
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


def normalize_row(raw_row: Dict[str, str], row_index: int) -> Dict[str, str]:
    row = {key.strip(): (value.strip() if value is not None else "") for key, value in raw_row.items() if key}

    missing = [column for column in REQUIRED_COLUMNS if not row.get(column)]
    if missing:
        raise ValueError(f"Row {row_index}: missing required columns/values: {', '.join(missing)}")

    turn_limit = row.get("turn-limit") or row.get("term-limit")
    if not turn_limit:
        raise ValueError("Row {}: missing required column/value: turn-limit (or term-limit)".format(row_index))

    return {
        "actor-model": row["actor-model"],
        "distinguisher-model": row["distinguisher-model"],
        "num-trials": row["num-trials"],
        "turn-limit": turn_limit,
        "experiment-name": row["experiment-name"],
    }


def build_command(row: Dict[str, str], args: argparse.Namespace) -> List[str]:
    script_path = Path(__file__).with_name("run_openrouter_single_stage_experiment.py")
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
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    failures = 0
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")

        for row_index, raw_row in enumerate(reader, start=2):
            row = normalize_row(raw_row, row_index)
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
