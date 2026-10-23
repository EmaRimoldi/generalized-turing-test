#!/usr/bin/env python3
"""Run an explicit batch of experiments described by a CSV file."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from turing_eval.cli import (
    BACKEND_OPTIONS,
    CONDITION_OPTIONS,
    CONDITIONS,
    EXPERIMENT_SCRIPT,
    add_backend_arguments,
    cli_arguments,
    validate_condition_options,
)
from turing_eval.models import MODEL_REGISTRY

REQUIRED_COLUMNS = {
    "run_name",
    "actor",
    "distinguisher",
    "num_trials",
    "turn_limit",
    "condition",
}


@dataclass
class BatchExperiment:
    run_name: str
    actor: str
    distinguisher: str
    condition: str
    target_model: str | None
    num_trials: int
    turn_limit: int
    specimen_turn_limit: int | None
    actor_hint_model: str | None
    num_queries: int | None
    num_turns: int | None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _require_field(row: dict[str, str], column: str, row_number: int) -> str:
    value = _clean(row.get(column))
    if value is None:
        raise ValueError(f"CSV row {row_number}: missing required value for {column!r}")
    return value


def _parse_int_field(
    row: dict[str, str], column: str, row_number: int, *, required: bool
) -> int | None:
    value = _clean(row.get(column))
    if value is None:
        if required:
            raise ValueError(f"CSV row {row_number}: missing required integer value for {column!r}")
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: invalid integer for {column!r}: {value!r}"
        ) from exc


def _parse_row(row: dict[str, str], row_number: int) -> BatchExperiment:
    condition = _require_field(row, "condition", row_number)
    if condition not in CONDITIONS:
        raise ValueError(
            f"CSV row {row_number}: invalid condition {condition!r}. Expected one of {sorted(CONDITIONS)}"
        )

    actor = _require_field(row, "actor", row_number)
    if actor not in MODEL_REGISTRY:
        raise ValueError(f"CSV row {row_number}: unknown actor model {actor!r}")

    distinguisher = _require_field(row, "distinguisher", row_number)
    if distinguisher not in MODEL_REGISTRY:
        raise ValueError(f"CSV row {row_number}: unknown distinguisher model {distinguisher!r}")
    target_model = _clean(row.get("target_model"))
    actor_hint_model = _clean(row.get("actor_hint_model"))
    for name, value in (("target_model", target_model), ("actor_hint_model", actor_hint_model)):
        if value is not None and value not in MODEL_REGISTRY:
            raise ValueError(f"CSV row {row_number}: unknown {name} {value!r}")

    experiment = BatchExperiment(
        run_name=_require_field(row, "run_name", row_number),
        actor=actor,
        distinguisher=distinguisher,
        condition=condition,
        target_model=target_model,
        num_trials=_parse_int_field(row, "num_trials", row_number, required=True),
        turn_limit=_parse_int_field(row, "turn_limit", row_number, required=True),
        specimen_turn_limit=_parse_int_field(
            row, "specimen_turn_limit", row_number, required=False
        ),
        actor_hint_model=actor_hint_model,
        num_queries=_parse_int_field(row, "num_queries", row_number, required=False),
        num_turns=_parse_int_field(row, "num_turns", row_number, required=False),
    )

    try:
        validate_condition_options(experiment)
    except ValueError as exc:
        raise ValueError(f"CSV row {row_number}: {exc}") from exc
    return experiment


def load_experiments(csv_path: Path) -> list[BatchExperiment]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")
        missing_columns = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing_columns)}")
        experiments = [
            _parse_row(row, row_number) for row_number, row in enumerate(reader, start=2)
        ]
    if not experiments:
        raise ValueError("CSV file contains no experiment rows")
    names = [experiment.run_name for experiment in experiments]
    if len(set(names)) != len(names):
        raise ValueError("CSV run_name values must be unique to avoid overwriting results")
    return experiments


def _build_command(args: argparse.Namespace, experiment: BatchExperiment) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        *cli_arguments(
            experiment,
            ("actor", "distinguisher", "condition", "num_trials", "turn_limit", "run_name"),
        ),
        "--output-dir",
        args.output_dir,
        *cli_arguments(experiment, CONDITION_OPTIONS),
        *cli_arguments(args, BACKEND_OPTIONS + ("resume",)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file describing one experiment per row.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume named CSV runs after validation.")
    add_backend_arguments(parser)
    args = parser.parse_args()

    experiments = load_experiments(Path(args.csv))
    existing = [
        experiment.run_name
        for experiment in experiments
        if (Path(args.output_dir) / experiment.run_name).exists()
    ]
    if existing and not args.resume:
        parser.error(
            f"Run directories already exist: {', '.join(existing)}. Choose a new --output-dir."
        )
    print(f"Running {len(experiments)} experiments from {args.csv}")

    failures = []
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[{index}/{len(experiments)}] {experiment.actor} -> {experiment.distinguisher} "
            f"| {experiment.condition} | run_name={experiment.run_name}"
        )
        command = _build_command(args, experiment)
        return_code = subprocess.run(command, check=False).returncode
        if return_code != 0:
            failures.append((experiment.run_name, return_code))
            print(f"  FAILED rc={return_code} - continuing")

    print(f"\nDone. {len(experiments) - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        for run_name, return_code in failures:
            print(f"  - {run_name}: rc={return_code}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
