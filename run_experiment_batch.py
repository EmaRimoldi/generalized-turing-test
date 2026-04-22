#!/usr/bin/env python3
"""Run an explicit batch of experiments described by a CSV file."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from turing_eval.backends.api_backend import PROVIDERS
from turing_eval.models import MODEL_REGISTRY

CONDITIONS = {
    "no_query_no_explanation",
    "no_query_no_explanation_explicit_model",
    "no_query_no_explanation_50_50_prior",
    "no_query_no_explanation_actor_model_hint",
    "no_prompt_self_pair",
    "yes_query_no_explanation",
    "yes_query_no_explanation_grand",
    "no_query_yes_explanation",
    "no_query_trusted_distinguisher",
}

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


def _parse_int_field(row: dict[str, str], column: str, row_number: int, *, required: bool) -> int | None:
    value = _clean(row.get(column))
    if value is None:
        if required:
            raise ValueError(f"CSV row {row_number}: missing required integer value for {column!r}")
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"CSV row {row_number}: invalid integer for {column!r}: {value!r}") from exc


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
    if condition == "no_prompt_self_pair" and actor != distinguisher:
        raise ValueError(
            f"CSV row {row_number}: condition 'no_prompt_self_pair' requires actor and distinguisher to match"
        )

    actor_hint_model = _clean(row.get("actor_hint_model"))
    if actor_hint_model is not None:
        if actor_hint_model not in MODEL_REGISTRY:
            raise ValueError(f"CSV row {row_number}: unknown actor_hint_model {actor_hint_model!r}")
        if condition != "no_query_no_explanation_actor_model_hint":
            raise ValueError(
                f"CSV row {row_number}: actor_hint_model is only valid for "
                "no_query_no_explanation_actor_model_hint"
            )

    specimen_required = condition in {"yes_query_no_explanation", "yes_query_no_explanation_grand"}
    target_model_required = condition == "no_query_trusted_distinguisher"
    specimen_turn_limit = _parse_int_field(
        row,
        "specimen_turn_limit",
        row_number,
        required=specimen_required,
    )
    target_model = _clean(row.get("target_model"))
    if target_model_required and target_model is None:
        raise ValueError(f"CSV row {row_number}: missing required value for 'target_model'")
    if target_model is not None and target_model not in MODEL_REGISTRY:
        raise ValueError(f"CSV row {row_number}: unknown target model {target_model!r}")
    if not target_model_required:
        target_model = None

    return BatchExperiment(
        run_name=_require_field(row, "run_name", row_number),
        actor=actor,
        distinguisher=distinguisher,
        condition=condition,
        target_model=target_model,
        num_trials=_parse_int_field(row, "num_trials", row_number, required=True),
        turn_limit=_parse_int_field(row, "turn_limit", row_number, required=True),
        specimen_turn_limit=specimen_turn_limit,
        actor_hint_model=actor_hint_model,
    )


def load_experiments(csv_path: Path) -> list[BatchExperiment]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")
        missing_columns = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing_columns)}")
        experiments = [_parse_row(row, row_number) for row_number, row in enumerate(reader, start=2)]
    if not experiments:
        raise ValueError("CSV file contains no experiment rows")
    return experiments


def _build_command(args: argparse.Namespace, experiment: BatchExperiment) -> list[str]:
    command = [
        sys.executable,
        "run_experiment.py",
        "--actor",
        experiment.actor,
        "--distinguisher",
        experiment.distinguisher,
        "--condition",
        experiment.condition,
        "--num-trials",
        str(experiment.num_trials),
        "--turn-limit",
        str(experiment.turn_limit),
        "--output-dir",
        args.output_dir,
        "--run-name",
        experiment.run_name,
    ]
    if experiment.specimen_turn_limit is not None:
        command.extend(["--specimen-turn-limit", str(experiment.specimen_turn_limit)])
    if experiment.target_model is not None:
        command.extend(["--target-model", experiment.target_model])
    if args.verbose:
        command.append("--verbose")
    if args.actor_backend is not None:
        command.extend(["--actor-backend", args.actor_backend])
    if args.distinguisher_backend is not None:
        command.extend(["--distinguisher-backend", args.distinguisher_backend])
    if args.actor_api_provider is not None:
        command.extend(["--actor-api-provider", args.actor_api_provider])
    if args.distinguisher_api_provider is not None:
        command.extend(["--distinguisher-api-provider", args.distinguisher_api_provider])
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if args.max_attempts is not None:
        command.extend(["--max-attempts", str(args.max_attempts)])
    if experiment.actor_hint_model is not None:
        command.extend(["--actor-hint-model", experiment.actor_hint_model])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file describing one experiment per row.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--actor-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--distinguisher-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--actor-api-provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--distinguisher-api-provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    experiments = load_experiments(Path(args.csv))
    print(f"Running {len(experiments)} experiments from {args.csv}")

    failures = []
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[{index}/{len(experiments)}] {experiment.actor} -> {experiment.distinguisher} "
            f"| {experiment.condition} | run_name={experiment.run_name}"
        )
        command = _build_command(args, experiment)
        return_code = subprocess.run(command).returncode
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
