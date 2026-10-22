#!/usr/bin/env python3
"""Run every ordered actor/distinguisher pairing from a model list."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from turing_eval.cli import (
    BACKEND_OPTIONS,
    CONDITION_OPTIONS,
    CONDITIONS,
    EXPERIMENT_SCRIPT,
    add_backend_arguments,
    add_condition_arguments,
    cli_arguments,
    positive_int,
    validate_condition_options,
)
from turing_eval.models import MODEL_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--num-trials", type=positive_int, default=5)
    parser.add_argument("--turn-limit", type=positive_int, default=40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-self-match", action="store_true")
    add_condition_arguments(parser)
    add_backend_arguments(parser)
    args = parser.parse_args()
    args.actor = args.distinguisher = args.models[0]
    try:
        validate_condition_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")

    if args.condition == "no_prompt_self_pair":
        pairs = [(model, model) for model in args.models]
    else:
        pairs = [
            (actor, judge)
            for actor, judge in itertools.product(args.models, repeat=2)
            if actor != judge or args.include_self_match
        ]
    if not pairs:
        parser.error("No pairs selected; add models or use --include-self-match")

    print(
        f"Running {len(pairs)} pairings x {args.num_trials} trials = {len(pairs) * args.num_trials} total trials"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    failures = []
    for index, (actor, distinguisher) in enumerate(pairs, 1):
        print(f"\n[{index}/{len(pairs)}] {actor} -> {distinguisher}")
        command = [
            sys.executable,
            str(EXPERIMENT_SCRIPT),
            "--actor",
            actor,
            "--distinguisher",
            distinguisher,
            *cli_arguments(args, ("condition", "num_trials", "turn_limit", "output_dir")),
            *cli_arguments(args, CONDITION_OPTIONS + BACKEND_OPTIONS),
        ]
        return_code = subprocess.run(command, check=False).returncode
        if return_code != 0:
            failures.append((actor, distinguisher, return_code))
            print(f"  FAILED rc={return_code} - continuing")

    print(f"\nDone. {len(pairs) - len(failures)} succeeded, {len(failures)} failed.")
    for actor, distinguisher, return_code in failures:
        print(f"  - {actor} -> {distinguisher}: rc={return_code}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
