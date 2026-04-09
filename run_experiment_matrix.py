#!/usr/bin/env python3
"""Run every (actor, distinguisher) pairing from a list of models, N trials each."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--turn-limit", type=int, default=None)
    parser.add_argument("--specimen-turn-limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-self-match", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--actor-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--distinguisher-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    pairs = []
    for actor, distinguisher in itertools.product(args.models, repeat=2):
        if actor == distinguisher and not args.include_self_match:
            continue
        pairs.append((actor, distinguisher))

    total_trials = len(pairs) * args.num_trials
    print(f"Running {len(pairs)} pairings x {args.num_trials} trials = {total_trials} total trials")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for index, (actor, distinguisher) in enumerate(pairs, 1):
        print(f"\n[{index}/{len(pairs)}] {actor} -> {distinguisher}")
        command = [
            sys.executable,
            "run_experiment.py",
            "--actor",
            actor,
            "--distinguisher",
            distinguisher,
            "--condition",
            args.condition,
            "--num-trials",
            str(args.num_trials),
            "--output-dir",
            str(out_root),
        ]
        if args.turn_limit is not None:
            command.extend(["--turn-limit", str(args.turn_limit)])
        if args.specimen_turn_limit is not None:
            command.extend(["--specimen-turn-limit", str(args.specimen_turn_limit)])
        if args.verbose:
            command.append("--verbose")
        if args.actor_backend is not None:
            command.extend(["--actor-backend", args.actor_backend])
        if args.distinguisher_backend is not None:
            command.extend(["--distinguisher-backend", args.distinguisher_backend])
        if args.max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.temperature is not None:
            command.extend(["--temperature", str(args.temperature)])
        return_code = subprocess.run(command).returncode
        if return_code != 0:
            failures.append((actor, distinguisher, return_code))
            print(f"  FAILED rc={return_code} - continuing")

    print(f"\nDone. {len(pairs) - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        for actor, distinguisher, return_code in failures:
            print(f"  - {actor} -> {distinguisher}: rc={return_code}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
