#!/usr/bin/env python3
"""Run a single actor/distinguisher pairing for N trials under one condition."""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from turing_eval.backends.base import GenerationConfig
from turing_eval.conditions import (
    run_trial_no_query_no_explanation,
    run_trial_no_query_yes_explanation,
    run_trial_yes_query_no_explanation,
)
from turing_eval.models import MODEL_REGISTRY, load_backend
from turing_eval.transcripts import save_transcript_json, write_csv

CONDITIONS = {
    "no_query_no_explanation": run_trial_no_query_no_explanation,
    "yes_query_no_explanation": run_trial_yes_query_no_explanation,
    "no_query_yes_explanation": run_trial_no_query_yes_explanation,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--distinguisher", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--turn-limit", type=int, default=40)
    parser.add_argument("--specimen-turn-limit", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--actor-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--distinguisher-backend", choices=["api", "hf"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    actor = load_backend(args.actor, generation, force_backend=args.actor_backend)
    distinguisher = load_backend(args.distinguisher, generation, force_backend=args.distinguisher_backend)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.actor}_vs_{args.distinguisher}_{args.condition}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = CONDITIONS[args.condition]
    results = []
    for trial_id in range(args.num_trials):
        print(f"\n=== Trial {trial_id + 1}/{args.num_trials} ===")
        kwargs: dict[str, object] = {
            "trial_id": trial_id,
            "actor": actor,
            "distinguisher": distinguisher,
            "turn_limit": args.turn_limit,
            "verbose": args.verbose,
        }
        if args.condition == "yes_query_no_explanation":
            specimen = load_backend(args.distinguisher, generation, force_backend=args.distinguisher_backend)
            kwargs["specimen"] = specimen
            kwargs["specimen_turn_limit"] = args.specimen_turn_limit
        try:
            result = runner(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"Trial {trial_id} crashed: {exc}")
            traceback.print_exc()
            result = {
                "trial_id": trial_id,
                "success": False,
                "turns": None,
                "answer": None,
                "error": str(exc),
            }
        transcript_path = out_dir / f"trial_{trial_id:03d}.json"
        save_transcript_json(
            result,
            transcript_path,
            actor.display_name,
            distinguisher.display_name,
            args.condition,
        )
        result["transcript_path"] = str(transcript_path)
        results.append(result)

    csv_path = out_dir / "results.csv"
    write_csv(results, csv_path, actor.display_name, distinguisher.display_name, args.condition)
    print(f"\nSaved {len(results)} trials to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
