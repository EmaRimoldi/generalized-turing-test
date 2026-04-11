#!/usr/bin/env python3
"""Run a single actor/distinguisher pairing for N trials under one condition."""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from turing_eval.backends.api_backend import PROVIDERS
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


def _is_valid_trial_result(result: dict[str, object]) -> bool:
    return bool(result.get("success")) or result.get("error") == "distinguisher_answered_in_opening"


def _failed_attempt_path(out_dir: Path, attempt_id: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return out_dir / f"failed_attempt_{attempt_id:03d}_{timestamp}.json"


def _describe_backend_route(backend) -> dict[str, object]:
    route = {
        "backend": getattr(backend, "backend_kind", None),
        "display_name": getattr(backend, "display_name", None),
        "model_id": getattr(backend, "model_id", None),
    }
    provider_name = getattr(backend, "provider_name", None)
    if provider_name is not None:
        route["provider"] = provider_name
    return route


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
    parser.add_argument("--actor-api-provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--distinguisher-api-provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens per response. Defaults to provider/model default when omitted.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()
    max_attempts = args.max_attempts if args.max_attempts is not None else (3 * args.num_trials)

    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    actor = load_backend(
        args.actor,
        generation,
        force_backend=args.actor_backend,
        force_api_provider=args.actor_api_provider,
    )
    distinguisher = load_backend(
        args.distinguisher,
        generation,
        force_backend=args.distinguisher_backend,
        force_api_provider=args.distinguisher_api_provider,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.actor}_vs_{args.distinguisher}_{args.condition}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = CONDITIONS[args.condition]
    results = []
    attempt_id = 0
    while len(results) < args.num_trials and attempt_id < max_attempts:
        trial_id = len(results)
        print(f"\n=== Trial {trial_id + 1}/{args.num_trials} (attempt {attempt_id + 1}/{max_attempts}) ===")
        kwargs: dict[str, object] = {
            "trial_id": trial_id,
            "actor": actor,
            "distinguisher": distinguisher,
            "turn_limit": args.turn_limit,
            "verbose": args.verbose,
        }
        if args.condition == "yes_query_no_explanation":
            specimen = load_backend(
                args.distinguisher,
                generation,
                force_backend=args.distinguisher_backend,
                force_api_provider=args.distinguisher_api_provider,
            )
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
        result["attempt_id"] = attempt_id
        result["actor_route"] = _describe_backend_route(actor)
        result["distinguisher_route"] = _describe_backend_route(distinguisher)
        if args.condition == "yes_query_no_explanation":
            result["specimen_route"] = _describe_backend_route(specimen)

        if _is_valid_trial_result(result):
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
        else:
            failed_path = _failed_attempt_path(out_dir, attempt_id)
            save_transcript_json(
                result,
                failed_path,
                actor.display_name,
                distinguisher.display_name,
                args.condition,
            )
        attempt_id += 1

    csv_path = out_dir / "results.csv"
    write_csv(results, csv_path, actor.display_name, distinguisher.display_name, args.condition)
    if len(results) < args.num_trials:
        print(
            f"\nWarning: reached max attempts ({max_attempts}) after collecting "
            f"{len(results)}/{args.num_trials} valid trials."
        )
    print(f"\nSaved {len(results)} valid trials to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
