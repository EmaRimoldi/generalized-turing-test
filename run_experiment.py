#!/usr/bin/env python3
"""Run a single actor/distinguisher pairing for N trials under one condition."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

from turing_eval.backends.base import GenerationConfig
from turing_eval.cli import (
    CONDITIONS,
    SPECIMEN_BUDGET_CONDITIONS,
    SPECIMEN_CONDITIONS,
    add_backend_arguments,
    add_condition_arguments,
    positive_int,
    validate_condition_options,
)
from turing_eval.models import MODEL_REGISTRY, load_backend
from turing_eval.transcripts import save_transcript_json, write_csv


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
    parser.add_argument("--num-trials", type=positive_int, default=5)
    parser.add_argument("--turn-limit", type=positive_int, default=40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)
    add_condition_arguments(parser)
    add_backend_arguments(parser)
    args = parser.parse_args()
    try:
        validate_condition_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    max_attempts = args.max_attempts if args.max_attempts is not None else (3 * args.num_trials)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run_name = args.run_name or f"{args.actor}_vs_{args.distinguisher}_{args.condition}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    if out_dir.exists():
        parser.error(
            f"Run directory already exists: {out_dir}. Choose a new --run-name or --output-dir."
        )

    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    target_display = MODEL_REGISTRY[args.target_model].display_name if args.target_model else None
    run_config = {
        "actor": args.actor,
        "distinguisher": args.distinguisher,
        "condition": args.condition,
        "target_model": args.target_model,
        "actor_hint_model": args.actor_hint_model,
        "num_trials": args.num_trials,
        "max_attempts": max_attempts,
        "turn_limit": args.turn_limit,
        "specimen_turn_limit": args.specimen_turn_limit if args.condition in SPECIMEN_BUDGET_CONDITIONS else None,
        "num_queries": args.num_queries,
        "num_turns": args.num_turns,
        "generation": asdict(generation),
    }
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

    out_dir.mkdir(parents=True, exist_ok=False)

    runner = CONDITIONS[args.condition]
    results = []
    attempt_id = 0
    while len(results) < args.num_trials and attempt_id < max_attempts:
        trial_id = len(results)
        print(
            f"\n=== Trial {trial_id + 1}/{args.num_trials} (attempt {attempt_id + 1}/{max_attempts}) ==="
        )
        kwargs: dict[str, object] = {
            "trial_id": trial_id,
            "actor": actor,
            "distinguisher": distinguisher,
            "turn_limit": args.turn_limit,
            "verbose": args.verbose,
        }
        if args.condition in SPECIMEN_CONDITIONS:
            specimen = load_backend(
                args.distinguisher,
                generation,
                force_backend=args.distinguisher_backend,
                force_api_provider=args.distinguisher_api_provider,
            )
            kwargs["specimen"] = specimen
            if args.condition in SPECIMEN_BUDGET_CONDITIONS:
                kwargs["specimen_turn_limit"] = args.specimen_turn_limit
            else:
                kwargs["num_queries"] = args.num_queries
        if args.condition == "no_query_trusted_distinguisher":
            kwargs["target_model"] = target_display
        if (
            args.condition == "no_query_no_explanation_actor_model_hint"
            and args.actor_hint_model is not None
        ):
            kwargs["actor_model_hint_display"] = MODEL_REGISTRY[args.actor_hint_model].display_name
        if args.condition == "no_query_controlled_turns":
            kwargs["num_turns"] = args.num_turns
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
        result["run_config"] = run_config
        result["actor_route"] = _describe_backend_route(actor)
        result["distinguisher_route"] = _describe_backend_route(distinguisher)
        if args.condition in SPECIMEN_CONDITIONS:
            result["specimen_route"] = _describe_backend_route(specimen)

        if _is_valid_trial_result(result):
            transcript_path = out_dir / f"trial_{trial_id:03d}.json"
            save_transcript_json(
                result,
                transcript_path,
                actor.display_name,
                distinguisher.display_name,
                args.condition,
                target_model=target_display,
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
                target_model=target_display,
            )
        attempt_id += 1

    csv_path = out_dir / "results.csv"
    write_csv(
        results,
        csv_path,
        actor.display_name,
        distinguisher.display_name,
        args.condition,
        target_model=target_display,
    )
    if len(results) < args.num_trials:
        print(
            f"\nWarning: reached max attempts ({max_attempts}) after collecting "
            f"{len(results)}/{args.num_trials} valid trials."
        )
    print(f"\nSaved {len(results)} valid trials to {out_dir}")
    return 0 if len(results) == args.num_trials else 1


if __name__ == "__main__":
    sys.exit(main())
