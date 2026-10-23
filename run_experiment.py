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
    TRUSTED_CONDITIONS,
    add_backend_arguments,
    add_condition_arguments,
    positive_int,
    validate_condition_options,
)
from turing_eval.models import MODEL_REGISTRY, load_backend
from turing_eval.resume import ResumeError, is_collected_trial, prepare_run, run_directory
from turing_eval.transcripts import save_transcript_json, write_csv


def _is_valid_trial_result(result: dict[str, object]) -> bool:
    return is_collected_trial(result)


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
    parser.add_argument("--resume", action="store_true", help="Resume a named run after strict validation.")
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
    if args.resume and not args.run_name:
        parser.error("--resume requires an explicit --run-name")
    if out_dir.exists() and not args.resume:
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
    try:
        with run_directory(out_dir, resume=args.resume):
            return _run(args, out_dir, generation, run_config, target_display)
    except ResumeError as exc:
        parser.error(str(exc))


def _run(args, out_dir, generation, run_config, target_display) -> int:
    max_attempts = run_config["max_attempts"]
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

    specimen = None
    if args.condition in SPECIMEN_CONDITIONS:
        if args.condition == "yes_query_trusted_distinguisher":
            # A trusted judge learns from the nominated target, independently of the judge's route.
            specimen = load_backend(args.target_model, generation)
        else:
            specimen = load_backend(
                args.distinguisher, generation,
                force_backend=args.distinguisher_backend,
                force_api_provider=args.distinguisher_api_provider,
            )
    identity = {
        "actor_model": actor.display_name, "distinguisher_model": distinguisher.display_name,
        "target_model": target_display, "condition": args.condition,
        "actor_route": _describe_backend_route(actor),
        "distinguisher_route": _describe_backend_route(distinguisher),
    }
    if specimen is not None:
        identity["specimen_route"] = _describe_backend_route(specimen)
    results, attempt_id = prepare_run(out_dir, run_config, identity)
    next_trial_id = max((row["trial_id"] for row in results), default=-1) + 1
    if results:
        print(f"Loaded {len(results)} recorded trials; target total is {args.num_trials}.")
    runner = CONDITIONS[args.condition]
    for invocation_attempt in range(max_attempts):
        if len(results) >= args.num_trials:
            break
        trial_id = next_trial_id
        print(
            f"\n=== Trial {trial_id + 1}/{args.num_trials} "
            f"(attempt {invocation_attempt + 1}/{max_attempts}, id {attempt_id}) ==="
        )
        kwargs: dict[str, object] = {
            "trial_id": trial_id,
            "actor": actor,
            "distinguisher": distinguisher,
            "turn_limit": args.turn_limit,
            "verbose": args.verbose,
        }
        if args.condition in SPECIMEN_CONDITIONS:
            kwargs["specimen"] = specimen
            if args.condition in SPECIMEN_BUDGET_CONDITIONS:
                kwargs["specimen_turn_limit"] = args.specimen_turn_limit
            else:
                kwargs["num_queries"] = args.num_queries
        if args.condition in TRUSTED_CONDITIONS:
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
            next_trial_id += 1
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
