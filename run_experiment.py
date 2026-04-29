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
    run_trial_no_query_controlled_turns,
    run_trial_no_query_no_explanation,
    run_trial_no_query_no_explanation_explicit_model,
    run_trial_no_query_no_explanation_50_50_prior,
    run_trial_no_query_no_explanation_actor_model_hint,
    run_trial_no_prompt_self_pair,
    run_trial_no_query_trusted_distinguisher,
    run_trial_no_query_yes_explanation,
    run_trial_yes_query_controlled_queries,
    run_trial_yes_query_no_explanation,
    run_trial_yes_query_no_explanation_grand,
)
from turing_eval.models import MODEL_REGISTRY, load_backend
from turing_eval.transcripts import save_transcript_json, write_csv

CONDITIONS = {
    "no_query_no_explanation": run_trial_no_query_no_explanation,
    "no_query_no_explanation_explicit_model": run_trial_no_query_no_explanation_explicit_model,
    "no_query_no_explanation_50_50_prior": run_trial_no_query_no_explanation_50_50_prior,
    "no_query_no_explanation_actor_model_hint": run_trial_no_query_no_explanation_actor_model_hint,
    "no_prompt_self_pair": run_trial_no_prompt_self_pair,
    "yes_query_no_explanation": run_trial_yes_query_no_explanation,
    "yes_query_no_explanation_grand": run_trial_yes_query_no_explanation_grand,
    "no_query_yes_explanation": run_trial_no_query_yes_explanation,
    "no_query_trusted_distinguisher": run_trial_no_query_trusted_distinguisher,
    "yes_query_controlled_queries": run_trial_yes_query_controlled_queries,
    "no_query_controlled_turns": run_trial_no_query_controlled_turns,
}


def _validate_condition_args(args: argparse.Namespace) -> None:
    if args.condition == "no_query_trusted_distinguisher" and args.target_model is None:
        raise SystemExit("--target-model is required for condition no_query_trusted_distinguisher")
    if args.condition != "no_query_trusted_distinguisher" and args.target_model is not None:
        raise SystemExit("--target-model is only valid for condition no_query_trusted_distinguisher")
    if args.condition == "no_prompt_self_pair" and args.actor != args.distinguisher:
        raise SystemExit("--condition no_prompt_self_pair requires --actor and --distinguisher to match")
    if args.actor_hint_model is not None and args.condition != "no_query_no_explanation_actor_model_hint":
        raise SystemExit(
            "--actor-hint-model is only valid for condition no_query_no_explanation_actor_model_hint"
        )
    if args.condition == "yes_query_controlled_queries" and args.num_queries is None:
        raise SystemExit("--num-queries is required for condition yes_query_controlled_queries")
    if args.condition != "yes_query_controlled_queries" and args.num_queries is not None:
        raise SystemExit("--num-queries is only valid for condition yes_query_controlled_queries")
    if args.condition == "no_query_controlled_turns" and args.num_turns is None:
        raise SystemExit("--num-turns is required for condition no_query_controlled_turns")
    if args.condition != "no_query_controlled_turns" and args.num_turns is not None:
        raise SystemExit("--num-turns is only valid for condition no_query_controlled_turns")


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
    parser.add_argument("--target-model", choices=sorted(MODEL_REGISTRY), default=None)
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
    parser.add_argument(
        "--actor-hint-model",
        default=None,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "For no_query_no_explanation_actor_model_hint only: OpenRouter slug substituted as {actor_model} "
            "in the distinguisher prompt (overrides the real actor's slug)."
        ),
    )
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--num-turns", type=int, default=None)
    args = parser.parse_args()
    _validate_condition_args(args)
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
        if args.condition in {
            "yes_query_no_explanation",
            "yes_query_no_explanation_grand",
            "yes_query_controlled_queries",
        }:
            specimen = load_backend(
                args.distinguisher,
                generation,
                force_backend=args.distinguisher_backend,
                force_api_provider=args.distinguisher_api_provider,
            )
            kwargs["specimen"] = specimen
            if args.condition in {"yes_query_no_explanation", "yes_query_no_explanation_grand"}:
                kwargs["specimen_turn_limit"] = args.specimen_turn_limit
            else:
                kwargs["num_queries"] = args.num_queries
        if args.condition == "no_query_trusted_distinguisher":
            kwargs["target_model"] = args.target_model
        if args.condition == "no_query_no_explanation_actor_model_hint" and args.actor_hint_model is not None:
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
        result["actor_route"] = _describe_backend_route(actor)
        result["distinguisher_route"] = _describe_backend_route(distinguisher)
        if args.condition in {
            "yes_query_no_explanation",
            "yes_query_no_explanation_grand",
            "yes_query_controlled_queries",
        }:
            result["specimen_route"] = _describe_backend_route(specimen)

        if _is_valid_trial_result(result):
            transcript_path = out_dir / f"trial_{trial_id:03d}.json"
            save_transcript_json(
                result,
                transcript_path,
                actor.display_name,
                distinguisher.display_name,
                args.condition,
                target_model=args.target_model,
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
                target_model=args.target_model,
            )
        attempt_id += 1

    csv_path = out_dir / "results.csv"
    write_csv(
        results,
        csv_path,
        actor.display_name,
        distinguisher.display_name,
        args.condition,
        target_model=args.target_model,
    )
    if len(results) < args.num_trials:
        print(
            f"\nWarning: reached max attempts ({max_attempts}) after collecting "
            f"{len(results)}/{args.num_trials} valid trials."
        )
    print(f"\nSaved {len(results)} valid trials to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
