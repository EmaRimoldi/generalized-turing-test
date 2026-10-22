"""Shared protocol registry, option validation, and launcher arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import conditions
from .backends.api_backend import PROVIDERS
from .models import MODEL_REGISTRY

EXPERIMENT_SCRIPT = Path(__file__).resolve().parents[1] / "run_experiment.py"
CONDITIONS = {
    name.removeprefix("run_trial_"): getattr(conditions, name)
    for name in (
        "run_trial_no_query_no_explanation",
        "run_trial_no_query_no_explanation_explicit_model",
        "run_trial_no_query_no_explanation_50_50_prior",
        "run_trial_no_query_no_explanation_actor_model_hint",
        "run_trial_no_prompt_self_pair",
        "run_trial_yes_query_no_explanation",
        "run_trial_yes_query_no_explanation_grand",
        "run_trial_no_query_yes_explanation",
        "run_trial_no_query_trusted_distinguisher",
        "run_trial_yes_query_controlled_queries",
        "run_trial_no_query_controlled_turns",
    )
}
SPECIMEN_BUDGET_CONDITIONS = {"yes_query_no_explanation", "yes_query_no_explanation_grand"}
SPECIMEN_CONDITIONS = SPECIMEN_BUDGET_CONDITIONS | {"yes_query_controlled_queries"}
CONDITION_OPTIONS = (
    "target_model",
    "actor_hint_model",
    "specimen_turn_limit",
    "num_queries",
    "num_turns",
)
BACKEND_OPTIONS = (
    "verbose",
    "actor_backend",
    "distinguisher_backend",
    "actor_api_provider",
    "distinguisher_api_provider",
    "max_new_tokens",
    "temperature",
    "max_attempts",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_condition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-model", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--actor-hint-model", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--specimen-turn-limit", type=positive_int, default=20)
    parser.add_argument("--num-queries", type=positive_int)
    parser.add_argument("--num-turns", type=positive_int)


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true")
    for role in ("actor", "distinguisher"):
        parser.add_argument(f"--{role}-backend", choices=["api", "hf"])
        parser.add_argument(f"--{role}-api-provider", choices=sorted(PROVIDERS))
    parser.add_argument(
        "--max-new-tokens", type=positive_int, help="Omit to use the served model default."
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-attempts", type=positive_int)


def cli_arguments(options: object, names: tuple[str, ...]) -> list[str]:
    """Forward present values, keeping CSV and matrix launchers consistent."""
    arguments = []
    for name in names:
        value = getattr(options, name, None)
        if value is None or value is False:
            continue
        arguments.append("--" + name.replace("_", "-"))
        if value is not True:
            arguments.append(str(value))
    return arguments


def validate_condition_options(options: object) -> None:
    """Reject incompatible options before starting any model interactions."""
    run_name = getattr(options, "run_name", None)
    if run_name is not None and (run_name in {".", ".."} or Path(run_name).name != run_name):
        raise ValueError("run_name must be a single directory name")
    condition = options.condition
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    if condition == "no_prompt_self_pair" and options.actor != options.distinguisher:
        raise ValueError("no_prompt_self_pair requires actor and distinguisher to match")
    required = {
        "target_model": "no_query_trusted_distinguisher",
        "num_queries": "yes_query_controlled_queries",
        "num_turns": "no_query_controlled_turns",
    }
    allowed = {**required, "actor_hint_model": "no_query_no_explanation_actor_model_hint"}
    for name, expected in allowed.items():
        value = getattr(options, name, None)
        flag = "--" + name.replace("_", "-")
        if value is None and condition == expected and name in required:
            raise ValueError(f"{name} ({flag}) is required for condition {condition}")
        if value is not None and condition != expected:
            raise ValueError(f"{name} is only valid for {expected} ({flag})")
    for name in ("num_trials", "turn_limit", "specimen_turn_limit", "num_queries", "num_turns"):
        value = getattr(options, name, None)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        condition in SPECIMEN_BUDGET_CONDITIONS
        and getattr(options, "specimen_turn_limit", None) is None
    ):
        raise ValueError(f"specimen_turn_limit is required for condition {condition}")
