#!/usr/bin/env python3
"""
Generate a batch CSV for run_openrouter_single_stage_batch.py.

The generated CSV contains every ordered (actor, distinguisher) pairing from the
provided model list. By default, self-pairs are included.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, List


def slugify_model_name(model_id: str) -> str:
    """Convert a model id into a filesystem-safe experiment suffix."""
    slug = model_id.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "model"


def iter_ordered_pairs(models: List[str], include_self_pairs: bool) -> Iterable[tuple[str, str]]:
    for actor_model in models:
        for distinguisher_model in models:
            if not include_self_pairs and actor_model == distinguisher_model:
                continue
            yield actor_model, distinguisher_model


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CSV of ordered actor/distinguisher experiment rows."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Space-separated OpenRouter model ids.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="Value written into the num-trials column for every row.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        required=True,
        help="Value written into the turn-limit column for every row.",
    )
    parser.add_argument(
        "--experiment-identifier",
        required=True,
        help="Prefix used in each experiment-name, e.g. 0408.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Destination CSV path. Defaults to <experiment-identifier>_experiments.csv in the current directory.",
    )
    parser.add_argument(
        "--exclude-self-pairs",
        action="store_true",
        help="Omit rows where actor-model and distinguisher-model are the same.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    models = [model.strip() for model in args.models if model.strip()]
    if not models:
        raise ValueError("At least one non-empty model id is required.")

    output_csv = Path(args.output_csv or f"{args.experiment_identifier}_experiments.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for actor_model, distinguisher_model in iter_ordered_pairs(
        models=models,
        include_self_pairs=not args.exclude_self_pairs,
    ):
        experiment_name = (
            f"{args.experiment_identifier}_"
            f"{slugify_model_name(actor_model)}_"
            f"{slugify_model_name(distinguisher_model)}"
        )
        rows.append(
            {
                "actor-model": actor_model,
                "distinguisher-model": distinguisher_model,
                "num-trials": args.num_trials,
                "turn-limit": args.turn_limit,
                "experiment-name": experiment_name,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "actor-model",
                "distinguisher-model",
                "num-trials",
                "turn-limit",
                "experiment-name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
