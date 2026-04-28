#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
VIS_PYTHON = REPO_ROOT / "visualizer" / "python"
if str(VIS_PYTHON) not in sys.path:
    sys.path.insert(0, str(VIS_PYTHON))

WEB_PUBLIC = REPO_ROOT / "visualizer" / "web" / "public"

from epsilon_visualizer.ingest import (  # noqa: E402
    IngestConfig,
    _serialize_pair_row,
    compute_pair_evidence_strict,
    trials_from_grouped_model_pair_json,
    write_visualizer_artifact_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grouped-json-a",
        default=str(WEB_PUBLIC / "condition_a_model_pair_results.json"),
        help="Grouped JSON for condition A (no_query_no_explanation).",
    )
    parser.add_argument(
        "--grouped-json-b",
        default=str(WEB_PUBLIC / "condition_b_model_pair_results.json"),
        help="Grouped JSON for condition B (yes_query_no_explanation).",
    )
    parser.add_argument(
        "--out",
        default=str(WEB_PUBLIC / "epsilon_visualizer_data.json"),
    )
    parser.add_argument("--epsilon-default", type=float, default=0.05)
    parser.add_argument("--min-valid-per-branch", type=int, default=3)
    parser.add_argument("--z", type=float, default=1.645)
    parser.add_argument("--grouped-no-impute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    grouped_a = Path(args.grouped_json_a).resolve()
    grouped_b = Path(args.grouped_json_b).resolve()

    trials_a, stats_a = trials_from_grouped_model_pair_json(
        grouped_a,
        condition="no_query_no_explanation",
        impute_uniform_if_missing=not args.grouped_no_impute,
    )
    trials_b, stats_b = trials_from_grouped_model_pair_json(
        grouped_b,
        condition="yes_query_no_explanation",
        impute_uniform_if_missing=not args.grouped_no_impute,
    )
    all_trials = trials_a + trials_b

    cfg = IngestConfig(
        results_root=REPO_ROOT,
        epsilon=args.epsilon_default,
        min_valid_per_branch=args.min_valid_per_branch,
        z=args.z,
        exclude_answered_in_opening=True,
        exclude_any_error=True,
    )
    pair_evidence = compute_pair_evidence_strict(trials=all_trials, cfg=cfg)

    conditions = sorted({t.condition for t in all_trials if t.condition})
    models_by_condition = {
        cond: sorted(
            {t.actor_model for t in all_trials if t.condition == cond and t.actor_model}
            | {t.distinguisher_model for t in all_trials if t.condition == cond and t.distinguisher_model}
        )
        for cond in conditions
    }
    models = sorted({m for xs in models_by_condition.values() for m in xs})

    artifact = {
        "mode": "strict",
        "availableModes": ["strict"],
        "epsilonDefault": args.epsilon_default,
        "minValidPerBranch": args.min_valid_per_branch,
        "confidenceMethod": {"kind": "plug_in_frequency"},
        "filtersDefault": {
            "excludeAnsweredInOpening": True,
            "excludeAnyError": True,
        },
        "models": models,
        "modelsByCondition": models_by_condition,
        "conditions": conditions,
        "pairs": [_serialize_pair_row(ev) for ev in pair_evidence],
        "formalismRef": "visualizer/docs/Turing_Ordering_Visualizer_Formalization_English.md",
        "dataSource": "grouped_json",
        "pairIngestStats": None,
        "conditionSources": {
            "no_query_no_explanation": {
                "groupedJson": str(grouped_a),
                "pairIngestStats": stats_a,
            },
            "yes_query_no_explanation": {
                "groupedJson": str(grouped_b),
                "pairIngestStats": stats_b,
            },
        },
    }

    write_visualizer_artifact_json(artifact, Path(args.out))
    print(f"Wrote: {args.out}")
    print(f"Conditions: {', '.join(conditions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
