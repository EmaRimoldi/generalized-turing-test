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

from epsilon_visualizer.ingest import build_visualizer_artifact, write_visualizer_artifact_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=None,
        help="Directory containing run folders with results.csv (used when neither --grouped-json nor --matrix-a is set and no default grouped JSON exists).",
    )
    parser.add_argument(
        "--matrix-a",
        dest="matrix_a",
        default=None,
        help="Aggregate Condition A matrix CSV (columns include valid_trials, answer_1_count, answer_0_count). "
        "When set, the visualizer uses only this file (no_query_no_explanation).",
    )
    parser.add_argument(
        "--grouped-json",
        dest="grouped_json",
        default=None,
        help="Grouped model-pair JSON (model_pair_results_grouped schema). Default path when unset: "
        "visualizer/web/public/condition_a_model_pair_results.json. Full grid: JSON cells, optional "
        "--grouped-matrix-fallback, then uniform 5/5 imputation unless --grouped-no-impute.",
    )
    parser.add_argument(
        "--grouped-matrix-fallback",
        dest="grouped_matrix_fallback",
        default=None,
        help="With --grouped-json: optional matrix CSV for extra measured cells (normally omitted; JSON is the source).",
    )
    parser.add_argument(
        "--grouped-no-impute",
        action="store_true",
        help="With --grouped-json: do not add uniform 5/5 trials for pairs absent from JSON and matrix (may leave missing_run).",
    )
    parser.add_argument("--out", default=str(WEB_PUBLIC / "epsilon_visualizer_data.json"))
    parser.add_argument("--epsilon-default", type=float, default=0.05)
    parser.add_argument("--min-valid-per-branch", type=int, default=3)
    parser.add_argument("--z", type=float, default=1.645)
    parser.add_argument("--include-answered-in-opening", action="store_true")
    parser.add_argument("--include-error-trials", action="store_true")
    args = parser.parse_args()

    default_grouped_json = WEB_PUBLIC / "condition_a_model_pair_results.json"
    if not args.matrix_a and not args.grouped_json and not args.results_root:
        if default_grouped_json.is_file():
            args.grouped_json = str(default_grouped_json)
        else:
            args.results_root = "RESULTS"

    n_sources = sum(bool(x) for x in (args.matrix_a, args.grouped_json, args.results_root))
    if n_sources > 1:
        parser.error("Use at most one of --matrix-a, --grouped-json, and --results-root")

    if args.grouped_json:
        fb = args.grouped_matrix_fallback
        fb_path = Path(fb).resolve() if fb else None
        artifact = build_visualizer_artifact(
            grouped_json=Path(args.grouped_json).resolve(),
            grouped_matrix_fallback_csv=fb_path,
            grouped_impute_uniform=not args.grouped_no_impute,
            epsilon=args.epsilon_default,
            min_valid_per_branch=args.min_valid_per_branch,
            z=args.z,
            exclude_answered_in_opening=not args.include_answered_in_opening,
            exclude_any_error=not args.include_error_trials,
        )
    elif args.matrix_a:
        artifact = build_visualizer_artifact(
            matrix_a_csv=Path(args.matrix_a).resolve(),
            epsilon=args.epsilon_default,
            min_valid_per_branch=args.min_valid_per_branch,
            z=args.z,
            exclude_answered_in_opening=not args.include_answered_in_opening,
            exclude_any_error=not args.include_error_trials,
        )
    else:
        root = Path(args.results_root or "RESULTS")
        artifact = build_visualizer_artifact(
            results_root=root,
            epsilon=args.epsilon_default,
            min_valid_per_branch=args.min_valid_per_branch,
            z=args.z,
            exclude_answered_in_opening=not args.include_answered_in_opening,
            exclude_any_error=not args.include_error_trials,
        )
    write_visualizer_artifact_json(artifact, Path(args.out))
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
