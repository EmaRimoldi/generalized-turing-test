#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from epsilon_visualizer.ingest import build_visualizer_artifact, write_visualizer_artifact_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="RESULTS", help="Directory containing run folders with results.csv.")
    parser.add_argument("--out", default="epsilon_visualizer_app/public/epsilon_visualizer_data.json")
    parser.add_argument("--epsilon-default", type=float, default=0.05)
    parser.add_argument("--min-valid-per-branch", type=int, default=3)
    parser.add_argument("--z", type=float, default=1.645)
    parser.add_argument("--include-answered-in-opening", action="store_true")
    parser.add_argument("--include-error-trials", action="store_true")
    args = parser.parse_args()

    artifact = build_visualizer_artifact(
        results_root=Path(args.results_root),
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

