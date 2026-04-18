#!/usr/bin/env python3
"""Riepilogo rapido dei results.csv della matrice B tier-0 (come nel messaggio guida)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Matrice B tier-0 senza gpt-5.4-pro (batch_configs/matrix_B_yes_query_no_explanation_tier0.csv)
DEFAULT_RUNS = [
    "B-010_claudeopus46_vs_claudeopus46",
    "B-012_claudeopus46_vs_gemini3pro",
    "B-016_gemini3pro_vs_claudeopus46",
    "B-018_gemini3pro_vs_gemini3pro",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("RESULTS"),
        help="Cartella radice (default: RESULTS nella cwd)",
    )
    args = parser.parse_args()
    root = args.results_dir

    for run in DEFAULT_RUNS:
        path = root / run / "results.csv"
        if not path.is_file():
            print(f"{run}: MISSING {path}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        success = sum(str(r.get("success", "")).lower() == "true" for r in rows)
        opening_invalid = sum(r.get("error") == "distinguisher_answered_in_opening" for r in rows)
        print(f"{run}: valid={len(rows)}, success={success}, opening_invalid={opening_invalid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
