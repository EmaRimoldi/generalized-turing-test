#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path


def parse_bool(value: str) -> bool:
    """Convert common CSV truthy strings to bool."""
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize_results_file(results_csv: Path) -> dict:
    """
    Summarize one results.csv into a single row.
    Assumes the parent folder name is the run_name.
    """
    run_name = results_csv.parent.name

    actor = ""
    distinguisher = ""
    target_model = ""
    specimen = ""
    valid_trials = 0
    clean_specimen_stops = 0
    answer_1_count = 0
    answer_0_count = 0

    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        for row in reader:
            if not actor:
                actor = row.get("actor_model", "").strip()
            if not distinguisher:
                distinguisher = row.get("distinguisher_model", "").strip()
            if not target_model:
                target_model = row.get("target_model", "").strip()
            if not specimen:
                if "specimen_model" in fieldnames:
                    specimen = row.get("specimen_model", "").strip()
                elif "specimen" in fieldnames:
                    specimen = row.get("specimen", "").strip()

            if parse_bool(row.get("success", "")):
                valid_trials += 1

            if parse_bool(row.get("specimen_stopped_cleanly", "")):
                clean_specimen_stops += 1

            answer = str(row.get("answer", "")).strip()
            if answer == "1":
                answer_1_count += 1
            elif answer == "0":
                answer_0_count += 1

    return {
        "run_name": run_name,
        "actor": actor,
        "distinguisher": distinguisher,
        "target_model": target_model,
        "specimen": specimen,
        "valid_trials": valid_trials,
        "clean_specimen_stops": clean_specimen_stops,
        "answer_1_count": answer_1_count,
        "answer_0_count": answer_0_count,
    }


def main() -> None:
    """
    Looks for results.csv files in subfolders of the current directory.
    Example structure:

    current_directory/
        summarize_results.py
        results/
            B-001/results.csv
            B-002/results.csv
            ...

    Default behavior:
    - assumes there is a folder called 'results' next to this script
    - writes summary to results_summary.csv next to this script
    """
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir / "results"
    output_csv = script_dir / "results_summary.csv"

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Could not find root results folder: {root_dir}\n"
            f"Put this script one level above the folder named 'results', "
            f"or edit root_dir in the script."
        )

    results_files = sorted(root_dir.rglob("results.csv"))

    if not results_files:
        raise FileNotFoundError(f"No results.csv files found under {root_dir}")

    summaries = [summarize_results_file(path) for path in results_files]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name",
            "actor",
            "distinguisher",
            "target_model",
            "specimen",
            "valid_trials",
            "clean_specimen_stops",
            "answer_1_count",
            "answer_0_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
