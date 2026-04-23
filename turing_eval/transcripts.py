"""Persistence of per-trial JSON transcripts and aggregate CSV rows.

CSV schema is designed to be a superset of the original single-stage OpenRouter
output: the first 7 columns are identical
and in the same order, so analysis scripts for that format read both files unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .environment import capture_environment

SUSAN_CSV_FIELDS = [
    "trial_id",
    "actor_model",
    "distinguisher_model",
    "success",
    "turns",
    "answer",
    "transcript_path",
]

EXTRA_CSV_FIELDS = [
    "condition",
    "target_model",
    "error",
    "reasoning",
    "specimen_turns",
    "specimen_stopped_cleanly",
]


def save_transcript_json(
    trial_result: Dict[str, object],
    path: Path,
    actor_model: str,
    distinguisher_model: str,
    condition: str,
    target_model: str | None = None,
) -> None:
    payload = {
        "trial_id": trial_result["trial_id"],
        "condition": condition,
        "actor_model": actor_model,
        "distinguisher_model": distinguisher_model,
        "target_model": target_model,
        "environment": capture_environment(),
        **{key: value for key, value in trial_result.items() if key != "trial_id"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_csv(
    results: List[Dict[str, object]],
    output_csv: Path,
    actor_model: str,
    distinguisher_model: str,
    condition: str,
    target_model: str | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = SUSAN_CSV_FIELDS + EXTRA_CSV_FIELDS
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "trial_id": row["trial_id"],
                    "actor_model": actor_model,
                    "distinguisher_model": distinguisher_model,
                    "success": row.get("success"),
                    "turns": row.get("turns"),
                    "answer": row.get("answer"),
                    "transcript_path": row.get("transcript_path"),
                    "condition": condition,
                    "target_model": row.get("target_model", target_model),
                    "error": row.get("error"),
                    "reasoning": row.get("reasoning"),
                    "specimen_turns": row.get("specimen_turns"),
                    "specimen_stopped_cleanly": row.get("specimen_stopped_cleanly"),
                }
            )


def load_trial_results(
    run_dir: Path,
    *,
    actor_model: str,
    distinguisher_model: str,
    condition: str,
    target_model: str | None = None,
) -> list[Dict[str, object]]:
    """Load previously completed trial JSON transcripts from a run directory."""
    results: list[Dict[str, object]] = []
    seen_trial_ids: set[int] = set()

    for transcript_path in sorted(run_dir.glob("trial_*.json")):
        try:
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping unreadable transcript {transcript_path}: {exc}")
            continue

        if payload.get("condition") != condition:
            print(f"Warning: skipping transcript with mismatched condition: {transcript_path}")
            continue
        if payload.get("actor_model") != actor_model:
            print(f"Warning: skipping transcript with mismatched actor model: {transcript_path}")
            continue
        if payload.get("distinguisher_model") != distinguisher_model:
            print(f"Warning: skipping transcript with mismatched distinguisher model: {transcript_path}")
            continue
        if payload.get("target_model") != target_model:
            print(f"Warning: skipping transcript with mismatched target model: {transcript_path}")
            continue

        trial_id = payload.get("trial_id")
        if not isinstance(trial_id, int):
            print(f"Warning: skipping transcript with invalid trial_id: {transcript_path}")
            continue
        if trial_id in seen_trial_ids:
            print(f"Warning: skipping duplicate trial_id {trial_id} in {transcript_path}")
            continue

        result = {
            key: value
            for key, value in payload.items()
            if key not in {"condition", "actor_model", "distinguisher_model", "environment"}
        }
        result["transcript_path"] = str(transcript_path)
        results.append(result)
        seen_trial_ids.add(trial_id)

    results.sort(key=lambda row: int(row["trial_id"]))
    return results
