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
    "num_queries",
    "num_turns",
    "controlled_budget_respected",
    "premature_answer_detected",
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
                    "num_queries": row.get("num_queries"),
                    "num_turns": row.get("num_turns"),
                    "controlled_budget_respected": row.get("controlled_budget_respected"),
                    "premature_answer_detected": row.get("premature_answer_detected"),
                }
            )
