from __future__ import annotations

import csv
from pathlib import Path

from scripts.render_results_summary import discover_result_csvs, load_rows, summarize_rows


def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_id",
                "actor_model",
                "distinguisher_model",
                "success",
                "turns",
                "answer",
                "transcript_path",
                "condition",
                "reasoning",
                "specimen_turns",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_rows_aggregates_trials(tmp_path: Path):
    _write_results_csv(
        tmp_path / "a" / "results.csv",
        [
            {
                "trial_id": 0,
                "actor_model": "model-a",
                "distinguisher_model": "model-b",
                "success": True,
                "turns": 4,
                "answer": 1,
                "transcript_path": "a.json",
                "condition": "no_query_no_explanation",
                "reasoning": None,
                "specimen_turns": None,
            },
            {
                "trial_id": 1,
                "actor_model": "model-a",
                "distinguisher_model": "model-b",
                "success": False,
                "turns": 6,
                "answer": 0,
                "transcript_path": "b.json",
                "condition": "no_query_no_explanation",
                "reasoning": None,
                "specimen_turns": None,
            },
        ],
    )
    _write_results_csv(
        tmp_path / "b" / "results.csv",
        [
            {
                "trial_id": 0,
                "actor_model": "model-a",
                "distinguisher_model": "model-c",
                "success": True,
                "turns": 5,
                "answer": 1,
                "transcript_path": "c.json",
                "condition": "yes_query_no_explanation",
                "reasoning": None,
                "specimen_turns": 3,
            }
        ],
    )

    csv_paths = discover_result_csvs(tmp_path)
    rows = load_rows(csv_paths)
    summary = summarize_rows(rows)

    assert len(csv_paths) == 2
    assert len(rows) == 3
    first = next(
        row
        for row in summary
        if row["condition"] == "no_query_no_explanation" and row["distinguisher_model"] == "model-b"
    )
    assert first["n_trials"] == 2
    assert first["n_completed"] == 1
    assert first["answer_1_rate"] == 0.5
    assert first["mean_turns"] == 5
    assert first["mean_specimen_turns"] is None

    second = next(
        row
        for row in summary
        if row["condition"] == "yes_query_no_explanation" and row["distinguisher_model"] == "model-c"
    )
    assert second["n_trials"] == 1
    assert second["n_completed"] == 1
    assert second["answer_1_rate"] == 1.0
    assert second["mean_turns"] == 5
    assert second["mean_specimen_turns"] == 3
