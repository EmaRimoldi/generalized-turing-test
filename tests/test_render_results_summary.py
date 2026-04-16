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
                "target_model",
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
                "target_model": None,
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
                "target_model": None,
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
                "target_model": None,
                "reasoning": None,
                "specimen_turns": 3,
            }
        ],
    )

    csv_paths = discover_result_csvs([tmp_path])
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
    assert first["target_model"] is None

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
    assert second["target_model"] is None


def test_summarize_rows_separates_trusted_distinguisher_target_models(tmp_path: Path):
    _write_results_csv(
        tmp_path / "trusted" / "results.csv",
        [
            {
                "trial_id": 0,
                "actor_model": "model-a",
                "distinguisher_model": "model-b",
                "success": True,
                "turns": 4,
                "answer": 1,
                "transcript_path": "a.json",
                "condition": "no_query_trusted_distinguisher",
                "target_model": "gemini-3-pro",
                "reasoning": None,
                "specimen_turns": None,
            },
            {
                "trial_id": 1,
                "actor_model": "model-a",
                "distinguisher_model": "model-b",
                "success": True,
                "turns": 6,
                "answer": 0,
                "transcript_path": "b.json",
                "condition": "no_query_trusted_distinguisher",
                "target_model": "claude-opus-4.6",
                "reasoning": None,
                "specimen_turns": None,
            },
        ],
    )

    summary = summarize_rows(load_rows(discover_result_csvs([tmp_path])))

    assert len(summary) == 2
    gemini = next(row for row in summary if row["target_model"] == "gemini-3-pro")
    claude = next(row for row in summary if row["target_model"] == "claude-opus-4.6")
    assert gemini["n_trials"] == 1
    assert gemini["answer_1_rate"] == 1.0
    assert claude["n_trials"] == 1
    assert claude["answer_0_rate"] == 1.0


def test_discover_result_csvs_accepts_multiple_roots(tmp_path: Path):
    first_root = tmp_path / "results"
    second_root = tmp_path / "artifacts"
    _write_results_csv(
        first_root / "run_a" / "results.csv",
        [
            {
                "trial_id": 0,
                "actor_model": "model-a",
                "distinguisher_model": "model-b",
                "success": True,
                "turns": 2,
                "answer": 1,
                "transcript_path": "a.json",
                "condition": "no_query_no_explanation",
                "target_model": None,
                "reasoning": None,
                "specimen_turns": None,
            }
        ],
    )
    _write_results_csv(
        second_root / "run_b" / "results.csv",
        [
            {
                "trial_id": 0,
                "actor_model": "model-c",
                "distinguisher_model": "model-d",
                "success": True,
                "turns": 3,
                "answer": 0,
                "transcript_path": "b.json",
                "condition": "yes_query_no_explanation",
                "target_model": None,
                "reasoning": None,
                "specimen_turns": None,
            }
        ],
    )

    csv_paths = discover_result_csvs([first_root, second_root])

    assert csv_paths == sorted(
        [
            first_root / "run_a" / "results.csv",
            second_root / "run_b" / "results.csv",
        ]
    )
