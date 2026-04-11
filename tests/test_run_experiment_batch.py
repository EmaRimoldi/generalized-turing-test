from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run_experiment_batch


def test_batch_runner_builds_one_command_per_csv_row(monkeypatch, tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,condition,ignored",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,,no_query_no_explanation,foo",
                "exp-b,ministral-8b-2512,qwen3-32b,3,15,8,yes_query_no_explanation,bar",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_experiment_batch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_batch.py",
            "--csv",
            str(csv_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--verbose",
            "--actor-backend",
            "hf",
            "--distinguisher-backend",
            "api",
            "--actor-api-provider",
            "openai",
            "--distinguisher-api-provider",
            "openrouter",
            "--max-new-tokens",
            "123",
            "--temperature",
            "0.4",
            "--max-attempts",
            "9",
        ],
    )

    rc = run_experiment_batch.main()

    assert rc == 0
    assert len(commands) == 2

    first = commands[0]
    assert first[:2] == [sys.executable, "run_experiment.py"]
    assert "--actor" in first and "gpt-5.4-pro" in first
    assert "--distinguisher" in first and "claude-opus-4.6" in first
    assert "--condition" in first and "no_query_no_explanation" in first
    assert "--num-trials" in first and "2" in first
    assert "--turn-limit" in first and "17" in first
    assert "--run-name" in first and "exp-a" in first
    assert "--specimen-turn-limit" not in first
    assert "--verbose" in first
    assert "--max-attempts" in first and "9" in first

    second = commands[1]
    assert "--specimen-turn-limit" in second and "8" in second
    assert "--condition" in second and "yes_query_no_explanation" in second


def test_batch_runner_requires_specimen_turn_limit_for_yes_query(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,ministral-8b-2512,qwen3-32b,3,15,yes_query_no_explanation",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specimen_turn_limit"):
        run_experiment_batch.load_experiments(csv_path)


def test_batch_runner_allows_blank_specimen_turn_limit_for_other_conditions(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,,no_query_no_explanation",
            ]
        ),
        encoding="utf-8",
    )

    experiments = run_experiment_batch.load_experiments(csv_path)

    assert len(experiments) == 1
    assert experiments[0].specimen_turn_limit is None


def test_batch_runner_rejects_missing_required_csv_columns(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,no_query_no_explanation",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="turn_limit"):
        run_experiment_batch.load_experiments(csv_path)
