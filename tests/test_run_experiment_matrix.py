from __future__ import annotations

import sys
from types import SimpleNamespace

import run_experiment_matrix


def test_matrix_runner_forwards_optional_flags(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_experiment_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix.py",
            "--models",
            "actor-a",
            "dist-b",
            "--condition",
            "yes_query_no_explanation",
            "--target-model",
            "gemini-3-pro",
            "--num-trials",
            "2",
            "--turn-limit",
            "17",
            "--specimen-turn-limit",
            "9",
            "--output-dir",
            str(tmp_path),
            "--include-self-match",
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
        ],
    )

    rc = run_experiment_matrix.main()

    assert rc == 0
    assert len(commands) == 4
    first = commands[0]
    assert "--turn-limit" in first and "17" in first
    assert "--specimen-turn-limit" in first and "9" in first
    assert "--target-model" in first and "gemini-3-pro" in first
    assert "--verbose" in first
    assert "--actor-backend" in first and "hf" in first
    assert "--distinguisher-backend" in first and "api" in first
    assert "--actor-api-provider" in first and "openai" in first
    assert "--distinguisher-api-provider" in first and "openrouter" in first
    assert "--max-new-tokens" in first and "123" in first
    assert "--temperature" in first and "0.4" in first
