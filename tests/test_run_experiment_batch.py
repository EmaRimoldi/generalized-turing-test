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

    def fake_run(command, *, check):
        assert check is False
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
    assert first[:2] == [sys.executable, str(run_experiment_batch.EXPERIMENT_SCRIPT)]
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


def test_batch_runner_accepts_no_query_trusted_distinguisher_condition(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,target_model,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,gemini-3-pro,2,17,no_query_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    experiments = run_experiment_batch.load_experiments(csv_path)

    assert len(experiments) == 1
    assert experiments[0].condition == "no_query_trusted_distinguisher"
    assert experiments[0].target_model == "gemini-3-pro"


def test_batch_runner_accepts_no_prompt_self_pair_condition(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,gpt-5.4-pro,2,17,no_prompt_self_pair",
            ]
        ),
        encoding="utf-8",
    )

    experiments = run_experiment_batch.load_experiments(csv_path)

    assert len(experiments) == 1
    assert experiments[0].condition == "no_prompt_self_pair"


def test_batch_runner_requires_matching_models_for_no_prompt_self_pair(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,no_prompt_self_pair",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires actor and distinguisher to match"):
        run_experiment_batch.load_experiments(csv_path)


def test_batch_runner_requires_target_model_for_no_query_trusted_distinguisher(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,no_query_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_model"):
        run_experiment_batch.load_experiments(csv_path)


def test_batch_runner_passes_actor_hint_model_for_actor_model_hint_condition(tmp_path, monkeypatch):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,condition,actor_hint_model",
                "exp-h,gemini-3-pro,gemini-3-pro,1,5,,no_query_no_explanation_actor_model_hint,claude-opus-4.6",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is False
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
        ],
    )

    assert run_experiment_batch.main() == 0
    assert commands[0].count("--actor-hint-model") == 1
    idx = commands[0].index("--actor-hint-model")
    assert commands[0][idx + 1] == "claude-opus-4.6"


def test_batch_runner_rejects_actor_hint_model_for_wrong_condition(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,condition,actor_hint_model",
                "exp-bad,gemini-3-pro,claude-opus-4.6,1,5,,no_query_no_explanation,claude-opus-4.6",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="actor_hint_model is only valid"):
        run_experiment_batch.load_experiments(csv_path)


def test_batch_runner_supports_controlled_query_and_turn_columns(monkeypatch, tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,num_queries,num_turns,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,,3,,yes_query_controlled_queries",
                "exp-b,ministral-8b-2512,qwen3-32b,3,15,,,4,no_query_controlled_turns",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is False
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
        ],
    )

    rc = run_experiment_batch.main()

    assert rc == 0
    assert "--num-queries" in commands[0] and "3" in commands[0]
    assert "--num-turns" not in commands[0]
    assert "--num-turns" in commands[1] and "4" in commands[1]
    assert "--num-queries" not in commands[1]


def test_batch_runner_requires_controlled_budget_columns(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,yes_query_controlled_queries",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_queries"):
        run_experiment_batch.load_experiments(csv_path)


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


def test_batch_runner_accepts_50_50_trusted_distinguisher_condition(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,target_model,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,gemini-3-pro,2,17,50_50_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    experiments = run_experiment_batch.load_experiments(csv_path)

    assert len(experiments) == 1
    assert experiments[0].condition == "50_50_trusted_distinguisher"
    assert experiments[0].target_model == "gemini-3-pro"



def test_batch_runner_accepts_yes_query_trusted_distinguisher_condition(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,target_model,num_trials,turn_limit,specimen_turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,gemini-3-pro,2,17,9,yes_query_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    experiments = run_experiment_batch.load_experiments(csv_path)

    assert len(experiments) == 1
    assert experiments[0].condition == "yes_query_trusted_distinguisher"
    assert experiments[0].target_model == "gemini-3-pro"
    assert experiments[0].specimen_turn_limit == 9



def test_batch_runner_requires_target_model_for_50_50_trusted_distinguisher(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,50_50_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_model"):
        run_experiment_batch.load_experiments(csv_path)



def test_batch_runner_requires_target_model_for_yes_query_trusted_distinguisher(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,num_trials,turn_limit,specimen_turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,2,17,9,yes_query_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_model"):
        run_experiment_batch.load_experiments(csv_path)



def test_batch_runner_requires_specimen_turn_limit_for_yes_query_trusted_distinguisher(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run_name,actor,distinguisher,target_model,num_trials,turn_limit,condition",
                "exp-a,gpt-5.4-pro,claude-opus-4.6,gemini-3-pro,2,17,yes_query_trusted_distinguisher",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specimen_turn_limit"):
        run_experiment_batch.load_experiments(csv_path)
