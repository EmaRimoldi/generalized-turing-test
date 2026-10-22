"""Catch invalid schedules before an expensive or destructive launch begins."""

import sys

import pytest

import run_experiment
import run_experiment_batch
import run_experiment_matrix


@pytest.mark.parametrize(
    "option,value",
    [
        ("--num-trials", "0"),
        ("--turn-limit", "-1"),
        ("--num-queries", "0"),
        ("--num-turns", "-1"),
        ("--max-attempts", "0"),
        ("--max-new-tokens", "-1"),
        ("--target-model", "qwen3-32b"),
    ],
)
def test_single_run_rejects_invalid_options_before_loading_models(
    monkeypatch, tmp_path, option, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("Loaded a model before validating options")

    monkeypatch.setattr(run_experiment, "load_backend", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "qwen3-32b",
            "--distinguisher",
            "gpt-5.4",
            "--condition",
            "no_query_no_explanation",
            "--output-dir",
            str(tmp_path),
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_experiment.main()
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_single_run_refuses_to_overwrite_a_recorded_run(monkeypatch, tmp_path):
    run = tmp_path / "recorded"
    run.mkdir()
    original = run / "trial_000.json"
    original.write_text('{"preserve":true}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "qwen3-32b",
            "--distinguisher",
            "gpt-5.4",
            "--condition",
            "no_query_no_explanation",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "recorded",
        ],
    )
    with pytest.raises(SystemExit):
        run_experiment.main()
    assert original.read_text() == '{"preserve":true}'


@pytest.mark.parametrize("extra", ["gpt-5.4,,", ",1,", ",,2"])
def test_batch_rejects_incompatible_options_on_any_row(tmp_path, extra):
    schedule = tmp_path / "batch.csv"
    schedule.write_text(
        "run_name,actor,distinguisher,num_trials,turn_limit,condition,target_model,num_queries,num_turns\n"
        "good,qwen3-32b,gpt-5.4,1,40,no_query_no_explanation,,,\n"
        f"bad,qwen3-32b,gpt-5.4,1,40,no_query_no_explanation,{extra}\n"
    )
    with pytest.raises(ValueError, match="CSV row 3"):
        run_experiment_batch.load_experiments(schedule)


@pytest.mark.parametrize("second_name", ["same", "../outside"])
def test_batch_rejects_reused_or_escaping_output_names(tmp_path, second_name):
    schedule = tmp_path / "batch.csv"
    schedule.write_text(
        "run_name,actor,distinguisher,num_trials,turn_limit,condition\n"
        "same,qwen3-32b,gpt-5.4,1,40,no_query_no_explanation\n"
        f"{second_name},qwen3-32b,gpt-5.4,1,40,no_query_no_explanation\n"
    )
    with pytest.raises(ValueError):
        run_experiment_batch.load_experiments(schedule)


def test_batch_validates_all_existing_directories_before_starting(monkeypatch, tmp_path):
    schedule = tmp_path / "batch.csv"
    schedule.write_text(
        "run_name,actor,distinguisher,num_trials,turn_limit,condition\n"
        "new,qwen3-32b,gpt-5.4,1,40,no_query_no_explanation\n"
        "existing,qwen3-32b,gpt-5.4,1,40,no_query_no_explanation\n"
    )
    (tmp_path / "existing").mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("Started a batch before checking all output directories")

    monkeypatch.setattr(run_experiment_batch.subprocess, "run", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["batch", "--csv", str(schedule), "--output-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        run_experiment_batch.main()


@pytest.mark.parametrize(
    "condition,option",
    [
        ("yes_query_controlled_queries", "--num-queries"),
        ("no_query_controlled_turns", "--num-turns"),
    ],
)
def test_matrix_forwards_resource_budgets_from_another_directory(
    monkeypatch, tmp_path, condition, option
):
    calls = []

    def fake_run(command, *, check):
        assert not check
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(run_experiment_matrix.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix",
            "--models",
            "qwen3-32b",
            "gpt-5.4",
            "--condition",
            condition,
            option,
            "3",
            "--output-dir",
            "results",
        ],
    )
    assert run_experiment_matrix.main() == 0
    assert len(calls) == 2
    assert calls[0][1] == str(run_experiment_matrix.EXPERIMENT_SCRIPT)
    assert calls[0][calls[0].index(option) + 1] == "3"
