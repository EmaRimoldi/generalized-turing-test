from __future__ import annotations

import csv
import json
import sys

import pytest

import run_experiment


class FakeBackend:
    def __init__(self, display_name: str) -> None:
        self.display_name = display_name
        self.model_id = display_name


def test_run_experiment_retries_infra_failures_and_keeps_only_valid_trials(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    actor_backend = FakeBackend("actor/model")
    distinguisher_backend = FakeBackend("dist/model")
    responses = iter(
        [
            {
                "success": False,
                "turns": None,
                "answer": None,
                "error": "openrouter HTTP 503 Service Unavailable",
            },
            {
                "success": False,
                "turns": 0,
                "answer": None,
                "error": "distinguisher_answered_in_opening",
                "first_message": "Question\n\n<answer>0</answer>",
                "raw_final_message": "Question\n\n<answer>0</answer>",
            },
            {
                "success": True,
                "turns": 2,
                "answer": 1,
                "first_message": "hello",
                "raw_final_message": "<answer>1</answer>",
            },
        ]
    )

    def fake_runner(**kwargs):
        result = dict(next(responses))
        result["trial_id"] = kwargs["trial_id"]
        return result

    def fake_load_backend(model_name, generation, force_backend=None, force_api_provider=None):
        if model_name == "gpt-5.4-pro":
            return actor_backend
        return distinguisher_backend

    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_no_explanation", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", fake_load_backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_no_explanation",
            "--num-trials",
            "2",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "retry-check",
            "--max-attempts",
            "5",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    run_path = out_dir / "retry-check"
    csv_path = run_path / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["trial_id"] == "0"
    assert rows[0]["error"] == "distinguisher_answered_in_opening"
    assert rows[1]["trial_id"] == "1"
    assert rows[1]["success"] == "True"
    assert rows[1]["error"] == ""
    failed_attempts = sorted(run_path.glob("failed_attempt_*.json"))
    assert len(failed_attempts) == 1
    failed_payload = json.loads(failed_attempts[0].read_text())
    assert failed_payload["error"] == "openrouter HTTP 503 Service Unavailable"
    assert failed_payload["attempt_id"] == 0
    assert (run_path / "trial_000.json").exists()
    assert (run_path / "trial_001.json").exists()


def test_run_experiment_fails_when_max_attempts_prevents_collecting_target(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "results"
    backend = FakeBackend("model")

    def fake_runner(**kwargs):
        return {
            "trial_id": kwargs["trial_id"],
            "success": False,
            "turns": None,
            "answer": None,
            "error": "openrouter returned non-JSON",
        }

    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_no_explanation", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_no_explanation",
            "--num-trials",
            "2",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "max-attempts-check",
            "--max-attempts",
            "2",
        ],
    )

    rc = run_experiment.main()

    assert rc == 1
    stdout = capsys.readouterr().out
    assert "Warning: reached max attempts (2) after collecting 0/2 valid trials." in stdout
    csv_path = out_dir / "max-attempts-check" / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []
    failed_attempts = sorted((out_dir / "max-attempts-check").glob("failed_attempt_*.json"))
    assert len(failed_attempts) == 2


def test_run_experiment_passes_api_provider_overrides(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    backend = FakeBackend("model")
    captured_calls = []

    def fake_runner(**kwargs):
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 1,
            "first_message": "hello",
            "raw_final_message": "<answer>1</answer>",
        }

    def fake_load_backend(model_name, generation, force_backend=None, force_api_provider=None):
        captured_calls.append((model_name, force_backend, force_api_provider))
        return backend

    monkeypatch.setitem(run_experiment.CONDITIONS, "yes_query_no_explanation", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", fake_load_backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "gpt-5.4-pro",
            "--condition",
            "yes_query_no_explanation",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "provider-override-check",
            "--actor-api-provider",
            "openai",
            "--distinguisher-api-provider",
            "openai",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_calls == [
        ("gpt-5.4-pro", None, "openai"),
        ("gpt-5.4-pro", None, "openai"),
        ("gpt-5.4-pro", None, "openai"),
    ]


def test_run_experiment_requires_target_model_for_trusted_condition(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_trusted_distinguisher",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_experiment.main()
    assert exc.value.code == 2
    assert "--target-model" in capsys.readouterr().err


def test_run_experiment_requires_matching_models_for_no_prompt_self_pair(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_prompt_self_pair",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_experiment.main()
    assert exc.value.code == 2
    assert "requires actor and distinguisher to match" in capsys.readouterr().err


def test_run_experiment_passes_target_model_to_runner_and_transcript(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    backend = FakeBackend("model")
    captured_runner_kwargs = []

    def fake_runner(**kwargs):
        captured_runner_kwargs.append(kwargs)
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 1,
            "target_model": kwargs["target_model"],
            "first_message": "hello",
            "raw_final_message": "<answer>1</answer>",
        }

    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_trusted_distinguisher", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_trusted_distinguisher",
            "--target-model",
            "gemini-3-pro",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "trusted-target-check",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_runner_kwargs[0]["target_model"] == "gemini-3-pro"
    csv_path = out_dir / "trusted-target-check" / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target_model"] == "gemini-3-pro"
    payload = json.loads((out_dir / "trusted-target-check" / "trial_000.json").read_text())
    assert payload["target_model"] == "gemini-3-pro"


def test_run_experiment_passes_num_queries_to_controlled_query_mode(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    backends = {
        "gpt-5.4-pro": FakeBackend("actor/model"),
        "claude-opus-4.6": FakeBackend("dist/model"),
    }
    captured_kwargs = {}

    def fake_runner(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 1,
            "specimen_turns": kwargs["num_queries"],
            "controlled_budget_respected": True,
            "first_message": "hello",
            "raw_final_message": "<answer>1</answer>",
        }

    monkeypatch.setitem(run_experiment.CONDITIONS, "yes_query_controlled_queries", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", lambda model_name, *args, **kwargs: backends[model_name])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "yes_query_controlled_queries",
            "--num-trials",
            "1",
            "--num-queries",
            "3",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "controlled-queries-check",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_kwargs["num_queries"] == 3
    assert captured_kwargs["turn_limit"] == 40
    assert captured_kwargs["specimen"].display_name == "dist/model"


def test_run_experiment_passes_num_turns_to_controlled_turn_mode(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    backend = FakeBackend("model")
    captured_kwargs = {}

    def fake_runner(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": kwargs["num_turns"],
            "answer": 0,
            "controlled_budget_respected": True,
            "first_message": "hello",
            "raw_final_message": "<answer>0</answer>",
        }

    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_controlled_turns", fake_runner)
    monkeypatch.setattr(run_experiment, "load_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_controlled_turns",
            "--num-trials",
            "1",
            "--num-turns",
            "4",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "controlled-turns-check",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_kwargs["num_turns"] == 4
    assert captured_kwargs["turn_limit"] == 40


def test_run_experiment_requires_num_queries_for_controlled_query_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "yes_query_controlled_queries",
            "--output-dir",
            "unused",
        ],
    )

    with pytest.raises(SystemExit):
        run_experiment.main()


def test_run_experiment_requires_num_turns_for_controlled_turn_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--actor",
            "gpt-5.4-pro",
            "--distinguisher",
            "claude-opus-4.6",
            "--condition",
            "no_query_controlled_turns",
            "--output-dir",
            "unused",
        ],
    )

    with pytest.raises(SystemExit):
        run_experiment.main()
