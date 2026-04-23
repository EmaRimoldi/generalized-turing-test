from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import run_experiment
from turing_eval.transcripts import save_transcript_json


class FakeBackend:
    def __init__(self, display_name: str) -> None:
        self.display_name = display_name
        self.model_id = display_name


def _write_existing_trial(
    run_dir: Path,
    *,
    trial_id: int,
    actor_model: str,
    distinguisher_model: str,
    condition: str,
    target_model: str | None = None,
    answer: int = 1,
) -> None:
    save_transcript_json(
        {
            "trial_id": trial_id,
            "success": True,
            "turns": 2,
            "answer": answer,
            "attempt_id": trial_id,
            "raw_final_message": f"<answer>{answer}</answer>",
        },
        run_dir / f"trial_{trial_id:03d}.json",
        actor_model,
        distinguisher_model,
        condition,
        target_model=target_model,
    )


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


def test_run_experiment_warns_when_max_attempts_prevents_collecting_target(monkeypatch, tmp_path, capsys):
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

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Warning: reached max attempts (2) after collecting 0/2 valid trials." in stdout
    csv_path = out_dir / "max-attempts-check" / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []
    failed_attempts = sorted((out_dir / "max-attempts-check").glob("failed_attempt_*.json"))
    assert len(failed_attempts) == 2


def test_run_experiment_resumes_from_existing_trial_transcripts(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    run_dir = out_dir / "resume-check"
    actor_backend = FakeBackend("actor/model")
    distinguisher_backend = FakeBackend("dist/model")
    _write_existing_trial(
        run_dir,
        trial_id=0,
        actor_model=actor_backend.display_name,
        distinguisher_model=distinguisher_backend.display_name,
        condition="no_query_no_explanation",
        answer=0,
    )
    _write_existing_trial(
        run_dir,
        trial_id=1,
        actor_model=actor_backend.display_name,
        distinguisher_model=distinguisher_backend.display_name,
        condition="no_query_no_explanation",
        answer=1,
    )

    seen_trial_ids = []

    def fake_runner(**kwargs):
        seen_trial_ids.append(kwargs["trial_id"])
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 1,
            "first_message": "hello",
            "raw_final_message": "<answer>1</answer>",
        }

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
            "4",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "resume-check",
            "--max-attempts",
            "5",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert seen_trial_ids == [2, 3]
    csv_path = run_dir / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["trial_id"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["answer"] for row in rows] == ["0", "1", "1", "1"]
    assert (run_dir / "trial_002.json").exists()
    assert (run_dir / "trial_003.json").exists()


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


def test_run_experiment_requires_target_model_for_trusted_condition(monkeypatch, tmp_path):
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

    try:
        run_experiment.main()
    except SystemExit as exc:
        assert "--target-model is required" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing --target-model")


def test_run_experiment_requires_target_model_for_50_50_trusted_condition(monkeypatch, tmp_path):
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
            "50_50_trusted_distinguisher",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )

    try:
        run_experiment.main()
    except SystemExit as exc:
        assert "--target-model is required" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing --target-model")


def test_run_experiment_requires_target_model_for_yes_query_trusted_condition(monkeypatch, tmp_path):
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
            "yes_query_trusted_distinguisher",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )

    try:
        run_experiment.main()
    except SystemExit as exc:
        assert "--target-model is required" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing --target-model")


def test_run_experiment_requires_matching_models_for_no_prompt_self_pair(monkeypatch, tmp_path):
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

    try:
        run_experiment.main()
    except SystemExit as exc:
        assert "requires --actor and --distinguisher to match" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for mismatched no_prompt_self_pair models")


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


def test_run_experiment_passes_target_model_for_50_50_condition(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"
    backend = FakeBackend("model")
    captured_runner_kwargs = []

    def fake_runner(**kwargs):
        captured_runner_kwargs.append(kwargs)
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 0,
            "target_model": kwargs["target_model"],
            "first_message": "hello",
            "raw_final_message": "<answer>0</answer>",
        }

    monkeypatch.setitem(run_experiment.CONDITIONS, "50_50_trusted_distinguisher", fake_runner)
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
            "50_50_trusted_distinguisher",
            "--target-model",
            "gemini-3-pro",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "trusted-50-50-target-check",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_runner_kwargs[0]["target_model"] == "gemini-3-pro"
    csv_path = out_dir / "trusted-50-50-target-check" / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target_model"] == "gemini-3-pro"
    payload = json.loads((out_dir / "trusted-50-50-target-check" / "trial_000.json").read_text())
    assert payload["target_model"] == "gemini-3-pro"


def test_run_experiment_passes_target_model_and_specimen_for_yes_query_trusted_condition(
    monkeypatch, tmp_path
):
    out_dir = tmp_path / "results"
    actor_backend = FakeBackend("actor/model")
    distinguisher_backend = FakeBackend("dist/model")
    specimen_backend = FakeBackend("target/model")
    captured_runner_kwargs = []
    load_calls = []

    def fake_runner(**kwargs):
        captured_runner_kwargs.append(kwargs)
        return {
            "trial_id": kwargs["trial_id"],
            "success": True,
            "turns": 1,
            "answer": 1,
            "target_model": kwargs["target_model"],
            "specimen_turns": kwargs["specimen_turn_limit"],
            "specimen_stopped_cleanly": True,
            "first_message": "hello",
            "raw_final_message": "<answer>1</answer>",
        }

    def fake_load_backend(model_name, generation, force_backend=None, force_api_provider=None):
        load_calls.append((model_name, force_backend, force_api_provider))
        if model_name == "gpt-5.4-pro":
            return actor_backend
        if model_name == "claude-opus-4.6":
            return distinguisher_backend
        if model_name == "gemini-3-pro":
            return specimen_backend
        raise AssertionError(f"Unexpected model load {model_name}")

    monkeypatch.setitem(run_experiment.CONDITIONS, "yes_query_trusted_distinguisher", fake_runner)
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
            "yes_query_trusted_distinguisher",
            "--target-model",
            "gemini-3-pro",
            "--specimen-turn-limit",
            "7",
            "--num-trials",
            "1",
            "--output-dir",
            str(out_dir),
            "--run-name",
            "yes-query-trusted-check",
        ],
    )

    rc = run_experiment.main()

    assert rc == 0
    assert captured_runner_kwargs[0]["target_model"] == "gemini-3-pro"
    assert captured_runner_kwargs[0]["specimen_turn_limit"] == 7
    assert captured_runner_kwargs[0]["specimen"] is specimen_backend
    assert load_calls == [
        ("gpt-5.4-pro", None, None),
        ("claude-opus-4.6", None, None),
        ("gemini-3-pro", None, None),
    ]
    csv_path = out_dir / "yes-query-trusted-check" / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target_model"] == "gemini-3-pro"
    assert rows[0]["specimen_turns"] == "7"
    payload = json.loads((out_dir / "yes-query-trusted-check" / "trial_000.json").read_text())
    assert payload["target_model"] == "gemini-3-pro"
    assert payload["specimen_route"]["display_name"] == "target/model"
