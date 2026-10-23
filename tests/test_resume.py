"""Exercise interruption recovery and refusal to mix experimental evidence."""

import csv
import json
import sys
from types import SimpleNamespace

import pytest

import run_experiment
import run_experiment_batch
from turing_eval import resume


@pytest.fixture
def launch(monkeypatch, tmp_path):
    calls = []
    backend = SimpleNamespace(display_name="model", model_id="model", backend_kind="api")
    monkeypatch.setattr(run_experiment, "load_backend", lambda *a, **kw: backend)

    def runner(**kwargs):
        calls.append(kwargs["trial_id"])
        return {"trial_id": kwargs["trial_id"], "success": True, "answer": 1, "turns": 1}

    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_no_explanation", runner)

    def invoke(*extra):
        monkeypatch.setattr(sys, "argv", [
            "run_experiment.py", "--actor", "qwen3-32b", "--distinguisher", "gpt-5.4",
            "--condition", "no_query_no_explanation", "--output-dir", str(tmp_path),
            "--run-name", "interrupted", "--num-trials", "2", "--max-attempts", "1", *extra,
        ])
        return run_experiment.main()

    return invoke, tmp_path / "interrupted", calls, backend


def test_resume_finishes_partial_run_and_rebuilds_csv_without_rewriting_trials(launch):
    invoke, path, calls, _ = launch
    assert invoke() == 1
    original = (path / "trial_000.json").read_bytes()
    # Simulate interruption before the aggregate CSV was published.
    (path / "results.csv").unlink()
    assert invoke("--resume") == 0
    assert calls == [0, 1]
    assert (path / "trial_000.json").read_bytes() == original
    assert json.loads((path / "trial_001.json").read_text())["attempt_id"] == 1
    with (path / "results.csv").open() as handle:
        assert [row["trial_id"] for row in csv.DictReader(handle)] == ["0", "1"]
    assert invoke("--resume") == 0
    assert calls == [0, 1]  # An already complete run makes no more model calls.
    assert invoke("--resume", "--num-trials", "3") == 0
    assert calls == [0, 1, 2]
    assert (path / "trial_000.json").read_bytes() == original


@pytest.mark.parametrize("extra", [
    ("--temperature", "0.5"), ("--turn-limit", "12"),
    ("--actor", "claude-opus-4.6"), ("--num-trials", "1"),
])
def test_resume_refuses_changed_configuration_without_modifying_evidence(launch, extra):
    invoke, path, calls, _ = launch
    assert invoke() == 1
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(SystemExit) as exc:
        invoke("--resume", *extra)
    assert exc.value.code == 2
    assert calls == [0]
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.parametrize("damage", [
    "unreadable", "wrong_model", "wrong_config", "wrong_id", "duplicate",
    "boolean_id", "failed_trial", "string_success", "missing_answer",
    "missing_manifest", "wrong_failed_attempt",
])
def test_resume_refuses_damaged_or_mixed_records(launch, damage):
    invoke, path, calls, _ = launch
    assert invoke() == 1
    trial = path / "trial_000.json"
    payload = json.loads(trial.read_text())
    if damage == "unreadable":
        trial.write_text("{")
    elif damage == "wrong_model":
        payload["actor_model"] = "other/model"
    elif damage == "wrong_config":
        payload["run_config"]["turn_limit"] = 100
    elif damage == "wrong_id":
        payload["trial_id"] = -1
    elif damage == "duplicate":
        (path / "trial_001.json").write_text(json.dumps(payload))
    elif damage == "boolean_id":
        payload["trial_id"] = True
    elif damage == "failed_trial":
        payload["success"] = False
    elif damage == "string_success":
        payload["success"] = "True"
    elif damage == "missing_answer":
        payload["answer"] = None
    elif damage == "missing_manifest":
        (path / "run_manifest.json").unlink()
    elif damage == "wrong_failed_attempt":
        failed = dict(payload, actor_model="other/model", success=False)
        (path / "failed_attempt_001_interrupted.json").write_text(json.dumps(failed))
    if damage not in {"unreadable", "missing_manifest"}:
        trial.write_text(json.dumps(payload))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(SystemExit):
        invoke("--resume")
    assert calls == [0]
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_resume_refuses_changed_route_and_implementation(launch, monkeypatch):
    invoke, path, calls, backend = launch
    assert invoke() == 1
    backend.model_id = "changed-route"
    with pytest.raises(SystemExit):
        invoke("--resume")
    backend.model_id = "model"
    monkeypatch.setattr(resume, "implementation_digest", lambda: "changed-code")
    with pytest.raises(SystemExit):
        invoke("--resume")
    assert calls == [0]


def test_resume_preserves_failed_attempts_and_advances_ids(launch, monkeypatch):
    invoke, path, calls, _ = launch
    assert invoke() == 1
    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_no_explanation", lambda **kw: {
        "trial_id": kw["trial_id"], "success": False, "answer": None, "error": "API unavailable",
    })
    assert invoke("--resume") == 1
    failed = next(path.glob("failed_attempt_*.json"))
    saved = failed.read_bytes()
    assert json.loads(saved)["attempt_id"] == 1
    monkeypatch.setitem(run_experiment.CONDITIONS, "no_query_no_explanation", lambda **kw: {
        "trial_id": kw["trial_id"], "success": False, "answer": None,
        "error": "distinguisher_answered_in_opening", "turns": 0,
    })
    assert invoke("--resume") == 0
    assert failed.read_bytes() == saved
    assert json.loads((path / "trial_001.json").read_text())["attempt_id"] == 2


def test_concurrent_resume_is_rejected_and_lock_released(tmp_path):
    path = tmp_path / "run"
    with resume.run_directory(path, resume=False):
        with pytest.raises(resume.ResumeError, match="Another process"):
            with resume.run_directory(path, resume=True):
                pytest.fail("Two writers entered the same run")
    with resume.run_directory(path, resume=True):
        pass


def test_batch_resume_forwards_flag_for_existing_named_runs(monkeypatch, tmp_path):
    schedule = tmp_path / "batch.csv"
    schedule.write_text(
        "run_name,actor,distinguisher,num_trials,turn_limit,condition\n"
        "partial,qwen3-32b,gpt-5.4,2,40,no_query_no_explanation\n"
    )
    (tmp_path / "partial").mkdir()
    commands = []
    monkeypatch.setattr(run_experiment_batch.subprocess, "run", lambda command, **kw: (
        commands.append(command) or SimpleNamespace(returncode=0)
    ))
    monkeypatch.setattr(sys, "argv", [
        "batch", "--csv", str(schedule), "--output-dir", str(tmp_path), "--resume",
    ])
    assert run_experiment_batch.main() == 0
    assert "--resume" in commands[0]
    assert commands[0][commands[0].index("--run-name") + 1] == "partial"
