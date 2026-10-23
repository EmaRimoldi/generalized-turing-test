"""Check recovered protocols through the maintained launcher and persistence path."""

import json
from pathlib import Path
import sys

import pytest

import run_experiment
import run_experiment_batch
from tests.test_conditions_runner import ScriptedBackend
from turing_eval.conditions import run_trial_yes_query_trusted_distinguisher
from turing_eval.models import MODEL_REGISTRY


@pytest.mark.parametrize("condition", [
    "50_50_trusted_distinguisher", "yes_query_trusted_distinguisher",
])
def test_trusted_launch_records_target_identity_and_independent_specimen_route(
    monkeypatch, tmp_path, condition,
):
    actor = "claude-opus-4.6"
    judge = "gpt-5.4-pro"
    target = "gemini-3-pro"
    judge_script = ["opening", "<answer>1</answer>"]
    if condition == "yes_query_trusted_distinguisher":
        judge_script = ["warmup", "STOP", *judge_script]
    backends = {
        actor: ScriptedBackend(MODEL_REGISTRY[actor].display_name, ["imitation"]),
        judge: ScriptedBackend(MODEL_REGISTRY[judge].display_name, judge_script),
        target: ScriptedBackend(MODEL_REGISTRY[target].display_name, ["specimen response"]),
    }
    loads = []

    def load(nickname, generation, **kwargs):
        loads.append((nickname, kwargs))
        return backends[nickname]

    monkeypatch.setattr(run_experiment, "load_backend", load)
    monkeypatch.setattr(sys, "argv", [
        "single", "--actor", actor, "--distinguisher", judge, "--target-model", target,
        "--condition", condition, "--num-trials", "1", "--run-name", "trusted",
        "--output-dir", str(tmp_path), "--distinguisher-api-provider", "openai",
    ])
    assert run_experiment.main() == 0
    payload = json.loads((tmp_path / "trusted" / "trial_000.json").read_text())
    assert payload["target_model"] == MODEL_REGISTRY[target].display_name
    assert payload["run_config"]["target_model"] == target
    assert MODEL_REGISTRY[target].display_name in payload["actor_prompt"]
    if condition == "yes_query_trusted_distinguisher":
        assert loads[-1] == (target, {})
        assert payload["specimen_route"]["display_name"] == MODEL_REGISTRY[target].display_name
        assert payload["specimen_turns"] == 1
        assert payload["specimen_stopped_cleanly"] is True
        assert payload["stage1_distinguisher_messages"][-1]["content"] == "STOP"
        assert payload["stage2_distinguisher_messages"][-1]["content"] == "<answer>1</answer>"


def test_trusted_specimen_budget_is_enforced_without_stop():
    actor = ScriptedBackend("actor", ["imitation"])
    judge = ScriptedBackend("judge", ["warmup", "one more", "opening", "<answer>0</answer>"])
    specimen = ScriptedBackend("target", ["sample"])
    result = run_trial_yes_query_trusted_distinguisher(
        0, actor, judge, specimen, "target", specimen_turn_limit=1, turn_limit=1,
    )
    assert result["specimen_turns"] == 1
    assert result["specimen_stopped_cleanly"] is False
    assert result["turns"] == 1
    assert result["answer"] == 0


def test_trusted_specimen_must_be_the_nominated_target():
    backend = ScriptedBackend("judge", [])
    with pytest.raises(ValueError, match="specimen must match"):
        run_trial_yes_query_trusted_distinguisher(0, backend, backend, backend, "other", 1, 1)


def test_recovered_historical_schedules_are_launchable():
    directory = Path(__file__).resolve().parents[1] / "configs" / "historical"
    experiments = [
        experiment for path in sorted(directory.glob("*.csv"))
        for experiment in run_experiment_batch.load_experiments(path)
    ]
    assert len(experiments) == 11
    assert sum(run.num_trials for run in experiments) == 60
    assert {run.condition for run in experiments} == {
        "no_query_trusted_distinguisher", "50_50_trusted_distinguisher",
        "yes_query_trusted_distinguisher",
    }
