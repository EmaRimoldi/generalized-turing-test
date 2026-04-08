from __future__ import annotations

from turing_eval.conditions import (
    run_trial_no_query_no_explanation,
    run_trial_no_query_yes_explanation,
    run_trial_yes_query_no_explanation,
)


class ScriptedBackend:
    def __init__(self, display_name: str, responses: list[str], model_id: str = "scripted") -> None:
        self.display_name = display_name
        self.model_id = model_id
        self._responses = list(responses)

    def generate(self, messages):
        if not self._responses:
            raise AssertionError("No scripted response left")
        return self._responses.pop(0)


def test_run_trial_no_query_no_explanation_success():
    actor = ScriptedBackend("actor/model", ["actor opening", "actor followup"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "question", "<answer>1</answer>"])
    result = run_trial_no_query_no_explanation(7, actor, distinguisher, turn_limit=5)
    assert result["trial_id"] == 7
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 1
    assert result["first_message"] == "d opening"
    assert result["raw_final_message"] == "<answer>1</answer>"
    assert result["actor_messages"][0]["role"] == "system"


def test_run_trial_yes_query_no_explanation_records_two_stages():
    actor = ScriptedBackend("actor/model", ["teach me", "STOP", "stage2 reply"])
    specimen = ScriptedBackend("dist/model", ["specimen answer"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "<answer>0</answer>"])
    result = run_trial_yes_query_no_explanation(
        3,
        actor,
        distinguisher,
        specimen,
        specimen_turn_limit=4,
        turn_limit=4,
    )
    assert result["success"] is True
    assert result["answer"] == 0
    assert result["specimen_turns"] == 1
    assert result["specimen_stopped_cleanly"] is True
    assert len(result["stage1_actor_messages"]) >= 3
    assert len(result["stage2_actor_messages"]) == 2


def test_run_trial_no_query_yes_explanation_extracts_reasoning():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend(
        "dist/model",
        [
            "d opening",
            "<answer>0</answer><reasoning>too generic</reasoning>",
        ],
    )
    result = run_trial_no_query_yes_explanation(11, actor, distinguisher, turn_limit=3)
    assert result["success"] is True
    assert result["answer"] == 0
    assert result["reasoning"] == "too generic"


def test_run_trial_yes_query_no_explanation_requires_matching_specimen():
    actor = ScriptedBackend("actor/model", ["STOP"])
    specimen = ScriptedBackend("wrong/model", [])
    distinguisher = ScriptedBackend("dist/model", [])
    try:
        run_trial_yes_query_no_explanation(
            1,
            actor,
            distinguisher,
            specimen,
            specimen_turn_limit=1,
            turn_limit=1,
        )
    except AssertionError as exc:
        assert "specimen must be the same model type" in str(exc)
    else:
        raise AssertionError("Expected assertion for mismatched specimen")
