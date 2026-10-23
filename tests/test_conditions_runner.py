from __future__ import annotations

from turing_eval.conditions import (
    run_trial_50_50_trusted_distinguisher,
    run_trial_yes_query_trusted_distinguisher,
    is_stop_signal,
    run_trial_no_query_controlled_turns,
    run_trial_no_query_no_explanation,
    run_trial_no_query_no_explanation_explicit_model,
    run_trial_no_prompt_self_pair,
    run_trial_no_query_trusted_distinguisher,
    run_trial_no_query_yes_explanation,
    run_trial_yes_query_controlled_queries,
    run_trial_yes_query_no_explanation,
)
from turing_eval.prompts import build_distinguisher_prompt_explicit_model_no_reasoning


class ScriptedBackend:
    def __init__(self, display_name: str, responses: list[str], model_id: str = "scripted") -> None:
        self.display_name = display_name
        self.model_id = model_id
        self._responses = list(responses)

    def generate(self, messages):
        if not self._responses:
            raise AssertionError("No scripted response left")
        return self._responses.pop(0)

    def generate_with_metadata(self, messages):
        return self.generate(messages), {}


def test_build_distinguisher_prompt_explicit_model_names_slug():
    slug = "anthropic/claude-opus-4.6"
    text = build_distinguisher_prompt_explicit_model_no_reasoning(slug)
    assert text.count(slug) >= 3
    assert "NOT" in text


def test_run_trial_no_query_no_explanation_explicit_model_success():
    actor = ScriptedBackend("actor/model", ["actor opening", "actor followup"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "question", "<answer>1</answer>"])
    result = run_trial_no_query_no_explanation_explicit_model(7, actor, distinguisher, turn_limit=5)
    assert result["trial_id"] == 7
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 2
    assert "dist/model" in result["distinguisher_messages"][0]["content"]


def test_run_trial_no_query_no_explanation_success():
    actor = ScriptedBackend("actor/model", ["actor opening", "actor followup"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "question", "<answer>1</answer>"])
    result = run_trial_no_query_no_explanation(7, actor, distinguisher, turn_limit=5)
    assert result["trial_id"] == 7
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 2
    assert result["first_message"] == "d opening"
    assert result["raw_final_message"] == "<answer>1</answer>"
    assert result["actor_messages"][0]["role"] == "user"


def test_run_trial_no_prompt_self_pair_opens_without_actor_prompt():
    actor = ScriptedBackend("shared/model", ["actor opening", "actor followup"])
    distinguisher = ScriptedBackend("shared/model", ["d opening", "question", "<answer>1</answer>"])
    result = run_trial_no_prompt_self_pair(8, actor, distinguisher, turn_limit=5)
    assert result["trial_id"] == 8
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 2
    assert result["first_message"] == "d opening"
    assert result["actor_prompt"] is None
    assert result["actor_messages"][0]["role"] == "user"
    assert result["actor_messages"][0]["content"] == "d opening"
    assert result["actor_messages"][1]["role"] == "assistant"
    assert result["actor_messages"][1]["content"] == "actor opening"
    assert result["distinguisher_messages"][2]["role"] == "user"
    assert result["distinguisher_messages"][2]["content"] == "actor opening"


def test_run_trial_no_prompt_self_pair_rejects_answer_in_opening():
    actor = ScriptedBackend("shared/model", ["unused"])
    distinguisher = ScriptedBackend("shared/model", ["<answer>1</answer>"])
    result = run_trial_no_prompt_self_pair(10, actor, distinguisher, turn_limit=3)
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert result["actor_prompt"] is None
    assert result["actor_messages"] == []


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
    assert result["turns"] == 1
    assert result["specimen_turns"] == 1
    assert result["specimen_stopped_cleanly"] is True
    assert len(result["stage1_actor_messages"]) >= 3
    assert len(result["stage2_actor_messages"]) == 2
    assert "specimen_messages" not in result


def test_run_trial_yes_query_no_explanation_counts_specimen_turns_by_specimen_replies():
    actor = ScriptedBackend("actor/model", ["question one", "question two", "STOP", "stage2 reply"])
    specimen = ScriptedBackend("dist/model", ["specimen answer one", "specimen answer two"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "<answer>1</answer>"])
    result = run_trial_yes_query_no_explanation(
        4,
        actor,
        distinguisher,
        specimen,
        specimen_turn_limit=4,
        turn_limit=4,
    )
    assert result["success"] is True
    assert result["specimen_turns"] == 2
    assert result["specimen_stopped_cleanly"] is True


def test_is_stop_signal_accepts_leading_stop_with_extra_text():
    assert is_stop_signal("STOP")
    assert is_stop_signal("STOP.")
    assert is_stop_signal(" stop\n\nI have enough information.")
    assert not is_stop_signal("STOPPING")
    assert not is_stop_signal("I should not stop yet")


def test_run_trial_no_query_no_explanation_accepts_uppercase_answer_tag():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "<ANSWER>1</ANSWER>"])
    result = run_trial_no_query_no_explanation(5, actor, distinguisher, turn_limit=3)
    assert result["success"] is True
    assert result["answer"] == 1


def test_run_trial_no_query_no_explanation_rejects_answer_in_opening():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("dist/model", ["<answer>1</answer>"])
    result = run_trial_no_query_no_explanation(6, actor, distinguisher, turn_limit=3)
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert result["raw_final_message"] == "<answer>1</answer>"
    assert result["actor_messages"] == []


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
    assert result["turns"] == 1
    assert result["reasoning"] == "too generic"


def test_run_trial_no_query_trusted_distinguisher_success():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("trusted/model", ["d opening", "<answer>1</answer>"])
    result = run_trial_no_query_trusted_distinguisher(
        12,
        actor,
        distinguisher,
        target_model="separate/target-model",
        turn_limit=3,
    )
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 1
    assert result["first_message"] == "d opening"
    assert "separate/target-model" in result["actor_prompt"]
    assert result["target_model"] == "separate/target-model"


def test_run_trial_no_query_trusted_distinguisher_rejects_answer_in_opening():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("trusted/model", ["<answer>0</answer>"])
    result = run_trial_no_query_trusted_distinguisher(
        13,
        actor,
        distinguisher,
        target_model="separate/target-model",
        turn_limit=3,
    )
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert "separate/target-model" in result["actor_prompt"]


def test_run_trial_yes_query_no_explanation_rejects_answer_in_opening():
    actor = ScriptedBackend("actor/model", ["teach me", "STOP"])
    specimen = ScriptedBackend("dist/model", ["specimen answer"])
    distinguisher = ScriptedBackend("dist/model", ["<answer>0</answer>"])
    result = run_trial_yes_query_no_explanation(
        9,
        actor,
        distinguisher,
        specimen,
        specimen_turn_limit=4,
        turn_limit=4,
    )
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert result["specimen_turns"] == 1
    assert result["stage2_actor_messages"] == []
    assert "specimen_messages" not in result


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


def test_run_trial_yes_query_controlled_queries_uses_exact_budget():
    actor = ScriptedBackend("actor/model", ["query one", "query two", "stage2 reply"])
    specimen = ScriptedBackend("dist/model", ["specimen answer one", "specimen answer two"])
    distinguisher = ScriptedBackend("dist/model", ["d opening", "<answer>1</answer>"])
    result = run_trial_yes_query_controlled_queries(
        12,
        actor,
        distinguisher,
        specimen,
        num_queries=2,
        turn_limit=4,
    )
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["specimen_turns"] == 2
    assert result["num_queries"] == 2
    assert result["controlled_budget_respected"] is True
    assert result["specimen_stopped_cleanly"] is None
    assert len(result["stage1_actor_messages"]) == 5
    assert len(result["stage2_actor_messages"]) == 2


def test_run_trial_no_query_controlled_turns_enforces_final_answer_turn():
    actor = ScriptedBackend("actor/model", ["actor round 1", "actor round 2", "actor round 3"])
    distinguisher = ScriptedBackend(
        "dist/model",
        [
            "opening question",
            "premature question <answer>0</answer>",
            "round 3 question",
            "<answer>1</answer>",
        ],
    )
    result = run_trial_no_query_controlled_turns(13, actor, distinguisher, num_turns=3)
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 3
    assert result["num_turns"] == 3
    assert result["controlled_budget_respected"] is True
    assert result["premature_answer_detected"] is True


def test_run_trial_50_50_trusted_distinguisher_success():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("trusted/model", ["d opening", "<answer>1</answer>"])
    result = run_trial_50_50_trusted_distinguisher(
        14,
        actor,
        distinguisher,
        target_model="separate/target-model",
        turn_limit=3,
    )
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["turns"] == 1
    assert result["first_message"] == "d opening"
    assert "separate/target-model" in result["actor_prompt"]
    assert result["target_model"] == "separate/target-model"



def test_run_trial_50_50_trusted_distinguisher_rejects_answer_in_opening():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    distinguisher = ScriptedBackend("trusted/model", ["<answer>0</answer>"])
    result = run_trial_50_50_trusted_distinguisher(
        15,
        actor,
        distinguisher,
        target_model="separate/target-model",
        turn_limit=3,
    )
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert "separate/target-model" in result["actor_prompt"]



def test_run_trial_yes_query_trusted_distinguisher_records_two_stages():
    actor = ScriptedBackend("actor/model", ["actor opening"])
    specimen = ScriptedBackend("target/model", ["specimen answer"])
    distinguisher = ScriptedBackend(
        "trusted/model",
        ["teach me about yourself", "STOP", "real opening", "<answer>1</answer>"],
    )
    result = run_trial_yes_query_trusted_distinguisher(
        16,
        actor,
        distinguisher,
        specimen,
        target_model="target/model",
        specimen_turn_limit=4,
        turn_limit=4,
    )
    assert result["success"] is True
    assert result["answer"] == 1
    assert result["target_model"] == "target/model"
    assert result["specimen_turns"] == 1
    assert result["specimen_stopped_cleanly"] is True
    assert result["first_message"] == "real opening"
    assert result["turns"] == 1
    assert len(result["actor_messages"]) == 2
    assert len(result["stage1_distinguisher_messages"]) == 4
    assert result["stage1_distinguisher_messages"][-1]["content"] == "STOP"
    assert result["stage2_distinguisher_messages"][0]["role"] == "user"
    assert "unknown agent" in result["stage2_distinguisher_messages"][0]["content"]



def test_run_trial_yes_query_trusted_distinguisher_rejects_answer_in_stage2_opening():
    actor = ScriptedBackend("actor/model", ["unused"])
    specimen = ScriptedBackend("target/model", ["specimen answer"])
    distinguisher = ScriptedBackend(
        "trusted/model",
        ["warmup question", "STOP", "<answer>0</answer>"],
    )
    result = run_trial_yes_query_trusted_distinguisher(
        17,
        actor,
        distinguisher,
        specimen,
        target_model="target/model",
        specimen_turn_limit=4,
        turn_limit=4,
    )
    assert result["success"] is False
    assert result["answer"] is None
    assert result["turns"] == 0
    assert result["error"] == "distinguisher_answered_in_opening"
    assert result["target_model"] == "target/model"
    assert result["specimen_turns"] == 1
    assert result["actor_messages"] == []
    assert len(result["stage1_distinguisher_messages"]) == 4
    assert result["stage1_distinguisher_messages"][-1]["content"] == "STOP"
    assert result["stage2_distinguisher_messages"][-1]["content"] == "<answer>0</answer>"
