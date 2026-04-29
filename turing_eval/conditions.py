"""Per-condition trial runners. One function per experimental condition."""
from __future__ import annotations

import copy
import re
from typing import Dict, Optional

from .backends.base import ChatBackend, ChatSession
from .prompts import (
    build_actor_prompt_no_query,
    build_actor_prompt_no_query_controlled_turns,
    build_actor_prompt_no_query_trusted_distinguisher,
    build_actor_prompt_yes_query,
    build_actor_prompt_yes_query_controlled_queries_final,
    build_actor_prompt_yes_query_controlled_queries_initial,
    build_actor_prompt_yes_query_controlled_queries_intermediate,
    build_distinguisher_prompt,
    build_distinguisher_prompt_actor_model_hint_no_reasoning,
    build_distinguisher_prompt_controlled_turns_final,
    build_distinguisher_prompt_controlled_turns_initial,
    build_distinguisher_prompt_controlled_turns_intermediate,
    build_distinguisher_prompt_explicit_model_no_reasoning,
    build_distinguisher_prompt_grand_experiment_no_reasoning,
    build_distinguisher_prompt_no_query_trusted_distinguisher,
    build_distinguisher_prompt_no_reasoning_50_50_prior,
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_STOP_RE = re.compile(r"^\s*STOP\b", re.IGNORECASE)


def extract_answer(text: str) -> Optional[int]:
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    stripped = match.group(1).strip()
    return int(stripped) if stripped in {"0", "1"} else None


def extract_reasoning(text: str) -> Optional[str]:
    match = _REASONING_RE.search(text)
    return match.group(1).strip() if match else None


def is_stop_signal(text: str) -> bool:
    return bool(_STOP_RE.match(text))


def _invalid_opening_result(
    *,
    trial_id: int,
    first_message: str,
    error: str,
    actor_prompt: Optional[str] = None,
    actor_messages: Optional[list[dict[str, object]]] = None,
    distinguisher_messages: Optional[list[dict[str, object]]] = None,
    specimen_turns: Optional[int] = None,
    specimen_stopped_cleanly: Optional[bool] = None,
    stage1_actor_messages: Optional[list[dict[str, object]]] = None,
    stage2_actor_messages: Optional[list[dict[str, object]]] = None,
    reasoning: Optional[str] = None,
    **extra_fields: object,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "trial_id": trial_id,
        "success": False,
        "turns": 0,
        "answer": None,
        "error": error,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": first_message,
    }
    if actor_messages is not None:
        result["actor_messages"] = actor_messages
    if distinguisher_messages is not None:
        result["distinguisher_messages"] = distinguisher_messages
    if specimen_turns is not None:
        result["specimen_turns"] = specimen_turns
    if specimen_stopped_cleanly is not None:
        result["specimen_stopped_cleanly"] = specimen_stopped_cleanly
    if stage1_actor_messages is not None:
        result["stage1_actor_messages"] = stage1_actor_messages
    if stage2_actor_messages is not None:
        result["stage2_actor_messages"] = stage2_actor_messages
    if reasoning is not None:
        result["reasoning"] = reasoning
    result.update(extra_fields)
    return result


def run_trial_no_query_no_explanation(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_no_query_no_explanation_explicit_model(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    """Like ``no_query_no_explanation`` (A), but distinguisher prompt names ``{distinguisher_model}`` explicitly."""
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_explicit_model_no_reasoning(distinguisher.display_name),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_no_query_no_explanation_50_50_prior(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    """Like ``no_query_no_explanation`` (A), but distinguisher prompt states an explicit 50/50 prior."""
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_no_reasoning_50_50_prior(),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_no_query_no_explanation_actor_model_hint(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
    actor_model_hint_display: Optional[str] = None,
) -> Dict[str, object]:
    """Like ``no_query_no_explanation`` (A), but tells the distinguisher the actor slug might be {actor_model}.

    ``actor_model_hint_display`` overrides the slug substituted into the distinguisher prompt (e.g. self-pair
    runs where the true actor matches the distinguisher but the prompt names a different model as the
    alternative).
    """
    hint = actor_model_hint_display if actor_model_hint_display is not None else actor.display_name
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_actor_model_hint_no_reasoning(hint),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
        "actor_model_hint_in_prompt": hint,
    }


def run_trial_no_prompt_self_pair(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=None,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=None)
    actor_reply = a_session.send(first_message)
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": None,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_yes_query_no_explanation(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    specimen: ChatBackend,
    specimen_turn_limit: int,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    assert specimen.display_name == distinguisher.display_name, (
        "specimen must be the same model type as the distinguisher"
    )

    actor_prompt = build_actor_prompt_yes_query(distinguisher.display_name)
    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    s_session = ChatSession(specimen, "specimen", initial_prompt=None)

    actor_to_specimen = a_session.generate_opening()
    if verbose:
        print("[A->S]", actor_to_specimen)

    specimen_turns = 0
    while not is_stop_signal(actor_to_specimen) and specimen_turns < specimen_turn_limit:
        specimen_reply = s_session.send(actor_to_specimen)
        if verbose:
            print("[S->A]", specimen_reply)
        specimen_turns += 1
        actor_to_specimen = a_session.send(specimen_reply)
        if verbose:
            print("[A->S]", actor_to_specimen)

    specimen_stopped_cleanly = is_stop_signal(actor_to_specimen)
    stage1_actor_messages = copy.deepcopy(a_session.messages)

    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            specimen_turns=specimen_turns,
            specimen_stopped_cleanly=specimen_stopped_cleanly,
            stage1_actor_messages=stage1_actor_messages,
            stage2_actor_messages=[],
            distinguisher_messages=copy.deepcopy(d_session.messages),
        )

    actor_reply = a_session.send(first_message)
    if verbose:
        print("[A]", actor_reply)
    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "specimen_turns": specimen_turns,
        "specimen_stopped_cleanly": specimen_stopped_cleanly,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "stage1_actor_messages": stage1_actor_messages,
        "stage2_actor_messages": copy.deepcopy(a_session.messages[len(stage1_actor_messages) :]),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_yes_query_no_explanation_grand(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    specimen: ChatBackend,
    specimen_turn_limit: int,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    """Like ``yes_query_no_explanation`` (C), but uses a more informative distinguisher prompt."""
    assert specimen.display_name == distinguisher.display_name, (
        "specimen must be the same model type as the distinguisher"
    )

    actor_prompt = build_actor_prompt_yes_query(distinguisher.display_name)
    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    s_session = ChatSession(specimen, "specimen", initial_prompt=None)

    actor_to_specimen = a_session.generate_opening()
    if verbose:
        print("[A->S]", actor_to_specimen)

    specimen_turns = 0
    while not is_stop_signal(actor_to_specimen) and specimen_turns < specimen_turn_limit:
        specimen_reply = s_session.send(actor_to_specimen)
        if verbose:
            print("[S->A]", specimen_reply)
        specimen_turns += 1
        actor_to_specimen = a_session.send(specimen_reply)
        if verbose:
            print("[A->S]", actor_to_specimen)

    specimen_stopped_cleanly = is_stop_signal(actor_to_specimen)
    stage1_actor_messages = copy.deepcopy(a_session.messages)

    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_grand_experiment_no_reasoning(distinguisher.display_name),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            specimen_turns=specimen_turns,
            specimen_stopped_cleanly=specimen_stopped_cleanly,
            stage1_actor_messages=stage1_actor_messages,
            stage2_actor_messages=[],
            distinguisher_messages=copy.deepcopy(d_session.messages),
        )

    actor_reply = a_session.send(first_message)
    if verbose:
        print("[A]", actor_reply)
    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "specimen_turns": specimen_turns,
        "specimen_stopped_cleanly": specimen_stopped_cleanly,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "stage1_actor_messages": stage1_actor_messages,
        "stage2_actor_messages": copy.deepcopy(a_session.messages[len(stage1_actor_messages) :]),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_yes_query_controlled_queries(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    specimen: ChatBackend,
    num_queries: int,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    assert specimen.display_name == distinguisher.display_name, (
        "specimen must be the same model type as the distinguisher"
    )
    if num_queries <= 0:
        raise ValueError("num_queries must be >= 1 for yes_query_controlled_queries")

    actor_prompt = build_actor_prompt_yes_query_controlled_queries_initial(
        distinguisher.display_name,
        num_queries,
    )
    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    s_session = ChatSession(specimen, "specimen", initial_prompt=None)

    actor_to_specimen = a_session.generate_opening()
    if verbose:
        print("[A->S]", actor_to_specimen)

    specimen_turns = 0
    last_specimen_reply = ""
    while specimen_turns < num_queries:
        specimen_reply = s_session.send(actor_to_specimen)
        last_specimen_reply = specimen_reply
        if verbose:
            print("[S->A]", specimen_reply)
        specimen_turns += 1
        if specimen_turns < num_queries:
            actor_to_specimen = a_session.send(
                build_actor_prompt_yes_query_controlled_queries_intermediate(
                    specimen_reply,
                    specimen_turns + 1,
                    num_queries,
                )
            )
            if verbose:
                print("[A->S]", actor_to_specimen)

    a_session.add_user_message(
        build_actor_prompt_yes_query_controlled_queries_final(
            last_specimen_reply,
            num_queries,
        )
    )
    stage1_actor_messages = copy.deepcopy(a_session.messages)
    controlled_budget_respected = specimen_turns == num_queries

    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            specimen_turns=specimen_turns,
            specimen_stopped_cleanly=None,
            stage1_actor_messages=stage1_actor_messages,
            stage2_actor_messages=[],
            distinguisher_messages=copy.deepcopy(d_session.messages),
            num_queries=num_queries,
            controlled_budget_respected=controlled_budget_respected,
        )

    actor_reply = a_session.send(first_message)
    if verbose:
        print("[A]", actor_reply)
    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "specimen_turns": specimen_turns,
        "specimen_stopped_cleanly": None,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "stage1_actor_messages": stage1_actor_messages,
        "stage2_actor_messages": copy.deepcopy(a_session.messages[len(stage1_actor_messages) :]),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
        "num_queries": num_queries,
        "controlled_budget_respected": controlled_budget_respected,
    }


def run_trial_no_query_controlled_turns(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    num_turns: int,
    turn_limit: int | None = None,
    verbose: bool = False,
) -> Dict[str, object]:
    if num_turns <= 0:
        raise ValueError("num_turns must be >= 1 for no_query_controlled_turns")

    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_controlled_turns_initial(num_turns),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query_controlled_turns(distinguisher.display_name, first_message)
    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    premature_answer_detected = extract_answer(first_message) is not None
    turns = 1
    while turns < num_turns:
        d_reply = d_session.send(
            build_distinguisher_prompt_controlled_turns_intermediate(
                actor_reply,
                turns + 1,
                num_turns,
            )
        )
        if verbose:
            print("[D]", d_reply)
        if extract_answer(d_reply) is not None:
            premature_answer_detected = True
        actor_reply = a_session.send(d_reply)
        turns += 1
        if verbose:
            print("[A]", actor_reply)

    d_reply = d_session.send(build_distinguisher_prompt_controlled_turns_final(actor_reply))
    if verbose:
        print("[D]", d_reply)
    answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
        "num_turns": num_turns,
        "controlled_budget_respected": turns == num_turns,
        "premature_answer_detected": premature_answer_detected,
    }


def run_trial_no_query_yes_explanation(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt(with_reasoning=True),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            reasoning=extract_reasoning(first_message),
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    reasoning = extract_reasoning(d_reply)
    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "reasoning": reasoning,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }


def run_trial_no_query_trusted_distinguisher(
    trial_id: int,
    actor: ChatBackend,
    distinguisher: ChatBackend,
    target_model: str,
    turn_limit: int,
    verbose: bool = False,
) -> Dict[str, object]:
    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        initial_prompt=build_distinguisher_prompt_no_query_trusted_distinguisher(target_model),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query_trusted_distinguisher(target_model, first_message)
    if extract_answer(first_message) is not None:
        return _invalid_opening_result(
            trial_id=trial_id,
            first_message=first_message,
            error="distinguisher_answered_in_opening",
            actor_prompt=actor_prompt,
            distinguisher_messages=copy.deepcopy(d_session.messages),
            actor_messages=[],
        )

    a_session = ChatSession(actor, "actor", initial_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    turns = 1
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    answer = extract_answer(d_reply)
    while answer is None and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        turns += 1
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        answer = extract_answer(d_reply)

    return {
        "trial_id": trial_id,
        "success": answer is not None,
        "turns": turns,
        "answer": answer,
        "target_model": target_model,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }
