"""Per-condition trial runners. One function per experimental condition."""
from __future__ import annotations

import copy
import re
from typing import Dict, Optional

from .backends.base import ChatBackend, ChatSession
from .prompts import (
    build_actor_prompt_no_query,
    build_actor_prompt_yes_query,
    build_distinguisher_prompt,
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
        system_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    a_session = ChatSession(actor, "actor", system_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    turns = 0
    while "<answer>" not in d_reply and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        turns += 1

    answer = extract_answer(d_reply)
    return {
        "trial_id": trial_id,
        "success": ("<answer>" in d_reply) and (answer is not None),
        "turns": turns,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
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
    a_session = ChatSession(actor, "actor", system_prompt=actor_prompt)
    s_session = ChatSession(specimen, "specimen", system_prompt=None)

    actor_to_specimen = a_session.generate_opening()
    if verbose:
        print("[A->S]", actor_to_specimen)

    specimen_turns = 0
    while not is_stop_signal(actor_to_specimen) and specimen_turns < specimen_turn_limit:
        specimen_reply = s_session.send(actor_to_specimen)
        if verbose:
            print("[S->A]", specimen_reply)
        actor_to_specimen = a_session.send(specimen_reply)
        if verbose:
            print("[A->S]", actor_to_specimen)
        specimen_turns += 1

    specimen_stopped_cleanly = is_stop_signal(actor_to_specimen)
    stage1_actor_messages = copy.deepcopy(a_session.messages)
    specimen_messages = copy.deepcopy(s_session.messages)

    d_session = ChatSession(
        distinguisher,
        "distinguisher",
        system_prompt=build_distinguisher_prompt(with_reasoning=False),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_reply = a_session.send(first_message)
    if verbose:
        print("[A]", actor_reply)
    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    turns = 0
    while "<answer>" not in d_reply and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        turns += 1

    answer = extract_answer(d_reply)
    return {
        "trial_id": trial_id,
        "success": ("<answer>" in d_reply) and (answer is not None),
        "turns": turns,
        "specimen_turns": specimen_turns,
        "specimen_stopped_cleanly": specimen_stopped_cleanly,
        "answer": answer,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "stage1_actor_messages": stage1_actor_messages,
        "specimen_messages": specimen_messages,
        "stage2_actor_messages": copy.deepcopy(a_session.messages[len(stage1_actor_messages) :]),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
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
        system_prompt=build_distinguisher_prompt(with_reasoning=True),
    )
    first_message = d_session.generate_opening()
    if verbose:
        print("[D]", first_message)

    actor_prompt = build_actor_prompt_no_query(distinguisher.display_name, first_message)
    a_session = ChatSession(actor, "actor", system_prompt=actor_prompt)
    actor_reply = a_session.generate_opening()
    if verbose:
        print("[A]", actor_reply)

    d_reply = d_session.send(actor_reply)
    if verbose:
        print("[D]", d_reply)

    turns = 0
    while "<answer>" not in d_reply and turns < turn_limit:
        actor_reply = a_session.send(d_reply)
        if verbose:
            print("[A]", actor_reply)
        d_reply = d_session.send(actor_reply)
        if verbose:
            print("[D]", d_reply)
        turns += 1

    answer = extract_answer(d_reply)
    reasoning = extract_reasoning(d_reply)
    return {
        "trial_id": trial_id,
        "success": ("<answer>" in d_reply) and (answer is not None),
        "turns": turns,
        "answer": answer,
        "reasoning": reasoning,
        "first_message": first_message,
        "actor_prompt": actor_prompt,
        "raw_final_message": d_reply,
        "actor_messages": copy.deepcopy(a_session.messages),
        "distinguisher_messages": copy.deepcopy(d_session.messages),
    }
