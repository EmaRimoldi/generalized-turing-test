#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

MODEL_ORDER = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-v3.2",
    "mistralai/mistral-large-2512",
    "qwen/qwen3-32b",
    "mistralai/ministral-8b-2512",
    "openai/gpt-5.4",
    "x-ai/grok-4.20",
]

CONDITION_BY_CODE = {
    "A": "no_query_no_explanation",
    "B": "yes_query_no_explanation",
}

ANSWER_TAG_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)


@dataclass
class PairAggregate:
    folder_name: str
    mtime: float
    actor: str
    distinguisher: str
    valid_trials: int
    answer_1_count: int
    answer_0_count: int
    parsed_opening_answers: int


def normalize_model_id(model_id: str, *, legacy_model_aliases: bool = False) -> str:
    s = (model_id or "").strip()
    if s == "grok-4.20":
        return "x-ai/grok-4.20"
    if legacy_model_aliases and s == "x-ai/grok-2":
        return "x-ai/grok-4.20"
    return s


def extract_answer(trial: dict) -> tuple[int | None, bool]:
    answer = trial.get("answer")
    opening = trial.get("error") == "distinguisher_answered_in_opening"
    if trial.get("error") and not opening:
        return None, False
    if not trial.get("success") and not opening:
        return None, False
    if type(answer) in (int, str) and answer in (0, 1, "0", "1"):
        return int(answer), False
    if trial.get("error") != "distinguisher_answered_in_opening":
        return None, False

    raw = trial.get("raw_final_message")
    if isinstance(raw, str):
        match = ANSWER_TAG_RE.search(raw)
        if match:
            return int(match.group(1)), True

    messages = trial.get("distinguisher_messages")
    if isinstance(messages, list):
        # Only generated assistant text can contain a verdict. The user prompt
        # contains answer-tag examples and must never be counted as a response.
        assistant = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        if assistant and isinstance(assistant[-1].get("content"), str):
            match = ANSWER_TAG_RE.search(assistant[-1]["content"])
            if match:
                return int(match.group(1)), True

    return None, False


def aggregate_folder(folder: Path, expected_condition: str, *, legacy_model_aliases: bool = False) -> PairAggregate | None:
    trial_paths = sorted(folder.glob("trial_*.json"))
    if not trial_paths:
        return None

    actor: str | None = None
    distinguisher: str | None = None
    valid = 0
    a1 = 0
    a0 = 0
    parsed_opening = 0

    for trial_path in trial_paths:
        try:
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trial, dict):
            continue
        condition = str(trial.get("condition") or "").strip()
        if condition and condition != expected_condition:
            continue
        current_actor = normalize_model_id(str(trial.get("actor_model") or ""), legacy_model_aliases=legacy_model_aliases)
        current_dist = normalize_model_id(str(trial.get("distinguisher_model") or ""), legacy_model_aliases=legacy_model_aliases)
        if current_actor not in MODEL_ORDER or current_dist not in MODEL_ORDER:
            continue
        if actor is None:
            actor = current_actor
            distinguisher = current_dist
        if actor != current_actor or distinguisher != current_dist:
            continue
        answer, from_opening = extract_answer(trial)
        if answer is None:
            continue
        valid += 1
        if answer == 1:
            a1 += 1
        else:
            a0 += 1
        if from_opening:
            parsed_opening += 1

    if actor is None or distinguisher is None or valid == 0:
        return None

    return PairAggregate(
        folder_name=folder.name,
        mtime=folder.stat().st_mtime,
        actor=actor,
        distinguisher=distinguisher,
        valid_trials=valid,
        answer_1_count=a1,
        answer_0_count=a0,
        parsed_opening_answers=parsed_opening,
    )


def build_grouped_json(results_root: Path, condition_code: str, *, legacy_model_aliases: bool = False) -> dict:
    expected_condition = CONDITION_BY_CODE[condition_code]
    by_pair: dict[tuple[str, str], PairAggregate] = {}
    duplicate_replacements: list[dict[str, str]] = []
    parsed_opening_total = 0
    run_folders_seen = 0

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name == "gpt54pro":
            continue
        if not child.name.startswith(f"{condition_code}-"):
            continue
        agg = aggregate_folder(child, expected_condition, legacy_model_aliases=legacy_model_aliases)
        if agg is None:
            continue
        run_folders_seen += 1
        key = (agg.actor, agg.distinguisher)
        old = by_pair.get(key)
        if old is None or agg.mtime > old.mtime:
            if old is not None:
                duplicate_replacements.append(
                    {
                        "actor": agg.actor,
                        "distinguisher": agg.distinguisher,
                        "kept": agg.folder_name,
                        "replaced": old.folder_name,
                    }
                )
            by_pair[key] = agg

    parsed_opening_total = sum(agg.parsed_opening_answers for agg in by_pair.values())
    records_by_actor = []
    for actor in MODEL_ORDER:
        rows = []
        for distinguisher in MODEL_ORDER:
            agg = by_pair.get((actor, distinguisher))
            if agg is None:
                continue
            rows.append(
                {
                    "distinguisher": distinguisher,
                    "valid_trials": agg.valid_trials,
                    "answer_1_count": agg.answer_1_count,
                    "answer_0_count": agg.answer_0_count,
                }
            )
        records_by_actor.append({"actor": actor, "rows": rows})

    expected_pairs = {(a, b) for a in MODEL_ORDER for b in MODEL_ORDER}
    present_pairs = set(by_pair.keys())
    missing_pairs = sorted(expected_pairs - present_pairs)

    return {
        "row_count": len(MODEL_ORDER) * len(MODEL_ORDER),
        "fields": ["actor", "distinguisher", "valid_trials", "answer_1_count", "answer_0_count"],
        "ordering": {
            "grouped_by": "actor",
            "within_actor_sorted_by": "distinguisher",
            "model_order": MODEL_ORDER,
        },
        "ingest_notes": {
            "condition": expected_condition,
            "results_root": str(results_root.resolve()),
            "run_folders_seen": run_folders_seen,
            "measured_directed_pairs": len(present_pairs),
            "expected_directed_pairs": len(expected_pairs),
            "missing_directed_pairs": [[a, b] for a, b in missing_pairs],
            "parsed_opening_answers": parsed_opening_total,
            "duplicate_replacements": duplicate_replacements,
            "legacy_model_aliases": legacy_model_aliases,
        },
        "records_by_actor": records_by_actor,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B"], required=True)
    parser.add_argument("--results-root", type=Path, required=True, help="Path to Results/A or Results/B.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-model-aliases", action="store_true", help="Opt in to historical Grok-2 → Grok-4.20 relabeling; inspect route metadata first.")
    args = parser.parse_args()

    doc = build_grouped_json(args.results_root.resolve(), args.condition, legacy_model_aliases=args.legacy_model_aliases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        f"Pairs: {doc['ingest_notes']['measured_directed_pairs']} / "
        f"{doc['ingest_notes']['expected_directed_pairs']}"
    )
    print(f"Opening answers parsed: {doc['ingest_notes']['parsed_opening_answers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
