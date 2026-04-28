#!/usr/bin/env python3
"""Build ``condition_b_model_pair_results.json`` from Drive ``Results/B`` run folders.

Each run folder is expected to hold ``results.csv`` and/or ``trial_*.json`` for
condition ``yes_query_no_explanation`` (README: Condition B, specimen warm-up).

Model roster defaults to ``ordering.model_order`` in
``visualizer/web/public/condition_a_model_pair_results.json`` (same 8-model grid
as Condition A after GPT removal).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path


def _normalize_model_id(model_id: str) -> str:
    s = (model_id or "").strip()
    if s in {"x-ai/grok-4.20", "x-ai/grok-2"}:
        return "grok-4.20"
    return s


def _load_model_order(grouped_a: Path) -> list[str]:
    raw = json.loads(grouped_a.read_text(encoding="utf-8"))
    order = [str(x).strip() for x in (raw.get("ordering") or {}).get("model_order") or [] if str(x).strip()]
    if not order:
        raise ValueError(f"No model_order in {grouped_a}")
    return order


def _accum_row(
    cnt: dict[tuple[str, str], list[int]],
    model_set: set[str],
    actor: str,
    distinguisher: str,
    condition: str | None,
    answer: object,
) -> None:
    if condition and condition != "yes_query_no_explanation":
        return
    a = _normalize_model_id(actor or "")
    d = _normalize_model_id(distinguisher or "")
    if a not in model_set or d not in model_set:
        return
    if answer not in (0, 1, "0", "1"):
        return
    ai = int(answer)
    key = (a, d)
    if ai == 1:
        cnt[key][0] += 1
    else:
        cnt[key][1] += 1


def _from_results_csv(path: Path, cnt: dict[tuple[str, str], list[int]], model_set: set[str]) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            _accum_row(
                cnt,
                model_set,
                row.get("actor_model", ""),
                row.get("distinguisher_model", ""),
                row.get("condition"),
                row.get("answer"),
            )


def _from_trial_jsons(folder: Path, cnt: dict[tuple[str, str], list[int]], model_set: set[str]) -> None:
    for p in sorted(folder.glob("trial_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _accum_row(
            cnt,
            model_set,
            str(d.get("actor_model") or ""),
            str(d.get("distinguisher_model") or ""),
            str(d.get("condition") or ""),
            d.get("answer"),
        )


def build_records(model_order: list[str], cnt: dict[tuple[str, str], list[int]]) -> tuple[list[dict], list[tuple[str, str]]]:
    records: list[dict] = []
    present: set[tuple[str, str]] = set()
    for actor in model_order:
        rows: list[dict] = []
        for dist in model_order:
            n1, n0 = cnt.get((actor, dist), [0, 0])
            vt = n1 + n0
            if vt > 0:
                present.add((actor, dist))
                rows.append(
                    {
                        "distinguisher": dist,
                        "valid_trials": vt,
                        "answer_1_count": n1,
                        "answer_0_count": n0,
                    }
                )
        records.append({"actor": actor, "rows": rows})
    expected = {(a, b) for a, b in product(model_order, model_order)}
    missing = sorted(expected - present, key=lambda t: (t[0], t[1]))
    return records, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-b-root",
        type=Path,
        default=Path(
            "/Users/emanuelerimoldi/Library/CloudStorage/GoogleDrive-emanuelerimoldi7@gmail.com/"
            "Il mio Drive/Turing Ordering Project /Results/B"
        ),
        help="Path to the Google Drive (synced) ``Results/B`` directory.",
    )
    parser.add_argument(
        "--grouped-a-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "web" / "public" / "condition_a_model_pair_results.json",
        help="Reference grouped JSON (for model_order only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "web" / "public" / "condition_b_model_pair_results.json",
    )
    args = parser.parse_args()

    model_order = _load_model_order(args.grouped_a_json)
    model_set = set(model_order)
    cnt: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    root = args.results_b_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    used_csv = 0
    used_json_fallback = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("B-"):
            continue
        csv_path = child / "results.csv"
        if csv_path.is_file():
            _from_results_csv(csv_path, cnt, model_set)
            used_csv += 1
        else:
            _from_trial_jsons(child, cnt, model_set)
            used_json_fallback += 1

    records_by_actor, missing_pairs = build_records(model_order, cnt)
    n_measured = sum(len(g["rows"]) for g in records_by_actor)

    doc = {
        "row_count": len(model_order) * len(model_order),
        "fields": ["actor", "distinguisher", "valid_trials", "answer_1_count", "answer_0_count"],
        "ordering": {
            "grouped_by": "actor",
            "within_actor_sorted_by": "distinguisher",
            "model_order": model_order,
        },
        "ingest_notes": {
            "condition": "yes_query_no_explanation",
            "results_b_root": str(root.resolve()),
            "run_folders_read": used_csv + used_json_fallback,
            "run_folders_with_results_csv": used_csv,
            "run_folders_json_only": used_json_fallback,
            "measured_directed_pairs": n_measured,
            "expected_directed_pairs": len(model_order) * len(model_order),
            "missing_directed_pairs": [[a, b] for a, b in missing_pairs],
        },
        "records_by_actor": records_by_actor,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Measured pairs: {n_measured} / {len(model_order) ** 2}")
    print(f"Missing pairs: {len(missing_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
