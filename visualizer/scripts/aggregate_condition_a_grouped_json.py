#!/usr/bin/env python3
"""Merge Condition A (no_query_no_explanation) aggregates from many results.csv trees into grouped JSON.

For each directed pair (actor, distinguisher), if multiple result folders exist, the newest
``results.csv`` (by file mtime) wins. Model ids are normalized toward the slugs in
``ordering.model_order`` (see ``CANONICAL_EQUIV`` for known aliases such as gpt-5.4 / gpt-5.4-pro).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CANONICAL_EQUIV: dict[str, str] = {
    # Map alternate slugs to the id used in the 9x9 public matrix, when appropriate.
    "openai/gpt-5.4-pro": "openai/gpt-5.4",
    "x-ai/grok-2": "grok-4.20",
    "x-ai/grok-4.20": "grok-4.20",
}

CONDITION = "no_query_no_explanation"


@dataclass
class CellAgg:
    path: Path
    mtime: float
    valid: int
    a1: int
    a0: int


def _norm_slug(s: str, order_set: set[str]) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    s = CANONICAL_EQUIV.get(s, s)
    if s in order_set:
        return s
    # Allow short name grok-4.20
    for o in order_set:
        if o.split("/")[-1] == s or o == s:
            return o
    return None


def _parse_int(x: str | None) -> int:
    if x is None or str(x).strip() == "":
        return 0
    try:
        return int(float(str(x).strip()))
    except ValueError:
        return 0


def _row_condition(row: dict[str, str], fieldnames: set[str]) -> str | None:
    if "condition" in fieldnames:
        return (row.get("condition") or "").strip() or None
    return None


def aggregate(
    results_root: Path,
    model_order: list[str],
) -> list[dict[str, object]]:
    order_set = set(model_order)
    by_pair: dict[tuple[str, str], CellAgg] = {}

    a_dir = results_root / "A"
    if a_dir.is_dir():
        globs = list(a_dir.rglob("results.csv"))
    else:
        globs = list(results_root.rglob("results.csv"))

    for csv_path in sorted(set(globs)):
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            continue
        a1 = 0
        a0 = 0
        valid = 0
        actor0: str | None = None
        dist0: str | None = None
        try:
            with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                fn = set(reader.fieldnames)
                if "actor_model" not in fn or "distinguisher_model" not in fn:
                    continue
                for row in reader:
                    c = _row_condition(row, fn) if "condition" in fn else None
                    if c is not None and c != CONDITION:
                        continue
                    actor = _norm_slug(row.get("actor_model", ""), order_set)
                    dist = _norm_slug(row.get("distinguisher_model", ""), order_set)
                    if not actor or not dist:
                        continue
                    if str(row.get("success", "")).lower() not in ("true", "1", "yes"):
                        continue
                    a = (row.get("answer") or "").strip()
                    if a not in ("0", "1"):
                        continue
                    if valid == 0:
                        actor0, dist0 = actor, dist
                    if actor0 != actor or dist0 != dist:
                        # Skip mixed-pair files (defensive; expected one pair per file)
                        break
                    valid += 1
                    if a == "1":
                        a1 += 1
                    else:
                        a0 += 1
        except OSError:
            continue
        if valid == 0 or actor0 is None or dist0 is None:
            continue
        key = (actor0, dist0)
        new = CellAgg(path=csv_path, mtime=mtime, valid=valid, a1=a1, a0=a0)
        old = by_pair.get(key)
        if old is None or new.mtime > old.mtime:
            by_pair[key] = new

    by_actor: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for a in model_order:
        for d in model_order:
            cell = by_pair.get((a, d))
            if cell is None:
                continue
            by_actor[a].append(
                {
                    "distinguisher": d,
                    "valid_trials": cell.valid,
                    "answer_1_count": cell.a1,
                    "answer_0_count": cell.a0,
                }
            )
    out = []
    for a in model_order:
        by_actor[a].sort(key=lambda r: (model_order.index(str(r["distinguisher"])), str(r["distinguisher"])))
        out.append({"actor": a, "rows": by_actor[a]})
    return out


def _load_grid(doc: dict[str, object]) -> dict[tuple[str, str], tuple[int, int, int]]:
    out: dict[tuple[str, str], tuple[int, int, int]] = {}
    for g in doc.get("records_by_actor") or []:
        actor = str(g.get("actor") or "").strip()
        for r in g.get("rows") or []:
            d = str(r.get("distinguisher") or "").strip()
            if not actor or not d:
                continue
            out[actor, d] = (
                int(r.get("valid_trials") or 0),
                int(r.get("answer_1_count") or 0),
                int(r.get("answer_0_count") or 0),
            )
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path, help="e.g. .../Turing Ordering Project/Results")
    p.add_argument(
        "--from-json",
        type=Path,
        help="Read ordering.model_order and (with --fill-missing) baseline cells (required).",
    )
    p.add_argument(
        "--fill-missing",
        action="store_true",
        help="For pairs absent in Drive, keep counts from --from-json (full 9x9 if baseline had them).",
    )
    p.add_argument("--out", type=Path, help="Write merged grouped JSON (optional).")
    args = p.parse_args()
    if not args.from_json:
        raise SystemExit("Need --from-json (template with model_order)")
    base_doc = json.loads(args.from_json.read_text(encoding="utf-8"))
    order = [str(x).strip() for x in (base_doc.get("ordering") or {}).get("model_order") or [] if str(x).strip()]
    if not order:
        raise SystemExit("Empty model_order")
    drive_blocks = aggregate(args.results_root, order)
    if args.fill_missing:
        base_grid = _load_grid(base_doc)
        by_pair: dict[tuple[str, str], tuple[int, int, int]] = dict(base_grid)
        for block in drive_blocks:
            a = str(block.get("actor") or "").strip()
            for r in block.get("rows") or []:
                d = str(r.get("distinguisher") or "").strip()
                by_pair[a, d] = (
                    int(r.get("valid_trials") or 0),
                    int(r.get("answer_1_count") or 0),
                    int(r.get("answer_0_count") or 0),
                )
        drive_only = sum(len(b["rows"]) for b in drive_blocks)
        records = []
        for a in order:
            rows: list[dict[str, int | str]] = []
            for d in order:
                vt, a1, a0 = by_pair.get((a, d), (0, 0, 0))
                if vt == 0 and a1 == 0 and a0 == 0:
                    continue
                rows.append(
                    {
                        "distinguisher": d,
                        "valid_trials": vt,
                        "answer_1_count": a1,
                        "answer_0_count": a0,
                    }
                )
            records.append({"actor": a, "rows": rows})
    else:
        records = drive_blocks
        drive_only = sum(len(b["rows"]) for b in drive_blocks)
    n_cells = sum(len(b["rows"]) for b in records)
    out_doc = {
        "row_count": 9 * 9,
        "fields": ["actor", "distinguisher", "valid_trials", "answer_1_count", "answer_0_count"],
        "ordering": {
            "grouped_by": "actor",
            "within_actor_sorted_by": "distinguisher",
            "model_order": order,
        },
        "records_by_actor": records,
    }
    print(
        f"Wrote {n_cells} non-empty directed cells. Drive contributed {drive_only} pair aggregates "
        f"(newest csv per pair in Results/A). Models: {len(order)}."
    )
    if args.out:
        args.out.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
