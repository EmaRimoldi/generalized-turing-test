#!/usr/bin/env python3
"""Monitoraggio in tempo reale di RESULTS/: trial completati, falliti, CSV e totali.

Apri un secondo terminale mentre gira run_experiment_batch.py e lancia:

  uv run python scripts/watch_experiment_progress.py --results-dir RESULTS --interval 5

Una sola istantanea (nessun loop):

  uv run python scripts/watch_experiment_progress.py --results-dir RESULTS --once
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

TRIAL_RE = re.compile(r"^trial_(\d+)\.json$")
FAILED_RE = re.compile(r"^failed_attempt_")


def _scan_run(run_dir: Path) -> dict[str, object]:
    trials = sorted(
        int(m.group(1))
        for p in run_dir.iterdir()
        if p.is_file() and (m := TRIAL_RE.match(p.name))
    )
    n_failed = sum(1 for p in run_dir.iterdir() if p.is_file() and FAILED_RE.match(p.name))
    csv_path = run_dir / "results.csv"
    csv_rows = 0
    success_true = 0
    answer_1 = 0
    answer_0 = 0
    opening_invalid = 0
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        csv_rows = len(rows)
        for r in rows:
            if str(r.get("success", "")).lower() == "true":
                success_true += 1
            ans = r.get("answer")
            if ans == "1" or ans == 1:
                answer_1 += 1
            elif ans == "0" or ans == 0:
                answer_0 += 1
            if r.get("error") == "distinguisher_answered_in_opening":
                opening_invalid += 1

    return {
        "trials_on_disk": len(trials),
        "trial_ids": trials,
        "failed_attempts": n_failed,
        "csv_rows": csv_rows,
        "success_true": success_true,
        "answer_1": answer_1,
        "answer_0": answer_0,
        "opening_invalid": opening_invalid,
        "csv_complete": csv_path.is_file() and csv_rows > 0,
    }


def _print_report(
    results_root: Path,
    expected_trials: int | None,
    clear_screen: bool,
    *,
    show_watch_hint: bool = True,
) -> None:
    if clear_screen:
        sys.stdout.write("\033[2J\033[H")

    subdirs = sorted(p for p in results_root.iterdir() if p.is_dir())
    if not subdirs:
        print(f"(nessuna sottocartella in {results_root})")
        return

    total_valid = 0
    total_failed = 0

    print(f"=== Esperimenti sotto {results_root.resolve()} ===\n")

    for d in subdirs:
        info = _scan_run(d)
        valid = info["trials_on_disk"]
        failed = info["failed_attempts"]
        total_valid += valid
        total_failed += failed

        exp_note = ""
        if expected_trials is not None:
            exp_note = f"  [{valid}/{expected_trials} trial validi raccolti]"

        line1 = (
            f"{d.name}\n"
            f"  trial .json validi: {valid}{exp_note}  |  failed_attempt: {failed}"
        )
        if info["csv_complete"]:
            line1 += "  |  results.csv: OK (coppia completata)"
        elif valid > 0:
            line1 += "  |  results.csv: assente finché il run della coppia non termina"

        print(line1)
        if info["csv_rows"]:
            print(
                f"  CSV: righe={info['csv_rows']}  success={info['success_true']}  "
                f"answer=1: {info['answer_1']}  answer=0: {info['answer_0']}  "
                f"opening_invalid={info['opening_invalid']}"
            )
        print()

    if expected_trials is not None:
        print(
            f"--- Totali: trial validi (tutti i run): {total_valid}  |  "
            f"failed_attempt (tutti i run): {total_failed} ---"
        )
    else:
        print(
            f"--- Totali: trial validi: {total_valid}  |  failed_attempt: {total_failed} ---"
        )
    if show_watch_hint and not clear_screen:
        print("(Ctrl+C per uscire dal watch)\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("RESULTS"))
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Secondi tra un aggiornamento e il successivo (default: 5)",
    )
    parser.add_argument("--once", action="store_true", help="Stampa una volta ed esci")
    parser.add_argument(
        "--expected-trials",
        type=int,
        default=None,
        help="Se impostato (es. 10), mostra avanzamento trial/run rispetto al target",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Non cancellare lo schermo tra un refresh e l'altro",
    )
    args = parser.parse_args()
    root = args.results_dir
    if not root.is_dir():
        print(f"Cartella inesistente: {root}", file=sys.stderr)
        return 1

    clear = not args.no_clear
    if args.once:
        _print_report(root, args.expected_trials, clear_screen=False, show_watch_hint=False)
        return 0

    try:
        while True:
            _print_report(root, args.expected_trials, clear_screen=clear, show_watch_hint=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nUscita.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
