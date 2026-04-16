#!/usr/bin/env python3
"""Aggregate experiment CSVs into summary tables and PNG visuals."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_result_csvs(results_roots: Iterable[Path]) -> List[Path]:
    csv_paths = set()
    for results_root in results_roots:
        csv_paths.update(results_root.glob("**/results.csv"))
    return sorted(csv_paths)


def load_rows(csv_paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def _to_int(value: object) -> int | None:
    if value in {"", "None", None}:
        return None
    return int(value)


def summarize_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["condition"],
                row.get("target_model", "") or "",
                row["actor_model"],
                row["distinguisher_model"],
            )
        ].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (condition, target_model, actor_model, distinguisher_model), items in sorted(grouped.items()):
        answers = [_to_int(item["answer"]) for item in items if _to_int(item["answer"]) is not None]
        turns = [_to_int(item["turns"]) for item in items if _to_int(item["turns"]) is not None]
        specimen_turns = [
            _to_int(item["specimen_turns"])
            for item in items
            if _to_int(item["specimen_turns"]) is not None
        ]
        completion_count = sum(str(item["success"]).lower() == "true" for item in items)
        answer_one_count = sum(answer == 1 for answer in answers)
        answer_zero_count = sum(answer == 0 for answer in answers)
        summary_rows.append(
            {
                "condition": condition,
                "target_model": target_model or None,
                "actor_model": actor_model,
                "distinguisher_model": distinguisher_model,
                "n_trials": len(items),
                "n_completed": completion_count,
                "completion_rate": completion_count / len(items),
                "n_answer_1": answer_one_count,
                "n_answer_0": answer_zero_count,
                "answer_1_rate": answer_one_count / len(items),
                "answer_0_rate": answer_zero_count / len(items),
                "mean_turns": mean(turns) if turns else None,
                "mean_specimen_turns": mean(specimen_turns) if specimen_turns else None,
            }
        )
    return summary_rows


def write_summary_csv(summary_rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "target_model",
        "actor_model",
        "distinguisher_model",
        "n_trials",
        "n_completed",
        "completion_rate",
        "n_answer_1",
        "n_answer_0",
        "answer_1_rate",
        "answer_0_rate",
        "mean_turns",
        "mean_specimen_turns",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def render_summary_table(summary_rows: List[Dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "condition",
        "target_model",
        "actor_model",
        "distinguisher_model",
        "n_trials",
        "completion_rate",
        "answer_1_rate",
        "mean_turns",
    ]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                row["condition"],
                "" if row["target_model"] is None else row["target_model"],
                row["actor_model"],
                row["distinguisher_model"],
                row["n_trials"],
                f"{row['completion_rate']:.2f}",
                f"{row['answer_1_rate']:.2f}",
                "" if row["mean_turns"] is None else f"{row['mean_turns']:.2f}",
            ]
        )

    fig_height = max(3.0, 0.4 * (len(table_rows) + 2))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    fig.suptitle("Turing Eval Summary Table", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_condition_heatmaps(summary_rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_condition: Dict[tuple[str, str | None], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_condition[(str(row["condition"]), row.get("target_model"))].append(row)

    outputs = []
    for (condition, target_model), rows in sorted(by_condition.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        actors = sorted({str(row["actor_model"]) for row in rows})
        distinguishers = sorted({str(row["distinguisher_model"]) for row in rows})
        matrix = [[math.nan for _ in distinguishers] for _ in actors]
        annotations = [["" for _ in distinguishers] for _ in actors]

        actor_index = {actor: idx for idx, actor in enumerate(actors)}
        distinguisher_index = {name: idx for idx, name in enumerate(distinguishers)}

        for row in rows:
            i = actor_index[str(row["actor_model"])]
            j = distinguisher_index[str(row["distinguisher_model"])]
            matrix[i][j] = float(row["answer_1_rate"])
            annotations[i][j] = f"p1={row['answer_1_rate']:.2f}\nn={row['n_trials']}"

        fig, ax = plt.subplots(figsize=(1.8 * len(distinguishers) + 3, 1.1 * len(actors) + 2))
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(distinguishers)), distinguishers, rotation=45, ha="right")
        ax.set_yticks(range(len(actors)), actors)
        title = f"{condition}: answer_1_rate"
        if target_model:
            title = f"{condition} [{target_model}]: answer_1_rate"
        ax.set_title(title)

        for i in range(len(actors)):
            for j in range(len(distinguishers)):
                if not math.isnan(matrix[i][j]):
                    ax.text(j, i, annotations[i][j], ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(answer=1)")
        fig.tight_layout()
        filename = condition if not target_model else f"{condition}__{target_model}"
        safe_filename = filename.replace("/", "_")
        out_path = output_dir / f"{safe_filename}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", dest="input_roots")
    parser.add_argument("--output-dir", default="results/summary")
    args = parser.parse_args()

    input_roots = [Path(root) for root in (args.input_roots or ["results"])]
    output_dir = Path(args.output_dir)
    csv_paths = discover_result_csvs(input_roots)
    if not csv_paths:
        formatted_roots = ", ".join(str(root) for root in input_roots)
        raise SystemExit(f"No results.csv files found under: {formatted_roots}")

    rows = load_rows(csv_paths)
    summary_rows = summarize_rows(rows)
    write_summary_csv(summary_rows, output_dir / "summary.csv")
    render_summary_table(summary_rows, output_dir / "summary_table.png")
    heatmaps = render_condition_heatmaps(summary_rows, output_dir)

    print(f"Loaded {len(rows)} trial rows from {len(csv_paths)} CSV files")
    print(f"Input roots: {', '.join(str(root) for root in input_roots)}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary_table.png'}")
    for path in heatmaps:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
