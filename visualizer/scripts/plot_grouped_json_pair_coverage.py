#!/usr/bin/env python3
"""Plot directed-pair coverage for ``model_pair_results_grouped.json``.

Writes a PNG heatmap: measured in JSON vs missing (before merge/impute).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "visualizer" / "python"))
from epsilon_visualizer.ingest import pair_counts_from_grouped_model_pair_json  # noqa: E402


def short(mid: str) -> str:
    return mid.split("/")[-1] if "/" in mid else mid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grouped-json",
        default=str(REPO_ROOT / "visualizer/web/public/condition_a_model_pair_results.json"),
        help="Path to grouped Condition A JSON (model_pair_results_grouped schema)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "visualizer/output/figures/pair_coverage.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    grouped = Path(args.grouped_json).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(grouped.read_text(encoding="utf-8"))
    order, counts = pair_counts_from_grouped_model_pair_json(grouped)
    n = len(order)
    present = set(counts)
    condition = raw.get("ingest_notes", {}).get("condition") or grouped.stem

    mat = np.zeros((n, n), dtype=float)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            mat[i, j] = 1.0 if (a, b) in present else 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.colors.ListedColormap(["#5c2a2a", "#1f5c45"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([short(m) for m in order], rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels([short(m) for m in order], fontsize=8)
    ax.set_xlabel("Distinguisher (B)", fontsize=11)
    ax.set_ylabel("Actor (A)", fontsize=11)
    ax.set_title(
        f"{condition} — directed pair coverage in grouped JSON\n"
        "(dark green = row present; brown = missing before optional imputation)",
        fontsize=12,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["missing", "measured"])
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
