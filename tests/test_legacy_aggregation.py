"""Historical identifier normalization must not silently discard measured pairs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("actor", ["grok-4.20", "x-ai/grok-4.20", "x-ai/grok-2"])
def test_legacy_gttq_aggregator_matches_reference_model_order(tmp_path, actor):
    template = tmp_path / "reference.json"
    template.write_text(json.dumps({"ordering": {"model_order": ["x-ai/grok-4.20"]}}))
    run = tmp_path / "runs" / "B-001"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "actor_model,distinguisher_model,condition,answer\n"
        f"{actor},x-ai/grok-4.20,yes_query_no_explanation,1\n"
    )
    output = tmp_path / "out.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "visualizer/scripts/legacy/aggregate_condition_b_grouped_json.py"),
            "--results-b-root",
            str(run.parent),
            "--grouped-a-json",
            str(template),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    record = json.loads(output.read_text())["records_by_actor"][0]
    assert record["actor"] == "x-ai/grok-4.20"
    assert record["rows"] == [
        {
            "distinguisher": "x-ai/grok-4.20",
            "valid_trials": 1,
            "answer_1_count": 1,
            "answer_0_count": 0,
        }
    ]
