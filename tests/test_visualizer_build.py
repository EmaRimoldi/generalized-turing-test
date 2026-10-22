from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multicondition_build_reproduces_published_evidence(tmp_path):
    output = tmp_path / "epsilon_visualizer_data.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "visualizer/scripts/build_multicondition_visualizer_data.py"),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(output.read_text(encoding="utf-8"))
    published = json.loads(
        (REPO_ROOT / "visualizer/web/public/epsilon_visualizer_data.json").read_text(encoding="utf-8")
    )

    assert actual["conditions"] == ["no_query_no_explanation", "yes_query_no_explanation"]
    assert len(actual["pairs"]) == 162
    assert actual == published
