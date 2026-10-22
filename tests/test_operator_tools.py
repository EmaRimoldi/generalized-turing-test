"""Check launcher and aggregation failures without external services."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.render_results_summary import summarize_rows, render_condition_heatmaps, render_summary_table
from scripts.watch_experiment_progress import _print_report
from visualizer.scripts.build_grouped_condition_json_from_results import extract_answer, normalize_model_id
from epsilon_visualizer.transitivity import condensation_heights, strict_pairs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("hf", [False, True])
def test_container_wrapper_forwards_options_without_keys_or_gpu(tmp_path, hf):
    name = "run-in-container-hf.sh" if hf else "run-in-container.sh"
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / "apptainer").mkdir()
    shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "apptainer" / ("turing-eval-hf.sif" if hf else "turing-eval.sif")).write_text("stub")
    binary = tmp_path / "apptainer"
    binary.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.update(PATH=f"{tmp_path}{os.pathsep}{env['PATH']}", HF_CACHE_DIR=str(tmp_path / "cache"),
               TURING_EVAL_API_TIMEOUT_SECONDS="120", TURING_EVAL_API_HTTP_RETRIES="2")
    result = subprocess.run(["/bin/bash", str(repo / "scripts" / name), "python", "--version"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert "TURING_EVAL_API_TIMEOUT_SECONDS=120" in args
    assert "TURING_EVAL_API_HTTP_RETRIES=2" in args
    assert "" not in args
    assert args[-2:] == ["python", "--version"]


def test_opening_answer_recovery_ignores_user_prompt_examples():
    trial = {"error": "distinguisher_answered_in_opening", "answer": None,
             "distinguisher_messages": [{"role": "user", "content": "Use <answer>0</answer>"},
                                        {"role": "assistant", "content": "Hello"}]}
    assert extract_answer(trial) == (None, False)
    trial["distinguisher_messages"][-1]["content"] = "<answer>1</answer>"
    assert extract_answer(trial) == (1, True)


def test_failed_trial_and_boolean_are_not_counted_as_verdicts():
    assert extract_answer({"answer": 1, "success": False, "error": "timeout"}) == (None, False)
    assert extract_answer({"answer": True, "success": True}) == (None, False)


def test_different_grok_versions_require_explicit_legacy_alias():
    assert normalize_model_id("x-ai/grok-2") == "x-ai/grok-2"
    assert normalize_model_id("x-ai/grok-2", legacy_model_aliases=True) == "x-ai/grok-4.20"


def test_progress_report_does_not_claim_partial_run_completed(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "trial_000.json").write_text("{}")
    (run / "results.csv").write_text("success,answer,error\nTrue,1,\n")
    _print_report(tmp_path, expected_trials=10, clear_screen=False, show_watch_hint=False)
    output = capsys.readouterr().out
    assert "completata" not in output
    assert "target raggiunto" not in output
    _print_report(tmp_path, expected_trials=1, clear_screen=False, show_watch_hint=False)
    assert "target raggiunto" in capsys.readouterr().out


def test_summary_answer_rates_exclude_missing_and_invalid_verdicts(tmp_path):
    base = dict(condition="A", actor_model="a", distinguisher_model="b", success="True")
    rows = [{**base, "answer": value} for value in ("1", "", "7", "malformed")]
    summary = summarize_rows(rows)
    assert summary[0]["n_trials"] == 4
    assert summary[0]["n_answered"] == 1
    assert summary[0]["answer_1_rate"] == 1
    empty = summarize_rows([{**base, "answer": ""}])
    assert empty[0]["answer_1_rate"] is None
    render_summary_table(empty, tmp_path / "table.png")
    assert render_condition_heatmaps(empty, tmp_path)


def test_condensation_cycle_and_subset_diagnostics():
    with pytest.raises(ValueError, match="acyclic"):
        condensation_heights(meta_nodes=["a", "b"], condensed_edges=[("a", "b"), ("b", "a")])
    assert strict_pairs(nodes=["a"], adj={"a": {"b"}}) == set()
