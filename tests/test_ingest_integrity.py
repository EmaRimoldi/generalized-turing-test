"""Regression cases for incomplete and malformed experimental evidence."""
import json
from pathlib import Path

import pytest

from epsilon_visualizer.graph import build_condensation_graph, build_supported_graph
from epsilon_visualizer.ingest import (
    IngestConfig, build_visualizer_artifact, compute_pair_evidence_strict,
    pair_counts_from_grouped_model_pair_json, trials_from_grouped_model_pair_json,
)
from epsilon_visualizer.thresholding import classify_pair
from epsilon_visualizer.types import TrialRecord


def trial(actor, judge, *, condition="A", turns=2, provenance="grouped_json"):
    return TrialRecord(actor, judge, condition, None, 1, True, None, False,
                       turns, None, None, provenance)


def grouped(tmp_path, models=("a", "b"), rows=None):
    path = tmp_path / "grouped.json"
    path.write_text(json.dumps({
        "ordering": {"model_order": list(models)},
        "records_by_actor": [{"actor": models[0], "rows": rows or []}],
    }))
    return path


def test_self_pair_uses_its_own_metadata():
    rows = compute_pair_evidence_strict(trials=[
        trial("a", "a", turns=2, provenance="matrix_csv"),
        trial("b", "b", turns=8), trial("a", "b", turns=5),
    ], cfg=IngestConfig(Path("."), .05, min_valid_per_branch=1))
    self_rows = {p.source: p for p in rows if p.source == p.target}
    assert self_rows["a"].avg_turn_count == 2
    assert self_rows["a"].genuine_data_provenance == "matrix_csv"
    assert self_rows["b"].avg_turn_count == 8


def test_missing_self_pair_does_not_crash():
    rows = compute_pair_evidence_strict(trials=[trial("a", "b")],
                                       cfg=IngestConfig(Path("."), .05))
    assert len(rows) == 4
    assert all(p.status_at_epsilon == "missing" for p in rows)


def test_conditions_keep_separate_model_universes():
    rows = compute_pair_evidence_strict(trials=[
        trial("a", "a"), trial("b", "b", condition="B"),
    ], cfg=IngestConfig(Path("."), .05))
    assert [(p.condition, p.source, p.target) for p in rows] == [("A", "a", "a"), ("B", "b", "b")]


def test_no_impute_preserves_missing_models_and_cells(tmp_path):
    path = grouped(tmp_path)
    trials, stats = trials_from_grouped_model_pair_json(path, impute_uniform_if_missing=False)
    assert trials == []
    assert stats["imputed_uniform"] == 0
    artifact = build_visualizer_artifact(grouped_json=path, grouped_impute_uniform=False, epsilon=.05)
    assert artifact["models"] == ["a", "b"]
    assert len(artifact["pairs"]) == 4
    assert all(p["statusAtEpsilon"] == "missing" for p in artifact["pairs"])


@pytest.mark.parametrize("counts", [(3, -1, 4), (3, 1, 1), (3.5, 1.5, 2), (True, 1, 0)])
def test_malformed_counts_are_not_imputed_as_measurements(tmp_path, counts):
    n, one, zero = counts
    path = grouped(tmp_path, rows=[dict(distinguisher="a", valid_trials=n,
                                      answer_1_count=one, answer_0_count=zero)])
    with pytest.raises(ValueError):
        pair_counts_from_grouped_model_pair_json(path)


def test_duplicate_cells_rejected(tmp_path):
    row = dict(distinguisher="a", valid_trials=1, answer_1_count=1, answer_0_count=0)
    with pytest.raises(ValueError):
        pair_counts_from_grouped_model_pair_json(grouped(tmp_path, rows=[row, row]))


def test_matrix_fallback_uses_same_grok_id(tmp_path):
    path = grouped(tmp_path, models=("x-ai/grok-4.20",))
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("actor,distinguisher,valid_trials,answer_1_count,answer_0_count\n"
                      "grok-4.20,x-ai/grok-4.20,3,3,0\n")
    trials, stats = trials_from_grouped_model_pair_json(path, matrix_fallback_csv=matrix)
    assert stats["from_matrix_fallback"] == 1
    assert len(trials) == 3


@pytest.mark.parametrize("estimate", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_estimate_is_missing(estimate):
    assert classify_pair(epsilon=.05, estimate=estimate, upper=None,
                         n_valid=10, min_valid=3).status == "missing"


@pytest.mark.parametrize("epsilon", [float("nan"), -.1, .6])
def test_invalid_ingest_threshold_rejected(epsilon):
    with pytest.raises(ValueError):
        IngestConfig(Path("."), epsilon)


def test_condensation_respects_selected_models():
    graph = build_condensation_graph(models=["a", "b"],
                                     supported_edges={"a": {"b", "c"}, "b": set(), "c": {"a"}})
    assert set(graph.member_to_node) == {"a", "b"}


def test_supported_graph_recomputes_requested_epsilon():
    rows = compute_pair_evidence_strict(trials=[trial("a", "a"), trial("b", "b"), trial("a", "b")],
                                       cfg=IngestConfig(Path("."), .05, min_valid_per_branch=1))
    # Replace the estimate while retaining the originally stored classification.
    from dataclasses import replace
    row = replace(next(p for p in rows if p.source == "a" and p.target == "b"),
                  estimate=.7, status_at_epsilon="unsupported")
    assert build_supported_graph(models=["a", "b"], pair_evidence=[row], epsilon=.3, min_valid=1)["a"] == {"b"}


def test_independent_utt_judge_and_target_cannot_be_pooled_as_gtt():
    with pytest.raises(ValueError, match="nominated-target"):
        compute_pair_evidence_strict(trials=[trial("a", "b", condition="no_query_trusted_distinguisher")],
                                     cfg=IngestConfig(Path("."), .05))


@pytest.mark.parametrize("args", [(2, 1), (-1, 3), (1, -1)])
def test_invalid_binomial_counts_rejected(args):
    from epsilon_visualizer.stats import wilson_interval
    with pytest.raises(ValueError):
        wilson_interval(*args)


def test_explicit_missing_fallback_does_not_silently_impute(tmp_path):
    with pytest.raises(FileNotFoundError):
        trials_from_grouped_model_pair_json(grouped(tmp_path), matrix_fallback_csv=tmp_path / "absent.csv")


def test_matrix_duplicate_cells_are_rejected(tmp_path):
    from epsilon_visualizer.ingest import pair_counts_from_matrix_a_aggregate_csv
    path = tmp_path / "matrix.csv"
    row = "a,b,3,2,1\n"
    path.write_text("actor,distinguisher,valid_trials,answer_1_count,answer_0_count\n" + row + row)
    with pytest.raises(ValueError, match="Duplicate"):
        pair_counts_from_matrix_a_aggregate_csv(path)
