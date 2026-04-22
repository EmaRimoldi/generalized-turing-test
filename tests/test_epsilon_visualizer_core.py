from __future__ import annotations

import tempfile
from pathlib import Path

from epsilon_visualizer.graph import build_condensation_graph
from epsilon_visualizer.ingest import trials_from_grouped_model_pair_json, trials_from_matrix_a_aggregate_csv
from epsilon_visualizer.thresholding import classify_pair
from epsilon_visualizer.transitivity import find_transitivity_violations


def test_threshold_classification_supported_unsupported_missing():
    # supported: d_hat = 0.02 <= epsilon
    r = classify_pair(epsilon=0.05, estimate=0.52, upper=0.90, n_valid=10, min_valid=3)
    assert r.status == "supported"

    # unsupported: d_hat = 0.10 > epsilon
    r = classify_pair(epsilon=0.01, estimate=0.60, upper=0.70, n_valid=10, min_valid=3)
    assert r.status == "unsupported"

    # missing: insufficient n
    r = classify_pair(epsilon=0.2, estimate=0.49, upper=0.51, n_valid=2, min_valid=3)
    assert r.status == "missing"


def test_scc_collapse_only_supported_edges():
    models = ["A", "B", "C"]
    supported = {"A": {"B"}, "B": {"A", "C"}, "C": set()}
    cg = build_condensation_graph(models=models, supported_edges=supported)

    # A and B should be in same SCC
    node_a = cg.member_to_node["A"]
    node_b = cg.member_to_node["B"]
    assert node_a == node_b
    assert cg.member_to_node["C"] != node_a

    # Condensation should have an edge from {A,B} to {C}
    assert (node_a, cg.member_to_node["C"]) in cg.edges


def test_transitivity_violations_ignore_reflexive_cycle_triples():
    models = ["A", "B", "C"]
    supported = {"A": {"B"}, "B": {"A", "C"}, "C": set()}
    count, triples = find_transitivity_violations(nodes=models, adj=supported)

    assert count == 1
    assert triples == [("A", "B", "C")]


def test_epsilon_sweep_behavior_small_example():
    # A->B: d_hat = 0.02
    # At eps=0.01: unsupported (0.52 > 0.51)
    # At eps=0.02: supported (d_hat <= epsilon)
    # At eps=0.06: supported
    r = classify_pair(epsilon=0.01, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "unsupported"
    r = classify_pair(epsilon=0.02, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "supported"
    r = classify_pair(epsilon=0.06, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "supported"


def test_matrix_a_csv_skips_invalid_and_expands_counts():
    csv_text = """id,actor,distinguisher,num_trials,turn_limit,valid_trials,answer_1_count,answer_0_count,avg # turns
X-1,a/a,a/a,10,40,0,0,0,
X-2,a/a,b/b,10,40,10,3,7,2.4
"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.csv"
        p.write_text(csv_text, encoding="utf-8")
        trials = trials_from_matrix_a_aggregate_csv(p)
    assert len(trials) == 10
    assert sum(1 for t in trials if t.answer == 1) == 3
    assert sum(1 for t in trials if t.answer == 0) == 7
    assert all(t.turn_count == 2 for t in trials)
    assert all(t.branch == "imitation" for t in trials)


def test_grouped_json_full_grid_with_imputation():
    doc = {
        "ordering": {"model_order": ["m/a", "m/b"]},
        "records_by_actor": [
            {
                "actor": "m/a",
                "rows": [
                    {"distinguisher": "m/a", "valid_trials": 10, "answer_1_count": 10, "answer_0_count": 0},
                    {"distinguisher": "m/b", "valid_trials": 10, "answer_1_count": 2, "answer_0_count": 8},
                ],
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "g.json"
        p.write_text(__import__("json").dumps(doc), encoding="utf-8")
        trials, stats = trials_from_grouped_model_pair_json(
            p,
            matrix_fallback_csv=None,
            impute_uniform_if_missing=True,
        )
    assert stats == {"from_json": 2, "from_matrix_fallback": 0, "imputed_uniform": 2}
    assert len(trials) == 40
