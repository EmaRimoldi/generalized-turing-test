from __future__ import annotations

from epsilon_visualizer.graph import build_condensation_graph, transitive_reduction
from epsilon_visualizer.thresholding import classify_pair


def test_threshold_classification_supported_tentative_unsupported_missing():
    # supported: upper below threshold
    r = classify_pair(epsilon=0.05, estimate=0.52, upper=0.54, n_valid=10, min_valid=3)
    assert r.status == "supported"

    # tentative: point below, upper above
    r = classify_pair(epsilon=0.02, estimate=0.51, upper=0.54, n_valid=10, min_valid=3)
    assert r.status == "tentative"

    # unsupported: point above
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


def test_transitive_reduction_dag():
    nodes = ["X", "Y", "Z"]
    edges = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
    reduced = transitive_reduction(nodes, edges)
    assert ("X", "Z") not in reduced
    assert ("X", "Y") in reduced
    assert ("Y", "Z") in reduced


def test_epsilon_sweep_behavior_small_example():
    # A->B: estimate 0.52, upper 0.56
    # At eps=0.01: unsupported (0.52 > 0.51)
    # At eps=0.02: tentative (0.52 <= 0.52, upper 0.56 > 0.52)
    # At eps=0.06: supported (upper 0.56 <= 0.56)
    r = classify_pair(epsilon=0.01, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "unsupported"
    r = classify_pair(epsilon=0.02, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "tentative"
    r = classify_pair(epsilon=0.06, estimate=0.52, upper=0.56, n_valid=10, min_valid=3)
    assert r.status == "supported"


