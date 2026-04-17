from __future__ import annotations

from epsilon_visualizer.transitivity import condensation_heights, find_transitivity_violations


def test_transitivity_violation_chain_break():
    nodes = ["A", "B", "C"]
    adj = {"A": {"B"}, "B": {"C"}, "C": set()}
    tv, triples = find_transitivity_violations(nodes=nodes, adj=adj)
    assert tv == 1
    assert triples == [("A", "B", "C")]


def test_transitivity_none_when_transitive():
    nodes = ["A", "B", "C"]
    adj = {"A": {"B", "C"}, "B": {"C"}, "C": set()}
    tv, _ = find_transitivity_violations(nodes=nodes, adj=adj)
    assert tv == 0


def test_condensation_height_sinks_zero():
    # C0 -> C1, C1 -> C2: sinks C2 h=0, C1 h=1, C0 h=2
    h = condensation_heights(meta_nodes=["C0", "C1", "C2"], condensed_edges=[("C0", "C1"), ("C1", "C2")])
    assert h["C2"] == 0
    assert h["C1"] == 1
    assert h["C0"] == 2
