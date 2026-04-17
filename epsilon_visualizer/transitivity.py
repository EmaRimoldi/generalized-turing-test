"""Transitivity diagnostics and SCC-DAG heights (Turing_Ordering_Visualizer_Formalization_English.md).

Edges are the empirical relation A succeeds B at a fixed threshold (here: supported pairs only).
A violation is a triple (A, B, C) with A -> B and B -> C but not A -> C.

Height on the condensation DAG (meta-nodes): h(C) = 0 if no outgoing edges;
else 1 + max{h(D) : C -> D}. Larger h = higher in the hierarchy (sources above sinks).
"""

from __future__ import annotations

from typing import Iterable


def find_transitivity_violations(
    *,
    nodes: list[str],
    adj: dict[str, set[str]],
) -> tuple[int, list[tuple[str, str, str]]]:
    """Return (TV count, sorted list of violating triples (A, B, C)).

    adj[u] = {v | edge u -> v}.
    """
    node_set = set(nodes)
    triples: list[tuple[str, str, str]] = []
    for a in nodes:
        for b in sorted(adj.get(a, ())):
            if b not in node_set:
                continue
            for c in sorted(adj.get(b, ())):
                if c not in node_set:
                    continue
                if c not in adj.get(a, ()):
                    triples.append((a, b, c))
    triples.sort()
    return len(triples), triples


def condensation_heights(
    *,
    meta_nodes: list[str],
    condensed_edges: Iterable[tuple[str, str]],
) -> dict[str, int]:
    """Markdown height on the quotient DAG (after collapsing SCCs)."""
    out: dict[str, set[str]] = {m: set() for m in meta_nodes}
    for u, v in condensed_edges:
        if u in out and v in out:
            out[u].add(v)

    memo: dict[str, int] = {}

    def height(x: str) -> int:
        if x in memo:
            return memo[x]
        succ = out.get(x, set())
        if not succ:
            memo[x] = 0
            return 0
        h = 1 + max(height(y) for y in succ)
        memo[x] = h
        return h

    return {m: height(m) for m in meta_nodes}


def strict_pairs(
    *,
    nodes: list[str],
    adj: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """Strict: A > B iff A -> B and not B -> A (supported-edge view)."""
    s: set[tuple[str, str]] = set()
    for a in nodes:
        for b in adj.get(a, ()):
            if a not in adj.get(b, ()):
                s.add((a, b))
    return s
