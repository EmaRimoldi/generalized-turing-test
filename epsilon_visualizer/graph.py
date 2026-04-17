from __future__ import annotations

from dataclasses import dataclass

from .types import PairEvidence


@dataclass(frozen=True)
class CondensationNode:
    id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class CondensationGraph:
    nodes: list[CondensationNode]
    edges: list[tuple[str, str]]  # node_id -> node_id
    member_to_node: dict[str, str]


def _tarjan_scc(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in edges.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes:
        if v not in indices:
            strongconnect(v)

    return sccs


def build_supported_graph(
    *,
    models: list[str],
    pair_evidence: list[PairEvidence],
    epsilon: float,
) -> dict[str, set[str]]:
    g: dict[str, set[str]] = {m: set() for m in models}
    for ev in pair_evidence:
        if ev.status_at_epsilon == "supported":
            g.setdefault(ev.source, set()).add(ev.target)
    return g


def build_condensation_graph(
    *,
    models: list[str],
    supported_edges: dict[str, set[str]],
) -> CondensationGraph:
    sccs = _tarjan_scc(models, supported_edges)
    # Deterministic node ids: sort members, then sort SCC list
    sorted_sccs = [tuple(sorted(comp)) for comp in sccs]
    sorted_sccs.sort(key=lambda t: (len(t), t))

    nodes: list[CondensationNode] = []
    member_to_node: dict[str, str] = {}
    for i, members in enumerate(sorted_sccs):
        node_id = f"C{i:03d}"
        nodes.append(CondensationNode(id=node_id, members=members))
        for m in members:
            member_to_node[m] = node_id

    condensed_edges: set[tuple[str, str]] = set()
    for src, tgts in supported_edges.items():
        for tgt in tgts:
            a = member_to_node[src]
            b = member_to_node[tgt]
            if a != b:
                condensed_edges.add((a, b))

    return CondensationGraph(
        nodes=nodes,
        edges=sorted(condensed_edges),
        member_to_node=member_to_node,
    )


def transitive_reduction(nodes: list[str], edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Transitive reduction for a DAG.

    Assumes the input graph is acyclic.
    """
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)

    def reachable(start: str, forbidden_edge: tuple[str, str]) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj.get(u, set()):
                if (u, v) == forbidden_edge:
                    continue
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    reduced: list[tuple[str, str]] = []
    for u, v in edges:
        # if v is reachable from u without using edge (u,v), then (u,v) is redundant
        if v not in reachable(u, (u, v)):
            reduced.append((u, v))
    return reduced

