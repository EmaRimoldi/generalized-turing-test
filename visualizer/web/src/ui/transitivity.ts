/** Transitivity TV_epsilon and SCC-DAG heights (see visualizer/docs/Turing_Ordering_Visualizer_Formalization_English.md). */

export function buildSupportedAdjacency(
  models: string[],
  edges: Array<[string, string]>
): Record<string, Set<string>> {
  const adj: Record<string, Set<string>> = {};
  for (const m of models) adj[m] = new Set();
  for (const [u, v] of edges) {
    if (adj[u]) adj[u].add(v);
  }
  return adj;
}

export function findTransitivityViolations(
  models: string[],
  adj: Record<string, Set<string>>
): { count: number; triples: Array<[string, string, string]> } {
  const triples: Array<[string, string, string]> = [];
  const set = new Set(models);
  for (const a of models) {
    for (const b of [...(adj[a] ?? [])].sort()) {
      if (!set.has(b)) continue;
      for (const c of [...(adj[b] ?? [])].sort()) {
        if (!set.has(c)) continue;
        if (new Set([a, b, c]).size < 3) continue;
        if (!(adj[a]?.has(c) ?? false)) triples.push([a, b, c]);
      }
    }
  }
  triples.sort((x, y) => x.join("\0").localeCompare(y.join("\0")));
  return { count: triples.length, triples };
}

/** Markdown height on condensation DAG: 0 at sinks, +1 along outgoing edges toward sources. */
export function condensationHeights(
  metaNodeIds: string[],
  edges: Array<[string, string]>
): Record<string, number> {
  const out: Record<string, Set<string>> = {};
  for (const id of metaNodeIds) out[id] = new Set();
  for (const [u, v] of edges) {
    if (out[u] && out[v]) out[u].add(v);
  }
  const memo = new Map<string, number>();
  function h(x: string): number {
    if (memo.has(x)) return memo.get(x)!;
    const succ = [...(out[x] ?? [])];
    if (succ.length === 0) {
      memo.set(x, 0);
      return 0;
    }
    const v = 1 + Math.max(...succ.map(h));
    memo.set(x, v);
    return v;
  }
  const r: Record<string, number> = {};
  for (const id of metaNodeIds) r[id] = h(id);
  return r;
}

/** Nudge node y so larger height sits higher on screen (smaller y). */
export function applyHeightToY(
  nodes: Array<{ id: string; x: number; y: number; w: number; h: number }>,
  heights: Record<string, number>,
  stride = 100
): void {
  const hs = Object.values(heights);
  const maxH = hs.length ? Math.max(...hs) : 0;
  for (const n of nodes) {
    const layer = maxH - (heights[n.id] ?? 0);
    n.y = stride * layer + n.h / 2 + 40;
  }
}
