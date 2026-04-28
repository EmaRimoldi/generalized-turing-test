import dagre from "dagre";

export interface DagNode {
  id: string;
  label: string;
  members: string[];
}

export interface DagEdge {
  from: string;
  to: string;
  kind: "supported" | "tentative";
}

export interface LayoutResult {
  nodes: Array<DagNode & { x: number; y: number; w: number; h: number }>;
  edges: DagEdge[];
  width: number;
  height: number;
  /** Center Y of each imitation-difficulty stratum (for guide lines), top = harder. */
  guideLineYs?: number[];
}

export function layoutDag(
  nodes: DagNode[],
  edges: DagEdge[]
): LayoutResult {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: "TB",
    nodesep: 40,
    ranksep: 60,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of nodes) {
    const circleBox = Math.min(172, Math.max(118, 82 + n.label.length * 4, 110 + n.members.length * 6));
    g.setNode(n.id, { width: circleBox + 42, height: circleBox + 34 });
  }
  for (const e of edges) {
    g.setEdge(e.from, e.to);
  }
  dagre.layout(g);

  const laid = nodes.map((n) => {
    const pos = g.node(n.id) as { x: number; y: number; width: number; height: number };
    return { ...n, x: pos.x, y: pos.y, w: pos.width, h: pos.height };
  });

  const xs = laid.map((n) => n.x + n.w / 2);
  const ys = laid.map((n) => n.y + n.h / 2);
  const minX = Math.min(...xs, 0);
  const minY = Math.min(...ys, 0);
  const maxX = Math.max(...xs, 800);
  const maxY = Math.max(...ys, 600);
  return {
    nodes: laid,
    edges,
    width: maxX - minX + 80,
    height: maxY - minY + 80,
  };
}

/** Recompute width/height after manually changing node positions. */
/**
 * Within each horizontal "layer" (similar y, from dagre TB or applyHeightToY), reorder
 * left→right: higher `score` first (e.g. stricter distinguishers). Preserves the layer’s
 * min/max x so the global width stays similar.
 */
export function orderLayerNodesByScore(
  nodes: Array<{ id: string; x: number; y: number; w: number; h: number }>,
  score: (id: string) => number,
  yBucketPx = 24
): void {
  const layers = new Map<number, typeof nodes>();
  for (const n of nodes) {
    const L = Math.round(n.y / yBucketPx);
    if (!layers.has(L)) layers.set(L, []);
    layers.get(L)!.push(n);
  }
  for (const layer of layers.values()) {
    if (layer.length < 2) continue;
    layer.sort((a, b) => score(b.id) - score(a.id) || a.id.localeCompare(b.id));
    const left = Math.min(...layer.map((n) => n.x));
    const right = Math.max(...layer.map((n) => n.x));
    const span = right - left;
    const m = layer.length;
    for (let i = 0; i < m; i += 1) {
      layer[i].x = m <= 1 ? left : left + (span * i) / (m - 1);
    }
  }
}

export function recomputeViewBox(laid: LayoutResult): LayoutResult {
  const xs = laid.nodes.map((n) => n.x + n.w / 2);
  const ys = laid.nodes.map((n) => n.y + n.h / 2);
  const minX = Math.min(...xs, 0);
  const minY = Math.min(...ys, 0);
  const maxX = Math.max(...xs, 800);
  const maxY = Math.max(...ys, 600);
  return {
    ...laid,
    width: maxX - minX + 80,
    height: maxY - minY + 80,
  };
}

/**
 * Row Y = "difficulty to imitate" (defender mean pDiff on off-diagonals): same tier ⇒ same y.
 * X: evenly spread within the row (preserves left-to-right order from dagre for ties).
 * Returns y-centers of strata (for optional horizontal guide lines).
 */
export function applyImitationDifficultyTiers(
  nodes: Array<{ id: string; x: number; y: number; w: number; h: number }>,
  tierById: Map<string, number>,
  nTiers: number,
  opts: { yTop: number; yStride: number; xMargin: number; xGap: number; minCanvasWidth: number }
): number[] {
  const { yTop, yStride, xMargin, xGap, minCanvasWidth } = opts;
  if (nTiers <= 0) {
    return [];
  }
  const layerCenters: number[] = Array.from(
    { length: nTiers },
    (_, t) => yTop + t * yStride
  );

  for (const n of nodes) {
    const ti = Math.min(tierById.get(n.id) ?? 0, nTiers - 1);
    n.y = layerCenters[ti] ?? yTop;
  }

  const byTier = new Map<number, Array<(typeof nodes)[number]>>();
  for (const n of nodes) {
    const ti = Math.min(tierById.get(n.id) ?? 0, nTiers - 1);
    if (!byTier.has(ti)) byTier.set(ti, []);
    byTier.get(ti)!.push(n);
  }

  for (const t of [...byTier.keys()].sort((a, b) => a - b)) {
    const group = byTier.get(t);
    if (!group || group.length === 0) continue;
    group.sort((a, b) => a.x - b.x);
    const m = group.length;
    const totalNodesW = group.reduce((s, p) => s + p.w, 0);
    const needed = totalNodesW + (m - 1) * xGap;
    const width = Math.max(minCanvasWidth, needed + 2 * xMargin);
    const inner = width - 2 * xMargin;
    if (m === 1) {
      group[0].x = xMargin + inner / 2;
    } else {
      for (let i = 0; i < m; i += 1) {
        const frac = i / (m - 1);
        group[i].x = xMargin + frac * inner;
      }
    }
  }

  return layerCenters;
}
