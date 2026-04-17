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
    const w = Math.max(140, 10 * n.label.length);
    const h = 44 + 16 * Math.min(4, n.members.length);
    g.setNode(n.id, { width: w, height: h });
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

