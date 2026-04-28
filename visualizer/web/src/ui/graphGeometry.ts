import type { DagEdge } from "./layout";

export type NodeWithRadius = { id: string; x: number; y: number; r: number; [k: string]: unknown };

/**
 * Quadratic edge between node centers (same logic as the main graph).
 */
export function edgePathDag(
  edge: DagEdge,
  idx: number,
  nodes: NodeWithRadius[],
  edges: DagEdge[]
): string {
  const from = nodes.find((n) => n.id === edge.from);
  const to = nodes.find((n) => n.id === edge.to);
  if (!from || !to) return "";

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / len;
  const uy = dy / len;
  const sx = from.x + ux * (from.r + 3);
  const sy = from.y + uy * (from.r + 3);
  const tx = to.x - ux * (to.r + 7);
  const ty = to.y - uy * (to.r + 7);
  const reciprocal = edges.some((e) => e.from === edge.to && e.to === edge.from);
  const nx = -uy;
  const ny = ux;
  const bend = reciprocal ? (edge.from < edge.to ? 48 : -48) : (idx % 2 === 0 ? 22 : -22);
  const cx = (sx + tx) / 2 + nx * bend;
  const cy = (sy + ty) / 2 + ny * bend;
  return `M ${sx.toFixed(2)} ${sy.toFixed(2)} Q ${cx.toFixed(2)} ${cy.toFixed(2)} ${tx.toFixed(2)} ${ty.toFixed(2)}`;
}

/**
 * Move circles apart **horizontally** only; same as the main app graph.
 */
export function resolveCircleCollisionsExport<T extends NodeWithRadius>(nodes: T[]): {
  nodes: T[];
  width: number;
  height: number;
} {
  const solved = nodes.map((n) => ({ ...n }));
  const margin = 34;
  const maxIterations = 120;

  for (let iter = 0; iter < maxIterations; iter += 1) {
    let moved = false;
    for (let i = 0; i < solved.length; i += 1) {
      for (let j = i + 1; j < solved.length; j += 1) {
        const a = solved[i]!;
        const b = solved[j]!;
        const dy = b.y - a.y;
        let dx = b.x - a.x;
        const dist = Math.hypot(dx, dy);
        const minDist = a.r + b.r + margin;
        if (dist >= minDist) continue;

        if (dist < 1e-3) {
          const half = minDist * 0.5;
          a.x -= half;
          b.x += half;
          moved = true;
          continue;
        }

        const needMag = Math.sqrt(Math.max(0, minDist * minDist - dy * dy));
        if (needMag <= 0) continue;

        const adx = Math.abs(dx);
        if (adx >= needMag - 1e-3) continue;

        const shortfall = (needMag - adx) * 0.5;
        if (a.x <= b.x) {
          a.x -= shortfall;
          b.x += shortfall;
        } else {
          a.x += shortfall;
          b.x += shortfall;
        }
        moved = true;
      }
    }
    if (!moved) break;
  }

  const minX = Math.min(...solved.map((n) => n.x - n.r - 56), 0);
  const minY = Math.min(...solved.map((n) => n.y - n.r - 56), 0);
  if (minX < 0 || minY < 0) {
    for (const n of solved) {
      n.x += Math.max(0, -minX);
      n.y += Math.max(0, -minY);
    }
  }

  const rightPad = 120;
  const width = Math.max(...solved.map((n) => n.x + n.r + 96 + rightPad), 1180);
  const height = Math.max(...solved.map((n) => n.y + n.r + 130), 660);
  return { nodes: solved, width, height };
}
