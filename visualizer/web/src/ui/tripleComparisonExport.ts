import {
  applyImitationDifficultyTiers,
  layoutDag,
  recomputeViewBox,
  type DagEdge,
  type DagNode,
  type LayoutResult,
} from "./layout";
import { strictnessTiers } from "./logic";
import { edgePathDag, resolveCircleCollisionsExport, type NodeWithRadius } from "./graphGeometry";
import { clusterLabel, labelFontSize, labelLines, nodeRadius } from "./graphLabels";

const GRAPH_PAD_TOP = 48;
const COL_GAP = 36;
const SVG_STYLE = `
  .te { fill: none; stroke: #28f0b8; stroke-width: 2.6; stroke-linecap: round; opacity: 0.86; }
  .tt { fill: none; stroke: #f2d84a; stroke-width: 2; stroke-dasharray: 8 9; stroke-linecap: round; opacity: 0.78; }
  .nc { fill: #0d1210; stroke: #d9fff4; stroke-width: 1.6; }
  .nr { fill: none; stroke: #28f0b8; stroke-width: 0.9; opacity: 0.36; }
  .nl { fill: #f3fff9; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-weight: 760; pointer-events: none; }
  .ns { fill: #6eb8a4; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-weight: 650; font-size: 9px; pointer-events: none; }
  .nsb { fill: #4a6d64; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-weight: 600; font-size: 8px; pointer-events: none; }
  .colt { fill: #b8d4c9; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 13px; }
  .gl { fill: none; stroke: #2a4a42; stroke-width: 0.8; stroke-opacity: 0.5; stroke-dasharray: 5 7; }
`;

type ExportNode = DagNode & {
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
  shortLabel: string;
  fontSize: number;
};

function laidToExportNodes(laid: LayoutResult, withShift: { yShift: number; guideLineYs: number[] }): {
  nodes: ExportNode[];
  guideLineYs: number[];
} {
  const guideLineYs = withShift.guideLineYs.map((y) => y + withShift.yShift);
  const nodes: ExportNode[] = laid.nodes.map((n) => {
    const shortLabel = clusterLabel(n.members);
    const r = nodeRadius(shortLabel, n.members.length);
    const fontSize = labelFontSize(shortLabel, n.members.length);
    return {
      ...n,
      y: n.y + withShift.yShift,
      r,
      shortLabel,
      fontSize,
    };
  });
  return { nodes, guideLineYs };
}

function buildStratifiedColumn(
  nodes: DagNode[],
  edges: DagEdge[],
  scoreByNodeId: Map<string, number>,
  tierOpt: { yTop: number; yStride: number; xMargin: number; xGap: number; minCanvasWidth: number }
): { render: { nodes: ExportNode[]; width: number; height: number; guideLineYs: number[] } } {
  const base = layoutDag(nodes, edges);
  const idToScore = new Map<string, number>();
  for (const n of base.nodes) {
    idToScore.set(n.id, scoreByNodeId.get(n.id) ?? 0);
  }
  const { tierById, nTiers } = strictnessTiers(idToScore);
  const nodeCopy = base.nodes.map((n) => ({ ...n }));
  const guideLineYsRaw =
    nTiers > 0
      ? applyImitationDifficultyTiers(nodeCopy, tierById, nTiers, tierOpt)
      : [];
  const laid = recomputeViewBox({ ...base, nodes: nodeCopy, guideLineYs: guideLineYsRaw });

  const minY = Math.min(...laid.nodes.map((n) => n.y - (nodeRadius(clusterLabel(n.members), n.members.length) + 56)), 0);
  const yShift = Math.max(0, -minY);
  const { nodes: withR, guideLineYs } = laidToExportNodes(laid, { yShift, guideLineYs: laid.guideLineYs ?? [] });

  const coll = resolveCircleCollisionsExport(withR as NodeWithRadius[]);
  return { render: { ...coll, guideLineYs, nodes: coll.nodes as ExportNode[] } };
}

function escapeXml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

function columnSvgFragment(
  col: { nodes: ExportNode[]; width: number; height: number; guideLineYs: number[] },
  edges: DagEdge[],
  opts: {
    scoreJ: Map<string, number>;
    scoreA: Map<string, number>;
    scoreS: Map<string, number>;
    mode: "combined" | "actor" | "judge";
  }
): string {
  const { scoreJ, scoreA, scoreS, mode } = opts;
  const lines: string[] = [];
  for (const y of col.guideLineYs) {
    lines.push(
      `<line class="gl" x1="0" y1="${y.toFixed(2)}" x2="${col.width}" y2="${y.toFixed(2)}" />`
    );
  }
  edges.forEach((edge, idx) => {
    const d = edgePathDag(edge, idx, col.nodes, edges);
    if (!d) return;
    const cls = edge.kind === "supported" ? "te" : "tt";
    const mid = edge.kind === "supported" ? "url(#m1)" : "url(#m2)";
    lines.push(`<path class="${cls}" d="${d}" marker-end="${mid}" />`);
  });
  for (const node of col.nodes) {
    const ll = labelLines(node.shortLabel);
    lines.push(
      `<circle class="nr" cx="${node.x}" cy="${node.y}" r="${node.r + 8}" />`,
      `<circle class="nc" cx="${node.x}" cy="${node.y}" r="${node.r}" />`
    );
    ll.forEach((line, lineIdx) => {
      const ty =
        node.y + (lineIdx - (ll.length - 1) / 2) * (node.fontSize + 2) + node.fontSize / 3;
      lines.push(
        `<text class="nl" x="${node.x}" y="${ty}" text-anchor="middle" font-size="${node.fontSize}">${escapeXml(line)}</text>`
      );
    });
    if (node.members.length > 1) {
      lines.push(
        `<text class="nsb" x="${node.x}" y="${node.y + node.r - 12}" text-anchor="middle">SCC</text>`
      );
    }
    const tx = node.x + node.r + 10;
    const sS = (scoreS.get(node.id) ?? 0).toFixed(3);
    const sJ = (scoreJ.get(node.id) ?? 0).toFixed(3);
    const sA = (scoreA.get(node.id) ?? 0).toFixed(3);
    if (mode === "combined") {
      lines.push(
        `<text class="ns" x="${tx}" y="${node.y - 4}" text-anchor="start">T=${sS}</text>`,
        `<text class="nsb" x="${tx}" y="${node.y + 8}" text-anchor="start">D=${sJ}  F=${sA}</text>`
      );
    } else if (mode === "actor") {
      lines.push(`<text class="ns" x="${tx}" y="${node.y + 2}" text-anchor="start">F=${sA}</text>`);
    } else {
      lines.push(`<text class="ns" x="${tx}" y="${node.y + 2}" text-anchor="start">D=${sJ}</text>`);
    }
  }
  return lines.join("\n");
}

export type TripleExportInput = {
  nodes: DagNode[];
  edges: DagEdge[];
  /** Graph node id → score (per node or per-cluster max over members) */
  scoreS: Map<string, number>;
  scoreJ: Map<string, number>;
  scoreA: Map<string, number>;
  graphView: "raw" | "condensation";
};

export function buildScoreMapsForGraphNodes(
  nodes: DagNode[],
  modelS: Map<string, number>,
  modelJ: Map<string, number>,
  modelA: Map<string, number>
): { scoreS: Map<string, number>; scoreJ: Map<string, number>; scoreA: Map<string, number> } {
  const scoreS = new Map<string, number>();
  const scoreJ = new Map<string, number>();
  const scoreA = new Map<string, number>();
  for (const n of nodes) {
    if (n.members.length === 1) {
      const m = n.members[0]!;
      scoreS.set(n.id, modelS.get(m) ?? 0);
      scoreJ.set(n.id, modelJ.get(m) ?? 0);
      scoreA.set(n.id, modelA.get(m) ?? 0);
    } else {
      scoreS.set(n.id, Math.max(0, ...n.members.map((x) => modelS.get(x) ?? 0)));
      scoreJ.set(n.id, Math.max(0, ...n.members.map((x) => modelJ.get(x) ?? 0)));
      scoreA.set(n.id, Math.max(0, ...n.members.map((x) => modelA.get(x) ?? 0)));
    }
  }
  return { scoreS, scoreJ, scoreA };
}

/**
 * Full-width SVG string: three parallel stratifications (S, A-only, J-only) with scores to the right of nodes.
 */
export function buildTripleComparisonSvgString(input: TripleExportInput): string {
  const { nodes, edges, scoreS, scoreJ, scoreA, graphView } = input;
  const tierOpt =
    graphView === "condensation"
      ? { yTop: 96, yStride: 168, xMargin: 88, xGap: 40, minCanvasWidth: 1100 }
      : { yTop: 88, yStride: 156, xMargin: 88, xGap: 40, minCanvasWidth: 1100 };

  const c1 = buildStratifiedColumn(nodes, edges, scoreS, tierOpt);
  const c2 = buildStratifiedColumn(nodes, edges, scoreA, tierOpt);
  const c3 = buildStratifiedColumn(nodes, edges, scoreJ, tierOpt);

  const title1 = "Turing score T = ½D + ½F";
  const title2 = "Fooling score F";
  const title3 = "Distinguishing score D";

  const innerH = Math.max(c1.render.height, c2.render.height, c3.render.height);
  const h = GRAPH_PAD_TOP + 18 + innerH;
  const w = c1.render.width + COL_GAP + c2.render.width + COL_GAP + c3.render.width;

  const mkCol = (
    x: number,
    title: string,
    col: typeof c1.render,
    mode: "combined" | "actor" | "judge"
  ) => `
  <g transform="translate(${x}, ${GRAPH_PAD_TOP})">
    <text class="colt" x="${col.width / 2}" y="0" text-anchor="middle">${escapeXml(title)}</text>
    <g transform="translate(0, 18)">
      ${columnSvgFragment(col, edges, { scoreJ, scoreA, scoreS, mode })}
    </g>
  </g>`;

  const body =
    mkCol(0, title1, c1.render, "combined") +
    mkCol(c1.render.width + COL_GAP, title2, c2.render, "actor") +
    mkCol(c1.render.width + COL_GAP + c2.render.width + COL_GAP, title3, c3.render, "judge");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
  <style>${SVG_STYLE}</style>
  <defs>
    <marker id="m1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#28f0b8" />
    </marker>
    <marker id="m2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#f2d84a" />
    </marker>
  </defs>
  <rect x="0" y="0" width="100%" height="100%" fill="#050706" />
  ${body}
</svg>`;
}

export async function downloadSvgStringAsPng(svgString: string, fileBase: string): Promise<void> {
  const dom = new DOMParser().parseFromString(svgString, "image/svg+xml");
  const svg = dom.documentElement;
  if (!svg || !svg.getAttribute("viewBox")) {
    const match = /viewBox="0 0 ([\d.]+) ([\d.]+)"/.exec(svgString);
    if (!match) throw new Error("No viewBox in triple SVG");
    const wi = match[1];
    const hi = match[2];
    if (!dom.documentElement.getAttribute("viewBox")) {
      dom.documentElement.setAttribute("viewBox", `0 0 ${wi} ${hi}`);
    }
  }
  if (!dom.documentElement.getAttribute("xmlns")) {
    dom.documentElement.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  }
  const w = dom.documentElement.getAttribute("width");
  const h = dom.documentElement.getAttribute("height");
  if (!w || !h) throw new Error("SVG size missing");
  const width = Number(w);
  const height = Number(h);
  const scale = 2;
  const ser = new XMLSerializer().serializeToString(dom);
  const blob = new Blob([ser], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  const loaded = new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("Triple PNG: image decode failed"));
  });
  image.src = url;
  await image.decode().catch(() => loaded);

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not available");
  ctx.fillStyle = "#050706";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);

  const out = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = out;
  link.download = `${fileBase}.png`;
  link.click();
}

export async function downloadTripleComparisonPng(input: TripleExportInput): Promise<void> {
  const svg = buildTripleComparisonSvgString(input);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  await downloadSvgStringAsPng(svg, `turing-triple-hierarchy-${stamp}`);
}
