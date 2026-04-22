import katex from "katex";
import "katex/dist/katex.min.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { PairRow, PairStatus, VisualizerArtifact } from "./types";
import { buildCondensation } from "./graphCompression";
import { defenderImitationStrictness, effectiveOrderingStatus, formatPct } from "./logic";
import { layoutDag, orderLayerNodesByScore, recomputeViewBox, type DagEdge, type DagNode } from "./layout";
import {
  applyHeightToY,
  buildSupportedAdjacency,
  condensationHeights,
  findTransitivityViolations,
} from "./transitivity";
import "./App.css";

const GRAPH_SVG_STYLE = `
  .edge-supported { fill: none; stroke: #28f0b8; stroke-width: 3; stroke-linecap: round; opacity: 0.86; }
  .edge-tentative { fill: none; stroke: #f2d84a; stroke-width: 2.4; stroke-dasharray: 8 9; stroke-linecap: round; opacity: 0.78; }
  .node-core { fill: #0d1210; stroke: #d9fff4; stroke-width: 1.8; }
  .node-core.selected { stroke: #ff4f83; stroke-width: 3; }
  .node-core.violation { stroke: #f2d84a; stroke-width: 2.6; }
  .node-ring { fill: none; stroke: #28f0b8; stroke-width: 1; opacity: 0.36; }
  .node-label { fill: #f3fff9; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-weight: 760; letter-spacing: 0; pointer-events: none; }
  .node-sub { fill: #90b7aa; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-weight: 650; letter-spacing: 0; pointer-events: none; }
`;

function useArtifact(): {
  loading: boolean;
  error: string | null;
  data: VisualizerArtifact | null;
} {
  const [data, setData] = useState<VisualizerArtifact | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const url = `${import.meta.env.BASE_URL}epsilon_visualizer_data.json`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${url}`);
        return r.json();
      })
      .then((j) => setData(j))
      .catch((e) => setError(String(e)));
  }, []);

  return { loading: !data && !error, error, data };
}

type Selected =
  | { kind: "none" }
  | { kind: "pair"; pair: PairRow }
  | { kind: "cluster"; clusterId: string; members: string[] };

type GraphNode = DagNode & {
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
  shortLabel: string;
  fontSize: number;
  inViolation: boolean;
};

type GraphRender = {
  nodes: GraphNode[];
  width: number;
  height: number;
};

function modelName(id: string): string {
  return id.split("/").pop() ?? id;
}

function conditionLabel(id: string): string {
  if (id === "no_query_no_explanation") return "Condition A · no query, no explanation";
  return id;
}

function clusterLabel(members: string[]): string {
  if (members.length === 1) return modelName(members[0]);
  return `${members.length} models`;
}

function labelLines(label: string): string[] {
  const parts = label.split(/[-_\s]+/).filter(Boolean);
  if (parts.length <= 1) return [label];
  const lines: string[] = [];
  let line = "";
  for (const part of parts) {
    const next = line ? `${line}-${part}` : part;
    if (next.length > 13 && line) {
      lines.push(line);
      line = part;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  if (lines.length <= 3) return lines;
  return [lines[0], lines[1], `${lines.slice(2).join("-").slice(0, 14)}...`];
}

function labelFontSize(label: string, memberCount: number): number {
  if (memberCount > 1) return 12;
  if (label.length <= 9) return 14;
  if (label.length <= 15) return 12;
  if (label.length <= 22) return 10;
  return 8;
}

function nodeRadius(label: string, memberCount: number): number {
  if (memberCount > 1) return Math.min(76, 54 + memberCount * 3);
  if (label.length <= 10) return 48;
  if (label.length <= 18) return 56;
  return 64;
}

function renderFormula(tex: string, displayMode = false): string {
  return katex.renderToString(tex, {
    displayMode,
    throwOnError: false,
    strict: false,
  });
}

function Formula({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = useMemo(() => renderFormula(tex, display), [tex, display]);
  return (
    <span
      className={display ? "math-display" : "math-inline"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/**
 * Separate overlapping node circles by moving **x only** so horizontal stratification
 * and dagre/height y-positions are preserved.
 */
function resolveCircleCollisions(nodes: GraphNode[]): GraphRender {
  const solved = nodes.map((n, idx) => ({ ...n, idx }));
  const margin = 34;
  const maxIterations = 120;

  for (let iter = 0; iter < maxIterations; iter += 1) {
    let moved = false;
    for (let i = 0; i < solved.length; i += 1) {
      for (let j = i + 1; j < solved.length; j += 1) {
        const a = solved[i];
        const b = solved[j];
        const dy = b.y - a.y;
        let dx = b.x - a.x;
        let dist = Math.hypot(dx, dy);
        const minDist = a.r + b.r + margin;
        if (dist >= minDist) continue;

        if (dist < 1e-3) {
          // Coincident centers: split along x; y stays fixed.
          const half = minDist * 0.5;
          a.x -= half;
          b.x += half;
          moved = true;
          continue;
        }

        // For fixed y, need |dx| >= sqrt(minDist^2 - dy^2) for Euclidean distance >= minDist.
        const needMag = Math.sqrt(Math.max(0, minDist * minDist - dy * dy));
        if (needMag <= 0) continue; // |dy| >= minDist: no 2D overlap; numerical noise only

        const adx = Math.abs(dx);
        if (adx >= needMag - 1e-3) continue;

        const shortfall = (needMag - adx) * 0.5;
        if (a.x <= b.x) {
          a.x -= shortfall;
          b.x += shortfall;
        } else {
          a.x += shortfall;
          b.x -= shortfall;
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

  const width = Math.max(...solved.map((n) => n.x + n.r + 96), 1180);
  const height = Math.max(...solved.map((n) => n.y + n.r + 96), 660);
  return { nodes: solved, width, height };
}

function statusColor(st: PairStatus): string {
  if (st === "supported") return "#28f0b8";
  if (st === "tentative") return "#f2d84a";
  if (st === "unsupported") return "#ff5a6f";
  return "#7a8883";
}

function edgePath(
  edge: DagEdge,
  idx: number,
  nodes: GraphNode[],
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

function viewBoxSize(viewBox: string): { width: number; height: number } {
  const parts = viewBox.split(/\s+/).map(Number);
  return { width: parts[2] || 1600, height: parts[3] || 900 };
}

async function exportSvgAsPng(svg: SVGSVGElement): Promise<void> {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const { width, height } = viewBoxSize(clone.getAttribute("viewBox") ?? "");
  const scale = 3;
  clone.setAttribute("width", String(width * scale));
  clone.setAttribute("height", String(height * scale));

  const text = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([text], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.decoding = "async";
  const loaded = new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("Snapshot image could not be decoded"));
  });
  image.src = url;
  await image.decode().catch(() => loaded);

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.fillStyle = "#050706";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  URL.revokeObjectURL(url);

  const pngUrl = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = pngUrl;
  link.download = `turing-ordering-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
  link.click();
}

export function App() {
  const { loading, error, data } = useArtifact();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [epsilon, setEpsilon] = useState(0.05);
  const [condition, setCondition] = useState<string>("");
  const [selected, setSelected] = useState<Selected>({ kind: "none" });
  const [graphView, setGraphView] = useState<"condensation" | "raw">("condensation");
  const [snapshotError, setSnapshotError] = useState<string | null>(null);

  useEffect(() => {
    if (!data) return;
    setEpsilon(data.epsilonDefault ?? 0.05);
    setCondition(data.conditions?.[0] ?? "");
  }, [data]);

  const minValid = data?.minValidPerBranch ?? 3;
  const strictPairs = useMemo(() => {
    if (!data) return [];
    return data.pairs.filter((p) => p.mode === "strict");
  }, [data]);

  const modelsForCondition = useMemo(() => {
    if (!data) return [];
    return data.modelsByCondition?.[condition] ?? data.models;
  }, [data, condition]);

  const modelsForConditionSet = useMemo(() => new Set(modelsForCondition), [modelsForCondition]);

  const pairsForCondition = useMemo(
    () =>
      strictPairs.filter(
        (p) =>
          p.condition === condition &&
          modelsForConditionSet.has(p.source) &&
          modelsForConditionSet.has(p.target)
      ),
    [strictPairs, condition, modelsForConditionSet]
  );

  const condensation = useMemo(() => {
    if (!data || !condition) return null;
    return buildCondensation({
      models: modelsForCondition,
      pairs: strictPairs,
      condition,
      epsilon,
      minValid,
    });
  }, [data, modelsForCondition, strictPairs, condition, epsilon, minValid]);

  const transitivity = useMemo(() => {
    if (!data) return { count: 0, triples: [] as Array<[string, string, string]> };
    const edges: Array<[string, string]> = [];
    for (const p of pairsForCondition) {
      if (effectiveOrderingStatus(p, epsilon, minValid) === "supported") edges.push([p.source, p.target]);
    }
    const adj = buildSupportedAdjacency(modelsForCondition, edges);
    return findTransitivityViolations(modelsForCondition, adj);
  }, [data, modelsForCondition, pairsForCondition, epsilon, minValid]);

  const violationMembers = useMemo(() => {
    const ids = new Set<string>();
    for (const triple of transitivity.triples) {
      for (const id of triple) ids.add(id);
    }
    return ids;
  }, [transitivity]);

  const laidCondensation = useMemo(() => {
    if (!data || !condensation) return null;
    const strictByModel = defenderImitationStrictness(strictPairs, condition, modelsForCondition);
    const clusterIds = condensation.clusters.map((c) => c.id);
    const heights = condensationHeights(clusterIds, condensation.supportedEdges);
    const nodes: DagNode[] = condensation.clusters.map((c) => ({
      id: c.id,
      label: clusterLabel(c.members),
      members: c.members,
    }));
    const scoreByClusterId = new Map<string, number>();
    for (const n of nodes) {
      const s =
        n.members.length > 0
          ? Math.max(...n.members.map((m) => strictByModel.get(m) ?? 0))
          : 0;
      scoreByClusterId.set(n.id, s);
    }
    const edges: DagEdge[] = [
      ...condensation.supportedEdges.map(([from, to]) => ({ from, to, kind: "supported" as const })),
      ...condensation.tentativeEdges.map(([from, to]) => ({ from, to, kind: "tentative" as const })),
    ];
    const base = layoutDag(nodes, edges);
    applyHeightToY(base.nodes, heights, 112);
    orderLayerNodesByScore(base.nodes, (id) => scoreByClusterId.get(id) ?? 0, 28);
    return recomputeViewBox(base);
  }, [data, condensation, strictPairs, condition, modelsForCondition]);

  const laidRaw = useMemo(() => {
    if (!data) return null;
    const strictByModel = defenderImitationStrictness(strictPairs, condition, modelsForCondition);
    const edges: DagEdge[] = [];
    for (const p of pairsForCondition) {
      const st = effectiveOrderingStatus(p, epsilon, minValid);
      if (st === "supported") edges.push({ from: p.source, to: p.target, kind: "supported" });
      if (st === "tentative") edges.push({ from: p.source, to: p.target, kind: "tentative" });
    }
    const nodes: DagNode[] = modelsForCondition.map((m) => ({
      id: m,
      label: modelName(m),
      members: [m],
    }));
    const base = layoutDag(nodes, edges);
    orderLayerNodesByScore(base.nodes, (id) => strictByModel.get(id) ?? 0, 22);
    return base;
  }, [data, modelsForCondition, pairsForCondition, strictPairs, condition, epsilon, minValid]);

  const summaryCounts = useMemo(() => {
    const c = { supported: 0, tentative: 0, unsupported: 0, missing: 0 };
    for (const p of pairsForCondition) {
      if (p.source === p.target) continue;
      const st = effectiveOrderingStatus(p, epsilon, minValid);
      c[st] += 1;
    }
    return c;
  }, [pairsForCondition, epsilon, minValid]);

  const displayLaid = graphView === "condensation" ? laidCondensation : laidRaw;
  const graphRender = useMemo<GraphRender>(() => {
    if (!displayLaid) return { nodes: [], width: 1180, height: 660 };
    const nodes = displayLaid.nodes.map((n) => {
      const shortLabel = clusterLabel(n.members);
      return {
        ...n,
        shortLabel,
        r: nodeRadius(shortLabel, n.members.length),
        fontSize: labelFontSize(shortLabel, n.members.length),
        inViolation: n.members.some((m) => violationMembers.has(m)),
      };
    });
    return resolveCircleCollisions(nodes);
  }, [displayLaid, violationMembers]);

  const graphNodes = graphRender.nodes;
  const graphViewBox = `0 0 ${graphRender.width} ${graphRender.height}`;

  const handleSnapshot = async () => {
    if (!svgRef.current) return;
    setSnapshotError(null);
    try {
      await exportSvgAsPng(svgRef.current);
    } catch (e) {
      setSnapshotError(String(e));
    }
  };

  if (loading) {
    return <div className="load-state">Loading strict evidence...</div>;
  }

  if (error) {
    return (
      <div className="load-state error-state">
        <h2>Data load failed</h2>
        <p>{error}</p>
        <code>python3 visualizer/scripts/serve_epsilon_visualizer.py</code>
      </div>
    );
  }

  if (!data) {
    return <div className="load-state">No data</div>;
  }

  const singleCondition = (data.conditions?.length ?? 0) <= 1;

  return (
    <div className="viz-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">strict evidence graph · Condition A only</p>
          <h1>Turing Ordering Field</h1>
        </div>
        <div className="stat-strip" aria-label="pair status counts">
          <StatusStat label="S" value={summaryCounts.supported} status="supported" />
          <StatusStat label="U" value={summaryCounts.unsupported} status="unsupported" />
          <StatusStat label="M" value={summaryCounts.missing} status="missing" />
          <div className="tv-pill">
            <span>TV_epsilon</span>
            <strong>{transitivity.count}</strong>
          </div>
        </div>
      </header>

      <section className="control-strip" aria-label="graph controls">
        <label>
          <span>Condition</span>
          {singleCondition ? (
            <span className="condition-static" title={condition}>
              {conditionLabel(condition)}
            </span>
          ) : (
            <select
              value={condition}
              onChange={(e) => {
                setCondition(e.target.value);
                setSelected({ kind: "none" });
              }}
            >
              {data.conditions.map((c) => (
                <option key={c} value={c}>
                  {conditionLabel(c)}
                </option>
              ))}
            </select>
          )}
        </label>

        <label className="epsilon-control">
          <span>epsilon {epsilon.toFixed(3)}</span>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.001}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
          />
          <small className="control-help">
            Higher epsilon is more permissive: more supported edges and possible SCC merges. Lower epsilon is stricter:
            fewer supported edges survive because d must be smaller.
          </small>
        </label>

        <label>
          <span>Graph</span>
          <select
            value={graphView}
            onChange={(e) => {
              setGraphView(e.target.value as "condensation" | "raw");
              setSelected({ kind: "none" });
            }}
          >
            <option value="condensation">SCC condensation</option>
            <option value="raw">Raw model graph</option>
          </select>
          <small className="control-help">
            Raw keeps every model separate. SCC condensation collapses mutually supported cycles into one circle.
          </small>
        </label>

        <button className="snapshot-button" type="button" onClick={handleSnapshot}>
          Snapshot HD
        </button>
      </section>

      {snapshotError && <div className="snapshot-error">{snapshotError}</div>}

      <section className="method-panel" aria-label="strict epsilon computation">
        <div>
          <p className="panel-kicker">Computation</p>
          {data.pairIngestStats && (
            <p className="ingest-stats">
              Directed pair sources: measured in grouped JSON{" "}
              <b>{data.pairIngestStats.from_json ?? 0}</b>, added from matrix fallback{" "}
              <b>{data.pairIngestStats.from_matrix_fallback ?? 0}</b>,               uniform imputation (10 trials, 5×‘1’ / 5×‘0’){" "}
              <b>{data.pairIngestStats.imputed_uniform ?? 0}</b>.
            </p>
          )}
          <p>
            For each actor <b>A</b> and distinguisher <b>B</b>, strict evidence combines the self-pair branch and the
            imitation branch:
          </p>
          <Formula
            display
            tex={
              "\\widehat P(\\mathrm{correct}) = \\frac{1}{2}\\widehat p(y=1\\mid B,B) + \\frac{1}{2}\\widehat p(y=0\\mid A,B)"
            }
          />
          <p>
            The distinguisher advantage above chance is:
          </p>
          <Formula display tex={"\\widehat d = \\widehat P(\\mathrm{correct}) - \\frac{1}{2}"} />
          <p>
            A <b>{"A -> B"}</b> edge is supported exactly when{" "}
            <Formula tex={"\\widehat d \\le \\varepsilon"} />, equivalently{" "}
            <Formula tex={"\\widehat P(\\mathrm{correct}) \\le \\frac{1}{2}+\\varepsilon"} />. Increasing{" "}
            <Formula tex={"\\varepsilon"} /> makes the relation more permissive; decreasing it demands that the
            distinguisher be closer to chance.
          </p>
          <p>
            For each branch, the estimate is the observed frequency <Formula tex={"\\widehat p=k/n"} />. The visualizer
            plugs those branch estimates into <Formula tex={"\\widehat P(\\mathrm{correct})"} /> and thresholds the
            resulting <Formula tex={"\\widehat d"} /> directly.
          </p>
        </div>
        <details className="legend-details">
          <summary>Legend</summary>
          <dl>
            <div>
              <dt>S</dt>
              <dd>supported: d is at most epsilon.</dd>
            </div>
            <div>
              <dt>U</dt>
              <dd>unsupported: d is larger than epsilon.</dd>
            </div>
            <div>
              <dt>M</dt>
              <dd>missing: strict-branch data are absent or insufficient.</dd>
            </div>
          </dl>
        </details>
      </section>

      <main className="workspace">
        <section className="graph-stage" aria-label="graph">
          <svg ref={svgRef} className="graph-canvas" viewBox={graphViewBox} role="img">
            <style>{GRAPH_SVG_STYLE}</style>
            <defs>
              <marker id="arrow-supported" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#28f0b8" />
              </marker>
              <marker id="arrow-tentative" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#f2d84a" />
              </marker>
              <pattern id="field-grid" width="42" height="42" patternUnits="userSpaceOnUse">
                <path d="M 42 0 H 0 V 42" fill="none" stroke="#18342d" strokeWidth="0.8" opacity="0.52" />
              </pattern>
            </defs>
            <rect x="0" y="0" width="100%" height="100%" fill="#050706" />
            <rect x="0" y="0" width="100%" height="100%" fill="url(#field-grid)" />
            {displayLaid?.edges.map((edge, idx) => (
              <path
                key={`${edge.from}-${edge.to}-${idx}`}
                className={edge.kind === "supported" ? "edge-supported" : "edge-tentative"}
                d={edgePath(edge, idx, graphNodes, displayLaid.edges)}
                markerEnd={edge.kind === "supported" ? "url(#arrow-supported)" : "url(#arrow-tentative)"}
              />
            ))}
            {graphNodes.map((node) => {
              const isSelected = selected.kind === "cluster" && selected.clusterId === node.id;
              const lines = labelLines(node.shortLabel);
              return (
                <g
                  key={node.id}
                  className="node-hit"
                  onClick={() => setSelected({ kind: "cluster", clusterId: node.id, members: node.members })}
                >
                  <circle className="node-ring" cx={node.x} cy={node.y} r={node.r + 10} />
                  <circle
                    className={`node-core${isSelected ? " selected" : ""}${node.inViolation ? " violation" : ""}`}
                    cx={node.x}
                    cy={node.y}
                    r={node.r}
                  />
                  {lines.map((line, lineIdx) => (
                    <text
                      key={`${line}-${lineIdx}`}
                      className="node-label"
                      x={node.x}
                      y={node.y + (lineIdx - (lines.length - 1) / 2) * (node.fontSize + 2) + node.fontSize / 3}
                      textAnchor="middle"
                      fontSize={node.fontSize}
                    >
                      {line}
                    </text>
                  ))}
                  {node.members.length > 1 && (
                    <text className="node-sub" x={node.x} y={node.y + node.r - 14} textAnchor="middle" fontSize={10}>
                      SCC
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </section>

        <aside className="inspector">
          <section>
            <p className="panel-kicker">Selection</p>
            {selected.kind === "none" && <p className="muted">Select a circle or matrix cell.</p>}
            {selected.kind === "cluster" && (
              <div>
                <h2>{selected.members.length === 1 ? modelName(selected.members[0]) : `${selected.members.length} models`}</h2>
                <ul className="member-list">
                  {selected.members.map((m) => (
                    <li key={m}>{modelName(m)}</li>
                  ))}
                </ul>
              </div>
            )}
            {selected.kind === "pair" && <PairDetails pair={selected.pair} epsilon={epsilon} minValid={minValid} />}
          </section>

          <section>
            <p className="panel-kicker">Transitivity</p>
            <h2>{transitivity.count} violating triples</h2>
            {transitivity.count === 0 ? (
              <p className="muted">No supported chain violates closure at this epsilon.</p>
            ) : (
              <ol className="triple-list">
                {transitivity.triples.slice(0, 12).map(([a, b, c]) => (
                  <li key={`${a}|${b}|${c}`}>
                    {modelName(a)}
                    {" -> "}
                    {modelName(b)}
                    {" -> "}
                    {modelName(c)}
                  </li>
                ))}
              </ol>
            )}
          </section>
        </aside>
      </main>

      <section className="matrix-section">
        <div className="matrix-heading">
          <h2>Actor / Distinguisher Matrix</h2>
          <p>
            S supported, U unsupported, M missing (no usable ordering edge: insufficient trials, missing run, or
            imitation/genuine branch filled only with uniform imputation). Diagonal “—” = self-pair (not an ordering
            edge).
          </p>
        </div>
        <PairMatrix
          models={modelsForCondition}
          pairs={pairsForCondition}
          epsilon={epsilon}
          minValid={minValid}
          onSelect={(pair) => setSelected({ kind: "pair", pair })}
        />
      </section>
    </div>
  );
}

function StatusStat({
  label,
  value,
  status,
}: {
  label: string;
  value: number;
  status: PairStatus;
}) {
  return (
    <div className="status-stat">
      <span style={{ color: statusColor(status) }}>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PairDetails({ pair, epsilon, minValid }: { pair: PairRow; epsilon: number; minValid: number }) {
  const st = effectiveOrderingStatus(pair, epsilon, minValid);
  return (
    <div className="pair-details">
      <h2>
        {modelName(pair.source)}
        {" -> "}
        {modelName(pair.target)}
      </h2>
      <div className="status-line">
        <span style={{ backgroundColor: statusColor(st) }} />
        {st}
      </div>
      <dl>
        <div>
          <dt>P(correct)</dt>
          <dd>{formatPct(pair.estimate)}</dd>
        </div>
        <div>
          <dt>d = P(correct) - 1/2</dt>
          <dd>{pair.dHat != null ? pair.dHat.toFixed(4) : "--"}</dd>
        </div>
        <div>
          <dt>epsilon* = max(0, d)</dt>
          <dd>{pair.epsilonStarHat != null ? pair.epsilonStarHat.toFixed(4) : "--"}</dd>
        </div>
        <div>
          <dt>nValid</dt>
          <dd>{pair.nValid}</dd>
        </div>
        <div>
          <dt>pSameRealB</dt>
          <dd>{formatPct(pair.details.pSameRealB)}</dd>
        </div>
        <div>
          <dt>pDiffImpostor</dt>
          <dd>{formatPct(pair.details.pDiffImpostor)}</dd>
        </div>
      </dl>
    </div>
  );
}

function PairMatrix({
  models,
  pairs,
  epsilon,
  minValid,
  onSelect,
}: {
  models: string[];
  pairs: PairRow[];
  epsilon: number;
  minValid: number;
  onSelect: (pair: PairRow) => void;
}) {
  const map = useMemo(() => {
    const m = new Map<string, PairRow>();
    for (const p of pairs) m.set(`${p.source}|||${p.target}`, p);
    return m;
  }, [pairs]);

  return (
    <div className="matrix-scroll">
      <table className="pair-matrix">
        <thead>
          <tr>
            <th>actor / distinguisher</th>
            {models.map((target) => (
              <th key={target}>{modelName(target)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((source) => (
            <tr key={source}>
              <th>{modelName(source)}</th>
              {models.map((target) => {
                const pair = map.get(`${source}|||${target}`);
                const isSelf = source === target;
                const st = !pair
                  ? "missing"
                  : isSelf
                    ? "self"
                    : effectiveOrderingStatus(pair, epsilon, minValid);
                return (
                  <td key={target}>
                    <button
                      type="button"
                      onClick={() => pair && onSelect(pair)}
                      disabled={!pair}
                      className={isSelf ? "cell-self" : `cell-${st}`}
                      title={
                        pair
                          ? isSelf
                            ? `${modelName(source)} self-pair: calibration only (not an ordering A→B edge)`
                            : `${modelName(source)} -> ${modelName(target)}: ${st}`
                          : "missing"
                      }
                    >
                      {isSelf ? "—" : st[0].toUpperCase()}
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
