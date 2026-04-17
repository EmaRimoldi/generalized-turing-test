import React, { useEffect, useMemo, useState } from "react";
import type { PairRow, VisualizerArtifact } from "./types";
import { buildCondensation } from "./graphCompression";
import { classifyAtEpsilon, formatPct } from "./logic";
import { layoutDag, recomputeViewBox, type DagEdge, type DagNode } from "./layout";
import {
  applyHeightToY,
  buildSupportedAdjacency,
  condensationHeights,
  findTransitivityViolations,
  transitiveReduction,
} from "./transitivity";

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

export function App() {
  const { loading, error, data } = useArtifact();
  const [epsilon, setEpsilon] = useState(0.05);
  const [condition, setCondition] = useState<string>("");
  const [mode, setMode] = useState<"strict" | "surrogate">("strict");
  const [selected, setSelected] = useState<Selected>({ kind: "none" });
  const [showReduction, setShowReduction] = useState(false);
  const [graphView, setGraphView] = useState<"condensation" | "raw">("condensation");

  useEffect(() => {
    if (!data) return;
    setEpsilon(data.epsilonDefault ?? 0.05);
    setCondition(data.conditions?.[0] ?? "");
    setMode((data.mode as "strict" | "surrogate") ?? "strict");
  }, [data]);

  const minValid = data?.minValidPerBranch ?? 3;

  const filteredPairs = useMemo(() => {
    if (!data) return [];
    return data.pairs.filter((p) => p.mode === mode);
  }, [data, mode]);

  const condensation = useMemo(() => {
    if (!data || !condition) return null;
    return buildCondensation({
      models: data.models,
      pairs: filteredPairs,
      condition,
      epsilon,
      minValid,
    });
  }, [data, filteredPairs, condition, epsilon, minValid]);

  const pairsForCond = useMemo(
    () => filteredPairs.filter((p) => p.condition === condition),
    [filteredPairs, condition]
  );

  const transitivity = useMemo(() => {
    if (!data) return { count: 0, triples: [] as Array<[string, string, string]> };
    const edges: Array<[string, string]> = [];
    for (const p of pairsForCond) {
      if (classifyAtEpsilon(p, epsilon, minValid) === "supported") edges.push([p.source, p.target]);
    }
    const adj = buildSupportedAdjacency(data.models, edges);
    return findTransitivityViolations(data.models, adj);
  }, [data, pairsForCond, epsilon, minValid]);

  const laid = useMemo(() => {
    if (!data || !condensation) return null;
    const clusterIds = condensation.clusters.map((c) => c.id);
    let sup = condensation.supportedEdges;
    if (showReduction && sup.length > 0) {
      sup = transitiveReduction(clusterIds, sup);
    }
    const heights = condensationHeights(clusterIds, sup);
    const nodes: DagNode[] = condensation.clusters.map((c) => ({
      id: c.id,
      label: c.members.length === 1 ? c.members[0] : `{${c.members.length}}`,
      members: c.members,
    }));
    const edges: DagEdge[] = [
      ...sup.map(([from, to]) => ({ from, to, kind: "supported" as const })),
      ...condensation.tentativeEdges.map(([from, to]) => ({
        from,
        to,
        kind: "tentative" as const,
      })),
    ];
    const base = layoutDag(nodes, edges);
    applyHeightToY(base.nodes, heights, 100);
    return recomputeViewBox(base);
  }, [data, condensation, showReduction]);

  /** Raw G_{ε,π}: may contain cycles; no transitive reduction / SCC height (those are for the quotient DAG). */
  const laidRaw = useMemo(() => {
    if (!data) return null;
    const edges: DagEdge[] = [];
    for (const p of pairsForCond) {
      const st = classifyAtEpsilon(p, epsilon, minValid);
      if (st === "supported") edges.push({ from: p.source, to: p.target, kind: "supported" });
      if (st === "tentative") edges.push({ from: p.source, to: p.target, kind: "tentative" });
    }
    const nodes: DagNode[] = data.models.map((m) => ({
      id: m,
      label: m,
      members: [m],
    }));
    return layoutDag(nodes, edges);
  }, [data, pairsForCond, epsilon, minValid]);

  const summaryCounts = useMemo(() => {
    const c = { supported: 0, tentative: 0, unsupported: 0, missing: 0 };
    for (const p of pairsForCond) {
      const st = classifyAtEpsilon(p, epsilon, minValid);
      c[st] += 1;
    }
    return c;
  }, [pairsForCond, epsilon, minValid]);

  if (loading) return <div style={{ padding: 16 }}>Loading…</div>;
  if (error)
    return (
      <div style={{ padding: 16, maxWidth: 720, fontFamily: "system-ui, sans-serif" }}>
        <h2 style={{ color: "crimson", marginTop: 0 }}>Impossibile caricare i dati</h2>
        <p style={{ color: "#333" }}>
          <b>Dettaglio:</b> {error}
        </p>
        <p style={{ color: "#333" }}>
          Apri l’app tramite un server HTTP locale (non aprire solo il file <code>index.html</code> con{" "}
          <code>file://</code>, altrimenti il browser blocca il caricamento del JSON).
        </p>
        <p style={{ color: "#333" }}>
          Dalla root del repo <code>llm-eval</code> esegui:
        </p>
        <pre
          style={{
            background: "#f4f4f4",
            padding: 12,
            borderRadius: 8,
            overflow: "auto",
          }}
        >
          python3 scripts/serve_epsilon_visualizer.py
        </pre>
        <p style={{ color: "#555", fontSize: 14 }}>
          Non serve <code>npm</code> sul cluster: lo script serve la cartella <code>epsilon_visualizer_app/dist</code>{" "}
          già compilata. Se <code>dist</code> manca, genera la build su una macchina con Node e copia la cartella, oppure
          chiedi un nodo interattivo e lancia <code>npm run build</code> in <code>epsilon_visualizer_app</code>.
        </p>
      </div>
    );
  if (!data) return <div style={{ padding: 16 }}>No data</div>;

  const displayLaid = graphView === "condensation" ? laid : laidRaw;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 12 }}>
      <h2 style={{ margin: "0 0 8px 0" }}>
        Epsilon Visualizer ({mode === "strict" ? "strict mode" : "surrogate mode"})
      </h2>
      <div style={{ marginBottom: 8, color: "#444", maxWidth: 920 }}>
        Turing ordering at threshold ε (see <code>{data.formalismRef ?? "formalization doc"}</code>). Edge{" "}
        <b>A → B</b> when the cell is <b>supported</b>: Wilson upper bound on plug-in P(correct) ≤ 1/2 + ε
        (equivalently ε* = max(0, d_hat) ≤ ε at the point estimate). SCC view collapses strong components using
        supported edges only; tentative edges are shown but do not merge clusters.
      </div>

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          padding: 10,
          border: "1px solid #ddd",
          borderRadius: 8,
          marginBottom: 12,
        }}
      >
        <label>
          Condition{" "}
          <select
            value={condition}
            onChange={(e) => {
              setCondition(e.target.value);
              setSelected({ kind: "none" });
            }}
          >
            {data.conditions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label>
          Mode{" "}
          <select
            value={mode}
            onChange={(e) => {
              setMode(e.target.value as "strict" | "surrogate");
              setSelected({ kind: "none" });
            }}
          >
            {(data.availableModes ?? ["strict"]).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label style={{ flex: 1 }}>
          ε = <b>{epsilon.toFixed(3)}</b>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.001}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>

        <label>
          Graph{" "}
          <select
            value={graphView}
            onChange={(e) => setGraphView(e.target.value as "condensation" | "raw")}
          >
            <option value="condensation">SCC condensation (buckets)</option>
            <option value="raw">Raw model graph</option>
          </select>
        </label>

        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={showReduction}
            onChange={(e) => setShowReduction(e.target.checked)}
          />
          transitive reduction (SCC DAG only)
        </label>

        <div style={{ marginLeft: "auto", fontSize: 12, color: "#333" }}>
          supported {summaryCounts.supported}, tentative {summaryCounts.tentative}
          , unsupported {summaryCounts.unsupported}, missing {summaryCounts.missing}
        </div>
      </div>

      <div
        style={{
          marginBottom: 12,
          padding: 10,
          border: "1px solid #e8b86d",
          borderRadius: 8,
          background: "#fffbf5",
          fontSize: 13,
        }}
      >
        <b>Transitivity diagnostic TV_ε</b> (supported edges on models): {transitivity.count} violating triples
        {transitivity.count > 0 && (
          <details style={{ marginTop: 6 }}>
            <summary>List (A, B, C) with A→B, B→C but not A→C</summary>
            <ul style={{ margin: "6px 0 0 0", maxHeight: 160, overflow: "auto" }}>
              {transitivity.triples.map(([a, b, c]) => (
                <li key={`${a}|${b}|${c}`}>
                  ({a}, {b}, {c})
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1.2fr", gap: 12 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
          <h3 style={{ margin: "0 0 8px 0" }}>
            Graph ({graphView === "condensation" ? "SCC condensation" : "raw models"})
          </h3>
          {graphView === "condensation" && (!laid || !condensation) ? (
            <div>Missing layout</div>
          ) : graphView === "raw" && !laidRaw ? (
            <div>Missing layout</div>
          ) : (
            <svg
              width="100%"
              viewBox={`0 0 ${displayLaid!.width} ${displayLaid!.height}`}
              style={{ background: "#fafafa", borderRadius: 8 }}
            >
              <defs>
                <marker
                  id="arrow"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#555" />
                </marker>
                <marker
                  id="arrowTent"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#999" />
                </marker>
              </defs>

              {displayLaid!.edges.map((e, idx) => {
                const from = displayLaid!.nodes.find((n) => n.id === e.from)!;
                const to = displayLaid!.nodes.find((n) => n.id === e.to)!;
                const x1 = from.x;
                const y1 = from.y + from.h / 2;
                const x2 = to.x;
                const y2 = to.y - to.h / 2;
                const dashed = e.kind === "tentative";
                return (
                  <line
                    key={idx}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={dashed ? "#999" : "#555"}
                    strokeWidth={dashed ? 2 : 2.5}
                    strokeDasharray={dashed ? "6 6" : undefined}
                    markerEnd={dashed ? "url(#arrowTent)" : "url(#arrow)"}
                  />
                );
              })}

              {displayLaid!.nodes.map((n) => {
                const isSelected =
                  selected.kind === "cluster" && selected.clusterId === n.id;
                return (
                  <g
                    key={n.id}
                    onClick={() =>
                      setSelected({ kind: "cluster", clusterId: n.id, members: n.members })
                    }
                    style={{ cursor: "pointer" }}
                  >
                    <rect
                      x={n.x - n.w / 2}
                      y={n.y - n.h / 2}
                      width={n.w}
                      height={n.h}
                      rx={8}
                      fill={isSelected ? "#e8f0ff" : "white"}
                      stroke={isSelected ? "#245bdb" : "#bbb"}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />
                    <text
                      x={n.x}
                      y={n.y - n.h / 2 + 18}
                      textAnchor="middle"
                      fontSize={12}
                      fill="#111"
                    >
                      {n.members.length === 1 ? n.members[0] : `Cluster (${n.members.length})`}
                    </text>
                    {n.members.length > 1 && (
                      <text
                        x={n.x}
                        y={n.y - n.h / 2 + 34}
                        textAnchor="middle"
                        fontSize={11}
                        fill="#555"
                      >
                        {n.members.slice(0, 3).join(", ")}
                        {n.members.length > 3 ? "…" : ""}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
          <h3 style={{ margin: "0 0 8px 0" }}>Details</h3>
          {selected.kind === "none" && (
            <div style={{ color: "#555" }}>
              Click a cluster in the graph, or a cell in the matrix.
            </div>
          )}
          {selected.kind === "cluster" && (
            <div>
              <div>
                <b>Cluster</b> {selected.clusterId}
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: "#333" }}>
                Members:
                <ul>
                  {selected.members.map((m) => (
                    <li key={m}>{m}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {selected.kind === "pair" && <PairDetails pair={selected.pair} epsilon={epsilon} minValid={minValid} />}
        </div>
      </div>

      <div style={{ marginTop: 12, border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
        <h3 style={{ margin: "0 0 8px 0" }}>Pair matrix (ordered pairs)</h3>
        {mode === "surrogate" && (
          <div
            style={{
              marginBottom: 8,
              padding: 8,
              borderRadius: 8,
              background: "#fff3cd",
              color: "#6b4e00",
            }}
          >
            <b>Surrogate mode:</b> this view thresholds an imitation-only proxy derived from `A vs B` trials.
            It is <b>not</b> the formal \(P_B^A(\mathrm{correct})\) quantity; use it to explore patterns when strict
            joins (self-pairs) are missing or unstable.
          </div>
        )}
        <PairMatrix
          models={data.models}
          pairs={pairsForCond}
          epsilon={epsilon}
          minValid={minValid}
          onSelect={(pair) => setSelected({ kind: "pair", pair })}
        />
      </div>
    </div>
  );
}

function statusColor(st: PairStatus): string {
  if (st === "supported") return "#1b8a3a";
  if (st === "tentative") return "#b08900";
  if (st === "unsupported") return "#b12a2a";
  return "#777";
}

function PairDetails({ pair, epsilon, minValid }: { pair: PairRow; epsilon: number; minValid: number }) {
  const st = classifyAtEpsilon(pair, epsilon, minValid);
  return (
    <div style={{ fontSize: 13 }}>
      <div>
        <b>{pair.source}</b> → <b>{pair.target}</b>
      </div>
      <div style={{ marginTop: 6 }}>
        status: <b style={{ color: statusColor(st) }}>{st}</b>
      </div>
      <div style={{ marginTop: 6, fontSize: 12, color: "#333" }}>
        <div>
          <b>{"d\u0302"}</b> (plug-in advantage) = <b>{"P\u0302"}</b>(correct) − ½:{" "}
          {pair.dHat != null ? pair.dHat.toFixed(4) : "—"}
        </div>
        <div>
          <b>{"\u03b5\u0302*"}</b> = max(0, {"d\u0302"}):{" "}
          {pair.epsilonStarHat != null ? pair.epsilonStarHat.toFixed(4) : "—"}
        </div>
        <div style={{ color: "#666", marginTop: 4 }}>
          Supported at ε uses Wilson upper on P(correct) ≤ ½ + ε (see formalization), not{" "}
          <b>{"\u03b5\u0302*"}</b> alone.
        </div>
      </div>
      <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div>
          <div>estimate (p_correct): {formatPct(pair.estimate)}</div>
          <div>upper bound: {formatPct(pair.upperBound)}</div>
          <div>lower bound: {formatPct(pair.lowerBound)}</div>
        </div>
        <div>
          <div>nValid: {pair.nValid}</div>
          <div>nExcluded: {pair.nExcluded}</div>
          <div>avg turns: {pair.details.averageTurnCount ?? "—"}</div>
        </div>
      </div>
      <div style={{ marginTop: 8 }}>
        <div>pSameRealB: {formatPct(pair.details.pSameRealB)}</div>
        <div>pDiffImpostor: {formatPct(pair.details.pDiffImpostor)}</div>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: "#444" }}>
        <div>Exclusions:</div>
        <pre style={{ whiteSpace: "pre-wrap" }}>
          {JSON.stringify(pair.details.exclusionCounts ?? {}, null, 2)}
        </pre>
      </div>
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
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            <th style={thStyle}>source \\ target</th>
            {models.map((t) => (
              <th key={t} style={thStyle}>
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((s) => (
            <tr key={s}>
              <th style={thStyle}>{s}</th>
              {models.map((t) => {
                const p = map.get(`${s}|||${t}`);
                const st = p ? classifyAtEpsilon(p, epsilon, minValid) : "missing";
                const bg =
                  st === "supported"
                    ? "#dff5e4"
                    : st === "tentative"
                      ? "#fff3cd"
                      : st === "unsupported"
                        ? "#fde2e2"
                        : "#f1f1f1";
                return (
                  <td
                    key={t}
                    onClick={() => p && onSelect(p)}
                    style={{
                      ...tdStyle,
                      background: bg,
                      cursor: p ? "pointer" : "not-allowed",
                      color: statusColor(st),
                    }}
                    title={p ? `${st} (nValid=${p.nValid})` : "missing"}
                  >
                    {st[0].toUpperCase()}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 6, color: "#555", fontSize: 12 }}>
        Legend: <b style={{ color: statusColor("supported") }}>S</b> supported,{" "}
        <b style={{ color: statusColor("tentative") }}>T</b> tentative,{" "}
        <b style={{ color: statusColor("unsupported") }}>U</b> unsupported,{" "}
        <b style={{ color: statusColor("missing") }}>M</b> missing
      </div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  border: "1px solid #ddd",
  padding: "4px 6px",
  position: "sticky",
  left: 0,
  background: "#fff",
  zIndex: 2,
  textAlign: "left",
  maxWidth: 220,
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  border: "1px solid #eee",
  padding: "4px 6px",
  textAlign: "center",
  minWidth: 34,
};

