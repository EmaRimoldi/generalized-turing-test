import katex from "katex";
import "katex/dist/katex.min.css";
import React, { useEffect, useMemo, useState } from "react";
import type { PairRow, PairStatus, VisualizerArtifact } from "./types";
import { clusterLabel, labelFontSize, labelLines, modelName, nodeRadius } from "./graphLabels";
import {
  actorScoreBySource,
  combinedScoreByModel,
  effectiveOrderingStatus,
  formatPct,
  judgeScoreByTarget,
} from "./logic";
import { edgeGeometry, resolveCircleCollisions } from "./graphGeometry";
import type { DagEdge, DagNode, LayoutResult } from "./layout";
import { buildSupportedAdjacency, findTransitivityViolations } from "./transitivity";
import "./App.css";

const GRAPH_SVG_STYLE = `
  .edge-supported { fill: none; stroke: #1f5c55; stroke-width: 2.5; stroke-linecap: round; opacity: 0.94; }
  .node-core { fill: #ffffff; stroke: #314d4a; stroke-width: 1.5; }
  .node-core.selected { stroke: #b13a58; stroke-width: 2.6; }
  .node-ring { fill: none; stroke: #d8e3e1; stroke-width: 1; opacity: 0.9; }
  .node-label { fill: #213937; font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-weight: 700; pointer-events: none; }
  .node-rank { fill: #7a5e21; font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-weight: 800; letter-spacing: 0.05em; pointer-events: none; }
  .node-strat { fill: #2b514c; font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-weight: 650; pointer-events: none; }
  .node-strat-sub { fill: #657a76; font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-weight: 600; pointer-events: none; }
`;

type RankingMode = "combined" | "actor" | "judge";

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

type RankedViolation = {
  triple: [string, string, string];
  defect: number | null;
};

type GraphNode = DagNode & {
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
  shortLabel: string;
  fontSize: number;
  primaryMetricValue: number;
  primaryMetricLabel: string;
  secondaryMetricText: string | null;
  rankText: string | null;
};

type GraphRender = {
  nodes: GraphNode[];
  width: number;
  height: number;
};

function conditionLabel(id: string): string {
  if (id === "no_query_no_explanation") return "GTT · no query, no explanation";
  if (id === "yes_query_no_explanation") return "GTTQ · query allowed, no explanation";
  return id;
}

function rankingModeLabel(mode: RankingMode): string {
  if (mode === "combined") return "Turing score T";
  if (mode === "actor") return "Fooling score F";
  return "Distinguishing score D";
}

function rankingModeShort(mode: RankingMode): string {
  if (mode === "combined") return "T";
  if (mode === "actor") return "F";
  return "D";
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

function statusColor(st: PairStatus): string {
  if (st === "supported") return "#28f0b8";
  if (st === "tentative") return "#f2d84a";
  if (st === "unsupported") return "#ff5a6f";
  return "#7a8883";
}

function buildCircleLayout(models: string[]): LayoutResult {
  const width = 1260;
  const height = 980;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(280, 290 + models.length * 8);
  const count = Math.max(models.length, 1);
  return {
    nodes: models.map((m, idx) => {
      const theta = -Math.PI / 2 + (2 * Math.PI * idx) / count;
      return {
        id: m,
        label: modelName(m),
        members: [m],
        x: cx + Math.cos(theta) * radius,
        y: cy + Math.sin(theta) * radius,
        w: 210,
        h: 110,
      };
    }),
    edges: [],
    width,
    height,
  };
}

function chooseScoreMap(
  mode: RankingMode,
  combined: Map<string, number>,
  actor: Map<string, number>,
  judge: Map<string, number>
): Map<string, number> {
  if (mode === "combined") return combined;
  if (mode === "actor") return actor;
  return judge;
}

function buildSecondaryMetricText(
  mode: RankingMode,
  model: string,
  combined: Map<string, number>,
  actor: Map<string, number>,
  judge: Map<string, number>,
  outDegree: Map<string, number>,
  inDegree: Map<string, number>
): string {
  const s = combined.get(model) ?? 0;
  const a = actor.get(model) ?? 0;
  const j = judge.get(model) ?? 0;
  const out = outDegree.get(model) ?? 0;
  const incoming = inDegree.get(model) ?? 0;
  if (mode === "combined") return `F=${a.toFixed(3)}  D=${j.toFixed(3)}  in=${incoming} out=${out}`;
  if (mode === "actor") return `T=${s.toFixed(3)}  D=${j.toFixed(3)}  in=${incoming} out=${out}`;
  return `T=${s.toFixed(3)}  F=${a.toFixed(3)}  in=${incoming} out=${out}`;
}

function decorateGraphRender(
  laid: LayoutResult | null,
  primaryScoreByModel: Map<string, number>,
  combinedScore: Map<string, number>,
  actorScore: Map<string, number>,
  judgeScore: Map<string, number>,
  outDegreeByModel: Map<string, number>,
  inDegreeByModel: Map<string, number>,
  rankIndex: Map<string, number>,
  rankingMode: RankingMode
): GraphRender {
  if (!laid) return { nodes: [], width: 1180, height: 660 };

  const nodes = laid.nodes.map((n) => {
    const shortLabel = clusterLabel(n.members);
    const id = n.members[0]!;
    const primaryMetricValue = n.members.length === 1 ? primaryScoreByModel.get(id) ?? 0 : 0;
    const secondaryMetricText =
      n.members.length === 1
        ? buildSecondaryMetricText(
            rankingMode,
            id,
            combinedScore,
            actorScore,
            judgeScore,
            outDegreeByModel,
            inDegreeByModel
          )
        : null;
    const rankText = n.members.length === 1 ? `#${rankIndex.get(id) ?? "?"}` : null;

    return {
      ...n,
      shortLabel,
      r: nodeRadius(shortLabel, n.members.length),
      fontSize: labelFontSize(shortLabel, n.members.length),
      primaryMetricValue,
      primaryMetricLabel: rankingModeShort(rankingMode),
      secondaryMetricText,
      rankText,
    };
  });

  return resolveCircleCollisions(nodes);
}

function pairJudgeAdvantage(row: PairRow, minValid: number): number | null {
  const im = row.details?.imitationDataProvenance as string | undefined;
  const ge = row.details?.genuineDataProvenance as string | undefined;
  if (row.source === row.target || row.nValid < minValid || im === "imputed_uniform" || ge === "imputed_uniform") {
    return null;
  }
  if (row.dHat != null && !Number.isNaN(row.dHat)) return row.dHat;
  if (row.estimate == null || Number.isNaN(row.estimate)) return null;
  return row.estimate - 0.5;
}

function pairCriticalEpsilon(row: PairRow, minValid: number): number | null {
  const d = pairJudgeAdvantage(row, minValid);
  if (d == null) return null;
  return Math.max(0, d);
}

export function App() {
  const { loading, error, data } = useArtifact();
  const [epsilon, setEpsilon] = useState(0.05);
  const [condition, setCondition] = useState<string>("");
  const [rankingMode, setRankingMode] = useState<RankingMode>("combined");
  const [selected, setSelected] = useState<Selected>({ kind: "none" });

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

  const graphModels = useMemo(() => {
    if (!data) return [];
    return data.modelsByCondition?.[condition] ?? data.models;
  }, [data, condition]);

  const graphModelsSet = useMemo(() => new Set(graphModels), [graphModels]);

  const pairsForCondition = useMemo(
    () =>
      strictPairs.filter(
        (p) =>
          p.condition === condition && graphModelsSet.has(p.source) && graphModelsSet.has(p.target)
      ),
    [strictPairs, condition, graphModelsSet]
  );

  const combinedScore = useMemo(
    () => combinedScoreByModel(strictPairs, condition, graphModels, minValid),
    [strictPairs, condition, graphModels, minValid]
  );
  const actorScore = useMemo(
    () => actorScoreBySource(strictPairs, condition, graphModels, minValid),
    [strictPairs, condition, graphModels, minValid]
  );
  const judgeScore = useMemo(
    () => judgeScoreByTarget(strictPairs, condition, graphModels, minValid),
    [strictPairs, condition, graphModels, minValid]
  );
  const selectedScore = useMemo(
    () => chooseScoreMap(rankingMode, combinedScore, actorScore, judgeScore),
    [rankingMode, combinedScore, actorScore, judgeScore]
  );

  const epsilonStarByPair = useMemo(() => {
    const out = new Map<string, number>();
    for (const p of pairsForCondition) {
      const eStar = pairCriticalEpsilon(p, minValid);
      if (eStar == null) continue;
      out.set(`${p.source}:::${p.target}`, eStar);
    }
    return out;
  }, [pairsForCondition, minValid]);

  const currentEdges = useMemo(() => {
    const edges: DagEdge[] = [];
    for (const p of pairsForCondition) {
      if (p.source === p.target) continue;
      if (effectiveOrderingStatus(p, epsilon, minValid) === "supported") {
        edges.push({ from: p.source, to: p.target, kind: "supported" });
      }
    }
    return edges;
  }, [pairsForCondition, epsilon, minValid]);

  const outDegreeByModel = useMemo(() => {
    const out = new Map<string, number>();
    for (const model of graphModels) out.set(model, 0);
    for (const edge of currentEdges) {
      out.set(edge.from, (out.get(edge.from) ?? 0) + 1);
    }
    return out;
  }, [graphModels, currentEdges]);

  const inDegreeByModel = useMemo(() => {
    const out = new Map<string, number>();
    for (const model of graphModels) out.set(model, 0);
    for (const edge of currentEdges) {
      out.set(edge.to, (out.get(edge.to) ?? 0) + 1);
    }
    return out;
  }, [graphModels, currentEdges]);

  const transitivity = useMemo(() => {
    const tuples = currentEdges.map((e) => [e.from, e.to] as [string, string]);
    const adj = buildSupportedAdjacency(graphModels, tuples);
    const raw = findTransitivityViolations(graphModels, adj);
    const triples: RankedViolation[] = raw.triples
      .map((triple) => {
        const [a, b, c] = triple;
        const ab = epsilonStarByPair.get(`${a}:::${b}`);
        const bc = epsilonStarByPair.get(`${b}:::${c}`);
        const ac = epsilonStarByPair.get(`${a}:::${c}`);
        const defect = ab != null && bc != null && ac != null ? ac - Math.max(ab, bc) : null;
        return { triple, defect };
      })
      .sort(
        (lhs, rhs) =>
          (rhs.defect ?? Number.NEGATIVE_INFINITY) - (lhs.defect ?? Number.NEGATIVE_INFINITY) ||
          lhs.triple.join("\0").localeCompare(rhs.triple.join("\0"))
      );
    return { count: raw.count, triples };
  }, [currentEdges, graphModels, epsilonStarByPair]);

  const summaryCounts = useMemo(() => {
    const c = { supported: 0, tentative: 0, unsupported: 0, missing: 0 };
    for (const p of pairsForCondition) {
      if (p.source === p.target) continue;
      c[effectiveOrderingStatus(p, epsilon, minValid)] += 1;
    }
    return c;
  }, [pairsForCondition, epsilon, minValid]);

  const graphOrder = useMemo(
    () =>
      [...graphModels].sort(
        (a, b) =>
          (inDegreeByModel.get(b) ?? 0) - (inDegreeByModel.get(a) ?? 0) ||
          (combinedScore.get(b) ?? 0) - (combinedScore.get(a) ?? 0) ||
          (actorScore.get(b) ?? 0) - (actorScore.get(a) ?? 0) ||
          (judgeScore.get(b) ?? 0) - (judgeScore.get(a) ?? 0) ||
          modelName(a).localeCompare(modelName(b))
      ),
    [graphModels, inDegreeByModel, combinedScore, actorScore, judgeScore]
  );
  const graphRankIndex = useMemo(
    () => new Map(graphOrder.map((m, idx) => [m, idx + 1])),
    [graphOrder]
  );

  const laidGraph = useMemo(() => {
    if (!graphModels.length) return null;
    const base = buildCircleLayout(graphOrder);
    return { ...base, edges: currentEdges };
  }, [graphModels.length, graphOrder, currentEdges]);

  const graphRender = useMemo(
    () =>
      decorateGraphRender(
        laidGraph,
        selectedScore,
        combinedScore,
        actorScore,
        judgeScore,
        outDegreeByModel,
        inDegreeByModel,
        graphRankIndex,
        rankingMode
      ),
    [
      laidGraph,
      selectedScore,
      combinedScore,
      actorScore,
      judgeScore,
      outDegreeByModel,
      inDegreeByModel,
      graphRankIndex,
      rankingMode,
    ]
  );

  const rankingCards = useMemo(() => {
    if (!data) return [];
    return data.conditions.map((id) => {
      const models = data.modelsByCondition?.[id] ?? data.models;
      const s = combinedScoreByModel(strictPairs, id, models, minValid);
      const a = actorScoreBySource(strictPairs, id, models, minValid);
      const j = judgeScoreByTarget(strictPairs, id, models, minValid);
      const condPairs = strictPairs.filter(
        (p) => p.condition === id && models.includes(p.source) && models.includes(p.target)
      );
      const out = new Map<string, number>();
      for (const model of models) out.set(model, 0);
      for (const p of condPairs) {
        if (p.source === p.target) continue;
        if (effectiveOrderingStatus(p, epsilon, minValid) === "supported") {
          out.set(p.source, (out.get(p.source) ?? 0) + 1);
        }
      }
      const primary = chooseScoreMap(rankingMode, s, a, j);
      const rows = [...models]
        .sort(
          (lhs, rhs) =>
            (primary.get(rhs) ?? 0) - (primary.get(lhs) ?? 0) ||
            (s.get(rhs) ?? 0) - (s.get(lhs) ?? 0) ||
            (a.get(rhs) ?? 0) - (a.get(lhs) ?? 0) ||
            (j.get(rhs) ?? 0) - (j.get(lhs) ?? 0) ||
            modelName(lhs).localeCompare(modelName(rhs))
        )
        .map((model, idx) => ({
          rank: idx + 1,
          model,
          combined: s.get(model) ?? 0,
          actor: a.get(model) ?? 0,
          judge: j.get(model) ?? 0,
          outDegree: out.get(model) ?? 0,
        }));
      return { id, rows };
    });
  }, [data, strictPairs, rankingMode, epsilon, minValid]);

  if (loading) {
    return <div className="load-state">Loading strict epsilon evidence...</div>;
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
          <p className="eyebrow">strict epsilon graph and comparative rankings</p>
          <h1>Turing Ordering Visualizer</h1>
        </div>
        <div className="stat-strip" aria-label="pair status counts">
          <StatusStat label="S" value={summaryCounts.supported} status="supported" />
          <StatusStat label="U" value={summaryCounts.unsupported} status="unsupported" />
          <StatusStat label="M" value={summaryCounts.missing} status="missing" />
          <div className="tv-pill">
            <span>TV_ε</span>
            <strong>{transitivity.count}</strong>
          </div>
        </div>
      </header>

      <section className="control-strip" aria-label="graph controls">
        <label>
          <span>Experiment</span>
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
            onInput={(e) => setEpsilon(Number((e.target as HTMLInputElement).value))}
          />
          <small className="control-help">
            Higher epsilon is more permissive: an edge survives whenever the judge advantage d(A,B) stays below the threshold.
          </small>
        </label>

        <label>
          <span>Ranking metric</span>
          <select value={rankingMode} onChange={(e) => setRankingMode(e.target.value as RankingMode)}>
            <option value="combined">Turing score T</option>
            <option value="actor">Fooling score F</option>
            <option value="judge">Distinguishing score D</option>
          </select>
          <small className="control-help">
            The metric changes node annotations and table order; the graph keeps the strict epsilon relation.
          </small>
        </label>
      </section>

      <section className="method-panel" aria-label="strict epsilon computation">
        <div>
          <p className="panel-kicker">Strict graph</p>
          <p className="en-summary" lang="en">
            For each ordered pair <b>(A,B)</b>, the judge-side correctness combines the self-pair branch and the imitation branch.
          </p>
          <Formula
            display
            tex={"\\widehat P(\\mathrm{correct}) = \\frac12\\widehat p(y=1\\mid B,B) + \\frac12\\widehat p(y=0\\mid A,B)"}
          />
          <Formula display tex={"\\widehat d(A,B)=\\widehat P(\\mathrm{correct})-\\frac12"} />
          <p className="en-summary" lang="en">
            The visualizer draws the directed edge <b>A → B</b> exactly when <b>d(A,B) ≤ ε</b>.
          </p>
          <Formula display tex={"A \\to B \\iff \\widehat d(A,B) \\le \\varepsilon"} />
          <p className="panel-kicker">Rankings</p>
          <p className="en-summary" lang="en">
            Scores aggregate the same pairwise outcomes over actor and distinguisher roles.
          </p>
          <Formula display tex={"D(B)=\\frac12\\widehat p(y=1\\mid B,B)+\\frac12\\operatorname*{mean}_{A\\ne B}\\widehat p(y=0\\mid A,B)"} />
          <Formula display tex={"F(M)=\\operatorname*{mean}_{B\\ne M}\\bigl(1-\\widehat p(y=0\\mid M,B)\\bigr)"} />
          <Formula display tex={"T(M)=\\frac12F(M)+\\frac12D(M)"} />
          <p>
            Tables can be sorted by <b>T</b>, <b>F</b>, or <b>D</b>. Circular node order follows current in-degree; edges follow the strict epsilon rule.
          </p>
        </div>
        <details className="legend-details">
          <summary>Legend</summary>
          <dl>
            <div>
              <dt>S</dt>
              <dd>supported: d(A,B) is at most epsilon.</dd>
            </div>
            <div>
              <dt>U</dt>
              <dd>unsupported: the pair exists, but the judge advantage stays above epsilon.</dd>
            </div>
            <div>
              <dt>M</dt>
              <dd>missing: no usable strict pair because a branch is absent, imputed, or under the validity threshold.</dd>
            </div>
          </dl>
        </details>
      </section>

      <main className="workspace">
        <section className="graph-stage" aria-label="graph">
          <div className="graph-stage-header">
            <div>
              <p className="panel-kicker">Strict graph</p>
              <h2>Directed epsilon relation</h2>
            </div>
            <p className="graph-stage-note">
              Visible edges: <b>{currentEdges.length}</b>. Nodes are ordered clockwise by current in-degree, i.e. how many models can fool that target at this epsilon.
            </p>
          </div>
          <GraphFigure
            title="Circular strict epsilon graph"
            subtitle="pairwise judge advantage"
            note="An edge is drawn iff d(A,B) = P(correct) - 1/2 is at most epsilon."
            formula="A → B iff d(A,B) ≤ ε"
            render={graphRender}
            edges={currentEdges}
            selected={selected}
            onSelect={setSelected}
          />
          <DegreeBarChart
            models={graphOrder}
            inDegree={inDegreeByModel}
            outDegree={outDegreeByModel}
          />
          <EpsilonCurve
            pairs={pairsForCondition}
            epsilon={epsilon}
            minValid={minValid}
          />
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
            <p className="muted">
              This count is computed on the current strict graph. For each violating chain A → B → C without A → C,
              we also show the defect <Formula tex={"\\Delta = \\varepsilon^*(A,C)-\\max\\{\\varepsilon^*(A,B),\\varepsilon^*(B,C)\\}"} /> when available.
              The list shows the 12 largest defects.
            </p>
            {transitivity.count === 0 ? (
              <p className="muted">No supported chain violates closure at this epsilon.</p>
            ) : (
              <ol className="triple-list">
                {transitivity.triples.slice(0, 12).map(({ triple, defect }) => {
                  const [a, b, c] = triple;
                  return (
                    <li key={`${a}|${b}|${c}`}>
                      {modelName(a)}
                      {" -> "}
                      {modelName(b)}
                      {" -> "}
                      {modelName(c)}
                      {defect != null ? `  (Δ=${defect.toFixed(3)})` : ""}
                    </li>
                  );
                })}
              </ol>
            )}
          </section>
        </aside>
      </main>

      <section className="matrix-section ranking-section">
        <div className="matrix-heading">
          <h2>Rankings by Experiment</h2>
          <p>
            Rows are sorted by <b>{rankingModeLabel(rankingMode)}</b>. Columns report:
            Turing score <b>T</b>, fooling score <b>F</b>, distinguishing score <b>D</b>, and current strict out-degree.
          </p>
        </div>
        <div className="ranking-grid">
          {rankingCards.map((card) => (
            <section key={card.id} className="ranking-card">
              <h2>{conditionLabel(card.id)}</h2>
              <p>
                Sort key: <b>{rankingModeLabel(rankingMode)}</b>. The graph itself still uses only the strict epsilon rule.
              </p>
              <div className="ranking-table-scroll">
                <table className="ranking-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>model</th>
                      <th>T</th>
                      <th>F</th>
                      <th>D</th>
                      <th>out_ε</th>
                    </tr>
                  </thead>
                  <tbody>
                    {card.rows.map((row) => (
                      <tr key={`${card.id}-${row.model}`}>
                        <td>{row.rank}</td>
                        <td>{modelName(row.model)}</td>
                        <td>{row.combined.toFixed(3)}</td>
                        <td>{row.actor.toFixed(3)}</td>
                        <td>{row.judge.toFixed(3)}</td>
                        <td>{row.outDegree}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ))}
        </div>
      </section>

      <section className="matrix-section">
        <div className="matrix-heading">
          <h2>Actor / Distinguisher Matrix</h2>
          <p>
            S supported, U unsupported, M missing. Diagonal “—” is the self-pair used only for calibration in{" "}
            <Formula tex={"\\widehat p(y=1\\mid B,B)"} /> and is not an ordering edge.
          </p>
        </div>
        <PairMatrix
          models={graphModels}
          pairs={pairsForCondition}
          epsilon={epsilon}
          minValid={minValid}
          onSelect={(pair) => setSelected({ kind: "pair", pair })}
        />
      </section>
    </div>
  );
}

function GraphFigure({
  title,
  subtitle,
  note,
  formula,
  render,
  edges,
  selected,
  onSelect,
}: {
  title: string;
  subtitle: string;
  note: string;
  formula: string;
  render: GraphRender;
  edges: DagEdge[];
  selected: Selected;
  onSelect: (selection: Selected) => void;
}) {
  const viewBox = `0 0 ${render.width} ${render.height}`;

  return (
    <section className="graph-figure">
      <header className="graph-figure-head">
        <div>
          <p className="panel-kicker">{subtitle}</p>
          <h2>{title}</h2>
        </div>
        <p className="graph-formula">{formula}</p>
      </header>
      <p className="graph-interaction-hint" role="note">
        {note}
      </p>
      <svg className="graph-canvas" viewBox={viewBox} role="img">
        <style>{GRAPH_SVG_STYLE}</style>
        <defs>
          <marker id="arrow-supported" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#136f63" />
          </marker>
        </defs>
        <rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />
        {edges.map((edge, idx) => {
          const geom = edgeGeometry(edge, idx, render.nodes, edges);
          return (
            <g key={`${edge.from}-${edge.to}-${idx}`}>
              <path className="edge-supported" d={geom.path} markerEnd="url(#arrow-supported)" />
            </g>
          );
        })}
        {render.nodes.map((node) => {
          const isSelected = selected.kind === "cluster" && selected.clusterId === node.id;
          const lines = labelLines(node.shortLabel);
          return (
            <g
              key={node.id}
              className="node-hit"
              onClick={() => onSelect({ kind: "cluster", clusterId: node.id, members: node.members })}
            >
              <circle className="node-ring" cx={node.x} cy={node.y} r={node.r + 10} />
              <circle className={`node-core${isSelected ? " selected" : ""}`} cx={node.x} cy={node.y} r={node.r} />
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
              {node.rankText && (
                <text className="node-rank" x={node.x} y={node.y - node.r - 18} textAnchor="middle" fontSize={12}>
                  {node.rankText}
                </text>
              )}
              <text className="node-strat" x={node.x} y={node.y + node.r + 18} textAnchor="middle" fontSize={11}>
                {`${node.primaryMetricLabel}=${node.primaryMetricValue.toFixed(3)}`}
              </text>
              {node.secondaryMetricText && (
                <text className="node-strat-sub" x={node.x} y={node.y + node.r + 34} textAnchor="middle" fontSize={9}>
                  {node.secondaryMetricText}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </section>
  );
}

function DegreeBarChart({
  models,
  inDegree,
  outDegree,
}: {
  models: string[];
  inDegree: Map<string, number>;
  outDegree: Map<string, number>;
}) {
  const maxDegree = Math.max(1, ...models.flatMap((model) => [inDegree.get(model) ?? 0, outDegree.get(model) ?? 0]));

  return (
    <section className="degree-panel" aria-label="in out degree chart">
      <div className="degree-panel-head">
        <div>
          <p className="panel-kicker">Degree profile</p>
          <h2>Fooling exposure at current epsilon</h2>
        </div>
        <p>in_ε counts how many actors fool the model; out_ε counts how many targets the model fools.</p>
      </div>
      <div className="degree-bars">
        <div className="degree-row degree-row-head" aria-hidden="true">
          <div />
          <div>in_ε</div>
          <div />
          <div>out_ε</div>
          <div />
        </div>
        {models.map((model) => {
          const incoming = inDegree.get(model) ?? 0;
          const outgoing = outDegree.get(model) ?? 0;
          return (
            <div className="degree-row" key={model}>
              <div className="degree-label">{modelName(model)}</div>
              <div className="degree-track" title={`in_ε=${incoming}`}>
                <span className="degree-fill degree-fill-in" style={{ width: `${(100 * incoming) / maxDegree}%` }} />
              </div>
              <div className="degree-value">{incoming}</div>
              <div className="degree-track" title={`out_ε=${outgoing}`}>
                <span className="degree-fill degree-fill-out" style={{ width: `${(100 * outgoing) / maxDegree}%` }} />
              </div>
              <div className="degree-value">{outgoing}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function EpsilonCurve({
  pairs,
  epsilon,
  minValid,
}: {
  pairs: PairRow[];
  epsilon: number;
  minValid: number;
}) {
  const width = 720;
  const height = 250;
  const pad = { top: 18, right: 20, bottom: 34, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const epsilonStars = useMemo(
    () =>
      pairs
        .filter((pair) => pair.source !== pair.target)
        .map((pair) => pairCriticalEpsilon(pair, minValid))
        .filter((value): value is number => value != null)
        .sort((a, b) => a - b),
    [pairs, minValid]
  );

  const maxEdges = Math.max(1, epsilonStars.length);
  const thresholds = useMemo(
    () => Array.from({ length: 101 }, (_, idx) => idx / 200),
    []
  );
  const points = thresholds.map((threshold) => {
    const count = epsilonStars.filter((value) => value <= threshold + 1e-12).length;
    return {
      threshold,
      count,
      x: pad.left + (threshold / 0.5) * plotW,
      y: pad.top + plotH - (count / maxEdges) * plotH,
    };
  });
  const path = points.map((point, idx) => `${idx === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
  const activeCount = epsilonStars.filter((value) => value <= epsilon + 1e-12).length;
  const activeX = pad.left + (Math.min(0.5, Math.max(0, epsilon)) / 0.5) * plotW;
  const activeY = pad.top + plotH - (activeCount / maxEdges) * plotH;

  return (
    <section className="epsilon-curve-panel" aria-label="epsilon edge curve">
      <div className="degree-panel-head">
        <div>
          <p className="panel-kicker">Epsilon curve</p>
          <h2>Edge count as epsilon varies</h2>
        </div>
        <p>
          The curve counts how many ordered pairs satisfy ε*(A,B) ≤ ε. The marker is the current slider value.
        </p>
      </div>
      <div className="epsilon-curve-wrap">
        <svg className="epsilon-curve" viewBox={`0 0 ${width} ${height}`} role="img">
          <line className="curve-axis" x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} />
          <line className="curve-axis" x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} />
          {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((tick) => {
            const x = pad.left + (tick / 0.5) * plotW;
            return (
              <g key={tick}>
                <line className="curve-grid" x1={x} y1={pad.top} x2={x} y2={pad.top + plotH} />
                <text className="curve-tick" x={x} y={height - 12} textAnchor="middle">
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}
          {[0, Math.round(maxEdges / 2), maxEdges].map((tick) => {
            const y = pad.top + plotH - (tick / maxEdges) * plotH;
            return (
              <g key={tick}>
                <line className="curve-grid" x1={pad.left} y1={y} x2={pad.left + plotW} y2={y} />
                <text className="curve-tick" x={pad.left - 10} y={y + 4} textAnchor="end">
                  {tick}
                </text>
              </g>
            );
          })}
          <path className="curve-line" d={path} />
          <line className="curve-marker-line" x1={activeX} y1={pad.top} x2={activeX} y2={pad.top + plotH} />
          <circle className="curve-marker" cx={activeX} cy={activeY} r={5} />
          <text className="curve-caption" x={pad.left + plotW} y={pad.top + 14} textAnchor="end">
            {`${activeCount}/${maxEdges} edges at ε=${epsilon.toFixed(3)}`}
          </text>
          <text className="curve-axis-label" x={pad.left + plotW / 2} y={height - 1} textAnchor="middle">
            epsilon
          </text>
          <text className="curve-axis-label" transform={`translate(12 ${pad.top + plotH / 2}) rotate(-90)`} textAnchor="middle">
            supported edges
          </text>
        </svg>
      </div>
    </section>
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
  const isSelf = pair.source === pair.target;
  const status = effectiveOrderingStatus(pair, epsilon, minValid);
  const pCorrect = pair.details.pCorrect ?? pair.estimate;
  const d = pairJudgeAdvantage(pair, minValid);
  const epsilonStar = pairCriticalEpsilon(pair, minValid);

  return (
    <div className="pair-details">
      <h2>
        {modelName(pair.source)}
        {" -> "}
        {modelName(pair.target)}
      </h2>
      <div className="status-line">
        <span style={{ backgroundColor: isSelf ? "#8ea4bf" : statusColor(status) }} />
        {isSelf ? "self-pair calibration" : status}
      </div>
      <dl>
        <div>
          <dt>P(correct)</dt>
          <dd>{formatPct(pCorrect)}</dd>
        </div>
        <div>
          <dt>d = P(correct) - 1/2</dt>
          <dd>{d != null ? d.toFixed(4) : "--"}</dd>
        </div>
        <div>
          <dt>epsilon* = max(0, d)</dt>
          <dd>{epsilonStar != null ? epsilonStar.toFixed(4) : "--"}</dd>
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
                            ? `${modelName(source)} self-pair: calibration only`
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
