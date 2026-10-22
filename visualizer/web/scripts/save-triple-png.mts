/**
 * Write triple-hierarchy PNGs to public/exports/ (not browser download).
 *   - triple-hierarchy-hd.png  — high-DPI render (default 300 DPI)
 *   - triple-hierarchy.png     — lighter preview (default 144 DPI)
 * Run: npm run export:triple-png
 * Env: CONDITION=<id> GRAPH_VIEW=raw|condensation EXPORT_DPI_HD EXPORT_DPI_STD
 */
import { mkdirSync, readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { buildCondensation } from "../src/ui/graphCompression";
import { clusterLabel, modelName } from "../src/ui/graphLabels";
import { applyImitationDifficultyTiers, layoutDag, recomputeViewBox, type DagEdge, type DagNode } from "../src/ui/layout";
import {
  actorFoolingAsSource,
  defenderImitationStrictness,
  effectiveOrderingStatus,
  stratificationLayoutScore,
  strictnessTiers,
} from "../src/ui/logic";
import type { VisualizerArtifact } from "../src/ui/types";
import { buildScoreMapsForGraphNodes, buildTripleComparisonSvgString } from "../src/ui/tripleComparisonExport";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = join(__dirname, "..");
const dataPath = join(webRoot, "public/epsilon_visualizer_data.json");
const outDir = join(webRoot, "public/exports");
const outPngHd = join(outDir, "triple-hierarchy-hd.png");
const outPngStd = join(outDir, "triple-hierarchy.png");

function parsePositiveInt(s: string | undefined, fallback: number): number {
  const n = s != null && s !== "" ? Number.parseInt(s, 10) : NaN;
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function isGptModel(id: string): boolean {
  return id.toLowerCase().includes("gpt");
}

function buildDisplayLaid(args: {
  data: VisualizerArtifact;
  condition: string;
  graphView: "raw" | "condensation";
  epsilon: number;
  minValid: number;
}): { nodes: DagNode[]; edges: DagEdge[]; graphView: "raw" | "condensation" } | null {
  const { data, condition, graphView, epsilon, minValid } = args;
  const strictPairs = data.pairs.filter((p) => p.mode === "strict");
  const baseModels = data.modelsByCondition?.[condition] ?? data.models;
  const graphModels = baseModels.filter((m) => !isGptModel(m));
  if (graphModels.length === 0) {
    return null;
  }
  const set = new Set(graphModels);
  const pairsForCondition = strictPairs.filter(
    (p) => p.condition === condition && set.has(p.source) && set.has(p.target)
  );

  const stratificationByModel = stratificationLayoutScore(strictPairs, condition, graphModels, minValid);
  const judgeScoreByModel = defenderImitationStrictness(strictPairs, condition, graphModels, minValid);
  const actorScoreByModel = actorFoolingAsSource(strictPairs, condition, graphModels, minValid);

  if (graphView === "condensation") {
    const condensation = buildCondensation({
      models: graphModels,
      pairs: strictPairs,
      condition,
      epsilon,
      minValid,
    });
    const strictByModel = stratificationByModel;
    const nodes: DagNode[] = condensation.clusters.map((c) => ({
      id: c.id,
      label: clusterLabel(c.members),
      members: c.members,
    }));
    const scoreByClusterId = new Map<string, number>();
    for (const n of nodes) {
      const s = n.members.length
        ? Math.max(...n.members.map((m) => strictByModel.get(m) ?? 0))
        : 0;
      scoreByClusterId.set(n.id, s);
    }
    const { tierById, nTiers } = strictnessTiers(scoreByClusterId);
    const edges: DagEdge[] = [
      ...condensation.supportedEdges.map(([from, to]) => ({ from, to, kind: "supported" as const })),
      ...condensation.tentativeEdges.map(([from, to]) => ({ from, to, kind: "tentative" as const })),
    ];
    const base = layoutDag(nodes, edges);
    const guideLineYs =
      nTiers > 0
        ? applyImitationDifficultyTiers(base.nodes, tierById, nTiers, {
            yTop: 96,
            yStride: 168,
            xMargin: 88,
            xGap: 40,
            minCanvasWidth: 1200,
          })
        : [];
    const laid = recomputeViewBox({ ...base, guideLineYs });
    return { nodes: laid.nodes, edges: laid.edges, graphView };
  }

  const edges: DagEdge[] = [];
  for (const p of pairsForCondition) {
    const st = effectiveOrderingStatus(p, epsilon, minValid);
    if (st === "supported") edges.push({ from: p.source, to: p.target, kind: "supported" });
    if (st === "tentative") edges.push({ from: p.source, to: p.target, kind: "tentative" });
  }
  const scoreByModelId = new Map<string, number>();
  for (const m of graphModels) {
    scoreByModelId.set(m, stratificationByModel.get(m) ?? 0);
  }
  const { tierById, nTiers } = strictnessTiers(scoreByModelId);
  const nodes: DagNode[] = graphModels.map((m) => ({
    id: m,
    label: modelName(m),
    members: [m],
  }));
  const base = layoutDag(nodes, edges);
  const guideLineYs =
    nTiers > 0
      ? applyImitationDifficultyTiers(base.nodes, tierById, nTiers, {
          yTop: 88,
          yStride: 156,
          xMargin: 88,
          xGap: 40,
          minCanvasWidth: 1200,
        })
      : [];
  const laid = recomputeViewBox({ ...base, guideLineYs });
  return { nodes: laid.nodes, edges: laid.edges, graphView };
}

async function main() {
  if (!existsSync(dataPath)) {
    throw new Error(`Data not found: ${dataPath}`);
  }
  const data = JSON.parse(readFileSync(dataPath, "utf-8")) as VisualizerArtifact;
  const condition = process.env.CONDITION || data.conditions?.[0] || "";
  const graphView = (process.env.GRAPH_VIEW as "raw" | "condensation") || "raw";
  const epsilon = data.epsilonDefault ?? 0.05;
  const minValid = data.minValidPerBranch ?? 3;
  if (!condition) {
    throw new Error("No condition in JSON and CONDITION not set");
  }

  const graphModels = (data.modelsByCondition?.[condition] ?? data.models).filter((m) => !isGptModel(m));
  const strictPairs = data.pairs.filter((p) => p.mode === "strict");
  const stratificationByModel = stratificationLayoutScore(strictPairs, condition, graphModels, minValid);
  const judgeScoreByModel = defenderImitationStrictness(strictPairs, condition, graphModels, minValid);
  const actorScoreByModel = actorFoolingAsSource(strictPairs, condition, graphModels, minValid);

  const display = buildDisplayLaid({ data, condition, graphView, epsilon, minValid });
  if (!display) {
    throw new Error("No models to export after GPT filter (check data / condition).");
  }

  const { scoreS, scoreJ, scoreA } = buildScoreMapsForGraphNodes(
    display.nodes,
    stratificationByModel,
    judgeScoreByModel,
    actorScoreByModel
  );

  const svg = buildTripleComparisonSvgString({
    nodes: display.nodes,
    edges: display.edges,
    scoreS,
    scoreJ,
    scoreA,
    graphView: display.graphView,
  });

  const dpiHd = parsePositiveInt(process.env.EXPORT_DPI_HD, 300);
  const dpiStd = parsePositiveInt(process.env.EXPORT_DPI_STD, 144);
  const buf = Buffer.from(svg, "utf-8");
  mkdirSync(outDir, { recursive: true });
  await sharp(buf, { density: dpiHd })
    .png()
    .toFile(outPngHd);
  await sharp(buf, { density: dpiStd })
    .png()
    .toFile(outPngStd);
  // eslint-disable-next-line no-console
  console.log(`Wrote ${outPngHd} (${dpiHd} DPI) and ${outPngStd} (${dpiStd} DPI)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
