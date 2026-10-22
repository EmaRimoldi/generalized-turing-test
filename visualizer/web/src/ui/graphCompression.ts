import type { PairRow, PairStatus } from "./types";
import { effectiveOrderingStatus } from "./logic";

export interface Cluster {
  id: string;
  members: string[];
}

export interface Condensation {
  clusters: Cluster[];
  memberToCluster: Record<string, string>;
  supportedEdges: Array<[string, string]>; // cluster -> cluster
  tentativeEdges: Array<[string, string]>; // cluster -> cluster
}

function tarjanScc(
  nodes: string[],
  edges: Record<string, Set<string>>
): string[][] {
  let index = 0;
  const stack: string[] = [];
  const onStack = new Set<string>();
  const indices = new Map<string, number>();
  const low = new Map<string, number>();
  const sccs: string[][] = [];

  function strongconnect(v: string) {
    indices.set(v, index);
    low.set(v, index);
    index += 1;
    stack.push(v);
    onStack.add(v);

    for (const w of edges[v] ?? []) {
      if (!indices.has(w)) {
        strongconnect(w);
        low.set(v, Math.min(low.get(v)!, low.get(w)!));
      } else if (onStack.has(w)) {
        low.set(v, Math.min(low.get(v)!, indices.get(w)!));
      }
    }

    if (low.get(v) === indices.get(v)) {
      const comp: string[] = [];
      while (true) {
        const w = stack.pop()!;
        onStack.delete(w);
        comp.push(w);
        if (w === v) break;
      }
      sccs.push(comp);
    }
  }

  for (const v of nodes) {
    if (!indices.has(v)) strongconnect(v);
  }
  return sccs;
}

export function buildCondensation(args: {
  models: string[];
  pairs: PairRow[];
  condition: string;
  epsilon: number;
  minValid: number;
}): Condensation {
  const { models, pairs, condition, epsilon, minValid } = args;
  const selected = new Set(models);
  const relevant = pairs.filter((p) => p.condition === condition && p.source !== p.target && selected.has(p.source) && selected.has(p.target));
  const status = (p: PairRow): PairStatus =>
    effectiveOrderingStatus(p, epsilon, minValid);

  // supported adjacency on models
  const supportedAdj: Record<string, Set<string>> = {};
  for (const m of models) supportedAdj[m] = new Set();
  for (const p of relevant) {
    if (status(p) === "supported") supportedAdj[p.source]?.add(p.target);
  }

  const sccs = tarjanScc(models, supportedAdj).map((c) => [...c].sort());
  sccs.sort((a, b) => (a.length - b.length) || a.join("|").localeCompare(b.join("|")));

  const clusters: Cluster[] = [];
  const memberToCluster: Record<string, string> = {};
  sccs.forEach((members, i) => {
    const id = `C${String(i).padStart(3, "0")}`;
    clusters.push({ id, members });
    for (const m of members) memberToCluster[m] = id;
  });

  const supportedEdgesSet = new Set<string>();
  const tentativeEdgesSet = new Set<string>();

  for (const p of relevant) {
    const st = status(p);
    const a = memberToCluster[p.source];
    const b = memberToCluster[p.target];
    if (!a || !b || a === b) continue;
    if (st === "supported") supportedEdgesSet.add(`${a}->${b}`);
    if (st === "tentative") tentativeEdgesSet.add(`${a}->${b}`);
  }

  const toPairs = (s: Set<string>): Array<[string, string]> =>
    [...s].sort().map((x) => {
      const [a, b] = x.split("->");
      return [a, b];
    });

  return {
    clusters,
    memberToCluster,
    supportedEdges: toPairs(supportedEdgesSet),
    tentativeEdges: toPairs(tentativeEdgesSet),
  };
}

