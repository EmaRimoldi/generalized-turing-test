import type { PairRow, PairStatus } from "./types";

export function classifyAtEpsilon(
  row: PairRow,
  epsilon: number,
  minValid: number
): PairStatus {
  if (
    row.estimate == null ||
    row.nValid < minValid
  ) {
    return "missing";
  }
  const dHat = row.estimate - 0.5;
  return dHat <= epsilon + 1e-12 ? "supported" : "unsupported";
}

/** Strict ordering status for graph/matrix: same as ``classifyAtEpsilon``, but missing if imitation/genuine uses uniform imputation. */
export function effectiveOrderingStatus(
  row: PairRow,
  epsilon: number,
  minValid: number
): PairStatus {
  const im = row.details?.imitationDataProvenance as string | undefined;
  const ge = row.details?.genuineDataProvenance as string | undefined;
  if (im === "imputed_uniform" || ge === "imputed_uniform") {
    return "missing";
  }
  return classifyAtEpsilon(row, epsilon, minValid);
}

/** Pure fooling relation: A -> B iff B is no better than chance+epsilon at catching A imitating B. */
export function foolingOrderingStatus(
  row: PairRow,
  epsilon: number,
  minValid: number
): PairStatus {
  if (row.source === row.target) return "missing";
  const im = row.details?.imitationDataProvenance as string | undefined;
  if (im === "imputed_uniform") {
    return "missing";
  }
  const pDiff = row.details?.pDiffImpostor;
  if (pDiff == null || Number.isNaN(pDiff) || row.nValid < minValid) {
    return "missing";
  }
  const dHat = pDiff - 0.5;
  return dHat <= epsilon + 1e-12 ? "supported" : "unsupported";
}

export function formatPct(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(100 * x).toFixed(1)}%`;
}

function hasUsablePairBranches(row: PairRow, minValid: number): boolean {
  const im = row.details?.imitationDataProvenance as string | undefined;
  const ge = row.details?.genuineDataProvenance as string | undefined;
  return !(
    row.source === row.target ||
    row.nValid < minValid ||
    im === "imputed_uniform" ||
    ge === "imputed_uniform"
  );
}

/** s_B = P(B says "same" | genuine B), the self-recognition rate of the target/judge B. */
export function pairSelfRecognition(row: PairRow, minValid: number): number | null {
  const ge = row.details?.genuineDataProvenance as string | undefined;
  if (row.nValid < minValid || ge === "imputed_uniform") return null;
  const v = row.details?.pSameRealB;
  return v == null || Number.isNaN(v) ? null : v;
}

/** r_{A→B} = P(B says "same" | A imitates B), the fooling rate of actor A against judge B. */
export function pairFoolingRate(row: PairRow, minValid: number): number | null {
  if (!hasUsablePairBranches(row, minValid)) return null;
  const pDiff = row.details?.pDiffImpostor;
  if (pDiff == null || Number.isNaN(pDiff)) return null;
  return 1 - pDiff;
}

/**
 * Actor-centric pairwise margin:
 *
 *   m(A,B) = 1/2 * ( r_{A→B} - s_B )
 *
 * Positive means A fools B better than B recognizes a genuine B.
 */
export function actorPairMargin(row: PairRow, minValid: number): number | null {
  const s = pairSelfRecognition(row, minValid);
  const r = pairFoolingRate(row, minValid);
  if (s == null || r == null) return null;
  return 0.5 * (r - s);
}

/** Minimal epsilon for the original blog relation A >=_epsilon B, rewritten from the actor side. */
export function actorPairEpsilonStar(row: PairRow, minValid: number): number | null {
  const m = actorPairMargin(row, minValid);
  if (m == null) return null;
  return Math.max(0, -m);
}

/** Empirical fooling graph: A ->_tau B iff m(A,B) >= tau. */
export function actorPairStatusAtThreshold(
  row: PairRow,
  tau: number,
  minValid: number
): PairStatus {
  const m = actorPairMargin(row, minValid);
  if (m == null) return "missing";
  return m + 1e-12 >= tau ? "supported" : "unsupported";
}

function marginsBySource(
  pairs: PairRow[],
  condition: string,
  models: string[],
  minValid: number
): Map<string, number[]> {
  const out = new Map<string, number[]>();
  for (const m of models) out.set(m, []);
  for (const p of pairs) {
    if (p.condition !== condition || p.source === p.target) continue;
    const m = actorPairMargin(p, minValid);
    if (m == null) continue;
    out.get(p.source)?.push(m);
  }
  return out;
}

/** Unified actor ranking: mean positive part of m(A,B), equal to the area under the threshold curve up to scaling. */
export function actorPositiveScoreBySource(
  pairs: PairRow[],
  condition: string,
  models: string[],
  minValid: number
): Map<string, number> {
  const grouped = marginsBySource(pairs, condition, models, minValid);
  const out = new Map<string, number>();
  for (const m of models) {
    const xs = grouped.get(m) ?? [];
    out.set(m, xs.length ? xs.reduce((acc, v) => acc + Math.max(0, v), 0) / xs.length : 0);
  }
  return out;
}

/** Mean signed actor margin across targets. */
export function actorMeanMarginBySource(
  pairs: PairRow[],
  condition: string,
  models: string[],
  minValid: number
): Map<string, number> {
  const grouped = marginsBySource(pairs, condition, models, minValid);
  const out = new Map<string, number>();
  for (const m of models) {
    const xs = grouped.get(m) ?? [];
    out.set(m, xs.length ? xs.reduce((acc, v) => acc + v, 0) / xs.length : 0);
  }
  return out;
}

/** Out-degree of the empirical fooling graph at threshold tau. */
export function actorOutDegreeBySource(
  pairs: PairRow[],
  condition: string,
  models: string[],
  tau: number,
  minValid: number
): Map<string, number> {
  const out = new Map<string, number>();
  for (const m of models) out.set(m, 0);
  for (const p of pairs) {
    if (p.condition !== condition || p.source === p.target) continue;
    if (actorPairStatusAtThreshold(p, tau, minValid) !== "supported") continue;
    out.set(p.source, (out.get(p.source) ?? 0) + 1);
  }
  return out;
}

/**
 * Stratification score for a distinguisher **B** (the `target` in pair rows), combining
 * the same two branches that feed strict `pCorrect` per pair:
 *
 * - **Self (B, B):** `pSameRealB` = P(answer = 1 | talking to a genuine B) — “is B right about being the real B?”
 * - **Off-diagonal:** mean of `pDiffImpostor` = P(answer = 0 | A → B) over all A ≠ B — “how often B flags imitators.”
 *
 * If both are present: **(pSame + meanPdiff) / 2**, mirroring the 50/50 weight in `pCorrect = ½·pSame + ½·pDiff` per (A, B).
 * If only one is present, that part alone; if neither, 0.
 */
export function defenderImitationStrictness(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  const pSameByD = new Map<string, number>();
  const pDiffsByD = new Map<string, number[]>();
  for (const m of models) {
    pDiffsByD.set(m, []);
  }
  for (const p of pairs) {
    if (p.condition !== condition) continue;
    const pSame = p.details?.pSameRealB;
    // Self-pairs are rendered as calibration-only rows with null estimates, so the
    // genuine-branch rate must be recovered from any pair row targeting B.
    if (pSame != null && !Number.isNaN(pSame) && !pSameByD.has(p.target)) {
      pSameByD.set(p.target, pSame);
    }
    if (p.source === p.target) continue;
    const v = p.details?.pDiffImpostor;
    if (v == null || Number.isNaN(v)) continue;
    pDiffsByD.get(p.target)?.push(v);
  }
  const out = new Map<string, number>();
  for (const m of models) {
    const pSame = pSameByD.get(m);
    const xs = pDiffsByD.get(m) ?? [];
    const meanPdiff = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
    if (pSame != null && meanPdiff != null) {
      out.set(m, 0.5 * pSame + 0.5 * meanPdiff);
    } else if (meanPdiff != null) {
      out.set(m, meanPdiff);
    } else if (pSame != null) {
      out.set(m, pSame);
    } else {
      out.set(m, 0);
    }
  }
  return out;
}

/**
 * For each model as **source** (actor) A in off-diagonals (A, T), A ≠ T: mean of
 * **(1 − pDiffImpostor)**, i.e. how often other distinguishers are **not** catching A as
 * an imitator — a simple “fooling / imitation” read across all targets.
 */
export function actorFoolingAsSource(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  const pDiffsByS = new Map<string, number[]>();
  for (const m of models) {
    pDiffsByS.set(m, []);
  }
  for (const p of pairs) {
    if (p.condition !== condition) continue;
    if (p.source === p.target) continue;
    const v = p.details?.pDiffImpostor;
    if (v == null || Number.isNaN(v)) continue;
    pDiffsByS.get(p.source)?.push(v);
  }
  const out = new Map<string, number>();
  for (const m of models) {
    const xs = pDiffsByS.get(m) ?? [];
    if (xs.length === 0) {
      out.set(m, 0);
    } else {
      out.set(m, xs.reduce((acc, pd) => acc + (1 - pd), 0) / xs.length);
    }
  }
  return out;
}

/** Previous "A" ranking: actor-side fooling ability across targets. */
export function actorScoreBySource(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  return actorFoolingAsSource(pairs, condition, models);
}

/** Previous "J" ranking: judge-side self/strictness ability across incoming actors. */
export function judgeScoreByTarget(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  return defenderImitationStrictness(pairs, condition, models);
}

/**
 * Single number per model for **graph row** stratification: half “as distinguisher/self”
 * (`defenderImitationStrictness`: ½pSame+½ mean pDiff as target) and half “as actor
 * across all others” (`actorFoolingAsSource`: mean 1−pDiff as source). ε—edges
 * are unchanged; this is layout only.
 */
export function stratificationLayoutScore(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  const j = defenderImitationStrictness(pairs, condition, models);
  const a = actorFoolingAsSource(pairs, condition, models);
  const out = new Map<string, number>();
  for (const m of models) {
    out.set(m, 0.5 * (j.get(m) ?? 0) + 0.5 * (a.get(m) ?? 0));
  }
  return out;
}

/** Previous "S" ranking: combined score = 1/2 A + 1/2 J. */
export function combinedScoreByModel(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  return stratificationLayoutScore(pairs, condition, models);
}

/**
 * Group models (or graph node ids) into discrete tiers: same **quantized** strictness
 * ⇒ same row index. Higher strictness ⇒ smaller tier index (used as upper rows).
 */
export function strictnessTiers(idToScore: Map<string, number>): {
  tierById: Map<string, number>;
  nTiers: number;
} {
  const quant = (s: number) => Math.round(s * 1000) / 1000;
  const ids = [...idToScore.keys()];
  if (ids.length === 0) {
    return { tierById: new Map(), nTiers: 0 };
  }
  const uniq = [...new Set(ids.map((id) => quant(idToScore.get(id) ?? 0)))].sort((a, b) => b - a);
  const tierById = new Map<string, number>();
  for (const id of ids) {
    const s = quant(idToScore.get(id) ?? 0);
    tierById.set(id, Math.max(0, uniq.indexOf(s)));
  }
  return { tierById, nTiers: uniq.length };
}
