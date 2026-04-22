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

export function formatPct(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(100 * x).toFixed(1)}%`;
}

/**
 * Per distinguisher (target) D: mean of P(answer=0 | imitation) = pDiffImpostor over
 * off-diagonal pairs (source, D) — "how often" imitators are caught, i.e. harder to fool.
 */
export function defenderImitationStrictness(
  pairs: PairRow[],
  condition: string,
  models: string[]
): Map<string, number> {
  const byTarget = new Map<string, number[]>();
  for (const m of models) {
    byTarget.set(m, []);
  }
  for (const p of pairs) {
    if (p.condition !== condition) continue;
    if (p.source === p.target) continue;
    const v = p.details?.pDiffImpostor;
    if (v == null || Number.isNaN(v)) continue;
    const arr = byTarget.get(p.target);
    if (arr) arr.push(v);
  }
  const out = new Map<string, number>();
  for (const m of models) {
    const xs = byTarget.get(m) ?? [];
    out.set(m, xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  }
  return out;
}
