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
