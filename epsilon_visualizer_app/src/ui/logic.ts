import type { PairRow, PairStatus } from "./types";

export function classifyAtEpsilon(
  row: PairRow,
  epsilon: number,
  minValid: number
): PairStatus {
  if (
    row.estimate == null ||
    row.upperBound == null ||
    row.nValid < minValid
  ) {
    return "missing";
  }
  const bound = 0.5 + epsilon;
  if (row.upperBound <= bound) return "supported";
  if (row.estimate <= bound) return "tentative";
  return "unsupported";
}

export function formatPct(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(100 * x).toFixed(1)}%`;
}

