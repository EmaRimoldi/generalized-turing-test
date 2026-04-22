export type PairStatus = "supported" | "tentative" | "unsupported" | "missing";
export type EvidenceMode = "strict";

export interface PairRow {
  source: string;
  target: string;
  condition: string;
  mode: EvidenceMode;
  nValid: number;
  nExcluded: number;
  estimate: number | null;
  lowerBound: number | null;
  upperBound: number | null;
  thresholdAtPoint: number | null;
  thresholdAtUpper: number | null;
  /** plug-in d_hat = P_hat(correct) - 1/2 (strict mode: combined branch estimate). */
  dHat?: number | null;
  /** max(0, d_hat); edge A->B at epsilon iff this <= epsilon. */
  epsilonStarHat?: number | null;
  statusAtEpsilon: PairStatus;
  details: {
    pSameRealB?: number | null;
    pDiffImpostor?: number | null;
    pCorrect?: number | null;
    averageTurnCount?: number | null;
    averageSpecimenTurnCount?: number | null;
    exclusionCounts?: Record<string, number>;
    rawPaths?: string[];
    imitationDataProvenance?: string | null;
    genuineDataProvenance?: string | null;
  };
}

export interface VisualizerArtifact {
  mode: EvidenceMode;
  availableModes?: EvidenceMode[];
  /** Present when JSON was built from aggregate matrix CSV vs per-trial results.csv. */
  dataSource?: "matrix_a_csv" | "results_csv" | "grouped_json";
  /** When built from grouped JSON: how directed pairs were resolved (JSON vs matrix vs imputed). */
  pairIngestStats?: Record<string, number> | null;
  epsilonDefault: number;
  minValidPerBranch: number;
  confidenceMethod: { kind: string; z?: number };
  filtersDefault: {
    excludeAnsweredInOpening: boolean;
    excludeAnyError: boolean;
  };
  models: string[];
  modelsByCondition?: Record<string, string[]>;
  conditions: string[];
  pairs: PairRow[];
  formalismRef?: string;
}
