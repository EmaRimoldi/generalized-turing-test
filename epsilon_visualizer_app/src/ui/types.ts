export type PairStatus = "supported" | "tentative" | "unsupported" | "missing";
export type EvidenceMode = "strict" | "surrogate";

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
  /** max(0, d_hat); edge A->B at epsilon iff this <= epsilon for point estimate; UI uses Wilson bands. */
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
  };
}

export interface VisualizerArtifact {
  mode: EvidenceMode;
  availableModes?: EvidenceMode[];
  epsilonDefault: number;
  minValidPerBranch: number;
  confidenceMethod: { kind: string; z: number };
  filtersDefault: {
    excludeAnsweredInOpening: boolean;
    excludeAnyError: boolean;
  };
  models: string[];
  conditions: string[];
  pairs: PairRow[];
  formalismRef?: string;
}

