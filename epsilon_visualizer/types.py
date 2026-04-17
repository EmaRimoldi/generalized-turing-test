from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ConditionName = str

Branch = Literal["genuine_target", "imitation"]
PairStatus = Literal["supported", "tentative", "unsupported", "missing"]
EvidenceMode = Literal["strict", "surrogate"]


@dataclass(frozen=True)
class TrialRecord:
    actor_model: str
    distinguisher_model: str
    condition: ConditionName
    branch: Optional[Branch]
    answer: Optional[int]  # 0/1
    valid: bool
    error: Optional[str]
    answered_in_opening: bool
    turn_count: Optional[int]
    specimen_turn_count: Optional[int]
    raw_path: Optional[str]


@dataclass(frozen=True)
class BranchEvidence:
    n_valid: int
    n_excluded: int
    p_hat: Optional[float]
    upper: Optional[float]
    lower: Optional[float]


@dataclass(frozen=True)
class PairEvidence:
    source: str
    target: str
    condition: ConditionName
    mode: EvidenceMode

    n_valid: int
    n_excluded: int

    estimate: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]

    threshold_at_point: Optional[float]
    threshold_at_upper: Optional[float]

    status_at_epsilon: PairStatus

    # secondary / provenance
    p_same_real_b: Optional[float]
    p_diff_impostor: Optional[float]
    p_correct: Optional[float]
    avg_turn_count: Optional[float]
    avg_specimen_turn_count: Optional[float]

    exclusions: dict[str, int]
    raw_paths: list[str]

