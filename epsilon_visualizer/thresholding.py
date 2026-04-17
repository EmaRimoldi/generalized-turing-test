from __future__ import annotations

from dataclasses import dataclass

from .types import PairStatus


@dataclass(frozen=True)
class ThresholdResult:
    status: PairStatus
    threshold_at_point: float | None
    threshold_at_upper: float | None


def classify_pair(
    *,
    epsilon: float,
    estimate: float | None,
    upper: float | None,
    n_valid: int,
    min_valid: int,
) -> ThresholdResult:
    """Classify a pair at a given epsilon.

    - supported: upper <= 0.5 + ε
    - tentative: estimate <= 0.5 + ε but upper > 0.5 + ε
    - unsupported: estimate > 0.5 + ε
    - missing: insufficient valid data or estimate missing
    """
    if estimate is None or upper is None or n_valid < min_valid:
        return ThresholdResult(status="missing", threshold_at_point=None, threshold_at_upper=None)

    thresh_point = max(0.0, float(estimate) - 0.5)
    thresh_upper = max(0.0, float(upper) - 0.5)
    bound = 0.5 + float(epsilon)
    if upper <= bound:
        return ThresholdResult(status="supported", threshold_at_point=thresh_point, threshold_at_upper=thresh_upper)
    if estimate <= bound:
        return ThresholdResult(status="tentative", threshold_at_point=thresh_point, threshold_at_upper=thresh_upper)
    return ThresholdResult(status="unsupported", threshold_at_point=thresh_point, threshold_at_upper=thresh_upper)

