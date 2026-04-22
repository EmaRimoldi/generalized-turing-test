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

    The ordering criterion uses the plug-in estimate of the formal quantity:

        d_hat = P_hat(correct) - 1/2

    - supported: d_hat <= ε
    - unsupported: d_hat > ε
    - missing: insufficient valid data or estimate missing

    The upper argument is accepted for JSON/API compatibility but is not used
    by the threshold rule.
    """
    if estimate is None or n_valid < min_valid:
        return ThresholdResult(status="missing", threshold_at_point=None, threshold_at_upper=None)

    thresh_point = max(0.0, float(estimate) - 0.5)
    if thresh_point <= float(epsilon) + 1e-12:
        return ThresholdResult(status="supported", threshold_at_point=thresh_point, threshold_at_upper=None)
    return ThresholdResult(status="unsupported", threshold_at_point=thresh_point, threshold_at_upper=None)
