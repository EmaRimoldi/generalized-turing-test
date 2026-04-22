from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinomSummary:
    n: int
    k: int

    @property
    def p_hat(self) -> float:
        return self.k / self.n if self.n else float("nan")


def wilson_interval(k: int, n: int, *, z: float = 1.645) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion.

    Default z=1.645 corresponds to ~95% one-sided; we still return a two-sided interval.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) + z2 / (4.0 * n)) / n)
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def wilson_upper(k: int, n: int, *, z: float = 1.645) -> float:
    return wilson_interval(k, n, z=z)[1]


def wilson_lower(k: int, n: int, *, z: float = 1.645) -> float:
    return wilson_interval(k, n, z=z)[0]

