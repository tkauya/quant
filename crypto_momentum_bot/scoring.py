from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreBreakdown:
    liquidity: float
    trend: float
    momentum: float
    obv: float
    relative_strength: float
    risk_reward: float

    @property
    def total(self) -> float:
        return round(
            self.liquidity
            + self.trend
            + self.momentum
            + self.obv
            + self.relative_strength
            + self.risk_reward,
            2,
        )


def classify(score: float) -> str:
    if score >= 85:
        return "Tier 1 opportunity"
    if score >= 70:
        return "Strong watchlist"
    if score >= 55:
        return "Early accumulation"
    return "Ignore"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
