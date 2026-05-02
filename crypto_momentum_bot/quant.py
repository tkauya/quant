from __future__ import annotations

import json
import re
from typing import Any


SCORE_COMPONENT_KEYS = (
    "liquidity",
    "trend",
    "momentum",
    "obv",
    "relative_strength",
    "risk_reward",
)


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def setup_type_from_note(note: str) -> str:
    lowered = (note or "").lower()
    if "breakout" in lowered:
        return "breakout"
    if "ema20 reclaim" in lowered:
        return "ema20 reclaim"
    if "compression" in lowered:
        return "compression"
    return "structure developing"


def score_breakdown_to_dict(breakdown: Any) -> dict[str, float]:
    return {
        key: round(float(getattr(breakdown, key, 0) or 0), 4)
        for key in SCORE_COMPONENT_KEYS
    }


def score_breakdown_from_notes(notes: str) -> dict[str, float]:
    match = re.search(r"ScoreBreakdown\(([^)]*)\)", notes or "")
    if not match:
        return {}
    result: dict[str, float] = {}
    for part in match.group(1).split(","):
        key, _, value = part.strip().partition("=")
        if key in SCORE_COMPONENT_KEYS:
            try:
                result[key] = round(float(value), 4)
            except ValueError:
                pass
    return result


def estimate_win_probability(signal: dict[str, Any], market_regime: str = "unknown") -> tuple[float, list[str]]:
    score = float(signal.get("score") or 0)
    notes: list[str] = []

    if score >= 85:
        probability = 0.57
        notes.append("base 57% from Tier 1 score")
    elif score >= 70:
        probability = 0.52
        notes.append("base 52% from Strong watchlist score")
    elif score >= 55:
        probability = 0.47
        notes.append("base 47% from Early accumulation score")
    else:
        probability = 0.40
        notes.append("base 40% from sub-55 score")

    if market_regime == "risk-on":
        probability += 0.03
        notes.append("+3% risk-on regime")
    elif market_regime == "risk-off":
        probability -= 0.04
        notes.append("-4% risk-off regime")

    relative_strength = str(signal.get("relative_strength") or "")
    if "BTC and ETH" in relative_strength:
        probability += 0.03
        notes.append("+3% outperforming BTC and ETH")
    elif "one benchmark" in relative_strength:
        probability += 0.01
        notes.append("+1% outperforming one benchmark")
    elif "lagging" in relative_strength:
        probability -= 0.02
        notes.append("-2% lagging benchmarks")

    trend_status = str(signal.get("trend_status") or "")
    if "4H bullish, 1H aligned" in trend_status:
        probability += 0.03
        notes.append("+3% aligned 4H/1H trend")
    elif "4H bullish" in trend_status:
        probability += 0.015
        notes.append("+1.5% 4H bullish trend")
    elif "downtrend" in trend_status:
        probability -= 0.05
        notes.append("-5% downtrend")

    obv_status = str(signal.get("obv_status") or "")
    if "crossed" in obv_status:
        probability += 0.02
        notes.append("+2% fresh OBV cross")
    elif "above" in obv_status:
        probability += 0.01
        notes.append("+1% OBV above MA")
    elif "below" in obv_status:
        probability -= 0.02
        notes.append("-2% OBV below MA")

    macd_status = str(signal.get("macd_status") or "")
    if "4H and 1H" in macd_status:
        probability += 0.025
        notes.append("+2.5% MACD expanding on 4H and 1H")
    elif "4H" in macd_status:
        probability += 0.015
        notes.append("+1.5% MACD expanding on 4H")
    elif "negative" in macd_status:
        probability -= 0.03
        notes.append("-3% negative MACD")

    try:
        rsi = float(signal.get("rsi") or 0)
    except Exception:
        rsi = 0
    if 55 <= rsi <= 68:
        probability += 0.015
        notes.append("+1.5% RSI in momentum band")
    elif rsi > 78:
        probability -= 0.02
        notes.append("-2% RSI extended")

    probability = max(0.25, min(0.72, probability))
    return round(probability, 4), notes


def calculate_ev(
    signal: dict[str, Any],
    account_size: float,
    risk_percent: float,
    market_regime: str = "unknown",
    quote_amount: float | None = None,
) -> dict[str, Any]:
    price = float(signal.get("price") or signal.get("reference_price") or 0)
    stop = float(signal.get("stop_loss") or 0)
    tp1 = float(signal.get("take_profit_1") or 0)
    tp2 = float(signal.get("take_profit_2") or 0)

    if price <= 0 or stop <= 0 or stop >= price:
        return {
            "ok": False,
            "ev_label": "unknown",
            "ev_notes": ["Invalid price or stop for EV calculation"],
        }

    risk_budget = max(float(account_size or 0) * (float(risk_percent or 0) / 100), 0)
    stop_distance = max(price - stop, price * 0.005)
    derived_quote = min(float(account_size or 0), (risk_budget / stop_distance) * price) if stop_distance else 0
    quote = float(quote_amount) if quote_amount is not None else derived_quote
    base_quantity = quote / price if price else 0

    risk_usdt = stop_distance * base_quantity
    reward_1_usdt = max(tp1 - price, 0) * base_quantity
    reward_2_usdt = max(tp2 - price, 0) * base_quantity
    risk_multiple_tp1 = reward_1_usdt / risk_usdt if risk_usdt else 0
    risk_multiple_tp2 = reward_2_usdt / risk_usdt if risk_usdt else 0
    blended_reward_r = (risk_multiple_tp1 * 0.55) + (risk_multiple_tp2 * 0.45)

    win_probability, notes = estimate_win_probability(signal, market_regime)
    expected_value_r = (win_probability * blended_reward_r) - ((1 - win_probability) * 1)
    expected_value_usdt = expected_value_r * risk_usdt

    if expected_value_r >= 0.35:
        ev_label = "positive"
    elif expected_value_r >= 0.05:
        ev_label = "marginal"
    else:
        ev_label = "negative"

    return {
        "ok": True,
        "risk_usdt": round(risk_usdt, 4),
        "reward_1_usdt": round(reward_1_usdt, 4),
        "reward_2_usdt": round(reward_2_usdt, 4),
        "risk_multiple_tp1": round(risk_multiple_tp1, 4),
        "risk_multiple_tp2": round(risk_multiple_tp2, 4),
        "blended_reward_r": round(blended_reward_r, 4),
        "win_probability": round(win_probability, 4),
        "expected_value_r": round(expected_value_r, 4),
        "expected_value_usdt": round(expected_value_usdt, 4),
        "ev_label": ev_label,
        "quote_amount": round(quote, 2),
        "base_quantity_estimate": round(base_quantity, 8),
        "risk_budget_usdt": round(risk_budget, 4),
        "ev_notes": notes,
    }
