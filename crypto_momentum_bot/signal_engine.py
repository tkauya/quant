from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .scoring import clamp


STRATEGY_VERSION = "signal_v2_obv_rs_structure"


@dataclass
class SignalWeights:
    obv_accumulation: float = 20
    relative_strength: float = 20
    volume_liquidity: float = 15
    momentum_alignment: float = 15
    breakout_structure: float = 15
    volatility_reward_risk: float = 10
    market_regime: float = 5


@dataclass
class SignalDecision:
    symbol: str
    strategy_version: str
    score: float
    tier: str
    signal_type: str
    rejected: bool
    rejection_reason: str
    entry_reason: str
    price: float
    entry_zone: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    trend_status: str
    volume_status: str
    obv_status: str
    rsi: float
    macd_status: str
    relative_strength: str
    relative_strength_btc: float
    relative_strength_eth: float
    market_regime: str
    setup_type: str
    volume_ratio: float
    component_scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def score_breakdown_json(self) -> str:
        return json.dumps(self.component_scores, default=str)

    @property
    def component_scores_json(self) -> str:
        return json.dumps(self.component_scores, default=str)


def classify_v2(score: float, rejected: bool = False) -> str:
    if rejected or score < 65:
        return "Tier C: Ignore"
    if score >= 80:
        return "Tier A: Strong trade"
    return "Tier B: Watchlist"


def score_signal(
    *,
    symbol: str,
    volume_rank: int,
    quote_volume: float,
    entry_tf: pd.DataFrame,
    btc_entry_tf: pd.DataFrame,
    eth_entry_tf: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    fifteen_m: pd.DataFrame | None = None,
    one_h: pd.DataFrame | None = None,
    four_h: pd.DataFrame | None = None,
    btc_four_h: pd.DataFrame | None = None,
    eth_four_h: pd.DataFrame | None = None,
    existing_open_trade: bool = False,
    weights: SignalWeights | None = None,
    strategy_version: str = STRATEGY_VERSION,
) -> SignalDecision:
    weights = weights or SignalWeights()
    rejection_reasons: list[str] = []

    if len(entry_tf) < 80:
        rejection_reasons.append("insufficient candle history")
    if len(btc_entry_tf) < 80 or len(eth_entry_tf) < 80:
        rejection_reasons.append("insufficient BTC/ETH benchmark history")

    df = entry_tf.copy()
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    price = float(latest["close"])
    atr = float(latest.get("atr14") or 0)
    volume_ratio = _safe_ratio(latest.get("volume"), latest.get("vol_ma20"))
    candle_range_pct = _safe_ratio(float(latest["high"]) - float(latest["low"]), price)
    quote_volume = float(quote_volume or 0)

    market_regime = _market_regime(btc_four_h, eth_four_h)
    trend_status = _trend_status(entry_tf, fifteen_m, one_h, four_h, daily)
    volume_status = _volume_status(volume_rank, quote_volume, volume_ratio, candle_range_pct)
    obv_status, obv_points_raw = _obv_state(df, latest, prev)
    macd_status = _macd_status(entry_tf, one_h)
    rs_btc, rs_eth, relative_strength = _relative_strength(df, btc_entry_tf, eth_entry_tf)
    structure, setup_type, breakout_distance = _structure(df)
    entry_zone, stop_loss, tp1, tp2, rr, stop_pct = _risk_plan(df)

    if quote_volume and quote_volume < 2_000_000:
        rejection_reasons.append("24h quote volume below liquidity floor")
    if existing_open_trade:
        rejection_reasons.append("existing open live trade on symbol")
    if rr < 1.5:
        rejection_reasons.append("reward/risk below 1.5")
    if stop_pct > 0.08:
        rejection_reasons.append("ATR stop too wide")
    if breakout_distance > 0.025:
        rejection_reasons.append("price too extended from breakout/reclaim level")
    if market_regime == "risk-off" and not (rs_btc > 0.015 and rs_eth > 0.015):
        rejection_reasons.append("BTC/ETH regime hostile without exceptional relative strength")
    if "exhaustion" in volume_status and rs_btc <= 0:
        rejection_reasons.append("possible exhaustion candle without BTC relative strength")

    component_scores = {
        "obv_accumulation": _weighted(obv_points_raw, weights.obv_accumulation),
        "relative_strength": _weighted(_rs_points(rs_btc, rs_eth), weights.relative_strength),
        "volume_liquidity": _weighted(_volume_points(volume_rank, quote_volume, volume_ratio, candle_range_pct), weights.volume_liquidity),
        "momentum_alignment": _weighted(_momentum_points(entry_tf, fifteen_m, one_h, four_h), weights.momentum_alignment),
        "breakout_structure": _weighted(_structure_points(setup_type, breakout_distance), weights.breakout_structure),
        "volatility_reward_risk": _weighted(_risk_points(rr, stop_pct), weights.volatility_reward_risk),
        "market_regime": _weighted(_regime_points(market_regime), weights.market_regime),
    }
    score = round(sum(component_scores.values()), 2)
    rejected = bool(rejection_reasons)
    tier = classify_v2(score, rejected)

    # Tier A requires the actual confirmation stack, not just a high numeric score.
    if tier.startswith("Tier A"):
        missing = []
        if "below" in obv_status:
            missing.append("OBV confirmation missing")
        if rs_btc <= 0 or rs_eth <= 0:
            missing.append("relative strength not positive vs BTC and ETH")
        if volume_ratio < 1.05:
            missing.append("volume expansion missing")
        if rr < 1.5:
            missing.append("reward/risk not acceptable")
        if market_regime == "risk-off":
            missing.append("market regime hostile")
        if missing:
            rejected = True
            rejection_reasons.extend(missing)
            tier = classify_v2(score, rejected)

    entry_reason = "; ".join(
        [
            strategy_version,
            setup_type,
            trend_status,
            volume_status,
            obv_status,
            relative_strength,
            macd_status,
            f"RR {rr}",
        ]
    )

    return SignalDecision(
        symbol=symbol,
        strategy_version=strategy_version,
        score=score,
        tier=tier,
        signal_type=setup_type,
        rejected=rejected,
        rejection_reason="; ".join(dict.fromkeys(rejection_reasons)),
        entry_reason=entry_reason,
        price=price,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_reward=rr,
        trend_status=trend_status,
        volume_status=volume_status,
        obv_status=obv_status,
        rsi=round(float(latest.get("rsi14") or 0), 2),
        macd_status=macd_status,
        relative_strength=relative_strength,
        relative_strength_btc=round(rs_btc, 6),
        relative_strength_eth=round(rs_eth, 6),
        market_regime=market_regime,
        setup_type=setup_type,
        volume_ratio=round(volume_ratio, 4),
        component_scores=component_scores,
        diagnostics={
            "volume_rank": volume_rank,
            "quote_volume": quote_volume,
            "atr": atr,
            "stop_pct": round(stop_pct, 6),
            "breakout_distance": round(breakout_distance, 6),
            "candle_range_pct": round(candle_range_pct, 6),
            "structure": structure,
        },
    )


def decision_to_signal_dict(decision: SignalDecision, timestamp: str, volume_rank: int) -> dict[str, Any]:
    data = asdict(decision)
    return {
        "timestamp": timestamp,
        "symbol": decision.symbol,
        "volume_rank": volume_rank,
        "price": decision.price,
        "score": decision.score,
        "tier": decision.tier,
        "trend_status": decision.trend_status,
        "volume_status": decision.volume_status,
        "obv_status": decision.obv_status,
        "rsi": decision.rsi,
        "macd_status": decision.macd_status,
        "relative_strength": decision.relative_strength,
        "entry_zone": decision.entry_zone,
        "stop_loss": decision.stop_loss,
        "take_profit_1": decision.take_profit_1,
        "take_profit_2": decision.take_profit_2,
        "risk_reward": decision.risk_reward,
        "notes": decision.entry_reason,
        "score_breakdown_json": decision.score_breakdown_json,
        "ev_json": "{}",
        "setup_type": decision.setup_type,
        "volume_ratio": decision.volume_ratio,
        "strategy_version": decision.strategy_version,
        "signal_type": decision.signal_type,
        "rejection_reason": decision.rejection_reason,
        "component_scores_json": decision.component_scores_json,
        "relative_strength_btc": decision.relative_strength_btc,
        "relative_strength_eth": decision.relative_strength_eth,
        "market_regime": decision.market_regime,
        "diagnostics_json": json.dumps(data.get("diagnostics") or {}, default=str),
    }


def is_live_eligible(signal: dict[str, Any]) -> bool:
    return float(signal.get("score") or 0) >= 80 and not str(signal.get("rejection_reason") or "").strip()


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator = float(denominator or 0)
        if denominator == 0:
            return 0.0
        return float(numerator or 0) / denominator
    except Exception:
        return 0.0


def _trend_label(df: pd.DataFrame | None) -> str:
    if df is None or len(df) < 50:
        return "unknown"
    latest = df.iloc[-1]
    if latest["close"] > latest["ema20"] > latest["ema50"]:
        return "bullish"
    if latest["close"] < latest["ema20"] < latest["ema50"]:
        return "bearish"
    return "neutral"


def _market_regime(btc: pd.DataFrame | None, eth: pd.DataFrame | None) -> str:
    btc_label = _trend_label(btc)
    eth_label = _trend_label(eth)
    btc_ret = _pct_change(btc, 6)
    eth_ret = _pct_change(eth, 6)
    if btc_ret < -0.025 and eth_ret < -0.025:
        return "risk-off"
    if btc_label == "bullish" and eth_label == "bullish":
        return "risk-on"
    if btc_label == "bearish" and eth_label == "bearish":
        return "risk-off"
    return "neutral"


def _trend_status(entry: pd.DataFrame, fifteen: pd.DataFrame | None, one_h: pd.DataFrame | None, four_h: pd.DataFrame | None, daily: pd.DataFrame | None) -> str:
    labels = {
        "5m": _trend_label(entry),
        "15m": _trend_label(fifteen),
        "1h": _trend_label(one_h),
        "4h": _trend_label(four_h),
        "1d": _trend_label(daily),
    }
    if labels["15m"] == labels["1h"] == "bullish" and labels["4h"] in {"bullish", "neutral"}:
        return "15m/1H bullish, 4H supportive"
    if labels["1h"] == "bullish" and labels["4h"] in {"bullish", "neutral"}:
        return "1H bullish, 4H supportive"
    if labels["5m"] == "bullish":
        return "5m bullish only"
    if labels["1h"] == "bearish" and labels["4h"] == "bearish":
        return "1H/4H bearish"
    return "mixed / basing"


def _volume_status(volume_rank: int, quote_volume: float, volume_ratio: float, candle_range_pct: float) -> str:
    if quote_volume and quote_volume < 2_000_000:
        return "low liquidity"
    if volume_ratio >= 2.5 and candle_range_pct > 0.035:
        return "possible exhaustion volume"
    if volume_ratio >= 1.8:
        return "major expansion"
    if volume_ratio >= 1.2:
        return "above average"
    if volume_ratio >= 0.8:
        return "normal"
    return "quiet"


def _obv_state(df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> tuple[str, float]:
    above = latest["obv"] > latest["obv_ma7"]
    crossed = prev["obv"] <= prev["obv_ma7"] and latest["obv"] > latest["obv_ma7"]
    obv_slope = float(df["obv"].diff().tail(5).mean() or 0)
    obv_accel = float(df["obv"].diff().diff().tail(5).mean() or 0)
    price_ret = _series_return(df["close"], 12)
    obv_ret = _series_return(df["obv"], 12)
    divergence = price_ret <= 0.005 and obv_ret > 0.01
    points = 0
    if above:
        points += 4
    if crossed:
        points += 3
    if obv_slope > 0:
        points += 2
    if obv_accel > 0:
        points += 1
    if divergence:
        points += 2
    if crossed:
        label = "crossed above OBV MA(7)"
    elif above and divergence:
        label = "OBV accumulation divergence"
    elif above:
        label = "above OBV MA(7)"
    else:
        label = "below OBV MA(7)"
    return label, clamp(points, 0, 10)


def _relative_strength(coin: pd.DataFrame, btc: pd.DataFrame, eth: pd.DataFrame) -> tuple[float, float, str]:
    periods = [12, 48, 288]
    btc_edges = []
    eth_edges = []
    for period in periods:
        if len(coin) > period and len(btc) > period and len(eth) > period:
            coin_ret = _series_return(coin["close"], period)
            btc_edges.append(coin_ret - _series_return(btc["close"], period))
            eth_edges.append(coin_ret - _series_return(eth["close"], period))
    rs_btc = sum(btc_edges) / len(btc_edges) if btc_edges else 0.0
    rs_eth = sum(eth_edges) / len(eth_edges) if eth_edges else 0.0
    if rs_btc > 0 and rs_eth > 0:
        label = "outperforming BTC and ETH"
    elif rs_btc > 0 or rs_eth > 0:
        label = "outperforming one benchmark"
    else:
        label = "lagging BTC/ETH"
    return rs_btc, rs_eth, label


def _macd_status(entry: pd.DataFrame, one_h: pd.DataFrame | None) -> str:
    latest = entry.iloc[-1]
    prev = entry.iloc[-2]
    entry_expanding = latest["macd_hist"] > 0 and latest["macd_hist"] > prev["macd_hist"]
    h1_expanding = False
    if one_h is not None and len(one_h) > 2:
        h1_expanding = one_h.iloc[-1]["macd_hist"] > 0 and one_h.iloc[-1]["macd_hist"] > one_h.iloc[-2]["macd_hist"]
    if entry_expanding and h1_expanding:
        return "positive expansion on entry and 1H"
    if entry_expanding:
        return "positive expansion on entry timeframe"
    if latest["macd_hist"] > 0:
        return "positive but fading"
    return "negative"


def _structure(df: pd.DataFrame) -> tuple[str, str, float]:
    latest = df.iloc[-1]
    recent = df.tail(24)
    prior = df.iloc[-25:-1] if len(df) > 25 else df.tail(24)
    high = float(prior["high"].max())
    low = float(prior["low"].min())
    price = float(latest["close"])
    compression = recent["range_pct"].tail(8).mean() < recent["range_pct"].mean() * 0.75
    expansion = _safe_ratio(latest["volume"], latest["vol_ma20"]) >= 1.2
    higher_lows = len(recent) >= 8 and recent["low"].iloc[-1] > recent["low"].iloc[-5] > recent["low"].iloc[-8]
    reclaim = latest["low"] <= latest["ema20"] <= latest["close"]
    breakout_distance = max((price - high) / price, 0) if price else 0
    if price >= high * 0.998 and expansion:
        return "recent resistance breakout with volume", "OBV breakout", breakout_distance
    if compression and expansion and price > high * 0.99:
        return "compression followed by expansion", "compression breakout", breakout_distance
    if reclaim:
        return "EMA20 reclaim", "pullback continuation", abs(price - float(latest["ema20"])) / price
    if higher_lows and price > latest["ema20"]:
        return "higher lows continuation", "relative strength continuation", abs(price - float(latest["ema20"])) / price
    return "structure developing", "structure developing", abs(price - float(latest["ema20"])) / price


def _risk_plan(df: pd.DataFrame) -> tuple[str, float, float, float, float, float]:
    latest = df.iloc[-1]
    price = float(latest["close"])
    atr = float(latest["atr14"])
    recent_low = float(df.tail(12)["low"].min())
    raw_stop = min(price - atr * 1.2, recent_low * 0.995)
    stop_loss = round(max(raw_stop, price * 0.92), 8)
    risk = max(price - stop_loss, price * 0.004)
    tp1 = round(price + risk * 1.5, 8)
    tp2 = round(price + risk * 2.5, 8)
    rr = round((tp2 - price) / risk, 2) if risk > 0 else 0
    stop_pct = risk / price if price else 0
    return f"{round(price * 0.995, 8)} - {round(price * 1.005, 8)}", stop_loss, tp1, tp2, rr, stop_pct


def _volume_points(volume_rank: int, quote_volume: float, volume_ratio: float, candle_range_pct: float) -> float:
    points = 3 if volume_rank <= 40 else 1
    points += 3 if volume_rank <= 10 else 2 if volume_rank <= 25 else 1
    points += 3 if volume_ratio >= 1.8 else 2 if volume_ratio >= 1.2 else 1 if volume_ratio >= 0.8 else 0
    points += 1 if quote_volume >= 10_000_000 or not quote_volume else 0
    if volume_ratio >= 2.5 and candle_range_pct > 0.035:
        points -= 3
    return clamp(points, 0, 10)


def _rs_points(rs_btc: float, rs_eth: float) -> float:
    points = 0
    points += 4 if rs_btc > 0.01 else 3 if rs_btc > 0 else 1 if rs_btc > -0.005 else 0
    points += 4 if rs_eth > 0.01 else 3 if rs_eth > 0 else 1 if rs_eth > -0.005 else 0
    points += 2 if rs_btc > 0 and rs_eth > 0 else 0
    return clamp(points, 0, 10)


def _momentum_points(entry: pd.DataFrame, fifteen: pd.DataFrame | None, one_h: pd.DataFrame | None, four_h: pd.DataFrame | None) -> float:
    latest = entry.iloc[-1]
    rsi = float(latest["rsi14"])
    points = 0
    for frame in [entry, fifteen, one_h, four_h]:
        label = _trend_label(frame)
        points += 1.8 if label == "bullish" else 0.8 if label == "neutral" else 0
    points += 2 if 50 <= rsi <= 72 else 1 if 45 <= rsi < 50 or 72 < rsi <= 80 else -1
    if latest["macd_hist"] > 0 and latest["macd_hist"] > entry.iloc[-2]["macd_hist"]:
        points += 1
    return clamp(points, 0, 10)


def _structure_points(setup_type: str, breakout_distance: float) -> float:
    base = {
        "OBV breakout": 9,
        "compression breakout": 8,
        "pullback continuation": 7,
        "relative strength continuation": 7,
        "structure developing": 3,
    }.get(setup_type, 3)
    if breakout_distance > 0.025:
        base -= 4
    elif breakout_distance <= 0.012:
        base += 1
    return clamp(base, 0, 10)


def _risk_points(rr: float, stop_pct: float) -> float:
    points = 0
    points += 5 if rr >= 2 else 3 if rr >= 1.5 else 0
    points += 3 if 0.004 <= stop_pct <= 0.04 else 1 if stop_pct <= 0.08 else 0
    points += 2 if rr >= 2.5 else 0
    return clamp(points, 0, 10)


def _regime_points(regime: str) -> float:
    if regime == "risk-on":
        return 10
    if regime == "neutral":
        return 6
    return 2


def _weighted(points_0_to_10: float, weight: float) -> float:
    return round(clamp(points_0_to_10, 0, 10) / 10 * weight, 4)


def _pct_change(df: pd.DataFrame | None, period: int) -> float:
    if df is None or len(df) <= period:
        return 0.0
    return _series_return(df["close"], period)


def _series_return(series: pd.Series, period: int) -> float:
    if len(series) <= period:
        return 0.0
    start = float(series.iloc[-period])
    end = float(series.iloc[-1])
    if start == 0:
        return 0.0
    return (end / start) - 1
