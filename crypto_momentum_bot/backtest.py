from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .indicators import enrich
from .market_data import BinanceMarketData, MarketTicker
from .scoring import ScoreBreakdown, classify, clamp
from .signal_engine import STRATEGY_VERSION, score_signal
from .settings import get_settings


@dataclass
class BacktestTrade:
    symbol: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    score: float
    tier: str
    outcome: str
    pnl_percent: float
    pnl_usdt: float
    r_multiple: float
    reason: str
    strategy_version: str = STRATEGY_VERSION
    signal_type: str = ""
    market_regime: str = ""
    score_bucket: str = ""


class BacktestEngine:
    def __init__(
        self,
        market_data: BinanceMarketData,
        universe_size: int = 10,
        bars: int = 720,
        min_score: float = 70,
        account_size: float = 1000,
        risk_percent: float = 1,
        fee_bps: float = 10,
        timeframe: str = "4h",
        symbols: list[str] | None = None,
        scalp_mode: bool = False,
        scalp_stop_atr: float = 0.6,
        scalp_tp1_r: float = 0.6,
        scalp_tp2_r: float = 1.1,
        scalp_max_hold_minutes: int = 90,
        scalp_min_risk_pct: float = 0.0015,
    ):
        self.market_data = market_data
        self.universe_size = universe_size
        self.bars = bars
        self.min_score = min_score
        self.account_size = account_size
        self.risk_percent = risk_percent
        self.fee_bps = fee_bps
        self.timeframe = timeframe
        self.symbols = symbols
        self.scalp_mode = scalp_mode
        self.scalp_stop_atr = scalp_stop_atr
        self.scalp_tp1_r = scalp_tp1_r
        self.scalp_tp2_r = scalp_tp2_r
        self.scalp_max_hold_minutes = scalp_max_hold_minutes
        self.scalp_min_risk_pct = scalp_min_risk_pct

    def run(self) -> dict[str, Any]:
        universe = self._universe()
        btc = enrich(self.market_data.fetch_ohlcv("BTC/USDT", self.timeframe, limit=self.bars + 100))
        eth = enrich(self.market_data.fetch_ohlcv("ETH/USDT", self.timeframe, limit=self.bars + 100))
        btc_4h = enrich(self.market_data.fetch_ohlcv("BTC/USDT", "4h", limit=max(120, int(self.bars / 48) + 100)))
        eth_4h = enrich(self.market_data.fetch_ohlcv("ETH/USDT", "4h", limit=max(120, int(self.bars / 48) + 100)))
        trades: list[BacktestTrade] = []
        errors: list[str] = []
        universe_errors = [ticker.symbol for ticker in universe if ticker.quote_volume < 0]
        errors.extend([f"{symbol}: not an active Binance spot USDT market" for symbol in universe_errors])
        universe = [ticker for ticker in universe if ticker.quote_volume >= 0]

        for rank, ticker in enumerate(universe, start=1):
            try:
                df = enrich(self.market_data.fetch_ohlcv(ticker.symbol, self.timeframe, limit=self.bars + 100))
                trend_1h = None
                trend_4h = None
                trend_15m = None
                trend_daily = None
                if self.timeframe == "5m" or self.scalp_mode:
                    trend_15m = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "15m", limit=max(120, int(self.bars / 3) + 100)))
                    trend_1h = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "1h", limit=max(120, int(self.bars / 12) + 100)))
                    trend_4h = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "4h", limit=max(120, int(self.bars / 48) + 100)))
                    trend_daily = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "1d", limit=180))
                trades.extend(self._run_symbol(ticker, rank, df, btc, eth, trend_1h, trend_4h, trend_15m, trend_daily, btc_4h, eth_4h))
            except Exception as exc:
                errors.append(f"{ticker.symbol}: {exc}")

        trades.sort(key=lambda trade: trade.entry_time)
        summary = self._summary(trades)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assumptions": {
                "universe": "Current top Binance spot USDT pairs by quote volume, not historical constituents",
                "universe_size": self.universe_size,
                "timeframe": self.timeframe,
                "bars_requested": self.bars,
                "min_score": self.min_score,
                "entry": f"Next {self.timeframe} candle open after signal",
                "exit": "Full size exits at TP1 or stop loss; if both touch same candle, stop is assumed first",
                "fees": f"{self.fee_bps} bps per side",
                "scalp_mode": self.scalp_mode,
                "strategy_version": STRATEGY_VERSION,
                "scalp_exit": "50% TP1, 50% TP2, hard time stop, or momentum-fade exit" if self.scalp_mode else None,
                "scalp_params": {
                    "stop_atr": self.scalp_stop_atr,
                    "tp1_r": self.scalp_tp1_r,
                    "tp2_r": self.scalp_tp2_r,
                    "max_hold_minutes": self.scalp_max_hold_minutes,
                    "min_risk_pct": self.scalp_min_risk_pct,
                } if self.scalp_mode else None,
            },
            "summary": summary,
            "symbols": [ticker.symbol for ticker in universe],
            "errors": errors,
            "trades": [asdict(trade) for trade in trades],
        }

    def _universe(self) -> list[MarketTicker]:
        if not self.symbols:
            return self.market_data.load_top_usdt_pairs(self.universe_size)
        self.market_data.ensure_markets_loaded()
        tickers = self.market_data.exchange.fetch_tickers(self.symbols)
        universe: list[MarketTicker] = []
        for rank, symbol in enumerate(self.symbols, start=1):
            market = self.market_data.exchange.market(symbol)
            if not self.market_data._is_tradeable_usdt_spot(symbol, market):
                universe.append(MarketTicker(symbol=symbol, base=symbol.split("/")[0], quote_volume=-1, last=0))
                continue
            ticker = tickers.get(symbol) or {}
            universe.append(
                MarketTicker(
                    symbol=symbol,
                    base=str(market.get("base") or symbol.split("/")[0]),
                    quote_volume=float(ticker.get("quoteVolume") or 0),
                    last=float(ticker.get("last") or ticker.get("close") or 0),
                )
            )
        return universe

    def _run_symbol(
        self,
        ticker: MarketTicker,
        volume_rank: int,
        df: pd.DataFrame,
        btc: pd.DataFrame,
        eth: pd.DataFrame,
        trend_1h: pd.DataFrame | None = None,
        trend_4h: pd.DataFrame | None = None,
        trend_15m: pd.DataFrame | None = None,
        trend_daily: pd.DataFrame | None = None,
        btc_4h: pd.DataFrame | None = None,
        eth_4h: pd.DataFrame | None = None,
    ) -> list[BacktestTrade]:
        trades: list[BacktestTrade] = []
        warmup = 80
        i = warmup
        max_i = len(df) - 2

        while i < max_i:
            if self.scalp_mode and not self._higher_tf_bullish(df.iloc[i]["timestamp"], trend_1h, trend_4h):
                i += 1
                continue
            ts = df.iloc[i]["timestamp"]
            signal = self._signal_at(
                ticker,
                volume_rank,
                df,
                btc,
                eth,
                i,
                self._slice_to(trend_15m, ts),
                self._slice_to(trend_1h, ts),
                self._slice_to(trend_4h, ts),
                self._slice_to(trend_daily, ts),
                self._slice_to(btc_4h, ts),
                self._slice_to(eth_4h, ts),
            )
            if signal["score"] < self.min_score:
                i += 1
                continue
            if signal.get("rejection_reason"):
                i += 1
                continue

            entry_index = i + 1
            entry_price = float(df.iloc[entry_index]["open"])
            stop_loss = float(signal["stop_loss"])
            take_profit_1 = float(signal["take_profit_1"])
            take_profit_2 = float(signal["take_profit_2"])

            if not (stop_loss < entry_price < take_profit_1 < take_profit_2):
                i += 1
                continue

            risk_per_unit = entry_price - stop_loss
            risk_usdt = self.account_size * (self.risk_percent / 100)
            risk_quantity = risk_usdt / risk_per_unit if risk_per_unit > 0 else 0
            max_quantity = self.account_size / entry_price if entry_price > 0 else 0
            quantity = min(risk_quantity, max_quantity)
            fills = (
                self._find_scalp_exits(df, entry_index, entry_price, stop_loss, take_profit_1, take_profit_2)
                if self.scalp_mode
                else self._find_momentum_exits(df, entry_index, entry_price, stop_loss, take_profit_1, take_profit_2)
            )
            exit_index = max(fill["index"] for fill in fills)
            exit_price = sum(fill["price"] * fill["fraction"] for fill in fills)
            outcome = "+".join(fill["label"] for fill in fills)
            gross_pnl = sum((fill["price"] - entry_price) * quantity * fill["fraction"] for fill in fills)
            fee = (entry_price * quantity + sum(fill["price"] * quantity * fill["fraction"] for fill in fills)) * (self.fee_bps / 10000)
            pnl_usdt = gross_pnl - fee
            pnl_percent = (pnl_usdt / (entry_price * quantity)) * 100 if quantity else 0
            r_multiple = pnl_usdt / risk_usdt if risk_usdt else 0

            trades.append(
                BacktestTrade(
                    symbol=ticker.symbol,
                    signal_time=self._ts(df.iloc[i]),
                    entry_time=self._ts(df.iloc[entry_index]),
                    exit_time=self._ts(df.iloc[exit_index]),
                    entry_price=round(entry_price, 8),
                    exit_price=round(exit_price, 8),
                    stop_loss=round(stop_loss, 8),
                    take_profit_1=round(take_profit_1, 8),
                    take_profit_2=round(take_profit_2, 8),
                    score=signal["score"],
                    tier=signal.get("tier", classify(signal["score"])),
                    outcome=outcome,
                    pnl_percent=round(pnl_percent, 4),
                    pnl_usdt=round(pnl_usdt, 4),
                    r_multiple=round(r_multiple, 4),
                    reason=signal["reason"],
                    strategy_version=signal.get("strategy_version", STRATEGY_VERSION),
                    signal_type=signal.get("signal_type", ""),
                    market_regime=signal.get("market_regime", ""),
                    score_bucket=self._score_bucket(signal["score"]),
                )
            )
            i = exit_index + 1

        return trades

    def _signal_at(
        self,
        ticker: MarketTicker,
        volume_rank: int,
        df: pd.DataFrame,
        btc: pd.DataFrame,
        eth: pd.DataFrame,
        i: int,
        fifteen_m: pd.DataFrame | None = None,
        one_h: pd.DataFrame | None = None,
        four_h: pd.DataFrame | None = None,
        daily: pd.DataFrame | None = None,
        btc_4h: pd.DataFrame | None = None,
        eth_4h: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        current = df.iloc[: i + 1]
        decision = score_signal(
            symbol=ticker.symbol,
            volume_rank=volume_rank,
            quote_volume=ticker.quote_volume,
            entry_tf=current,
            btc_entry_tf=btc.iloc[: i + 1],
            eth_entry_tf=eth.iloc[: i + 1],
            daily=daily,
            fifteen_m=fifteen_m,
            one_h=one_h,
            four_h=four_h,
            btc_four_h=btc_4h,
            eth_four_h=eth_4h,
        )
        return {
            "symbol": ticker.symbol,
            "score": decision.score,
            "stop_loss": decision.stop_loss,
            "take_profit_1": decision.take_profit_1,
            "take_profit_2": decision.take_profit_2,
            "reason": decision.entry_reason,
            "rejection_reason": decision.rejection_reason,
            "strategy_version": decision.strategy_version,
            "tier": decision.tier,
            "signal_type": decision.signal_type,
            "market_regime": decision.market_regime,
        }

    def _find_momentum_exits(
        self,
        df: pd.DataFrame,
        entry_index: int,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
    ) -> list[dict[str, Any]]:
        risk = entry_price - stop_loss
        remaining = 1.0
        fills: list[dict[str, Any]] = []
        tp1_done = False
        tp2_done = False
        trail = stop_loss
        high_water = entry_price

        for j in range(entry_index, len(df)):
            row = df.iloc[j]
            low = float(row["low"])
            high = float(row["high"])
            if low <= trail:
                fills.append({"index": j, "price": trail, "fraction": remaining, "label": "trail_stop" if trail > stop_loss else "stop"})
                return fills
            if not tp1_done and high >= take_profit_1:
                fills.append({"index": j, "price": take_profit_1, "fraction": 0.33, "label": "tp1"})
                remaining -= 0.33
                tp1_done = True
            if not tp2_done and high >= take_profit_2:
                fills.append({"index": j, "price": take_profit_2, "fraction": 0.33, "label": "tp2"})
                remaining -= 0.33
                tp2_done = True
            high_water = max(high_water, high)
            if tp1_done:
                trail = max(trail, entry_price, high_water - risk)
        last_index = len(df) - 1
        if remaining > 0:
            fills.append({"index": last_index, "price": float(df.iloc[last_index]["close"]), "fraction": remaining, "label": "open_mark"})
        return fills

    def _find_scalp_exits(
        self,
        df: pd.DataFrame,
        entry_index: int,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
    ) -> list[dict[str, Any]]:
        max_bars = max(1, int(self.scalp_max_hold_minutes / self._timeframe_minutes()))
        remaining = 1.0
        fills: list[dict[str, Any]] = []
        tp1_done = False
        last_index = min(len(df) - 1, entry_index + max_bars)

        for j in range(entry_index, last_index + 1):
            row = df.iloc[j]
            low = float(row["low"])
            high = float(row["high"])
            if low <= stop_loss:
                fills.append({"index": j, "price": stop_loss, "fraction": remaining, "label": "stop"})
                return fills
            if not tp1_done and high >= take_profit_1:
                fills.append({"index": j, "price": take_profit_1, "fraction": 0.5, "label": "tp1"})
                remaining = 0.5
                tp1_done = True
            if high >= take_profit_2:
                fills.append({"index": j, "price": take_profit_2, "fraction": remaining, "label": "tp2"})
                return fills
            if j > entry_index + 1 and remaining > 0:
                prev = df.iloc[j - 1]
                momentum_faded = row["macd_hist"] < prev["macd_hist"] and row["rsi14"] < 50 and row["close"] < row["ema20"]
                if momentum_faded:
                    fills.append({"index": j, "price": float(row["close"]), "fraction": remaining, "label": "fade"})
                    return fills

        if remaining > 0:
            fills.append({"index": last_index, "price": float(df.iloc[last_index]["close"]), "fraction": remaining, "label": "time_stop"})
        return fills

    def _trend_status(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        if latest["close"] > latest["ema20"] > latest["ema50"]:
            return "4H bullish"
        if latest["close"] > latest["ema20"]:
            return "4H reclaiming EMA20"
        return "not bullish"

    def _volume_status(self, latest: pd.Series) -> str:
        ratio = float(latest["volume"] / latest["vol_ma20"]) if latest["vol_ma20"] else 0
        if ratio >= 1.8:
            return "major expansion"
        if ratio >= 1.2:
            return "above average"
        if ratio >= 0.8:
            return "normal"
        return "quiet"

    def _obv_status(self, latest: pd.Series, prev: pd.Series) -> str:
        if prev["obv"] <= prev["obv_ma7"] and latest["obv"] > latest["obv_ma7"]:
            return "crossed above OBV MA(7)"
        if latest["obv"] > latest["obv_ma7"]:
            return "above OBV MA(7)"
        return "below OBV MA(7)"

    def _macd_status(self, latest: pd.Series, prev: pd.Series) -> str:
        if latest["macd_hist"] > 0 and latest["macd_hist"] > prev["macd_hist"]:
            return "positive expansion on 4H"
        if latest["macd_hist"] > 0:
            return "positive but fading"
        return "negative"

    def _relative_strength(self, coin: pd.DataFrame, btc: pd.DataFrame, eth: pd.DataFrame) -> str:
        coin_ret = coin["close"].pct_change(12).iloc[-1]
        btc_ret = btc["close"].pct_change(12).iloc[-1]
        eth_ret = eth["close"].pct_change(12).iloc[-1]
        if coin_ret > btc_ret and coin_ret > eth_ret:
            return "outperforming BTC and ETH"
        if coin_ret > btc_ret or coin_ret > eth_ret:
            return "outperforming one benchmark"
        return "lagging BTC/ETH"

    def _risk_plan(self, df: pd.DataFrame) -> tuple[float, float, float, float]:
        latest = df.iloc[-1]
        price = float(latest["close"])
        atr = float(latest["atr14"])
        recent_low = float(df.tail(10)["low"].min())
        raw_stop = min(price - atr * 1.2, recent_low * 0.995)
        stop_loss = round(max(raw_stop, price * 0.7), 8)
        risk = max(price - stop_loss, price * 0.005)
        tp1 = round(price + risk * 1.5, 8)
        tp2 = round(price + risk * 2.5, 8)
        rr = round(((1.5 * 0.33) + (2.5 * 0.33) + (2.5 * 0.34)), 2) if risk > 0 else 0
        return stop_loss, tp1, tp2, rr

    def _scalp_risk_plan(self, df: pd.DataFrame) -> tuple[float, float, float, float]:
        latest = df.iloc[-1]
        price = float(latest["close"])
        atr = float(latest["atr14"])
        risk = max(atr * self.scalp_stop_atr, price * self.scalp_min_risk_pct)
        stop_loss = round(price - risk, 8)
        tp1 = round(price + risk * self.scalp_tp1_r, 8)
        tp2 = round(price + risk * self.scalp_tp2_r, 8)
        rr = round((self.scalp_tp1_r * 0.5) + (self.scalp_tp2_r * 0.5), 2)
        return stop_loss, tp1, tp2, rr

    def _higher_tf_bullish(
        self,
        timestamp: Any,
        trend_1h: pd.DataFrame | None,
        trend_4h: pd.DataFrame | None,
    ) -> bool:
        return self._trend_df_bullish(timestamp, trend_1h) or self._trend_df_bullish(timestamp, trend_4h)

    def _trend_df_bullish(self, timestamp: Any, df: pd.DataFrame | None) -> bool:
        if df is None or df.empty:
            return False
        current = df[df["timestamp"] <= timestamp]
        if current.empty:
            return False
        latest = current.iloc[-1]
        return bool(latest["close"] > latest["ema20"] > latest["ema50"])

    def _timeframe_minutes(self) -> int:
        suffix = self.timeframe[-1]
        amount = int(self.timeframe[:-1])
        if suffix == "m":
            return amount
        if suffix == "h":
            return amount * 60
        return amount * 1440

    def _score(
        self,
        volume_rank: int,
        latest: pd.Series,
        trend_status: str,
        volume_status: str,
        obv_status: str,
        macd_status: str,
        relative_strength: str,
        rr: float,
    ) -> ScoreBreakdown:
        liquidity = clamp(21 - (volume_rank * 1.0), 4, 20)
        trend = 18 if trend_status == "4H bullish" else 10 if "EMA20" in trend_status else 2
        rsi = float(latest["rsi14"])
        rsi_points = 8 if 50 <= rsi <= 75 else 4 if 45 <= rsi < 50 or 75 < rsi <= 82 else 1
        macd_points = 8 if "expansion" in macd_status else 3 if "positive" in macd_status else 0
        momentum = clamp(rsi_points + macd_points + 4, 0, 20)
        volume_points = 8 if volume_status == "major expansion" else 6 if volume_status == "above average" else 3 if volume_status == "normal" else 0
        obv_points = 12 if "crossed" in obv_status else 9 if "above" in obv_status else 1
        obv_score = clamp(volume_points + obv_points, 0, 20)
        rs = 10 if "BTC and ETH" in relative_strength else 6 if "one benchmark" in relative_strength else 1
        rr_score = 10 if rr >= 1.5 else 4 if rr >= 1 else 0
        return ScoreBreakdown(liquidity, trend, momentum, obv_score, rs, rr_score)

    def _summary(self, trades: list[BacktestTrade]) -> dict[str, Any]:
        pnl = sum(trade.pnl_usdt for trade in trades)
        winners = [trade for trade in trades if trade.pnl_usdt > 0]
        losers = [trade for trade in trades if trade.pnl_usdt < 0]
        gross_win = sum(trade.pnl_usdt for trade in winners)
        gross_loss = abs(sum(trade.pnl_usdt for trade in losers))
        average_win = gross_win / len(winners) if winners else 0
        average_loss = gross_loss / len(losers) if losers else 0
        win_rate = len(winners) / len(trades) if trades else 0
        loss_rate = len(losers) / len(trades) if trades else 0
        equity = []
        running = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in trades:
            running += trade.pnl_usdt
            peak = max(peak, running)
            max_drawdown = min(max_drawdown, running - peak)
            equity.append(running)
        returns = [trade.r_multiple for trade in trades]
        mean_r = sum(returns) / len(returns) if returns else 0
        variance = sum((item - mean_r) ** 2 for item in returns) / len(returns) if returns else 0
        sharpe = mean_r / (variance ** 0.5) if variance > 0 else 0
        return {
            "strategy_version": STRATEGY_VERSION,
            "trades": len(trades),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate_percent": round((len(winners) / len(trades)) * 100, 2) if trades else 0,
            "average_win_usdt": round(average_win, 4),
            "average_loss_usdt": round(average_loss, 4),
            "expectancy_usdt": round((win_rate * average_win) - (loss_rate * average_loss), 4),
            "total_pnl_usdt": round(pnl, 4),
            "return_percent": round((pnl / self.account_size) * 100, 4) if self.account_size else 0,
            "average_r": round(sum(trade.r_multiple for trade in trades) / len(trades), 4) if trades else 0,
            "sharpe_simple": round(sharpe, 4),
            "max_drawdown_usdt": round(max_drawdown, 4),
            "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else None,
            "tp1_exits": len([trade for trade in trades if trade.outcome == "tp1"]),
            "stop_exits": len([trade for trade in trades if trade.outcome == "stop"]),
            "open_marks": len([trade for trade in trades if trade.outcome == "open_mark"]),
            "average_duration_hours": round(self._average_duration_hours(trades), 4),
            "by_coin": self._group_performance(trades, "symbol"),
            "by_market_regime": self._group_performance(trades, "market_regime"),
            "by_score_bucket": self._group_performance(trades, "score_bucket"),
            "by_signal_type": self._group_performance(trades, "signal_type"),
        }

    def _ts(self, row: pd.Series) -> str:
        value = row["timestamp"]
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def _slice_to(self, df: pd.DataFrame | None, timestamp: Any) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        return df[df["timestamp"] <= timestamp].copy()

    def _score_bucket(self, score: float) -> str:
        if score >= 90:
            return "90+"
        if score >= 80:
            return "80-90"
        if score >= 70:
            return "70-80"
        if score >= 60:
            return "60-70"
        return "<60"

    def _average_duration_hours(self, trades: list[BacktestTrade]) -> float:
        durations = []
        for trade in trades:
            try:
                start = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                durations.append((end - start).total_seconds() / 3600)
            except Exception:
                continue
        return sum(durations) / len(durations) if durations else 0

    def _group_performance(self, trades: list[BacktestTrade], field: str) -> dict[str, Any]:
        groups: dict[str, list[BacktestTrade]] = {}
        for trade in trades:
            key = str(getattr(trade, field) or "unknown")
            groups.setdefault(key, []).append(trade)
        return {key: self._compact_group_summary(rows) for key, rows in sorted(groups.items())}

    def _compact_group_summary(self, trades: list[BacktestTrade]) -> dict[str, Any]:
        winners = [trade for trade in trades if trade.pnl_usdt > 0]
        losers = [trade for trade in trades if trade.pnl_usdt < 0]
        gross_win = sum(trade.pnl_usdt for trade in winners)
        gross_loss = abs(sum(trade.pnl_usdt for trade in losers))
        avg_win = gross_win / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        win_rate = len(winners) / len(trades) if trades else 0
        loss_rate = len(losers) / len(trades) if trades else 0
        return {
            "trades": len(trades),
            "win_rate_percent": round(win_rate * 100, 2),
            "total_pnl_usdt": round(sum(trade.pnl_usdt for trade in trades), 4),
            "expectancy_usdt": round((win_rate * avg_win) - (loss_rate * avg_loss), 4),
            "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else None,
            "average_r": round(sum(trade.r_multiple for trade in trades) / len(trades), 4) if trades else 0,
        }


def write_outputs(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"backtest_{stamp}.json"
    csv_path = out_dir / f"backtest_trades_{stamp}.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["trades"][0].keys()) if result["trades"] else ["symbol"])
        writer.writeheader()
        writer.writerows(result["trades"])
    return json_path, csv_path


def normalize_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    aliases = {
        "RIPPLE": "XRP",
        "XRP": "XRP",
        "SOLANA": "SOL",
        "SOLANO": "SOL",
        "SOL": "SOL",
    }
    symbols: list[str] = []
    for raw in value.split(","):
        token = raw.strip().upper()
        if not token:
            continue
        if "/" in token:
            symbols.append(token)
        else:
            symbols.append(f"{aliases.get(token, token)}/USDT")
    return symbols


def bars_for_days(days: float, timeframe: str) -> int:
    units = {"m": 24 * 60, "h": 24, "d": 1}
    suffix = timeframe[-1]
    amount = int(timeframe[:-1])
    per_day = units[suffix] / amount
    return int(days * per_day)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Binance momentum strategy.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--bars", type=int, default=720)
    parser.add_argument("--days", type=float, default=None, help="Lookback days. Overrides --bars using 4H candles.")
    parser.add_argument("--days-list", default=None, help="Comma-separated lookback days, e.g. 3,7,30")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--timeframes", default=None, help="Comma-separated timeframes, e.g. 1h,4h,1d")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols/bases, e.g. ZEC,FET,XRP,SOL,TAO")
    parser.add_argument("--min-score", type=float, default=70)
    parser.add_argument("--account-size", type=float, default=1000)
    parser.add_argument("--risk-percent", type=float, default=1)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--scalp-mode", action="store_true")
    parser.add_argument("--scalp-stop-atr", type=float, default=0.6)
    parser.add_argument("--scalp-tp1-r", type=float, default=0.6)
    parser.add_argument("--scalp-tp2-r", type=float, default=1.1)
    parser.add_argument("--scalp-max-hold-minutes", type=int, default=90)
    parser.add_argument("--scalp-min-risk-pct", type=float, default=0.0015)
    parser.add_argument("--out", type=Path, default=Path("data/backtests"))
    args = parser.parse_args()

    day_values = [float(item.strip()) for item in args.days_list.split(",")] if args.days_list else [args.days]
    timeframes = [item.strip() for item in args.timeframes.split(",")] if args.timeframes else [args.timeframe]
    requested_symbols = normalize_symbols(args.symbols)
    results = []

    for timeframe in timeframes:
        for days in day_values:
            bars = bars_for_days(days, timeframe) if days is not None else args.bars
            engine = BacktestEngine(
                BinanceMarketData(get_settings()),
                universe_size=args.top,
                bars=bars,
                min_score=args.min_score,
                account_size=args.account_size,
                risk_percent=args.risk_percent,
                fee_bps=args.fee_bps,
                timeframe=timeframe,
                symbols=requested_symbols,
                scalp_mode=args.scalp_mode,
                scalp_stop_atr=args.scalp_stop_atr,
                scalp_tp1_r=args.scalp_tp1_r,
                scalp_tp2_r=args.scalp_tp2_r,
                scalp_max_hold_minutes=args.scalp_max_hold_minutes,
                scalp_min_risk_pct=args.scalp_min_risk_pct,
            )
            result = engine.run()
            json_path, csv_path = write_outputs(result, args.out)
            results.append(
                {
                    "timeframe": timeframe,
                    "days": days,
                    "summary": result["summary"],
                    "symbols": result["symbols"],
                    "errors": result["errors"],
                    "json": str(json_path),
                    "csv": str(csv_path),
                }
            )
    print(json.dumps(results[0] if len(results) == 1 else {"runs": results}, indent=2))


if __name__ == "__main__":
    main()
