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
    score: float
    tier: str
    outcome: str
    pnl_percent: float
    pnl_usdt: float
    r_multiple: float
    reason: str


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
    ):
        self.market_data = market_data
        self.universe_size = universe_size
        self.bars = bars
        self.min_score = min_score
        self.account_size = account_size
        self.risk_percent = risk_percent
        self.fee_bps = fee_bps

    def run(self) -> dict[str, Any]:
        universe = self.market_data.load_top_usdt_pairs(self.universe_size)
        btc = enrich(self.market_data.fetch_ohlcv("BTC/USDT", "4h", limit=self.bars + 100))
        eth = enrich(self.market_data.fetch_ohlcv("ETH/USDT", "4h", limit=self.bars + 100))
        trades: list[BacktestTrade] = []
        errors: list[str] = []

        for rank, ticker in enumerate(universe, start=1):
            try:
                df = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "4h", limit=self.bars + 100))
                trades.extend(self._run_symbol(ticker, rank, df, btc, eth))
            except Exception as exc:
                errors.append(f"{ticker.symbol}: {exc}")

        trades.sort(key=lambda trade: trade.entry_time)
        summary = self._summary(trades)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assumptions": {
                "universe": "Current top Binance spot USDT pairs by quote volume, not historical constituents",
                "universe_size": self.universe_size,
                "timeframe": "4h",
                "bars_requested": self.bars,
                "min_score": self.min_score,
                "entry": "Next 4H candle open after signal",
                "exit": "Full size exits at TP1 or stop loss; if both touch same candle, stop is assumed first",
                "fees": f"{self.fee_bps} bps per side",
            },
            "summary": summary,
            "symbols": [ticker.symbol for ticker in universe],
            "errors": errors,
            "trades": [asdict(trade) for trade in trades],
        }

    def _run_symbol(
        self,
        ticker: MarketTicker,
        volume_rank: int,
        df: pd.DataFrame,
        btc: pd.DataFrame,
        eth: pd.DataFrame,
    ) -> list[BacktestTrade]:
        trades: list[BacktestTrade] = []
        warmup = 80
        i = warmup
        max_i = len(df) - 2

        while i < max_i:
            signal = self._signal_at(ticker.symbol, volume_rank, df, btc, eth, i)
            if signal["score"] < self.min_score:
                i += 1
                continue

            entry_index = i + 1
            entry_price = float(df.iloc[entry_index]["open"])
            stop_loss = float(signal["stop_loss"])
            take_profit_1 = float(signal["take_profit_1"])

            if not (stop_loss < entry_price < take_profit_1):
                i += 1
                continue

            exit_index, exit_price, outcome = self._find_exit(df, entry_index, stop_loss, take_profit_1)
            risk_per_unit = entry_price - stop_loss
            risk_usdt = self.account_size * (self.risk_percent / 100)
            quantity = risk_usdt / risk_per_unit if risk_per_unit > 0 else 0
            gross_pnl = (exit_price - entry_price) * quantity
            fee = (entry_price + exit_price) * quantity * (self.fee_bps / 10000)
            pnl_usdt = gross_pnl - fee
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100 - (self.fee_bps * 2 / 100)
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
                    score=signal["score"],
                    tier=classify(signal["score"]),
                    outcome=outcome,
                    pnl_percent=round(pnl_percent, 4),
                    pnl_usdt=round(pnl_usdt, 4),
                    r_multiple=round(r_multiple, 4),
                    reason=signal["reason"],
                )
            )
            i = exit_index + 1

        return trades

    def _signal_at(
        self,
        symbol: str,
        volume_rank: int,
        df: pd.DataFrame,
        btc: pd.DataFrame,
        eth: pd.DataFrame,
        i: int,
    ) -> dict[str, Any]:
        current = df.iloc[: i + 1]
        latest = current.iloc[-1]
        prev = current.iloc[-2]
        trend_status = self._trend_status(current)
        volume_status = self._volume_status(latest)
        obv_status = self._obv_status(latest, prev)
        macd_status = self._macd_status(latest, prev)
        relative_strength = self._relative_strength(current, btc.iloc[: i + 1], eth.iloc[: i + 1])
        stop_loss, tp1, rr = self._risk_plan(current)
        score = self._score(volume_rank, latest, trend_status, volume_status, obv_status, macd_status, relative_strength, rr)
        return {
            "symbol": symbol,
            "score": score.total,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "reason": "; ".join([trend_status, volume_status, obv_status, macd_status, relative_strength]),
        }

    def _find_exit(self, df: pd.DataFrame, entry_index: int, stop_loss: float, take_profit_1: float) -> tuple[int, float, str]:
        for j in range(entry_index, len(df)):
            row = df.iloc[j]
            hit_stop = float(row["low"]) <= stop_loss
            hit_tp = float(row["high"]) >= take_profit_1
            if hit_stop:
                return j, stop_loss, "stop"
            if hit_tp:
                return j, take_profit_1, "tp1"
        last_index = len(df) - 1
        return last_index, float(df.iloc[last_index]["close"]), "open_mark"

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

    def _risk_plan(self, df: pd.DataFrame) -> tuple[float, float, float]:
        latest = df.iloc[-1]
        price = float(latest["close"])
        atr = float(latest["atr14"])
        recent_low = float(df.tail(10)["low"].min())
        raw_stop = min(price - atr * 1.2, recent_low * 0.995)
        stop_loss = round(max(raw_stop, price * 0.7), 8)
        risk = max(price - stop_loss, price * 0.005)
        tp1 = round(price + risk * 1.5, 8)
        rr = round((tp1 - price) / risk, 2) if risk > 0 else 0
        return stop_loss, tp1, rr

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
        return {
            "trades": len(trades),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate_percent": round((len(winners) / len(trades)) * 100, 2) if trades else 0,
            "total_pnl_usdt": round(pnl, 4),
            "return_percent": round((pnl / self.account_size) * 100, 4) if self.account_size else 0,
            "average_r": round(sum(trade.r_multiple for trade in trades) / len(trades), 4) if trades else 0,
            "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else None,
            "tp1_exits": len([trade for trade in trades if trade.outcome == "tp1"]),
            "stop_exits": len([trade for trade in trades if trade.outcome == "stop"]),
            "open_marks": len([trade for trade in trades if trade.outcome == "open_mark"]),
        }

    def _ts(self, row: pd.Series) -> str:
        value = row["timestamp"]
        return value.isoformat() if hasattr(value, "isoformat") else str(value)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Binance momentum strategy.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--bars", type=int, default=720)
    parser.add_argument("--days", type=float, default=None, help="Lookback days. Overrides --bars using 4H candles.")
    parser.add_argument("--min-score", type=float, default=70)
    parser.add_argument("--account-size", type=float, default=1000)
    parser.add_argument("--risk-percent", type=float, default=1)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/backtests"))
    args = parser.parse_args()

    bars = int(args.days * 6) if args.days is not None else args.bars

    engine = BacktestEngine(
        BinanceMarketData(get_settings()),
        universe_size=args.top,
        bars=bars,
        min_score=args.min_score,
        account_size=args.account_size,
        risk_percent=args.risk_percent,
        fee_bps=args.fee_bps,
    )
    result = engine.run()
    json_path, csv_path = write_outputs(result, args.out)
    print(json.dumps({"summary": result["summary"], "symbols": result["symbols"], "errors": result["errors"], "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
