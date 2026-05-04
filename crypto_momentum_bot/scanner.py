from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .db import Database
from .indicators import enrich
from .live_execution import GatedLiveExecutor
from .market_data import BinanceMarketData, MarketTicker
from .quant import calculate_ev, parse_json_dict, score_breakdown_from_notes, score_breakdown_to_dict, setup_type_from_note
from .scoring import ScoreBreakdown, classify, clamp

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MomentumScanner:
    def __init__(self, db: Database, market_data: BinanceMarketData):
        self.db = db
        self.market_data = market_data
        self.live_executor = GatedLiveExecutor(db, market_data)

    def scan(self) -> dict[str, Any]:
        started = now_iso()
        errors: list[str] = []
        saved = 0
        try:
            tickers = self.market_data.load_top_usdt_pairs(40)
            btc_4h = enrich(self.market_data.fetch_ohlcv("BTC/USDT", "4h"))
            eth_4h = enrich(self.market_data.fetch_ohlcv("ETH/USDT", "4h"))
            btc_5m = enrich(self.market_data.fetch_ohlcv("BTC/USDT", "5m"))
            eth_5m = enrich(self.market_data.fetch_ohlcv("ETH/USDT", "5m"))
            btc_trend = self._trend_label(btc_4h)
            eth_trend = self._trend_label(eth_4h)
            market_regime = self._market_regime(btc_trend, eth_trend)

            for rank, ticker in enumerate(tickers, start=1):
                try:
                    signal = self._build_signal(ticker, rank, btc_5m, eth_5m)
                    signal_id = self._insert_signal(signal)
                    saved += 1
                    self._maybe_create_live_order_intent(signal_id, signal)
                    self._maybe_create_paper_trade(signal_id, signal)
                except Exception as exc:
                    message = f"{ticker.symbol}: {exc}"
                    logger.exception("Signal build failed for %s", ticker.symbol)
                    errors.append(message)

            self.db.execute(
                """
                UPDATE bot_status
                SET last_scan_time = ?, coins_scanned = ?, market_regime = ?,
                    btc_trend = ?, eth_trend = ?, errors = ?
                WHERE id = 1
                """,
                (started, len(tickers), market_regime, btc_trend, eth_trend, "\n".join(errors)),
            )
            self.log("INFO", f"Scan completed: {saved} signals stored", {"errors": errors[:5]})
            return {"saved": saved, "coins_scanned": len(tickers), "errors": errors}
        except Exception as exc:
            logger.exception("Scan failed")
            self.db.execute("UPDATE bot_status SET errors = ? WHERE id = 1", (str(exc),))
            self.log("ERROR", "Scan failed", {"error": str(exc)})
            raise

    def _build_signal(
        self,
        ticker: MarketTicker,
        volume_rank: int,
        btc_5m: pd.DataFrame,
        eth_5m: pd.DataFrame,
    ) -> dict[str, Any]:
        daily = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "1d"))
        four_h = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "4h"))
        one_h = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "1h"))
        five_m = enrich(self.market_data.fetch_ohlcv(ticker.symbol, "5m"))

        latest_5m = five_m.iloc[-1]
        prev_5m = five_m.iloc[-2]
        latest_1h = one_h.iloc[-1]
        prev_1h = one_h.iloc[-2]
        price = float(latest_5m["close"])

        trend_status = self._trend_status(daily, four_h, one_h, five_m)
        volume_status = self._volume_status(latest_5m)
        obv_status = self._obv_status(latest_5m, prev_5m)
        macd_status = self._macd_status(latest_5m, prev_5m, latest_1h, prev_1h)
        relative_strength = self._relative_strength(five_m, btc_5m, eth_5m)
        structure = self._structure_note(five_m)
        entry_zone, stop_loss, tp1, tp2, rr = self._risk_plan(five_m)
        breakdown = self._score(volume_rank, latest_5m, latest_1h, trend_status, volume_status, obv_status, macd_status, relative_strength, rr)
        breakdown_json = score_breakdown_to_dict(breakdown)
        setup_type = setup_type_from_note(structure)
        volume_ratio = round(float(latest_5m["volume"] / latest_5m["vol_ma20"]), 4) if latest_5m["vol_ma20"] else 0
        notes = "; ".join(["5m momentum swing profile", structure, f"Score parts {breakdown}"])

        return {
            "timestamp": now_iso(),
            "symbol": ticker.symbol,
            "volume_rank": volume_rank,
            "price": price,
            "score": breakdown.total,
            "tier": classify(breakdown.total),
            "trend_status": trend_status,
            "volume_status": volume_status,
            "obv_status": obv_status,
            "rsi": round(float(latest_5m["rsi14"]), 2),
            "macd_status": macd_status,
            "relative_strength": relative_strength,
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "risk_reward": rr,
            "notes": notes,
            "score_breakdown_json": json.dumps(breakdown_json),
            "setup_type": setup_type,
            "volume_ratio": volume_ratio,
        }

    def _trend_label(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        if latest["close"] > latest["ema20"] > latest["ema50"]:
            return "bullish"
        if latest["close"] < latest["ema20"] < latest["ema50"]:
            return "bearish"
        return "mixed"

    def _market_regime(self, btc_trend: str, eth_trend: str) -> str:
        if btc_trend == "bullish" and eth_trend == "bullish":
            return "risk-on"
        if btc_trend == "bearish" and eth_trend == "bearish":
            return "risk-off"
        return "selective / mixed"

    def _trend_status(self, daily: pd.DataFrame, four_h: pd.DataFrame, one_h: pd.DataFrame, five_m: pd.DataFrame) -> str:
        d = self._trend_label(daily)
        h4 = self._trend_label(four_h)
        h1 = self._trend_label(one_h)
        m5 = self._trend_label(five_m)
        if m5 == "bullish" and h1 == "bullish" and h4 in {"bullish", "mixed"} and d in {"bullish", "mixed"}:
            return "5m bullish, 1H aligned"
        if m5 == "bullish" and h1 in {"bullish", "mixed"}:
            return "5m bullish, 1H mixed"
        if m5 == "mixed":
            return "mixed / basing"
        return "downtrend"

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
        crossed = prev["obv"] <= prev["obv_ma7"] and latest["obv"] > latest["obv_ma7"]
        if crossed:
            return "crossed above OBV MA(7)"
        if latest["obv"] > latest["obv_ma7"]:
            return "above OBV MA(7)"
        return "below OBV MA(7)"

    def _macd_status(self, latest_4h: pd.Series, prev_4h: pd.Series, latest_1h: pd.Series, prev_1h: pd.Series) -> str:
        expanding_4h = latest_4h["macd_hist"] > 0 and latest_4h["macd_hist"] > prev_4h["macd_hist"]
        expanding_1h = latest_1h["macd_hist"] > 0 and latest_1h["macd_hist"] > prev_1h["macd_hist"]
        if expanding_4h and expanding_1h:
            return "positive expansion on 5m and 1H"
        if expanding_4h:
            return "positive expansion on 5m"
        if latest_4h["macd_hist"] > 0:
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

    def _structure_note(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        recent = df.tail(20)
        high_20 = recent["high"].max()
        compression = recent["range_pct"].tail(6).mean() < recent["range_pct"].mean() * 0.75
        if latest["close"] >= high_20 * 0.995:
            return "breakout near 20-candle high"
        if latest["close"] > latest["ema20"] and compression:
            return "compression above EMA20"
        if latest["low"] <= latest["ema20"] <= latest["close"]:
            return "EMA20 reclaim"
        return "structure developing"

    def _risk_plan(self, df: pd.DataFrame) -> tuple[str, float, float, float, float]:
        latest = df.iloc[-1]
        price = float(latest["close"])
        atr = float(latest["atr14"])
        recent_low = float(df.tail(10)["low"].min())
        raw_stop = min(price - atr * 1.2, recent_low * 0.995)
        stop_loss = round(max(raw_stop, price * 0.7), 8)
        risk = max(price - stop_loss, price * 0.005)
        tp1 = round(price + risk * 1.5, 8)
        tp2 = round(price + risk * 2.5, 8)
        rr = round((tp2 - price) / risk, 2) if risk > 0 else 0
        entry_low = round(price * 0.995, 8)
        entry_high = round(price * 1.005, 8)
        return f"{entry_low} - {entry_high}", stop_loss, tp1, tp2, rr

    def _score(
        self,
        volume_rank: int,
        latest_4h: pd.Series,
        latest_1h: pd.Series,
        trend_status: str,
        volume_status: str,
        obv_status: str,
        macd_status: str,
        relative_strength: str,
        rr: float,
    ) -> ScoreBreakdown:
        liquidity = clamp(21 - (volume_rank * 0.5), 4, 20)
        trend = 20 if "5m bullish, 1H aligned" in trend_status else 14 if "5m bullish" in trend_status else 9 if "basing" in trend_status else 2
        rsi = float(latest_4h["rsi14"])
        rsi_points = 8 if 50 <= rsi <= 75 else 4 if 45 <= rsi < 50 or 75 < rsi <= 82 else 1
        macd_points = 8 if "5m and 1H" in macd_status else 6 if "5m" in macd_status else 3 if "positive" in macd_status else 0
        one_h_points = 4 if latest_1h["close"] > latest_1h["ema20"] else 0
        momentum = clamp(rsi_points + macd_points + one_h_points, 0, 20)
        volume_points = 8 if volume_status == "major expansion" else 6 if volume_status == "above average" else 3 if volume_status == "normal" else 0
        obv_points = 12 if "crossed" in obv_status else 9 if "above" in obv_status else 1
        obv_score = clamp(volume_points + obv_points, 0, 20)
        rs = 10 if "BTC and ETH" in relative_strength else 6 if "one benchmark" in relative_strength else 1
        rr_score = 10 if rr >= 2.2 else 7 if rr >= 1.5 else 3 if rr >= 1 else 0
        return ScoreBreakdown(liquidity, trend, momentum, obv_score, rs, rr_score)

    def _insert_signal(self, signal: dict[str, Any]) -> int:
        return self.db.execute(
            """
            INSERT INTO signals (
                timestamp, symbol, volume_rank, price, score, tier, trend_status,
                volume_status, obv_status, rsi, macd_status, relative_strength,
                entry_zone, stop_loss, take_profit_1, take_profit_2, risk_reward, notes,
                score_breakdown_json, ev_json, setup_type, volume_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal["timestamp"],
                signal["symbol"],
                signal["volume_rank"],
                signal["price"],
                signal["score"],
                signal["tier"],
                signal["trend_status"],
                signal["volume_status"],
                signal["obv_status"],
                signal["rsi"],
                signal["macd_status"],
                signal["relative_strength"],
                signal["entry_zone"],
                signal["stop_loss"],
                signal["take_profit_1"],
                signal["take_profit_2"],
                signal["risk_reward"],
                signal["notes"],
                signal.get("score_breakdown_json", "{}"),
                signal.get("ev_json", "{}"),
                signal.get("setup_type", ""),
                signal.get("volume_ratio"),
            ),
        )

    def _maybe_create_paper_trade(self, signal_id: int, signal: dict[str, Any]) -> None:
        status = self.db.fetch_one("SELECT mode FROM bot_status WHERE id = 1") or {}
        if status.get("mode") != "paper-trading":
            return
        if signal["score"] < 70:
            return
        existing = self.db.fetch_one(
            "SELECT id FROM trades WHERE symbol = ? AND status = 'open' LIMIT 1",
            (signal["symbol"],),
        )
        if existing:
            return
        settings = self.db.fetch_one("SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1") or {}
        account_size = float(settings.get("account_size") or 0)
        risk_pct = min(float(settings.get("risk_per_trade_percent") or 1), 10)
        risk_usdt = account_size * (risk_pct / 100)
        stop_distance = max(float(signal["price"]) - float(signal["stop_loss"]), signal["price"] * 0.005)
        risk_size = risk_usdt / stop_distance if stop_distance > 0 else 0
        max_size = account_size / float(signal["price"]) if signal["price"] else 0
        position_size = round(min(risk_size, max_size), 8)
        self.db.execute(
            """
            INSERT INTO trades (
                signal_id, symbol, entry_time, entry_price, stop_loss, take_profit_1,
                take_profit_2, status, lifecycle_state, entry_reason, score_at_entry, position_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 'bought', ?, ?, ?)
            """,
            (
                signal_id,
                signal["symbol"],
                now_iso(),
                signal["price"],
                signal["stop_loss"],
                signal["take_profit_1"],
                signal["take_profit_2"],
                f"Paper entry from {signal['tier']}: {signal['notes'][:180]}",
                signal["score"],
                position_size,
            ),
        )
        self.log("INFO", f"Paper trade opened for {signal['symbol']}", signal)

    def _maybe_create_live_order_intent(self, signal_id: int, signal: dict[str, Any]) -> None:
        if signal["score"] < 70:
            return
        existing = self.db.fetch_one(
            """
            SELECT id, lifecycle_state, status, seen_count FROM live_order_intents
            WHERE symbol = ?
              AND lifecycle_state NOT IN ('ignored', 'executed', 'archived')
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (signal["symbol"],),
        )
        if existing:
            lifecycle = "stale" if existing.get("lifecycle_state") == "stale" else "seen"
            self.db.execute(
                """
                UPDATE live_order_intents
                SET signal_id = ?,
                    last_seen_at = ?,
                    scan_seen_at = ?,
                    seen_count = COALESCE(seen_count, 1) + 1,
                    lifecycle_state = ?,
                    reference_price = ?,
                    stop_loss = ?,
                    take_profit_1 = ?,
                    take_profit_2 = ?
                WHERE id = ?
                """,
                (
                    signal_id,
                    now_iso(),
                    signal["timestamp"],
                    lifecycle,
                    signal["price"],
                    signal["stop_loss"],
                    signal["take_profit_1"],
                    signal["take_profit_2"],
                    existing["id"],
                ),
            )
            return
        open_trade = self.db.fetch_one(
            """
            SELECT id FROM trades
            WHERE symbol = ? AND status = 'open' AND execution_type = 'live'
            LIMIT 1
            """,
            (signal["symbol"],),
        )
        if open_trade:
            return
        signal_with_id = dict(signal)
        signal_with_id["id"] = signal_id
        try:
            intent = self.live_executor.build_intent_from_signal(signal_with_id)
            validation = self.live_executor.validate_intent(intent)
            status = self.db.fetch_one("SELECT market_regime FROM bot_status WHERE id = 1") or {}
            settings = self.db.fetch_one("SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1") or {}
            ev = calculate_ev(
                signal,
                float(settings.get("account_size") or 0),
                float(settings.get("risk_per_trade_percent") or 0),
                str(status.get("market_regime") or "unknown"),
                quote_amount=intent.quote_amount,
            )
        except Exception as exc:
            self.log("WARNING", f"Live intent validation failed for {signal['symbol']}", {"error": str(exc)})
            return
        self.db.execute(
            """
            INSERT INTO live_order_intents (
                timestamp, signal_id, symbol, side, order_type, quote_amount,
                base_quantity, reference_price, stop_loss, take_profit_1,
                take_profit_2, status, lifecycle_state, first_seen_at, last_seen_at,
                scan_seen_at, seen_count, validation_json, reason, ev_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', 'new', ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                now_iso(),
                signal_id,
                intent.symbol,
                intent.side.value,
                intent.order_type.value,
                intent.quote_amount,
                intent.base_quantity,
                intent.reference_price,
                intent.stop_loss,
                intent.take_profit_1,
                intent.take_profit_2,
                signal["timestamp"],
                signal["timestamp"],
                signal["timestamp"],
                json.dumps(validation, default=str),
                intent.reason,
                json.dumps(ev, default=str),
            ),
        )
        self.log("INFO", f"Draft live order intent created for {signal['symbol']}", {"validation": validation})

    def get_live_order_intents(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM live_order_intents ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return self._enrich_live_intents(rows)

    def get_live_order_intent_detail(self, intent_id: int) -> dict[str, Any]:
        intent = self.db.fetch_one("SELECT * FROM live_order_intents WHERE id = ?", (intent_id,))
        if not intent:
            raise ValueError("Live order intent not found")
        enriched = self._enrich_live_intents([intent])[0]
        signal = None
        if intent.get("signal_id"):
            signal = self.db.fetch_one("SELECT * FROM signals WHERE id = ?", (intent["signal_id"],))
            if signal:
                signal = self._enrich_signal_quant(signal)
        logs = self.db.fetch_all(
            """
            SELECT * FROM event_log
            WHERE message LIKE ? OR context LIKE ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (f"%{intent_id}%", f"%{intent_id}%"),
        )
        return {"intent": enriched, "signal": signal, "event_log": logs}

    def get_quant_snapshot(self) -> dict[str, Any]:
        self.recover_stuck_submitting_intents()
        self.mark_stale_intents()
        db_status = self.db.fetch_one("SELECT * FROM bot_status WHERE id = 1") or {}
        settings = self.db.fetch_one("SELECT * FROM settings WHERE id = 1") or {}
        signals = [self._enrich_signal_quant(row) for row in self._latest_signals(limit=80)]
        intents = self.get_live_order_intents(limit=100)
        lifecycle_counts: dict[str, int] = {}
        for intent in intents:
            state = str(intent.get("status") or "unknown")
            lifecycle_counts[state] = lifecycle_counts.get(state, 0) + 1
        return {
            "bot_status": db_status,
            "settings": settings,
            "live_gates": {
                "database_enabled": bool(settings.get("live_trading_enabled")),
                "env_enabled": bool(self.market_data.settings.enable_binance_live_trading),
                "legacy_unlock": bool(self.market_data.settings.live_trading_unlocked),
                "ready_for_submission": bool(
                    settings.get("live_trading_enabled")
                    and self.market_data.settings.enable_binance_live_trading
                    and self.market_data.settings.live_trading_unlocked
                ),
            },
            "lifecycle_counts": lifecycle_counts,
            "signals": signals,
            "live_intents": intents,
            "market_regime": db_status.get("market_regime", "unknown"),
            "last_refreshed_at": now_iso(),
        }

    def get_dashboard_lifecycle(self) -> dict[str, Any]:
        self.mark_stale_intents()
        refresh = self.refresh_live_portfolio_state()
        trades = self.get_trade_history(limit=500)
        active_states = {"open", "submitted", "bought", "exit_strategy_pending", "exit_strategy_active", "partial", "partially_closed"}
        active = [
            trade for trade in trades
            if trade.get("execution_type") == "live"
            and (trade.get("status") in {"open", "partial", "partially_closed"} or trade.get("lifecycle_state") in active_states)
        ]
        issues = [
            trade for trade in active
            if trade.get("exit_strategy_state") in {"pending", "failed"}
            or trade.get("exit_order_status") in {"failed", "unprotected", ""}
        ]
        closed = [
            trade for trade in trades
            if trade.get("status") in {"closed", "manually_closed", "cancelled", "failed"}
            or trade.get("lifecycle_state") in {"closed", "manually_closed", "cancelled", "failed"}
        ][:100]
        intent_rows = self.get_live_order_intents(limit=300)
        new_intents = [row for row in intent_rows if row.get("lifecycle_state") == "new" and row.get("status") == "draft"]
        seen_intents = [
            row for row in intent_rows
            if row.get("lifecycle_state") in {"seen", "stale"} and row.get("status") in {"draft", "approved", "failed"}
        ]
        ignored_intents = [
            row for row in intent_rows
            if row.get("lifecycle_state") in {"ignored", "archived"} or row.get("status") == "rejected"
        ][:100]
        executed_intents = [row for row in intent_rows if row.get("lifecycle_state") == "executed"][:100]
        errors = list(refresh.get("errors") or [])
        status = self.db.fetch_one("SELECT errors FROM bot_status WHERE id = 1") or {}
        if status.get("errors"):
            errors.append(status["errors"])
        strategy_summary = self._strategy_performance_summary(active, closed, issues)
        return {
            "last_refreshed_at": now_iso(),
            "refresh": refresh,
            "errors": errors[:8],
            "summary": strategy_summary,
            "active_trades": active,
            "exit_issues": issues,
            "closed_trades": closed,
            "intents": {
                "new": new_intents,
                "seen": seen_intents,
                "ignored": ignored_intents,
                "executed": executed_intents,
            },
            "recent_events": self.db.fetch_all("SELECT * FROM trade_events ORDER BY timestamp DESC LIMIT 80"),
        }

    def _strategy_performance_summary(
        self,
        active: list[dict[str, Any]],
        closed: list[dict[str, Any]],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        closed_pnl = sum(float(trade.get("pnl_usdt") or 0) for trade in closed)
        active_pnl = sum(float(trade.get("pnl_usdt") or 0) for trade in active)
        active_exposure = sum(
            float(trade.get("current_value_usdt") or trade.get("entry_value_usdt") or 0)
            for trade in active
        )
        winners = [float(trade.get("pnl_usdt") or 0) for trade in closed if float(trade.get("pnl_usdt") or 0) > 0]
        losers = [float(trade.get("pnl_usdt") or 0) for trade in closed if float(trade.get("pnl_usdt") or 0) < 0]
        closed_count = len(closed)
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        r_values = [self._trade_r_multiple(trade) for trade in closed + active]
        closed_r_values = [self._trade_r_multiple(trade) for trade in closed]
        open_risk = sum(self._open_risk_usdt(trade) for trade in active)
        return {
            "global_pnl_usdt": round(closed_pnl + active_pnl, 4),
            "realized_pnl_usdt": round(closed_pnl, 4),
            "unrealized_pnl_usdt": round(active_pnl, 4),
            "active_pnl_usdt": round(active_pnl, 4),
            "closed_pnl_usdt": round(closed_pnl, 4),
            "in_trade_amount_usdt": round(active_exposure, 4),
            "open_risk_usdt": round(open_risk, 4),
            "closed_trades": closed_count,
            "open_trades": len(active),
            "exit_issue_count": len(issues),
            "win_rate_percent": round((len(winners) / closed_count) * 100, 2) if closed_count else 0,
            "average_winner_usdt": round(gross_profit / len(winners), 4) if winners else 0,
            "average_loser_usdt": round(sum(losers) / len(losers), 4) if losers else 0,
            "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else (round(gross_profit, 4) if gross_profit else 0),
            "average_r": round(sum(r_values) / len(r_values), 4) if r_values else 0,
            "closed_average_r": round(sum(closed_r_values) / len(closed_r_values), 4) if closed_r_values else 0,
            "expectancy_usdt": round(closed_pnl / closed_count, 4) if closed_count else 0,
            "expectancy_r": round(sum(closed_r_values) / len(closed_r_values), 4) if closed_r_values else 0,
        }

    def _trade_initial_risk_usdt(self, trade: dict[str, Any]) -> float:
        entry = float(trade.get("entry_price") or 0)
        stop = float(trade.get("stop_loss") or 0)
        size = float(trade.get("position_size") or 0)
        risk_per_unit = max(entry - stop, 0)
        return risk_per_unit * size

    def _trade_r_multiple(self, trade: dict[str, Any]) -> float:
        initial_risk = self._trade_initial_risk_usdt(trade)
        if initial_risk <= 0:
            return 0.0
        return float(trade.get("pnl_usdt") or 0) / initial_risk

    def _open_risk_usdt(self, trade: dict[str, Any]) -> float:
        current = float(trade.get("current_price") or trade.get("entry_price") or 0)
        stop = float(trade.get("stop_loss") or 0)
        remaining = float(trade.get("remaining_position_size") or trade.get("position_size") or 0)
        if current <= 0 or stop <= 0 or remaining <= 0:
            return 0.0
        return max(current - stop, 0) * remaining

    def _enrich_signal_quant(self, signal: dict[str, Any]) -> dict[str, Any]:
        item = dict(signal)
        status = self.db.fetch_one("SELECT market_regime FROM bot_status WHERE id = 1") or {}
        settings = self.db.fetch_one("SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1") or {}
        ev = parse_json_dict(item.get("ev_json"))
        if not ev or not ev.get("ok"):
            ev = calculate_ev(
                item,
                float(settings.get("account_size") or 0),
                float(settings.get("risk_per_trade_percent") or 0),
                str(status.get("market_regime") or "unknown"),
            )
        breakdown = parse_json_dict(item.get("score_breakdown_json"))
        if not breakdown:
            breakdown = score_breakdown_from_notes(str(item.get("notes") or ""))
        item["score_breakdown"] = breakdown
        item["ev"] = ev
        item["setup_type"] = item.get("setup_type") or setup_type_from_note(str(item.get("notes") or ""))
        item["quote_amount"] = ev.get("quote_amount")
        item["risk_usdt"] = ev.get("risk_usdt")
        item["expected_value_r"] = ev.get("expected_value_r")
        item["expected_value_usdt"] = ev.get("expected_value_usdt")
        item["win_probability"] = ev.get("win_probability")
        return item

    def _enrich_live_intents(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prices: dict[str, float] = {}
        if rows:
            try:
                prices = self.market_data.fetch_last_prices(sorted({row["symbol"] for row in rows}))
            except Exception as exc:
                logger.warning("Could not fetch live prices for live intents: %s", exc)

        status = self.db.fetch_one("SELECT market_regime FROM bot_status WHERE id = 1") or {}
        settings = self.db.fetch_one("SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1") or {}
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            validation = parse_json_dict(item.get("validation_json"))
            signal = self.db.fetch_one("SELECT * FROM signals WHERE id = ?", (item.get("signal_id"),)) if item.get("signal_id") else None
            signal_like = dict(signal or {})
            signal_like.update(
                {
                    "symbol": item.get("symbol"),
                    "price": item.get("reference_price"),
                    "reference_price": item.get("reference_price"),
                    "stop_loss": item.get("stop_loss"),
                    "take_profit_1": item.get("take_profit_1"),
                    "take_profit_2": item.get("take_profit_2"),
                    "score": signal_like.get("score") or 0,
                    "tier": signal_like.get("tier") or "",
                }
            )
            ev = parse_json_dict(item.get("ev_json"))
            if not ev or not ev.get("ok"):
                ev = calculate_ev(
                    signal_like,
                    float(settings.get("account_size") or 0),
                    float(settings.get("risk_per_trade_percent") or 0),
                    str(status.get("market_regime") or "unknown"),
                    quote_amount=float(item.get("quote_amount") or 0),
                )
            current_price = prices.get(str(item.get("symbol")), item.get("reference_price"))
            reference = float(item.get("reference_price") or 0)
            drift_pct = ((float(current_price) - reference) / reference) * 100 if current_price and reference else 0
            item["validation"] = validation
            item["ev"] = ev
            item["current_price"] = round(float(current_price), 8) if current_price else None
            item["price_drift_percent"] = round(drift_pct, 4)
            item["is_stale"] = abs(drift_pct) >= 1.0
            item["current_score"] = signal_like.get("score")
            item["score_breakdown"] = parse_json_dict(signal_like.get("score_breakdown_json"))
            if not item["score_breakdown"]:
                item["score_breakdown"] = score_breakdown_from_notes(str(signal_like.get("notes") or ""))
            item["setup_type"] = signal_like.get("setup_type") or setup_type_from_note(str(signal_like.get("notes") or ""))
            item["risk_usdt"] = ev.get("risk_usdt")
            item["expected_value_r"] = ev.get("expected_value_r")
            item["expected_value_usdt"] = ev.get("expected_value_usdt")
            item["win_probability"] = ev.get("win_probability")
            enriched.append(item)
        return enriched

    def recover_stuck_submitting_intents(self, max_age_minutes: int = 5) -> int:
        rows = self.db.fetch_all("SELECT * FROM live_order_intents WHERE status = 'submitting'")
        recovered = 0
        for row in rows:
            try:
                timestamp = row.get("submitted_at") or row.get("timestamp")
                started = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                age_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
            except Exception:
                age_minutes = max_age_minutes + 1
            if age_minutes < max_age_minutes:
                continue
            existing_trade = self.db.fetch_one(
                "SELECT id FROM trades WHERE signal_id = ? AND symbol = ? AND execution_type = 'live' LIMIT 1",
                (row.get("signal_id"), row.get("symbol")),
            )
            if existing_trade:
                continue
            validation = parse_json_dict(row.get("validation_json"))
            validation["recovery_error"] = "Intent was left in submitting state without a matching live trade record"
            self.db.execute(
                "UPDATE live_order_intents SET status = 'failed', lifecycle_state = 'stale', validation_json = ? WHERE id = ?",
                (json.dumps(validation, default=str), row["id"]),
            )
            self.log("ERROR", f"Recovered stuck live intent {row['id']} as failed", {"intent": row})
            recovered += 1
        return recovered

    def mark_stale_intents(self, max_age_hours: int = 6) -> int:
        rows = self.db.fetch_all(
            """
            SELECT id, last_seen_at, timestamp FROM live_order_intents
            WHERE lifecycle_state IN ('new', 'seen')
              AND status IN ('draft', 'approved')
            """
        )
        updated = 0
        for row in rows:
            try:
                value = row.get("last_seen_at") or row.get("timestamp")
                seen_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                continue
            age_hours = (datetime.now(timezone.utc) - seen_at).total_seconds() / 3600
            if age_hours >= max_age_hours:
                self.db.execute(
                    "UPDATE live_order_intents SET lifecycle_state = 'stale', last_status_reason = ? WHERE id = ?",
                    (f"Not seen in the last {max_age_hours} hours", row["id"]),
                )
                updated += 1
        return updated

    def close_trade(self, trade_id: int, exit_price: float, reason: str) -> None:
        trade = self.db.fetch_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        if not trade:
            raise ValueError("Trade not found")
        pnl_percent = ((exit_price - trade["entry_price"]) / trade["entry_price"]) * 100
        pnl_usdt = None
        if trade.get("position_size"):
            pnl_usdt = (exit_price - trade["entry_price"]) * trade["position_size"]
        self.db.execute(
            """
            UPDATE trades SET exit_time = ?, exit_price = ?, status = 'closed',
                lifecycle_state = 'closed',
                pnl_percent = ?, pnl_usdt = ?, exit_reason = ?
            WHERE id = ?
            """,
            (now_iso(), exit_price, round(pnl_percent, 4), round(pnl_usdt or 0, 4), reason, trade_id),
        )
        self.log("INFO", f"Paper trade {trade_id} closed", {"exit_price": exit_price, "reason": reason})
        self.record_trade_event(trade_id, "trade_closed", f"Trade {trade_id} closed", {"exit_price": exit_price, "reason": reason})

    def mark_trade_manually_closed(self, trade_id: int, reason: str = "User marked manually closed") -> dict[str, Any]:
        trade = self.db.fetch_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        if not trade:
            raise ValueError("Trade not found")
        current_price = trade.get("exit_price") or trade.get("entry_price")
        try:
            prices = self.market_data.fetch_last_prices([trade["symbol"]])
            current_price = prices.get(trade["symbol"], current_price)
        except Exception as exc:
            self.log("WARNING", f"Could not fetch price while marking trade {trade_id} manually closed", {"error": str(exc)})
        exit_price = float(current_price or trade["entry_price"])
        pnl_percent = ((exit_price - float(trade["entry_price"])) / float(trade["entry_price"])) * 100
        pnl_usdt = (exit_price - float(trade["entry_price"])) * float(trade.get("remaining_position_size") or trade.get("position_size") or 0)
        now = now_iso()
        self.db.execute(
            """
            UPDATE trades
            SET status = 'manually_closed',
                lifecycle_state = 'manually_closed',
                exit_time = ?,
                exit_price = ?,
                pnl_percent = ?,
                pnl_usdt = ?,
                exit_reason = ?,
                manually_closed_at = ?,
                remaining_position_size = 0
            WHERE id = ?
            """,
            (now, round(exit_price, 8), round(pnl_percent, 4), round(pnl_usdt, 4), reason, now, trade_id),
        )
        payload = {"trade_id": trade_id, "symbol": trade["symbol"], "exit_price": exit_price, "reason": reason}
        self.record_trade_event(trade_id, "user_marked_manually_closed", f"User marked trade {trade_id} manually closed", payload, level="WARNING")
        return payload

    def get_open_positions(self) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM trades WHERE status IN ('open', 'partial', 'partially_closed') ORDER BY entry_time DESC"
        )
        return self._with_live_pnl(rows)

    def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))
        return self._with_live_pnl(rows)

    def get_portfolio(self) -> dict[str, Any]:
        self.refresh_live_portfolio_state()
        settings = self.db.fetch_one("SELECT account_size FROM settings WHERE id = 1") or {}
        account_size = float(settings.get("account_size") or 0)
        trades = self.get_trade_history(limit=100000)

        open_trades = [trade for trade in trades if trade.get("status") in {"open", "partial", "partially_closed"}]
        closed_trades = [trade for trade in trades if trade.get("status") in {"closed", "manually_closed"}]
        realized_pnl = sum(float(trade.get("pnl_usdt") or 0) for trade in closed_trades)
        unrealized_pnl = sum(float(trade.get("unrealized_pnl_usdt") or 0) for trade in open_trades)
        total_pnl = realized_pnl + unrealized_pnl
        open_exposure = sum(float(trade.get("current_value_usdt") or 0) for trade in open_trades)
        entry_exposure = sum(float(trade.get("entry_value_usdt") or 0) for trade in open_trades)
        winners = [trade for trade in closed_trades if float(trade.get("pnl_usdt") or 0) > 0]
        losers = [trade for trade in closed_trades if float(trade.get("pnl_usdt") or 0) < 0]

        by_symbol: dict[str, dict[str, Any]] = {}
        for trade in trades:
            symbol = trade.get("symbol") or "unknown"
            row = by_symbol.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "open_count": 0,
                    "closed_count": 0,
                    "pnl_usdt": 0.0,
                    "exposure_usdt": 0.0,
                },
            )
            if trade.get("status") in {"open", "partial", "partially_closed"}:
                row["open_count"] += 1
                row["exposure_usdt"] += float(trade.get("current_value_usdt") or 0)
                row["pnl_usdt"] += float(trade.get("unrealized_pnl_usdt") or 0)
            else:
                row["closed_count"] += 1
                row["pnl_usdt"] += float(trade.get("pnl_usdt") or 0)

        return {
            "summary": {
                "account_size": round(account_size, 2),
                "open_trades": len(open_trades),
                "closed_trades": len(closed_trades),
                "total_trades": len(trades),
                "realized_pnl_usdt": round(realized_pnl, 4),
                "unrealized_pnl_usdt": round(unrealized_pnl, 4),
                "total_pnl_usdt": round(total_pnl, 4),
                "total_return_percent": round((total_pnl / account_size) * 100, 4) if account_size else 0,
                "open_exposure_usdt": round(open_exposure, 4),
                "entry_exposure_usdt": round(entry_exposure, 4),
                "win_rate_percent": round((len(winners) / len(closed_trades)) * 100, 2) if closed_trades else 0,
                "winners": len(winners),
                "losers": len(losers),
                "live_open": len([trade for trade in open_trades if trade.get("execution_type") == "live"]),
                "paper_open": len([trade for trade in open_trades if trade.get("execution_type") != "live"]),
            },
            "trades": trades,
            "by_symbol": sorted(
                [
                    {
                        **row,
                        "pnl_usdt": round(float(row["pnl_usdt"]), 4),
                        "exposure_usdt": round(float(row["exposure_usdt"]), 4),
                    }
                    for row in by_symbol.values()
                ],
                key=lambda item: float(item.get("pnl_usdt") or 0),
            ),
            "charts": {
                "pnl_by_symbol": [
                    {"label": row["symbol"], "value": row["pnl_usdt"]}
                    for row in sorted(by_symbol.values(), key=lambda item: float(item.get("pnl_usdt") or 0))
                ],
                "exposure_by_symbol": [
                    {"label": row["symbol"], "value": round(float(row["exposure_usdt"]), 4)}
                    for row in sorted(by_symbol.values(), key=lambda item: float(item.get("exposure_usdt") or 0), reverse=True)
                    if float(row.get("exposure_usdt") or 0) > 0
                ],
            },
        }

    def refresh_live_portfolio_state(self) -> dict[str, Any]:
        """
        Single lightweight reconciliation path for live trades.

        For each open live trade:
        - sync open protective sell orders from Binance
        - read account fills since entry
        - if sell fills cover the position, close the trade and write PnL
        - if partially sold, update remaining_position_size and realized PnL so far

        This intentionally assumes one open live trade per symbol, which the scanner now enforces.
        """
        rows = self.db.fetch_all(
            """
            SELECT * FROM trades
            WHERE status IN ('open', 'partial', 'partially_closed')
              AND execution_type = 'live'
            ORDER BY entry_time ASC
            """
        )
        refreshed: list[dict[str, Any]] = []
        errors: list[str] = []
        for trade in rows:
            try:
                refreshed.append(self._refresh_live_trade_from_exchange(trade))
            except Exception as exc:
                message = f"{trade.get('symbol')} trade {trade.get('id')}: {exc}"
                errors.append(message)
                self.db.execute(
                    "UPDATE trades SET last_reconciled_at = ?, last_reconcile_error = ? WHERE id = ?",
                    (now_iso(), str(exc), trade.get("id")),
                )
                self.log("ERROR", "Live portfolio refresh failed", {"trade": trade, "error": str(exc)})
        return {"refreshed": refreshed, "errors": errors}

    def _refresh_live_trade_from_exchange(self, trade: dict[str, Any]) -> dict[str, Any]:
        trade_id = int(trade["id"])
        symbol = str(trade["symbol"])
        entry_time = str(trade["entry_time"])
        entry_price = float(trade.get("entry_price") or 0)
        position_size = float(trade.get("position_size") or 0)
        if position_size <= 0 or entry_price <= 0:
            return {"trade_id": trade_id, "symbol": symbol, "status": "skipped", "reason": "missing entry size/price"}

        exit_sync = self.live_executor.sync_exit_orders_for_trade(trade_id)
        since_ms = self._timestamp_ms(entry_time)
        fills = self.market_data.private_exchange().fetch_my_trades(symbol, since=since_ms)
        sell_fills = [fill for fill in fills if str(fill.get("side") or "").lower() == "sell"]

        sold_qty = sum(float(fill.get("amount") or 0) for fill in sell_fills)
        sell_proceeds = sum(float(fill.get("cost") or 0) for fill in sell_fills)
        sell_fees = self._quote_fees_usdt(sell_fills)
        average_exit = sell_proceeds / sold_qty if sold_qty > 0 else None
        remaining_qty = max(position_size - sold_qty, 0)
        sold_fraction = min(sold_qty / position_size, 1) if position_size else 0
        estimated_buy_fee = entry_price * sold_qty * 0.001
        realized_pnl = sell_proceeds - (entry_price * sold_qty) - sell_fees - estimated_buy_fee
        realized_pct = (realized_pnl / (entry_price * position_size)) * 100 if position_size else 0

        if sold_qty >= position_size * 0.98 and average_exit:
            self.db.execute(
                """
                UPDATE trades
                SET status = 'closed',
                    lifecycle_state = 'closed',
                    exit_time = ?,
                    exit_price = ?,
                    pnl_percent = ?,
                    pnl_usdt = ?,
                    exit_reason = ?,
                    remaining_position_size = 0,
                    closed_detected_at = ?,
                    last_reconciled_at = ?,
                    last_reconcile_error = ''
                WHERE id = ?
                """,
                (
                    self._last_fill_time(sell_fills) or now_iso(),
                    round(float(average_exit), 8),
                    round(realized_pct, 4),
                    round(realized_pnl, 4),
                    "Closed from Binance sell fills on refresh",
                    now_iso(),
                    now_iso(),
                    trade_id,
                ),
            )
            self.record_trade_event(trade_id, "trade_closed", f"Trade {trade_id} closed from Binance fills", {"fills": sell_fills, "realized_pnl": realized_pnl})
            status = "closed"
        elif sold_qty > 0:
            self.db.execute(
                """
                UPDATE trades
                SET status = 'partially_closed',
                    lifecycle_state = 'partially_closed',
                    remaining_position_size = ?,
                    pnl_percent = ?,
                    pnl_usdt = ?,
                    exit_reason = ?,
                    last_reconciled_at = ?,
                    last_reconcile_error = ''
                WHERE id = ?
                """,
                (
                    round(remaining_qty, 8),
                    round(realized_pct, 4),
                    round(realized_pnl, 4),
                    f"Partially sold on Binance: {sold_fraction:.1%} filled",
                    now_iso(),
                    trade_id,
                ),
            )
            self.record_trade_event(trade_id, "trade_partially_closed", f"Trade {trade_id} partially sold on Binance", {"sold_qty": sold_qty, "remaining_qty": remaining_qty})
            status = "partial"
        else:
            balance_total = self._base_asset_total(symbol)
            remaining_reference = float(trade.get("remaining_position_size") or position_size)
            if balance_total is not None and balance_total <= remaining_reference * 0.02 and not exit_sync.get("protected"):
                self.db.execute(
                    """
                    UPDATE trades
                    SET status = 'manually_closed',
                        lifecycle_state = 'manually_closed',
                        exit_time = ?,
                        exit_reason = ?,
                        manually_closed_at = ?,
                        closed_detected_at = ?,
                        remaining_position_size = 0,
                        last_reconciled_at = ?,
                        last_reconcile_error = ''
                    WHERE id = ?
                    """,
                    (
                        now_iso(),
                        "Detected no remaining Binance balance/open exit orders during refresh",
                        now_iso(),
                        now_iso(),
                        now_iso(),
                        trade_id,
                    ),
                )
                self.record_trade_event(
                    trade_id,
                    "manual_close_detected",
                    f"Manual close detected for trade {trade_id}",
                    {"base_asset_total": balance_total, "remaining_reference": remaining_reference, "exit_sync": exit_sync},
                    level="WARNING",
                )
                status = "manually_closed"
            else:
                lifecycle_state = "exit_strategy_active" if exit_sync.get("protected") else "exit_strategy_pending"
                self.db.execute(
                    """
                    UPDATE trades
                    SET remaining_position_size = COALESCE(remaining_position_size, position_size),
                        lifecycle_state = ?,
                        last_reconciled_at = ?,
                        last_reconcile_error = ''
                    WHERE id = ?
                    """,
                    (lifecycle_state, now_iso(), trade_id),
                )
                status = "open"

        return {
            "trade_id": trade_id,
            "symbol": symbol,
            "status": status,
            "sold_qty": round(sold_qty, 8),
            "remaining_qty": round(remaining_qty, 8),
            "average_exit": round(float(average_exit), 8) if average_exit else None,
            "realized_pnl_usdt": round(realized_pnl, 4),
            "exit_sync": exit_sync.get("status"),
        }

    def _base_asset_total(self, symbol: str) -> float | None:
        try:
            base = symbol.split("/")[0]
            balance = self.market_data.private_exchange().fetch_balance()
            total = (balance.get("total") or {}).get(base)
            if total is None:
                free = (balance.get("free") or {}).get(base) or 0
                used = (balance.get("used") or {}).get(base) or 0
                total = float(free) + float(used)
            return float(total or 0)
        except Exception as exc:
            self.log("WARNING", f"Could not fetch base balance for {symbol}", {"error": str(exc)})
            return None

    def _timestamp_ms(self, value: str) -> int | None:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def _last_fill_time(self, fills: list[dict[str, Any]]) -> str | None:
        timestamps = [int(fill.get("timestamp") or 0) for fill in fills if fill.get("timestamp")]
        if not timestamps:
            return None
        return datetime.fromtimestamp(max(timestamps) / 1000, timezone.utc).isoformat()

    def _quote_fees_usdt(self, fills: list[dict[str, Any]]) -> float:
        total = 0.0
        for fill in fills:
            fees = list(fill.get("fees") or [])
            fee = fill.get("fee")
            if fee:
                fees.append(fee)
            for item in fees:
                currency = str(item.get("currency") or "").upper()
                if currency in {"USDT", "FDUSD", "USDC"}:
                    total += float(item.get("cost") or 0)
        return total

    def get_performance(self) -> dict[str, Any]:
        self.refresh_live_portfolio_state()
        trades = self.get_trade_history(limit=100000)
        status = self.db.fetch_one("SELECT * FROM bot_status WHERE id = 1") or {}
        settings = self.db.fetch_one("SELECT * FROM settings WHERE id = 1") or {}
        signals = self._latest_signals(limit=200)

        open_trades = [trade for trade in trades if trade.get("status") in {"open", "partial", "partially_closed"}]
        closed_trades = [trade for trade in trades if trade.get("status") in {"closed", "manually_closed"}]
        realized_pnl = sum(float(trade.get("pnl_usdt") or 0) for trade in closed_trades)
        unrealized_pnl = sum(float(trade.get("unrealized_pnl_usdt") or 0) for trade in open_trades)
        total_pnl = realized_pnl + unrealized_pnl
        account_size = float(settings.get("account_size") or 0)
        total_return = (total_pnl / account_size) * 100 if account_size else 0
        winners = [trade for trade in closed_trades if float(trade.get("pnl_usdt") or 0) > 0]
        win_rate = (len(winners) / len(closed_trades)) * 100 if closed_trades else 0
        signal_scores = [float(signal.get("score") or 0) for signal in signals]
        avg_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0

        return {
            "summary": {
                "account_size": round(account_size, 2),
                "realized_pnl_usdt": round(realized_pnl, 4),
                "unrealized_pnl_usdt": round(unrealized_pnl, 4),
                "total_pnl_usdt": round(total_pnl, 4),
                "total_return_percent": round(total_return, 4),
                "open_positions": len(open_trades),
                "closed_trades": len(closed_trades),
                "win_rate_percent": round(win_rate, 2),
                "avg_signal_score": round(avg_score, 2),
                "market_regime": status.get("market_regime", "unknown"),
            },
            "charts": {
                "equity_curve": self._equity_curve(trades),
                "open_pnl_by_symbol": [
                    {"label": trade["symbol"], "value": round(float(trade.get("unrealized_pnl_usdt") or 0), 4)}
                    for trade in sorted(open_trades, key=lambda item: float(item.get("unrealized_pnl_usdt") or 0))
                ],
                "signal_tiers": self._count_by(signals, "tier"),
                "trade_status": self._count_by(trades, "status"),
                "score_buckets": self._score_buckets(signals),
            },
        }

    def _latest_signals(self, limit: int = 200) -> list[dict[str, Any]]:
        status = self.db.fetch_one("SELECT last_scan_time FROM bot_status WHERE id = 1") or {}
        if status.get("last_scan_time"):
            return self.db.fetch_all(
                """
                SELECT * FROM signals
                WHERE id IN (
                    SELECT MAX(id) FROM signals
                    WHERE ignored = 0 AND timestamp >= ?
                    GROUP BY symbol
                )
                ORDER BY score DESC, volume_rank ASC
                LIMIT ?
                """,
                (status["last_scan_time"], limit),
            )
        return self.db.fetch_all(
            "SELECT * FROM signals WHERE ignored = 0 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def _equity_curve(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cumulative = 0.0
        points: list[dict[str, Any]] = [{"label": "Start", "value": 0.0}]
        ordered = sorted(trades, key=lambda item: item.get("exit_time") or item.get("entry_time") or "")
        for trade in ordered:
            pnl = float(trade.get("pnl_usdt") or 0)
            cumulative += pnl
            label = trade.get("symbol", "")
            if trade.get("status") == "open":
                label = f"{label} open"
            points.append({"label": label, "value": round(cumulative, 4)})
        return points

    def _count_by(self, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return [{"label": label, "value": value} for label, value in sorted(counts.items())]

    def _score_buckets(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets = {
            "Ignore <55": 0,
            "Early 55-69": 0,
            "Strong 70-84": 0,
            "Tier 1 85+": 0,
        }
        for signal in signals:
            score = float(signal.get("score") or 0)
            if score >= 85:
                buckets["Tier 1 85+"] += 1
            elif score >= 70:
                buckets["Strong 70-84"] += 1
            elif score >= 55:
                buckets["Early 55-69"] += 1
            else:
                buckets["Ignore <55"] += 1
        return [{"label": label, "value": value} for label, value in buckets.items()]

    def _with_live_pnl(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        open_rows = [row for row in rows if row.get("status") in {"open", "partial", "partially_closed"}]
        prices: dict[str, float] = {}
        if open_rows:
            try:
                prices = self.market_data.fetch_last_prices(sorted({row["symbol"] for row in open_rows}))
            except Exception as exc:
                logger.warning("Could not fetch live prices for open positions: %s", exc)
                self.db.execute("UPDATE bot_status SET errors = ? WHERE id = 1", (f"PnL price refresh failed: {exc}",))

        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["current_price"] = item.get("exit_price")
            item["unrealized_pnl_percent"] = None
            item["unrealized_pnl_usdt"] = None
            item["time_in_trade"] = ""
            item["status_notes"] = item.get("exit_reason") or ""

            if item.get("status") in {"open", "partial", "partially_closed"}:
                current_price = prices.get(item["symbol"], item["entry_price"])
                entry_price = float(item["entry_price"])
                original_size = float(item.get("position_size") or 0)
                remaining_size = float(item.get("remaining_position_size") or original_size)
                realized_pnl = float(item.get("pnl_usdt") or 0) if item.get("execution_type") == "live" else 0.0
                entry_value = entry_price * remaining_size
                current_value = current_price * remaining_size
                unrealized_pnl = (current_price - entry_price) * remaining_size
                total_pnl = realized_pnl + unrealized_pnl
                original_entry_value = entry_price * original_size
                pnl_percent = (total_pnl / original_entry_value) * 100 if original_entry_value else 0
                item["current_price"] = round(current_price, 8)
                item["remaining_position_size"] = round(remaining_size, 8)
                item["entry_value_usdt"] = round(entry_value, 4)
                item["current_value_usdt"] = round(current_value, 4)
                item["unrealized_pnl_percent"] = round(pnl_percent, 4)
                item["unrealized_pnl_usdt"] = round(unrealized_pnl, 4)
                item["realized_pnl_usdt"] = round(realized_pnl, 4)
                item["pnl_percent"] = item["unrealized_pnl_percent"]
                item["pnl_usdt"] = round(total_pnl, 4)
                item["time_in_trade"] = self._time_in_trade(item["entry_time"])
                item["status_notes"] = self._position_status_note(item, current_price)
            else:
                entry_price = float(item.get("entry_price") or 0)
                exit_price = float(item.get("exit_price") or 0)
                position_size = float(item.get("position_size") or 0)
                item["entry_value_usdt"] = round(entry_price * position_size, 4) if position_size else None
                item["current_value_usdt"] = round(exit_price * position_size, 4) if exit_price and position_size else None
            enriched.append(item)
        return enriched

    def _time_in_trade(self, entry_time: str) -> str:
        try:
            start = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - start
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            return f"{hours}h {minutes}m"
        except Exception:
            return ""

    def _position_status_note(self, trade: dict[str, Any], current_price: float) -> str:
        stop = trade.get("stop_loss")
        tp1 = trade.get("take_profit_1")
        tp2 = trade.get("take_profit_2")
        if stop and current_price <= float(stop):
            return "At or below invalidation"
        if tp2 and current_price >= float(tp2):
            return "At or above TP2"
        if tp1 and current_price >= float(tp1):
            return "At or above TP1"
        return "Open, mark-to-market"

    def export_trades_csv(self) -> str:
        rows = self.get_trade_history(limit=100000)
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()) if rows else ["id", "symbol", "status"])
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()

    def log(self, level: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "INSERT INTO event_log (timestamp, level, message, context) VALUES (?, ?, ?, ?)",
            (now_iso(), level, message, json.dumps(context or {}, default=str)),
        )

    def record_trade_event(
        self,
        trade_id: int | None,
        event_type: str,
        message: str,
        context: dict[str, Any] | None = None,
        level: str = "INFO",
        intent_id: int | None = None,
        symbol: str | None = None,
    ) -> None:
        if trade_id and not symbol:
            trade = self.db.fetch_one("SELECT symbol FROM trades WHERE id = ?", (trade_id,))
            symbol = trade.get("symbol") if trade else None
        self.db.execute(
            """
            INSERT INTO trade_events (timestamp, trade_id, intent_id, symbol, event_type, level, message, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now_iso(), trade_id, intent_id, symbol, event_type, level, message, json.dumps(context or {}, default=str)),
        )
