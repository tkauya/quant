from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any

from .db import Database
from .market_data import BinanceMarketData


class LiveExecutionDisabled(RuntimeError):
    pass


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quote_amount: float
    base_quantity: float | None
    reference_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reason: str
    signal_id: int | None = None


@dataclass
class ExchangeFilterResult:
    ok: bool
    adjusted_quantity: float | None
    adjusted_quote_amount: float
    notes: list[str]


class BinanceFilterHelper:
    def __init__(self, market_data: BinanceMarketData):
        self.market_data = market_data

    def validate_intent(self, intent: OrderIntent) -> ExchangeFilterResult:
        self.market_data.ensure_markets_loaded()
        market = self.market_data.exchange.market(intent.symbol)
        filters = {
            item.get("filterType"): item
            for item in market.get("info", {}).get("filters", [])
        }

        notes: list[str] = []

        min_notional = self._decimal_filter(filters, "NOTIONAL", "minNotional")
        if min_notional is None:
            min_notional = self._decimal_filter(filters, "MIN_NOTIONAL", "minNotional")

        quote_amount = Decimal(str(intent.quote_amount))

        if min_notional is not None and quote_amount < min_notional:
            notes.append(f"Quote amount {quote_amount} is below min notional {min_notional}")

        adjusted_quantity = None

        if intent.base_quantity is not None:
            quantity = Decimal(str(intent.base_quantity))

            step_size = self._decimal_filter(filters, "LOT_SIZE", "stepSize")
            min_qty = self._decimal_filter(filters, "LOT_SIZE", "minQty")

            if step_size:
                quantity = self._round_down(quantity, step_size)
                notes.append(f"Rounded quantity to step size {step_size}")

            if min_qty is not None and quantity < min_qty:
                notes.append(f"Quantity {quantity} is below min quantity {min_qty}")

            adjusted_quantity = float(quantity)

        return ExchangeFilterResult(
            ok=not any("below" in note for note in notes),
            adjusted_quantity=adjusted_quantity,
            adjusted_quote_amount=float(quote_amount),
            notes=notes,
        )

    def _decimal_filter(
        self,
        filters: dict[str, Any],
        filter_name: str,
        field: str,
    ) -> Decimal | None:
        value = (filters.get(filter_name) or {}).get(field)
        return Decimal(str(value)) if value else None

    def _round_down(self, value: Decimal, step: Decimal) -> Decimal:
        return (value / step).to_integral_value(rounding=ROUND_DOWN) * step

    def round_quantity_down(self, symbol: str, quantity: float) -> float:
        market = self.market_data.exchange.market(symbol)
        filters = {
            item.get("filterType"): item
            for item in market.get("info", {}).get("filters", [])
        }
        step_size = self._decimal_filter(filters, "LOT_SIZE", "stepSize")
        value = Decimal(str(quantity))
        if step_size:
            value = self._round_down(value, step_size)
        return float(value)


class RiskGuard:
    def __init__(self, db: Database):
        self.db = db

    def validate_intent(self, intent: OrderIntent) -> list[str]:
        settings = self.db.fetch_one(
            "SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1"
        ) or {}

        account_size = float(settings.get("account_size") or 0)
        risk_percent = float(settings.get("risk_per_trade_percent") or 0)

        max_risk = account_size * (risk_percent / 100)

        stop_distance = max(intent.reference_price - intent.stop_loss, 0)
        estimated_base = (
            intent.quote_amount / intent.reference_price
            if intent.reference_price
            else 0
        )
        estimated_risk = stop_distance * estimated_base

        problems: list[str] = []

        if intent.side != OrderSide.BUY:
            problems.append("Only long spot buy intents are currently supported")

        if intent.order_type != OrderType.MARKET:
            problems.append("Only market orders are currently supported")

        if intent.quote_amount <= 0:
            problems.append("Quote amount must be positive")

        tolerance = max(0.01, max_risk * 0.001)
        if estimated_risk > max_risk + tolerance:
            problems.append(
                f"Estimated risk {estimated_risk:.4f} exceeds max risk {max_risk:.4f}"
            )

        if intent.stop_loss >= intent.reference_price:
            problems.append("Stop loss must be below reference price")

        return problems


class GatedLiveExecutor:
    """
    Live executor with explicit safety gates.

    It can:
    - build order intents
    - validate order intents
    - approve/reject draft intents
    - submit approved market buys only if live_trading_enabled = 1
    - store execution records in trades

    Important:
    - The live entry is still permissioned: approved intent + explicit submit.
    - After a live buy fills, it can automatically place a protective Spot OCO
      sell using TP1 and stop_loss.
    - TP2 is retained as a planned runner target for later staged exits.
    """

    def __init__(self, db: Database, market_data: BinanceMarketData):
        self.db = db
        self.market_data = market_data
        self.filters = BinanceFilterHelper(market_data)
        self.risk_guard = RiskGuard(db)

    def build_intent_from_signal(self, signal: dict[str, Any]) -> OrderIntent:
        settings = self.db.fetch_one(
            "SELECT account_size, risk_per_trade_percent FROM settings WHERE id = 1"
        ) or {}

        account_size = float(settings.get("account_size") or 0)
        risk_percent = float(settings.get("risk_per_trade_percent") or 0)

        risk_usdt = account_size * (risk_percent / 100)

        price = float(signal["price"])
        stop = float(signal["stop_loss"])

        stop_distance = max(price - stop, price * 0.005)

        quote_amount = (
            min(
                account_size,
                (risk_usdt / stop_distance) * price,
            )
            if stop_distance
            else 0
        )
        quote_amount = max(0.0, int((quote_amount * 0.995) * 100) / 100)

        return OrderIntent(
            signal_id=signal.get("id"),
            symbol=signal["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quote_amount=quote_amount,
            base_quantity=None,
            reference_price=price,
            stop_loss=stop,
            take_profit_1=float(signal["take_profit_1"]),
            take_profit_2=float(signal["take_profit_2"]),
            reason=f"Signal score {signal['score']}: {signal['tier']}",
        )

    def validate_intent(self, intent: OrderIntent) -> dict[str, Any]:
        risk_problems = self.risk_guard.validate_intent(intent)
        filter_result = self.filters.validate_intent(intent)

        return {
            "ok": not risk_problems and filter_result.ok,
            "risk_problems": risk_problems,
            "filter_notes": filter_result.notes,
            "adjusted_quantity": filter_result.adjusted_quantity,
            "adjusted_quote_amount": filter_result.adjusted_quote_amount,
        }

    def approve_intent(self, intent_id: int) -> dict[str, Any]:
        intent = self.db.fetch_one(
            "SELECT * FROM live_order_intents WHERE id = ?",
            (intent_id,),
        )

        if not intent:
            raise ValueError("Live order intent not found")

        if intent["status"] != "draft":
            raise ValueError(
                f"Only draft intents can be approved. Current status: {intent['status']}"
            )

        validation = json.loads(intent.get("validation_json") or "{}")

        if not validation.get("ok"):
            raise ValueError(f"Cannot approve invalid intent: {validation}")

        self.db.execute(
            "UPDATE live_order_intents SET status = 'approved' WHERE id = ?",
            (intent_id,),
        )

        self._log("INFO", f"Live order intent {intent_id} approved", intent)

        return self.db.fetch_one(
            "SELECT * FROM live_order_intents WHERE id = ?",
            (intent_id,),
        ) or {}

    def reject_intent(
        self,
        intent_id: int,
        reason: str = "Rejected manually",
    ) -> None:
        intent = self.db.fetch_one(
            "SELECT * FROM live_order_intents WHERE id = ?",
            (intent_id,),
        )

        if not intent:
            raise ValueError("Live order intent not found")

        self.db.execute(
            """
            UPDATE live_order_intents
            SET status = 'rejected',
                reason = reason || ?
            WHERE id = ?
            """,
            (f" | {reason}", intent_id),
        )

        self._log(
            "INFO",
            f"Live order intent {intent_id} rejected",
            {"intent_id": intent_id, "reason": reason},
        )

    def submit_approved_intent(self, intent_id: int) -> dict[str, Any]:
        if not self.market_data.settings.enable_binance_live_trading:
            raise LiveExecutionDisabled("ENABLE_BINANCE_LIVE_TRADING is not enabled in .env")

        if not self.market_data.settings.live_trading_unlocked:
            raise LiveExecutionDisabled("LIVE_TRADING_UNLOCKED is not enabled in .env")

        settings = self.db.fetch_one(
            "SELECT live_trading_enabled FROM settings WHERE id = 1"
        ) or {}

        if int(settings.get("live_trading_enabled") or 0) != 1:
            raise LiveExecutionDisabled("Live trading is disabled in settings")

        intent = self.db.fetch_one(
            "SELECT * FROM live_order_intents WHERE id = ?",
            (intent_id,),
        )

        if not intent:
            raise ValueError("Live order intent not found")

        if intent["status"] != "approved":
            raise LiveExecutionDisabled(
                f"Intent must be approved before execution. Current status: {intent['status']}"
            )

        existing_trade = self.db.fetch_one(
            "SELECT id, exchange_order_id FROM trades WHERE signal_id = ? AND symbol = ? AND execution_type = 'live' LIMIT 1",
            (intent.get("signal_id"), intent.get("symbol")),
        )
        if existing_trade:
            raise LiveExecutionDisabled(
                f"Intent appears already submitted as live trade {existing_trade['id']}"
            )

        validation = json.loads(intent.get("validation_json") or "{}")

        if not validation.get("ok"):
            raise LiveExecutionDisabled(f"Intent validation failed: {validation}")

        symbol = intent["symbol"]
        quote_amount = float(
            validation.get("adjusted_quote_amount") or intent["quote_amount"]
        )

        if quote_amount <= 0:
            raise LiveExecutionDisabled("Quote amount must be positive")

        self.db.execute(
            """
            UPDATE live_order_intents
            SET status = 'submitting',
                submitted_at = ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), intent_id),
        )

        try:
            order = self.market_data.private_exchange().create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=None,
                price=None,
                params={
                    "quoteOrderQty": quote_amount,
                },
            )

        except Exception as exc:
            self.db.execute(
                """
                UPDATE live_order_intents
                SET status = 'failed',
                    validation_json = ?
                WHERE id = ?
                """,
                (
                    json.dumps(
                        {
                            "previous_validation": validation,
                            "execution_error": str(exc),
                        },
                        default=str,
                    ),
                    intent_id,
                ),
            )

            self._log(
                "ERROR",
                f"Live order failed for {symbol}",
                {
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "quote_amount": quote_amount,
                    "error": str(exc),
                },
            )

            raise

        average_price = self._extract_average_fill_price(order)
        filled_quantity = self._extract_filled_quantity(order)
        exchange_order_id = self._extract_order_id(order)

        if average_price is None:
            average_price = float(intent["reference_price"])

        if filled_quantity is None:
            filled_quantity = 0.0

        trade_id = self.db.execute(
            """
            INSERT INTO trades (
                signal_id,
                symbol,
                entry_time,
                entry_price,
                stop_loss,
                take_profit_1,
                take_profit_2,
                status,
                entry_reason,
                exit_reason,
                score_at_entry,
                position_size,
                exchange_order_id,
                execution_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, '', ?, ?, ?, 'live')
            """,
            (
                intent.get("signal_id"),
                symbol,
                datetime.now(timezone.utc).isoformat(),
                average_price,
                intent.get("stop_loss"),
                intent.get("take_profit_1"),
                intent.get("take_profit_2"),
                f"LIVE entry from approved intent {intent_id}: {intent.get('reason', '')}",
                None,
                filled_quantity,
                exchange_order_id,
            ),
        )

        protective_exit: dict[str, Any] = {
            "enabled": bool(self.market_data.settings.auto_place_protective_oco),
            "status": "disabled",
        }
        if self.market_data.settings.auto_place_protective_oco:
            protective_exit = self._place_momentum_exit_orders(
                trade_id=trade_id,
                symbol=symbol,
                filled_quantity=filled_quantity,
                average_price=average_price,
                stop_loss=float(intent.get("stop_loss")),
                take_profit_1=float(intent.get("take_profit_1")),
                take_profit_2=float(intent.get("take_profit_2")),
                entry_order=order,
            )

        submitted_payload = {
            "previous_validation": validation,
            "submitted": True,
            "exchange_order_id": exchange_order_id,
            "average_price": average_price,
            "filled_quantity": filled_quantity,
            "protective_exit": protective_exit,
        }

        self.db.execute(
            """
            UPDATE live_order_intents
            SET status = 'submitted',
                validation_json = ?,
                exchange_order_id = ?,
                submitted_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(submitted_payload, default=str),
                exchange_order_id,
                datetime.now(timezone.utc).isoformat(),
                intent_id,
            ),
        )

        self._log(
            "INFO",
            f"Live order submitted for {symbol}",
            {
                "intent_id": intent_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "quote_amount": quote_amount,
                "average_price": average_price,
                "filled_quantity": filled_quantity,
                "exchange_order_id": exchange_order_id,
                "protective_exit": protective_exit,
                "raw_order": order,
            },
        )

        return {
            "intent_id": intent_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "quote_amount": quote_amount,
            "average_price": average_price,
            "filled_quantity": filled_quantity,
            "exchange_order_id": exchange_order_id,
            "protective_exit": protective_exit,
            "status": "submitted",
            "raw_order": order,
        }

    def retry_exit_orders_for_trade(self, trade_id: int) -> dict[str, Any]:
        trade = self.db.fetch_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        if not trade:
            raise ValueError("Trade not found")
        if trade.get("status") != "open" or trade.get("execution_type") != "live":
            raise ValueError("Only open live trades can have exit orders retried")
        quantity = self.filters.round_quantity_down(
            trade["symbol"],
            float(trade.get("remaining_position_size") or trade.get("position_size") or 0) * 0.999,
        )
        result = self._place_full_stop_only(
            trade_id=trade_id,
            symbol=trade["symbol"],
            quantity=quantity,
            stop_loss=float(trade["stop_loss"]),
        )
        self._log("WARNING", f"Exit orders retried for trade {trade_id}", result)
        return result

    def _place_protective_oco(
        self,
        trade_id: int,
        symbol: str,
        filled_quantity: float,
        average_price: float,
        stop_loss: float,
        take_profit_1: float,
        entry_order: dict[str, Any],
    ) -> dict[str, Any]:
        if filled_quantity <= 0:
            return self._record_exit_order_failure(
                trade_id,
                "Filled quantity was zero; protective OCO was not placed",
            )

        if not (stop_loss < average_price < take_profit_1):
            return self._record_exit_order_failure(
                trade_id,
                f"Invalid protective prices: stop={stop_loss}, entry={average_price}, tp1={take_profit_1}",
            )

        net_quantity = min(
            filled_quantity - self._extract_base_fee(entry_order, symbol),
            filled_quantity * 0.999,
        )
        quantity = self.filters.round_quantity_down(symbol, net_quantity)

        if quantity <= 0:
            return self._record_exit_order_failure(
                trade_id,
                "Rounded protective OCO quantity was zero",
            )

        exchange = self.market_data.private_exchange()
        params = self._oco_params(symbol, quantity, take_profit_1, stop_loss)

        try:
            order_list = exchange.privatePostOrderListOco(params)
        except Exception as exc:
            self._log(
                "ERROR",
                f"Protective OCO failed for trade {trade_id}",
                {"trade_id": trade_id, "symbol": symbol, "params": params, "error": str(exc)},
            )
            return self._record_exit_order_failure(trade_id, str(exc), params)

        order_list_id = self._extract_order_list_id(order_list)
        payload = {
            "status": "placed",
            "order_list_id": order_list_id,
            "symbol": symbol,
            "quantity": quantity,
            "take_profit_1": take_profit_1,
            "stop_loss": stop_loss,
            "stop_limit_price": params.get("belowPrice"),
            "params": params,
            "raw_order": order_list,
        }
        self.db.execute(
            """
            UPDATE trades
            SET exit_order_list_id = ?,
                exit_order_status = 'placed',
                exit_order_json = ?
            WHERE id = ?
            """,
            (order_list_id, json.dumps(payload, default=str), trade_id),
        )
        self._log("INFO", f"Protective OCO placed for trade {trade_id}", payload)
        return payload

    def _place_momentum_exit_orders(
        self,
        trade_id: int,
        symbol: str,
        filled_quantity: float,
        average_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        entry_order: dict[str, Any],
    ) -> dict[str, Any]:
        if filled_quantity <= 0:
            return self._record_exit_order_failure(trade_id, "Filled quantity was zero; exits were not placed")
        if not (stop_loss < average_price < take_profit_1 < take_profit_2):
            return self._record_exit_order_failure(
                trade_id,
                f"Invalid exit prices: stop={stop_loss}, entry={average_price}, tp1={take_profit_1}, tp2={take_profit_2}",
            )

        net_quantity = min(filled_quantity - self._extract_base_fee(entry_order, symbol), filled_quantity * 0.999)
        qty1 = self.filters.round_quantity_down(symbol, net_quantity * 0.33)
        qty2 = self.filters.round_quantity_down(symbol, net_quantity * 0.33)
        qty3 = self.filters.round_quantity_down(symbol, max(net_quantity - qty1 - qty2, 0))
        if min(qty1, qty2, qty3) <= 0:
            return self._record_exit_order_failure(trade_id, "One or more staged exit quantities rounded to zero")
        if min(qty1, qty2, qty3) * average_price < 6:
            fallback = self._place_protective_oco(
                trade_id,
                symbol,
                filled_quantity,
                average_price,
                stop_loss,
                take_profit_1,
                entry_order,
            )
            fallback["fallback_reason"] = "Staged exit slices were below practical Binance minimum notional"
            return fallback

        orders: list[dict[str, Any]] = []
        try:
            orders.append(self._place_oco_slice(trade_id, symbol, qty1, take_profit_1, stop_loss, "tp1_33"))
            orders.append(self._place_oco_slice(trade_id, symbol, qty2, take_profit_2, stop_loss, "tp2_33"))
            orders.append(self._place_stop_slice(trade_id, symbol, qty3, stop_loss, "runner_34_stop"))
        except Exception as exc:
            self._log(
                "ERROR",
                f"Momentum exit order placement failed for trade {trade_id}",
                {"trade_id": trade_id, "symbol": symbol, "placed_orders": orders, "error": str(exc)},
            )
            fallback = self._try_emergency_stop(
                trade_id,
                symbol,
                max(net_quantity - sum(float(order.get("quantity") or 0) for order in orders), 0),
                stop_loss,
                orders,
                str(exc),
            )
            if fallback.get("status") == "emergency_stop_placed":
                return fallback
            return self._record_exit_order_failure(
                trade_id,
                f"Momentum exit order placement failed: {exc}",
                {"placed_orders": orders},
            )

        payload = {
            "status": "placed",
            "model": "momentum_33_33_34",
            "symbol": symbol,
            "quantity_total": net_quantity,
            "orders": orders,
        }
        self.db.execute(
            """
            UPDATE trades
            SET exit_order_status = 'placed',
                exit_order_json = ?
            WHERE id = ?
            """,
            (json.dumps(payload, default=str), trade_id),
        )
        self._log("INFO", f"Momentum exit orders placed for trade {trade_id}", payload)
        return payload

    def _place_oco_slice(self, trade_id: int, symbol: str, quantity: float, take_profit: float, stop_loss: float, label: str) -> dict[str, Any]:
        exchange = self.market_data.private_exchange()
        params = self._oco_params(symbol, quantity, take_profit, stop_loss)
        raw = exchange.privatePostOrderListOco(params)
        return {"label": label, "type": "OCO", "quantity": quantity, "params": params, "raw_order": raw}

    def _oco_params(self, symbol: str, quantity: float, take_profit: float, stop_loss: float) -> dict[str, Any]:
        market = self.market_data.exchange.market(symbol)
        stop_limit_price = stop_loss * 0.995
        stop_price = self.market_data.exchange.price_to_precision(symbol, stop_loss)
        stop_limit = self.market_data.exchange.price_to_precision(symbol, stop_limit_price)
        if float(stop_limit) >= float(stop_price):
            stop_limit = self.market_data.exchange.price_to_precision(symbol, stop_loss * 0.99)
        return {
            "symbol": market["id"],
            "side": "SELL",
            "quantity": self.market_data.exchange.amount_to_precision(symbol, quantity),
            "aboveType": "LIMIT_MAKER",
            "abovePrice": self.market_data.exchange.price_to_precision(symbol, take_profit),
            "belowType": "STOP_LOSS_LIMIT",
            "belowStopPrice": stop_price,
            "belowPrice": stop_limit,
            "belowTimeInForce": "GTC",
        }

    def _place_stop_slice(self, trade_id: int, symbol: str, quantity: float, stop_loss: float, label: str) -> dict[str, Any]:
        exchange = self.market_data.private_exchange()
        market = self.market_data.exchange.market(symbol)
        stop_limit_price = stop_loss * 0.995
        params = {
            "symbol": market["id"],
            "side": "SELL",
            "type": "STOP_LOSS_LIMIT",
            "timeInForce": "GTC",
            "quantity": self.market_data.exchange.amount_to_precision(symbol, quantity),
            "stopPrice": self.market_data.exchange.price_to_precision(symbol, stop_loss),
            "price": self.market_data.exchange.price_to_precision(symbol, stop_limit_price),
        }
        raw = exchange.privatePostOrder(params)
        return {"label": label, "type": "STOP_LOSS_LIMIT", "quantity": quantity, "params": params, "raw_order": raw}

    def _place_full_stop_only(self, trade_id: int, symbol: str, quantity: float, stop_loss: float) -> dict[str, Any]:
        if quantity <= 0:
            return self._record_exit_order_failure(trade_id, "Retry failed: stop quantity rounded to zero")
        try:
            stop_order = self._place_stop_slice(trade_id, symbol, quantity, stop_loss, "full_protective_stop")
        except Exception as exc:
            return self._record_exit_order_failure(trade_id, f"Full protective stop retry failed: {exc}")
        payload = {"status": "stop_only_placed", "reason": "OCO/staged exits failed; full protective stop placed", "order": stop_order}
        self.db.execute(
            """
            UPDATE trades
            SET exit_order_status = 'stop_only_placed',
                exit_order_json = ?
            WHERE id = ?
            """,
            (json.dumps(payload, default=str), trade_id),
        )
        return payload

    def _try_emergency_stop(
        self,
        trade_id: int,
        symbol: str,
        quantity: float,
        stop_loss: float,
        placed_orders: list[dict[str, Any]],
        original_error: str,
    ) -> dict[str, Any]:
        try:
            quantity = self.filters.round_quantity_down(symbol, quantity)
            if quantity <= 0:
                raise ValueError("No remaining quantity available for emergency stop")
            stop_order = self._place_stop_slice(trade_id, symbol, quantity, stop_loss, "emergency_stop")
            payload = {
                "status": "emergency_stop_placed",
                "reason": "Staged exits failed; emergency stop placed for remaining quantity",
                "original_error": original_error,
                "placed_orders": placed_orders,
                "emergency_stop": stop_order,
            }
            self.db.execute(
                """
                UPDATE trades
                SET exit_order_status = 'emergency_stop_placed',
                    exit_order_json = ?
                WHERE id = ?
                """,
                (json.dumps(payload, default=str), trade_id),
            )
            self._log("ERROR", f"Emergency stop placed for trade {trade_id}", payload)
            return payload
        except Exception as fallback_exc:
            return {
                "status": "failed",
                "reason": "Staged exits failed and emergency stop failed",
                "original_error": original_error,
                "fallback_error": str(fallback_exc),
                "placed_orders": placed_orders,
            }

    def _record_exit_order_failure(
        self,
        trade_id: int,
        reason: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"status": "failed", "reason": reason, "params": params or {}}
        self.db.execute(
            """
            UPDATE trades
            SET exit_order_status = 'failed',
                exit_order_json = ?
            WHERE id = ?
            """,
            (json.dumps(payload, default=str), trade_id),
        )
        return payload

    def _extract_base_fee(self, order: dict[str, Any], symbol: str) -> float:
        base = symbol.split("/")[0]
        fees = order.get("fees") or []
        fee = order.get("fee")
        if fee:
            fees.append(fee)
        fills = order.get("fills") or order.get("info", {}).get("fills") or []
        for fill in fills:
            commission = fill.get("commission")
            commission_asset = fill.get("commissionAsset")
            if commission and commission_asset:
                fees.append({"cost": commission, "currency": commission_asset})
        return sum(
            float(item.get("cost") or 0)
            for item in fees
            if str(item.get("currency") or "").upper() == base
        )

    def _extract_order_list_id(self, order_list: dict[str, Any]) -> str | None:
        for key in ("orderListId", "listClientOrderId", "id"):
            value = order_list.get(key) or order_list.get("info", {}).get(key)
            if value:
                return str(value)
        return None

    def _extract_order_id(self, order: dict[str, Any]) -> str | None:
        order_id = order.get("id")

        if order_id:
            return str(order_id)

        info_order_id = order.get("info", {}).get("orderId")

        if info_order_id:
            return str(info_order_id)

        return None

    def _extract_average_fill_price(self, order: dict[str, Any]) -> float | None:
        average = order.get("average")

        if average:
            return float(average)

        price = order.get("price")

        if price:
            return float(price)

        cost = order.get("cost")
        filled = order.get("filled")

        if cost and filled and float(filled) > 0:
            return float(cost) / float(filled)

        fills = order.get("fills") or order.get("info", {}).get("fills") or []

        total_cost = 0.0
        total_qty = 0.0

        for fill in fills:
            fill_price = float(fill.get("price") or 0)
            fill_qty = float(fill.get("qty") or fill.get("quantity") or 0)

            total_cost += fill_price * fill_qty
            total_qty += fill_qty

        if total_cost > 0 and total_qty > 0:
            return total_cost / total_qty

        return None

    def _extract_filled_quantity(self, order: dict[str, Any]) -> float | None:
        filled = order.get("filled")

        if filled:
            return float(filled)

        amount = order.get("amount")

        if amount:
            return float(amount)

        executed_qty = order.get("info", {}).get("executedQty")

        if executed_qty:
            return float(executed_qty)

        fills = order.get("fills") or order.get("info", {}).get("fills") or []

        total_qty = 0.0

        for fill in fills:
            fill_qty = float(fill.get("qty") or fill.get("quantity") or 0)
            total_qty += fill_qty

        if total_qty > 0:
            return total_qty

        return None

    def _log(
        self,
        level: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO event_log (timestamp, level, message, context)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                level,
                message,
                json.dumps(context or {}, default=str),
            ),
        )
