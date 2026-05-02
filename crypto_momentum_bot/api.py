from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from .db import Database
from .scanner import MomentumScanner
from .scheduler import BotScheduler


class SettingsUpdate(BaseModel):
    account_size: Optional[float] = Field(default=None, ge=0)
    risk_per_trade_percent: Optional[float] = Field(default=None, ge=0.01, le=10)
    scan_interval_minutes: Optional[int] = Field(default=None, ge=5, le=60)
    alerts_enabled: Optional[bool] = None
    alert_method: Optional[str] = None


class ModeUpdate(BaseModel):
    mode: str
    confirmation: Optional[str] = None


def normalize_confirmation(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().strip("\"'").strip()


class CloseTradeRequest(BaseModel):
    exit_price: float = Field(gt=0)
    reason: str = "manual close"


class LiveReadinessRequest(BaseModel):
    include_account_check: bool = False
    confirmation: Optional[str] = None


class TradePermissionTestRequest(BaseModel):
    symbol: str = "DOGE/USDT"
    quote_amount: float = Field(default=6.0, ge=5.0, le=25.0)
    confirmation: Optional[str] = None


class RejectIntentRequest(BaseModel):
    reason: str = "Rejected manually"


def create_router(db: Database, scanner: MomentumScanner, scheduler: BotScheduler) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/status")
    def status():
        live_env_enabled = bool(scanner.market_data.settings.enable_binance_live_trading)
        live_legacy_unlock = bool(scanner.market_data.settings.live_trading_unlocked)
        return {
            "bot_status": db.fetch_one("SELECT * FROM bot_status WHERE id = 1"),
            "settings": db.fetch_one("SELECT * FROM settings WHERE id = 1"),
            "live_gates": {
                "database_enabled": bool(
                    (db.fetch_one("SELECT live_trading_enabled FROM settings WHERE id = 1") or {}).get(
                        "live_trading_enabled"
                    )
                ),
                "env_enabled": live_env_enabled,
                "legacy_unlock": live_legacy_unlock,
                "ready_for_submission": live_env_enabled and live_legacy_unlock,
            },
            "recent_logs": db.fetch_all("SELECT * FROM event_log ORDER BY timestamp DESC LIMIT 20"),
        }

    @router.get("/signals")
    def signals(limit: int = 80):
        status = db.fetch_one("SELECT last_scan_time FROM bot_status WHERE id = 1") or {}
        if status.get("last_scan_time"):
            return db.fetch_all(
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
        return db.fetch_all(
            "SELECT * FROM signals WHERE ignored = 0 ORDER BY score DESC, timestamp DESC LIMIT ?",
            (limit,),
        )

    @router.get("/trades")
    def trades(limit: int = 100):
        return scanner.get_trade_history(limit)

    @router.get("/positions")
    def positions():
        return scanner.get_open_positions()

    @router.get("/performance")
    def performance():
        return scanner.get_performance()

    @router.get("/portfolio")
    def portfolio():
        return scanner.get_portfolio()

    @router.get("/quant/snapshot")
    def quant_snapshot():
        snapshot = scanner.get_quant_snapshot()
        snapshot["scan_status"] = scheduler.scan_status()
        return snapshot

    @router.get("/scan-status")
    def scan_status():
        return scheduler.scan_status()

    @router.get("/live/readiness")
    def live_readiness():
        return scanner.market_data.live_readiness(include_account_check=False)

    @router.get("/live/intents")
    def live_order_intents(limit: int = 100):
        return scanner.get_live_order_intents(limit)

    @router.get("/live/intents/{intent_id}")
    def live_order_intent_detail(intent_id: int):
        try:
            return scanner.get_live_order_intent_detail(intent_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.post("/live/intents/{intent_id}/approve")
    def approve_live_order_intent(intent_id: int):
        try:
            return scanner.live_executor.approve_intent(intent_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/live/intents/{intent_id}/reject")
    def reject_live_order_intent(intent_id: int, req: RejectIntentRequest):
        try:
            scanner.live_executor.reject_intent(intent_id, reason=req.reason)
            return {"status": "rejected"}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/live/intents/{intent_id}/submit")
    def submit_live_order_intent(intent_id: int):
        try:
            return scanner.live_executor.submit_approved_intent(intent_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/live/readiness")
    def live_readiness_check(req: LiveReadinessRequest):
        if req.include_account_check and req.confirmation != "TEST BINANCE ACCOUNT":
            raise HTTPException(
                status_code=400,
                detail="Type TEST BINANCE ACCOUNT to confirm sending configured API credentials to Binance for an account-readiness check.",
            )
        return scanner.market_data.live_readiness(include_account_check=req.include_account_check)

    @router.post("/live/trade-permission-test")
    def live_trade_permission_test(req: TradePermissionTestRequest):
        if req.confirmation != "TEST SPOT TRADE PERMISSION":
            raise HTTPException(
                status_code=400,
                detail="Type TEST SPOT TRADE PERMISSION to confirm sending a Binance /api/v3/order/test market BUY payload. This does not place an order.",
            )
        result = scanner.market_data.test_spot_trade_permission(req.symbol, req.quote_amount)
        scanner.log("WARNING", "Spot trade permission test executed", {"symbol": req.symbol, "ok": result.get("ok")})
        return result

    @router.post("/start")
    def start():
        scheduler.start()
        return {"ok": True, "scan": scheduler.run_now_async()}

    @router.post("/stop")
    def stop():
        scheduler.stop()
        return {"ok": True}

    @router.post("/scan")
    def scan():
        return {"ok": True, "scan": scheduler.run_now_async()}

    @router.post("/settings")
    def update_settings(update: SettingsUpdate):
        current = db.fetch_one("SELECT * FROM settings WHERE id = 1") or {}
        values = {
            "account_size": update.account_size if update.account_size is not None else current["account_size"],
            "risk_per_trade_percent": update.risk_per_trade_percent
            if update.risk_per_trade_percent is not None
            else current["risk_per_trade_percent"],
            "scan_interval_minutes": update.scan_interval_minutes
            if update.scan_interval_minutes is not None
            else current["scan_interval_minutes"],
            "alerts_enabled": int(update.alerts_enabled) if update.alerts_enabled is not None else current["alerts_enabled"],
            "alert_method": update.alert_method if update.alert_method is not None else current["alert_method"],
        }
        db.execute(
            """
            UPDATE settings
            SET account_size = ?, risk_per_trade_percent = ?, scan_interval_minutes = ?,
                alerts_enabled = ?, alert_method = ?
            WHERE id = 1
            """,
            (
                values["account_size"],
                values["risk_per_trade_percent"],
                values["scan_interval_minutes"],
                values["alerts_enabled"],
                values["alert_method"],
            ),
        )
        scheduler.reschedule()
        return db.fetch_one("SELECT * FROM settings WHERE id = 1")

    @router.post("/mode")
    def update_mode(update: ModeUpdate):
        allowed = {"alert-only", "paper-trading", "live-trading"}
        if update.mode not in allowed:
            raise HTTPException(status_code=400, detail="Invalid mode")
        live_enabled = 0
        if update.mode == "live-trading":
            confirmation = normalize_confirmation(update.confirmation)
            if confirmation != "ENABLE LIVE TRADING":
                raise HTTPException(status_code=400, detail="Type ENABLE LIVE TRADING to confirm live mode.")
            if not scanner.market_data.settings.enable_binance_live_trading:
                raise HTTPException(
                    status_code=400,
                    detail="Phrase accepted, but ENABLE_BINANCE_LIVE_TRADING is missing or not 1 in .env. Add ENABLE_BINANCE_LIVE_TRADING=1 and restart the server.",
                )
            if not scanner.market_data.settings.live_trading_unlocked:
                raise HTTPException(
                    status_code=400,
                    detail="Set LIVE_TRADING_UNLOCKED=true in .env and restart the server first.",
                )
            live_enabled = 1
        db.execute("UPDATE bot_status SET mode = ? WHERE id = 1", (update.mode,))
        db.execute("UPDATE settings SET live_trading_enabled = ? WHERE id = 1", (live_enabled,))
        scanner.log("WARNING" if live_enabled else "INFO", f"Mode changed to {update.mode}")
        return db.fetch_one("SELECT * FROM bot_status WHERE id = 1")

    @router.post("/signals/{signal_id}/ignore")
    def ignore_signal(signal_id: int):
        db.execute("UPDATE signals SET ignored = 1 WHERE id = ?", (signal_id,))
        return {"ok": True}

    @router.post("/trades/{trade_id}/close")
    def close_trade(trade_id: int, req: CloseTradeRequest):
        scanner.close_trade(trade_id, req.exit_price, req.reason)
        return {"ok": True}

    @router.get("/trades/export.csv")
    def export_trades():
        csv_data = scanner.export_trades_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=trade_history.csv"},
        )

    return router
