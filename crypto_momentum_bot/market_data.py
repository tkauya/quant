from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from .settings import AppSettings

logger = logging.getLogger(__name__)


EXCLUDED_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "DAI",
    "USDP",
    "EUR",
    "GBP",
    "TRY",
    "BRL",
    "AUD",
    "AEUR",
    "EURI",
    "PAXG",
    "USDE",
    "USDS",
    "SUSD",
    "USD1",
    "RLUSD",
    "XUSD",
    "XAUT",
    "WBETH",
    "BETH",
    "WBTC",
    "TAO",
    "U",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")
STANDARD_BASE_RE = re.compile(r"^[A-Z0-9]{2,12}$")
UNUSUAL_SYMBOL_PARTS = (
    "UP/USDT",
    "DOWN/USDT",
    "BULL/USDT",
    "BEAR/USDT",
)


@dataclass
class MarketTicker:
    symbol: str
    base: str
    quote_volume: float
    last: float


class BinanceMarketData:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.user_excluded_bases = {
            base.strip().upper()
            for base in settings.excluded_bases_extra.split(",")
            if base.strip()
        }
        self.exchange = ccxt.binance(
            {
                "apiKey": settings.binance_api_key or None,
                "secret": settings.binance_api_secret or None,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self._markets_lock = threading.Lock()

    def ensure_markets_loaded(self) -> None:
        if self.exchange.markets:
            return
        with self._markets_lock:
            if not self.exchange.markets:
                self.exchange.load_markets()

    def credential_status(self) -> dict[str, Any]:
        api_key = self.settings.binance_api_key or ""
        secret = self.settings.binance_api_secret or ""
        return {
            "api_key_configured": bool(api_key),
            "api_secret_configured": bool(secret),
            "api_key_hint": self._mask_secret(api_key),
            "credentials_complete": bool(api_key and secret),
        }

    def live_readiness(self, include_account_check: bool = False) -> dict[str, Any]:
        credential_status = self.credential_status()
        checks = [
            {
                "name": "API key configured",
                "ok": credential_status["api_key_configured"],
                "detail": "BINANCE_API_KEY is present" if credential_status["api_key_configured"] else "Add BINANCE_API_KEY to .env",
            },
            {
                "name": "API secret configured",
                "ok": credential_status["api_secret_configured"],
                "detail": "BINANCE_API_SECRET is present" if credential_status["api_secret_configured"] else "Add BINANCE_API_SECRET to .env",
            },
            {
                "name": "Live order executor",
                "ok": bool(self.settings.enable_binance_live_trading),
                "detail": "ENABLE_BINANCE_LIVE_TRADING=1"
                if self.settings.enable_binance_live_trading
                else "ENABLE_BINANCE_LIVE_TRADING is false",
            },
            {
                "name": "Live trading unlock",
                "ok": bool(self.settings.live_trading_unlocked),
                "detail": "LIVE_TRADING_UNLOCKED=true" if self.settings.live_trading_unlocked else "LIVE_TRADING_UNLOCKED is false",
            },
        ]
        account_check: dict[str, Any] = {
            "requested": include_account_check,
            "ok": None,
            "detail": "Not run",
        }
        if include_account_check:
            if not credential_status["credentials_complete"]:
                account_check = {"requested": True, "ok": False, "detail": "Missing API key or secret"}
            else:
                try:
                    balance = self.exchange.fetch_balance()
                    non_zero_assets = sum(
                        1
                        for value in (balance.get("total") or {}).values()
                        if isinstance(value, (int, float)) and value > 0
                    )
                    account_check = {
                        "requested": True,
                        "ok": True,
                        "detail": f"Authenticated account check passed; non-zero assets: {non_zero_assets}",
                    }
                except Exception as exc:
                    account_check = {"requested": True, "ok": False, "detail": str(exc)}
        return {
            "credentials": credential_status,
            "checks": checks,
            "account_check": account_check,
            "trade_permission_test": {"requested": False, "ok": None, "detail": "Not run"},
            "live_order_execution_enabled": bool(
                self.settings.enable_binance_live_trading and self.settings.live_trading_unlocked
            ),
            "mode_note": "Live submission requires env gates, DB live flag, and an approved intent.",
        }

    def _mask_secret(self, value: str) -> str:
        if not value:
            return ""
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"

    def test_spot_trade_permission(self, symbol: str = "DOGE/USDT", quote_amount: float = 6.0) -> dict[str, Any]:
        credential_status = self.credential_status()
        if not credential_status["credentials_complete"]:
            return {"requested": True, "ok": False, "detail": "Missing API key or secret"}

        self.ensure_markets_loaded()
        market = self.exchange.market(symbol)
        if not self._is_tradeable_usdt_spot(symbol, market):
            return {"requested": True, "ok": False, "detail": f"{symbol} is not an allowed active spot USDT market"}

        params = {
            "symbol": market["id"],
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": self.exchange.amount_to_precision(symbol, quote_amount),
        }
        try:
            self.exchange.privatePostOrderTest(params)
            return {
                "requested": True,
                "ok": True,
                "detail": f"Binance /api/v3/order/test accepted a spot market BUY test for {symbol}. No order was placed.",
            }
        except Exception as exc:
            return {
                "requested": True,
                "ok": False,
                "detail": f"Binance order/test rejected the request: {exc}",
            }

    def load_top_usdt_pairs(self, limit: int = 40) -> list[MarketTicker]:
        markets = self.exchange.load_markets()
        tickers: dict[str, dict[str, Any]] = self.exchange.fetch_tickers()
        candidates: list[MarketTicker] = []

        for symbol, market in markets.items():
            if not self._is_tradeable_usdt_spot(symbol, market):
                continue
            base = str(market.get("base", ""))
            ticker = tickers.get(symbol) or {}
            quote_volume = float(ticker.get("quoteVolume") or 0)
            last = float(ticker.get("last") or ticker.get("close") or 0)
            if quote_volume <= 0 or last <= 0:
                continue
            candidates.append(MarketTicker(symbol=symbol, base=base, quote_volume=quote_volume, last=last))

        candidates.sort(key=lambda item: item.quote_volume, reverse=True)
        return candidates[:limit]

    def _is_tradeable_usdt_spot(self, symbol: str, market: dict[str, Any]) -> bool:
        base = str(market.get("base", ""))
        info = market.get("info") or {}
        permissions = set(info.get("permissions") or [])

        if market.get("quote") != "USDT":
            return False
        if not market.get("spot") or not market.get("active"):
            return False
        if market.get("swap") or market.get("future") or market.get("option") or market.get("contract"):
            return False
        if info.get("status") not in (None, "TRADING"):
            return False
        if info.get("isSpotTradingAllowed") is False:
            return False
        if permissions and "SPOT" not in permissions:
            return False
        if base in EXCLUDED_BASES or base in self.user_excluded_bases or base.endswith(LEVERAGED_SUFFIXES):
            return False
        if not STANDARD_BASE_RE.fullmatch(base):
            return False
        if any(part in symbol for part in UNUSUAL_SYMBOL_PARTS):
            return False
        return True

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 210) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna().reset_index(drop=True)

    def fetch_last_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        tickers = self.exchange.fetch_tickers(symbols)
        prices: dict[str, float] = {}
        for symbol, ticker in tickers.items():
            last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            if last:
                prices[symbol] = float(last)
        return prices
