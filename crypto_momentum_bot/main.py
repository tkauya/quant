from __future__ import annotations

import logging
import base64
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .db import Database
from .market_data import BinanceMarketData
from .scanner import MomentumScanner
from .scheduler import BotScheduler
from .settings import get_settings

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler("logs/bot.log"), logging.StreamHandler()],
)

settings = get_settings()
db = Database(settings.database_path)
market_data = BinanceMarketData(settings)
scanner = MomentumScanner(db, market_data)
scheduler = BotScheduler(db, scanner.scan)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    scheduler.shutdown()


app = FastAPI(title="Binance Momentum Alert Bot", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def require_dashboard_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    if not settings.dashboard_auth_enabled:
        return await call_next(request)

    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")

    authenticated = False
    if scheme.lower() == "basic" and token:
        try:
            decoded = base64.b64decode(token).decode("utf-8")
            username, _, password = decoded.partition(":")
            authenticated = secrets.compare_digest(username, settings.dashboard_username) and secrets.compare_digest(
                password, settings.dashboard_password
            )
        except Exception:
            authenticated = False

    if authenticated:
        return await call_next(request)

    return Response(
        content="Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Binance Momentum Bot"'},
    )


app.include_router(create_router(db, scanner, scheduler))
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")
