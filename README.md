# Binance Momentum Alert Bot

Alert-only / paper-trading Binance spot scanner for the top 40 USDT pairs by 24h quote volume.

## Architecture

- `crypto_momentum_bot/main.py` - FastAPI app, lifecycle, dashboard route.
- `crypto_momentum_bot/api.py` - JSON API and dashboard controls.
- `crypto_momentum_bot/scanner.py` - top-volume universe selection, indicators, scoring, signal persistence, paper-trade creation.
- `crypto_momentum_bot/market_data.py` - Binance spot data via `ccxt`.
- `crypto_momentum_bot/indicators.py` - EMA, RSI, MACD, OBV, ATR helpers.
- `crypto_momentum_bot/scoring.py` - 100-point signal model and tier classification.
- `crypto_momentum_bot/quant.py` - heuristic EV, risk/reward, and quant dashboard helpers.
- `crypto_momentum_bot/db.py` - SQLite schema, migrations, query helpers.
- `crypto_momentum_bot/settings.py` - environment and runtime config.
- `crypto_momentum_bot/static/dashboard.html` - lightweight dashboard.

## Safety Defaults

- Starts in `alert-only` mode.
- `live_trading_enabled` is `0` by default.
- `live-trading` requires `ENABLE_BINANCE_LIVE_TRADING=1`, `LIVE_TRADING_UNLOCKED=true`, dashboard confirmation, and an approved live intent.
- Live order intents start as `draft`; they do not submit until approved and explicitly submitted.
- Position size is capped by configured risk percentage and stop-loss distance.

## Live Readiness And Execution

The dashboard includes a **Live Readiness** panel and a **Live Order Intents** table.

To check Binance API readiness:

1. Create restricted Binance API keys.
2. Prefer read-only keys for account-readiness testing, then add spot trading only when you are ready.
3. Add keys to `.env`:

```bash
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
ENABLE_BINANCE_LIVE_TRADING=0
LIVE_TRADING_UNLOCKED=false
```

4. Restart the app.
5. Use **Refresh Readiness**.
6. Use **Test Binance Account** only when you are comfortable sending the configured API credentials to Binance for an account-readiness check.

Live mode requires both env gates:

```env
ENABLE_BINANCE_LIVE_TRADING=1
LIVE_TRADING_UNLOCKED=true
```

`crypto_momentum_bot/live_execution.py` contains the gated live execution flow:

- `OrderIntent` models a proposed order.
- `BinanceFilterHelper` validates lot-size/min-notional style filters.
- `RiskGuard` checks account-size and risk-per-trade constraints.
- `GatedLiveExecutor` builds, validates, approves, rejects, and submits approved live intents.
- `live_order_intents` stores future order-intent drafts/audit records.
- During scans, qualifying signals (`score >= 70`) create draft live-order intents using the same sizing inputs as paper trades.
- Draft intents can move to `approved` or `rejected` through `/api/live/intents/{id}/approve` and `/api/live/intents/{id}/reject`.
- `/api/live/intents/{id}/submit` submits only approved, valid intents while the live gates are enabled.
- `/api/quant/snapshot` powers the internal quant view with score breakdowns, heuristic EV, lifecycle counts, and enriched live intents.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

API keys are optional for public market-data scanning. Keep live-trading disabled unless you intentionally extend the project later.

## Run Locally

```bash
source .venv/bin/activate
uvicorn crypto_momentum_bot.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Use **Refresh Scan** to run a scan immediately. Use **Start Bot** to schedule repeated scans every `scan_interval_minutes`.

## Main Endpoints

- `GET /api/status`
- `GET /api/signals`
- `GET /api/trades`
- `GET /api/positions`
- `GET /api/performance`
- `GET /api/quant/snapshot`
- `GET /api/live/readiness`
- `GET /api/live/intents/{id}`
- `POST /api/live/readiness`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/scan`
- `POST /api/settings`
- `POST /api/mode`
- `POST /api/signals/{id}/ignore`
- `POST /api/trades/{id}/close`
- `GET /api/trades/export.csv`

## Database Schema

The SQLite schema is created automatically on startup. See `crypto_momentum_bot/db.py` for the full DDL.

## VPS Notes

For phone access, do not expose the dashboard without authentication.

Recommended simple path:

1. Use a small VPS.
2. Copy the project to the VPS.
3. Set `.env` with Binance keys, live gates, and dashboard auth:

```env
BOT_HOST=0.0.0.0
BOT_PORT=8000
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=use_a_long_random_password
ENABLE_BINANCE_LIVE_TRADING=1
LIVE_TRADING_UNLOCKED=true
```

4. Run the app with `uvicorn crypto_momentum_bot.main:app --host 0.0.0.0 --port 8000`.
5. Put Caddy or Nginx in front of it with HTTPS.
6. Persist `data/bot.sqlite3` so trade history and settings survive restarts.

Safer private option: use Tailscale on the VPS and phone, then keep the bot off the public internet entirely.

### Common VPS Commands

From your computer, SSH into the VPS:

```bash
ssh root@YOUR_VPS_IP
```

Then update and restart the bot:

```bash
cd /root/quant
git pull
systemctl restart quant-bot
```

Check service status:

```bash
systemctl status quant-bot --no-pager
```

View recent logs:

```bash
journalctl -u quant-bot -n 80 --no-pager
```

If the dashboard looks stale after code changes, restart the service and refresh the browser:

```bash
systemctl restart quant-bot
```

## Railway Deployment

Railway is a good fit for phone access because it gives you a public HTTPS URL and simple environment-variable management.

This repo includes:

- `railway.json` with the Railway start command.
- `.python-version` for Nixpacks.
- `/health` for unauthenticated service checks.
- Basic Auth for the dashboard and API when `DASHBOARD_AUTH_ENABLED=true`.

Railway setup:

1. Push this project to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Add a Railway volume and mount it at `/data`.
4. Set these Railway variables:

```env
DATABASE_PATH=/data/bot.sqlite3
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=use_a_long_random_password
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
ENABLE_BINANCE_LIVE_TRADING=1
LIVE_TRADING_UNLOCKED=true
AUTO_PLACE_PROTECTIVE_OCO=true
EXCLUDED_BASES_EXTRA=TAO
```

5. Deploy. Railway will run:

```bash
uvicorn crypto_momentum_bot.main:app --host 0.0.0.0 --port $PORT
```

6. Open the Railway domain on your phone and sign in with the dashboard username/password.

Do not use Railway without `DASHBOARD_AUTH_ENABLED=true`. The dashboard contains live order controls.
