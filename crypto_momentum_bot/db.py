from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    volume_rank INTEGER NOT NULL,
    price REAL NOT NULL,
    score REAL NOT NULL,
    tier TEXT NOT NULL,
    trend_status TEXT NOT NULL,
    volume_status TEXT NOT NULL,
    obv_status TEXT NOT NULL,
    rsi REAL,
    macd_status TEXT NOT NULL,
    relative_strength TEXT NOT NULL,
    entry_zone TEXT NOT NULL,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    risk_reward REAL,
    notes TEXT NOT NULL DEFAULT '',
    ignored INTEGER NOT NULL DEFAULT 0,
    score_breakdown_json TEXT NOT NULL DEFAULT '{}',
    ev_json TEXT NOT NULL DEFAULT '{}',
    setup_type TEXT NOT NULL DEFAULT '',
    volume_ratio REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    exit_time TEXT,
    exit_price REAL,
    status TEXT NOT NULL,
    pnl_percent REAL,
    pnl_usdt REAL,
    entry_reason TEXT NOT NULL DEFAULT '',
    exit_reason TEXT NOT NULL DEFAULT '',
    score_at_entry REAL,
    position_size REAL,
    exchange_order_id TEXT,
    exit_order_list_id TEXT,
    exit_order_status TEXT NOT NULL DEFAULT '',
    exit_order_json TEXT NOT NULL DEFAULT '{}',
    remaining_position_size REAL,
    tp1_filled INTEGER NOT NULL DEFAULT 0,
    tp2_filled INTEGER NOT NULL DEFAULT 0,
    runner_stop REAL,
    high_water_price REAL,
    execution_type TEXT NOT NULL DEFAULT 'paper',
    submitted_at TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS bot_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    is_running INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'alert-only',
    last_scan_time TEXT,
    coins_scanned INTEGER NOT NULL DEFAULT 0,
    market_regime TEXT NOT NULL DEFAULT 'unknown',
    btc_trend TEXT NOT NULL DEFAULT 'unknown',
    eth_trend TEXT NOT NULL DEFAULT 'unknown',
    errors TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    account_size REAL NOT NULL DEFAULT 1000,
    risk_per_trade_percent REAL NOT NULL DEFAULT 1,
    scan_interval_minutes INTEGER NOT NULL DEFAULT 10,
    alerts_enabled INTEGER NOT NULL DEFAULT 0,
    alert_method TEXT NOT NULL DEFAULT 'none',
    live_trading_enabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS live_order_intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quote_amount REAL NOT NULL,
    base_quantity REAL,
    reference_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    status TEXT NOT NULL DEFAULT 'draft',
    validation_json TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    ev_json TEXT NOT NULL DEFAULT '{}',
    exchange_order_id TEXT,
    submitted_at TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);

INSERT OR IGNORE INTO bot_status (id) VALUES (1);
INSERT OR IGNORE INTO settings (id) VALUES (1);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_live_order_intents_status ON live_order_intents(status);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        migrations = {
            "signals": {
                "score_breakdown_json": "TEXT NOT NULL DEFAULT '{}'",
                "ev_json": "TEXT NOT NULL DEFAULT '{}'",
                "setup_type": "TEXT NOT NULL DEFAULT ''",
                "volume_ratio": "REAL",
            },
            "trades": {
                "exchange_order_id": "TEXT",
                "exit_order_list_id": "TEXT",
                "exit_order_status": "TEXT NOT NULL DEFAULT ''",
                "exit_order_json": "TEXT NOT NULL DEFAULT '{}'",
                "remaining_position_size": "REAL",
                "tp1_filled": "INTEGER NOT NULL DEFAULT 0",
                "tp2_filled": "INTEGER NOT NULL DEFAULT 0",
                "runner_stop": "REAL",
                "high_water_price": "REAL",
                "execution_type": "TEXT NOT NULL DEFAULT 'paper'",
                "submitted_at": "TEXT",
            },
            "live_order_intents": {
                "ev_json": "TEXT NOT NULL DEFAULT '{}'",
                "exchange_order_id": "TEXT",
                "submitted_at": "TEXT",
            },
        }

        for table, columns in migrations.items():
            existing = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, ddl in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def fetch_one(self, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return dict(row) if row else None

    def fetch_all(self, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, tuple(params)).fetchall()]

    def execute(self, query: str, params: Iterable[Any] = ()) -> int:
        with self.connect() as conn:
            cur = conn.execute(query, tuple(params))
            return int(cur.lastrowid or 0)

    def executemany(self, query: str, params: list[Iterable[Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(query, params)
