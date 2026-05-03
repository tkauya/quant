from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    binance_api_key: str = ""
    binance_api_secret: str = ""
    database_path: Path = Field(default=Path("data/bot.sqlite3"))
    bot_host: str = "127.0.0.1"
    bot_port: int = 8000
    excluded_bases_extra: str = "TAO"
    enable_binance_live_trading: bool = False
    live_trading_unlocked: bool = False
    auto_place_protective_oco: bool = True
    dashboard_auth_enabled: bool = False
    dashboard_username: str = "admin"
    dashboard_password: str = ""


@lru_cache
def get_settings() -> AppSettings:
    settings = AppSettings()
    try:
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        settings.database_path = Path("data/bot.sqlite3")
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    return settings
