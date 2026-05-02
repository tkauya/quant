from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .db import Database

logger = logging.getLogger(__name__)


class BotScheduler:
    def __init__(self, db: Database, scan_callable):
        self.db = db
        self.scan_callable = scan_callable
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
        self.reschedule()
        self.db.execute("UPDATE bot_status SET is_running = 1 WHERE id = 1")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.remove_all_jobs()
        self.db.execute("UPDATE bot_status SET is_running = 0 WHERE id = 1")

    def reschedule(self) -> None:
        settings = self.db.fetch_one("SELECT scan_interval_minutes FROM settings WHERE id = 1") or {}
        minutes = int(settings.get("scan_interval_minutes") or 10)
        if self.scheduler.running:
            self.scheduler.add_job(
                self.scan_callable,
                "interval",
                minutes=minutes,
                id="market_scan",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                next_run_time=None,
            )

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
