from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from .db import Database

logger = logging.getLogger(__name__)


class BotScheduler:
    def __init__(self, db: Database, scan_callable):
        self.db = db
        self.scan_callable = scan_callable
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self._scan_lock = threading.Lock()
        self._scan_status: dict[str, Any] = {
            "is_scanning": False,
            "last_started_at": None,
            "last_finished_at": None,
            "last_error": None,
            "last_result": None,
        }

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
                self._run_scan_sync,
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

    def run_now_async(self) -> dict[str, Any]:
        if not self._scan_lock.acquire(blocking=False):
            return {"queued": False, "reason": "scan already running", "scan_status": self.scan_status()}
        started = datetime.now(timezone.utc).isoformat()
        self._scan_status.update(
            {
                "is_scanning": True,
                "last_started_at": started,
                "last_error": None,
            }
        )
        thread = threading.Thread(target=self._run_scan_with_lock, name="market-scan", daemon=True)
        thread.start()
        return {"queued": True, "scan_status": self.scan_status()}

    def scan_status(self) -> dict[str, Any]:
        return dict(self._scan_status)

    def _run_scan_sync(self) -> None:
        if not self._scan_lock.acquire(blocking=False):
            return
        started = datetime.now(timezone.utc).isoformat()
        self._scan_status.update(
            {
                "is_scanning": True,
                "last_started_at": started,
                "last_error": None,
            }
        )
        self._run_scan_with_lock()

    def _run_scan_with_lock(self) -> None:
        try:
            result = self.scan_callable()
            self._scan_status["last_result"] = result
        except Exception as exc:
            logger.exception("Scheduled scan failed")
            self._scan_status["last_error"] = str(exc)
        finally:
            self._scan_status["is_scanning"] = False
            self._scan_status["last_finished_at"] = datetime.now(timezone.utc).isoformat()
            self._scan_lock.release()
