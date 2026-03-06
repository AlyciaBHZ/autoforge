"""Global LLM rate limiting for HPC / harness workloads.

This module provides a dependency-free, cross-process rate limiter that
coordinates through a SQLite database file (often placed on a shared FS).

Why SQLite:
  - Available in stdlib
  - Works across processes and hosts when backed by shared storage
  - Transactional updates for token bucket state

Limits:
  - Best-effort. External providers may still rate-limit independently.
  - Token counts are typically estimated pre-call (input approx + max_tokens).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitSpec:
    enabled: bool
    namespace: str
    requests_per_minute: int
    tokens_per_minute: int
    db_path: Path

    def is_active(self) -> bool:
        if not self.enabled:
            return False
        return (self.requests_per_minute > 0) or (self.tokens_per_minute > 0)


class SqliteRateLimiter:
    """Cross-process RPM/TPM limiter backed by SQLite token buckets."""

    def __init__(self, *, spec: RateLimitSpec) -> None:
        self.spec = spec
        self._db_path = spec.db_path
        self._namespace = (spec.namespace or "global").strip() or "global"
        self._rpm = max(0, int(spec.requests_per_minute or 0))
        self._tpm = max(0, int(spec.tokens_per_minute or 0))
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level=None,  # manual transactions
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS token_bucket (
                        key TEXT PRIMARY KEY,
                        capacity REAL NOT NULL,
                        available REAL NOT NULL,
                        refill_per_sec REAL NOT NULL,
                        last_ts REAL NOT NULL
                    )
                    """
                )
            finally:
                conn.close()
        except Exception as e:
            logger.debug("Rate limiter DB init failed: %s", e)

    def _bucket_key(self, kind: str) -> str:
        return f"{self._namespace}:{kind}"

    def _refill_rate(self, per_minute: int) -> float:
        if per_minute <= 0:
            return 0.0
        return float(per_minute) / 60.0

    def _acquire_once(self, *, req: int, tok: int) -> float:
        if self._rpm <= 0:
            req = 0
        if self._tpm <= 0:
            tok = 0
        if req <= 0 and tok <= 0:
            return 0.0

        now = float(time.time())

        # Prevent impossible-to-satisfy single acquisitions (would block forever).
        if self._tpm > 0 and tok > self._tpm:
            tok = self._tpm

        try:
            conn = self._connect()
        except Exception:
            return 0.25

        try:
            try:
                conn.execute("BEGIN IMMEDIATE;")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    return 0.25
                raise

            wait_s = 0.0
            if req > 0 and self._rpm > 0:
                wait_s = max(wait_s, self._reserve_bucket(
                    conn,
                    key=self._bucket_key("req"),
                    need=float(req),
                    capacity=float(self._rpm),
                    refill_per_sec=self._refill_rate(self._rpm),
                    now=now,
                ))

            if tok > 0 and self._tpm > 0:
                wait_s = max(wait_s, self._reserve_bucket(
                    conn,
                    key=self._bucket_key("tok"),
                    need=float(tok),
                    capacity=float(self._tpm),
                    refill_per_sec=self._refill_rate(self._tpm),
                    now=now,
                ))

            conn.execute("COMMIT;")
            return float(wait_s)
        except sqlite3.OperationalError as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            if "locked" in str(e).lower():
                return 0.25
            logger.debug("Rate limiter sqlite operational error: %s", e)
            return 0.5
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            logger.debug("Rate limiter acquire failed: %s", e)
            return 0.5
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _reserve_bucket(
        self,
        conn: sqlite3.Connection,
        *,
        key: str,
        need: float,
        capacity: float,
        refill_per_sec: float,
        now: float,
    ) -> float:
        if need <= 0.0 or capacity <= 0.0 or refill_per_sec <= 0.0:
            return 0.0

        row = conn.execute(
            "SELECT capacity, available, refill_per_sec, last_ts FROM token_bucket WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            # Initialize full.
            conn.execute(
                "INSERT INTO token_bucket(key, capacity, available, refill_per_sec, last_ts) VALUES (?,?,?,?,?)",
                (key, float(capacity), float(capacity), float(refill_per_sec), float(now)),
            )
            available = float(capacity)
            last_ts = float(now)
        else:
            _cap, available, _rps, last_ts = row
            available = float(available)
            last_ts = float(last_ts)

        # Refill using current config rates/capacity.
        available = min(float(capacity), available + max(0.0, now - last_ts) * float(refill_per_sec))

        if available + 1e-9 < need:
            # Not enough: write back the refilled state and return wait time.
            conn.execute(
                "UPDATE token_bucket SET capacity=?, available=?, refill_per_sec=?, last_ts=? WHERE key=?",
                (float(capacity), float(available), float(refill_per_sec), float(now), key),
            )
            deficit = max(0.0, float(need) - float(available))
            return float(deficit) / float(refill_per_sec)

        # Reserve.
        available = float(available) - float(need)
        conn.execute(
            "UPDATE token_bucket SET capacity=?, available=?, refill_per_sec=?, last_ts=? WHERE key=?",
            (float(capacity), float(available), float(refill_per_sec), float(now), key),
        )
        return 0.0

    async def acquire(self, *, estimated_tokens: int, requests: int = 1) -> None:
        """Block until capacity is available, then reserve.

        estimated_tokens should be a conservative estimate. A good default is:
          (estimated_input_tokens + max_tokens)
        """
        if not self.spec.is_active():
            return
        req = max(0, int(requests or 0))
        tok = max(0, int(estimated_tokens or 0))
        if req <= 0 and tok <= 0:
            return

        while True:
            wait_s = await asyncio.to_thread(self._acquire_once, req=req, tok=tok)
            if wait_s <= 0.0:
                return
            await asyncio.sleep(max(0.05, min(10.0, float(wait_s))))

