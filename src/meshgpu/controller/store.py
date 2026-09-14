"""SQLite-backed metadata store for controller state."""
from __future__ import annotations

import hmac
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Store:
    """
    Single-writer SQLite store. All writes go through one lock; reads are
    read-only cursors. Suitable for MVP single-controller deployment.
    """

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(database_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                incarnation TEXT NOT NULL,
                credential TEXT NOT NULL,
                expires_at REAL NOT NULL,
                capability_json TEXT NOT NULL,
                last_heartbeat REAL NOT NULL DEFAULT 0,
                suspected INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                task TEXT NOT NULL,
                backend TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                lease_epoch INTEGER NOT NULL DEFAULT 1,
                placement_version INTEGER NOT NULL DEFAULT 1,
                placement_json TEXT,
                checkpoint_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leases (
                job_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                lease_epoch INTEGER NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (job_id, worker_id)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                global_step INTEGER NOT NULL,
                manifest_json TEXT NOT NULL,
                committed_at REAL NOT NULL
            );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def upsert_worker(
        self,
        worker_id: str,
        incarnation: str,
        credential: str,
        expires_at: float,
        capability: dict,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO workers (worker_id, incarnation, credential, expires_at,
                                     capability_json, last_heartbeat)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    incarnation=excluded.incarnation,
                    credential=excluded.credential,
                    expires_at=excluded.expires_at,
                    capability_json=excluded.capability_json,
                    last_heartbeat=excluded.last_heartbeat,
                    suspected=0
                """,
                (
                    worker_id,
                    incarnation,
                    credential,
                    expires_at,
                    json.dumps(capability),
                    time.time(),
                ),
            )
            self._conn.commit()

    def authenticate_worker(self, worker_id: str, credential: str) -> bool:
        """Validate a worker credential and its expiration time."""
        with self._lock:
            row = self._conn.execute(
                "SELECT credential, expires_at FROM workers WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
        if not row or time.time() >= row["expires_at"]:
            return False
        return hmac.compare_digest(str(row["credential"]), credential)

    def get_worker(self, worker_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
        return dict(row) if row else None

    def heartbeat(
        self,
        worker_id: str,
        incarnation: str,
        credential: str | None = None,
    ) -> bool:
        """Update last_heartbeat if identity and (when supplied) credential match."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT incarnation, credential, expires_at FROM workers WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
            if (
                not row
                or row["incarnation"] != incarnation
                or now >= row["expires_at"]
                or (
                    credential is not None
                    and not hmac.compare_digest(str(row["credential"]), credential)
                )
            ):
                return False
            self._conn.execute(
                "UPDATE workers SET last_heartbeat=?, suspected=0 WHERE worker_id=?",
                (now, worker_id),
            )
            self._conn.commit()
            return True

    def mark_suspected(self, worker_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE workers SET suspected=1 WHERE worker_id=?", (worker_id,)
            )
            self._conn.commit()

    def list_workers(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM workers").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(self, spec: dict) -> str:
        now = time.time()
        job_id = spec["job_id"]
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (job_id, job_name, state, task, backend,
                                  spec_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    spec["job_name"],
                    "pending",
                    spec["task"],
                    spec["backend"],
                    json.dumps(spec),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def transition_job(
        self,
        job_id: str,
        new_state: str,
        *,
        expected_state: str | None = None,
        reason: str = "",
        placement: dict | None = None,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Transition a job, optionally using an atomic expected-state check.

        ``expected_state`` turns this into a compare-and-swap operation.  The
        scheduler uses that form for admission/cancellation so concurrent HTTP
        calls cannot both advance the same job or issue duplicate leases.
        """
        now = time.time()
        with self._lock:
            updates: list[Any] = [new_state, now]
            extra = ""
            if placement is not None:
                extra += ", placement_json=?"
                updates.append(json.dumps(placement))
            if checkpoint_id is not None:
                extra += ", checkpoint_id=?"
                updates.append(checkpoint_id)
            updates.append(job_id)
            where = " WHERE job_id=?"
            if expected_state is not None:
                where += " AND state=?"
                updates.append(getattr(expected_state, "value", expected_state))
            self._conn.execute(
                f"UPDATE jobs SET state=?, updated_at=?{extra}{where}",
                updates,
            )
            changed = self._conn.execute("SELECT changes()").fetchone()[0]
            self._conn.commit()
            return bool(changed)

    def bump_lease_epoch(self, job_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT lease_epoch FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            new_epoch = row["lease_epoch"] + 1
            self._conn.execute(
                "UPDATE jobs SET lease_epoch=? WHERE job_id=?", (new_epoch, job_id)
            )
            self._conn.commit()
            return new_epoch

    def list_jobs(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Leases
    # ------------------------------------------------------------------

    def grant_lease(
        self, job_id: str, worker_id: str, lease_epoch: int, duration_s: float
    ) -> float:
        expires_at = time.time() + duration_s
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO leases (job_id, worker_id, lease_epoch, expires_at)
                VALUES (?,?,?,?)
                ON CONFLICT(job_id, worker_id) DO UPDATE SET
                    lease_epoch=excluded.lease_epoch,
                    expires_at=excluded.expires_at
                """,
                (job_id, worker_id, lease_epoch, expires_at),
            )
            self._conn.commit()
        return expires_at

    def revoke_leases(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE job_id=?", (job_id,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def commit_checkpoint(
        self,
        checkpoint_id: str,
        job_id: str,
        global_step: int,
        manifest: dict,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO checkpoints (checkpoint_id, job_id, global_step,
                                         manifest_json, committed_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    checkpoint_id,
                    job_id,
                    global_step,
                    json.dumps(manifest),
                    time.time(),
                ),
            )
            self._conn.execute(
                "UPDATE jobs SET checkpoint_id=? WHERE job_id=?",
                (checkpoint_id, job_id),
            )
            self._conn.commit()

    def get_latest_checkpoint(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM checkpoints WHERE job_id=?
                ORDER BY committed_at DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
