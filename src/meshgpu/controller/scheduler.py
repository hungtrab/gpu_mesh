"""Job scheduler: admission, placement, lease grant and heartbeat monitoring."""
from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
from collections.abc import Callable

from meshgpu.controller.store import Store
from meshgpu.protocol.messages import (
    CapabilityReport,
    JobSpec,
    JobState,
    LeaseGrant,
)

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 5.0
SUSPECT_AFTER_MISSED = 3
LEASE_DURATION_S = 60.0
CREDENTIAL_DURATION_S = 24 * 3600


class Scheduler:
    def __init__(
        self,
        store: Store,
        on_worker_lost: Callable[[str, str], None] | None = None,
        join_token: str | None = None,
    ) -> None:
        if join_token == "":
            raise ValueError("join_token must be non-empty or None for dev mode")
        self._store = store
        self._on_worker_lost = on_worker_lost
        self._join_token = join_token
        self._missed: dict[str, int] = {}  # worker_id → consecutive missed heartbeats

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def register_worker(self, join_token: str, capability: CapabilityReport) -> dict:
        """Validate join token, issue credential and record worker."""
        if not isinstance(join_token, str) or not join_token:
            raise PermissionError("join token must be non-empty")
        # A configured token is pre-shared for the MVP.  ``None`` preserves a
        # deliberately insecure local-dev mode, which the HTTP app warns about.
        if self._join_token is not None and not hmac.compare_digest(
            join_token, self._join_token
        ):
            raise PermissionError("invalid join token")
        credential = secrets.token_urlsafe(32)
        expires_at = time.time() + CREDENTIAL_DURATION_S
        self._store.upsert_worker(
            worker_id=capability.worker_id,
            incarnation=capability.worker_incarnation,
            credential=credential,
            expires_at=expires_at,
            capability=capability.model_dump(),
        )
        log.info("registered worker %s (%s)", capability.worker_id, capability.provider)
        return {
            "worker_id": capability.worker_id,
            "credential": credential,
            "expires_at": expires_at,
        }

    # ------------------------------------------------------------------
    # Heartbeat processing
    # ------------------------------------------------------------------

    def process_heartbeat(
        self,
        worker_id: str,
        incarnation: str,
        credential: str | None = None,
    ) -> bool:
        ok = self._store.heartbeat(worker_id, incarnation, credential)
        if ok:
            self._missed[worker_id] = 0
        return ok

    async def run_heartbeat_monitor(self) -> None:
        """Background task: mark workers as suspected after missed heartbeats."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            threshold = time.time() - HEARTBEAT_INTERVAL_S * SUSPECT_AFTER_MISSED
            for w in self._store.list_workers():
                wid = w["worker_id"]
                if w["last_heartbeat"] < threshold and not w["suspected"]:
                    missed = self._missed.get(wid, 0) + 1
                    self._missed[wid] = missed
                    if missed >= SUSPECT_AFTER_MISSED:
                        log.warning("worker %s suspected (missed %d heartbeats)", wid, missed)
                        self._store.mark_suspected(wid)
                        if self._on_worker_lost:
                            self._on_worker_lost(wid, w["incarnation"])

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def submit_job(self, spec: JobSpec) -> str:
        """Create job record in PENDING state; return job_id."""
        job_id = self._store.create_job(spec.model_dump(mode="json"))
        log.info("job %s submitted (%s/%s)", job_id, spec.task, spec.backend)
        return job_id

    def admit_job(self, job_id: str) -> LeaseGrant | None:
        """
        Move PENDING → PREFLIGHT → RESERVED; issue leases to placement workers.
        Returns LeaseGrant to be forwarded to agents, or None if not feasible.
        """
        job = self._store.get_job(job_id)
        if not job:
            return None
        if job["state"] != JobState.PENDING:
            return None

        # Claim the pending job atomically.  A second concurrent admission call
        # must lose this CAS and return without issuing another lease epoch.
        if not self._store.transition_job(
            job_id,
            JobState.PREFLIGHT,
            expected_state=JobState.PENDING,
            reason="scheduler_admit",
        ):
            return None
        spec = JobSpec.model_validate_json(job["spec_json"])

        # Verify all placement workers are registered and not suspected
        for entry in spec.placement:
            w = self._store.get_worker(entry.worker_id)
            if not w or w["suspected"] or time.time() >= float(w["expires_at"]):
                log.warning("job %s: worker %s unavailable", job_id, entry.worker_id)
                self._store.transition_job(
                    job_id,
                    JobState.FAILED,
                    expected_state=JobState.PREFLIGHT,
                    reason="worker_unavailable",
                )
                return None

        lease_epoch = self._store.bump_lease_epoch(job_id)
        expires_at = time.time() + LEASE_DURATION_S

        for entry in spec.placement:
            self._store.grant_lease(
                job_id, entry.worker_id, lease_epoch, LEASE_DURATION_S
            )

        if not self._store.transition_job(
            job_id,
            JobState.RESERVED,
            expected_state=JobState.PREFLIGHT,
            reason="leases_granted",
        ):
            # Cancellation (or another terminal transition) won the race
            # while leases were being written.  Never leave those leases live.
            self._store.revoke_leases(job_id)
            return None
        log.info("job %s reserved (epoch=%d)", job_id, lease_epoch)

        return LeaseGrant(
            job_id=job_id,
            lease_epoch=lease_epoch,
            placement_version=spec.placement_version,
            job_spec=spec,
            expires_at=expires_at,
        )

    def cancel_job(self, job_id: str, reason: str = "user_cancel") -> bool:
        job = self._store.get_job(job_id)
        if not job:
            return False
        terminal = {JobState.STOPPED, JobState.CANCELLED, JobState.FAILED}
        if job["state"] in terminal or job["state"] == JobState.CANCELLING:
            return False
        if not self._store.transition_job(
            job_id,
            JobState.CANCELLING,
            expected_state=job["state"],
            reason=reason,
        ):
            return False
        self._store.revoke_leases(job_id)
        self._store.transition_job(
            job_id,
            JobState.CANCELLED,
            expected_state=JobState.CANCELLING,
            reason=reason,
        )
        log.info("job %s cancelled: %s", job_id, reason)
        return True

    def mark_job_running(self, job_id: str) -> None:
        self._store.transition_job(job_id, JobState.RUNNING, reason="warmup_complete")

    def mark_job_failed(self, job_id: str, reason: str) -> None:
        self._store.revoke_leases(job_id)
        self._store.transition_job(job_id, JobState.FAILED, reason=reason)
        log.error("job %s failed: %s", job_id, reason)

    def get_job_status(self, job_id: str) -> dict | None:
        return self._store.get_job(job_id)

    def list_jobs(self) -> list[dict]:
        return self._store.list_jobs()
