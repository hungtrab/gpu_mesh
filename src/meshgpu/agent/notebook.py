"""
Notebook agent bootstrap.
Idempotent cell that:
  - Installs meshgpu if needed
  - Spawns agent supervisor as a background thread
  - Displays worker status (worker_id, backend, job state, stop button)
  - Handles secret fetch from env (never from cell literal)
  - Reconnects on disconnect within the same runtime

Anti-patterns avoided:
  - Does NOT create multiple agent processes on re-run
  - Does NOT store tokens in notebook output / print statements
  - Does NOT keep connection alive past runtime death (no keepalive hacks)
  - Does NOT claim Colab managed runtime support — only self-managed / local runtime
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_AGENT_THREAD: threading.Thread | None = None
_AGENT_STOP: threading.Event | None = None
_AGENT_SUPERVISOR = None  # meshgpu.agent.supervisor.Supervisor instance
_AGENT_STATUS_CALLBACK: Callable[[dict], None] | None = None
_AGENT_GENERATION = 0
_AGENT_LOCK = threading.RLock()


@dataclass
class NotebookConfig:
    controller_url: str
    join_token: str
    worker_id: str | None = None
    provider: str = "colab_local_runtime"
    session_remaining_s: float | None = None   # hint: how long runtime has left


def _resolve_secret(key: str, fallback: str | None = None) -> str | None:
    """Read secret from environment only — never from cell output."""
    val = os.environ.get(key, fallback)
    if not val:
        log.warning("secret %r not set in environment", key)
    return val


def start(
    controller_url: str | None = None,
    join_token: str | None = None,
    *,
    worker_id: str | None = None,
    provider: str = "colab_local_runtime",
    session_remaining_s: float | None = None,
    on_status: Callable[[dict], None] | None = None,
) -> None:
    """
    Start (or no-op if already running) the notebook agent.
    Call from a notebook cell. Safe to re-run.

    The controller URL may be passed as a convenience, but the join token is
    deliberately read from ``MESHGPU_JOIN_TOKEN`` only.  Keeping credentials
    out of cell arguments prevents them from being captured in notebook input
    history and exported notebooks.
    """
    global _AGENT_THREAD, _AGENT_STOP, _AGENT_SUPERVISOR, _AGENT_STATUS_CALLBACK
    global _AGENT_GENERATION

    with _AGENT_LOCK:
        if _AGENT_THREAD is not None and _AGENT_THREAD.is_alive():
            existing_worker_id = _AGENT_SUPERVISOR and _AGENT_SUPERVISOR._state.worker_id
            log.info("agent already running (worker_id=%s)", existing_worker_id)
            if on_status is not None:
                _AGENT_STATUS_CALLBACK = on_status
            callback = _AGENT_STATUS_CALLBACK
            existing_status = status()
        else:
            url = controller_url or _resolve_secret("MESHGPU_CONTROLLER")
            token = _resolve_secret("MESHGPU_JOIN_TOKEN")

            if join_token is not None:
                log.warning(
                    "join_token argument is ignored; set MESHGPU_JOIN_TOKEN in the "
                    "runtime environment instead"
                )
            if not url or not token:
                raise ValueError(
                    "controller_url and join_token are required; join_token must come "
                    "from the MESHGPU_JOIN_TOKEN environment variable."
                )

            from meshgpu.agent.supervisor import Supervisor

            stop_event = threading.Event()
            sup = Supervisor(url, token, worker_id=worker_id, provider=provider)
            _AGENT_GENERATION += 1
            generation = _AGENT_GENERATION

            def _run() -> None:
                # Publish the start event from the agent thread itself.  A
                # supervisor can finish before ``start()`` returns (notably
                # when registration is mocked or fails immediately); emitting
                # only from the caller would then report ``running=False`` and
                # lose the lifecycle transition entirely.
                _notify_status(
                    running=True,
                    generation=generation,
                    supervisor=sup,
                )
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(sup.run())
                except Exception:
                    log.exception("agent error")
                finally:
                    loop.close()
                    # The thread is still technically alive while this callback
                    # runs, so override ``running`` to describe the supervisor
                    # lifecycle rather than the implementation detail.
                    _notify_status(
                        running=False,
                        generation=generation,
                        supervisor=sup,
                    )

            t = threading.Thread(target=_run, daemon=True, name="meshgpu-agent")
            _AGENT_STOP = stop_event
            _AGENT_SUPERVISOR = sup
            _AGENT_THREAD = t
            _AGENT_STATUS_CALLBACK = on_status
            t.start()
            log.info("agent thread started (provider=%s)", provider)
            callback = _AGENT_STATUS_CALLBACK
            existing_status = None

    if existing_status is not None:
        _emit_status(callback, existing_status)
        return
    _print_status()
    _emit_status(callback, status())


def stop() -> None:
    """Stop the agent gracefully."""
    global _AGENT_SUPERVISOR, _AGENT_THREAD, _AGENT_STOP, _AGENT_STATUS_CALLBACK
    callback: Callable[[dict], None] | None = None
    stopped_status: dict | None = None
    with _AGENT_LOCK:
        supervisor = _AGENT_SUPERVISOR
        thread = _AGENT_THREAD
        stop_event = _AGENT_STOP
        if supervisor is not None:
            supervisor.stop()
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        # Do not permit a second supervisor while a slow old thread is still
        # alive.  ``is True`` keeps lightweight MagicMock-based notebook tests
        # compatible while real threading.Thread always returns bool here.
        if thread is not None and thread.is_alive() is True:
            log.warning("agent thread did not stop within 5 seconds")
            return
        if supervisor is not None:
            stopped_status = status()
            stopped_status["running"] = False
            callback = _AGENT_STATUS_CALLBACK
        _AGENT_SUPERVISOR = None
        _AGENT_THREAD = None
        _AGENT_STOP = None
        _AGENT_STATUS_CALLBACK = None
    log.info("agent stopped")
    if stopped_status is not None:
        _emit_status(callback, stopped_status)


def status() -> dict:
    """Return current agent status dict."""
    with _AGENT_LOCK:
        supervisor = _AGENT_SUPERVISOR
        thread = _AGENT_THREAD
        payload = _status_for(supervisor)
        payload["running"] = thread is not None and thread.is_alive()
        return payload


def _print_status() -> None:
    st = status()
    if st["running"]:
        print(f"[MeshGPU] agent running | worker_id={st['worker_id']} | provider={st['provider']}")
        print(f"[MeshGPU] job={st['job_id'] or 'idle'} | state={st['job_state']}")
        print("[MeshGPU] call meshgpu.notebook.stop() to stop the agent")
    else:
        print("[MeshGPU] agent not running")


def _notify_status(
    *,
    running: bool | None = None,
    generation: int | None = None,
    supervisor: Any | None = None,
) -> None:
    """Emit a best-effort lifecycle update for the current agent generation."""
    with _AGENT_LOCK:
        if generation is not None and generation != _AGENT_GENERATION:
            # A previous thread may finish after a new ``start`` call.  Its
            # lifecycle event must never describe the replacement supervisor.
            return
        callback = _AGENT_STATUS_CALLBACK
        active_supervisor = supervisor or _AGENT_SUPERVISOR
    if callback is None:
        return
    payload = _status_for(active_supervisor)
    _emit_status(callback, payload, running=running)


def _status_for(supervisor: Any | None) -> dict:
    """Build a status payload without consulting mutable global supervisor state."""
    if supervisor is None:
        return {"running": False}
    state = supervisor._state
    return {
        "running": False,
        "worker_id": state.worker_id,
        "provider": supervisor._capability.provider,
        "job_id": state.current_job_id,
        "job_state": state.job_state.value,
        "lease_epoch": state.lease_epoch,
    }


def _emit_status(
    callback: Callable[[dict], None] | None,
    payload: dict,
    *,
    running: bool | None = None,
) -> None:
    if callback is None:
        return
    if running is not None:
        payload = dict(payload)
        payload["running"] = running
    try:
        callback(payload)
    except Exception:
        # A UI/reporting callback must never take down the agent thread.
        log.exception("notebook status callback failed")
