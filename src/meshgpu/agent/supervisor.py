"""
Agent supervisor: registers with controller, manages GPU subprocess,
sends heartbeats, handles lease and job lifecycle.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import random
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import cast

import aiohttp

from meshgpu.agent.probe import build_capability_report
from meshgpu.agent.provider import ProviderKind, assert_eligible_for_worker
from meshgpu.controller.scheduler import HEARTBEAT_INTERVAL_S
from meshgpu.protocol.messages import (
    JobState,
    RegisterRequest,
)

log = logging.getLogger(__name__)

_RECONNECT_DELAY_BASE_S = 2.0
_RECONNECT_DELAY_MAX_S = 60.0
_RECONNECT_JITTER_S = 1.0
_MAX_RECONNECT_ATTEMPTS = 10
_HEARTBEAT_INTERVAL_S = HEARTBEAT_INTERVAL_S


@dataclass
class AgentState:
    worker_id: str
    incarnation: str = field(default_factory=lambda: str(uuid.uuid4()))
    credential: str = ""
    controller_url: str = ""
    current_job_id: str | None = None
    lease_epoch: int = 0
    job_state: JobState = JobState.PENDING
    gpu_pids: list[int] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        controller_url: str,
        join_token: str,
        worker_id: str | None = None,
        provider: str = "self_managed",
    ) -> None:
        try:
            provider_kind = ProviderKind(provider)
        except ValueError as exc:
            raise ValueError(f"unknown provider {provider!r}") from exc
        assert_eligible_for_worker(provider_kind)
        self._controller_url = controller_url.rstrip("/")
        self._join_token = join_token
        self._capability = build_capability_report(worker_id, provider)
        # Registration and every heartbeat must use the same incarnation.
        # A new Supervisor instance gets a new capability incarnation, while
        # reconnects within that instance retain it.
        self._state = AgentState(
            worker_id=self._capability.worker_id,
            incarnation=self._capability.worker_incarnation,
        )
        self._state.controller_url = controller_url
        self._session: aiohttp.ClientSession | None = None
        self._stop = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mp_context = mp.get_context("spawn")
        self._gpu_processes: dict[int, mp.Process] = {}
        self._gpu_commands: dict[int, Connection] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _register(self) -> bool:
        req = RegisterRequest(
            join_token=self._join_token,
            capability=self._capability,
        )
        try:
            session = self._session
            if session is None:
                raise RuntimeError("supervisor HTTP session is not initialized")
            async with session.post(
                f"{self._controller_url}/v1/workers/register",
                json=req.model_dump(mode="json"),
            ) as resp:
                if resp.status != 200:
                    log.error("registration failed: %s", resp.status)
                    return False
                data = await resp.json()
                self._state.credential = data["credential"]
                log.info(
                    "registered as %s (expires %.0f)",
                    self._state.worker_id,
                    data["expires_at"],
                )
                return True
        except Exception:
            log.exception("registration error")
            return False

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        ws_url = (
            self._controller_url.replace("http://", "ws://")
            .replace("https://", "wss://")
            + f"/v1/workers/{self._state.worker_id}/ws"
        )
        attempt = 0
        session = self._session
        if session is None:
            raise RuntimeError("supervisor HTTP session is not initialized")
        while not self._stop.is_set():
            try:
                async with session.ws_connect(
                    ws_url,
                    headers={"Authorization": f"Bearer {self._state.credential}"},
                ) as ws:
                    attempt = 0
                    log.info("heartbeat ws connected")
                    while not self._stop.is_set():
                        gpu_free = self._poll_gpu_free()
                        await ws.send_json({
                            "_type": "heartbeat",
                            "worker_id": self._state.worker_id,
                            "worker_incarnation": self._state.incarnation,
                            "lease_epoch": self._state.lease_epoch,
                            "job_id": self._state.current_job_id,
                            "gpu_free_vram_bytes": gpu_free,
                            "ts": time.time(),
                        })
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=10)
                        if msg.get("_type") == "error":
                            log.error("heartbeat error from controller: %s", msg)
                            break
                        await self._sleep_or_stop(_HEARTBEAT_INTERVAL_S)
            except Exception:
                attempt += 1
                if attempt >= _MAX_RECONNECT_ATTEMPTS:
                    log.error("too many reconnect attempts (%d); giving up", attempt)
                    self._stop.set()
                    return
                delay = min(
                    _RECONNECT_DELAY_BASE_S * (2 ** (attempt - 1)),
                    _RECONNECT_DELAY_MAX_S,
                ) + random.uniform(0, _RECONNECT_JITTER_S)
                log.warning(
                    "heartbeat ws disconnected (attempt %d/%d); reconnecting in %.1fs",
                    attempt, _MAX_RECONNECT_ATTEMPTS, delay,
                )
                await self._sleep_or_stop(delay)

    def _poll_gpu_free(self) -> list[int]:
        try:
            import torch
            return [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # GPU subprocess management
    # ------------------------------------------------------------------

    def _launch_gpu_worker(self, stage_id: int, device_index: int) -> int:
        """Spawn a managed process that owns one GPU device and return its PID.

        The process is intentionally only a resource/lifecycle owner here.  A
        future stage-RPC launcher can attach a model executor to the same
        boundary without changing registration or shutdown semantics.
        """
        if stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if device_index < 0:
            raise ValueError("device_index must be non-negative")

        existing = self._gpu_processes.get(stage_id)
        if existing is not None and existing.is_alive() and existing.pid is not None:
            return existing.pid
        if existing is not None:
            self._discard_gpu_process(stage_id, existing)

        from meshgpu.agent.gpu_process import run_gpu_process

        parent, child = self._mp_context.Pipe(duplex=True)
        process = cast(mp.Process, self._mp_context.Process(
            target=run_gpu_process,
            args=(device_index, child),
            name=f"meshgpu-gpu-stage-{stage_id}",
            daemon=True,
        ))
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()

        # Wait for the explicit ready/error handshake.  This turns an invalid
        # device into an immediate launch error instead of a later heartbeat
        # mystery.  The process remains alive after the handshake.
        try:
            if not parent.poll(10.0):
                process.terminate()
                process.join(timeout=2.0)
                raise RuntimeError(
                    f"GPU worker stage={stage_id} did not become ready within 10 seconds"
                )
            message = parent.recv()
            if message.get("event") != "ready":
                raise RuntimeError(
                    f"GPU worker stage={stage_id} failed to start: "
                    f"{message.get('message', message)}"
                )
        except Exception:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            parent.close()
            raise

        self._gpu_processes[stage_id] = process
        self._gpu_commands[stage_id] = parent
        if process.pid is None:  # pragma: no cover - multiprocessing invariant
            self._discard_gpu_process(stage_id, process)
            raise RuntimeError("GPU worker started without a PID")
        self._state.gpu_pids.append(process.pid)
        log.info(
            "GPU worker started stage=%d device=%d pid=%d",
            stage_id,
            device_index,
            process.pid,
        )
        return process.pid

    def _kill_gpu_workers(self) -> None:
        for stage_id, process in list(self._gpu_processes.items()):
            command = self._gpu_commands.get(stage_id)
            if command is not None:
                try:
                    command.send("stop")
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=3.0)
            if process.is_alive():
                log.warning("terminating GPU worker stage=%d pid=%s", stage_id, process.pid)
                process.terminate()
                process.join(timeout=2.0)
            self._discard_gpu_process(stage_id, process)
        self._state.gpu_pids.clear()

    def _discard_gpu_process(self, stage_id: int, process: mp.Process) -> None:
        command = self._gpu_commands.pop(stage_id, None)
        if command is not None:
            try:
                command.close()
            except OSError:
                pass
        self._gpu_processes.pop(stage_id, None)
        if process.pid is not None:
            self._state.gpu_pids = [pid for pid in self._state.gpu_pids if pid != process.pid]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._session = aiohttp.ClientSession()
        try:
            for attempt in range(_MAX_RECONNECT_ATTEMPTS):
                if self._stop.is_set():
                    return
                if await self._register():
                    break
                await self._sleep_or_stop(_retry_delay(attempt + 1))
            else:
                log.error("could not register; exiting")
                return

            await self._heartbeat_loop()
        finally:
            if self._session:
                await self._session.close()
            self._loop = None

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        else:
            self._stop.set()
        self._kill_gpu_workers()

    async def _sleep_or_stop(self, delay: float) -> None:
        """Sleep until a retry interval elapses or stop() is requested."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


def _retry_delay(attempt: int) -> float:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    return min(
        _RECONNECT_DELAY_BASE_S * (2 ** (attempt - 1)),
        _RECONNECT_DELAY_MAX_S,
    ) + random.uniform(0, _RECONNECT_JITTER_S)
