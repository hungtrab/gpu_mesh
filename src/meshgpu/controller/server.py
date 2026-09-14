"""Controller HTTP/WebSocket server (FastAPI)."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from meshgpu.controller.scheduler import Scheduler
from meshgpu.controller.store import Store
from meshgpu.protocol.messages import (
    CancelRequest,
    Heartbeat,
    JobSpec,
    RegisterRequest,
)

log = logging.getLogger(__name__)


def build_app(db_path: str = "meshgpu.db", join_token: str | None = None) -> FastAPI:
    configured_token = (
        join_token
        if join_token is not None
        else os.environ.get("MESHGPU_JOIN_TOKEN")
    )
    if configured_token == "":
        raise ValueError("MESHGPU_JOIN_TOKEN must be non-empty when configured")
    store = Store(db_path)
    if configured_token is None:
        log.warning(
            "MESHGPU_JOIN_TOKEN is not configured; controller registration is in dev mode"
        )
    scheduler = Scheduler(store, join_token=configured_token)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Own the monitor task and SQLite connection for the app lifetime."""
        task = asyncio.create_task(scheduler.run_heartbeat_monitor())
        app.state.heartbeat_monitor_task = task
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            app.state.heartbeat_monitor_task = None
            store.close()

    app = FastAPI(
        title="MeshGPU Controller",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # REST endpoints
    # ------------------------------------------------------------------

    @app.post("/v1/workers/register")
    async def register(req: RegisterRequest) -> JSONResponse:
        try:
            result = scheduler.register_worker(req.join_token, req.capability)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/v1/jobs")
    async def submit_job(spec: JobSpec) -> JSONResponse:
        job_id = scheduler.submit_job(spec)
        return JSONResponse({"job_id": job_id, "state": "pending"})

    @app.post("/v1/jobs/{job_id}/admit")
    async def admit_job(job_id: str) -> JSONResponse:
        grant = scheduler.admit_job(job_id)
        if grant is None:
            raise HTTPException(status_code=409, detail="job cannot be admitted")
        return JSONResponse(grant.model_dump(mode="json"))

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        status = scheduler.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse(status)

    @app.get("/v1/jobs")
    async def list_jobs() -> JSONResponse:
        return JSONResponse(scheduler.list_jobs())

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, req: CancelRequest) -> JSONResponse:
        ok = scheduler.cancel_job(job_id, req.reason)
        if not ok:
            raise HTTPException(status_code=409, detail="cannot cancel job in current state")
        return JSONResponse({"job_id": job_id, "state": "cancelled"})

    @app.get("/v1/workers")
    async def list_workers() -> JSONResponse:
        # ``Store`` needs the credential for authenticated heartbeats, but the
        # worker listing is a controller-facing discovery endpoint.  Never
        # expose bearer credentials in that public response: anyone who can
        # read it could impersonate the worker over the WebSocket channel.
        workers = []
        for worker in store.list_workers():
            public_worker = dict(worker)
            public_worker.pop("credential", None)
            workers.append(public_worker)
        return JSONResponse(workers)

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ------------------------------------------------------------------
    # WebSocket: persistent heartbeat + command channel
    # ------------------------------------------------------------------

    @app.websocket("/v1/workers/{worker_id}/ws")
    async def worker_ws(websocket: WebSocket, worker_id: str) -> None:
        authorization = websocket.headers.get("authorization", "")
        scheme, _, credential = authorization.partition(" ")
        if (
            scheme.lower() != "bearer"
            or not credential
            or not store.authenticate_worker(worker_id, credential)
        ):
            await websocket.close(code=1008, reason="authentication failed")
            return
        await websocket.accept()
        log.info("ws connected: worker %s", worker_id)
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("_type")

                if msg_type == "heartbeat":
                    hb = Heartbeat.model_validate(data)
                    if hb.worker_id != worker_id:
                        await websocket.send_json({"_type": "error", "code": "auth_failed"})
                        break
                    ok = scheduler.process_heartbeat(
                        hb.worker_id,
                        hb.worker_incarnation,
                        credential,
                    )
                    if not ok:
                        await websocket.send_json({"_type": "error", "code": "auth_failed"})
                        break
                    await websocket.send_json({
                        "_type": "heartbeat_ack",
                        "lease_epoch": hb.lease_epoch,
                        "deadline": hb.ts + 300,
                    })

                else:
                    log.warning("unknown ws message type %r from %s", msg_type, worker_id)

        except WebSocketDisconnect:
            log.info("ws disconnected: worker %s", worker_id)
        except Exception:
            log.exception("ws error for worker %s", worker_id)
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                # The client may already have closed the connection.
                pass

    return app
