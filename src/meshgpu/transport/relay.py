"""
Tensor relay — forwards frames between workers when direct connection unavailable.
Outbound workers (notebooks) connect out to relay via WSS/443.

Design constraints from plan.md §6.2:
  - Separate control/data connections so large tensors don't block heartbeats.
  - Per-job credit limit and byte quota; sender blocks when receiver is slow.
  - Relay does NOT interpret CUDA pointers or execute model ops.
  - Routes are job-scoped and torn down when job ends or lease expires.

MVP: asyncio WebSocket relay, single process, in-memory frame queues.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import ssl
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

log = logging.getLogger(__name__)

# Max bytes buffered per route before sender is backpressured
_ROUTE_BUFFER_BYTES = 32 * 1024 * 1024   # 32 MiB
_FRAME_DEADLINE_S = 30.0                  # drop route if idle > 30s
_MAX_FRAME_BYTES = 8 * 1024 * 1024       # 8 MiB per frame hard limit
_MAX_RELAY_MESSAGE_BYTES = _MAX_FRAME_BYTES + 64 * 1024
_RELAY_PATH = "/v1/relay"
RELAY_TOKEN_HEADER = "X-MeshGPU-Relay-Token"
_RELAY_QUEUE_MESSAGES = 64
_RELAY_PAIR_TIMEOUT_S = 120.0
# A stage forward can legitimately take longer than the WebSocket library's
# 20-second default on a cold GPU or across a public relay.  Keep the
# connection alive, but allow a slow WAN round trip and a long first forward
# before declaring the peer dead.
_RELAY_PING_INTERVAL_S = 30.0
_RELAY_PING_TIMEOUT_S = 300.0
RelayRole = Literal["worker", "gateway"]


@dataclass
class Route:
    """One directional A→B forwarding route."""
    job_id: str
    src_worker: str
    dst_worker: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    buffered_bytes: int = 0
    credit_bytes: int = _ROUTE_BUFFER_BYTES
    last_activity: float = field(default_factory=time.time)
    closed: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def route_key(self) -> str:
        return f"{self.job_id}/{self.src_worker}->{self.dst_worker}"

    def can_enqueue(self, frame_bytes: int) -> bool:
        return self.buffered_bytes + frame_bytes <= self.credit_bytes

    async def enqueue(self, frame: bytes) -> bool:
        if len(frame) > _MAX_FRAME_BYTES:
            raise ValueError(f"frame {len(frame)} > max {_MAX_FRAME_BYTES}")
        async with self._lock:
            if self.closed or not self.can_enqueue(len(frame)) or self.queue.full():
                return False
            self.queue.put_nowait(frame)
            self.buffered_bytes += len(frame)
            self.last_activity = time.time()
            return True

    async def mark_dequeued(self, frame_bytes: int) -> None:
        async with self._lock:
            self.buffered_bytes = max(0, self.buffered_bytes - frame_bytes)
            self.last_activity = time.time()

    def close(self) -> None:
        self.closed = True


class RelayServer:
    """
    In-process relay that bridges WebSocket connections.
    Each worker opens two connections: one control WSS, one data WSS.
    Routes are registered per job; frames forwarded without interpretation.
    """

    def __init__(self) -> None:
        self._routes: dict[str, Route] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Route management (called by controller side)
    # ------------------------------------------------------------------

    async def register_route(
        self, job_id: str, src_worker: str, dst_worker: str
    ) -> str:
        route = Route(job_id=job_id, src_worker=src_worker, dst_worker=dst_worker)
        key = route.route_key()
        async with self._lock:
            if key in self._routes and not self._routes[key].closed:
                log.warning("route %s already exists, replacing", key)
            self._routes[key] = route
        log.info("route registered: %s", key)
        return key

    async def close_route(self, job_id: str, src_worker: str, dst_worker: str) -> None:
        key = f"{job_id}/{src_worker}->{dst_worker}"
        async with self._lock:
            if key in self._routes:
                self._routes[key].close()
                del self._routes[key]
        log.info("route closed: %s", key)

    async def close_job_routes(self, job_id: str) -> None:
        async with self._lock:
            to_remove = [k for k in self._routes if k.startswith(f"{job_id}/")]
            for k in to_remove:
                self._routes[k].close()
                del self._routes[k]
        log.info("all routes closed for job %s", job_id)

    # ------------------------------------------------------------------
    # Frame forwarding (called per received frame)
    # ------------------------------------------------------------------

    async def forward(
        self,
        job_id: str,
        src_worker: str,
        dst_worker: str,
        frame: bytes,
    ) -> bool:
        """Enqueue frame for forwarding; return False if route not found or full."""
        key = f"{job_id}/{src_worker}->{dst_worker}"
        async with self._lock:
            route = self._routes.get(key)
        if route is None or route.closed:
            log.warning("forward: route %s not found", key)
            return False
        accepted = await route.enqueue(frame)
        if not accepted:
            log.warning("forward: route %s buffer full or closed", key)
        return accepted

    async def drain_to_worker(
        self,
        job_id: str,
        dst_worker: str,
        ws_send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """
        Drain all frames destined for dst_worker and send via ws_send coroutine.
        Call this from the worker's receive loop.
        """
        async with self._lock:
            routes = [
                r for r in self._routes.values()
                if r.job_id == job_id and r.dst_worker == dst_worker and not r.closed
            ]

        for route in routes:
            while not route.queue.empty():
                frame: bytes | None = None
                try:
                    frame = route.queue.get_nowait()
                    await ws_send(frame)
                    await route.mark_dequeued(len(frame))
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    # Do not lose a frame if the destination send fails.
                    if frame is not None:
                        async with route._lock:
                            if not route.closed and not route.queue.full():
                                route.queue.put_nowait(frame)
                    raise

    # ------------------------------------------------------------------
    # Idle route cleanup
    # ------------------------------------------------------------------

    async def gc_loop(self) -> None:
        """Background task: remove routes idle beyond deadline."""
        while True:
            now = time.time()
            async with self._lock:
                stale = [
                    k for k, r in self._routes.items()
                    if now - r.last_activity > _FRAME_DEADLINE_S
                ]
                for k in stale:
                    log.info("gc: closing idle route %s", k)
                    self._routes[k].close()
                    del self._routes[k]
            await asyncio.sleep(10)

    def route_count(self) -> int:
        return len(self._routes)

    def stats(self) -> dict:
        return {
            "routes": self.route_count(),
            "total_buffered_bytes": sum(r.buffered_bytes for r in self._routes.values()),
        }


@dataclass
class _SocketPair:
    """The two outbound WebSockets belonging to one stage route."""

    route_key: str
    worker: Any | None = None
    gateway: Any | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    bridge_task: asyncio.Task[None] | None = None
    closed: bool = False


class _BoundedMessageQueue:
    """Byte-bounded queue used in one direction of a relay bridge."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_RELAY_QUEUE_MESSAGES)
        self._condition = asyncio.Condition()
        self._buffered_bytes = 0

    async def put(self, message: bytes) -> None:
        if not isinstance(message, bytes):
            raise TypeError("relay only forwards binary WebSocket messages")
        if len(message) > _MAX_RELAY_MESSAGE_BYTES:
            raise ValueError(
                f"relay message {len(message)} > max {_MAX_RELAY_MESSAGE_BYTES}"
            )
        async with self._condition:
            while (
                self._buffered_bytes + len(message) > _ROUTE_BUFFER_BYTES
                or self._queue.full()
            ):
                await self._condition.wait()
            self._queue.put_nowait(message)
            self._buffered_bytes += len(message)

    async def get(self) -> bytes:
        message = await self._queue.get()
        async with self._condition:
            self._buffered_bytes -= len(message)
            self._condition.notify_all()
        return message


class WebSocketRelay:
    """Public WebSocket rendezvous for workers that cannot accept inbound TCP.

    A worker and a gateway both open outbound WebSockets to this endpoint. The
    relay authenticates the small handshake using ``relay_token``, pairs one
    ``worker`` with one ``gateway`` by ``job_id/stage_id``, then forwards the
    authenticated binary Stage-RPC stream without decoding it. Every
    direction has both a message-count and byte bound, so a slow notebook
    backpressures its sender instead of causing unbounded relay RAM.

    This is intentionally a trusted single-process MVP. Put it behind TLS
    (normally a reverse proxy on port 443) and use a separate token per
    deployment; it is not a public multi-tenant broker.
    """

    def __init__(
        self,
        *,
        relay_token: str | None = None,
        path: str = _RELAY_PATH,
        pair_timeout_s: float = _RELAY_PAIR_TIMEOUT_S,
    ) -> None:
        if relay_token == "":
            raise ValueError("relay_token must be non-empty or None for dev mode")
        if not path.startswith("/") or path.endswith("/"):
            raise ValueError("relay path must start with / and not end with /")
        if (
            isinstance(pair_timeout_s, bool)
            or not isinstance(pair_timeout_s, (int, float))
            or pair_timeout_s <= 0
        ):
            raise ValueError("pair_timeout_s must be positive")
        self._relay_token = relay_token
        self._path = path
        self._pair_timeout_s = float(pair_timeout_s)
        self._pairs: dict[str, _SocketPair] = {}
        self._lock = asyncio.Lock()

    async def serve(
        self,
        host: str,
        port: int,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> Any:
        """Start the public relay and return the closeable WebSocket server."""
        from websockets.asyncio.server import serve

        return await serve(
            self._handle_connection,
            host,
            port,
            ssl=ssl_context,
            max_size=_MAX_RELAY_MESSAGE_BYTES,
            max_queue=_RELAY_QUEUE_MESSAGES,
            ping_interval=_RELAY_PING_INTERVAL_S,
            ping_timeout=_RELAY_PING_TIMEOUT_S,
        )

    async def close(self) -> None:
        """Close all currently paired sockets and forget their routes."""
        async with self._lock:
            pairs = list(self._pairs.values())
            self._pairs.clear()
            for pair in pairs:
                pair.closed = True
                pair.ready.set()
            bridge_tasks = [
                pair.bridge_task
                for pair in pairs
                if pair.bridge_task is not None
            ]
        for task in bridge_tasks:
            task.cancel()
        if bridge_tasks:
            await asyncio.gather(*bridge_tasks, return_exceptions=True)
        sockets = [
            socket
            for pair in pairs
            for socket in (pair.worker, pair.gateway)
            if socket is not None
        ]
        await asyncio.gather(
            *(
                _close_socket(socket, code=1001, reason="relay shutdown")
                for socket in sockets
            ),
            return_exceptions=True,
        )

    def route_count(self) -> int:
        return len(self._pairs)

    async def _handle_connection(self, ws: Any) -> None:
        if not self._authorized(ws):
            await _close_socket(ws, code=1008, reason="relay authentication failed")
            return
        try:
            route_key, role = _parse_relay_route(ws, expected_path=self._path)
        except (TypeError, ValueError) as exc:
            await _close_socket(ws, code=1008, reason=str(exc)[:123])
            return

        pair = await self._attach(route_key, role, ws)
        if pair is None:
            await _close_socket(ws, code=1013, reason="relay route already has this role")
            return
        try:
            ready_task = asyncio.create_task(pair.ready.wait())
            wait_tasks: set[asyncio.Task[Any]] = {ready_task}
            wait_closed = getattr(ws, "wait_closed", None)
            closed_task: asyncio.Task[Any] | None = None
            if callable(wait_closed):
                closed_task = asyncio.create_task(wait_closed())
                wait_tasks.add(closed_task)
            try:
                done, _pending = await asyncio.wait(
                    wait_tasks,
                    timeout=self._pair_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in wait_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*wait_tasks, return_exceptions=True)
            if not done:
                await _close_socket(ws, code=1013, reason="relay peer did not connect")
                return
            if closed_task is not None and closed_task in done and not pair.ready.is_set():
                return
            if pair.closed or pair.bridge_task is None:
                return
            await pair.bridge_task
        finally:
            await self._detach(pair, route_key, role, ws)

    def _authorized(self, ws: Any) -> bool:
        if self._relay_token is None:
            return True
        request = getattr(ws, "request", None)
        headers = getattr(request, "headers", None) or {}
        token = headers.get(RELAY_TOKEN_HEADER, "")
        return isinstance(token, str) and hmac.compare_digest(token, self._relay_token)

    async def _attach(
        self,
        route_key: str,
        role: RelayRole,
        ws: Any,
    ) -> _SocketPair | None:
        async with self._lock:
            pair = self._pairs.get(route_key)
            if pair is None or pair.closed:
                pair = _SocketPair(route_key=route_key)
                self._pairs[route_key] = pair
            if getattr(pair, role) is not None:
                return None
            setattr(pair, role, ws)
            if pair.worker is not None and pair.gateway is not None:
                pair.bridge_task = asyncio.create_task(self._bridge(pair))
                pair.ready.set()
            return pair

    async def _detach(
        self,
        pair: _SocketPair,
        route_key: str,
        role: RelayRole,
        ws: Any,
    ) -> None:
        peer: Any | None = None
        async with self._lock:
            if self._pairs.get(route_key) is not pair:
                return
            if getattr(pair, role) is ws:
                setattr(pair, role, None)
            peer = pair.gateway if role == "worker" else pair.worker
            pair.closed = True
            pair.ready.set()
            self._pairs.pop(route_key, None)
        if peer is not None:
            await _close_socket(peer, code=1001, reason="relay peer disconnected")

    async def _bridge(self, pair: _SocketPair) -> None:
        assert pair.worker is not None and pair.gateway is not None
        worker_to_gateway = _BoundedMessageQueue()
        gateway_to_worker = _BoundedMessageQueue()
        tasks = {
            asyncio.create_task(_read_to_queue(pair.worker, worker_to_gateway)),
            asyncio.create_task(_write_from_queue(pair.gateway, worker_to_gateway)),
            asyncio.create_task(_read_to_queue(pair.gateway, gateway_to_worker)),
            asyncio.create_task(_write_from_queue(pair.worker, gateway_to_worker)),
        }
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                _close_socket(pair.worker, code=1000, reason="relay route closed"),
                _close_socket(pair.gateway, code=1000, reason="relay route closed"),
                return_exceptions=True,
            )


async def _read_to_queue(ws: Any, queue: _BoundedMessageQueue) -> None:
    async for message in ws:
        await queue.put(message)


async def _write_from_queue(ws: Any, queue: _BoundedMessageQueue) -> None:
    while True:
        await ws.send(await queue.get())


async def _close_socket(ws: Any, *, code: int, reason: str) -> None:
    try:
        await ws.close(code=code, reason=reason)
    except (ConnectionError, OSError, RuntimeError):
        pass


def _parse_relay_route(ws: Any, *, expected_path: str) -> tuple[str, RelayRole]:
    request = getattr(ws, "request", None)
    path = getattr(request, "path", None)
    if not isinstance(path, str):
        raise ValueError("relay connection has no request path")
    parsed = urlsplit(path)
    if parsed.path != expected_path:
        raise ValueError(f"relay path must be {expected_path!r}")

    values = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)

    def one(name: str) -> str:
        items = values.get(name, [])
        if len(items) != 1 or not items[0] or len(items[0]) > 256:
            raise ValueError(f"relay query parameter {name!r} must occur once")
        return items[0]

    job_id = one("job_id")
    stage_id = one("stage_id")
    role_value = one("role")
    try:
        stage_number = int(stage_id)
    except ValueError as exc:
        raise ValueError("relay stage_id must be an integer") from exc
    if not 0 <= stage_number < (1 << 32):
        raise ValueError("relay stage_id must fit in an unsigned 32-bit field")
    if role_value not in {"worker", "gateway"}:
        raise ValueError("relay role must be worker or gateway")
    return f"{job_id}/{stage_number}", cast(RelayRole, role_value)


def make_relay_url(
    base_url: str,
    *,
    job_id: str | int,
    stage_id: int,
    role: RelayRole,
    worker_id: str | None = None,
) -> str:
    """Build a worker/gateway route URL without putting credentials in it."""
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("relay base URL must not be empty")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("relay URL must use ws:// or wss:// and include a host")
    if role not in {"worker", "gateway"}:
        raise ValueError("relay role must be worker or gateway")
    if isinstance(job_id, bool) or not str(job_id) or len(str(job_id)) > 256:
        raise ValueError("job_id must be non-empty and at most 256 characters")
    if isinstance(stage_id, bool) or not isinstance(stage_id, int):
        raise TypeError("stage_id must be an integer")
    if not 0 <= stage_id < (1 << 32):
        raise ValueError("stage_id must fit in an unsigned 32-bit field")
    if worker_id is not None and (not worker_id or len(worker_id) > 256):
        raise ValueError("worker_id must be non-empty and at most 256 characters")
    existing = parse_qsl(parsed.query, keep_blank_values=True)
    reserved = {"job_id", "stage_id", "role", "worker_id"}
    if any(key in reserved for key, _value in existing):
        raise ValueError("relay URL already contains a reserved route parameter")
    path = parsed.path.rstrip("/") or _RELAY_PATH
    query = existing + [
        ("job_id", str(job_id)),
        ("stage_id", str(stage_id)),
        ("role", role),
    ]
    if worker_id is not None:
        query.append(("worker_id", worker_id))
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))


async def run_outbound_stage(
    endpoint: Any,
    relay_url: str,
    credential: str,
    *,
    job_id: str | int,
    stage_id: int,
    relay_token: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    max_reconnect_attempts: int = 0,
    reconnect_delay_base_s: float = 2.0,
    reconnect_delay_max_s: float = 60.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Connect a stage outbound and serve it through a relay.

    ``max_reconnect_attempts`` counts retries after the initial connection;
    zero gives deterministic one-shot smoke-test behavior. A stop event can
    interrupt the bounded backoff while a notebook is being shut down.
    """
    if not credential:
        raise ValueError("stage credential must not be empty")
    if relay_token == "":
        raise ValueError("relay_token must be non-empty or None")
    if (
        isinstance(max_reconnect_attempts, bool)
        or not isinstance(max_reconnect_attempts, int)
        or max_reconnect_attempts < 0
    ):
        raise ValueError("max_reconnect_attempts must be a non-negative integer")
    if (
        reconnect_delay_base_s <= 0
        or reconnect_delay_max_s <= 0
        or reconnect_delay_base_s > reconnect_delay_max_s
    ):
        raise ValueError("reconnect delays must be positive and ordered")
    url = make_relay_url(
        relay_url,
        job_id=job_id,
        stage_id=stage_id,
        role="worker",
    )
    # websockets 15 requires an explicit SSL context for ``wss://``.  Passing
    # ``ssl=None`` is not equivalent to the default context there and fails
    # before the TCP connection is attempted.  Keep plaintext ``ws://``
    # available for trusted local/test deployments.
    parsed_url = urlsplit(url)
    connect_ssl = ssl_context
    if parsed_url.scheme == "wss" and connect_ssl is None:
        connect_ssl = ssl.create_default_context()
    headers = {"Authorization": f"Bearer {credential}"}
    if relay_token is not None:
        headers[RELAY_TOKEN_HEADER] = relay_token
    from websockets.asyncio.client import connect

    retries = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        ws: Any | None = None
        error: BaseException | None = None
        try:
            ws = await connect(
                url,
                additional_headers=headers,
                ssl=connect_ssl,
                max_size=_MAX_RELAY_MESSAGE_BYTES,
                ping_interval=_RELAY_PING_INTERVAL_S,
                ping_timeout=_RELAY_PING_TIMEOUT_S,
            )
            await endpoint.serve_connection(ws)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
            log.warning("outbound stage connection failed: %s", exc)
        finally:
            if ws is not None:
                await _close_socket(ws, code=1000, reason="stage worker stopping")

        if retries >= max_reconnect_attempts:
            if error is not None:
                raise ConnectionError("outbound stage exhausted reconnect attempts") from error
            return
        retries += 1
        delay = min(
            reconnect_delay_base_s * (2 ** (retries - 1)),
            reconnect_delay_max_s,
        )
        if stop_event is None:
            await asyncio.sleep(delay)
        else:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                return
