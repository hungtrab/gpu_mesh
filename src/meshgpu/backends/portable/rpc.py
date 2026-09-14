"""Inference RPC for a single portable pipeline stage.

The local ``StageWorker`` API is synchronous because it is also used by the
correctness reference.  This module provides an asynchronous equivalent for a
stage that lives in another process or machine:

* control messages describe one operation and its tensor fields;
* tensor fields travel through the authenticated, chunked ``DataConn``;
* the server executes only the explicit ``forward_inference`` operation;
* KV cache remains in the server-side stage worker.

Training is intentionally not hidden behind this interface.  Backward needs an
explicit activation/gradient protocol and checkpoint boundary; callers should
use the portable in-process trainer until that contract is implemented.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import math
import ssl
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from meshgpu.backends.portable.stage_worker import ForwardResult, StageWorker
from meshgpu.protocol.schema import MAX_TENSOR_BYTES, DType, TensorMeta
from meshgpu.transport.connection import MAX_WEBSOCKET_MESSAGE_BYTES, DataConn

log = logging.getLogger(__name__)

_RPC_REQUEST = "stage_rpc_request"
_RPC_RESPONSE = "stage_rpc_response"
_RPC_ERROR = "stage_rpc_error"
_ALLOWED_FIELDS = frozenset({"hidden", "input_ids", "position_ids", "attention_mask"})
_MAX_ID_LENGTH = 256
_MAX_REQUEST_INPUT_BYTES = MAX_TENSOR_BYTES
_MAX_PENDING_REQUESTS = 128
_RPC_PING_INTERVAL_S = 30.0
_RPC_PING_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class StageRpcIdentity:
    """Numeric wire identity used by ``FrameHeader`` for one stage link."""

    cluster_id: int
    job_id: int
    lease_epoch: int
    worker_incarnation: int
    peer_worker_incarnation: int | None = None

    def __post_init__(self) -> None:
        _validate_uint("cluster_id", self.cluster_id, 16)
        for name, value in (
            ("job_id", self.job_id),
            ("lease_epoch", self.lease_epoch),
            ("worker_incarnation", self.worker_incarnation),
        ):
            _validate_uint(name, value, 32)
        if self.peer_worker_incarnation is not None:
            _validate_uint(
                "peer_worker_incarnation", self.peer_worker_incarnation, 32
            )


@dataclass
class _PendingRequest:
    future: asyncio.Future
    method: str
    operation_id: int
    attempt_id: str
    response: dict[str, Any] | None = None
    output: tuple[TensorMeta, bytes] | None = None


@dataclass
class _ServerRequest:
    request_id: str
    method: str
    operation_id: int
    attempt_id: str
    reset_kv: bool
    cache_key: str | None
    fields: dict[str, str]
    length: int = 0
    tensors: dict[str, tuple[TensorMeta, bytes]] = field(default_factory=dict)
    dispatched: bool = False


def _dtype_from_torch(dtype: torch.dtype) -> DType:
    mapping = {
        torch.float32: DType.FLOAT32,
        torch.float16: DType.FLOAT16,
        torch.bfloat16: DType.BFLOAT16,
        torch.int8: DType.INT8,
        torch.int32: DType.INT32,
        torch.int64: DType.INT64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported RPC tensor dtype: {dtype}") from exc


def _torch_dtype(dtype: DType) -> torch.dtype:
    mapping = {
        DType.FLOAT32: torch.float32,
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.INT8: torch.int8,
        DType.INT32: torch.int32,
        DType.INT64: torch.int64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported RPC tensor dtype: {dtype.value}") from exc


def _tensor_wire_parts(tensor: torch.Tensor, tensor_id: str) -> tuple[TensorMeta, bytes]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"RPC tensor must be torch.Tensor, got {type(tensor).__name__}")
    if not tensor_id or len(tensor_id) > _MAX_ID_LENGTH:
        raise ValueError("RPC tensor_id must be non-empty and at most 256 characters")
    cpu = tensor.detach().to(device="cpu").contiguous()
    meta = TensorMeta(
        tensor_id=tensor_id,
        dtype=_dtype_from_torch(cpu.dtype),
        shape=tuple(int(dim) for dim in cpu.shape),
    )
    meta.validate()
    # Viewing as bytes avoids NumPy dtype gaps (notably bfloat16).
    raw = cpu.view(torch.uint8).numpy().tobytes()
    if len(raw) != meta.byte_length:
        raise RuntimeError(
            f"serialized tensor has {len(raw)} bytes; metadata declares {meta.byte_length}"
        )
    return meta, raw


def _tensor_from_wire(meta: TensorMeta, raw: bytes) -> torch.Tensor:
    meta.validate()
    if len(raw) != meta.byte_length:
        raise ValueError(
            f"tensor {meta.tensor_id!r} has {len(raw)} bytes; expected {meta.byte_length}"
        )
    # bytearray gives frombuffer writable storage; clone owns the result after
    # the request state is discarded.
    return torch.frombuffer(
        bytearray(raw), dtype=_torch_dtype(meta.dtype)
    ).reshape(meta.shape).clone()


async def _to_thread_drain_on_cancel(
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a worker call and wait for it even if its owner task is cancelled.

    Cancelling ``asyncio.to_thread`` cancels the asyncio wrapper, but cannot
    stop the underlying executor thread.  The RPC server must drain that
    thread before releasing connection-owned state, otherwise a late
    ``forward_inference`` can recreate a KV cache after cleanup has run.
    """
    worker_task = asyncio.create_task(
        asyncio.to_thread(function, *args, **kwargs)
    )
    try:
        return await asyncio.shield(worker_task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(worker_task)
        except BaseException:
            # The cancellation of the RPC task is the relevant result.  The
            # worker exception has still been consumed by this await.
            pass
        raise


def _header_dict(identity: StageRpcIdentity) -> dict[str, int]:
    return {
        "cluster_id": identity.cluster_id,
        "job_id": identity.job_id,
        "lease_epoch": identity.lease_epoch,
        "worker_incarnation": identity.worker_incarnation,
    }


class StageRpcClient:
    """Async client for one remote stage endpoint."""

    def __init__(
        self,
        ws: Any,
        data: DataConn,
        *,
        request_timeout_s: float = 300.0,
        vocab_size: int | None = None,
        max_position_embeddings: int | None = None,
    ) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self._ws = ws
        self._data = data
        self._pending: dict[str, _PendingRequest] = {}
        self._tensor_payloads: dict[str, tuple[TensorMeta, bytes]] = {}
        self._call_lock = asyncio.Lock()
        self._wire_attempt = 0
        self._closed = False
        self._request_timeout_s = request_timeout_s
        # Optional metadata lets the HTTP/session layer validate requests
        # before sending them.  The server remains authoritative regardless.
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # A named cache is mandatory for safe multiplexing.  Direct callers
        # may omit ``cache_key``; give that connection a private default rather
        # than sharing the StageWorker's legacy process-wide cache.
        self._default_cache_key = f"rpc-client:{uuid.uuid4().hex}"
        self._recv_task = asyncio.create_task(self._recv_loop())

    @classmethod
    async def connect(
        cls,
        url: str,
        credential: str,
        identity: StageRpcIdentity,
        *,
        ssl_context: ssl.SSLContext | None = None,
        relay_token: str | None = None,
        request_timeout_s: float = 300.0,
        vocab_size: int | None = None,
        max_position_embeddings: int | None = None,
    ) -> StageRpcClient:
        if not url:
            raise ValueError("stage RPC url must not be empty")
        if not credential:
            raise ValueError("stage RPC credential must not be empty")
        if relay_token == "":
            raise ValueError("relay_token must be non-empty or None")
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"ws", "wss"} or not parsed_url.netloc:
            raise ValueError("stage RPC url must use ws:// or wss:// and include a host")
        # websockets 15 rejects ``ssl=None`` for a secure WebSocket URL.  A
        # caller may still provide a custom context (for a private CA); when
        # it doesn't, use the platform trust store.
        if parsed_url.scheme == "wss" and ssl_context is None:
            ssl_context = ssl.create_default_context()
        from websockets.asyncio.client import connect as websocket_connect

        headers = {"Authorization": f"Bearer {credential}"}
        if relay_token is not None:
            from meshgpu.transport.relay import RELAY_TOKEN_HEADER

            headers[RELAY_TOKEN_HEADER] = relay_token
        ws = await websocket_connect(
            url,
            additional_headers=headers,
            ssl=ssl_context,
            max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
            ping_interval=_RPC_PING_INTERVAL_S,
            ping_timeout=_RPC_PING_TIMEOUT_S,
        )
        data = DataConn(
            ws,
            **_header_dict(identity),
            credential=credential,
            peer_worker_incarnation=identity.peer_worker_incarnation,
        )
        return cls(
            ws,
            data,
            request_timeout_s=request_timeout_s,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
        )

    async def forward_inference(
        self,
        hidden: torch.Tensor | None,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        operation_id: int,
        attempt_id: str,
        reset_kv: bool = False,
        cache_key: str | None = None,
    ) -> ForwardResult:
        tensors = {
            name: tensor
            for name, tensor in {
                "hidden": hidden,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }.items()
            if tensor is not None
        }
        control, output = await self._request(
            method="forward_inference",
            operation_id=operation_id,
            attempt_id=attempt_id,
            reset_kv=reset_kv,
            cache_key=cache_key,
            tensors=tensors,
        )
        if output is None:
            raise RuntimeError("stage RPC response omitted forward output")
        return ForwardResult(
            stage_id=int(control["stage_id"]),
            operation_id=int(control["operation_id"]),
            attempt_id=str(control["attempt_id"]),
            output=_tensor_from_wire(*output),
            kv_caches=[],
        )

    async def clear_kv(self, cache_key: str | None = None) -> None:
        await self._request(
            method="clear_kv",
            operation_id=0,
            attempt_id="control",
            cache_key=cache_key,
        )

    async def clear_all_kv(self) -> None:
        await self._request(
            method="clear_all_kv",
            operation_id=0,
            attempt_id="control",
        )

    async def kv_cache_length(self, cache_key: str | None = None) -> int:
        control, _ = await self._request(
            method="kv_cache_length",
            operation_id=0,
            attempt_id="control",
            cache_key=cache_key,
        )
        return int(control["kv_length"])

    async def trim_kv(self, length: int, cache_key: str | None = None) -> None:
        if length < 0:
            raise ValueError("cache length must be non-negative")
        await self._request(
            method="trim_kv",
            operation_id=0,
            attempt_id="control",
            cache_key=cache_key,
            extra={"length": length},
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._fail_pending(ConnectionError("stage RPC client closed"))
        self._recv_task.cancel()
        try:
            await self._recv_task
        except asyncio.CancelledError:
            pass
        await self._ws.close()

    async def __aenter__(self) -> StageRpcClient:
        if self._closed:
            raise ConnectionError("stage RPC client is closed")
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.close()

    async def _request(
        self,
        *,
        method: str,
        operation_id: int,
        attempt_id: str,
        reset_kv: bool = False,
        cache_key: str | None = None,
        tensors: dict[str, torch.Tensor] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], tuple[TensorMeta, bytes] | None]:
        if self._closed:
            raise ConnectionError("stage RPC client is closed")
        if not attempt_id or len(attempt_id) > _MAX_ID_LENGTH:
            raise ValueError("attempt_id must be non-empty and at most 256 characters")
        if cache_key is not None and len(cache_key) > _MAX_ID_LENGTH:
            raise ValueError("cache_key is too long")
        if cache_key == "":
            raise ValueError("cache_key must be non-empty when provided")
        if isinstance(operation_id, bool) or not isinstance(operation_id, int):
            raise ValueError("operation_id must be a non-negative integer")
        if not 0 <= operation_id < (1 << 32):
            raise ValueError("operation_id must fit in an unsigned 32-bit field")
        effective_cache_key = cache_key
        if cache_key is None and method != "clear_all_kv":
            effective_cache_key = self._default_cache_key

        async with self._call_lock:
            if self._closed:
                raise ConnectionError("stage RPC client is closed")
            request_id = uuid.uuid4().hex
            field_ids: dict[str, str] = {}
            wire_tensors: list[tuple[str, TensorMeta, bytes]] = []
            for field_name, tensor in (tensors or {}).items():
                if field_name not in _ALLOWED_FIELDS:
                    raise ValueError(f"unsupported stage RPC tensor field: {field_name!r}")
                tensor_id = f"rpc:{request_id}:{field_name}"
                meta, raw = _tensor_wire_parts(tensor, tensor_id)
                field_ids[field_name] = tensor_id
                wire_tensors.append((field_name, meta, raw))

            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = _PendingRequest(
                future=future,
                method=method,
                operation_id=operation_id,
                attempt_id=attempt_id,
            )
            message: dict[str, Any] = {
                "_type": _RPC_REQUEST,
                "request_id": request_id,
                "method": method,
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "reset_kv": reset_kv,
                "cache_key": effective_cache_key,
                "tensors": field_ids,
                **(extra or {}),
            }
            self._wire_attempt = (self._wire_attempt + 1) & 0xFFFFFFFF
            frame_attempt = self._wire_attempt
            deadline = asyncio.get_running_loop().time() + self._request_timeout_s

            def remaining_timeout() -> float:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError("stage RPC request deadline exceeded")
                return remaining

            try:
                await asyncio.wait_for(
                    self._data.send_control(message), remaining_timeout()
                )
                for _field_name, meta, raw in wire_tensors:
                    await asyncio.wait_for(
                        self._data.send_tensor(raw, meta, attempt_id=frame_attempt),
                        remaining_timeout(),
                    )
                control, output = await asyncio.wait_for(future, remaining_timeout())
                return control, output
            except BaseException as exc:
                # A timeout/cancellation may happen after credit was acquired
                # but before the corresponding chunk was published.  The
                # connection's credit state is then unknowable; close it
                # instead of allowing a later request to deadlock or overrun.
                if isinstance(exc, (asyncio.TimeoutError, asyncio.CancelledError)):
                    await self.close()
                raise
            finally:
                # Successful requests must release their future and any
                # received tensor payload too.  Keeping this cleanup in a
                # finally block is important: the return above makes code
                # placed after the try/except unreachable on the happy path.
                self._pending.pop(request_id, None)
                self._drop_tensor_payloads(request_id)

    async def _recv_loop(self) -> None:
        try:
            await self._data.recv_messages(self._on_tensor, self._on_control)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("stage RPC receive loop failed: %s", exc)
            self._fail_pending(ConnectionError(f"stage RPC receive failed: {exc}"))
        else:
            self._fail_pending(ConnectionError("stage RPC connection closed"))
        finally:
            self._closed = True

    def _on_tensor(self, _operation_id: int, meta: TensorMeta, raw: bytes) -> None:
        request_id = _request_id_from_tensor_id(meta.tensor_id)
        if not request_id or request_id not in self._pending:
            log.warning("discarding tensor for unknown stage RPC request %r", request_id)
            return
        self._tensor_payloads[meta.tensor_id] = (meta, raw)
        self._finish_ready_requests()

    def _on_control(self, message: dict[str, Any]) -> None:
        if message.get("_type") not in {_RPC_RESPONSE, _RPC_ERROR}:
            return
        request_id = message.get("request_id")
        if not isinstance(request_id, str):
            log.warning("ignoring stage RPC response without a string request_id")
            return
        pending = self._pending.get(request_id)
        if pending is None:
            log.warning("ignoring response for unknown stage RPC request %r", request_id)
            return
        pending.response = message
        self._finish_ready_requests()

    def _finish_ready_requests(self) -> None:
        for request_id, pending in list(self._pending.items()):
            response = pending.response
            if response is None:
                continue
            if response.get("_type") == _RPC_ERROR:
                if not pending.future.done():
                    pending.future.set_exception(
                        RuntimeError(str(response.get("message", "stage RPC failed")))
                    )
                self._pending.pop(request_id, None)
                self._drop_tensor_payloads(request_id)
                continue
            if (
                response.get("operation_id") != pending.operation_id
                or response.get("attempt_id") != pending.attempt_id
            ):
                if not pending.future.done():
                    pending.future.set_exception(
                        RuntimeError(
                            "stage RPC response operation/attempt does not match request"
                        )
                    )
                self._pending.pop(request_id, None)
                self._drop_tensor_payloads(request_id)
                continue
            output_id = response.get("output_tensor_id")
            if pending.method == "forward_inference" and not isinstance(output_id, str):
                if not pending.future.done():
                    pending.future.set_exception(
                        RuntimeError("stage RPC forward response omitted output tensor ID")
                    )
                self._pending.pop(request_id, None)
                self._drop_tensor_payloads(request_id)
                continue
            if output_id is not None:
                if (
                    not isinstance(output_id, str)
                    or not output_id.startswith(f"rpc:{request_id}:")
                ):
                    if not pending.future.done():
                        pending.future.set_exception(
                            RuntimeError("stage RPC response has an invalid output tensor ID")
                        )
                    self._pending.pop(request_id, None)
                    self._drop_tensor_payloads(request_id)
                    continue
                output = self._tensor_payloads.get(output_id)
                if output is None:
                    continue
                pending.output = output
                self._tensor_payloads.pop(output_id, None)
            if not pending.future.done():
                pending.future.set_result((response, pending.output))
            self._pending.pop(request_id, None)
            self._drop_tensor_payloads(request_id)

    def _fail_pending(self, error: BaseException) -> None:
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()
        self._tensor_payloads.clear()

    def _drop_tensor_payloads(self, request_id: str) -> None:
        prefix = f"rpc:{request_id}:"
        for tensor_id in list(self._tensor_payloads):
            if tensor_id.startswith(prefix):
                self._tensor_payloads.pop(tensor_id, None)


class StageRpcServer:
    """Serve one ``StageWorker`` over authenticated WebSocket RPC."""

    def __init__(
        self,
        worker: StageWorker,
        identity: StageRpcIdentity,
        *,
        credential: str | None = None,
    ) -> None:
        if credential == "":
            raise ValueError("stage RPC credential must be non-empty or None")
        self._worker = worker
        self._identity = identity
        self._credential = credential

    async def serve_connection(self, ws: Any) -> None:
        if not self._authorized(ws):
            await ws.close(code=1008, reason="authentication failed")
            return
        data = DataConn(
            ws,
            **_header_dict(self._identity),
            credential=self._credential or "",
            peer_worker_incarnation=self._identity.peer_worker_incarnation,
        )
        requests: dict[str, _ServerRequest] = {}
        tasks: set[asyncio.Task] = set()
        owned_cache_keys: set[str] = set()
        # Cache names supplied by a client are labels, not global identities.
        # Namespace them per authenticated connection so two clients using
        # the same label cannot share or overwrite a StageWorker cache.
        cache_namespace = f"rpc-connection:{uuid.uuid4().hex}:"
        send_lock = asyncio.Lock()

        def spawn(coroutine: Any) -> None:
            task = asyncio.create_task(coroutine)
            tasks.add(task)

            def finish(completed: asyncio.Task) -> None:
                tasks.discard(completed)
                if completed.cancelled():
                    return
                try:
                    error = completed.exception()
                except asyncio.CancelledError:
                    return
                if error is not None:
                    log.warning("stage RPC connection task failed: %s", error)

            task.add_done_callback(finish)

        def on_control(message: dict[str, Any]) -> None:
            if message.get("_type") != _RPC_REQUEST:
                return
            try:
                request = _parse_request(message)
            except (TypeError, ValueError) as exc:
                spawn(
                    self._send_error(data, send_lock, message.get("request_id"), str(exc))
                )
                return
            if request.request_id in requests:
                spawn(
                    self._send_error(data, send_lock, request.request_id, "duplicate request_id")
                )
                return
            if len(requests) >= _MAX_PENDING_REQUESTS:
                spawn(
                    self._send_error(
                        data,
                        send_lock,
                        request.request_id,
                        "too many pending stage RPC requests",
                    )
                )
                return
            if request.method != "clear_all_kv":
                # StageRpcClient normally supplies a private default key, but
                # raw protocol clients may omit one.  Give those requests a
                # private named cache as well instead of using the worker's
                # process-wide legacy default context.
                client_cache_key = request.cache_key or "default"
                request.cache_key = cache_namespace + client_cache_key
            requests[request.request_id] = request
            # A cache becomes owned by this WebSocket when this connection
            # creates it through forward_inference.  Merely probing or
            # trimming an arbitrary cache key must not grant clear-all
            # authority over another client's cache.
            if request.method == "forward_inference":
                assert request.cache_key is not None
                owned_cache_keys.add(request.cache_key)
            spawn(maybe_dispatch(request.request_id))

        def on_tensor(_operation_id: int, meta: TensorMeta, raw: bytes) -> None:
            request_id = _request_id_from_tensor_id(meta.tensor_id)
            request = requests.get(request_id)
            if request is None or meta.tensor_id not in request.fields.values():
                spawn(
                    self._send_error(data, send_lock, request_id, "tensor without a request")
                )
                return
            if request.dispatched:
                # The request has already captured its complete input set;
                # late/duplicate chunks must not mutate the tensors that the
                # execution task is reading.
                log.warning(
                    "discarding late tensor %s for dispatched request %s",
                    meta.tensor_id,
                    request_id,
                )
                return
            if request.length + len(raw) > _MAX_REQUEST_INPUT_BYTES:
                request.dispatched = True
                requests.pop(request_id, None)
                spawn(
                    self._send_error(
                        data,
                        send_lock,
                        request_id,
                        "stage RPC request input exceeds byte limit",
                    )
                )
                return
            request.length += len(raw)
            field_name = next(
                name for name, tensor_id in request.fields.items()
                if tensor_id == meta.tensor_id
            )
            if field_name in request.tensors:
                request.dispatched = True
                requests.pop(request_id, None)
                spawn(
                    self._send_error(
                        data,
                        send_lock,
                        request_id,
                        f"duplicate tensor field: {field_name}",
                    )
                )
                return
            request.tensors[field_name] = (meta, raw)
            spawn(maybe_dispatch(request_id))

        async def maybe_dispatch(request_id: str) -> None:
            request = requests.get(request_id)
            if request is None or request.dispatched:
                return
            if any(field not in request.tensors for field in request.fields):
                return
            request.dispatched = True

            async def run_request() -> None:
                succeeded = False
                try:
                    succeeded = await self._execute_request(
                        data,
                        send_lock,
                        request,
                        owned_cache_keys=owned_cache_keys,
                    )
                    if succeeded and request.method == "clear_kv":
                        if request.cache_key is not None:
                            owned_cache_keys.discard(request.cache_key)
                    elif succeeded and request.method == "clear_all_kv":
                        owned_cache_keys.clear()
                finally:
                    requests.pop(request_id, None)

            spawn(run_request())

        try:
            await data.recv_messages(on_tensor, on_control)
        finally:
            for task in list(tasks):
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for cache_key in owned_cache_keys:
                self._worker.clear_kv(cache_key)

    async def serve(
        self,
        host: str,
        port: int,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> Any:
        """Start a WebSocket server and return its closeable server object."""
        from websockets.asyncio.server import serve as websocket_serve

        return await websocket_serve(
            self.serve_connection,
            host,
            port,
            ssl=ssl_context,
            max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
            ping_interval=_RPC_PING_INTERVAL_S,
            ping_timeout=_RPC_PING_TIMEOUT_S,
        )

    async def _execute_request(
        self,
        data: DataConn,
        send_lock: asyncio.Lock,
        request: _ServerRequest,
        *,
        owned_cache_keys: set[str],
    ) -> bool:
        try:
            tensors = {
                field_name: _tensor_from_wire(*request.tensors[field_name])
                for field_name in request.fields
            }
            if request.method == "forward_inference":
                result = await _to_thread_drain_on_cancel(
                    self._worker.forward_inference,
                    tensors.get("hidden"),
                    input_ids=tensors.get("input_ids"),
                    position_ids=tensors.get("position_ids"),
                    attention_mask=tensors.get("attention_mask"),
                    operation_id=request.operation_id,
                    attempt_id=request.attempt_id,
                    reset_kv=request.reset_kv,
                    cache_key=request.cache_key,
                )
                meta, raw = _tensor_wire_parts(
                    result.output, f"rpc:{request.request_id}:output"
                )
                async with send_lock:
                    await data.send_tensor(raw, meta, attempt_id=_frame_attempt(request.request_id))
                    await data.send_control({
                        "_type": _RPC_RESPONSE,
                        "request_id": request.request_id,
                        "stage_id": self._worker._ctx.stage_id,
                        "operation_id": result.operation_id,
                        "attempt_id": result.attempt_id,
                        "output_tensor_id": meta.tensor_id,
                        "kv_length": self._worker.kv_cache_length(request.cache_key),
                    })
                return True
            elif request.method == "clear_kv":
                await _to_thread_drain_on_cancel(
                    self._worker.clear_kv,
                    request.cache_key,
                )
                await self._send_ok(data, send_lock, request)
                return True
            elif request.method == "clear_all_kv":
                # ``clear_all_kv`` is scoped to this authenticated WebSocket.
                # StageWorker also serves other clients, so calling its
                # process-wide clear_all_kv here would destroy their sessions.
                for cache_key in tuple(owned_cache_keys):
                    await _to_thread_drain_on_cancel(self._worker.clear_kv, cache_key)
                await self._send_ok(data, send_lock, request)
                return True
            elif request.method == "kv_cache_length":
                length = await _to_thread_drain_on_cancel(
                    self._worker.kv_cache_length,
                    request.cache_key,
                )
                await self._send_ok(data, send_lock, request, kv_length=length)
                return True
            elif request.method == "trim_kv":
                await _to_thread_drain_on_cancel(
                    self._worker.trim_kv,
                    request.length,
                    request.cache_key,
                )
                await self._send_ok(data, send_lock, request)
                return True
            else:  # pragma: no cover - _parse_request rejects this
                raise ValueError(f"unsupported stage RPC method: {request.method!r}")
        except Exception as exc:
            log.exception("stage RPC request %s failed", request.request_id)
            await self._send_error(data, send_lock, request.request_id, str(exc))
            return False

    async def _send_ok(
        self,
        data: DataConn,
        send_lock: asyncio.Lock,
        request: _ServerRequest,
        **extra: Any,
    ) -> None:
        async with send_lock:
            await data.send_control({
                "_type": _RPC_RESPONSE,
                "request_id": request.request_id,
                "stage_id": self._worker._ctx.stage_id,
                "operation_id": request.operation_id,
                "attempt_id": request.attempt_id,
                **extra,
            })

    async def _send_error(
        self,
        data: DataConn,
        send_lock: asyncio.Lock,
        request_id: Any,
        message: str,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            return
        async with send_lock:
            await data.send_control({
                "_type": _RPC_ERROR,
                "request_id": request_id,
                "message": message,
            })

    def _authorized(self, ws: Any) -> bool:
        if self._credential is None:
            return True
        headers = getattr(ws, "request_headers", None)
        if headers is None:
            headers = getattr(getattr(ws, "request", None), "headers", None)
        if headers is None:
            headers = {}
        authorization = headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token, self._credential)


def _parse_request(message: dict[str, Any]) -> _ServerRequest:
    request_id = message.get("request_id")
    method = message.get("method")
    operation_id = message.get("operation_id")
    attempt_id = message.get("attempt_id")
    fields = message.get("tensors", {})
    if (
        not isinstance(request_id, str)
        or not request_id
        or len(request_id) > _MAX_ID_LENGTH
    ):
        raise ValueError("invalid request_id")
    if method not in {
        "forward_inference",
        "clear_kv",
        "clear_all_kv",
        "kv_cache_length",
        "trim_kv",
    }:
        raise ValueError(f"unsupported stage RPC method: {method!r}")
    if (
        isinstance(operation_id, bool)
        or not isinstance(operation_id, int)
        or not 0 <= operation_id < (1 << 32)
    ):
        raise ValueError("operation_id must be a non-negative uint32 integer")
    if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > _MAX_ID_LENGTH:
        raise ValueError("invalid attempt_id")
    if not isinstance(fields, dict):
        raise ValueError("tensors must be a mapping")
    normalized_fields: dict[str, str] = {}
    tensor_ids: set[str] = set()
    for field_name, tensor_id in fields.items():
        if field_name not in _ALLOWED_FIELDS:
            raise ValueError(f"unsupported tensor field: {field_name!r}")
        if not isinstance(tensor_id, str) or not tensor_id:
            raise ValueError(f"invalid tensor_id for field {field_name!r}")
        if tensor_id in tensor_ids:
            raise ValueError("tensor IDs must be unique within a request")
        tensor_ids.add(tensor_id)
        normalized_fields[field_name] = tensor_id
    cache_key = message.get("cache_key")
    if cache_key is not None and (
        not isinstance(cache_key, str)
        or not cache_key
        or len(cache_key) > _MAX_ID_LENGTH
    ):
        raise ValueError("invalid cache_key")
    reset_kv = message.get("reset_kv", False)
    if not isinstance(reset_kv, bool):
        raise ValueError("reset_kv must be boolean")
    request = _ServerRequest(
        request_id=request_id,
        method=method,
        operation_id=operation_id,
        attempt_id=attempt_id,
        reset_kv=reset_kv,
        cache_key=cache_key,
        fields=normalized_fields,
    )
    length = message.get("length", 0)
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("length must be an integer")
    request.length = length
    if request.length < 0:
        raise ValueError("length must be non-negative")
    if request.length > _MAX_REQUEST_INPUT_BYTES:
        raise ValueError("stage RPC request input exceeds byte limit")
    if method != "forward_inference" and normalized_fields:
        raise ValueError(f"method {method!r} does not accept tensor fields")
    return request


def _request_id_from_tensor_id(tensor_id: str) -> str:
    parts = tensor_id.split(":")
    if len(parts) < 3 or parts[0] != "rpc":
        return ""
    return parts[1]


def _frame_attempt(request_id: str) -> int:
    # Stable within a process and representable by the uint32 wire field.
    return uuid.UUID(request_id).int & 0xFFFFFFFF


def _validate_uint(name: str, value: int, bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must fit in an unsigned {bits}-bit field")


async def connect_stage_clients(
    urls: Sequence[str],
    credentials: Sequence[str],
    identities: Sequence[StageRpcIdentity],
    *,
    ssl_context: ssl.SSLContext | None = None,
    relay_token: str | None = None,
    request_timeout_s: float = 300.0,
    vocab_size: int | None = None,
    max_position_embeddings: int | None = None,
) -> list[StageRpcClient]:
    """Connect an ordered set of remote stages, closing partial results on error.

    ``urls[i]``, ``credentials[i]`` and ``identities[i]`` describe the same
    stage.  Connections are opened concurrently so a WAN deployment does not
    pay one handshake round-trip per stage.  The returned list preserves input
    order, which is the pipeline layer order.
    """
    if not urls:
        raise ValueError("at least one stage URL is required")
    if len(credentials) != len(urls) or len(identities) != len(urls):
        raise ValueError(
            "urls, credentials and identities must have the same number of stages"
        )
    if any(not url for url in urls):
        raise ValueError("stage URLs must not be empty")
    if any(not credential for credential in credentials):
        raise ValueError("stage credentials must not be empty")
    if relay_token == "":
        raise ValueError("relay_token must be non-empty or None")

    async def connect_one(index: int) -> StageRpcClient:
        return await StageRpcClient.connect(
            urls[index],
            credentials[index],
            identities[index],
            ssl_context=ssl_context,
            relay_token=relay_token,
            request_timeout_s=request_timeout_s,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
        )

    results = await asyncio.gather(
        *(connect_one(index) for index in range(len(urls))),
        return_exceptions=True,
    )
    clients: list[StageRpcClient] = []
    first_error: BaseException | None = None
    for result in results:
        if isinstance(result, BaseException):
            if first_error is None:
                first_error = result
        else:
            clients.append(result)
    if first_error is not None:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        if isinstance(first_error, asyncio.CancelledError):
            raise first_error
        raise ConnectionError(
            f"could not connect all stage RPC clients: {first_error}"
        ) from first_error
    return clients
