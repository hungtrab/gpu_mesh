"""Authenticated WebSocket connection for control and data planes."""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
from collections.abc import Callable
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection

from meshgpu.protocol.schema import FrameHeader, TensorMeta
from meshgpu.transport.credit import CreditManager
from meshgpu.transport.frame import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    TensorAssembler,
    TensorFrame,
    chunk_tensor,
)

log = logging.getLogger(__name__)

# Credit window: 32 MiB default; sender blocks when receiver hasn't ACKed
_DEFAULT_CREDIT_BYTES = 32 * 1024 * 1024
_DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
_DEFAULT_MAX_PENDING_TENSORS = 128
_CONNECTION_PING_INTERVAL_S = 30.0
_CONNECTION_PING_TIMEOUT_S = 300.0
# WebSocket's message limit must include the tag and per-chunk FrameHeader;
# otherwise the default 1 MiB chunk is rejected before our own validator sees it.
MAX_WEBSOCKET_MESSAGE_BYTES = MAX_CHUNK_SIZE + FrameHeader.SIZE + 1 + 64 * 1024

# Wire prefix bytes: 1 byte type tag
_TAG_CONTROL = b"\x00"   # JSON control message
_TAG_TENSOR_HEADER = b"\x01"  # FrameHeader bytes (44 B) + TensorMeta JSON
_TAG_TENSOR_CHUNK = b"\x02"  # raw chunk payload
_TAG_CREDIT = b"\x03"    # credit grant: 4-byte big-endian uint


class DataConn:
    """
    Single authenticated WebSocket connection for the tensor data plane.

    Protocol per tensor:
      1. sender → TENSOR_HEADER (FrameHeader + TensorMeta JSON)
      2. sender → TENSOR_CHUNK × chunk_count
      3. receiver → CREDIT (byte count released back)
    """

    def __init__(
        self,
        ws: ClientConnection,
        *,
        cluster_id: int,
        job_id: int,
        lease_epoch: int,
        worker_incarnation: int,
        credential: str,
        credit_bytes: int = _DEFAULT_CREDIT_BYTES,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        peer_worker_incarnation: int | None = None,
        max_pending_tensors: int = _DEFAULT_MAX_PENDING_TENSORS,
    ) -> None:
        if isinstance(credit_bytes, bool) or not isinstance(credit_bytes, int):
            raise TypeError("credit_bytes must be an integer")
        if credit_bytes < 1:
            raise ValueError("credit_bytes must be positive")
        if chunk_size < 1 or chunk_size > MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be in [1, {MAX_CHUNK_SIZE}], got {chunk_size}"
            )
        # The first chunk is sent before the receiver can return any credit;
        # allowing a larger chunk would make a valid connection wait forever.
        if chunk_size > credit_bytes:
            raise ValueError("chunk_size must not exceed the initial credit window")
        if isinstance(max_pending_tensors, bool) or not isinstance(
            max_pending_tensors, int
        ):
            raise TypeError("max_pending_tensors must be an integer")
        if max_pending_tensors < 1:
            raise ValueError("max_pending_tensors must be positive")
        self._ws = ws
        self._cluster_id = cluster_id
        self._job_id = job_id
        self._lease_epoch = lease_epoch
        self._worker_incarnation = worker_incarnation
        self._credential = credential
        self._peer_worker_incarnation = peer_worker_incarnation
        self._chunk_size = chunk_size
        self._max_pending_tensors = max_pending_tensors
        self._credit = CreditManager(credit_bytes, credit_bytes * 2)
        self._op_seq = 0
        self._tensor_send_lock = asyncio.Lock()

    def _next_op_id(self) -> int:
        self._op_seq += 1
        return self._op_seq

    async def send_tensor(
        self,
        raw: bytes,
        meta: TensorMeta,
        *,
        attempt_id: int,
    ) -> int:
        """Send a tensor; returns operation_id used."""
        # Keep a tensor's header/chunk sequence together when multiple caller
        # tasks share one connection.  Credit messages intentionally do not
        # use this lock: the receiver must be able to grant credit while a
        # sender is waiting for that credit.
        async with self._tensor_send_lock:
            op_id = self._next_op_id()
            frames = chunk_tensor(
                raw,
                meta,
                cluster_id=self._cluster_id,
                job_id=self._job_id,
                lease_epoch=self._lease_epoch,
                worker_incarnation=self._worker_incarnation,
                operation_id=op_id,
                attempt_id=attempt_id,
                chunk_size=self._chunk_size,
            )
            # Send header frame (metadata only, no payload)
            hdr_msg = _TAG_TENSOR_HEADER + frames[0].header.pack() + json.dumps(
                {
                    "tensor_id": meta.tensor_id,
                    "dtype": meta.dtype.value,
                    "shape": list(meta.shape),
                    "layout": meta.layout,
                }
            ).encode()
            await self._ws.send(hdr_msg)

            for frame in frames:
                await self._credit.acquire(frame.header.byte_length)
                await self._ws.send(
                    _TAG_TENSOR_CHUNK + frame.header.pack() + frame.payload
                )

            return op_id

    async def grant_credit(self, n: int) -> None:
        """Tell sender we consumed n bytes."""
        import struct
        if isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > 0xFFFFFFFF:
            raise ValueError(f"credit amount out of range: {n}")
        await self._ws.send(_TAG_CREDIT + struct.pack("!I", n))

    async def send_control(self, msg: dict) -> None:
        await self._ws.send(_TAG_CONTROL + json.dumps(msg).encode())

    async def recv_messages(
        self,
        on_tensor: Callable[[int, TensorMeta, bytes], None],
        on_control: Callable[[dict], None],
    ) -> None:
        """Receive loop; calls handlers inline (run in background task)."""
        import struct

        assemblers: dict[int, TensorAssembler] = {}
        pending_meta: dict[int, TensorMeta] = {}
        # Keep the descriptor received on the header so chunk zero can be
        # checked against it.  Without this, a peer could advertise one
        # checksum/length in the header and send a different descriptor in
        # the first chunk while still passing TensorAssembler validation.
        pending_headers: dict[int, FrameHeader] = {}

        async for raw in self._ws:
            if not isinstance(raw, bytes) or len(raw) < 1:
                continue
            tag, body = raw[:1], raw[1:]

            if tag == _TAG_CONTROL:
                try:
                    on_control(json.loads(body))
                except Exception:
                    log.exception("error in control handler")

            elif tag == _TAG_TENSOR_HEADER:
                hdr = FrameHeader.unpack(body[: FrameHeader.SIZE])
                if not self._header_matches_context(hdr):
                    continue
                if hdr.chunk_index != 0:
                    raise ValueError("tensor header must describe chunk_index=0")
                if hdr.byte_length < 1 or hdr.byte_length > MAX_CHUNK_SIZE:
                    raise ValueError(
                        "tensor header first-chunk byte_length must be in "
                        f"[1, {MAX_CHUNK_SIZE}]"
                    )
                meta_json = json.loads(body[FrameHeader.SIZE :])
                from meshgpu.protocol.schema import DType
                meta = TensorMeta(
                    tensor_id=meta_json["tensor_id"],
                    dtype=DType(meta_json["dtype"]),
                    shape=tuple(meta_json["shape"]),
                    layout=meta_json.get("layout", "contiguous"),
                )
                meta.validate()
                op_id = hdr.operation_id
                if hdr.chunk_count > meta.byte_length:
                    raise ValueError("chunk_count exceeds tensor byte length")
                if op_id in assemblers:
                    raise ValueError(f"duplicate tensor header for operation_id={op_id}")
                if len(assemblers) >= self._max_pending_tensors:
                    raise ValueError(
                        "too many pending tensor headers; peer must send chunks "
                        "before opening more tensor operations"
                    )
                pending_meta[op_id] = meta
                pending_headers[op_id] = hdr
                assemblers[op_id] = TensorAssembler(
                    meta,
                    hdr.chunk_count,
                    operation_id=hdr.operation_id,
                    attempt_id=hdr.attempt_id,
                )

            elif tag == _TAG_TENSOR_CHUNK:
                hdr = FrameHeader.unpack(body[: FrameHeader.SIZE])
                if not self._header_matches_context(hdr):
                    # The websocket message has already been consumed.  Return
                    # its payload credit even though the frame is stale, or a
                    # sender that retries on this live connection can block
                    # forever after its lease changes.
                    await self.grant_credit(len(body) - FrameHeader.SIZE)
                    continue
                op_id = hdr.operation_id
                assembler = assemblers.get(op_id)
                chunk_meta = pending_meta.get(op_id)
                if assembler is None or chunk_meta is None:
                    raise ValueError(f"tensor chunk without header for operation_id={op_id}")
                payload = body[FrameHeader.SIZE :]
                header_descriptor = pending_headers.get(op_id)
                if header_descriptor is None:
                    raise RuntimeError(
                        f"tensor header disappeared for operation_id={op_id}"
                    )
                if hdr.chunk_index == 0 and (
                    hdr.byte_length != header_descriptor.byte_length
                    or hdr.payload_checksum != header_descriptor.payload_checksum
                ):
                    raise ValueError(
                        "tensor chunk zero does not match its tensor header descriptor"
                    )
                done = assembler.feed(
                    TensorFrame(header=hdr, meta=chunk_meta, payload=payload)
                )
                if done:
                    assembled = assembler.assemble()
                    assemblers.pop(op_id, None)
                    m = pending_meta.pop(op_id, None)
                    pending_headers.pop(op_id, None)
                    if m is None:  # defensive: assembler/meta lifetimes must match
                        raise RuntimeError(
                            f"tensor metadata disappeared for operation_id={op_id}"
                        )
                    try:
                        on_tensor(op_id, m, assembled)
                    except Exception:
                        log.exception("error in tensor handler")
                await self.grant_credit(len(payload))

            elif tag == _TAG_CREDIT:
                if len(body) != 4:
                    raise ValueError("credit message must contain exactly four bytes")
                n = struct.unpack("!I", body[:4])[0]
                await self._credit.release_async(n)

    def _header_matches_context(self, header: FrameHeader) -> bool:
        """Validate routing identity and discard stale lease/incarnation data."""
        if header.cluster_id != self._cluster_id or header.job_id != self._job_id:
            raise ValueError("frame routing identity does not match this connection")
        if header.lease_epoch != self._lease_epoch or (
            self._peer_worker_incarnation is not None
            and header.worker_incarnation != self._peer_worker_incarnation
        ):
            log.warning(
                "discarding stale frame operation=%d lease=%d incarnation=%d",
                header.operation_id,
                header.lease_epoch,
                header.worker_incarnation,
            )
            return False
        return True


async def connect(
    url: str,
    credential: str,
    *,
    cluster_id: int,
    job_id: int,
    lease_epoch: int,
    worker_incarnation: int,
    ssl_ctx: ssl.SSLContext | None = None,
    credit_bytes: int = _DEFAULT_CREDIT_BYTES,
    peer_worker_incarnation: int | None = None,
) -> DataConn:
    from websockets.asyncio.client import connect as websocket_connect

    parsed_url = urlsplit(url)
    if parsed_url.scheme == "wss" and ssl_ctx is None:
        ssl_ctx = ssl.create_default_context()
    headers = {"Authorization": f"Bearer {credential}"}
    ws = await websocket_connect(
        url,
        additional_headers=headers,
        ssl=ssl_ctx,
        max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
        ping_interval=_CONNECTION_PING_INTERVAL_S,
        ping_timeout=_CONNECTION_PING_TIMEOUT_S,
    )
    return DataConn(
        ws,
        cluster_id=cluster_id,
        job_id=job_id,
        lease_epoch=lease_epoch,
        worker_incarnation=worker_incarnation,
        credential=credential,
        credit_bytes=credit_bytes,
        peer_worker_incarnation=peer_worker_incarnation,
    )
