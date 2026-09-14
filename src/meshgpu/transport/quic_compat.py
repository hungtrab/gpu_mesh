"""
QUIC-compatible transport abstraction.

Provides an `AbstractTransport` protocol so the rest of the codebase is
transport-agnostic.  Today's default is WebSocket (TLS over TCP).  Swapping
to QUIC (or RDMA) requires only a new concrete implementation of this protocol.

Why QUIC matters for MeshGPU:
  - 0-RTT reconnect: after a worker reconnects it resumes immediately without
    a TCP + TLS handshake round-trip.
  - Stream multiplexing without head-of-line blocking: control frames, tensor
    chunks and credit updates can flow on independent QUIC streams; one large
    chunk does not delay a heartbeat.
  - Loss recovery without retransmitting buffered-but-already-received data
    (selective ACK is first-class in QUIC vs bolted-on for TCP).

RDMA note:
  RDMA (via libibverbs / rdma-core) bypasses the kernel network stack and
  DMA-transfers tensor bytes directly between GPU/NIC buffers.  The integration
  point is the same AbstractTransport interface; the implementation requires
  hardware support and is gated on P6.

QuicTransport stub:
  Requires `pip install aioquic`.  Not wired into the main path; present to
  document the integration surface and allow early experimentation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AbstractTransport(ABC):
    """
    Bidirectional, message-oriented transport.
    Each `send` delivers an opaque bytes payload; each `recv` iteration yields
    the next received payload.
    """

    @abstractmethod
    async def send(self, data: bytes) -> None: ...

    @abstractmethod
    def recv(self) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


@dataclass
class TransportMetrics:
    bytes_sent: int = 0
    bytes_recv: int = 0
    messages_sent: int = 0
    messages_recv: int = 0
    reconnects: int = 0


# ---------------------------------------------------------------------------
# WebSocket transport (current default)
# ---------------------------------------------------------------------------

class WebSocketTransport(AbstractTransport):
    """
    Wraps an aiohttp ClientWebSocketResponse or ServerWebSocketResponse.
    Provides the AbstractTransport contract over WebSocket frames.
    """

    def __init__(self, ws, metrics: TransportMetrics | None = None) -> None:
        self._ws = ws
        self._metrics = metrics or TransportMetrics()
        self._closed = False

    async def send(self, data: bytes) -> None:
        await self._ws.send_bytes(data)
        self._metrics.bytes_sent += len(data)
        self._metrics.messages_sent += 1

    async def recv(self) -> AsyncIterator[bytes]:
        async for msg in self._ws:
            import aiohttp
            if msg.type == aiohttp.WSMsgType.BINARY:
                self._metrics.bytes_recv += len(msg.data)
                self._metrics.messages_recv += 1
                yield msg.data
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break

    async def close(self) -> None:
        self._closed = True
        await self._ws.close()

    @property
    def is_connected(self) -> bool:
        return not self._closed and not self._ws.closed


# ---------------------------------------------------------------------------
# QUIC transport stub (requires aioquic)
# ---------------------------------------------------------------------------

class QuicTransport(AbstractTransport):
    """
    QUIC-based transport using aioquic.

    Multiplexes streams:
      - Stream 0: control messages (JSON / protobuf)
      - Stream 1: tensor chunks
      - Stream 2: credit tokens

    This is a STUB — the interface is defined but connect() is not
    implemented until aioquic is added as a dependency.

    To use in production:
      pip install aioquic
      Set MESHGPU_TRANSPORT=quic in the environment.
    """

    def __init__(self, host: str, port: int, tls_cert: str | None = None) -> None:
        self._host = host
        self._port = port
        self._tls_cert = tls_cert
        self._protocol = None
        self._metrics = TransportMetrics()

    async def connect(self) -> None:
        try:
            import aioquic  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "aioquic is required for QUIC transport: pip install aioquic"
            )
        raise NotImplementedError(
            "QuicTransport.connect() is a stub for P6 — use WebSocketTransport for now"
        )

    async def send(self, data: bytes) -> None:
        raise NotImplementedError("QuicTransport is a stub")

    async def recv(self) -> AsyncIterator[bytes]:
        raise NotImplementedError("QuicTransport is a stub")
        yield  # make it a generator

    async def close(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Transport factory
# ---------------------------------------------------------------------------

_TRANSPORTS = {"websocket": WebSocketTransport, "quic": QuicTransport}


def get_transport_class(name: str = "websocket") -> type:
    """
    Return transport class by name.
    Override via MESHGPU_TRANSPORT env var.
    """
    import os
    name = os.environ.get("MESHGPU_TRANSPORT", name).lower()
    cls = _TRANSPORTS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown transport {name!r}. Available: {list(_TRANSPORTS)}"
        )
    return cls


def make_transport(ws, name: str = "websocket") -> AbstractTransport:
    """Convenience: wrap an existing WebSocket as the default transport."""
    cls = get_transport_class(name)
    if cls is WebSocketTransport:
        return WebSocketTransport(ws)
    raise ValueError(f"Cannot wrap ws with {name!r} transport; create directly")
