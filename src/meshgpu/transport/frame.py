"""Frame assembly, disassembly and validation for tensor data plane."""
from __future__ import annotations

from dataclasses import dataclass

from meshgpu.protocol.schema import (
    FrameHeader,
    TensorMeta,
    checksum32,
)

DEFAULT_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB
MIN_CHUNK_SIZE = 64 * 1024  # 64 KiB
MAX_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


@dataclass
class TensorFrame:
    header: FrameHeader
    meta: TensorMeta
    payload: bytes  # raw tensor bytes for this chunk

    def validate_checksum(self) -> None:
        got = checksum32(self.payload)
        if got != self.header.payload_checksum:
            raise ValueError(
                f"checksum mismatch: expected {self.header.payload_checksum:#010x}, "
                f"got {got:#010x}"
            )

    def validate_size(self) -> None:
        self.meta.validate()
        if self.header.chunk_count < 1:
            raise ValueError("header chunk_count must be positive")
        if self.header.chunk_index < 0 or self.header.chunk_index >= self.header.chunk_count:
            raise ValueError(
                f"chunk_index {self.header.chunk_index} outside "
                f"[0, {self.header.chunk_count})"
            )
        if len(self.payload) != self.header.byte_length:
            raise ValueError(
                f"payload length {len(self.payload)} != header byte_length "
                f"{self.header.byte_length}"
            )
        if self.header.byte_length > MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk byte_length {self.header.byte_length} exceeds limit {MAX_CHUNK_SIZE}"
            )


def chunk_tensor(
    raw: bytes,
    meta: TensorMeta,
    *,
    cluster_id: int,
    job_id: int,
    lease_epoch: int,
    worker_incarnation: int,
    operation_id: int,
    attempt_id: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[TensorFrame]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError(f"raw tensor data must be bytes-like, got {type(raw).__name__}")
    if isinstance(raw, (bytearray, memoryview)):
        raw = bytes(raw)
    meta.validate()
    if len(raw) != meta.byte_length:
        raise ValueError(
            f"raw bytes {len(raw)} != expected {meta.byte_length} for {meta}"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1 or chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be in [1, {MAX_CHUNK_SIZE}], got {chunk_size}"
        )

    chunk_count = (len(raw) + chunk_size - 1) // chunk_size or 1
    if chunk_count > 0xFFFF:
        raise ValueError(
            f"tensor requires {chunk_count} chunks; maximum is {0xFFFF}"
        )
    chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)] or [b""]
    frames = []
    for idx, chunk in enumerate(chunks):
        hdr = FrameHeader(
            protocol_version=1,
            cluster_id=cluster_id,
            job_id=job_id,
            lease_epoch=lease_epoch,
            worker_incarnation=worker_incarnation,
            operation_id=operation_id,
            attempt_id=attempt_id,
            chunk_index=idx,
            chunk_count=chunk_count,
            byte_length=len(chunk),
            payload_checksum=checksum32(chunk),
        )
        frames.append(TensorFrame(header=hdr, meta=meta, payload=chunk))
    return frames


class TensorAssembler:
    """Reassemble chunked frames into a complete tensor."""

    def __init__(
        self,
        meta: TensorMeta,
        chunk_count: int,
        *,
        operation_id: int | None = None,
        attempt_id: int | None = None,
    ) -> None:
        meta.validate()
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
            raise TypeError("chunk_count must be an integer")
        if chunk_count < 1:
            raise ValueError("chunk_count must be positive")
        if chunk_count > 0xFFFF:
            raise ValueError("chunk_count exceeds the uint16 wire limit")
        if chunk_count > meta.byte_length:
            raise ValueError("chunk_count exceeds tensor byte length")
        self._meta = meta
        self._chunk_count = chunk_count
        self._operation_id = operation_id
        self._attempt_id = attempt_id
        self._chunks: dict[int, bytes] = {}
        self._received_bytes = 0

    def feed(self, frame: TensorFrame) -> bool:
        """Return True when all chunks have arrived."""
        frame.validate_size()
        frame.validate_checksum()
        if frame.meta != self._meta:
            raise ValueError("tensor metadata changed while assembling chunks")
        if frame.header.chunk_count != self._chunk_count:
            raise ValueError(
                f"chunk_count {frame.header.chunk_count} != expected {self._chunk_count}"
            )
        if (
            self._operation_id is not None
            and frame.header.operation_id != self._operation_id
        ):
            raise ValueError("operation_id changed while assembling chunks")
        if self._attempt_id is not None and frame.header.attempt_id != self._attempt_id:
            raise ValueError("attempt_id changed while assembling chunks")
        idx = frame.header.chunk_index
        if idx in self._chunks:
            # idempotent: same chunk received twice is fine if payload matches
            if self._chunks[idx] != frame.payload:
                raise ValueError(f"conflicting payload for chunk {idx}")
            return self.complete
        if self._received_bytes + len(frame.payload) > self._meta.byte_length:
            raise ValueError(
                f"received {self._received_bytes + len(frame.payload)} bytes, "
                f"exceeds tensor byte_length {self._meta.byte_length}"
            )
        self._chunks[idx] = frame.payload
        self._received_bytes += len(frame.payload)
        return self.complete

    @property
    def complete(self) -> bool:
        return len(self._chunks) == self._chunk_count

    def assemble(self) -> bytes:
        if not self.complete:
            raise RuntimeError("not all chunks received")
        raw = b"".join(self._chunks[i] for i in range(self._chunk_count))
        if len(raw) != self._meta.byte_length:
            raise ValueError(
                f"assembled {len(raw)} bytes, expected {self._meta.byte_length}"
            )
        return raw
