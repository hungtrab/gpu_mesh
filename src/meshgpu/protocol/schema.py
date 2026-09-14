"""Wire protocol schema — frame metadata and operation identity."""
from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import ClassVar

PROTOCOL_VERSION = 1
FRAME_MAGIC = b"MGF\x01"  # MeshGPU Frame v1
MAX_TENSOR_BYTES = 512 * 1024 * 1024  # 512 MiB hard limit per frame


class Phase(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    CHECKPOINT = "checkpoint"


class DType(str, enum.Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    INT32 = "int32"
    INT64 = "int64"

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self]


_DTYPE_ITEMSIZE: dict[DType, int] = {
    DType.FLOAT32: 4,
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.INT8: 1,
    DType.INT4: 1,  # packed; caller must account for sub-byte packing
    DType.INT32: 4,
    DType.INT64: 8,
}


@dataclass(frozen=True, slots=True)
class TensorMeta:
    tensor_id: str
    dtype: DType
    shape: tuple[int, ...]
    layout: str = "contiguous"  # only "contiguous" supported in v0

    @property
    def byte_length(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * self.dtype.itemsize

    def validate(self) -> None:
        if not isinstance(self.tensor_id, str) or not self.tensor_id:
            raise ValueError("tensor_id must not be empty")
        if not isinstance(self.dtype, DType):
            raise TypeError(f"dtype must be a DType, got {type(self.dtype).__name__}")
        if self.dtype is DType.INT4:
            raise ValueError(
                "INT4 wire tensors are unsupported until an explicit packed-byte "
                "encoding is implemented"
            )
        if not self.shape:
            raise ValueError("shape must not be empty")
        for d in self.shape:
            if isinstance(d, bool) or not isinstance(d, int):
                raise TypeError(f"shape dimensions must be integers, got {type(d).__name__}")
            if d <= 0:
                raise ValueError(f"shape dimension must be positive, got {d}")
        if self.byte_length > MAX_TENSOR_BYTES:
            raise ValueError(
                f"tensor byte_length {self.byte_length} exceeds limit {MAX_TENSOR_BYTES}"
            )
        if self.layout != "contiguous":
            raise ValueError(f"unsupported layout: {self.layout!r}")


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """Fixed-size binary header prefixed to every frame payload."""

    STRUCT_FMT: ClassVar[str] = "!4sHHIIIIIHHQI"
    SIZE: ClassVar[int] = struct.calcsize("!4sHHIIIIIHHQI")  # 44 bytes

    protocol_version: int
    cluster_id: int
    job_id: int
    lease_epoch: int
    worker_incarnation: int
    operation_id: int
    attempt_id: int
    chunk_index: int
    chunk_count: int
    byte_length: int  # payload bytes in this chunk
    payload_checksum: int  # CRC32 of payload bytes

    def __post_init__(self) -> None:
        """Reject values that cannot be represented by the wire struct."""
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version != PROTOCOL_VERSION
        ):
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        _check_uint("cluster_id", self.cluster_id, 16)
        for name, value in (
            ("job_id", self.job_id),
            ("lease_epoch", self.lease_epoch),
            ("worker_incarnation", self.worker_incarnation),
            ("operation_id", self.operation_id),
            ("attempt_id", self.attempt_id),
        ):
            _check_uint(name, value, 32)
        _check_uint("chunk_index", self.chunk_index, 16)
        _check_uint("chunk_count", self.chunk_count, 16)
        _check_uint("byte_length", self.byte_length, 64)
        _check_uint("payload_checksum", self.payload_checksum, 32)
        if self.chunk_count < 1:
            raise ValueError("chunk_count must be positive")
        if self.chunk_index >= self.chunk_count:
            raise ValueError(
                f"chunk_index {self.chunk_index} outside [0, {self.chunk_count})"
            )

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FMT,
            FRAME_MAGIC,
            self.protocol_version,
            self.cluster_id,
            self.job_id,
            self.lease_epoch,
            self.worker_incarnation,
            self.operation_id,
            self.attempt_id,
            self.chunk_index,
            self.chunk_count,
            self.byte_length,
            self.payload_checksum,
        )

    @classmethod
    def unpack(cls, data: bytes) -> FrameHeader:
        if len(data) < cls.SIZE:
            raise ValueError(f"header too short: {len(data)} < {cls.SIZE}")
        (
            magic,
            version,
            cluster_id,
            job_id,
            lease_epoch,
            worker_incarnation,
            operation_id,
            attempt_id,
            chunk_index,
            chunk_count,
            byte_length,
            checksum,
        ) = struct.unpack_from(cls.STRUCT_FMT, data)
        if magic != FRAME_MAGIC:
            raise ValueError(f"bad frame magic: {magic!r}")
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {version}")
        return cls(
            protocol_version=version,
            cluster_id=cluster_id,
            job_id=job_id,
            lease_epoch=lease_epoch,
            worker_incarnation=worker_incarnation,
            operation_id=operation_id,
            attempt_id=attempt_id,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            byte_length=byte_length,
            payload_checksum=checksum,
        )


def checksum32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def _check_uint(name: str, value: int, bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must fit in an unsigned {bits}-bit field")
