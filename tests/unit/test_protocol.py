"""Unit tests for protocol schema and frame assembly."""
import pytest

from meshgpu.protocol.schema import (
    DType,
    FrameHeader,
    TensorMeta,
)
from meshgpu.transport.frame import TensorAssembler, chunk_tensor


def make_meta(shape=(4, 8), dtype=DType.FLOAT32) -> TensorMeta:
    return TensorMeta(tensor_id="t1", dtype=dtype, shape=shape)


# ------------------------------------------------------------------
# TensorMeta validation
# ------------------------------------------------------------------

def test_meta_byte_length():
    meta = make_meta((4, 8), DType.FLOAT32)
    assert meta.byte_length == 4 * 8 * 4


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [(DType.INT32, 4), (DType.INT64, 8)],
)
def test_integer_metadata_has_wire_itemsize(dtype, itemsize):
    assert TensorMeta("ids", dtype, (3, 2)).byte_length == 3 * 2 * itemsize


def test_meta_rejects_zero_dim():
    meta = TensorMeta(tensor_id="t", dtype=DType.FLOAT32, shape=(0, 8))
    with pytest.raises(ValueError, match="positive"):
        meta.validate()


def test_meta_rejects_empty_shape():
    meta = TensorMeta(tensor_id="t", dtype=DType.FLOAT32, shape=())
    with pytest.raises(ValueError, match="shape must not be empty"):
        meta.validate()


def test_meta_rejects_empty_id():
    meta = TensorMeta(tensor_id="", dtype=DType.FLOAT32, shape=(4,))
    with pytest.raises(ValueError, match="tensor_id"):
        meta.validate()


def test_meta_rejects_unsupported_layout():
    meta = TensorMeta(tensor_id="t", dtype=DType.FLOAT32, shape=(4,), layout="strided")
    with pytest.raises(ValueError, match="layout"):
        meta.validate()


def test_meta_rejects_unimplemented_int4_wire_encoding():
    meta = TensorMeta(tensor_id="weights", dtype=DType.INT4, shape=(8,))
    with pytest.raises(ValueError, match="INT4 wire tensors"):
        meta.validate()


def test_meta_rejects_non_string_tensor_id():
    meta = TensorMeta(tensor_id=123, dtype=DType.FLOAT32, shape=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tensor_id"):
        meta.validate()


# ------------------------------------------------------------------
# FrameHeader pack/unpack round-trip
# ------------------------------------------------------------------

def test_frame_header_roundtrip():
    hdr = FrameHeader(
        protocol_version=1,
        cluster_id=1,
        job_id=42,
        lease_epoch=7,
        worker_incarnation=3,
        operation_id=99,
        attempt_id=1,
        chunk_index=0,
        chunk_count=1,
        byte_length=128,
        payload_checksum=0xDEADBEEF,
    )
    unpacked = FrameHeader.unpack(hdr.pack())
    assert unpacked == hdr


def test_frame_header_bad_magic():
    data = b"\x00\x00\x00\x00" + b"\x00" * (FrameHeader.SIZE - 4)
    with pytest.raises(ValueError, match="magic"):
        FrameHeader.unpack(data)


def test_frame_header_rejects_boolean_protocol_version():
    with pytest.raises(ValueError, match="protocol version"):
        FrameHeader(
            protocol_version=True,
            cluster_id=1,
            job_id=42,
            lease_epoch=7,
            worker_incarnation=3,
            operation_id=99,
            attempt_id=1,
            chunk_index=0,
            chunk_count=1,
            byte_length=1,
            payload_checksum=0,
        )


@pytest.mark.parametrize("chunk_size", [True, 1.5])
def test_chunk_tensor_rejects_non_integer_chunk_size(chunk_size):
    meta = TensorMeta("t", DType.INT8, (1,))
    with pytest.raises(TypeError, match="chunk_size"):
        chunk_tensor(
            b"x",
            meta,
            cluster_id=1,
            job_id=1,
            lease_epoch=1,
            worker_incarnation=1,
            operation_id=1,
            attempt_id=1,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", -1),
        ("job_id", 1 << 32),
        ("chunk_index", 1),
        ("chunk_count", 0),
        ("byte_length", -1),
    ],
)
def test_frame_header_rejects_invalid_wire_values(field, value):
    values = dict(
        protocol_version=1,
        cluster_id=1,
        job_id=42,
        lease_epoch=7,
        worker_incarnation=3,
        operation_id=99,
        attempt_id=1,
        chunk_index=0,
        chunk_count=1,
        byte_length=128,
        payload_checksum=0xDEADBEEF,
    )
    values[field] = value
    if field == "chunk_index":
        values["chunk_count"] = 1
    with pytest.raises(ValueError):
        FrameHeader(**values)


# ------------------------------------------------------------------
# chunk_tensor + TensorAssembler round-trip
# ------------------------------------------------------------------

def _make_raw(meta: TensorMeta) -> bytes:
    return (bytes(range(256)) * (meta.byte_length // 256 + 1))[: meta.byte_length]


def _round_trip(meta: TensorMeta, chunk_size: int) -> bytes:
    raw = _make_raw(meta)
    frames = chunk_tensor(
        raw,
        meta,
        cluster_id=1,
        job_id=1,
        lease_epoch=1,
        worker_incarnation=1,
        operation_id=1,
        attempt_id=1,
        chunk_size=chunk_size,
    )
    chunk_count = frames[0].header.chunk_count
    asm = TensorAssembler(meta, chunk_count)
    for frame in frames:
        done = asm.feed(frame)
    assert done
    return asm.assemble()


def test_round_trip_single_chunk():
    meta = make_meta((8, 8))
    raw = _make_raw(meta)
    assert _round_trip(meta, 4096) == raw


def test_round_trip_multi_chunk():
    meta = make_meta((64, 64))  # 64*64*4 = 16384 bytes
    raw = _make_raw(meta)
    result = _round_trip(meta, 1024)
    assert result == raw


def test_checksum_mismatch_detected():
    from meshgpu.transport.frame import TensorFrame
    meta = make_meta((4,))
    raw = _make_raw(meta)[: meta.byte_length]
    frames = chunk_tensor(
        raw, meta,
        cluster_id=1, job_id=1, lease_epoch=1,
        worker_incarnation=1, operation_id=1, attempt_id=1,
    )
    f = frames[0]
    bad_frame = TensorFrame(header=f.header, meta=meta, payload=b"\xFF" * len(f.payload))
    with pytest.raises(ValueError, match="checksum"):
        bad_frame.validate_checksum()


def test_duplicate_chunk_same_payload_ok():
    meta = make_meta((4,))
    raw = _make_raw(meta)[: meta.byte_length]
    frames = chunk_tensor(
        raw, meta,
        cluster_id=1, job_id=1, lease_epoch=1,
        worker_incarnation=1, operation_id=1, attempt_id=1,
    )
    asm = TensorAssembler(meta, 1)
    asm.feed(frames[0])
    asm.feed(frames[0])  # duplicate — same payload, should be fine
    assert asm.assemble() == raw


def test_assembler_rejects_declared_chunks_larger_than_tensor():
    """Malformed chunk streams must be rejected before buffering excess bytes."""
    from meshgpu.protocol.schema import checksum32
    from meshgpu.transport.frame import TensorFrame

    meta = TensorMeta("small", DType.INT8, (4,))
    payload = b"12345"  # five bytes for a four-byte tensor
    frame = TensorFrame(
        header=FrameHeader(
            protocol_version=1,
            cluster_id=1,
            job_id=1,
            lease_epoch=1,
            worker_incarnation=1,
            operation_id=1,
            attempt_id=1,
            chunk_index=0,
            chunk_count=1,
            byte_length=len(payload),
            payload_checksum=checksum32(payload),
        ),
        meta=meta,
        payload=payload,
    )
    assembler = TensorAssembler(meta, 1)

    with pytest.raises(ValueError, match="exceeds tensor byte_length"):
        assembler.feed(frame)
    assert not assembler.complete
