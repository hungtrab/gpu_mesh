"""End-to-end tests for the binary DataConn framing contract."""

import json
from dataclasses import replace

import pytest

from meshgpu.protocol.schema import DType, TensorMeta
from meshgpu.transport.connection import (
    _TAG_TENSOR_CHUNK,
    _TAG_TENSOR_HEADER,
    DataConn,
)
from meshgpu.transport.frame import TensorFrame, chunk_tensor


class _FakeWebSocket:
    def __init__(self, incoming=None):
        self.sent: list[bytes] = []
        self._incoming = list(incoming or [])

    async def send(self, data: bytes) -> None:
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._incoming:
            raise StopAsyncIteration
        return self._incoming.pop(0)


def _conn(
    ws,
    *,
    peer_incarnation=4,
    worker_incarnation=9,
    chunk_size=4,
    credit_bytes=16,
    max_pending_tensors=128,
):
    return DataConn(
        ws,
        cluster_id=1,
        job_id=2,
        lease_epoch=3,
        worker_incarnation=worker_incarnation,
        credential="test",
        peer_worker_incarnation=peer_incarnation,
        chunk_size=chunk_size,
        credit_bytes=credit_bytes,
        max_pending_tensors=max_pending_tensors,
    )


def _header_message(frame: TensorFrame) -> bytes:
    return _TAG_TENSOR_HEADER + frame.header.pack() + json.dumps(
        {
            "tensor_id": frame.meta.tensor_id,
            "dtype": frame.meta.dtype.value,
            "shape": list(frame.meta.shape),
            "layout": frame.meta.layout,
        }
    ).encode()


def _chunk_message(frame: TensorFrame) -> bytes:
    return _TAG_TENSOR_CHUNK + frame.header.pack() + frame.payload


@pytest.mark.asyncio
async def test_send_and_receive_tensor_round_trip():
    sender_ws = _FakeWebSocket()
    sender = _conn(sender_ws)
    meta = TensorMeta("hidden", DType.FLOAT32, (3,))
    raw = b"abcdefghijkl"

    operation_id = await sender.send_tensor(raw, meta, attempt_id=7)
    receiver_ws = _FakeWebSocket(sender_ws.sent)
    receiver = _conn(receiver_ws, peer_incarnation=9)
    received = []

    await receiver.recv_messages(
        lambda op, received_meta, payload: received.append((op, received_meta, payload)),
        lambda _: None,
    )

    assert operation_id == 1
    assert received == [(1, meta, raw)]
    # One credit message is returned for each payload chunk.
    assert len(receiver_ws.sent) == 3


@pytest.mark.asyncio
async def test_receiver_assembles_interleaved_operations():
    meta_a = TensorMeta("a", DType.FLOAT32, (2,))
    meta_b = TensorMeta("b", DType.FLOAT32, (2,))
    frames_a = chunk_tensor(
        b"12345678", meta_a, cluster_id=1, job_id=2, lease_epoch=3,
        worker_incarnation=4, operation_id=10, attempt_id=1, chunk_size=4,
    )
    frames_b = chunk_tensor(
        b"ABCDEFGH", meta_b, cluster_id=1, job_id=2, lease_epoch=3,
        worker_incarnation=4, operation_id=11, attempt_id=2, chunk_size=4,
    )
    incoming = [
        _header_message(frames_a[0]),
        _header_message(frames_b[0]),
        _chunk_message(frames_a[1]),
        _chunk_message(frames_b[0]),
        _chunk_message(frames_a[0]),
        _chunk_message(frames_b[1]),
    ]
    ws = _FakeWebSocket(incoming)
    conn = _conn(ws)
    received = []

    await conn.recv_messages(
        lambda op, meta, payload: received.append((op, meta.tensor_id, payload)),
        lambda _: None,
    )

    assert received == [(10, "a", b"12345678"), (11, "b", b"ABCDEFGH")]


@pytest.mark.asyncio
async def test_stale_frames_are_discarded():
    meta = TensorMeta("stale", DType.FLOAT32, (1,))
    frames = chunk_tensor(
        b"1234", meta, cluster_id=1, job_id=2, lease_epoch=99,
        worker_incarnation=4, operation_id=1, attempt_id=1, chunk_size=4,
    )
    ws = _FakeWebSocket([_header_message(frames[0]), _chunk_message(frames[0])])
    conn = _conn(ws)
    received = []

    await conn.recv_messages(
        lambda *args: received.append(args),
        lambda _: None,
    )

    assert received == []
    assert len(ws.sent) == 1  # consumed stale payload credit is returned


def test_connection_rejects_deadlocking_credit_configuration():
    with pytest.raises(ValueError, match="initial credit"):
        _conn(_FakeWebSocket(), chunk_size=8, credit_bytes=4)


def test_connection_rejects_non_integer_credit_configuration():
    with pytest.raises(TypeError, match="credit_bytes"):
        _conn(_FakeWebSocket(), credit_bytes=16.0)
    with pytest.raises(TypeError, match="credit_bytes"):
        _conn(_FakeWebSocket(), credit_bytes=True)


def test_connection_rejects_invalid_pending_tensor_limit():
    with pytest.raises(ValueError, match="max_pending_tensors"):
        _conn(_FakeWebSocket(), max_pending_tensors=0)
    with pytest.raises(TypeError, match="max_pending_tensors"):
        _conn(_FakeWebSocket(), max_pending_tensors=1.5)


@pytest.mark.asyncio
async def test_receiver_bounds_unfinished_tensor_headers():
    meta_a = TensorMeta("a", DType.INT8, (1,))
    meta_b = TensorMeta("b", DType.INT8, (1,))
    frame_a = chunk_tensor(
        b"a", meta_a, cluster_id=1, job_id=2, lease_epoch=3,
        worker_incarnation=4, operation_id=10, attempt_id=1,
    )[0]
    frame_b = chunk_tensor(
        b"b", meta_b, cluster_id=1, job_id=2, lease_epoch=3,
        worker_incarnation=4, operation_id=11, attempt_id=1,
    )[0]
    receiver = _conn(
        _FakeWebSocket([_header_message(frame_a), _header_message(frame_b)]),
        max_pending_tensors=1,
    )

    with pytest.raises(ValueError, match="too many pending tensor headers"):
        await receiver.recv_messages(lambda *_args: None, lambda _msg: None)


@pytest.mark.asyncio
async def test_receiver_checks_chunk_zero_against_header_descriptor():
    meta = TensorMeta("tensor", DType.INT8, (2,))
    frame = chunk_tensor(
        b"ab", meta, cluster_id=1, job_id=2, lease_epoch=3,
        worker_incarnation=4, operation_id=12, attempt_id=1,
    )[0]
    # The header is allowed to carry only a descriptor; it must nevertheless
    # describe the same first chunk that arrives on the wire.
    inconsistent_header = TensorFrame(
        header=replace(frame.header, byte_length=1),
        meta=frame.meta,
        payload=frame.payload,
    )
    receiver = _conn(
        _FakeWebSocket([_header_message(inconsistent_header), _chunk_message(frame)])
    )

    with pytest.raises(ValueError, match="chunk zero"):
        await receiver.recv_messages(lambda *_args: None, lambda _msg: None)
