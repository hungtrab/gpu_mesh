"""WebSocket stage-RPC tests using real local endpoints."""

import asyncio
import threading

import pytest
import torch

from meshgpu.backends.portable.pipeline import (
    build_pipeline,
    pipeline_decode_step,
    pipeline_decode_step_async,
    pipeline_prefill,
    pipeline_prefill_async,
)
from meshgpu.backends.portable.rpc import (
    StageRpcClient,
    StageRpcIdentity,
    StageRpcServer,
    connect_stage_clients,
)
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.session import InferenceSession
from meshgpu.models.llama_dense import LlamaConfig


def test_stage_rpc_identity_rejects_values_that_do_not_fit_wire_header():
    with pytest.raises(ValueError, match="cluster_id"):
        StageRpcIdentity(
            cluster_id=1 << 16,
            job_id=1,
            lease_epoch=1,
            worker_incarnation=1,
        )


@pytest.mark.asyncio
async def test_connect_stage_clients_closes_partial_connections_on_failure(monkeypatch):
    class FakeClient:
        def __init__(self, url: str):
            self.url = url
            self.closed = False

        async def close(self):
            self.closed = True

    connected: list[FakeClient] = []

    async def fake_connect(cls, url, _credential, _identity, **_kwargs):
        if url == "bad":
            raise OSError("unreachable")
        client = FakeClient(url)
        connected.append(client)
        return client

    monkeypatch.setattr(StageRpcClient, "connect", classmethod(fake_connect))
    identities = [
        StageRpcIdentity(1, 2, 3, 10, 20),
        StageRpcIdentity(1, 2, 3, 10, 21),
    ]

    with pytest.raises(ConnectionError, match="could not connect all"):
        await connect_stage_clients(
            ["good", "bad"],
            ["secret", "secret"],
            identities,
        )

    assert [client.url for client in connected] == ["good"]
    assert connected[0].closed


@pytest.mark.asyncio
async def test_stage_rpc_client_rejects_non_websocket_url_before_connecting():
    with pytest.raises(ValueError, match="ws:// or wss://"):
        await StageRpcClient.connect(
            "http://stage.example",
            "rpc-secret",
            StageRpcIdentity(1, 2, 3, 4),
        )
    with pytest.raises(ValueError, match="worker_incarnation"):
        StageRpcIdentity(
            cluster_id=1,
            job_id=1,
            lease_epoch=1,
            worker_incarnation=1 << 32,
        )


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )


async def _connect_pair(worker, *, server_incarnation: int, client_incarnation: int):
    server_endpoint = StageRpcServer(
        worker,
        StageRpcIdentity(
            cluster_id=11,
            job_id=22,
            lease_epoch=3,
            worker_incarnation=server_incarnation,
            peer_worker_incarnation=client_incarnation,
        ),
        credential="rpc-secret",
    )
    websocket_server = await server_endpoint.serve("127.0.0.1", 0)
    port = websocket_server.sockets[0].getsockname()[1]
    client = await StageRpcClient.connect(
        f"ws://127.0.0.1:{port}",
        "rpc-secret",
        StageRpcIdentity(
            cluster_id=11,
            job_id=22,
            lease_epoch=3,
            worker_incarnation=client_incarnation,
            peer_worker_incarnation=server_incarnation,
        ),
        request_timeout_s=10.0,
    )
    return client, websocket_server


async def _close_pair(client, websocket_server):
    await client.close()
    websocket_server.close()
    await websocket_server.wait_closed()


@pytest.mark.asyncio
async def test_stage_rpc_forward_decode_and_cache_controls():
    cfg = _tiny_cfg()
    server_worker = build_pipeline(cfg, 1, [torch.device("cpu")])[0]
    reference_worker = build_pipeline(cfg, 1, [torch.device("cpu")])[0]
    reference_worker._model.load_state_dict(server_worker._model.state_dict())
    client, websocket_server = await _connect_pair(
        server_worker,
        server_incarnation=202,
        client_incarnation=101,
    )
    try:
        prompt = torch.tensor([[3, 7, 11, 13]], dtype=torch.long)
        remote = await client.forward_inference(
            None,
            input_ids=prompt,
            operation_id=1,
            attempt_id="prefill",
            cache_key="request-a",
        )
        assert not client._pending
        expected = pipeline_prefill([reference_worker], prompt)
        torch.testing.assert_close(remote.output, expected)
        assert await client.kv_cache_length("request-a") == prompt.shape[1]

        next_token = remote.output[:, -1, :].argmax(dim=-1, keepdim=True)
        position_ids = torch.full_like(next_token, prompt.shape[1])
        remote_decode = await client.forward_inference(
            None,
            input_ids=next_token,
            position_ids=position_ids,
            operation_id=2,
            attempt_id="decode",
            cache_key="request-a",
        )
        assert not client._pending
        expected_decode = pipeline_decode_step(
            [reference_worker],
            next_token,
            operation_id=2,
            cache_key=None,
        )
        torch.testing.assert_close(remote_decode.output, expected_decode)

        await client.trim_kv(3, "request-a")
        assert await client.kv_cache_length("request-a") == 3
        await client.clear_kv("request-a")
        assert await client.kv_cache_length("request-a") == 0
    finally:
        await _close_pair(client, websocket_server)


@pytest.mark.asyncio
async def test_clear_all_kv_is_scoped_to_the_authenticated_connection():
    """One RPC client must not be able to erase another client's KV cache."""
    cfg = _tiny_cfg()
    server_worker = build_pipeline(cfg, 1, [torch.device("cpu")])[0]
    client_a, websocket_server = await _connect_pair(
        server_worker,
        server_incarnation=212,
        client_incarnation=111,
    )
    client_b = None
    try:
        port = websocket_server.sockets[0].getsockname()[1]
        client_b = await StageRpcClient.connect(
            f"ws://127.0.0.1:{port}",
            "rpc-secret",
            StageRpcIdentity(
                cluster_id=11,
                job_id=22,
                lease_epoch=3,
                worker_incarnation=111,
                peer_worker_incarnation=212,
            ),
            request_timeout_s=10.0,
        )
        prompt = torch.tensor([[3, 7, 11]], dtype=torch.long)
        await client_a.forward_inference(
            None,
            input_ids=prompt,
            operation_id=20,
            attempt_id="client-a",
            cache_key="same-client-label",
        )
        await client_b.forward_inference(
            None,
            input_ids=prompt,
            operation_id=21,
            attempt_id="client-b",
            cache_key="same-client-label",
        )
        assert await client_a.kv_cache_length("same-client-label") == 3
        assert await client_b.kv_cache_length("same-client-label") == 3

        await client_b.clear_all_kv()

        assert await client_b.kv_cache_length("same-client-label") == 0
        assert await client_a.kv_cache_length("same-client-label") == 3
        await client_a.clear_all_kv()
    finally:
        if client_b is not None:
            await client_b.close()
        await _close_pair(client_a, websocket_server)


@pytest.mark.asyncio
async def test_async_pipeline_matches_local_two_stage_pipeline():
    cfg = _tiny_cfg()
    remote_workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    local_workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    for remote, local in zip(remote_workers, local_workers):
        local._model.load_state_dict(remote._model.state_dict())

    pairs = []
    clients = []
    try:
        for stage_id, worker in enumerate(remote_workers):
            client, websocket_server = await _connect_pair(
                worker,
                server_incarnation=200 + stage_id,
                client_incarnation=100 + stage_id,
            )
            pairs.append((client, websocket_server))
            clients.append(client)

        prompt = torch.tensor([[2, 5, 8, 13]], dtype=torch.long)
        remote_logits = await pipeline_prefill_async(
            clients,
            prompt,
            operation_id=10,
            cache_key="request-b",
        )
        local_logits = pipeline_prefill(local_workers, prompt, operation_id=10)
        torch.testing.assert_close(remote_logits, local_logits)

        next_token = remote_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        remote_decode = await pipeline_decode_step_async(
            clients,
            next_token,
            operation_id=11,
            cache_key="request-b",
        )
        local_decode = pipeline_decode_step(
            local_workers,
            next_token,
            operation_id=11,
        )
        torch.testing.assert_close(remote_decode, local_decode)
    finally:
        for client, websocket_server in pairs:
            await _close_pair(client, websocket_server)


@pytest.mark.asyncio
async def test_inference_session_can_drive_remote_stage_and_cleans_cache():
    cfg = _tiny_cfg()
    server_worker = build_pipeline(cfg, 1, [torch.device("cpu")])[0]
    client, websocket_server = await _connect_pair(
        server_worker,
        server_incarnation=302,
        client_incarnation=301,
    )
    try:
        session = InferenceSession(
            "remote-session",
            [1, 4, 9],
            [client],
            SamplingParams(temperature=0.0),
            max_new_tokens=3,
        )
        results = [result async for result in session.run()]
        assert 1 <= len(results) <= 3
        assert all(0 <= result.token_id < cfg.vocab_size for result in results)
        assert results[-1].is_last
        assert await client.kv_cache_length() == 0
    finally:
        await _close_pair(client, websocket_server)


@pytest.mark.asyncio
async def test_disconnect_drains_inflight_worker_before_cache_cleanup():
    """A canceled RPC must not recreate KV after connection cleanup."""
    cfg = _tiny_cfg()
    worker = build_pipeline(cfg, 1, [torch.device("cpu")])[0]
    original_forward = worker.forward_inference
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_forward(*args, **kwargs):
        started.set()
        release.wait(5)
        try:
            return original_forward(*args, **kwargs)
        finally:
            finished.set()

    worker.forward_inference = slow_forward
    client, websocket_server = await _connect_pair(
        worker,
        server_incarnation=402,
        client_incarnation=401,
    )
    request = asyncio.create_task(
        client.forward_inference(
            None,
            input_ids=torch.tensor([[3, 7, 11]]),
            operation_id=40,
            attempt_id="disconnect",
            cache_key="disconnecting",
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await client.close()

        # Let the executor call finish only after the server has observed the
        # disconnect.  Without draining, cleanup runs first and this late
        # forward recreates a cache that survives the connection.
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await asyncio.gather(request, return_exceptions=True)
        await asyncio.sleep(0.05)
        assert not worker._kv_caches_by_key
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
        await _close_pair(client, websocket_server)
