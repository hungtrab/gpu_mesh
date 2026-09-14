"""End-to-end HTTP generation through independent remote stage endpoints."""

import asyncio

import pytest
import torch

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from meshgpu.backends.portable.pipeline import build_pipeline, pipeline_prefill_async
from meshgpu.backends.portable.rpc import (
    StageRpcIdentity,
    StageRpcServer,
    connect_stage_clients,
)
from meshgpu.inference.server import build_inference_app
from meshgpu.models.llama_dense import LlamaConfig
from meshgpu.transport.relay import WebSocketRelay, make_relay_url, run_outbound_stage

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


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


@pytest.mark.asyncio
async def test_remote_stage_gateway_generates_and_releases_kv():
    cfg = _tiny_cfg()
    remote_workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    endpoints = []
    servers = []
    for stage_id, worker in enumerate(remote_workers):
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=7,
                job_id=8,
                lease_epoch=9,
                worker_incarnation=200 + stage_id,
                peer_worker_incarnation=301,
            ),
            credential="gateway-secret",
        )
        server = await endpoint.serve("127.0.0.1", 0)
        servers.append(server)
        port = server.sockets[0].getsockname()[1]
        endpoints.append(f"ws://127.0.0.1:{port}")

    clients = []
    try:
        clients = await connect_stage_clients(
            endpoints,
            ["gateway-secret", "gateway-secret"],
            [
                StageRpcIdentity(
                    cluster_id=7,
                    job_id=8,
                    lease_epoch=9,
                    worker_incarnation=301,
                    peer_worker_incarnation=200 + stage_id,
                )
                for stage_id in range(2)
            ],
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
        )
        app = build_inference_app(clients, kv_slots=64)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.post(
                "/v1/generate",
                json={
                    "prompt_ids": [1, 4, 9],
                    "max_new_tokens": 2,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert len(payload["token_ids"]) == 2
        assert all(0 <= token < cfg.vocab_size for token in payload["token_ids"])
        for client in clients:
            assert await client.kv_cache_length() == 0
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        for server in servers:
            server.close()
            await server.wait_closed()


@pytest.mark.asyncio
async def test_two_outbound_workers_pair_through_relay():
    """Two inbound-inaccessible stages can be reached through one relay."""
    cfg = _tiny_cfg()
    remote_workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    relay = WebSocketRelay(relay_token="relay-secret")
    relay_server = await relay.serve("127.0.0.1", 0)
    relay_port = relay_server.sockets[0].getsockname()[1]
    relay_url = f"ws://127.0.0.1:{relay_port}/v1/relay"

    stage_endpoints = []
    worker_tasks = []
    for stage_id, worker in enumerate(remote_workers):
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=7,
                job_id=8,
                lease_epoch=9,
                worker_incarnation=200 + stage_id,
                peer_worker_incarnation=301,
            ),
            credential="gateway-secret",
        )
        stage_endpoints.append(endpoint)
        worker_tasks.append(
            asyncio.create_task(
                run_outbound_stage(
                    endpoint,
                    relay_url,
                    "gateway-secret",
                    job_id=8,
                    stage_id=stage_id,
                    relay_token="relay-secret",
                )
            )
        )

    clients = []
    try:
        clients = await connect_stage_clients(
            [
                make_relay_url(
                    relay_url,
                    job_id=8,
                    stage_id=stage_id,
                    role="gateway",
                )
                for stage_id in range(2)
            ],
            ["gateway-secret", "gateway-secret"],
            [
                StageRpcIdentity(
                    cluster_id=7,
                    job_id=8,
                    lease_epoch=9,
                    worker_incarnation=301,
                    peer_worker_incarnation=200 + stage_id,
                )
                for stage_id in range(2)
            ],
            relay_token="relay-secret",
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
        )
        logits = await pipeline_prefill_async(
            clients,
            torch.tensor([[1, 4, 9]], dtype=torch.long),
            operation_id=1,
            attempt_id="relay-test",
        )
        assert logits.shape == (1, 3, cfg.vocab_size)
        assert relay.route_count() == 2
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        relay_server.close()
        await relay_server.wait_closed()
        await relay.close()


@pytest.mark.asyncio
async def test_relay_releases_unpaired_worker_after_disconnect():
    """A dead notebook must not block its replacement from claiming a route."""
    from websockets.asyncio.client import connect

    relay = WebSocketRelay(relay_token="relay-secret", pair_timeout_s=1.0)
    relay_server = await relay.serve("127.0.0.1", 0)
    relay_port = relay_server.sockets[0].getsockname()[1]
    worker_url = make_relay_url(
        f"ws://127.0.0.1:{relay_port}/v1/relay",
        job_id=8,
        stage_id=0,
        role="worker",
    )
    ws = await connect(
        worker_url,
        additional_headers={"X-MeshGPU-Relay-Token": "relay-secret"},
    )
    try:
        for _ in range(20):
            if relay.route_count() == 1:
                break
            await asyncio.sleep(0.005)
        assert relay.route_count() == 1
    finally:
        await ws.close()
    for _ in range(20):
        if relay.route_count() == 0:
            break
        await asyncio.sleep(0.005)
    try:
        assert relay.route_count() == 0
    finally:
        relay_server.close()
        await relay_server.wait_closed()
        await relay.close()
