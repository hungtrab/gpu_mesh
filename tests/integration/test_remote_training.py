"""Correctness tests for backward and optimizer RPC over WebSockets."""

import asyncio

import pytest
import torch

from meshgpu.backends.native.lora_recipe import LoRAConfig, lora_state_dict
from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.backends.portable.rpc import (
    StageRpcClient,
    StageRpcIdentity,
    StageRpcServer,
)
from meshgpu.models.llama_dense import LlamaConfig
from meshgpu.training.remote_ttt import RemoteTaskTTTSession
from meshgpu.training.ttt import TaskTTTConfig
from meshgpu.transport.relay import WebSocketRelay, make_relay_url, run_outbound_stage


def _tiny_cfg(*, tied_embeddings: bool = False) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=tied_embeddings,
    )


async def _start_pair(workers):
    servers = []
    clients = []
    for stage_id, worker in enumerate(workers):
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=41,
                job_id=42,
                lease_epoch=1,
                worker_incarnation=100 + stage_id,
                peer_worker_incarnation=200 + stage_id,
            ),
            credential="training-secret",
        )
        server = await endpoint.serve("127.0.0.1", 0)
        servers.append(server)
        port = server.sockets[0].getsockname()[1]
        clients.append(
            await StageRpcClient.connect(
                f"ws://127.0.0.1:{port}",
                "training-secret",
                StageRpcIdentity(
                    cluster_id=41,
                    job_id=42,
                    lease_epoch=1,
                    worker_incarnation=200 + stage_id,
                    peer_worker_incarnation=100 + stage_id,
                ),
                request_timeout_s=30.0,
            )
        )
    return clients, servers


async def _stop_pair(clients, servers) -> None:
    await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
    for server in servers:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_remote_ttt_backward_updates_and_reset_restores_all_stages() -> None:
    torch.manual_seed(19)
    workers = build_pipeline(_tiny_cfg(), 2, [torch.device("cpu")] * 2)
    clients, servers = await _start_pair(workers)
    try:
        lora = LoRAConfig(
            rank=2,
            alpha=4.0,
            use_rslora=True,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        session = RemoteTaskTTTSession(
            clients,
            TaskTTTConfig(
                lora=lora,
                learning_rate=2e-2,
                max_grad_norm=0.25,
                max_steps=1,
            ),
        )
        summaries = await session.configure(batch_size=1, sequence_length=4)
        assert len(summaries) == 2
        assert all(int(summary["trainable_parameters"]) > 0 for summary in summaries)
        assert all(summary["memory"]["batch_size"] == 1 for summary in summaries)
        baseline = [
            {name: value.detach().clone() for name, value in lora_state_dict(worker._model).items()}
            for worker in workers
        ]

        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        results = await session.adapt(ids, ids, steps=1, operation_id_start=17)
        assert len(results) == 1
        assert results[0].n_valid_tokens == ids.numel()
        assert math_is_finite_positive(results[0].loss)
        assert math_is_finite_positive(results[0].grad_norm)
        assert any(
            not torch.equal(value, baseline[stage_id][name])
            for stage_id, worker in enumerate(workers)
            for name, value in lora_state_dict(worker._model).items()
        )

        await session.reset_task()
        for stage_id, worker in enumerate(workers):
            for name, value in lora_state_dict(worker._model).items():
                torch.testing.assert_close(value, baseline[stage_id][name])
            assert not worker._saved
    finally:
        await _stop_pair(clients, servers)


@pytest.mark.asyncio
async def test_remote_ttt_runs_through_outbound_relay() -> None:
    """The actual two-Kaggle topology carries backward as well as forward."""
    torch.manual_seed(23)
    workers = build_pipeline(_tiny_cfg(), 2, [torch.device("cpu")] * 2)
    relay = WebSocketRelay(relay_token="relay-secret")
    relay_server = await relay.serve("127.0.0.1", 0)
    relay_port = relay_server.sockets[0].getsockname()[1]
    relay_base = f"ws://127.0.0.1:{relay_port}/v1/relay"
    worker_tasks = []
    for stage_id, worker in enumerate(workers):
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=51,
                job_id=52,
                lease_epoch=1,
                worker_incarnation=500 + stage_id,
                peer_worker_incarnation=600,
            ),
            credential=f"stage-{stage_id}-secret",
        )
        worker_tasks.append(
            asyncio.create_task(
                run_outbound_stage(
                    endpoint,
                    relay_base,
                    f"stage-{stage_id}-secret",
                    job_id=52,
                    stage_id=stage_id,
                    relay_token="relay-secret",
                )
            )
        )

    clients = []
    try:
        clients = await asyncio.gather(
            *(
                StageRpcClient.connect(
                    make_relay_url(
                        relay_base,
                        job_id=52,
                        stage_id=stage_id,
                        role="gateway",
                    ),
                    f"stage-{stage_id}-secret",
                    StageRpcIdentity(
                        cluster_id=51,
                        job_id=52,
                        lease_epoch=1,
                        worker_incarnation=600,
                        peer_worker_incarnation=500 + stage_id,
                    ),
                    relay_token="relay-secret",
                    request_timeout_s=30.0,
                )
                for stage_id in range(2)
            )
        )
        session = RemoteTaskTTTSession(
            clients,
            TaskTTTConfig(
                lora=LoRAConfig(
                    rank=2,
                    alpha=4.0,
                    use_rslora=True,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    modules_to_save=["embed_tokens", "lm_head"],
                ),
                learning_rate=1e-2,
                max_steps=1,
            ),
        )
        results = await session.adapt(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
            steps=1,
        )
        assert len(results) == 1
        assert results[0].n_valid_tokens == 4
        assert math_is_finite_positive(results[0].loss)
        assert math_is_finite_positive(results[0].grad_norm)
        assert relay.route_count() == 2
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        relay_server.close()
        await relay_server.wait_closed()
        await relay.close()


@pytest.mark.asyncio
async def test_remote_ttt_rejects_tied_embeddings_before_training() -> None:
    """Do not silently update two endpoint copies for a tied model."""
    workers = build_pipeline(
        _tiny_cfg(tied_embeddings=True),
        2,
        [torch.device("cpu")] * 2,
    )
    clients, servers = await _start_pair(workers)
    try:
        session = RemoteTaskTTTSession(
            clients,
            TaskTTTConfig(
                lora=LoRAConfig(rank=2, alpha=4.0),
                max_steps=1,
            ),
        )
        with pytest.raises(RuntimeError, match="tied word embeddings"):
            await session.configure(batch_size=1, sequence_length=4)
        assert not any(worker._saved for worker in workers)
    finally:
        await _stop_pair(clients, servers)


def math_is_finite_positive(value: float) -> bool:
    return bool(torch.isfinite(torch.tensor(value)) and value > 0)
