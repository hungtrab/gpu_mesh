"""Runtime LoRA memory admission tests without requiring a physical GPU."""
from __future__ import annotations

import pytest
import torch

from meshgpu.backends.native.lora_recipe import LoRAConfig, iter_lora_modules
from meshgpu.backends.portable.memory_guard import check_training_memory
from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.backends.portable.rpc import StageRpcIdentity, StageRpcServer
from meshgpu.models.llama_dense import LlamaConfig


def _cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )


def _lora() -> LoRAConfig:
    return LoRAConfig(
        rank=256,
        alpha=32.0,
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


def _cuda_stage():
    return build_pipeline(_cfg(), 1, [torch.device("cpu")])[0]


def _fake_cuda_memory(monkeypatch, *, free: int, total: int) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (free, total))


def test_cpu_stage_is_feasible_and_reports_incremental_memory() -> None:
    worker = _cuda_stage()
    estimate = check_training_memory(
        worker,
        _lora(),
        batch_size=2,
        sequence_length=32,
        activation_checkpointing=True,
    )

    assert estimate.feasible
    assert estimate.device == "cpu"
    assert estimate.new_adapter_parameters > 0
    assert estimate.adapter_bytes > 0
    assert estimate.gradient_bytes > 0
    assert estimate.optimizer_bytes > 0
    assert estimate.activation_bytes > 0


def test_cuda_guard_rejects_before_any_adapter_mutation(monkeypatch) -> None:
    worker = _cuda_stage()
    worker._ctx.device = torch.device("cuda:0")
    _fake_cuda_memory(monkeypatch, free=512 * 1024**2, total=16 * 1024**3)

    estimate = check_training_memory(worker, _lora())

    assert not estimate.feasible
    assert "insufficient_memory" in estimate.reason
    assert not list(iter_lora_modules(worker._model))


def test_cuda_guard_accounts_for_shape_and_checkpointing(monkeypatch) -> None:
    worker = _cuda_stage()
    worker._ctx.device = torch.device("cuda:0")
    _fake_cuda_memory(monkeypatch, free=16 * 1024**3, total=16 * 1024**3)

    without = check_training_memory(
        worker,
        _lora(),
        batch_size=2,
        sequence_length=64,
        activation_checkpointing=False,
    )
    with_checkpointing = check_training_memory(
        worker,
        _lora(),
        batch_size=2,
        sequence_length=64,
        activation_checkpointing=True,
    )

    assert without.feasible and with_checkpointing.feasible
    assert without.activation_bytes > with_checkpointing.activation_bytes
    assert without.required_bytes > with_checkpointing.required_bytes


def test_guard_requires_batch_and_sequence_together() -> None:
    with pytest.raises(ValueError, match="provided together"):
        check_training_memory(_cuda_stage(), _lora(), batch_size=1)


def test_rpc_rejects_low_vram_before_apply_lora(monkeypatch) -> None:
    worker = _cuda_stage()
    worker._ctx.device = torch.device("cuda:0")
    _fake_cuda_memory(monkeypatch, free=512 * 1024**2, total=16 * 1024**3)
    server = StageRpcServer(
        worker,
        StageRpcIdentity(
            cluster_id=1,
            job_id=2,
            lease_epoch=1,
            worker_incarnation=3,
        ),
        credential="secret",
    )

    with pytest.raises(RuntimeError, match="insufficient_memory"):
        server._configure_training(
            {
                "lora": _lora().to_dict(),
                "learning_rate": 1e-3,
                "batch_size": 1,
                "sequence_length": 16,
            },
            "owner",
        )
    assert server._training_optimizer is None
    assert not list(iter_lora_modules(worker._model))
