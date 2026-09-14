"""CPU-safe correctness tests for the native FSDP recipe helpers."""

import pytest
import torch

from meshgpu.backends.native.fsdp_recipe import TrainingConfig, training_step
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage


def _stage() -> LlamaStage:
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
    )
    return LlamaStage(cfg, 0, 1, has_embedding=True, has_lm_head=True)


def test_training_step_runs_without_cuda_on_cpu():
    torch.manual_seed(3)
    model = _stage()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainingConfig(
        gradient_accumulation_steps=2,
        mixed_precision="float32",
    )
    input_ids = torch.randint(0, 32, (2, 4))
    labels = input_ids.roll(-1, dims=1)
    labels[:, -1] = -100

    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = training_step(model, optimizer, None, input_ids, labels, cfg)

    assert result["n_valid_tokens"] == 6
    assert result["loss"] > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_training_step_reports_zero_valid_tokens():
    model = _stage()
    cfg = TrainingConfig(mixed_precision="float32")
    input_ids = torch.zeros((1, 3), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)

    result = training_step(model, None, None, input_ids, labels, cfg)

    assert result["n_valid_tokens"] == 0
    assert result["loss"] == 0.0


def test_training_config_rejects_invalid_precision():
    with pytest.raises(ValueError, match="mixed_precision"):
        TrainingConfig(mixed_precision="int8")
