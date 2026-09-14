"""NVARC-compatible LoRA coverage for attention, MLP, and endpoints."""

import math

import torch
import torch.nn as nn

from meshgpu.backends.native.lora_recipe import (
    LoRAConfig,
    LoRAEmbedding,
    LoRALinear,
    apply_lora,
    lora_config,
    lora_state_dict,
    trainable_parameters,
)
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage


def _cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )


def test_nvarc_recipe_covers_attention_mlp_and_full_endpoint_modules() -> None:
    stage = LlamaStage(_cfg(), 0, 2, has_embedding=True, has_lm_head=True)
    cfg = LoRAConfig(
        rank=4,
        alpha=8.0,
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
    apply_lora(stage, cfg)

    for layer in stage.layers:
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            assert isinstance(projection, LoRALinear)
            assert math.isclose(projection.scale, 4.0)
            assert projection.use_rslora
    assert stage.embed_tokens.weight.requires_grad
    assert stage.lm_head.weight.requires_grad
    assert not stage.layers[0].self_attn.q_proj.base.weight.requires_grad
    assert len(trainable_parameters(stage)) > 0

    saved = lora_state_dict(stage)
    assert "embed_tokens.weight" in saved
    assert "lm_head.weight" in saved
    assert lora_config(stage) is not None
    assert lora_config(stage).to_dict() == cfg.to_dict()


def test_endpoint_modules_to_save_are_optional_on_other_stages() -> None:
    stage = LlamaStage(_cfg(), 1, 2, has_embedding=False, has_lm_head=False)
    apply_lora(
        stage,
        LoRAConfig(
            rank=2,
            alpha=4.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
        ),
    )
    assert not hasattr(stage, "embed_tokens")
    assert not hasattr(stage, "lm_head")
    assert all(
        isinstance(layer.self_attn.q_proj, LoRALinear) for layer in stage.layers
    )


def test_embedding_target_has_zero_delta_and_can_merge() -> None:
    embedding = nn.Embedding(12, 6)
    adapter = LoRAEmbedding(embedding, rank=3, alpha=6.0, dropout=0.0, use_rslora=True)
    ids = torch.tensor([[1, 4, 1]])
    with torch.no_grad():
        before = adapter(ids).clone()
        adapter.lora_B.fill_(0.2)
        expected = embedding(ids) + (
            adapter.lora_A[ids] @ adapter.lora_B
        ) * adapter.scale
        after = adapter(ids)
    torch.testing.assert_close(after, expected)
    merged = adapter.merge()
    torch.testing.assert_close(merged(ids), after)
    assert not torch.equal(before, after)
