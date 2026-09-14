"""Tests for LoRA recipe — adapter injection, training, checkpoint."""
import pytest
import torch

from meshgpu.backends.native.lora_recipe import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    trainable_parameters,
)
from meshgpu.backends.portable.pipeline import build_pipeline, pipeline_train_step
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage

TINY_CFG = LlamaConfig(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, max_position_embeddings=32,
)
DEVICE = torch.device("cpu")


def _make_stage(with_head: bool = True) -> LlamaStage:
    torch.manual_seed(0)
    return LlamaStage(
        TINY_CFG, 0, TINY_CFG.num_hidden_layers,
        has_embedding=True, has_lm_head=with_head, device=DEVICE,
    )


# ------------------------------------------------------------------
# Adapter injection
# ------------------------------------------------------------------

def test_apply_lora_injects_adapters():
    stage = _make_stage()
    cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
    apply_lora(stage, cfg)
    # Check all q_proj and v_proj are replaced
    for layer in stage.layers:
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)


def test_apply_lora_is_idempotent_and_reactivates_existing_adapters():
    stage = _make_stage()
    cfg = LoRAConfig(rank=4, alpha=8.0)
    apply_lora(stage, cfg)
    q_proj = stage.layers[0].self_attn.q_proj
    assert isinstance(q_proj, LoRALinear)
    with torch.no_grad():
        q_proj.lora_B.fill_(0.25)
    q_before = q_proj

    # Simulate a restore path that froze every parameter before re-applying the
    # recipe.  The second call must not stack a second adapter or reset state.
    for parameter in stage.parameters():
        parameter.requires_grad_(False)
    apply_lora(stage, cfg)

    assert stage.layers[0].self_attn.q_proj is q_before
    assert stage.layers[0].self_attn.q_proj.lora_A.requires_grad
    assert stage.layers[0].self_attn.q_proj.lora_B.requires_grad
    torch.testing.assert_close(
        stage.layers[0].self_attn.q_proj.lora_B,
        torch.full_like(stage.layers[0].self_attn.q_proj.lora_B, 0.25),
    )


def test_apply_lora_rejects_incomplete_target_set():
    stage = _make_stage()
    del stage.layers[0].self_attn.v_proj
    with pytest.raises(ValueError, match="target modules missing"):
        apply_lora(stage, LoRAConfig(rank=2, target_modules=["q_proj", "v_proj"]))


def test_apply_lora_rejects_incompatible_existing_dropout():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=2, alpha=4.0, dropout=0.25))
    with pytest.raises(ValueError, match="dropout"):
        apply_lora(stage, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))


def test_base_weights_frozen_after_lora():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))
    for name, p in stage.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad, f"{name} should require grad"
        else:
            assert not p.requires_grad, f"{name} should be frozen"


def test_trainable_params_only_lora():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))
    trainable = trainable_parameters(stage)
    assert len(trainable) > 0
    total = sum(p.numel() for p in stage.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    assert trainable_n < total * 0.1  # LoRA << full params


# ------------------------------------------------------------------
# Forward pass unchanged (B=0 initialized)
# ------------------------------------------------------------------

def test_lora_forward_runs():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 8))
    with torch.no_grad():
        logits, _ = stage(torch.empty(0), input_ids=ids)
    assert logits.shape == (1, 8, TINY_CFG.vocab_size)


def test_lora_forward_supports_half_precision_base():
    """Adapters remain usable when imported inference weights are fp16."""
    stage = _make_stage().half()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
    with torch.no_grad():
        logits, _ = stage(torch.empty(0), input_ids=ids)
    assert logits.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lora_adapters_follow_cuda_base_device():
    """Injecting LoRA after moving a stage to CUDA must keep one device."""
    device = torch.device("cuda:0")
    stage = _make_stage().to(device).half()
    apply_lora(stage, LoRAConfig(rank=2, alpha=4.0))

    for layer in stage.layers:
        for projection in (layer.self_attn.q_proj, layer.self_attn.v_proj):
            assert isinstance(projection, LoRALinear)
            assert projection.lora_A.device == device
            assert projection.lora_B.device == device

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4), device=device)
    with torch.no_grad():
        logits, _ = stage(torch.empty(0, device=device), input_ids=ids)
    assert logits.device == device
    assert logits.dtype == torch.float16


def test_lora_init_zero_delta():
    """With lora_B init=0, forward should match base model exactly."""
    torch.manual_seed(42)
    stage_base = _make_stage()
    torch.manual_seed(42)
    stage_lora = _make_stage()
    apply_lora(stage_lora, LoRAConfig(rank=4, alpha=8.0))

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
    with torch.no_grad():
        base_out, _ = stage_base(torch.empty(0), input_ids=ids)
        lora_out, _ = stage_lora(torch.empty(0), input_ids=ids)

    torch.testing.assert_close(base_out, lora_out, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------
# Training: only adapter params update
# ------------------------------------------------------------------

def test_lora_training_updates_adapters_only():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))
    opt = torch.optim.Adam(trainable_parameters(stage), lr=1e-2)

    # Snapshot frozen base weight and lora_B (gets non-zero gradient even when lora_B=0)
    base_w = next(
        p.data.clone()
        for p in stage.parameters()
        if not p.requires_grad
    )
    lora_B_before = stage.layers[0].self_attn.q_proj.lora_B.data.clone()

    # One training step via pipeline
    workers = build_pipeline(TINY_CFG, 1, [DEVICE])
    workers[0]._model = stage
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
    labels = ids.clone()

    pipeline_train_step(workers, ids, labels, [opt], operation_id=1, attempt_id="a0")

    # Base weight must not change (frozen)
    base_w_after = next(
        p.data
        for p in stage.parameters()
        if not p.requires_grad
    )
    torch.testing.assert_close(base_w, base_w_after)

    # lora_B gets gradient d(loss)/d(B) = d(loss)/d(out).T @ (x @ A.T) ≠ 0
    assert not torch.equal(lora_B_before, stage.layers[0].self_attn.q_proj.lora_B.data)


# ------------------------------------------------------------------
# Checkpoint: lora_state_dict / load round-trip
# ------------------------------------------------------------------

def test_lora_checkpoint_roundtrip(tmp_path):
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))

    # Mutate adapter
    with torch.no_grad():
        for layer in stage.layers:
            layer.self_attn.q_proj.lora_A.fill_(3.14)

    sd = lora_state_dict(stage)
    assert all("lora_A" in k or "lora_B" in k for k in sd)

    # New stage: load adapters back
    stage2 = _make_stage()
    apply_lora(stage2, LoRAConfig(rank=4, alpha=8.0))
    load_lora_state_dict(stage2, sd)

    for k, v in sd.items():
        torch.testing.assert_close(v, stage2.state_dict()[k])


# ------------------------------------------------------------------
# Merge: merged weight equals base + ΔW
# ------------------------------------------------------------------

def test_lora_merge_correctness():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=4, alpha=8.0))

    lora_linear = stage.layers[0].self_attn.q_proj
    assert isinstance(lora_linear, LoRALinear)

    with torch.no_grad():
        lora_linear.lora_A.fill_(0.1)
        lora_linear.lora_B.fill_(0.2)

    merged = lora_linear.merge()
    expected = lora_linear.base.weight.data + (
        lora_linear.lora_B @ lora_linear.lora_A
    ) * lora_linear.scale
    torch.testing.assert_close(merged.weight.data, expected, atol=1e-5, rtol=1e-5)


def test_lora_state_dict_rejects_frozen_or_unknown_keys():
    stage = _make_stage()
    apply_lora(stage, LoRAConfig(rank=2, alpha=4.0))
    state = lora_state_dict(stage)
    state["layers.0.self_attn.q_proj.base.weight"] = torch.zeros(32, 32)

    with pytest.raises(ValueError, match="unknown keys"):
        load_lora_state_dict(stage, state)
