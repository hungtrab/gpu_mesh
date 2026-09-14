"""
P1 correctness tests for LlamaStage and pipeline.

Reference: single-stage (monolithic) vs 2-stage split.
Tolerance: atol=1e-5, rtol=1e-4 on FP32 with dropout off, fixed seed.
"""
import pytest
import torch

from meshgpu.backends.portable.pipeline import (
    build_pipeline,
    pipeline_decode_step,
    pipeline_prefill,
    pipeline_train_step,
)
from meshgpu.models.llama_dense import (
    GQAttention,
    LlamaConfig,
    LlamaStage,
    _precompute_freqs,
)

# Tiny model for fast CI
TINY_CFG = LlamaConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    rms_norm_eps=1e-5,
)
DEVICE = torch.device("cpu")
ATOL, RTOL = 1e-4, 1e-3  # CPU FP32 tolerance


def test_config_rejects_invalid_numeric_values():
    with pytest.raises(TypeError, match="num_hidden_layers"):
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2.0,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
    with pytest.raises(ValueError, match="rms_norm_eps"):
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            rms_norm_eps=float("nan"),
        )


def test_sdpa_uses_compact_kv_for_grouped_query_attention(monkeypatch):
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        attention_implementation="sdpa",
    )
    attention = GQAttention(cfg)
    cos, sin = _precompute_freqs(cfg.head_dim, cfg.max_position_embeddings)
    observed: dict[str, object] = {}
    original = torch.nn.functional.scaled_dot_product_attention

    def spy(query, key, value, **kwargs):
        observed["query_heads"] = query.shape[1]
        observed["key_heads"] = key.shape[1]
        observed["enable_gqa"] = kwargs.get("enable_gqa", False)
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        spy,
    )
    x = torch.randn(1, 4, cfg.hidden_size)
    position_ids = torch.arange(4).unsqueeze(0)
    output, _ = attention(x, cos, sin, position_ids, None, None, True)

    assert output.shape == (1, 4, cfg.hidden_size)
    assert observed == {"query_heads": 4, "key_heads": 2, "enable_gqa": True}
def _seed(n: int = 42) -> None:
    torch.manual_seed(n)


def _make_single_stage() -> LlamaStage:
    _seed()
    return LlamaStage(
        TINY_CFG, 0, TINY_CFG.num_hidden_layers,
        has_embedding=True, has_lm_head=True, device=DEVICE,
    )


def _copy_weights_to_pipeline(
    mono: LlamaStage,
    workers: list,
) -> None:
    """Copy monolithic weights into the split pipeline workers."""
    for worker in workers:
        stage: LlamaStage = worker._model
        if stage.has_embedding:
            stage.embed_tokens.load_state_dict(mono.embed_tokens.state_dict())
        if stage.has_lm_head:
            stage.norm.load_state_dict(mono.norm.state_dict())
            stage.lm_head.load_state_dict(mono.lm_head.state_dict())
        for local_i, global_i in enumerate(
            range(stage.layer_start, stage.layer_end)
        ):
            stage.layers[local_i].load_state_dict(
                mono.layers[global_i].state_dict()
            )


# ------------------------------------------------------------------
# Forward sharding
# ------------------------------------------------------------------

def test_single_stage_forward():
    mono = _make_single_stage()
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 8))
    with torch.no_grad():
        logits, _ = mono(torch.empty(0), input_ids=ids)
    assert logits.shape == (1, 8, TINY_CFG.vocab_size)


def test_embedding_stage_missing_input_ids_is_a_clear_error():
    model = _make_single_stage()
    with pytest.raises(ValueError, match="requires input_ids"):
        model(torch.empty(0))


def test_stage_rejects_bad_cache_layout_and_positions():
    model = LlamaStage(
        TINY_CFG,
        0,
        1,
        has_embedding=True,
        has_lm_head=False,
        device=DEVICE,
    )
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="KV cache entries"):
        model(
            torch.empty(0),
            input_ids=ids,
            kv_caches=[],
        )
    with pytest.raises(ValueError, match="outside"):
        model(
            torch.empty(0),
            input_ids=ids,
            position_ids=torch.tensor([[0, TINY_CFG.max_position_embeddings]]),
        )
    with pytest.raises(ValueError, match="input_ids"):
        model(torch.empty(0), input_ids=torch.tensor([[TINY_CFG.vocab_size, 1]]))


def test_stage_rejects_partial_or_mismatched_kv_cache():
    model = LlamaStage(
        TINY_CFG,
        0,
        2,
        has_embedding=True,
        has_lm_head=False,
        device=DEVICE,
    )
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="every layer or for none"):
        model(
            torch.empty(0),
            input_ids=ids,
            kv_caches=[(torch.zeros(1, 2, 1, 16), torch.zeros(1, 2, 1, 16)), None],
        )

    wrong_dtype = [
        (
            torch.zeros(1, 2, 1, 16, dtype=torch.float64),
            torch.zeros(1, 2, 1, 16, dtype=torch.float64),
        )
    ] * 2
    with pytest.raises(ValueError, match="same dtype"):
        model(torch.empty(0), input_ids=ids, kv_caches=wrong_dtype)


def test_decoder_prefill_is_causal():
    """Adding future tokens must not change logits at an earlier position."""
    model = _make_single_stage()
    ids = torch.tensor([[3, 7, 11, 13]])
    with torch.no_grad():
        full_logits, _ = model(torch.empty(0), input_ids=ids)
        first_logits, _ = model(torch.empty(0), input_ids=ids[:, :1])
    torch.testing.assert_close(
        full_logits[:, :1], first_logits, atol=ATOL, rtol=RTOL
    )


def test_2stage_forward_matches_mono():
    mono = _make_single_stage()
    _seed()
    workers = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    _copy_weights_to_pipeline(mono, workers)

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 8))

    with torch.no_grad():
        ref_logits, _ = mono(torch.empty(0), input_ids=ids)

    pipe_logits = pipeline_prefill(workers, ids, operation_id=1, attempt_id="t0")

    torch.testing.assert_close(
        pipe_logits, ref_logits.cpu(), atol=ATOL, rtol=RTOL
    )


def test_4stage_forward_matches_mono():
    mono = _make_single_stage()
    _seed()
    workers = build_pipeline(TINY_CFG, 4, [DEVICE] * 4)
    _copy_weights_to_pipeline(mono, workers)

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 6))
    with torch.no_grad():
        ref_logits, _ = mono(torch.empty(0), input_ids=ids)

    pipe_logits = pipeline_prefill(workers, ids, operation_id=1, attempt_id="t0")
    torch.testing.assert_close(
        pipe_logits, ref_logits.cpu(), atol=ATOL, rtol=RTOL
    )


# ------------------------------------------------------------------
# KV cache: decode token matches reference
# ------------------------------------------------------------------

def test_kv_cache_decode():
    _seed()
    workers = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])

    prompt_ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
    # Prefill
    logits_prefill = pipeline_prefill(workers, prompt_ids, operation_id=1, attempt_id="t0")
    next_token = logits_prefill[:, -1, :].argmax(dim=-1, keepdim=True)

    # Run monolithic reference for one decode step
    _seed()
    workers_ref = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    for w, wref in zip(workers, workers_ref):
        wref._model.load_state_dict(w._model.state_dict())

    ref_prefill = pipeline_prefill(workers_ref, prompt_ids, operation_id=1, attempt_id="t0")
    ref_next = ref_prefill[:, -1, :].argmax(dim=-1, keepdim=True)

    assert next_token.item() == ref_next.item()


def test_cached_stage_infers_rope_positions_from_prefix_length():
    _seed()
    model = _make_single_stage()
    prefix = torch.tensor([[4, 8, 12, 16]])
    next_token = torch.tensor([[20]])
    with torch.no_grad():
        _, caches = model(torch.empty(0), input_ids=prefix)
        implicit, _ = model(
            torch.empty(0),
            input_ids=next_token,
            kv_caches=caches,
        )
        explicit, _ = model(
            torch.empty(0),
            input_ids=next_token,
            position_ids=torch.tensor([[prefix.shape[1]]]),
            kv_caches=caches,
        )
    torch.testing.assert_close(implicit, explicit, atol=ATOL, rtol=RTOL)


def test_multi_token_decode_matches_single_token_decode():
    """A packed decode query must use consecutive positions, not one position."""
    _seed()
    packed = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    sequential = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    for packed_worker, sequential_worker in zip(packed, sequential):
        sequential_worker._model.load_state_dict(packed_worker._model.state_dict())

    prompt = torch.tensor([[4, 8, 12]])
    pipeline_prefill(packed, prompt, operation_id=1, attempt_id="packed")
    pipeline_prefill(sequential, prompt, operation_id=1, attempt_id="sequential")
    future = torch.tensor([[15, 16, 17]])
    packed_logits = pipeline_decode_step(
        packed, future, operation_id=2, attempt_id="packed"
    )
    sequential_logits = []
    for index, token in enumerate(future[0].tolist(), start=2):
        sequential_logits.append(
            pipeline_decode_step(
                sequential,
                torch.tensor([[token]]),
                operation_id=index,
                attempt_id="sequential",
            )
        )
    sequential_logits = torch.cat(sequential_logits, dim=1)
    torch.testing.assert_close(packed_logits, sequential_logits, atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------
# Training: loss goes down over several steps
# ------------------------------------------------------------------

def test_training_loss_decreases():
    _seed()
    workers = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    optimizers = [
        torch.optim.Adam(w._model.parameters(), lr=1e-2)
        for w in workers
    ]

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 8))
    labels = ids.clone()

    losses = []
    for step in range(5):
        info = pipeline_train_step(
            workers, ids, labels, optimizers,
            operation_id=step + 1, attempt_id=f"a{step}",
        )
        losses.append(info["loss"])

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ------------------------------------------------------------------
# Training: valid token normalization
# ------------------------------------------------------------------

def test_training_ignores_padding():
    _seed()
    workers = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    optimizers = [torch.optim.SGD(w._model.parameters(), lr=0) for w in workers]

    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 8))
    labels = ids.clone()
    labels[0, :4] = -100  # mask first 4 tokens

    info = pipeline_train_step(
        workers, ids, labels, optimizers,
        operation_id=1, attempt_id="a0",
    )
    assert info["n_valid_tokens"] == 4


def test_pipeline_gradient_scaler_unscales_before_optimizer_step():
    _seed()
    workers = build_pipeline(TINY_CFG, 1, [DEVICE])
    optimizer = torch.optim.SGD(workers[0]._model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu")
    before = [parameter.detach().clone() for parameter in workers[0]._model.parameters()]

    pipeline_train_step(
        workers,
        torch.randint(0, TINY_CFG.vocab_size, (1, 4)),
        torch.randint(0, TINY_CFG.vocab_size, (1, 4)),
        [optimizer],
        operation_id=901,
        attempt_id="scaled",
        gradient_scaler=scaler,
    )

    assert all(parameter.grad is None for parameter in workers[0]._model.parameters())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, workers[0]._model.parameters())
    )


def test_pipeline_gradient_scaler_skips_all_stages_on_one_stage_overflow():
    """A pipeline update is atomic when one stage produces an inf gradient."""
    _seed()
    workers = build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])
    optimizers = [
        torch.optim.SGD(worker._model.parameters(), lr=1e-2)
        for worker in workers
    ]
    scaler = torch.amp.GradScaler("cpu")
    before = [
        [parameter.detach().clone() for parameter in worker._model.parameters()]
        for worker in workers
    ]

    overflowing_parameter = next(workers[0]._model.parameters())
    hook = overflowing_parameter.register_hook(
        lambda grad: torch.full_like(grad, float("inf"))
    )
    try:
        pipeline_train_step(
            workers,
            torch.randint(0, TINY_CFG.vocab_size, (1, 4)),
            torch.randint(0, TINY_CFG.vocab_size, (1, 4)),
            optimizers,
            operation_id=902,
            attempt_id="one-stage-overflow",
            gradient_scaler=scaler,
        )
    finally:
        hook.remove()

    for previous_stage, worker in zip(before, workers):
        for previous, current in zip(previous_stage, worker._model.parameters()):
            torch.testing.assert_close(previous, current)
        assert all(parameter.grad is None for parameter in worker._model.parameters())
