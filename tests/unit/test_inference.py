"""Tests for inference admission, sampling and session."""
import asyncio
import threading

import pytest
import torch

from meshgpu.backends.portable.pipeline import (
    _async_stage_call,
    build_pipeline,
    pipeline_decode_step,
    pipeline_prefill,
)
from meshgpu.inference.admission import (
    AdmissionConfig,
    AdmissionController,
    KVBudget,
    MemoryPreflightResult,
)
from meshgpu.inference.sampling import SamplingParams, greedy_sample, top_p_sample
from meshgpu.inference.server import _model_spec_from_workers, _stream_sse
from meshgpu.inference.session import InferenceSession
from meshgpu.models.llama_dense import LlamaConfig
from meshgpu.planner.placement import ModelSpec

TINY_CFG = LlamaConfig(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=64,
)
DEVICE = torch.device("cpu")


def _make_workers():
    torch.manual_seed(0)
    return build_pipeline(TINY_CFG, 2, [DEVICE, DEVICE])


@pytest.mark.asyncio
async def test_async_stage_call_drains_sync_worker_before_cancel() -> None:
    """Cancellation must not leave a local stage mutating state in a thread."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class SlowWorker:
        def forward(self) -> str:
            started.set()
            release.wait(timeout=2.0)
            finished.set()
            return "done"

    task = asyncio.create_task(_async_stage_call(SlowWorker(), "forward"))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    asyncio.get_running_loop().call_later(0.01, release.set)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()


# ------------------------------------------------------------------
# Admission
# ------------------------------------------------------------------

def _make_admission(max_concurrent=2, kv_slots=1000):
    cfg = AdmissionConfig(
        max_prompt_tokens=512, max_new_tokens=128,
        max_concurrent_requests=max_concurrent,
        kv_slots=kv_slots,
    )
    kv = KVBudget(total_slots=kv_slots)
    return AdmissionController(cfg, kv), kv


def test_admit_basic():
    ctrl, _ = _make_admission()
    ok, reason = ctrl.admit(10, 50)
    assert ok
    assert ctrl.active_requests == 1


def test_admit_exceeds_concurrent():
    ctrl, _ = _make_admission(max_concurrent=1)
    ctrl.admit(10, 50)
    ok, reason = ctrl.admit(10, 50)
    assert not ok
    assert "concurrent" in reason


def test_admit_exceeds_prompt_limit():
    ctrl, _ = _make_admission()
    ok, reason = ctrl.admit(9999, 50)
    assert not ok
    assert "prompt_len" in reason


def test_admit_kv_full():
    ctrl, _ = _make_admission(kv_slots=10)
    ok, _ = ctrl.admit(8, 5)  # reserves 13 > 10
    assert not ok


def test_release_frees_slot():
    ctrl, _ = _make_admission(max_concurrent=1)
    ctrl.admit(10, 50)
    ctrl.release(10, 50)
    ok, _ = ctrl.admit(10, 50)
    assert ok


def test_kv_budget_derive():
    cfg = AdmissionConfig()
    # 4 GiB usable, 4 layers, 2 kv_heads, dim=128, fp16
    usable = 4 * 1024**3
    cfg.derive_kv_slots(usable, 4, 2, 128, 2)
    assert cfg.kv_slots > 0


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------

def test_greedy_returns_argmax():
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    assert int(greedy_sample(logits)) == 1


def test_top_p_with_zero_temp_is_greedy():
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    assert int(top_p_sample(logits, 0.0, 1.0)) == 1


def test_top_p_samples_within_vocab():
    torch.manual_seed(42)
    logits = torch.randn(1, 50)
    token = int(top_p_sample(logits, 1.0, 0.9))
    assert 0 <= token < 50


def test_seeded_top_p_sampling_is_reproducible():
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=123)
    logits = torch.randn(1, 50)
    from meshgpu.inference.sampling import make_sampling_generator

    first = make_sampling_generator(params)
    second = make_sampling_generator(params)
    ids_a = [
        int(top_p_sample(logits, params.temperature, params.top_p, generator=first))
        for _ in range(5)
    ]
    ids_b = [
        int(top_p_sample(logits, params.temperature, params.top_p, generator=second))
        for _ in range(5)
    ]
    assert ids_a == ids_b


def test_sampling_params_reject_non_finite_values():
    with pytest.raises(ValueError):
        SamplingParams(temperature=float("nan"))
    with pytest.raises(ValueError):
        SamplingParams(top_p=float("inf"))


def test_zero_kv_budget_rejects_requests():
    from meshgpu.inference.admission import AdmissionController

    controller = AdmissionController(AdmissionConfig(), KVBudget(total_slots=0))
    ok, reason = controller.admit(1, 1)
    assert not ok
    assert "KV cache" in reason


def test_memory_preflight_result_is_explicit():
    rejected = MemoryPreflightResult(
        feasible=False,
        reason="stage 0 peak exceeds usable VRAM",
        details={"gpu": 0, "peak_bytes": 12},
    )
    assert not rejected.feasible
    assert rejected.details["gpu"] == 0


def test_loaded_cuda_preflight_metadata_maps_to_model_spec():
    """The automatic guard must be able to derive metadata from live stages."""
    workers = _make_workers()
    model = _model_spec_from_workers(workers, torch, ModelSpec)
    assert isinstance(model, ModelSpec)
    assert model.num_layers == TINY_CFG.num_hidden_layers
    assert model.num_kv_heads == TINY_CFG.num_key_value_heads


def test_prefill_failure_clears_partial_kv_cache():
    workers = _make_workers()
    original = workers[1].forward_inference

    def fail(*args, **kwargs):
        raise RuntimeError("injected downstream failure")

    workers[1].forward_inference = fail
    with pytest.raises(RuntimeError, match="injected downstream failure"):
        pipeline_prefill(
            workers,
            torch.tensor([[1, 2, 3]]),
            cache_key="rollback-prefill",
        )
    assert all(worker.kv_cache_length("rollback-prefill") == 0 for worker in workers)
    workers[1].forward_inference = original


def test_decode_failure_restores_pre_operation_kv_lengths():
    workers = _make_workers()
    prompt = torch.tensor([[1, 2, 3]])
    pipeline_prefill(workers, prompt, cache_key="rollback-decode")
    before = [worker.kv_cache_length("rollback-decode") for worker in workers]
    original = workers[1].forward_inference

    def fail(*args, **kwargs):
        raise RuntimeError("injected decode failure")

    workers[1].forward_inference = fail
    with pytest.raises(RuntimeError, match="injected decode failure"):
        pipeline_decode_step(
            workers,
            torch.tensor([[4]]),
            operation_id=2,
            cache_key="rollback-decode",
        )
    assert [worker.kv_cache_length("rollback-decode") for worker in workers] == before
    workers[1].forward_inference = original


# ------------------------------------------------------------------
# Session: greedy decode produces tokens
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_generates_tokens():
    workers = _make_workers()
    prompt = list(range(4))
    session = InferenceSession(
        session_id="s0",
        prompt_ids=prompt,
        workers=workers,
        sampling=SamplingParams(temperature=0.0),
        max_new_tokens=5,
    )

    results = []
    async for r in session.run():
        results.append(r)

    assert len(results) == 5
    # Sequence numbers should be monotone 1..5
    assert [r.seq_num for r in results] == list(range(1, 6))
    # Last token marked
    assert results[-1].is_last


@pytest.mark.asyncio
async def test_session_greedy_deterministic():
    prompt = [1, 2, 3]
    workers1 = _make_workers()
    workers2 = _make_workers()

    async def collect(workers):
        s = InferenceSession("x", prompt, workers, SamplingParams(temperature=0.0), 4)
        return [r.token_id async for r in s.run()]

    ids1 = await collect(workers1)
    ids2 = await collect(workers2)
    assert ids1 == ids2


@pytest.mark.asyncio
async def test_session_kv_cache_consistency():
    """Prefill + 3 decode tokens should not raise and token ids should be valid."""
    workers = _make_workers()
    session = InferenceSession(
        "s1", [5, 6, 7, 8], workers,
        SamplingParams(temperature=0.0), 3,
    )
    ids = [r.token_id async for r in session.run()]
    assert all(0 <= t < TINY_CFG.vocab_size for t in ids)


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_independent_kv_caches():
    """Requests sharing stage workers must not overwrite one another's context."""
    torch.manual_seed(123)
    shared = _make_workers()
    reference_a = _make_workers()
    reference_b = _make_workers()
    for source, target in zip(shared, reference_a):
        target._model.load_state_dict(source._model.state_dict())
    for source, target in zip(shared, reference_b):
        target._model.load_state_dict(source._model.state_dict())

    prompt_a = [2, 4, 6]
    prompt_b = [11, 13, 17, 19]
    sampling = SamplingParams(temperature=0.0)

    async def generate(workers, session_id, prompt):
        session = InferenceSession(
            session_id=session_id,
            prompt_ids=prompt,
            workers=workers,
            sampling=sampling,
            max_new_tokens=4,
        )
        return [result.token_id async for result in session.run()]

    expected_a, expected_b = await asyncio.gather(
        generate(reference_a, "reference-a", prompt_a),
        generate(reference_b, "reference-b", prompt_b),
    )
    actual_a, actual_b = await asyncio.gather(
        generate(shared, "shared-a", prompt_a),
        generate(shared, "shared-b", prompt_b),
    )

    assert actual_a == expected_a
    assert actual_b == expected_b
    assert all(not worker._kv_caches_by_key for worker in shared)


@pytest.mark.asyncio
async def test_stream_sse_closes_session_generator_after_terminal_token():
    workers = _make_workers()
    controller, _ = _make_admission(max_concurrent=1, kv_slots=100)
    assert controller.admit(2, 1)[0]
    session = InferenceSession(
        "stream-cleanup",
        [1, 2],
        workers,
        SamplingParams(temperature=0.0),
        1,
    )

    events = [
        event
        async for event in _stream_sse(session, controller, 2, 1)
    ]

    assert any('"event": "done"' in event for event in events)
    assert all(not worker._kv_caches_by_key for worker in workers)
    assert controller.active_requests == 0
