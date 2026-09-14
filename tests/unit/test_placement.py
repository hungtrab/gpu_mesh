"""Tests for placement planner."""

import pytest

from meshgpu.planner.memory import GiB
from meshgpu.planner.placement import (
    InferenceWorkload,
    ModelSpec,
    TrainingWorkload,
    WorkerSpec,
    _inference_peak,
    _proportional_split,
    _trainable_layer_params,
    _training_peak,
    plan_inference,
    plan_training,
)

SMALL_MODEL = ModelSpec(
    num_layers=4,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=256,
    param_count=2_000_000,
    dtype_bytes=2,
)
WORKLOAD = InferenceWorkload(batch_size=1, max_prompt_tokens=512, max_new_tokens=128)


def _make_worker(i: int, vram_gib: float = 16.0) -> WorkerSpec:
    b = int(vram_gib * GiB)
    return WorkerSpec(
        worker_id=f"w{i}",
        device_index=0,
        total_vram_bytes=b,
        free_vram_bytes=b,
    )


def test_plan_2_workers_feasible():
    workers = [_make_worker(0), _make_worker(1)]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    assert report.feasible
    assert len(report.assignments) == 2
    assert report.assignments[0].layer_start == 0
    assert report.assignments[-1].layer_end == SMALL_MODEL.num_layers


def test_plan_layers_cover_all():
    workers = [_make_worker(i) for i in range(3)]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    assert report.feasible
    total = sum(a.layer_end - a.layer_start for a in report.assignments)
    assert total == SMALL_MODEL.num_layers


def test_plan_first_has_embedding_last_has_head():
    workers = [_make_worker(0), _make_worker(1)]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    assert report.assignments[0].has_embedding
    assert not report.assignments[0].has_lm_head
    assert not report.assignments[-1].has_embedding
    assert report.assignments[-1].has_lm_head


def test_plan_infeasible_when_vram_too_small():
    # 10 MiB GPUs — way too small
    workers = [WorkerSpec(
        worker_id=f"w{i}", device_index=0,
        total_vram_bytes=10 * 1024 * 1024,
        free_vram_bytes=10 * 1024 * 1024,
    ) for i in range(2)]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    assert not report.feasible
    assert "exceeds" in report.reason


def test_plan_bottleneck_identified():
    # One large, one small GPU
    workers = [
        WorkerSpec("big", 0, int(32 * GiB), int(32 * GiB)),
        WorkerSpec("small", 0, int(4 * GiB), int(4 * GiB)),
    ]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    if report.feasible:
        assert report.bottleneck_stage is not None


def test_plan_ttft_estimated_with_goodput():
    workers = [
        WorkerSpec("w0", 0, int(16 * GiB), int(16 * GiB), goodput_mbit_s=1000.0),
        WorkerSpec("w1", 0, int(16 * GiB), int(16 * GiB), goodput_mbit_s=1000.0),
    ]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    if report.feasible:
        assert report.estimated_ttft_s is not None
        assert report.estimated_ttft_s >= 0


def test_plan_summary_string():
    workers = [_make_worker(0), _make_worker(1)]
    report = plan_inference(workers, SMALL_MODEL, WORKLOAD)
    summary = report.summary()
    assert "stage0" in summary
    assert "stage1" in summary


@pytest.mark.parametrize(
    ("n", "weights"),
    [
        (5, [100.0, 100.0, 0.0]),
        (11, [1.0, 0.0, 0.0, 0.0]),
        (9, [0.0, 0.0, 0.0]),
        (17, [0.1, 2.5, 9.0, 0.2]),
    ],
)
def test_proportional_split_conserves_items_and_stage_ownership(n, weights):
    counts = _proportional_split(n, weights, sum(weights))
    assert sum(counts) == n
    assert all(count >= 1 for count in counts)


def test_proportional_split_rejects_nan_and_infinite_weights():
    with pytest.raises(ValueError, match="weights"):
        _proportional_split(4, [1.0, float("nan")], 1.0)
    with pytest.raises(ValueError, match="weights"):
        _proportional_split(4, [1.0, float("inf")], 1.0)


def test_inference_peak_accounts_for_endpoint_logits_and_attention_backend():
    sdpa_model = ModelSpec(
        num_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=4096,
        param_count=2_000_000,
        attention_implementation="sdpa",
    )
    eager_model = ModelSpec(
        **{
            **sdpa_model.__dict__,
            "attention_implementation": "eager",
        }
    )
    workload = InferenceWorkload(
        batch_size=2,
        max_prompt_tokens=128,
        max_new_tokens=64,
        max_concurrent=2,
    )
    interior = _inference_peak(
        sdpa_model, workload, 2, is_first=False, is_last=False
    )
    last = _inference_peak(sdpa_model, workload, 2, is_first=False, is_last=True)
    eager = _inference_peak(eager_model, workload, 2, is_first=False, is_last=False)

    assert last.workspace_bytes > interior.workspace_bytes
    assert eager.workspace_bytes > interior.workspace_bytes
    assert eager.total > interior.total


def test_training_peak_reflects_activation_checkpointing():
    workload_without = TrainingWorkload(
        batch_size=2,
        sequence_length=128,
        gradient_accumulation_steps=4,
        activation_checkpointing=False,
    )
    workload_with = TrainingWorkload(
        batch_size=2,
        sequence_length=128,
        gradient_accumulation_steps=4,
        activation_checkpointing=True,
    )
    without = _training_peak(
        SMALL_MODEL, workload_without, 3, is_first=True, is_last=False
    )
    with_checkpointing = _training_peak(
        SMALL_MODEL, workload_with, 3, is_first=True, is_last=False
    )
    assert with_checkpointing.workspace_bytes < without.workspace_bytes
    assert with_checkpointing.total < without.total


def test_lora_endpoint_parameters_stay_on_endpoint_stages():
    model = ModelSpec(
        **{
            **SMALL_MODEL.__dict__,
            "adapter_param_count": 5_100,
            "adapter_layer_param_count": 4_000,
            "adapter_embedding_param_count": 1_000,
            "adapter_lm_head_param_count": 100,
        }
    )

    assert _trainable_layer_params(model, 2, is_first=True) == 3_000
    assert _trainable_layer_params(model, 2, is_last=True) == 2_100
    assert _trainable_layer_params(model, 2) == 2_000
    first = _training_peak(
        model,
        TrainingWorkload(batch_size=1, sequence_length=8),
        2,
        is_first=True,
        is_last=False,
    )
    interior = _training_peak(
        model,
        TrainingWorkload(batch_size=1, sequence_length=8),
        2,
        is_first=False,
        is_last=False,
    )
    assert first.adapter_bytes - interior.adapter_bytes == 1_000 * 4


def test_planner_rejects_context_budget_before_memory_placement():
    model = ModelSpec(**{**SMALL_MODEL.__dict__, "max_position_embeddings": 600})
    with pytest.raises(ValueError, match="exceeds model context"):
        plan_inference(
            [_make_worker(0), _make_worker(1)],
            model,
            WORKLOAD,
        )


def test_training_planner_rejects_sequence_beyond_model_context():
    model = ModelSpec(**{**SMALL_MODEL.__dict__, "max_position_embeddings": 256})
    with pytest.raises(ValueError, match="exceeds model context"):
        plan_training(
            [_make_worker(0), _make_worker(1)],
            model,
            TrainingWorkload(batch_size=1, sequence_length=512),
        )
