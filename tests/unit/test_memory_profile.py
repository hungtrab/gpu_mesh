"""Tests for measured, workload-keyed memory planning."""
from __future__ import annotations

import pytest
import torch

from meshgpu.planner.memory_profile import (
    MemoryProfile,
    StageMemoryMeasurement,
    WorkloadSignature,
    plan_from_profile,
    profile_cuda_phase,
)
from meshgpu.planner.placement import ModelSpec, WorkerSpec


def _profile() -> MemoryProfile:
    signature = WorkloadSignature(
        model_id="qwen3-test@rev1",
        mode="inference",
        compute_dtype="float16",
        batch_size=1,
        prompt_tokens=32,
        max_new_tokens=16,
        attention_backend="sdpa",
    )
    profile = MemoryProfile(signature, hardware_profile="test-gpu", software_profile="test")
    for stage_id, (start, end), peak in [
        (0, (0, 2), 2 * 1024**3),
        (1, (2, 4), 3 * 1024**3),
    ]:
        profile.add(
            StageMemoryMeasurement(
                stage_id=stage_id,
                worker_id=f"w{stage_id}",
                device_index=stage_id,
                layer_start=start,
                layer_end=end,
                peak_allocated_bytes=peak,
                peak_reserved_bytes=peak,
                phase="prefill",
            )
        )
    return profile


def _model() -> ModelSpec:
    return ModelSpec(
        num_layers=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=97,
        param_count=1000,
    )


def _workers() -> list[WorkerSpec]:
    return [
        WorkerSpec("w0", 0, 10 * 1024**3, 10 * 1024**3),
        WorkerSpec("w1", 1, 10 * 1024**3, 10 * 1024**3),
    ]


def test_profile_roundtrip_and_key(tmp_path) -> None:
    profile = _profile()
    path = tmp_path / "memory.json"
    profile.save(path)
    loaded = MemoryProfile.load(path)
    assert loaded.signature.key == profile.signature.key
    assert loaded.find(0, 0, 2, worker_id="w0") is not None


def test_measured_plan_considers_exact_ranges() -> None:
    report = plan_from_profile(_workers(), _model(), _profile())
    assert report.feasible
    assert report.profile_key == _profile().signature.key
    assert report.bottleneck_stage in {0, 1}
    assert all(item.peak.cuda_context_bytes == 0 for item in report.assignments)


def test_missing_exact_measurement_is_rejected() -> None:
    profile = _profile()
    profile.measurements.pop()
    report = plan_from_profile(_workers(), _model(), profile)
    assert not report.feasible
    assert "missing exact" in report.reason


def test_non_exact_mode_is_a_marked_static_preview() -> None:
    profile = _profile()
    profile.measurements.clear()
    report = plan_from_profile(_workers(), _model(), profile, require_exact=False)
    assert report.feasible
    assert not report.safe_for_admission
    assert "static estimate only" in report.warnings[0]
    assert not report.as_preflight().feasible
    assert report.as_preflight().details["safe_for_admission"] is False


def test_ttt_default_optimizer_has_a_static_preview() -> None:
    signature = WorkloadSignature(
        model_id="qwen3-test@rev1",
        mode="ttt",
        compute_dtype="float16",
        batch_size=1,
        sequence_length=32,
        lora_rank=8,
    )
    profile = MemoryProfile(signature, hardware_profile="test-gpu", software_profile="test")
    report = plan_from_profile(_workers(), _model(), profile, require_exact=False)
    assert report.feasible
    assert not report.safe_for_admission
    assert "static estimate only" in report.warnings[0]


def test_workload_signature_rejects_invalid_exactness_fields() -> None:
    with pytest.raises(ValueError, match="compute_dtype"):
        WorkloadSignature(
            model_id="model",
            mode="inference",
            compute_dtype="int4",
            batch_size=1,
        )
    with pytest.raises(TypeError, match="gradient_accumulation_steps"):
        WorkloadSignature(
            model_id="model",
            mode="ttt",
            compute_dtype="float16",
            batch_size=1,
            gradient_accumulation_steps=True,
        )


def test_cpu_phase_profile_has_zero_cuda_counters() -> None:
    value, allocated, reserved = profile_cuda_phase(torch.device("cpu"), lambda: 17)
    assert value == 17
    assert allocated == 0
    assert reserved == 0
