"""Tests for the fit-first capacity acceptance contract."""
from __future__ import annotations

import torch

from meshgpu.benchmarks.capacity import (
    CapacityContract,
    CapacityRun,
    measure_capacity_run,
    outputs_close,
    run_capacity_gate,
)


def _contract() -> CapacityContract:
    return CapacityContract(
        model_id="qwen3-test",
        model_revision="revision-1",
        compute_dtype="float16",
        prompt_ids=(1, 4, 9),
        max_new_tokens=5,
    )


def test_contract_fingerprint_is_stable_and_covers_workload() -> None:
    first = _contract()
    second = _contract()
    assert first.fingerprint == second.fingerprint
    assert first.context_tokens == 8
    changed = CapacityContract(
        model_id=first.model_id,
        model_revision=first.model_revision,
        compute_dtype=first.compute_dtype,
        prompt_ids=first.prompt_ids,
        max_new_tokens=first.max_new_tokens + 1,
    )
    assert changed.fingerprint != first.fingerprint


def test_measure_capacity_run_snapshots_nested_cpu_output() -> None:
    output = measure_capacity_run(
        "reference",
        lambda _contract: {"ids": [1, 2], "logits": torch.ones(2)},
        _contract(),
        ["cpu"],
    )
    assert output.succeeded
    assert output.devices == ("cpu",)
    assert output.output["ids"] == [1, 2]
    assert torch.equal(output.output["logits"], torch.ones(2))
    assert output.peak_allocated_by_device == {}
    assert output.peak_reserved_by_device == {}


def test_outputs_close_requires_matching_structure_and_values() -> None:
    expected = {"ids": torch.tensor([1, 2]), "logits": [torch.tensor([1.0, 2.0])]}
    actual = {"ids": torch.tensor([1, 2]), "logits": [torch.tensor([1.0, 2.0 + 1e-6])]}
    assert outputs_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert not outputs_close({"ids": torch.tensor([1, 3])}, expected, atol=1e-5, rtol=1e-4)


def test_gate_is_pending_without_two_physical_cuda_devices() -> None:
    report = run_capacity_gate(
        _contract(),
        single_runner=lambda _contract: torch.tensor([1, 2]),
        sharded_runner=lambda _contract: torch.tensor([1, 2]),
        reference_runner=lambda _contract: torch.tensor([1, 2]),
        single_device="cpu",
        sharded_devices=["cpu", "cpu"],
    )
    assert report.status == "pending_hardware"
    assert not report.passed
    assert report.correctness_ok is True


def test_gate_passes_only_when_real_oom_and_sharded_reference_match(monkeypatch) -> None:
    runs = {
        "single_gpu": CapacityRun("single_gpu", "cuda_oom", ("cuda:0",)),
        "sharded": CapacityRun("sharded", "success", ("cuda:0", "cuda:1"), output=[7, 8]),
        "reference": CapacityRun("reference", "success", ("cpu",), output=[7, 8]),
    }

    def fake_measure(name, _runner, _contract, _devices):
        return runs[name]

    monkeypatch.setattr("meshgpu.benchmarks.capacity.measure_capacity_run", fake_measure)
    # The runner results are synthetic here; mock only the physical topology
    # probe so this unit test exercises decision logic, not hardware claims.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    report = run_capacity_gate(
        _contract(),
        single_runner=lambda _contract: None,
        sharded_runner=lambda _contract: None,
        reference_runner=lambda _contract: None,
        single_device="cuda:0",
        sharded_devices=["cuda:0", "cuda:1"],
    )
    assert report.status == "passed"
    assert report.passed
    assert report.correctness_ok is True


def test_gate_rejects_successful_single_gpu_baseline(monkeypatch) -> None:
    runs = {
        "single_gpu": CapacityRun("single_gpu", "success", ("cuda:0",), output=[7, 8]),
        "sharded": CapacityRun("sharded", "success", ("cuda:0", "cuda:1"), output=[7, 8]),
        "reference": CapacityRun("reference", "success", ("cpu",), output=[7, 8]),
    }
    monkeypatch.setattr(
        "meshgpu.benchmarks.capacity.measure_capacity_run",
        lambda name, _runner, _contract, _devices: runs[name],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    report = run_capacity_gate(
        _contract(),
        single_runner=lambda _contract: None,
        sharded_runner=lambda _contract: None,
        reference_runner=lambda _contract: None,
        single_device="cuda:0",
        sharded_devices=["cuda:0", "cuda:1"],
    )
    assert report.status == "failed"
    assert any("did not produce" in reason for reason in report.reasons)


def test_contract_rejects_empty_quantization_name() -> None:
    try:
        CapacityContract(
            model_id="model",
            model_revision="rev",
            compute_dtype="float16",
            prompt_ids=(1,),
            max_new_tokens=1,
            quantization="",
        )
    except ValueError as exc:
        assert "quantization" in str(exc)
    else:  # pragma: no cover - assertion keeps the test explicit
        raise AssertionError("empty quantization name should be rejected")
