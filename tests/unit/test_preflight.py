"""Tests for capacity checks that run before CUDA stage construction."""
from __future__ import annotations

import torch
from click.testing import CliRunner

from meshgpu.cli.main import cli
from meshgpu.planner.job import lora_param_count_from_manifest
from meshgpu.planner.placement import ModelSpec, _trainable_layer_params
from meshgpu.planner.preflight import plan_cuda_training_from_manifest
from tests.unit.test_hf_import import _save_shards, _tiny_cfg


def _fake_cuda(monkeypatch, *, total: int, free: int, count: int = 2) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    class Properties:
        total_memory = total

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: Properties(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (free, total),
    )


def test_lora_count_uses_real_projection_shapes(tmp_path) -> None:
    artifact = _save_shards(_tiny_cfg(), 2, tmp_path)

    # q and v are [out, in] = [16, 16] in this GQA fixture.
    assert lora_param_count_from_manifest(artifact, 4) == 4 * (32 + 32) * 4


def test_cuda_training_preflight_returns_loader_ranges(monkeypatch, tmp_path) -> None:
    artifact = _save_shards(_tiny_cfg(), 2, tmp_path)
    _fake_cuda(monkeypatch, total=16 * 1024**3, free=16 * 1024**3)
    monkeypatch.setattr(
        "meshgpu.planner.preflight._conservative_attention_model",
        lambda model, devices, *, compute_dtype: (model, []),
    )

    report = plan_cuda_training_from_manifest(
        artifact,
        [torch.device("cuda:0"), torch.device("cuda:1")],
        batch_size=1,
        sequence_length=32,
        recipe="lora",
        lora_rank=4,
    )

    assert report.feasible
    assert [(item.layer_start, item.layer_end) for item in report.assignments] == [
        (0, 2),
        (2, 4),
    ]
    assert all(item.peak.adapter_bytes > 0 for item in report.assignments)
    assert any("pre-load" in warning for warning in report.warnings)


def test_cuda_training_preflight_rejects_before_loader_on_small_cards(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = _save_shards(_tiny_cfg(), 2, artifact_root)
    _fake_cuda(monkeypatch, total=1 * 1024**3, free=1 * 1024**3)
    monkeypatch.setattr(
        "meshgpu.planner.preflight._conservative_attention_model",
        lambda model, devices, *, compute_dtype: (model, []),
    )

    result = plan_cuda_training_from_manifest(
        artifact,
        [torch.device("cuda:0"), torch.device("cuda:1")],
        batch_size=1,
        sequence_length=32,
        recipe="full",
    )

    assert not result.feasible
    assert "exceeds usable" in result.reason


def test_full_training_peak_includes_endpoint_optimizer_state() -> None:
    model = ModelSpec(
        num_layers=4,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=4,
        vocab_size=32,
        param_count=1000,
        per_layer_param_count=10,
        embedding_param_count=100,
        lm_head_param_count=200,
    )

    assert _trainable_layer_params(model, 2, is_first=True) == 120
    assert _trainable_layer_params(model, 2, is_last=True) == 220


def test_cli_rejects_cuda_training_before_building_workers(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = _save_shards(_tiny_cfg(), 2, artifact_root)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "unused"}\n')
    _fake_cuda(monkeypatch, total=1 * 1024**3, free=1 * 1024**3)
    monkeypatch.setattr(
        "meshgpu.planner.preflight._conservative_attention_model",
        lambda model, devices, *, compute_dtype: (model, []),
    )

    def should_not_load(*args, **kwargs):
        raise AssertionError("stage loader must not run after preflight rejection")

    monkeypatch.setattr(
        "meshgpu.backends.portable.pipeline.build_pipeline_from_manifest",
        should_not_load,
    )
    result = CliRunner().invoke(
        cli,
        [
            "fine-tune",
            "--manifest",
            str(artifact),
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "out"),
            "--sequence-length",
            "32",
            "--devices",
            "cuda:0,cuda:1",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "rejected before loading weights" in result.output


def test_cli_rejects_cuda_inference_before_building_workers(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = _save_shards(_tiny_cfg(), 2, artifact_root)
    _fake_cuda(monkeypatch, total=1 * 1024**3, free=1 * 1024**3)
    monkeypatch.setattr(
        "meshgpu.planner.preflight._conservative_attention_model",
        lambda model, devices, *, compute_dtype: (model, []),
    )

    def should_not_load(*args, **kwargs):
        raise AssertionError("stage loader must not run after preflight rejection")

    monkeypatch.setattr(
        "meshgpu.backends.portable.pipeline.build_pipeline_from_manifest",
        should_not_load,
    )
    result = CliRunner().invoke(
        cli,
        [
            "serve",
            "--manifest",
            str(artifact),
            "--devices",
            "cuda:0,cuda:1",
            "--max-prompt-tokens",
            "16",
            "--max-new-tokens",
            "8",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "rejected before loading weights" in result.output
