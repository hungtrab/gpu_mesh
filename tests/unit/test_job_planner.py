"""Tests for YAML-job planning without touching a real GPU."""

import pytest

from meshgpu.planner.job import plan_job, report_to_dict


def _model():
    return {
        "num_layers": 4,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "dtype_bytes": 2,
    }


def _workers():
    return [
        {"worker_id": "gpu-a", "device_index": 0, "total_vram_gib": 16},
        {"worker_id": "gpu-b", "device_index": 1, "total_vram_gib": 16},
    ]


def test_plan_job_inference_returns_stage_mapping():
    report = plan_job(
        {
            "task": "inference",
            "model": _model(),
            "placement": {"workers": _workers()},
            "inference": {"max_prompt_tokens": 128, "max_new_tokens": 32},
        }
    )

    assert report.feasible
    assert [(item.layer_start, item.layer_end) for item in report.assignments] == [
        (0, 2),
        (2, 4),
    ]
    assert report_to_dict(report)["assignments"][0]["budget"]["usable_bytes"] > 0


def test_plan_job_training_includes_optimizer_memory():
    report = plan_job(
        {
            "task": "training",
            "model": _model(),
            "placement": {"workers": _workers()},
            "training": {"sequence_length": 128, "optimizer": "adamw"},
        }
    )

    assert report.feasible
    assert all(item.peak.optimizer_bytes > 0 for item in report.assignments)
    assert "training microbatch" in report.reason


def test_plan_job_propagates_attention_and_checkpointing_fields():
    model = {**_model(), "attention_implementation": "eager", "max_position_embeddings": 256}
    report = plan_job(
        {
            "task": "training",
            "model": model,
            "placement": {"workers": _workers()},
            "training": {
                "sequence_length": 128,
                "activation_checkpointing": True,
            },
        }
    )
    assert report.feasible
    assert all(item.peak.workspace_bytes > 0 for item in report.assignments)


def test_plan_job_does_not_coerce_non_boolean_checkpointing():
    with pytest.raises(TypeError, match="activation_checkpointing"):
        plan_job(
            {
                "task": "training",
                "model": _model(),
                "placement": {"workers": _workers()},
                "training": {"activation_checkpointing": "false"},
            }
        )


def test_plan_job_requires_worker_capacity():
    with pytest.raises(ValueError, match="VRAM"):
        plan_job(
            {
                "model": _model(),
                "placement": {"workers": ["gpu-a"]},
            }
        )


def test_plan_job_accepts_worker_specs_and_size_units():
    report = plan_job(
        {
            "model": _model(),
            "worker_specs": {
                "gpu-a": {"total_vram": "16GiB", "free_vram": "15GiB"},
                "gpu-b": {"total_vram": "16GiB", "free_vram": "15GiB"},
            },
            "placement": {"workers": ["gpu-a", "gpu-b"]},
            "resources": {"reserve_min_gib": "1GiB"},
            "inference": {"max_prompt_tokens": 64, "max_new_tokens": 16},
        }
    )

    assert len(report.assignments) == 2
