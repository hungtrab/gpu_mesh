"""Pre-load CUDA capacity checks for local training.

The model loader is deliberately allowed to allocate only after this module
has inspected the immutable artifact and the current physical GPUs.  The
check is conservative and static; it is a guard against an obviously unsafe
launch, not a substitute for a measured warmup profile.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from meshgpu.planner.job import (
    lora_param_breakdown_from_manifest,
    model_spec_from_manifest,
)
from meshgpu.planner.memory import GiB
from meshgpu.planner.placement import (
    InferenceWorkload,
    ModelSpec,
    PlacementReport,
    TrainingWorkload,
    WorkerSpec,
    plan_inference,
    plan_training,
)

log = logging.getLogger(__name__)


def collect_cuda_worker_specs(
    devices: Sequence[torch.device | str],
    *,
    torch_module: Any = torch,
) -> list[WorkerSpec]:
    """Read current physical VRAM and build one planner worker per device.

    Device identities are canonicalized before the duplicate check.  Passing
    ``cuda`` twice therefore cannot accidentally make one card look like two
    independent workers.
    """
    if not devices:
        raise ValueError("at least one CUDA device is required")
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot run CUDA training preflight")

    count = int(cuda.device_count())
    if count < 1:
        raise RuntimeError("CUDA reports no physical devices")

    canonical: list[int] = []
    for position, raw_device in enumerate(devices):
        try:
            device = torch_module.device(raw_device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid CUDA device at stage {position}: {raw_device!r}") from exc
        if device.type != "cuda":
            raise ValueError(
                "CUDA training preflight requires every stage device to be CUDA; "
                f"stage {position} is {device}"
            )
        index = device.index
        if index is None:
            index = int(cuda.current_device())
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device index {index} at stage {position} is unavailable; "
                f"CUDA reports {count} device(s)"
            )
        canonical.append(int(index))

    if len(set(canonical)) != len(canonical):
        raise ValueError(
            "CUDA training preflight requires one distinct CUDA device per stage"
        )

    workers: list[WorkerSpec] = []
    for stage_id, device_index in enumerate(canonical):
        try:
            properties = cuda.get_device_properties(device_index)
            free_bytes, total_bytes = cuda.mem_get_info(device_index)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(
                f"could not read VRAM for CUDA device {device_index}: {exc}"
            ) from exc
        total = int(getattr(properties, "total_memory"))
        free = int(free_bytes)
        if total < 1 or free < 0 or free > total:
            raise RuntimeError(
                f"CUDA device {device_index} returned invalid VRAM values: "
                f"free={free}, total={total}"
            )
        workers.append(
            WorkerSpec(
                worker_id=f"stage-{stage_id}",
                device_index=device_index,
                total_vram_bytes=total,
                free_vram_bytes=free,
            )
        )
    return workers


def plan_cuda_training_from_manifest(
    manifest_or_dir: Any,
    devices: Sequence[torch.device | str],
    *,
    artifact_root: str | Path | None = None,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int = 1,
    activation_checkpointing: bool = False,
    recipe: str = "lora",
    lora_rank: int = 16,
    optimizer: str = "adamw",
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int = GiB,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    modules_to_save: tuple[str, ...] = (),
) -> PlacementReport:
    """Plan a local training launch before any stage weights reach a GPU.

    The artifact is inspected on the ``meta`` device, LoRA memory is counted
    from the real projection shapes, and physical ``free/total`` VRAM is read
    immediately before placement.  The returned assignments are safe to pass
    as ``layer_ranges`` to ``build_pipeline_from_manifest``.
    """
    if recipe not in {"full", "lora"}:
        raise ValueError(f"unsupported training recipe: {recipe!r}")
    if recipe == "lora" and (
        isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 1
    ):
        raise ValueError("lora_rank must be a positive integer")
    if not isinstance(activation_checkpointing, bool):
        raise TypeError("activation_checkpointing must be a boolean")

    model = model_spec_from_manifest(manifest_or_dir, artifact_root=artifact_root)
    if recipe == "lora":
        breakdown = lora_param_breakdown_from_manifest(
            manifest_or_dir,
            lora_rank,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            artifact_root=artifact_root,
        )
        if breakdown["other"]:
            raise ValueError(
                "modules_to_save contains non-endpoint modules that the planner "
                "cannot assign safely: "
                f"{breakdown['other']} parameters"
            )
        model = replace(
            model,
            adapter_param_count=breakdown["total"],
            adapter_layer_param_count=breakdown["layer"],
            adapter_embedding_param_count=breakdown["embedding"],
            adapter_lm_head_param_count=breakdown["lm_head"],
            # LoRALinear deliberately owns fp32 Parameters regardless of the
            # frozen artifact's compute dtype.
            adapter_dtype_bytes=4,
        )
    else:
        # A base artifact should normally already have zero adapters.  Do not
        # let a stale adapter-bearing manifest silently become a full recipe.
        if model.adapter_param_count:
            raise ValueError(
                "full fine-tuning requires a base artifact without LoRA parameters"
            )

    model, attention_warnings = _conservative_attention_model(
        model,
        devices,
        compute_dtype=_manifest_compute_dtype(manifest_or_dir),
    )
    workers = collect_cuda_worker_specs(devices)
    report = plan_training(
        workers,
        model,
        TrainingWorkload(
            batch_size=batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            activation_checkpointing=activation_checkpointing,
        ),
        max_vram_fraction=max_vram_fraction,
        reserve_min_bytes=reserve_min_bytes,
        optimizer=optimizer,
    )
    report.warnings[:0] = attention_warnings
    report.warnings.append(
        "pre-load CUDA estimate is conservative but static; run the post-load "
        "warmup/profile before publishing this configuration as capacity-verified"
    )
    return report


def plan_cuda_inference_from_manifest(
    manifest_or_dir: Any,
    devices: Sequence[torch.device | str],
    *,
    artifact_root: str | Path | None = None,
    max_prompt_tokens: int,
    max_new_tokens: int,
    batch_size: int = 1,
    max_concurrent: int = 1,
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int = GiB,
) -> PlacementReport:
    """Plan a local inference launch before loading any model weights."""
    model = model_spec_from_manifest(manifest_or_dir, artifact_root=artifact_root)
    model, attention_warnings = _conservative_attention_model(
        model,
        devices,
        compute_dtype=_manifest_compute_dtype(manifest_or_dir),
    )
    workers = collect_cuda_worker_specs(devices)
    report = plan_inference(
        workers,
        model,
        InferenceWorkload(
            batch_size=batch_size,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_concurrent=max_concurrent,
        ),
        max_vram_fraction=max_vram_fraction,
        reserve_min_bytes=reserve_min_bytes,
    )
    report.warnings[:0] = attention_warnings
    report.warnings.append(
        "pre-load CUDA estimate is conservative but static; run a full warmup "
        "at the advertised context/concurrency before calling the capacity gate"
    )
    return report


def _conservative_attention_model(
    model: ModelSpec,
    devices: Sequence[torch.device | str],
    *,
    compute_dtype: str,
) -> tuple[ModelSpec, list[str]]:
    """Use eager workspace accounting unless a fused SDPA path is verified."""
    if model.attention_implementation == "eager":
        return model, []
    if model.attention_implementation == "flash_attention_2":
        return replace(model, attention_implementation="eager"), [
            "flash_attention_2 was not verified before loading; using the "
            "conservative eager attention workspace estimate",
        ]

    from meshgpu.backends.portable.attention import verify_sdpa_backend

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(compute_dtype)
    if dtype is None:
        raise ValueError(f"unsupported compute dtype: {compute_dtype!r}")
    failed = False
    for raw_device in devices:
        device = torch.device(raw_device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        try:
            result = verify_sdpa_backend(
                (
                    1,
                    model.num_attention_heads,
                    min(model.max_position_embeddings or 128, 128),
                    model.head_dim,
                ),
                device=device,
                dtype=dtype,
            )
        except Exception as exc:  # pragma: no cover - defensive CUDA boundary
            log.warning("SDPA verification failed on %s: %s", device, exc)
            failed = True
            continue
        if not result.verified or not result.fused:
            failed = True
    if not failed:
        return model, []
    return replace(model, attention_implementation="eager"), [
        "at least one GPU did not verify a fused SDPA kernel before loading; "
        "using the conservative eager attention workspace estimate",
    ]


def _manifest_compute_dtype(manifest_or_dir: Any) -> str:
    """Read the artifact dtype without loading its config or weights."""
    from meshgpu.artifacts.manifest import ModelManifest

    if isinstance(manifest_or_dir, ModelManifest):
        return manifest_or_dir.meta.compute_dtype
    path = Path(manifest_or_dir)
    root = path if path.is_dir() else path.parent
    manifest = ModelManifest.load(root / "manifest.json")
    return manifest.meta.compute_dtype
