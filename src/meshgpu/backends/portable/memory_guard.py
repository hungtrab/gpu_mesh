"""Runtime CUDA memory admission for remote LoRA training.

The static planner runs before a local launch, but a Kaggle worker loads its
stage in a separate process and may have unrelated allocations already
present.  This guard therefore probes the live device immediately before
adapter/optimizer allocation and again before each training forward.

The estimate is intentionally conservative.  A rejection is preferable to
letting a remote task discover CUDA OOM after another stage has already
created an autograd graph.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from meshgpu.backends.native.lora_recipe import (
    LoRAConfig,
    estimate_lora_parameters,
)
from meshgpu.planner.memory import GiB

_ADAPTER_DTYPE_BYTES = 4
_OPTIMIZER_BYTES_PER_PARAMETER = 12
_GRADIENT_BYTES_PER_PARAMETER = 4
_DEFAULT_RESERVE_FRACTION = 0.10
_DEFAULT_SAFETY_FRACTION = 0.20


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    """One stage's live CUDA admission result."""

    stage_id: int
    device: str
    feasible: bool
    reason: str
    free_bytes: int
    total_bytes: int
    reserve_bytes: int
    usable_bytes: int
    required_bytes: int
    adapter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    safety_bytes: int
    batch_size: int | None
    sequence_length: int | None
    trainable_parameters: int
    new_adapter_parameters: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_training_memory(
    worker: Any,
    lora_config: LoRAConfig,
    *,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    activation_checkpointing: bool = False,
    optimizer_state_needed: bool = True,
    new_adapter_storage_needed: bool = True,
    reserve_min_bytes: int = GiB,
    reserve_fraction: float = _DEFAULT_RESERVE_FRACTION,
    safety_fraction: float = _DEFAULT_SAFETY_FRACTION,
) -> TrainingMemoryEstimate:
    """Estimate one remote stage's incremental training peak.

    Base weights are already resident when this function is called, so live
    free VRAM is the correct starting point.  Adapter factors are fp32 in the
    MeshGPU recipe; gradients and AdamW state are charged conservatively in
    fp32 as well.  If the batch shape is omitted, the check covers static
    adapter/optimizer allocation only; the server must call it again with the
    real shape before forward.
    """
    _validate_optional_positive(batch_size, "batch_size")
    _validate_optional_positive(sequence_length, "sequence_length")
    if (batch_size is None) != (sequence_length is None):
        raise ValueError("batch_size and sequence_length must be provided together")
    if not isinstance(activation_checkpointing, bool):
        raise TypeError("activation_checkpointing must be boolean")
    if not isinstance(optimizer_state_needed, bool):
        raise TypeError("optimizer_state_needed must be boolean")
    if not isinstance(new_adapter_storage_needed, bool):
        raise TypeError("new_adapter_storage_needed must be boolean")
    _validate_non_negative_int(reserve_min_bytes, "reserve_min_bytes")
    _validate_fraction(reserve_fraction, "reserve_fraction", allow_zero=True)
    _validate_fraction(safety_fraction, "safety_fraction", allow_zero=True)

    model = getattr(worker, "_model", None)
    context = getattr(worker, "_ctx", None)
    if model is None or context is None:
        raise TypeError("worker must expose _model and _ctx")
    if not isinstance(lora_config, LoRAConfig):
        raise TypeError("lora_config must be a LoRAConfig")
    lora = estimate_lora_parameters(model, lora_config)
    trainable = lora.trainable_parameters
    adapter_bytes = (
        lora.new_adapter_parameters * _ADAPTER_DTYPE_BYTES
        if new_adapter_storage_needed
        else 0
    )
    gradient_bytes = trainable * _GRADIENT_BYTES_PER_PARAMETER
    optimizer_bytes = (
        trainable * _OPTIMIZER_BYTES_PER_PARAMETER
        if optimizer_state_needed
        else 0
    )
    activation_bytes = 0
    if batch_size is not None and sequence_length is not None:
        activation_bytes = _estimate_activation_bytes(
            model,
            batch_size,
            sequence_length,
            activation_checkpointing=activation_checkpointing,
            is_last=bool(getattr(context, "is_last", False)),
        )

    incremental = adapter_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    safety_bytes = math.ceil(incremental * safety_fraction)

    device = torch.device(getattr(context, "device"))
    stage_id = int(getattr(context, "stage_id", -1))
    if device.type != "cuda":
        return TrainingMemoryEstimate(
            stage_id=stage_id,
            device=str(device),
            feasible=True,
            reason="non-CUDA stage; live CUDA admission not required",
            free_bytes=0,
            total_bytes=0,
            reserve_bytes=0,
            usable_bytes=0,
            required_bytes=incremental + safety_bytes,
            adapter_bytes=adapter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_bytes=optimizer_bytes,
            activation_bytes=activation_bytes,
            safety_bytes=safety_bytes,
            batch_size=batch_size,
            sequence_length=sequence_length,
            trainable_parameters=trainable,
            new_adapter_parameters=lora.new_adapter_parameters,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for a CUDA training stage")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"could not read CUDA memory for {device}: {exc}") from exc
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    if total_bytes < 1 or free_bytes < 0 or free_bytes > total_bytes:
        raise RuntimeError(
            f"CUDA returned invalid memory for {device}: "
            f"free={free_bytes}, total={total_bytes}"
        )
    reserve_bytes = max(
        reserve_min_bytes,
        math.ceil(total_bytes * reserve_fraction),
    )
    usable_bytes = max(0, free_bytes - reserve_bytes)
    required_bytes = incremental + safety_bytes
    feasible = required_bytes <= usable_bytes
    if feasible:
        reason = (
            f"stage {stage_id} needs {_format_bytes(required_bytes)} additional VRAM; "
            f"{_format_bytes(usable_bytes)} is usable after reserve"
        )
    else:
        reason = (
            f"insufficient_memory: stage {stage_id} needs "
            f"{_format_bytes(required_bytes)} additional VRAM but only "
            f"{_format_bytes(usable_bytes)} is usable after reserve "
            f"(free={_format_bytes(free_bytes)}, reserve={_format_bytes(reserve_bytes)})"
        )
    return TrainingMemoryEstimate(
        stage_id=stage_id,
        device=str(device),
        feasible=feasible,
        reason=reason,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        reserve_bytes=reserve_bytes,
        usable_bytes=usable_bytes,
        required_bytes=required_bytes,
        adapter_bytes=adapter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        safety_bytes=safety_bytes,
        batch_size=batch_size,
        sequence_length=sequence_length,
        trainable_parameters=trainable,
        new_adapter_parameters=lora.new_adapter_parameters,
    )


def _estimate_activation_bytes(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    *,
    activation_checkpointing: bool,
    is_last: bool,
) -> int:
    hidden = _model_int(model, "hidden_size")
    intermediate = _model_int(model, "intermediate_size")
    heads = _model_int(model, "num_attention_heads")
    kv_heads = _model_int(model, "num_key_value_heads", default=heads)
    head_dim = _model_int(model, "head_dim", default=max(1, hidden // heads))
    vocab = _model_int(model, "vocab_size", default=0)
    layer_count = len(getattr(model, "layers"))
    dtype_bytes = _base_dtype_bytes(model)
    retained_layers = 1 if activation_checkpointing else layer_count
    boundary = batch_size * sequence_length * hidden * dtype_bytes
    internal = (
        batch_size
        * sequence_length
        * retained_layers
        * (
            hidden
            + 2 * intermediate
            + (heads + 2 * kv_heads) * head_dim
        )
        * dtype_bytes
    )
    backend = _attention_backend(model)
    if backend in {"sdpa", "flash_attention_2"}:
        attention = batch_size * sequence_length * hidden * dtype_bytes
    else:
        attention = (
            batch_size
            * heads
            * sequence_length
            * sequence_length
            * dtype_bytes
            * 3
        )
    logits = (
        batch_size * sequence_length * vocab * dtype_bytes
        if is_last
        else 0
    )
    # One GPU copy is retained by autograd and the remote boundary may create
    # another temporary during serialization.  The safety fraction handles
    # allocator fragmentation and kernel-specific scratch space.
    return internal + attention + logits + 2 * boundary


def _model_int(model: torch.nn.Module, name: str, default: int | None = None) -> int:
    owners = (
        model,
        getattr(model, "cfg", None),
        getattr(model, "config", None),
    )
    for owner in owners:
        if owner is None:
            continue
        value = getattr(owner, name, None)
        if value is not None:
            value = int(value)
            if value < 1:
                raise ValueError(f"model {name} must be positive")
            return value
    if default is not None:
        return default
    raise ValueError(f"model does not expose {name}")


def _attention_backend(model: torch.nn.Module) -> str:
    for owner in (
        model,
        getattr(model, "cfg", None),
        getattr(model, "config", None),
    ):
        if owner is None:
            continue
        for name in ("attn_implementation", "attention_implementation", "_attn_implementation"):
            value = getattr(owner, name, None)
            if isinstance(value, str) and value:
                return value
    return "eager"


def _base_dtype_bytes(model: torch.nn.Module) -> int:
    for name, parameter in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            return int(parameter.element_size())
    return 4


def _validate_optional_positive(value: int | None, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None")


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_fraction(value: float, name: str, *, allow_zero: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (value < 0 if allow_zero else value <= 0)
        or value > 1
    ):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}")


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"
