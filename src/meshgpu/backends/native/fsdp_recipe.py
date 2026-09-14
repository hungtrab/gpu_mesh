"""
FSDP2 training recipe for single-node multi-GPU.
Wraps supported decoder blocks with torch.distributed.fsdp.fully_shard (FSDP2).

Requirements:
  - PyTorch >= 2.4 with torch.distributed
  - Multiple CUDA GPUs visible
  - Launched via torchrun or equivalent

This module is imported only when native FSDP2 training is requested;
import errors are caught at the caller so CPU-only environments still work.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import torch

log = logging.getLogger(__name__)


@dataclass
class FSDPConfig:
    mixed_precision: str = "bfloat16"   # "float16", "bfloat16", "float32"
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    sharding_strategy: str = "full_shard"  # FSDP2 default
    grad_reduce_dtype: str = "float32"
    checkpoint_activations_policy: str = "block"  # wrap per transformer block

    def __post_init__(self) -> None:
        allowed = {"float32", "float16", "bfloat16"}
        if self.mixed_precision not in allowed:
            raise ValueError(f"unsupported mixed_precision: {self.mixed_precision!r}")
        if self.grad_reduce_dtype not in allowed:
            raise ValueError(f"unsupported grad_reduce_dtype: {self.grad_reduce_dtype!r}")


@dataclass
class TrainingConfig:
    global_batch_tokens: int = 131072    # 128k tokens per global step
    micro_batch_tokens: int = 4096       # tokens per microbatch
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.1
    ignore_index: int = -100
    mixed_precision: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.global_batch_tokens < 1 or self.micro_batch_tokens < 1:
            raise ValueError("batch token counts must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.mixed_precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported mixed_precision: {self.mixed_precision!r}")


def setup_fsdp(
    model: Any,
    cfg: FSDPConfig,
    device_mesh: Any | None = None,
) -> Any:
    """
    Wrap model with FSDP2 fully_shard.
    Returns the sharded model ready for training.
    """
    try:
        import torch
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as e:
        raise RuntimeError(
            f"FSDP2 requires PyTorch >= 2.4 with torch.distributed: {e}"
        ) from e

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if cfg.mixed_precision not in dtype_map:
        raise ValueError(f"unsupported mixed_precision: {cfg.mixed_precision!r}")
    if cfg.grad_reduce_dtype not in dtype_map:
        raise ValueError(f"unsupported grad_reduce_dtype: {cfg.grad_reduce_dtype!r}")
    if cfg.cpu_offload:
        raise NotImplementedError(
            "FSDP2 CPU offload is not enabled in this recipe; use a profile-backed "
            "portable/offload backend instead"
        )
    if cfg.sharding_strategy != "full_shard":
        raise ValueError(
            "this FSDP2 recipe supports only sharding_strategy='full_shard'"
        )
    if cfg.checkpoint_activations_policy != "block":
        raise ValueError(
            "this FSDP2 recipe supports only checkpoint_activations_policy='block'"
        )
    param_dtype = dtype_map[cfg.mixed_precision]

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=dtype_map[cfg.grad_reduce_dtype],
        output_dtype=param_dtype,
    )

    shard_kwargs: dict[str, Any] = {"mp_policy": mp_policy}
    if device_mesh is not None:
        shard_kwargs["mesh"] = device_mesh

    # Wrap each transformer block independently for fine-grained sharding.
    # Qwen3 is optional, so its official block class is discovered lazily and
    # never makes the base package depend on Transformers.
    block_types = _decoder_block_types()
    for layer in model.modules():
        if isinstance(layer, block_types):
            fully_shard(cast(Any, layer), **shard_kwargs)

    # Wrap the root model
    fully_shard(model, **shard_kwargs)

    if cfg.activation_checkpointing:
        _apply_activation_checkpointing(model)

    return model


def _apply_activation_checkpointing(model: Any) -> None:
    block_types = _decoder_block_types()
    block_count = sum(1 for module in model.modules() if isinstance(module, block_types))
    if block_count == 0:
        raise RuntimeError(
            "activation checkpointing was requested, but no supported decoder "
            "blocks were found in the model"
        )
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        def check_fn(module: Any) -> bool:
            return isinstance(module, block_types)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        log.info("activation checkpointing applied to %d decoder blocks", block_count)
    except Exception as exc:
        # A configured memory-saving feature cannot degrade to a normal
        # uncheckpointed run: that would make the planner's fit decision
        # unsound and usually surfaces later as an avoidable CUDA OOM.
        raise RuntimeError("could not apply activation checkpointing") from exc


def _decoder_block_types() -> tuple[type, ...]:
    """Return block classes supported by the native recipe."""
    from meshgpu.models.llama_dense import LlamaDecoderLayer

    block_types: list[type] = [LlamaDecoderLayer]
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        block_types.append(Qwen3DecoderLayer)
    return tuple(block_types)


def compute_global_grad_norm(
    model: Any,
    max_norm: float,
) -> float:
    """
    Compute global gradient norm across all FSDP shards and clip.
    Must be called after loss.backward() and before optimizer.step().
    """
    if max_norm <= 0 or not math.isfinite(max_norm):
        raise ValueError("max_norm must be finite and positive")
    try:
        import torch

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm
        )
        return float(total_norm)
    except Exception as exc:
        # A failed clip must abort the optimizer boundary.  Returning zero
        # would make the caller believe gradients were safely clipped and can
        # silently apply an unbounded update after an FSDP/runtime error.
        raise RuntimeError("global gradient norm computation failed") from exc


def training_step(
    model: Any,
    optimizer: Any,
    scaler: Any | None,
    input_ids: Any,
    labels: Any,
    cfg: TrainingConfig,
) -> dict[str, float]:
    """
    Single microbatch training step.
    Caller accumulates gradients and calls optimizer.step() after
    gradient_accumulation_steps microbatches.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError(f"PyTorch required: {e}") from e

    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape [batch, sequence]")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must have at least one parameter") from exc
    input_ids = input_ids.to(model_device)
    labels = labels.to(model_device)
    autocast_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[cfg.mixed_precision]
    autocast_enabled = model_device.type in {"cuda", "cpu"} and autocast_dtype != torch.float32

    with torch.autocast(
        device_type=model_device.type,
        dtype=autocast_dtype,
        enabled=autocast_enabled,
    ):
        logits = _forward_logits(model, input_ids, model_device)
        valid_mask = labels != cfg.ignore_index
        n_valid = int(valid_mask.sum().item())
        denominator = max(n_valid, 1)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=cfg.ignore_index,
            reduction="sum",
        ) / denominator

    # Scale for accumulation
    loss_scaled = loss / cfg.gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(loss_scaled).backward()
    else:
        loss_scaled.backward()

    return {"loss": loss.item(), "n_valid_tokens": n_valid}


def _forward_logits(model: Any, input_ids: Any, model_device: Any) -> Any:
    """Run either a MeshGPU stage or a native HF causal-LM model."""
    if getattr(model, "has_embedding", False) or getattr(model, "meshgpu_adapter", None):
        logits, _ = model(
            torch.empty(0, device=model_device),
            input_ids=input_ids,
            use_cache=False,
        )
        return logits
    output = model(input_ids=input_ids, use_cache=False)
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("native model output does not expose causal-LM logits")
