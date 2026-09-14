"""
LoRA recipe — adds low-rank adapter layers to a frozen LlamaStage.

Adds LoRA to q_proj and v_proj (configurable). Base weights are frozen.
Gradient flows through the adapter only. Compatible with both
portable pipeline training and native single-node FSDP2.

Note from plan.md §9.4:
  - Base model must fit on the GPU(s); LoRA reduces optimizer state only.
  - ΔW = B×A; averaging factors from different replicas is a different algorithm.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.0

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("LoRA rank must be an integer")
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        if not self.target_modules:
            raise ValueError("LoRA target_modules must not be empty")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("LoRA target_modules must contain non-empty strings")


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = linear
        self.base.requires_grad_(False)
        in_f, out_f = linear.in_features, linear.out_features
        # Keep the adapter parameters alongside the frozen base module.  A
        # newly-created Parameter otherwise defaults to CPU, which makes a
        # LoRA-wrapped CUDA layer fail on its first forward pass.  Adapters
        # intentionally stay fp32 for stable training; ``forward`` handles
        # the cast back to the base output dtype.
        adapter_device = linear.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_f, device=adapter_device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, rank, device=adapter_device, dtype=torch.float32)
        )
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Keep adapters in their parameter dtype (normally fp32 even when the
        # frozen base is fp16/bf16), then cast the low-rank result back to the
        # base output dtype before adding it.
        adapter_x = self.dropout(x).to(self.lora_A.dtype)
        lora_out = adapter_x @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out.to(base_out.dtype) * self.scale

    def merge(self) -> nn.Linear:
        """Return a new Linear with ΔW merged in (for export/inference)."""
        delta = (self.lora_B @ self.lora_A) * self.scale
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(stage: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """
    Freeze base weights and inject LoRA adapters in-place.
    Returns the same stage (modified).
    """
    layers = getattr(stage, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("LoRA stage must expose a decoder `layers` ModuleList")
    # Validate the complete target set before mutating the stage.  A partial
    # adapter is worse than a loud failure: it would silently train only some
    # layers and make checkpoints impossible to interpret.
    target_modules = tuple(dict.fromkeys(cfg.target_modules))
    targets: list[tuple[Any, str, nn.Module]] = []
    missing: list[str] = []
    for layer in layers:
        attn: Any = getattr(layer, "self_attn", None)
        if attn is None:
            missing.extend(f"{type(layer).__name__}.{name}" for name in target_modules)
            continue
        for mod_name in target_modules:
            if not hasattr(attn, mod_name):
                missing.append(f"{type(attn).__name__}.{mod_name}")
                continue
            original = getattr(attn, mod_name)
            if not isinstance(original, (nn.Linear, LoRALinear)):
                raise TypeError(
                    f"LoRA target {type(attn).__name__}.{mod_name} must be nn.Linear "
                    f"or LoRALinear, got {type(original).__name__}"
                )
            if isinstance(original, LoRALinear):
                if (
                    original.rank != cfg.rank
                    or not math.isclose(
                        original.scale, cfg.scale, rel_tol=1e-6, abs_tol=1e-8
                    )
                    or not math.isclose(
                        original.dropout_p, cfg.dropout, rel_tol=1e-6, abs_tol=1e-8
                    )
                ):
                    raise ValueError(
                        f"existing LoRA target {mod_name} has rank/scale/dropout "
                        "incompatible with the requested LoRAConfig"
                    )
            targets.append((attn, mod_name, original))
    if missing:
        raise ValueError("LoRA target modules missing: " + ", ".join(sorted(set(missing))))

    # Freeze everything, including an HF-backed Qwen3 stage.  The replacement
    # below then adds only the adapter Parameters back as trainable.  This is
    # PEFT-compatible q_proj/v_proj semantics while keeping the stage
    # checkpoint layout independent of a full-model wrapper.
    for p in stage.parameters():
        p.requires_grad_(False)

    n_replaced = 0
    for attn, mod_name, original in targets:
        if isinstance(original, LoRALinear):
            # Make repeated calls idempotent.  This matters when a resumed
            # TTT session or CLI path applies the recipe to an already wrapped
            # stage: do not stack adapters or accidentally leave them frozen.
            original.lora_A.requires_grad_(True)
            original.lora_B.requires_grad_(True)
        else:
            assert isinstance(original, nn.Linear)
            setattr(
                attn,
                mod_name,
                LoRALinear(original, cfg.rank, cfg.alpha, cfg.dropout),
            )
        n_replaced += 1

    if not n_replaced:
        raise ValueError("LoRA injection found no target modules")

    trainable = sum(p.numel() for p in stage.parameters() if p.requires_grad)
    total = sum(p.numel() for p in stage.parameters())
    log.info(
        "LoRA applied: %d adapters, trainable params=%d / %d (%.2f%%)",
        n_replaced, trainable, total, 100.0 * trainable / max(total, 1),
    )
    return stage


def trainable_parameters(stage: nn.Module) -> list[nn.Parameter]:
    return [p for p in stage.parameters() if p.requires_grad]


def lora_state_dict(stage: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter weights (for lightweight checkpoint)."""
    return {
        k: v for k, v in stage.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state_dict(stage: nn.Module, adapter_sd: dict[str, torch.Tensor]) -> None:
    """Load only adapter weights; base weights untouched."""
    missing, unexpected = stage.load_state_dict(adapter_sd, strict=False)
    non_lora_missing = [k for k in missing if "lora" not in k]
    if non_lora_missing:
        log.warning("unexpected missing non-lora keys: %s", non_lora_missing)
    lora_unexpected = [k for k in unexpected if "lora" in k]
    if lora_unexpected:
        log.warning("unexpected lora keys: %s", lora_unexpected)
