"""Memory budget and placement feasibility checks."""
from __future__ import annotations

import math
from dataclasses import dataclass

GiB = 1024 ** 3
GB = 1_000_000_000


@dataclass
class GpuBudget:
    device_index: int
    total_vram_bytes: int
    free_vram_at_admission: int
    user_budget_fraction: float = 0.85
    reserve_min_bytes: int = GiB  # 1 GiB floor

    def __post_init__(self) -> None:
        for name, value in (
            ("device_index", self.device_index),
            ("total_vram_bytes", self.total_vram_bytes),
            ("free_vram_at_admission", self.free_vram_at_admission),
            ("reserve_min_bytes", self.reserve_min_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if self.total_vram_bytes < 0 or self.free_vram_at_admission < 0:
            raise ValueError("VRAM values must be non-negative")
        if self.free_vram_at_admission > self.total_vram_bytes:
            raise ValueError("free VRAM cannot exceed total VRAM")
        if (
            isinstance(self.user_budget_fraction, bool)
            or not isinstance(self.user_budget_fraction, (int, float))
            or not math.isfinite(float(self.user_budget_fraction))
            or not 0 < self.user_budget_fraction <= 1
        ):
            raise ValueError("user_budget_fraction must be in (0, 1]")
        if self.reserve_min_bytes < 0:
            raise ValueError("reserve_min_bytes must be non-negative")

    @property
    def reserve_bytes(self) -> int:
        return max(self.reserve_min_bytes, int(self.total_vram_bytes * 0.10))

    @property
    def usable_bytes(self) -> int:
        user_cap = int(self.total_vram_bytes * self.user_budget_fraction)
        return min(user_cap, self.free_vram_at_admission - self.reserve_bytes)


@dataclass
class TensorPeak:
    """All sources of VRAM consumption for one GPU during one phase."""
    weight_bytes: int = 0
    kv_cache_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    comm_buffer_bytes: int = 0
    workspace_bytes: int = 0
    cuda_context_bytes: int = 128 * 1024 * 1024  # ~128 MiB typical
    # Adapter parameters can use a different dtype from the frozen base model
    # (the built-in LoRA recipe deliberately keeps them in fp32).  Keep this
    # separate from ``weight_bytes`` so reports cannot hide that difference.
    # It is appended to preserve the positional constructor of the original
    # planner API for callers that used ``TensorPeak(..., cuda_context_bytes)``.
    adapter_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total(self) -> int:
        return (
            self.weight_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.comm_buffer_bytes
            + self.workspace_bytes
            + self.cuda_context_bytes
            + self.adapter_bytes
        )


def estimate_weight_bytes(param_count: int, dtype_bytes: int) -> int:
    if (
        isinstance(param_count, bool)
        or not isinstance(param_count, int)
        or isinstance(dtype_bytes, bool)
        or not isinstance(dtype_bytes, int)
        or param_count < 0
        or dtype_bytes < 1
    ):
        raise ValueError("param_count must be non-negative and dtype_bytes positive")
    return param_count * dtype_bytes


def estimate_kv_cache_bytes(
    batch_size: int,
    cached_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    values = (
        batch_size,
        cached_tokens,
        num_layers,
        num_kv_heads,
        head_dim,
        dtype_bytes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("KV dimensions and dtype_bytes must be integers")
    if any(
        value < 0
        for value in (batch_size, cached_tokens, num_layers, num_kv_heads, head_dim)
    ) or dtype_bytes < 1:
        raise ValueError("KV dimensions must be non-negative and dtype_bytes positive")
    return 2 * batch_size * cached_tokens * num_layers * num_kv_heads * head_dim * dtype_bytes


def estimate_activation_boundary_bytes(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype_bytes: int,
) -> int:
    values = (batch_size, seq_len, hidden_dim, dtype_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("activation dimensions and dtype_bytes must be integers")
    if any(value < 0 for value in (batch_size, seq_len, hidden_dim)) or dtype_bytes < 1:
        raise ValueError("activation dimensions must be non-negative and dtype_bytes positive")
    return batch_size * seq_len * hidden_dim * dtype_bytes


def estimate_adam_optimizer_bytes(param_count: int) -> int:
    # FP32 master weight (4) + FP32 momentum (4) + FP32 variance (4)
    if isinstance(param_count, bool) or not isinstance(param_count, int):
        raise TypeError("param_count must be an integer")
    if param_count < 0:
        raise ValueError("param_count must be non-negative")
    return param_count * (4 + 4 + 4)


@dataclass
class FeasibilityResult:
    feasible: bool
    reason: str
    peak_per_gpu: list[int]
    usable_per_gpu: list[int]
    margin_per_gpu: list[int]

    def report(self) -> str:
        lines = [f"feasible: {self.feasible}", f"reason: {self.reason}"]
        for i, (peak, usable, margin) in enumerate(
            zip(self.peak_per_gpu, self.usable_per_gpu, self.margin_per_gpu)
        ):
            lines.append(
                f"  gpu{i}: peak={peak/GiB:.2f} GiB  usable={usable/GiB:.2f} GiB  "
                f"margin={margin/GiB:+.2f} GiB"
            )
        return "\n".join(lines)


def check_feasibility(
    budgets: list[GpuBudget],
    peaks: list[TensorPeak],
) -> FeasibilityResult:
    if not budgets:
        raise ValueError("at least one GPU budget is required")
    if len(budgets) != len(peaks):
        raise ValueError("budgets and peaks must have the same length")

    peak_vals = [p.total for p in peaks]
    usable_vals = [b.usable_bytes for b in budgets]
    margin_vals = [u - p for u, p in zip(usable_vals, peak_vals)]

    for i, margin in enumerate(margin_vals):
        if margin < 0:
            return FeasibilityResult(
                feasible=False,
                reason=(
                    f"gpu{i} peak {peak_vals[i] / GiB:.2f} GiB exceeds usable "
                    f"{usable_vals[i] / GiB:.2f} GiB"
                ),
                peak_per_gpu=peak_vals,
                usable_per_gpu=usable_vals,
                margin_per_gpu=margin_vals,
            )

    return FeasibilityResult(
        feasible=True,
        reason="all gpus within budget",
        peak_per_gpu=peak_vals,
        usable_per_gpu=usable_vals,
        margin_per_gpu=margin_vals,
    )


def estimate_transfer_time_s(bytes_: int, goodput_mbit_s: float) -> float:
    if isinstance(bytes_, bool) or not isinstance(bytes_, int) or bytes_ < 0:
        raise ValueError("bytes_ must be a non-negative integer")
    if (
        isinstance(goodput_mbit_s, bool)
        or not isinstance(goodput_mbit_s, (int, float))
        or not math.isfinite(float(goodput_mbit_s))
        or goodput_mbit_s <= 0
    ):
        return float("inf")
    return bytes_ / (goodput_mbit_s * 1e6 / 8)
