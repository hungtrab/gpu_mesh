"""Runtime SDPA backend verification.

``scaled_dot_product_attention`` is a dispatch API, not a promise that a
fused CUDA kernel was selected.  The profiler-based helper below records the
actual operator family for a representative shape and returns ``math`` or
``unknown`` when a fused path was not observed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionBackendReport:
    requested: str
    selected: str
    verified: bool
    fused: bool
    device: str
    dtype: str
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    elapsed_ms: float
    warning: str | None = None


def verify_sdpa_backend(
    query_shape: tuple[int, int, int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    causal: bool = True,
    requested: str = "sdpa",
) -> AttentionBackendReport:
    """Execute representative SDPA and identify the operator from a trace."""
    if requested != "sdpa":
        raise ValueError("verify_sdpa_backend only verifies requested='sdpa'")
    if len(query_shape) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in query_shape
    ):
        raise ValueError("query_shape must be [batch, heads, query, head_dim]")
    device = torch.device(device)
    batch, heads, query_len, head_dim = query_shape
    key_len = query_len
    key_shape = (batch, heads, key_len, head_dim)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    started = time.perf_counter()
    try:
        query = torch.randn(query_shape, device=device, dtype=dtype)
        key = torch.randn(key_shape, device=device, dtype=dtype)
        value = torch.randn_like(key)
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=causal,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        return AttentionBackendReport(
            requested=requested,
            selected="unavailable",
            verified=False,
            fused=False,
            device=str(device),
            dtype=str(dtype),
            query_shape=query_shape,
            key_shape=key_shape,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            warning=f"SDPA execution failed: {exc}",
        )

    selected = _kernel_from_events(prof.key_averages())
    fused = selected in {"flash", "efficient"}
    warning = None
    if selected == "unknown":
        warning = "SDPA ran but the profiler did not identify its kernel"
    elif not fused:
        warning = "SDPA selected the math backend; no fused kernel was observed"
    return AttentionBackendReport(
        requested=requested,
        selected=selected,
        verified=selected != "unknown",
        fused=fused,
        device=str(device),
        dtype=str(dtype),
        query_shape=query_shape,
        key_shape=key_shape,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        warning=warning,
    )


def _kernel_from_events(events) -> str:
    names = [str(event.key).lower() for event in events]
    if any("scaled_dot_product_flash_attention" in name for name in names):
        return "flash"
    if any(
        marker in name
        for name in names
        for marker in (
            "scaled_dot_product_efficient_attention",
            "efficient_attention_backward",
        )
    ):
        return "efficient"
    if any("scaled_dot_product_attention_math" in name for name in names):
        return "math"
    return "unknown"
