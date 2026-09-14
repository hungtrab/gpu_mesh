"""Unit tests for memory planner."""

from meshgpu.planner.memory import (
    GiB,
    GpuBudget,
    TensorPeak,
    check_feasibility,
    estimate_kv_cache_bytes,
    estimate_transfer_time_s,
    estimate_weight_bytes,
)


def make_budget(total_gib=16, free_gib=14, fraction=0.85, reserve_min_gib=1):
    return GpuBudget(
        device_index=0,
        total_vram_bytes=int(total_gib * GiB),
        free_vram_at_admission=int(free_gib * GiB),
        user_budget_fraction=fraction,
        reserve_min_bytes=int(reserve_min_gib * GiB),
    )


def test_usable_bytes_respects_reserve():
    b = make_budget(total_gib=16, free_gib=14, fraction=0.85)
    assert b.usable_bytes <= int(16 * GiB * 0.85)
    assert b.usable_bytes > 0


def test_feasibility_passes_when_fits():
    budget = make_budget(total_gib=16, free_gib=14)
    peak = TensorPeak(weight_bytes=int(4 * GiB), cuda_context_bytes=0)
    result = check_feasibility([budget], [peak])
    assert result.feasible


def test_feasibility_fails_when_exceeds():
    budget = make_budget(total_gib=8, free_gib=8)
    peak = TensorPeak(weight_bytes=int(16 * GiB), cuda_context_bytes=0)
    result = check_feasibility([budget], [peak])
    assert not result.feasible
    assert "exceeds" in result.reason


def test_weight_bytes_fp16():
    assert estimate_weight_bytes(7_000_000_000, 2) == 14_000_000_000


def test_kv_cache_bytes():
    # batch=1, 8192 tokens, 32 layers, 8 kv_heads, head_dim=128, fp16 (2 bytes)
    b = estimate_kv_cache_bytes(1, 8192, 32, 8, 128, 2)
    # 2 * 1 * 8192 * 32 * 8 * 128 * 2 = 1073741824 = 1 GiB
    assert b == GiB


def test_transfer_time():
    # 32 MiB at 1 Gbit/s = 32*1024*1024 / (1e9/8) ≈ 0.268s
    t = estimate_transfer_time_s(32 * 1024 * 1024, 1000.0)
    assert abs(t - 0.268) < 0.01


def test_transfer_time_zero_goodput():
    assert estimate_transfer_time_s(1024, 0) == float("inf")
