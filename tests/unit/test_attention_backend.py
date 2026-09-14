"""SDPA dispatch must be reported rather than assumed."""
from __future__ import annotations

import pytest
import torch

from meshgpu.backends.portable.attention import verify_sdpa_backend


def test_cpu_sdpa_execution_is_verified_as_math_or_known_kernel() -> None:
    report = verify_sdpa_backend((1, 2, 4, 8), device="cpu")
    assert report.verified
    assert report.selected in {"math", "efficient", "flash"}
    assert report.device == "cpu"


def test_invalid_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        verify_sdpa_backend((1, 2, 0, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_report_contains_actual_dispatch_label() -> None:
    report = verify_sdpa_backend(
        (1, 4, 16, 32),
        device=torch.device("cuda:0"),
        dtype=torch.float16,
    )
    assert report.verified
    assert report.selected in {"math", "efficient", "flash"}
