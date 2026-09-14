"""Same-host boundary transport tests (CUDA cases are hardware-gated)."""
from __future__ import annotations

import pytest
import torch

from meshgpu.transport.local_cuda import boundary_transport_name, move_boundary, probe_cuda_path


def test_cpu_boundary_is_explicit_and_preserves_values() -> None:
    tensor = torch.randn(2, 3, requires_grad=True)
    moved = move_boundary(tensor, torch.device("cpu"), transport="local_cuda")
    assert moved is tensor
    assert boundary_transport_name(torch.device("cpu"), torch.device("cpu")) == "same_device"
    torch.testing.assert_close(moved, tensor)


def test_unknown_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported local boundary transport"):
        move_boundary(torch.ones(1), torch.device("cpu"), transport="tcp")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_path_and_device_copy_preserve_values() -> None:
    source = torch.device("cuda:0")
    destination = torch.device("cuda:1") if torch.cuda.device_count() > 1 else source
    path = probe_cuda_path(source, destination)
    assert path.mode in {"same_device", "peer", "host_fallback"}
    tensor = torch.randn(2, 3, device=source)
    moved = move_boundary(tensor, destination, transport="local_cuda")
    assert moved.device == destination
    torch.testing.assert_close(moved.cpu(), tensor.cpu())
