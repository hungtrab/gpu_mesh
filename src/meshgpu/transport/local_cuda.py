"""Same-host CUDA boundary transport.

The portable pipeline historically materialized every stage boundary on CPU.
This module makes that choice explicit: ``local_cuda`` keeps boundary tensors
device-resident and uses a peer copy when CUDA reports peer access.  When peer
access is unavailable, the fallback is an intentionally visible host staging
copy.  The caller can inspect the selected path and must not advertise it as
zero-copy without a hardware probe.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CudaPath:
    """Result of probing one same-host source/destination pair."""

    source: torch.device
    destination: torch.device
    mode: str  # same_device, peer, host_fallback, non_cuda
    peer_access: bool = False


def probe_cuda_path(source: torch.device, destination: torch.device) -> CudaPath:
    """Determine whether a boundary can use CUDA peer access."""
    source = torch.device(source)
    destination = torch.device(destination)
    # ``torch.device("cuda")`` means the current device, not necessarily
    # cuda:0.  Resolve it before comparing/probing so a process launched with
    # CUDA_VISIBLE_DEVICES or a non-zero current device cannot silently route
    # a boundary to the wrong GPU.
    source = _resolve_cuda_device(source)
    destination = _resolve_cuda_device(destination)
    if source == destination:
        return CudaPath(source, destination, "same_device", peer_access=True)
    if source.type != "cuda" or destination.type != "cuda":
        return CudaPath(source, destination, "non_cuda", peer_access=False)
    if not torch.cuda.is_available():
        return CudaPath(source, destination, "host_fallback", peer_access=False)
    src_index = 0 if source.index is None else source.index
    dst_index = 0 if destination.index is None else destination.index
    try:
        peer = bool(torch.cuda.can_device_access_peer(src_index, dst_index))
    except (RuntimeError, TypeError, AttributeError):
        peer = False
    return CudaPath(
        source,
        destination,
        "peer" if peer else "host_fallback",
        peer_access=peer,
    )


def move_boundary(
    tensor: torch.Tensor,
    destination: torch.device,
    *,
    transport: str = "cpu",
) -> torch.Tensor:
    """Move a stage boundary according to the selected transport policy.

    ``cpu`` preserves the original portable behavior.  ``local_cuda`` uses a
    direct ``Tensor.to(cuda_device)`` for a peer-capable pair and an explicit
    CPU staging hop otherwise.  The latter is deliberately synchronous: the
    first correctness path should not hide lifetime hazards behind streams.
    """
    destination = _resolve_cuda_device(torch.device(destination))
    if transport not in {"cpu", "local_cuda"}:
        raise ValueError(f"unsupported local boundary transport: {transport!r}")
    if tensor.device == destination:
        return tensor
    if transport == "cpu":
        # Make the legacy CPU policy real even when both stages happen to be
        # CUDA stages: force a host staging hop before loading the destination
        # module.  The old direct ``tensor.to(destination)`` let PyTorch choose
        # a peer copy, contradicting the transport report and making capacity/
        # bandwidth measurements impossible to interpret.
        return tensor.detach().to("cpu").to(destination)

    path = probe_cuda_path(tensor.device, destination)
    if path.mode == "peer":
        return tensor.to(destination, non_blocking=True)
    if path.mode == "host_fallback":
        # ``detach`` is safe at a pipeline boundary: StageWorker creates a
        # fresh leaf for training backward, while inference is no-grad.
        return tensor.detach().to("cpu").to(destination)
    return tensor.to(destination)


def boundary_transport_name(source: torch.device, destination: torch.device) -> str:
    """Return a stable report label for the path that would be used."""
    return probe_cuda_path(source, destination).mode


def _resolve_cuda_device(device: torch.device) -> torch.device:
    """Resolve an unindexed CUDA device without touching CUDA on CPU paths."""
    if device.type != "cuda" or device.index is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    # Keep the original semantic for an unavailable CUDA runtime; the actual
    # tensor transfer will raise the useful PyTorch error at the call site.
    return device
