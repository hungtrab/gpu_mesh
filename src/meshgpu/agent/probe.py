"""Probe local hardware and software capabilities."""
from __future__ import annotations

import platform
import socket
import sys

import psutil  # type: ignore[import-untyped]

from meshgpu.agent.provider import ProviderKind, SupportLevel, check_eligibility
from meshgpu.protocol.messages import (
    Backend,
    CapabilityReport,
    GpuInfo,
    WorkerMode,
)


def probe_gpus() -> list[GpuInfo]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            free_bytes, total_bytes = mem

            # Smoke-test supported dtypes
            dtypes: list[str] = ["float32"]
            try:
                t = torch.zeros(1, dtype=torch.float16, device=f"cuda:{i}")
                del t
                dtypes.append("float16")
            except Exception:
                pass
            try:
                t = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{i}")
                del t
                dtypes.append("bfloat16")
            except Exception:
                pass

            gpus.append(
                GpuInfo(
                    device_index=i,
                    name=props.name,
                    total_vram_bytes=total_bytes,
                    free_vram_bytes=free_bytes,
                    compute_capability=(props.major, props.minor),
                    supported_dtypes=dtypes,
                )
            )
        return gpus
    except ImportError:
        return []


def probe_torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not_installed"


def probe_cuda_driver() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda or None
    except ImportError:
        pass
    return None


def probe_host_memory() -> tuple[int, int]:
    """Return (total_ram_bytes, available_ram_bytes)."""
    mem = psutil.virtual_memory()
    return mem.total, mem.available


def probe_scratch_disk(path: str = "/tmp") -> int:
    usage = psutil.disk_usage(path)
    return usage.free


def build_capability_report(
    worker_id: str | None = None,
    provider: str = "self_managed",
    pinned_memory_limit_bytes: int = 256 * 1024 * 1024,
    session_remaining_s: float | None = None,
) -> CapabilityReport:
    wid = worker_id or socket.gethostname()
    gpus = probe_gpus()
    total_ram, _ = probe_host_memory()
    scratch = probe_scratch_disk()

    backends: list[Backend] = [Backend.PORTABLE_PIPELINE]
    try:
        import torch
        if torch.cuda.is_available():
            backends.append(Backend.NATIVE)
    except ImportError:
        pass

    # Capability must reflect the provider matrix.  Advertising a managed
    # Colab notebook as a distributed worker would make the scheduler admit a
    # job that the provider policy explicitly disallows.
    try:
        provider_kind = ProviderKind(provider)
    except ValueError:
        provider_kind = ProviderKind.UNKNOWN
    eligibility = check_eligibility(provider_kind)
    modes: list[WorkerMode] = []
    if eligibility.client != SupportLevel.UNSUPPORTED:
        modes.append(WorkerMode.CLIENT)
    if eligibility.single_runtime_job != SupportLevel.UNSUPPORTED:
        modes.append(WorkerMode.SINGLE_RUNTIME_JOB)
    if eligibility.distributed_worker not in {
        SupportLevel.UNSUPPORTED,
        SupportLevel.CLIENT_ONLY,
    }:
        modes.append(WorkerMode.DISTRIBUTED_WORKER)

    adapters = ["llama_dense_v1"]
    # Report only adapters that can actually be constructed in this runtime.
    # The base package keeps Transformers optional, so a worker without the
    # extra must not advertise Qwen3 just because the controller knows it.
    try:
        import transformers

        if hasattr(transformers, "Qwen3Config"):
            adapters.append("qwen3_hf_v1")
    except ImportError:
        pass

    return CapabilityReport(
        worker_id=wid,
        provider=provider,
        os=platform.system(),
        arch=platform.machine(),
        python_version=sys.version.split()[0],
        torch_version=probe_torch_version(),
        cuda_driver_version=probe_cuda_driver(),
        gpus=gpus,
        host_ram_bytes=total_ram,
        pinned_memory_limit_bytes=pinned_memory_limit_bytes,
        scratch_disk_bytes=scratch,
        supported_modes=modes,
        available_backends=backends,
        available_model_adapters=adapters,
        session_remaining_s=session_remaining_s,
    )
