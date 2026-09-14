"""Fit-first capacity acceptance harness.

The throughput benchmark answers "how fast is this configuration?".  This
module answers the more important first question for MeshGPU: with the exact
same model revision, dtype, prompt and output budget, did an unsharded run
actually fail for lack of CUDA memory while the sharded run completed?

The harness deliberately does not manufacture an OOM from a planner estimate.
It records a real runner result and exposes ``pending_hardware`` when the host
cannot provide at least two CUDA devices.  Callers running destructive OOM
experiments should invoke each runner in a fresh process; a CUDA context that
has hit OOM is not a safe isolation boundary for a later experiment in the
same process.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

CapacityStatus = Literal["success", "cuda_oom", "error", "skipped"]
GateStatus = Literal["passed", "pending_hardware", "failed"]
CapacityRunner = Callable[["CapacityContract"], Any]


@dataclass(frozen=True)
class CapacityContract:
    """Immutable workload identity shared by all compared runners."""

    model_id: str
    model_revision: str
    compute_dtype: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    batch_size: int = 1
    cpu_offload: bool = False
    quantization: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id.strip()
            or not isinstance(self.model_revision, str)
            or not self.model_revision.strip()
        ):
            raise ValueError("model_id and model_revision must not be empty")
        if self.compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported compute_dtype: {self.compute_dtype!r}")
        if isinstance(self.prompt_ids, (str, bytes, bytearray)) or not isinstance(
            self.prompt_ids, Sequence
        ):
            raise TypeError("prompt_ids must be a sequence of token IDs")
        # The contract is frozen and its fingerprint is used to compare
        # independent runner processes.  Normalize list-like input once so a
        # caller cannot mutate the workload identity after construction.
        normalized_prompt_ids = tuple(self.prompt_ids)
        object.__setattr__(self, "prompt_ids", normalized_prompt_ids)
        if not normalized_prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in normalized_prompt_ids
        ):
            raise ValueError("prompt_ids must contain non-negative integer token IDs")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be positive")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be positive")
        if not isinstance(self.cpu_offload, bool):
            raise TypeError("cpu_offload must be a boolean")
        if self.cpu_offload:
            raise ValueError(
                "capacity acceptance requires cpu_offload=False; offload changes the workload"
            )
        if self.quantization is not None and (
            not isinstance(self.quantization, str) or not self.quantization.strip()
        ):
            raise ValueError("quantization must be a non-empty name or None")

    @property
    def context_tokens(self) -> int:
        return len(self.prompt_ids) + self.max_new_tokens

    @property
    def fingerprint(self) -> str:
        payload = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "compute_dtype": self.compute_dtype,
            "prompt_ids": list(self.prompt_ids),
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "cpu_offload": self.cpu_offload,
            "quantization": self.quantization,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:24]


@dataclass
class CapacityRun:
    """One measured runner invocation."""

    name: str
    status: CapacityStatus
    devices: tuple[str, ...]
    elapsed_s: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    output: Any = None
    error: str | None = None
    # Per-device maps are appended after the original fields to keep the
    # positional constructor backward-compatible for callers of the early
    # benchmark API.
    peak_allocated_by_device: dict[str, int] = field(default_factory=dict)
    peak_reserved_by_device: dict[str, int] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def peak_bytes(self) -> int:
        return max(self.peak_allocated_bytes, self.peak_reserved_bytes)


@dataclass
class CapacityGateReport:
    """Decision and evidence for one single-vs-sharded capacity gate."""

    contract: CapacityContract
    status: GateStatus
    single_gpu: CapacityRun
    sharded: CapacityRun
    reference: CapacityRun | None
    correctness_ok: bool | None
    reasons: list[str] = field(default_factory=list)
    atol: float = 1e-5
    rtol: float = 1e-4

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def hardware_ready(self) -> bool:
        shape_ready = (
            len(self.single_gpu.devices) == 1
            and _canonical_device_name(self.single_gpu.devices[0]).startswith("cuda")
            and len(self.sharded.devices) >= 2
            and all(
                _canonical_device_name(device).startswith("cuda")
                for device in self.sharded.devices
            )
            and len({_canonical_device_name(device) for device in self.sharded.devices}) >= 2
        )
        if not shape_ready:
            return False
        try:
            return _physical_cuda_topology_available(
                torch.device(self.single_gpu.devices[0]),
                tuple(torch.device(device) for device in self.sharded.devices),
            )
        except (RuntimeError, TypeError, ValueError):
            return False

    def summary(self) -> str:
        lines = [
            f"status={self.status} contract={self.contract.fingerprint}",
            f"single_gpu={self.single_gpu.status} peak={self.single_gpu.peak_bytes}B "
            f"by_device={self.single_gpu.peak_allocated_by_device}",
            f"sharded={self.sharded.status} peak={self.sharded.peak_bytes}B "
            f"by_device={self.sharded.peak_allocated_by_device}",
        ]
        if self.reference is not None:
            lines.append(f"reference={self.reference.status}")
        if self.correctness_ok is not None:
            lines.append(f"correctness_ok={self.correctness_ok}")
        lines.extend(f"reason={reason}" for reason in self.reasons)
        return "\n".join(lines)


def measure_capacity_run(
    name: str,
    runner: CapacityRunner,
    contract: CapacityContract,
    devices: Sequence[torch.device | str],
) -> CapacityRun:
    """Run one callback and capture real CUDA peak counters.

    The callback receives the same immutable contract as every other runner.
    It may return token IDs, logits, or a nested structure of tensors; the
    result is copied to CPU before the callback's model can mutate it.
    """
    parsed_devices = tuple(torch.device(device) for device in devices)
    if not parsed_devices:
        raise ValueError("at least one device is required")
    device_names = tuple(str(device) for device in parsed_devices)
    cuda_devices = [device for device in parsed_devices if device.type == "cuda"]
    start = time.perf_counter()
    try:
        if cuda_devices:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for a CUDA capacity runner")
            for device in cuda_devices:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
        output = runner(contract)
        if cuda_devices:
            for device in cuda_devices:
                torch.cuda.synchronize(device)
        peak_allocated_by_device = _peak_memory_by_device(cuda_devices, reserved=False)
        peak_reserved_by_device = _peak_memory_by_device(cuda_devices, reserved=True)
        peak_allocated = max(peak_allocated_by_device.values(), default=0)
        peak_reserved = max(peak_reserved_by_device.values(), default=0)
        return CapacityRun(
            name=name,
            status="success",
            devices=device_names,
            elapsed_s=time.perf_counter() - start,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            peak_allocated_by_device=peak_allocated_by_device,
            peak_reserved_by_device=peak_reserved_by_device,
            output=_snapshot_output(output),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        status: CapacityStatus = "cuda_oom" if _is_cuda_oom(exc, cuda_devices) else "error"
        peak_allocated_by_device = _peak_memory_by_device(cuda_devices, reserved=False)
        peak_reserved_by_device = _peak_memory_by_device(cuda_devices, reserved=True)
        if cuda_devices and torch.cuda.is_available():
            # Release cached blocks, but do not imply that the CUDA context is
            # safe for another destructive OOM experiment.  The caller's
            # process-isolation policy remains authoritative.
            for device in cuda_devices:
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
        return CapacityRun(
            name=name,
            status=status,
            devices=device_names,
            elapsed_s=elapsed,
            peak_allocated_bytes=max(peak_allocated_by_device.values(), default=0),
            peak_reserved_bytes=max(peak_reserved_by_device.values(), default=0),
            peak_allocated_by_device=peak_allocated_by_device,
            peak_reserved_by_device=peak_reserved_by_device,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_capacity_gate(
    contract: CapacityContract,
    *,
    single_runner: CapacityRunner,
    sharded_runner: CapacityRunner,
    reference_runner: CapacityRunner | None,
    single_device: torch.device | str,
    sharded_devices: Sequence[torch.device | str],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> CapacityGateReport:
    """Evaluate the genuine one-GPU-OOM / multi-GPU-success acceptance gate."""
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float))
        or not math.isfinite(float(atol))
        or atol < 0
        or isinstance(rtol, bool)
        or not isinstance(rtol, (int, float))
        or not math.isfinite(float(rtol))
        or rtol < 0
    ):
        raise ValueError("atol and rtol must be finite and non-negative")

    try:
        requested_single = torch.device(single_device)
        requested_sharded = tuple(torch.device(device) for device in sharded_devices)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid capacity device: {exc}") from exc
    if not requested_sharded:
        raise ValueError("sharded_devices must contain at least one device")

    # A deliberate OOM is not a safe way to discover that the host has no
    # suitable topology.  Refuse to invoke user runners in that situation;
    # this also prevents a one-GPU host from poisoning its only CUDA context.
    requested_cuda = (
        requested_single.type == "cuda"
        or any(device.type == "cuda" for device in requested_sharded)
    )
    if requested_cuda and not _physical_cuda_topology_available(
        requested_single,
        requested_sharded,
    ):
        reason = (
            "requires one valid physical CUDA device for the baseline and at least "
            "two distinct valid CUDA devices for the sharded run"
        )
        return CapacityGateReport(
            contract=contract,
            status="pending_hardware",
            single_gpu=CapacityRun(
                name="single_gpu",
                status="skipped",
                devices=(str(requested_single),),
                error=reason,
            ),
            sharded=CapacityRun(
                name="sharded",
                status="skipped",
                devices=tuple(str(device) for device in requested_sharded),
                error=reason,
            ),
            reference=None,
            correctness_ok=None,
            reasons=[reason],
            atol=atol,
            rtol=rtol,
        )

    single = measure_capacity_run("single_gpu", single_runner, contract, [single_device])
    sharded = measure_capacity_run("sharded", sharded_runner, contract, sharded_devices)
    reference = (
        measure_capacity_run("reference", reference_runner, contract, ["cpu"])
        if reference_runner is not None
        else None
    )
    report = CapacityGateReport(
        contract=contract,
        status="failed",
        single_gpu=single,
        sharded=sharded,
        reference=reference,
        correctness_ok=None,
        atol=atol,
        rtol=rtol,
    )

    if not report.hardware_ready:
        report.status = "pending_hardware"
        report.reasons.append(
            "requires one physical CUDA device for the baseline and at least two "
            "distinct CUDA devices for the sharded run"
        )

    if single.status != "cuda_oom":
        report.reasons.append(
            "single-GPU baseline did not produce a real CUDA out-of-memory failure"
        )
    if not sharded.succeeded:
        report.reasons.append(
            f"sharded run did not complete ({sharded.status}"
            f"{': ' + sharded.error if sharded.error else ''})"
        )
    if reference is None:
        report.reasons.append("a correctness reference runner is required for acceptance")
    elif not reference.succeeded:
        report.reasons.append(
            f"reference run did not complete ({reference.status}"
            f"{': ' + reference.error if reference.error else ''})"
        )
    elif sharded.succeeded:
        report.correctness_ok = outputs_close(
            sharded.output,
            reference.output,
            atol=atol,
            rtol=rtol,
        )
        if not report.correctness_ok:
            report.reasons.append("sharded output differs from the reference output")

    if report.hardware_ready and not report.reasons:
        report.status = "passed"
    elif report.status != "pending_hardware":
        report.status = "failed"
    return report


def outputs_close(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    """Compare token IDs/logits without accepting a changed structure."""
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            return False
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False
        if actual.is_floating_point() or actual.is_complex():
            return bool(torch.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False))
        return bool(torch.equal(actual, expected))
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            return False
        if set(actual) != set(expected):
            return False
        return all(
            outputs_close(actual[key], expected[key], atol=atol, rtol=rtol)
            for key in actual
        )
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes, bytearray)):
            return False
        return len(actual) == len(expected) and all(
            outputs_close(left, right, atol=atol, rtol=rtol)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _snapshot_output(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_output(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_output(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_output(item) for item in value]
    return value


def _is_cuda_oom(exc: BaseException, devices: Sequence[torch.device]) -> bool:
    if not any(device.type == "cuda" for device in devices):
        return False
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _canonical_device_name(device: str) -> str:
    """Normalize ``cuda`` and ``cuda:0`` before checking physical identity."""
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return str(device)
    if parsed.type == "cuda" and parsed.index is None:
        try:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except (RuntimeError, AssertionError):
            index = 0
        return f"cuda:{index}"
    return str(parsed)


def _physical_cuda_topology_available(
    single_device: torch.device,
    sharded_devices: Sequence[torch.device],
) -> bool:
    """Check the requested CUDA ordinals against the live physical runtime."""
    if single_device.type != "cuda" or len(sharded_devices) < 2:
        return False
    if any(device.type != "cuda" for device in sharded_devices):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        count = int(torch.cuda.device_count())
        single_index = (
            int(single_device.index)
            if single_device.index is not None
            else int(torch.cuda.current_device())
        )
        sharded_indices = [
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
            for device in sharded_devices
        ]
    except (RuntimeError, TypeError, ValueError, AssertionError):
        return False
    if single_index < 0 or single_index >= count:
        return False
    return all(0 <= index < count for index in sharded_indices) and len(
        set(sharded_indices)
    ) >= 2


def _peak_memory_by_device(
    devices: Sequence[torch.device],
    *,
    reserved: bool,
) -> dict[str, int]:
    """Read per-device peak counters, tolerating a poisoned OOM context."""
    result: dict[str, int] = {}
    reader = torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
    for device in devices:
        try:
            result[_canonical_device_name(str(device))] = int(reader(device))
        except (RuntimeError, AssertionError):
            # Some CUDA versions reject a memory-stat query after an OOM.  The
            # run remains classified correctly; missing evidence is reported
            # as an empty per-device entry instead of masking the real error.
            continue
    return result
