"""Measured memory profiles and profile-backed placement selection.

Static parameter arithmetic is useful for a first rejection, but it cannot
know allocator fragmentation, attention workspaces or the selected CUDA
kernel.  This module records those observations with an explicit workload
signature and lets admission require an exact profile instead of guessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import torch

from meshgpu.planner.memory import GpuBudget, TensorPeak, check_feasibility
from meshgpu.planner.placement import (
    InferenceWorkload,
    ModelSpec,
    StageAssignment,
    TrainingWorkload,
    WorkerSpec,
    plan_inference,
    plan_training,
)

MemoryMode = Literal["inference", "ttt"]


@dataclass(frozen=True)
class WorkloadSignature:
    """All inputs that can materially change a stage peak."""

    model_id: str
    mode: MemoryMode
    compute_dtype: str
    batch_size: int
    prompt_tokens: int = 0
    max_new_tokens: int = 0
    sequence_length: int = 0
    gradient_accumulation_steps: int = 1
    lora_rank: int = 0
    optimizer: str = "none"
    attention_backend: str = "sdpa"
    checkpointing: bool = False
    max_concurrent: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"inference", "ttt"}:
            raise ValueError(f"unsupported memory profile mode: {self.mode!r}")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.compute_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported compute_dtype: {self.compute_dtype!r}")
        if self.attention_backend not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                f"unsupported attention_backend: {self.attention_backend!r}"
            )
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must be a non-empty string")
        if self.mode == "ttt" and self.optimizer.lower() not in {"none", "adam", "adamw"}:
            raise ValueError(
                "TTT memory profiles support optimizer 'adam' or 'adamw'"
            )
        for name in (
            "batch_size",
            "prompt_tokens",
            "max_new_tokens",
            "max_concurrent",
            "sequence_length",
            "gradient_accumulation_steps",
            "lora_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class StageMemoryMeasurement:
    """One exact stage/range observation from an isolated profiling run."""

    stage_id: int
    worker_id: str
    device_index: int
    layer_start: int
    layer_end: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    weight_bytes: int = 0
    adapter_bytes: int = 0
    kv_cache_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    logits_bytes: int = 0
    temporary_bytes: int = 0
    communication_bytes: int = 0
    allocator_margin_bytes: int = 0
    phase: str = "unknown"
    measured_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name.endswith("_bytes") and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.stage_id < 0 or self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("invalid stage or layer range in memory measurement")

    @property
    def component_bytes(self) -> int:
        return sum(
            getattr(self, name)
            for name in (
                "weight_bytes",
                "adapter_bytes",
                "kv_cache_bytes",
                "activation_bytes",
                "gradient_bytes",
                "optimizer_bytes",
                "logits_bytes",
                "temporary_bytes",
                "communication_bytes",
                "allocator_margin_bytes",
            )
        )

    @property
    def peak_bytes(self) -> int:
        """Conservative peak: measured reserved memory wins over components."""
        return max(self.peak_allocated_bytes, self.peak_reserved_bytes, self.component_bytes)


@dataclass
class MemoryProfile:
    signature: WorkloadSignature
    hardware_profile: str
    software_profile: str
    measurements: list[StageMemoryMeasurement] = field(default_factory=list)

    def add(self, measurement: StageMemoryMeasurement) -> None:
        existing = [
            item
            for item in self.measurements
            if not (
                item.stage_id == measurement.stage_id
                and item.worker_id == measurement.worker_id
                and item.layer_start == measurement.layer_start
                and item.layer_end == measurement.layer_end
                and item.phase == measurement.phase
            )
        ]
        existing.append(measurement)
        self.measurements = existing

    def find(
        self,
        stage_id: int,
        layer_start: int,
        layer_end: int,
        *,
        worker_id: str | None = None,
    ) -> StageMemoryMeasurement | None:
        matches = [
            item
            for item in self.measurements
            if item.stage_id == stage_id
            and item.layer_start == layer_start
            and item.layer_end == layer_end
            and (worker_id is None or item.worker_id == worker_id)
        ]
        # A complete profile may contain prefill/decode or forward/backward
        # observations for the same stage.  Admission must use the largest
        # exact peak, not whichever phase happened to be written last.
        return max(matches, key=lambda item: item.peak_bytes, default=None)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": asdict(self.signature),
            "hardware_profile": self.hardware_profile,
            "software_profile": self.software_profile,
            "measurements": [asdict(item) for item in self.measurements],
        }
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
            with temporary.open("rb") as written:
                os.fsync(written.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> MemoryProfile:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("memory profile must be a JSON object")
        signature = WorkloadSignature(**payload["signature"])
        measurements = [StageMemoryMeasurement(**item) for item in payload["measurements"]]
        return cls(
            signature=signature,
            hardware_profile=str(payload["hardware_profile"]),
            software_profile=str(payload["software_profile"]),
            measurements=measurements,
        )


def profile_cuda_phase(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    synchronize: bool = True,
) -> tuple[Any, int, int]:
    """Run ``fn`` and return ``(result, peak_allocated, peak_reserved)``.

    CUDA peak counters are reset immediately before the phase.  The function
    intentionally propagates CUDA OOM so the capacity harness can classify it
    as a real baseline failure rather than converting it into a fake estimate.
    CPU callers get zero counters and still exercise the phase.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return fn(), 0, 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    if synchronize:
        torch.cuda.synchronize(device)
    return (
        result,
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


@dataclass
class MeasuredPlacementReport:
    feasible: bool
    reason: str
    assignments: list[StageAssignment]
    bottleneck_stage: int | None
    profile_key: str
    warnings: list[str] = field(default_factory=list)
    safe_for_admission: bool = True

    def summary(self) -> str:
        lines = [
            f"feasible={self.feasible} safe_for_admission={self.safe_for_admission} "
            f"reason={self.reason!r} profile={self.profile_key}",
        ]
        for assignment in self.assignments:
            margin = assignment.budget.usable_bytes - assignment.peak.total
            lines.append(
                f"  stage{assignment.stage_id} layers=[{assignment.layer_start},"
                f"{assignment.layer_end}) peak={assignment.peak.total / (1024 ** 3):.2f}GiB "
                f"margin={margin / (1024 ** 3):+.2f}GiB"
            )
        return "\n".join(lines)

    def as_preflight(self):
        """Convert the report to the inference server's explicit decision type.

        A static estimate or a profile with incomplete identity is useful for
        diagnostics, but it must not silently become a production admission.
        ``safe_for_admission`` is therefore part of the conversion rather
        than an informal warning that callers could overlook.
        """
        from meshgpu.inference.admission import MemoryPreflightResult

        details = {
            "profile_key": self.profile_key,
            "safe_for_admission": self.safe_for_admission,
            "bottleneck_stage": self.bottleneck_stage,
            "warnings": list(self.warnings),
        }
        return MemoryPreflightResult(
            feasible=self.feasible and self.safe_for_admission,
            reason=self.reason,
            details=details,
        )


def plan_from_profile(
    workers: list[WorkerSpec],
    model: ModelSpec,
    profile: MemoryProfile,
    *,
    require_exact: bool = True,
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int = 1024 ** 3,
) -> MeasuredPlacementReport:
    """Choose the best contiguous split from exact stage measurements.

    All candidate cuts are considered.  A missing measurement is a hard
    rejection when ``require_exact`` is true; this is the safe default for
    production admission.  ``require_exact=False`` computes a clearly marked
    static preview for UI/planning only; it is never safe for admission.
    """
    if not workers:
        raise ValueError("workers must not be empty")
    if len(workers) > model.num_layers:
        raise ValueError("each stage must own at least one layer")
    budgets = [
        GpuBudget(
            device_index=worker.device_index,
            total_vram_bytes=worker.total_vram_bytes,
            free_vram_at_admission=worker.free_vram_bytes,
            user_budget_fraction=max_vram_fraction,
            reserve_min_bytes=reserve_min_bytes,
        )
        for worker in workers
    ]

    best: tuple[float, list[StageAssignment], bool, str, list[str]] | None = None
    missing_for_all: list[str] = []
    for ranges in _contiguous_partitions(model.num_layers, len(workers)):
        assignments: list[StageAssignment] = []
        peaks: list[TensorPeak] = []
        missing: list[str] = []
        for stage_id, ((layer_start, layer_end), worker, budget) in enumerate(
            zip(ranges, workers, budgets)
        ):
            measurement = profile.find(
                stage_id,
                layer_start,
                layer_end,
                worker_id=worker.worker_id,
            )
            if measurement is None:
                missing.append(
                    f"stage{stage_id}:{worker.worker_id}[{layer_start},{layer_end})"
                )
                continue
            peak_total = measurement.peak_bytes
            peak = TensorPeak(
                # The measured peak already includes the CUDA context and
                # allocator reservation.  Do not add the static context floor
                # a second time.
                workspace_bytes=peak_total,
                cuda_context_bytes=0,
            )
            peaks.append(peak)
            assignments.append(
                StageAssignment(
                    stage_id=stage_id,
                    worker_id=worker.worker_id,
                    device_index=worker.device_index,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    has_embedding=stage_id == 0,
                    has_lm_head=stage_id == len(workers) - 1,
                    peak=peak,
                    budget=budget,
                )
            )
        if missing:
            missing_for_all.extend(missing)
            if require_exact:
                continue
            # The static preview is produced once below.  Do not mix an
            # incomplete candidate with exact candidates.
            continue
        feasibility = check_feasibility(budgets, peaks)
        margins = [budget.usable_bytes - peak.total for budget, peak in zip(budgets, peaks)]
        score = float(min(margins))
        candidate: tuple[float, list[StageAssignment], bool, str, list[str]] = (
            score,
            assignments,
            feasibility.feasible,
            feasibility.reason,
            [],
        )
        if best is None or score > best[0]:
            best = candidate

    if best is None:
        if not require_exact:
            static = _static_preview(workers, model, profile.signature)
            if static is not None:
                warning = (
                    "static estimate only; exact stage memory measurements are missing; "
                    "not safe for admission"
                )
                return MeasuredPlacementReport(
                    feasible=static.feasible,
                    reason=static.reason,
                    assignments=static.assignments,
                    bottleneck_stage=static.bottleneck_stage,
                    profile_key=profile.signature.key,
                    warnings=[warning],
                    safe_for_admission=False,
                )
        warning = "missing exact stage memory measurements"
        if not require_exact:
            warning += "; static preview was unavailable for this workload"
        missing_suffix = (
            ": " + ", ".join(sorted(set(missing_for_all)))
            if missing_for_all
            else ""
        )
        return MeasuredPlacementReport(
            feasible=False,
            reason=warning + missing_suffix,
            assignments=[],
            bottleneck_stage=None,
            profile_key=profile.signature.key,
            warnings=[warning],
            safe_for_admission=False,
        )

    _, assignments, feasible, reason, warnings = best
    bottleneck = None
    if assignments:
        bottleneck = min(
            assignments,
            key=lambda item: item.budget.usable_bytes - item.peak.total,
        ).stage_id
    if profile.hardware_profile == "unknown" or profile.software_profile == "unknown":
        warnings.append("profile hardware/software identity is incomplete")
    safe_for_admission = feasible and not warnings
    return MeasuredPlacementReport(
        feasible=feasible,
        reason=reason,
        assignments=assignments,
        bottleneck_stage=bottleneck,
        profile_key=profile.signature.key,
        warnings=warnings,
        safe_for_admission=safe_for_admission,
    )


def _static_preview(
    workers: list[WorkerSpec],
    model: ModelSpec,
    signature: WorkloadSignature,
):
    """Build a non-admission placement report from the existing cost model."""
    # A profile is keyed by the requested/observed attention backend.  Use it
    # for the static diagnostic even when the caller's bare ModelSpec retained
    # its default backend; exact measured placement never uses this estimate.
    if model.attention_implementation != signature.attention_backend:
        model = replace(model, attention_implementation=signature.attention_backend)
    if signature.mode == "inference":
        if signature.prompt_tokens < 1 or signature.max_new_tokens < 1:
            return None
        return plan_inference(
            workers,
            model,
            InferenceWorkload(
                batch_size=signature.batch_size,
                max_prompt_tokens=signature.prompt_tokens,
                max_new_tokens=signature.max_new_tokens,
                max_concurrent=signature.max_concurrent,
            ),
        )
    sequence_length = signature.sequence_length or (
        signature.prompt_tokens + signature.max_new_tokens
    )
    if sequence_length < 1:
        return None
    # ``optimizer='none'`` is a useful sentinel for inference signatures and
    # was the historical default.  A TTT static preview still needs a valid
    # optimizer footprint; use the documented AdamW footprint while keeping
    # the report explicitly marked as non-admission-safe by the caller.
    optimizer = signature.optimizer
    if optimizer.lower() == "none":
        optimizer = "adamw"
    return plan_training(
        workers,
        model,
        TrainingWorkload(
            batch_size=signature.batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=signature.gradient_accumulation_steps,
            activation_checkpointing=signature.checkpointing,
        ),
        optimizer=optimizer,
    )


def _contiguous_partitions(n_layers: int, n_stages: int):
    """Yield every ordered non-empty contiguous partition of layer indices."""
    if n_stages == 1:
        yield [(0, n_layers)]
        return
    for cut in range(1, n_layers - n_stages + 2):
        for suffix in _contiguous_partitions(n_layers - cut, n_stages - 1):
            yield [(0, cut)] + [(start + cut, end + cut) for start, end in suffix]
