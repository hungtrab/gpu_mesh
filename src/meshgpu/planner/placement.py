"""
Contiguous layer placement planner (v0).
Partitions layers across available workers, estimates feasibility,
and generates a PlacementReport before job admission.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from meshgpu.planner.memory import (
    GiB,
    GpuBudget,
    TensorPeak,
    check_feasibility,
    estimate_activation_boundary_bytes,
    estimate_adam_optimizer_bytes,
    estimate_kv_cache_bytes,
    estimate_transfer_time_s,
    estimate_weight_bytes,
)

log = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    worker_id: str
    device_index: int
    total_vram_bytes: int
    free_vram_bytes: int
    compute_score: float = 1.0      # relative compute; 1.0 = baseline
    goodput_mbit_s: float | None = None  # to next stage; None = unknown

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, int):
            raise TypeError("device_index must be an integer")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        for name, value in (
            ("total_vram_bytes", self.total_vram_bytes),
            ("free_vram_bytes", self.free_vram_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.free_vram_bytes > self.total_vram_bytes:
            raise ValueError("free_vram_bytes cannot exceed total_vram_bytes")
        if (
            isinstance(self.compute_score, bool)
            or not isinstance(self.compute_score, (int, float))
            or not math.isfinite(float(self.compute_score))
            or self.compute_score <= 0
        ):
            raise ValueError("compute_score must be finite and positive")
        if self.goodput_mbit_s is not None and (
            isinstance(self.goodput_mbit_s, bool)
            or not isinstance(self.goodput_mbit_s, (int, float))
            or not math.isfinite(float(self.goodput_mbit_s))
            or self.goodput_mbit_s < 0
        ):
            raise ValueError("goodput_mbit_s must be finite and non-negative")


@dataclass
class ModelSpec:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    param_count: int
    dtype_bytes: int = 2            # fp16 default
    # Optional exact component counts obtained from a manifest/profile.  The
    # old arithmetic remains available when these fields are absent.
    per_layer_param_count: int | None = None
    embedding_param_count: int | None = None
    lm_head_param_count: int | None = None
    adapter_param_count: int = 0
    # This is the backend selected by the model adapter, not merely a label
    # for the model family.  It materially changes the attention workspace:
    # the handwritten eager path materialises scores, while SDPA/Flash are
    # expected to use a memory-efficient kernel when the runtime verifies it.
    attention_implementation: str = "sdpa"
    max_position_embeddings: int | None = None
    # LoRA parameters are fp32 in the portable/native recipe even when the
    # frozen base weights are fp16/bf16.  A separate field prevents the
    # planner from under-counting resident, gradient and optimizer memory.
    adapter_dtype_bytes: int = 4

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "vocab_size",
            "param_count",
            "dtype_bytes",
            "adapter_param_count",
            "adapter_dtype_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        positive = (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_kv_heads",
            "head_dim",
            "vocab_size",
            "dtype_bytes",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError("num_kv_heads must not exceed num_attention_heads")
        for name in ("param_count", "adapter_param_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "per_layer_param_count",
            "embedding_param_count",
            "lm_head_param_count",
        ):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer or None")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        if self.attention_implementation not in {
            "eager",
            "sdpa",
            "flash_attention_2",
        }:
            raise ValueError(
                "attention_implementation must be one of 'eager', 'sdpa', "
                "or 'flash_attention_2'"
            )
        if self.max_position_embeddings is not None:
            if (
                isinstance(self.max_position_embeddings, bool)
                or not isinstance(self.max_position_embeddings, int)
            ):
                raise TypeError("max_position_embeddings must be an integer or None")
            if self.max_position_embeddings < 1:
                raise ValueError("max_position_embeddings must be positive")
        if self.adapter_dtype_bytes < 1:
            raise ValueError("adapter_dtype_bytes must be positive")


@dataclass
class InferenceWorkload:
    batch_size: int
    max_prompt_tokens: int
    max_new_tokens: int
    max_concurrent: int = 1

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "max_prompt_tokens",
            "max_new_tokens",
            "max_concurrent",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass
class TrainingWorkload:
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "sequence_length", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be a boolean")


@dataclass
class StageAssignment:
    stage_id: int
    worker_id: str
    device_index: int
    layer_start: int
    layer_end: int
    has_embedding: bool
    has_lm_head: bool
    peak: TensorPeak
    budget: GpuBudget


@dataclass
class PlacementReport:
    feasible: bool
    reason: str
    assignments: list[StageAssignment]
    bottleneck_stage: int | None
    estimated_ttft_s: float | None          # time to first token
    estimated_decode_ms_per_token: float | None
    boundary_bytes_per_forward: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"feasible={self.feasible} reason={self.reason!r}"]
        for a in self.assignments:
            lines.append(
                f"  stage{a.stage_id} worker={a.worker_id} "
                f"layers=[{a.layer_start},{a.layer_end}) "
                f"peak={a.peak.total/GiB:.2f}GiB "
                f"margin={(a.budget.usable_bytes - a.peak.total)/GiB:+.2f}GiB"
            )
        if self.bottleneck_stage is not None:
            lines.append(f"  bottleneck=stage{self.bottleneck_stage}")
        return "\n".join(lines)


def plan_inference(
    workers: list[WorkerSpec],
    model: ModelSpec,
    workload: InferenceWorkload,
    *,
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int = GiB,
    layer_ranges: Sequence[tuple[int, int]] | None = None,
) -> PlacementReport:
    """
    Partition model layers across workers in order.
    Uses capacity-proportional split: workers with more usable VRAM get more layers.
    """
    n_stages = len(workers)
    n_layers = model.num_layers
    if not workers:
        raise ValueError("workers must not be empty")
    if n_layers < 1:
        raise ValueError("model.num_layers must be positive")
    if n_stages > n_layers:
        raise ValueError(
            f"cannot place {n_layers} layers across {n_stages} stages; "
            "each stage must own at least one layer"
        )
    if (
        model.max_position_embeddings is not None
        and workload.max_prompt_tokens + workload.max_new_tokens
        > model.max_position_embeddings
    ):
        raise ValueError(
            "inference prompt plus output budget exceeds model context: "
            f"{workload.max_prompt_tokens + workload.max_new_tokens} > "
            f"{model.max_position_embeddings}"
        )
    budgets = [
        GpuBudget(
            device_index=w.device_index,
            total_vram_bytes=w.total_vram_bytes,
            free_vram_at_admission=w.free_vram_bytes,
            user_budget_fraction=max_vram_fraction,
            reserve_min_bytes=reserve_min_bytes,
        )
        for w in workers
    ]

    # Choose the contiguous partition from the complete peak model.  This is
    # still capacity-aware, but unlike a raw VRAM-proportional split it also
    # accounts for endpoint-only embedding/lm-head weights and mode-specific
    # KV/activation/communication costs.
    layer_counts = (
        _layer_counts_from_ranges(layer_ranges, n_layers, n_stages)
        if layer_ranges is not None
        else _choose_layer_counts(
            n_layers,
            budgets,
            lambda layer_count, is_first, is_last: _inference_peak(
                model, workload, layer_count, is_first=is_first, is_last=is_last
            ),
        )
    )

    # Estimate peak per stage
    peaks: list[TensorPeak] = []
    assignments: list[StageAssignment] = []
    layer_cursor = 0

    for i, (w, b, n_l) in enumerate(zip(workers, budgets, layer_counts)):
        is_first = i == 0
        is_last = i == n_stages - 1
        ls, le = layer_cursor, layer_cursor + n_l
        layer_cursor = le

        peak = _inference_peak(
            model,
            workload,
            n_l,
            is_first=is_first,
            is_last=is_last,
        )
        peaks.append(peak)
        assignments.append(StageAssignment(
            stage_id=i,
            worker_id=w.worker_id,
            device_index=w.device_index,
            layer_start=ls,
            layer_end=le,
            has_embedding=is_first,
            has_lm_head=is_last,
            peak=peak,
            budget=b,
        ))

    result = check_feasibility(budgets, peaks)
    if not result.feasible:
        return PlacementReport(
            feasible=False,
            reason=result.reason,
            assignments=assignments,
            bottleneck_stage=None,
            estimated_ttft_s=None,
            estimated_decode_ms_per_token=None,
            boundary_bytes_per_forward=0,
        )

    # Bottleneck: stage with smallest margin
    margins = [b.usable_bytes - p.total for b, p in zip(budgets, peaks)]
    bottleneck = int(min(range(n_stages), key=lambda i: margins[i]))

    # Boundary traffic estimate (one boundary per forward step)
    boundary_bytes = estimate_activation_boundary_bytes(
        workload.batch_size,
        workload.max_prompt_tokens,
        model.hidden_size,
        model.dtype_bytes,
    )

    # TTFT estimate: sum transfer times across all boundaries
    warnings = []
    if model.attention_implementation in {"sdpa", "flash_attention_2"}:
        warnings.append(
            f"static peak assumes memory-efficient {model.attention_implementation} "
            "dispatch; verify the selected kernel on each target GPU"
        )
    ttft = None
    decode_ms = None
    goodputs = [worker.goodput_mbit_s for worker in workers]
    if all(goodput is not None and goodput > 0 for goodput in goodputs):
        boundary_times = []
        for i in range(n_stages - 1):
            gp = goodputs[i]
            assert gp is not None  # narrowed by the all(...) guard above
            t = estimate_transfer_time_s(boundary_bytes, gp)
            boundary_times.append(t)
        ttft = sum(boundary_times)
        # Decode: hidden_size * dtype_bytes * batch per token
        decode_boundary = model.hidden_size * model.dtype_bytes * workload.batch_size
        decode_times = []
        for i in range(n_stages - 1):
            gp = goodputs[i]
            assert gp is not None  # narrowed by the all(...) guard above
            decode_times.append(estimate_transfer_time_s(decode_boundary, gp))
        decode_ms = sum(decode_times) * 1000
        if ttft > 5.0:
            warnings.append(
                f"estimated TTFT {ttft:.1f}s — network may be the bottleneck"
            )

    return PlacementReport(
        feasible=True,
        reason="all stages within budget",
        assignments=assignments,
        bottleneck_stage=bottleneck,
        estimated_ttft_s=ttft,
        estimated_decode_ms_per_token=decode_ms,
        boundary_bytes_per_forward=boundary_bytes,
        warnings=warnings,
    )


def plan_training(
    workers: list[WorkerSpec],
    model: ModelSpec,
    workload: TrainingWorkload,
    *,
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int = GiB,
    optimizer: str = "adamw",
    layer_ranges: Sequence[tuple[int, int]] | None = None,
) -> PlacementReport:
    """Plan a fixed pipeline training microbatch, including optimizer state."""
    n_stages = len(workers)
    if not workers:
        raise ValueError("workers must not be empty")
    if model.num_layers < 1:
        raise ValueError("model.num_layers must be positive")
    if n_stages > model.num_layers:
        raise ValueError(
            f"cannot place {model.num_layers} layers across {n_stages} stages; "
            "each stage must own at least one layer"
        )
    if (
        model.max_position_embeddings is not None
        and workload.sequence_length > model.max_position_embeddings
    ):
        raise ValueError(
            "training sequence_length exceeds model context: "
            f"{workload.sequence_length} > {model.max_position_embeddings}"
        )
    if optimizer.lower() not in {"adam", "adamw"}:
        raise ValueError(f"unsupported optimizer for planner: {optimizer!r}")

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
    layer_counts = (
        _layer_counts_from_ranges(layer_ranges, model.num_layers, n_stages)
        if layer_ranges is not None
        else _choose_layer_counts(
            model.num_layers,
            budgets,
            lambda layer_count, is_first, is_last: _training_peak(
                model,
                workload,
                layer_count,
                is_first=is_first,
                is_last=is_last,
            ),
        )
    )
    assignments: list[StageAssignment] = []
    layer_cursor = 0
    for index, (worker, budget, layer_count) in enumerate(
        zip(workers, budgets, layer_counts)
    ):
        is_first = index == 0
        is_last = index == n_stages - 1
        layer_start = layer_cursor
        layer_end = layer_start + layer_count
        layer_cursor = layer_end
        peak = _training_peak(
            model,
            workload,
            layer_count,
            is_first=is_first,
            is_last=is_last,
        )
        assignments.append(
            StageAssignment(
                stage_id=index,
                worker_id=worker.worker_id,
                device_index=worker.device_index,
                layer_start=layer_start,
                layer_end=layer_end,
                has_embedding=is_first,
                has_lm_head=is_last,
                peak=peak,
                budget=budget,
            )
        )

    result = check_feasibility(budgets, [assignment.peak for assignment in assignments])
    if not result.feasible:
        return PlacementReport(
            feasible=False,
            reason=result.reason,
            assignments=assignments,
            bottleneck_stage=None,
            estimated_ttft_s=None,
            estimated_decode_ms_per_token=None,
            boundary_bytes_per_forward=0,
        )

    margins = [
        assignment.budget.usable_bytes - assignment.peak.total
        for assignment in assignments
    ]
    bottleneck = int(min(range(n_stages), key=lambda index: margins[index]))
    boundary_bytes = estimate_activation_boundary_bytes(
        workload.batch_size,
        workload.sequence_length,
        model.hidden_size,
        model.dtype_bytes,
    )
    return PlacementReport(
        feasible=True,
        reason="all stages within budget for one training microbatch",
        assignments=assignments,
        bottleneck_stage=bottleneck,
        estimated_ttft_s=None,
        estimated_decode_ms_per_token=None,
        boundary_bytes_per_forward=boundary_bytes,
    )


def _proportional_split(n: int, weights: Sequence[float], total: float) -> list[int]:
    """Split ``n`` items proportionally, with one item reserved per slot.

    The old implementation rounded each raw quota independently and then
    tried to repair the difference by decrementing the largest slots.  That
    can lose items when several large quotas hit the one-item floor.  This is
    the largest-remainder method applied to the ``n - k`` items left after the
    mandatory allocation, so the returned counts always sum to ``n``.

    ``total`` is retained for API compatibility with the planner callers.  A
    zero total (for example, all workers having no usable memory) selects an
    equal split; otherwise the actual finite weight sum is used as the
    normalization denominator so inconsistent caller metadata cannot violate
    the conservation invariant.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    k = len(weights)
    if n < 1 or k < 1:
        raise ValueError("n and weights must be non-empty positive values")
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(float(total))
        or total < 0
    ):
        raise ValueError("total must be a finite non-negative number")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight < 0
        for weight in weights
    ):
        raise ValueError("weights must be non-negative")
    if n < k:
        raise ValueError(f"cannot split {n} layers across {k} stages")

    weight_sum = float(sum(float(weight) for weight in weights))
    if total <= 0 or weight_sum <= 0:
        normalized = [1.0 / k] * k
    else:
        normalized = [float(weight) / weight_sum for weight in weights]

    remaining = n - k
    quotas = [remaining * weight for weight in normalized]
    floors = [math.floor(quota) for quota in quotas]
    counts = [1 + floor for floor in floors]
    remainder = remaining - sum(floors)
    order = sorted(
        range(k),
        key=lambda index: (
            -(quotas[index] - floors[index]),
            -normalized[index],
            index,
        ),
    )
    for index in order[:remainder]:
        counts[index] += 1

    assert sum(counts) == n
    return counts


def _layer_counts_from_ranges(
    layer_ranges: Sequence[tuple[int, int]],
    n_layers: int,
    n_stages: int,
) -> list[int]:
    """Validate an already-built stage partition and return its layer counts."""
    if isinstance(layer_ranges, (str, bytes)) or len(layer_ranges) != n_stages:
        raise ValueError(
            f"layer_ranges must contain exactly {n_stages} stage ranges"
        )
    counts: list[int] = []
    expected_start = 0
    for index, raw_range in enumerate(layer_ranges):
        if (
            not isinstance(raw_range, (tuple, list))
            or len(raw_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_range
            )
        ):
            raise TypeError(f"layer_ranges[{index}] must be a pair of integers")
        layer_start, layer_end = raw_range
        if (
            layer_start != expected_start
            or layer_end <= layer_start
            or layer_end > n_layers
        ):
            raise ValueError(
                "layer_ranges must be non-empty and contiguous from zero; "
                f"range {index} was [{layer_start}, {layer_end}) after "
                f"{expected_start}"
            )
        counts.append(layer_end - layer_start)
        expected_start = layer_end
    if expected_start != n_layers:
        raise ValueError(
            f"layer_ranges cover [0, {expected_start}), expected [0, {n_layers})"
        )
    return counts


def _choose_layer_counts(
    n_layers: int,
    budgets: list[GpuBudget],
    peak_for_count: Callable[[int, bool, bool], TensorPeak],
) -> list[int]:
    """Choose a contiguous split that maximizes the smallest stage margin.

    Dynamic programming keeps this bounded at ``O(stages * layers²)`` rather
    than enumerating every composition.  The score is meaningful even when
    no placement fits: the resulting report then identifies the least-bad
    bottleneck instead of depending on an arbitrary proportional rounding.
    """
    n_stages = len(budgets)
    if n_stages < 1:
        raise ValueError("at least one budget is required")
    if n_layers < n_stages:
        raise ValueError("each stage must own at least one layer")

    # ``scores[allocated]`` is the best minimum margin for the processed
    # prefix.  Parent pointers reconstruct the winning layer count per stage.
    scores: list[float | None] = [None] * (n_layers + 1)
    scores[0] = float("inf")
    parents: list[list[int | None]] = []

    for stage_id, budget in enumerate(budgets):
        next_scores: list[float | None] = [None] * (n_layers + 1)
        next_parents: list[int | None] = [None] * (n_layers + 1)
        remaining_stages = n_stages - stage_id - 1
        for allocated, prefix_score in enumerate(scores):
            if prefix_score is None:
                continue
            max_count = n_layers - allocated - remaining_stages
            for layer_count in range(1, max_count + 1):
                new_allocated = allocated + layer_count
                peak = peak_for_count(
                    layer_count,
                    stage_id == 0,
                    stage_id == n_stages - 1,
                )
                margin = budget.usable_bytes - peak.total
                score = min(prefix_score, float(margin))
                previous_score = next_scores[new_allocated]
                if previous_score is None or score > previous_score:
                    next_scores[new_allocated] = score
                    next_parents[new_allocated] = allocated
        scores = next_scores
        parents.append(next_parents)

    if scores[n_layers] is None:
        raise RuntimeError("could not construct a non-empty layer partition")
    counts = [0] * n_stages
    allocated = n_layers
    for stage_id in range(n_stages - 1, -1, -1):
        previous = parents[stage_id][allocated]
        if previous is None:
            raise RuntimeError("layer partition reconstruction failed")
        counts[stage_id] = allocated - previous
        allocated = previous
    assert allocated == 0
    assert sum(counts) == n_layers
    return counts


def _inference_peak(
    model: ModelSpec,
    workload: InferenceWorkload,
    layer_count: int,
    *,
    is_first: bool,
    is_last: bool,
) -> TensorPeak:
    effective_batch = workload.batch_size * workload.max_concurrent
    prompt_tokens = workload.max_prompt_tokens
    total_tokens = prompt_tokens + workload.max_new_tokens
    base_params = _base_layer_params(model, layer_count, is_first, is_last)
    adapter_params = _adapter_params_for_layers(model, layer_count)
    weight_bytes = estimate_weight_bytes(base_params, model.dtype_bytes)
    adapter_bytes = estimate_weight_bytes(adapter_params, model.adapter_dtype_bytes)
    kv_bytes = estimate_kv_cache_bytes(
        workload.batch_size * workload.max_concurrent,
        workload.max_prompt_tokens + workload.max_new_tokens,
        layer_count,
        model.num_kv_heads,
        model.head_dim,
        model.dtype_bytes,
    )
    activation_bytes = estimate_activation_boundary_bytes(
        effective_batch,
        prompt_tokens,
        model.hidden_size,
        model.dtype_bytes,
    )

    # A stage executes one decoder block at a time, so the local intermediate
    # activation estimate is per-layer rather than multiplied by the number of
    # layers.  It still covers the largest common temporary tensors (MLP and
    # q/k/v/output projections) that are not represented by the boundary.
    internal_activation_bytes = (
        effective_batch
        * prompt_tokens
        * (
            model.hidden_size
            + 2 * model.intermediate_size
            + (model.num_attention_heads + 2 * model.num_kv_heads) * model.head_dim
        )
        * model.dtype_bytes
    )
    attention_workspace_bytes = _attention_workspace_bytes(
        model,
        effective_batch,
        prompt_tokens,
        total_tokens,
        training=False,
    )
    logits_bytes = (
        effective_batch * prompt_tokens * model.vocab_size * model.dtype_bytes
        if is_last
        else 0
    )
    return TensorPeak(
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_bytes,
        activation_bytes=activation_bytes,
        # One boundary on the embedding stage, two on an interior/final stage.
        comm_buffer_bytes=activation_bytes if is_first else 2 * activation_bytes,
        workspace_bytes=internal_activation_bytes + attention_workspace_bytes + logits_bytes,
        adapter_bytes=adapter_bytes,
    )


def _training_peak(
    model: ModelSpec,
    workload: TrainingWorkload,
    layer_count: int,
    *,
    is_first: bool,
    is_last: bool,
) -> TensorPeak:
    base_params = _base_layer_params(model, layer_count, is_first, is_last)
    adapter_params = _adapter_params_for_layers(model, layer_count)
    weight_bytes = estimate_weight_bytes(base_params, model.dtype_bytes)
    adapter_bytes = estimate_weight_bytes(adapter_params, model.adapter_dtype_bytes)
    # With an adapter present, base weights are resident but frozen.  Only the
    # adapter parameters need gradient and Adam state; full fine-tuning
    # artifacts have adapter_param_count == 0 and retain the old full-state
    # estimate.
    trainable_params = _trainable_layer_params(
        model,
        layer_count,
        is_first=is_first,
        is_last=is_last,
    )
    trainable_dtype_bytes = (
        model.adapter_dtype_bytes if model.adapter_param_count else model.dtype_bytes
    )
    trainable_bytes = estimate_weight_bytes(trainable_params, trainable_dtype_bytes)
    activation_bytes = estimate_activation_boundary_bytes(
        workload.batch_size,
        workload.sequence_length,
        model.hidden_size,
        model.dtype_bytes,
    )
    retained_layers = 1 if workload.activation_checkpointing else layer_count
    internal_activation_bytes = (
        workload.batch_size
        * workload.sequence_length
        * retained_layers
        * (
            model.hidden_size
            + 2 * model.intermediate_size
            + (model.num_attention_heads + 2 * model.num_kv_heads) * model.head_dim
        )
        * model.dtype_bytes
    )
    attention_workspace_bytes = _attention_workspace_bytes(
        model,
        workload.batch_size,
        workload.sequence_length,
        workload.sequence_length,
        training=True,
    )
    logits_bytes = (
        workload.batch_size
        * workload.sequence_length
        * model.vocab_size
        * model.dtype_bytes
        if is_last
        else 0
    )
    return TensorPeak(
        weight_bytes=weight_bytes,
        adapter_bytes=adapter_bytes,
        gradient_bytes=trainable_bytes,
        optimizer_bytes=estimate_adam_optimizer_bytes(trainable_params),
        activation_bytes=activation_bytes,
        comm_buffer_bytes=activation_bytes if is_first else 2 * activation_bytes,
        workspace_bytes=internal_activation_bytes + attention_workspace_bytes + logits_bytes,
    )


def _attention_workspace_bytes(
    model: ModelSpec,
    effective_batch: int,
    prefill_tokens: int,
    total_tokens: int,
    *,
    training: bool,
) -> int:
    """Estimate attention temporaries for one stage.

    The eager reference implementation explicitly creates a ``[B, H, Q, K]``
    score tensor.  For inference the largest score is either prompt prefill
    (``P x P``) or one-token decode against the full cache (``1 x T``).  The
    backward graph retains additional score/probability storage, hence the
    larger training multiplier.  SDPA and Flash are modelled as linear in
    sequence length because their memory-efficient contract avoids the score
    matrix; exact CUDA allocator usage still belongs to a measured profile.
    """
    if model.attention_implementation == "eager":
        score_elements = effective_batch * model.num_attention_heads * max(
            prefill_tokens * prefill_tokens,
            total_tokens,
        )
        # Eager keeps score and probability-like intermediates live at some
        # point in the forward/backward peak.  Keep this conservative rather
        # than presenting the estimate as an exact allocator measurement.
        return score_elements * model.dtype_bytes * (3 if training else 2)

    # Fused/SDPA paths may allocate a small linear scratch buffer.  Count a
    # hidden-sized output scratch so the estimate is not unrealistically zero,
    # but do not charge a quadratic score matrix.
    return (
        effective_batch
        * max(prefill_tokens, total_tokens)
        * model.hidden_size
        * model.dtype_bytes
    )


def _trainable_layer_params(
    model: ModelSpec,
    n_layers: int,
    *,
    is_first: bool = False,
    is_last: bool = False,
) -> int:
    """Return trainable parameters for one stage.

    Full fine-tuning updates endpoint modules too: the first stage owns the
    embedding table and the last stage owns the final norm/lm head.  The old
    estimate counted only decoder blocks, which understated gradient and Adam
    state on both endpoint stages.  LoRA remains layer-only because the built
    in recipe injects adapters into attention projections.
    """
    if model.adapter_param_count:
        # ``adapter_param_count`` is the complete-model count as obtained from
        # a manifest.  Distribute it by contiguous layer ownership; endpoint
        # modules currently do not carry LoRA adapters.
        # Round up: an under-estimate can admit a request that OOMs.  The
        # complete-model count is not necessarily divisible by the number of
        # layers, so conservative ownership beats integer truncation here.
        return math.ceil(model.adapter_param_count * n_layers / max(model.num_layers, 1))
    return _layer_params(model, n_layers, is_first=is_first, is_last=is_last)


def _layer_params(model: ModelSpec, n_layers: int, is_first: bool, is_last: bool) -> int:
    """Rough parameter count for a stage with n_layers decoder blocks."""
    return _base_layer_params(model, n_layers, is_first, is_last) + _adapter_params_for_layers(
        model, n_layers
    )


def _base_layer_params(
    model: ModelSpec,
    n_layers: int,
    is_first: bool,
    is_last: bool,
) -> int:
    """Return base-model parameters, excluding adapter parameters."""
    H, intermediate, V = model.hidden_size, model.intermediate_size, model.vocab_size
    Hkv, D = model.num_kv_heads, model.head_dim
    Ha = model.num_attention_heads

    # Per-layer: q,k,v,o projections + gate,up,down + 2 layer norms.  A
    # manifest may provide the exact count (Qwen includes q/k RMSNorm), in
    # which case use it instead of a family-specific approximation.
    if model.per_layer_param_count is not None:
        per_layer = model.per_layer_param_count
    else:
        attn = Ha * D * H + Hkv * D * H + Hkv * D * H + Ha * D * H
        mlp = H * intermediate + H * intermediate + intermediate * H
        norms = 2 * H
        per_layer = attn + mlp + norms

    total = per_layer * n_layers
    if is_first:
        total += model.embedding_param_count if model.embedding_param_count is not None else V * H
    if is_last:
        total += (
            model.lm_head_param_count
            if model.lm_head_param_count is not None
            else H * V + H
        )
    return total


def _adapter_params_for_layers(model: ModelSpec, n_layers: int) -> int:
    """Conservatively assign complete-model adapters to a stage's layers."""
    if model.adapter_param_count == 0:
        return 0
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    return math.ceil(model.adapter_param_count * n_layers / max(model.num_layers, 1))
