"""Dependency-correct 1F1B pipeline schedule.

The scheduler runs in one Python process for the current portable prototype,
so it does not claim wall-clock overlap. It does, however, execute stage-level
forward/backward actions in the same warmup/steady/drain order as a distributed
1F1B engine and keeps boundary gradients and saved activations explicit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count
from threading import Lock
from typing import Literal

import torch

from meshgpu.backends.portable.pipeline import (
    _accumulate_tied_embedding_grads,
    sync_tied_parameters,
)
from meshgpu.backends.portable.stage_worker import StageWorker

log = logging.getLogger(__name__)


@dataclass
class MicroBatchResult:
    micro_idx: int
    loss: float
    n_valid_tokens: int


Action = tuple[Literal["F", "B"], int]
_SCHEDULE_OPERATION_SEQ = count(1)
_SCHEDULE_OPERATION_LOCK = Lock()


def _reserve_operation_ids(count_needed: int) -> int:
    """Reserve a contiguous, process-wide operation-id range."""
    with _SCHEDULE_OPERATION_LOCK:
        first = next(_SCHEDULE_OPERATION_SEQ)
        for _ in range(count_needed - 1):
            next(_SCHEDULE_OPERATION_SEQ)
    return first


def build_1f1b_actions(n_stages: int, n_microbatches: int) -> list[list[Action]]:
    """Build per-stage FIFO actions for a classic 1F1B schedule."""
    if n_stages < 1:
        raise ValueError("n_stages must be positive")
    if n_microbatches < 1:
        raise ValueError("n_microbatches must be positive")

    actions: list[list[Action]] = []
    for stage in range(n_stages):
        warmup = min(n_microbatches, n_stages - stage - 1)
        stage_actions: list[Action] = [("F", micro) for micro in range(warmup)]
        next_forward = warmup
        next_backward = 0
        while next_forward < n_microbatches:
            stage_actions.append(("F", next_forward))
            next_forward += 1
            stage_actions.append(("B", next_backward))
            next_backward += 1
        while next_backward < n_microbatches:
            stage_actions.append(("B", next_backward))
            next_backward += 1
        actions.append(stage_actions)
    return actions


def pipeline_train_1f1b(
    workers: list[StageWorker],
    micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
    optimizers: list[torch.optim.Optimizer],
    *,
    ignore_index: int = -100,
    job_id: str = "1f1b",
) -> list[MicroBatchResult]:
    """Run one optimizer step using a dependency-aware 1F1B schedule.

    Losses are divided by the total number of valid labels across all
    microbatches, so the accumulated gradient is invariant to microbatch
    boundaries. Optimizers step once after the drain phase.
    """
    if not workers:
        raise ValueError("workers must not be empty")
    if len(workers) != len(optimizers):
        raise ValueError("workers and optimizers must have the same length")
    if not micro_batches:
        return []

    total_valid = sum(
        int((labels != ignore_index).sum().item())
        for _, labels in micro_batches
    )
    normalizer = max(total_valid, 1)
    for micro_idx, (input_ids, labels) in enumerate(micro_batches):
        if input_ids.ndim != 2 or labels.ndim != 2:
            raise ValueError(
                f"microbatch {micro_idx}: input_ids and labels must have shape [batch, sequence]"
            )
        if input_ids.shape != labels.shape:
            raise ValueError(
                f"microbatch {micro_idx}: input_ids and labels must have the same shape"
            )
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError(f"microbatch {micro_idx} must be non-empty")
    sync_tied_parameters(workers)
    actions = build_1f1b_actions(len(workers), len(micro_batches))
    pointers = [0] * len(workers)
    forward_outputs: dict[tuple[int, int], torch.Tensor] = {}
    backward_grads: dict[tuple[int, int], torch.Tensor | None] = {}
    losses: dict[int, float] = {}
    valid_counts: dict[int, int] = {}
    op_base = _reserve_operation_ids(len(micro_batches))
    operation_ids = {op_base + micro_idx for micro_idx in range(len(micro_batches))}

    try:
        # A tick tries one action per stage. Ascending order propagates forward
        # readiness; descending retries allow backward readiness to propagate in
        # the same tick when the next stage already produced its gradient.
        while any(
            pointer < len(stage_actions)
            for pointer, stage_actions in zip(pointers, actions)
        ):
            progressed = False
            for stage_idx, worker in enumerate(workers):
                pointer = pointers[stage_idx]
                if pointer >= len(actions[stage_idx]):
                    continue
                kind, micro_idx = actions[stage_idx][pointer]
                op_id = op_base + micro_idx

                if kind == "F":
                    hidden = None
                    if stage_idx > 0:
                        upstream = forward_outputs.get((stage_idx - 1, micro_idx))
                        if upstream is None:
                            continue
                        # Use the same explicit boundary path as the regular
                        # pipeline.  In particular, local_cuda keeps the
                        # tensor on-device while CPU transport makes the
                        # network/serialization boundary observable.
                        hidden = worker._move_input(upstream.detach()).requires_grad_(True)
                    ids, _ = micro_batches[micro_idx]
                    result = worker.forward_train(
                        hidden,
                        input_ids=ids if stage_idx == 0 else None,
                        operation_id=op_id,
                        attempt_id=f"{job_id}_m{micro_idx}",
                    )
                    if stage_idx < len(workers) - 1:
                        forward_outputs[(stage_idx, micro_idx)] = result.output
                    pointers[stage_idx] += 1
                    progressed = True
                    continue

                # Backward action: the last stage creates the loss gradient; every
                # earlier stage consumes a gradient sent by its downstream peer.
                if stage_idx == len(workers) - 1:
                    saved = worker._saved.get(op_id)
                    if saved is None:
                        continue
                    _, labels = micro_batches[micro_idx]
                    labels_dev = labels.to(worker._ctx.device)
                    valid = int((labels_dev != ignore_index).sum().item())
                    logits = saved["out"]
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        labels_dev.view(-1),
                        ignore_index=ignore_index,
                        reduction="sum",
                    ) / normalizer
                    loss.backward()
                    losses[micro_idx] = float(loss.item())
                    valid_counts[micro_idx] = valid
                    last_x = saved["x"]
                    grad = last_x.grad if last_x.requires_grad else None
                    worker._saved.pop(op_id, None)
                    worker.clear_kv()
                    if stage_idx > 0:
                        backward_grads[(stage_idx - 1, micro_idx)] = (
                            _boundary_gradient(worker, grad)
                            if grad is not None
                            else None
                        )
                else:
                    grad_key = (stage_idx, micro_idx)
                    if grad_key not in backward_grads:
                        continue
                    grad = backward_grads.pop(grad_key)
                    bwd = worker.backward_train(
                        grad,
                        operation_id=op_id,
                        attempt_id=f"{job_id}_m{micro_idx}",
                    )
                    if stage_idx > 0:
                        backward_grads[(stage_idx - 1, micro_idx)] = bwd.grad_input
                pointers[stage_idx] += 1
                progressed = True

            if not progressed:
                pending = [
                    actions[i][pointers[i]]
                    for i in range(len(workers))
                    if pointers[i] < len(actions[i])
                ]
                raise RuntimeError(f"1F1B schedule deadlocked; pending actions={pending}")

        # A tied multi-stage model has two Parameter objects: the input
        # embedding receives its own gradient and the replicated LM head
        # receives the loss gradient.  Combine them before stepping the owner,
        # then copy the result to the replica on the final stage.
        _accumulate_tied_embedding_grads(workers)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        sync_tied_parameters(workers)

        return [
            MicroBatchResult(
                micro_idx=micro_idx,
                loss=losses[micro_idx],
                n_valid_tokens=valid_counts[micro_idx],
            )
            for micro_idx in range(len(micro_batches))
        ]
    except BaseException:
        for worker in workers:
            for operation_id in operation_ids:
                worker._saved.pop(operation_id, None)
            worker.clear_kv()
        # Backward may have populated gradients on downstream stages before
        # an upstream stage or transport fails.  Those gradients belong to an
        # abandoned logical step and must not leak into the next retry.
        for optimizer in optimizers:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception as exc:
                log.warning("could not clear optimizer gradients after 1F1B failure: %s", exc)
        raise


def split_into_micro_batches(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    n_micro: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split a global batch [B, T] into n_micro non-empty microbatches."""
    if n_micro < 1:
        raise ValueError("n_micro must be positive")
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must have shape [batch, sequence]")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape")
    batch_size = input_ids.shape[0]
    if batch_size < n_micro:
        raise ValueError(f"batch_size {batch_size} < n_micro {n_micro}")
    base, rem = divmod(batch_size, n_micro)
    slices = []
    start = 0
    for index in range(n_micro):
        end = start + base + (1 if index < rem else 0)
        slices.append((input_ids[start:end], labels[start:end]))
        start = end
    return slices


def _boundary_gradient(worker: StageWorker, gradient: torch.Tensor) -> torch.Tensor:
    """Keep 1F1B reverse boundaries on the selected transport path."""
    detached = gradient.detach()
    if worker._ctx.transport == "cpu":
        return detached.cpu()
    if worker._ctx.transport == "local_cuda":
        return detached
    raise ValueError(f"unsupported stage transport: {worker._ctx.transport!r}")
