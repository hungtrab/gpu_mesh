"""
DiLoCo: Distributed Low-Communication training (Douillard et al. 2023).

Core idea:
  - Each worker (or cluster of workers) runs H local gradient steps independently.
  - After H steps, workers synchronize only their outer gradient:
      Δ_i = θ_ref - θ_i   (outer gradient = displacement from reference)
  - Outer optimizer (e.g. SGD with momentum) aggregates outer gradients:
      θ_ref ← outer_opt(mean(Δ_i))
  - Workers reset to new θ_ref and continue.

Advantages vs. standard DDP:
  - Communication once every H steps instead of every step → H× bandwidth reduction.
  - Each inner loop can use any optimizer; outer loop uses a slow global optimizer.
  - Compatible with LoRA: only adapter params participate in outer sync.

This implementation:
  - "Workers" are local replicas (StageWorker lists for pipeline parallelism).
  - All-reduce is simulated in-process (mean of tensors from all replicas).
  - For actual distributed deployment, replace `_allreduce` with NCCL / gloo.
  - Only LoRA adapter parameters (lora_A, lora_B) participate in the outer sync
    when `lora_only=True`; base model weights stay frozen as in LoRA fine-tuning.

Reference:
  Douillard et al. "DiLoCo: Distributed Low-Communication Large Language Model
  Training." arXiv 2311.08105, 2023.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from meshgpu.backends.portable.stage_worker import StageWorker

log = logging.getLogger(__name__)


@dataclass
class DiLoCoConfig:
    inner_steps: int = 100        # H: local steps between outer syncs
    outer_lr: float = 0.7         # outer SGD learning rate
    outer_momentum: float = 0.9   # outer SGD momentum
    lora_only: bool = True        # sync only LoRA adapter params

    def __post_init__(self) -> None:
        if isinstance(self.inner_steps, bool) or not isinstance(self.inner_steps, int):
            raise TypeError("inner_steps must be an integer")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if not math.isfinite(self.outer_lr) or self.outer_lr <= 0:
            raise ValueError("outer_lr must be finite and positive")
        if not math.isfinite(self.outer_momentum) or not 0 <= self.outer_momentum < 1:
            raise ValueError("outer_momentum must be finite and in [0, 1)")


@dataclass
class DiLoCoStepResult:
    inner_step: int
    outer_step: int
    loss: float
    n_valid_tokens: int
    synced: bool = False          # True on outer-sync steps


class DiLoCoTrainer:
    """
    Federated LoRA trainer using DiLoCo outer synchronization.

    Each "replica" is a list of StageWorkers representing one GPU cluster.
    In-process simulation: all replicas live in the same process.
    """

    def __init__(
        self,
        replicas: list[list[StageWorker]],   # [replica][stage]
        inner_opts: list[list[torch.optim.Optimizer]],  # [replica][stage]
        cfg: DiLoCoConfig | None = None,
        train_fn: Callable | None = None,
    ) -> None:
        if len(replicas) < 1:
            raise ValueError("need at least 1 replica")
        if len(replicas) != len(inner_opts):
            raise ValueError("replicas and inner_opts must have same length")
        if any(not workers for workers in replicas):
            raise ValueError("each replica must contain at least one stage")
        if any(len(workers) != len(opts) for workers, opts in zip(replicas, inner_opts)):
            raise ValueError("each replica needs one inner optimizer per stage")

        self._replicas = replicas
        self._inner_opts = inner_opts
        self._cfg = cfg or DiLoCoConfig()

        # Build outer optimizer: one param group per adapter param per replica
        # For the outer update we operate on the reference model (replica 0's params)
        ref_params = _adapter_params(replicas[0]) if self._cfg.lora_only \
            else _all_params(replicas[0])
        if not ref_params:
            selected = "LoRA adapter" if self._cfg.lora_only else "trainable"
            raise ValueError(f"no {selected} parameters found in the reference replica")
        _validate_replica_parameter_layout(replicas, lora_only=self._cfg.lora_only)
        self._outer_opt = torch.optim.SGD(
            ref_params,
            lr=self._cfg.outer_lr,
            momentum=self._cfg.outer_momentum,
        )
        # Save reference checkpoint at init
        self._ref_state = _snapshot_params(replicas[0], lora_only=self._cfg.lora_only)

        self._inner_step = 0
        self._outer_step = 0

        # train_fn: (workers, input_ids, labels, optimizers) →
        # {"loss": float, "n_valid_tokens": int}
        # Default: use pipeline_train_step
        if train_fn is None:
            from meshgpu.backends.portable.pipeline import pipeline_train_step
            self._train_fn = pipeline_train_step
        else:
            self._train_fn = train_fn

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
        max_outer_steps: int = 10,
    ):
        """
        Yield DiLoCoStepResult per inner step.
        After every `inner_steps` steps, outer sync fires and replicas reset.
        """
        if max_outer_steps < 1:
            raise ValueError("max_outer_steps must be positive")
        op_seq = 0

        for input_ids, labels in data_iter:
            if self._outer_step >= max_outer_steps:
                break

            # Run one inner step on each replica (in-process, sequentially)
            losses: list[float] = []
            tokens: list[int] = []
            for r, (workers, opts) in enumerate(zip(self._replicas, self._inner_opts)):
                op_seq += 1
                info = self._train_fn(
                    workers, input_ids, labels, opts,
                    operation_id=op_seq,
                    attempt_id=f"diloco_r{r}_s{self._inner_step}",
                )
                losses.append(float(info["loss"]))
                tokens.append(int(info["n_valid_tokens"]))

            self._inner_step += 1
            avg_loss = sum(losses) / len(losses)
            avg_tokens = sum(tokens) // len(tokens)

            synced = False
            if self._inner_step % self._cfg.inner_steps == 0:
                self._outer_sync()
                synced = True

            yield DiLoCoStepResult(
                inner_step=self._inner_step,
                outer_step=self._outer_step,
                loss=avg_loss,
                n_valid_tokens=avg_tokens,
                synced=synced,
            )

    # ------------------------------------------------------------------
    # Outer synchronization
    # ------------------------------------------------------------------

    def _outer_sync(self) -> None:
        """
        Compute outer gradients (Δ = θ_ref - θ_local) for all replicas,
        all-reduce, apply outer optimizer, broadcast new reference to replicas.
        """
        lora_only = self._cfg.lora_only

        # Compute outer grad as average displacement across replicas
        outer_grads = _allreduce_outer_grads(
            self._replicas,
            self._ref_state,
            lora_only=lora_only,
        )

        # The outer optimizer is attached to replica 0's Parameter objects,
        # but those parameters contain replica 0's *local* model after inner
        # training.  Restore the reference values before applying the outer
        # displacement; otherwise the outer update starts from the wrong point
        # and produces a topology-dependent result.
        _load_params(self._replicas[0], self._ref_state, lora_only=lora_only)

        # Apply outer gradient to reference model (replica 0)
        ref_params = _adapter_params(self._replicas[0]) if lora_only \
            else _all_params(self._replicas[0])

        self._outer_opt.zero_grad()
        for param, grad in zip(ref_params, outer_grads):
            param.grad = grad.to(param.device)
        self._outer_opt.step()

        # Snapshot new reference
        self._ref_state = _snapshot_params(self._replicas[0], lora_only=lora_only)

        # Broadcast new reference to all replicas
        for replica in self._replicas:
            _load_params(replica, self._ref_state, lora_only=lora_only)

        # Reset inner optimizers
        for opts in self._inner_opts:
            for opt in opts:
                opt.zero_grad(set_to_none=True)

        self._outer_step += 1
        log.info(
            "DiLoCo outer sync %d: ref params updated, broadcast to %d replicas",
            self._outer_step, len(self._replicas),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter_params(workers: list[StageWorker]) -> list[nn.Parameter]:
    """Return LoRA adapter parameters (lora_A, lora_B) from all stage models."""
    params: list[nn.Parameter] = []
    for w in workers:
        for name, p in w._model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                params.append(p)
    return params


def _all_params(workers: list[StageWorker]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for w in workers:
        params.extend(p for p in w._model.parameters() if p.requires_grad)
    return params


def _snapshot_params(
    workers: list[StageWorker],
    lora_only: bool,
) -> list[torch.Tensor]:
    """Return a list of detached parameter tensors (CPU copy)."""
    params = _adapter_params(workers) if lora_only else _all_params(workers)
    return [p.data.detach().clone().cpu() for p in params]


def _load_params(
    workers: list[StageWorker],
    snapshot: list[torch.Tensor],
    lora_only: bool,
) -> None:
    """Overwrite model parameters from a snapshot."""
    params = _adapter_params(workers) if lora_only else _all_params(workers)
    if len(params) != len(snapshot):
        raise ValueError(
            f"parameter snapshot has {len(snapshot)} tensors, but replica has {len(params)}"
        )
    for p, snap in zip(params, snapshot):
        if tuple(p.shape) != tuple(snap.shape):
            raise ValueError(
                f"parameter shape mismatch: replica={tuple(p.shape)}, "
                f"snapshot={tuple(snap.shape)}"
            )
        p.data.copy_(snap.to(p.device))


def _allreduce_outer_grads(
    replicas: list[list[StageWorker]],
    ref_state: list[torch.Tensor],
    lora_only: bool,
) -> list[torch.Tensor]:
    """
    Outer gradient = mean over replicas of (θ_ref - θ_local).
    Returns list of gradients aligned with ref_state.
    """
    n = len(replicas)
    if n < 1:
        raise ValueError("need at least one replica")
    if len(ref_state) == 0:
        raise ValueError("reference parameter snapshot must not be empty")
    grad_sum: list[torch.Tensor] = [torch.zeros_like(s) for s in ref_state]

    for workers in replicas:
        params = _adapter_params(workers) if lora_only else _all_params(workers)
        if len(params) != len(ref_state):
            raise ValueError(
                f"replica parameter count {len(params)} does not match reference "
                f"count {len(ref_state)}"
            )
        for i, (ref, p) in enumerate(zip(ref_state, params)):
            if tuple(ref.shape) != tuple(p.shape):
                raise ValueError(
                    f"replica parameter shape {tuple(p.shape)} does not match "
                    f"reference shape {tuple(ref.shape)}"
                )
            # Outer grad: displacement from reference (pull back toward reference)
            delta = ref - p.data.detach().cpu()
            grad_sum[i] += delta

    return [g / n for g in grad_sum]


def _validate_replica_parameter_layout(
    replicas: list[list[StageWorker]],
    *,
    lora_only: bool,
) -> None:
    """Fail early when replicas cannot participate in the same outer update."""
    reference = _selected_named_params(replicas[0], lora_only=lora_only)
    reference_layout = [
        (name, tuple(param.shape)) for name, param in reference
    ]
    for replica_index, workers in enumerate(replicas[1:], start=1):
        params = _selected_named_params(workers, lora_only=lora_only)
        if len(params) != len(reference_layout):
            raise ValueError(
                f"replica {replica_index} has {len(params)} selected parameters; "
                f"reference has {len(reference_layout)}"
            )
        layout = [(name, tuple(param.shape)) for name, param in params]
        if layout != reference_layout:
            raise ValueError(
                f"replica {replica_index} parameter names/shapes do not match reference"
            )


def _selected_named_params(
    workers: list[StageWorker],
    *,
    lora_only: bool,
) -> list[tuple[str, nn.Parameter]]:
    """Return the exact ordered parameter layout participating in outer sync."""
    selected: list[tuple[str, nn.Parameter]] = []
    for worker in workers:
        for name, param in worker._model.named_parameters():
            if lora_only:
                if "lora_A" not in name and "lora_B" not in name:
                    continue
            elif not param.requires_grad:
                continue
            selected.append((name, param))
    return selected
