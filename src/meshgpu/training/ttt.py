"""Task-scoped LoRA test-time training for a portable pipeline.

One ``TaskTTTSession`` owns one adapter baseline.  Adaptation data may change
the adapter for the current task, but ``reset_task`` restores the baseline and
clears optimizer state before the next task.  Base model weights never enter
the task snapshot, which avoids a second full-model VRAM allocation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from meshgpu.backends.native.lora_recipe import (
    LoRAConfig,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    trainable_parameters,
)
from meshgpu.backends.portable.pipeline import pipeline_train_step
from meshgpu.backends.portable.stage_worker import StageWorker


@dataclass
class TaskTTTConfig:
    """Bounded LoRA adaptation configuration."""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_steps: int = 1
    optimizer: str = "adamw"

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.optimizer.lower() != "adamw":
            raise ValueError("TaskTTT currently supports only AdamW")


class TaskTTTSession:
    """Own a reusable, resettable LoRA adaptation session."""

    def __init__(
        self,
        workers: list[StageWorker],
        cfg: TaskTTTConfig | None = None,
    ) -> None:
        if not workers:
            raise ValueError("workers must not be empty")
        self.workers = workers
        self.cfg = cfg or TaskTTTConfig()
        for worker in workers:
            # ``apply_lora`` is idempotent for an already wrapped stage.  Run
            # it unconditionally so a partially restored stage cannot slip
            # through merely because one adapter parameter happens to exist.
            apply_lora(worker._model, self.cfg.lora)
        self._optimizers = [self._make_optimizer(worker) for worker in workers]
        self._baseline = [
            {
                name: value.detach().cpu().clone()
                for name, value in lora_state_dict(worker._model).items()
            }
            for worker in workers
        ]
        self._task_steps = 0

    def adapt(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        steps: int | None = None,
        operation_id_start: int = 1,
        attempt_prefix: str = "ttt",
    ) -> list[dict[str, float]]:
        """Run bounded LoRA updates on one task's labelled examples."""
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("input_ids and labels must have the same shape [batch, sequence]")
        n_steps = self.cfg.max_steps if steps is None else steps
        if n_steps < 1:
            raise ValueError("steps must be positive")
        results: list[dict[str, float]] = []
        for step in range(n_steps):
            result = pipeline_train_step(
                self.workers,
                input_ids,
                labels,
                self._optimizers,
                operation_id=operation_id_start + step,
                attempt_id=f"{attempt_prefix}_{self._task_steps + step}",
                max_grad_norm=self.cfg.max_grad_norm,
            )
            results.append(result)
        self._task_steps += n_steps
        return results

    def reset_task(self) -> None:
        """Restore adapter baseline and clear all task-local state."""
        for worker, baseline, optimizer in zip(self.workers, self._baseline, self._optimizers):
            load_lora_state_dict(worker._model, baseline)
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            worker.clear_all_kv()
            worker.clear_saved()
        self._task_steps = 0

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a CPU-only, stage-qualified adapter snapshot for persistence."""
        state: dict[str, torch.Tensor] = {}
        for stage_id, worker in enumerate(self.workers):
            for name, value in lora_state_dict(worker._model).items():
                state[f"stage{stage_id}.{name}"] = value.detach().cpu().clone()
        return state

    def _make_optimizer(self, worker: StageWorker) -> torch.optim.Optimizer:
        params = trainable_parameters(worker._model)
        if not params:
            raise ValueError(f"stage {worker._ctx.stage_id} has no trainable LoRA parameters")
        return torch.optim.AdamW(
            params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
