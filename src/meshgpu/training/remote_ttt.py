"""Task-scoped LoRA adaptation through remote stage RPC.

This module is the training counterpart of ``remote-serve``.  Each Kaggle
kernel keeps its own stage model and autograd graph; the gateway sends only
the hidden activation forward and the boundary gradient backward.  No graph,
logits tensor, optimizer state, or base-model weights cross the relay.

The session intentionally uses one in-flight microbatch and one optimizer
step at a time.  That is the correctness path for TTT: it bounds activation
memory and makes a reconnect/retry boundary unambiguous.  Continuous batching
or a pipelined 1F1B WAN schedule can be added without changing this wire
contract.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from meshgpu.backends.portable.rpc import StageRpcClient
from meshgpu.training.ttt import TaskTTTConfig

_MAX_UINT32 = 1 << 32


@dataclass(frozen=True)
class RemoteTTTStepResult:
    """Metrics for one completed remote optimizer boundary."""

    step: int
    loss: float
    grad_norm: float
    n_valid_tokens: int
    overflow: bool = False


class RemoteTaskTTTSession:
    """Run resettable LoRA TTT over an ordered set of remote stages."""

    def __init__(
        self,
        clients: Sequence[StageRpcClient],
        cfg: TaskTTTConfig | None = None,
    ) -> None:
        if not clients:
            raise ValueError("clients must not be empty")
        if any(not isinstance(client, StageRpcClient) for client in clients):
            raise TypeError("clients must contain StageRpcClient instances")
        self.clients = list(clients)
        self.cfg = cfg or TaskTTTConfig()
        self._configured = False
        self._task_steps = 0

    @property
    def task_steps(self) -> int:
        return self._task_steps

    async def configure(
        self,
        *,
        batch_size: int | None = None,
        sequence_length: int | None = None,
    ) -> list[dict[str, object]]:
        """Configure all stages; identical repeated calls are idempotent."""
        _validate_optional_shape(batch_size, "batch_size")
        _validate_optional_shape(sequence_length, "sequence_length")
        if (batch_size is None) != (sequence_length is None):
            raise ValueError("batch_size and sequence_length must be provided together")
        configured = await asyncio.gather(
            *(
                client.configure_training(
                    self.cfg.lora,
                    learning_rate=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    activation_checkpointing=self.cfg.activation_checkpointing,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
                for client in self.clients
            ),
            return_exceptions=True,
        )
        errors = [result for result in configured if isinstance(result, BaseException)]
        if errors:
            # One stage may have completed configuration before another stage
            # rejected it (for example a tied-endpoint model).  Reset any
            # successful stage while the connections are still alive so a
            # failed configure cannot leave a half-owned task behind.
            await asyncio.gather(
                *(client.reset_training() for client in self.clients),
                return_exceptions=True,
            )
            raise errors[0]

        results = [result for result in configured if isinstance(result, dict)]
        if len(results) != len(self.clients):  # pragma: no cover - defensive
            raise RuntimeError("remote stage configuration returned an invalid result")
        try:
            for index, summary in enumerate(results):
                if int(summary.get("stage_id", -1)) != index:
                    raise RuntimeError(
                        f"remote stage order mismatch: client {index} reported "
                        f"stage_id={summary.get('stage_id')!r}"
                    )
                expected_last = index == len(self.clients) - 1
                if bool(summary.get("is_last", False)) != expected_last:
                    raise RuntimeError(
                        f"remote stage {index} has inconsistent is_last metadata"
                    )
                if bool(summary.get("tied_embeddings", False)):
                    raise RuntimeError(
                        "remote TTT does not support tied word embeddings; use an "
                        "untied artifact or local portable training"
                    )
        except BaseException:
            await asyncio.gather(
                *(client.reset_training() for client in self.clients),
                return_exceptions=True,
            )
            raise
        self._configured = True
        return results

    async def adapt(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        steps: int | None = None,
        operation_id_start: int = 1,
        attempt_prefix: str = "remote-ttt",
    ) -> list[RemoteTTTStepResult]:
        """Run bounded remote forward/backward/optimizer steps.

        ``input_ids`` and ``labels`` are copied to the final/first stage over
        the authenticated tensor channel.  The final stage computes the loss
        locally, so the gateway never materializes model logits.
        """
        _validate_batch(input_ids, labels)
        _validate_client_metadata(
            self.clients,
            input_ids,
            labels,
            ignore_index=self.cfg.ignore_index,
        )
        n_steps = self.cfg.max_steps if steps is None else steps
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("steps must be a positive integer")
        if (
            isinstance(operation_id_start, bool)
            or not isinstance(operation_id_start, int)
            or operation_id_start < 0
            or operation_id_start + n_steps > _MAX_UINT32
        ):
            raise ValueError("operation_id_start and steps must fit in uint32")
        if not attempt_prefix or len(attempt_prefix) > 200:
            raise ValueError("attempt_prefix must be non-empty and reasonably short")
        if not self._configured:
            await self.configure(
                batch_size=int(input_ids.shape[0]),
                sequence_length=int(input_ids.shape[1]),
            )

        results: list[RemoteTTTStepResult] = []
        for local_step in range(n_steps):
            operation_id = operation_id_start + local_step
            attempt_id = f"{attempt_prefix}_{self._task_steps}"
            if len(attempt_id) > 256:
                raise ValueError("generated attempt_id exceeds the RPC limit")

            loss: float
            n_valid: int
            try:
                loss, n_valid = await self._forward(
                    input_ids,
                    labels,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                )
            except BaseException:
                await self._abort(operation_id, attempt_id)
                raise

            try:
                await self._backward(operation_id=operation_id, attempt_id=attempt_id)
            except BaseException:
                await self._abort(operation_id, attempt_id)
                raise

            try:
                norms = await asyncio.gather(
                    *(client.gradient_norm() for client in self.clients)
                )
                grad_norm = _global_norm(norms)
                if not math.isfinite(grad_norm):
                    # ``skip_update`` is explicit because a later stage may be
                    # the only one reporting non-finite gradients.  Every
                    # stage must discard its local gradient so a bad update
                    # cannot leak into the next puzzle.
                    await asyncio.gather(
                        *(
                            client.optimizer_step(skip_update=True)
                            for client in self.clients
                        )
                    )
                    # The global norm is already non-finite; the explicit
                    # remote skip makes every stage discard its gradients.
                    overflow = True
                else:
                    clip_coef = (
                        min(1.0, self.cfg.max_grad_norm / grad_norm)
                        if grad_norm > 0
                        else 1.0
                    )
                    overflow_results = await asyncio.gather(
                        *(
                            client.optimizer_step(clip_coef=clip_coef)
                            for client in self.clients
                        )
                    )
                    overflow = any(overflow_results)
            except BaseException:
                await self._abort(operation_id, attempt_id)
                raise

            results.append(
                RemoteTTTStepResult(
                    step=self._task_steps + 1,
                    loss=loss,
                    grad_norm=grad_norm,
                    n_valid_tokens=n_valid,
                    overflow=overflow,
                )
            )
            # Count a step as soon as its optimizer boundary has completed.
            # If a later step fails, callers can see exactly how much work
            # survived and can choose reset/reconnect policy explicitly.
            self._task_steps += 1

        return results

    async def reset_task(self) -> None:
        """Restore every stage's configure-time adapter baseline."""
        if not self._configured:
            return
        await asyncio.gather(*(client.reset_training() for client in self.clients))
        self._task_steps = 0

    async def _forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        operation_id: int,
        attempt_id: str,
    ) -> tuple[float, int]:
        hidden: torch.Tensor | None = None
        final_loss: float | None = None
        final_valid = 0
        for stage_index, client in enumerate(self.clients):
            is_last = stage_index == len(self.clients) - 1
            result = await client.forward_train(
                hidden,
                input_ids=input_ids if stage_index == 0 else None,
                labels=labels if is_last else None,
                operation_id=operation_id,
                attempt_id=attempt_id,
                ignore_index=self.cfg.ignore_index,
            )
            if is_last:
                if result.loss is None:
                    raise RuntimeError("final remote stage did not return a training loss")
                final_loss = result.loss
                final_valid = result.n_valid_tokens
                if result.output is not None:
                    raise RuntimeError("final remote stage unexpectedly returned logits")
            else:
                if result.output is None:
                    raise RuntimeError("non-final remote stage omitted hidden output")
                hidden = result.output
        assert final_loss is not None  # the non-empty client check guarantees this
        return final_loss, final_valid

    async def _backward(self, *, operation_id: int, attempt_id: str) -> None:
        grad: torch.Tensor | None = None
        for client_index in range(len(self.clients) - 1, -1, -1):
            result = await self.clients[client_index].backward_train(
                grad,
                operation_id=operation_id,
                attempt_id=attempt_id,
            )
            if client_index > 0:
                if result.grad_input is None:
                    raise RuntimeError(
                        f"remote stage {client_index} did not return a boundary gradient"
                    )
                grad = result.grad_input
            else:
                if result.grad_input is not None:
                    raise RuntimeError("stage 0 must not return a previous-stage gradient")

    async def _abort(self, operation_id: int, attempt_id: str) -> None:
        await asyncio.gather(
            *(
                client.abort_training(operation_id, attempt_id)
                for client in self.clients
            ),
            return_exceptions=True,
        )


def _validate_batch(input_ids: torch.Tensor, labels: torch.Tensor) -> None:
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("input_ids and labels must be torch.Tensor instances")
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape [batch, sequence]")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids and labels must be non-empty")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must use int32 or int64 dtype")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must use int32 or int64 dtype")


def _validate_client_metadata(
    clients: Sequence[StageRpcClient],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> None:
    """Reject bad IDs before a remote stage creates an autograd graph."""
    vocab_sizes = {
        int(client.vocab_size)
        for client in clients
        if client.vocab_size is not None
    }
    if len(vocab_sizes) > 1:
        raise ValueError(f"remote stages disagree on vocab_size: {sorted(vocab_sizes)}")
    if vocab_sizes:
        vocab_size = next(iter(vocab_sizes))
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= vocab_size:
            raise ValueError("input_ids contain a token outside the remote vocabulary")
        invalid_labels = (labels != ignore_index) & (
            (labels < 0) | (labels >= vocab_size)
        )
        if bool(invalid_labels.any()):
            raise ValueError("labels contain a token outside the remote vocabulary")

    contexts = {
        int(client.max_position_embeddings)
        for client in clients
        if client.max_position_embeddings is not None
    }
    if len(contexts) > 1:
        raise ValueError(
            "remote stages disagree on max_position_embeddings: "
            f"{sorted(contexts)}"
        )
    if contexts:
        context_limit = next(iter(contexts))
    else:
        context_limit = None
    if context_limit is not None and input_ids.shape[1] > context_limit:
        raise ValueError(
            "remote TTT sequence length exceeds max_position_embeddings: "
            f"{input_ids.shape[1]} > {context_limit}"
        )


def _validate_optional_shape(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")


def _global_norm(norms: Sequence[float]) -> float:
    squared = 0.0
    for value in norms:
        if not math.isfinite(value):
            return math.inf
        if value < 0:
            raise ValueError("stage gradient norms must be non-negative")
        squared += value * value
    return math.sqrt(squared)
