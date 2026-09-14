"""
Portable pipeline coordinator — orchestrates N stages for inference and training.
Phase 1: in-process, single machine (correctness reference).
Phase 2: cross-machine via transport layer.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from typing import Any

import torch

from meshgpu.backends.portable.stage_worker import OptimizerLike, StageContext, StageWorker
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage

log = logging.getLogger(__name__)


def build_pipeline_from_manifest(
    manifest_or_dir,
    *,
    artifact_root=None,
    devices: list[torch.device] | None = None,
    job_id: str = "local",
    transport: str = "cpu",
    activation_checkpointing: bool = False,
    layer_ranges: Sequence[tuple[int, int]] | None = None,
) -> list[StageWorker]:
    """
    Load a saved manifest (written by import_from_hf) and return pipeline workers
    with real weights.  Delegates to artifacts.hf_import.build_pipeline_from_manifest.
    """
    from meshgpu.artifacts.hf_import import build_pipeline_from_manifest as _impl
    return _impl(
        manifest_or_dir,
        artifact_root=artifact_root,
        devices=devices,
        job_id=job_id,
        transport=transport,
        activation_checkpointing=activation_checkpointing,
        layer_ranges=layer_ranges,
    )


def build_pipeline(
    cfg: LlamaConfig,
    num_stages: int,
    devices: list[torch.device],
    job_id: str = "local",
    transport: str = "cpu",
    activation_checkpointing: bool = False,
) -> list[StageWorker]:
    """
    Partition cfg.num_hidden_layers evenly across num_stages.
    Returns list of StageWorker in stage order.
    """
    if num_stages < 1 or num_stages > cfg.num_hidden_layers:
        raise ValueError(
            f"num_stages must be in [1, {cfg.num_hidden_layers}], got {num_stages}"
        )
    if len(devices) != num_stages:
        raise ValueError(f"need {num_stages} devices, got {len(devices)}")

    n = cfg.num_hidden_layers
    base, rem = divmod(n, num_stages)
    boundaries: list[tuple[int, int]] = []
    start = 0
    for i in range(num_stages):
        end = start + base + (1 if i < rem else 0)
        boundaries.append((start, end))
        start = end

    workers = []
    for i, (ls, le) in enumerate(boundaries):
        is_first = i == 0
        is_last = i == num_stages - 1
        stage = LlamaStage(
            cfg,
            layer_start=ls,
            layer_end=le,
            has_embedding=is_first,
            has_lm_head=is_last,
            device=devices[i],
            activation_checkpointing=activation_checkpointing,
        )
        ctx = StageContext(
            stage_id=i,
            layer_start=ls,
            layer_end=le,
            device=devices[i],
            is_first=is_first,
            is_last=is_last,
            job_id=job_id,
            attempt_id="local",
            transport=transport,
        )
        workers.append(StageWorker(stage, ctx))
        log.debug("stage %d: layers [%d, %d) on %s", i, ls, le, devices[i])

    return workers


# ---------------------------------------------------------------------------
# In-process pipeline inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def pipeline_prefill(
    workers: list[StageWorker],
    input_ids: torch.Tensor,
    *,
    operation_id: int = 1,
    attempt_id: str = "a0",
    cache_key: str | None = None,
) -> torch.Tensor:
    """
    Run prefill through all stages in sequence.
    Returns logits [B, S, vocab] from last stage.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    B, S = input_ids.shape
    if B < 1 or S < 1:
        raise ValueError("input_ids must have a non-empty batch and sequence")
    if not workers:
        raise ValueError("workers must not be empty")
    max_positions = getattr(workers[0]._model, "cfg", None)
    max_positions = getattr(max_positions, "max_position_embeddings", None)
    if max_positions is not None and S > max_positions:
        raise ValueError(
            f"prefill sequence length {S} exceeds model context "
            f"{max_positions}"
        )
    hidden = None

    # Clear this request's cache context upfront so replayed prefill doesn't
    # accumulate old KV.  Named contexts are essential when the same stage
    # workers serve multiple HTTP sessions concurrently.
    for w in workers:
        w.clear_kv(cache_key)

    try:
        for i, worker in enumerate(workers):
            result = worker.forward_inference(
                hidden,
                input_ids=input_ids if i == 0 else None,
                operation_id=operation_id,
                attempt_id=attempt_id,
                reset_kv=False,  # already cleared above
                cache_key=cache_key,
            )
            hidden = result.output
    except BaseException:
        # A stage may have appended K/V before a downstream stage fails.  Do
        # not leave a partially-prefilled request behind for a retry.
        _clear_kv_best_effort(workers, cache_key)
        raise

    assert hidden is not None  # guarded by the non-empty workers check
    return hidden  # logits at last stage


@torch.no_grad()
def pipeline_decode_step(
    workers: list[StageWorker],
    next_token: torch.Tensor,
    *,
    operation_id: int,
    attempt_id: str = "a0",
    cache_key: str | None = None,
) -> torch.Tensor:
    """
    Run one decode step; KV caches already populated from prefill.
    Returns logits [B, 1, vocab].
    """
    if next_token.ndim != 2 or next_token.shape[0] < 1 or next_token.shape[1] < 1:
        raise ValueError("next_token must have shape [batch, sequence] with sequence >= 1")

    batch_size, query_len = next_token.shape
    if not workers:
        raise ValueError("workers must not be empty")
    max_positions = getattr(workers[0]._model, "cfg", None)
    max_positions = getattr(max_positions, "max_position_embeddings", None)
    cache_lengths: list[int] = []
    expected_kv_len: int | None = None
    for i, worker in enumerate(workers):
        kv_len = worker.kv_cache_length(cache_key)
        cache_lengths.append(kv_len)
        if expected_kv_len is None:
            expected_kv_len = kv_len
        elif kv_len != expected_kv_len:
            raise RuntimeError(
                "pipeline KV caches are out of sync: "
                f"stage 0 has {expected_kv_len} tokens, stage {i} has {kv_len}"
            )
        if kv_len == 0:
            raise RuntimeError(
                "pipeline_decode_step requires a prefilled KV cache; "
                "call pipeline_prefill first"
            )
        if max_positions is not None and kv_len + query_len > max_positions:
            raise ValueError(
                f"decode would reach position {kv_len + query_len - 1}, "
                f"beyond model context {max_positions}"
            )
    hidden = None
    try:
        for i, worker in enumerate(workers):
            kv_len = cache_lengths[i]
            position_ids = torch.arange(
                kv_len,
                kv_len + query_len,
                dtype=torch.long,
                device=next_token.device,
            ).unsqueeze(0).expand(batch_size, -1)
            result = worker.forward_inference(
                hidden,
                input_ids=next_token if i == 0 else None,
                position_ids=position_ids,
                operation_id=operation_id,
                attempt_id=attempt_id,
                reset_kv=False,
                cache_key=cache_key,
            )
            hidden = result.output
    except BaseException:
        _restore_kv_lengths_best_effort(workers, cache_lengths, cache_key)
        raise
    assert hidden is not None  # guarded by the non-empty workers check
    return hidden


# ---------------------------------------------------------------------------
# Async pipeline for local and remote stage workers
# ---------------------------------------------------------------------------

async def pipeline_prefill_async(
    workers: list[Any],
    input_ids: torch.Tensor,
    *,
    operation_id: int = 1,
    attempt_id: str = "a0",
    cache_key: str | None = None,
) -> torch.Tensor:
    """Run prefill when stages may be local ``StageWorker`` or RPC clients."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have a non-empty batch and sequence")
    if not workers:
        raise ValueError("workers must not be empty")
    max_positions = _max_positions(workers[0])
    if max_positions is not None and input_ids.shape[1] > max_positions:
        raise ValueError(
            f"prefill sequence length {input_ids.shape[1]} exceeds model context "
            f"{max_positions}"
        )

    for worker in workers:
        await _async_stage_call(worker, "clear_kv", cache_key)

    hidden: torch.Tensor | None = None
    try:
        for index, worker in enumerate(workers):
            result = await _async_stage_call(
                worker,
                "forward_inference",
                hidden,
                input_ids=input_ids if index == 0 else None,
                operation_id=operation_id,
                attempt_id=attempt_id,
                reset_kv=False,
                cache_key=cache_key,
            )
            hidden = result.output
    except BaseException:
        await _clear_kv_async_best_effort(workers, cache_key)
        raise
    assert hidden is not None  # guarded by the non-empty workers check
    return hidden


async def pipeline_decode_step_async(
    workers: list[Any],
    next_token: torch.Tensor,
    *,
    operation_id: int,
    attempt_id: str = "a0",
    cache_key: str | None = None,
) -> torch.Tensor:
    """Run one decode step across local or remote stage workers."""
    if next_token.ndim != 2 or next_token.shape[0] < 1 or next_token.shape[1] < 1:
        raise ValueError("next_token must have shape [batch, sequence] with sequence >= 1")
    if not workers:
        raise ValueError("workers must not be empty")
    batch_size, query_len = next_token.shape
    max_positions = _max_positions(workers[0])
    cache_lengths: list[int] = []
    expected_kv_len: int | None = None
    for index, worker in enumerate(workers):
        kv_len = int(await _async_stage_call(worker, "kv_cache_length", cache_key))
        cache_lengths.append(kv_len)
        if expected_kv_len is None:
            expected_kv_len = kv_len
        elif kv_len != expected_kv_len:
            raise RuntimeError(
                "pipeline KV caches are out of sync: "
                f"stage 0 has {expected_kv_len} tokens, stage {index} has {kv_len}"
            )
        if kv_len == 0:
            raise RuntimeError(
                "pipeline_decode_step_async requires a prefilled KV cache; "
                "call pipeline_prefill_async first"
            )
        if max_positions is not None and kv_len + query_len > max_positions:
            raise ValueError(
                f"decode would reach position {kv_len + query_len - 1}, "
                f"beyond model context {max_positions}"
            )
    hidden: torch.Tensor | None = None
    try:
        for index, worker in enumerate(workers):
            kv_len = cache_lengths[index]
            position_ids = torch.arange(
                kv_len,
                kv_len + query_len,
                dtype=torch.long,
                device=next_token.device,
            ).unsqueeze(0).expand(batch_size, -1)
            result = await _async_stage_call(
                worker,
                "forward_inference",
                hidden,
                input_ids=next_token if index == 0 else None,
                position_ids=position_ids,
                operation_id=operation_id,
                attempt_id=attempt_id,
                reset_kv=False,
                cache_key=cache_key,
            )
            hidden = result.output
    except BaseException:
        await _restore_kv_lengths_async_best_effort(workers, cache_lengths, cache_key)
        raise
    assert hidden is not None  # guarded by the non-empty workers check
    return hidden


async def _async_stage_call(worker: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a stage method without abandoning synchronous work on cancellation.

    Local ``StageWorker`` methods run in an executor because they are
    synchronous.  Cancelling the coroutine returned by ``to_thread`` does not
    stop the underlying thread, though.  Shield the executor task and drain it
    after cancellation so a request cleanup cannot race a still-running stage
    call (in particular, a call that is mutating a KV cache).
    """
    method = getattr(worker, method_name)
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # The executor thread cannot be cancelled safely.  Drain its result
        # before propagating cancellation, otherwise the caller may clear or
        # reuse state while the abandoned method is still touching it.
        try:
            await asyncio.shield(task)
        except BaseException:
            # Preserve the original cancellation even if the stage call
            # itself failed while we were draining it.
            pass
        raise


def _clear_kv_best_effort(workers: Sequence[Any], cache_key: str | None) -> None:
    """Clear local inference state without masking the original exception."""
    for worker in workers:
        try:
            worker.clear_kv(cache_key)
        except Exception as exc:
            log.warning("could not roll back stage KV cache: %s", exc)


def _restore_kv_lengths_best_effort(
    workers: Sequence[Any],
    lengths: Sequence[int],
    cache_key: str | None,
) -> None:
    """Restore a partially-mutated local decode to its pre-operation lengths."""
    for worker, length in zip(workers, lengths):
        try:
            if length == 0:
                worker.clear_kv(cache_key)
            else:
                worker.trim_kv(length, cache_key)
        except Exception as exc:
            log.warning("could not roll back stage KV cache to %d: %s", length, exc)


async def _clear_kv_async_best_effort(
    workers: Sequence[Any],
    cache_key: str | None,
) -> None:
    """Async counterpart of ``_clear_kv_best_effort`` for local/RPC stages."""
    for worker in workers:
        try:
            await _async_stage_call(worker, "clear_kv", cache_key)
        except Exception as exc:
            log.warning("could not roll back remote stage KV cache: %s", exc)


async def _restore_kv_lengths_async_best_effort(
    workers: Sequence[Any],
    lengths: Sequence[int],
    cache_key: str | None,
) -> None:
    """Restore a partially-mutated async decode to its pre-operation lengths."""
    for worker, length in zip(workers, lengths):
        try:
            if length == 0:
                await _async_stage_call(worker, "clear_kv", cache_key)
            else:
                await _async_stage_call(worker, "trim_kv", length, cache_key)
        except Exception as exc:
            log.warning("could not roll back remote stage KV cache to %d: %s", length, exc)


def _max_positions(worker: Any) -> int | None:
    model = getattr(worker, "_model", None)
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        return getattr(cfg, "max_position_embeddings", None)
    return getattr(worker, "max_position_embeddings", None)


# ---------------------------------------------------------------------------
# In-process pipeline training (GPipe-style flush schedule)
# ---------------------------------------------------------------------------

def pipeline_train_step(
    workers: list[StageWorker],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizers: Sequence[OptimizerLike],
    *,
    operation_id: int = 1,
    attempt_id: str = "a0",
    ignore_index: int = -100,
    loss_normalizer: int | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    step_optimizers: bool = True,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one training microbatch and leave no partial operation on failure."""
    if any(operation_id in worker._saved for worker in workers):
        raise RuntimeError(
            f"operation_id={operation_id} already has an active training forward"
        )
    try:
        return _pipeline_train_step_impl(
            workers,
            input_ids,
            labels,
            optimizers,
            operation_id=operation_id,
            attempt_id=attempt_id,
            ignore_index=ignore_index,
            loss_normalizer=loss_normalizer,
            gradient_scaler=gradient_scaler,
            step_optimizers=step_optimizers,
            max_grad_norm=max_grad_norm,
        )
    except BaseException:
        _cleanup_training_operation(workers, operation_id)
        raise


def _pipeline_train_step_impl(
    workers: list[StageWorker],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizers: Sequence[OptimizerLike],
    *,
    operation_id: int = 1,
    attempt_id: str = "a0",
    ignore_index: int = -100,
    loss_normalizer: int | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    step_optimizers: bool = True,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """
    One microbatch forward + backward + optimizer step across all stages.
    Loss normalized by valid (non-ignored) tokens.
    Returns dict with loss and grad norm.
    """
    if not workers:
        raise ValueError("workers must not be empty")
    if len(optimizers) != len(workers):
        raise ValueError(
            f"need one optimizer per worker, got {len(optimizers)} for {len(workers)}"
        )
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must have shape [batch, sequence]")
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have the same shape, "
            f"got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if loss_normalizer is not None and loss_normalizer < 1:
        raise ValueError("loss_normalizer must be positive when provided")

    sync_tied_parameters(workers)

    # Forward — pass detached boundary tensors between stages (simulates transport)
    hidden = None
    for i, worker in enumerate(workers):
        result = worker.forward_train(
            hidden,
            input_ids=input_ids if i == 0 else None,
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
        if not worker._ctx.is_last:
            # The next StageWorker owns the destination-device transfer and
            # creates the leaf used for inter-stage backward.  Moving here
            # would first bounce a CPU boundary back to the source GPU and
            # defeat the explicit CPU fallback; with local_cuda it would also
            # prevent the next stage from selecting the peer/host path.
            hidden = result.output.detach()

    # Loss: use the last stage's saved output tensor — it still has grad_fn
    last_saved = workers[-1]._saved[operation_id]
    logits: torch.Tensor = last_saved["out"]  # [B, S, V] on device, has grad_fn
    labels_dev = labels.to(workers[-1]._ctx.device)

    valid_mask = labels_dev != ignore_index
    n_valid = int(valid_mask.sum().item())
    denominator = max(n_valid if loss_normalizer is None else loss_normalizer, 1)

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels_dev.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    ) / denominator

    if gradient_scaler is not None:
        gradient_scaler.scale(loss).backward()
    else:
        loss.backward()

    # Retrieve gradient at the last stage's input boundary (the detached hidden from prev stage)
    last_x = last_saved["x"]
    grad = last_x.grad if (not workers[-1]._ctx.is_first and last_x.requires_grad) else None
    workers[-1]._saved.pop(operation_id, None)
    workers[-1].clear_kv()

    # Backward through earlier stages (last-1 → 0)
    for i in range(len(workers) - 2, -1, -1):
        bwd = workers[i].backward_train(
            grad,
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
        grad = bwd.grad_input

    _accumulate_tied_embedding_grads(workers)

    if step_optimizers:
        if max_grad_norm is not None:
            if max_grad_norm <= 0:
                raise ValueError("max_grad_norm must be positive when provided")
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for worker in workers
                    for parameter in worker._model.parameters()
                    if parameter.grad is not None and parameter.requires_grad
                ],
                max_grad_norm,
            )
        overflow = False
        if gradient_scaler is not None:
            # A scaler must unscale every optimizer before stepping and must be
            # updated exactly once for the whole pipeline step.  Calling
            # ``GradScaler.update`` once per stage would make the scale depend
            # on topology and can make later stages observe a different scale.
            if not all(isinstance(opt, torch.optim.Optimizer) for opt in optimizers):
                raise TypeError(
                    "gradient_scaler requires real torch.optim.Optimizer instances "
                    "when step_optimizers=True"
                )
            real_optimizers = [
                optimizer
                for optimizer in optimizers
                if isinstance(optimizer, torch.optim.Optimizer)
            ]
            for optimizer in real_optimizers:
                gradient_scaler.unscale_(optimizer)
            overflow = not _all_gradients_finite(workers)
            if not overflow:
                for optimizer in real_optimizers:
                    gradient_scaler.step(optimizer)
            gradient_scaler.update()
            for optimizer in real_optimizers:
                optimizer.zero_grad(set_to_none=True)
            if overflow:
                log.warning("pipeline training step overflow; all optimizer steps skipped")
        else:
            for worker, stage_optimizer in zip(workers, optimizers):
                worker.optimizer_step(stage_optimizer, scaler=None)
        if not overflow:
            sync_tied_parameters(workers)

    return {"loss": loss.item(), "n_valid_tokens": n_valid}


def _all_gradients_finite(workers: list[StageWorker]) -> bool:
    """Check all stage gradients before a scaler updates any optimizer."""
    for worker in workers:
        for parameter in worker._model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                return False
    return True


def _cleanup_training_operation(
    workers: list[StageWorker],
    operation_id: int,
) -> None:
    """Drop saved autograd graphs and default training KV after a failed step."""
    for worker in workers:
        worker._saved.pop(operation_id, None)
        worker.clear_kv()


def _tied_embedding_pair(
    workers: list[StageWorker],
) -> tuple[torch.nn.Parameter, torch.nn.Parameter] | None:
    if len(workers) < 2:
        return None
    cfg = getattr(workers[0]._model, "cfg", None)
    if not getattr(cfg, "tie_word_embeddings", False):
        return None
    first = getattr(workers[0]._model, "embed_tokens", None)
    last = getattr(workers[-1]._model, "lm_head", None)
    if first is None or last is None:
        return None
    return first.weight, last.weight


def sync_tied_parameters(workers: list[StageWorker]) -> None:
    """Keep replicated tied embedding/lm-head weights byte-identical."""
    pair = _tied_embedding_pair(workers)
    if pair is None:
        return
    embedding, lm_head = pair
    with torch.no_grad():
        lm_head.copy_(embedding.to(lm_head.device, dtype=lm_head.dtype))


def _accumulate_tied_embedding_grads(workers: list[StageWorker]) -> None:
    """Aggregate input-embedding and output-head gradients for tied weights."""
    pair = _tied_embedding_pair(workers)
    if pair is None:
        return
    embedding, lm_head = pair
    if lm_head.grad is not None:
        head_grad = lm_head.grad.to(embedding.device, dtype=embedding.dtype)
        if embedding.grad is None:
            embedding.grad = head_grad
        else:
            embedding.grad.add_(head_grad)
        # The owner optimizer applies the combined update once.  The replica
        # is refreshed from it after the optimizer boundary.
        lm_head.grad = None
