"""
Inference recovery: replay a known-good token prefix to rebuild KV caches
after a stage worker is replaced.

Scenario:
  - Request was mid-generation when stage S crashed.
  - Stage S is replaced with a fresh StageWorker (loaded from checkpoint).
  - All stages need fresh KV caches because the dead worker's state is lost.

Strategy — full prefix replay (always correct):
  1. Run pipeline_prefill on the confirmed prefix tokens.
  2. All stages rebuild their KV caches in one forward pass.
  3. Decode continues from position len(prefix).

Partial replay (stages 0..S-1 already warm) is a P5 optimization.

Public API:
  replay_prefix(workers, prefix_ids) → RecoveryResult
  resume_session(workers, prefix_ids, ...) → InferenceSession (decode-ready)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch

from meshgpu.backends.portable.pipeline import pipeline_prefill, pipeline_prefill_async
from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.session import InferenceSession

log = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    ok: bool
    prefix_len: int
    error: str | None = None
    # Logits predicting the token immediately after the replayed prefix.
    # Keeping this result is what lets a resumed session continue without
    # feeding the last prefix token through the model a second time.
    next_logits: torch.Tensor | None = None


def replay_prefix(
    workers: list[StageWorker],
    prefix_ids: torch.Tensor,
    *,
    operation_id: int = 0,
    attempt_id: str = "recovery",
    cache_key: str | None = None,
) -> RecoveryResult:
    """
    Rebuild KV caches for all stages by replaying prefix_ids.
    prefix_ids: [1, T] (batch_size=1; recovery is per-request).
    """
    if not workers:
        raise ValueError("workers must not be empty")
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    if prefix_ids.shape[0] != 1:
        raise ValueError("replay_prefix only supports batch_size=1")

    try:
        next_logits = pipeline_prefill(
            workers,
            prefix_ids,
            operation_id=operation_id,
            attempt_id=attempt_id,
            cache_key=cache_key,
        )
        log.info(
            "recovery: replayed %d prefix tokens across %d stages",
            prefix_ids.shape[1], len(workers),
        )
        return RecoveryResult(
            ok=True,
            prefix_len=prefix_ids.shape[1],
            next_logits=next_logits,
        )
    except Exception as exc:
        log.exception("recovery: replay_prefix failed")
        for worker in workers:
            worker.clear_kv(cache_key)
        return RecoveryResult(ok=False, prefix_len=0, error=str(exc))


async def resume_session(
    workers: list[StageWorker],
    session_id: str,
    prefix_ids: list[int],
    max_new_tokens: int,
    sampling: SamplingParams,
    *,
    operation_id: int = 0,
    confirmed_seq_num: int = 0,
) -> InferenceSession:
    """
    Replay prefix to rebuild KV caches, then return an InferenceSession
    ready to generate new tokens from position len(prefix_ids).

    confirmed_seq_num: the highest seq_num the client already received —
    the resumed session continues from confirmed_seq_num+1 so the client
    can detect duplicate-free continuations.

    Usage:
        session = await resume_session(workers, sid, prefix, max_new, sampling)
        async for result in session.run():
            emit(result)   # seq_num continues from confirmed_seq_num+1
    """
    if not workers:
        raise ValueError("workers must not be empty")
    if not prefix_ids:
        raise ValueError("prefix_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if confirmed_seq_num < 0:
        raise ValueError("confirmed_seq_num must be non-negative")
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long)
    if _has_async_stage(workers):
        result = await replay_prefix_async(
            workers,
            prefix_tensor,
            operation_id=operation_id,
            attempt_id=f"recover_{session_id}",
            cache_key=f"session:{session_id}",
        )
    else:
        result = await asyncio.to_thread(
            replay_prefix,
            workers,
            prefix_tensor,
            operation_id=operation_id,
            attempt_id=f"recover_{session_id}",
            cache_key=f"session:{session_id}",
        )
    if not result.ok:
        raise RuntimeError(f"prefix replay failed: {result.error}")

    session = InferenceSession(
        session_id=session_id,
        prompt_ids=prefix_ids,
        workers=workers,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        skip_prefill=True,        # KV already built by replay
        start_seq_num=confirmed_seq_num,
        initial_logits=result.next_logits,
    )
    log.info(
        "recovery: session %s ready, prefix_len=%d, seq_num_offset=%d",
        session_id, result.prefix_len, confirmed_seq_num,
    )
    return session


async def replay_prefix_async(
    workers: list[StageWorker],
    prefix_ids: torch.Tensor,
    *,
    operation_id: int = 0,
    attempt_id: str = "recovery",
    cache_key: str | None = None,
) -> RecoveryResult:
    """Async counterpart used when one or more stages are remote RPC clients."""
    if not workers:
        raise ValueError("workers must not be empty")
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    if prefix_ids.shape[0] != 1:
        raise ValueError("replay_prefix only supports batch_size=1")

    try:
        next_logits = await pipeline_prefill_async(
            workers,
            prefix_ids,
            operation_id=operation_id,
            attempt_id=attempt_id,
            cache_key=cache_key,
        )
        log.info(
            "recovery: replayed %d prefix tokens across %d stages",
            prefix_ids.shape[1], len(workers),
        )
        return RecoveryResult(
            ok=True,
            prefix_len=prefix_ids.shape[1],
            next_logits=next_logits,
        )
    except Exception as exc:
        log.exception("recovery: async replay_prefix failed")
        for worker in workers:
            try:
                clear = worker.clear_kv
                if asyncio.iscoroutinefunction(clear):
                    await clear(cache_key)
                else:
                    await asyncio.to_thread(clear, cache_key)
            except Exception as cleanup_exc:
                log.warning("recovery cache cleanup failed: %s", cleanup_exc)
        return RecoveryResult(ok=False, prefix_len=0, error=str(exc))


def _has_async_stage(workers: list[StageWorker]) -> bool:
    return any(
        asyncio.iscoroutinefunction(getattr(worker, "forward_inference", None))
        for worker in workers
    )
