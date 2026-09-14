"""
Continuous / dynamic batching for the portable pipeline.

Design:
  - Up to `max_batch_size` sequences are admitted and multiplexed in a
    round-robin decode loop.  This is continuous admission/cache management;
    the current portable reference does not pretend to fuse unrelated KV
    caches into one tensor batch.
  - New requests join the active set on the next decode iteration when a slot is free.
  - Sequences that finish are evicted from active; their result queues persist
    until all tokens are consumed by stream().
  - seq_id → Queue is registered in _queues at submit() time so stream() can
    always find its queue regardless of eviction timing.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

import torch

from meshgpu.backends.portable.pipeline import pipeline_decode_step, pipeline_prefill
from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.inference.sampling import SamplingParams, make_sampling_generator
from meshgpu.inference.session import SessionResult, _sample

log = logging.getLogger(__name__)


@dataclass
class _Sequence:
    seq_id: str
    prompt_ids: list[int]
    sampling: SamplingParams
    max_new_tokens: int
    result_queue: asyncio.Queue
    generator: torch.Generator | None = None
    generator_device: torch.device | None = None
    generated: list[int] = field(default_factory=list)
    seq_num: int = 0
    finished: bool = False
    cancelled: bool = False
    started_at: float = field(default_factory=time.time)

    def last_token(self) -> int:
        return self.generated[-1] if self.generated else self.prompt_ids[-1]

    @property
    def total_generated(self) -> int:
        return len(self.generated)


class ContinuousBatcher:
    """
    Drives a decode loop across multiple concurrent sequences.

    Usage:
        batcher = ContinuousBatcher(workers, max_batch_size=8)
        batcher.start()
        seq_id = await batcher.submit(prompt_ids, sampling, max_new_tokens)
        async for result in batcher.stream(seq_id):
            yield result
        batcher.stop()
    """

    def __init__(
        self,
        workers: list[StageWorker],
        max_batch_size: int = 8,
    ) -> None:
        if not workers:
            raise ValueError("workers must not be empty")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._workers = workers
        self._max_batch = max_batch_size
        self._active: dict[str, _Sequence] = {}
        self._pending: asyncio.Queue[_Sequence] = asyncio.Queue()
        self._pending_by_id: dict[str, _Sequence] = {}
        # seq_id → result queue; persists even after eviction so stream() works
        self._queues: dict[str, asyncio.Queue] = {}
        self._stream_consumers: set[str] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._stopping = False
        self._loop_task: asyncio.Task | None = None
        # ``asyncio.to_thread`` cannot be forcefully stopped.  Keep task
        # handles so stop() can cancel the scheduler, drain in-flight model
        # calls, and only then release their KV caches.
        self._inflight: set[asyncio.Task] = set()
        self._op_seq = 0

    def _next_op(self) -> int:
        self._op_seq += 1
        return self._op_seq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        if self._loop_task is not None and not self._loop_task.done():
            raise RuntimeError(
                "batcher is still stopping; await wait_stopped() before restarting"
            )
        self._running = True
        self._stopping = False
        self._loop_task = asyncio.ensure_future(self._decode_loop())

    def stop(self) -> None:
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._stopping = True
        # Wake every consumer.  Without a terminal item, a caller awaiting
        # ``stream()`` would hang forever when the decode task is cancelled.
        for seq in list(self._active.values()):
            if not seq.finished:
                seq.finished = True
                seq.cancelled = True
                seq.result_queue.put_nowait(
                    SessionResult(token_id=-1, seq_num=seq.seq_num, is_last=True)
                )
        while True:
            try:
                seq = self._pending.get_nowait()
            except asyncio.QueueEmpty:
                break
            seq.finished = True
            seq.cancelled = True
            self._pending_by_id.pop(seq.seq_id, None)
            seq.result_queue.put_nowait(
                SessionResult(token_id=-1, seq_num=seq.seq_num, is_last=True)
            )
        if self._loop_task:
            self._loop_task.cancel()

    async def wait_stopped(self) -> None:
        """Wait until a canceled decode loop has drained its worker threads."""
        task = self._loop_task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def submit(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        max_new_tokens: int,
    ) -> str:
        """Submit a new request; returns seq_id for streaming."""
        if not self._running:
            raise RuntimeError("batcher is not running; call start() first")
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        model_cfg = getattr(self._workers[0]._model, "cfg", None)
        vocab_size = getattr(model_cfg, "vocab_size", None)
        prompt_ids = list(prompt_ids)
        if vocab_size is not None and any(
            token < 0 or token >= vocab_size for token in prompt_ids
        ):
            raise ValueError("prompt contains an out-of-vocabulary token")
        max_positions = getattr(model_cfg, "max_position_embeddings", None)
        if max_positions is not None and len(prompt_ids) + max_new_tokens > max_positions:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds model context {max_positions}"
            )
        seq_id = uuid.uuid4().hex[:12]
        q: asyncio.Queue = asyncio.Queue()
        seq = _Sequence(
            seq_id=seq_id,
            prompt_ids=prompt_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            result_queue=q,
        )
        # Queue registration and the running check share the same lock as
        # cancellation.  This prevents a stream from closing between the
        # check and enqueue, leaving an unowned request in the pending queue.
        async with self._lock:
            if not self._running:
                raise RuntimeError("batcher is not running; call start() first")
            self._queues[seq_id] = q
            self._pending_by_id[seq_id] = seq
            await self._pending.put(seq)
        return seq_id

    async def stream(self, seq_id: str):
        """Async generator: yields SessionResult for each generated token."""
        # Wait for the queue to be registered (submit() does it synchronously)
        q = self._queues.get(seq_id)
        if q is None:
            raise RuntimeError(f"seq {seq_id} not found — call submit() first")

        async with self._lock:
            if seq_id in self._stream_consumers:
                raise RuntimeError(f"seq {seq_id} already has an active stream consumer")
            self._stream_consumers.add(seq_id)
        completed = False
        try:
            while True:
                result: SessionResult = await q.get()
                yield result
                if result.is_last:
                    completed = True
                    self._queues.pop(seq_id, None)
                    break
        finally:
            if not completed:
                await self._cancel_sequence(seq_id)
                self._queues.pop(seq_id, None)
            async with self._lock:
                self._stream_consumers.discard(seq_id)

    # ------------------------------------------------------------------
    # Decode loop
    # ------------------------------------------------------------------

    async def _decode_loop(self) -> None:
        try:
            while self._running:
                await self._admit_pending()

                if not self._active:
                    await asyncio.sleep(0.01)
                    continue

                finished_ids = []
                for seq_id, seq in list(self._active.items()):
                    if seq.finished:
                        finished_ids.append(seq_id)
                        continue
                    result = await self._run_sync(self._decode_one_sync, seq)
                    if result is not None and not seq.cancelled:
                        await seq.result_queue.put(result)
                    if seq.finished:
                        finished_ids.append(seq_id)

                async with self._lock:
                    for sid in finished_ids:
                        finished_seq = self._active.get(sid)
                        if finished_seq is not None:
                            del self._active[sid]
                            self._clear_sequence_cache(finished_seq)
        finally:
            # A canceled asyncio wrapper does not stop the executor thread.
            # Drain every outstanding call before touching caches.  This is
            # the same state-ownership rule used by the stage RPC server.
            try:
                if self._inflight:
                    await asyncio.gather(*tuple(self._inflight), return_exceptions=True)
                for seq in list(self._active.values()):
                    self._clear_sequence_cache(seq)
                self._active.clear()
            finally:
                self._running = False
                self._stopping = False

    async def _admit_pending(self) -> None:
        for _ in range(self._max_batch):
            # Dequeue, remove from the pending index, and publish active state
            # as one critical section.  Otherwise stream().aclose() can cancel
            # after the queue pop but before _active is populated, leaving an
            # unowned request whose KV cache is never reclaimed.
            async with self._lock:
                if len(self._active) >= self._max_batch:
                    break
                try:
                    seq = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._pending_by_id.pop(seq.seq_id, None)
                if seq.cancelled:
                    continue
                self._active[seq.seq_id] = seq

            # Prefill is blocking and must not run while holding the lock;
            # stream cancellation can now mark the active sequence finished
            # while this call is in flight, and the result is discarded below.

            result = await self._run_sync(self._prefill_sync, seq)
            if result is not None and not seq.cancelled:
                await seq.result_queue.put(result)

    async def _cancel_sequence(self, seq_id: str) -> None:
        """Cancel a pending/active sequence and let the loop clean its cache."""
        async with self._lock:
            sequence = self._active.get(seq_id) or self._pending_by_id.get(seq_id)
            if sequence is None:
                return
            sequence.cancelled = True
            sequence.finished = True
            self._pending_by_id.pop(seq_id, None)

    async def _run_sync(self, function, *args):
        """Run a blocking stage operation and retain it across cancellation."""
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._inflight.add(task)

        def discard(done: asyncio.Task) -> None:
            self._inflight.discard(done)

        task.add_done_callback(discard)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight.discard(task)

    # ------------------------------------------------------------------
    # Sync ops called via asyncio.to_thread
    # ------------------------------------------------------------------

    def _prefill_sync(self, seq: _Sequence) -> SessionResult | None:
        try:
            input_ids = torch.tensor([seq.prompt_ids], dtype=torch.long)
            logits = pipeline_prefill(
                self._workers, input_ids,
                operation_id=self._next_op(),
                attempt_id=f"pf_{seq.seq_id}",
                cache_key=f"seq:{seq.seq_id}",
            )
            next_id = int(
                _sample(
                    logits[:, -1, :],
                    seq.sampling,
                    self._generator_for(seq, logits),
                )
            )
            seq.generated.append(next_id)
            seq.seq_num += 1
            is_last = (
                (seq.sampling.eos_token_id is not None and next_id == seq.sampling.eos_token_id)
                or seq.total_generated >= seq.max_new_tokens
            )
            seq.finished = is_last
            if is_last:
                self._clear_sequence_cache(seq)
            return SessionResult(token_id=next_id, seq_num=seq.seq_num, is_last=is_last)
        except Exception:
            log.exception("prefill failed for seq %s", seq.seq_id)
            self._clear_sequence_cache(seq)
            seq.finished = True
            return SessionResult(token_id=-1, seq_num=0, is_last=True)

    def _decode_one_sync(self, seq: _Sequence) -> SessionResult | None:
        if seq.finished:
            return None
        try:
            next_input = torch.tensor([[seq.last_token()]], dtype=torch.long)
            logits = pipeline_decode_step(
                self._workers, next_input,
                operation_id=self._next_op(),
                attempt_id=f"dc_{seq.seq_id}",
                cache_key=f"seq:{seq.seq_id}",
            )
            next_id = int(
                _sample(
                    logits[:, -1, :],
                    seq.sampling,
                    self._generator_for(seq, logits),
                )
            )
            seq.generated.append(next_id)
            seq.seq_num += 1
            is_last = (
                (seq.sampling.eos_token_id is not None and next_id == seq.sampling.eos_token_id)
                or seq.total_generated >= seq.max_new_tokens
            )
            seq.finished = is_last
            if is_last:
                self._clear_sequence_cache(seq)
            return SessionResult(token_id=next_id, seq_num=seq.seq_num, is_last=is_last)
        except Exception:
            log.exception("decode failed for seq %s", seq.seq_id)
            self._clear_sequence_cache(seq)
            seq.finished = True
            return SessionResult(token_id=-1, seq_num=seq.seq_num, is_last=True)

    def _clear_sequence_cache(self, seq: _Sequence) -> None:
        for worker in self._workers:
            try:
                worker.clear_kv(f"seq:{seq.seq_id}")
            except Exception as exc:
                # Cleanup must not kill the scheduler or hide the original
                # inference result.  The next request gets a fresh namespaced
                # cache; a failed clear is surfaced in logs for operators.
                log.warning(
                    "could not clear batcher cache for seq %s on stage: %s",
                    seq.seq_id,
                    exc,
                )

    @staticmethod
    def _generator_for(seq: _Sequence, logits: torch.Tensor) -> torch.Generator | None:
        """Create one request RNG on the device where its logits are sampled."""
        if seq.sampling.seed is None:
            return None
        device = torch.device(logits.device)
        if seq.generator is None:
            seq.generator = make_sampling_generator(seq.sampling, device=device)
            seq.generator_device = device
        elif seq.generator_device != device:
            raise RuntimeError(
                "sampling logits changed device during one sequence: "
                f"{seq.generator_device} -> {device}"
            )
        return seq.generator

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()
