"""
Inference session owner.
Owns prompt token IDs, confirmed prefix, sampling state and decode loop.
One session = one active request; KV cache lives in stage workers.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import torch

from meshgpu.backends.portable.pipeline import (
    _async_stage_call,
    pipeline_decode_step_async,
    pipeline_prefill_async,
)
from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.inference.sampling import (
    SamplingParams,
    greedy_sample,
    make_sampling_generator,
    top_p_sample,
)

log = logging.getLogger(__name__)


@dataclass
class SessionResult:
    token_id: int
    seq_num: int       # monotone; client uses this to dedupe replayed tokens
    is_last: bool
    text: str = ""     # decoded text (empty until tokenizer is wired)


class InferenceSession:
    """
    Drives prefill + decode for one request through the pipeline.

    KV caches are owned by stage workers; this class owns:
      - prompt token IDs
      - generated token sequence + sequence numbers
      - sampling state
    """

    def __init__(
        self,
        session_id: str,
        prompt_ids: list[int],
        workers: list[StageWorker],
        sampling: SamplingParams,
        max_new_tokens: int,
        *,
        skip_prefill: bool = False,
        start_seq_num: int = 0,
        initial_logits: torch.Tensor | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not workers:
            raise ValueError("workers must not be empty")
        if start_seq_num < 0:
            raise ValueError("start_seq_num must be non-negative")
        model_cfg = getattr(getattr(workers[0], "_model", None), "cfg", None)
        vocab_size = getattr(model_cfg, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(workers[0], "vocab_size", None)
        if vocab_size is not None and any(
            token < 0 or token >= vocab_size for token in prompt_ids
        ):
            raise ValueError("prompt contains an out-of-vocabulary token")
        max_positions = getattr(model_cfg, "max_position_embeddings", None)
        if max_positions is None:
            max_positions = getattr(workers[0], "max_position_embeddings", None)
        if max_positions is not None and len(prompt_ids) + max_new_tokens > max_positions:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds model context {max_positions}"
            )
        self.session_id = session_id
        self._prompt_ids = list(prompt_ids)
        self._workers = workers
        self._sampling = sampling
        # The output logits may live on the final CUDA stage.  A CPU generator
        # cannot be passed to ``torch.multinomial``/``torch.rand`` on CUDA, so
        # create the seeded generator lazily on the first logits device.  The
        # device is then pinned for the lifetime of this session; silently
        # recreating it would repeat the random stream after a topology bug.
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None
        self._max_new_tokens = max_new_tokens
        self._skip_prefill = skip_prefill
        self._initial_logits = initial_logits
        self._cache_key = f"session:{session_id}"
        self._generated: list[int] = []
        self._seq_num = start_seq_num
        self._started_at = time.time()
        self._done = False
        self._run_started = False
        self._op_seq = 0

    def _next_op(self) -> int:
        self._op_seq += 1
        return self._op_seq

    async def run(self) -> AsyncGenerator[SessionResult, None]:
        """Yield one SessionResult per generated token."""
        if self._run_started:
            raise RuntimeError(f"session {self.session_id} can only be run once")
        self._run_started = True
        attempt_id = str(uuid.uuid4())[:8]

        input_ids = torch.tensor([self._prompt_ids], dtype=torch.long)
        try:
            if self._skip_prefill and self._initial_logits is not None:
                # Recovery replay already computed the distribution for the
                # token after the confirmed prefix.  Reusing it avoids
                # appending the final prefix token a second time.
                logits = self._initial_logits
            else:
                # A legacy caller may set skip_prefill without supplying the
                # replay logits.  Rebuild the prefix rather than silently
                # decoding the last prompt token twice.
                logits = await pipeline_prefill_async(
                    self._workers,
                    input_ids,
                    operation_id=self._next_op(),
                    attempt_id=attempt_id,
                    cache_key=self._cache_key,
                )

            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
                raise RuntimeError(f"pipeline returned invalid logits shape {tuple(logits.shape)}")

            # Greedy/top-p from last position
            next_id = int(
                _sample(
                    logits[:, -1, :],
                    self._sampling,
                    self._generator_for(logits),
                )
            )
            self._generated.append(next_id)
            self._seq_num += 1

            eos_token_id = self._sampling.eos_token_id
            first_is_last = (
                (eos_token_id is not None and next_id == eos_token_id)
                or len(self._generated) >= self._max_new_tokens
            )
            yield SessionResult(
                token_id=next_id,
                seq_num=self._seq_num,
                is_last=first_is_last,
            )
            if first_is_last:
                return

            # Decode loop
            for _ in range(self._max_new_tokens - 1):
                next_input = torch.tensor([[next_id]], dtype=torch.long)
                logits = await pipeline_decode_step_async(
                    self._workers,
                    next_input,
                    operation_id=self._next_op(),
                    attempt_id=attempt_id,
                    cache_key=self._cache_key,
                )

                next_id = int(
                    _sample(
                        logits[:, -1, :],
                        self._sampling,
                        self._generator_for(logits),
                    )
                )
                self._generated.append(next_id)
                self._seq_num += 1

                is_last = (
                    (eos_token_id is not None and next_id == eos_token_id)
                    or len(self._generated) >= self._max_new_tokens
                )
                yield SessionResult(
                    token_id=next_id,
                    seq_num=self._seq_num,
                    is_last=is_last,
                )

                if is_last:
                    break
        finally:
            for worker in self._workers:
                try:
                    await _clear_kv_async(worker, self._cache_key)
                except Exception as exc:
                    # The worker may be the very component that failed.  Do
                    # not replace the useful inference exception with cleanup
                    # noise; the server-side lease/recovery path owns it now.
                    log.warning(
                        "could not clear session %s cache on stage cleanup: %s",
                        self.session_id,
                        exc,
                    )
            self._done = True

    def _generator_for(self, logits: torch.Tensor) -> torch.Generator | None:
        """Return the seeded RNG compatible with the logits' physical device."""
        if self._sampling.seed is None:
            return None
        device = torch.device(logits.device)
        if self._generator is None:
            self._generator = make_sampling_generator(self._sampling, device=device)
            self._generator_device = device
        elif self._generator_device != device:
            raise RuntimeError(
                "sampling logits changed device during one session: "
                f"{self._generator_device} -> {device}"
            )
        return self._generator

    @property
    def generated_ids(self) -> list[int]:
        return list(self._generated)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._started_at

    @property
    def tokens_per_second(self) -> float:
        n = len(self._generated)
        t = self.elapsed_s
        return n / t if t > 0 else 0.0


def _sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    if params.temperature == 0.0:
        return int(greedy_sample(logits))
    return int(
        top_p_sample(
            logits,
            params.temperature,
            params.top_p,
            generator=generator,
        )
    )


async def _clear_kv_async(worker: StageWorker, cache_key: str) -> None:
    """Clear either a local synchronous worker or an async RPC stage."""
    await _async_stage_call(worker, "clear_kv", cache_key)
