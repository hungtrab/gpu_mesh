"""Speculative decoding for two independent portable pipelines.

The draft and target pipelines each own a KV cache.  This implementation
keeps those caches aligned after every accept/reject decision; rejected
proposals are trimmed before the replacement token is appended.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from meshgpu.backends.portable.pipeline import pipeline_decode_step, pipeline_prefill
from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.inference.sampling import SamplingParams, make_sampling_generator

log = logging.getLogger(__name__)


@dataclass
class SpeculativeResult:
    generated_ids: list[int]
    n_draft_tokens: int
    n_accepted: int
    n_rejected: int
    acceptance_rate: float


def speculative_decode(
    draft_workers: list[StageWorker],
    target_workers: list[StageWorker],
    prompt_ids: list[int],
    max_new_tokens: int,
    lookahead: int = 4,
    sampling: SamplingParams | None = None,
    *,
    operation_id: int = 1,
    cache_key: str | None = None,
) -> SpeculativeResult:
    """Run speculative decoding and always release both pipeline caches."""
    base_cache_key = cache_key or f"spec:{operation_id}"
    draft_cache_key = f"{base_cache_key}:draft"
    target_cache_key = f"{base_cache_key}:target"
    try:
        return _speculative_decode_impl(
            draft_workers,
            target_workers,
            prompt_ids,
            max_new_tokens,
            lookahead,
            sampling,
            operation_id=operation_id,
            draft_cache_key=draft_cache_key,
            target_cache_key=target_cache_key,
        )
    finally:
        for worker in [*draft_workers, *target_workers]:
            worker.clear_kv(draft_cache_key)
            worker.clear_kv(target_cache_key)


def _speculative_decode_impl(
    draft_workers: list[StageWorker],
    target_workers: list[StageWorker],
    prompt_ids: list[int],
    max_new_tokens: int,
    lookahead: int = 4,
    sampling: SamplingParams | None = None,
    *,
    operation_id: int = 1,
    draft_cache_key: str,
    target_cache_key: str,
) -> SpeculativeResult:
    """Generate tokens with draft proposals and target verification.

    At the start of every round both caches represent exactly ``context_ids``
    and the corresponding ``*_next_logits`` predict the next token. The target
    evaluates all draft tokens in one multi-token decode call. KV entries past
    the accepted context are then removed before continuing.
    """
    if not draft_workers or not target_workers:
        raise ValueError("draft_workers and target_workers must not be empty")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if lookahead < 1:
        raise ValueError("lookahead must be positive")
    if sampling is None:
        sampling = SamplingParams(temperature=0.0)
    generated: list[int] = []
    context_ids = list(prompt_ids)
    n_draft = 0
    n_accepted = 0
    n_rejected = 0
    op_id = operation_id

    prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
    draft_next_logits = pipeline_prefill(
        draft_workers,
        prompt_t,
        operation_id=op_id,
        attempt_id="spec_pf_d",
        cache_key=draft_cache_key,
    )
    target_next_logits = pipeline_prefill(
        target_workers,
        prompt_t,
        operation_id=op_id + 1,
        attempt_id="spec_pf_t",
        cache_key=target_cache_key,
    )
    op_id += 2
    # Draft proposals and target accept/rejection decisions can finish on
    # different physical devices (for example a one-GPU draft beside a
    # two-stage target).  A generator is device-specific in PyTorch; keep one
    # isolated stream per logits owner instead of passing a CPU generator into
    # a CUDA multinomial/rand operation.
    draft_generator = make_sampling_generator(
        sampling,
        device=draft_next_logits.device,
    )
    target_generator = make_sampling_generator(
        sampling,
        device=target_next_logits.device,
    )

    while len(generated) < max_new_tokens:
        old_context_len = len(context_ids)
        remaining = max_new_tokens - len(generated)
        k = min(lookahead, remaining)

        # Draft K tokens. The first token comes from the prefill distribution;
        # each later token consumes the previous proposal and advances draft KV.
        draft_tokens: list[int] = []
        draft_probs: list[torch.Tensor] = []
        for i in range(k):
            if i == 0:
                logits = draft_next_logits
            else:
                logits = pipeline_decode_step(
                    draft_workers,
                    torch.tensor([[draft_tokens[-1]]], dtype=torch.long),
                    operation_id=op_id,
                    attempt_id=f"spec_d{i}",
                    cache_key=draft_cache_key,
                )
                op_id += 1
            p = _sampling_distribution(logits[0, -1, :], sampling)
            draft_probs.append(p)
            draft_tokens.append(_sample_distribution(p, sampling, draft_generator))

        # Target's current next-token distribution verifies draft_tokens[0].
        # Feeding all K proposals yields distributions for tokens 1..K and
        # therefore also the bonus distribution at index K.
        target_decode_logits = pipeline_decode_step(
            target_workers,
            torch.tensor([draft_tokens], dtype=torch.long),
            operation_id=op_id,
            attempt_id="spec_t_verify",
            cache_key=target_cache_key,
        )
        op_id += 1
        target_logits = torch.cat(
            [target_next_logits[:, -1:, :], target_decode_logits],
            dim=1,
        )

        n_evaluated = 0
        accepted_this_round = 0
        rejected = False
        stop_after_round = False
        final_token: int | None = None

        for i, proposed in enumerate(draft_tokens):
            n_evaluated += 1
            p_target = _sampling_distribution(target_logits[0, i, :], sampling)
            p_draft = draft_probs[i]
            accepted = _accept(
                proposed,
                p_target,
                p_draft,
                sampling,
                target_generator,
            )

            if accepted:
                generated.append(proposed)
                context_ids.append(proposed)
                accepted_this_round += 1
                n_accepted += 1
                if (
                    sampling.eos_token_id is not None
                    and proposed == sampling.eos_token_id
                ) or len(generated) >= max_new_tokens:
                    stop_after_round = True
                    break
                continue

            rejected = True
            n_rejected += 1
            final_token = _sample_rejection(
                p_target,
                p_draft,
                sampling,
                target_generator,
            )
            generated.append(final_token)
            context_ids.append(final_token)
            if (
                sampling.eos_token_id is not None
                and final_token == sampling.eos_token_id
            ) or len(generated) >= max_new_tokens:
                stop_after_round = True
            break

        if not rejected and not stop_after_round:
            # All K proposals were accepted. The target's K-th decode output
            # supplies the bonus token, which is not counted as a draft token.
            p_bonus = _sampling_distribution(target_logits[0, k, :], sampling)
            final_token = _sample_distribution(p_bonus, sampling, target_generator)
            generated.append(final_token)
            context_ids.append(final_token)
            if (
                sampling.eos_token_id is not None
                and final_token == sampling.eos_token_id
            ) or len(generated) >= max_new_tokens:
                stop_after_round = True

        # Only target-evaluated proposals contribute to the accounting. This
        # keeps accepted + rejected == draft tokens even on early rejection.
        n_draft += n_evaluated

        if stop_after_round:
            break

        if rejected:
            # Both pipelines contain speculative suffixes. Roll them back to
            # the accepted prefix, then consume the replacement token in each.
            accepted_len = old_context_len + accepted_this_round
            _trim_pipeline_cache(draft_workers, accepted_len, cache_key=draft_cache_key)
            _trim_pipeline_cache(target_workers, accepted_len, cache_key=target_cache_key)
            replacement = torch.tensor([[final_token]], dtype=torch.long)
            draft_next_logits = pipeline_decode_step(
                draft_workers,
                replacement,
                operation_id=op_id,
                attempt_id="spec_d_resync",
                cache_key=draft_cache_key,
            )
            target_next_logits = pipeline_decode_step(
                target_workers,
                replacement,
                operation_id=op_id + 1,
                attempt_id="spec_t_resync",
                cache_key=target_cache_key,
            )
            op_id += 2
        else:
            # All K accepted and a non-terminal bonus was emitted. Draft has
            # cached only through draft[K-2]; append draft[K-1] first so both
            # caches represent old_context + K, then append the bonus.
            last_draft = torch.tensor([[draft_tokens[-1]]], dtype=torch.long)
            draft_decode_logits = pipeline_decode_step(
                draft_workers,
                last_draft,
                operation_id=op_id,
                attempt_id="spec_d_tail",
                cache_key=draft_cache_key,
            )
            target_next_logits = pipeline_decode_step(
                target_workers,
                torch.tensor([[final_token]], dtype=torch.long),
                operation_id=op_id + 1,
                attempt_id="spec_t_bonus",
                cache_key=target_cache_key,
            )
            draft_next_logits = pipeline_decode_step(
                draft_workers,
                torch.tensor([[final_token]], dtype=torch.long),
                operation_id=op_id + 2,
                attempt_id="spec_d_bonus",
                cache_key=draft_cache_key,
            )
            # The tail call's output is intentionally unused: the bonus call
            # returns the distribution needed for the next draft token.
            del draft_decode_logits
            op_id += 3

        if _cache_len(draft_workers, cache_key=draft_cache_key) != len(context_ids):
            raise RuntimeError("draft KV cache desynchronized after speculative step")
        if _cache_len(target_workers, cache_key=target_cache_key) != len(context_ids):
            raise RuntimeError("target KV cache desynchronized after speculative step")

        log.debug(
            "spec step: k=%d accepted=%d/%d rejected=%s final=%s",
            k,
            accepted_this_round,
            n_evaluated,
            rejected,
            final_token,
        )

    acceptance_rate = n_accepted / n_draft if n_draft else 0.0
    result = SpeculativeResult(
        generated_ids=generated,
        n_draft_tokens=n_draft,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        acceptance_rate=acceptance_rate,
    )
    return result


def _sample_distribution(
    probabilities: torch.Tensor,
    sampling: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    if sampling.temperature == 0.0:
        return int(probabilities.argmax())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _sampling_distribution(
    logits: torch.Tensor,
    sampling: SamplingParams,
) -> torch.Tensor:
    """Convert logits to the exact distribution used by speculative sampling."""
    logits = logits.float()
    if sampling.temperature == 0.0:
        probabilities = torch.zeros_like(logits)
        probabilities[logits.argmax()] = 1.0
        return probabilities

    probabilities = F.softmax(logits / sampling.temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=-1,
    )
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative - sorted_probs <= sampling.top_p
    sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.zeros_like(probabilities).scatter(-1, sorted_indices, sorted_probs)


def _accept(
    proposed: int,
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    sampling: SamplingParams,
    generator: torch.Generator | None = None,
) -> bool:
    if sampling.temperature == 0.0:
        return int(p_target.argmax()) == proposed
    ratio = (p_target[proposed] / p_draft[proposed].clamp(min=1e-8)).clamp(max=1.0)
    return torch.rand((), device=ratio.device, generator=generator).item() < ratio.item()


def _sample_rejection(
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    sampling: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    if sampling.temperature == 0.0:
        return int(p_target.argmax())
    adjusted = (p_target - p_draft).clamp(min=0)
    total = adjusted.sum()
    if total <= 0:
        return _sample_distribution(p_target, sampling, generator)
    return int(torch.multinomial(adjusted / total, 1, generator=generator).item())


def _cache_len(workers: list[StageWorker], *, cache_key: str | None = None) -> int:
    lengths = {worker.kv_cache_length(cache_key) for worker in workers}
    if len(lengths) != 1:
        raise RuntimeError(f"pipeline stages have inconsistent KV lengths: {sorted(lengths)}")
    return lengths.pop()


def _trim_pipeline_cache(
    workers: list[StageWorker],
    length: int,
    *,
    cache_key: str | None = None,
) -> None:
    """Drop speculative KV suffixes while preserving stage ownership."""
    if length < 0:
        raise ValueError("cache length must be non-negative")
    for worker in workers:
        worker.trim_kv(length, cache_key)
