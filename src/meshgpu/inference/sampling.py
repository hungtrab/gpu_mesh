"""Sampling utilities: greedy, top-p."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.0      # 0 = greedy
    top_p: float = 1.0
    eos_token_id: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(float(self.top_p))
            or not 0 <= self.top_p <= 1
        ):
            raise ValueError("top_p must be a finite number in [0, 1]")
        if self.eos_token_id is not None and (
            isinstance(self.eos_token_id, bool)
            or not isinstance(self.eos_token_id, int)
            or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a non-negative integer or None")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


def make_sampling_generator(
    params: SamplingParams,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator | None:
    """Create an isolated RNG for a seeded request, or ``None`` otherwise."""
    if params.seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    return generator


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax over last dimension."""
    return logits.argmax(dim=-1)


def top_p_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.
    logits: [batch, vocab] or [vocab]
    Returns scalar token id tensor.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if not math.isfinite(top_p) or not 0 <= top_p <= 1:
        raise ValueError("top_p must be a finite number in [0, 1]")
    if temperature <= 0:
        return greedy_sample(logits)

    logits = logits.float() / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    # (shift by 1 so we always keep the top token)
    remove = cumulative - sorted_probs > top_p
    sorted_probs[remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    next_token = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, next_token).squeeze(-1)
