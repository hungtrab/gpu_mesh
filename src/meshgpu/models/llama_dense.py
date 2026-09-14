"""
Llama dense decoder adapter (v1).
Defines the stage contract for a decoder-only Transformer with GQA.
Only layers explicitly assigned to this stage are loaded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"({self.num_attention_heads} % {self.num_key_value_heads} != 0)"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "num_key_value_heads must not exceed num_attention_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotary embeddings, got {self.head_dim}")
        if not isinstance(self.rms_norm_eps, (int, float)) or not math.isfinite(
            float(self.rms_norm_eps)
        ) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if not isinstance(self.rope_theta, (int, float)) or not math.isfinite(
            float(self.rope_theta)
        ) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                "attention_implementation must be 'eager' or 'sdpa', got "
                f"{self.attention_implementation!r}"
            )

    @property
    def head_dim_computed(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def _precompute_freqs(
    dim: int, max_seq: int, theta: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [max_seq, D//2] → select positions → [B, S, D//2]
    # Expand to full head_dim: [B, 1, S, D]
    cos = torch.cat([cos, cos], dim=-1)[position_ids].unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1)[position_ids].unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.num_attention_heads
        Hkv = cfg.num_key_value_heads
        D = cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_size, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, Hkv * D, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        is_causal: bool | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape
        H = self.cfg.num_attention_heads
        Hkv = self.cfg.num_key_value_heads
        D = self.cfg.head_dim

        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, S, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, Hkv, D).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin, position_ids)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v)

        groups = H // Hkv
        if self.cfg.attention_implementation == "sdpa":
            sdpa_is_causal = bool(is_causal) if attention_mask is None else False
            sdpa_scale = 1.0 / math.sqrt(D)
            if groups > 1:
                # Keep K/V compact on the memory-sensitive path.  ``enable_gqa``
                # lets PyTorch dispatch GQA without materialising a repeated
                # [B, H, S, D] copy.  Torch 2.4 builds that predate the keyword
                # fall back to the equivalent explicit expansion below.
                try:
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=sdpa_is_causal,
                        scale=sdpa_scale,
                        enable_gqa=True,
                    )
                except TypeError as exc:
                    if "enable_gqa" not in str(exc):
                        raise
                    k = k.repeat_interleave(groups, dim=1)
                    v = v.repeat_interleave(groups, dim=1)
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=sdpa_is_causal,
                        scale=sdpa_scale,
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=sdpa_is_causal,
                    scale=sdpa_scale,
                )
        else:
            # The eager reference path has no native GQA switch, so expand K/V
            # only here.  It is intentionally not used by SDPA.
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            scale = math.sqrt(D)
            attn = torch.matmul(q, k.transpose(2, 3)) / scale
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)
        return self.o_proj(out), new_cache


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = GQAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        is_causal: bool | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x, new_cache = self.self_attn(
            self.input_layernorm(x),
            cos,
            sin,
            position_ids,
            kv_cache,
            attention_mask,
            is_causal,
        )
        x = residual + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Stage — owns a slice of layers [layer_start, layer_end)
# ---------------------------------------------------------------------------

class KVEntry(NamedTuple):
    k: torch.Tensor
    v: torch.Tensor


class LlamaStage(nn.Module):
    """
    One pipeline stage: a contiguous slice of decoder layers,
    optionally with embedding table (stage 0) and LM head (last stage).
    """

    supports_cache_flag = True

    def __init__(
        self,
        cfg: LlamaConfig,
        layer_start: int,
        layer_end: int,
        *,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        device: torch.device | None = None,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= layer_start < layer_end <= cfg.num_hidden_layers:
            raise ValueError(
                f"invalid layer range [{layer_start}, {layer_end}) for "
                f"{cfg.num_hidden_layers} layers"
            )
        self.cfg = cfg
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.activation_checkpointing = activation_checkpointing

        if has_embedding:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(cfg) for _ in range(layer_end - layer_start)]
        )
        if has_lm_head:
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            if has_embedding and cfg.tie_word_embeddings:
                # A one-stage model can preserve HF's true shared parameter.
                # Multi-stage replicas are synchronized by the portable
                # pipeline/trainer boundary instead.
                self.lm_head.weight = self.embed_tokens.weight

        if device is not None:
            self.to(device)

        # RoPE buffers — precomputed, not trainable
        cos, sin = _precompute_freqs(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta,
            device=device,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor] | None] | None,
    ]:
        """
        Args:
            x: hidden states [B, S, H] (ignored when has_embedding and input_ids given)
            input_ids: [B, S] only for stage 0
            position_ids: [B, S] — required
            kv_caches: list of (k, v) per layer, or None per layer
            use_cache: retain and return K/V tensors for inference; training
                callers should disable this to avoid holding graph references
            attention_mask: additive mask [B, 1, S_q, S_kv]
        Returns:
            (output_hidden or logits, new_kv_caches)
        """
        if self.has_embedding:
            if input_ids is None:
                raise ValueError("the embedding stage requires input_ids")
            if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
                raise ValueError(
                    "input_ids must have shape [batch, sequence] and be non-empty"
                )
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must use int32 or int64 dtype")
            token_min = int(input_ids.min().item())
            token_max = int(input_ids.max().item())
            if token_min < 0 or token_max >= self.cfg.vocab_size:
                raise ValueError(
                    "input_ids contain a value outside "
                    f"[0, {self.cfg.vocab_size})"
                )
            B = input_ids.shape[0]
            x = self.embed_tokens(input_ids)
        else:
            if input_ids is not None:
                raise ValueError("input_ids are only accepted by the embedding stage")
            if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1:
                raise ValueError(
                    "a non-embedding stage requires non-empty hidden states [B, S, H]"
                )
            if x.shape[-1] != self.cfg.hidden_size:
                raise ValueError(
                    f"hidden state width {x.shape[-1]} != model hidden_size "
                    f"{self.cfg.hidden_size}"
                )
            B = x.shape[0]

        implicit_position_ids = position_ids is None
        if position_ids is None:
            S = x.shape[1]
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            if position_ids.ndim != 2 or position_ids.shape[1] != x.shape[1]:
                raise ValueError(
                    "position_ids must have shape [batch or 1, sequence] matching input"
                )
            if position_ids.shape[0] not in (1, B):
                raise ValueError("position_ids batch dimension must be 1 or input batch")
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("position_ids must use int32 or int64 dtype")
            position_ids = position_ids.to(x.device)
            if position_ids.shape[0] == 1 and B > 1:
                position_ids = position_ids.expand(B, -1)
        position_min = int(position_ids.min().item())
        position_max = int(position_ids.max().item())
        if position_min < 0 or position_max >= self.cfg.max_position_embeddings:
            raise ValueError(
                "position_ids contain a value outside "
                f"[0, {self.cfg.max_position_embeddings})"
            )

        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)

        if not use_cache:
            # A caller cannot accidentally feed an inference cache through a
            # training forward.  More importantly, this keeps the returned
            # cache-free path from retaining graph-attached K/V tensors.
            kv_caches = None
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        elif len(kv_caches) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} KV cache entries, got {len(kv_caches)}"
            )

        cache_presence = [cached is not None for cached in kv_caches]
        if any(cache_presence) and not all(cache_presence):
            raise ValueError(
                "KV cache must be provided for every layer or for none of them"
            )

        past_len: int | None = None
        for cached in kv_caches:
            if cached is None:
                continue
            if len(cached) != 2:
                raise ValueError("each KV cache entry must contain key and value tensors")
            key, value = cached
            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise ValueError("KV cache tensors must have matching shape [B, Hkv, S, D]")
            if key.device != x.device or value.device != x.device:
                raise ValueError("KV cache tensors must be on the same device as input")
            if key.dtype != x.dtype or value.dtype != x.dtype:
                raise ValueError("KV cache tensors must use the same dtype as input")
            if (
                key.shape[0] != B
                or key.shape[1] != self.cfg.num_key_value_heads
                or key.shape[3] != self.cfg.head_dim
            ):
                raise ValueError("KV cache shape does not match model configuration")
            current_len = int(key.shape[2])
            if past_len is None:
                past_len = current_len
            elif current_len != past_len:
                raise ValueError("KV cache sequence lengths are inconsistent across layers")
        past_len = past_len or 0
        if past_len + x.shape[1] > self.cfg.max_position_embeddings:
            raise ValueError(
                f"sequence would reach {past_len + x.shape[1]} positions, "
                f"but model context is {self.cfg.max_position_embeddings}"
            )
        if implicit_position_ids and past_len:
            # A direct stage caller may provide a cache without explicitly
            # supplying positions.  Continue from the cached prefix instead
            # of silently reusing RoPE positions 0..S-1.
            position_ids = torch.arange(
                past_len,
                past_len + x.shape[1],
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)

        # Decoder-only attention must not see future tokens during prefill.
        # For decode, ``past_len`` accounts for cached keys, while the current
        # query positions come from position_ids.  Keep this mask explicit so
        # the same stage contract works for both training and inference.
        efficient_causal = False
        if attention_mask is None:
            key_positions = torch.arange(
                past_len + x.shape[1], device=x.device, dtype=torch.long
            )
            allowed = key_positions.view(1, 1, 1, -1) <= position_ids.view(B, 1, -1, 1)
            standard_prefill = (
                past_len == 0
                and torch.equal(
                    position_ids,
                    torch.arange(x.shape[1], device=x.device, dtype=position_ids.dtype)
                    .unsqueeze(0)
                    .expand(B, -1),
                )
            )
            # SDPA can generate a causal mask internally for a standard
            # prefill.  A single-token decode may attend to every cached key,
            # so it needs no mask and must explicitly disable is_causal (the
            # square causal rule would otherwise expose only key zero).
            if self.cfg.attention_implementation == "sdpa" and standard_prefill:
                efficient_causal = True
                attention_mask = None
            elif self.cfg.attention_implementation == "sdpa" and x.shape[1] == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros(
                    (B, 1, x.shape[1], past_len + x.shape[1]),
                    device=x.device,
                    dtype=x.dtype,
                )
                attention_mask.masked_fill_(
                    ~allowed,
                    torch.finfo(x.dtype).min,
                )
        else:
            attention_mask = attention_mask.to(x.device, dtype=x.dtype)

        new_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i]
            if self.activation_checkpointing and self.training and not any(
                cached is not None for cached in kv_caches
            ):
                def run_layer(value: torch.Tensor, _layer=layer) -> torch.Tensor:
                    result, _ = _layer(
                        value,
                        cos,
                        sin,
                        position_ids,
                        None,
                        attention_mask,
                        efficient_causal,
                    )
                    return result

                x = torch.utils.checkpoint.checkpoint(
                    run_layer,
                    x,
                    use_reentrant=False,
                )
                new_caches.append(None)
            else:
                x, cache = layer(
                    x,
                    cos,
                    sin,
                    position_ids,
                    cache,
                    attention_mask,
                    efficient_causal,
                )
                # Training calls use_cache=False.  The layer still returns
                # freshly-built K/V tensors for its attention computation, but
                # retaining those graph-attached tensors in the stage object
                # would keep the entire backward graph alive until the next
                # operation.  Only inference owns/reuses a KV cache.
                new_caches.append(cache if use_cache else None)

        if self.has_lm_head:
            x = self.lm_head(self.norm(x))

        return x, new_caches if use_cache else None
