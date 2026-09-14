"""Hugging Face Qwen3 dense pipeline stage.

This module deliberately keeps the decoder implementation in Transformers.
MeshGPU owns only the stage boundary, the per-stage cache, and the optional
embedding/output head.  That separation is important: a Qwen checkpoint must
be validated against the official Hugging Face model before any distributed
transport is involved.

The adapter targets the Qwen3 API shipped by Transformers 4.56 and newer.  It
uses ``DynamicCache`` rather than converting Qwen's cache to the legacy Llama
tuple format, so GQA, RoPE and future cache metadata remain owned by the HF
implementation.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def _qwen_symbols() -> dict[str, Any]:
    """Import Qwen symbols lazily so the base package stays HF-optional."""
    try:
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RMSNorm,
            Qwen3RotaryEmbedding,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Qwen3 support requires Transformers with the qwen3 model; "
            "install `meshgpu[hf]`"
        ) from exc
    return {
        "DynamicCache": DynamicCache,
        "Qwen3DecoderLayer": Qwen3DecoderLayer,
        "Qwen3RMSNorm": Qwen3RMSNorm,
        "Qwen3RotaryEmbedding": Qwen3RotaryEmbedding,
        "create_causal_mask": create_causal_mask,
        "create_sliding_window_causal_mask": create_sliding_window_causal_mask,
    }


class Qwen3Stage(nn.Module):
    """A contiguous Qwen3 decoder range with optional boundary modules.

    ``layer_start`` and ``layer_end`` are global model indices.  The local
    ``ModuleList`` is indexed from zero for compact stage checkpoints, while
    each official ``Qwen3DecoderLayer`` receives its global index so cache
    entries and sliding/full attention metadata remain correct.
    """

    meshgpu_adapter = "qwen3_hf_v1"
    supports_hf_cache = True

    def __init__(
        self,
        config: Any,
        layer_start: int,
        layer_end: int,
        *,
        has_embedding: bool,
        has_lm_head: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        attn_implementation: str = "sdpa",
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if layer_start < 0 or layer_end <= layer_start:
            raise ValueError("Qwen3 stage must own a non-empty layer range")
        if layer_end > int(config.num_hidden_layers):
            raise ValueError("Qwen3 stage layer range exceeds config")
        if attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "unsupported Qwen3 attention implementation: "
                f"{attn_implementation!r}"
            )

        symbols = _qwen_symbols()
        self.config = copy.deepcopy(config)
        # Transformers resolves this field on a full PreTrainedModel.  A
        # standalone stage needs to set it explicitly before constructing the
        # official decoder layers and before creating the causal mask.
        self.config._attn_implementation = attn_implementation
        self.cfg = self.config  # common MeshGPU stage metadata hook
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.attn_implementation = attn_implementation
        self.activation_checkpointing = activation_checkpointing
        self.vocab_size = int(self.config.vocab_size)
        self.max_position_embeddings = int(self.config.max_position_embeddings)
        self.hidden_size = int(self.config.hidden_size)

        # ``with torch.device`` avoids an intermediate CPU allocation when a
        # caller constructs a stage on CUDA or meta.  The loader can use
        # ``to_empty`` for the meta case before loading the selected shard.
        construction_device = torch.device(device) if device is not None else torch.device("cpu")
        with torch.device(construction_device):
            if has_embedding:
                self.embed_tokens = nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.pad_token_id,
                )
            self.layers = nn.ModuleList(
                [
                    symbols["Qwen3DecoderLayer"](self.config, global_idx)
                    for global_idx in range(layer_start, layer_end)
                ]
            )
            if has_lm_head:
                self.norm = symbols["Qwen3RMSNorm"](
                    self.config.hidden_size,
                    eps=self.config.rms_norm_eps,
                )
                self.lm_head = nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=False,
                )
                if has_embedding and bool(self.config.tie_word_embeddings):
                    # A one-stage Qwen model can preserve the true HF weight
                    # sharing.  With multiple stages the portable pipeline
                    # owns replicated endpoint weights and synchronizes them
                    # at the optimizer boundary.
                    self.lm_head.weight = self.embed_tokens.weight
            self.rotary_emb = symbols["Qwen3RotaryEmbedding"](
                config=self.config,
                device=construction_device,
            )

        if dtype is not None:
            self.to(dtype=dtype)
        if device is not None and construction_device.type != "meta":
            self.to(device=construction_device)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        kv_caches: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any | None]:
        """Run this stage using the official Qwen3 decoder blocks."""
        has_input_ids = input_ids is not None
        has_hidden = hidden.numel() != 0
        if has_input_ids == has_hidden:
            raise ValueError("stage must receive exactly one of input_ids or hidden")
        if input_ids is not None:
            if not self.has_embedding:
                raise ValueError("only the first Qwen3 stage accepts input_ids")
            if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
                raise ValueError("input_ids must have shape [batch, sequence] and be non-empty")
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must use int32 or int64 dtype")
            token_min = int(input_ids.min().item())
            token_max = int(input_ids.max().item())
            if token_min < 0 or token_max >= self.vocab_size:
                raise ValueError(
                    f"input_ids contain a value outside [0, {self.vocab_size})"
                )
            x = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        else:
            if self.has_embedding:
                raise ValueError("the first Qwen3 stage expects input_ids")
            x = hidden

        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Qwen3 hidden must have shape [batch, seq, {self.hidden_size}], "
                f"got {tuple(x.shape)}"
            )
        batch_size, query_len = x.shape[:2]
        if position_ids is None:
            past_len = self.cache_length(kv_caches) if use_cache and kv_caches is not None else 0
            position_ids = torch.arange(
                past_len,
                past_len + query_len,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device=x.device, dtype=torch.long)
            if position_ids.shape != (batch_size, query_len):
                raise ValueError(
                    "position_ids must have shape "
                    f"[{batch_size}, {query_len}], got {tuple(position_ids.shape)}"
                )
        position_min = int(position_ids.min().item())
        position_max = int(position_ids.max().item())
        if position_min < 0 or position_max >= self.max_position_embeddings:
            raise ValueError(
                "position_ids contain a value outside "
                f"[0, {self.max_position_embeddings})"
            )

        if use_cache:
            if kv_caches is None:
                kv_caches = self._new_cache()
            past_len = self.cache_length(kv_caches)
        else:
            # A training stage must not accidentally retain an inference cache
            # or build graph-attached key/value tensors.
            kv_caches = None
            past_len = 0

        if cache_position is None:
            cache_position = torch.arange(
                past_len,
                past_len + query_len,
                device=x.device,
                dtype=torch.long,
            )
        else:
            cache_position = cache_position.to(device=x.device, dtype=torch.long)
            if cache_position.numel() != query_len:
                raise ValueError("cache_position length must equal the query length")

        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            create_mask = self._mask_factory()
            masks = {
                "full_attention": create_mask(
                    config=self.config,
                    input_embeds=x,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=kv_caches,
                    position_ids=position_ids,
                )
            }
            if self._has_sliding_layers:
                masks["sliding_attention"] = self._sliding_mask_factory()(
                    config=self.config,
                    input_embeds=x,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=kv_caches,
                    position_ids=position_ids,
                )

        position_embeddings = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            layer_mask = masks.get(getattr(layer, "attention_type", "full_attention"))
            if self.activation_checkpointing and self.training and not use_cache:
                # Keep all non-tensor arguments in the closure.  The official
                # layer remains untouched; checkpointing only controls whether
                # its intermediates are retained by autograd.
                def run_layer(value: torch.Tensor, _layer=layer, _mask=layer_mask) -> torch.Tensor:
                    return _layer(
                        value,
                        attention_mask=_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    run_layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=kv_caches,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        if self.has_lm_head:
            x = self.lm_head(self.norm(x))
        return x, kv_caches

    @property
    def _has_sliding_layers(self) -> bool:
        return any(
            getattr(layer, "attention_type", "full_attention") == "sliding_attention"
            for layer in self.layers
        )

    def _mask_factory(self) -> Callable[..., Any]:
        return _qwen_symbols()["create_causal_mask"]

    def _sliding_mask_factory(self) -> Callable[..., Any]:
        return _qwen_symbols()["create_sliding_window_causal_mask"]

    def _new_cache(self) -> Any:
        return _qwen_symbols()["DynamicCache"](config=self.config)

    def cache_length(self, cache: Any | None) -> int:
        """Return this stage's cache length, including non-zero global indices."""
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length(self.layer_start))
        if isinstance(cache, (list, tuple)) and cache:
            first = cache[0]
            return int(first[0].shape[-2])
        raise TypeError(f"unsupported Qwen3 cache type: {type(cache)!r}")

    def trim_cache(self, cache: Any | None, length: int) -> None:
        if cache is None:
            if length:
                raise RuntimeError("cannot trim an empty Qwen3 cache to a non-zero length")
            return
        if length < 0:
            raise ValueError("cache length must be non-negative")
        if hasattr(cache, "crop"):
            cache.crop(length)
            return
        raise TypeError(f"Qwen3 cache does not support trimming: {type(cache)!r}")


def qwen3_config_from_dict(values: dict[str, Any]) -> Any:
    """Build a Qwen3Config without importing Transformers at package import time."""
    try:
        from transformers import Qwen3Config
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Qwen3 support requires `meshgpu[hf]`") from exc
    return Qwen3Config.from_dict(values)


def qwen3_config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize the exact HF config used by a Qwen3 artifact."""
    if not hasattr(config, "to_dict"):
        raise TypeError("config must provide Hugging Face to_dict()")
    values = config.to_dict()
    if not isinstance(values, dict):
        raise TypeError("Hugging Face config.to_dict() must return a dict")
    return values
