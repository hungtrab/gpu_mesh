"""
Portable pipeline stage worker.
Runs in a subprocess per GPU device; communicates via multiprocessing queues
for the local path and via transport.connection for cross-machine path.

Phase 1 (P1): in-process queues only (single-machine correctness test).
Phase 2 (P2): swap queues for network transport.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class OptimizerLike(Protocol):
    """Minimal optimizer surface needed by a stage boundary."""

    def step(self) -> None: ...

    def zero_grad(self, *, set_to_none: bool = False) -> None: ...


@dataclass
class StageContext:
    stage_id: int
    layer_start: int
    layer_end: int
    device: torch.device
    is_first: bool
    is_last: bool
    job_id: str
    attempt_id: str
    # ``cpu`` is the backwards-compatible network/portable path.  ``local_cuda``
    # keeps same-host CUDA boundaries on device and reports a host fallback if
    # peer access is unavailable.
    transport: str = "cpu"

    def __post_init__(self) -> None:
        for name, value in (
            ("stage_id", self.stage_id),
            ("layer_start", self.layer_start),
            ("layer_end", self.layer_end),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("StageContext must own a non-empty layer range")
        self.device = torch.device(self.device)
        if self.transport not in {"cpu", "local_cuda"}:
            raise ValueError(f"unsupported stage transport: {self.transport!r}")
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.attempt_id:
            raise ValueError("attempt_id must not be empty")


@dataclass
class ForwardResult:
    stage_id: int
    operation_id: int
    attempt_id: str
    output: torch.Tensor           # detached; CPU or source device per transport
    kv_caches: Any                 # kept on device
    # For backward: store the boundary tensor with grad_fn so we can call .backward()
    _boundary_tensor: torch.Tensor | None = field(default=None, repr=False)


@dataclass
class BackwardResult:
    stage_id: int
    operation_id: int
    attempt_id: str
    grad_input: torch.Tensor | None  # dL/d_input to send to previous stage; None for stage 0


class StageWorker:
    """
    Owns one LlamaStage, executes forward and backward for pipeline training.
    KV caches are owned here and never transmitted per-token.
    """

    def __init__(self, model: nn.Module, ctx: StageContext) -> None:
        self._model = model
        self._ctx = ctx
        self._kv_caches: Any = []
        # ``_kv_caches`` is retained as the backwards-compatible default
        # context.  Real inference uses one named context per request so
        # concurrent sessions never overwrite one another's KV state.
        self._kv_caches_by_key: dict[str, Any] = {}
        # A stage model may be shared by several asyncio worker threads.  The
        # lock protects both model execution and cache replacement; named
        # caches provide isolation, while the lock prevents races inside a
        # single stage when two requests arrive at the same time.
        self._inference_lock = threading.RLock()
        # Saved activations keyed by operation_id for backward
        self._saved: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Inference (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(
        self,
        hidden: torch.Tensor | None,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        operation_id: int,
        attempt_id: str,
        reset_kv: bool = False,
        cache_key: str | None = None,
    ) -> ForwardResult:
        with self._inference_lock:
            if reset_kv:
                self.clear_kv(cache_key)

            x = self._move_input(hidden)
            if input_ids is not None:
                input_ids = input_ids.to(self._ctx.device)
            if position_ids is not None:
                position_ids = position_ids.to(self._ctx.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._ctx.device)

            kv_in = self._get_kv(cache_key)
            out, new_kv = self._call_model(
                x,
                input_ids=input_ids,
                position_ids=position_ids,
                kv_caches=_cache_or_none(kv_in),
                attention_mask=attention_mask,
                use_cache=True,
            )
            self._set_kv(cache_key, new_kv)
        return ForwardResult(
            stage_id=self._ctx.stage_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            output=self._boundary_output(out),
            kv_caches=new_kv,
        )

    # ------------------------------------------------------------------
    # Training forward (with grad)
    # ------------------------------------------------------------------

    def forward_train(
        self,
        hidden: torch.Tensor | None,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        operation_id: int,
        attempt_id: str,
    ) -> ForwardResult:
        if not self._ctx.is_first and hidden is None:
            raise ValueError(f"stage {self._ctx.stage_id} requires a hidden-state input")
        if self._ctx.is_first and input_ids is None:
            raise ValueError("the first stage requires input_ids")
        if operation_id in self._saved:
            raise RuntimeError(
                f"operation_id={operation_id} already has an active training forward"
            )

        x = self._move_input(hidden)
        # ``Tensor.to(device)`` returns a non-leaf tensor when the stage is on
        # another GPU.  Detach *after* the transfer so ``x.grad`` is populated
        # reliably for the inter-stage backward message.
        if not self._ctx.is_first:
            x = x.detach().requires_grad_(True)

        if input_ids is not None:
            input_ids = input_ids.to(self._ctx.device)
        if position_ids is not None:
            position_ids = position_ids.to(self._ctx.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._ctx.device)

        out, new_kv = self._call_model(
            x,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Save for backward
        self._saved[operation_id] = {
            "x": x,
            "out": out,
            "attempt_id": attempt_id,
        }
        self._kv_caches = new_kv if new_kv is not None else []

        return ForwardResult(
            stage_id=self._ctx.stage_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            output=self._boundary_output(out.detach()),
            kv_caches=new_kv,
            _boundary_tensor=x if not self._ctx.is_first else None,
        )

    # ------------------------------------------------------------------
    # Training backward
    # ------------------------------------------------------------------

    def backward_train(
        self,
        grad_output: torch.Tensor | None,
        *,
        operation_id: int,
        attempt_id: str,
    ) -> BackwardResult:
        saved = self._saved.get(operation_id)
        if saved is None:
            raise RuntimeError(
                f"no saved state for operation_id={operation_id} attempt={attempt_id}"
            )
        if saved["attempt_id"] != attempt_id:
            raise RuntimeError(
                f"attempt_id mismatch: saved={saved['attempt_id']}, got={attempt_id}"
            )
        # Remove the graph from the retry table before invoking autograd.  If
        # backward raises, the caller must not accidentally reuse a partially
        # consumed graph under the same operation id.
        self._saved.pop(operation_id, None)

        out: torch.Tensor = saved["out"]
        x: torch.Tensor = saved["x"]

        if grad_output is not None:
            grad_output = self._move_input(grad_output)

        try:
            # Loss is computed at the last stage; for earlier stages
            # ``grad_output`` is the gradient received from downstream.
            out.backward(gradient=grad_output)

            grad_input = None
            if not self._ctx.is_first and x.grad is not None:
                # Preserve the same-host CUDA path for the reverse boundary.
                # The CPU path remains the wire-compatible default, while a
                # local CUDA pipeline can pass dL/dx device-to-device.
                grad_input = self._boundary_output(x.grad)

            return BackwardResult(
                stage_id=self._ctx.stage_id,
                operation_id=operation_id,
                attempt_id=attempt_id,
                grad_input=grad_input,
            )
        finally:
            # Training KV tensors are attached to the autograd graph and are
            # not reusable for the next training batch.  Release them even
            # when autograd raises, otherwise a failed operation retains the
            # graph and leaks memory into the next retry.
            self._kv_caches = []

    def optimizer_step(self, optimizer: OptimizerLike, scaler: Any | None) -> None:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _get_kv(self, cache_key: str | None) -> Any:
        if cache_key is None:
            return self._kv_caches
        return self._kv_caches_by_key.get(cache_key, [])

    def _set_kv(self, cache_key: str | None, caches: Any) -> None:
        if cache_key is None:
            self._kv_caches = caches
        else:
            self._kv_caches_by_key[cache_key] = caches

    def clear_kv(self, cache_key: str | None = None) -> None:
        """Drop one request's KV cache, or the legacy default cache."""
        with self._inference_lock:
            if cache_key is None:
                self._kv_caches = []
            else:
                self._kv_caches_by_key.pop(cache_key, None)

    def clear_all_kv(self) -> None:
        """Drop the default cache and every named request cache."""
        with self._inference_lock:
            self._kv_caches = []
            self._kv_caches_by_key.clear()

    def kv_cache_length(self, cache_key: str | None = None) -> int:
        """Return the cached sequence length for a request context."""
        with self._inference_lock:
            caches = self._get_kv(cache_key)
            if not caches:
                return 0
            cache_length = getattr(self._model, "cache_length", None)
            if callable(cache_length):
                return int(cache_length(caches))
            first = caches[0]
            return int(first[0].shape[2])

    def trim_kv(self, length: int, cache_key: str | None = None) -> None:
        """Trim a request cache to ``length`` tokens in place."""
        if length < 0:
            raise ValueError("cache length must be non-negative")
        with self._inference_lock:
            caches = self._get_kv(cache_key)
            if not caches:
                if length:
                    raise RuntimeError("cannot trim an empty cache to a non-zero length")
                return
            trim_cache = getattr(self._model, "trim_cache", None)
            if callable(trim_cache):
                trim_cache(caches, length)
                return
            trimmed = []
            for key, value in caches:
                if key.shape[2] < length:
                    raise RuntimeError(
                        f"cache length {key.shape[2]} is shorter than requested {length}"
                    )
                trimmed.append((key[:, :, :length, :], value[:, :, :length, :]))
            self._set_kv(cache_key, trimmed)

    def clear_saved(self) -> None:
        self._saved.clear()

    def _move_input(self, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            return torch.empty(0, device=self._ctx.device)
        from meshgpu.transport.local_cuda import move_boundary

        return move_boundary(
            hidden,
            self._ctx.device,
            transport=self._ctx.transport,
        )

    def _boundary_output(self, output: torch.Tensor) -> torch.Tensor:
        """Detach output while retaining its source device for local CUDA."""
        if self._ctx.transport == "local_cuda":
            return output.detach()
        if self._ctx.transport == "cpu":
            return output.detach().cpu()
        raise ValueError(f"unsupported stage transport: {self._ctx.transport!r}")

    def _call_model(self, hidden: torch.Tensor, *, use_cache: bool, **kwargs: Any) -> Any:
        # Explicitly pass the cache policy to both official HF stages and the
        # built-in Llama stage.  Other stage adapters keep their historical
        # call contract unless they opt into this flag.
        if (
            getattr(self._model, "supports_hf_cache", False)
            or getattr(self._model, "supports_cache_flag", False)
        ):
            kwargs["use_cache"] = use_cache
        return self._model(hidden, **kwargs)


def _cache_or_none(cache: Any) -> Any | None:
    """Normalize the legacy empty-list cache without probing HF cache objects."""
    if cache is None:
        return None
    if isinstance(cache, (list, tuple)) and not cache:
        return None
    return cache
