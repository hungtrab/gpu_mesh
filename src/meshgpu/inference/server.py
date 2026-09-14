"""
Inference HTTP server — streaming token generation via SSE.
Mounts on a separate port from the controller.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.inference.admission import (
    AdmissionConfig,
    AdmissionController,
    KVBudget,
    MemoryPreflightResult,
)
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.session import InferenceSession
from meshgpu.inference.tokenizer import Tokenizer, TokenizerError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt_ids: list[int] | None = Field(
        default=None,
        min_length=1,
        description="Tokenized prompt token IDs",
    )
    prompt: str | None = Field(default=None, min_length=1, description="Raw text prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, description="Optional per-request sampling seed")
    stream: bool = Field(default=True)

    @model_validator(mode="after")
    def exactly_one_prompt(self) -> GenerateRequest:
        if (self.prompt_ids is None) == (self.prompt is None):
            raise ValueError("provide exactly one of prompt or prompt_ids")
        return self


class TokenEvent(BaseModel):
    token_id: int
    seq_num: int
    is_last: bool
    text: str = ""


class GenerateResponse(BaseModel):
    session_id: str
    token_ids: list[int]
    seq_nums: list[int]
    tokens_per_second: float
    text: str = ""


class MemoryPreflight(Protocol):
    """Synchronous request-level memory check supplied by the planner."""

    def __call__(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
    ) -> MemoryPreflightResult | bool | tuple[bool, str]: ...


def build_cuda_memory_preflight(
    workers: list[Any],
    *,
    max_concurrent: int = 1,
    max_vram_fraction: float = 0.85,
    reserve_min_bytes: int | None = None,
) -> MemoryPreflight | None:
    """Build a conservative request-time CUDA memory guard.

    The guard is intentionally local and synchronous: it re-reads free VRAM
    immediately before each request and runs the same static cost model used by
    ``meshgpu plan``.  It is a rejection guard, not proof that an unprofiled
    kernel cannot allocate more; callers with exact measurements should still
    provide ``memory_preflight`` explicitly.  CPU and remote RPC workers do not
    expose enough physical capacity metadata here and return ``None``.
    """
    import torch

    from meshgpu.planner.memory import GiB, TensorPeak, check_feasibility
    from meshgpu.planner.placement import (
        InferenceWorkload,
        ModelSpec,
        WorkerSpec,
        plan_inference,
    )

    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int):
        raise TypeError("max_concurrent must be an integer")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    if reserve_min_bytes is None:
        reserve_min_bytes = GiB
    if not workers or not torch.cuda.is_available():
        return None

    stage_devices: list[torch.device] = []
    stage_ranges: list[tuple[int, int]] = []
    for worker in workers:
        device = _local_worker_device(worker)
        if device is None:
            return None
        if device.type != "cuda":
            return None
        stage_devices.append(device)
        context = getattr(worker, "_ctx", None)
        layer_start = getattr(context, "layer_start", None)
        layer_end = getattr(context, "layer_end", None)
        if (
            isinstance(layer_start, bool)
            or not isinstance(layer_start, int)
            or isinstance(layer_end, bool)
            or not isinstance(layer_end, int)
        ):
            return None
        stage_ranges.append((layer_start, layer_end))
    canonical_devices = [
        int(device.index if device.index is not None else torch.cuda.current_device())
        for device in stage_devices
    ]
    if len(set(canonical_devices)) != len(canonical_devices):
        raise ValueError(
            "automatic CUDA memory preflight requires one distinct CUDA device "
            "per stage; pass an explicit preflight for intentional co-location"
        )

    model = _model_spec_from_workers(workers, torch, ModelSpec)
    if model is None:
        # Every worker reached this point as a local CUDA worker.  Disabling
        # the guard because a malformed/custom stage lacks planner metadata
        # would be a dangerous fail-open path.  Such callers can still opt in
        # to a measured callback through ``build_inference_app``.
        raise RuntimeError(
            "automatic CUDA memory preflight cannot derive a complete model "
            "specification from the loaded stages; provide an explicit measured "
            "memory_preflight callback"
        )

    attention_reports: list[dict[str, Any]] = []
    attention_warning: str | None = None
    if model.attention_implementation == "sdpa":
        # SDPA is a dispatch API.  If any participating GPU falls back to the
        # math implementation, charge the conservative quadratic workspace in
        # the request guard rather than assuming a fused kernel was selected.
        from meshgpu.backends.portable.attention import (
            AttentionBackendReport,
            verify_sdpa_backend,
        )

        first_model = getattr(workers[0], "_model", None)
        if first_model is None:
            raise RuntimeError("local CUDA worker is missing its model")
        parameter = next(
            (
                parameter
                for name, parameter in first_model.named_parameters()
                if "lora_A" not in name and "lora_B" not in name
            ),
            None,
        )
        probe_dtype = parameter.dtype if parameter is not None else torch.float32
        probe_tokens = min(model.max_position_embeddings or 128, 128)
        selected: list[AttentionBackendReport | None] = []
        for device in stage_devices:
            try:
                backend_report = verify_sdpa_backend(
                    (
                        1,
                        model.num_attention_heads,
                        probe_tokens,
                        model.head_dim,
                    ),
                    device=device,
                    dtype=probe_dtype,
                )
            except Exception as exc:  # pragma: no cover - defensive backend boundary
                attention_reports.append(
                    {
                        "device": str(device),
                        "selected": "unavailable",
                        "verified": False,
                        "fused": False,
                        "warning": (
                            f"SDPA verification raised {type(exc).__name__}: {exc}"
                        ),
                    }
                )
                selected.append(None)
                continue
            attention_reports.append(
                {
                    "device": str(device),
                    "selected": backend_report.selected,
                    "verified": backend_report.verified,
                    "fused": backend_report.fused,
                    "warning": backend_report.warning,
                }
            )
            selected.append(backend_report)
        if any(item is None or not item.verified or not item.fused for item in selected):
            attention_warning = (
                "at least one GPU did not verify a fused SDPA kernel; "
                "the automatic guard uses the conservative eager attention estimate"
            )
            model = replace(model, attention_implementation="eager")
    elif model.attention_implementation == "flash_attention_2":
        # flash-attn has independent package/SM constraints.  The generic
        # SDPA profiler cannot prove that an official Qwen layer will dispatch
        # to it, so use the conservative quadratic estimate until the caller
        # supplies an exact profile.  The stage runtime remains unchanged.
        attention_warning = (
            "flash_attention_2 is not verified by the automatic guard; "
            "using the conservative eager attention estimate"
        )
        attention_reports.extend(
            {
                "device": str(device),
                "selected": "unverified",
                "verified": False,
                "fused": False,
                "warning": "flash-attn dispatch requires an explicit hardware profile",
            }
            for device in stage_devices
        )
        model = replace(model, attention_implementation="eager")

    def check(prompt_ids: list[int], max_new_tokens: int) -> MemoryPreflightResult:
        if not isinstance(prompt_ids, list) or not prompt_ids:
            return MemoryPreflightResult(
                feasible=False,
                reason="prompt_ids must be a non-empty list",
            )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            return MemoryPreflightResult(
                feasible=False,
                reason="max_new_tokens must be an integer",
            )
        if max_new_tokens < 1:
            return MemoryPreflightResult(
                feasible=False,
                reason="max_new_tokens must be positive",
            )

        worker_specs: list[WorkerSpec] = []
        try:
            for index, (worker, device) in enumerate(zip(workers, stage_devices)):
                total = int(torch.cuda.get_device_properties(device).total_memory)
                free, _ = torch.cuda.mem_get_info(device)
                worker_specs.append(
                    WorkerSpec(
                        worker_id=str(
                            getattr(getattr(worker, "_ctx", None), "stage_id", index)
                        ),
                        device_index=canonical_devices[index],
                        total_vram_bytes=total,
                        free_vram_bytes=int(free),
                    )
                )
            static_report = plan_inference(
                worker_specs,
                model,
                InferenceWorkload(
                    batch_size=1,
                    max_prompt_tokens=len(prompt_ids),
                    max_new_tokens=max_new_tokens,
                    max_concurrent=max_concurrent,
                ),
                max_vram_fraction=max_vram_fraction,
                reserve_min_bytes=reserve_min_bytes,
                layer_ranges=stage_ranges,
            )
            # The local workers are already loaded when this app factory is
            # called.  Their resident weights and CUDA context are therefore
            # already reflected in ``free``; charging them a second time would
            # reject valid requests.  Keep only the incremental request peak.
            dynamic_assignments = [
                replace(
                    assignment,
                    peak=TensorPeak(
                        kv_cache_bytes=assignment.peak.kv_cache_bytes,
                        activation_bytes=assignment.peak.activation_bytes,
                        comm_buffer_bytes=assignment.peak.comm_buffer_bytes,
                        workspace_bytes=assignment.peak.workspace_bytes,
                        cuda_context_bytes=0,
                        adapter_bytes=0,
                    ),
                )
                for assignment in static_report.assignments
            ]
            dynamic_feasibility = check_feasibility(
                [assignment.budget for assignment in static_report.assignments],
                [assignment.peak for assignment in dynamic_assignments],
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            return MemoryPreflightResult(
                feasible=False,
                reason=f"could not compute CUDA memory preflight: {exc}",
            )
        warnings = list(static_report.warnings)
        if attention_warning is not None:
            warnings.append(attention_warning)
        return MemoryPreflightResult(
            feasible=dynamic_feasibility.feasible,
            reason=dynamic_feasibility.reason,
            details={
                "static_estimate": True,
                "resident_weights_already_loaded": True,
                "warnings": warnings,
                "attention_dispatch": attention_reports,
                "stages": [
                    {
                        "stage_id": assignment.stage_id,
                        "device_index": assignment.device_index,
                        "layer_start": assignment.layer_start,
                        "layer_end": assignment.layer_end,
                        "peak_bytes": assignment.peak.total,
                        "usable_bytes": assignment.budget.usable_bytes,
                        "margin_bytes": (
                            assignment.budget.usable_bytes - assignment.peak.total
                        ),
                    }
                    for assignment in dynamic_assignments
                ],
            },
        )

    return check


def _local_worker_device(worker: Any):
    """Return a local stage device, or ``None`` for an RPC-like worker."""
    model = getattr(worker, "_model", None)
    context = getattr(worker, "_ctx", None)
    device = getattr(context, "device", None)
    if model is None or device is None:
        return None
    try:
        import torch

        return torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return None


def _model_spec_from_workers(_workers: list[Any], _torch_module: Any, model_spec_type: Any):
    """Derive conservative planner metadata from loaded local stage modules.

    This helper returns ``None`` when the live stages do not expose enough
    identity to make a safe estimate.  The CUDA preflight caller treats that
    as a hard startup error; it must never silently turn a malformed local
    CUDA model into an unguarded service.
    """
    workers = _workers
    if not workers:
        return None
    first_model = getattr(workers[0], "_model", None)
    cfg = getattr(first_model, "cfg", None)
    if cfg is None:
        return None
    required = (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    )
    try:
        values = {name: int(getattr(cfg, name)) for name in required}
    except (AttributeError, TypeError, ValueError):
        return None
    max_positions = getattr(cfg, "max_position_embeddings", None)
    if max_positions is not None:
        try:
            max_positions = int(max_positions)
        except (TypeError, ValueError):
            return None

    stage_ranges: list[tuple[int, int]] = []
    param_count = 0
    per_layer_candidates: list[int] = []
    embedding_count: int | None = None
    lm_head_count: int | None = None
    adapter_count = 0
    dtype_bytes = 0
    adapter_dtype_bytes = 4
    seen_first = False
    seen_last = False
    attention: str | None = None

    for index, worker in enumerate(workers):
        model = getattr(worker, "_model", None)
        context = getattr(worker, "_ctx", None)
        if model is None or context is None:
            return None
        stage_cfg = getattr(model, "cfg", None)
        if stage_cfg is None:
            return None
        try:
            if any(int(getattr(stage_cfg, name)) != values[name] for name in required):
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        stage_max_positions = getattr(stage_cfg, "max_position_embeddings", None)
        if stage_max_positions is not None:
            try:
                stage_max_positions = int(stage_max_positions)
            except (TypeError, ValueError):
                return None
        if stage_max_positions != max_positions:
            return None

        start = getattr(context, "layer_start", None)
        end = getattr(context, "layer_end", None)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > values["num_hidden_layers"]
        ):
            return None
        stage_ranges.append((start, end))
        layer_count = end - start

        is_first = getattr(context, "is_first", None)
        is_last = getattr(context, "is_last", None)
        if not isinstance(is_first, bool) or not isinstance(is_last, bool):
            return None
        if is_first != (index == 0) or is_last != (index == len(workers) - 1):
            return None
        seen_first |= is_first
        seen_last |= is_last

        parameters = list(model.named_parameters())
        if not parameters:
            return None
        param_count += sum(int(parameter.numel()) for _, parameter in parameters)
        layer_params = sum(
            int(parameter.numel())
            for name, parameter in parameters
            if name.startswith("layers.")
            and "lora_A" not in name
            and "lora_B" not in name
        )
        if layer_params % layer_count != 0:
            # A formula based on an average layer would be an unsafe
            # under-estimate for a heterogeneous or malformed stage.
            return None
        per_layer_candidates.append(layer_params // layer_count)

        if is_first:
            embedding_count = sum(
                int(parameter.numel())
                for name, parameter in parameters
                if name.startswith("embed_tokens.")
            ) or None
            if embedding_count is None:
                return None
        if is_last:
            lm_head_count = sum(
                int(parameter.numel())
                for name, parameter in parameters
                if name.startswith("norm.") or name.startswith("lm_head.")
            ) or None
            if lm_head_count is None:
                return None

        adapter_count += sum(
            int(parameter.numel())
            for name, parameter in parameters
            if "lora_A" in name or "lora_B" in name
        )
        base_dtypes = [
            int(parameter.element_size())
            for name, parameter in parameters
            if "lora_A" not in name and "lora_B" not in name
        ]
        if not base_dtypes:
            return None
        dtype_bytes = max(dtype_bytes, max(base_dtypes))
        adapter_dtypes = [
            int(parameter.element_size())
            for name, parameter in parameters
            if "lora_A" in name or "lora_B" in name
        ]
        if adapter_dtypes:
            adapter_dtype_bytes = max(adapter_dtype_bytes, max(adapter_dtypes))

        stage_attention = (
            getattr(model, "attn_implementation", None)
            or getattr(stage_cfg, "attention_implementation", None)
            or getattr(stage_cfg, "_attn_implementation", "sdpa")
        )
        if not isinstance(stage_attention, str):
            return None
        if attention is None:
            attention = stage_attention
        elif attention != stage_attention:
            return None

    expected_start = 0
    for start, end in stage_ranges:
        if start != expected_start:
            return None
        expected_start = end
    if expected_start != values["num_hidden_layers"] or not seen_first or not seen_last:
        return None
    if not per_layer_candidates or len(set(per_layer_candidates)) != 1:
        return None
    if dtype_bytes < 1 or attention is None:
        return None

    try:
        return model_spec_type(
            num_layers=values["num_hidden_layers"],
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            num_attention_heads=values["num_attention_heads"],
            num_kv_heads=values["num_key_value_heads"],
            head_dim=values["head_dim"],
            vocab_size=values["vocab_size"],
            param_count=param_count,
            dtype_bytes=dtype_bytes,
            per_layer_param_count=per_layer_candidates[0],
            embedding_param_count=embedding_count,
            lm_head_param_count=lm_head_count,
            adapter_param_count=adapter_count,
            attention_implementation=attention,
            max_position_embeddings=max_positions,
            adapter_dtype_bytes=adapter_dtype_bytes,
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_inference_app(
    workers: list[Any],
    admission_cfg: AdmissionConfig | None = None,
    kv_slots: int = 8192,
    tokenizer: Tokenizer | None = None,
    memory_preflight: MemoryPreflight | None = None,
) -> FastAPI:
    if not workers:
        raise ValueError("workers must not be empty")
    cfg = admission_cfg or AdmissionConfig(max_new_tokens=4096, kv_slots=kv_slots)
    budget_slots = cfg.kv_slots if cfg.kv_slots > 0 else kv_slots
    kv_budget = KVBudget(total_slots=budget_slots)
    controller = AdmissionController(cfg, kv_budget)
    if memory_preflight is None:
        memory_preflight = build_cuda_memory_preflight(
            workers,
            # Size the static estimate for the largest number of requests the
            # admission controller can allow.  A per-request estimate here
            # would be fail-open: four individually safe requests could still
            # exceed one stage's KV/activation budget together.
            max_concurrent=cfg.max_concurrent_requests,
        )

    app = FastAPI(title="MeshGPU Inference", version="0.1.0")

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok",
            "active_requests": controller.active_requests,
            "kv_available": kv_budget.available,
        }

    @app.post("/v1/generate", response_model=None)
    async def generate(req: GenerateRequest):
        prompt_ids = _resolve_prompt(req, tokenizer, workers)
        prompt_len = len(prompt_ids)
        max_positions = _worker_attr(workers[0], "max_position_embeddings")
        if max_positions is not None and prompt_len + req.max_new_tokens > max_positions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt ({prompt_len}) + max_new_tokens ({req.max_new_tokens}) "
                    f"exceeds model context {max_positions}"
                ),
            )
        if memory_preflight is not None:
            try:
                decision = _run_memory_preflight(
                    memory_preflight,
                    prompt_ids,
                    req.max_new_tokens,
                )
            except Exception as exc:
                # A preflight failure is safer as a structured capacity
                # rejection than as a 500 followed by an admitted request.
                # In particular, never continue when a live VRAM probe or
                # measured profile could not make a decision.
                log.exception("memory preflight failed before admission")
                raise HTTPException(
                    status_code=507,
                    detail={
                        "code": "insufficient_memory",
                        "reason": f"memory preflight failed: {exc}",
                    },
                ) from exc
            if not decision.feasible:
                raise HTTPException(
                    status_code=507,
                    detail={
                        "code": "insufficient_memory",
                        "reason": decision.reason,
                        **decision.details,
                    },
                )
        ok, reason = controller.admit(prompt_len, req.max_new_tokens)
        if not ok:
            raise HTTPException(status_code=429, detail=reason)

        session_id = str(uuid.uuid4())
        try:
            sampling = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                seed=req.seed,
                eos_token_id=tokenizer.eos_token_id if tokenizer is not None else None,
            )
            session = InferenceSession(
                session_id=session_id,
                prompt_ids=prompt_ids,
                workers=workers,
                sampling=sampling,
                max_new_tokens=req.max_new_tokens,
            )
        except Exception:
            # Admission succeeded, so every construction failure must release
            # its reservation before the error reaches the client.
            controller.release(prompt_len, req.max_new_tokens)
            raise

        if req.stream:
            return StreamingResponse(
                _stream_sse(
                    session,
                    controller,
                    prompt_len,
                    req.max_new_tokens,
                    tokenizer,
                    cfg.request_deadline_s,
                ),
                media_type="text/event-stream",
                headers={
                    "X-Session-Id": session_id,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                token_ids = []
                seq_nums = []

                async def _collect() -> None:
                    async for result in session.run():
                        token_ids.append(result.token_id)
                        seq_nums.append(result.seq_num)

                try:
                    await asyncio.wait_for(_collect(), cfg.request_deadline_s)
                except asyncio.TimeoutError as exc:
                    raise HTTPException(
                        status_code=504,
                        detail="inference request deadline exceeded",
                    ) from exc
                return GenerateResponse(
                    session_id=session_id,
                    token_ids=token_ids,
                    seq_nums=seq_nums,
                    tokens_per_second=session.tokens_per_second,
                    text=tokenizer.decode(token_ids) if tokenizer is not None else "",
                )
            finally:
                controller.release(prompt_len, req.max_new_tokens)

    return app


def _resolve_prompt(
    request: GenerateRequest,
    tokenizer: Tokenizer | None,
    workers: list[StageWorker],
) -> list[int]:
    if request.prompt_ids is not None:
        prompt_ids = list(request.prompt_ids)
    elif tokenizer is not None and request.prompt is not None:
        try:
            prompt_ids = tokenizer.encode(request.prompt, add_special_tokens=True)
        except (TokenizerError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not tokenize prompt: {exc}",
            ) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="raw text prompt requires a tokenizer loaded with the model artifact",
        )
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="prompt produced no tokens")

    vocab_size = _worker_attr(workers[0], "vocab_size") if workers else None
    if vocab_size is not None and any(token < 0 or token >= vocab_size for token in prompt_ids):
        raise HTTPException(status_code=400, detail="prompt contains an out-of-vocabulary token")
    return prompt_ids


def _run_memory_preflight(
    checker: MemoryPreflight,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> MemoryPreflightResult:
    """Normalize planner callbacks while requiring an explicit decision."""
    raw = checker(prompt_ids, max_new_tokens)
    if isinstance(raw, MemoryPreflightResult):
        return raw
    if isinstance(raw, bool):
        return MemoryPreflightResult(
            feasible=raw,
            reason="memory profile accepted" if raw else "memory profile rejected",
        )
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
        return MemoryPreflightResult(feasible=raw[0], reason=str(raw[1]))
    raise TypeError(
        "memory_preflight must return MemoryPreflightResult, bool, or (bool, reason)"
    )


def _worker_attr(worker: StageWorker, name: str):
    model_cfg = getattr(getattr(worker, "_model", None), "cfg", None)
    value = getattr(model_cfg, name, None)
    return value if value is not None else getattr(worker, name, None)


async def _stream_sse(
    session: InferenceSession,
    controller: AdmissionController,
    prompt_len: int,
    max_new_tokens: int,
    tokenizer: Tokenizer | None = None,
    deadline_s: float = 300.0,
) -> AsyncIterator[str]:
    previous_text = ""
    session_iterator = session.run()
    try:
        iterator = session_iterator.__aiter__()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + deadline_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("inference request deadline exceeded")
            try:
                result = await asyncio.wait_for(iterator.__anext__(), remaining)
            except StopAsyncIteration:
                break
            delta = ""
            if tokenizer is not None:
                previous_text, delta = tokenizer.delta_decode(
                    session.generated_ids,
                    previous_text,
                )
            event = TokenEvent(
                token_id=result.token_id,
                seq_num=result.seq_num,
                is_last=result.is_last,
                text=delta,
            )
            yield f"data: {event.model_dump_json()}\n\n"
            if result.is_last:
                break
        # Final stats event
        stats = {
            "event": "done",
            "tokens_per_second": session.tokens_per_second,
            "total_tokens": len(session.generated_ids),
        }
        yield f"data: {json.dumps(stats)}\n\n"
    except Exception as e:
        err = {"event": "error", "message": str(e)}
        yield f"data: {json.dumps(err)}\n\n"
        log.exception("inference session error: %s", session.session_id)
    finally:
        # If the client disconnects immediately after the final token, the
        # async generator is suspended at its last ``yield``.  Explicitly
        # close it so StageWorker finally blocks release named KV caches now,
        # rather than waiting for garbage collection.
        try:
            await session_iterator.aclose()
        finally:
            controller.release(prompt_len, max_new_tokens)
