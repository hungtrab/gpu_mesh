"""Parse a small, explicit YAML job spec into a memory placement report."""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from meshgpu.planner.placement import (
    InferenceWorkload,
    ModelSpec,
    PlacementReport,
    TrainingWorkload,
    WorkerSpec,
    plan_inference,
    plan_training,
)

_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}


def plan_job(spec: dict[str, Any], *, base_dir: str | Path = ".") -> PlacementReport:
    """Build a report from a YAML-decoded job dictionary.

    Worker VRAM is intentionally required in the spec (or in the optional
    ``worker_specs`` map).  Guessing capacity would make an infeasible job look
    safe, so a list of worker IDs alone fails with an actionable message.
    """
    if not isinstance(spec, dict):
        raise ValueError("job spec must be a mapping")
    model_raw = spec.get("model")
    if not isinstance(model_raw, dict):
        raise ValueError("job spec requires a model mapping")
    model = _parse_model(model_raw, base_dir=Path(base_dir))
    workers = _parse_workers(spec)
    if not workers:
        raise ValueError(
            "job spec requires placement.workers with VRAM values; "
            "use worker_specs or inline worker mappings"
        )

    resources = spec.get("resources") or {}
    if not isinstance(resources, dict):
        raise ValueError("resources must be a mapping")
    fraction_value = resources.get("max_vram_fraction")
    if fraction_value is None:
        fraction_value = spec.get("max_vram_fraction", 0.85)
    fraction = float(cast(float | int | str, fraction_value))
    reserve_value = resources.get("reserve_min_gib", 1)
    reserve = _size_bytes(reserve_value, default_unit="gib")
    task = str(spec.get("task", "inference")).lower()
    if task == "inference":
        inference_raw = spec.get("inference") or {}
        if not isinstance(inference_raw, dict):
            raise ValueError("inference must be a mapping")
        max_concurrent_value = inference_raw.get(
            "max_concurrent_requests", inference_raw.get("max_concurrent", 1)
        )
        inference_workload = InferenceWorkload(
            batch_size=int(inference_raw.get("batch_size", 1)),
            max_prompt_tokens=int(inference_raw.get("max_prompt_tokens", 2048)),
            max_new_tokens=int(inference_raw.get("max_new_tokens", 256)),
            max_concurrent=int(cast(int | float | str, max_concurrent_value)),
        )
        return plan_inference(
            workers,
            model,
            inference_workload,
            max_vram_fraction=fraction,
            reserve_min_bytes=reserve,
        )
    if task == "training":
        training_raw = spec.get("training") or {}
        if not isinstance(training_raw, dict):
            raise ValueError("training must be a mapping")
        training_workload = TrainingWorkload(
            batch_size=int(training_raw.get("batch_size", 1)),
            sequence_length=int(training_raw.get("sequence_length", 512)),
            gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 1)),
            activation_checkpointing=_read_bool(
                training_raw.get("activation_checkpointing", False),
                "training.activation_checkpointing",
            ),
        )
        return plan_training(
            workers,
            model,
            training_workload,
            max_vram_fraction=fraction,
            reserve_min_bytes=reserve,
            optimizer=str(training_raw.get("optimizer", "adamw")),
        )
    raise ValueError(f"unsupported task for planner: {task!r}")


def _read_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def report_to_dict(report: PlacementReport) -> dict[str, Any]:
    """Serialize derived properties that ``dataclasses.asdict`` omits."""
    result = asdict(report)
    for assignment in result["assignments"]:
        budget = assignment["budget"]
        # These are properties, so calculate them from the serialized fields.
        reserve = max(
            budget["reserve_min_bytes"],
            int(budget["total_vram_bytes"] * 0.10),
        )
        budget["reserve_bytes"] = reserve
        budget["usable_bytes"] = min(
            int(budget["total_vram_bytes"] * budget["user_budget_fraction"]),
            budget["free_vram_at_admission"] - reserve,
        )
    result["summary"] = report.summary()
    return result


def _parse_model(raw: dict[str, Any], *, base_dir: Path) -> ModelSpec:
    manifest_ref = raw.get("manifest")
    if manifest_ref:
        path = Path(manifest_ref)
        if not path.is_absolute():
            path = base_dir / path
        return model_spec_from_manifest(
            path if path.is_dir() else path.parent,
        )

    required = (
        "num_layers", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_kv_heads", "head_dim", "vocab_size",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"model is missing fields: {', '.join(missing)}")
    attention_implementation = raw.get(
        "attention_implementation",
        raw.get("attn_implementation", "sdpa"),
    )
    if (
        "attention_implementation" in raw
        and "attn_implementation" in raw
        and raw["attention_implementation"] != raw["attn_implementation"]
    ):
        raise ValueError(
            "model attention_implementation and attn_implementation disagree"
        )
    if not isinstance(attention_implementation, str):
        raise TypeError("model attention_implementation must be a string")
    return ModelSpec(
        **{key: int(raw[key]) for key in required},
        param_count=int(raw.get("param_count", 0)),
        dtype_bytes=int(raw.get("dtype_bytes", 2)),
        attention_implementation=attention_implementation,
        max_position_embeddings=(
            int(raw["max_position_embeddings"])
            if raw.get("max_position_embeddings") is not None
            else None
        ),
    )


def model_spec_from_manifest(
    manifest_or_dir: Any,
    *,
    artifact_root: str | Path | None = None,
) -> ModelSpec:
    """Derive an exact-enough planner model from an immutable artifact.

    Only tensor metadata is read (``map_location='meta'``); no model weights
    are materialized.  This makes the helper suitable for a preflight that
    must run *before* a CUDA stage is constructed.  The component counts are
    taken from the shard state dicts instead of assuming that every decoder
    family has the same parameter formula.
    """
    from meshgpu.artifacts.manifest import ModelManifest

    if isinstance(manifest_or_dir, ModelManifest):
        manifest = manifest_or_dir
        root = Path(artifact_root) if artifact_root is not None else Path(".")
    else:
        path = Path(manifest_or_dir)
        root = path if path.is_dir() else path.parent
        manifest = ModelManifest.load(root / "manifest.json")

    if manifest.meta.adapter == "qwen3_hf_v1":
        from meshgpu.artifacts.qwen_import import load_qwen_config

        cfg = load_qwen_config(manifest, artifact_root=root)
    else:
        from meshgpu.artifacts.hf_import import (
            load_llama_config,
            validate_manifest_config,
        )

        cfg = load_llama_config(manifest, artifact_root=root)
        validate_manifest_config(manifest, cfg)

    if manifest.config_hash:
        # Hash the serialized config file rather than normalizing a
        # reconstructed object.  Hugging Face configs are not dataclasses and
        # their ``from_dict``/``to_dict`` round-trip can also add or remove
        # implementation-only fields (Qwen3 is one example).  The artifact
        # hash covers the exact JSON that the loader validated and used.
        if manifest.config_path is not None:
            from meshgpu.artifacts.hf_import import _resolve_artifact_path

            config_path = _resolve_artifact_path(root, manifest.config_path)
            try:
                config_values = json.loads(config_path.read_text())
            except FileNotFoundError as exc:
                raise ValueError(
                    f"artifact config file not found: {config_path}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"artifact config file is not valid JSON: {config_path}"
                ) from exc
            if not isinstance(config_values, dict):
                raise ValueError("artifact config file must contain a JSON object")
        elif hasattr(cfg, "to_dict"):
            config_values = cfg.to_dict()
        else:
            config_values = asdict(cfg)
        actual_config_hash = ModelManifest.compute_config_hash(config_values)
        if actual_config_hash != manifest.config_hash:
            raise ValueError(
                "artifact config hash mismatch: the planner config does not match "
                f"manifest (expected {manifest.config_hash}, got {actual_config_hash})"
            )

    component_counts = _manifest_component_counts(manifest, root)
    if component_counts["per_layer"] is None:
        raise ValueError(
            "artifact decoder layers do not have a uniform, complete parameter "
            "schema; refusing to approximate preflight memory"
        )
    if component_counts["embedding"] is None or component_counts["lm_head"] is None:
        raise ValueError(
            "artifact is missing a complete embedding or output-head parameter set"
        )
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(
        manifest.meta.compute_dtype
    )
    if dtype_bytes is None:  # ManifestMeta normally rejects this first.
        raise ValueError(
            f"unsupported manifest compute_dtype: {manifest.meta.compute_dtype!r}"
        )

    param_count = 0
    for shard in manifest.shards:
        state = _load_verified_shard_tensors(root, shard)
        if not isinstance(state, dict) or any(
            not isinstance(key, str) for key in state
        ):
            raise ValueError(f"shard {shard.shard_id} must contain a string-keyed state dict")
        param_count += sum(int(tensor.numel()) for tensor in state.values())

    return ModelSpec(
        num_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        num_attention_heads=int(cfg.num_attention_heads),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        vocab_size=int(cfg.vocab_size),
        param_count=param_count,
        dtype_bytes=dtype_bytes,
        per_layer_param_count=component_counts["per_layer"],
        embedding_param_count=component_counts["embedding"],
        lm_head_param_count=component_counts["lm_head"],
        adapter_param_count=int(component_counts["adapter"] or 0),
        attention_implementation=_model_attention_from_manifest(manifest, cfg),
        max_position_embeddings=int(cfg.max_position_embeddings),
        adapter_dtype_bytes=4,
    )


def lora_param_breakdown_from_manifest(
    manifest_or_dir: Any,
    rank: int,
    *,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    modules_to_save: tuple[str, ...] = (),
    artifact_root: str | Path | None = None,
) -> dict[str, int]:
    """Count adapter parameters and keep endpoint modules attached to stages.

    The count is obtained from the actual projection shapes in the artifact,
    so GQA and official Qwen blocks are handled without hard-coded output
    dimensions.  Missing or malformed targets fail closed instead of letting
    a training preflight under-estimate optimizer memory.  ``other`` contains
    full saved modules that are neither decoder-layer nor embedding/head
    modules; callers that cannot represent those modules must reject them.
    """
    from meshgpu.artifacts.manifest import ModelManifest

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    targets = tuple(dict.fromkeys(target_modules))
    if not targets or any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("target_modules must contain non-empty strings")
    saved_modules = tuple(dict.fromkeys(modules_to_save))
    if any(not isinstance(name, str) or not name for name in saved_modules):
        raise ValueError("modules_to_save must contain non-empty strings")

    if isinstance(manifest_or_dir, ModelManifest):
        manifest = manifest_or_dir
        root = Path(artifact_root) if artifact_root is not None else Path(".")
    else:
        path = Path(manifest_or_dir)
        root = path if path.is_dir() else path.parent
        manifest = ModelManifest.load(root / "manifest.json")

    # Load and verify each shard once.  Besides avoiding duplicate disk/hash
    # work, keeping this snapshot makes every count in this function refer to
    # exactly the same immutable artifact files.
    shard_states = [
        (shard, _load_verified_shard_tensors(root, shard))
        for shard in manifest.shards
    ]
    layer_targets = tuple(
        target for target in targets if not _is_endpoint_module_name(target)
    )
    endpoint_targets = tuple(
        target for target in targets if _is_endpoint_module_name(target)
    )

    per_layer: dict[int, int] = {}
    for shard, state in shard_states:
        for local_index in range(shard.layer_end - shard.layer_start):
            global_index = shard.layer_start + local_index
            layer_total = 0
            matched_weights: dict[str, Any] = {}
            for target in layer_targets:
                matches = [
                    (key, tensor)
                    for key, tensor in state.items()
                    if _is_layer_target_weight(key, local_index, target)
                ]
                if not matches:
                    raise ValueError(
                        f"artifact shard {shard.shard_id} is missing LoRA target "
                        f"{target!r} in layer {local_index}"
                    )
                for key, tensor in matches:
                    # PEFT-style target aliases (for example ``q_proj`` and
                    # ``self_attn.q_proj``) can resolve to the same module.
                    # Count the module once, exactly as ``apply_lora`` does.
                    matched_weights[key] = tensor
            for key, tensor in matched_weights.items():
                _validate_lora_weight(tensor, key)
                module_path = key[: -len(".weight")]
                if any(
                    _matches_module_name(module_path, saved_name)
                    for saved_name in saved_modules
                ):
                    raise ValueError(
                        f"LoRA target {module_path!r} cannot also be listed in "
                        "modules_to_save"
                    )
                out_features, in_features = int(tensor.shape[0]), int(tensor.shape[1])
                layer_total += rank * (in_features + out_features)
            if global_index in per_layer:
                raise ValueError(f"duplicate artifact layer {global_index}")
            per_layer[global_index] = layer_total

    expected = set(range(manifest.meta.num_layers))
    if set(per_layer) != expected:
        missing = sorted(expected - set(per_layer))
        extra = sorted(set(per_layer) - expected)
        raise ValueError(
            "artifact layers do not cover the model for LoRA counting: "
            f"missing={missing}, extra={extra}"
        )
    layer_total = sum(per_layer.values())
    total = layer_total
    endpoint_counts = {"embedding": 0, "lm_head": 0, "other": 0}
    matched_endpoint_weights: dict[str, Any] = {}
    for target in endpoint_targets:
        matches = [
            (key, tensor)
            for _shard, state in shard_states
            for key, tensor in state.items()
            if _is_endpoint_target_weight(key, target)
        ]
        if not matches:
            raise ValueError(
                f"artifact does not contain LoRA endpoint target {target!r}"
            )
        for key, tensor in matches:
            # As above, aliases must not count one endpoint twice.
            matched_endpoint_weights[key] = tensor
    for key, tensor in matched_endpoint_weights.items():
        _validate_lora_weight(tensor, key)
        module_path = key[: -len(".weight")]
        if any(
            _matches_module_name(module_path, saved_name)
            for saved_name in saved_modules
        ):
            raise ValueError(
                f"LoRA target {module_path!r} cannot also be listed in "
                "modules_to_save"
            )
        out_features, in_features = int(tensor.shape[0]), int(tensor.shape[1])
        count = rank * (in_features + out_features)
        total += count
        if _matches_module_name(module_path, "embed_tokens"):
            endpoint_counts["embedding"] += count
        elif _matches_module_name(module_path, "lm_head"):
            endpoint_counts["lm_head"] += count
        else:  # guarded by _is_endpoint_module_name, defensive only
            endpoint_counts["other"] += count

    if saved_modules:
        for _shard, state in shard_states:
            # apply_lora enables every parameter below each matching module,
            # not just weight.  Build a union of matching module paths first
            # so a module with a bias (or nested parameters) is counted once.
            matched_module_paths: set[str] = set()
            for key in state:
                if not isinstance(key, str) or "." not in key:
                    continue
                module_path = key.rsplit(".", 1)[0]
                if any(
                    module_path == name or module_path.endswith("." + name)
                    for name in saved_modules
                ):
                    matched_module_paths.add(module_path)
            for key, tensor in state.items():
                if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                    continue
                parameter_path = key.rsplit(".", 1)[0] if "." in key else ""
                matching_paths = [
                    module_path
                    for module_path in matched_module_paths
                    if parameter_path == module_path
                    or parameter_path.startswith(module_path + ".")
                ]
                if not matching_paths:
                    continue
                count = int(tensor.numel())
                total += count
                if any(_matches_module_name(path, "embed_tokens") for path in matching_paths):
                    endpoint_counts["embedding"] += count
                elif any(_matches_module_name(path, "lm_head") for path in matching_paths):
                    endpoint_counts["lm_head"] += count
                else:
                    endpoint_counts["other"] += count
        for saved_name in saved_modules:
            if not any(
                _matches_module_name(path, saved_name)
                for _shard, state in shard_states
                for path in _module_paths_with_parameters(state)
            ):
                raise ValueError(
                    f"artifact does not contain modules_to_save target {saved_name!r}"
                )
    return {
        "total": total,
        "layer": layer_total,
        **endpoint_counts,
    }


def lora_param_count_from_manifest(
    manifest_or_dir: Any,
    rank: int,
    *,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    modules_to_save: tuple[str, ...] = (),
    artifact_root: str | Path | None = None,
) -> int:
    """Return the total adapter parameter count for compatibility callers."""
    return lora_param_breakdown_from_manifest(
        manifest_or_dir,
        rank,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        artifact_root=artifact_root,
    )["total"]


def _matches_module_name(path: str, name: str) -> bool:
    return path == name or path.endswith("." + name)


def _is_endpoint_module_name(name: str) -> bool:
    return _matches_module_name(name, "embed_tokens") or _matches_module_name(
        name, "lm_head"
    )


def _module_path_from_weight_key(key: Any) -> str | None:
    if not isinstance(key, str) or not key.endswith(".weight"):
        return None
    return key[: -len(".weight")]


def _is_layer_target_weight(key: Any, local_index: int, target: str) -> bool:
    module_path = _module_path_from_weight_key(key)
    if module_path is None:
        return False
    prefix = f"layers.{local_index}."
    return module_path.startswith(prefix) and _matches_module_name(module_path, target)


def _is_endpoint_target_weight(key: Any, target: str) -> bool:
    module_path = _module_path_from_weight_key(key)
    return module_path is not None and _matches_module_name(module_path, target)


def _validate_lora_weight(tensor: Any, key: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"LoRA target {key!r} is not a tensor")
    if tensor.ndim != 2 or any(int(dim) < 1 for dim in tensor.shape):
        raise ValueError(f"LoRA target {key!r} must be a non-empty rank-2 weight")


def _module_paths_with_parameters(state: dict[str, Any]) -> set[str]:
    return {
        module_path
        for key in state
        if (module_path := _module_path_from_weight_key(key)) is not None
    }


def _model_attention_from_manifest(manifest: Any, cfg: Any) -> str:
    """Return the adapter's recorded attention backend.

    The manifest is the immutable artifact contract.  The config is retained
    as a fallback for older artifacts that predate the extra metadata field.
    """
    extra = getattr(manifest.meta, "extra", {})
    recorded = extra.get("attn_implementation")
    if recorded is not None:
        if not isinstance(recorded, str):
            raise TypeError("manifest attn_implementation must be a string")
        return recorded
    return str(
        getattr(cfg, "attention_implementation", None)
        or getattr(cfg, "_attn_implementation", "sdpa")
    )


def _load_shard_tensors(path: Path) -> dict[str, Any]:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"model shard not found: {path}")
    return torch.load(path, map_location="meta", weights_only=True)


def _load_verified_shard_tensors(root: Path, shard: Any) -> dict[str, Any]:
    """Read planner metadata only after verifying the manifest evidence.

    Preflight must not size a different file from the one the loader will
    execute.  In particular, a replaced shard could otherwise change the
    parameter/adapter count between planning and loading, or escape through a
    symlink if the path were joined naively.
    """
    from meshgpu.artifacts.hf_import import _resolve_artifact_path, _sha256_file

    path = _resolve_artifact_path(root, shard.path)
    if not path.exists():
        raise FileNotFoundError(f"model shard not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != shard.byte_length:
        raise ValueError(
            f"shard {shard.shard_id} size mismatch: expected {shard.byte_length}, "
            f"got {actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != shard.sha256:
        raise ValueError(
            f"shard {shard.shard_id} hash mismatch: expected {shard.sha256}, "
            f"got {actual_sha}"
        )
    state = _load_shard_tensors(path)
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError(f"shard {shard.shard_id} must contain a string-keyed state dict")
    if set(state) != set(shard.tensor_names):
        raise ValueError(
            f"shard {shard.shard_id} tensor_names do not match its state dict"
        )
    return state


def _manifest_component_counts(manifest: Any, root: Path) -> dict[str, int | None]:
    """Read exact parameter components without materializing tensor storage."""
    layer_counts: dict[int, int] = {}
    embedding = 0
    lm_head = 0
    adapter = 0
    for shard in manifest.shards:
        for name, tensor in _load_verified_shard_tensors(root, shard).items():
            count = int(tensor.numel())
            if name == "embed_tokens.weight":
                embedding += count
            elif name in {"lm_head.weight", "norm.weight"}:
                lm_head += count
            elif name.startswith("layers."):
                local_index = int(name.split(".", 2)[1])
                if "lora_A" in name or "lora_B" in name:
                    adapter += count
                else:
                    layer_counts[shard.layer_start + local_index] = (
                        layer_counts.get(shard.layer_start + local_index, 0) + count
                    )
    per_layer = None
    if layer_counts and len(layer_counts) == manifest.meta.num_layers:
        values = set(layer_counts.values())
        if len(values) == 1:
            per_layer = values.pop()
    return {
        "per_layer": per_layer,
        "embedding": embedding or None,
        "lm_head": lm_head or None,
        "adapter": adapter,
    }


def _parse_workers(spec: dict[str, Any]) -> list[WorkerSpec]:
    placement = spec.get("placement") or {}
    if not isinstance(placement, dict):
        raise ValueError("placement must be a mapping")
    raw_workers = placement.get("workers", spec.get("workers", []))
    specs = spec.get("worker_specs", {})
    if not isinstance(specs, dict):
        raise ValueError("worker_specs must be a mapping")
    if not isinstance(raw_workers, list):
        raise ValueError("placement.workers must be a list")
    result = []
    for index, raw in enumerate(raw_workers):
        if isinstance(raw, str):
            worker_spec = specs.get(raw)
            if worker_spec is not None and not isinstance(worker_spec, dict):
                raise ValueError(f"worker_specs[{raw!r}] must be a mapping")
            raw = {"worker_id": raw, **(worker_spec or {})}
        if not isinstance(raw, dict):
            raise ValueError("each placement worker must be a string or mapping")
        worker_id = str(raw.get("worker_id", raw.get("id", f"worker-{index}")))
        total = _read_size(raw, "total_vram", "total_vram_bytes", "total_vram_gib")
        free = _read_size(raw, "free_vram", "free_vram_bytes", "free_vram_gib", default=total)
        result.append(
            WorkerSpec(
                worker_id=worker_id,
                device_index=int(raw.get("device_index", index)),
                total_vram_bytes=total,
                free_vram_bytes=free,
                compute_score=float(raw.get("compute_score", 1.0)),
                goodput_mbit_s=(
                    float(raw["goodput_mbit_s"])
                    if raw.get("goodput_mbit_s") is not None else None
                ),
            )
        )
    return result


def _read_size(
    raw: dict[str, Any],
    generic: str,
    bytes_key: str,
    unit_key: str,
    *,
    default: int | None = None,
) -> int:
    if bytes_key in raw:
        return _size_bytes(raw[bytes_key])
    if unit_key in raw:
        return _size_bytes(raw[unit_key], default_unit="gib")
    if generic in raw:
        return _size_bytes(raw[generic])
    if default is not None:
        return default
    raise ValueError(f"worker is missing {bytes_key} or {unit_key} (VRAM capacity)")


def _size_bytes(value: Any, *, default_unit: str = "b") -> int:
    if isinstance(value, bool):
        raise ValueError("size cannot be boolean")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("size cannot be negative")
        return int(value * _SIZE_UNITS[default_unit])
    if not isinstance(value, str):
        raise ValueError(f"invalid size value: {value!r}")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", value)
    if not match:
        raise ValueError(f"invalid size value: {value!r}")
    unit = match.group(2).lower() or default_unit
    if unit not in _SIZE_UNITS:
        raise ValueError(f"unknown size unit: {unit!r}")
    return int(float(match.group(1)) * _SIZE_UNITS[unit])
