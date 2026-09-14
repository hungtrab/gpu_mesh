"""
Topology change: reshard a model across a different number of stages.

Scenario:
  - Training was running with N stages (N GPUs).
  - One or more workers died; only M < N replacement workers are available.
  - Or: M > N new workers joined and we want to rebalance.
  - Solution: load the last committed checkpoint into a fresh M-stage pipeline.

Algorithm:
  1. Load each stage shard from the checkpoint (N shards, each covering a layer range).
  2. Build a new M-stage pipeline with the correct layer partition.
  3. For each new stage, collect all layers it should own, copying parameters from
     whichever old stage owned those layers.
  4. Return the new list of StageWorker, ready to resume training.

Key invariant: layer ranges in checkpoint are contiguous and cover [0, num_layers).
The new partition is recalculated with build_pipeline's even-split logic.
"""
from __future__ import annotations

import logging
from typing import cast

import torch

from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.backends.portable.stage_worker import StageWorker
from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage

log = logging.getLogger(__name__)


def reshard_from_checkpoint(
    cfg: LlamaConfig,
    checkpoint_id: str,
    coord: CheckpointCoordinator,
    old_num_stages: int,
    new_num_stages: int,
    devices: list[torch.device] | None = None,
) -> list[StageWorker]:
    """
    Reconstruct a new pipeline with new_num_stages stages from a checkpoint
    that was saved with old_num_stages stages.

    Returns a list of StageWorker (model weights loaded, optimizer NOT restored —
    caller should call trainer.replace_worker or rebuild optimizers).  LoRA
    checkpoints are rejected because their wrapper configuration is not part
    of the current checkpoint format.
    """
    n = cfg.num_hidden_layers
    for name, value in (
        ("old_num_stages", old_num_stages),
        ("new_num_stages", new_num_stages),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= value <= n:
            raise ValueError(
                f"{name} must be in [1, {n}], got {value}"
            )
    if devices is None:
        devices = [torch.device("cpu")] * new_num_stages
    if len(devices) != new_num_stages:
        raise ValueError(f"need {new_num_stages} devices, got {len(devices)}")

    # --- Load old stage state dicts ---
    old_states: list[dict] = []
    old_ranges: list[tuple[int, int]] = []

    old_base, old_rem = divmod(n, old_num_stages)
    cursor = 0
    for i in range(old_num_stages):
        n_l = old_base + (1 if i < old_rem else 0)
        old_ranges.append((cursor, cursor + n_l))
        cursor += n_l
        payload = coord.load_stage(checkpoint_id, stage=i)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint stage {i} payload is not a mapping")
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise ValueError(
                f"checkpoint stage {i} has no valid model state dict"
            )
        old_states.append(model_state)

    lora_keys = [
        key
        for state in old_states
        for key in state
        if "lora_A" in key or "lora_B" in key or ".base." in key
    ]
    if lora_keys:
        raise ValueError(
            "LoRA checkpoints cannot be resharded safely: the checkpoint "
            "does not contain enough adapter configuration to reconstruct "
            "the wrapped layers"
        )

    log.info(
        "reshard: loaded %d old stages, layer ranges %s",
        old_num_stages, old_ranges,
    )

    # --- Build full merged state dict (embed + all layers + lm_head) ---
    full_model = LlamaStage(
        cfg, 0, n,
        has_embedding=True,
        has_lm_head=True,
    )
    # Copy layer by layer from old stage state dicts
    _merge_into_full(full_model, old_states, old_ranges, cfg, n)

    # --- Build new pipeline and distribute weights ---
    new_workers = build_pipeline(cfg, new_num_stages, devices)
    new_base, new_rem = divmod(n, new_num_stages)
    new_cursor = 0
    for i, worker in enumerate(new_workers):
        n_l = new_base + (1 if i < new_rem else 0)
        ls, le = new_cursor, new_cursor + n_l
        new_cursor = le
        _copy_stage_weights(full_model, cast(LlamaStage, worker._model), ls, le,
                            is_first=(i == 0), is_last=(i == new_num_stages - 1))
        log.info("reshard: stage %d ← layers [%d, %d)", i, ls, le)

    log.info("reshard complete: %d → %d stages", old_num_stages, new_num_stages)
    return new_workers


def _merge_into_full(
    full_model: LlamaStage,
    old_states: list[dict],
    old_ranges: list[tuple[int, int]],
    cfg: LlamaConfig,
    num_layers: int,
) -> None:
    """
    Fill full_model's state dict by copying from each old stage's state dict.
    embed_tokens comes from stage 0; norm + lm_head come from last stage.
    """
    if len(old_states) != len(old_ranges):
        raise ValueError("old stage states and layer ranges must have the same length")
    full_sd = full_model.state_dict()
    assigned: set[str] = set()

    for i, (sd, (ls, le)) in enumerate(zip(old_states, old_ranges)):
        is_first = i == 0
        is_last = i == len(old_states) - 1

        # Remap layer keys: old stage stores "layers.0.*" but full model uses
        # "layers.<ls>.*" through "layers.<le-1>.*"
        for old_key, value in sd.items():
            if old_key.startswith("layers."):
                parts = old_key.split(".", 2)
                if len(parts) != 3:
                    raise ValueError(f"invalid layer key in checkpoint: {old_key!r}")
                try:
                    rel_idx = int(parts[1])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid layer index in checkpoint key: {old_key!r}"
                    ) from exc
                if not 0 <= rel_idx < le - ls:
                    raise ValueError(
                        f"checkpoint key {old_key!r} is outside stage {i} range "
                        f"[{ls}, {le})"
                    )
            new_key = _remap_key(old_key, ls, is_first, is_last)
            if new_key not in full_sd:
                raise ValueError(
                    f"unsupported or misplaced checkpoint key {old_key!r} "
                    f"(mapped to {new_key!r})"
                )
            if new_key in assigned:
                raise ValueError(f"duplicate checkpoint key after remap: {new_key!r}")
            if not torch.is_tensor(value):
                raise ValueError(f"checkpoint value for {old_key!r} is not a tensor")
            expected = full_sd[new_key]
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError(
                    f"checkpoint shape mismatch for {old_key!r}: "
                    f"got {tuple(value.shape)}, expected {tuple(expected.shape)}"
                )
            full_sd[new_key] = value
            assigned.add(new_key)

    missing = sorted(set(full_sd) - assigned)
    if missing:
        raise ValueError(
            "checkpoint does not cover the full model; missing keys: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )

    full_model.load_state_dict(full_sd, strict=True)


def _remap_key(key: str, layer_start: int, is_first: bool, is_last: bool) -> str:
    """
    Stage stores layers as "layers.0.*", "layers.1.*" etc.
    Full model uses absolute indices "layers.<layer_start+i>.*".
    Embed / norm / lm_head are not remapped.
    """
    if key.startswith("layers."):
        parts = key.split(".", 2)
        if len(parts) != 3 or not parts[2]:
            raise ValueError(f"invalid layer key: {key!r}")
        try:
            rel_idx = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid layer index in key: {key!r}") from exc
        if rel_idx < 0:
            raise ValueError(f"layer index must be non-negative: {key!r}")
        abs_idx = layer_start + rel_idx
        return f"layers.{abs_idx}.{parts[2]}"
    if key.startswith("embed_tokens.") and not is_first:
        raise ValueError(f"embedding key {key!r} belongs only to the first stage")
    if (key.startswith("norm.") or key.startswith("lm_head.")) and not is_last:
        raise ValueError(f"output key {key!r} belongs only to the last stage")
    # embed_tokens, norm, lm_head stay as-is
    return key


def _copy_stage_weights(
    full_model: LlamaStage,
    stage: LlamaStage,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
) -> None:
    """Copy weights from full_model into stage for layers [layer_start, layer_end)."""
    if not 0 <= layer_start < layer_end <= full_model.cfg.num_hidden_layers:
        raise ValueError(f"invalid target layer range [{layer_start}, {layer_end})")
    if (stage.layer_start, stage.layer_end) != (layer_start, layer_end):
        raise ValueError(
            "target stage range does not match requested range: "
            f"stage=[{stage.layer_start}, {stage.layer_end}), "
            f"requested=[{layer_start}, {layer_end})"
        )
    if stage.has_embedding != is_first or stage.has_lm_head != is_last:
        raise ValueError("target stage boundary flags do not match its position")
    full_sd = full_model.state_dict()
    stage_sd = stage.state_dict()

    for key in list(stage_sd.keys()):
        if key.startswith("layers."):
            parts = key.split(".", 2)
            rel_idx = int(parts[1])
            abs_key = f"layers.{layer_start + rel_idx}.{parts[2]}"
            if abs_key not in full_sd:
                raise ValueError(f"full model is missing source key {abs_key!r}")
            stage_sd[key] = full_sd[abs_key]
        elif key in full_sd:
            stage_sd[key] = full_sd[key]
        else:
            raise ValueError(f"full model is missing source key {key!r}")

    stage.load_state_dict(stage_sd, strict=True)
