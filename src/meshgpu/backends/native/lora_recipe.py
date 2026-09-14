"""LoRA recipes shared by native and portable MeshGPU training.

The implementation intentionally stays small and dependency-free.  It follows
the parts of PEFT's LoRA contract that matter for a sharded decoder stage:

* target names match module leaf names (``q_proj`` also matches
  ``layers.0.self_attn.q_proj``);
* ``use_rslora`` selects :math:`alpha / sqrt(r)` scaling;
* ``modules_to_save`` keeps complete modules trainable.  This is the
  important distinction for NVARC-style recipes where the input embedding and
  output head are saved as full weights rather than low-rank factors;
* injection is idempotent and validates the complete layer target set before
  changing a stage.

This is deliberately separate from QLoRA/quantization.  The base module is
frozen, adapters stay in fp32, and the stage's compute dtype is preserved.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for a stage-local LoRA recipe.

    ``modules_to_save`` uses module leaf names, just like PEFT.  Missing
    endpoint modules are allowed on a stage that does not own that endpoint:
    stage 0 may have ``embed_tokens`` while the last stage has ``lm_head``.
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.0
    use_rslora: bool = False
    modules_to_save: list[str] = field(default_factory=list)

    @property
    def scale(self) -> float:
        """Return the LoRA multiplier used by the forward path."""
        return self.alpha / math.sqrt(self.rank) if self.use_rslora else self.alpha / self.rank

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("LoRA rank must be an integer")
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("LoRA alpha must be numeric")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("LoRA dropout must be numeric")
        if not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        if not isinstance(self.use_rslora, bool):
            raise TypeError("LoRA use_rslora must be boolean")
        _validate_names("target_modules", self.target_modules, allow_empty=False)
        _validate_names("modules_to_save", self.modules_to_save, allow_empty=True)

        # Normalize duplicate names once so serialization and validation are
        # deterministic.  Preserve caller order because it makes config logs
        # and checkpoint metadata easier to compare.
        self.target_modules = list(dict.fromkeys(self.target_modules))
        self.modules_to_save = list(dict.fromkeys(self.modules_to_save))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe adapter metadata for a remote worker/checkpoint."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "dropout": self.dropout,
            "use_rslora": self.use_rslora,
            "modules_to_save": list(self.modules_to_save),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LoRAConfig:
        if not isinstance(values, dict):
            raise TypeError("LoRA config must be a mapping")
        allowed = {
            "rank",
            "alpha",
            "target_modules",
            "dropout",
            "use_rslora",
            "modules_to_save",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown LoRA config fields: {unknown}")
        return cls(**values)


@dataclass(frozen=True)
class LoRAMemoryEstimate:
    """Parameter counts needed to budget one adapter injection."""

    new_adapter_parameters: int
    adapter_parameters: int
    saved_module_parameters: int
    trainable_parameters: int


def _validate_names(name: str, values: Any, *, allow_empty: bool) -> None:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"LoRA {name} must be a list of strings")
    if not allow_empty and not values:
        raise ValueError(f"LoRA {name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"LoRA {name} must contain non-empty strings")


def _scale(alpha: float, rank: int, use_rslora: bool) -> float:
    return float(alpha / math.sqrt(rank) if use_rslora else alpha / rank)


class LoRALinear(nn.Module):
    """Drop-in replacement for :class:`torch.nn.Linear` with a low-rank delta."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear base must be nn.Linear")
        self.base = linear
        self.base.requires_grad_(False)
        in_f, out_f = linear.in_features, linear.out_features
        # Keep adapter parameters on the base device.  They remain fp32 for
        # stable updates even when the imported model is bf16/fp16.
        adapter_device = linear.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_f, device=adapter_device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, rank, device=adapter_device, dtype=torch.float32)
        )
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.use_rslora = bool(use_rslora)
        self.scale = _scale(alpha, rank, self.use_rslora)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        """Expose the frozen base weight for code that inspects Linear modules."""
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_x = self.dropout(x).to(self.lora_A.dtype)
        lora_out = adapter_x @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out.to(base_out.dtype) * self.scale

    def merge(self) -> nn.Linear:
        """Return a new Linear with the adapter delta merged into its weight."""
        delta = (self.lora_B @ self.lora_A) * self.scale
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
            if self.base.bias is not None:
                assert merged.bias is not None
                merged.bias.copy_(self.base.bias)
        return merged


class LoRAEmbedding(nn.Module):
    """Embedding equivalent of LoRA for recipes that target embeddings.

    ``lora_A`` is indexed by token and ``lora_B`` projects the rank dimension
    back to the embedding width.  For PEFT-compatible NVARC recipes callers
    normally use ``modules_to_save=["embed_tokens"]`` instead, which keeps the
    full embedding trainable and avoids changing the intended semantics.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("LoRAEmbedding base must be nn.Embedding")
        if dropout:
            # Token-wise dropout is ambiguous for an embedding lookup and is
            # not part of the endpoint recipe.  Rejecting it avoids silently
            # applying a different mask from the Linear implementation.
            raise ValueError("LoRAEmbedding does not support non-zero dropout")
        self.base = embedding
        self.base.requires_grad_(False)
        device = embedding.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(embedding.num_embeddings, rank, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, embedding.embedding_dim, device=device, dtype=torch.float32)
        )
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.use_rslora = bool(use_rslora)
        self.scale = _scale(alpha, rank, self.use_rslora)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base(input_ids)
        adapter = self.lora_A[input_ids] @ self.lora_B
        return base_out + adapter.to(base_out.dtype) * self.scale

    def merge(self) -> nn.Embedding:
        """Return a standalone embedding with the delta merged."""
        merged = nn.Embedding(
            self.base.num_embeddings,
            self.base.embedding_dim,
            padding_idx=self.base.padding_idx,
            max_norm=self.base.max_norm,
            norm_type=self.base.norm_type,
            scale_grad_by_freq=self.base.scale_grad_by_freq,
            sparse=self.base.sparse,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = (self.lora_A @ self.lora_B) * self.scale
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
        return merged


def _matches(path: str, name: str) -> bool:
    return path == name or path.endswith("." + name)


def _parent_and_attr(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent: nn.Module = root
    for piece in pieces[:-1]:
        child = getattr(parent, piece, None)
        if child is None:
            raise ValueError(f"module path disappeared during LoRA injection: {path!r}")
        parent = child
    return parent, pieces[-1]


def _layer_index(path: str) -> int | None:
    pieces = path.split(".")
    if len(pieces) < 2 or pieces[0] != "layers":
        return None
    try:
        return int(pieces[1])
    except ValueError:
        return None


def _all_named_modules(stage: nn.Module) -> list[tuple[str, nn.Module]]:
    # ``named_modules`` is a snapshot: replacements below must not alter the
    # collection while we validate it.
    return [(path, module) for path, module in stage.named_modules() if path]


def _check_existing_adapter(original: nn.Module, cfg: LoRAConfig, path: str) -> None:
    if not isinstance(original, (LoRALinear, LoRAEmbedding)):
        return
    if (
        original.rank != cfg.rank
        or not math.isclose(original.scale, cfg.scale, rel_tol=1e-6, abs_tol=1e-8)
        or not math.isclose(original.dropout_p, cfg.dropout, rel_tol=1e-6, abs_tol=1e-8)
        or bool(original.use_rslora) != cfg.use_rslora
    ):
        raise ValueError(
            f"existing LoRA target {path!r} has rank/scale/dropout/rsLoRA "
            "incompatible with the requested LoRAConfig"
        )


def _adapter_for(original: nn.Module, cfg: LoRAConfig) -> nn.Module:
    if isinstance(original, (LoRALinear, LoRAEmbedding)):
        return original
    if isinstance(original, nn.Linear):
        return LoRALinear(
            original,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            use_rslora=cfg.use_rslora,
        )
    if isinstance(original, nn.Embedding):
        return LoRAEmbedding(
            original,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            use_rslora=cfg.use_rslora,
        )
    raise TypeError(
        f"LoRA target must be nn.Linear/nn.Embedding or an existing adapter, "
        f"got {type(original).__name__}"
    )


def estimate_lora_parameters(stage: nn.Module, cfg: LoRAConfig) -> LoRAMemoryEstimate:
    """Estimate adapter allocation and trainable counts without mutation.

    This mirrors the target matching rules used by apply_lora, but only reads
    module shapes.  It is used by the remote CUDA memory gate before adapter
    tensors and Adam state are allocated.
    """
    layers = getattr(stage, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise TypeError("LoRA stage must expose a non-empty decoder layers ModuleList")
    named_modules = _all_named_modules(stage)
    targets: list[tuple[str, nn.Module]] = []
    missing: list[str] = []
    for target_name in cfg.target_modules:
        found = [
            (path, module)
            for path, module in named_modules
            if _matches(path, target_name)
        ]
        layer_found = [
            (path, module)
            for path, module in found
            if _layer_index(path) is not None
        ]
        if layer_found:
            found_indices = {_layer_index(path) for path, _ in layer_found}
            missing.extend(
                f"layers.{index}.{target_name}"
                for index in sorted(set(range(len(layers))) - found_indices)
            )
        elif not found and target_name not in {"embed_tokens", "lm_head"}:
            missing.append(target_name)
        for path, module in found:
            if (path, module) not in targets:
                targets.append((path, module))
    target_paths = {path for path, _ in targets}
    saved: list[tuple[str, nn.Module]] = []
    saved_paths: set[str] = set()
    for save_name in cfg.modules_to_save:
        for path, module in named_modules:
            if not _matches(path, save_name):
                continue
            if path in target_paths:
                raise ValueError(
                    f"LoRA target {path!r} cannot also be listed in modules_to_save"
                )
            if path not in saved_paths:
                saved.append((path, module))
                saved_paths.add(path)
    if missing:
        raise ValueError(
            "LoRA target modules missing: " + ", ".join(sorted(set(missing)))
        )
    if not targets and not saved:
        raise ValueError("LoRA injection found no target or modules_to_save")

    new_adapter_parameters = 0
    adapter_parameters = 0
    for path, module in targets:
        _check_existing_adapter(module, cfg, path)
        if isinstance(module, LoRALinear):
            count = module.lora_A.numel() + module.lora_B.numel()
        elif isinstance(module, LoRAEmbedding):
            count = module.lora_A.numel() + module.lora_B.numel()
        elif isinstance(module, nn.Linear):
            count = cfg.rank * (module.in_features + module.out_features)
            new_adapter_parameters += count
        elif isinstance(module, nn.Embedding):
            count = (
                module.num_embeddings * cfg.rank
                + cfg.rank * module.embedding_dim
            )
            new_adapter_parameters += count
        else:
            raise TypeError(
                f"LoRA target {path!r} must be nn.Linear or nn.Embedding, "
                f"got {type(module).__name__}"
            )
        adapter_parameters += count
    saved_module_parameters = 0
    for path, module in saved:
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            raise ValueError(
                f"modules_to_save {path!r} refers to an existing LoRA wrapper"
            )
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise TypeError(
                f"modules_to_save {path!r} must refer to nn.Linear or nn.Embedding, "
                f"got {type(module).__name__}"
            )
        saved_module_parameters += sum(
            parameter.numel() for parameter in module.parameters()
        )
    return LoRAMemoryEstimate(
        new_adapter_parameters=new_adapter_parameters,
        adapter_parameters=adapter_parameters,
        saved_module_parameters=saved_module_parameters,
        trainable_parameters=adapter_parameters + saved_module_parameters,
    )


def apply_lora(stage: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """Freeze a stage and inject/activate the configured adapters in-place.

    Decoder-layer targets are required in every local layer.  Endpoint targets
    are optional because a stage may not own the embedding or lm_head.  All
    validation happens before freezing or replacing modules, so a typo cannot
    leave a half-adapted stage behind.
    """
    layers = getattr(stage, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise TypeError("LoRA stage must expose a non-empty decoder `layers` ModuleList")
    existing_config = getattr(stage, "_meshgpu_lora_config", None)
    if existing_config is not None:
        try:
            existing = LoRAConfig.from_dict(dict(existing_config))
        except (TypeError, ValueError) as exc:
            raise ValueError("stage contains invalid existing LoRA metadata") from exc
        existing_values = existing.to_dict()
        requested_values = cfg.to_dict()
        if existing_values != requested_values:
            differing = [
                name
                for name in existing_values
                if existing_values[name] != requested_values[name]
            ]
            detail = ", ".join(differing) or "recipe fields"
            raise ValueError(
                "stage is already configured with a different LoRA recipe; "
                f"differing fields: {detail}; reset or rebuild the stage before "
                "reconfiguring"
            )

    named_modules = _all_named_modules(stage)
    target_paths: list[tuple[str, nn.Module]] = []
    missing: list[str] = []

    for target_name in cfg.target_modules:
        found = [
            (path, module)
            for path, module in named_modules
            if _matches(path, target_name)
        ]
        layer_found = [
            (path, module)
            for path, module in found
            if _layer_index(path) is not None
        ]
        if layer_found:
            found_indices = {_layer_index(path) for path, _ in layer_found}
            expected_indices = set(range(len(layers)))
            missing_indices = sorted(expected_indices - found_indices)
            if missing_indices:
                missing.extend(
                    f"layers.{index}.{target_name}" for index in missing_indices
                )
        elif not found and target_name not in {"embed_tokens", "lm_head"}:
            missing.append(target_name)
        for path, module in found:
            if (path, module) not in target_paths:
                target_paths.append((path, module))

    target_path_set = {path for path, _ in target_paths}
    save_paths: list[tuple[str, nn.Module]] = []
    save_path_set: set[str] = set()
    for save_name in cfg.modules_to_save:
        found = [
            (path, module)
            for path, module in named_modules
            if _matches(path, save_name)
        ]
        for path, module in found:
            if path in target_path_set:
                raise ValueError(
                    f"LoRA target {path!r} cannot also be listed in modules_to_save"
                )
            if path not in save_path_set:
                save_paths.append((path, module))
                save_path_set.add(path)
        # Missing modules_to_save are intentionally allowed: endpoint names
        # are absent from the other stages of a sharded model.

    if missing:
        raise ValueError(
            "LoRA target modules missing: " + ", ".join(sorted(set(missing)))
        )
    if not target_paths and not save_paths:
        raise ValueError("LoRA injection found no target or modules_to_save")

    # Validate every existing target before mutating anything.
    for path, original in target_paths:
        _check_existing_adapter(original, cfg, path)
        if not isinstance(
            original,
            (nn.Linear, nn.Embedding, LoRALinear, LoRAEmbedding),
        ):
            raise TypeError(
                f"LoRA target {path!r} must be nn.Linear or nn.Embedding, "
                f"got {type(original).__name__}"
            )
    for path, module in save_paths:
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            raise ValueError(
                f"modules_to_save {path!r} refers to an existing LoRA wrapper; "
                "use its base module or configure it as a target instead"
            )
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise TypeError(
                f"modules_to_save {path!r} must refer to nn.Linear or nn.Embedding, "
                f"got {type(module).__name__}"
            )

    # Freeze the complete stage first.  The selected full modules and fresh /
    # existing adapter factors are re-enabled below.
    for parameter in stage.parameters():
        parameter.requires_grad_(False)

    n_adapters = 0
    for path, original in target_paths:
        parent, attr = _parent_and_attr(stage, path)
        adapter = _adapter_for(original, cfg)
        setattr(parent, attr, adapter)
        if isinstance(adapter, (LoRALinear, LoRAEmbedding)):
            adapter.lora_A.requires_grad_(True)
            adapter.lora_B.requires_grad_(True)
        n_adapters += 1

    for path, _original in save_paths:
        parent, attr = _parent_and_attr(stage, path)
        module = getattr(parent, attr)
        module.requires_grad_(True)

    # Metadata is a plain attribute rather than a registered buffer, so it is
    # not serialized as a tensor and remains safe for old checkpoints.
    stage._meshgpu_lora_config = cfg.to_dict()  # type: ignore[attr-defined]
    stage._meshgpu_modules_to_save = tuple(  # type: ignore[attr-defined]
        path for path, _module in save_paths
    )

    trainable = sum(
        parameter.numel()
        for parameter in stage.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in stage.parameters())
    log.info(
        "LoRA applied: %d adapters, full modules=%d, trainable params=%d / %d (%.2f%%)",
        n_adapters,
        len(save_paths),
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return stage


def trainable_parameters(stage: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in stage.parameters() if parameter.requires_grad]


def _saved_module_paths(stage: nn.Module) -> tuple[str, ...]:
    values = getattr(stage, "_meshgpu_modules_to_save", ())
    return tuple(str(value) for value in values)


def lora_state_dict(stage: nn.Module) -> dict[str, torch.Tensor]:
    """Return adapter factors plus explicitly saved full-module parameters."""
    state = stage.state_dict()
    selected: dict[str, torch.Tensor] = {
        key: value
        for key, value in state.items()
        if "lora_A" in key or "lora_B" in key
    }
    save_paths = _saved_module_paths(stage)
    if save_paths:
        for key, value in state.items():
            if any(key == path or key.startswith(path + ".") for path in save_paths):
                selected[key] = value
    return selected


def load_lora_state_dict(stage: nn.Module, adapter_sd: dict[str, torch.Tensor]) -> None:
    """Load adapter/full-saved tensors without changing frozen base weights."""
    if not isinstance(adapter_sd, dict):
        raise TypeError("adapter state must be a mapping")
    expected = lora_state_dict(stage)
    missing = sorted(set(expected) - set(adapter_sd))
    if missing:
        raise ValueError(f"adapter state is missing required keys: {missing}")
    # Only the explicitly trainable subset is accepted.  Checking against the
    # complete model state would allow a caller to smuggle a frozen
    # base.weight (or an unrelated buffer) into load_state_dict and silently
    # mutate the base model during task reset/resume.
    unknown = sorted(set(adapter_sd) - set(expected))
    if unknown:
        raise ValueError(f"adapter state contains unknown keys: {unknown}")
    for key, value in adapter_sd.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"adapter state value for {key!r} must be a tensor")
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(
                f"adapter state shape mismatch for {key!r}: "
                f"expected {tuple(expected[key].shape)}, got {tuple(value.shape)}"
            )
    # ``strict=False`` is intentional: all frozen base weights are absent from
    # an adapter snapshot.  The explicit checks above make the trainable part
    # strict without producing misleading warnings for expected base misses.
    stage.load_state_dict(adapter_sd, strict=False)


def lora_config(stage: nn.Module) -> LoRAConfig | None:
    """Return the injection config recorded by :func:`apply_lora`, if any."""
    values = getattr(stage, "_meshgpu_lora_config", None)
    if values is None:
        return None
    return LoRAConfig.from_dict(dict(values))


def iter_lora_modules(stage: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield all low-rank wrappers; useful for export and diagnostics."""
    return (
        (path, module)
        for path, module in stage.named_modules()
        if isinstance(module, (LoRALinear, LoRAEmbedding))
    )
