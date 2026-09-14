"""
Request admission control.
Reserve KV budget before accepting a request; reject when at capacity.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryPreflightResult:
    """Caller-visible memory decision made before request admission."""

    feasible: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.details, dict):
            raise TypeError("details must be a mapping")


@dataclass
class KVBudget:
    total_slots: int       # max cached tokens across all live sessions
    reserved: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_slots, bool)
            or not isinstance(self.total_slots, int)
            or isinstance(self.reserved, bool)
            or not isinstance(self.reserved, int)
        ):
            raise TypeError("KV slot counts must be integers")
        if self.total_slots < 0:
            raise ValueError("total_slots must be non-negative")
        if self.reserved < 0 or self.reserved > self.total_slots:
            raise ValueError("reserved must be within the total slot budget")

    @property
    def available(self) -> int:
        with self._lock:
            return self.total_slots - self.reserved

    def try_reserve(self, prompt_len: int, max_new: int) -> bool:
        if (
            isinstance(prompt_len, bool)
            or not isinstance(prompt_len, int)
            or isinstance(max_new, bool)
            or not isinstance(max_new, int)
        ):
            raise TypeError("token reservations must be integers")
        if prompt_len < 0 or max_new < 0:
            raise ValueError("token reservations must be non-negative")
        needed = prompt_len + max_new
        with self._lock:
            if needed > self.total_slots - self.reserved:
                return False
            self.reserved += needed
            return True

    def release(self, prompt_len: int, max_new: int) -> None:
        if (
            isinstance(prompt_len, bool)
            or not isinstance(prompt_len, int)
            or isinstance(max_new, bool)
            or not isinstance(max_new, int)
        ):
            raise TypeError("token reservations must be integers")
        if prompt_len < 0 or max_new < 0:
            raise ValueError("token reservations must be non-negative")
        with self._lock:
            self.reserved = max(0, self.reserved - (prompt_len + max_new))


@dataclass
class AdmissionConfig:
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 256
    max_concurrent_requests: int = 4
    request_deadline_s: float = 300.0
    kv_slots: int = 0  # 0 = derived from GPU budget; set by planner

    def __post_init__(self) -> None:
        for name, value in (
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_new_tokens", self.max_new_tokens),
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("kv_slots", self.kv_slots),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if not math.isfinite(self.request_deadline_s) or self.request_deadline_s <= 0:
            raise ValueError("request_deadline_s must be finite and positive")
        if self.kv_slots < 0:
            raise ValueError("kv_slots must be non-negative")

    def derive_kv_slots(
        self,
        usable_vram_bytes: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int,
    ) -> None:
        values = (
            usable_vram_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("VRAM and model dimensions must be integers")
        if usable_vram_bytes < 0:
            raise ValueError("usable_vram_bytes must be non-negative")
        if any(value < 1 for value in (num_layers, num_kv_heads, head_dim, dtype_bytes)):
            raise ValueError("model dimensions and dtype_bytes must be positive")
        # bytes per token per layer: 2 (k+v) * num_kv_heads * head_dim * dtype_bytes
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        if bytes_per_token <= 0:
            self.kv_slots = 0
            return
        # Reserve 40% of usable VRAM for KV; rest is weights + activations.
        kv_budget = int(usable_vram_bytes * 0.4)
        self.kv_slots = max(0, kv_budget // bytes_per_token)


class AdmissionController:
    def __init__(self, cfg: AdmissionConfig, kv_budget: KVBudget) -> None:
        self._cfg = cfg
        self._kv = kv_budget
        self._active: int = 0
        self._lock = threading.Lock()
        # Keep a small reservation ledger so a duplicated HTTP cleanup path
        # cannot release the same request twice.  Identical requests are
        # counted independently.
        self._reservations: dict[tuple[int, int], int] = {}

    def admit(self, prompt_len: int, max_new: int) -> tuple[bool, str]:
        """Try to admit a request. Returns (ok, reason)."""
        if (
            isinstance(prompt_len, bool)
            or not isinstance(prompt_len, int)
            or isinstance(max_new, bool)
            or not isinstance(max_new, int)
        ):
            return False, "token counts must be integers"
        if prompt_len < 1 or max_new < 1:
            return False, "token counts must be positive"
        if prompt_len > self._cfg.max_prompt_tokens:
            return False, f"prompt_len {prompt_len} > limit {self._cfg.max_prompt_tokens}"
        if max_new > self._cfg.max_new_tokens:
            return False, f"max_new_tokens {max_new} > limit {self._cfg.max_new_tokens}"

        with self._lock:
            if self._active >= self._cfg.max_concurrent_requests:
                return False, "max concurrent requests reached"
            if not self._kv.try_reserve(prompt_len, max_new):
                return False, f"KV cache full (available={self._kv.available} slots)"
            self._active += 1
            key = (prompt_len, max_new)
            self._reservations[key] = self._reservations.get(key, 0) + 1

        return True, "ok"

    def release(self, prompt_len: int, max_new: int) -> None:
        with self._lock:
            key = (prompt_len, max_new)
            count = self._reservations.get(key, 0)
            if count == 0:
                return
            if count == 1:
                del self._reservations[key]
            else:
                self._reservations[key] = count - 1
            self._active = max(0, self._active - 1)
            self._kv.release(prompt_len, max_new)

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active
