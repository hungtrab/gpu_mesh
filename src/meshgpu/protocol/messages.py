"""Control-plane JSON messages (not tensor payload)."""
from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobState(str, enum.Enum):
    PENDING = "pending"
    PREFLIGHT = "preflight"
    RESERVED = "reserved"
    LOADING = "loading"
    WARMUP = "warmup"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    DRAINING = "draining"
    RECOVERING = "recovering"
    STOPPED = "stopped"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskKind(str, enum.Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class Backend(str, enum.Enum):
    NATIVE = "native"
    PORTABLE_PIPELINE = "portable_pipeline"


class WorkerMode(str, enum.Enum):
    CLIENT = "client"
    SINGLE_RUNTIME_JOB = "single_runtime_job"
    DISTRIBUTED_WORKER = "distributed_worker"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Msg(BaseModel):
    msg_id: str = Field(default_factory=_uid)
    ts: float = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Registration / capability
# ---------------------------------------------------------------------------

class GpuInfo(BaseModel):
    device_index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: tuple[int, int]
    supported_dtypes: list[str]


class CapabilityReport(BaseModel):
    worker_id: str
    worker_incarnation: str = Field(default_factory=_uid)
    provider: str
    os: str
    arch: str
    python_version: str
    torch_version: str
    cuda_driver_version: str | None
    gpus: list[GpuInfo]
    host_ram_bytes: int
    pinned_memory_limit_bytes: int
    scratch_disk_bytes: int
    supported_modes: list[WorkerMode]
    available_backends: list[Backend]
    available_model_adapters: list[str]
    controller_rtt_ms: float | None = None
    storage_rtt_ms: float | None = None
    data_goodput_mbit_s: float | None = None
    session_remaining_s: float | None = None


class RegisterRequest(Msg):
    join_token: str
    capability: CapabilityReport


class RegisterResponse(Msg):
    worker_id: str
    credential: str
    expires_at: float


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat(Msg):
    worker_id: str
    worker_incarnation: str
    lease_epoch: int
    job_id: str | None
    gpu_free_vram_bytes: list[int]  # per device


class HeartbeatAck(Msg):
    lease_epoch: int
    deadline: float  # unix timestamp; worker must checkpoint or stop by then


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

class PlacementEntry(BaseModel):
    worker_id: str
    device_index: int
    stage_id: int
    layer_start: int
    layer_end: int  # exclusive


class JobSpec(BaseModel):
    job_id: str = Field(default_factory=_uid)
    job_name: str
    task: TaskKind
    backend: Backend
    model_manifest: str
    model_adapter: str
    compute_dtype: str
    placement: list[PlacementEntry]
    placement_version: int = 1
    max_vram_fraction: float = 0.85
    reserve_min_gib: float = 1.0
    allow_relay: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class LeaseGrant(Msg):
    job_id: str
    lease_epoch: int
    placement_version: int
    job_spec: JobSpec
    expires_at: float


class JobStateChange(Msg):
    job_id: str
    worker_id: str
    worker_incarnation: str
    lease_epoch: int
    attempt_id: str
    old_state: JobState
    new_state: JobState
    reason: str


class CancelRequest(Msg):
    job_id: str
    reason: str = "user_cancel"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

class PreflightRequest(Msg):
    job_id: str
    job_spec: JobSpec
    peer_workers: list[str]


class PreflightResult(BaseModel):
    feasible: bool
    reason: str
    peak_vram_bytes_per_device: list[int]
    usable_vram_bytes_per_device: list[int]
    estimated_goodput_mbit_s: float | None


class PreflightResponse(Msg):
    job_id: str
    worker_id: str
    result: PreflightResult


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorCode(str, enum.Enum):
    PROTOCOL_ERROR = "protocol_error"
    AUTH_FAILED = "auth_failed"
    LEASE_EXPIRED = "lease_expired"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    WORKER_LOST = "worker_lost"
    INVALID_STATE = "invalid_state"
    NOT_SUPPORTED = "not_supported"
    INTERNAL = "internal"


class ErrorMsg(Msg):
    code: ErrorCode
    message: str
    job_id: str | None = None
    attempt_id: str | None = None
