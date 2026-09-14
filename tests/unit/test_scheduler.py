"""Controller admission and registration concurrency tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from meshgpu.controller.scheduler import Scheduler
from meshgpu.controller.store import Store
from meshgpu.protocol.messages import (
    Backend,
    CapabilityReport,
    GpuInfo,
    JobSpec,
    JobState,
    PlacementEntry,
    TaskKind,
    WorkerMode,
)


def _capability(worker_id: str = "worker-1") -> CapabilityReport:
    return CapabilityReport(
        worker_id=worker_id,
        provider="self_managed",
        os="linux",
        arch="x86_64",
        python_version="3.12",
        torch_version="2.10",
        cuda_driver_version=None,
        gpus=[
            GpuInfo(
                device_index=0,
                name="test",
                total_vram_bytes=8 * 1024**3,
                free_vram_bytes=7 * 1024**3,
                compute_capability=(8, 0),
                supported_dtypes=["float32"],
            )
        ],
        host_ram_bytes=16 * 1024**3,
        pinned_memory_limit_bytes=4 * 1024**3,
        scratch_disk_bytes=32 * 1024**3,
        supported_modes=[WorkerMode.DISTRIBUTED_WORKER],
        available_backends=[Backend.PORTABLE_PIPELINE],
        available_model_adapters=["llama_dense_v1"],
    )


def _job(job_id: str = "job-1") -> JobSpec:
    return JobSpec(
        job_id=job_id,
        job_name="test",
        task=TaskKind.INFERENCE,
        backend=Backend.PORTABLE_PIPELINE,
        model_manifest="manifest",
        model_adapter="llama_dense_v1",
        compute_dtype="float32",
        placement=[],
    )


@pytest.fixture
def scheduler(tmp_path):
    store = Store(tmp_path / "nested" / "controller.db")
    value = Scheduler(store, join_token="join-secret")
    yield value, store
    store.close()


def test_registration_requires_join_token(scheduler):
    value, _ = scheduler
    with pytest.raises(PermissionError, match="non-empty"):
        value.register_worker("", _capability())
    with pytest.raises(PermissionError):
        value.register_worker("wrong", _capability())

    result = value.register_worker("join-secret", _capability())
    assert result["worker_id"] == "worker-1"


def test_scheduler_rejects_empty_configured_token(scheduler):
    _, store = scheduler
    with pytest.raises(ValueError, match="non-empty"):
        Scheduler(store, join_token="")


def test_admission_rejects_expired_worker(scheduler):
    value, store = scheduler
    value.register_worker("join-secret", _capability())
    with store._lock:
        store._conn.execute(
            "UPDATE workers SET expires_at=? WHERE worker_id=?",
            (0.0, "worker-1"),
        )
        store._conn.commit()

    spec = _job()
    spec.placement = [
        PlacementEntry(
            worker_id="worker-1",
            device_index=0,
            stage_id=0,
            layer_start=0,
            layer_end=1,
        )
    ]
    store.create_job(spec.model_dump(mode="json"))

    assert value.admit_job(spec.job_id) is None
    assert store.get_job(spec.job_id)["state"] == JobState.FAILED


def test_concurrent_admission_claims_job_once(scheduler):
    value, store = scheduler
    store.create_job(_job().model_dump(mode="json"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: value.admit_job("job-1"), range(8)))

    grants = [result for result in results if result is not None]
    assert len(grants) == 1
    assert store.get_job("job-1")["state"] == JobState.RESERVED
