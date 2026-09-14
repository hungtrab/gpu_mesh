"""Controller HTTP security regression tests."""
import json

import pytest

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from meshgpu.controller.server import build_app
from meshgpu.protocol.messages import (
    Backend,
    CapabilityReport,
    GpuInfo,
    WorkerMode,
)

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


def test_controller_rejects_empty_join_token(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        build_app(tmp_path / "controller.db", join_token="")


def _capability() -> CapabilityReport:
    return CapabilityReport(
        worker_id="public-list-worker",
        provider="self_managed",
        os="linux",
        arch="x86_64",
        python_version="3.13",
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


@pytest.mark.asyncio
async def test_worker_listing_does_not_expose_bearer_credential(tmp_path):
    app = build_app(tmp_path / "controller.db", join_token="join-secret")
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
    try:
        registered = await client.post(
            "/v1/workers/register",
            json={
                "join_token": "join-secret",
                "capability": _capability().model_dump(mode="json"),
            },
        )
        assert registered.status_code == 200
        credential = registered.json()["credential"]

        listed = await client.get("/v1/workers")
        assert listed.status_code == 200
        body = listed.json()
        assert body[0]["worker_id"] == "public-list-worker"
        assert all("credential" not in worker for worker in body)
        assert credential not in json.dumps(body)
    finally:
        await client.aclose()
