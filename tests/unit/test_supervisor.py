"""Tests for supervisor-owned GPU subprocess lifecycle."""

import pytest

from meshgpu.agent.supervisor import Supervisor


def _supervisor() -> Supervisor:
    return Supervisor(
        "http://127.0.0.1:1",
        "test-token",
        worker_id="test-worker",
        provider="self_managed",
    )


def test_launch_gpu_worker_starts_ready_process():
    supervisor = _supervisor()
    try:
        pid = supervisor._launch_gpu_worker(stage_id=0, device_index=0)
        assert pid > 0
        assert supervisor._gpu_processes[0].is_alive()
        assert supervisor._launch_gpu_worker(stage_id=0, device_index=0) == pid
    finally:
        supervisor._kill_gpu_workers()

    assert supervisor._state.gpu_pids == []
    assert supervisor._gpu_processes == {}


@pytest.mark.parametrize("stage_id,device_index", [(-1, 0), (0, -1)])
def test_launch_gpu_worker_rejects_negative_indices(stage_id, device_index):
    with pytest.raises(ValueError):
        _supervisor()._launch_gpu_worker(stage_id, device_index)
