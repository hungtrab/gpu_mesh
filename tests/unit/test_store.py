"""Unit tests for controller Store."""
import time

import pytest

from meshgpu.controller.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_upsert_and_get_worker(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() + 3600, {"gpu": []})
    w = store.get_worker("w1")
    assert w["worker_id"] == "w1"
    assert w["suspected"] == 0


def test_heartbeat_updates_timestamp(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() + 3600, {})
    ok = store.heartbeat("w1", "inc1")
    assert ok
    w = store.get_worker("w1")
    assert w["last_heartbeat"] > 0


def test_heartbeat_rejects_wrong_incarnation(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() + 3600, {})
    ok = store.heartbeat("w1", "wrong")
    assert not ok


def test_heartbeat_rejects_expired_credential(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() - 1, {})
    assert not store.heartbeat("w1", "inc1", "cred")


def test_heartbeat_rejects_rotated_credential(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() + 3600, {})
    assert not store.heartbeat("w1", "inc1", "old-cred")


def test_mark_suspected(store):
    store.upsert_worker("w1", "inc1", "cred", time.time() + 3600, {})
    store.mark_suspected("w1")
    w = store.get_worker("w1")
    assert w["suspected"] == 1


def test_create_and_transition_job(store):
    spec = {
        "job_id": "j1",
        "job_name": "test",
        "task": "inference",
        "backend": "portable_pipeline",
        "model_manifest": "m.json",
        "model_adapter": "llama_dense_v1",
        "compute_dtype": "float16",
        "placement": [],
        "placement_version": 1,
        "max_vram_fraction": 0.85,
        "reserve_min_gib": 1.0,
        "allow_relay": True,
        "extra": {},
    }
    job_id = store.create_job(spec)
    assert job_id == "j1"
    job = store.get_job("j1")
    assert job["state"] == "pending"

    store.transition_job("j1", "running", reason="test")
    assert store.get_job("j1")["state"] == "running"


def test_transition_expected_state_is_atomic(store):
    spec = {
        "job_id": "cas-job",
        "job_name": "test",
        "task": "inference",
        "backend": "portable_pipeline",
        "model_manifest": "m.json",
        "model_adapter": "llama_dense_v1",
        "compute_dtype": "float16",
        "placement": [],
        "placement_version": 1,
        "max_vram_fraction": 0.85,
        "reserve_min_gib": 1.0,
        "allow_relay": True,
        "extra": {},
    }
    store.create_job(spec)

    assert store.transition_job("cas-job", "preflight", expected_state="pending")
    assert not store.transition_job("cas-job", "preflight", expected_state="pending")
    assert store.get_job("cas-job")["state"] == "preflight"


def test_bump_lease_epoch(store):
    spec = {
        "job_id": "j2", "job_name": "t", "task": "training",
        "backend": "native", "model_manifest": "m", "model_adapter": "a",
        "compute_dtype": "float16", "placement": [], "placement_version": 1,
        "max_vram_fraction": 0.85, "reserve_min_gib": 1.0, "allow_relay": False, "extra": {},
    }
    store.create_job(spec)
    e1 = store.bump_lease_epoch("j2")
    e2 = store.bump_lease_epoch("j2")
    assert e2 == e1 + 1


def test_commit_checkpoint(store):
    spec = {
        "job_id": "j3", "job_name": "t", "task": "training",
        "backend": "native", "model_manifest": "m", "model_adapter": "a",
        "compute_dtype": "float16", "placement": [], "placement_version": 1,
        "max_vram_fraction": 0.85, "reserve_min_gib": 1.0, "allow_relay": False, "extra": {},
    }
    store.create_job(spec)
    store.commit_checkpoint("ck1", "j3", 100, {"shards": []})
    ck = store.get_latest_checkpoint("j3")
    assert ck["checkpoint_id"] == "ck1"
    assert ck["global_step"] == 100
