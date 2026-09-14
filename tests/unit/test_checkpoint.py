"""Tests for checkpoint save/commit/restore."""
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest
import torch

from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage

TINY_CFG = LlamaConfig(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, max_position_embeddings=32,
)


def _save_shared_checkpoint(root, step):
    return CheckpointCoordinator(root).save(
        job_id="shared", global_step=step,
        model_states=[{"weight": torch.ones(1)}], optimizer_states=[{}],
    )


@pytest.mark.parametrize("executor_type", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_concurrent_commits_preserve_all_history(tmp_path, executor_type):
    reader = CheckpointCoordinator(tmp_path)
    pool = (
        ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"))
        if executor_type is ProcessPoolExecutor else ThreadPoolExecutor(max_workers=4)
    )
    with pool as executor:
        ids = list(executor.map(_save_shared_checkpoint, [tmp_path] * 8, range(8)))
    restarted = CheckpointCoordinator(tmp_path)
    for step, checkpoint_id in enumerate(ids):
        assert restarted.load_manifest(checkpoint_id)["global_step"] == step
        assert reader.load_stage(checkpoint_id, 0)["global_step"] == step


def test_stale_coordinator_preserves_earlier_commit(tmp_path):
    first = CheckpointCoordinator(tmp_path)
    second = CheckpointCoordinator(tmp_path)
    ids = [coordinator.save(
        job_id="shared", global_step=step,
        model_states=[{"weight": torch.ones(1)}], optimizer_states=[{}],
    ) for step, coordinator in enumerate((first, second))]
    restarted = CheckpointCoordinator(tmp_path)
    assert [restarted.load_manifest(cid)["global_step"] for cid in ids] == [0, 1]
    assert first.committed_id == ids[-1]
    assert first.load_latest()["global_step"] == 1


def _make_stage(i: int) -> LlamaStage:
    return LlamaStage(
        TINY_CFG, i, i + 1,
        has_embedding=(i == 0),
        has_lm_head=(i == TINY_CFG.num_hidden_layers - 1),
    )


def test_save_and_load(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)

    ckpt_id = coord.save(
        job_id="j1",
        global_step=10,
        model_states=[stage.state_dict()],
        optimizer_states=[opt.state_dict()],
    )

    assert coord.committed_id == ckpt_id
    payload = coord.load_stage(ckpt_id, stage=0)
    assert payload["global_step"] == 10
    assert "model" in payload
    assert "optimizer" in payload


def test_verify_passes(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)
    ckpt_id = coord.save(
        job_id="j1", global_step=5,
        model_states=[stage.state_dict()],
        optimizer_states=[opt.state_dict()],
    )
    bad = coord.verify(ckpt_id)
    assert bad == []


def test_verify_detects_corruption(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)
    ckpt_id = coord.save(
        job_id="j1", global_step=5,
        model_states=[stage.state_dict()],
        optimizer_states=[opt.state_dict()],
    )
    # Corrupt the shard
    shard = tmp_path / "ckpts" / ckpt_id / "stage_0.pt"
    shard.write_bytes(b"corrupted")
    bad = coord.verify(ckpt_id)
    assert len(bad) > 0


def test_committed_pointer_persists(tmp_path):
    root = tmp_path / "ckpts"
    coord1 = CheckpointCoordinator(root)
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)
    ckpt_id = coord1.save(
        job_id="j1", global_step=1,
        model_states=[stage.state_dict()],
        optimizer_states=[opt.state_dict()],
    )

    # New coordinator reads same root — should find committed pointer
    coord2 = CheckpointCoordinator(root)
    assert coord2.committed_id == ckpt_id


def test_checkpoint_pointer_cannot_escape_root(tmp_path):
    root = tmp_path / "ckpts"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"checkpoint_id": "attacker"}')
    (root / "committed.json").write_text(
        '{"path": "' + str(outside) + '", "checkpoint_id": "attacker"}'
    )

    coord = CheckpointCoordinator(root)
    assert coord.committed_id is None
    assert coord.load_latest() is None


def test_checkpoint_manifest_without_pointer_is_not_loadable(tmp_path, monkeypatch):
    """A fully written but unpublished snapshot must not be resumable."""
    root = tmp_path / "ckpts"
    coord = CheckpointCoordinator(root)
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)

    real_replace = os.replace

    def fail_pointer(source, destination):
        if Path(destination).name == "committed.json":
            raise OSError("simulated pointer publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("meshgpu.checkpoints.coordinator.os.replace", fail_pointer)
    with pytest.raises(OSError, match="pointer publication"):
        coord.save(
            job_id="j1",
            global_step=1,
            model_states=[stage.state_dict()],
            optimizer_states=[opt.state_dict()],
        )

    checkpoint_id = next(path.name for path in root.iterdir() if path.is_dir())
    assert coord.committed_id is None
    with pytest.raises(ValueError, match="not committed"):
        coord.load_manifest(checkpoint_id)
    with pytest.raises(ValueError, match="not committed"):
        coord.load_stage(checkpoint_id, stage=0)


def test_resume_weights(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    stage = _make_stage(0)
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)

    # Save
    ckpt_id = coord.save(
        job_id="j1", global_step=20,
        model_states=[stage.state_dict()],
        optimizer_states=[opt.state_dict()],
    )

    # Modify model
    with torch.no_grad():
        for p in stage.parameters():
            p.fill_(0.0)

    # Restore
    payload = coord.load_stage(ckpt_id, stage=0)
    stage.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])

    # Weights should be back to original (non-zero)
    total = sum(p.abs().sum().item() for p in stage.parameters())
    assert total > 0


def test_save_rejects_ambiguous_or_incomplete_metadata(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    with pytest.raises(ValueError, match="scheduler_state or scheduler_states"):
        coord.save(
            job_id="j1",
            global_step=0,
            model_states=[{"weight": torch.ones(1)}],
            optimizer_states=[{}],
            scheduler_state={},
            scheduler_states=[{}],
        )
    with pytest.raises(ValueError, match="rng_states"):
        coord.save(
            job_id="j1",
            global_step=0,
            model_states=[{"weight": torch.ones(1)}, {"weight": torch.ones(1)}],
            optimizer_states=[{}, {}],
            rng_states=[{}],
        )


def test_load_stage_rejects_valid_hash_for_invalid_payload(tmp_path):
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    checkpoint_id = coord.save(
        job_id="j1",
        global_step=0,
        model_states=[{"weight": torch.ones(1)}],
        optimizer_states=[{}],
    )
    shard = tmp_path / "ckpts" / checkpoint_id / "stage_0.pt"
    from meshgpu.checkpoints import coordinator as checkpoint_module

    checkpoint_module._atomic_torch_save({"not_model": {}}, shard)
    manifest_path = shard.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["byte_length"] = shard.stat().st_size
    manifest["shards"][0]["sha256"] = checkpoint_module._sha256_file(shard)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no model state"):
        coord.load_stage(checkpoint_id, stage=0)
