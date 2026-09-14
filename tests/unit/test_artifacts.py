"""Tests for artifact store and manifest."""
import hashlib

import pytest

from meshgpu.artifacts.manifest import ManifestMeta, ModelManifest, ShardEntry
from meshgpu.artifacts.store import ArtifactStore


def _make_manifest() -> ModelManifest:
    meta = ManifestMeta(
        model_family="llama_dense",
        model_id="test-rev-001",
        adapter="llama_dense_v1",
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        compute_dtype="float16",
        weight_format="safetensors",
    )
    shard = ShardEntry(
        shard_id="shard_0",
        path="shards/shard_0",
        byte_length=8,
        sha256="",  # filled by test
        layer_start=0,
        layer_end=4,
        tensor_names=["layer.0.weight"],
    )
    return ModelManifest(manifest_id="test-manifest-001", meta=meta, shards=[shard])


def test_manifest_save_load(tmp_path):
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    m.save(path)
    loaded = ModelManifest.load(path)
    assert loaded.manifest_id == m.manifest_id
    assert loaded.meta.model_family == "llama_dense"


def test_manifest_shards_for_layers():
    m = _make_manifest()
    assert len(m.shards_for_layers(0, 4)) == 1
    assert len(m.shards_for_layers(4, 8)) == 0


def test_store_ingest_and_verify(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    # Create a fake shard file
    shard_data = b"fake_weight_data"
    sha = hashlib.sha256(shard_data).hexdigest()
    src = tmp_path / "shard_0.bin"
    src.write_bytes(shard_data)

    m = _make_manifest()
    m.shards[0].sha256 = sha
    m.shards[0].byte_length = len(shard_data)
    store.save_manifest(m)

    store.ingest_shard("test-manifest-001", m.shards[0], src)
    assert store.verify_shard("test-manifest-001", m.shards[0])


def test_store_detects_hash_mismatch(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    shard_data = b"correct_data"
    src = tmp_path / "shard.bin"
    src.write_bytes(shard_data)

    m = _make_manifest()
    m.shards[0].sha256 = "deadbeef" * 8  # wrong hash
    m.shards[0].byte_length = len(shard_data)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        store.ingest_shard("test-manifest-001", m.shards[0], src)


def test_store_ingest_does_not_publish_partial_copy(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    shard_data = b"complete shard"
    src = tmp_path / "shard.bin"
    src.write_bytes(shard_data)

    m = _make_manifest()
    m.shards[0].sha256 = hashlib.sha256(shard_data).hexdigest()
    m.shards[0].byte_length = len(shard_data)
    destination = store.shard_path(m.manifest_id, m.shards[0].shard_id)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"previous committed shard")

    def interrupted_copy(_source, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr("meshgpu.artifacts.store.shutil.copy2", interrupted_copy)
    with pytest.raises(OSError, match="simulated interruption"):
        store.ingest_shard(m.manifest_id, m.shards[0], src)

    assert destination.read_bytes() == b"previous committed shard"
    assert not list(destination.parent.glob(".*.tmp"))
