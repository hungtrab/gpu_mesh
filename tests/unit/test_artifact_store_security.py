"""Path-safety regressions for the local artifact store."""

import pytest

from meshgpu.artifacts.store import ArtifactStore


@pytest.mark.parametrize("value", ["", ".", "..", "../outside", "/tmp/outside"])
def test_manifest_id_cannot_escape_store(tmp_path, value):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.manifest_path(value)


def test_shard_id_cannot_escape_store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.shard_path("model-1", "../outside.pt")
