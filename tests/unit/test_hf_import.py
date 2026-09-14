"""Tests for HF → MeshGPU weight importer.

Does NOT require transformers installed — tests are self-contained.
import_from_hf itself is skipped when transformers is absent.
"""
from __future__ import annotations

import dataclasses
import json
import types
import uuid
from pathlib import Path

import pytest
import torch

from meshgpu.artifacts.hf_import import (
    _resolve_model_file,
    build_pipeline_from_manifest,
    build_stage_from_manifest,
    hf_config_to_llama,
    remap_keys,
    split_layers,
)
from meshgpu.artifacts.manifest import ManifestMeta, ModelManifest, ShardEntry
from meshgpu.models.llama_dense import LlamaConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
    )


def _fake_hf_cfg(**overrides):
    """Minimal mock of a HF LlamaConfig."""
    base = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _make_full_sd(n_layers: int = 4, hidden_size: int = 16, intermediate_size: int = 32,
                  num_heads: int = 2, num_kv_heads: int = 2, head_dim: int = 8,
                  vocab: int = 64) -> dict:
    """Build a fake HF-style state dict matching LlamaStage weight shapes."""
    sd = {}
    hidden_dim = hidden_size
    intermediate_dim = intermediate_size
    q_size = num_heads * head_dim      # q_proj out features
    kv_size = num_kv_heads * head_dim  # k/v_proj out features
    sd["model.embed_tokens.weight"] = torch.randn(vocab, hidden_dim)
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[f"{p}input_layernorm.weight"] = torch.randn(hidden_dim)
        sd[f"{p}self_attn.q_proj.weight"] = torch.randn(q_size, hidden_dim)
        sd[f"{p}self_attn.k_proj.weight"] = torch.randn(kv_size, hidden_dim)
        sd[f"{p}self_attn.v_proj.weight"] = torch.randn(kv_size, hidden_dim)
        sd[f"{p}self_attn.o_proj.weight"] = torch.randn(hidden_dim, q_size)
        sd[f"{p}mlp.gate_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
        sd[f"{p}mlp.up_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
        sd[f"{p}mlp.down_proj.weight"] = torch.randn(hidden_dim, intermediate_dim)
        sd[f"{p}post_attention_layernorm.weight"] = torch.randn(hidden_dim)
    sd["model.norm.weight"] = torch.randn(hidden_dim)
    sd["lm_head.weight"] = torch.randn(vocab, hidden_dim)
    return sd


def test_resolve_model_file_accepts_huggingface_cache_blob_symlink(tmp_path, monkeypatch):
    import huggingface_hub.constants as hf_constants

    cache = tmp_path / "hf-cache"
    snapshot = cache / "models--test--model" / "snapshots" / "revision"
    blob = cache / "blobs" / "sha256-value"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))

    assert _resolve_model_file(snapshot, "model.safetensors") == blob.resolve()


def test_resolve_model_file_rejects_arbitrary_external_symlink(tmp_path):
    root = tmp_path / "checkpoint"
    outside = tmp_path / "outside.safetensors"
    root.mkdir()
    outside.write_bytes(b"not trusted")
    (root / "model.safetensors").symlink_to(outside)

    with pytest.raises(ValueError, match="trusted Hugging Face cache"):
        _resolve_model_file(root, "model.safetensors")


def _save_shards(cfg: LlamaConfig, n_stages: int, tmp_path: Path) -> Path:
    """Save a tiny model as MeshGPU shards and return the directory."""
    from meshgpu.artifacts.hf_import import _sha256_file, split_layers

    full_sd = _make_full_sd(
        n_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        vocab=cfg.vocab_size,
    )

    (tmp_path / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    ranges = split_layers(cfg.num_hidden_layers, n_stages)
    shards = []
    for stage_idx, (ls, le) in enumerate(ranges):
        is_first = stage_idx == 0
        is_last = stage_idx == n_stages - 1
        sd = remap_keys(full_sd, ls, le, is_first=is_first, is_last=is_last)
        shard_id = f"stage_{stage_idx:02d}"
        shard_file = tmp_path / f"{shard_id}.pt"
        torch.save(sd, shard_file)
        shards.append(ShardEntry(
            shard_id=shard_id,
            path=f"{shard_id}.pt",
            byte_length=shard_file.stat().st_size,
            sha256=_sha256_file(shard_file),
            layer_start=ls,
            layer_end=le,
            tensor_names=sorted(sd.keys()),
        ))

    meta = ManifestMeta(
        model_family="llama_dense",
        model_id="test/tiny",
        adapter="llama_dense_v1",
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        compute_dtype="float32",
        weight_format="pytorch",
        tied_embeddings=False,
        extra={
            "num_stages": n_stages,
            "intermediate_size": cfg.intermediate_size,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": cfg.rope_theta,
        },
    )
    manifest = ModelManifest(manifest_id=uuid.uuid4().hex[:8], meta=meta, shards=shards)
    manifest.save(tmp_path / "manifest.json")
    return tmp_path


# ---------------------------------------------------------------------------
# split_layers
# ---------------------------------------------------------------------------

class TestSplitLayers:
    def test_even(self):
        ranges = split_layers(8, 4)
        assert ranges == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_remainder_goes_to_front(self):
        ranges = split_layers(5, 2)
        # 5 = 3 + 2; stage 0 gets the extra
        assert ranges[0] == (0, 3)
        assert ranges[1] == (3, 5)

    def test_single_stage(self):
        assert split_layers(4, 1) == [(0, 4)]

    def test_n_equals_stages(self):
        ranges = split_layers(3, 3)
        assert ranges == [(0, 1), (1, 2), (2, 3)]

    def test_contiguous(self):
        ranges = split_layers(7, 3)
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]
        assert ranges[-1][1] == 7

    def test_too_many_stages_raises(self):
        with pytest.raises(ValueError):
            split_layers(2, 5)


# ---------------------------------------------------------------------------
# hf_config_to_llama
# ---------------------------------------------------------------------------

class TestHfConfigToLlama:
    def test_basic_fields(self):
        hf = _fake_hf_cfg()
        cfg = hf_config_to_llama(hf)
        assert cfg.vocab_size == 64
        assert cfg.hidden_size == 16
        assert cfg.intermediate_size == 32
        assert cfg.num_hidden_layers == 4
        assert cfg.num_attention_heads == 2
        assert cfg.num_key_value_heads == 2
        assert cfg.max_position_embeddings == 64

    def test_head_dim_inferred(self):
        hf = _fake_hf_cfg(hidden_size=32, num_attention_heads=4)
        cfg = hf_config_to_llama(hf)
        assert cfg.head_dim == 8  # 32 // 4

    def test_head_dim_explicit(self):
        hf = _fake_hf_cfg(hidden_size=32, num_attention_heads=4, head_dim=16)
        cfg = hf_config_to_llama(hf)
        assert cfg.head_dim == 16

    def test_defaults_applied(self):
        # rms_norm_eps and rope_theta should have defaults
        hf = _fake_hf_cfg()
        cfg = hf_config_to_llama(hf)
        assert cfg.rms_norm_eps == pytest.approx(1e-5)
        assert cfg.rope_theta == pytest.approx(10000.0)

    def test_mha_fallback(self):
        # No num_key_value_heads → falls back to num_attention_heads (MHA)
        hf = _fake_hf_cfg()
        del hf.num_key_value_heads
        cfg = hf_config_to_llama(hf)
        assert cfg.num_key_value_heads == hf.num_attention_heads


# ---------------------------------------------------------------------------
# remap_keys
# ---------------------------------------------------------------------------

class TestRemapKeys:
    def _sd(self, n=4):
        return _make_full_sd(n_layers=n)

    def test_layer_keys_remapped(self):
        sd = self._sd()
        out = remap_keys(sd, 0, 2, is_first=True, is_last=False)
        assert "layers.0.self_attn.q_proj.weight" in out
        assert "layers.1.self_attn.q_proj.weight" in out
        # global layer 2 should NOT be present
        assert "layers.2.self_attn.q_proj.weight" not in out

    def test_local_index_offset(self):
        sd = self._sd()
        out = remap_keys(sd, 2, 4, is_first=False, is_last=True)
        # global 2 → local 0, global 3 → local 1
        assert "layers.0.self_attn.q_proj.weight" in out
        assert "layers.1.self_attn.q_proj.weight" in out
        assert "layers.2.self_attn.q_proj.weight" not in out

    def test_embed_only_in_first(self):
        sd = self._sd()
        first = remap_keys(sd, 0, 2, is_first=True, is_last=False)
        not_first = remap_keys(sd, 2, 4, is_first=False, is_last=True)
        assert "embed_tokens.weight" in first
        assert "embed_tokens.weight" not in not_first

    def test_head_only_in_last(self):
        sd = self._sd()
        last = remap_keys(sd, 2, 4, is_first=False, is_last=True)
        not_last = remap_keys(sd, 0, 2, is_first=True, is_last=False)
        assert "norm.weight" in last
        assert "lm_head.weight" in last
        assert "norm.weight" not in not_last
        assert "lm_head.weight" not in not_last

    def test_tied_embeddings(self):
        sd = _make_full_sd(num_heads=2, num_kv_heads=2, head_dim=8)
        del sd["lm_head.weight"]
        out = remap_keys(sd, 2, 4, is_first=False, is_last=True, tie_word_embeddings=True)
        assert "lm_head.weight" in out
        assert torch.equal(out["lm_head.weight"], sd["model.embed_tokens.weight"])

    def test_no_extra_global_keys(self):
        sd = self._sd()
        out = remap_keys(sd, 0, 2, is_first=True, is_last=False)
        for k in out:
            assert not k.startswith("model."), f"HF prefix leaked: {k}"

    def test_all_sublayers_present(self):
        sd = self._sd()
        out = remap_keys(sd, 0, 1, is_first=True, is_last=False)
        for sub in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                    "post_attention_layernorm"):
            assert f"layers.0.{sub}.weight" in out, f"missing layers.0.{sub}.weight"


# ---------------------------------------------------------------------------
# build_pipeline_from_manifest
# ---------------------------------------------------------------------------

class TestBuildPipelineFromManifest:
    def test_returns_correct_number_of_workers(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        workers = build_pipeline_from_manifest(tmp_path)
        assert len(workers) == 2

    def test_stage_order(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        workers = build_pipeline_from_manifest(tmp_path)
        assert workers[0]._ctx.is_first
        assert workers[1]._ctx.is_last
        assert not workers[0]._ctx.is_last
        assert not workers[1]._ctx.is_first

    def test_layer_ranges(self, tmp_path):
        cfg = _tiny_cfg()  # 4 layers
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        workers = build_pipeline_from_manifest(tmp_path)
        assert workers[0]._ctx.layer_start == 0
        assert workers[0]._ctx.layer_end == 2
        assert workers[1]._ctx.layer_start == 2
        assert workers[1]._ctx.layer_end == 4

    def test_inference_forward_pass(self, tmp_path):
        from meshgpu.backends.portable.pipeline import pipeline_prefill
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        workers = build_pipeline_from_manifest(tmp_path)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits = pipeline_prefill(workers, ids)
        assert logits.shape == (1, 8, cfg.vocab_size)

    def test_weights_loaded_not_random(self, tmp_path):
        """Weights from saved shards must differ from freshly initialized weights."""
        from meshgpu.backends.portable.pipeline import build_pipeline
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)

        loaded = build_pipeline_from_manifest(tmp_path)
        fresh = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)

        # At least one parameter should differ (weights come from our random shard, not
        # from the fresh random init — both are random but drawn independently)
        loaded_sd = loaded[0]._model.state_dict()
        fresh_sd = fresh[0]._model.state_dict()
        any_differ = any(
            not torch.equal(loaded_sd[k], fresh_sd[k]) for k in loaded_sd
        )
        assert any_differ, "loaded and fresh weights are identical — load may be a no-op"

    def test_single_stage(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=1, tmp_path=tmp_path)
        workers = build_pipeline_from_manifest(tmp_path)
        assert len(workers) == 1
        assert workers[0]._ctx.is_first
        assert workers[0]._ctx.is_last

    def test_planner_ranges_can_reshard_existing_artifact(self, tmp_path):
        """A placement plan may choose cuts different from conversion-time cuts."""
        from meshgpu.backends.portable.pipeline import pipeline_prefill

        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        original = build_pipeline_from_manifest(tmp_path)
        planned = build_pipeline_from_manifest(
            tmp_path,
            devices=[torch.device("cpu")] * 3,
            layer_ranges=[(0, 1), (1, 3), (3, 4)],
        )
        assert [(worker._ctx.layer_start, worker._ctx.layer_end) for worker in planned] == [
            (0, 1), (1, 3), (3, 4)
        ]
        ids = torch.tensor([[1, 5, 9, 13]])
        expected = pipeline_prefill(original, ids)
        actual = pipeline_prefill(planned, ids)
        torch.testing.assert_close(actual, expected)

    def test_planner_ranges_must_cover_all_layers(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="contiguous"):
            build_pipeline_from_manifest(
                tmp_path,
                devices=[torch.device("cpu")] * 2,
                layer_ranges=[(0, 1), (2, 4)],
            )

    def test_build_stage_loads_only_selected_shard(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)

        worker = build_stage_from_manifest(tmp_path, 1, device=torch.device("cpu"))

        assert worker._ctx.stage_id == 1
        assert worker._ctx.layer_start == 2
        assert worker._ctx.layer_end == 4
        assert worker._ctx.is_last
        assert not worker._ctx.is_first

    def test_build_stage_reports_missing_selected_shard(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        (tmp_path / "stage_01.pt").unlink()

        with pytest.raises(FileNotFoundError, match="shard not found"):
            build_stage_from_manifest(tmp_path, 1)

    def test_missing_shard_raises(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        (tmp_path / "stage_01.pt").unlink()
        with pytest.raises(FileNotFoundError):
            build_pipeline_from_manifest(tmp_path)

    def test_manifest_object_accepted(self, tmp_path):
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        manifest = ModelManifest.load(tmp_path / "manifest.json")
        workers = build_pipeline_from_manifest(manifest, artifact_root=tmp_path)
        assert len(workers) == 2

    def test_fallback_no_config_json(self, tmp_path):
        """Works even when config.json is absent (reconstructs from manifest.meta.extra)."""
        cfg = _tiny_cfg()
        _save_shards(cfg, n_stages=2, tmp_path=tmp_path)
        (tmp_path / "config.json").unlink()
        workers = build_pipeline_from_manifest(tmp_path)
        assert len(workers) == 2


# ---------------------------------------------------------------------------
# import_from_hf — skip if transformers absent
# ---------------------------------------------------------------------------

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestImportFromHf:
    def test_tiny_model(self, tmp_path):
        from transformers import LlamaConfig as HFLlamaConfig
        from transformers import LlamaForCausalLM

        from meshgpu.artifacts.hf_import import import_from_hf

        hf_cfg = HFLlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(hf_cfg)
        model_dir = tmp_path / "hf_model"
        model.save_pretrained(model_dir)

        manifest = import_from_hf(str(model_dir), num_stages=2,
                                   out_dir=tmp_path / "shards", dtype="float32")
        assert len(manifest.shards) == 2
        assert manifest.meta.num_layers == 4
        assert manifest.config_path is not None
        assert (tmp_path / "shards" / manifest.config_path).exists()
        assert all((tmp_path / "shards" / shard.path).exists() for shard in manifest.shards)

    def test_pipeline_runs_after_import(self, tmp_path):
        from transformers import LlamaConfig as HFLlamaConfig
        from transformers import LlamaForCausalLM

        from meshgpu.artifacts.hf_import import import_from_hf
        from meshgpu.backends.portable.pipeline import pipeline_prefill

        hf_cfg = HFLlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(hf_cfg)
        model_dir = tmp_path / "hf_model"
        model.save_pretrained(model_dir)

        out = tmp_path / "shards"
        import_from_hf(str(model_dir), num_stages=2, out_dir=out, dtype="float32")
        workers = build_pipeline_from_manifest(out)
        ids = torch.randint(0, 64, (1, 4))
        logits = pipeline_prefill(workers, ids)
        assert logits.shape == (1, 4, 64)

    def test_pipeline_logits_match_huggingface_reference(self, tmp_path):
        """The importer must preserve Llama causal logits, not just shapes."""
        from transformers import LlamaConfig as HFLlamaConfig
        from transformers import LlamaForCausalLM

        from meshgpu.artifacts.hf_import import import_from_hf
        from meshgpu.backends.portable.pipeline import pipeline_prefill

        torch.manual_seed(7)
        hf_cfg = HFLlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        reference = LlamaForCausalLM(hf_cfg).eval()
        model_dir = tmp_path / "hf_model"
        reference.save_pretrained(model_dir)
        artifact = tmp_path / "shards"
        import_from_hf(
            str(model_dir),
            num_stages=2,
            out_dir=artifact,
            dtype="float32",
            include_tokenizer=False,
        )
        workers = build_pipeline_from_manifest(artifact)
        ids = torch.tensor([[1, 5, 9, 13]])
        with torch.no_grad():
            expected = reference(input_ids=ids).logits
        actual = pipeline_prefill(workers, ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_tied_embeddings_pipeline_matches_reference(self, tmp_path):
        """Replicated stage-0 embedding and stage-last head remain equivalent."""
        from transformers import LlamaConfig as HFLlamaConfig
        from transformers import LlamaForCausalLM

        from meshgpu.artifacts.hf_import import import_from_hf
        from meshgpu.backends.portable.pipeline import pipeline_prefill

        torch.manual_seed(11)
        hf_cfg = HFLlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, tie_word_embeddings=True,
        )
        reference = LlamaForCausalLM(hf_cfg).eval()
        model_dir = tmp_path / "hf_model"
        reference.save_pretrained(model_dir)
        artifact = tmp_path / "shards"
        import_from_hf(
            str(model_dir),
            num_stages=2,
            out_dir=artifact,
            dtype="float32",
            include_tokenizer=False,
        )
        workers = build_pipeline_from_manifest(artifact)
        ids = torch.tensor([[1, 5, 9, 13]])
        with torch.no_grad():
            expected = reference(input_ids=ids).logits
        actual = pipeline_prefill(workers, ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
