"""Tests for export_checkpoint and load_exported_stage."""
import pytest
import torch

from meshgpu.artifacts.export import _merge_lora_inplace, export_checkpoint, load_exported_stage
from meshgpu.artifacts.manifest import ModelManifest
from meshgpu.backends.native.lora_recipe import LoRAConfig, LoRALinear, apply_lora
from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.models.llama_dense import LlamaConfig, LlamaStage


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_and_save_checkpoint(cfg, tmp_path, num_stages=2) -> tuple[CheckpointCoordinator, str]:
    """Build a 2-stage model, save a checkpoint, return coordinator + ckpt_id."""
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    n = cfg.num_hidden_layers
    base, rem = divmod(n, num_stages)
    stages = []
    layer_cursor = 0
    for i in range(num_stages):
        n_l = base + (1 if i < rem else 0)
        ls, le = layer_cursor, layer_cursor + n_l
        layer_cursor = le
        stage = LlamaStage(cfg, ls, le, has_embedding=(i == 0), has_lm_head=(i == num_stages - 1))
        stages.append(stage)

    opts = [torch.optim.Adam(s.parameters(), lr=1e-3) for s in stages]
    ckpt_id = coord.save(
        job_id="test",
        global_step=1,
        model_states=[s.state_dict() for s in stages],
        optimizer_states=[o.state_dict() for o in opts],
    )
    return coord, ckpt_id, stages


class TestExportCheckpoint:
    def test_produces_manifest(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        assert isinstance(manifest, ModelManifest)
        assert len(manifest.shards) == 2

    def test_manifest_saved_to_disk(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        assert (out_dir / "manifest.json").exists()

    def test_manifest_references_revision_specific_config_and_shards(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)

        assert manifest.config_path is not None
        assert (out_dir / manifest.config_path).exists()
        assert all(
            shard.path != "stage_00_layers_0000_0001.pt"
            for shard in manifest.shards
        )
        assert all((out_dir / shard.path).exists() for shard in manifest.shards)

    def test_shard_files_exist(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        for shard in manifest.shards:
            shard_path = out_dir / shard.path
            assert shard_path.exists(), f"missing shard {shard_path}"

    def test_sha256_in_manifest_matches_file(self, tmp_path):
        import hashlib
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        for shard in manifest.shards:
            p = out_dir / shard.path
            h = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            assert h.hexdigest() == shard.sha256

    def test_layer_ranges_cover_all_layers(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        assert manifest.shards[0].layer_start == 0
        assert manifest.shards[-1].layer_end == cfg.num_hidden_layers

    def test_meta_fields(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        assert manifest.meta.model_family == "llama_dense"
        assert manifest.meta.num_layers == cfg.num_hidden_layers
        assert manifest.meta.vocab_size == cfg.vocab_size


class TestLoadExportedStage:
    def test_roundtrip_loads_correct_weights(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, original_stages = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)

        # Load stage 0 back
        loaded = load_exported_stage(manifest, 0, cfg, out_dir)
        assert isinstance(loaded, LlamaStage)

        orig_sd = original_stages[0].state_dict()
        loaded_sd = loaded.state_dict()
        for k in orig_sd:
            assert k in loaded_sd, f"key {k} missing from loaded"

    def test_hash_mismatch_raises(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)

        # Corrupt the shard file
        shard0_path = out_dir / manifest.shards[0].path
        shard0_path.write_bytes(shard0_path.read_bytes() + b"\x00corrupt")

        with pytest.raises(ValueError, match="hash mismatch"):
            load_exported_stage(manifest, 0, cfg, out_dir)

    def test_config_mismatch_is_rejected_before_loading(self, tmp_path):
        cfg = _tiny_cfg()
        coord, ckpt_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_checkpoint(coord, ckpt_id, out_dir, cfg, num_stages=2)
        incompatible = LlamaConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
        )

        with pytest.raises(ValueError, match="manifest/config mismatch"):
            load_exported_stage(manifest, 0, incompatible, out_dir)


class TestMergeLoraInplace:
    def test_lora_replaced_with_linear(self, tmp_path):
        cfg = _tiny_cfg()
        stage = LlamaStage(cfg, 0, 2, has_embedding=True, has_lm_head=True)
        apply_lora(stage, LoRAConfig(rank=2, alpha=1.0))

        # Confirm LoRALinear is present before merge
        assert isinstance(stage.layers[0].self_attn.q_proj, LoRALinear)

        _merge_lora_inplace(stage)

        # After merge: should be nn.Linear
        import torch.nn as nn
        assert isinstance(stage.layers[0].self_attn.q_proj, nn.Linear)

    def test_merge_noop_when_no_lora(self, tmp_path):
        import torch.nn as nn
        cfg = _tiny_cfg()
        stage = LlamaStage(cfg, 0, 2, has_embedding=True, has_lm_head=True)

        # No LoRA applied — merge should be a no-op
        _merge_lora_inplace(stage)
        assert isinstance(stage.layers[0].self_attn.q_proj, nn.Linear)

    def test_merged_weights_match_lora_formula(self):
        """LoRA merge: W_merged = W_base + (B @ A) * scale."""
        import torch.nn as nn
        cfg = _tiny_cfg()
        stage = LlamaStage(cfg, 0, 1, has_embedding=False, has_lm_head=False)
        apply_lora(stage, LoRAConfig(rank=2, alpha=2.0))

        attn = stage.layers[0].self_attn
        q_lora: LoRALinear = attn.q_proj

        # Record expected merged weight before merge
        with torch.no_grad():
            expected = q_lora.base.weight + (q_lora.lora_B @ q_lora.lora_A) * q_lora.scale

        _merge_lora_inplace(stage)

        merged_q = attn.q_proj
        assert isinstance(merged_q, nn.Linear)
        assert torch.allclose(merged_q.weight, expected, atol=1e-5)


def test_lora_checkpoint_export_merges_adapter_without_losing_delta(tmp_path):
    cfg = _tiny_cfg()
    lora_cfg = LoRAConfig(rank=2, alpha=4.0)
    stage = LlamaStage(cfg, 0, cfg.num_hidden_layers, has_embedding=True, has_lm_head=True)
    apply_lora(stage, lora_cfg)
    with torch.no_grad():
        stage.layers[0].self_attn.q_proj.lora_B.fill_(0.1)

    optimizer = torch.optim.Adam(
        [parameter for parameter in stage.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    coordinator = CheckpointCoordinator(tmp_path / "ckpts")
    checkpoint_id = coordinator.save(
        job_id="lora-export",
        global_step=1,
        model_states=[stage.state_dict()],
        optimizer_states=[optimizer.state_dict()],
    )
    manifest = export_checkpoint(
        coordinator,
        checkpoint_id,
        tmp_path / "export",
        cfg,
        num_stages=1,
        compute_dtype="float32",
        lora_config=lora_cfg,
    )
    loaded = load_exported_stage(manifest, 0, cfg, tmp_path / "export")
    ids = torch.tensor([[3, 5, 7]])
    with torch.no_grad():
        expected, _ = stage(torch.empty(0), input_ids=ids)
        actual, _ = loaded(torch.empty(0), input_ids=ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_lora_export_requires_recipe_metadata(tmp_path):
    cfg = _tiny_cfg()
    stage = LlamaStage(cfg, 0, cfg.num_hidden_layers, has_embedding=True, has_lm_head=True)
    apply_lora(stage, LoRAConfig(rank=2, alpha=4.0))
    optimizer = torch.optim.Adam(
        [parameter for parameter in stage.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    coordinator = CheckpointCoordinator(tmp_path / "ckpts")
    checkpoint_id = coordinator.save(
        job_id="lora-export",
        global_step=1,
        model_states=[stage.state_dict()],
        optimizer_states=[optimizer.state_dict()],
    )
    with pytest.raises(ValueError, match="lora_config"):
        export_checkpoint(
            coordinator,
            checkpoint_id,
            tmp_path / "export",
            cfg,
            num_stages=1,
            compute_dtype="float32",
        )


def test_export_preserves_slash_model_id_in_metadata(tmp_path):
    cfg = _tiny_cfg()
    coord, checkpoint_id, _ = _make_and_save_checkpoint(cfg, tmp_path)
    manifest = export_checkpoint(
        coord,
        checkpoint_id,
        tmp_path / "export",
        cfg,
        num_stages=2,
        model_id="acme/TinyLlama",
    )

    assert manifest.meta.model_id == "acme/TinyLlama"
    assert "/" not in manifest.manifest_id


def test_qwen_export_loader_rejects_incompatible_caller_config(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Qwen3 is unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from meshgpu.artifacts.qwen_import import remap_qwen_keys
    from meshgpu.models.qwen3_hf import Qwen3Stage

    cfg = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_sliding_window=False,
    )
    reference = Qwen3ForCausalLM(cfg)
    stages = []
    for stage_id, (start, end) in enumerate([(0, 1), (1, 2)]):
        first = stage_id == 0
        last = stage_id == 1
        stage = Qwen3Stage(
            cfg,
            start,
            end,
            has_embedding=first,
            has_lm_head=last,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        stage.load_state_dict(
            remap_qwen_keys(
                reference.state_dict(),
                start,
                end,
                is_first=first,
                is_last=last,
                tie_word_embeddings=bool(cfg.tie_word_embeddings),
            )
        )
        stages.append(stage)
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    opts = [torch.optim.Adam(stage.parameters(), lr=1e-3) for stage in stages]
    checkpoint_id = coord.save(
        job_id="qwen-config-check",
        global_step=1,
        model_states=[stage.state_dict() for stage in stages],
        optimizer_states=[opt.state_dict() for opt in opts],
    )
    manifest = export_checkpoint(
        coord,
        checkpoint_id,
        tmp_path / "export",
        cfg,
        num_stages=2,
        compute_dtype="float32",
        attn_implementation="sdpa",
    )
    incompatible = Qwen3Config.from_dict({**cfg.to_dict(), "hidden_size": 24})
    with pytest.raises(ValueError, match="manifest/config mismatch"):
        load_exported_stage(manifest, 0, incompatible, tmp_path / "export")


def test_qwen_checkpoint_export_roundtrips_through_qwen_loader(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Qwen3 is unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from meshgpu.artifacts.qwen_import import remap_qwen_keys
    from meshgpu.models.qwen3_hf import Qwen3Stage

    cfg = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_sliding_window=False,
    )
    torch.manual_seed(77)
    reference = Qwen3ForCausalLM(cfg)
    stages = []
    for stage_id, (start, end) in enumerate([(0, 1), (1, 2)]):
        first = stage_id == 0
        last = stage_id == 1
        stage = Qwen3Stage(
            cfg,
            start,
            end,
            has_embedding=first,
            has_lm_head=last,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        stage.load_state_dict(
            remap_qwen_keys(
                reference.state_dict(),
                start,
                end,
                is_first=first,
                is_last=last,
                tie_word_embeddings=bool(cfg.tie_word_embeddings),
            )
        )
        stages.append(stage)

    coord = CheckpointCoordinator(tmp_path / "ckpts")
    opts = [torch.optim.Adam(stage.parameters(), lr=1e-3) for stage in stages]
    checkpoint_id = coord.save(
        job_id="qwen-export",
        global_step=1,
        model_states=[stage.state_dict() for stage in stages],
        optimizer_states=[opt.state_dict() for opt in opts],
    )
    manifest = export_checkpoint(
        coord,
        checkpoint_id,
        tmp_path / "export",
        cfg,
        num_stages=2,
        compute_dtype="float32",
        attn_implementation="sdpa",
    )
    assert manifest.meta.adapter == "qwen3_hf_v1"

    from meshgpu.artifacts.export import load_exported_stage

    loaded = [
        load_exported_stage(manifest, stage_id, cfg, tmp_path / "export")
        for stage_id in range(2)
    ]
    from meshgpu.backends.portable.pipeline import pipeline_prefill
    from meshgpu.backends.portable.stage_worker import StageContext, StageWorker

    workers = [
        StageWorker(
            stage,
            StageContext(
                stage_id,
                stage.layer_start,
                stage.layer_end,
                torch.device("cpu"),
                stage_id == 0,
                stage_id == 1,
                "qwen-export",
                "attempt",
            ),
        )
        for stage_id, stage in enumerate(loaded)
    ]
    ids = torch.tensor([[1, 4, 9, 16]])
    with torch.no_grad():
        expected = reference(input_ids=ids, use_cache=False).logits
    actual = pipeline_prefill(workers, ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
