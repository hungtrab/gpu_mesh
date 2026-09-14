"""Tests for topology change: reshard_from_checkpoint."""
import pytest
import torch

from meshgpu.backends.native.lora_recipe import LoRAConfig, apply_lora
from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.backends.portable.reshard import _remap_key, reshard_from_checkpoint
from meshgpu.backends.portable.trainer import PortableTrainer, TrainerConfig
from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_pipeline(cfg, num_stages, devices=None):
    if devices is None:
        devices = [torch.device("cpu")] * num_stages
    return build_pipeline(cfg, num_stages, devices)


def _save_checkpoint(cfg, workers, coord):
    opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
    return coord.save(
        job_id="test",
        global_step=1,
        model_states=[w._model.state_dict() for w in workers],
        optimizer_states=[o.state_dict() for o in opts],
    )


class TestRemapKey:
    def test_layer_key_remapped(self):
        assert _remap_key("layers.0.self_attn.q_proj.weight", 2, False, False) == \
               "layers.2.self_attn.q_proj.weight"

    def test_layer_key_rel1_remapped(self):
        assert _remap_key("layers.1.mlp.gate_proj.weight", 3, False, False) == \
               "layers.4.mlp.gate_proj.weight"

    def test_embed_unchanged(self):
        assert _remap_key("embed_tokens.weight", 0, True, False) == "embed_tokens.weight"

    def test_lm_head_unchanged(self):
        assert _remap_key("lm_head.weight", 0, False, True) == "lm_head.weight"

    def test_norm_unchanged(self):
        assert _remap_key("norm.weight", 0, False, True) == "norm.weight"


class TestReshardSameStages:
    def test_2to2_preserves_weights(self, tmp_path):
        """Reshard 2→2 stages: weights should match original."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=2)
        ckpt_id = _save_checkpoint(cfg, workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=2, new_num_stages=2,
        )
        assert len(new_workers) == 2

        # Each stage should have the same weight tensors
        for i, (old_w, new_w) in enumerate(zip(workers, new_workers)):
            old_sd = old_w._model.state_dict()
            new_sd = new_w._model.state_dict()
            for k in old_sd:
                assert k in new_sd, f"stage {i}: key {k} missing"
                assert torch.allclose(old_sd[k].float(), new_sd[k].float(), atol=1e-5), \
                    f"stage {i}: weight {k} mismatch"


class TestReshardScaleDown:
    def test_4to2_fewer_stages(self, tmp_path):
        """Reshard 4→2 stages: new workers get correct layer ranges."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=4)
        ckpt_id = _save_checkpoint(cfg, workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=4, new_num_stages=2,
        )
        assert len(new_workers) == 2
        # Stage 0: layers 0-1, stage 1: layers 2-3
        assert new_workers[0]._model.layer_start == 0
        assert new_workers[0]._model.layer_end == 2
        assert new_workers[1]._model.layer_start == 2
        assert new_workers[1]._model.layer_end == 4

    def test_4to2_forward_runs(self, tmp_path):
        """After reshard, forward inference should produce correct-shaped output."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=4)
        ckpt_id = _save_checkpoint(cfg, workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=4, new_num_stages=2,
        )
        from meshgpu.backends.portable.pipeline import pipeline_prefill
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        logits = pipeline_prefill(new_workers, ids, operation_id=1, attempt_id="test")
        assert logits.shape == (1, 4, cfg.vocab_size)

    def test_4to2_weights_preserved(self, tmp_path):
        """Reshard 4→2: layer 0 and 1 weights from old stage 0 and 1 end up in new stage 0."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        old_workers = _make_pipeline(cfg, num_stages=4)
        ckpt_id = _save_checkpoint(cfg, old_workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=4, new_num_stages=2,
        )
        # Layer 0 of new_stage_0 should match layer 0 of old_stage_0
        old_l0 = old_workers[0]._model.layers[0].state_dict()
        new_l0 = new_workers[0]._model.layers[0].state_dict()
        for k in old_l0:
            assert k in new_l0
            assert torch.allclose(old_l0[k].float(), new_l0[k].float(), atol=1e-5), \
                f"layer 0 key {k} mismatch after 4→2 reshard"


class TestReshardScaleUp:
    def test_2to4_more_stages(self, tmp_path):
        """Reshard 2→4 stages: one layer per stage (4-layer model)."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=2)
        ckpt_id = _save_checkpoint(cfg, workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=2, new_num_stages=4,
        )
        assert len(new_workers) == 4
        for i, w in enumerate(new_workers):
            assert w._model.layer_start == i
            assert w._model.layer_end == i + 1

    def test_2to4_forward_runs(self, tmp_path):
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=2)
        ckpt_id = _save_checkpoint(cfg, workers, coord)
        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord, old_num_stages=2, new_num_stages=4
        )
        from meshgpu.backends.portable.pipeline import pipeline_prefill
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        logits = pipeline_prefill(new_workers, ids, operation_id=1, attempt_id="test")
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestReshardDeviceMismatch:
    def test_wrong_device_count_raises(self, tmp_path):
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        workers = _make_pipeline(cfg, num_stages=2)
        ckpt_id = _save_checkpoint(cfg, workers, coord)
        with pytest.raises(ValueError, match="devices"):
            reshard_from_checkpoint(
                cfg, ckpt_id, coord,
                old_num_stages=2, new_num_stages=3,
                devices=[torch.device("cpu"), torch.device("cpu")],  # only 2, need 3
            )


def test_reshard_rejects_lora_checkpoint_instead_of_silently_corrupting_weights(tmp_path):
    """LoRA metadata is not in the stage state dict, so resharding must fail closed."""
    cfg = _tiny_cfg()
    coord = CheckpointCoordinator(tmp_path / "ckpts")
    workers = _make_pipeline(cfg, num_stages=2)
    for worker in workers:
        apply_lora(worker._model, LoRAConfig(rank=2, alpha=4.0))
    ckpt_id = _save_checkpoint(cfg, workers, coord)

    with pytest.raises(ValueError, match="LoRA"):
        reshard_from_checkpoint(
            cfg,
            ckpt_id,
            coord,
            old_num_stages=2,
            new_num_stages=1,
        )


class TestReshardThenTrain:
    def test_reshard_then_train_step(self, tmp_path):
        """After reshard, training should produce a loss."""
        cfg = _tiny_cfg()
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        old_workers = _make_pipeline(cfg, num_stages=4)
        ckpt_id = _save_checkpoint(cfg, old_workers, coord)

        new_workers = reshard_from_checkpoint(
            cfg, ckpt_id, coord,
            old_num_stages=4, new_num_stages=2,
        )
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in new_workers]
        trainer = PortableTrainer(
            new_workers, opts,
            cfg=TrainerConfig(max_steps=2),
        )
        ids = torch.randint(0, cfg.vocab_size, (2, 4))
        labels = ids.clone()
        results = list(trainer.train(iter([(ids, labels)] * 3)))
        assert len(results) == 2
        for r in results:
            assert r.loss > 0
