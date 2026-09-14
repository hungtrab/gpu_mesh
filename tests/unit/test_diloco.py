"""Tests for DiLoCo federated LoRA trainer."""
import pytest
import torch

from meshgpu.backends.native.lora_recipe import LoRAConfig, apply_lora
from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.models.llama_dense import LlamaConfig
from meshgpu.training.diloco import (
    DiLoCoConfig,
    DiLoCoStepResult,
    DiLoCoTrainer,
    _adapter_params,
    _allreduce_outer_grads,
    _snapshot_params,
)


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_replica(cfg, num_stages=2):
    workers = build_pipeline(cfg, num_stages, [torch.device("cpu")] * num_stages)
    for w in workers:
        apply_lora(w._model, LoRAConfig(rank=2, alpha=1.0))
    return workers


def _make_inner_opts(workers):
    return [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]


def _batches(cfg, n=20, batch=2, seq_len=4):
    for _ in range(n):
        ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
        yield ids, ids.clone()


class TestAdapterParamHelpers:
    def test_adapter_params_only_lora(self):
        cfg = _tiny_cfg()
        workers = _make_replica(cfg)
        params = _adapter_params(workers)
        assert len(params) > 0
        for p in params:
            # Should only be lora tensors; non-lora params not included
            assert p.requires_grad

    def test_snapshot_and_load_roundtrip(self):
        cfg = _tiny_cfg()
        workers = _make_replica(cfg)
        snap = _snapshot_params(workers, lora_only=True)
        params_before = [p.data.clone() for p in _adapter_params(workers)]

        # Corrupt params
        for p in _adapter_params(workers):
            p.data.fill_(999.0)

        # Restore
        from meshgpu.training.diloco import _load_params
        _load_params(workers, snap, lora_only=True)
        params_after = [p.data.clone() for p in _adapter_params(workers)]

        for b, a in zip(params_before, params_after):
            assert torch.allclose(b, a, atol=1e-5)


class TestDiLoCoTrainer:
    def test_requires_at_least_one_replica(self):
        with pytest.raises(ValueError, match="replica"):
            DiLoCoTrainer([], [], cfg=DiLoCoConfig())

    def test_yields_step_results(self):
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        r1 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0, r1],
            [_make_inner_opts(r0), _make_inner_opts(r1)],
            cfg=DiLoCoConfig(inner_steps=5),
        )
        results = []
        for r in trainer.train(_batches(cfg, n=6), max_outer_steps=2):
            results.append(r)

        assert len(results) > 0
        for r in results:
            assert isinstance(r, DiLoCoStepResult)
            assert r.loss > 0

    def test_outer_sync_fires_at_inner_steps_boundary(self):
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0],
            [_make_inner_opts(r0)],
            cfg=DiLoCoConfig(inner_steps=3),
        )
        results = list(trainer.train(_batches(cfg, n=6), max_outer_steps=2))
        synced = [r for r in results if r.synced]
        # With inner_steps=3 and 6 batches, should fire outer sync at steps 3 and 6
        assert len(synced) == 2

    def test_outer_step_increments_on_sync(self):
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0],
            [_make_inner_opts(r0)],
            cfg=DiLoCoConfig(inner_steps=2),
        )
        results = list(trainer.train(_batches(cfg, n=6), max_outer_steps=3))
        outer_steps = [r.outer_step for r in results if r.synced]
        assert outer_steps == [1, 2, 3]

    def test_inner_step_increments_monotonically(self):
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0],
            [_make_inner_opts(r0)],
            cfg=DiLoCoConfig(inner_steps=5),
        )
        results = list(trainer.train(_batches(cfg, n=5), max_outer_steps=1))
        steps = [r.inner_step for r in results]
        assert steps == list(range(1, len(steps) + 1))

    def test_replicas_get_same_weights_after_sync(self):
        """After outer sync, all replicas should have identical adapter weights."""
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        r1 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0, r1],
            [_make_inner_opts(r0), _make_inner_opts(r1)],
            cfg=DiLoCoConfig(inner_steps=3),
        )
        # Run exactly inner_steps batches to trigger one sync
        list(trainer.train(_batches(cfg, n=3), max_outer_steps=1))

        p0 = _adapter_params(r0)
        p1 = _adapter_params(r1)
        for a, b in zip(p0, p1):
            assert torch.allclose(a.data, b.data, atol=1e-5), \
                "replicas diverged after outer sync"

    def test_outer_sync_applies_displacement_to_reference(self):
        """The outer step must start from theta_ref, not replica 0's local theta."""
        cfg = _tiny_cfg()
        replicas = [_make_replica(cfg), _make_replica(cfg)]
        for replica in replicas:
            for parameter in _adapter_params(replica):
                parameter.data.fill_(1.0)

        trainer = DiLoCoTrainer(
            replicas,
            [_make_inner_opts(replica) for replica in replicas],
            cfg=DiLoCoConfig(
                inner_steps=1,
                outer_lr=0.7,
                outer_momentum=0.0,
            ),
        )
        for parameter in _adapter_params(replicas[0]):
            parameter.data.fill_(0.8)
        for parameter in _adapter_params(replicas[1]):
            parameter.data.fill_(0.6)

        # mean(theta_ref - theta_local) = (0.2 + 0.4) / 2 = 0.3;
        # SGD from theta_ref=1.0 with lr=0.7 therefore yields 0.79.
        trainer._outer_sync()

        for replica in replicas:
            for parameter in _adapter_params(replica):
                torch.testing.assert_close(
                    parameter,
                    torch.full_like(parameter, 0.79),
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_loss_decreases_overfit(self):
        """Single replica should overfit a fixed batch over many outer steps."""
        cfg = _tiny_cfg()
        torch.manual_seed(3)
        r0 = _make_replica(cfg)
        trainer = DiLoCoTrainer(
            [r0],
            [_make_inner_opts(r0)],
            cfg=DiLoCoConfig(inner_steps=5, outer_lr=0.5),
        )
        fixed_ids = torch.randint(0, cfg.vocab_size, (2, 4))
        fixed_labels = fixed_ids.clone()
        data = [(fixed_ids, fixed_labels)] * 50

        results = list(trainer.train(iter(data), max_outer_steps=10))
        first = results[0].loss
        last = results[-1].loss
        assert last < first, f"loss did not decrease: {first:.4f} → {last:.4f}"


class TestOuterGradAllReduce:
    def test_allreduce_mean_of_displacements(self):
        cfg = _tiny_cfg()
        r0 = _make_replica(cfg)
        r1 = _make_replica(cfg)

        # Set known params
        for p in _adapter_params(r0):
            p.data.fill_(1.0)
        for p in _adapter_params(r1):
            p.data.fill_(3.0)

        ref = _snapshot_params(r0, lora_only=True)
        # ref is 1.0; r0 displacement = ref - r0 = 0; r1 displacement = ref - r1 = 1-3 = -2
        # mean = -1.0
        grads = _allreduce_outer_grads([r0, r1], ref, lora_only=True)
        for g in grads:
            assert torch.allclose(g, torch.full_like(g, -1.0), atol=1e-5), \
                f"expected -1.0, got {g}"
