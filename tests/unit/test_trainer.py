"""Tests for PortableTrainer — gradient accumulation, grad norm, checkpoint."""
import random

import pytest
import torch

from meshgpu.backends.portable.pipeline import build_pipeline, pipeline_train_step
from meshgpu.backends.portable.trainer import PortableTrainer, TrainerConfig, _noop_opt
from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_pipeline(cfg, num_stages=2):
    devices = [torch.device("cpu")] * num_stages
    return build_pipeline(cfg, num_stages, devices)


def _constant_batches(vocab_size, n, seq_len=4, batch=2):
    for _ in range(n):
        ids = torch.randint(0, vocab_size, (batch, seq_len))
        labels = ids.clone()
        yield ids, labels


def _opts(workers, lr=1e-3, cls=torch.optim.Adam):
    return [cls(w._model.parameters(), lr=lr) for w in workers]


class TestNoopOpt:
    def test_step_does_nothing(self):
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.ones(4)
        opt = torch.optim.SGD([param], lr=1.0)
        noop = _noop_opt(opt)
        noop.step()
        assert (param == 0).all()

    def test_zero_grad_does_nothing(self):
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.ones(4)
        opt = torch.optim.SGD([param], lr=1.0)
        noop = _noop_opt(opt)
        noop.zero_grad(set_to_none=True)
        assert param.grad is not None

    def test_state_dict_delegates(self):
        param = torch.nn.Parameter(torch.zeros(4))
        opt = torch.optim.SGD([param], lr=1.0)
        noop = _noop_opt(opt)
        assert "param_groups" in noop.state_dict()


class TestPortableTrainerBasic:
    def test_yields_step_results(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(
            workers, _opts(workers),
            cfg=TrainerConfig(max_steps=3, log_every_steps=1),
        )
        results = list(trainer.train(_constant_batches(cfg.vocab_size, 5)))
        assert len(results) == 3
        for r in results:
            assert r.loss > 0
            assert r.step > 0

    def test_loss_decreases_over_steps(self):
        """Overfit on a single repeated batch — loss must strictly decrease."""
        cfg = _tiny_cfg()
        torch.manual_seed(42)
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(
            workers, _opts(workers, lr=5e-2),
            cfg=TrainerConfig(max_steps=30, log_every_steps=100),
        )
        # Repeat the same (ids, labels) batch so the model can memorize
        fixed_ids = torch.randint(0, cfg.vocab_size, (2, 4))
        fixed_labels = fixed_ids.clone()
        data = [(fixed_ids, fixed_labels)] * 30
        results = list(trainer.train(iter(data)))
        first_loss = results[0].loss
        last_loss = results[-1].loss
        assert last_loss < first_loss, f"loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"

    def test_stops_at_max_steps(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(
            workers, _opts(workers, lr=1e-4, cls=torch.optim.SGD),
            cfg=TrainerConfig(max_steps=2),
        )
        results = list(trainer.train(_constant_batches(cfg.vocab_size, 100)))
        assert len(results) == 2

    def test_grad_norm_is_nonneg(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(workers, _opts(workers), cfg=TrainerConfig(max_steps=2))
        results = list(trainer.train(_constant_batches(cfg.vocab_size, 5)))
        for r in results:
            assert r.grad_norm >= 0


class TestGradientAccumulation:
    def test_accumulation_produces_fewer_yields(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(
            workers, _opts(workers),
            cfg=TrainerConfig(max_steps=4, gradient_accumulation_steps=2),
        )
        results = list(trainer.train(_constant_batches(cfg.vocab_size, 8)))
        assert len(results) == 4

    def test_step_count_field(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        trainer = PortableTrainer(workers, _opts(workers), cfg=TrainerConfig(max_steps=3))
        steps = [r.step for r in trainer.train(_constant_batches(cfg.vocab_size, 10))]
        assert steps == [1, 2, 3]


class TestClipGradNorm:
    def test_clip_returns_finite(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg, num_stages=1)
        trainer = PortableTrainer(workers, _opts(workers), cfg=TrainerConfig(max_grad_norm=0.5))
        results = list(trainer.train(_constant_batches(cfg.vocab_size, 3, seq_len=4)))
        for r in results:
            assert r.grad_norm != float("inf")

    def test_no_grads_returns_zero(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg, num_stages=1)
        trainer = PortableTrainer(workers, _opts(workers), cfg=TrainerConfig())
        norm = trainer._clip_grad_norm()
        assert norm == 0.0


def test_amp_overflow_skips_every_stage_and_scheduler(monkeypatch):
    """A single overflowing stage must not allow a partial pipeline update."""
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg)
    opts = _opts(workers, lr=1e-2, cls=torch.optim.SGD)
    schedulers = [
        torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        for opt in opts
    ]
    scaler = torch.amp.GradScaler("cpu")
    trainer = PortableTrainer(
        workers,
        opts,
        schedulers=schedulers,
        scaler=scaler,
        cfg=TrainerConfig(max_steps=1, max_grad_norm=1.0),
    )

    before = [next(worker._model.parameters()).detach().clone() for worker in workers]

    def inject_one_stage_overflow(workers, _input_ids, _labels, _optimizers, **kwargs):
        active_scaler = kwargs["gradient_scaler"]
        for index, worker in enumerate(workers):
            parameter = next(worker._model.parameters())
            active_scaler.scale(parameter.sum()).backward()
            if index == 0:
                parameter.grad.fill_(float("inf"))
        return {"loss": 1.0, "n_valid_tokens": 1}

    monkeypatch.setattr(
        "meshgpu.backends.portable.trainer.pipeline_train_step",
        inject_one_stage_overflow,
    )
    ids = torch.ones(1, 2, dtype=torch.long)
    result = next(iter(trainer.train(iter([(ids, ids.clone())]))))

    assert result.overflow
    assert all(
        torch.equal(previous, next(worker._model.parameters()))
        for previous, worker in zip(before, workers)
    )
    assert all(scheduler.last_epoch == 0 for scheduler in schedulers)


def test_accumulation_normalizes_by_total_valid_tokens():
    """Two microbatches must equal one concatenated batch by gradient."""
    cfg = _tiny_cfg()
    torch.manual_seed(123)
    accumulated = _make_pipeline(cfg)
    combined = _make_pipeline(cfg)
    for left, right in zip(accumulated, combined):
        right._model.load_state_dict(left._model.state_dict())

    ids_a = torch.randint(0, cfg.vocab_size, (1, 4))
    labels_a = ids_a.clone()
    ids_b = torch.randint(0, cfg.vocab_size, (1, 4))
    labels_b = ids_b.clone()
    labels_b[:, :3] = -100
    total_valid = 5
    opts_a = _opts(accumulated, cls=torch.optim.SGD)
    opts_b = _opts(combined, cls=torch.optim.SGD)

    for index, (ids, labels) in enumerate(((ids_a, labels_a), (ids_b, labels_b)), start=1):
        pipeline_train_step(
            accumulated,
            ids,
            labels,
            [_noop_opt(opt) for opt in opts_a],
            operation_id=index,
            loss_normalizer=total_valid,
        )
    pipeline_train_step(
        combined,
        torch.cat([ids_a, ids_b]),
        torch.cat([labels_a, labels_b]),
        [_noop_opt(opt) for opt in opts_b],
        operation_id=1,
        loss_normalizer=total_valid,
    )

    for accumulated_worker, combined_worker in zip(accumulated, combined):
        for left, right in zip(
            accumulated_worker._model.parameters(),
            combined_worker._model.parameters(),
        ):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)


def test_tied_embeddings_share_parameter_in_one_stage():
    cfg = _tiny_cfg()
    cfg.tie_word_embeddings = True
    worker = _make_pipeline(cfg, num_stages=1)[0]

    assert worker._model.embed_tokens.weight is worker._model.lm_head.weight


def test_tied_embeddings_stay_synced_across_training_stages():
    cfg = _tiny_cfg()
    cfg.tie_word_embeddings = True
    workers = _make_pipeline(cfg, num_stages=2)
    opts = _opts(workers, lr=1e-2, cls=torch.optim.SGD)
    trainer = PortableTrainer(
        workers,
        opts,
        cfg=TrainerConfig(max_steps=2, log_every_steps=100),
    )
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    results = list(trainer.train(iter([(ids, ids.clone())] * 2)))

    assert len(results) == 2
    torch.testing.assert_close(
        workers[0]._model.embed_tokens.weight,
        workers[-1]._model.lm_head.weight,
    )


def test_failed_accumulation_group_discards_partial_gradients():
    """A failed later microbatch cannot contaminate the next retry."""
    cfg = _tiny_cfg()
    torch.manual_seed(44)
    workers = _make_pipeline(cfg)
    trainer = PortableTrainer(
        workers,
        _opts(workers, lr=1e-2, cls=torch.optim.SGD),
        cfg=TrainerConfig(max_steps=1, gradient_accumulation_steps=2),
    )
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    bad_labels = ids.clone()
    bad_labels[0, 0] = cfg.vocab_size

    with pytest.raises(IndexError):
        list(trainer.train(iter([(ids, ids.clone()), (ids, bad_labels)])))

    assert trainer._accum_loss == 0.0
    assert trainer._accum_tokens == 0
    assert all(
        parameter.grad is None
        for worker in workers
        for parameter in worker._model.parameters()
    )

    result = next(iter(trainer.train(iter([(ids, ids.clone()), (ids, ids.clone())]))))
    assert result.n_valid_tokens == 8


def test_all_masked_accumulation_group_is_skipped():
    """Padding-only groups must not call the pipeline with a zero normalizer."""
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg)
    trainer = PortableTrainer(
        workers,
        _opts(workers, cls=torch.optim.SGD),
        cfg=TrainerConfig(max_steps=1),
    )
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    masked = torch.full_like(ids, -100)
    valid_results = list(
        trainer.train(iter([(ids, masked), (ids, ids.clone())]))
    )

    assert len(valid_results) == 1
    assert valid_results[0].step == 1
    assert valid_results[0].n_valid_tokens == 4


def test_checkpoint_resume_restores_rng_state(tmp_path):
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg, num_stages=1)
    coordinator = CheckpointCoordinator(tmp_path / "checkpoints")
    trainer = PortableTrainer(
        workers,
        _opts(workers, cls=torch.optim.SGD),
        checkpoint=coordinator,
        cfg=TrainerConfig(max_steps=1, checkpoint_every_steps=1),
    )
    batch = next(_constant_batches(cfg.vocab_size, 1))
    random.seed(321)
    torch.manual_seed(321)
    result = next(iter(trainer.train(iter([batch]))))
    checkpoint_id = result.checkpoint_id
    assert checkpoint_id is not None

    expected_torch = torch.rand(3)
    expected_python = random.random()
    trainer.resume(checkpoint_id)

    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert random.random() == expected_python


def test_checkpoint_resume_restores_data_cursor(tmp_path):
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg, num_stages=1)
    coordinator = CheckpointCoordinator(tmp_path / "checkpoints")
    trainer = PortableTrainer(
        workers,
        _opts(workers, cls=torch.optim.SGD),
        checkpoint=coordinator,
        cfg=TrainerConfig(
            max_steps=2,
            gradient_accumulation_steps=2,
            checkpoint_every_steps=1,
        ),
    )
    batches = list(_constant_batches(cfg.vocab_size, 4))
    result = next(iter(trainer.train(iter(batches[:2]))))
    checkpoint_id = result.checkpoint_id
    assert checkpoint_id is not None
    assert trainer.data_cursor == 2

    payload = coordinator.load_stage(checkpoint_id, stage=0)
    assert payload["data_cursor"] == {
        "schema": "packed_batch_v1",
        "batches_consumed": 2,
        "gradient_accumulation_steps": 2,
    }

    trainer.resume(checkpoint_id)
    assert trainer.data_cursor == 2
    assert trainer.data_cursor_exact


def test_explicit_checkpoint_is_allowed_after_one_yield(tmp_path):
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg, num_stages=1)
    trainer = PortableTrainer(
        workers,
        _opts(workers, cls=torch.optim.SGD),
        checkpoint=CheckpointCoordinator(tmp_path / "checkpoints"),
        cfg=TrainerConfig(max_steps=2),
    )
    batch = next(_constant_batches(cfg.vocab_size, 1))
    result = next(iter(trainer.train(iter([batch]))))

    assert result.step == 1
    assert trainer.save_checkpoint()


def test_failed_accumulation_rolls_back_data_cursor(tmp_path):
    cfg = _tiny_cfg()
    workers = _make_pipeline(cfg, num_stages=1)
    trainer = PortableTrainer(
        workers,
        _opts(workers, cls=torch.optim.SGD),
        checkpoint=CheckpointCoordinator(tmp_path / "checkpoints"),
        cfg=TrainerConfig(max_steps=1, gradient_accumulation_steps=2),
    )
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    bad_labels = ids.clone()
    bad_labels[0, 0] = cfg.vocab_size

    with pytest.raises(IndexError):
        list(trainer.train(iter([(ids, ids.clone()), (ids, bad_labels)])))

    assert trainer.data_cursor == 0
