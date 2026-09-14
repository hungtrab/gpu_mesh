"""Tests for 1F1B pipeline schedule."""
import pytest
import torch

from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.backends.portable.schedule_1f1b import (
    pipeline_train_1f1b,
    split_into_micro_batches,
)
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_pipeline(cfg, num_stages=2):
    return build_pipeline(cfg, num_stages, [torch.device("cpu")] * num_stages)


def _make_micro_batches(cfg, n=4, batch=2, seq_len=4):
    return [
        (
            torch.randint(0, cfg.vocab_size, (batch, seq_len)),
            torch.randint(0, cfg.vocab_size, (batch, seq_len)),
        )
        for _ in range(n)
    ]


class TestSplitIntoMicroBatches:
    def test_even_split(self):
        ids = torch.arange(8).unsqueeze(1).expand(8, 4)
        labels = ids.clone()
        slices = split_into_micro_batches(ids, labels, n_micro=4)
        assert len(slices) == 4
        for ids_m, _ in slices:
            assert ids_m.shape[0] == 2

    def test_uneven_split(self):
        ids = torch.zeros(5, 4, dtype=torch.long)
        labels = ids.clone()
        slices = split_into_micro_batches(ids, labels, n_micro=2)
        assert len(slices) == 2
        sizes = [s[0].shape[0] for s in slices]
        assert sum(sizes) == 5

    def test_too_few_samples_raises(self):
        ids = torch.zeros(2, 4, dtype=torch.long)
        labels = ids.clone()
        with pytest.raises(ValueError, match="n_micro"):
            split_into_micro_batches(ids, labels, n_micro=4)


class TestPipelineTrain1F1B:
    def test_returns_results_per_microbatch(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        micros = _make_micro_batches(cfg, n=4)
        results = pipeline_train_1f1b(workers, micros, opts)
        assert len(results) == 4

    def test_micro_indices_complete(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        micros = _make_micro_batches(cfg, n=3)
        results = pipeline_train_1f1b(workers, micros, opts)
        indices = sorted(r.micro_idx for r in results)
        assert indices == list(range(3))

    def test_all_losses_positive(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        micros = _make_micro_batches(cfg, n=4)
        results = pipeline_train_1f1b(workers, micros, opts)
        for r in results:
            assert r.loss > 0, f"microbatch {r.micro_idx} loss={r.loss}"

    def test_empty_microbatches_returns_empty(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        results = pipeline_train_1f1b(workers, [], opts)
        assert results == []

    def test_single_microbatch(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        micros = _make_micro_batches(cfg, n=1)
        results = pipeline_train_1f1b(workers, micros, opts)
        assert len(results) == 1
        assert results[0].micro_idx == 0

    def test_n_valid_tokens_positive(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        micros = _make_micro_batches(cfg, n=2)
        results = pipeline_train_1f1b(workers, micros, opts)
        for r in results:
            assert r.n_valid_tokens > 0

    def test_loss_decreases_with_1f1b(self):
        """Overfit a single repeated microbatch — loss should trend down."""
        cfg = _tiny_cfg()
        torch.manual_seed(7)
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=5e-2) for w in workers]

        fixed_ids = torch.randint(0, cfg.vocab_size, (4, 4))
        fixed_labels = fixed_ids.clone()
        micros_one = [(fixed_ids, fixed_labels)]

        first_loss = None
        last_loss = None
        for _ in range(15):
            results = pipeline_train_1f1b(workers, micros_one, opts)
            loss = results[0].loss
            if first_loss is None:
                first_loss = loss
            last_loss = loss

        assert last_loss < first_loss, f"loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"

    def test_tied_embeddings_remain_synchronized_after_step(self):
        cfg = _tiny_cfg()
        cfg.tie_word_embeddings = True
        workers = _make_pipeline(cfg)
        opts = [
            torch.optim.SGD(worker._model.parameters(), lr=1e-2)
            for worker in workers
        ]
        ids = torch.randint(0, cfg.vocab_size, (2, 4))
        pipeline_train_1f1b(workers, [(ids, ids.clone())], opts)

        torch.testing.assert_close(
            workers[0]._model.embed_tokens.weight,
            workers[-1]._model.lm_head.weight,
        )

    def test_failure_clears_gradients_before_retry(self, monkeypatch):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        ids = torch.randint(0, cfg.vocab_size, (2, 4))

        def fail_backward(*_args, **_kwargs):
            raise RuntimeError("simulated upstream failure")

        monkeypatch.setattr(workers[0], "backward_train", fail_backward)
        with pytest.raises(RuntimeError, match="simulated upstream failure"):
            pipeline_train_1f1b(workers, [(ids, ids.clone())], opts)

        assert all(
            parameter.grad is None
            for worker in workers
            for parameter in worker._model.parameters()
        )
        assert all(not worker._saved for worker in workers)
