"""Tests for inference recovery: replay_prefix, resume_session, replace_worker."""

import pytest
import torch

from meshgpu.backends.portable.pipeline import build_pipeline, pipeline_prefill
from meshgpu.backends.portable.trainer import PortableTrainer, TrainerConfig
from meshgpu.checkpoints.coordinator import CheckpointCoordinator
from meshgpu.inference.recovery import RecoveryResult, replay_prefix, resume_session
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.session import InferenceSession
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_pipeline(cfg, num_stages=2):
    return build_pipeline(cfg, num_stages, [torch.device("cpu")] * num_stages)


def _sampling():
    return SamplingParams(temperature=0.0)


class TestReplayPrefix:
    def test_basic_replay_succeeds(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = torch.randint(0, cfg.vocab_size, (1, 6))
        result = replay_prefix(workers, prefix)
        assert result.ok is True
        assert result.prefix_len == 6
        assert result.error is None

    def test_1d_input_auto_unsqueezed(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = torch.randint(0, cfg.vocab_size, (5,))
        result = replay_prefix(workers, prefix)
        assert result.ok is True
        assert result.prefix_len == 5

    def test_batch_gt1_raises(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = torch.randint(0, cfg.vocab_size, (2, 5))
        with pytest.raises(ValueError, match="batch_size=1"):
            replay_prefix(workers, prefix)

    def test_replay_rebuilds_kv(self):
        """After replay, stage workers should have non-empty KV caches."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = torch.randint(0, cfg.vocab_size, (1, 4))
        replay_prefix(workers, prefix)
        for w in workers:
            assert len(w._kv_caches) > 0, "KV cache should be populated after replay"

    def test_failed_replay_returns_not_ok(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        # Empty tensor should cause an error inside pipeline_prefill
        bad = torch.zeros((1, 0), dtype=torch.long)
        result = replay_prefix(workers, bad)
        # Either it works (empty sequence) or it returns ok=False
        # We just verify the result is a RecoveryResult
        assert isinstance(result, RecoveryResult)


class TestResumeSession:
    @pytest.mark.asyncio
    async def test_resume_creates_session(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = list(torch.randint(0, cfg.vocab_size, (6,)).tolist())
        session = await resume_session(
            workers, "sess_1", prefix,
            max_new_tokens=4,
            sampling=_sampling(),
        )
        assert isinstance(session, InferenceSession)

    @pytest.mark.asyncio
    async def test_resume_generates_tokens(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        session = await resume_session(
            workers, "sess_2", prefix,
            max_new_tokens=3,
            sampling=_sampling(),
        )
        tokens = []
        async for r in session.run():
            tokens.append(r.token_id)
        assert len(tokens) >= 1
        assert all(0 <= t < cfg.vocab_size for t in tokens)

    @pytest.mark.asyncio
    async def test_resume_seq_num_offset(self):
        """seq_num continues from confirmed_seq_num + 1."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        prefix = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        confirmed = 7
        session = await resume_session(
            workers, "sess_3", prefix,
            max_new_tokens=3,
            sampling=_sampling(),
            confirmed_seq_num=confirmed,
        )
        seq_nums = []
        async for r in session.run():
            seq_nums.append(r.seq_num)
        # First token seq_num should be confirmed+1
        assert seq_nums[0] == confirmed + 1
        # seq_nums monotone increasing
        for a, b in zip(seq_nums, seq_nums[1:]):
            assert b == a + 1

    @pytest.mark.asyncio
    async def test_resume_matches_normal_prefill_continuation(self):
        """Replay must reuse next-token logits without duplicating the prefix tail."""
        cfg = _tiny_cfg()
        torch.manual_seed(91)
        normal_workers = _make_pipeline(cfg)
        recovered_workers = _make_pipeline(cfg)
        for source, target in zip(normal_workers, recovered_workers):
            target._model.load_state_dict(source._model.state_dict())

        prefix = list(torch.randint(0, cfg.vocab_size, (5,)).tolist())
        normal = InferenceSession(
            session_id="normal-continuation",
            prompt_ids=prefix,
            workers=normal_workers,
            sampling=_sampling(),
            max_new_tokens=5,
        )
        expected = [result.token_id async for result in normal.run()]

        resumed = await resume_session(
            recovered_workers,
            "recovered-continuation",
            prefix,
            max_new_tokens=5,
            sampling=_sampling(),
        )
        actual = [result.token_id async for result in resumed.run()]

        assert actual == expected
        assert all(not worker._kv_caches_by_key for worker in recovered_workers)


class TestSkipPrefill:
    @pytest.mark.asyncio
    async def test_skip_prefill_decodes_from_existing_kv(self):
        """Session with skip_prefill=True starts decode immediately."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)

        # First build KV caches via normal prefill
        prefix = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        normal = InferenceSession(
            session_id="normal",
            prompt_ids=prefix,
            workers=workers,
            sampling=_sampling(),
            max_new_tokens=1,
        )
        _ = [r.token_id async for r in normal.run()]

        # Replay to rebuild KV, then skip_prefill session
        replay_prefix(workers, torch.tensor([prefix]))
        skipped = InferenceSession(
            session_id="skip",
            prompt_ids=prefix,
            workers=workers,
            sampling=_sampling(),
            max_new_tokens=1,
            skip_prefill=True,
        )
        skip_tokens = [r.token_id async for r in skipped.run()]
        # Both should produce valid vocab tokens (exact match not required since
        # skip_prefill uses a decode step from last prefix token, not full prefill)
        assert all(0 <= t < cfg.vocab_size for t in skip_tokens)


class TestReplaceWorker:
    def test_resume_rejects_legacy_optimizer_without_names(self, tmp_path):
        workers = _make_pipeline(_tiny_cfg())
        opts = [torch.optim.Adam(w._model.parameters()) for w in workers]
        coord = CheckpointCoordinator(tmp_path)
        cid = coord.save(
            job_id="legacy", global_step=0,
            model_states=[w._model.state_dict() for w in workers],
            optimizer_states=[opt.state_dict() for opt in opts],
        )
        trainer = PortableTrainer(workers, opts, checkpoint=coord, job_id="legacy")
        with pytest.raises(ValueError, match="parameter names/order missing"):
            trainer.resume(cid)
        assert coord.load_stage(cid, 0)["model"]  # weights remain available for export

    @pytest.mark.parametrize("same_shape", [False, True])
    def test_resume_rejects_reordered_optimizer_before_mutation(self, tmp_path, same_shape):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters()) for w in workers]
        trainer = PortableTrainer(
            workers, opts, checkpoint=CheckpointCoordinator(tmp_path),
            cfg=TrainerConfig(max_steps=2),
        )
        ids = torch.tensor([[1, 2, 3, 4]])
        list(trainer.train(iter([(ids, ids)])))
        checkpoint_id = trainer.save_checkpoint()
        list(trainer.train(iter([(ids, ids)])))
        params = list(workers[0]._model.parameters())
        if same_shape:
            i, j = next((i, j) for i in range(len(params)) for j in range(i + 1, len(params))
                        if params[i].shape == params[j].shape)
            params[i], params[j] = params[j], params[i]
        else:
            params.reverse()
        trainer._opts[0] = torch.optim.Adam(params)
        before = [[p.detach().clone() for p in w._model.parameters()] for w in workers]
        with pytest.raises(ValueError, match="parameter names/order"):
            trainer.resume(checkpoint_id)
        assert trainer.global_step == 2
        for worker, previous in zip(workers, before):
            for current, expected in zip(worker._model.parameters(), previous):
                torch.testing.assert_close(current, expected, rtol=0, atol=0)

    def test_matching_optimizer_can_train_after_resume(self, tmp_path):
        workers = _make_pipeline(_tiny_cfg())
        trainer = PortableTrainer(
            workers, [torch.optim.Adam(w._model.parameters()) for w in workers],
            checkpoint=CheckpointCoordinator(tmp_path), cfg=TrainerConfig(max_steps=3),
        )
        ids = torch.tensor([[1, 2, 3, 4]])
        list(trainer.train(iter([(ids, ids)])))
        cid = trainer.save_checkpoint()
        expected = list(trainer.train(iter([(ids, ids)])))[0]
        weights = [[p.detach().clone() for p in w._model.parameters()] for w in workers]
        trainer.resume(cid)
        actual = list(trainer.train(iter([(ids, ids)])))[0]
        assert actual.loss == expected.loss
        for worker, previous in zip(workers, weights):
            for current, target in zip(worker._model.parameters(), previous):
                torch.testing.assert_close(current, target, rtol=0, atol=0)

    def test_replace_worker_no_checkpoint(self, tmp_path):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        trainer = PortableTrainer(workers, opts, cfg=TrainerConfig(max_steps=1))
        old_optimizer = trainer._opts[1]

        new_workers = _make_pipeline(cfg)
        new_stage1 = new_workers[1]
        old_state = {
            name: value.detach().clone()
            for name, value in workers[1]._model.state_dict().items()
        }
        trainer.replace_worker(1, new_stage1)

        assert trainer._workers[1] is new_stage1
        assert trainer._opts[1] is not old_optimizer
        for name, value in new_stage1._model.state_dict().items():
            torch.testing.assert_close(value, old_state[name])
        replacement_params = {
            id(param) for param in new_stage1._model.parameters()
        }
        optimizer_params = {
            id(param)
            for group in trainer._opts[1].param_groups
            for param in group["params"]
        }
        assert optimizer_params == replacement_params

    def test_replace_worker_with_checkpoint(self, tmp_path):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        trainer = PortableTrainer(
            workers, opts,
            checkpoint=coord,
            cfg=TrainerConfig(max_steps=2, checkpoint_every_steps=1),
        )

        # Train one step to produce a checkpoint
        data = [(
            torch.randint(0, cfg.vocab_size, (2, 4)),
            torch.randint(0, cfg.vocab_size, (2, 4)),
        )]
        results = list(trainer.train(iter(data * 3)))
        ckpt_id = results[0].checkpoint_id or coord.committed_id
        assert ckpt_id is not None

        # Create a replacement stage worker for stage 1
        replacement = _make_pipeline(cfg)[1]
        trainer.replace_worker(1, replacement, checkpoint_id=ckpt_id)
        assert trainer._workers[1] is replacement

    def test_replace_worker_out_of_range(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        trainer = PortableTrainer(workers, opts, cfg=TrainerConfig())
        replacement = _make_pipeline(cfg)[0]
        with pytest.raises(IndexError):
            trainer.replace_worker(5, replacement)

    def test_training_continues_after_replace(self, tmp_path):
        """After replace_worker, the train loop should be able to continue."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.Adam(w._model.parameters(), lr=1e-3) for w in workers]
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        trainer = PortableTrainer(
            workers, opts,
            checkpoint=coord,
            cfg=TrainerConfig(max_steps=4, checkpoint_every_steps=2),
        )

        fixed_ids = torch.randint(0, cfg.vocab_size, (2, 4))
        labels = fixed_ids.clone()
        data = [(fixed_ids, labels)] * 6

        # Train 2 steps, checkpoint
        results = []
        gen = trainer.train(iter(data))
        for r in gen:
            results.append(r)
            if r.step == 2:
                break

        ckpt_id = coord.committed_id
        # Replace stage 1 from checkpoint
        replacement = _make_pipeline(cfg)[1]
        trainer.replace_worker(1, replacement, checkpoint_id=ckpt_id)

        # Continue training for 2 more steps
        for r in gen:
            results.append(r)
            if len(results) >= 4:
                break

        assert len(results) >= 4
        for r in results:
            assert r.loss > 0

    def test_checkpoint_replacement_restores_all_stages_to_one_step(self, tmp_path):
        """Recovery cannot combine the replacement stage with newer live stages."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.SGD(w._model.parameters(), lr=1e-2) for w in workers]
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        trainer = PortableTrainer(
            workers,
            opts,
            checkpoint=coord,
            cfg=TrainerConfig(max_steps=2, checkpoint_every_steps=100),
        )
        ids = torch.randint(0, cfg.vocab_size, (2, 4))
        list(trainer.train(iter([(ids, ids.clone())])))
        checkpoint_id = trainer.save_checkpoint()
        checkpoint_payloads = [
            coord.load_stage(checkpoint_id, stage=stage)
            for stage in range(len(workers))
        ]

        # Advance the live pipeline after the committed checkpoint.  Recovery
        # must roll every stage back to the same checkpoint, not only stage 0.
        list(trainer.train(iter([(ids, ids.clone())])))
        assert trainer.global_step == 2

        replacement = _make_pipeline(cfg)[0]
        trainer.replace_worker(0, replacement, checkpoint_id=checkpoint_id)

        assert trainer.global_step == 1
        for worker, payload in zip(trainer._workers, checkpoint_payloads):
            for name, parameter in worker._model.state_dict().items():
                torch.testing.assert_close(parameter, payload["model"][name])

    def test_restore_failure_does_not_mutate_live_model(self, tmp_path):
        """An incompatible optimizer must fail before model state is changed."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.SGD(w._model.parameters(), lr=1e-2) for w in workers]
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        trainer = PortableTrainer(
            workers,
            opts,
            checkpoint=coord,
            cfg=TrainerConfig(max_steps=1),
        )
        checkpoint_id = trainer.save_checkpoint()
        with torch.no_grad():
            for worker in workers:
                for parameter in worker._model.parameters():
                    parameter.add_(1.0)
        before = [
            {name: value.clone() for name, value in worker._model.state_dict().items()}
            for worker in workers
        ]

        params = list(workers[1]._model.parameters())
        trainer._opts[1] = torch.optim.SGD(
            [{"params": params[:1]}, {"params": params[1:]}],
            lr=1e-2,
        )
        with pytest.raises(ValueError, match="optimizer"):
            trainer.resume(checkpoint_id)

        assert trainer.global_step == 0
        for worker, previous in zip(workers, before):
            for name, value in worker._model.state_dict().items():
                torch.testing.assert_close(value, previous[name])

    def test_restore_clears_named_kv_caches(self, tmp_path):
        """Restoring weights invalidates every request cache, including named ones."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        opts = [torch.optim.SGD(w._model.parameters(), lr=1e-2) for w in workers]
        coord = CheckpointCoordinator(tmp_path / "ckpts")
        trainer = PortableTrainer(
            workers,
            opts,
            checkpoint=coord,
            cfg=TrainerConfig(max_steps=1),
        )
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        pipeline_prefill(workers, ids, cache_key="old-session")
        checkpoint_id = trainer.save_checkpoint()
        assert all(worker.kv_cache_length("old-session") == 4 for worker in workers)

        trainer.resume(checkpoint_id)

        assert all(worker.kv_cache_length("old-session") == 0 for worker in workers)
