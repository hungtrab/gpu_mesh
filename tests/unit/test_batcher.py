"""Tests for ContinuousBatcher — submit, stream, eviction."""
import asyncio

import pytest
import torch

from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.inference.batcher import ContinuousBatcher
from meshgpu.inference.sampling import SamplingParams
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


class TestContinuousBatcher:
    @pytest.mark.asyncio
    async def test_single_request_generates_tokens(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=4)
        batcher.start()

        prompt = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        seq_id = await batcher.submit(prompt, _sampling(), max_new_tokens=3)

        tokens = []
        async for result in batcher.stream(seq_id):
            tokens.append(result.token_id)

        batcher.stop()
        assert len(tokens) >= 1
        assert all(0 <= t < cfg.vocab_size for t in tokens)

    @pytest.mark.asyncio
    async def test_stream_ends_at_max_new(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=4)
        batcher.start()

        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        seq_id = await batcher.submit(prompt, _sampling(), max_new_tokens=2)

        tokens = []
        async for result in batcher.stream(seq_id):
            tokens.append(result.token_id)
            if result.is_last:
                break

        batcher.stop()
        assert len(tokens) <= 2

    @pytest.mark.asyncio
    async def test_seq_nums_monotone(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=4)
        batcher.start()

        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        seq_id = await batcher.submit(prompt, _sampling(), max_new_tokens=4)

        seq_nums = []
        async for result in batcher.stream(seq_id):
            seq_nums.append(result.seq_num)

        batcher.stop()
        assert seq_nums == sorted(seq_nums)
        assert seq_nums == list(range(1, len(seq_nums) + 1))

    @pytest.mark.asyncio
    async def test_stop_cancels_loop(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=4)
        batcher.start()
        batcher.stop()
        # Allow the event loop to process the cancellation
        if batcher._loop_task:
            try:
                await batcher._loop_task
            except (asyncio.CancelledError, Exception):
                pass
            assert batcher._loop_task.done()

    @pytest.mark.asyncio
    async def test_restart_waits_for_old_loop_to_drain(self):
        batcher = ContinuousBatcher(_make_pipeline(_tiny_cfg()), max_batch_size=1)
        batcher.start()
        batcher.stop()
        with pytest.raises(RuntimeError, match="still stopping"):
            batcher.start()
        await batcher.wait_stopped()
        batcher.start()
        batcher.stop()
        await batcher.wait_stopped()

    @pytest.mark.asyncio
    async def test_duplicate_stream_consumers_are_rejected(self):
        cfg = _tiny_cfg()
        batcher = ContinuousBatcher(_make_pipeline(cfg), max_batch_size=1)
        batcher.start()
        seq_id = await batcher.submit([1, 2], _sampling(), max_new_tokens=2)
        first = batcher.stream(seq_id)
        result = await first.__anext__()
        assert result.token_id >= 0
        second = batcher.stream(seq_id)
        with pytest.raises(RuntimeError, match="active stream consumer"):
            await second.__anext__()
        await first.aclose()
        batcher.stop()
        await batcher.wait_stopped()

    @pytest.mark.asyncio
    async def test_closing_stream_cancels_active_sequence_and_clears_cache(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=1)
        batcher.start()
        seq_id = await batcher.submit([1, 2, 3], _sampling(), max_new_tokens=8)

        stream = batcher.stream(seq_id)
        first = await stream.__anext__()
        assert not first.is_last
        await stream.aclose()

        for _ in range(20):
            if batcher.active_count == 0:
                break
            await asyncio.sleep(0.01)
        batcher.stop()
        await batcher.wait_stopped()
        assert batcher.active_count == 0
        assert all(not worker._kv_caches_by_key for worker in workers)

    @pytest.mark.asyncio
    async def test_stop_wakes_pending_stream(self):
        cfg = _tiny_cfg()
        batcher = ContinuousBatcher(_make_pipeline(cfg), max_batch_size=1)
        batcher.start()
        seq_id = await batcher.submit([1, 2, 3], _sampling(), max_new_tokens=3)

        batcher.stop()
        results = [result async for result in batcher.stream(seq_id)]

        assert len(results) == 1
        assert results[0].token_id == -1
        assert results[0].is_last

    @pytest.mark.asyncio
    async def test_submit_rejects_context_overflow(self):
        cfg = _tiny_cfg()
        batcher = ContinuousBatcher(_make_pipeline(cfg), max_batch_size=1)
        batcher.start()
        with pytest.raises(ValueError, match="exceeds model context"):
            await batcher.submit([1] * 31, _sampling(), max_new_tokens=2)
        batcher.stop()

    @pytest.mark.asyncio
    async def test_active_count_goes_to_zero(self):
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=4)
        batcher.start()

        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        seq_id = await batcher.submit(prompt, _sampling(), max_new_tokens=2)

        async for _ in batcher.stream(seq_id):
            pass

        # Give loop one more tick to evict
        await asyncio.sleep(0.05)
        batcher.stop()
        assert batcher.active_count == 0

    @pytest.mark.asyncio
    async def test_two_sequences_keep_separate_kv_caches(self):
        """Dynamic admission must not make the second prompt continue the first."""
        cfg = _tiny_cfg()
        workers = _make_pipeline(cfg)
        batcher = ContinuousBatcher(workers, max_batch_size=2)
        batcher.start()

        first = await batcher.submit([2, 3, 5], _sampling(), max_new_tokens=3)
        second = await batcher.submit([7, 11, 13, 17], _sampling(), max_new_tokens=3)
        first_tokens, second_tokens = await asyncio.gather(
            _collect(batcher, first),
            _collect(batcher, second),
        )

        batcher.stop()
        if batcher._loop_task:
            try:
                await batcher._loop_task
            except asyncio.CancelledError:
                pass
        assert len(first_tokens) == 3
        assert len(second_tokens) == 3
        assert all(not worker._kv_caches_by_key for worker in workers)


async def _collect(batcher: ContinuousBatcher, seq_id: str) -> list[int]:
    return [result.token_id async for result in batcher.stream(seq_id)]
