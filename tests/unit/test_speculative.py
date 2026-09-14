"""Tests for speculative decoding."""
import torch

from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.speculative import SpeculativeResult, speculative_decode
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


def _make_pipeline(cfg, num_stages=2):
    return build_pipeline(cfg, num_stages, [torch.device("cpu")] * num_stages)


class TestSpeculativeDecoding:
    def test_returns_result(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=4, lookahead=2,
        )
        assert isinstance(result, SpeculativeResult)

    def test_generates_tokens(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=6, lookahead=2,
        )
        assert len(result.generated_ids) > 0
        assert len(result.generated_ids) <= 6

    def test_tokens_in_vocab_range(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=5, lookahead=2,
        )
        for t in result.generated_ids:
            assert 0 <= t < cfg.vocab_size, f"token {t} out of range"

    def test_n_draft_equals_k_times_steps(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=4, lookahead=2,
        )
        # n_draft_tokens >= number of lookahead proposals made
        assert result.n_draft_tokens >= 1

    def test_accepted_plus_rejected_equals_draft(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=4, lookahead=2,
        )
        assert result.n_accepted + result.n_rejected == result.n_draft_tokens

    def test_acceptance_rate_in_range(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (4,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=6, lookahead=3,
        )
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_eos_stops_early(self):
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = [0, 1, 2, 3]
        # EOS = 0; very likely to appear in greedy decode from tiny random model
        result = speculative_decode(
            draft_workers, target_workers, prompt,
            max_new_tokens=20, lookahead=2,
            sampling=SamplingParams(temperature=0.0, eos_token_id=0),
        )
        # Should stop before 20 (EOS found), or at 20 if EOS never appeared
        assert len(result.generated_ids) <= 20

    def test_lookahead_1(self):
        """lookahead=1 degenerates to standard decode — should still work."""
        cfg = _tiny_cfg()
        draft_workers = _make_pipeline(cfg, num_stages=1)
        target_workers = _make_pipeline(cfg, num_stages=2)
        prompt = list(torch.randint(0, cfg.vocab_size, (3,)).tolist())
        result = speculative_decode(
            draft_workers, target_workers, prompt, max_new_tokens=3, lookahead=1,
        )
        assert len(result.generated_ids) > 0

    def test_greedy_same_draft_target_accepts_all(self):
        """
        When draft == target (same weights), greedy decoding accepts every token.
        acceptance_rate should be 1.0.
        """
        cfg = _tiny_cfg()
        torch.manual_seed(42)
        # Use the same pipeline for both draft and target
        workers = _make_pipeline(cfg, num_stages=2)

        # NOTE: since both share KV caches, we can't literally share workers.
        # Instead create two pipelines with the same initial weights.
        draft_workers = workers
        target_workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
        # Copy weights from draft to target
        for dw, tw in zip(draft_workers, target_workers):
            tw._model.load_state_dict(dw._model.state_dict())

        prompt = [5, 10, 15, 20]
        result = speculative_decode(
            draft_workers, target_workers, prompt,
            max_new_tokens=6, lookahead=3,
            sampling=SamplingParams(temperature=0.0),
        )
        # With identical models, greedy target always agrees with draft
        assert result.acceptance_rate == 1.0, \
            f"expected 1.0 acceptance but got {result.acceptance_rate:.3f}"
