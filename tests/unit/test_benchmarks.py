"""Tests for benchmark utilities — correct structure, no division-by-zero, etc."""

from meshgpu.benchmarks.throughput import (
    BenchmarkReport,
    BenchResult,
    bench_decode,
    bench_pipeline_bubble,
    bench_prefill,
    bench_speculative,
    run_all,
)
from meshgpu.models.llama_dense import LlamaConfig


def _tiny_cfg():
    # max_position_embeddings must exceed worst-case KV accumulation.
    # Speculative decode appends k+1 tokens to target KV per round regardless of
    # acceptance — with lookahead=4 and 0% acceptance: 16 + 5*max_new = 16+160 = 176.
    # Use 512 to be safe for all bench_* defaults.
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512,
    )


class TestBenchResult:
    def test_str_has_name(self):
        r = BenchResult("my_bench", 1000.0, 1.0, 0.1, 5)
        assert "my_bench" in str(r)

    def test_report_summary(self):
        report = BenchmarkReport()
        r = BenchResult("test", 500.0, 2.0, 0.2, 3)
        report.add(r)
        s = report.summary()
        assert "MeshGPU Benchmark Report" in s
        assert "test" in s


class TestBenchPrefill:
    def test_returns_bench_result(self):
        cfg = _tiny_cfg()
        r = bench_prefill(cfg, num_stages=2, prompt_len=8, n_bench=2)
        assert isinstance(r, BenchResult)

    def test_tps_positive(self):
        cfg = _tiny_cfg()
        r = bench_prefill(cfg, num_stages=2, prompt_len=8, n_bench=2)
        assert r.tokens_per_second > 0

    def test_latency_positive(self):
        cfg = _tiny_cfg()
        r = bench_prefill(cfg, num_stages=2, prompt_len=4, n_bench=2)
        assert r.mean_latency_ms > 0


class TestBenchDecode:
    def test_returns_bench_result(self):
        cfg = _tiny_cfg()
        r = bench_decode(cfg, num_stages=2, prompt_len=4, n_decode=4, n_bench=2)
        assert isinstance(r, BenchResult)

    def test_tps_positive(self):
        cfg = _tiny_cfg()
        r = bench_decode(cfg, num_stages=2, prompt_len=4, n_decode=4, n_bench=2)
        assert r.tokens_per_second > 0


class TestBenchPipelineBubble:
    def test_returns_two_results(self):
        cfg = _tiny_cfg()
        gpipe_r, f1b1_r = bench_pipeline_bubble(
            cfg, num_stages=2, n_micro=2, batch=2, seq_len=4, n_bench=2
        )
        assert isinstance(gpipe_r, BenchResult)
        assert isinstance(f1b1_r, BenchResult)

    def test_both_have_positive_tps(self):
        cfg = _tiny_cfg()
        gpipe_r, f1b1_r = bench_pipeline_bubble(
            cfg, num_stages=2, n_micro=2, batch=2, seq_len=4, n_bench=2
        )
        assert gpipe_r.tokens_per_second > 0
        assert f1b1_r.tokens_per_second > 0


class TestBenchSpeculative:
    def test_returns_two_results(self):
        cfg = _tiny_cfg()
        std_r, spec_r = bench_speculative(
            cfg, num_stages=2, prompt_len=4, max_new=4, lookahead=2, n_bench=2
        )
        assert isinstance(std_r, BenchResult)
        assert isinstance(spec_r, BenchResult)

    def test_speedup_in_extra(self):
        cfg = _tiny_cfg()
        _, spec_r = bench_speculative(
            cfg, num_stages=2, prompt_len=4, max_new=4, lookahead=2, n_bench=2
        )
        assert "speedup" in spec_r.extra
        assert spec_r.extra["speedup"] >= 0


class TestRunAll:
    def test_run_all_quick(self):
        cfg = _tiny_cfg()
        report = run_all(cfg=cfg, num_stages=2, device="cpu", quick=True)
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) >= 4

    def test_all_tps_positive(self):
        cfg = _tiny_cfg()
        report = run_all(cfg=cfg, num_stages=2, device="cpu", quick=True)
        for r in report.results:
            assert r.tokens_per_second > 0, f"{r.name} has tps={r.tokens_per_second}"
