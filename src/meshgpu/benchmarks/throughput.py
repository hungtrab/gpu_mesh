"""
MeshGPU throughput benchmarks.

Measures:
  1. Prefill throughput — tokens/s for varying prompt lengths
  2. Decode throughput — tokens/s for sustained decode (batch = 1)
  3. Pipeline bubble — GPipe vs 1F1B wall-clock comparison
  4. Speculative decoding speedup — vs standard greedy decode
  5. End-to-end tokens/s — full prefill + N decode steps

Results are printed to stdout and returned as BenchmarkReport for programmatic use.

Usage (CLI):
    python -m meshgpu.benchmarks.throughput --model tiny --stages 2 --device cpu

Usage (Python):
    from meshgpu.benchmarks.throughput import run_all
    report = run_all(cfg, num_stages=2)
    print(report.summary())
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

import torch

from meshgpu.backends.portable.pipeline import (
    build_pipeline,
    pipeline_decode_step,
    pipeline_prefill,
    pipeline_train_step,
)
from meshgpu.backends.portable.schedule_1f1b import (
    pipeline_train_1f1b,
    split_into_micro_batches,
)
from meshgpu.inference.sampling import SamplingParams
from meshgpu.inference.speculative import speculative_decode
from meshgpu.models.llama_dense import LlamaConfig

log = logging.getLogger(__name__)


@dataclass
class BenchResult:
    name: str
    tokens_per_second: float
    mean_latency_ms: float
    std_latency_ms: float
    samples: int
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name:<35} "
            f"tok/s={self.tokens_per_second:>10.1f}  "
            f"lat={self.mean_latency_ms:>8.1f}ms ±{self.std_latency_ms:.1f}ms  "
            f"(n={self.samples})"
        )


@dataclass
class BenchmarkReport:
    results: list[BenchResult] = field(default_factory=list)
    cfg_summary: str = ""

    def summary(self) -> str:
        lines = ["", "=" * 80, "MeshGPU Benchmark Report", "-" * 80]
        if self.cfg_summary:
            lines.append(self.cfg_summary)
            lines.append("-" * 80)
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 80)
        return "\n".join(lines)

    def add(self, r: BenchResult) -> None:
        self.results.append(r)
        print(r)


def _time_fn(fn, n_warmup=2, n_bench=10) -> list[float]:
    """Run fn() n_warmup + n_bench times; return n_bench latencies in seconds."""
    for _ in range(n_warmup):
        fn()
    lats = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn()
        lats.append(time.perf_counter() - t0)
    return lats


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_prefill(
    cfg: LlamaConfig,
    num_stages: int,
    prompt_len: int = 128,
    batch: int = 1,
    device: torch.device = torch.device("cpu"),
    n_bench: int = 5,
) -> BenchResult:
    """Prefill throughput: tokens processed per second."""
    devices = [device] * num_stages
    ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len))

    op = [0]
    def fn():
        # Fresh workers so KV caches don't accumulate
        workers = build_pipeline(cfg, num_stages, devices)
        op[0] += 1
        pipeline_prefill(workers, ids, operation_id=op[0], attempt_id="bench")

    lats = _time_fn(fn, n_warmup=2, n_bench=n_bench)
    total_tokens = batch * prompt_len
    tps = total_tokens / statistics.mean(lats)
    return BenchResult(
        name=f"prefill  prompt={prompt_len} batch={batch} stages={num_stages}",
        tokens_per_second=tps,
        mean_latency_ms=statistics.mean(lats) * 1000,
        std_latency_ms=statistics.stdev(lats) * 1000 if len(lats) > 1 else 0.0,
        samples=n_bench,
    )


def bench_decode(
    cfg: LlamaConfig,
    num_stages: int,
    prompt_len: int = 32,
    n_decode: int = 32,
    device: torch.device = torch.device("cpu"),
    n_bench: int = 3,
) -> BenchResult:
    """Decode throughput: tokens per second (batch=1)."""
    devices = [device] * num_stages
    ids = torch.randint(0, cfg.vocab_size, (1, prompt_len))

    op = [0]
    def fn():
        # Fresh workers each call so KV caches don't accumulate across bench iterations
        workers = build_pipeline(cfg, num_stages, devices)
        op[0] += 1
        logits = pipeline_prefill(workers, ids, operation_id=op[0], attempt_id="bench_pf")
        next_id = int(logits[0, -1, :].argmax())
        for _ in range(n_decode):
            op[0] += 1
            logits = pipeline_decode_step(
                workers,
                torch.tensor([[next_id]]),
                operation_id=op[0],
                attempt_id="bench_dc",
            )
            next_id = int(logits[0, -1, :].argmax())

    lats = _time_fn(fn, n_warmup=1, n_bench=n_bench)
    tps = n_decode / statistics.mean(lats)
    return BenchResult(
        name=f"decode   n_steps={n_decode} stages={num_stages}",
        tokens_per_second=tps,
        mean_latency_ms=statistics.mean(lats) * 1000 / n_decode,
        std_latency_ms=(statistics.stdev(lats) * 1000 / n_decode) if len(lats) > 1 else 0.0,
        samples=n_bench,
        extra={"prompt_len": prompt_len, "n_decode": n_decode},
    )


def bench_pipeline_bubble(
    cfg: LlamaConfig,
    num_stages: int,
    n_micro: int = 4,
    batch: int = 4,
    seq_len: int = 8,
    device: torch.device = torch.device("cpu"),
    n_bench: int = 3,
) -> tuple[BenchResult, BenchResult]:
    """
    Compare GPipe (pipeline_train_step looped) vs 1F1B wall-clock time
    for the same total number of samples.
    """
    devices = [device] * num_stages
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    labels = ids.clone()
    micros = split_into_micro_batches(ids, labels, n_micro)

    # GPipe: run each microbatch sequentially with full step
    def gpipe_fn():
        workers = build_pipeline(cfg, num_stages, devices)
        opts = [torch.optim.SGD(w._model.parameters(), lr=1e-4) for w in workers]
        for i, (mb_ids, mb_labels) in enumerate(micros):
            pipeline_train_step(workers, mb_ids, mb_labels, opts, operation_id=i, attempt_id="g")

    def f1b1_fn():
        workers = build_pipeline(cfg, num_stages, devices)
        opts = [torch.optim.SGD(w._model.parameters(), lr=1e-4) for w in workers]
        pipeline_train_1f1b(workers, micros, opts)

    gpipe_lats = _time_fn(gpipe_fn, n_warmup=1, n_bench=n_bench)
    f1b1_lats = _time_fn(f1b1_fn, n_warmup=1, n_bench=n_bench)

    total_tokens = batch * seq_len
    tps_gpipe = total_tokens / statistics.mean(gpipe_lats)
    tps_1f1b = total_tokens / statistics.mean(f1b1_lats)

    gpipe_r = BenchResult(
        name=f"train_gpipe n_micro={n_micro} stages={num_stages}",
        tokens_per_second=tps_gpipe,
        mean_latency_ms=statistics.mean(gpipe_lats) * 1000,
        std_latency_ms=statistics.stdev(gpipe_lats) * 1000 if len(gpipe_lats) > 1 else 0.0,
        samples=n_bench,
    )
    f1b1_r = BenchResult(
        name=f"train_1f1b n_micro={n_micro} stages={num_stages}",
        tokens_per_second=tps_1f1b,
        mean_latency_ms=statistics.mean(f1b1_lats) * 1000,
        std_latency_ms=statistics.stdev(f1b1_lats) * 1000 if len(f1b1_lats) > 1 else 0.0,
        samples=n_bench,
        extra={"gpipe_ms": statistics.mean(gpipe_lats) * 1000},
    )
    return gpipe_r, f1b1_r


def bench_speculative(
    cfg: LlamaConfig,
    num_stages: int,
    prompt_len: int = 16,
    max_new: int = 32,
    lookahead: int = 4,
    device: torch.device = torch.device("cpu"),
    n_bench: int = 3,
) -> tuple[BenchResult, BenchResult]:
    """Compare standard greedy decode vs speculative decode wall-clock time."""
    devices = [device] * num_stages
    ids = list(torch.randint(0, cfg.vocab_size, (prompt_len,)).tolist())
    sampling = SamplingParams(temperature=0.0)

    # Standard decode — fresh workers each call so KV doesn't accumulate
    def standard_fn():
        workers = build_pipeline(cfg, num_stages, devices)
        ids_t = torch.tensor([ids])
        logits = pipeline_prefill(workers, ids_t, operation_id=1, attempt_id="s_pf")
        next_id = int(logits[0, -1, :].argmax())
        for step in range(max_new):
            logits = pipeline_decode_step(
                workers,
                torch.tensor([[next_id]]),
                operation_id=step + 2,
                attempt_id="s_dc",
            )
            next_id = int(logits[0, -1, :].argmax())

    # Speculative decode — fresh pipelines each call
    def spec_fn():
        draft = build_pipeline(cfg, 1, [device])
        target = build_pipeline(cfg, num_stages, devices)
        speculative_decode(draft, target, ids, max_new, lookahead=lookahead, sampling=sampling)

    std_lats = _time_fn(standard_fn, n_warmup=1, n_bench=n_bench)
    spec_lats = _time_fn(spec_fn, n_warmup=1, n_bench=n_bench)

    tps_std = max_new / statistics.mean(std_lats)
    tps_spec = max_new / statistics.mean(spec_lats)

    std_r = BenchResult(
        name=f"decode_standard  n={max_new} stages={num_stages}",
        tokens_per_second=tps_std,
        mean_latency_ms=statistics.mean(std_lats) * 1000,
        std_latency_ms=statistics.stdev(std_lats) * 1000 if len(std_lats) > 1 else 0.0,
        samples=n_bench,
    )
    spec_r = BenchResult(
        name=f"decode_speculative k={lookahead} stages={num_stages}",
        tokens_per_second=tps_spec,
        mean_latency_ms=statistics.mean(spec_lats) * 1000,
        std_latency_ms=statistics.stdev(spec_lats) * 1000 if len(spec_lats) > 1 else 0.0,
        samples=n_bench,
        extra={"speedup": tps_spec / tps_std if tps_std > 0 else 0},
    )
    return std_r, spec_r


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_all(
    cfg: LlamaConfig | None = None,
    num_stages: int = 2,
    device: str = "cpu",
    quick: bool = False,
) -> BenchmarkReport:
    """Run all benchmarks and return report."""
    if cfg is None:
        cfg = LlamaConfig(
            vocab_size=32000, hidden_size=512, intermediate_size=1024,
            num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
            head_dim=64, max_position_embeddings=2048,
        )
    dev = torch.device(device)
    n_bench = 3 if quick else 10
    report = BenchmarkReport(
        cfg_summary=(
            f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
            f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
            f"stages={num_stages} device={device}"
        )
    )

    print("\nRunning benchmarks...")
    for plen in ([32] if quick else [32, 128, 512]):
        r = bench_prefill(cfg, num_stages, prompt_len=plen, device=dev, n_bench=n_bench)
        report.add(r)

    decode_r = bench_decode(
        cfg,
        num_stages,
        n_decode=(16 if quick else 64),
        device=dev,
        n_bench=n_bench,
    )
    report.add(decode_r)

    gpipe_r, f1b1_r = bench_pipeline_bubble(
        cfg,
        num_stages,
        device=dev,
        n_bench=max(2, n_bench // 3),
    )
    report.add(gpipe_r)
    report.add(f1b1_r)

    std_r, spec_r = bench_speculative(cfg, num_stages, device=dev, n_bench=max(2, n_bench // 3))
    report.add(std_r)
    report.add(spec_r)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _tiny_cfg_for_bench() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=1024, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=512,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=["tiny", "default"])
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = _tiny_cfg_for_bench() if args.model == "tiny" else None
    report = run_all(cfg=cfg, num_stages=args.stages, device=args.device, quick=args.quick)
    print(report.summary())
