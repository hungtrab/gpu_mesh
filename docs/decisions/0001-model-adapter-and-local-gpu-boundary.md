# ADR 0001: Model adapter and same-host stage boundary

Status: accepted for the current prototype

Date: 2026-09-13

## Context

MeshGPU has two separate correctness risks: reimplementing a model block while
also distributing it, and silently moving every stage boundary through CPU
memory. The first makes parity failures difficult to diagnose. The second
makes a two-GPU machine behave like a CPU/network pipeline even when the GPUs
are in the same host.

The capacity milestone is deliberately fit-first. It must compare the same
model revision, dtype, prompt, batch and output budget, and it must use a real
single-GPU CUDA OOM as evidence. A planner estimate or a CPU test is not a
passing capacity result.

## Decision

1. Qwen3 is supported through the official Hugging Face `Qwen3DecoderLayer`,
   `Qwen3RMSNorm`, rotary embedding and cache implementation. MeshGPU only
   owns the contiguous layer range, endpoint embedding/head, stage cache and
   boundary protocol. `meshgpu convert` records the source revision and
   selected attention implementation in the manifest.
2. Llama remains available through the small standalone dense adapter. Its
   implementation is a correctness reference, not a claim that every Llama
   variant is supported.
3. The portable pipeline exposes an explicit `transport` policy: `cpu` is the
   deterministic fallback used by remote/TCP-compatible paths; `local_cuda`
   keeps same-host CUDA boundaries device-resident when peer access is
   available, and otherwise performs an explicit host staging copy. The latter
   is not advertised as zero-copy.
4. Activation checkpointing is opt-in per stage and only applies during
   training. It is wired through both Llama and Qwen artifact loaders; cache
   enabled inference never uses it.
5. Admission may use exact workload memory profiles. A static estimate is
   marked `safe_for_admission=false`, and the capacity harness returns
   `pending_hardware` when the required physical CUDA topology is unavailable.
   The harness records per-device allocated/reserved peaks and compares a
   sharded result to a reference output.
6. The first training adaptation recipe is custom LoRA with PEFT-compatible
   target semantics (`q_proj`/`v_proj` by default). It is intentionally
   separate from QLoRA: quantizer, FSDP and kernel compatibility have not been
   established, so `qlora` is not exposed by the CLI.

## Consequences

- A Qwen artifact can be loaded one stage at a time without materializing the
  complete model on one GPU. Unsupported Qwen configurations fail during
  import or stage-schema validation instead of producing a plausible but wrong
  result.
- `local_cuda` can reduce host copies on a multi-GPU host, but it does not make
  disjoint VRAM a single CUDA address space. Device placement and peer access
  still need to be probed on the target machine.
- CPU fallback remains necessary for correctness tests and remote stage
  execution. Network transport, relay and QUIC are separate concerns from
  same-host peer copies.
- No release documentation may claim “two GPUs fit” until a capacity report
  has a real single-GPU OOM, a successful sharded run, matching reference
  output, and the per-GPU peak evidence.

## Verification required before promotion

- Qwen reference parity for logits, cached decode, loss, gradients and one
  optimizer step.
- Same-host CUDA test with at least two physical devices, including peer and
  host-fallback paths.
- Capacity acceptance using one immutable `CapacityContract`; run destructive
  OOM trials in isolated processes.
- At least 100 optimizer steps plus checkpoint/resume and export/inference on
  the model and dtype that will be advertised.
