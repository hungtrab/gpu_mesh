"""Task-scoped LoRA adaptation and reset tests."""
from __future__ import annotations

import pytest
import torch

from meshgpu.backends.native.lora_recipe import LoRAConfig
from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.models.llama_dense import LlamaConfig
from meshgpu.training.ttt import TaskTTTConfig, TaskTTTSession


def _cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_implementation="sdpa",
    )


def test_task_ttt_updates_only_lora_and_reset_restores_baseline() -> None:
    torch.manual_seed(4)
    workers = build_pipeline(_cfg(), 2, [torch.device("cpu")] * 2)
    base_before = [
        {name: value.detach().clone() for name, value in worker._model.named_parameters()}
        for worker in workers
    ]
    session = TaskTTTSession(
        workers,
        TaskTTTConfig(
            lora=LoRAConfig(rank=2, alpha=4.0),
            learning_rate=5e-2,
            max_steps=2,
        ),
    )
    baseline = session.adapter_state_dict()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    before = session.adapter_state_dict()
    results = session.adapt(input_ids, labels, steps=2)
    assert len(results) == 2
    after = session.adapter_state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)

    for worker, before_params in zip(workers, base_before):
        for name, value in worker._model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                original_name = name.replace(".base.weight", ".weight")
                torch.testing.assert_close(value, before_params[original_name])

    session.reset_task()
    reset = session.adapter_state_dict()
    for key, value in baseline.items():
        torch.testing.assert_close(reset[key], value)
    assert all(not worker._kv_caches_by_key for worker in workers)
    assert all(not optimizer.state for optimizer in session._optimizers)


def test_task_ttt_rejects_empty_and_bad_shapes() -> None:
    workers = build_pipeline(_cfg(), 2, [torch.device("cpu")] * 2)
    session = TaskTTTSession(workers)
    with pytest.raises(ValueError, match="same shape"):
        session.adapt(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2, dtype=torch.long))


def test_task_ttt_can_run_two_tasks_from_same_baseline() -> None:
    torch.manual_seed(5)
    workers = build_pipeline(_cfg(), 2, [torch.device("cpu")] * 2)
    session = TaskTTTSession(
        workers,
        TaskTTTConfig(lora=LoRAConfig(rank=2, alpha=2.0), learning_rate=1e-2, max_steps=1),
    )
    task = torch.tensor([[3, 4, 5, 6]])
    session.adapt(task, task, steps=1)
    output_after_first = session.adapter_state_dict()
    session.reset_task()
    baseline = session.adapter_state_dict()
    assert any(not torch.equal(output_after_first[key], baseline[key]) for key in baseline)
    session.adapt(task, task, steps=1)
    output_after_second = session.adapter_state_dict()
    for key in output_after_first:
        torch.testing.assert_close(output_after_first[key], output_after_second[key])


def test_qwen_hf_stage_supports_task_lora() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Qwen3 is unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from meshgpu.artifacts.qwen_import import remap_qwen_keys
    from meshgpu.backends.portable.stage_worker import StageContext, StageWorker
    from meshgpu.models.qwen3_hf import Qwen3Stage

    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_sliding_window=False,
    )
    torch.manual_seed(8)
    reference = Qwen3ForCausalLM(cfg)
    workers = []
    for stage_id, (start, end) in enumerate([(0, 1), (1, 2)]):
        first = stage_id == 0
        last = stage_id == 1
        stage = Qwen3Stage(
            cfg,
            start,
            end,
            has_embedding=first,
            has_lm_head=last,
            dtype=torch.float32,
        )
        stage.load_state_dict(
            remap_qwen_keys(
                reference.state_dict(),
                start,
                end,
                is_first=first,
                is_last=last,
            )
        )
        workers.append(
            StageWorker(
                stage,
                StageContext(
                    stage_id,
                    start,
                    end,
                    torch.device("cpu"),
                    first,
                    last,
                    "qwen-ttt",
                    "attempt",
                ),
            )
        )
    session = TaskTTTSession(
        workers,
        TaskTTTConfig(
            lora=LoRAConfig(rank=2, alpha=2.0),
            learning_rate=1e-2,
            max_steps=1,
        ),
    )
    result = session.adapt(
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[2, 3, 4, 5]]),
    )
    assert len(result) == 1
    assert all(
        any("lora_A" in name for name, _ in worker._model.named_parameters())
        for worker in workers
    )
