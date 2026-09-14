# ruff: noqa: E402
"""Qwen3 HF adapter tests: official-block parity and stage artifact loading."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("installed Transformers has no Qwen3 support", allow_module_level=True)

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from meshgpu.artifacts.qwen_import import (
    build_qwen_pipeline_from_manifest,
    import_qwen_from_hf,
    remap_qwen_keys,
    split_layers,
)  # noqa: E402
from meshgpu.backends.portable.pipeline import (  # noqa: E402
    pipeline_decode_step,
    pipeline_prefill,
    pipeline_train_step,
)
from meshgpu.backends.portable.stage_worker import StageContext, StageWorker  # noqa: E402
from meshgpu.models.qwen3_hf import Qwen3Stage  # noqa: E402
from meshgpu.planner.job import model_spec_from_manifest  # noqa: E402


def _config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_sliding_window=False,
    )


def _workers_from_reference(model: Qwen3ForCausalLM, n_stages: int = 2) -> list[StageWorker]:
    cfg = model.config
    workers: list[StageWorker] = []
    ranges = split_layers(cfg.num_hidden_layers, n_stages)
    for stage_id, (layer_start, layer_end) in enumerate(ranges):
        first = stage_id == 0
        last = stage_id == n_stages - 1
        stage = Qwen3Stage(
            cfg,
            layer_start,
            layer_end,
            has_embedding=first,
            has_lm_head=last,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        state = remap_qwen_keys(
            model.state_dict(),
            layer_start,
            layer_end,
            is_first=first,
            is_last=last,
            tie_word_embeddings=bool(cfg.tie_word_embeddings),
        )
        missing, unexpected = stage.load_state_dict(state, strict=True)
        assert not missing and not unexpected
        workers.append(
            StageWorker(
                stage,
                StageContext(
                    stage_id,
                    layer_start,
                    layer_end,
                    torch.device("cpu"),
                    first,
                    last,
                    "qwen-test",
                    "attempt",
                ),
            )
        )
    return workers


def test_qwen_stage_uses_official_blocks_and_matches_reference() -> None:
    torch.manual_seed(12)
    reference = Qwen3ForCausalLM(_config()).eval()
    workers = _workers_from_reference(reference)
    assert all(
        type(layer).__name__ == "Qwen3DecoderLayer"
        for worker in workers
        for layer in worker._model.layers
    )

    input_ids = torch.tensor([[1, 5, 9, 13]])
    with torch.no_grad():
        expected = reference(input_ids=input_ids, use_cache=False).logits
    actual = pipeline_prefill(workers, input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert [worker.kv_cache_length() for worker in workers] == [4, 4]


def test_qwen_cached_decode_matches_reference() -> None:
    torch.manual_seed(13)
    reference = Qwen3ForCausalLM(_config()).eval()
    workers = _workers_from_reference(reference)
    input_ids = torch.tensor([[2, 3, 7]])
    pipeline_prefill(workers, input_ids)
    next_token = torch.tensor([[11]])
    actual = pipeline_decode_step(workers, next_token, operation_id=2)

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=reference.config)
    with torch.no_grad():
        reference(input_ids=input_ids, past_key_values=cache, use_cache=True)
        expected = reference(input_ids=next_token, past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert [worker.kv_cache_length() for worker in workers] == [4, 4]


def test_qwen_import_and_manifest_pipeline_preserve_logits(tmp_path) -> None:
    torch.manual_seed(14)
    reference = Qwen3ForCausalLM(_config()).eval()
    model_dir = tmp_path / "qwen3"
    reference.save_pretrained(model_dir, safe_serialization=True)
    artifact_dir = tmp_path / "artifact"

    manifest = import_qwen_from_hf(
        str(model_dir),
        num_stages=2,
        out_dir=artifact_dir,
        dtype="float32",
        include_tokenizer=False,
    )
    assert manifest.meta.model_family == "qwen3"
    assert manifest.meta.adapter == "qwen3_hf_v1"
    workers = build_qwen_pipeline_from_manifest(artifact_dir)
    input_ids = torch.tensor([[4, 8, 15, 16]])
    with torch.no_grad():
        expected = reference(input_ids=input_ids, use_cache=False).logits
    actual = pipeline_prefill(workers, input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_generic_manifest_builder_dispatches_qwen(tmp_path) -> None:
    reference = Qwen3ForCausalLM(_config()).eval()
    model_dir = tmp_path / "qwen3"
    reference.save_pretrained(model_dir, safe_serialization=True)
    artifact_dir = tmp_path / "artifact"
    import_qwen_from_hf(str(model_dir), 2, artifact_dir, dtype="float32", include_tokenizer=False)

    from meshgpu.backends.portable.pipeline import build_pipeline_from_manifest

    workers = build_pipeline_from_manifest(artifact_dir)
    assert workers[0]._model.meshgpu_adapter == "qwen3_hf_v1"
    assert workers[1]._ctx.is_last


def test_qwen_manifest_can_load_planner_selected_ranges(tmp_path) -> None:
    torch.manual_seed(23)
    reference = Qwen3ForCausalLM(_config()).eval()
    model_dir = tmp_path / "qwen3"
    reference.save_pretrained(model_dir, safe_serialization=True)
    artifact_dir = tmp_path / "artifact"
    import_qwen_from_hf(str(model_dir), 2, artifact_dir, dtype="float32", include_tokenizer=False)

    original = build_qwen_pipeline_from_manifest(artifact_dir)
    planned = build_qwen_pipeline_from_manifest(
        artifact_dir,
        devices=[torch.device("cpu")] * 3,
        layer_ranges=[(0, 1), (1, 3), (3, 4)],
    )
    input_ids = torch.tensor([[4, 8, 15, 16]])
    expected = pipeline_prefill(original, input_ids)
    actual = pipeline_prefill(planned, input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_qwen_manifest_feeds_planner_without_dataclass_assumptions(tmp_path) -> None:
    reference = Qwen3ForCausalLM(_config()).eval()
    model_dir = tmp_path / "qwen3"
    reference.save_pretrained(model_dir, safe_serialization=True)
    artifact_dir = tmp_path / "artifact"
    manifest = import_qwen_from_hf(
        str(model_dir),
        num_stages=2,
        out_dir=artifact_dir,
        dtype="float32",
        include_tokenizer=False,
    )

    spec = model_spec_from_manifest(artifact_dir)

    assert spec.num_layers == manifest.meta.num_layers
    assert spec.hidden_size == manifest.meta.hidden_size
    assert spec.attention_implementation == "sdpa"
    assert spec.param_count > 0


def test_qwen_import_requires_safetensors(tmp_path) -> None:
    reference = Qwen3ForCausalLM(_config())
    model_dir = tmp_path / "qwen3-bin"
    reference.save_pretrained(model_dir, safe_serialization=False)
    with pytest.raises(RuntimeError, match="safetensors"):
        import_qwen_from_hf(str(model_dir), 2, tmp_path / "artifact", include_tokenizer=False)


def test_qwen_pipeline_backward_and_update_match_official_model() -> None:
    torch.manual_seed(19)
    reference = Qwen3ForCausalLM(_config())
    workers = _workers_from_reference(reference)
    pipeline_optimizers = [
        torch.optim.SGD(worker._model.parameters(), lr=1e-2)
        for worker in workers
    ]

    input_ids = torch.tensor([[3, 5, 8, 13]])
    labels = torch.tensor([[5, 8, 13, -100]])
    valid = int((labels != -100).sum().item())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-2)
    reference_logits = reference(input_ids=input_ids, use_cache=False).logits
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / valid
    reference_loss.backward()
    reference_optimizer.step()

    pipeline_info = pipeline_train_step(
        workers,
        input_ids,
        labels,
        pipeline_optimizers,
        operation_id=91,
        attempt_id="qwen-train",
    )
    torch.testing.assert_close(
        torch.tensor(pipeline_info["loss"]),
        reference_loss.detach(),
        atol=1e-5,
        rtol=1e-4,
    )

    for stage_id, worker in enumerate(workers):
        for local_name, value in worker._model.state_dict().items():
            if local_name == "embed_tokens.weight":
                reference_name = "model.embed_tokens.weight"
            elif local_name == "norm.weight":
                reference_name = "model.norm.weight"
            elif local_name == "lm_head.weight":
                reference_name = "lm_head.weight"
            elif local_name.startswith("layers."):
                local_index, suffix = local_name.split(".", 2)[1:]
                global_index = int(local_index) + (0 if stage_id == 0 else 2)
                reference_name = f"model.layers.{global_index}.{suffix}"
            else:  # pragma: no cover - defensive for future Qwen state fields
                continue
            torch.testing.assert_close(
                value,
                reference.state_dict()[reference_name],
                atol=2e-5,
                rtol=2e-4,
            )


def test_qwen_single_stage_preserves_tied_embedding_parameter() -> None:
    cfg = _config()
    cfg.tie_word_embeddings = True
    stage = Qwen3Stage(
        cfg,
        0,
        cfg.num_hidden_layers,
        has_embedding=True,
        has_lm_head=True,
        dtype=torch.float32,
    )
    assert stage.embed_tokens.weight is stage.lm_head.weight
