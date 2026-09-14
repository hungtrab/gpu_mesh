"""Tests for JSONL packing and causal next-token labels."""

import pytest

from meshgpu.inference.tokenizer import Tokenizer
from meshgpu.training.data import CausalTextBatcher, iter_jsonl_text, make_causal_batches


class CharTokenizer:
    vocab_size = 256
    eos_token_id = 0
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids if token > 1)

    def save_pretrained(self, path):
        pass


def _tokenizer():
    return Tokenizer(CharTokenizer(), "char")


def test_jsonl_accepts_string_and_text_records(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('"one"\n{"text": "two"}\n\n')
    assert list(iter_jsonl_text(path)) == ["one", "two"]


def test_jsonl_reports_bad_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "ok"}\nnot-json\n')
    with pytest.raises(ValueError, match="data.jsonl:2"):
        list(iter_jsonl_text(path))


def test_labels_are_shifted_and_batches_have_expected_shape():
    batches = list(
        make_causal_batches(
            ["abcdefghij"],
            _tokenizer(),
            sequence_length=3,
            batch_size=2,
            add_eos=False,
            repeat=False,
        )
    )
    assert len(batches) == 1
    inputs, labels = batches[0]
    assert inputs.shape == (2, 3)
    assert labels.shape == (2, 3)
    assert inputs[0].tolist() == [ord("a"), ord("b"), ord("c")]
    assert labels[0].tolist() == [ord("b"), ord("c"), ord("d")]
    assert inputs[1].tolist() == [ord("e"), ord("f"), ord("g")]
    assert labels[1].tolist() == [ord("f"), ord("g"), ord("h")]


def test_final_partial_sample_masks_padding():
    batches = list(
        CausalTextBatcher(
            ["abcdef"],
            _tokenizer(),
            sequence_length=3,
            batch_size=1,
            add_eos=False,
            repeat=False,
            drop_last=False,
        )
    )
    assert len(batches) == 2
    assert batches[0][1].tolist() == [[ord("b"), ord("c"), ord("d")]]
    assert batches[1][0].tolist() == [[ord("e"), 0, 0]]
    assert batches[1][1].tolist() == [[ord("f"), -100, -100]]


def test_repeat_can_be_bounded():
    batches = list(
        make_causal_batches(
            ["abcdef"],
            _tokenizer(),
            sequence_length=2,
            batch_size=1,
            add_eos=False,
            repeat=True,
            max_batches=3,
        )
    )
    assert len(batches) == 3


def test_repeated_short_corpus_can_fill_a_complete_sequence():
    batches = make_causal_batches(
        ["ab"],
        _tokenizer(),
        sequence_length=4,
        batch_size=1,
        add_eos=False,
        repeat=True,
        drop_last=True,
        max_batches=1,
    )

    inputs, labels = next(batches)
    assert inputs.shape == (1, 4)
    assert labels.shape == (1, 4)


def test_skip_batches_resumes_the_same_packed_stream():
    all_batches = list(
        make_causal_batches(
            ["abcdefghijklmnop"],
            _tokenizer(),
            sequence_length=2,
            batch_size=1,
            add_eos=False,
            repeat=False,
        )
    )
    resumed = list(
        make_causal_batches(
            ["abcdefghijklmnop"],
            _tokenizer(),
            sequence_length=2,
            batch_size=1,
            add_eos=False,
            repeat=False,
            skip_batches=2,
        )
    )
    assert len(resumed) == len(all_batches) - 2
    for actual, expected in zip(resumed, all_batches[2:]):
        assert actual[0].tolist() == expected[0].tolist()
        assert actual[1].tolist() == expected[1].tolist()


def test_skip_batches_validation():
    with pytest.raises(ValueError, match="skip_batches"):
        CausalTextBatcher(["abc"], _tokenizer(), sequence_length=2, batch_size=1, skip_batches=-1)
    with pytest.raises(TypeError, match="skip_batches"):
        CausalTextBatcher(["abc"], _tokenizer(), sequence_length=2, batch_size=1, skip_batches=True)
