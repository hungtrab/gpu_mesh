"""Tests for the optional tokenizer boundary without HuggingFace downloads."""
from pathlib import Path

import pytest

from meshgpu.inference.tokenizer import Tokenizer, TokenizerError, load_tokenizer_from_artifact


class FakeTokenizer:
    vocab_size = 128
    eos_token_id = 0
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, **kwargs):
        ids = [ord(char) % 100 + 2 for char in text]
        if kwargs.get("add_special_tokens"):
            return [self.bos_token_id, *ids]
        return ids

    def decode(self, token_ids, **kwargs):
        ids = [
            token
            for token in token_ids
            if not (kwargs.get("skip_special_tokens") and token < 2)
        ]
        return "".join(chr(token - 2) for token in ids)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "tokenizer.json").write_text("fake")


def test_encode_normalizes_special_tokens():
    tokenizer = Tokenizer(FakeTokenizer(), "fake")
    assert tokenizer.encode("ab") == [1, 99, 100]
    assert tokenizer.encode("ab", add_special_tokens=False) == [99, 100]


def test_decode_and_delta_are_cumulative():
    tokenizer = Tokenizer(FakeTokenizer(), "fake")
    assert tokenizer.decode([1, 99, 100]) == "ab"
    assert tokenizer.delta_decode([99], "") == ("a", "a")
    assert tokenizer.delta_decode([99, 100], "a") == ("ab", "b")


def test_out_of_range_ids_are_rejected():
    class BadTokenizer(FakeTokenizer):
        def encode(self, text, **kwargs):
            return [self.vocab_size]

    tokenizer = Tokenizer(BadTokenizer(), "fake")
    with pytest.raises(TokenizerError):
        tokenizer.encode("a")

def test_decode_rejects_out_of_range_id():
    tokenizer = Tokenizer(FakeTokenizer(), "fake")
    with pytest.raises(ValueError):
        tokenizer.decode([128])


def test_save_pretrained_creates_directory(tmp_path):
    tokenizer = Tokenizer(FakeTokenizer(), "fake")
    destination = tokenizer.save_pretrained(tmp_path / "tokenizer")
    assert destination == tmp_path / "tokenizer"
    assert (destination / "tokenizer.json").exists()


def test_artifact_tokenizer_path_traversal_rejected(tmp_path):
    with pytest.raises(TokenizerError, match="escapes"):
        load_tokenizer_from_artifact(tmp_path, "../outside")
