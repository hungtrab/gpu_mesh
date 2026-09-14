"""Small, explicit tokenizer boundary for inference and training.

MeshGPU deliberately does not reimplement a subword tokenizer.  The model
adapter owns tensors and execution; this module owns the text/token boundary
and keeps the optional HuggingFace dependency out of the core runtime.

Remote model code is never enabled here.  A tokenizer may be loaded from a
local directory or from the HuggingFace Hub, but it must use the standard
``AutoTokenizer`` implementation supplied by ``transformers``.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class TokenizerError(RuntimeError):
    """Raised when a text/tokenizer operation cannot be completed safely."""


class TokenizerBackend(Protocol):
    eos_token_id: int | None
    bos_token_id: int | None
    pad_token_id: int | None
    vocab_size: int

    def encode(self, text: str, **kwargs: Any) -> Any: ...
    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...
    def save_pretrained(self, path: str | Path) -> Any: ...


class Tokenizer:
    """Validated wrapper around a HuggingFace-compatible tokenizer.

    The wrapper normalizes the two common HF return forms (``list[int]`` and
    ``BatchEncoding``) and exposes only operations MeshGPU needs.  It also
    provides cumulative decoding, which is useful for producing correct text
    deltas for byte-pair tokenizers.
    """

    def __init__(self, backend: TokenizerBackend, source: str | Path) -> None:
        self._backend = backend
        self.source = str(source)

    @property
    def backend(self) -> TokenizerBackend:
        return self._backend

    @property
    def vocab_size(self) -> int:
        value = getattr(self._backend, "vocab_size", None)
        if value is None:
            raise TokenizerError("tokenizer does not expose vocab_size")
        return int(value)

    @property
    def eos_token_id(self) -> int | None:
        return _optional_int(getattr(self._backend, "eos_token_id", None))

    @property
    def bos_token_id(self) -> int | None:
        return _optional_int(getattr(self._backend, "bos_token_id", None))

    @property
    def pad_token_id(self) -> int | None:
        return _optional_int(getattr(self._backend, "pad_token_id", None))

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text:
            return []
        kwargs: dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
        }
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be positive")
            kwargs["max_length"] = max_length
        encoded = self._backend.encode(text, **kwargs)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        if not isinstance(encoded, (list, tuple)):
            raise TokenizerError(
                f"tokenizer returned unsupported encode result: {type(encoded).__name__}"
            )
        token_ids = [int(token_id) for token_id in encoded]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
            raise TokenizerError("tokenizer returned an ID outside its vocabulary")
        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("token ID outside tokenizer vocabulary")
        return str(
            self._backend.decode(
                ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        )

    def delta_decode(
        self,
        token_ids: Sequence[int],
        previous_text: str = "",
        *,
        skip_special_tokens: bool = True,
    ) -> tuple[str, str]:
        """Return ``(new_text, delta)`` using cumulative decoding.

        Decoding each BPE token in isolation is incorrect for many tokenizers.
        Cumulative decoding makes the server's text stream stable; if a
        tokenizer changes an earlier rendering, the returned delta is the
        safest full suffix available from the two strings.
        """
        text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if text.startswith(previous_text):
            return text, text[len(previous_text):]
        # This can happen around special-token normalization.  Do not emit a
        # fabricated diff; return the complete current rendering so a client
        # can replace its accumulated text deterministically.
        return text, text

    def save_pretrained(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self._backend.save_pretrained(str(destination))
        return destination

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        local_files_only: bool = False,
    ) -> Tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TokenizerError(
                "HuggingFace tokenizer support requires the optional dependency; "
                "install with `pip install 'meshgpu[hf]'`"
            ) from exc

        try:
            backend = AutoTokenizer.from_pretrained(
                str(source),
                use_fast=True,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise TokenizerError(f"could not load tokenizer from {source!r}: {exc}") from exc
        return cls(backend, source)


def load_tokenizer(source: str | Path, *, local_files_only: bool = False) -> Tokenizer:
    """Load a tokenizer from a model ID or local tokenizer directory."""
    return Tokenizer.from_pretrained(source, local_files_only=local_files_only)


def load_tokenizer_from_artifact(
    artifact_dir: str | Path,
    tokenizer_path: str | None = None,
) -> Tokenizer:
    """Load the tokenizer referenced by an artifact manifest.

    ``tokenizer_path`` is treated as a relative path under ``artifact_dir``;
    absolute paths and path traversal are rejected to avoid loading an
    unrelated host directory from an untrusted manifest.
    """
    root = Path(artifact_dir).resolve()
    relative = tokenizer_path or "tokenizer"
    candidate = Path(relative)
    if ".." in candidate.parts:
        raise TokenizerError("artifact tokenizer_path escapes artifact directory")
    if not relative or candidate == Path(".") or candidate.is_absolute():
        raise TokenizerError("artifact tokenizer_path must be relative")
    tokenizer_dir = (root / candidate).resolve()
    if root not in tokenizer_dir.parents and tokenizer_dir != root:
        raise TokenizerError("artifact tokenizer_path escapes artifact directory")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer directory not found: {tokenizer_dir}")
    return load_tokenizer(tokenizer_dir, local_files_only=True)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
