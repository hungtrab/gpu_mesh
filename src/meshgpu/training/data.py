"""Minimal causal-language-model data pipeline for the portable trainer.

The runtime consumes tensors, not tokenizers or dataset objects. This module
keeps that boundary explicit: JSONL text is tokenized, documents are packed
into fixed-length streams, and labels are shifted by one token.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch

from meshgpu.inference.tokenizer import Tokenizer


def iter_jsonl_text(path: str | Path) -> Iterator[str]:
    """Yield text records from a JSONL file.

    Records may be JSON strings or objects containing a string ``text`` field.
    Malformed records fail with the line number instead of becoming empty data.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            if isinstance(record, str):
                text = record
            elif isinstance(record, dict) and isinstance(record.get("text"), str):
                text = record["text"]
            else:
                raise ValueError(
                    f"JSONL record at {source}:{line_number} must be a string "
                    "or an object with a string 'text' field"
                )
            if text:
                yield text


class CausalTextBatcher:
    """Pack text into fixed-shape ``(input_ids, labels)`` batches.

    Each sample is made from ``sequence_length + 1`` packed tokens. The first
    ``sequence_length`` become inputs and the one-token-shifted suffix becomes
    labels. If ``drop_last=False``, the final short sample is padded and its
    padded labels are ``ignore_index``.
    """

    def __init__(
        self,
        source: str | Path | Iterable[str],
        tokenizer: Tokenizer,
        *,
        sequence_length: int,
        batch_size: int,
        add_eos: bool = True,
        repeat: bool = True,
        drop_last: bool = True,
        ignore_index: int = -100,
        max_batches: int | None = None,
        skip_batches: int = 0,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive when provided")
        if isinstance(skip_batches, bool) or not isinstance(skip_batches, int):
            raise TypeError("skip_batches must be an integer")
        if skip_batches < 0:
            raise ValueError("skip_batches must be non-negative")
        if isinstance(source, (str, Path)):
            self._source_path: Path | None = Path(source)
            self._texts: list[str] | None = None
        else:
            self._source_path = None
            self._texts = list(source)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.add_eos = add_eos
        self.repeat = repeat
        self.drop_last = drop_last
        self.ignore_index = ignore_index
        self.max_batches = max_batches
        self.skip_batches = skip_batches

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        samples = self._iter_samples()
        batches_emitted = 0
        batches_to_skip = self.skip_batches
        while True:
            batch: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(self.batch_size):
                try:
                    batch.append(next(samples))
                except StopIteration:
                    break
            if not batch or (len(batch) < self.batch_size and self.drop_last):
                return
            inputs = _stack_padded([sample[0] for sample in batch], self._pad_id)
            labels = _stack_padded(
                [sample[1] for sample in batch],
                lambda: self.ignore_index,
            )
            if batches_to_skip:
                batches_to_skip -= 1
                continue
            yield inputs, labels
            batches_emitted += 1
            if self.max_batches is not None and batches_emitted >= self.max_batches:
                return

    def _iter_samples(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[int] = []
        yielded_any = False

        while True:
            saw_tokens = False
            for text in self._iter_texts():
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                if self.add_eos and self.tokenizer.eos_token_id is not None:
                    token_ids.append(self.tokenizer.eos_token_id)
                if not token_ids:
                    continue
                saw_tokens = True
                buffer.extend(token_ids)

                while len(buffer) >= self.sequence_length + 1:
                    chunk = buffer[: self.sequence_length + 1]
                    del buffer[: self.sequence_length + 1]
                    yielded_any = True
                    yield _make_sample(chunk)

            if not self.repeat or not saw_tokens:
                break
            # Preserve a short tail and let the next epoch complete it. This
            # is a packed stream, so no token is discarded at an epoch boundary.
            # A corpus shorter than one sample therefore becomes valid once
            # repeated epochs fill the buffer.

        if buffer and not self.drop_last and len(buffer) >= 2:
            actual_input_len = min(len(buffer) - 1, self.sequence_length)
            input_ids = buffer[:actual_input_len]
            labels = buffer[1 : actual_input_len + 1]
            input_ids.extend([self._pad_id()] * (self.sequence_length - len(input_ids)))
            labels.extend([self.ignore_index] * (self.sequence_length - len(labels)))
            yielded_any = True
            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )

        if not yielded_any:
            raise ValueError(
                "dataset produced no complete sequence; provide more text or lower "
                "sequence_length"
            )

    def _iter_texts(self) -> Iterator[str]:
        if self._source_path is not None:
            return iter_jsonl_text(self._source_path)
        return iter(self._texts or [])

    def _pad_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        return 0


def make_causal_batches(
    source: str | Path | Iterable[str],
    tokenizer: Tokenizer,
    *,
    sequence_length: int,
    batch_size: int,
    add_eos: bool = True,
    repeat: bool = True,
    drop_last: bool = True,
    ignore_index: int = -100,
    max_batches: int | None = None,
    skip_batches: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Return an iterator of packed next-token training batches.

    ``skip_batches`` is a deterministic resume cursor: skipped batches are
    consumed from the same packed stream before the first yielded batch.  It
    is deliberately expressed in *emitted batch* units rather than token
    units, so the cursor remains valid for a fixed dataset/batch configuration
    and cannot silently change meaning when padding is present.
    """
    return iter(
        CausalTextBatcher(
            source,
            tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            add_eos=add_eos,
            repeat=repeat,
            drop_last=drop_last,
            ignore_index=ignore_index,
            max_batches=max_batches,
            skip_batches=skip_batches,
        )
    )


def _make_sample(chunk: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(chunk) < 2:
        raise ValueError("a causal sample needs at least two tokens")
    return (
        torch.tensor(chunk[:-1], dtype=torch.long),
        torch.tensor(chunk[1:], dtype=torch.long),
    )


def _stack_padded(
    tensors: list[torch.Tensor],
    fill_value: int | Callable[[], int],
) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot stack an empty batch")
    width = max(int(tensor.numel()) for tensor in tensors)
    fill = int(fill_value() if callable(fill_value) else fill_value)
    result = torch.full((len(tensors), width), fill, dtype=torch.long)
    for row, tensor in enumerate(tensors):
        result[row, : tensor.numel()] = tensor
    return result
