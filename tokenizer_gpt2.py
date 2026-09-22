"""GPT-2 BPE helpers aligned with ModelConfig.vocab_size (50,304)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


GPT2_VOCAB = 50_257
PADDED_VOCAB = 50_304
EOT_TOKEN = 50256  # <|endoftext|>


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode_text(text: str) -> list[int]:
    return get_encoding().encode_ordinary(text)


def decode_tokens(tokens: list[int]) -> str:
    return get_encoding().decode(tokens)
