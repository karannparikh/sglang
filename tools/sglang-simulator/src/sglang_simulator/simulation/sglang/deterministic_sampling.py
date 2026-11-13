from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from typing import NamedTuple

DETERMINISTIC_TOKEN_ALGORITHM = "sglang-simulator-token-v1"


class DeterministicRequestState(NamedTuple):
    request_digest: bytes
    output_position: int
    excluded_token_ids: tuple[int, ...] = ()


def _update_bytes(hasher, value: bytes) -> None:
    hasher.update(struct.pack("<Q", len(value)))
    hasher.update(value)


def _update_text(hasher, value: str | None) -> None:
    _update_bytes(hasher, b"" if value is None else value.encode("utf-8"))


def build_request_digest(
    *,
    model_identity: str,
    prompt_token_ids: Iterable[int],
    cache_salt: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
    sampling_seed: int | None,
) -> bytes:
    hasher = hashlib.sha256()
    _update_text(hasher, DETERMINISTIC_TOKEN_ALGORITHM)
    _update_text(hasher, model_identity)
    _update_text(hasher, cache_salt)

    prompt_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
    hasher.update(struct.pack("<Q", len(prompt_token_ids)))
    for token_id in prompt_token_ids:
        hasher.update(struct.pack("<q", token_id))

    for value in (
        temperature,
        top_p,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
    ):
        hasher.update(struct.pack("<d", float(value)))
    hasher.update(struct.pack("<q", int(top_k)))
    if sampling_seed is None:
        hasher.update(b"\x00")
    else:
        hasher.update(b"\x01")
        hasher.update(struct.pack("<q", int(sampling_seed)))
    return hasher.digest()


def sample_token_id(
    *,
    state: DeterministicRequestState,
    vocab_size: int,
) -> int:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
    if state.output_position < 0:
        raise ValueError(
            f"output_position must be non-negative, got {state.output_position}."
        )

    excluded = {
        token_id for token_id in state.excluded_token_ids if 0 <= token_id < vocab_size
    }
    if len(excluded) >= vocab_size:
        raise ValueError("No token remains after applying stop-token exclusions.")

    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(state.request_digest)
    hasher.update(struct.pack("<Q", state.output_position))
    token_id = int.from_bytes(hasher.digest(), "little") % vocab_size
    while token_id in excluded:
        token_id = (token_id + 1) % vocab_size
    return token_id
