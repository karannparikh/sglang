import os
import platform
import sys
from array import array
from functools import lru_cache
from hashlib import sha256
from typing import Any, Optional


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_native_hash_module() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        abs_path = os.path.dirname(os.path.abspath(__file__))
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        return load_extension_with_recovery(
            name="hicache_hash_cpp",
            sources=[f"{abs_path}/hash_binding.cpp"],
            extra_cflags=extra_cflags,
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to load HiCache native hash extension") from exc


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("q", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("q", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("q", raw[:logical_len]), logical_len, 1, is_bigram


def _python_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int]
) -> str | list[str]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )
    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    def logical_unit(index: int) -> tuple[int, ...]:
        if is_bigram:
            return int(raw[index]), int(raw[index + 1])
        token = raw[index]
        if isinstance(token, tuple):
            return tuple(int(value) for value in token)
        return (int(token),)

    def hash_range(start: int, end: int, previous: Optional[bytes]) -> bytes:
        units = [logical_unit(index) for index in range(start, end)]
        values = [value for unit in units for value in unit]
        if any(value < 0 or value > (1 << 63) - 1 for value in values):
            raise OverflowError("token id must be a non-negative int64")
        wide = any(value > (1 << 32) - 1 for value in values)
        digest = sha256()
        if previous is not None:
            digest.update(previous)
        if wide:
            digest.update(b"\xff")
        width = 8 if wide else 4
        for value in values:
            digest.update(value.to_bytes(width, "little"))
        return digest.digest()

    if page_size is None:
        return hash_range(0, logical_len, prior_digest).hex()
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    hashes = []
    previous = prior_digest
    for start in range(0, logical_len, page_size):
        previous = hash_range(start, min(start + page_size, logical_len), previous)
        hashes.append(previous.hex())
    return hashes


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    if os.getenv("SGLANG_SIMULATOR_BOOTSTRAP") == "1":
        return _python_hash(token_ids, prior_digest, page_size)
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    return _load_native_hash_module().get_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )
