import hashlib
import importlib.util
from array import array
from pathlib import Path

NATIVE_HASH_PATH = (
    Path(__file__).parents[3] / "python/sglang/srt/mem_cache/cpp_utils/native_hash.py"
)
SPEC = importlib.util.spec_from_file_location("simulator_native_hash", NATIVE_HASH_PATH)
assert SPEC is not None and SPEC.loader is not None
NATIVE_HASH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE_HASH)
get_native_hash = NATIVE_HASH.get_native_hash


class HashKey:
    def __init__(self, token_ids, *, is_bigram=False):
        self.token_ids = token_ids
        self.is_bigram = is_bigram

    def __len__(self):
        return len(self.token_ids) - 1 if self.is_bigram else len(self.token_ids)

    def raw_token_ids(self):
        return self.token_ids


def _hash(values, prior=None):
    digest = hashlib.sha256()
    if prior is not None:
        digest.update(prior)
    for value in values:
        digest.update(value.to_bytes(4, "little"))
    return digest.digest()


def test_simulator_hash_fallback_preserves_page_chaining(monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    prior = _hash([7, 8, 9])
    expected_first = _hash([1, 2], prior)
    expected_second = _hash([3, 4], expected_first)

    assert get_native_hash([1, 2, 3, 4], prior, 2) == [
        expected_first.hex(),
        expected_second.hex(),
    ]
    assert get_native_hash([1, 2, 3, 4], prior) == _hash([1, 2, 3, 4], prior).hex()


def test_simulator_hash_fallback_supports_bigrams_and_wide_tokens(monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    bigram = HashKey(array("q", [1, 2, 3]), is_bigram=True)

    assert get_native_hash(bigram, None) == get_native_hash([(1, 2), (2, 3)], None)

    wide = (1 << 32) + 1
    digest = hashlib.sha256(
        b"\xff" + wide.to_bytes(8, "little") + (2).to_bytes(8, "little")
    ).hexdigest()
    assert get_native_hash([wide, 2], None) == digest
