import sys

import pytest
from sglang_simulator.simulation.sglang.redis_stub import install_redis_stub


def test_install_redis_stub_supports_imports_and_rejects_use(monkeypatch):
    for name in (
        "redis",
        "redis.asyncio",
        "redis.asyncio.cluster",
        "redis.asyncio.retry",
        "redis.backoff",
        "redis.exceptions",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    install_redis_stub()

    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import RedisClusterException, RedisError

    assert issubclass(RedisClusterException, Exception)
    assert issubclass(RedisError, Exception)
    with pytest.raises(RuntimeError, match="shared Redis image cache"):
        RedisCluster.from_url("redis://unused")
