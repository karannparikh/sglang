"""Keep unused Redis-backed multimodal caching out of simulator-only runtimes."""

import importlib.machinery
import sys
import types


class RedisError(Exception):
    pass


class RedisClusterException(Exception):
    pass


class _UnsupportedRedis:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "SGLang Simulator does not support Dynamo's shared Redis image cache."
        )

    @classmethod
    def from_url(cls, *_args, **_kwargs):
        return cls()


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<sglang-simulator-stub:{name}>"
    module.__package__ = name if package else name.rsplit(".", 1)[0]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader=None,
        is_package=package,
    )
    if package:
        module.__path__ = []
        module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    return module


def install_redis_stub() -> None:
    if "redis" in sys.modules:
        return

    package = _module("redis", package=True)
    asyncio_package = _module("redis.asyncio", package=True)
    cluster = _module("redis.asyncio.cluster")
    retry = _module("redis.asyncio.retry")
    backoff = _module("redis.backoff")
    exceptions = _module("redis.exceptions")

    cluster.RedisCluster = _UnsupportedRedis
    retry.Retry = _UnsupportedRedis
    backoff.NoBackoff = _UnsupportedRedis
    exceptions.RedisError = RedisError
    exceptions.RedisClusterException = RedisClusterException

    asyncio_package.cluster = cluster
    asyncio_package.retry = retry
    package.asyncio = asyncio_package
    package.backoff = backoff
    package.exceptions = exceptions
