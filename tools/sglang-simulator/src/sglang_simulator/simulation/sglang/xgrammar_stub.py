"""Keep unused structured-generation imports out of simulator-only runtimes."""

import importlib.machinery
import sys
import types


class _XGrammarValue:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema

        return core_schema.any_schema()

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        for name, value in kwargs.items():
            setattr(self, name, value)


class _UnsupportedXGrammar(_XGrammarValue):
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "SGLang Simulator does not support grammar-constrained generation."
        )


def _placeholder(module: types.ModuleType, name: str):
    base = (
        _UnsupportedXGrammar
        if name in {"GrammarCompiler", "GrammarMatcher", "TokenizerInfo"}
        else _XGrammarValue
    )
    value = type(name, (base,), {})
    setattr(module, name, value)
    return value


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
    module.__getattr__ = lambda attr: _placeholder(module, attr)
    sys.modules[name] = module
    return module


def install_xgrammar_stub() -> None:
    if "xgrammar" in sys.modules:
        return

    package = _module("xgrammar", package=True)
    structural_tag = _module("xgrammar.structural_tag")
    package.structural_tag = structural_tag
