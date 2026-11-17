"""Provide the unquantized simulator without importing optional quantization backends."""

import importlib.machinery
import sys
import types
from pathlib import Path

_PACKAGE = "sglang.srt.layers.quantization"


class _UnsupportedQuantization:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "SGLang Simulator does not support quantized model execution."
        )


def _placeholder_type(module: types.ModuleType, name: str):
    value = type(name, (_UnsupportedQuantization,), {})
    setattr(module, name, value)
    return value


def _new_module(name: str, *, package: bool = False) -> types.ModuleType:
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


def install_unquantized_registry() -> None:
    if _PACKAGE in sys.modules:
        return

    import sglang

    package_path = Path(sglang.__file__).parent / "srt" / "layers" / "quantization"
    module = _new_module(_PACKAGE, package=True)
    module.__file__ = str(package_path / "__init__.py")
    module.__path__ = [str(package_path)]
    module.__spec__.submodule_search_locations = module.__path__

    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    def get_quantization_config(quantization: str):
        raise ValueError(
            "SGLang Simulator supports only unquantized dummy-model execution; "
            f"quantization method {quantization!r} is unavailable."
        )

    module.QuantizationConfig = QuantizationConfig
    module.QUANTIZATION_METHODS = {}
    module.get_quantization_config = get_quantization_config

    compressed_package_name = f"{_PACKAGE}.compressed_tensors"
    compressed_package = _new_module(compressed_package_name, package=True)
    module.compressed_tensors = compressed_package

    compressed_module_name = f"{compressed_package_name}.compressed_tensors"
    compressed_module = _new_module(compressed_module_name)
    compressed_module.__getattr__ = lambda name: _placeholder_type(
        compressed_module,
        name,
    )
    compressed_module.CompressedTensorsFusedMoEMethod = _placeholder_type(
        compressed_module,
        "CompressedTensorsFusedMoEMethod",
    )
    compressed_package.compressed_tensors = compressed_module

    schemes_name = f"{compressed_package_name}.schemes"
    schemes = _new_module(schemes_name)
    schemes.__getattr__ = lambda name: (
        (4, 8) if name == "WNA16_SUPPORTED_BITS" else _placeholder_type(schemes, name)
    )
    schemes.CompressedTensorsMxInt4MoE = _placeholder_type(
        schemes,
        "CompressedTensorsMxInt4MoE",
    )
    compressed_package.schemes = schemes
