import importlib.machinery
import sys
import types


class _UnavailableKernelOp:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(
            "SGLang Simulator does not execute compiled sgl_kernel operation "
            f"{self.name}."
        )

    def __getattr__(self, _name: str):
        return self


class _NoOpKernelOp(_UnavailableKernelOp):
    def __call__(self, *_args, **_kwargs):
        return None


class _UnavailableTorchOps:
    _NO_OPS = frozenset({"init_cpu_threads_env", "initialize"})

    def __init__(self):
        self._operations: dict[str, _UnavailableKernelOp] = {}

    def __getattr__(self, name: str) -> _UnavailableKernelOp:
        operation = self._operations.get(name)
        if operation is None:
            operation_class = (
                _NoOpKernelOp if name in self._NO_OPS else _UnavailableKernelOp
            )
            operation = operation_class(name)
            self._operations[name] = operation
        return operation


def _unavailable_kernel(*_args, **_kwargs):
    raise RuntimeError(
        "SGLang Simulator does not execute compiled sgl_kernel operations."
    )


def install_load_utils_stub() -> None:
    """Install import-safe kernel stubs before SGLang loads compiled extensions."""
    import torch

    if not isinstance(torch.ops.sgl_kernel, _UnavailableTorchOps):
        torch.ops.sgl_kernel = _UnavailableTorchOps()

    package_name = "sgl_kernel"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = "<sglang-simulator-kernel-stub>"
        package.__package__ = package_name
        package.__path__ = []
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        package.__spec__.submodule_search_locations = package.__path__
        package.__getattr__ = lambda _name: _unavailable_kernel
        sys.modules[package_name] = package

    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = package_name
        sys.modules[module_name] = module
        package.load_utils = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None
