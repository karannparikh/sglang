import sys
import types

import pytest
from sglang_simulator.simulation.sglang import sgl_kernel_hook


def test_install_load_utils_stub_adds_fail_closed_torch_ops(monkeypatch):
    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(sgl_kernel=object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delitem(sys.modules, "sgl_kernel", raising=False)
    monkeypatch.delitem(sys.modules, "sgl_kernel.load_utils", raising=False)

    sgl_kernel_hook.install_load_utils_stub()

    operation = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu
    assert operation.default is operation
    with pytest.raises(RuntimeError, match="fused_sigmoid_mul_cpu"):
        operation(None)

    assert torch.ops.sgl_kernel.init_cpu_threads_env([0, 1]) is None
    assert torch.ops.sgl_kernel.initialize(1, 0) is None
