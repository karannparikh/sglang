"""Fail-closed simulator activation during Python site initialization."""

import os
import sys
import traceback

_BOOTSTRAP_ENV = "SGLANG_SIMULATOR_BOOTSTRAP"


def activate_simulator_or_exit() -> None:
    if os.environ.get(_BOOTSTRAP_ENV) != "1":
        return

    try:
        if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
            import torch

            # Some model-specific import-time checks probe the target GPU even
            # though the simulator never executes a real model forward.
            torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)

        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        install_simulator_hooks()
    except BaseException:
        traceback.print_exc()
        sys.stderr.write(
            "SGLang Simulator activation failed; refusing to start without hooks.\n"
        )
        sys.stderr.flush()
        raise SystemExit(1)
