"""Activate SGLang Simulator in virtual environments before application imports."""

import importlib.machinery
import importlib.util
import os
import sys

from _sglang_simulator_startup import activate_simulator_or_exit

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_previous_sitecustomize() -> None:
    path = [entry for entry in sys.path if os.path.abspath(entry or ".") != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is None or spec.origin == __file__ or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


activate_simulator_or_exit()
_chain_previous_sitecustomize()
