"""Keep unused audio-transcription imports out of simulator-only runtimes."""

import importlib.machinery
import sys
import types


def _unsupported_audio(*_args, **_kwargs):
    raise RuntimeError("SGLang Simulator does not support audio transcription.")


def install_soundfile_stub() -> None:
    if "soundfile" in sys.modules:
        return

    module = types.ModuleType("soundfile")
    module.__file__ = "<sglang-simulator-stub:soundfile>"
    module.__package__ = ""
    module.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)
    module.__getattr__ = lambda _name: _unsupported_audio
    sys.modules["soundfile"] = module
