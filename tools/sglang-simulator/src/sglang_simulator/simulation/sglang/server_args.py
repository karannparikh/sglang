from sglang_simulator.compat import apply_simulator_server_args
from sglang_simulator.hook import BaseHook


class C_ServerArgsHook(BaseHook):
    HOOK_CLASS_NAME = "ServerArgs"
    HOOK_MODULE_NAME = "sglang.srt.server_args"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            original_post_init(self)
            apply_simulator_server_args(self)

        target.__post_init__ = wrapped_post_init
