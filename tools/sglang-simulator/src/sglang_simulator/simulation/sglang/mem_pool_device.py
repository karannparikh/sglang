from sglang_simulator.hook import BaseHook


def _install_meta_payload_method(target, method_name: str) -> None:
    original_method = getattr(target, method_name)

    def wrapped_method(self, *args, **kwargs):
        logical_device = self.device
        self.device = "meta"
        try:
            return original_method(self, *args, **kwargs)
        finally:
            self.device = logical_device

    setattr(target, method_name, wrapped_method)


class C_MHATokenToKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"

    @classmethod
    def hook(cls, target):
        _install_meta_payload_method(target, "_create_buffers_normal")


class C_MLATokenToKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "MLATokenToKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"

    @classmethod
    def hook(cls, target):
        _install_meta_payload_method(target, "_create_buffers")
