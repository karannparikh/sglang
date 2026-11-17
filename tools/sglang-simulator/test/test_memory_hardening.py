from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
ASSETS = Path(__file__).parent / "assets"
_RESULT_PREFIX = "SGLANG_SIMULATOR_MEMORY_RESULT="
_MIB = 1024 * 1024
_RSS_DELTA_LIMIT = 10 * _MIB
_LIFECYCLE_RSS_BAND = 16 * _MIB
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _rss_bytes() -> int:
    import psutil

    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    return sum(child.memory_info().rss for child in processes if child.is_running())


def _write_sim_config(directory: Path) -> Path:
    replay_path = directory / "replay.json"
    replay_path.write_text(
        json.dumps({"[[1, 1024]]": 0.001, "[[1025, 0]]": 0.01}),
        encoding="utf-8",
    )
    config_path = directory / "sim_config.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {
                        "name": "a100_sxm",
                        "hbm_capacity_gb": 80,
                    },
                    "disk_read_bandwidth_gb": 8,
                    "disk_write_bandwidth_gb": 8,
                    "memory_read_bandwidth_gb": 64,
                    "memory_write_bandwidth_gb": 64,
                    "num_device_per_node": 8,
                },
                "predictor": {
                    "name": "replay",
                    "database_path": str(replay_path),
                    "miss_strategy": "knn",
                    "miss_knn_k": 1,
                },
                "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _make_runner(directory: Path, *, model_path: Path | None = None):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(directory))
    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    return SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(model_path or ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            max_total_tokens=4096,
            page_size=256,
            skip_tokenizer_init=True,
        )
    )


def _probe_virtual_kv(capacity: int) -> dict:
    import torch
    from sglang_simulator.simulation.sglang.mem_pool_device import (
        _install_meta_payload_method,
    )

    class LogicalKVPool:
        def __init__(self):
            self.device = "cpu"

        def create_buffers(self):
            self.payload = torch.empty(
                (capacity, 32, 128),
                dtype=torch.float16,
                device=self.device,
            )

    _install_meta_payload_method(LogicalKVPool, "create_buffers")
    pool = LogicalKVPool()
    pool.create_buffers()
    return {
        "is_meta": pool.payload.is_meta,
        "logical_bytes": pool.payload.numel() * pool.payload.element_size(),
        "rss_bytes": _rss_bytes(),
    }


def _probe_logical_transfer(logical_bytes: int) -> dict:
    from sglang.srt.disaggregation.simulator.conn import SimulatorTransferTask

    tasks = [
        SimulatorTransferTask(
            room=index,
            is_last_chunk=True,
            targets=[],
            target_aux_item_lens=[],
            logical_bytes=logical_bytes,
            modeled_latency_s=0.001,
            link_name=None,
            aux_index=None,
            send_record=None,
        )
        for index in range(50_000)
    ]
    return {
        "tasks": len(tasks),
        "logical_bytes": sum(task.logical_bytes for task in tasks),
        "rss_bytes": _rss_bytes(),
    }


async def _run_lifecycle_round(runner, round_index: int) -> None:
    for index in range(5):
        await runner.engine.async_generate(
            input_ids=[round_index, index, 1],
            sampling_params={"ignore_eos": True, "max_new_tokens": 4},
            rid=f"complete-{round_index}-{index}",
        )

    for index in range(5):
        rid = f"abort-{round_index}-{index}"
        task = asyncio.create_task(
            runner.engine.async_generate(
                input_ids=[round_index, index, 2],
                sampling_params={"ignore_eos": True, "max_new_tokens": 4096},
                rid=rid,
            )
        )
        await asyncio.sleep(0)
        runner.engine.tokenizer_manager.abort_request(rid)
        with suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(task, timeout=1)

    for index in range(5):
        with suppress(TimeoutError):
            await asyncio.wait_for(
                runner.engine.async_generate(
                    input_ids=[round_index, index, 3],
                    sampling_params={
                        "ignore_eos": True,
                        "max_new_tokens": 4096,
                    },
                    rid=f"timeout-{round_index}-{index}",
                ),
                timeout=1e-6,
            )


def _probe_lifecycle() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        runner = _make_runner(Path(tmp))
        rss_samples = []
        try:
            for round_index in range(4):
                runner.engine.loop.run_until_complete(
                    _run_lifecycle_round(runner, round_index)
                )
                gc.collect()
                rss_samples.append(_rss_bytes())
        finally:
            runner.shutdown()
    return {"rss_samples": rss_samples}


def _probe_weight_opens() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        audit_path = directory / "weight-opens.jsonl"
        model_path = directory / "model"
        shutil.copytree(ASSETS / "qwen3-8b", model_path)
        (model_path / "model.safetensors").write_bytes(
            b"simulator must not open this weight file"
        )
        active = False

        def audit_hook(event, args):
            nonlocal active
            if active or event != "open" or not args:
                return
            path = (
                os.fsdecode(os.fspath(args[0]))
                if isinstance(args[0], (str, bytes))
                else ""
            )
            if not str(path).lower().endswith(_WEIGHT_SUFFIXES) or not Path(
                path
            ).resolve().is_relative_to(model_path):
                return
            active = True
            try:
                with audit_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(str(path)) + "\n")
            finally:
                active = False

        sys.addaudithook(audit_hook)
        sitecustomize_dir = directory / "audit-site"
        sitecustomize_dir.mkdir()
        (sitecustomize_dir / "sitecustomize.py").write_text(
            """
import json
import os
import sys

_active = False
_audit_path = os.environ.get("SGLANG_SIMULATOR_WEIGHT_AUDIT_PATH")
_model_root = os.environ.get("SGLANG_SIMULATOR_MODEL_ROOT")
_weight_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

def _audit_hook(event, args):
    global _active
    if _active or _audit_path is None or event != "open" or not args:
        return
    path = (
        os.fsdecode(os.fspath(args[0]))
        if isinstance(args[0], (str, bytes))
        else ""
    )
    real_path = os.path.realpath(path)
    if (
        not path.lower().endswith(_weight_suffixes)
        or _model_root is None
        or not (
            real_path == _model_root
            or real_path.startswith(_model_root + os.path.sep)
        )
    ):
        return
    _active = True
    try:
        with open(_audit_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(path) + "\\n")
    finally:
        _active = False

sys.addaudithook(_audit_hook)
""".lstrip(),
            encoding="utf-8",
        )
        os.environ["SGLANG_SIMULATOR_WEIGHT_AUDIT_PATH"] = str(audit_path)
        os.environ["SGLANG_SIMULATOR_MODEL_ROOT"] = str(model_path.resolve())
        os.environ["PYTHONPATH"] = os.pathsep.join(
            (
                str(sitecustomize_dir),
                os.environ.get("PYTHONPATH", ""),
            )
        )
        runner = _make_runner(directory, model_path=model_path)
        runner.shutdown()
        paths = (
            [json.loads(line) for line in audit_path.read_text().splitlines()]
            if audit_path.exists()
            else []
        )
    return {"weight_paths": paths}


def _run_probe(name: str, value: int | None = None) -> dict:
    command = [sys.executable, __file__, name]
    if value is not None:
        command.append(str(value))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert result_lines, completed.stdout
    return json.loads(result_lines[-1].removeprefix(_RESULT_PREFIX))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS regression gate")
def test_ten_x_logical_kv_capacity_adds_less_than_ten_mib_idle_rss():
    baseline = _run_probe("virtual-kv", 100_000)
    scaled = _run_probe("virtual-kv", 1_000_000)

    assert baseline["is_meta"] and scaled["is_meta"]
    assert scaled["logical_bytes"] == 10 * baseline["logical_bytes"]
    assert abs(scaled["rss_bytes"] - baseline["rss_bytes"]) < _RSS_DELTA_LIMIT


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS regression gate")
def test_hundred_x_logical_transfer_does_not_scale_peak_rss():
    baseline = _run_probe("logical-transfer", 1_000_000)
    scaled = _run_probe("logical-transfer", 100_000_000)

    assert scaled["logical_bytes"] == 100 * baseline["logical_bytes"]
    assert abs(scaled["rss_bytes"] - baseline["rss_bytes"]) < _RSS_DELTA_LIMIT


@pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS regression gate")
def test_request_abort_timeout_cycles_return_to_bounded_rss_band():
    samples = _run_probe("lifecycle")["rss_samples"]

    assert len(samples) == 4
    assert max(samples[1:]) - min(samples[1:]) < _LIFECYCLE_RSS_BAND
    assert samples[-1] - samples[0] < _LIFECYCLE_RSS_BAND


def test_simulator_startup_does_not_open_model_weight_files():
    assert _run_probe("weight-opens")["weight_paths"] == []


if __name__ == "__main__":
    probe = sys.argv[1]
    value = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if probe == "virtual-kv":
        result = _probe_virtual_kv(value)
    elif probe == "logical-transfer":
        result = _probe_logical_transfer(value)
    elif probe == "lifecycle":
        result = _probe_lifecycle()
    elif probe == "weight-opens":
        result = _probe_weight_opens()
    else:
        raise ValueError(f"Unknown memory probe: {probe}")
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))
