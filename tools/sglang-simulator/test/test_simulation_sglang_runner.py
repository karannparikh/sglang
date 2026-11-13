from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from sglang_simulator.simulation.sglang.deterministic_sampling import (
    DeterministicRequestState,
    build_request_digest,
    sample_token_id,
)
from sglang_simulator.simulation.sglang.model_runner import (
    C_ForwardBatchHook,
    C_ModelRunnerHook,
    _build_request_state,
)

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def make_fixed_dataset(
    start_token: int,
    count: int,
    *,
    input_length: int = 1025,
    output_length: int = 1,
) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[start_token + i] * input_length,
                input_length=input_length,
                output_length=output_length,
                custom_params={"created_time": i / 10},
            )
            for i in range(count)
        ]
    )


def _write_sim_config(tmp_path: Path) -> Path:
    table_path = tmp_path / "replay.json"
    table_path.write_text(
        json.dumps({"[[1, 1024]]": 0.001, "[[1025, 0]]": 0.01}),
        encoding="utf-8",
    )
    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "replay",
            "database_path": str(table_path),
            "miss_strategy": "knn",
            "miss_knn_k": 1,
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def make_sglang_runner(tmp_path: Path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from sglang.srt.server_args import ServerArgs

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            max_total_tokens=4 * 1024,
            page_size=256,
            skip_tokenizer_init=True,
        )
    )
    runner.clear_hicache_storage()
    return runner


def _generate(
    runner,
    *,
    input_ids: list[int],
    rid: str,
    cache_salt: str | None = None,
    temperature: float = 1.0,
) -> dict:
    return runner.engine.loop.run_until_complete(
        runner.engine.async_generate(
            input_ids=input_ids,
            sampling_params={
                "ignore_eos": True,
                "max_new_tokens": 8,
                "temperature": temperature,
            },
            rid=rid,
            cache_salt=cache_salt,
        )
    )


def _make_deterministic_request(**sampling_overrides):
    sampling_params = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        sampling_seed=None,
        stop_token_ids=None,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
        logit_bias=None,
        watermark=None,
        beam_width=None,
        min_new_tokens=0,
    )
    for name, value in sampling_overrides.items():
        setattr(sampling_params, name, value)
    return SimpleNamespace(
        sampling_params=sampling_params,
        return_logprob=False,
        return_sampling_mask=False,
        grammar=None,
        custom_logit_processor=None,
        origin_input_ids=[11, 12, 13],
        output_ids=[],
        cache_salt=None,
        eos_token_ids=(),
    )


def test_deterministic_token_algorithm_known_vector():
    digest = build_request_digest(
        model_identity="model@revision",
        prompt_token_ids=[11, 12, 13],
        cache_salt="tenant-a",
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        sampling_seed=123,
    )
    states = [
        DeterministicRequestState(
            request_digest=digest,
            output_position=position,
            excluded_token_ids=(0, 1),
        )
        for position in range(4)
    ]

    assert [sample_token_id(state=state, vocab_size=32000) for state in states] == [
        20328,
        18258,
        22015,
        7731,
    ]


def test_deterministic_token_algorithm_skips_excluded_tokens():
    state = DeterministicRequestState(
        request_digest=b"request",
        output_position=0,
        excluded_token_ids=(0, 1, 3, 99),
    )

    assert sample_token_id(state=state, vocab_size=4) == 2
    with pytest.raises(ValueError, match="No token remains"):
        sample_token_id(
            state=state._replace(excluded_token_ids=(0, 1, 2, 3)),
            vocab_size=4,
        )


def test_request_digest_is_stable_across_decode_positions():
    req = _make_deterministic_request()
    initial = _build_request_state(model_identity="model@revision", req=req)

    req.output_ids = [20, 21]
    decoded = _build_request_state(model_identity="model@revision", req=req)
    salted = _build_request_state(
        model_identity="model@revision",
        req=SimpleNamespace(**{**req.__dict__, "cache_salt": "tenant-b"}),
    )
    other_sampling = _build_request_state(
        model_identity="model@revision",
        req=_make_deterministic_request(temperature=0.5),
    )

    assert initial.request_digest == decoded.request_digest
    assert decoded.output_position == 2
    assert initial.request_digest != salted.request_digest
    assert initial.request_digest != other_sampling.request_digest


@pytest.mark.parametrize(
    ("sampling_override", "feature_name"),
    [
        ({"json_schema": "{}"}, "structured output"),
        ({"min_new_tokens": 1}, "min_new_tokens"),
    ],
)
def test_deterministic_output_rejects_logits_dependent_features(
    sampling_override, feature_name
):
    req = _make_deterministic_request(**sampling_override)

    with pytest.raises(ValueError, match=feature_name):
        _build_request_state(model_identity="model@revision", req=req)


def test_forward_batch_hook_tracks_generation_position_and_stop_tokens():
    class MockForwardBatch:
        @classmethod
        def init_new(cls, batch, model_runner, **kwargs):
            return SimpleNamespace(batch_size=len(batch.reqs))

    C_ForwardBatchHook.hook(MockForwardBatch)
    req = _make_deterministic_request(stop_token_ids={3})
    req.output_ids = [20, 21]
    req.eos_token_ids = {2}
    batch = SimpleNamespace(reqs=[req])
    model_runner = SimpleNamespace(simulator_model_identity="model@revision")

    forward_batch = MockForwardBatch.init_new(batch, model_runner)
    state = forward_batch.simulator_sampling_states[0]

    assert state.output_position == 2
    assert state.excluded_token_ids == (2, 3)


def test_model_runner_uses_zero_width_logits():
    class FakeLogitsProcessorOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModelRunnerOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    logits_processor_module = ModuleType("sglang.srt.layers.logits_processor")
    logits_processor_module.LogitsProcessorOutput = FakeLogitsProcessorOutput
    model_runner_module = ModuleType("sglang.srt.model_executor.model_runner")
    model_runner_module.ModelRunnerOutput = FakeModelRunnerOutput

    class MockModelRunner:
        pass

    C_ModelRunnerHook.hook(MockModelRunner)
    runner = MockModelRunner()
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(vocab_size=32000)
    forward_batch = SimpleNamespace(batch_size=2)

    with patch.dict(
        sys.modules,
        {
            "sglang.srt.layers.logits_processor": logits_processor_module,
            "sglang.srt.model_executor.model_runner": model_runner_module,
        },
    ):
        output = runner.forward(forward_batch)

        forward_batch.simulator_sampling_states = [
            DeterministicRequestState(
                request_digest=b"request-a",
                output_position=0,
            ),
            DeterministicRequestState(
                request_digest=b"request-b",
                output_position=4,
            ),
        ]
        next_token_ids = runner.sample(output, forward_batch)

    assert output.logits_output.next_token_logits.shape == (2, 0)
    assert output.logits_output.next_token_logits.numel() == 0
    assert next_token_ids.shape == (2,)
    assert next_token_ids.dtype == torch.int64


def test_deterministic_output_tracks_request_identity(tmp_path):
    runner = make_sglang_runner(tmp_path)
    try:
        prompt = [100, 101, 102, 103]
        first = _generate(runner, input_ids=prompt, rid="cold")
        warm = _generate(runner, input_ids=prompt, rid="warm")
        other_prompt = _generate(
            runner,
            input_ids=[100, 101, 102, 104],
            rid="other-prompt",
        )
        other_salt = _generate(
            runner,
            input_ids=prompt,
            rid="other-salt",
            cache_salt="tenant-b",
        )
        other_sampling = _generate(
            runner,
            input_ids=prompt,
            rid="other-sampling",
            temperature=0.5,
        )
    finally:
        runner.shutdown()

    assert len(first["output_ids"]) == 8
    assert first["output_ids"] == warm["output_ids"]
    assert warm["meta_info"]["cached_tokens"] > 0
    assert first["output_ids"] != other_prompt["output_ids"]
    assert first["output_ids"] != other_salt["output_ids"]
    assert first["output_ids"] != other_sampling["output_ids"]


def test_benchmark_sglang_runs_paged_decode(tmp_path):
    runner = make_sglang_runner(tmp_path)
    dataset = make_fixed_dataset(
        1000,
        2,
        input_length=1024,
        output_length=2,
    )

    try:
        metrics = runner.benchmark(
            BenchmarkConfig(request_rate=10, ignore_request_timestamp=False),
            dataset=dataset,
        )
        request_stats = runner.get_request_stats()
    finally:
        with patch.object(atexit, "unregister", wraps=atexit.unregister) as unregister:
            runner.shutdown()
            runner.shutdown()

        unregister.assert_called_once_with(runner.engine.shutdown)

    assert metrics["completed"] == len(dataset)
    assert metrics["total_input"] == 2 * 1024
    assert metrics["total_output"] == 2 * 2
    assert metrics["mean_tpot_ms"] > 0
    assert all(
        idx == 0 or req["created_time"] > 0 for idx, req in enumerate(request_stats)
    )
