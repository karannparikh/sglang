import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang_simulator.hardware_policy import build_startup_summary
from sglang_simulator.profiles import (
    H200_LAB_MIXED_TOPOLOGY,
    H200_LAB_PROFILE,
    H200_LAB_PROFILE_NAME,
    H200_LAB_TWO_ZONES_TOPOLOGY,
    SIMULATOR_TOPOLOGY_PRESETS,
    SimulatorProfileError,
    apply_simulator_profile,
)
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.time_predictor import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.h200_lab import (
    H200_LAB_DECODE_ENGINE_MS,
    H200_LAB_PREFILL_ENGINE_MS,
    H200LabTimePredictor,
    _sample_profile_ms,
)

REPO_ROOT = Path(__file__).parents[3]


def _simulator_topology_class():
    path = REPO_ROOT / "python/sglang/srt/disaggregation/simulator_topology.py"
    spec = importlib.util.spec_from_file_location("simulator_topology_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimulatorTopology


def _write_config(
    tmp_path: Path,
    monkeypatch,
    **overrides,
) -> dict:
    config = {"profile": H200_LAB_PROFILE_NAME, **overrides}
    path = tmp_path / "simulator.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(path))
    return config


def _profile_target(**overrides) -> dict:
    target = {
        "chunked_prefill_size": None,
        "context_length": None,
        "disaggregation_mode": "prefill",
        "disaggregation_simulator_topology": None,
        "kv_cache_dtype": "auto",
        "max_prefill_tokens": 16_384,
        "max_running_requests": None,
        "max_total_tokens": None,
        "page_size": None,
        "random_seed": None,
        "tp_size": 1,
    }
    target.update(overrides)
    return target


def _predictor(*, role: str, seed: int = 17, logical_rank: int = 0):
    return H200LabTimePredictor(
        None,
        hw=None,
        config=SchedulerConfig(),
        role=role,
        seed=seed,
        logical_rank=logical_rank,
    )


def test_h200_profile_pins_external_capacity_and_timing_tables():
    assert dict(H200_LAB_PROFILE.server_arg_defaults) == {
        "tp_size": 8,
        "page_size": 256,
        "max_total_tokens": 3_970_304,
        "context_length": 1_048_576,
        "max_running_requests": 32,
        "max_prefill_tokens": 8_192,
        "chunked_prefill_size": 8_192,
        "kv_cache_dtype": "fp8_e4m3",
        "random_seed": 0,
    }
    assert H200_LAB_DECODE_ENGINE_MS == (
        (1, (3.6, 6.2, 7.0, 8.4)),
        (2, (4.1, 7.2, 7.8, 8.7)),
        (4, (6.2, 8.0, 8.5, 9.7)),
        (8, (7.5, 8.8, 9.3, 11.7)),
        (16, (8.6, 10.0, 10.4, 12.0)),
        (24, (9.8, 11.0, 11.4, 12.9)),
        (32, (10.4, 12.0, 12.7, 14.8)),
    )
    assert H200_LAB_PREFILL_ENGINE_MS == (
        (8_192, (130.0, 1_000.0, 2_500.0, 8_500.0)),
        (24_576, (260.0, 2_000.0, 5_000.0, 17_000.0)),
        (49_152, (1_050.0, 3_000.0, 5_400.0, 16_000.0)),
        (98_304, (2_450.0, 5_300.0, 6_800.0, 21_000.0)),
        (196_608, (6_500.0, 14_800.0, 22_000.0, 57_000.0)),
        (393_216, (19_000.0, 46_000.0, 73_000.0, 124_000.0)),
    )


def test_profile_defaults_and_explicit_override_precedence(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch)
    target = _profile_target(max_running_requests=16)

    apply_simulator_profile(
        target,
        explicit_fields=frozenset({"max_running_requests"}),
    )

    assert target["max_running_requests"] == 16
    assert target["max_total_tokens"] == 3_970_304
    assert target["tp_size"] == 8
    assert target["random_seed"] == 0


def test_profile_override_conflict_is_rejected(tmp_path, monkeypatch):
    _write_config(
        tmp_path,
        monkeypatch,
        profile_overrides={"max_running_requests": 24},
    )

    with pytest.raises(SimulatorProfileError, match="Ambiguous simulator profile"):
        apply_simulator_profile(
            _profile_target(max_running_requests=16),
            explicit_fields=frozenset({"max_running_requests"}),
        )


def test_profile_override_can_restore_a_stock_default(tmp_path, monkeypatch):
    _write_config(
        tmp_path,
        monkeypatch,
        profile_overrides={"tp_size": 1},
    )
    target = _profile_target()

    apply_simulator_profile(target, explicit_fields=frozenset())

    assert target["tp_size"] == 1


@pytest.mark.parametrize("role", ["null", "encode"])
def test_profile_rejects_unmeasured_worker_roles(tmp_path, monkeypatch, role):
    _write_config(tmp_path, monkeypatch)

    with pytest.raises(SimulatorProfileError, match="supports only disaggregated"):
        apply_simulator_profile(
            _profile_target(disaggregation_mode=role),
            explicit_fields=frozenset({"disaggregation_mode"}),
        )


def test_profile_rejects_a_second_predictor(tmp_path, monkeypatch):
    _write_config(
        tmp_path,
        monkeypatch,
        predictor={"name": "replay"},
    )

    with pytest.raises(SimulatorProfileError, match="owns its timing predictor"):
        apply_simulator_profile(
            _profile_target(),
            explicit_fields=frozenset(),
        )


def test_profile_selection_constructs_the_sampled_predictor(monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_LOGICAL_RANK", "3")
    ConfigManager._raw_config = {"profile": H200_LAB_PROFILE_NAME}
    try:
        predictor = ConfigManager.get_inference_time_predictor(
            None,
            None,
            SchedulerConfig(random_seed=11),
            role="prefill",
        )
    finally:
        ConfigManager.reset_config_cache()

    assert isinstance(predictor, H200LabTimePredictor)
    assert predictor.seed == 11
    assert predictor.logical_rank == 3


def test_sampling_is_reproducible_and_rank_local():
    batch = ScheduleBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=128) for _ in range(8)]
    )
    rank_zero = _predictor(role="decode", logical_rank=0)
    rank_zero_again = _predictor(role="decode", logical_rank=0)
    rank_one = _predictor(role="decode", logical_rank=1)

    zero_samples = [rank_zero.predict_infer_time(batch) for _ in range(4)]

    assert zero_samples == [rank_zero_again.predict_infer_time(batch) for _ in range(4)]
    assert zero_samples != [rank_one.predict_infer_time(batch) for _ in range(4)]


def test_quantile_and_workload_interpolation_and_clamping():
    assert _sample_profile_ms(
        H200_LAB_DECODE_ENGINE_MS,
        workload=3,
        percentile=0.50,
    ) == pytest.approx(5.15)
    assert _sample_profile_ms(
        H200_LAB_DECODE_ENGINE_MS,
        workload=64,
        percentile=0.90,
    ) == pytest.approx(12.0)
    assert _sample_profile_ms(
        H200_LAB_DECODE_ENGINE_MS,
        workload=1,
        percentile=0.0,
    ) == pytest.approx(0.35)
    assert _sample_profile_ms(
        H200_LAB_DECODE_ENGINE_MS,
        workload=1,
        percentile=1.0,
    ) == pytest.approx(8.75)


def test_sub_8k_prefill_prorates_work_and_zero_cache_hit_costs_zero():
    predictor = _predictor(role="prefill")
    predictor._rng = SimpleNamespace(random=lambda: 0.50)

    half_anchor = ScheduleBatch(
        reqs=[ScheduleRequest(extend_length=4_096, past_kv_length=100_000)]
    )
    cache_hit = ScheduleBatch(
        reqs=[ScheduleRequest(extend_length=0, past_kv_length=100_000)]
    )

    assert predictor.predict_infer_time(half_anchor) == pytest.approx(0.067)
    assert predictor.predict_infer_time(cache_hit) == 0.0


def test_decode_profile_rejects_prefill_work():
    predictor = _predictor(role="decode")
    with pytest.raises(ValueError, match="batch containing prefill work"):
        predictor.predict_infer_time(
            ScheduleBatch(reqs=[ScheduleRequest(extend_length=2, past_kv_length=100)])
        )


def test_capacity_defaults_resolve_to_reference_block_count():
    defaults = dict(H200_LAB_PROFILE.server_arg_defaults)
    assert defaults["max_total_tokens"] // defaults["page_size"] == 15_509
    assert defaults["max_total_tokens"] % defaults["page_size"] == 0


def test_topology_presets_resolve_reference_links():
    simulator_topology = _simulator_topology_class()
    two_zones = simulator_topology.from_json(
        SIMULATOR_TOPOLOGY_PRESETS[H200_LAB_TWO_ZONES_TOPOLOGY].topology_json
    )
    mixed = simulator_topology.from_json(
        SIMULATOR_TOPOLOGY_PRESETS[H200_LAB_MIXED_TOPOLOGY].topology_json
    )

    assert (
        two_zones.link_between(
            ("zone-a", "rack-1", "node-1"),
            ("zone-a", "rack-1", "node-1"),
        ).bandwidth_gb_per_s
        == 450
    )
    assert (
        two_zones.link_between(
            ("zone-a", "rack-1", "node-1"),
            ("zone-b", "rack-1", "node-1"),
        ).fixed_latency_ms
        == 5
    )
    assert (
        mixed.link_between(
            ("rack-1", "node-1"),
            ("rack-2", "node-1"),
        ).bandwidth_gb_per_s
        == 12.5
    )
    assert json.loads(
        SIMULATOR_TOPOLOGY_PRESETS[H200_LAB_TWO_ZONES_TOPOLOGY].topology_json
    )["links"] == {
        "default": {
            "bandwidth_gb_per_s": 1.25,
            "fixed_latency_ms": 5.0,
        },
        "node": {
            "bandwidth_gb_per_s": 450.0,
            "fixed_latency_ms": 0.01,
        },
        "rack": {
            "bandwidth_gb_per_s": 50.0,
            "fixed_latency_ms": 0.05,
        },
        "zone": {
            "bandwidth_gb_per_s": 12.5,
            "fixed_latency_ms": 0.5,
        },
    }


def test_topology_preset_rejects_inline_topology(tmp_path, monkeypatch):
    _write_config(
        tmp_path,
        monkeypatch,
        topology_preset=H200_LAB_TWO_ZONES_TOPOLOGY,
    )

    with pytest.raises(SimulatorProfileError, match="cannot be combined"):
        apply_simulator_profile(
            _profile_target(disaggregation_simulator_topology="{}"),
            explicit_fields=frozenset({"disaggregation_simulator_topology"}),
        )


def test_startup_summary_reports_profile_seed_capacity_and_topology():
    scheduler_config = SchedulerConfig(
        tp_size=8,
        context_length=1_048_576,
        max_running_requests=32,
        max_prefill_tokens=8_192,
        chunked_prefill_size=8_192,
        random_seed=23,
        kv_cache_dtype_name="fp8_e4m3",
    )
    topology = SIMULATOR_TOPOLOGY_PRESETS[H200_LAB_TWO_ZONES_TOPOLOGY].topology_json

    summary = build_startup_summary(
        predictor_name="h200-lab-sampled-v1",
        predictor_mode=H200_LAB_PROFILE_NAME,
        simulation_mode="OFFLINE",
        profile_name=H200_LAB_PROFILE_NAME,
        random_seed=23,
        logical_rank=4,
        scheduler_config=scheduler_config,
        max_total_tokens=3_970_304,
        page_size=256,
        hicache_ratio=None,
        disaggregation_backend="simulator",
        disaggregation_bandwidth_gb_per_s=64,
        disaggregation_fixed_latency_ms=1,
        disaggregation_topology=topology,
        disaggregation_location="zone-a/rack-1/node-1",
        topology_preset=H200_LAB_TWO_ZONES_TOPOLOGY,
        deterministic_sampling_version="sglang-simulator-token-v1",
    )
    parsed = json.loads(
        summary.removeprefix("SGLang Simulator effective configuration: ")
    )

    assert parsed["compute"] == {
        "logical_rank": 4,
        "predictor": "h200-lab-sampled-v1",
        "predictor_mode": H200_LAB_PROFILE_NAME,
        "profile": H200_LAB_PROFILE_NAME,
        "random_seed": 23,
        "simulation_mode": "OFFLINE",
    }
    assert parsed["logical_kv"]["max_total_tokens"] == 3_970_304
    assert parsed["logical_kv"]["kv_cache_dtype"] == "fp8_e4m3"
    assert parsed["pd"]["model"]["preset"] == H200_LAB_TWO_ZONES_TOPOLOGY
    assert parsed["pd"]["model"]["links"]["default"] == {
        "bandwidth_gb_per_s": 1.25,
        "fixed_latency_ms": 5.0,
    }
