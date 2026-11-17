import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sglang_simulator.hardware_policy as hardware_policy
from sglang_simulator.compat import (
    SGLangCompatibilityError,
    apply_simulator_server_args,
)
from sglang_simulator.hardware_policy import (
    PHYSICAL_SERVER_ARG_OVERRIDES,
    SIMULATOR_ARG_POLICY,
    SimulatorArgPolicy,
    SimulatorPolicyError,
    build_startup_summary,
    evaluate_simulator_policy,
    hardware_sensitive_server_arg_fields,
    recorded_policy_outcomes,
    registry_validation_errors,
    requested_server_arg_values,
)
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.sglang.server_args import C_ServerArgsHook
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.simulation.utils import calc_iteration_metrics

REPO_ROOT = Path(__file__).parents[3]


def test_hardware_policy_registry_tracks_current_server_arg_groups():
    assert set(SIMULATOR_ARG_POLICY) == hardware_sensitive_server_arg_fields(REPO_ROOT)
    assert registry_validation_errors(REPO_ROOT) == []


def test_registry_guard_rejects_invalid_and_stale_metadata(monkeypatch):
    valid = SIMULATOR_ARG_POLICY["device"]

    with monkeypatch.context() as context:
        context.setitem(
            hardware_policy._registry,
            "device",
            SimulatorArgPolicy(
                category="invalid",
                subsystem=valid.subsystem,
                rationale=valid.rationale,
            ),
        )
        assert "device: invalid category 'invalid'" in registry_validation_errors(
            REPO_ROOT
        )

    with monkeypatch.context() as context:
        context.setitem(
            hardware_policy._registry,
            "device",
            SimulatorArgPolicy(
                category="modeled",
                subsystem="invalid",
                rationale=valid.rationale,
            ),
        )
        errors = registry_validation_errors(REPO_ROOT)
        assert "device: invalid subsystem 'invalid'" in errors
        assert "device: modeled field has no consumer assertion" in errors

    with monkeypatch.context() as context:
        context.setitem(
            hardware_policy._registry,
            "future_hardware_knob",
            valid,
        )
        assert any(
            error.startswith("Stale simulator classifications:")
            for error in registry_validation_errors(REPO_ROOT)
        )

    with monkeypatch.context() as context:
        context.setattr(
            hardware_policy,
            "_registered_field_sequence",
            [*hardware_policy._registered_field_sequence, "device"],
        )
        assert (
            "Duplicate simulator classifications are present"
            in registry_validation_errors(REPO_ROOT)
        )


def test_warn_no_op_only_surfaces_explicit_or_non_default_values():
    defaults = {"attention_backend": None}

    untouched = evaluate_simulator_policy(
        {"attention_backend": None},
        defaults=defaults,
    )
    requested = evaluate_simulator_policy(
        {"attention_backend": "flashinfer"},
        defaults=defaults,
    )

    assert untouched.warning_fields == ()
    assert requested.warning_fields == ("attention_backend",)
    assert "--attention-backend" in requested.warning_messages[0]


def test_rejected_hardware_value_names_the_argument_and_reason():
    with pytest.raises(
        SimulatorPolicyError,
        match=r"--dcp-size=2.+physical distributed groups",
    ):
        evaluate_simulator_policy(
            {"dcp_size": 2},
            defaults={"dcp_size": 1},
        )


def test_policy_preserves_logical_topology_before_physical_override(monkeypatch):
    target = {
        "tp_size": 8,
        "ep_size": 4,
        "dp_size": 2,
        "pp_size": 2,
        "attn_cp_size": 1,
    }
    defaults = {name: 1 for name in target}
    monkeypatch.delenv("SGLANG_SIMULATOR_REQUESTED_SERVER_ARGS", raising=False)
    monkeypatch.delenv("SGLANG_SIMULATOR_POLICY_OUTCOMES", raising=False)

    with patch(
        "sglang_simulator.hardware_policy._server_arg_defaults",
        return_value=defaults,
    ):
        apply_simulator_server_args(target)

    assert requested_server_arg_values() == {
        "attn_cp_size": 1,
        "dp_size": 2,
        "ep_size": 4,
        "pp_size": 2,
        "tp_size": 8,
    }
    assert {
        name: target[name]
        for name in ("tp_size", "ep_size", "dp_size", "pp_size", "attn_cp_size")
    } == {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
    }


def test_apply_rejects_unmodeled_physical_runtime_before_override():
    with patch(
        "sglang_simulator.hardware_policy._server_arg_defaults",
        return_value={"device": None},
    ):
        with pytest.raises(SGLangCompatibilityError, match=r"--device='cuda'"):
            apply_simulator_server_args({"device": "cuda"})


def test_external_harness_server_args_are_classified_during_construction(
    monkeypatch,
):
    class ExternalServerArgs:
        def __init__(self):
            self.tp_size = 4
            self.attention_backend = "flashinfer"
            self.__post_init__()

        def __post_init__(self):
            self.post_init_called = True

    monkeypatch.delenv("SGLANG_SIMULATOR_REQUESTED_SERVER_ARGS", raising=False)
    monkeypatch.delenv("SGLANG_SIMULATOR_POLICY_OUTCOMES", raising=False)
    with patch(
        "sglang_simulator.hardware_policy._server_arg_defaults",
        return_value={
            "tp_size": 1,
            "attention_backend": None,
            "speculative_algorithm": None,
        },
    ):
        C_ServerArgsHook.hook(ExternalServerArgs)
        server_args = ExternalServerArgs()
        apply_simulator_server_args(server_args)
        server_args.speculative_algorithm = "EAGLE"
        with pytest.raises(
            SGLangCompatibilityError,
            match=r"--speculative-algorithm='EAGLE'",
        ):
            apply_simulator_server_args(server_args)

    assert server_args.post_init_called
    assert server_args.tp_size == 1
    assert server_args.attention_backend == "torch_native"
    assert requested_server_arg_values()["tp_size"] == 4
    assert recorded_policy_outcomes() == {"warn_no_op": ["attention_backend"]}


def test_requested_logical_topology_feeds_predictor_with_json_precedence(
    monkeypatch,
):
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_REQUESTED_SERVER_ARGS",
        json.dumps(
            {
                "attn_cp_size": 2,
                "dp_size": 2,
                "ep_size": 4,
                "pp_size": 2,
                "tp_size": 8,
            }
        ),
    )
    ConfigManager._raw_config = {"scheduler": {"tp_size": 16}}
    config = SchedulerConfig()
    try:
        ConfigManager.set_scheduler_config(config)
    finally:
        ConfigManager.reset_config_cache()

    assert config.tp_size == 16
    assert config.ep_size == 4
    assert config.dp_size == 2
    assert config.pp_size == 2
    assert config.cp_size == 2


def test_startup_summary_is_stable_and_bounded(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_POLICY_OUTCOMES",
        json.dumps({"warn_no_op": ["attention_backend"]}),
    )
    scheduler_config = SimpleNamespace(
        tp_size=8,
        ep_size=4,
        dp_size=2,
        pp_size=2,
        cp_size=1,
    )
    kwargs = {
        "predictor_name": "replay",
        "predictor_mode": "knn",
        "simulation_mode": "OFFLINE",
        "scheduler_config": scheduler_config,
        "max_total_tokens": 1_000_000,
        "page_size": 256,
        "hicache_ratio": 2.0,
        "disaggregation_backend": "simulator",
        "disaggregation_bandwidth_gb_per_s": 64.0,
        "disaggregation_fixed_latency_ms": 1.0,
        "disaggregation_topology": None,
        "disaggregation_location": None,
        "deterministic_sampling_version": "sglang-simulator-token-v1",
    }

    first = build_startup_summary(**kwargs)
    second = build_startup_summary(**kwargs)
    prefix = "SGLang Simulator effective configuration: "
    summary = json.loads(first.removeprefix(prefix))

    assert first == second
    assert first.startswith(prefix)
    assert summary["logical_topology"] == {
        "attn_cp_size": 1,
        "dp_size": 2,
        "ep_size": 4,
        "pp_size": 2,
        "tp_size": 8,
    }
    assert summary["logical_kv"] == {
        "device_payload": "meta",
        "hicache_ratio": 2.0,
        "host_payload": "meta",
        "max_total_tokens": 1_000_000,
        "page_size": 256,
    }
    assert summary["explicit_outcomes"] == {"warn_no_op": ["attention_backend"]}
    assert summary["physical_runtime"] == dict(PHYSICAL_SERVER_ARG_OVERRIDES)
    assert set(summary) == {
        "classification_counts",
        "compute",
        "deterministic_sampling",
        "explicit_outcomes",
        "logical_kv",
        "logical_topology",
        "pd",
        "physical_runtime",
    }
    assert recorded_policy_outcomes() == {"warn_no_op": ["attention_backend"]}


def test_iteration_metrics_expose_bounded_simulator_observability():
    metrics = calc_iteration_metrics(
        [
            {
                "predicted_compute_latency": 0.25,
                "forward_latency": 0.25,
                "l2_load_latency": 0.5,
                "h2d_load_bytes": 4096,
                "cpu_overhead": 0.1,
            },
            {
                "predicted_compute_latency": 0.75,
                "forward_latency": 0.75,
                "l2_load_latency": 0.25,
                "h2d_load_bytes": 8192,
                "cpu_overhead": 0.1,
            },
        ],
        virtual_kv_capacity_tokens=1_000_000,
        virtual_kv_page_size=256,
    )

    assert metrics["simulator_predicted_compute_time_s"] == 1.0
    assert metrics["simulator_logical_transfer_bytes"] == 12_288
    assert metrics["simulator_logical_transfer_time_s"] == 0.75
    assert metrics["simulator_virtual_kv_capacity_tokens"] == 1_000_000
    assert metrics["simulator_virtual_kv_page_size"] == 256
