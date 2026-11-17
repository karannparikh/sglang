from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import msgspec

H200_LAB_PROFILE_NAME = "h200-lab-disaggregated-v1"
H200_LAB_TWO_ZONES_TOPOLOGY = "h200-lab-two-zones-v1"
H200_LAB_MIXED_TOPOLOGY = "h200-lab-mixed-v1"
H200_LAB_PREDICTOR_NAME = "h200-lab-sampled-v1"

_CONFIG_PATH_ENV = "SGLANG_SIMULATOR_CONFIG_PATH"


class SimulatorProfileError(ValueError):
    pass


class SimulatorProfile(msgspec.Struct, frozen=True):
    name: str
    predictor_name: str
    server_arg_defaults: tuple[tuple[str, Any], ...]
    supported_roles: tuple[str, ...]


class SimulatorTopologyPreset(msgspec.Struct, frozen=True):
    name: str
    topology_json: str


def _topology_json(*, levels: tuple[str, ...], links: dict[str, dict]) -> str:
    return json.dumps(
        {"levels": list(levels), "links": links},
        sort_keys=True,
        separators=(",", ":"),
    )


H200_LAB_PROFILE = SimulatorProfile(
    name=H200_LAB_PROFILE_NAME,
    predictor_name=H200_LAB_PREDICTOR_NAME,
    server_arg_defaults=(
        ("tp_size", 8),
        ("page_size", 256),
        ("max_total_tokens", 3_970_304),
        ("context_length", 1_048_576),
        ("max_running_requests", 32),
        ("max_prefill_tokens", 8_192),
        ("chunked_prefill_size", 8_192),
        ("kv_cache_dtype", "fp8_e4m3"),
        ("random_seed", 0),
    ),
    supported_roles=("prefill", "decode"),
)

H200_LAB_TWO_ZONES = SimulatorTopologyPreset(
    name=H200_LAB_TWO_ZONES_TOPOLOGY,
    topology_json=_topology_json(
        levels=("zone", "rack", "node"),
        links={
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
            "default": {
                "bandwidth_gb_per_s": 1.25,
                "fixed_latency_ms": 5.0,
            },
        },
    ),
)

H200_LAB_MIXED = SimulatorTopologyPreset(
    name=H200_LAB_MIXED_TOPOLOGY,
    topology_json=_topology_json(
        levels=("rack", "node"),
        links={
            "node": {
                "bandwidth_gb_per_s": 450.0,
                "fixed_latency_ms": 0.01,
            },
            "rack": {
                "bandwidth_gb_per_s": 50.0,
                "fixed_latency_ms": 0.05,
            },
            "default": {
                "bandwidth_gb_per_s": 12.5,
                "fixed_latency_ms": 0.5,
            },
        },
    ),
)

SIMULATOR_PROFILES = MappingProxyType(
    {
        H200_LAB_PROFILE.name: H200_LAB_PROFILE,
    }
)
SIMULATOR_TOPOLOGY_PRESETS = MappingProxyType(
    {
        H200_LAB_TWO_ZONES.name: H200_LAB_TWO_ZONES,
        H200_LAB_MIXED.name: H200_LAB_MIXED,
    }
)


def load_simulator_config() -> dict[str, Any]:
    config_path = os.getenv(_CONFIG_PATH_ENV)
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise SimulatorProfileError(
            f"{_CONFIG_PATH_ENV} does not name a simulator config file: {config_path}"
        )
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise SimulatorProfileError("Simulator configuration must be a JSON object")
    return raw


def get_selected_profile(
    raw_config: Mapping[str, Any] | None = None,
) -> SimulatorProfile | None:
    raw_config = load_simulator_config() if raw_config is None else raw_config
    name = raw_config.get("profile")
    if name is None:
        return None
    if not isinstance(name, str) or name not in SIMULATOR_PROFILES:
        raise SimulatorProfileError(
            f"Unknown simulator profile {name!r}. Supported: "
            f"{', '.join(SIMULATOR_PROFILES)}"
        )
    return SIMULATOR_PROFILES[name]


def get_selected_topology_preset(
    raw_config: Mapping[str, Any] | None = None,
) -> SimulatorTopologyPreset | None:
    raw_config = load_simulator_config() if raw_config is None else raw_config
    name = raw_config.get("topology_preset")
    if name is None:
        return None
    if not isinstance(name, str) or name not in SIMULATOR_TOPOLOGY_PRESETS:
        raise SimulatorProfileError(
            f"Unknown simulator topology preset {name!r}. Supported: "
            f"{', '.join(SIMULATOR_TOPOLOGY_PRESETS)}"
        )
    return SIMULATOR_TOPOLOGY_PRESETS[name]


def apply_simulator_profile(
    target,
    *,
    explicit_fields: frozenset[str],
) -> tuple[str | None, str | None]:
    raw_config = load_simulator_config()
    profile = get_selected_profile(raw_config)
    topology = get_selected_topology_preset(raw_config)
    if profile is None and topology is None:
        return None, None

    if profile is not None and raw_config.get("predictor") is not None:
        raise SimulatorProfileError(
            f"Simulator profile {profile.name!r} owns its timing predictor; "
            "remove the predictor section instead of mixing timing models"
        )

    profile_overrides = raw_config.get("profile_overrides", {})
    if not isinstance(profile_overrides, Mapping):
        raise SimulatorProfileError("profile_overrides must be a JSON object")
    if profile is None and profile_overrides:
        raise SimulatorProfileError(
            "profile_overrides requires a named simulator profile"
        )

    values = _target_values(target)
    if profile is not None:
        profile_fields = dict(profile.server_arg_defaults)
        unknown_overrides = sorted(set(profile_overrides) - set(profile_fields))
        if unknown_overrides:
            raise SimulatorProfileError(
                f"Profile {profile.name!r} cannot override fields outside its "
                f"default set: {unknown_overrides}"
            )
        for name, profile_default in profile.server_arg_defaults:
            if name in profile_overrides:
                override = profile_overrides[name]
                if (
                    name in explicit_fields
                    and name in values
                    and values[name] != override
                ):
                    raise SimulatorProfileError(
                        f"Ambiguous simulator profile value for --"
                        f"{name.replace('_', '-')}: command/runtime value "
                        f"{values[name]!r} conflicts with profile_overrides "
                        f"value {override!r}"
                    )
                _set_target_value(target, name, override)
            elif name not in explicit_fields:
                _set_target_value(target, name, profile_default)

        role = _target_values(target).get("disaggregation_mode")
        if role is not None and role not in profile.supported_roles:
            raise SimulatorProfileError(
                f"Simulator profile {profile.name!r} supports only disaggregated "
                f"roles {profile.supported_roles}; got disaggregation_mode={role!r}"
            )

    if topology is not None:
        topology_field = "disaggregation_simulator_topology"
        existing_topology = _target_values(target).get(topology_field)
        if topology_field in explicit_fields and existing_topology is not None:
            raise SimulatorProfileError(
                f"Topology preset {topology.name!r} cannot be combined with an "
                "explicit --disaggregation-simulator-topology value"
            )
        _set_target_value(target, topology_field, topology.topology_json)

    return (
        profile.name if profile is not None else None,
        topology.name if topology is not None else None,
    )


def infer_explicit_fields(
    target,
    *,
    defaults: Mapping[str, Any],
    explicit_fields: frozenset[str],
) -> frozenset[str]:
    values = _target_values(target)
    inferred = {
        name
        for name, value in values.items()
        if name in defaults and value != defaults[name]
    }
    return frozenset((*explicit_fields, *inferred))


def _target_values(target) -> dict[str, Any]:
    if isinstance(target, Mapping):
        return dict(target)
    return vars(target)


def _set_target_value(target, name: str, value: Any) -> None:
    if isinstance(target, dict):
        target[name] = value
    else:
        setattr(target, name, value)
