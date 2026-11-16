from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

DEFAULT_LINK_NAME = "default"
_TOPOLOGY_FIELDS = {"levels", "links"}
_LINK_FIELDS = {"bandwidth_gb_per_s", "fixed_latency_ms"}


class SimulatorTopologyError(ValueError):
    pass


class SimulatorLink(msgspec.Struct, frozen=True):
    name: str
    bandwidth_gb_per_s: float
    fixed_latency_ms: float


class SimulatorTopology(msgspec.Struct, frozen=True):
    levels: tuple[str, ...]
    links: tuple[SimulatorLink, ...]

    @classmethod
    def from_json(cls, text: str) -> SimulatorTopology:
        if not isinstance(text, str) or not text:
            raise SimulatorTopologyError(
                "--disaggregation-simulator-topology must be a non-empty JSON object"
            )
        try:
            raw = json.loads(text, object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as exc:
            raise SimulatorTopologyError(
                f"--disaggregation-simulator-topology is not valid JSON: {exc.msg}"
            ) from exc
        return cls.from_config(raw)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> SimulatorTopology:
        if not isinstance(raw, Mapping):
            raise SimulatorTopologyError(
                "--disaggregation-simulator-topology must be a JSON object"
            )
        unknown_fields = sorted(set(raw) - _TOPOLOGY_FIELDS)
        if unknown_fields:
            raise SimulatorTopologyError(
                f"Simulator topology has unknown fields: {', '.join(unknown_fields)}"
            )

        raw_levels = raw.get("levels")
        if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
            raise SimulatorTopologyError(
                "Simulator topology levels must be a non-empty ordered list"
            )
        if any(not isinstance(level, str) or not level for level in raw_levels):
            raise SimulatorTopologyError(
                "Simulator topology levels must be non-empty strings"
            )
        levels = tuple(raw_levels)
        if len(set(levels)) != len(levels):
            raise SimulatorTopologyError("Simulator topology levels must be unique")
        if DEFAULT_LINK_NAME in levels:
            raise SimulatorTopologyError(
                f"Simulator topology level name {DEFAULT_LINK_NAME!r} is reserved"
            )

        raw_links = raw.get("links")
        if not isinstance(raw_links, Mapping):
            raise SimulatorTopologyError(
                "Simulator topology links must be a JSON object"
            )
        expected_names = (*levels, DEFAULT_LINK_NAME)
        missing_names = sorted(set(expected_names) - set(raw_links))
        unknown_names = sorted(set(raw_links) - set(expected_names))
        if missing_names or unknown_names:
            raise SimulatorTopologyError(
                "Simulator topology links must match levels plus 'default'; "
                f"missing={missing_names}, unknown={unknown_names}"
            )

        links = tuple(
            _parse_link(name=name, raw=raw_links[name]) for name in expected_names
        )
        return cls(levels=levels, links=links)

    def to_json(self) -> str:
        raw = {
            "levels": list(self.levels),
            "links": {
                link.name: {
                    "bandwidth_gb_per_s": link.bandwidth_gb_per_s,
                    "fixed_latency_ms": link.fixed_latency_ms,
                }
                for link in self.links
            },
        }
        return json.dumps(raw, separators=(",", ":"))

    def validate_location(self, location: Sequence[str]) -> None:
        if len(location) != len(self.levels) or any(
            not isinstance(part, str) or not part for part in location
        ):
            raise SimulatorTopologyError(
                "--disaggregation-simulator-location must have exactly "
                f"{len(self.levels)} non-empty components ({'/'.join(self.levels)})"
            )

    def link_between(
        self, source: Sequence[str], destination: Sequence[str]
    ) -> SimulatorLink:
        self.validate_location(source)
        self.validate_location(destination)
        shared_depth = 0
        for source_part, destination_part in zip(source, destination):
            if source_part != destination_part:
                break
            shared_depth += 1
        link_name = (
            DEFAULT_LINK_NAME if shared_depth == 0 else self.levels[shared_depth - 1]
        )
        return next(link for link in self.links if link.name == link_name)


def parse_simulator_location(location: str) -> tuple[str, ...]:
    if not isinstance(location, str) or not location:
        raise SimulatorTopologyError(
            "--disaggregation-simulator-location must be a non-empty path"
        )
    parts = tuple(location.split("/"))
    if any(not part for part in parts):
        raise SimulatorTopologyError(
            "--disaggregation-simulator-location components must be non-empty"
        )
    return parts


def load_simulator_topology(text: str | None) -> SimulatorTopology | None:
    return None if text is None else SimulatorTopology.from_json(text)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise SimulatorTopologyError(
                f"Simulator topology JSON contains duplicate key {name!r}"
            )
        result[name] = value
    return result


def _parse_link(*, name: str, raw: Any) -> SimulatorLink:
    if not isinstance(raw, Mapping):
        raise SimulatorTopologyError(
            f"Simulator topology link {name!r} must be a JSON object"
        )
    missing_fields = sorted(_LINK_FIELDS - set(raw))
    unknown_fields = sorted(set(raw) - _LINK_FIELDS)
    if missing_fields or unknown_fields:
        raise SimulatorTopologyError(
            f"Simulator topology link {name!r} must contain exactly "
            "bandwidth_gb_per_s and fixed_latency_ms; "
            f"missing={missing_fields}, unknown={unknown_fields}"
        )

    bandwidth = _finite_number(
        raw["bandwidth_gb_per_s"],
        field=f"Simulator topology link {name!r} bandwidth_gb_per_s",
    )
    fixed_latency = _finite_number(
        raw["fixed_latency_ms"],
        field=f"Simulator topology link {name!r} fixed_latency_ms",
    )
    if bandwidth <= 0:
        raise SimulatorTopologyError(
            f"Simulator topology link {name!r} bandwidth_gb_per_s must be positive"
        )
    if fixed_latency < 0:
        raise SimulatorTopologyError(
            f"Simulator topology link {name!r} fixed_latency_ms must be non-negative"
        )
    return SimulatorLink(
        name=name,
        bandwidth_gb_per_s=bandwidth,
        fixed_latency_ms=0.0 if fixed_latency == 0 else fixed_latency,
    )


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulatorTopologyError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SimulatorTopologyError(f"{field} must be a finite number")
    return result
