"""Early compatibility checks for the SGLang surfaces used by the simulator."""

import inspect
import logging
from importlib import metadata

from sglang_simulator import hardware_policy
from sglang_simulator.hardware_policy import (
    PHYSICAL_SERVER_ARG_OVERRIDES,
    SimulatorPolicyError,
    evaluate_simulator_policy,
    policy_context_recorded,
    record_policy_context,
    recorded_policy_outcomes,
    server_arg_values,
)
from sglang_simulator.profiles import (
    SimulatorProfileError,
    apply_simulator_profile,
    infer_explicit_fields,
)

logger = logging.getLogger(__name__)

SIMULATOR_SERVER_ARG_OVERRIDES = PHYSICAL_SERVER_ARG_OVERRIDES


class SGLangCompatibilityError(RuntimeError):
    pass


def _sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _require_parameters(function, required: set[str], surface: str) -> None:
    parameters = set(inspect.signature(function).parameters)
    missing = required - parameters
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing {surface} parameters: "
            f"{', '.join(sorted(missing))}. The simulator must be adapted to "
            "this SGLang revision before it can run."
        )


def apply_simulator_server_args(target, *, explicit_fields=None) -> None:
    """Classify requested values, then apply simulator physical-runtime values."""
    requested_values = server_arg_values(target)
    already_prepared = policy_context_recorded() and all(
        requested_values.get(name) == expected
        for name, expected in SIMULATOR_SERVER_ARG_OVERRIDES.items()
    )
    defaults = hardware_policy._server_arg_defaults()
    effective_explicit_fields = infer_explicit_fields(
        target,
        defaults=defaults,
        explicit_fields=frozenset(explicit_fields or ()),
    )
    try:
        if not already_prepared:
            apply_simulator_profile(
                target,
                explicit_fields=effective_explicit_fields,
            )
            requested_values = server_arg_values(target)
        evaluation = evaluate_simulator_policy(
            requested_values,
            explicit_fields=effective_explicit_fields,
            ignored_fields=(
                frozenset(SIMULATOR_SERVER_ARG_OVERRIDES)
                if already_prepared
                else frozenset()
            ),
            defaults=defaults,
        )
    except (SimulatorPolicyError, SimulatorProfileError) as error:
        raise SGLangCompatibilityError(str(error)) from error

    previous_warnings = set(recorded_policy_outcomes()["warn_no_op"])
    for field, message in zip(
        evaluation.warning_fields,
        evaluation.warning_messages,
        strict=True,
    ):
        if field not in previous_warnings:
            logger.warning("SGLang Simulator: %s", message)
    if already_prepared:
        return

    record_policy_context(
        requested_values=requested_values,
        evaluation=evaluation,
    )

    if isinstance(target, dict):
        target.update(SIMULATOR_SERVER_ARG_OVERRIDES)
        return

    for name, value in SIMULATOR_SERVER_ARG_OVERRIDES.items():
        setattr(target, name, value)


def validate_simulator_server_args(server_args) -> None:
    """Fail early if a process bypassed a simulator-owned launch entry point."""
    mismatches = [
        f"{name}={getattr(server_args, name)!r} (expected {expected!r})"
        for name, expected in SIMULATOR_SERVER_ARG_OVERRIDES.items()
        if getattr(server_args, name) != expected
    ]
    if mismatches:
        raise SGLangCompatibilityError(
            "SGLang Simulator server arguments were not prepared by a supported "
            f"entry point: {', '.join(mismatches)}"
        )


def validate_launch_runtime() -> None:
    from sglang.srt.entrypoints.http_server import launch_server

    _require_parameters(
        launch_server,
        {"server_args", "run_scheduler_process_func", "run_detokenizer_process_func"},
        "launch_server",
    )


def validate_benchmark_runtime() -> None:
    from sglang.benchmark import serving

    missing = [
        name
        for name in (
            "BenchmarkMetrics",
            "calculate_metrics",
            "cli_main",
            "get_request",
            "run_benchmark",
        )
        if not hasattr(serving, name)
    ]
    if missing:
        raise SGLangCompatibilityError(
            f"SGLang {_sglang_version()} is missing benchmark surfaces: "
            f"{', '.join(missing)}. The simulator must be adapted to this "
            "SGLang revision before it can run."
        )
