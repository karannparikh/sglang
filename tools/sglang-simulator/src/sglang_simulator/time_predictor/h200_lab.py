from __future__ import annotations

import random
from itertools import pairwise

from sglang_simulator.profiles import H200_LAB_PREDICTOR_NAME
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch

_QUANTILE_PROBABILITIES = (0.50, 0.90, 0.95, 0.99)
_PREFILL_FIXED_OVERHEAD_MS = 4.0
_SEED_MASK = (1 << 64) - 1

H200_LAB_DECODE_ENGINE_MS = (
    (1, (3.6, 6.2, 7.0, 8.4)),
    (2, (4.1, 7.2, 7.8, 8.7)),
    (4, (6.2, 8.0, 8.5, 9.7)),
    (8, (7.5, 8.8, 9.3, 11.7)),
    (16, (8.6, 10.0, 10.4, 12.0)),
    (24, (9.8, 11.0, 11.4, 12.9)),
    (32, (10.4, 12.0, 12.7, 14.8)),
)

H200_LAB_PREFILL_ENGINE_MS = (
    (8_192, (130.0, 1_000.0, 2_500.0, 8_500.0)),
    (24_576, (260.0, 2_000.0, 5_000.0, 17_000.0)),
    (49_152, (1_050.0, 3_000.0, 5_400.0, 16_000.0)),
    (98_304, (2_450.0, 5_300.0, 6_800.0, 21_000.0)),
    (196_608, (6_500.0, 14_800.0, 22_000.0, 57_000.0)),
    (393_216, (19_000.0, 46_000.0, 73_000.0, 124_000.0)),
)


class H200LabTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *,
        role: str,
        seed: int,
        logical_rank: int,
    ):
        super().__init__(model, hw, config)
        if role not in ("prefill", "decode"):
            raise ValueError(
                f"{H200_LAB_PREDICTOR_NAME} supports only prefill and decode "
                f"roles, got {role!r}"
            )
        if logical_rank < 0:
            raise ValueError("Simulator logical rank must be non-negative")
        self.role = role
        self.seed = int(seed)
        self.logical_rank = logical_rank
        self._rng = random.Random(_rank_seed(seed=self.seed, rank=logical_rank))

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        if self.role == "decode":
            if not batch.is_decode():
                raise ValueError(
                    f"{H200_LAB_PREDICTOR_NAME} has no measured decode latency "
                    "for a batch containing prefill work"
                )
            latency_ms = _sample_profile_ms(
                H200_LAB_DECODE_ENGINE_MS,
                workload=batch.batch_size,
                percentile=self._rng.random(),
            )
            return latency_ms / 1000.0

        recomputed_tokens = batch.num_context_tokens
        if recomputed_tokens == 0:
            return 0.0
        latency_ms = _sample_profile_ms(
            H200_LAB_PREFILL_ENGINE_MS,
            workload=recomputed_tokens,
            percentile=self._rng.random(),
        )
        first_anchor = H200_LAB_PREFILL_ENGINE_MS[0][0]
        if recomputed_tokens < first_anchor:
            work_fraction = recomputed_tokens / first_anchor
            latency_ms = _PREFILL_FIXED_OVERHEAD_MS + work_fraction * (
                latency_ms - _PREFILL_FIXED_OVERHEAD_MS
            )
        return latency_ms / 1000.0


def _rank_seed(*, seed: int, rank: int) -> int:
    mixed_rank = (rank + 0x9E3779B97F4A7C15) & _SEED_MASK
    mixed_rank ^= mixed_rank >> 30
    mixed_rank = (mixed_rank * 0xBF58476D1CE4E5B9) & _SEED_MASK
    mixed_rank ^= mixed_rank >> 27
    mixed_rank = (mixed_rank * 0x94D049BB133111EB) & _SEED_MASK
    mixed_rank ^= mixed_rank >> 31
    return (int(seed) & _SEED_MASK) ^ mixed_rank


def _sample_profile_ms(
    profile: tuple[tuple[int, tuple[float, float, float, float]], ...],
    *,
    workload: int,
    percentile: float,
) -> float:
    p50, p90, p95, p99 = _interpolate_workload_quantiles(profile, workload)
    lower = max(0.0, p50 - (p90 - p50) * (0.50 / 0.40))
    upper = max(0.0, p99 + (p99 - p95) * (0.01 / 0.04))
    inverse_cdf = (
        (0.0, lower),
        (_QUANTILE_PROBABILITIES[0], p50),
        (_QUANTILE_PROBABILITIES[1], p90),
        (_QUANTILE_PROBABILITIES[2], p95),
        (_QUANTILE_PROBABILITIES[3], p99),
        (1.0, upper),
    )
    return _interpolate(
        inverse_cdf,
        min(max(float(percentile), 0.0), 1.0),
    )


def _interpolate_workload_quantiles(
    profile: tuple[tuple[int, tuple[float, float, float, float]], ...],
    workload: int,
) -> tuple[float, float, float, float]:
    if workload <= profile[0][0]:
        return profile[0][1]
    for (left_workload, left), (right_workload, right) in pairwise(profile):
        if workload <= right_workload:
            ratio = (workload - left_workload) / (right_workload - left_workload)
            return tuple(
                left_value + ratio * (right_value - left_value)
                for left_value, right_value in zip(left, right)
            )
    return profile[-1][1]


def _interpolate(
    points: tuple[tuple[float, float], ...],
    value: float,
) -> float:
    if value <= points[0][0]:
        return points[0][1]
    for (left_x, left_y), (right_x, right_y) in pairwise(points):
        if value <= right_x:
            ratio = (value - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return points[-1][1]
