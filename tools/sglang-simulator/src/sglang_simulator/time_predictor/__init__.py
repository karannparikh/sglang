from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.h200_lab import H200LabTimePredictor
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sglang_simulator.time_predictor.replay import ReplayTimePredictor

__all__ = (
    "ScheduleRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "AISimulateTimePredictor",
    "H200LabTimePredictor",
    "MLTimePredictor",
    "ReplayTimePredictor",
)


def __getattr__(name):
    if name == "AISimulateTimePredictor":
        from sglang_simulator.time_predictor.aisimulate import (
            AISimulateTimePredictor,
        )

        return AISimulateTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
