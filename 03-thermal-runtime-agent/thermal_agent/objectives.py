"""Agent 1: optimization objectives and constraints, with a thermal budget.

The AutoOptAgent slide lists latency, throughput, memory and energy as
objectives. This adds the missing one: a thermal budget. Energy is related but
not the same thing: a configuration can be efficient in joules and still hold
the chip at a temperature that triggers hardware throttling.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import THERMAL_STATES


@dataclass(frozen=True)
class Objectives:
    # Hard-ish constraint: stay below this temperature. Set it just under the
    # temperature where the hardware starts throttling.
    max_temp_c: float = 53.0
    # Safety margin applied to forecasts.
    temp_margin_c: float = 1.0
    # How far ahead the controller plans when deciding to back off. Short
    # horizons exploit thermal mass for bursts (interactive use), long
    # horizons plan for sustained load.
    horizon_s: float = 120.0
    # Extra margin required before stepping back up, to stop hunting.
    raise_margin_c: float = 1.0
    # Soft throughput floor, reported by the evaluator but never traded
    # against the thermal constraint.
    min_tok_per_s: float = 0.0
    # Coarse-state devices (no raw temperature): back off at this state and
    # recover when the state falls back to the up state.
    coarse_down_state: str = "serious"
    coarse_up_state: str = "nominal"
    # Quality constraint (the "accuracy" constraint on the AutoOptAgent slide).
    # A configuration whose quality score is below this is never chosen, even to
    # avoid heat. Scores are relative to the top configuration.
    min_quality: float = 0.0
    # Exchange rate in the objective function. Utility of a configuration is
    #   tok_per_s / reference_tok_per_s + quality_weight * quality
    # so quality_weight says how many "units of relative speed" one unit of
    # quality is worth. It only matters when knobs change quality.
    quality_weight: float = 2.0

    @property
    def limit_c(self) -> float:
        return self.max_temp_c - self.temp_margin_c

    def validate(self) -> None:
        if self.temp_margin_c < 0 or self.raise_margin_c < 0:
            raise ValueError("margins must be non negative")
        if self.horizon_s <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= self.min_quality <= 1.0 or self.quality_weight < 0:
            raise ValueError("min_quality must be in [0, 1] and quality_weight non negative")
        for s in (self.coarse_down_state, self.coarse_up_state):
            if s not in THERMAL_STATES:
                raise ValueError("unknown thermal state: " + s)
