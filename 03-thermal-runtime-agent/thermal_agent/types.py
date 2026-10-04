"""Shared types for the thermal runtime agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Coarse thermal states, ordered. These mirror what phone operating systems
# expose to apps (iOS reports four states and no raw temperature).
THERMAL_STATES = ("nominal", "fair", "serious", "critical")


def state_rank(state: str) -> int:
    return THERMAL_STATES.index(state)


@dataclass(frozen=True)
class Observation:
    """One telemetry reading.

    power_w is the mean power over the interval that ends at t_s.
    temp_c may be None on devices that only expose a coarse thermal state.
    """

    t_s: float
    power_w: float
    tok_per_s: float
    temp_c: Optional[float] = None
    thermal_state: str = "nominal"
    throttled: bool = False
    battery_pct: Optional[float] = None


# A configuration maps a knob name to a level index.
Config = Dict[str, int]


def config_key(config: Config) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(config.items()))
