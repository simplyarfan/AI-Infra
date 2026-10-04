"""A simulated device with a thermal mass and reactive hardware throttling.

This exists so the agent logic can be developed and tested without hardware.
It is a model, not a measurement. The presets are illustrative and are not
fitted to any real device. Fit them to real traces with
thermal_agent.thermal_model.fit_thermal_model once real data exists.

Model:
  - A power cap level sets the fraction of dynamic power the chip may use.
  - Temperature follows the first order model in thermal_model.py.
  - Hardware throttling is reactive: once the temperature passes t_throttle the
    chip is forced down to throttle_cap until it cools below t_release. This
    gives the sawtooth behaviour seen on real phones under sustained load.
  - Throughput has a memory bound part and a compute bound part that scales
    with frequency, which scales with the cube root of power.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import List, Optional

from .knobs import Knob, power_cap_knob
from .types import Observation

DEFAULT_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class SimParams:
    t_amb: float = 25.0
    r_th: float = 8.0          # C per W
    tau_s: float = 60.0
    p_idle: float = 0.5        # W
    p_full: float = 6.0        # W at the top power cap level
    base_tps: float = 77.0     # tokens per second at full power
    mem_frac: float = 0.35     # memory bound share of throughput
    t_throttle: float = 55.0   # hardware starts throttling here
    t_release: float = 50.0    # and releases below this
    throttle_cap: float = 0.03
    noise_tps: float = 0.0     # relative gaussian noise on throughput
    seed: int = 0


PRESETS = {
    # Small thermal mass, heavy reactive throttling, no fan.
    "phone_severe": SimParams(),
    # Larger thermal mass and surface: under the same load it settles below the
    # throttle point, like the iPad in the measurements.
    "tablet_cool": SimParams(r_th=3.0, tau_s=180.0, base_tps=97.0),
}


def thermal_state_for(temp_c: float, t_throttle: float) -> str:
    if temp_c >= t_throttle:
        return "critical"
    if temp_c >= t_throttle - 6.0:
        return "serious"
    if temp_c >= t_throttle - 12.0:
        return "fair"
    return "nominal"


class SimPowerCapActuator:
    """Actuator view of the device. knob is what the agent sees, which can
    differ from the device physics to test robustness to wrong priors."""

    def __init__(self, device: "SimulatedDevice", knob: Knob) -> None:
        self._device = device
        self.knob = knob

    def get_level(self) -> int:
        return self._device._level

    def set_level(self, idx: int) -> None:
        self._device._level = max(0, min(self.knob.top, int(idx)))


class SimulatedDevice:
    def __init__(
        self,
        params: Optional[SimParams] = None,
        fractions=DEFAULT_FRACTIONS,
        coarse_only: bool = False,
        start_temp: Optional[float] = None,
        agent_knob: Optional[Knob] = None,
    ) -> None:
        self.params = params or PRESETS["phone_severe"]
        self._amb = self.params.t_amb
        self._caps = tuple(fractions)
        knob = agent_knob or power_cap_knob(fractions=self._caps, mem_frac=self.params.mem_frac)
        if len(knob.levels) != len(self._caps):
            raise ValueError("agent knob must have one level per physical cap")
        self.actuator = SimPowerCapActuator(self, knob)
        self.coarse_only = coarse_only
        self._level = len(self._caps) - 1
        self.temp = self._amb if start_temp is None else start_temp
        self.clock = 0.0
        self.power_w = 0.0
        self.tps = 0.0
        self.hw_throttled = False
        self.throttle_events = 0
        self._rng = random.Random(self.params.seed)

    @property
    def actuators(self) -> List[SimPowerCapActuator]:
        return [self.actuator]

    def set_ambient(self, t_amb: float) -> None:
        self._amb = t_amb

    def advance(self, dt: float) -> None:
        p = self.params
        cap_set = self._caps[self._level]
        cap = min(cap_set, p.throttle_cap) if self.hw_throttled else cap_set
        power = p.p_idle + (p.p_full - p.p_idle) * cap
        t_ss = self._amb + p.r_th * power
        self.temp = t_ss + (self.temp - t_ss) * math.exp(-dt / p.tau_s)
        speed = p.base_tps * (p.mem_frac + (1.0 - p.mem_frac) * cap ** (1.0 / 3.0))
        if p.noise_tps:
            speed *= 1.0 + self._rng.gauss(0.0, p.noise_tps)
        self.power_w, self.tps = power, speed
        if not self.hw_throttled and self.temp >= p.t_throttle:
            self.hw_throttled = True
            self.throttle_events += 1
        elif self.hw_throttled and self.temp <= p.t_release:
            self.hw_throttled = False
        self.clock += dt

    def read(self) -> Observation:
        return Observation(
            t_s=self.clock,
            power_w=self.power_w,
            tok_per_s=self.tps,
            temp_c=None if self.coarse_only else self.temp,
            thermal_state=thermal_state_for(self.temp, self.params.t_throttle),
            throttled=self.hw_throttled,
        )
