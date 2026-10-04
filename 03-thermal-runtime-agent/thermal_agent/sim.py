"""A simulated device with a thermal mass and reactive hardware throttling.

This exists so the agent logic can be developed and tested without hardware.
It is a model, not a measurement. The presets are illustrative and are not
fitted to any real device. Fit them to real traces with
thermal_agent.thermal_model.fit_thermal_model once real data exists.

Model:
  - Each knob level scales dynamic power, throughput and quality by a constant
    factor (the ground truth), and the factors multiply across knobs. The
    default device has one knob, a power cap. model-level devices (see
    make_phone_device) have token pacing, quantization and KV cache precision,
    whose true effects can differ from the priors the agent starts with.
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

from .knobs import (
    Knob, KnobLevel, kv_precision_knob, power_cap_knob, quantization_knob, token_pacing_knob,
)
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


class SimKnobActuator:
    """Actuator view of one knob on the device. knob is what the agent sees,
    which can differ from the device physics to test robustness to wrong priors."""

    def __init__(self, device: "SimulatedDevice", knob: Knob) -> None:
        self._device = device
        self.knob = knob

    def get_level(self) -> int:
        return self._device._levels[self.knob.name]

    def set_level(self, idx: int) -> None:
        self._device._levels[self.knob.name] = max(0, min(self.knob.top, int(idx)))


# Kept so older code and tests that import the original name still work.
SimPowerCapActuator = SimKnobActuator


class SimulatedDevice:
    def __init__(
        self,
        params: Optional[SimParams] = None,
        fractions=DEFAULT_FRACTIONS,
        coarse_only: bool = False,
        start_temp: Optional[float] = None,
        agent_knob: Optional[Knob] = None,
        knobs: Optional[List[Knob]] = None,
        agent_knobs: Optional[List[Knob]] = None,
    ) -> None:
        self.params = params or PRESETS["phone_severe"]
        self._amb = self.params.t_amb
        if knobs is None:
            # Original single knob device: a power cap with a DVFS law.
            self._caps = tuple(fractions)
            truth = [power_cap_knob(fractions=self._caps, mem_frac=self.params.mem_frac)]
            seen = [agent_knob] if agent_knob is not None else truth
        else:
            self._caps = ()
            truth = list(knobs)
            seen = list(agent_knobs) if agent_knobs is not None else truth
        if len(seen) != len(truth):
            raise ValueError("agent knobs must match the physical knobs one to one")
        for t, a in zip(truth, seen):
            if t.name != a.name or len(t.levels) != len(a.levels):
                raise ValueError("agent knob must match the physical levels")
        self._truth = truth
        self._levels = {k.name: k.top for k in truth}
        self._actuators = [SimKnobActuator(self, a) for a in seen]
        self.actuator = self._actuators[0]
        self.coarse_only = coarse_only
        self.temp = self._amb if start_temp is None else start_temp
        self.clock = 0.0
        self.power_w = 0.0
        self.tps = 0.0
        self.hw_throttled = False
        self.throttle_events = 0
        self._rng = random.Random(self.params.seed)

    def _scale(self, attr: str) -> float:
        v = 1.0
        for k in self._truth:
            v *= getattr(k.levels[self._levels[k.name]], attr)
        return v

    @property
    def quality(self) -> float:
        return self._scale("quality_scale")

    def steady_state(self, levels: Optional[dict] = None) -> float:
        """Closed form steady temperature at a configuration (no throttling)."""
        p = self.params
        saved = self._levels
        self._levels = dict(levels) if levels is not None else saved
        try:
            power = p.p_idle + (p.p_full - p.p_idle) * self._scale("power_scale")
        finally:
            self._levels = saved
        return self._amb + p.r_th * power

    @property
    def actuators(self) -> List[SimKnobActuator]:
        return list(self._actuators)

    def set_ambient(self, t_amb: float) -> None:
        self._amb = t_amb

    def advance(self, dt: float) -> None:
        p = self.params
        p_scale = self._scale("power_scale")
        s_scale = self._scale("speed_scale")
        frac = min(p_scale, p.throttle_cap) if self.hw_throttled else p_scale
        power = p.p_idle + (p.p_full - p.p_idle) * frac
        t_ss = self._amb + p.r_th * power
        self.temp = t_ss + (self.temp - t_ss) * math.exp(-dt / p.tau_s)

        def dvfs(x: float) -> float:
            return p.mem_frac + (1.0 - p.mem_frac) * x ** (1.0 / 3.0)

        speed = p.base_tps * s_scale
        if self.hw_throttled and frac < p_scale:
            speed *= dvfs(frac) / dvfs(p_scale)
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


# --- A phone with model level knobs ---------------------------------------------
# Ground truth for the simulated phone's knobs. ILLUSTRATIVE: these are scenarios
# for testing the agent, not measurements of any phone. The agent starts from the
# priors in knobs.py, which differ from these on purpose.

def _quant_truth(cools: bool):
    """True effect of quantization. Fewer bits always speeds decode up (fewer
    bytes per token). Whether watts also fall is the open question: if the
    memory system stays saturated, tokens just come faster at the same power
    (cools=False). If less data movement really lowers power, they fall too."""
    from .knobs import QUANT_LEVELS
    top_bits = QUANT_LEVELS[-1][1]
    out = []
    for label, bits, quality in QUANT_LEVELS:
        rel = bits / top_bits
        speed = 1.0 / (0.40 + 0.60 * rel)
        power = (1.0 - 0.8 * (1.0 - rel) * 0.7) if cools else 1.0
        out.append(KnobLevel(label, power_scale=power, speed_scale=speed, quality_scale=quality))
    return Knob("quantization", tuple(out), switch_cost=1)


def _kv_truth():
    spec = (("KV4", 0.98, 1.02, 0.97), ("KV8", 0.99, 1.01, 0.995), ("KV16", 1.0, 1.0, 1.0))
    return Knob("kv_precision", tuple(KnobLevel(l, p, sp, q) for l, p, sp, q in spec), switch_cost=1)


PHONE_SCENARIOS = {
    # Fewer bits lower the watts as well as raising the speed.
    "phone_quant_cools": True,
    # Fewer bits only make tokens faster. The agent expects some cooling and has to discover there is none.
    "phone_quant_flat": False,
}


def make_phone_device(scenario: str = "phone_quant_flat", params: Optional[SimParams] = None, **kw) -> SimulatedDevice:
    """A hot phone with no platform power cap. The agent can pace tokens, change
    the quantization of the model, and change the KV cache precision."""
    truth = [token_pacing_knob(), _quant_truth(PHONE_SCENARIOS[scenario]), _kv_truth()]
    seen = [token_pacing_knob(), quantization_knob(), kv_precision_knob()]
    return SimulatedDevice(params or PRESETS["phone_severe"], knobs=truth, agent_knobs=seen, **kw)
