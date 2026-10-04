"""Agent 4a: configuration prediction.

Chooses the next configuration. This is a model predictive controller rather
than a black box search, which matters for a live edge device:

  - Moving DOWN (thermal pressure or violation) may jump several levels at once,
    because being too hot is the expensive mistake. It picks the highest
    throughput configuration whose forecast stays under the limit.
  - Moving UP is exploration, so it is restricted to one step on one knob, and
    only if the candidate's steady state is under the limit. This is the safe
    exploration rule: never try something the model says will overheat.
  - After backing off from a configuration, going back to anything that draws as
    much power is blocked for tabu_s seconds, which stops hunting.

Predicted power and throughput come from measured data when the configuration
has been visited, and otherwise from the knob priors scaled by what the device
has actually been doing.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import ConfigPerfDataset
from .evaluation import Assessment
from .knobs import Knob
from .objectives import Objectives
from .thermal_model import ThermalParams
from .types import Config, Observation, state_rank


class ConfigPredictor:
    def __init__(
        self,
        knobs: Sequence[Knob],
        objectives: Objectives,
        dataset: ConfigPerfDataset,
        idle_power_w: float = 0.0,
        tabu_s: float = 90.0,
        min_stat_n: int = 3,
        alpha: float = 0.3,
    ) -> None:
        self.knobs = list(knobs)
        self.obj = objectives
        self.dataset = dataset
        self.idle = idle_power_w
        self.tabu_s = tabu_s
        self.min_stat_n = min_stat_n
        self.alpha = alpha
        self.full_dyn_power: Optional[float] = None
        self.full_tps: Optional[float] = None
        self._block_power: Optional[float] = None
        self._block_until: float = -1.0
        self._coarse_block_idx: Optional[int] = None

    # priors ---------------------------------------------------------------
    def _scale(self, config: Config, attr: str) -> float:
        v = 1.0
        for k in self.knobs:
            v *= getattr(k.levels[config[k.name]], attr)
        return v

    def seed(self, config: Config, power_w: float, tok_per_s: float) -> None:
        self.full_dyn_power = max(1e-6, (power_w - self.idle) / self._scale(config, "power_scale"))
        self.full_tps = tok_per_s / self._scale(config, "speed_scale")

    def learn(self, config: Config, obs: Observation) -> None:
        """Refine the full power and speed estimates from an unthrottled reading."""
        if obs.throttled or self.full_dyn_power is None:
            return
        dyn = max(1e-6, (obs.power_w - self.idle) / self._scale(config, "power_scale"))
        tps = obs.tok_per_s / self._scale(config, "speed_scale")
        self.full_dyn_power += self.alpha * (dyn - self.full_dyn_power)
        self.full_tps += self.alpha * (tps - self.full_tps)

    def predict(self, config: Config) -> Tuple[float, float]:
        st = self.dataset.stats(config, include_throttled=False, min_n=self.min_stat_n)
        if st is not None:
            return st.mean_power, st.mean_tps
        return (
            self.idle + self.full_dyn_power * self._scale(config, "power_scale"),
            self.full_tps * self._scale(config, "speed_scale"),
        )

    # candidate generation ---------------------------------------------------
    def _all_configs(self) -> List[Config]:
        sizes = [len(k.levels) for k in self.knobs]
        total = 1
        for s in sizes:
            total *= s
        if total > 512:
            return []
        names = [k.name for k in self.knobs]
        return [dict(zip(names, combo)) for combo in itertools.product(*[range(s) for s in sizes])]

    def _neighbours_up(self, current: Config) -> List[Config]:
        out = []
        for k in sorted(self.knobs, key=lambda k: k.switch_cost):
            if current[k.name] < k.top:
                c = dict(current)
                c[k.name] += 1
                out.append(c)
        return out

    def clear_block(self) -> None:
        self._block_power = None
        self._block_until = -1.0
        self._coarse_block_idx = None

    def _blocked(self, power_w: float, now: float) -> bool:
        return (
            self._block_power is not None
            and now < self._block_until
            and power_w >= self._block_power - 1e-9
        )

    # decision -------------------------------------------------------------
    def propose(
        self, current: Config, assessment: Assessment, obs: Observation,
        params: ThermalParams, now: float,
    ) -> Tuple[Config, str]:
        if self.full_dyn_power is None:
            self.seed(current, obs.power_w, obs.tok_per_s)
        if obs.temp_c is None:
            return self._propose_coarse(current, assessment, obs, now)

        o = self.obj
        status = assessment.status
        cur_power, _ = self.predict(current)

        if status in ("violation", "thermal_pressure"):
            cur_scale = self._scale(current, "power_scale")
            pool = [c for c in self._all_configs() if self._scale(c, "power_scale") < cur_scale - 1e-12]
            if not pool:
                return current, "already at the lowest configuration"
            scored = []
            for c in pool:
                p, t = self.predict(c)
                scored.append((c, p, t, params.predict(obs.temp_c, p, o.horizon_s)))
            safe = [s for s in scored if s[3] <= o.limit_c]
            if safe:
                best = max(safe, key=lambda s: (round(s[2], 6), -s[1]))
                why = "back off: forecast %.1fC under limit %.1fC" % (best[3], o.limit_c)
            else:
                best = min(scored, key=lambda s: s[3])
                why = "back off: no safe configuration, taking the coolest (forecast %.1fC)" % best[3]
            self._block_power = cur_power
            self._block_until = now + self.tabu_s
            return best[0], why

        if status == "headroom":
            options = []
            for c in self._neighbours_up(current):
                p, t = self.predict(c)
                if self._blocked(p, now):
                    continue
                if params.steady_state(p) <= o.limit_c:
                    options.append((c, p, t))
            if options:
                best = max(options, key=lambda s: s[2])
                return best[0], "step up: steady state %.1fC under limit %.1fC" % (
                    params.steady_state(best[1]), o.limit_c)
            return current, "headroom but no safe step up"

        return current, "hold"

    def _primary_knob(self) -> Knob:
        return sorted(self.knobs, key=lambda k: k.switch_cost)[0]

    def _propose_coarse(self, current: Config, assessment: Assessment, obs: Observation, now: float) -> Tuple[Config, str]:
        """Rule based fallback for devices that only report a coarse state."""
        knob = self._primary_knob()
        idx = current[knob.name]
        if assessment.status in ("violation", "thermal_pressure"):
            step = 2 if assessment.status == "violation" else 1
            new = max(0, idx - step)
            if new == idx:
                return current, "already at the lowest level"
            self._coarse_block_idx = idx
            self._block_until = now + self.tabu_s
            c = dict(current)
            c[knob.name] = new
            return c, "coarse back off at state " + obs.thermal_state
        if assessment.status == "headroom":
            if (self._coarse_block_idx is not None and now < self._block_until
                    and idx + 1 >= self._coarse_block_idx):
                return current, "coarse headroom but recently backed off from that level"
            c = dict(current)
            c[knob.name] = min(knob.top, idx + 1)
            return c, "coarse step up at state " + obs.thermal_state
        return current, "hold"
