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

Predicted power and throughput come from a Bayesian surrogate (surrogate.py)
that starts at the knob priors and is corrected by every observation, so what
the agent learns about one configuration carries over to configurations it has
not tried. Candidates are ranked by the objective function from Agent 1:

    utility = tok_per_s / reference_tok_per_s + quality_weight * quality

subject to the thermal limit and the quality floor. With knobs that do not
change quality this reduces to "highest throughput that stays cool".
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import ConfigPerfDataset
from .evaluation import Assessment
from .knobs import Knob
from .objectives import Objectives
from .surrogate import Surrogate
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
        surrogate: Optional[Surrogate] = None,
    ) -> None:
        self.knobs = list(knobs)
        self.obj = objectives
        self.dataset = dataset
        self.idle = idle_power_w
        self.tabu_s = tabu_s
        # Agent 4 raises this after a failed exploration (see optimizer.py).
        self.tabu_scale = 1.0
        self.surrogate = surrogate or Surrogate(self.knobs)
        self._block_power: Optional[float] = None
        self._block_until: float = -1.0
        self._coarse_block_idx: Optional[int] = None

    # learning ---------------------------------------------------------------
    @property
    def seeded(self) -> bool:
        return self.surrogate.seeded

    def seed(self, config: Config, power_w: float, tok_per_s: float) -> None:
        self.surrogate.update(config, max(1e-6, power_w - self.idle), tok_per_s)

    def learn(self, config: Config, obs: Observation) -> None:
        """Add one unthrottled reading. Throttled readings describe the
        hardware governor, not the configuration, so they are left out."""
        if obs.throttled or not self.seeded:
            return
        self.surrogate.update(config, max(1e-6, obs.power_w - self.idle), obs.tok_per_s)

    def fit_from_dataset(self, dataset: ConfigPerfDataset) -> int:
        """Warm start from an earlier session's records. Returns how many were used."""
        n = 0
        for r in dataset.records:
            if r.throttled or set(r.config) != {k.name for k in self.knobs}:
                continue
            self.surrogate.update(r.config, max(1e-6, r.power_w - self.idle), r.tok_per_s)
            n += 1
        return n

    # prediction ---------------------------------------------------------------
    def predict(self, config: Config) -> Tuple[float, float]:
        """Predicted (total power in watts, tokens per second)."""
        dyn, tps = self.surrogate.predict(config)
        return self.idle + dyn, tps

    def quality(self, config: Config) -> float:
        q = 1.0
        for k in self.knobs:
            q *= k.levels[config[k.name]].quality_scale
        return q

    def _top_config(self) -> Config:
        return {k.name: k.top for k in self.knobs}

    def utility(self, config: Config, ref_tps: float) -> float:
        return self.predict(config)[1] / ref_tps + self.obj.quality_weight * self.quality(config)

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
        params: ThermalParams, now: float, can_raise=None,
    ) -> Tuple[Config, str]:
        """can_raise(knob_name) says whether the transformation agent would
        accept a raise of that knob right now. Candidates that need a held
        raise are skipped, so the configuration that was vetted is the one that
        gets applied, not a partly applied version of it."""
        if not self.seeded:
            self.seed(current, obs.power_w, obs.tok_per_s)
        if obs.temp_c is None:
            return self._propose_coarse(current, assessment, obs, now)

        o = self.obj
        status = assessment.status
        cur_power, _ = self.predict(current)
        ref_tps = max(1e-9, self.predict(self._top_config())[1])

        if status in ("violation", "thermal_pressure"):
            cur_dyn = cur_power - self.idle
            def reachable(c: Config) -> bool:
                return can_raise is None or all(
                    c[k.name] <= current[k.name] or can_raise(k.name) for k in self.knobs)

            cooler = [c for c in self._all_configs()
                      if self.predict(c)[0] - self.idle < cur_dyn * (1.0 - 1e-3) and reachable(c)]
            if not cooler:
                return current, "already at the lowest configuration"
            pool = [c for c in cooler if self.quality(c) >= o.min_quality]
            if not pool:
                return current, "no cooler configuration meets the quality floor"
            scored = []
            for c in pool:
                p, _t = self.predict(c)
                scored.append((c, p, self.utility(c, ref_tps), params.predict(obs.temp_c, p, o.horizon_s)))
            safe = [s for s in scored if s[3] <= o.limit_c]
            if safe:
                best = max(safe, key=lambda s: (round(s[2], 6), -s[1]))
                why = "back off: forecast %.1fC under limit %.1fC" % (best[3], o.limit_c)
            else:
                best = min(scored, key=lambda s: s[3])
                why = "back off: no safe configuration, taking the coolest (forecast %.1fC)" % best[3]
            self._block_power = cur_power
            self._block_until = now + self.tabu_s * self.tabu_scale
            return best[0], why

        if status == "headroom":
            cur_u = self.utility(current, ref_tps)
            options = []
            for c in self._neighbours_up(current):
                p, _t = self.predict(c)
                if self._blocked(p, now) or self.quality(c) < o.min_quality:
                    continue
                if can_raise is not None and not all(
                        c[k.name] <= current[k.name] or can_raise(k.name) for k in self.knobs):
                    continue
                u = self.utility(c, ref_tps)
                if params.steady_state(p) <= o.limit_c and u > cur_u + 1e-9:
                    options.append((c, p, u))
            if options:
                best = max(options, key=lambda s: s[2])
                return best[0], "step up: steady state %.1fC under limit %.1fC" % (
                    params.steady_state(best[1]), o.limit_c)
            return current, "headroom but no safe step up that improves the objective"

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
