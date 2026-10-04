"""Agent 2 (baseline) and Agent 3 (evaluation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dataset import ConfigPerfDataset
from .knobs import Knob
from .objectives import Objectives
from .thermal_model import ThermalParams
from .types import Config, Observation, config_key, state_rank


# --- Agent 2: baseline -------------------------------------------------------
@dataclass
class BaselineReport:
    config: Config
    power_w: float
    tok_per_s: float
    temp_start_c: Optional[float]
    temp_end_c: Optional[float]


class BaselineAgent:
    """Runs the starting configuration for a few ticks and records what it
    gives. These numbers seed the predictor's estimate of full power and speed."""

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self._obs: List[Observation] = []
        self._config: Optional[Config] = None
        self.report: Optional[BaselineReport] = None

    @property
    def done(self) -> bool:
        return self.report is not None

    def add(self, obs: Observation, config: Config) -> None:
        if self.done:
            return
        if self._config is None:
            self._config = dict(config)
        if not obs.throttled:
            self._obs.append(obs)
        if len(self._obs) >= self.ticks:
            n = len(self._obs)
            temps = [o.temp_c for o in self._obs if o.temp_c is not None]
            self.report = BaselineReport(
                config=dict(self._config),
                power_w=sum(o.power_w for o in self._obs) / n,
                tok_per_s=sum(o.tok_per_s for o in self._obs) / n,
                temp_start_c=temps[0] if temps else None,
                temp_end_c=temps[-1] if temps else None,
            )


def known_good_config(dataset: ConfigPerfDataset, knobs: Sequence[Knob], objectives: Objectives,
                      min_samples: int = 30) -> Optional[Config]:
    """The "default / prior configuration" input to Agent 2, read from the
    Configuration-Performance Dataset.

    Returns the best configuration that earlier sessions ran for long enough to
    trust, never saw hardware throttling in, and finished under the temperature
    limit. Best means highest value of the objective function from Agent 1.
    None if there is no such configuration."""
    names = {k.name for k in knobs}
    groups: Dict[tuple, list] = {}
    for r in dataset.records:
        if set(r.config) == names:
            groups.setdefault(config_key(r.config), []).append(r)
    cands = []
    for key, rs in groups.items():
        if len(rs) < min_samples or any(r.throttled for r in rs):
            continue
        temps = [r.temp_c for r in rs if r.temp_c is not None][-10:]
        if not temps or sum(temps) / len(temps) > objectives.limit_c:
            continue
        cfg = dict(key)
        q = 1.0
        for k in knobs:
            q *= k.levels[cfg[k.name]].quality_scale
        cands.append((cfg, sum(r.tok_per_s for r in rs) / len(rs), q))
    if not cands:
        return None
    ref = max(c[1] for c in cands)
    best = max(cands, key=lambda c: c[1] / ref + objectives.quality_weight * c[2])
    return best[0]


# --- Agent 3: evaluation -----------------------------------------------------
@dataclass
class Assessment:
    status: str               # violation | thermal_pressure | headroom | ok
    needs_optimization: bool  # the "Is optimization needed?" gate on the slide
    reasons: List[str]
    forecast_c: Optional[float] = None

    @property
    def system_state(self) -> str:
        """The same verdict in the slide's vocabulary for Agent 3: underutilization,
        overload, instability, constraint violation (or stable)."""
        if self.status == "thermal_pressure" and any("flapping" in r for r in self.reasons):
            return "instability"
        return {"violation": "constraint_violation", "thermal_pressure": "overload",
                "headroom": "underutilization", "ok": "stable"}[self.status]


class EvaluationAgent:
    """Compares the live state with the objectives and decides whether the
    optimisation loop needs to run."""

    def __init__(self, objectives: Objectives) -> None:
        objectives.validate()
        self.obj = objectives

    def assess(self, obs: Observation, params: ThermalParams, at_top: bool, flapping: bool = False) -> Assessment:
        o = self.obj
        if obs.temp_c is None:
            return self._assess_coarse(obs, at_top)

        forecast = params.predict(obs.temp_c, obs.power_w, o.horizon_s)
        if obs.throttled or obs.temp_c >= o.max_temp_c:
            why = "hardware throttling active" if obs.throttled else "temperature at or above budget"
            return Assessment("violation", True, [why], forecast)
        if flapping:
            return Assessment("thermal_pressure", True, ["hardware throttling is flapping"], forecast)
        if forecast > o.limit_c:
            return Assessment(
                "thermal_pressure", True,
                ["forecast %.1fC in %.0fs exceeds limit %.1fC" % (forecast, o.horizon_s, o.limit_c)],
                forecast,
            )
        steady = params.steady_state(obs.power_w)
        if (not at_top) and steady < o.limit_c - o.raise_margin_c:
            return Assessment(
                "headroom", True,
                ["steady state %.1fC leaves room under limit %.1fC" % (steady, o.limit_c)],
                forecast,
            )
        return Assessment("ok", False, [], forecast)

    def _assess_coarse(self, obs: Observation, at_top: bool) -> Assessment:
        o = self.obj
        rank = state_rank(obs.thermal_state)
        if obs.throttled or rank >= state_rank("critical"):
            return Assessment("violation", True, ["critical thermal state or throttling"], None)
        if rank >= state_rank(o.coarse_down_state):
            return Assessment("thermal_pressure", True, ["thermal state " + obs.thermal_state], None)
        if rank <= state_rank(o.coarse_up_state) and not at_top:
            return Assessment("headroom", True, ["thermal state " + obs.thermal_state], None)
        return Assessment("ok", False, [], None)
