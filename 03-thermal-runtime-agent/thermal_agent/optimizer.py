"""Agent 4: optimization and reasoning (the orchestrator).

On the AutoOptAgent slide this agent sits between the evaluation gate and the
three optimization subagents. Its jobs are to interpret the system state,
decide whether and how to optimize, coordinate the subagents (4a predicts, 4b
applies, 4c observes), and manage the explore versus exploit trade-off.

Here that means:

  interpret   Agent 3's verdict becomes a mode: back_off (violation or
              overload), explore (underutilization, so try a cautious step up),
              or hold.
  coordinate  call 4a for a candidate, optionally let an advisor propose a
              different one, and hand the result to 4b.
  explore / exploit
              Exploring is a step up that might overheat. If one is followed
              soon by a back off, the exploration failed and the block on
              returning to that power level is doubled, up to a cap. A long calm
              stretch resets it. A drift event also resets it, because what was
              learned about the old conditions no longer applies.
  advisor     An optional reasoning component, for example an LLM running on a
              slower timescale or off the device, can suggest a configuration.
              Nothing it says is applied without passing the same safety rules
              as 4a's own proposals. The fast loop stays safe whatever the
              advisor says.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from .evaluation import Assessment
from .observer import Observer
from .predictor import ConfigPredictor
from .thermal_model import ThermalParams
from .types import Config, Observation


class Advisor(Protocol):
    def advise(self, summary: Dict) -> Optional[Config]:
        """Return a full configuration to try instead of 4a's candidate, or None."""


@dataclass
class Plan:
    mode: str                       # back_off | explore | hold
    target: Config
    note: str
    proposed_by: str = "4a"         # "4a" or "advisor"
    vetoed: Optional[str] = None    # why an advisor proposal was rejected, if one was


class OptimizationReasoningAgent:
    def __init__(self, predictor: ConfigPredictor, advisor: Optional[Advisor] = None,
                 explore_window_s: float = 120.0, calm_s: float = 300.0,
                 max_tabu_scale: float = 8.0) -> None:
        self.predictor = predictor
        self.advisor = advisor
        self.explore_window_s = explore_window_s
        self.calm_s = calm_s
        self.max_tabu_scale = max_tabu_scale
        self._last_explore_t: Optional[float] = None
        self._last_trouble_t: float = 0.0
        self.failed_explorations = 0
        self.advisor_accepted = 0
        self.advisor_vetoed = 0

    # interpret --------------------------------------------------------------
    @staticmethod
    def mode_for(assessment: Assessment) -> str:
        if assessment.status in ("violation", "thermal_pressure"):
            return "back_off"
        if assessment.status == "headroom":
            return "explore"
        return "hold"

    # coordinate ---------------------------------------------------------------
    def plan(self, current: Config, assessment: Assessment, obs: Observation,
             params: ThermalParams, now: float, can_raise=None) -> Plan:
        mode = self.mode_for(assessment)
        self._manage_exploration(mode, now, params)
        target, note = self.predictor.propose(current, assessment, obs, params, now, can_raise)
        plan = Plan(mode, target, note, "4a")
        if self.advisor is not None and assessment.needs_optimization:
            summary = {
                "mode": mode, "system_state": assessment.system_state, "reasons": list(assessment.reasons),
                "temp_c": obs.temp_c, "forecast_c": assessment.forecast_c, "thermal_state": obs.thermal_state,
                "current": dict(current), "candidate": dict(target),
            }
            proposal = self.advisor.advise(summary)
            if proposal is not None and proposal != target:
                ok, why = self._vet(current, proposal, target, mode, obs, params, now, can_raise)
                if ok:
                    self.advisor_accepted += 1
                    plan = Plan(mode, dict(proposal), "advisor: " + why, "advisor")
                else:
                    self.advisor_vetoed += 1
                    plan.vetoed = why
        return plan

    def note_applied(self, plan: Plan, applied: bool, now: float) -> None:
        if applied and plan.mode == "explore":
            self._last_explore_t = now

    def reoptimize(self, observer: Observer, rollout_error_c: float) -> None:
        """Called when Agent 8 reports that conditions changed."""
        observer.correct_ambient(rollout_error_c)
        self.predictor.clear_block()
        self.predictor.tabu_scale = 1.0
        self._last_explore_t = None
        self._last_trouble_t = 0.0

    # explore / exploit ----------------------------------------------------------
    def _manage_exploration(self, mode: str, now: float, params: ThermalParams) -> None:
        # An exploration needs about three time constants for its heat to show up,
        # so a back off inside that window counts as the exploration failing.
        window = max(self.explore_window_s, 3.0 * params.tau_s)
        if mode == "back_off":
            self._last_trouble_t = now
            if self._last_explore_t is not None and now - self._last_explore_t <= window:
                self.failed_explorations += 1
                self.predictor.tabu_scale = min(self.max_tabu_scale, self.predictor.tabu_scale * 2.0)
            self._last_explore_t = None
        elif now - self._last_trouble_t >= self.calm_s:
            self.predictor.tabu_scale = 1.0

    # safety vetting of advisor proposals ---------------------------------------------
    def _vet(self, current: Config, proposal: Config, candidate: Config, mode: str,
             obs: Observation, params: ThermalParams, now: float, can_raise=None):
        p = self.predictor
        names = {k.name for k in p.knobs}
        if set(proposal) != names:
            return False, "proposal does not cover every knob"
        for k in p.knobs:
            if not 0 <= proposal[k.name] <= k.top:
                return False, "level out of range for " + k.name
        if p.quality(proposal) < p.obj.min_quality:
            return False, "below the quality floor"
        if can_raise is not None and any(
                proposal[k.name] > current[k.name] and not can_raise(k.name) for k in p.knobs):
            return False, "a raise is held by the dwell time"
        if obs.temp_c is None:
            return False, "no temperature to vet against"
        cur_p, _ = p.predict(current)
        new_p, _ = p.predict(proposal)
        if new_p <= cur_p - 1e-9:
            # Cooling move. It must be at least as cool a forecast as 4a's own pick.
            horizon = p.obj.horizon_s
            f_new = params.predict(obs.temp_c, new_p, horizon)
            f_cand = params.predict(obs.temp_c, p.predict(candidate)[0], horizon)
            if mode != "back_off":
                return False, "cooling move while no back off is needed"
            if f_new <= p.obj.limit_c or f_new <= f_cand + 1e-9:
                return True, "cooling move, forecast %.1fC" % f_new
            return False, "cooling move forecasts %.1fC, worse than the safe pick" % f_new
        # Anything that draws more power is exploration and gets the strict rule.
        if mode != "explore":
            return False, "raising power while not in an exploring state"
        changed = [k.name for k in p.knobs if proposal[k.name] != current[k.name]]
        if len(changed) != 1 or proposal[changed[0]] != current[changed[0]] + 1:
            return False, "exploration may only raise one knob by one level"
        if p._blocked(new_p, now):
            return False, "blocked after a recent back off"
        if params.steady_state(new_p) > p.obj.limit_c:
            return False, "steady state %.1fC is over the limit" % params.steady_state(new_p)
        return True, "safe single step up"
