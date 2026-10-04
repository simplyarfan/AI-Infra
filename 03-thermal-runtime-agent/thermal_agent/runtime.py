"""The AutoOptAgent lifecycle, instantiated for the thermal use case.

The slide's use case box is "LLM Inference KV Cache Optimization". This is the
same lifecycle with a different use case: keep an edge device under its thermal
budget while the model runs. Mapping (my reading of the slide, to be checked
against the filing):

    1   Objectives and constraints  objectives.Objectives (adds a thermal budget, a quality floor)
    2   Baseline                    evaluation.BaselineAgent (starts from a prior configuration)
    3   Evaluation + gate           evaluation.EvaluationAgent ("is optimization needed?")
    4   Optimization and reasoning  optimizer.OptimizationReasoningAgent (orchestrator)
    4a  Configuration prediction    predictor.ConfigPredictor over surrogate.Surrogate
    4b  Transformation              transformer.Transformer (applies knob changes)
    4c  Observability and feedback  observer.Observer (records, identifies the thermal model)
    8   Deployment and monitoring   transformer.DeploymentMonitor (drift, reuse)
    DB  Configuration-Performance   dataset.ConfigPerfDataset

One call to tick() is one pass around the loop, and the Decision it returns
records which stages ran, so the path through the slide can be read off:

    4c observe -> 2 baseline -> 8 monitor -> 3 evaluate -> gate
        -> no:  8 hold
        -> yes: 4 plan -> 4a predict -> 4b apply -> (4c on the next tick)

The loop is deterministic and has no sleeps, so a simulator can drive it step by
step. run_realtime() wraps it for real devices.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .dataset import ConfigPerfDataset
from .evaluation import BaselineAgent, EvaluationAgent
from .objectives import Objectives
from .observer import Observer
from .optimizer import Advisor, OptimizationReasoningAgent
from .predictor import ConfigPredictor
from .thermal_model import ThermalParams
from .transformer import DeploymentMonitor, Transformer
from .types import Config


@dataclass
class AgentConfig:
    objectives: Objectives = field(default_factory=Objectives)
    prior: ThermalParams = ThermalParams(25.0, 8.0, 60.0)
    baseline_ticks: int = 5
    idle_power_w: float = 0.0
    min_dwell_s: Dict[int, float] = field(default_factory=lambda: {0: 8.0, 1: 30.0, 2: 120.0})
    tabu_s: float = 90.0
    window_s: float = 120.0
    assume_cold_start: bool = True
    drift_threshold_c: float = 1.0
    drift_patience: int = 3
    # Start from what an earlier session in the dataset says worked (Agent 8 reuse,
    # Agent 2 prior configuration) and train the surrogate on its records.
    warm_start: bool = False
    # Optional reasoning component for Agent 4. Everything it proposes is vetted.
    advisor: Optional[Advisor] = None


@dataclass
class Decision:
    t_s: float
    status: str
    reasons: List[str]
    before: Config
    after: Config
    applied: bool
    note: str
    forecast_c: Optional[float]
    drift: bool
    params: ThermalParams
    stages: List[str] = field(default_factory=list)
    mode: str = ""
    system_state: str = ""
    proposed_by: str = ""
    advisor_vetoed: Optional[str] = None

    def trace(self) -> str:
        return " -> ".join(self.stages)


class ThermalAgent:
    def __init__(self, telemetry, actuators: Sequence, config: Optional[AgentConfig] = None,
                 dataset: Optional[ConfigPerfDataset] = None) -> None:
        self.cfg = config or AgentConfig()
        self.telemetry = telemetry
        self.dataset = dataset if dataset is not None else ConfigPerfDataset()
        knobs = [a.knob for a in actuators]
        self.knobs = knobs
        # 4c, 2, 3, 4a, 4, 4b, 8
        self.observer = Observer(self.dataset, self.cfg.prior, self.cfg.window_s,
                                 self.cfg.assume_cold_start)
        self.baseline = BaselineAgent(self.cfg.baseline_ticks)
        self.evaluator = EvaluationAgent(self.cfg.objectives)
        self.predictor = ConfigPredictor(knobs, self.cfg.objectives, self.dataset,
                                         self.cfg.idle_power_w, self.cfg.tabu_s)
        self.optimizer = OptimizationReasoningAgent(self.predictor, self.cfg.advisor)
        self.transformer = Transformer(actuators, self.cfg.min_dwell_s)
        self.monitor = DeploymentMonitor(self.cfg.drift_threshold_c, self.cfg.drift_patience)
        self.decisions: List[Decision] = []
        self.reoptimizations = 0
        self.warm_started: Optional[Config] = None
        if self.cfg.warm_start and self.dataset.records:
            self.predictor.fit_from_dataset(self.dataset)
            start = self.monitor.reuse(self.dataset, knobs, self.cfg.objectives)
            if start is not None:
                for a in actuators:
                    a.set_level(start[a.knob.name])
                self.warm_started = start

    def _at_top(self, config: Config) -> bool:
        return all(config[k.name] == k.top for k in self.knobs)

    def tick(self) -> Decision:
        stages: List[str] = []
        obs = self.telemetry.read()
        before = self.transformer.current()

        # 4c: observe, record the (x, y) pair, keep the thermal model current.
        self.observer.record(obs, before)
        stages.append("4c observe")

        # 2: baseline on the starting (default or prior) configuration.
        was_done = self.baseline.done
        self.baseline.add(obs, before)
        if self.baseline.done and not was_done:
            rep = self.baseline.report
            self.predictor.seed(rep.config, rep.power_w, rep.tok_per_s)
        elif self.baseline.done:
            self.predictor.learn(before, obs)
        else:
            stages.append("2 baseline")

        # 8: monitor. Drift means the world changed, which triggers re-optimization.
        params = self.observer.params()
        err = self.observer.rollout_error()
        drift = self.monitor.check(err)
        stages.append("8 monitor")
        if drift:
            self.optimizer.reoptimize(self.observer, err)
            self.reoptimizations += 1
            params = self.observer.params()
            stages.append("8 drift -> re-optimize")

        # 3: evaluate against the objectives, then the gate.
        assessment = self.evaluator.assess(obs, params, self._at_top(before), self.observer.flapping())
        stages.append("3 evaluate (%s)" % assessment.system_state)
        gate = assessment.needs_optimization and (self.baseline.done or assessment.status == "violation")
        stages.append("gate: " + ("optimization needed" if gate else "not needed"))

        applied, note = False, "baseline window" if not self.baseline.done else "hold"
        after = dict(before)
        mode, proposed_by, vetoed = "hold", "", None
        if gate:
            # 4 coordinates 4a (predict) and 4b (apply).
            plan = self.optimizer.plan(before, assessment, obs, params, obs.t_s,
                                       lambda name: self.transformer.can_raise(name, obs.t_s))
            mode, proposed_by, vetoed = plan.mode, plan.proposed_by, plan.vetoed
            stages.append("4 plan (%s)" % plan.mode)
            stages.append("4a predict" if plan.proposed_by == "4a" else "4a predict, advisor chosen")
            applied, hold = self.transformer.apply(plan.target, obs.t_s)
            after = self.transformer.current()
            self.optimizer.note_applied(plan, applied, obs.t_s)
            note = plan.note + (" (" + hold + ")" if hold else "")
            stages.append("4b apply" if applied else "4b no change")
        else:
            stages.append("8 deploy and hold")

        d = Decision(
            t_s=obs.t_s, status=assessment.status, reasons=assessment.reasons, before=before,
            after=after, applied=applied, note=note, forecast_c=assessment.forecast_c,
            drift=drift, params=params, stages=stages, mode=mode,
            system_state=assessment.system_state, proposed_by=proposed_by, advisor_vetoed=vetoed,
        )
        self.decisions.append(d)
        return d

    def run_realtime(self, duration_s: float, interval_s: float = 1.0,
                     sleep: Callable[[float], None] = time.sleep) -> None:
        end = time.time() + duration_s
        while time.time() < end:
            self.tick()
            sleep(interval_s)
