"""The runtime loop that ties the agents together.

Mapping to the AutoOptAgent diagram (my reading of the slide, to be checked
against the filing):

    Agent 1  Objectives and constraints  objectives.Objectives (adds a thermal budget)
    Agent 2  Baseline                    evaluation.BaselineAgent
    Agent 3  Evaluation                  evaluation.EvaluationAgent
    Agent 4a Configuration prediction    predictor.ConfigPredictor
    Agent 4b Transformation              transformer.Transformer
    Agent 4c Observability and feedback  observer.Observer
    Agent 8  Deployment and monitoring   transformer.DeploymentMonitor
    Dataset  Configuration-Performance   dataset.ConfigPerfDataset

One call to tick() is one pass around the loop: observe, record, evaluate,
predict, apply. The loop is deterministic and has no sleeps, so a simulator can
drive it step by step. run_realtime() wraps it for real devices.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .dataset import ConfigPerfDataset
from .evaluation import Assessment, BaselineAgent, EvaluationAgent
from .objectives import Objectives
from .observer import Observer
from .predictor import ConfigPredictor
from .thermal_model import ThermalParams
from .transformer import DeploymentMonitor, Transformer
from .types import Config, Observation


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


class ThermalAgent:
    def __init__(self, telemetry, actuators: Sequence, config: Optional[AgentConfig] = None,
                 dataset: Optional[ConfigPerfDataset] = None) -> None:
        self.cfg = config or AgentConfig()
        self.telemetry = telemetry
        self.dataset = dataset if dataset is not None else ConfigPerfDataset()
        knobs = [a.knob for a in actuators]
        self.observer = Observer(self.dataset, self.cfg.prior, self.cfg.window_s,
                                 self.cfg.assume_cold_start)
        self.baseline = BaselineAgent(self.cfg.baseline_ticks)
        self.evaluator = EvaluationAgent(self.cfg.objectives)
        self.predictor = ConfigPredictor(knobs, self.cfg.objectives, self.dataset,
                                         self.cfg.idle_power_w, self.cfg.tabu_s)
        self.transformer = Transformer(actuators, self.cfg.min_dwell_s)
        self.monitor = DeploymentMonitor(self.cfg.drift_threshold_c, self.cfg.drift_patience)
        self.knobs = knobs
        self.decisions: List[Decision] = []
        self.reoptimizations = 0

    def _at_top(self, config: Config) -> bool:
        return all(config[k.name] == k.top for k in self.knobs)

    def tick(self) -> Decision:
        obs = self.telemetry.read()
        before = self.transformer.current()
        self.observer.record(obs, before)

        was_done = self.baseline.done
        self.baseline.add(obs, before)
        if self.baseline.done and not was_done:
            rep = self.baseline.report
            self.predictor.seed(rep.config, rep.power_w, rep.tok_per_s)
        elif self.baseline.done:
            self.predictor.learn(before, obs)

        params = self.observer.params()
        drift = self.monitor.check(self.observer.rollout_error())
        if drift:
            self.observer.correct_ambient(self.observer.rollout_error())
            self.predictor.clear_block()
            self.reoptimizations += 1
            params = self.observer.params()

        assessment = self.evaluator.assess(obs, params, self._at_top(before), self.observer.flapping())

        applied, note = False, "baseline window"
        after = dict(before)
        if self.baseline.done or assessment.status == "violation":
            target, why = self.predictor.propose(before, assessment, obs, params, obs.t_s)
            applied, hold = self.transformer.apply(target, obs.t_s)
            after = self.transformer.current()
            note = why + (" (" + hold + ")" if hold else "")

        d = Decision(
            t_s=obs.t_s, status=assessment.status, reasons=assessment.reasons, before=before,
            after=after, applied=applied, note=note, forecast_c=assessment.forecast_c,
            drift=drift, params=params,
        )
        self.decisions.append(d)
        return d

    def run_realtime(self, duration_s: float, interval_s: float = 1.0,
                     sleep: Callable[[float], None] = time.sleep) -> None:
        end = time.time() + duration_s
        while time.time() < end:
            self.tick()
            sleep(interval_s)
