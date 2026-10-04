"""Agent 4b (transformation) and Agent 8 (deployment and monitoring)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from .dataset import ConfigPerfDataset
from .evaluation import known_good_config
from .objectives import Objectives
from .types import Config


class Transformer:
    """Applies a configuration through the actuators.

    Moves that lower intensity are applied immediately, because they make the
    device safer. Moves that raise intensity respect a minimum dwell time per
    knob, scaled by how expensive the knob is to switch, so the agent cannot
    thrash an expensive action such as moving a model between chips.
    """

    def __init__(self, actuators: Sequence, min_dwell_s: Mapping[int, float]) -> None:
        self.actuators = {a.knob.name: a for a in actuators}
        self.min_dwell_s = dict(min_dwell_s)
        self._last_change: Dict[str, float] = {}

    def current(self) -> Config:
        return {name: a.get_level() for name, a in self.actuators.items()}

    def can_raise(self, name: str, now: float) -> bool:
        """Whether a knob may be raised now, or is still inside its dwell time."""
        act = self.actuators[name]
        dwell = self.min_dwell_s.get(act.knob.switch_cost, 0.0)
        last = self._last_change.get(name)
        return last is None or now - last >= dwell

    def apply(self, target: Config, now: float) -> Tuple[bool, str]:
        changed = False
        notes = []
        for name, level in target.items():
            act = self.actuators[name]
            cur = act.get_level()
            level = max(0, min(act.knob.top, level))
            if level == cur:
                continue
            if level > cur:
                dwell = self.min_dwell_s.get(act.knob.switch_cost, 0.0)
                last = self._last_change.get(name)
                if last is not None and now - last < dwell:
                    notes.append("%s: raise held by dwell time" % name)
                    continue
            act.set_level(level)
            self._last_change[name] = now
            changed = True
        return changed, "; ".join(notes)


class DeploymentMonitor:
    """Agent 8: deployment and monitoring.

    On the slide this agent deploys the chosen configuration, monitors long
    term performance, and either reuses what it knows or triggers
    re-optimisation when conditions change.

      check()  watches how well the thermal model is predicting. If the error
               stays large for several ticks the world has changed (ambient
               temperature, a case, charging, another app) and it returns True,
               which is the trigger to re-optimise.
      reuse()  the other half: when a session starts, return the configuration
               the dataset says already worked, so the agent does not rediscover
               the same limits every launch.
    """

    def __init__(self, threshold_c: float = 1.0, patience: int = 3) -> None:
        self.threshold_c = threshold_c
        self.patience = patience
        self._streak = 0
        self.drift_events = 0

    def check(self, rollout_error_c: float) -> bool:
        if abs(rollout_error_c) > self.threshold_c:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            self._streak = 0
            self.drift_events += 1
            return True
        return False

    def reuse(self, dataset: ConfigPerfDataset, knobs: Sequence, objectives: Objectives) -> Optional[Config]:
        return known_good_config(dataset, knobs, objectives)
