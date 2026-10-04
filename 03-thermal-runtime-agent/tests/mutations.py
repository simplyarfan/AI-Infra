"""Pytest plugin that deliberately breaks the framework, selected by the MUT
environment variable. Used by scripts/mutation_check.py to show the tests can
actually fail. Not a test module."""
import os

MUT = os.environ.get("MUT", "")

import thermal_agent.backends.nvidia as nvidia
import thermal_agent.evaluation as evaluation
import thermal_agent.observer as observer
import thermal_agent.optimizer as optimizer
import thermal_agent.predictor as predictor
import thermal_agent.surrogate as surrogate
import thermal_agent.transformer as transformer
from thermal_agent.evaluation import Assessment

if MUT == "agent_never_acts":
    transformer.Transformer.apply = lambda self, target, now: (False, "mutated")

elif MUT == "evaluator_ignores_heat":
    evaluation.EvaluationAgent.assess = (
        lambda self, obs, params, at_top, flapping=False: Assessment("ok", False, [], None))

elif MUT == "model_never_fitted":
    observer.Observer._maybe_refit = lambda self: None

elif MUT == "drift_never_detected":
    transformer.DeploymentMonitor.check = lambda self, err: False

elif MUT == "step_up_jumps_to_top":
    def _jump(self, current):
        out = []
        for k in self.knobs:
            if current[k.name] < k.top:
                c = dict(current)
                c[k.name] = k.top
                out.append(c)
        return out
    predictor.ConfigPredictor._neighbours_up = _jump

elif MUT == "coarse_state_ignored":
    evaluation.EvaluationAgent._assess_coarse = (
        lambda self, obs, at_top: Assessment("ok", False, [], None))

elif MUT == "software_cap_counted_as_throttle":
    nvidia.HW_SLOWDOWN_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80

elif MUT == "back_off_one_level_only":
    _orig = predictor.ConfigPredictor.propose

    def _one_level(self, current, assessment, obs, params, now, can_raise=None):
        target, why = _orig(self, current, assessment, obs, params, now, can_raise)
        if sum(target.values()) < sum(current.values()):
            target = {k: max(target[k], current[k] - 1) for k in current}
        return target, why
    predictor.ConfigPredictor.propose = _one_level

elif MUT == "lowering_waits_for_dwell":
    def _apply(self, target, now):
        changed, notes = False, []
        for name, level in target.items():
            act = self.actuators[name]
            cur = act.get_level()
            level = max(0, min(act.knob.top, level))
            if level == cur:
                continue
            dwell = self.min_dwell_s.get(act.knob.switch_cost, 0.0)
            last = self._last_change.get(name)
            if last is not None and now - last < dwell:
                notes.append("held by dwell")
                continue
            act.set_level(level)
            self._last_change[name] = now
            changed = True
        return changed, "; ".join(notes)
    transformer.Transformer.apply = _apply

elif MUT == "advisor_unvetted":
    optimizer.OptimizationReasoningAgent._vet = lambda self, *a, **k: (True, "mutated")

elif MUT == "quality_floor_ignored":
    predictor.ConfigPredictor.quality = lambda self, config: 1.0

elif MUT == "surrogate_never_learns":
    surrogate.Surrogate.update = lambda self, *a, **k: None

elif MUT == "failed_exploration_not_penalised":
    optimizer.OptimizationReasoningAgent._manage_exploration = lambda self, mode, now, params: None

elif MUT == "warm_start_ignored":
    transformer.DeploymentMonitor.reuse = lambda self, ds, knobs, obj: None

elif MUT == "dataset_not_fed_to_the_predictor":
    predictor.ConfigPredictor.fit_from_dataset = lambda self, ds: 0

elif MUT == "gate_always_open":
    _assess = evaluation.EvaluationAgent.assess

    def _open(self, obs, params, at_top, flapping=False):
        a = _assess(self, obs, params, at_top, flapping)
        a.needs_optimization = True
        return a
    evaluation.EvaluationAgent.assess = _open

elif MUT == "held_raises_not_respected_by_4a":
    transformer.Transformer.can_raise = lambda self, name, now: True

elif MUT == "drift_does_not_reach_agent_4":
    optimizer.OptimizationReasoningAgent.reoptimize = lambda self, observer, err: None
