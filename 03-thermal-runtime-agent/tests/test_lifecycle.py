"""The AutoOptAgent lifecycle, stage by stage.

Each section is one box or arrow on the slide: Agent 3's gate and states,
Agent 2's prior configuration, Agent 4's orchestration and its safety vetting of
an advisor, Agent 8's reuse, and the path a tick takes through all of them.
"""
import re

import pytest

from thermal_agent import (
    AgentConfig, Assessment, ConfigPerfDataset, ConfigPredictor, Objectives, Observation,
    OptimizationReasoningAgent, Plan, Record, SimulatedDevice, ThermalAgent, ThermalParams,
    kv_precision_knob, known_good_config, make_phone_device, PRESETS, quantization_knob,
    token_pacing_knob, run_session,
)
from thermal_agent.transformer import DeploymentMonitor

OBJ = Objectives()
PARAMS = ThermalParams(25.0, 8.0, 60.0)


def knobs():
    return [token_pacing_knob(), quantization_knob(), kv_precision_knob()]


TOP = {"token_pacing": 4, "quantization": 3, "kv_precision": 2}


# ------------------------------------------------------ Agent 3: states and gate
@pytest.mark.parametrize("status,needs,state", [
    ("violation", True, "constraint_violation"),
    ("thermal_pressure", True, "overload"),
    ("headroom", True, "underutilization"),
    ("ok", False, "stable"),
])
def test_agent3_reports_the_slides_system_states(status, needs, state):
    assert Assessment(status, needs, []).system_state == state


def test_agent3_flapping_is_instability():
    a = Assessment("thermal_pressure", True, ["hardware throttling is flapping"])
    assert a.system_state == "instability"


# ------------------------------------------------- Agent 2: prior configuration
def _rec(cfg, n, tps, temp, throttled=False, t0=0.0):
    return [Record(t_s=t0 + i, config=dict(cfg), tok_per_s=tps, power_w=3.0, temp_c=temp,
                   throttled=throttled) for i in range(n)]


def _dataset(*groups):
    ds = ConfigPerfDataset()
    for g in groups:
        for r in g:
            ds.add(r)
    return ds


def test_known_good_config_picks_a_config_that_ran_cool_and_unthrottled():
    cfg = dict(TOP, token_pacing=0)
    ds = _dataset(_rec(cfg, 40, 40.0, 48.0))
    assert known_good_config(ds, knobs(), OBJ) == cfg


def test_known_good_config_rejects_throttled_hot_or_thinly_sampled_configs():
    cfg = dict(TOP, token_pacing=0)
    assert known_good_config(_dataset(_rec(cfg, 40, 40.0, 48.0, throttled=True)), knobs(), OBJ) is None
    assert known_good_config(_dataset(_rec(cfg, 40, 40.0, 54.0)), knobs(), OBJ) is None
    assert known_good_config(_dataset(_rec(cfg, 5, 40.0, 48.0)), knobs(), OBJ) is None
    assert known_good_config(ConfigPerfDataset(), knobs(), OBJ) is None


def test_known_good_config_follows_the_objective_function():
    fast_lossy = {"token_pacing": 2, "quantization": 0, "kv_precision": 2}
    slow_clean = {"token_pacing": 1, "quantization": 3, "kv_precision": 2}
    ds = _dataset(_rec(fast_lossy, 40, 80.0, 48.0), _rec(slow_clean, 40, 40.0, 48.0))
    assert known_good_config(ds, knobs(), Objectives(quality_weight=0.1)) == fast_lossy
    assert known_good_config(ds, knobs(), Objectives(quality_weight=50.0)) == slow_clean


def test_known_good_config_ignores_records_from_a_different_set_of_knobs():
    ds = _dataset(_rec({"power_cap": 3}, 40, 40.0, 48.0))
    assert known_good_config(ds, knobs(), OBJ) is None


# ------------------------------------------------------ Agent 4: orchestration
def _predictor(power_at_top=6.0, objectives=OBJ):
    p = ConfigPredictor(knobs(), objectives, ConfigPerfDataset())
    p.seed(TOP, power_at_top, 77.0)
    return p


def _obs(temp=50.0, power=4.0):
    return Observation(t_s=100.0, power_w=power, tok_per_s=50.0, temp_c=temp)


def test_agent4_interprets_each_verdict_as_a_mode():
    m = OptimizationReasoningAgent.mode_for
    assert m(Assessment("violation", True, [])) == "back_off"
    assert m(Assessment("thermal_pressure", True, [])) == "back_off"
    assert m(Assessment("headroom", True, [])) == "explore"
    assert m(Assessment("ok", False, [])) == "hold"


def test_agent4_doubles_the_block_after_a_failed_exploration_and_caps_it():
    p = _predictor()
    a = OptimizationReasoningAgent(p, max_tabu_scale=4.0)
    cur = dict(TOP, token_pacing=2)
    pressure = Assessment("thermal_pressure", True, ["hot"], 60.0)
    explored = Plan("explore", cur, "step up")
    a.note_applied(explored, True, 10.0)
    a.plan(cur, pressure, _obs(), PARAMS, 20.0)          # trouble soon after exploring
    assert p.tabu_scale == 2.0 and a.failed_explorations == 1
    a.note_applied(explored, True, 30.0)
    a.plan(cur, pressure, _obs(), PARAMS, 40.0)
    assert p.tabu_scale == 4.0
    a.note_applied(explored, True, 50.0)
    a.plan(cur, pressure, _obs(), PARAMS, 60.0)
    assert p.tabu_scale == 4.0                           # capped


def test_agent4_does_not_penalise_a_back_off_unrelated_to_exploration():
    p = _predictor()
    a = OptimizationReasoningAgent(p)
    a.plan(dict(TOP, token_pacing=2), Assessment("thermal_pressure", True, ["hot"], 60.0), _obs(), PARAMS, 20.0)
    assert p.tabu_scale == 1.0 and a.failed_explorations == 0


def test_agent4_calm_stretch_resets_the_block_and_drift_does_too():
    p = _predictor()
    p.tabu_scale = 4.0
    a = OptimizationReasoningAgent(p, calm_s=100.0)
    a.plan(dict(TOP), Assessment("ok", False, []), _obs(40.0, 1.0), PARAMS, 500.0)
    assert p.tabu_scale == 1.0
    p.tabu_scale = 4.0

    class Obs:
        def correct_ambient(self, err):
            self.err = err
    o = Obs()
    a.reoptimize(o, 2.5)
    assert p.tabu_scale == 1.0 and o.err == 2.5


# ------------------------------------------- Agent 4: vetting an advisor
class Fixed:
    def __init__(self, cfg):
        self.cfg = cfg

    def advise(self, summary):
        return dict(self.cfg) if self.cfg is not None else None


def _plan_with(advisor, status, cur, temp, power_at_top=6.0, objectives=OBJ):
    p = _predictor(power_at_top, objectives)
    a = OptimizationReasoningAgent(p, advisor)
    forecast = 60.0 if status != "headroom" else 40.0
    ass = Assessment(status, True, ["x"], forecast)
    return a, a.plan(cur, ass, _obs(temp, power=p.predict(cur)[0]), PARAMS, 100.0)


def test_advisor_cannot_raise_power_while_the_device_is_overheating():
    cur = dict(TOP, token_pacing=2)
    a, plan = _plan_with(Fixed(TOP), "thermal_pressure", cur, 51.0)
    assert plan.proposed_by == "4a" and plan.vetoed and plan.target != TOP
    assert a.advisor_vetoed == 1


def test_advisor_cooling_proposal_is_accepted_when_it_forecasts_safe():
    cur = dict(TOP, token_pacing=2)
    coolest = {"token_pacing": 0, "quantization": 0, "kv_precision": 0}
    a, plan = _plan_with(Fixed(coolest), "thermal_pressure", cur, 51.0)
    assert plan.proposed_by == "advisor" and plan.target == coolest and a.advisor_accepted == 1


def test_advisor_proposal_below_the_quality_floor_is_vetoed():
    cur = dict(TOP, token_pacing=2)
    lossy = {"token_pacing": 0, "quantization": 0, "kv_precision": 0}
    a, plan = _plan_with(Fixed(lossy), "thermal_pressure", cur, 51.0, objectives=Objectives(min_quality=0.99))
    assert plan.proposed_by == "4a" and "quality" in plan.vetoed


def test_advisor_with_malformed_levels_is_vetoed():
    cur = dict(TOP, token_pacing=2)
    for bad in ({"token_pacing": 0}, dict(cur, token_pacing=99)):
        a, plan = _plan_with(Fixed(bad), "thermal_pressure", cur, 51.0)
        assert plan.proposed_by == "4a" and plan.vetoed


def test_advisor_single_step_up_is_accepted_only_when_the_steady_state_is_safe():
    # Two knobs below the top, so 4a's own pick and the advisor's can differ.
    cur = {"token_pacing": 2, "quantization": 3, "kv_precision": 1}
    kv_up = dict(cur, kv_precision=2)
    a, safe = _plan_with(Fixed(kv_up), "headroom", cur, 40.0, power_at_top=2.0)
    assert safe.proposed_by == "advisor" and safe.target == kv_up and a.advisor_accepted == 1
    _, unsafe = _plan_with(Fixed(kv_up), "headroom", cur, 40.0, power_at_top=6.0)
    assert unsafe.proposed_by == "4a" and "limit" in unsafe.vetoed


def test_advisor_cannot_jump_two_levels_when_exploring():
    cur = dict(TOP, token_pacing=1)
    _, plan = _plan_with(Fixed(dict(cur, token_pacing=3)), "headroom", cur, 40.0, power_at_top=2.0)
    assert plan.proposed_by == "4a" and "one level" in plan.vetoed


def test_a_reckless_advisor_never_makes_a_session_worse_than_having_none():
    class Reckless:
        def advise(self, summary):
            return {"token_pacing": 4, "quantization": 3, "kv_precision": 2}

    results = {}
    for name, adv in (("none", None), ("reckless", Reckless())):
        dev = make_phone_device("phone_quant_cools")
        agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ, advisor=adv))
        results[name] = run_session(dev, agent, 600.0, 1.0, OBJ.max_temp_c)[1]
    assert results["reckless"].time_throttled_s == 0
    assert results["reckless"].peak_temp_c <= results["none"].peak_temp_c + 1.0


# ---------------------------------------------- the path a tick takes
def test_every_tick_follows_a_legal_path_through_the_slide():
    dev = make_phone_device("phone_quant_flat")
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    for _ in range(200):
        dev.advance(1.0)
        d = agent.tick()
        assert d.stages[0] == "4c observe"
        i_mon = d.stages.index("8 monitor")
        i_eval = next(i for i, s in enumerate(d.stages) if s.startswith("3 evaluate"))
        i_gate = next(i for i, s in enumerate(d.stages) if s.startswith("gate"))
        assert i_mon < i_eval < i_gate
        if "gate: optimization needed" in d.stages:
            assert d.stages[i_gate + 1].startswith("4 plan")
            assert d.stages[i_gate + 2].startswith("4a predict")
            assert d.stages[-1] in ("4b apply", "4b no change")
            assert "8 deploy and hold" not in d.stages
        else:
            assert d.stages[-1] == "8 deploy and hold"
            assert not any(s.startswith("4") and s != "4c observe" for s in d.stages)


def test_a_stable_device_never_reaches_agent_4():
    # With a prior that matches the device there is nothing to optimize.
    dev = SimulatedDevice(PRESETS["tablet_cool"])
    agent = ThermalAgent(dev, dev.actuators,
                         AgentConfig(objectives=OBJ, prior=ThermalParams(25.0, 3.0, 180.0)))
    for _ in range(120):
        dev.advance(1.0)
        agent.tick()
    after_baseline = agent.decisions[10:]
    assert all("gate: not needed" in d.stages for d in after_baseline)
    assert all(not any(s.startswith("4 plan") for s in d.stages) for d in after_baseline)


def test_baseline_runs_before_the_first_optimization_unless_the_device_is_in_violation():
    dev = make_phone_device("phone_quant_flat")
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    stages = []
    for _ in range(8):
        dev.advance(1.0)
        stages.append(agent.tick().stages)
    assert all("2 baseline" in s for s in stages[:4])
    assert not any("2 baseline" in s for s in stages[6:])


def test_drift_is_routed_through_agent_8_to_agent_4():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    drifted = None
    for t in range(500):
        if t == 250:
            dev.set_ambient(35.0)
        dev.advance(1.0)
        d = agent.tick()
        if d.drift and drifted is None:
            drifted = d
    assert drifted is not None
    assert "8 drift -> re-optimize" in drifted.stages
    assert drifted.stages.index("8 monitor") < drifted.stages.index("8 drift -> re-optimize")


# --------------------------------------------------- Agent 8: reuse
def test_agent8_reuse_returns_the_known_good_configuration():
    cfg = dict(TOP, token_pacing=0)
    ds = _dataset(_rec(cfg, 40, 40.0, 48.0))
    assert DeploymentMonitor().reuse(ds, knobs(), OBJ) == cfg


def test_a_warm_started_agent_begins_where_the_last_session_settled():
    obj = OBJ
    dev1 = make_phone_device("phone_quant_flat")
    first = ThermalAgent(dev1, dev1.actuators, AgentConfig(objectives=obj))
    run_session(dev1, first, 900.0, 1.0, obj.max_temp_c)
    settled = dict(first.transformer.current())

    dev2 = make_phone_device("phone_quant_flat")
    second = ThermalAgent(dev2, dev2.actuators, AgentConfig(objectives=obj, warm_start=True),
                          dataset=first.dataset)
    assert second.warm_started is not None
    assert second.warm_started == second.transformer.current()
    assert second.predictor.surrogate.n_updates > 0           # trained on the old records
    _, warm = run_session(dev2, second, 900.0, 1.0, obj.max_temp_c)

    dev3 = make_phone_device("phone_quant_flat")
    cold = ThermalAgent(dev3, dev3.actuators, AgentConfig(objectives=obj))
    _, cold_m = run_session(dev3, cold, 900.0, 1.0, obj.max_temp_c)

    assert warm.time_throttled_s == 0
    assert warm.level_changes < cold_m.level_changes
    assert settled  # the first session did settle somewhere


def test_without_warm_start_the_dataset_is_not_reused():
    dev = make_phone_device("phone_quant_flat")
    ds = _dataset(_rec(dict(TOP, token_pacing=0), 40, 40.0, 48.0))
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ), dataset=ds)
    assert agent.warm_started is None
    assert agent.transformer.current() == TOP
