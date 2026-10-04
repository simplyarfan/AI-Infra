"""End to end on a simulated phone whose levers are model level: token pacing,
quantization and KV cache precision. There is no platform power cap, which is
the situation on a real iPhone.

Two scenarios differ only in what the simulator's hidden physics does when the
model is quantized harder: either the watts fall as well (phone_quant_cools) or
only the speed rises (phone_quant_flat). The agent starts with the same priors in
both and has to cope with whichever world it is in.

Scope: simulator only. These show that the lifecycle finds a good configuration
in a world that obeys the model, and they state the cost of doing so honestly.
They say nothing yet about how a real phone responds to these knobs.
"""
import pytest

from thermal_agent import (
    AgentConfig, Objectives, ThermalAgent, make_phone_device, objective_value,
    run_model_level_comparison, run_session,
)

OBJ = Objectives()
REF_TPS = 77.0          # the simulated phone's base throughput at the top configuration
SCENARIOS = ["phone_quant_flat", "phone_quant_cools"]


@pytest.fixture(scope="module", params=SCENARIOS)
def run(request):
    return request.param, run_model_level_comparison(request.param, 900.0, objectives=OBJ)


def test_agent_never_triggers_hardware_throttling(run):
    _, res = run
    m = res["thermal_agent"][1]
    assert m.time_throttled_s == 0 and m.throttle_events == 0
    assert m.peak_temp_c <= OBJ.max_temp_c


def test_unmanaged_top_configuration_throttles_in_the_same_scenario(run):
    _, res = run
    m = res["fixed_top"][1]
    assert m.throttle_events >= 10 and m.time_throttled_s > 200 and m.time_over_budget_s > 0


def test_agent_lands_close_to_the_best_fixed_configuration_chosen_with_hindsight(run):
    _, res = run
    agent = objective_value(res["thermal_agent"][1], REF_TPS, OBJ)
    oracle = objective_value(res["best_fixed_with_hindsight"][1], REF_TPS, OBJ)
    assert res["best_fixed_with_hindsight"][1].time_throttled_s == 0
    assert agent >= 0.97 * oracle


def test_agent_does_not_thrash(run):
    _, res = run
    assert res["thermal_agent"][1].level_changes <= 12


def test_agent_beats_the_os_governor_only_when_a_model_knob_really_cools():
    flat = run_model_level_comparison("phone_quant_flat", 900.0, objectives=OBJ)
    cools = run_model_level_comparison("phone_quant_cools", 900.0, objectives=OBJ)
    # If quantization cools, staying under the budget is a net win in throughput.
    assert cools["thermal_agent"][1].mean_tps > cools["fixed_top"][1].mean_tps
    # If nothing cools efficiently, staying under the budget costs throughput.
    # The unmanaged phone averages more tokens per second (it throttles hard
    # but runs at full speed in between); the agent buys a cooler, steadier device.
    assert flat["thermal_agent"][1].tail_tps < flat["fixed_top"][1].mean_tps
    assert flat["thermal_agent"][1].peak_temp_c < flat["fixed_top"][1].peak_temp_c - 3.0


def test_agent_ends_on_the_same_configuration_as_the_hindsight_oracle_when_quant_is_flat():
    res = run_model_level_comparison("phone_quant_flat", 900.0, objectives=OBJ)
    assert res["thermal_agent"][0][-1]["config"] == res["best_fixed_with_hindsight"][0][-1]["config"]


def test_quality_floor_is_never_violated_even_to_avoid_heat():
    obj = Objectives(min_quality=0.99)
    dev = make_phone_device("phone_quant_flat")
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=obj))
    rows, m = run_session(dev, agent, 900.0, 1.0, obj.max_temp_c)
    assert min(r["quality"] for r in rows) >= 0.99
    assert m.time_throttled_s == 0


def test_quality_weight_changes_what_the_agent_gives_up():
    final = {}
    for w in (0.1, 50.0):
        obj = Objectives(quality_weight=w)
        dev = make_phone_device("phone_quant_cools")
        agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=obj))
        rows, m = run_session(dev, agent, 900.0, 1.0, obj.max_temp_c)
        final[w] = m
    assert final[50.0].mean_quality > final[0.1].mean_quality
    assert final[0.1].mean_tps > final[50.0].mean_tps


def test_agent_works_with_only_token_pacing_the_one_lever_any_ios_app_has():
    dev = make_phone_device("phone_quant_flat")
    agent = ThermalAgent(dev, dev.actuators[:1], AgentConfig(objectives=OBJ))
    rows, m = run_session(dev, agent, 900.0, 1.0, OBJ.max_temp_c)
    assert m.time_throttled_s == 0 and m.mean_quality == 1.0
    assert m.peak_temp_c <= OBJ.max_temp_c


def test_agent_works_from_the_coarse_thermal_state_alone_with_model_knobs():
    unmanaged = run_session(make_phone_device("phone_quant_flat"), None, 900.0, 1.0, OBJ.max_temp_c)[1]
    dev = make_phone_device("phone_quant_flat", coarse_only=True)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    m = run_session(dev, agent, 900.0, 1.0, OBJ.max_temp_c)[1]
    assert m.time_throttled_s < 0.1 * unmanaged.time_throttled_s
    assert all(d.forecast_c is None for d in agent.decisions)


def test_the_surrogate_ends_up_following_what_the_device_really_does():
    dev = make_phone_device("phone_quant_flat")
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    run_session(dev, agent, 900.0, 1.0, OBJ.max_temp_c)
    cfg = agent.transformer.current()
    pred_power, pred_tps = agent.predictor.predict(cfg)
    last = dev.read()
    assert pred_power == pytest.approx(last.power_w, rel=0.05)
    assert pred_tps == pytest.approx(last.tok_per_s, rel=0.05)


def test_runs_are_deterministic():
    a = run_model_level_comparison("phone_quant_flat", 300.0, objectives=OBJ)["thermal_agent"][1]
    b = run_model_level_comparison("phone_quant_flat", 300.0, objectives=OBJ)["thermal_agent"][1]
    assert a == b
