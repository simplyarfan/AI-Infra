"""End to end scenarios: the agent closing the loop on a simulated device.

These are the tests that matter for the claim. Each one states a behaviour the
runtime controller is supposed to have, and checks it against a baseline run on
an identical simulated device.

Scope note: these run on a simulator, so they test that the control logic does
what it is designed to do on a device that obeys the model. They do not show
that a real phone or GPU obeys the model. That needs the hardware runs.
"""
import pytest

from thermal_agent import (
    AgentConfig, Knob, KnobLevel, Objectives, PRESETS, SimParams, SimulatedDevice,
    ThermalAgent, ThermalParams, power_cap_knob, run_comparison, run_session,
)
from thermal_agent.sim import DEFAULT_FRACTIONS

OBJ = Objectives()


def managed(params, duration=600.0, cfg=None, **device_kwargs):
    dev = SimulatedDevice(params, **device_kwargs)
    agent = ThermalAgent(dev, dev.actuators, cfg or AgentConfig(objectives=OBJ))
    rows, m = run_session(dev, agent, duration, 1.0, OBJ.max_temp_c)
    return dev, agent, rows, m


@pytest.fixture(scope="module")
def severe():
    return run_comparison("phone_severe", 600.0)


@pytest.fixture(scope="module")
def cool():
    return run_comparison("tablet_cool", 600.0)


# ------------------------------------------------------- hot device (the point)
def test_agent_avoids_hardware_throttling_entirely(severe):
    _, m = severe["thermal_agent"]
    assert m.time_throttled_s == 0
    assert m.throttle_events == 0
    assert m.peak_temp_c <= OBJ.max_temp_c


def test_unmanaged_full_power_does_throttle_in_the_same_scenario(severe):
    # Guards against the agent "winning" a scenario that was never hard.
    _, m = severe["fixed_full_power"]
    assert m.throttle_events >= 5
    assert m.time_throttled_s > 100
    assert m.time_over_budget_s > 0


def test_agent_sustains_more_throughput_than_unmanaged_full_power(severe):
    agent = severe["thermal_agent"][1]
    base = severe["fixed_full_power"][1]
    assert agent.mean_tps > 1.03 * base.mean_tps
    assert agent.tail_tps > 1.05 * base.tail_tps


def test_agent_uses_less_energy_per_token_than_unmanaged_full_power(severe):
    agent = severe["thermal_agent"][1]
    base = severe["fixed_full_power"][1]
    assert agent.j_per_token < 0.9 * base.j_per_token


def test_agent_matches_a_hand_tuned_static_cap_without_being_told_it(severe):
    agent = severe["thermal_agent"][1]
    static = severe["fixed_50_percent"][1]
    assert agent.mean_tps >= 0.97 * static.mean_tps


def test_agent_does_not_thrash(severe):
    _, m = severe["thermal_agent"]
    assert m.level_changes <= 6


# ---------------------------------------------------- cool device (do no harm)
def test_agent_does_not_hurt_a_device_that_never_overheats(cool):
    agent = cool["thermal_agent"][1]
    full = cool["fixed_full_power"][1]
    assert full.throttle_events == 0
    assert agent.throttle_events == 0
    assert agent.mean_tps >= 0.98 * full.mean_tps
    assert agent.tail_tps >= 0.99 * full.tail_tps


def test_agent_ends_back_at_full_power_on_a_cool_device():
    dev, agent, _rows, _m = managed(PRESETS["tablet_cool"], 600.0)
    assert dev.actuator.get_level() == dev.actuator.knob.top


# ------------------------------------------------------------ online learning
def test_agent_learns_the_thermal_model_online():
    _dev, agent, _rows, _m = managed(PRESETS["phone_severe"], 600.0)
    assert agent.observer.fitted
    p = agent.observer.params()
    assert p.r_th == pytest.approx(8.0, rel=0.25)
    assert p.tau_s == pytest.approx(60.0, rel=0.35)


def test_agent_copes_with_a_badly_wrong_prior(severe):
    wrong = AgentConfig(objectives=OBJ, prior=ThermalParams(25.0, 3.0, 200.0))
    _dev, _agent, _rows, m = managed(PRESETS["phone_severe"], 600.0, cfg=wrong)
    unmanaged = severe["fixed_full_power"][1]
    assert m.time_throttled_s < 0.1 * unmanaged.time_throttled_s
    assert m.mean_tps > unmanaged.mean_tps


def test_agent_recovers_from_wrong_power_priors(severe):
    # Power priors that are nonlinearly wrong (a square law instead of linear).
    right = power_cap_knob(fractions=DEFAULT_FRACTIONS, mem_frac=0.35)
    wrong = Knob("power_cap", tuple(
        KnobLevel(l.label, l.power_scale ** 2, l.speed_scale) for l in right.levels), 0)
    _dev, _agent, _rows, m = managed(PRESETS["phone_severe"], 900.0, agent_knob=wrong)
    unmanaged = severe["fixed_full_power"][1]
    assert m.peak_temp_c <= OBJ.max_temp_c + 2.0
    assert m.time_throttled_s < 0.25 * unmanaged.time_throttled_s
    assert m.tail_tps > unmanaged.tail_tps


# ------------------------------------------------------------ safety behaviour
def test_agent_backs_off_immediately_when_the_device_starts_hot():
    dev = SimulatedDevice(PRESETS["phone_severe"], start_temp=54.0)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    rows, m = run_session(dev, agent, 300.0, 1.0, OBJ.max_temp_c)
    first = agent.decisions[0]
    assert first.status == "violation"
    assert sum(first.after.values()) < sum(first.before.values())
    assert m.throttle_events == 0
    assert m.peak_temp_c < 55.0


def test_raising_the_configuration_only_ever_takes_single_steps():
    # Safe exploration: going up is one step on one knob at a time.
    _dev, agent, _rows, _m = managed(PRESETS["tablet_cool"], 600.0)
    ups = [d for d in agent.decisions if d.applied and sum(d.after.values()) > sum(d.before.values())]
    assert ups, "scenario should include at least one step up"
    for d in ups:
        deltas = [d.after[k] - d.before[k] for k in d.before]
        assert max(deltas) == 1 and sum(deltas) == 1


def test_backing_off_may_jump_several_levels_at_once():
    _dev, agent, _rows, _m = managed(PRESETS["phone_severe"], 120.0)
    downs = [d for d in agent.decisions if d.applied and sum(d.after.values()) < sum(d.before.values())]
    assert downs
    assert max(sum(d.before.values()) - sum(d.after.values()) for d in downs) >= 2


def test_every_decision_is_logged_and_the_dataset_is_populated():
    _dev, agent, rows, _m = managed(PRESETS["phone_severe"], 300.0)
    assert len(agent.decisions) == len(rows)
    assert len(agent.dataset.records) == len(rows)
    assert len(agent.dataset.configs()) >= 2
    assert all(d.note for d in agent.decisions)


# -------------------------------------------------------------- drift / change
def test_agent_adapts_when_the_ambient_temperature_jumps():
    params = PRESETS["phone_severe"]
    dev = SimulatedDevice(params)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    level_before = None
    throttled_after = 0
    for step in range(1200):
        if step == 400:
            level_before = dev.actuator.get_level()
            dev.set_ambient(params.t_amb + 10.0)
        dev.advance(1.0)
        agent.tick()
        if step >= 400 and dev.hw_throttled:
            throttled_after += 1
    assert agent.monitor.drift_events >= 1
    assert agent.reoptimizations >= 1
    assert dev.actuator.get_level() < level_before
    assert throttled_after <= 30


# ------------------------------------------------- phones without a temperature
def test_agent_works_from_the_coarse_thermal_state_alone(severe):
    # iOS style: no raw temperature, only nominal / fair / serious / critical.
    dev, _agent, _rows, m = managed(PRESETS["phone_severe"], 600.0, coarse_only=True)
    unmanaged = severe["fixed_full_power"][1]
    assert m.throttle_events == 0
    assert m.time_throttled_s == 0
    assert m.mean_tps > unmanaged.mean_tps


def test_coarse_mode_makes_no_use_of_a_temperature_it_does_not_have():
    dev = SimulatedDevice(PRESETS["phone_severe"], coarse_only=True)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=OBJ))
    for _ in range(60):
        dev.advance(1.0)
        d = agent.tick()
        assert d.forecast_c is None


# ------------------------------------------------------------------ robustness
def test_throughput_noise_does_not_cause_extra_switching():
    noisy = SimParams(noise_tps=0.03, seed=5)
    _dev, _agent, _rows, m = managed(noisy, 600.0)
    assert m.level_changes <= 8
    assert m.time_throttled_s == 0


def test_runs_are_deterministic():
    _a, _b, _c, m1 = managed(PRESETS["phone_severe"], 300.0)
    _d, _e, _f, m2 = managed(PRESETS["phone_severe"], 300.0)
    assert m1 == m2


def test_a_stricter_thermal_budget_trades_throughput_for_temperature():
    loose = Objectives(max_temp_c=53.0, temp_margin_c=1.0)
    strict = Objectives(max_temp_c=47.0, temp_margin_c=1.0)
    res = {}
    for name, obj in (("loose", loose), ("strict", strict)):
        dev = SimulatedDevice(PRESETS["phone_severe"])
        agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=obj))
        res[name] = run_session(dev, agent, 900.0, 1.0, obj.max_temp_c)[1]
    assert res["strict"].peak_temp_c < res["loose"].peak_temp_c
    assert res["strict"].mean_tps < res["loose"].mean_tps
    assert res["strict"].time_over_budget_s == 0
