"""Agent 4a's engine (the surrogate) and the knobs it reasons about."""
import math

import pytest

from thermal_agent import (
    Objectives, Surrogate, kv_precision_knob, power_cap_knob, quantization_knob, token_pacing_knob,
)
from thermal_agent.surrogate import solve

TOP = {"token_pacing": 4, "quantization": 3, "kv_precision": 2}


def knobs():
    return [token_pacing_knob(), quantization_knob(), kv_precision_knob()]


# ------------------------------------------------------------------- the knobs
def test_quantization_fewer_bits_means_faster_but_lower_quality():
    k = quantization_knob()
    speeds = [l.speed_scale for l in k.levels]
    quality = [l.quality_scale for l in k.levels]
    assert speeds == sorted(speeds, reverse=True)       # top level (most bits) is the slowest
    assert quality == sorted(quality)                   # and the highest quality
    assert k.levels[k.top].speed_scale == pytest.approx(1.0)
    assert k.levels[k.top].quality_scale == 1.0


def test_quantization_speed_is_less_than_proportional_to_bits():
    k = quantization_knob()
    bits_ratio = 8.5 / 3.9
    assert 1.0 < k.levels[0].speed_scale < bits_ratio


def test_quantization_cooling_parameter_controls_the_power_prior():
    flat = quantization_knob(cooling=0.0)
    cool = quantization_knob(cooling=1.0)
    assert all(l.power_scale == pytest.approx(1.0) for l in flat.levels)
    assert cool.levels[0].power_scale < flat.levels[0].power_scale
    assert all(0 < l.power_scale <= 1.0 for l in cool.levels)


def test_token_pacing_prior_is_linear_in_duty_cycle():
    k = token_pacing_knob()
    for l, d in zip(k.levels, (0.4, 0.55, 0.7, 0.85, 1.0)):
        assert l.power_scale == pytest.approx(d) and l.speed_scale == pytest.approx(d)
    assert k.switch_cost == 0


def test_dvfs_power_cap_is_a_better_lever_than_pacing():
    # The reason the agent has to learn which knob cools efficiently.
    cap = power_cap_knob(fractions=(0.5, 1.0))
    pace = token_pacing_knob(duty=(0.5, 1.0))
    assert cap.levels[0].speed_scale > pace.levels[0].speed_scale


def test_model_knobs_cost_more_to_switch_than_cheap_knobs():
    assert quantization_knob().switch_cost > token_pacing_knob().switch_cost
    assert kv_precision_knob().switch_cost > token_pacing_knob().switch_cost


# --------------------------------------------------------------- the objective
def test_objectives_validate_quality_fields():
    Objectives(min_quality=0.9, quality_weight=1.0).validate()
    with pytest.raises(ValueError):
        Objectives(min_quality=1.5).validate()
    with pytest.raises(ValueError):
        Objectives(quality_weight=-1.0).validate()


# ------------------------------------------------------------------- surrogate
def test_surrogate_equals_the_priors_before_any_data():
    s = Surrogate(knobs())
    q4 = dict(TOP, quantization=1)
    p_top, t_top = s.predict(TOP)
    p_q4, t_q4 = s.predict(q4)
    lvl = quantization_knob().levels[1]
    assert p_q4 / p_top == pytest.approx(lvl.power_scale, rel=1e-6)
    assert t_q4 / t_top == pytest.approx(lvl.speed_scale, rel=1e-6)


def test_surrogate_pins_the_baseline_from_a_few_readings():
    s = Surrogate(knobs())
    for _ in range(5):
        s.update(TOP, 6.0, 77.0)
    p, t = s.predict(TOP)
    assert p == pytest.approx(6.0, rel=0.01) and t == pytest.approx(77.0, rel=0.01)


def test_surrogate_learns_that_a_knob_does_not_cool_when_the_data_says_so():
    s = Surrogate(knobs())
    for _ in range(5):
        s.update(TOP, 6.0, 77.0)
    q4 = dict(TOP, quantization=1)
    prior_power = s.predict(q4)[0]
    assert prior_power < 5.5                      # the prior expects some cooling
    for _ in range(12):
        s.update(q4, 6.0, 77.0 * 1.30)            # truth: same watts, 30 percent faster
    power_factor, speed_factor = s.effect("quantization", 1)
    assert power_factor == pytest.approx(1.0, abs=0.03)
    assert speed_factor == pytest.approx(1.30, abs=0.04)


def test_surrogate_learning_carries_over_to_unvisited_configurations():
    s = Surrogate(knobs())
    for _ in range(5):
        s.update(TOP, 6.0, 77.0)
    q4 = dict(TOP, quantization=1)
    for _ in range(12):
        s.update(q4, 6.0, 77.0 * 1.30)
    never_visited = dict(q4, token_pacing=1)      # pacing 55 percent on top of Q4
    p, _ = s.predict(never_visited)
    # Pacing prior is 0.55 and Q4 is now known not to cool, so about 3.3 W.
    assert p == pytest.approx(6.0 * 0.55, rel=0.08)


def test_surrogate_forgets_so_a_changed_device_is_followed():
    s = Surrogate(knobs(), forgetting=0.99)
    for _ in range(100):
        s.update(TOP, 6.0, 77.0)
    for _ in range(300):
        s.update(TOP, 4.5, 60.0)                  # e.g. the battery got low
    p, t = s.predict(TOP)
    assert p == pytest.approx(4.5, rel=0.05) and t == pytest.approx(60.0, rel=0.05)


def test_surrogate_ignores_nonpositive_readings():
    s = Surrogate(knobs())
    s.update(TOP, 0.0, 10.0)
    s.update(TOP, 5.0, 0.0)
    assert s.n_updates == 0 and not s.seeded


def test_linear_solver_and_its_failure_mode():
    x = solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert x == pytest.approx([1.0, 3.0])
    with pytest.raises(ValueError):
        solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
