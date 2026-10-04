"""The simulated device. The agent is only as credible as the simulator it is
tested on, so the simulator's own physics is tested directly."""
import pytest

from thermal_agent import PRESETS, SimParams, SimulatedDevice, power_cap_knob


def run(device, seconds, dt=1.0):
    for _ in range(int(seconds / dt)):
        device.advance(dt)


def test_steady_state_matches_the_closed_form():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    dev.actuator.set_level(2)                       # 50 percent power cap
    run(dev, 1200)
    p = dev.params
    expected = p.t_amb + p.r_th * (p.p_idle + (p.p_full - p.p_idle) * 0.5)
    assert dev.temp == pytest.approx(expected, abs=0.05)
    assert not dev.hw_throttled


def test_full_power_triggers_repeated_hardware_throttling():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    peak = 0.0
    for _ in range(600):
        dev.advance(1.0)
        peak = max(peak, dev.temp)
    assert dev.throttle_events >= 3                  # sawtooth, not a single event
    assert peak < dev.params.t_throttle + 1.5        # throttling does cap the temperature


def test_throttling_costs_a_large_share_of_throughput():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    free, throttled = [], []
    for _ in range(600):
        dev.advance(1.0)
        (throttled if dev.hw_throttled else free).append(dev.tps)
    assert throttled and free
    assert sum(throttled) / len(throttled) < 0.75 * (sum(free) / len(free))


def test_hysteresis_keeps_throttle_on_until_temperature_drops():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    while not dev.hw_throttled:
        dev.advance(1.0)
    dev.advance(1.0)
    assert dev.hw_throttled
    assert dev.temp > dev.params.t_release


def test_decode_is_memory_bound_so_power_falls_faster_than_speed():
    full = SimulatedDevice(PRESETS["phone_severe"])
    full.advance(1.0)
    half = SimulatedDevice(PRESETS["phone_severe"])
    half.actuator.set_level(2)
    half.advance(1.0)
    dyn_full = full.power_w - full.params.p_idle
    dyn_half = half.power_w - half.params.p_idle
    assert dyn_half / dyn_full == pytest.approx(0.5, abs=0.01)
    assert half.tps / full.tps > 0.8


def test_tablet_never_throttles_at_full_power():
    dev = SimulatedDevice(PRESETS["tablet_cool"])
    run(dev, 1800)
    assert dev.throttle_events == 0
    assert dev.temp < dev.params.t_throttle


def test_coarse_only_device_hides_temperature_but_reports_state():
    dev = SimulatedDevice(PRESETS["phone_severe"], coarse_only=True)
    dev.advance(1.0)
    o = dev.read()
    assert o.temp_c is None
    assert o.thermal_state == "nominal"


def test_thermal_state_escalates_as_the_device_heats():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    seen = []
    for _ in range(300):
        dev.advance(1.0)
        s = dev.read().thermal_state
        if not seen or seen[-1] != s:
            seen.append(s)
    assert seen[0] == "nominal"
    assert "fair" in seen and "serious" in seen


def test_ambient_change_moves_the_steady_state():
    dev = SimulatedDevice(PRESETS["phone_severe"])
    dev.actuator.set_level(0)
    run(dev, 900)
    before = dev.temp
    dev.set_ambient(dev.params.t_amb + 10.0)
    run(dev, 900)
    assert dev.temp == pytest.approx(before + 10.0, abs=0.2)


def test_agent_knob_must_match_the_physical_levels():
    with pytest.raises(ValueError):
        SimulatedDevice(PRESETS["phone_severe"], agent_knob=power_cap_knob(fractions=(0.5, 1.0)))


def test_noise_is_reproducible_for_a_seed():
    params = SimParams(noise_tps=0.05, seed=11)
    a, b = SimulatedDevice(params), SimulatedDevice(params)
    seq_a, seq_b = [], []
    for _ in range(20):
        a.advance(1.0)
        b.advance(1.0)
        seq_a.append(a.tps)
        seq_b.append(b.tps)
    assert seq_a == seq_b
    assert len(set(seq_a)) > 1
