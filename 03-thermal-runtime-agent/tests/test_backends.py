"""Hardware backends, tested with fake nvidia-smi output and fake runners.

These do not prove the code works on a real GPU. They prove the parsing, the
command construction, and the wiring into the agent are right, so that the
first real run fails only because of the hardware, not because of a typo.
"""
import time

import pytest

from thermal_agent import AgentConfig, Objectives, ThermalAgent, ThermalParams
from thermal_agent.backends.nvidia import (
    FIELDS_NEW, FIELDS_OLD, NvidiaPowerLimitActuator, NvidiaTelemetry, build_power_knob,
    parse_smi_line, state_from_temp,
)
from thermal_agent.backends.ollama import BackgroundWorkload


# ------------------------------------------------------------------- parsing
def test_parse_a_normal_line():
    s = parse_smi_line("63, 148.20, 0x0000000000000000")
    assert s["temp_c"] == 63.0 and s["power_w"] == pytest.approx(148.2)
    assert s["hw_throttled"] is False


@pytest.mark.parametrize("mask,expected", [
    ("0x0000000000000001", False),    # idle
    ("0x0000000000000004", False),    # software power cap: the agent's own cap
    ("0x0000000000000008", True),     # hardware slowdown
    ("0x0000000000000020", True),     # software thermal slowdown
    ("0x0000000000000040", True),     # hardware thermal slowdown
    ("0x0000000000000068", True),     # several at once
    ("0x0000000000000084", True),     # power brake plus software cap
])
def test_throttle_reason_bits(mask, expected):
    assert parse_smi_line("70, 150.0, " + mask)["hw_throttled"] is expected


def test_parse_not_available_fields_become_none():
    s = parse_smi_line("[N/A], [N/A], [N/A]")
    assert s["temp_c"] is None and s["power_w"] is None and s["hw_throttled"] is False


def test_parse_rejects_a_malformed_line():
    with pytest.raises(ValueError):
        parse_smi_line("63, 148.2")


def test_state_from_temp_thresholds():
    assert state_from_temp(50) == "nominal"
    assert state_from_temp(72) == "fair"
    assert state_from_temp(82) == "serious"
    assert state_from_temp(90) == "critical"


# ----------------------------------------------------------------- telemetry
class FakeSmi:
    """Stands in for nvidia-smi. Temperature rises on every query."""

    def __init__(self, new_field_supported=True, start=40.0, rise=1.5, power=180.0):
        self.new_ok = new_field_supported
        self.temp, self.rise, self.power = start, rise, power
        self.limit_commands = []
        self.queries = 0

    def __call__(self, args):
        if args and args[0] == "-pl":
            self.limit_commands.append(int(args[1]))
            return 0, "Power limit set", ""
        field = args[0].split("=", 1)[1]
        if field == FIELDS_NEW and not self.new_ok:
            return 2, "", "Field is not a valid field to query."
        self.queries += 1
        self.temp += self.rise
        return 0, "%.1f, %.1f, 0x0000000000000000" % (self.temp, self.power), ""


def counter_clock():
    state = {"t": 0.0}

    def clock():
        state["t"] += 1.0
        return state["t"]
    return clock


def test_telemetry_reads_an_observation():
    tel = NvidiaTelemetry(lambda: 55.0, runner=FakeSmi(), clock=counter_clock())
    o = tel.read()
    assert o.tok_per_s == 55.0 and o.power_w == pytest.approx(180.0)
    assert o.temp_c is not None and o.throttled is False


def test_telemetry_falls_back_to_the_older_field_name():
    runner = FakeSmi(new_field_supported=False)
    tel = NvidiaTelemetry(lambda: 1.0, runner=runner, clock=counter_clock())
    assert tel.read().temp_c is not None


def test_telemetry_raises_clearly_when_nvidia_smi_fails():
    with pytest.raises(RuntimeError):
        NvidiaTelemetry(lambda: 1.0, runner=lambda a: (1, "", "not found"), clock=counter_clock())


# ------------------------------------------------------------------ actuator
def test_power_knob_is_ordered_and_bounded():
    knob, watts = build_power_knob(default_w=180.0, min_w=100.0, max_w=200.0, n_levels=6)
    assert watts == sorted(watts)
    assert watts[0] >= 100 and watts[-1] <= 180
    assert knob.levels[-1].power_scale == pytest.approx(1.0)
    assert [l.power_scale for l in knob.levels] == sorted(l.power_scale for l in knob.levels)


def test_actuator_sends_the_right_command_and_clamps():
    knob, watts = build_power_knob(180.0, 100.0, 200.0, n_levels=4)
    runner = FakeSmi()
    act = NvidiaPowerLimitActuator(knob, watts, runner=runner)
    act.set_level(1)
    assert runner.limit_commands == [watts[1]] and act.get_level() == 1
    act.set_level(99)
    assert act.get_level() == knob.top


def test_actuator_reports_missing_privileges():
    knob, watts = build_power_knob(180.0, 100.0, 200.0, n_levels=4)
    act = NvidiaPowerLimitActuator(knob, watts, runner=lambda a: (1, "", "Insufficient Permissions"))
    with pytest.raises(RuntimeError, match="administrator"):
        act.set_level(0)


def test_actuator_needs_one_wattage_per_level():
    knob, watts = build_power_knob(180.0, 100.0, 200.0, n_levels=4)
    with pytest.raises(ValueError):
        NvidiaPowerLimitActuator(knob, watts[:-1])


# ------------------------------------------------------------ full wiring
def test_agent_lowers_the_gpu_power_limit_when_telemetry_runs_hot():
    runner = FakeSmi(start=40.0, rise=1.5, power=180.0)
    knob, watts = build_power_knob(180.0, 100.0, 200.0, n_levels=5)
    actuator = NvidiaPowerLimitActuator(knob, watts, runner=runner)
    telemetry = NvidiaTelemetry(lambda: 60.0, runner=runner, clock=counter_clock())
    cfg = AgentConfig(
        objectives=Objectives(max_temp_c=75.0, temp_margin_c=2.0, horizon_s=60.0),
        prior=ThermalParams(30.0, 0.3, 40.0),
    )
    agent = ThermalAgent(telemetry, [actuator], cfg)
    for _ in range(15):
        agent.tick()
    assert runner.limit_commands, "agent should have lowered the power limit"
    assert min(runner.limit_commands) < watts[-1]
    assert actuator.get_level() < knob.top


# ---------------------------------------------------------- background load
def test_background_workload_reports_recent_speed():
    def gen():
        time.sleep(0.005)
        return 10, 42.0

    w = BackgroundWorkload(gen)
    assert w.tok_per_s() == 0.0
    w.start()
    deadline = time.time() + 2.0
    while w.tok_per_s() == 0.0 and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert w.tok_per_s() == pytest.approx(42.0)


def test_background_workload_surfaces_errors_instead_of_dying_silently():
    def gen():
        raise RuntimeError("server down")

    w = BackgroundWorkload(gen)
    w.start()
    deadline = time.time() + 2.0
    while w.error is None and time.time() < deadline:
        time.sleep(0.01)
    w.stop()
    assert isinstance(w.error, RuntimeError)
