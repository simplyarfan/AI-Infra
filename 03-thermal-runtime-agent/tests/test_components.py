"""Individual agent components: objectives, dataset, knobs, evaluation,
baseline, transformation, and the deployment monitor."""
import pytest

from thermal_agent import (
    AgentConfig, BaselineAgent, ConfigPerfDataset, DeploymentMonitor, EvaluationAgent, Knob,
    KnobLevel, Objectives, Observation, ThermalParams, Transformer, knob_from_sweep,
    power_cap_knob,
)
from thermal_agent.dataset import Record

PARAMS = ThermalParams(25.0, 8.0, 60.0)


# ---------------------------------------------------------------- objectives
def test_objectives_limit_applies_margin():
    assert Objectives(max_temp_c=53.0, temp_margin_c=1.0).limit_c == pytest.approx(52.0)


def test_objectives_validation_rejects_nonsense():
    with pytest.raises(ValueError):
        Objectives(temp_margin_c=-1.0).validate()
    with pytest.raises(ValueError):
        Objectives(horizon_s=0.0).validate()
    with pytest.raises(ValueError):
        Objectives(coarse_down_state="boiling").validate()


# ------------------------------------------------------------------- dataset
def rec(config, tps, power, temp=40.0, throttled=False, t=0.0):
    return Record(t_s=t, config=config, tok_per_s=tps, power_w=power, temp_c=temp, throttled=throttled)


def test_dataset_stats_average_per_configuration():
    ds = ConfigPerfDataset()
    ds.add(rec({"k": 1}, 60.0, 3.0))
    ds.add(rec({"k": 1}, 64.0, 3.4))
    ds.add(rec({"k": 2}, 80.0, 5.0))
    s = ds.stats({"k": 1})
    assert s.n == 2
    assert s.mean_tps == pytest.approx(62.0)
    assert s.mean_power == pytest.approx(3.2)
    assert s.j_per_token == pytest.approx(3.2 / 62.0)
    assert ds.stats({"k": 9}) is None


def test_dataset_excludes_throttled_readings_by_default():
    ds = ConfigPerfDataset()
    ds.add(rec({"k": 1}, 80.0, 5.0))
    ds.add(rec({"k": 1}, 40.0, 1.0, throttled=True))
    assert ds.stats({"k": 1}).mean_tps == pytest.approx(80.0)
    both = ds.stats({"k": 1}, include_throttled=True)
    assert both.mean_tps == pytest.approx(60.0)
    assert both.throttled_fraction == pytest.approx(0.5)


def test_dataset_min_n():
    ds = ConfigPerfDataset()
    ds.add(rec({"k": 1}, 80.0, 5.0))
    assert ds.stats({"k": 1}, min_n=3) is None


def test_dataset_round_trips_through_jsonl(tmp_path):
    ds = ConfigPerfDataset()
    ds.add(rec({"k": 1}, 60.0, 3.0, t=1.0))
    ds.add(rec({"k": 2}, 80.0, 5.0, temp=None, t=2.0))
    path = str(tmp_path / "ds.jsonl")
    ds.save(path)
    loaded = ConfigPerfDataset.load(path)
    assert len(loaded.records) == 2
    assert loaded.stats({"k": 2}).mean_tps == pytest.approx(80.0)
    assert loaded.stats({"k": 2}).mean_temp is None


# --------------------------------------------------------------------- knobs
def test_power_cap_knob_is_monotonic_and_normalised():
    k = power_cap_knob()
    p = [l.power_scale for l in k.levels]
    s = [l.speed_scale for l in k.levels]
    assert p == sorted(p) and s == sorted(s)
    assert p[-1] == pytest.approx(1.0) and s[-1] == pytest.approx(1.0)


def test_memory_bound_workload_loses_less_speed_than_power():
    half = len([f for f in (0.3, 0.4, 0.5)]) - 1
    k = power_cap_knob(mem_frac=0.8)
    lvl = k.levels[half]            # 50 percent power
    assert lvl.power_scale == pytest.approx(0.5)
    assert lvl.speed_scale > 0.9    # decode is memory bound, so speed barely drops
    compute_bound = power_cap_knob(mem_frac=0.1).levels[half]
    assert compute_bound.speed_scale < lvl.speed_scale


def write_sweep(tmp_path, rows_req, rows_tel):
    req = tmp_path / "req.csv"
    tel = tmp_path / "tel.csv"
    req.write_text("config,t_s,tokens,tok_per_s\n" + "".join("%s,0,100,%s\n" % r for r in rows_req))
    tel.write_text("config,t_s,temp_c,sm_mhz,mem_mhz,power_w,util_pct,throttle_reasons\n"
                   + "".join("%s,0,60,1,1,%s,99,0x0\n" % r for r in rows_tel))
    return str(req), str(tel)


def test_knob_from_sweep_builds_measured_priors(tmp_path):
    req, tel = write_sweep(
        tmp_path,
        [("100W", 50.0), ("100W", 52.0), ("150W", 58.0), ("150W", 60.0)],
        [("100W", 95.0), ("150W", 145.0)],
    )
    k = knob_from_sweep(req, tel)
    assert [l.label for l in k.levels] == ["100W", "150W"]
    assert k.levels[-1].power_scale == pytest.approx(1.0)
    assert k.levels[-1].speed_scale == pytest.approx(1.0)
    assert k.levels[0].power_scale == pytest.approx(95.0 / 145.0)
    assert k.levels[0].speed_scale == pytest.approx(51.0 / 59.0)


def test_knob_from_sweep_needs_two_configurations(tmp_path):
    req, tel = write_sweep(tmp_path, [("100W", 50.0)], [("100W", 95.0)])
    with pytest.raises(ValueError):
        knob_from_sweep(req, tel)


# ---------------------------------------------------------------- evaluation
def obs(temp=40.0, power=3.0, state="nominal", throttled=False):
    return Observation(t_s=10.0, power_w=power, tok_per_s=60.0, temp_c=temp,
                       thermal_state=state, throttled=throttled)


@pytest.fixture
def evaluator():
    return EvaluationAgent(Objectives(max_temp_c=53.0, temp_margin_c=1.0, horizon_s=120.0))


def test_hardware_throttling_is_a_violation(evaluator):
    a = evaluator.assess(obs(temp=40.0, throttled=True), PARAMS, at_top=False)
    assert a.status == "violation" and a.needs_optimization


def test_temperature_over_budget_is_a_violation(evaluator):
    assert evaluator.assess(obs(temp=54.0), PARAMS, at_top=False).status == "violation"


def test_forecast_over_limit_is_thermal_pressure(evaluator):
    # 40C at 6W heads toward 73C, so in 120s it is well past the limit.
    a = evaluator.assess(obs(temp=40.0, power=6.0), PARAMS, at_top=True)
    assert a.status == "thermal_pressure"
    assert a.forecast_c > 52.0


def test_cool_and_not_at_top_has_headroom(evaluator):
    assert evaluator.assess(obs(temp=30.0, power=1.0), PARAMS, at_top=False).status == "headroom"


def test_cool_at_top_is_ok(evaluator):
    a = evaluator.assess(obs(temp=30.0, power=1.0), PARAMS, at_top=True)
    assert a.status == "ok" and not a.needs_optimization


def test_flapping_hardware_throttle_is_pressure(evaluator):
    a = evaluator.assess(obs(temp=40.0, power=1.0), PARAMS, at_top=False, flapping=True)
    assert a.status == "thermal_pressure"


@pytest.mark.parametrize("state,at_top,expected", [
    ("critical", False, "violation"),
    ("serious", False, "thermal_pressure"),
    ("fair", False, "ok"),
    ("nominal", False, "headroom"),
    ("nominal", True, "ok"),
])
def test_coarse_state_only_assessment(evaluator, state, at_top, expected):
    o = Observation(t_s=1.0, power_w=0.0, tok_per_s=50.0, temp_c=None, thermal_state=state)
    assert evaluator.assess(o, PARAMS, at_top=at_top).status == expected


# ------------------------------------------------------------------ baseline
def test_baseline_averages_unthrottled_readings_only():
    b = BaselineAgent(ticks=3)
    cfg = {"k": 3}
    b.add(Observation(1, 4.0, 70.0, 30.0), cfg)
    b.add(Observation(2, 1.0, 30.0, 31.0, throttled=True), cfg)   # ignored
    assert not b.done
    b.add(Observation(3, 6.0, 80.0, 32.0), cfg)
    b.add(Observation(4, 5.0, 90.0, 33.0), cfg)
    assert b.done
    assert b.report.power_w == pytest.approx(5.0)
    assert b.report.tok_per_s == pytest.approx(80.0)
    assert b.report.temp_start_c == 30.0 and b.report.temp_end_c == 33.0


# --------------------------------------------------------------- transformer
class FakeActuator:
    def __init__(self, name="k", n=5, cost=0):
        self.knob = Knob(name, tuple(KnobLevel(str(i), (i + 1) / n, (i + 1) / n) for i in range(n)), cost)
        self.level = n - 1

    def get_level(self):
        return self.level

    def set_level(self, i):
        self.level = i


def test_lowering_is_immediate_even_inside_dwell():
    act = FakeActuator()
    t = Transformer([act], {0: 30.0})
    assert t.apply({"k": 3}, now=0.0)[0]
    assert t.apply({"k": 1}, now=1.0)[0]       # lowering ignores dwell
    assert act.level == 1


def test_raising_respects_dwell_then_proceeds():
    act = FakeActuator()
    t = Transformer([act], {0: 30.0})
    t.apply({"k": 1}, now=0.0)
    changed, note = t.apply({"k": 2}, now=10.0)
    assert not changed and "dwell" in note and act.level == 1
    assert t.apply({"k": 2}, now=31.0)[0] and act.level == 2


def test_expensive_knobs_have_longer_dwell():
    cheap, costly = FakeActuator("cheap", cost=0), FakeActuator("chip", cost=2)
    t = Transformer([cheap, costly], {0: 5.0, 2: 120.0})
    t.apply({"cheap": 1, "chip": 1}, now=0.0)
    t.apply({"cheap": 2, "chip": 2}, now=10.0)
    assert cheap.level == 2 and costly.level == 1


def test_transformer_clamps_levels():
    act = FakeActuator(n=3)
    t = Transformer([act], {0: 0.0})
    t.apply({"k": -5}, now=0.0)
    assert act.level == 0
    t.apply({"k": 99}, now=1.0)
    assert act.level == 2


# ------------------------------------------------------------------- monitor
def test_monitor_needs_a_sustained_error():
    m = DeploymentMonitor(threshold_c=1.0, patience=3)
    assert not m.check(2.0) and not m.check(2.0)
    assert m.check(2.0)
    assert m.drift_events == 1


def test_monitor_streak_resets_on_a_good_reading():
    m = DeploymentMonitor(threshold_c=1.0, patience=3)
    m.check(2.0)
    m.check(2.0)
    m.check(0.1)
    assert not m.check(2.0)
    assert m.drift_events == 0
