"""Simulation harness: run a policy against a simulated device and score it."""
from __future__ import annotations

import csv
import itertools
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

from .objectives import Objectives
from .runtime import AgentConfig, ThermalAgent
from .sim import PHONE_SCENARIOS, PRESETS, SimulatedDevice, make_phone_device


@dataclass
class SessionMetrics:
    mean_tps: float
    tail_tps: float            # mean over the last quarter of the session
    peak_temp_c: float
    time_throttled_s: float
    time_over_budget_s: float
    j_per_token: float
    level_changes: int
    throttle_events: int
    mean_quality: float = 1.0   # 1.0 when no knob changes output quality


def run_session(device: SimulatedDevice, agent: Optional[ThermalAgent] = None,
                duration_s: float = 600.0, dt: float = 1.0,
                budget_c: float = 53.0) -> Tuple[List[dict], SessionMetrics]:
    rows: List[dict] = []
    while device.clock < duration_s - 1e-9:
        device.advance(dt)
        if agent is not None:
            agent.tick()
        rows.append({
            "t_s": round(device.clock, 3), "temp_c": round(device.temp, 3),
            "power_w": round(device.power_w, 3), "tok_per_s": round(device.tps, 3),
            "throttled": int(device.hw_throttled), "level": device.actuator.get_level(),
            "config": ";".join("%s=%d" % (a.knob.name, a.get_level()) for a in device.actuators),
            "quality": round(device.quality, 4),
        })
    return rows, summarise(rows, dt, budget_c, device.throttle_events)


def summarise(rows: List[dict], dt: float, budget_c: float, throttle_events: int) -> SessionMetrics:
    n = len(rows)
    tail = rows[int(n * 0.75):]
    tokens = sum(r["tok_per_s"] * dt for r in rows)
    energy = sum(r["power_w"] * dt for r in rows)
    changes = sum(1 for a, b in zip(rows, rows[1:]) if a["config"] != b["config"])
    return SessionMetrics(
        mean_tps=sum(r["tok_per_s"] for r in rows) / n,
        tail_tps=sum(r["tok_per_s"] for r in tail) / len(tail),
        peak_temp_c=max(r["temp_c"] for r in rows),
        time_throttled_s=sum(r["throttled"] for r in rows) * dt,
        time_over_budget_s=sum(1 for r in rows if r["temp_c"] > budget_c) * dt,
        j_per_token=energy / tokens if tokens else float("inf"),
        level_changes=changes,
        throttle_events=throttle_events,
        mean_quality=sum(r["quality"] for r in rows) / n,
    )


def _level_for_fraction(device: SimulatedDevice, fraction: float) -> int:
    caps = device._caps
    return min(range(len(caps)), key=lambda i: abs(caps[i] - fraction))


def run_comparison(preset: str = "phone_severe", duration_s: float = 600.0, dt: float = 1.0,
                   objectives: Optional[Objectives] = None) -> Dict[str, Tuple[List[dict], SessionMetrics]]:
    """Fixed full power (hardware throttling only), a hand tuned static cap, and
    the thermal agent, on identical simulated devices."""
    obj = objectives or Objectives()
    params = PRESETS[preset]
    results = {}

    dev = SimulatedDevice(params)
    results["fixed_full_power"] = run_session(dev, None, duration_s, dt, obj.max_temp_c)

    dev = SimulatedDevice(params)
    dev.actuator.set_level(_level_for_fraction(dev, 0.5))
    results["fixed_50_percent"] = run_session(dev, None, duration_s, dt, obj.max_temp_c)

    dev = SimulatedDevice(params)
    agent = ThermalAgent(dev, dev.actuators, AgentConfig(objectives=obj))
    results["thermal_agent"] = run_session(dev, agent, duration_s, dt, obj.max_temp_c)
    return results


def write_comparison_csv(path: str, by_preset: Dict[str, Dict[str, Tuple[List[dict], SessionMetrics]]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["preset", "policy", "mean_tok_per_s", "tail_tok_per_s", "peak_temp_c",
                    "time_throttled_s", "time_over_budget_s", "j_per_token", "level_changes",
                    "throttle_events", "mean_quality"])
        for preset, res in by_preset.items():
            for policy, (_rows, m) in res.items():
                w.writerow([preset, policy, round(m.mean_tps, 2), round(m.tail_tps, 2),
                            round(m.peak_temp_c, 2), m.time_throttled_s, m.time_over_budget_s,
                            round(m.j_per_token, 3), m.level_changes, m.throttle_events,
                            round(m.mean_quality, 4)])


# --- model level knobs on a phone -------------------------------------------------
def oracle_static_levels(make_device: Callable[[], SimulatedDevice], objectives: Objectives) -> dict:
    """The best single fixed configuration, chosen with full knowledge of the
    device's true physics. It is the strongest static policy there is, and the
    bar a runtime agent has to be compared with. Best means highest value of
    the objective function whose steady state stays under the thermal limit."""
    probe = make_device()
    names = [k.name for k in probe._truth]
    sizes = [len(k.levels) for k in probe._truth]
    p = probe.params

    def speed_quality(levels):
        saved = probe._levels
        probe._levels = dict(levels)
        try:
            return p.base_tps * probe._scale("speed_scale"), probe.quality
        finally:
            probe._levels = saved

    top = {k.name: k.top for k in probe._truth}
    ref = speed_quality(top)[0]
    best, best_u = None, -1e9
    for combo in itertools.product(*[range(n) for n in sizes]):
        levels = dict(zip(names, combo))
        if probe.steady_state(levels) > objectives.limit_c:
            continue
        tps, q = speed_quality(levels)
        u = tps / ref + objectives.quality_weight * q
        if u > best_u:
            best, best_u = levels, u
    return best if best is not None else {n: 0 for n in names}


def run_model_level_comparison(scenario: str = "phone_quant_flat", duration_s: float = 900.0,
                               dt: float = 1.0, objectives: Optional[Objectives] = None,
                               agent_config: Optional[AgentConfig] = None
                               ) -> Dict[str, Tuple[List[dict], SessionMetrics]]:
    """A hot phone whose only levers are token pacing, quantization and KV cache
    precision. Unmanaged (everything at the top), the best fixed configuration
    chosen with hindsight, and the agent."""
    obj = objectives or Objectives()
    results = {}

    dev = make_phone_device(scenario)
    results["fixed_top"] = run_session(dev, None, duration_s, dt, obj.max_temp_c)

    oracle = oracle_static_levels(lambda: make_phone_device(scenario), obj)
    dev = make_phone_device(scenario)
    for a in dev.actuators:
        a.set_level(oracle[a.knob.name])
    results["best_fixed_with_hindsight"] = run_session(dev, None, duration_s, dt, obj.max_temp_c)

    dev = make_phone_device(scenario)
    cfg = agent_config or AgentConfig(objectives=obj)
    agent = ThermalAgent(dev, dev.actuators, cfg)
    results["thermal_agent"] = run_session(dev, agent, duration_s, dt, obj.max_temp_c)
    return results


def objective_value(m: SessionMetrics, ref_tps: float, objectives: Objectives) -> float:
    """Value of the Agent 1 objective function for a whole session:
    mean throughput relative to a reference plus quality_weight times mean
    quality. Only meaningful between policies that meet the thermal constraint;
    an unmanaged run that overheats can score high and still be unacceptable."""
    return m.mean_tps / ref_tps + objectives.quality_weight * m.mean_quality
