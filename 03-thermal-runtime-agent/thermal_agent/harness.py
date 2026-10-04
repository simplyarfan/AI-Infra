"""Simulation harness: run a policy against a simulated device and score it."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

from .objectives import Objectives
from .runtime import AgentConfig, ThermalAgent
from .sim import PRESETS, SimulatedDevice


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
        })
    return rows, summarise(rows, dt, budget_c, device.throttle_events)


def summarise(rows: List[dict], dt: float, budget_c: float, throttle_events: int) -> SessionMetrics:
    n = len(rows)
    tail = rows[int(n * 0.75):]
    tokens = sum(r["tok_per_s"] * dt for r in rows)
    energy = sum(r["power_w"] * dt for r in rows)
    changes = sum(1 for a, b in zip(rows, rows[1:]) if a["level"] != b["level"])
    return SessionMetrics(
        mean_tps=sum(r["tok_per_s"] for r in rows) / n,
        tail_tps=sum(r["tok_per_s"] for r in tail) / len(tail),
        peak_temp_c=max(r["temp_c"] for r in rows),
        time_throttled_s=sum(r["throttled"] for r in rows) * dt,
        time_over_budget_s=sum(1 for r in rows if r["temp_c"] > budget_c) * dt,
        j_per_token=energy / tokens if tokens else float("inf"),
        level_changes=changes,
        throttle_events=throttle_events,
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
                    "throttle_events"])
        for preset, res in by_preset.items():
            for policy, (_rows, m) in res.items():
                w.writerow([preset, policy, round(m.mean_tps, 2), round(m.tail_tps, 2),
                            round(m.peak_temp_c, 2), m.time_throttled_s, m.time_over_budget_s,
                            round(m.j_per_token, 3), m.level_changes, m.throttle_events])
