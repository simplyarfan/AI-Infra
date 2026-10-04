"""Knobs: the actions the agent can take, with priors on their effect.

A knob is an ordered list of levels from lowest to highest intensity. Each level
carries a prior on how much power it draws and how much throughput it gives,
both relative to the top level. Priors are replaced by measurements as the agent
runs, and can be built from a real sweep with knob_from_sweep.

switch_cost orders actions by how expensive they are to change:
    0 cheap      (power cap, draft length, token pacing)
    1 medium     (swap to a different quantisation)
    2 expensive  (move between GPU and NPU, offload to cloud)
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class KnobLevel:
    label: str
    power_scale: float   # power relative to the top level, (0, 1]
    speed_scale: float   # throughput relative to the top level, (0, 1]


@dataclass(frozen=True)
class Knob:
    name: str
    levels: Tuple[KnobLevel, ...]
    switch_cost: int = 0

    @property
    def top(self) -> int:
        return len(self.levels) - 1


def power_cap_knob(
    fractions: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    mem_frac: float = 0.35,
    name: str = "power_cap",
) -> Knob:
    """Power cap knob with a DVFS prior.

    Dynamic power scales roughly with frequency cubed, so frequency scales with
    the cube root of the power fraction. Throughput has a memory bound part
    (mem_frac, which does not care about core frequency) and a compute bound
    part that scales with frequency. LLM decode is mostly memory bound, so on a
    GPU mem_frac is high and a power cap costs little speed.
    """
    levels = tuple(
        KnobLevel(
            label="%d%%" % round(f * 100),
            power_scale=f,
            speed_scale=mem_frac + (1.0 - mem_frac) * f ** (1.0 / 3.0),
        )
        for f in fractions
    )
    return Knob(name=name, levels=levels, switch_cost=0)


def knob_from_sweep(requests_csv: str, telemetry_csv: str, name: str = "power_cap") -> Knob:
    """Build a power cap knob from the CSVs written by gpu_thermal_sweep.py.

    Config labels look like "170W". Priors are the measured mean throughput and
    mean power of each configuration relative to the highest one. Because these
    are total board power, use idle_power_w=0 in the agent config.
    """
    tps = {}
    pw = {}
    with open(requests_csv, newline="") as f:
        for row in csv.DictReader(f):
            tps.setdefault(row["config"], []).append(float(row["tok_per_s"]))
    with open(telemetry_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("power_w") not in (None, "", "None"):
                pw.setdefault(row["config"], []).append(float(row["power_w"]))
    labels = [l for l in tps if l in pw]
    if len(labels) < 2:
        raise ValueError("need at least two configurations with both throughput and power")

    def watts(label: str) -> int:
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    labels.sort(key=watts)
    mean_tps = {l: sum(tps[l]) / len(tps[l]) for l in labels}
    mean_pw = {l: sum(pw[l]) / len(pw[l]) for l in labels}
    tps_max = max(mean_tps.values())
    pw_max = max(mean_pw.values())
    levels = tuple(
        KnobLevel(label=l, power_scale=mean_pw[l] / pw_max, speed_scale=mean_tps[l] / tps_max)
        for l in labels
    )
    return Knob(name=name, levels=levels, switch_cost=0)
