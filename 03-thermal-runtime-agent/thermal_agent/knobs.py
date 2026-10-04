"""Knobs: the actions the agent can take, with priors on their effect.

A knob is an ordered list of levels from the most conservative (coolest) to the
most demanding. The top level is the setting the user would pick if heat were
not a concern. Each level carries a prior on how much power it draws, how much
throughput it gives, and how much output quality it keeps, all relative to the
top level. Priors are replaced by measurements as the agent runs
(surrogate.py), and a power cap knob can be built from a real sweep with
knob_from_sweep.

Two kinds of knob:
  - Platform knobs change how hard the hardware runs: power_cap_knob (needs a
    platform API, such as nvidia-smi), token_pacing_knob (any app can do this).
  - Model knobs change the model or serving configuration, which is what the
    AutoOptAgent transformation agent (4b) is about: quantization_knob,
    kv_precision_knob.

A model knob can make a level faster than the top level (fewer bits means fewer
bytes read per token), so speed_scale may be above 1. Whether a model knob also
lowers power is an empirical question about each device, which is why the
numbers below are priors and not facts.

switch_cost orders actions by how expensive they are to change:
    0 cheap      (power cap, token pacing, draft length)
    1 medium     (swap to a different quantisation, change KV cache precision)
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
    power_scale: float   # dynamic power relative to the top level
    speed_scale: float   # throughput relative to the top level (can exceed 1)
    quality_scale: float = 1.0   # output quality relative to the top level, (0, 1]


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


# --- Model level and portable knobs ------------------------------------------
# PRIORS, NOT MEASUREMENTS. They come from simple arguments (decode reads every
# weight once per token, so speed follows bytes per weight) and from placeholder
# quality numbers. Replace them with measured speed, power and an offline quality
# score for the model you actually ship. The agent corrects speed and power
# online; it cannot correct quality, because it has no way to grade an answer.

# (label, approximate bits per weight, placeholder quality relative to the top level)
QUANT_LEVELS = (
    ("Q3", 3.9, 0.92),
    ("Q4", 4.8, 0.97),
    ("Q5", 5.7, 0.99),
    ("Q8", 8.5, 1.00),
)

# Share of per-token time that does not depend on weight bytes (activations,
# KV cache reads, dequantisation, sampling). Keeps the speed prior from
# implying that halving the bits doubles the speed.
NON_WEIGHT_TIME_SHARE = 0.30


def quantization_knob(levels=QUANT_LEVELS, cooling: float = 0.5, name: str = "quantization") -> Knob:
    """Weight quantization. Top level is the highest precision.

    Speed prior: time per token = share + (1 - share) * bits / top_bits.
    Power prior: `cooling` is how much of the bytes saved turns into watts
    saved. 0 means fewer bits only make tokens faster, with no change in watts
    (the memory bandwidth stays saturated). 1 means watts fall in proportion to
    bytes. The default is an uninformed middle value, and the agent measures
    the real effect.
    """
    top_bits = levels[-1][1]
    out = []
    for label, bits, quality in levels:
        rel = bits / top_bits
        time_per_token = NON_WEIGHT_TIME_SHARE + (1.0 - NON_WEIGHT_TIME_SHARE) * rel
        out.append(KnobLevel(label, power_scale=1.0 - cooling * (1.0 - rel) * (1.0 - NON_WEIGHT_TIME_SHARE),
                             speed_scale=1.0 / time_per_token, quality_scale=quality))
    return Knob(name=name, levels=tuple(out), switch_cost=1)


def kv_precision_knob(name: str = "kv_precision") -> Knob:
    """KV cache precision (FP16, INT8, INT4), the example on the AutoOptAgent
    slide. The effect grows with context length, so the priors are small."""
    spec = (("KV4", 0.94, 1.06, 0.97), ("KV8", 0.97, 1.03, 0.995), ("KV16", 1.0, 1.0, 1.0))
    return Knob(name=name, levels=tuple(KnobLevel(l, p, sp, q) for l, p, sp, q in spec), switch_cost=1)


def token_pacing_knob(duty=(0.4, 0.55, 0.7, 0.85, 1.0), name: str = "token_pacing") -> Knob:
    """Idle time inserted between tokens or requests. Any app can do this, with
    no platform API, which is why it matters on a phone: iOS apps cannot set a
    power cap. Prior: average power and throughput both scale with the duty
    cycle. A DVFS power cap is more efficient than this (power falls faster than
    speed), which is exactly the kind of difference the agent should learn."""
    levels = tuple(KnobLevel("%d%%" % round(d * 100), power_scale=d, speed_scale=d) for d in duty)
    return Knob(name=name, levels=levels, switch_cost=0)
