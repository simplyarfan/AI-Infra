"""NVIDIA GPU backend: telemetry through nvidia-smi and a power limit actuator.

NOT YET RUN ON REAL HARDWARE. The parsing and command construction are unit
tested with fake nvidia-smi output; the live behaviour is validated only once
this runs on the RTX 5060 Ti.
"""
from __future__ import annotations

import subprocess
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..knobs import Knob, KnobLevel
from ..types import Observation

# Throttle reason bits from nvidia-smi. The software power cap bit (0x4) is
# deliberately not counted: that is the cap the agent sets itself.
HW_SLOWDOWN_BITS = 0x8 | 0x20 | 0x40 | 0x80
FIELDS_NEW = "temperature.gpu,power.draw,clocks_event_reasons.active"
FIELDS_OLD = "temperature.gpu,power.draw,clocks_throttle_reasons.active"

Runner = Callable[[List[str]], Tuple[int, str, str]]


def default_runner(args: List[str]) -> Tuple[int, str, str]:
    r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def parse_smi_line(line: str) -> Dict[str, object]:
    """Parse one csv line of: temperature.gpu, power.draw, throttle reasons."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        raise ValueError("unexpected nvidia-smi line: " + repr(line))
    mask_text = parts[2]
    try:
        mask = int(mask_text, 16) if mask_text.lower().startswith("0x") else int(mask_text)
    except ValueError:
        mask = 0
    return {
        "temp_c": _float(parts[0]),
        "power_w": _float(parts[1]),
        "reasons_mask": mask,
        "hw_throttled": bool(mask & HW_SLOWDOWN_BITS),
    }


def state_from_temp(temp_c: float, thresholds: Tuple[float, float, float] = (70.0, 80.0, 88.0)) -> str:
    fair, serious, critical = thresholds
    if temp_c >= critical:
        return "critical"
    if temp_c >= serious:
        return "serious"
    if temp_c >= fair:
        return "fair"
    return "nominal"


class NvidiaTelemetry:
    def __init__(self, throughput_fn: Callable[[], float], runner: Runner = default_runner,
                 clock: Callable[[], float] = None) -> None:
        import time
        self._tps = throughput_fn
        self._run = runner
        self._clock = clock or time.time
        self._t0 = self._clock()
        self._fields = self._pick_fields()

    def _pick_fields(self) -> str:
        for f in (FIELDS_NEW, FIELDS_OLD):
            rc, out, _ = self._run(["--query-gpu=" + f, "--format=csv,noheader,nounits"])
            if rc == 0 and out:
                return f
        raise RuntimeError("nvidia-smi query failed. Is the driver installed and nvidia-smi on PATH?")

    def read(self) -> Observation:
        rc, out, err = self._run(["--query-gpu=" + self._fields, "--format=csv,noheader,nounits"])
        if rc != 0 or not out:
            raise RuntimeError("nvidia-smi failed: " + err)
        s = parse_smi_line(out.splitlines()[0])
        temp = s["temp_c"]
        return Observation(
            t_s=self._clock() - self._t0,
            power_w=s["power_w"] or 0.0,
            tok_per_s=self._tps(),
            temp_c=temp,
            thermal_state=state_from_temp(temp) if temp is not None else "nominal",
            throttled=bool(s["hw_throttled"]),
        )


def build_power_knob(default_w: float, min_w: float, max_w: float, n_levels: int = 6,
                     mem_frac: float = 0.8, name: str = "power_limit") -> Tuple[Knob, List[int]]:
    """Power limit knob between the card's minimum and default limit.

    Returns the knob and the watts for each level. Speed priors assume decode is
    mostly memory bound (mem_frac). Replace them with knob_from_sweep results as
    soon as a real sweep exists.
    """
    lo = max(min_w, 0.4 * default_w)
    hi = min(max_w, default_w)
    watts = sorted({int(round(lo + (hi - lo) * i / (n_levels - 1))) for i in range(n_levels)})
    levels = []
    for w in watts:
        frac = w / hi
        levels.append(KnobLevel(label="%dW" % w, power_scale=frac,
                                speed_scale=mem_frac + (1.0 - mem_frac) * frac ** (1.0 / 3.0)))
    return Knob(name=name, levels=tuple(levels), switch_cost=0), watts


class NvidiaPowerLimitActuator:
    def __init__(self, knob: Knob, watts_by_level: Sequence[int], runner: Runner = default_runner) -> None:
        if len(watts_by_level) != len(knob.levels):
            raise ValueError("one wattage per knob level is required")
        self.knob = knob
        self._watts = list(watts_by_level)
        self._run = runner
        self._level = len(watts_by_level) - 1

    def get_level(self) -> int:
        return self._level

    def set_level(self, idx: int) -> None:
        idx = max(0, min(self.knob.top, int(idx)))
        rc, out, err = self._run(["-pl", str(self._watts[idx])])
        if rc != 0:
            raise RuntimeError("could not set power limit (administrator rights needed): " + (err or out))
        self._level = idx
