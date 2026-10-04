"""Agent 4c: observability and feedback.

Records every observation into the Configuration-Performance Dataset, keeps a
sliding window, and identifies the device's thermal model online. The model
starts from a prior and is replaced by a fitted one as soon as the data can
identify it. It also reports how well the model has been predicting recent
temperatures, which the deployment monitor uses to detect drift.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .dataset import ConfigPerfDataset, Record
from .thermal_model import ThermalParams, fit_thermal_model
from .types import Config, Observation


class Observer:
    def __init__(
        self,
        dataset: ConfigPerfDataset,
        prior: ThermalParams,
        window_s: float = 120.0,
        assume_cold_start: bool = True,
        refit_every: int = 5,
    ) -> None:
        self.dataset = dataset
        self.prior = prior
        self.window_s = window_s
        self.assume_cold_start = assume_cold_start
        self.refit_every = refit_every
        self._obs: List[Observation] = []
        self._params = prior
        self._ambient_set = False
        self.fitted = False
        self._since_fit = 0
        self._throttle_transitions: List[float] = []

    # recording -----------------------------------------------------------
    def record(self, obs: Observation, config: Config, context: Optional[Dict[str, float]] = None) -> None:
        ctx = dict(context or {})
        if obs.battery_pct is not None:
            ctx["battery_pct"] = obs.battery_pct
        self.dataset.add(Record(
            t_s=obs.t_s, config=dict(config), tok_per_s=obs.tok_per_s, power_w=obs.power_w,
            temp_c=obs.temp_c, throttled=obs.throttled, context=ctx,
        ))
        if self._obs and self._obs[-1].throttled != obs.throttled:
            self._throttle_transitions.append(obs.t_s)
        self._obs.append(obs)
        cutoff = obs.t_s - self.window_s
        while self._obs and self._obs[0].t_s < cutoff:
            self._obs.pop(0)
        self._throttle_transitions = [t for t in self._throttle_transitions if t >= cutoff]
        if not self._ambient_set and obs.temp_c is not None:
            if self.assume_cold_start:
                self._params = ThermalParams(obs.temp_c, self.prior.r_th, self.prior.tau_s)
            self._ambient_set = True
        self._since_fit += 1
        if self._since_fit >= self.refit_every:
            self._since_fit = 0
            self._maybe_refit()

    # model ----------------------------------------------------------------
    def params(self) -> ThermalParams:
        return self._params

    def _maybe_refit(self) -> None:
        pts = [o for o in self._obs if o.temp_c is not None]
        if len(pts) < 8:
            return
        times = [o.t_s for o in pts]
        temps = [o.temp_c for o in pts]
        powers = [o.power_w for o in pts]
        fit = fit_thermal_model(times, temps, powers)
        if fit is not None and abs(fit.t_amb - self._params.t_amb) > 25.0:
            fit = None
        if fit is None:
            fit = fit_thermal_model(times, temps, powers, t_amb=self._params.t_amb)
        if fit is not None:
            self._params = fit
            self.fitted = True

    def rollout_error(self, span_s: float = 15.0) -> float:
        """Actual minus predicted temperature over the last span_s seconds,
        rolling the model forward through the powers that were really applied.
        Near zero when the model is right, so it is not fooled by the agent's
        own configuration changes."""
        if len(self._obs) < 3:
            return 0.0
        now = self._obs[-1].t_s
        start = None
        for i, o in enumerate(self._obs):
            if o.t_s >= now - span_s:
                start = i
                break
        if start is None or start >= len(self._obs) - 1:
            return 0.0
        seg = self._obs[start:]
        if any(o.temp_c is None for o in seg):
            return 0.0
        pred = seg[0].temp_c
        for prev, cur in zip(seg, seg[1:]):
            pred = self._params.predict(pred, cur.power_w, cur.t_s - prev.t_s)
        return seg[-1].temp_c - pred

    def correct_ambient(self, err_c: float, span_s: float = 15.0) -> None:
        """Absorb a model error into the ambient estimate. For a constant
        power the error after span_s is delta_amb * (1 - exp(-span / tau))."""
        gain = 1.0 - math.exp(-span_s / self._params.tau_s)
        if gain <= 1e-6:
            return
        p = self._params
        self._params = ThermalParams(p.t_amb + err_c / gain, p.r_th, p.tau_s)

    def flapping(self) -> bool:
        """Hardware throttling switching on and off repeatedly."""
        return len(self._throttle_transitions) >= 3

    @property
    def last(self) -> Optional[Observation]:
        return self._obs[-1] if self._obs else None
