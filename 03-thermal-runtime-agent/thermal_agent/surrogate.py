"""Surrogate model for Agent 4a: a Bayesian model of the black box y = f(x).

The AutoOptAgent slide gives 4a the job of modelling a black box function
y = f(x) from configurations x to performance y, and predicting the next
candidate x. Here x is a setting for every knob and y is (power, throughput).

The model is log linear in the knob levels:

    log y(x) = a + sum over knobs k of theta[k][level of k in x]

with theta[k][top] fixed at zero. That is a multiplicative model: each knob
level scales the result by a constant factor, the same assumption the knob
priors make. It is fitted by Bayesian linear regression:

  - The prior mean for each theta is the log of the knob prior, so before any
    data the model equals the priors.
  - Each observation updates the posterior in closed form.
  - Old data is discounted slowly (forgetting), so a device that changes
    (battery, case, background load) does not leave the model stuck.

Two things this buys over a table of visited configurations. First, learning
generalises: after seeing the agent at one level of a knob, the model also
corrects its beliefs about that level inside configurations it has not visited.
Second, one observation at a configuration with several non-top knobs is
split between them in proportion to how uncertain each prior is, and later
observations at other configurations separate the effects. This is the usual
identifiability limit of any model fitted from a controller that moves one
knob at a time.

No numpy: the matrices are about 20 by 20.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .knobs import Knob
from .types import Config


def solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


class _LogLinear:
    """Bayesian linear regression on log y with sparse 0/1 features."""

    def __init__(self, prior_mean: Sequence[float], prior_sd: Sequence[float],
                 noise_sd: float, forgetting: float) -> None:
        self.dim = len(prior_mean)
        self.prior_mean = list(prior_mean)
        self.prior_prec = [1.0 / (sd * sd) for sd in prior_sd]
        self.noise_prec = 1.0 / (noise_sd * noise_sd)
        self.forgetting = forgetting
        self.data_prec = [[0.0] * self.dim for _ in range(self.dim)]
        self.data_eta = [0.0] * self.dim
        self.n = 0
        self._mean: Optional[List[float]] = None

    def update(self, active: Sequence[int], y_log: float) -> None:
        rho = self.forgetting
        if rho < 1.0:
            for i in range(self.dim):
                row = self.data_prec[i]
                for j in range(self.dim):
                    row[j] *= rho
                self.data_eta[i] *= rho
        for i in active:
            for j in active:
                self.data_prec[i][j] += self.noise_prec
            self.data_eta[i] += self.noise_prec * y_log
        self.n += 1
        self._mean = None

    def mean(self) -> List[float]:
        if self._mean is None:
            lam = [row[:] for row in self.data_prec]
            eta = self.data_eta[:]
            for i in range(self.dim):
                lam[i][i] += self.prior_prec[i]
                eta[i] += self.prior_prec[i] * self.prior_mean[i]
            self._mean = solve(lam, eta)
        return self._mean

    def predict_log(self, active: Sequence[int]) -> float:
        m = self.mean()
        return sum(m[i] for i in active)


class Surrogate:
    """Predicts (dynamic power in watts, tokens per second) for any configuration."""

    def __init__(self, knobs: Sequence[Knob], prior_sd: float = 0.25, noise_sd: float = 0.05,
                 forgetting: float = 0.995, intercept_sd: float = 3.0) -> None:
        self.knobs = list(knobs)
        self._index: Dict[Tuple[str, int], int] = {}
        n = 1                                   # feature 0 is the intercept
        for k in self.knobs:
            for lvl in range(len(k.levels)):
                if lvl != k.top:
                    self._index[(k.name, lvl)] = n
                    n += 1
        self.dim = n
        sd = [intercept_sd] + [prior_sd] * (n - 1)
        pm = [0.0] * n
        sm = [0.0] * n
        for (name, lvl), i in self._index.items():
            k = next(k for k in self.knobs if k.name == name)
            top = k.levels[k.top]
            pm[i] = math.log(k.levels[lvl].power_scale / top.power_scale)
            sm[i] = math.log(k.levels[lvl].speed_scale / top.speed_scale)
        self._power = _LogLinear(pm, sd, noise_sd, forgetting)
        self._speed = _LogLinear(sm, sd, noise_sd, forgetting)
        self.seeded = False

    # features ----------------------------------------------------------------
    def _active(self, config: Config) -> List[int]:
        out = [0]
        for k in self.knobs:
            i = self._index.get((k.name, config[k.name]))
            if i is not None:
                out.append(i)
        return out

    # learning ------------------------------------------------------------------
    def update(self, config: Config, dyn_power_w: float, tok_per_s: float) -> None:
        if dyn_power_w <= 0 or tok_per_s <= 0:
            return
        act = self._active(config)
        self._power.update(act, math.log(dyn_power_w))
        self._speed.update(act, math.log(tok_per_s))
        self.seeded = True

    @property
    def n_updates(self) -> int:
        return self._power.n

    # prediction ----------------------------------------------------------------
    def predict(self, config: Config) -> Tuple[float, float]:
        act = self._active(config)
        return math.exp(self._power.predict_log(act)), math.exp(self._speed.predict_log(act))

    def effect(self, knob_name: str, level: int) -> Tuple[float, float]:
        """Learned power and speed factor of one knob level relative to the top
        level of that knob (1.0 for the top). For inspection and tests."""
        i = self._index.get((knob_name, level))
        if i is None:
            return 1.0, 1.0
        return math.exp(self._power.mean()[i]), math.exp(self._speed.mean()[i])
