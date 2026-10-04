"""First order thermal model and online identification.

The device is modelled as a thermal resistance and capacitance:

    dT/dt = (t_amb + r_th * P - T) / tau

so for constant power P the temperature moves exponentially toward the steady
state t_amb + r_th * P. Three numbers (t_amb, r_th, tau) are enough to forecast
where a given power level will take the device, which is what a runtime
controller needs.

Everything here is standard library only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ThermalParams:
    t_amb: float   # ambient or idle temperature, C
    r_th: float    # thermal resistance, C per watt
    tau_s: float   # thermal time constant, seconds

    def steady_state(self, power_w: float) -> float:
        return self.t_amb + self.r_th * power_w

    def predict(self, temp_c: float, power_w: float, horizon_s: float) -> float:
        """Temperature after horizon_s seconds at constant power."""
        t_ss = self.steady_state(power_w)
        return t_ss + (temp_c - t_ss) * math.exp(-horizon_s / self.tau_s)

    def max_sustainable_power(self, limit_c: float) -> float:
        return max(0.0, (limit_c - self.t_amb) / self.r_th)


def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def _lstsq(rows: List[List[float]], ys: List[float]) -> Optional[List[float]]:
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    return _solve(ata, aty)


def fit_thermal_model(
    times: Sequence[float],
    temps: Sequence[float],
    powers: Sequence[float],
    t_amb: Optional[float] = None,
    min_tau_s: float = 5.0,
    max_tau_s: float = 3600.0,
) -> Optional[ThermalParams]:
    """Fit (t_amb, r_th, tau) from a temperature and power trace.

    powers[k + 1] is the mean power over the interval from times[k] to
    times[k + 1] (powers[0] is unused).

    If t_amb is None all three parameters are fitted, which needs the power to
    vary during the trace (otherwise ambient and resistance cannot be told
    apart and None is returned). If t_amb is given, only r_th and tau are
    fitted, which works even at constant power.

    Returns None when the data cannot identify the model.
    """
    n = len(times)
    if n < 5 or len(temps) != n or len(powers) != n:
        return None
    rows: List[List[float]] = []
    ys: List[float] = []
    for k in range(n - 1):
        dt = times[k + 1] - times[k]
        if dt <= 0:
            continue
        ys.append((temps[k + 1] - temps[k]) / dt)
        p = powers[k + 1]
        if t_amb is None:
            rows.append([p, temps[k], 1.0])
        else:
            rows.append([p, temps[k] - t_amb])
    if len(rows) < 4:
        return None
    if max(temps) - min(temps) < 0.2:
        return None
    if t_amb is None:
        used = [powers[k + 1] for k in range(n - 1)]
        if max(used) - min(used) < 0.05 * max(1e-9, max(used)):
            return None
    sol = _lstsq(rows, ys)
    if sol is None:
        return None
    a, b = sol[0], sol[1]
    if b >= 0 or a <= 0:
        return None
    tau = -1.0 / b
    if not (min_tau_s <= tau <= max_tau_s):
        return None
    r_th = a * tau
    if not (0.05 <= r_th <= 500.0):
        return None
    amb = t_amb if t_amb is not None else sol[2] * tau
    return ThermalParams(t_amb=amb, r_th=r_th, tau_s=tau)
