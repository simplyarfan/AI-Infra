"""The thermal model and its online identification.

Why this matters: the controller forecasts temperature with this model, so if
the model cannot be fitted from live data, or is fitted wrongly, every decision
downstream is wrong.
"""
import random

import pytest

from thermal_agent.thermal_model import ThermalParams, fit_thermal_model

TRUE = ThermalParams(t_amb=25.0, r_th=8.0, tau_s=60.0)


def trace(params, powers, dt=1.0):
    """Temperature trace for a list of per interval powers. powers_out[k + 1] is
    the power over the interval ending at times[k + 1]."""
    temp = params.t_amb
    times, temps, pw = [0.0], [temp], [powers[0]]
    for i, p in enumerate(powers):
        temp = params.predict(temp, p, dt)
        times.append((i + 1) * dt)
        temps.append(temp)
        pw.append(p)
    return times, temps, pw


def varying_power(n=240):
    return [6.0 if (i // 40) % 2 == 0 else 2.0 for i in range(n)]


def test_steady_state_and_prediction_basics():
    assert TRUE.steady_state(5.0) == pytest.approx(65.0)
    assert TRUE.predict(40.0, 5.0, 0.0) == pytest.approx(40.0)
    assert TRUE.predict(40.0, 5.0, 1e6) == pytest.approx(65.0)
    mid = TRUE.predict(40.0, 5.0, 30.0)
    assert 40.0 < mid < 65.0


def test_prediction_cools_when_power_drops():
    assert TRUE.predict(60.0, 1.0, 120.0) < 60.0


def test_max_sustainable_power():
    assert TRUE.max_sustainable_power(53.0) == pytest.approx(3.5)
    assert TRUE.max_sustainable_power(10.0) == 0.0


def test_fit_recovers_all_three_parameters_when_power_varies():
    fit = fit_thermal_model(*trace(TRUE, varying_power()))
    assert fit is not None
    assert fit.t_amb == pytest.approx(25.0, abs=1.5)
    assert fit.r_th == pytest.approx(8.0, rel=0.05)
    assert fit.tau_s == pytest.approx(60.0, rel=0.05)


def test_fit_with_known_ambient_works_at_constant_power():
    fit = fit_thermal_model(*trace(TRUE, [6.0] * 120), t_amb=25.0)
    assert fit is not None
    assert fit.r_th == pytest.approx(8.0, rel=0.05)
    assert fit.tau_s == pytest.approx(60.0, rel=0.05)


def test_fit_refuses_when_ambient_is_not_identifiable():
    # Constant power and unknown ambient: resistance and ambient are confounded.
    assert fit_thermal_model(*trace(TRUE, [6.0] * 120)) is None


def test_fit_refuses_flat_temperature():
    times = [float(i) for i in range(30)]
    assert fit_thermal_model(times, [40.0] * 30, [3.0] * 30, t_amb=25.0) is None


def test_fit_refuses_too_little_data():
    assert fit_thermal_model([0.0, 1.0, 2.0], [30.0, 31.0, 32.0], [3.0, 3.0, 3.0], t_amb=25.0) is None


def test_fit_survives_sensor_noise():
    rng = random.Random(7)
    times, temps, pw = trace(TRUE, varying_power(400))
    noisy = [t + rng.gauss(0.0, 0.05) for t in temps]
    fit = fit_thermal_model(times, noisy, pw)
    assert fit is not None
    assert fit.r_th == pytest.approx(8.0, rel=0.2)
    assert fit.tau_s == pytest.approx(60.0, rel=0.25)


def test_fit_handles_irregular_sampling():
    # Same physics, uneven sample spacing.
    rng = random.Random(3)
    temp, t = TRUE.t_amb, 0.0
    times, temps, pw = [0.0], [temp], [6.0]
    for i in range(200):
        dt = rng.choice([0.5, 1.0, 2.0])
        p = 6.0 if (i // 40) % 2 == 0 else 2.0
        temp = TRUE.predict(temp, p, dt)
        t += dt
        times.append(t)
        temps.append(temp)
        pw.append(p)
    fit = fit_thermal_model(times, temps, pw)
    assert fit is not None
    assert fit.r_th == pytest.approx(8.0, rel=0.08)
