import numpy as np
import pytest
from scipy.optimize import minimize

from jlucid.control.energy import solve_lqr_tracking


def _cost_and_energy(a, x0, x_t, s, rho, horizon, dt, u):
    steps = len(u)
    n = a.shape[0]
    ad = np.eye(n) + a * dt
    bd = np.eye(n) * dt
    e = x0 - x_t
    w = ad @ x_t - x_t
    total = 0.0
    ctrl = 0.0
    for k in range(steps):
        total += (e @ s @ e) * dt + rho * (u[k] @ u[k]) * dt
        ctrl += (u[k] @ u[k]) * dt
        e = ad @ e + bd @ u[k] + w
    return total, ctrl


def test_2d_matches_bruteforce_optimum():
    """Discrete LQR tracking minimizes the same discretized cost (2-D)."""
    a = np.diag([-1.0, -2.0])
    n = 2
    x0 = np.zeros(n)
    x_t = np.array([1.0, 0.0])
    s = np.eye(n)
    rho, horizon, dt = 1.0, 0.1, 0.001
    steps = int(round(horizon / dt))
    sol = solve_lqr_tracking(a, x0, x_t, s, rho=rho, horizon=horizon, dt=dt)

    def cost(u_flat):
        u = u_flat.reshape(steps, n)
        c, _ = _cost_and_energy(a, x0, x_t, s, rho, horizon, dt, u)
        return c

    res = minimize(cost, np.zeros(steps * n), method="SLSQP",
                   options={"maxiter": 3000, "ftol": 1e-10})
    _, ctrl_opt = _cost_and_energy(a, x0, x_t, s, rho, horizon, dt,
                                   res.x.reshape(steps, n))
    assert sol["cost"] == pytest.approx(res.fun, rel=0.01)
    assert sol["energy_total"] == pytest.approx(ctrl_opt, rel=0.03)


def test_monotonicity_soft_lqr():
    """Soft-terminal LQR: higher rho reduces control energy, and for a
    Hurwitz system with a nonzero target a shorter horizon accumulates less
    state-error penalty, so control energy is also lower (hard-terminal
    LQR would show the opposite horizon trend; Paper 6 uses soft tracking)."""
    rng = np.random.default_rng(1)
    n = 4
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = -np.array([1.0, 0.8, 0.6, 0.4])
    a = (q * lam) @ q.T
    x0 = np.zeros(n)
    x_t = np.ones(n)
    s = np.eye(n)
    e1 = solve_lqr_tracking(a, x0, x_t, s, rho=1.0)["energy_total"]
    e2 = solve_lqr_tracking(a, x0, x_t, s, rho=2.0)["energy_total"]
    e3 = solve_lqr_tracking(a, x0, x_t, s, rho=1.0, horizon=0.5)["energy_total"]
    assert e1 > 0
    assert e2 < e1
    assert e3 < e1


def test_deterministic():
    a = np.diag([-1.0, -2.0])
    s = np.eye(2)
    r1 = solve_lqr_tracking(a, np.zeros(2), np.ones(2), s)
    r2 = solve_lqr_tracking(a, np.zeros(2), np.ones(2), s)
    assert r1["energy_total"] == r2["energy_total"]


def test_energy_sum():
    a = np.diag([-1.0, -2.0, -0.5])
    s = np.eye(3)
    r = solve_lqr_tracking(a, np.zeros(3), np.ones(3), s)
    assert r["energy_total"] == pytest.approx(float(r["energy_node"].sum()))


def test_1d_matches_bruteforce_optimum():
    a = np.array([[-1.0]])
    s = np.eye(1)
    rho, horizon, dt = 1.0, 1.0, 0.01
    steps = int(round(horizon / dt))
    r = solve_lqr_tracking(a, np.zeros(1), np.ones(1), s, rho=rho,
                           horizon=horizon, dt=dt)

    def cost(u_flat):
        return _cost_and_energy(a, np.zeros(1), np.ones(1), s, rho, horizon,
                                dt, u_flat.reshape(steps, 1))[0]

    res = minimize(cost, np.zeros(steps), method="SLSQP",
                   options={"maxiter": 3000, "ftol": 1e-10})
    assert r["cost"] == pytest.approx(res.fun, rel=0.01)


def test_zero_s_cost():
    a = np.diag([-1.0, -2.0])
    r = solve_lqr_tracking(a, np.zeros(2), np.ones(2), np.zeros((2, 2)))
    assert r["energy_total"] == 0.0
    assert np.allclose(r["u"], 0.0)


def test_shape_validation():
    with pytest.raises(ValueError):
        solve_lqr_tracking(np.ones((2, 3)), np.zeros(2), np.ones(2), np.eye(2))
    with pytest.raises(ValueError):
        solve_lqr_tracking(np.eye(2), np.zeros(3), np.ones(2), np.eye(2))
    with pytest.raises(ValueError):
        solve_lqr_tracking(np.eye(2), np.zeros(2), np.ones(2),
                           np.array([[1.0, 1.0], [0.0, 1.0]]))
