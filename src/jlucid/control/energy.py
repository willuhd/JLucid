"""Finite-horizon LQR tracking / control energy (Paper 6 eqs. 11-14, D2-8)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..config import DT, RHO, T_HORIZON


def solve_lqr_tracking(
    a: np.ndarray,
    x0: np.ndarray,
    x_t: np.ndarray,
    s: np.ndarray,
    rho: float = RHO,
    horizon: float = T_HORIZON,
    dt: float = DT,
) -> dict:
    """Solve the finite-horizon LQR tracking problem.

    Cost: J = int_0^T [(x_T - x)^T S (x_T - x) + rho u^T u] dt,
    dx/dt = A x + B u, B = I.  Soft terminal constraint (free final state).

    We integrate the continuous-time differential Riccati equation for the
    tracking gain P (n x n) and affine term g (n,):

      dP/dtau = A^T P + P A - P P / rho + S
      dg/dtau = A^T g - P g / rho + P A x_T

    forward in tau in [0, T] with P(0)=0, g(0)=0 (so P(T)=0, g(T)=0), then
    the optimal control is u(t) = -(1/rho)(P(T-t) e(t) + g(T-t)), with
    e(t) = x(t) - x_T evolving as e' = A e + u + A x_T.

    The energy and cost are obtained by numerical integration with the same
    time step as the plan (Paper 6 uses dt = 0.001, T = 1).  Using the ODE
    avoids 1000 dense 90x90 solves per subject-transition and is ~100x
    faster, while reproducing the discrete recursion's energy within a few
    percent (verified against brute-force optimization in the tests).
    """
    a = np.asarray(a, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = a.shape[0]
    if a.ndim != 2 or a.shape != (n, n):
        raise ValueError("A must be square")
    if x0.shape != (n,) or x_t.shape != (n,):
        raise ValueError("x0/xT must be (n,)")
    if s.shape != (n, n) or not np.allclose(s, s.T):
        raise ValueError("S must be symmetric (n, n)")
    n_steps = max(1, int(round(horizon / dt)))

    def ric(t: float, pg: np.ndarray) -> np.ndarray:
        p = pg[: n * n].reshape(n, n)
        g = pg[n * n :]
        dp = a.T @ p + p @ a - (1.0 / rho) * (p @ p) + s
        dg = a.T @ g - (1.0 / rho) * (p @ g) + p @ (a @ x_t)
        return np.concatenate([dp.ravel(), dg])

    sol = solve_ivp(
        ric,
        [0.0, horizon],
        np.zeros(n * n + n),
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati ODE failed: {sol.message}")

    ts = np.linspace(0.0, horizon, n_steps + 1)
    e = x0 - x_t
    u = np.zeros((n_steps, n))
    x = np.zeros((n_steps + 1, n))
    x[0] = x_t + e
    cost = 0.0
    for i in range(n_steps):
        t = ts[i]
        pg = sol.sol(horizon - t)  # P/g at time t
        p = pg[: n * n].reshape(n, n)
        g = pg[n * n :]
        uk = -(1.0 / rho) * (p @ e + g)
        u[i] = uk
        cost += float((e @ s @ e) * dt + rho * (uk @ uk) * dt)
        e = e + dt * (a @ e + uk + a @ x_t)
        x[i + 1] = x_t + e
    energy_node = np.sum(u**2, axis=0) * dt
    energy_total = float(np.sum(energy_node))
    return {"u": u, "x": x, "energy_node": energy_node,
            "energy_total": energy_total, "cost": float(cost)}
