import numpy as np
import pytest

from jlucid.ode.jlucid2.comparison import bootstrap_ci, fit_linear_dynamics, predict_linear, r2


def test_fit_linear_dynamics_recovers_planted_a():
    rng = np.random.default_rng(0)
    d = 6
    A = np.diag([-0.05, -0.08, -0.11, -0.14, -0.17, -0.20])
    n = 4000
    m = 1.0
    z = np.full((n, d), m)
    for t in range(n - 1):
        z[t + 1] = m + (np.eye(d) + A) @ (z[t] - m) + rng.normal(0, 1e-3, d)
    labels = np.zeros(n, dtype=int)
    mask = np.ones(n, dtype=bool)
    # center so the through-origin LS matches the planted A
    zc = z - z.mean(axis=0)
    A_global, A_state = fit_linear_dynamics(zc, labels, mask, k_states=3)
    assert np.allclose(A_global, A, atol=0.05)
    assert np.allclose(A_state[0], A, atol=0.05)


def test_predict_linear_nan_on_missing_state():
    A_state = np.full((3, 4, 4), np.nan)
    A_state[1] = np.eye(4) * -0.1
    labels = np.array([1, 1, 1])
    out = predict_linear(np.ones(4), np.eye(4), 3, labels, A_state)
    assert out is not None
    labels2 = np.array([1, 2, 1])  # state 2 missing -> NaN
    assert predict_linear(np.ones(4), np.eye(4), 3, labels2, A_state) is None


def test_r2_and_bootstrap():
    y = np.arange(10.0)[:, None]
    assert r2(y, y) == 1.0
    mean, lo, hi = bootstrap_ci(np.arange(10.0))
    assert mean == pytest.approx(4.5)
    assert lo <= mean <= hi
