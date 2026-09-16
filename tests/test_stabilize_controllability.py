import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from jlucid.control.controllability import (
    aggregate_controllability,
    average_controllability,
    modal_controllability,
    network_masks,
)
from jlucid.control.stabilize import assert_hurwitz, hurwitz_shift, stabilize


def test_stabilize_hurwitz():
    rng = np.random.default_rng(0)
    fc = rng.normal(size=(6, 6))
    fc = (fc + fc.T) / 2
    a = stabilize(fc)
    assert_hurwitz(a)
    assert np.linalg.eigvalsh(a).max() < 0


def test_stabilize_formula():
    fc = np.array([[2.0, 0.0], [0.0, 1.0]])
    a = stabilize(fc, c=1.0)
    assert np.allclose(a, np.abs(fc) / 3.0 - np.eye(2))


def test_hurwitz_shift_symmetric_negative_eigs():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(5, 5))
    h = hurwitz_shift(a, margin=1e-3)
    assert_hurwitz(h)
    ev = np.linalg.eigvalsh(h)
    assert ev.max() < 0
    # not the FC |A| recipe
    assert not np.allclose(h, stabilize(a))


def test_average_controllability_matches_numerical_gramian():
    rng = np.random.default_rng(1)
    n = 6
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = (q * (-np.arange(1, n + 1))) @ q.T
    phi = average_controllability(a)
    ts = np.linspace(0, 10, 8001)
    gram = np.zeros((n, n))
    dt = ts[1] - ts[0]
    for t in ts:
        e = expm(a * t)
        gram += (e @ e.T) * dt
    trace_gram = np.trace(gram)
    # phi_i summed over nodes equals trace of Gramian with B=I
    assert phi.sum() == pytest.approx(trace_gram, rel=0.05)
    w, v = np.linalg.eigh(a)
    assert np.allclose(phi, (v**2 / (-2.0 * w)).sum(axis=1))


def test_modal_controllability_bounds():
    rng = np.random.default_rng(2)
    n = 5
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = (q * (-np.linspace(0.2, 1.0, n))) @ q.T
    psi = modal_controllability(a)
    assert (psi > 0).all() and (psi < 1).all()


def test_network_masks_and_aggregation():
    membership = pd.DataFrame({
        "idx": [1, 2, 3, 4, 5, 6],
        "network_group": ["DMN", "DMN", "VIS", "VIS", "Subcortical", "Subcortical"],
    })
    masks = network_masks(membership)
    assert masks["DMN"].sum() == 2
    assert masks["VIS"].sum() == 2
    assert masks["Subcortical"].sum() == 2
    phi = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    psi = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    agg = aggregate_controllability(phi, psi, membership)
    ind = agg[(agg.metric == "avg") & (agg.level == "individual")].value.iloc[0]
    assert ind == pytest.approx(phi.mean())
    dmn = agg[(agg.metric == "avg") & (agg.group == "DMN")].value.iloc[0]
    assert dmn == pytest.approx(1.5)
