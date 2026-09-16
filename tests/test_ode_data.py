import numpy as np
import pandas as pd
import pytest

from jlucid.ode.data import make_folds, soft_state_prob, valid_windows


def test_soft_state_prob_onehot_and_censor():
    labels = np.array([0, 1, -1, 2, 0], dtype=np.int8)
    p = soft_state_prob(labels, k=3)
    assert p.shape == (5, 3)
    assert np.allclose(p[0], [1, 0, 0])
    assert np.allclose(p[1], [0, 1, 0])
    assert np.allclose(p[2], 0)         # censored row is all-zero
    assert np.allclose(p[3], [0, 0, 1])


def test_soft_state_prob_ema_sums_to_one_on_valid():
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int8)
    p = soft_state_prob(labels, k=3, ema_window=3)
    valid = labels >= 0
    assert np.allclose(p[valid].sum(axis=1), 1.0)
    assert np.allclose(p[~valid], 0.0)


def test_valid_windows_counts():
    mask = np.ones(12, dtype=bool)
    mask[5] = False
    w = valid_windows(mask, 4)
    assert w.shape == (12,)
    assert not w[5]  # window ending at censored frame invalid
    assert w[4]      # window 1..4 fully usable


def test_make_folds_stratified_balanced():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "dx_group": np.where(rng.random(n) < 0.4, "ADHD", "TDC"),
        "site": rng.choice(["1", "3", "5"], n),
    })
    fold = make_folds(df, n_splits=5, seed=42)
    assert set(np.unique(fold)) == {0, 1, 2, 3, 4}
    tab = pd.crosstab(fold, df["dx_group"])
    assert (tab["ADHD"] / tab.sum(axis=1)).std() < 0.05

