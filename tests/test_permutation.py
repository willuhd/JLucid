import numpy as np
import pandas as pd

from jlucid.stats.permutation import (
    fdr_bh,
    permutation_p_value,
    residualize_metric,
    two_sample_perm_test,
    two_sample_perm_test_residual,
)


def test_two_sample_perm_test_null_uniform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = rng.normal(size=30)
    res = two_sample_perm_test(x, y, n_perm=500, seed=1)
    assert 0.0 < res["p_value"] <= 1.0
    assert res["null"].shape == (500,)


def test_two_sample_perm_test_rejects_shifted():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 40)
    y = rng.normal(2, 1, 40)
    res = two_sample_perm_test(x, y, n_perm=999, seed=3)
    assert res["p_value"] < 0.01


def test_fdr_bh_monotonic():
    p = np.array([0.5, 0.01, 0.03, 0.001, 0.2])
    q = fdr_bh(p)
    order = np.argsort(p)
    assert (q[order] <= 1.0).all()
    assert np.all(np.diff(q[order]) >= 0)


def test_residualize_removes_linear_covariate():
    rng = np.random.default_rng(0)
    n = 200
    cov = rng.normal(size=n)
    y = 2.0 * cov + rng.normal(0, 0.5, n)
    resid = residualize_metric(y, pd.DataFrame({"cov": cov}))
    assert np.isfinite(resid).all()
    assert abs(resid.mean()) < 1e-8
    assert abs(np.corrcoef(resid, cov)[0, 1]) < 1e-8


def test_residualize_preserves_nan():
    rng = np.random.default_rng(1)
    n = 50
    cov = rng.normal(size=n)
    y = rng.normal(size=n)
    y[3] = np.nan
    cov[10] = np.nan
    resid = residualize_metric(y, pd.DataFrame({"cov": cov}))
    assert np.isnan(resid[3]) and np.isnan(resid[10])
    assert np.isfinite(resid).sum() == n - 2


def test_permutation_p_value_hand_computed():
    null = np.array([-0.5, 0.2, -0.1, 1.0, 0.4])
    # |null| >= 0.3: -0.5, 0.4, 1.0 -> 3
    assert permutation_p_value(null, 0.3) == (3 + 1) / (5 + 1)
    assert permutation_p_value(null, 2.0) == (0 + 1) / (5 + 1)


def test_residual_permutation_not_spuriously_significant():
    rng = np.random.default_rng(2)
    n_per = 40
    group = np.array([0] * n_per + [1] * n_per)
    cov = rng.normal(size=2 * n_per)
    y = 3.0 * cov + rng.normal(0, 1.0, 2 * n_per)
    cov_df = pd.DataFrame({"cov": cov,
                           "site": np.random.default_rng(3).integers(0, 3, 2 * n_per).astype(str)})
    res = two_sample_perm_test_residual(y[group == 0], y[group == 1], cov_df,
                                       n_perm=999, seed=3)
    assert res["p_value"] > 0.01


def test_residual_permutation_detects_true_group_effect():
    rng = np.random.default_rng(4)
    n_per = 40
    cov = rng.normal(size=2 * n_per)
    eff = np.array([0.0] * n_per + [1.5] * n_per)
    y = 2.0 * cov + eff + rng.normal(0, 1.0, 2 * n_per)
    res = two_sample_perm_test_residual(y[:n_per], y[n_per:],
                                       pd.DataFrame({"cov": cov}), n_perm=999, seed=5)
    assert res["p_value"] < 0.05


def test_residual_permutation_alignment_requires_group_order():
    """Regression for P1: covariates must be in pooled (ADHD-then-TDC) order.

    If covariates are passed in interleaved subject order, the residualized
    permutation silently mispairs values with covariates.  Here we build a
    strong covariate that differs by group; with correct alignment the test
    removes the covariate effect, while the misaligned path would be biased.
    """
    rng = np.random.default_rng(7)
    n_per = 40
    # subjects in interleaved order: ADHD/TDC alternating
    group = np.array([0, 1] * n_per)  # 0=ADHD first half? use explicit
    # build x (ADHD) and y (TDC) with a strong confound
    cov_adhd = rng.normal(3.0, 0.5, n_per)
    cov_tdc = rng.normal(0.0, 0.5, n_per)
    y_adhd = 2.0 * cov_adhd + rng.normal(0, 0.1, n_per)
    y_tdc = 2.0 * cov_tdc + rng.normal(0, 0.1, n_per)
    # interleaved subject order
    cov_inter = np.concatenate([np.column_stack([cov_adhd, cov_tdc])[i] for i in range(n_per)])
    # Actually build aligned: x-then-y
    cov_aligned = np.concatenate([cov_adhd, cov_tdc])
    res_aligned = two_sample_perm_test_residual(y_adhd, y_tdc, pd.DataFrame({"cov": cov_aligned}), n_perm=999, seed=1)
    # misaligned: shuffle cov to interleaved order (wrong pairing)
    perm = np.empty(2 * n_per)
    perm[0::2] = cov_aligned[:n_per]
    perm[1::2] = cov_aligned[n_per:]
    res_wrong = two_sample_perm_test_residual(y_adhd, y_tdc, pd.DataFrame({"cov": perm}), n_perm=999, seed=1)
    # Correctly residualized: the group confound is removed -> not tiny p.
    assert res_aligned["p_value"] > 0.01
    # The wrong pairing should generally produce a different (often tiny) p.
    assert res_wrong["p_value"] != res_aligned["p_value"]


def test_residual_d_matches_observed_sign():
    """F4: Cohen's d on residuals must share the sign of the tested contrast.

    Raw ADHD mean can exceed TDC while the residual mean difference is
    negative (a confound). Headline d is computed on resid_x / resid_y.
    """
    from jlucid.stats.effects import cohens_d

    rng = np.random.default_rng(11)
    n = 40
    # strong confound: ADHD have higher covariate and therefore higher raw y,
    # but a negative residual group effect
    cov_adhd = rng.normal(2.0, 0.3, n)
    cov_tdc = rng.normal(0.0, 0.3, n)
    y_adhd = 3.0 * cov_adhd - 1.0 + rng.normal(0, 0.05, n)
    y_tdc = 3.0 * cov_tdc + rng.normal(0, 0.05, n)
    assert y_adhd.mean() > y_tdc.mean()
    cov = pd.DataFrame({"cov": np.concatenate([cov_adhd, cov_tdc])})
    res = two_sample_perm_test_residual(y_adhd, y_tdc, cov, n_perm=200, seed=1)
    d_raw = cohens_d(y_adhd, y_tdc)
    d_res = cohens_d(res["resid_x"], res["resid_y"])
    assert d_raw > 0
    assert res["observed"] < 0
    assert np.sign(d_res) == np.sign(res["observed"])
    assert "resid_x" in res and "resid_y" in res
