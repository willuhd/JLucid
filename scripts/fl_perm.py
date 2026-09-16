"""Freedman-Lane permutation inference module (fixes the 04_metrics_stats bug).

Old bug: null t's reused the OBSERVED design's pinv(X'X) while permuting the
target column -> mis-scaled null SEs. Freedman-Lane is the correct procedure:
  1. Residualize y AND the target column x_t on the covariates C (including
     intercept): y~ = y - C(C'y)^-, x~ = x_t - C(C'x_t)^-.
  2. Permute x~ (label-shuffle of the residualized target), refit the FULL
     model [1, x~_perm, C] on ORIGINAL y, recompute t from scratch
     (fresh lstsq + fresh pinv per permutation).
Familywise: max-|t| statistic across the metric family per permutation.

Also provides: literal majority-sign rule for leading eigenvectors
(count of positive loadings > half -> flip), per paper 7.
"""
import numpy as np


def residualize(y, X):
    """Remove the linear span of X (with intercept) from y."""
    Xd = np.column_stack([np.ones(len(y)), X]) if X is not None and X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def fl_perm_test(y, target, covs=None, n_perm=5000, seed=0, return_null=False):
    """Freedman-Lane permutation test for the target's coefficient t.

    y: (n,) outcome; target: (n,) predictor of interest; covs: (n, p) or None.
    Returns dict(t_obs, p_perm, beta, se) — p_perm is two-sided, (b+1)/(n_perm+1).
    """
    y = np.asarray(y, float); t_ = np.asarray(target, float)
    ok = np.isfinite(y) & np.isfinite(t_)
    if covs is not None:
        covs = np.asarray(covs, float)
        ok &= np.isfinite(covs).all(1)
    n_ok = int(ok.sum())
    y, t_ = y[ok], t_[ok]
    C = covs[ok] if covs is not None else None
    Xd = np.column_stack([np.ones(n_ok), t_] + ([C] if C is not None else []))
    ti = 1

    def fit_t(Xm):
        b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        r = y - Xm @ b
        dof = n_ok - Xm.shape[1]
        s2 = r @ r / dof
        XtXi = np.linalg.pinv(Xm.T @ Xm)
        v = b[ti] / np.sqrt(max(s2 * XtXi[ti, ti], 1e-300))
        return v if np.isfinite(v) else 0.0

    t_obs = fit_t(Xd)
    # residualized target under H0 (Freedman-Lane step 1)
    t_res = residualize(t_, C)
    rng = np.random.default_rng(seed)
    t_null = np.empty(n_perm)
    for i in range(n_perm):
        tp = t_res[rng.permutation(n_ok)]
        Xi = np.column_stack([np.ones(n_ok), tp] + ([C] if C is not None else []))
        t_null[i] = fit_t(Xi)
    t_null[~np.isfinite(t_null)] = 0.0
    p = (np.sum(np.abs(t_null) >= np.abs(t_obs)) + 1) / (n_perm + 1)
    out = dict(t=float(t_obs), p_perm=float(p), n=n_ok)
    if return_null:
        out['t_null'] = t_null
    return out


def fl_perm_test_yres(y, target, covs=None, n_perm=5000, seed=0):
    """Canonical Freedman-Lane variant: permute the reduced-model RESIDUALS of y.

    Same observable t; differs from fl_perm_test only in what is permuted
    (y-residuals vs residualized predictor). Used for the dual-scheme
    agreement check.
    """
    y = np.asarray(y, float); t_ = np.asarray(target, float)
    ok = np.isfinite(y) & np.isfinite(t_)
    if covs is not None:
        covs = np.asarray(covs, float)
        ok &= np.isfinite(covs).all(1)
    n_ok = int(ok.sum())
    y, t_ = y[ok], t_[ok]
    C = covs[ok] if covs is not None else None
    Xd = np.column_stack([np.ones(n_ok), t_] + ([C] if C is not None else []))
    ti = 1

    def fit_t(Xm):
        b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        r = y - Xm @ b
        dof = n_ok - Xm.shape[1]
        s2 = r @ r / dof
        XtXi = np.linalg.pinv(Xm.T @ Xm)
        v = b[ti] / np.sqrt(max(s2 * XtXi[ti, ti], 1e-300))
        return v if np.isfinite(v) else 0.0

    t_obs = fit_t(Xd)
    y_res = residualize(y, C)
    bC, *_ = np.linalg.lstsq(np.column_stack([np.ones(n_ok)] + ([C] if C is not None else [])), y, rcond=None)
    Crep = np.column_stack([np.ones(n_ok)] + ([C] if C is not None else []))
    y_recon = Crep @ bC
    rng = np.random.default_rng(seed)
    t_null = np.empty(n_perm)
    for i in range(n_perm):
        y_p = y_recon + y_res[rng.permutation(n_ok)]
        t_null[i] = fit_t_with_y(Xd, y_p)
    t_null[~np.isfinite(t_null)] = 0.0
    p = (np.sum(np.abs(t_null) >= np.abs(t_obs)) + 1) / (n_perm + 1)
    return dict(t=float(t_obs), p_perm=float(p), n=n_ok)


def fit_t_with_y(Xm, yv):
    b, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    r = yv - Xm @ b
    dof = len(yv) - Xm.shape[1]
    s2 = r @ r / dof
    XtXi = np.linalg.pinv(Xm.T @ Xm)
    v = b[1] / np.sqrt(max(s2 * XtXi[1, 1], 1e-300))
    return v if np.isfinite(v) else 0.0


def fl_family_test(metrics, y, covs=None, n_perm=5000, seed=0):
    """Familywise Freedman-Lane: single permutation set, max-|t| across the family.

    metrics: dict name -> (n,) predictor. All must share the same valid mask.
    Returns per-metric t, familywise p (max-stat corrected), and the family max-t null.
    """
    y = np.asarray(y, float)
    names = list(metrics)
    M = np.column_stack([np.asarray(metrics[n_], float) for n_ in names])
    ok = np.isfinite(y) & np.isfinite(M).all(1)
    if covs is not None:
        ok &= np.isfinite(np.asarray(covs, float)).all(1)
    n_ok = int(ok.sum())
    y_ok = y[ok]; M_ok = M[ok]
    C = np.asarray(covs, float)[ok] if covs is not None else None
    k = len(names)

    def fit_ts(Xm_extra):
        # Xm_extra: (n, k) permuted residualized targets
        ts = np.empty(k)
        for j in range(k):
            Xd = np.column_stack([np.ones(n_ok), Xm_extra[:, j]] + ([C] if C is not None else []))
            b, *_ = np.linalg.lstsq(Xd, y_ok, rcond=None)
            r = y_ok - Xd @ b
            dof = n_ok - Xd.shape[1]
            s2 = r @ r / dof
            XtXi = np.linalg.pinv(Xd.T @ Xd)
            ts[j] = b[1] / np.sqrt(s2 * XtXi[1, 1])
        return ts

    t_obs = fit_ts(M_ok)
    M_res = np.column_stack([residualize(M_ok[:, j], C) for j in range(k)])
    rng = np.random.default_rng(seed)
    max_null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n_ok)
        Mm = M_res[p, :]  # same permutation applied to all family members
        ts = fit_ts(Mm)
        max_null[i] = np.abs(ts).max()
    fw = np.empty(k)
    for j in range(k):
        fw[j] = (np.sum(max_null >= abs(t_obs[j])) + 1) / (n_perm + 1)
    return {names[j]: dict(t=float(t_obs[j]), p_famwise=float(fw[j])) for j in range(k)}, max_null


def literal_majority_sign(v):
    """Paper-7 literal rule: if MORE THAN HALF the loadings are positive, flip.

    (The old extraction used mean(v)>0 which differs when loadings are
    non-uniform.) Applied row-wise to v: (T, N) -> sign-corrected copy.
    """
    v = np.atleast_2d(v)
    flip = (v > 0).sum(1) > v.shape[1] / 2.0
    out = v.copy()
    out[flip] *= -1.0
    return out
