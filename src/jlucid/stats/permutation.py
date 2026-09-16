"""Nonparametric permutation tests (D12; used for group stats in step 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import N_PERMUTATIONS, SEED


def residualize_metric(
    values: np.ndarray,
    covariates: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """OLS-residualize a metric on covariates (pooled fit; D2-11).

    Categorical covariates should already be dummy-encoded by the caller
    (site is passed as dummies); an intercept column is added.  Rows with
    NaN in values or any covariate are kept as NaN in the output.
    """
    values = np.asarray(values, dtype=float)
    if isinstance(covariates, pd.DataFrame):
        cols = [c for c in covariates.columns if c != "intercept"]
        x = covariates[cols].copy()
        for c in x.columns:
            if x[c].dtype == object or str(x[c].dtype).startswith("category"):
                x = pd.get_dummies(x, columns=[c], drop_first=False)
        x = x.astype(float)
        x = np.asarray(x)
    else:
        x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(values)
    if x.shape[0] != n:
        raise ValueError("covariates rows must match values length")
    design = np.column_stack([np.ones(n), x])
    ok = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    resid = np.full(n, np.nan)
    if ok.sum() == 0:
        return resid
    beta, *_ = np.linalg.lstsq(design[ok], values[ok], rcond=None)
    resid[ok] = values[ok] - design[ok] @ beta
    return resid


def permutation_p_value(null: np.ndarray, observed: float) -> float:
    """Two-sided permutation p-value: (count(|null| >= |obs|) + 1) / (n + 1)."""
    null = np.asarray(null, dtype=float)
    count = int(np.sum(np.abs(null) >= abs(observed)))
    return (count + 1) / (len(null) + 1)


def two_sample_perm_test_residual(
    x: np.ndarray,
    y: np.ndarray,
    covariates: pd.DataFrame | np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict:
    """Permutation test on covariate-residualized group means (D2-11).

    Residuals come from one pooled OLS fit; the permutation then shuffles
    group labels and compares the residual mean difference two-sided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    # complete-case mask across values and covariates BEFORE splitting
    if isinstance(covariates, pd.DataFrame):
        cov_arr = covariates.select_dtypes(include=[np.number]).to_numpy()
        if cov_arr.size and cov_arr.shape[0] == len(pooled):
            ok = np.isfinite(pooled) & np.all(np.isfinite(cov_arr), axis=1)
        else:
            ok = np.isfinite(pooled)
    else:
        cov_arr = np.asarray(covariates, dtype=float)
        ok = np.isfinite(pooled) & np.all(np.isfinite(cov_arr), axis=1) if cov_arr.ndim > 1 else (np.isfinite(pooled) & np.isfinite(cov_arr))
    if not ok.all():
        pooled = pooled[ok]
        if isinstance(covariates, pd.DataFrame):
            covariates = covariates.loc[np.asarray(ok)]
        else:
            covariates = np.asarray(covariates)[ok]
    # split point = number of surviving ADHD rows (x occupies the first len(x)
    # positions of pooled, then y follows)
    n_x_filt = int(ok[: len(x)].sum())
    resid = residualize_metric(pooled, covariates)
    rx, ry = resid[: n_x_filt], resid[n_x_filt:]
    if len(rx) == 0 or len(ry) == 0:
        return {"observed": float("nan"), "p_value": float("nan"),
                "null": np.array([]), "n_perm": 0,
                "resid_x": rx, "resid_y": ry}
    obs = float(rx.mean() - ry.mean())
    rng = np.random.default_rng(seed)
    pooled_r = np.concatenate([rx, ry])
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(pooled_r)
        null[p] = perm[: len(rx)].mean() - perm[len(rx):].mean()
    return {"observed": obs, "p_value": permutation_p_value(null, obs),
            "null": null, "n_perm": n_perm,
            "resid_x": rx, "resid_y": ry}


def two_sample_perm_test(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERMUTATIONS,
                         seed: int = SEED) -> dict:
    """Two-sided permutation test on the mean difference."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pooled = np.concatenate([x, y])
    nx = len(x)
    obs = float(np.mean(x) - np.mean(y))
    rng = np.random.default_rng(seed)
    count = 0
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(pooled)
        stat = perm[:nx].mean() - perm[nx:].mean()
        null[p] = stat
        count += abs(stat) >= abs(obs)
    p_value = (count + 1) / (n_perm + 1)
    return {"observed": obs, "p_value": float(p_value), "null": null,
            "n_perm": n_perm}


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p_values, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out
