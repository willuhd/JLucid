"""Effect sizes and correlation comparisons (step-2 Phase E, D2-11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .permutation import fdr_bh


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Cohen's d with pooled SD."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sd_pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                        / (len(x) + len(y) - 2))
    if sd_pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / sd_pooled)


def fisher_z_corr(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Compare two independent Pearson correlations via Fisher z."""
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    r2 = float(np.clip(r2, -0.999999, 0.999999))
    z1 = float(np.arctanh(r1))
    z2 = float(np.arctanh(r2))
    se = float(np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1)))
    z = (z1 - z2) / se
    p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    return z, p


def pearson_r(x, y) -> float:
    """Pearson r; NaN if fewer than 3 finite pairs or a constant vector."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    if float(np.std(x[m], ddof=1)) == 0.0 or float(np.std(y[m], ddof=1)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def h4_avg_modal(
    ind: pd.DataFrame,
    qc: pd.DataFrame,
    covariates: pd.DataFrame,
    min_n: int = 10,
    site_col: str = "site",
) -> dict:
    """Site-aware H4: individual avg vs modal controllability.

    ``ind`` has columns ``sid``, ``avg``, ``modal``. ``qc`` has
    ``scan_dir_id``, ``dx_group``, and ``site_col``. ``covariates`` is
    indexed by ``scan_dir_id`` (dummy-encoded, same as group stats).

    Headline test = Fisher-z of the two within-group correlations after
    residualizing avg and modal on the pooled covariates (includes site).
    Per-site raw r / Fisher-z are returned for the published table; they
    are not the FDR-family statistic.
    """
    from .permutation import residualize_metric

    merged = ind.copy()
    merged["sid"] = merged["sid"].astype(str)
    q = qc.copy()
    q["scan_dir_id"] = q["scan_dir_id"].astype(str)
    merged = merged.merge(
        q[["scan_dir_id", "dx_group", site_col]],
        left_on="sid", right_on="scan_dir_id", how="inner",
    )
    merged = merged.dropna(subset=["avg", "modal", "dx_group"])

    pooled = {}
    for grp in ("ADHD", "TDC", "all"):
        sub = merged if grp == "all" else merged[merged.dx_group == grp]
        pooled[grp] = {
            "r": pearson_r(sub["avg"], sub["modal"]),
            "n": int(len(sub)),
        }

    per_site = []
    for site, g in merged.groupby(site_col):
        adhd = g[g.dx_group == "ADHD"]
        tdc = g[g.dx_group == "TDC"]
        row = {
            "site": str(site),
            "n_adhd": int(len(adhd)),
            "n_tdc": int(len(tdc)),
            "r_adhd": pearson_r(adhd["avg"], adhd["modal"]) if len(adhd) >= 3 else float("nan"),
            "r_tdc": pearson_r(tdc["avg"], tdc["modal"]) if len(tdc) >= 3 else float("nan"),
            "z": float("nan"),
            "p": float("nan"),
            "tested": False,
        }
        if len(adhd) >= min_n and len(tdc) >= min_n:
            row["z"], row["p"] = fisher_z_corr(
                row["r_adhd"], len(adhd), row["r_tdc"], len(tdc),
            )
            row["tested"] = True
        per_site.append(row)

    # residualize avg and modal on the same pooled design, then correlate
    # within group (one within-site-adjusted r per group)
    ccov = covariates.loc[merged["scan_dir_id"]].reset_index(drop=True)
    avg_r = residualize_metric(merged["avg"].to_numpy(), ccov)
    mod_r = residualize_metric(merged["modal"].to_numpy(), ccov)
    merged = merged.assign(avg_resid=avg_r, modal_resid=mod_r)
    resid = {}
    for grp in ("ADHD", "TDC"):
        sub = merged[merged.dx_group == grp]
        resid[grp] = {
            "r": pearson_r(sub["avg_resid"], sub["modal_resid"]),
            "n": int(len(sub)),
        }
    z_resid, p_resid = fisher_z_corr(
        resid["ADHD"]["r"], resid["ADHD"]["n"],
        resid["TDC"]["r"], resid["TDC"]["n"],
    )
    # Headline H4: Stouffer combination of per-site Fisher-z. Residualizing
    # avg/modal on site dummies and then pooling still leaves a composition
    # residue (r ~ 0.90 vs 0.98 here); the per-site tests are the
    # within-site evidence.
    tested = [r for r in per_site if r["tested"] and np.isfinite(r["z"])]
    if tested:
        zs = np.array([r["z"] for r in tested], dtype=float)
        z_stouffer = float(zs.sum() / np.sqrt(len(zs)))
        p_stouffer = 2.0 * (1.0 - float(norm.cdf(abs(z_stouffer))))
    else:
        z_stouffer, p_stouffer = float("nan"), float("nan")
    return {
        "pooled": pooled,
        "per_site": per_site,
        "residualized": resid,
        "z_residualized": float(z_resid),
        "p_residualized": float(p_resid),
        "z": float(z_stouffer),
        "p": float(p_stouffer),
        "n_sites_tested": int(len(tested)),
        "n_adhd": resid["ADHD"]["n"],
        "n_tdc": resid["TDC"]["n"],
    }


def fdr_table(p_values: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """FDR-corrected p-value table sorted by raw p."""
    p = np.asarray(p_values, dtype=float)
    q = fdr_bh(p)
    df = pd.DataFrame({"label": labels, "p_raw": p, "p_fdr": q})
    return df.sort_values("p_raw").reset_index(drop=True)
