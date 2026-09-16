"""ADHD-200 symptom-score hygiene and H4 (inattention) correlations.

Does not write the QC table. Sentinels stay in the source files; callers
get a cleaned *copy*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

from .permutation import residualize_metric


SENTINEL_LOW = -100.0
SENTINEL_HIGH = 900.0
SYMPTOM_COLS = ("inattentive", "hyper_impulsive", "adhd_index")
H4_JSPACE = ("C_k", "r_eff", "dmn_contamination", "L_FPN", "L_DAN")


def clean_sentinels(
    values,
    *,
    low: float = SENTINEL_LOW,
    high: float = SENTINEL_HIGH,
) -> np.ndarray:
    """Map ADHD-200 missing codes (−999, 999, non-finite) to NaN.

    Valid questionnaire scores in this release sit well inside (0, 200).
    """
    x = pd.to_numeric(values, errors="coerce")
    x = np.asarray(x, dtype=float)
    out = x.copy()
    bad = ~np.isfinite(out) | (out <= float(low)) | (out >= float(high))
    out[bad] = np.nan
    return out


def clean_symptom_frame(df: pd.DataFrame, cols: tuple[str, ...] = SYMPTOM_COLS) -> pd.DataFrame:
    """Return a copy with sentinel symptom scores replaced by NaN."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = clean_sentinels(out[c])
    return out


def rank_within(series: pd.Series, group: pd.Series) -> np.ndarray:
    """Average ranks inside each group; NaN stays NaN."""
    s = pd.to_numeric(series, errors="coerce")
    g = pd.Series(group).astype(str)
    ranked = s.groupby(g, dropna=False).rank(method="average")
    return ranked.to_numpy(dtype=float)


def spearman_rho(x, y) -> dict:
    """Spearman ρ and two-sided p. NaN if fewer than 5 finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    empty = {"rho": float("nan"), "p": float("nan"), "n": n}
    if n < 5:
        return empty
    if float(np.std(x[m], ddof=1)) == 0.0 or float(np.std(y[m], ddof=1)) == 0.0:
        return empty
    rho, p = spearmanr(x[m], y[m])
    return {"rho": float(rho), "p": float(p), "n": n}


def steiger_t(r_xy: float, r_xz: float, r_yz: float, n: int) -> dict:
    """Williams/Steiger test of r(x,y) vs r(x,z) on the same sample.

    Two-sided p; ``t`` has the sign of ``r_xy - r_xz``.
    """
    n = int(n)
    if n < 5 or not all(np.isfinite(v) for v in (r_xy, r_xz, r_yz)):
        return {"t": float("nan"), "p": float("nan"), "df": max(n - 3, 0), "n": n}
    r_xy = float(np.clip(r_xy, -0.999999, 0.999999))
    r_xz = float(np.clip(r_xz, -0.999999, 0.999999))
    r_yz = float(np.clip(r_yz, -0.999999, 0.999999))
    det = (1.0 - r_xy ** 2 - r_xz ** 2 - r_yz ** 2) + 2.0 * r_xy * r_xz * r_yz
    num = (r_xy - r_xz) * np.sqrt((n - 3) * (1.0 + r_yz))
    den = np.sqrt(max(2.0 * det, 0.0))
    if den <= 0.0 or not np.isfinite(den):
        return {"t": float("nan"), "p": float("nan"), "df": n - 3, "n": n}
    tval = float(num / den)
    df = n - 3
    p = float(2.0 * student_t.sf(abs(tval), df))
    return {"t": tval, "p": p, "df": int(df), "n": n}


def _pair(frame: pd.DataFrame, a: str, b: str) -> dict:
    return spearman_rho(frame[a], frame[b])


def h4_block(frame: pd.DataFrame, jcol: str, inatt: str, hyper: str) -> dict:
    """One J-space metric vs inattention and hyperactivity."""
    r_in = _pair(frame, jcol, inatt)
    r_hy = _pair(frame, jcol, hyper)
    r_ih = _pair(frame, inatt, hyper)
    n = int(min(r_in["n"], r_hy["n"]))
    ste = steiger_t(r_in["rho"], r_hy["rho"], r_ih["rho"], n)
    return {
        "metric": jcol,
        "n": n,
        "rho_inatt": r_in["rho"],
        "p_inatt": r_in["p"],
        "rho_hyper": r_hy["rho"],
        "p_hyper": r_hy["p"],
        "rho_inatt_hyper": r_ih["rho"],
        "steiger_t": ste["t"],
        "steiger_p": ste["p"],
        "abs_inatt_gt_abs_hyper": (
            bool(abs(r_in["rho"]) > abs(r_hy["rho"]))
            if np.isfinite(r_in["rho"]) and np.isfinite(r_hy["rho"]) else False
        ),
    }


def evaluate_h4(
    frame: pd.DataFrame,
    *,
    j_cols: tuple[str, ...] = H4_JSPACE,
    inatt_col: str = "inattentive",
    hyper_col: str = "hyper_impulsive",
    measure_col: str = "adhd_measure",
    dx_col: str = "dx_group",
) -> dict:
    """H4 on an already-merged, sentinel-cleaned table.

    Primary: ADHD-only, scores ranked within ``adhd_measure`` so the two
    instruments share a scale.  Per-measure raw scores and a TDC-included
    pool are sensitivities, not the headline.
    """
    work = clean_symptom_frame(frame, cols=(inatt_col, hyper_col))
    adhd = work[work[dx_col].astype(str) == "ADHD"].copy()
    adhd = adhd.dropna(subset=[inatt_col, hyper_col])
    if measure_col in adhd.columns:
        adhd["inatt_rank"] = rank_within(adhd[inatt_col], adhd[measure_col])
        adhd["hyper_rank"] = rank_within(adhd[hyper_col], adhd[measure_col])
    else:
        adhd["inatt_rank"] = adhd[inatt_col].rank(method="average")
        adhd["hyper_rank"] = adhd[hyper_col].rank(method="average")

    primary = [
        h4_block(adhd, j, "inatt_rank", "hyper_rank")
        for j in j_cols if j in adhd.columns
    ]

    per_measure = []
    if measure_col in adhd.columns:
        for meas, sub in adhd.groupby(measure_col, dropna=False):
            if len(sub) < 10:
                continue
            for j in j_cols:
                if j not in sub.columns:
                    continue
                row = h4_block(sub, j, inatt_col, hyper_col)
                row["adhd_measure"] = None if pd.isna(meas) else (
                    int(meas) if float(meas).is_integer() else meas
                )
                per_measure.append(row)

    pooled = work.dropna(subset=[inatt_col, hyper_col]).copy()
    if measure_col in pooled.columns:
        pooled["inatt_rank"] = rank_within(pooled[inatt_col], pooled[measure_col])
        pooled["hyper_rank"] = rank_within(pooled[hyper_col], pooled[measure_col])
    else:
        pooled["inatt_rank"] = pooled[inatt_col].rank(method="average")
        pooled["hyper_rank"] = pooled[hyper_col].rank(method="average")
    sensitivity_all = [
        h4_block(pooled, j, "inatt_rank", "hyper_rank")
        for j in j_cols if j in pooled.columns
    ]

    residual = []
    cov_src = adhd.copy()
    if {"age", "site"}.issubset(cov_src.columns) and len(cov_src) >= 10:
        sex = cov_src["gender"] if "gender" in cov_src.columns else cov_src.get("sex")
        motion = cov_src["mean_fd"] if "mean_fd" in cov_src.columns else cov_src.get("max_motion_mm")
        cov = pd.DataFrame({
            "age": pd.to_numeric(cov_src["age"], errors="coerce"),
            "sex": pd.to_numeric(sex, errors="coerce"),
            "motion": pd.to_numeric(motion, errors="coerce"),
            "site": cov_src["site"].astype(str),
        })
        cov = pd.get_dummies(cov, columns=["site"], drop_first=False)
        for j in j_cols:
            if j not in cov_src.columns:
                continue
            jr = residualize_metric(cov_src[j].to_numpy(), cov)
            ir = residualize_metric(cov_src["inatt_rank"].to_numpy(), cov)
            hr = residualize_metric(cov_src["hyper_rank"].to_numpy(), cov)
            tmp = pd.DataFrame({j: jr, "inatt_rank": ir, "hyper_rank": hr})
            residual.append(h4_block(tmp, j, "inatt_rank", "hyper_rank"))

    ck = next((r for r in primary if r["metric"] == "C_k"), None)
    headline = {
        "supported": False,
        "reason": "C_k missing",
    }
    if ck is not None:
        rho = ck["rho_inatt"]
        # predicted: higher C_k → lower inattention, and |ρ_inatt| > |ρ_hyper|
        direction = bool(np.isfinite(rho) and rho < 0 and ck["p_inatt"] < 0.05)
        stronger = bool(ck["abs_inatt_gt_abs_hyper"] and ck["steiger_p"] < 0.05)
        if direction and stronger:
            headline = {"supported": True, "reason": "C_k vs inatt negative and reliably stronger than hyper"}
        elif direction:
            headline = {
                "supported": False,
                "reason": "C_k vs inatt is negative at p<0.05 but not reliably stronger than hyper",
            }
        elif np.isfinite(rho) and abs(rho) < 0.15 and ck["p_inatt"] >= 0.05:
            headline = {
                "supported": False,
                "reason": "ADHD-only C_k vs inattention is null (H4 not supported)",
            }
        else:
            headline = {
                "supported": False,
                "reason": (
                    f"C_k vs inatt rho={rho:.3f} p={ck['p_inatt']:.3g}; "
                    "predicted negative + stronger-than-hyper not met"
                ),
            }

    return {
        "n_adhd_valid": int(len(adhd)),
        "n_pooled_valid": int(len(pooled)),
        "primary": "ADHD-only, Spearman on ranks within adhd_measure",
        "headline": headline,
        "primary_rows": primary,
        "residualized_rows": residual,
        "per_measure_rows": per_measure,
        "sensitivity_all_dx_rows": sensitivity_all,
    }
