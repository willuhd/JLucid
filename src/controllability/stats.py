"""Statistical testing — strictly paper 6 §2.6.
Primary: 5000-permutation (shuffle ADHD/HC labels) + FDR q<0.05.
Secondary: Welch's t, Cohen's d.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from typing import Dict, Tuple

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a-1)*va + (n_b-1)*vb) / (n_a+n_b-2 + 1e-12))
    return float((ma - mb) / (pooled + 1e-12))

def welch_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p_uncorrected": float(p), "cohens_d": cohens_d(a,b),
            "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=1)),
            "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=1))}

def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 42, alternative: str = "two-sided") -> Dict[str, float]:
    """Permutation test for difference in means (paper 6 §2.6).
    P = (#|Tperm| >= |Tobs|)/B  for two-sided.
    Returns p-value and observed statistic.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a,b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        pa = pooled[:n_a]
        pb = pooled[n_a:]
        perm_diff = float(np.mean(pa) - np.mean(pb))
        if alternative == "two-sided":
            if abs(perm_diff) >= abs(obs) - 1e-12:
                count += 1
        elif alternative == "greater":
            if perm_diff >= obs - 1e-12:
                count += 1
        else:
            if perm_diff <= obs + 1e-12:
                count += 1
    # add 1 to avoid p=0 (standard)
    p = (count + 1) / (n_perm + 1)
    return {"obs_diff": obs, "p_perm": float(p), "n_perm": n_perm, "count": int(count)}

def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR. Returns (p_fdr, rejected bool)."""
    from statsmodels.stats.multitest import multipletests
    pvals = np.asarray(pvals, float)
    rejected, p_corr, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return p_corr, rejected

def test_levels(avg_a, modal_a, avg_b, modal_b, network_avg_a=None, network_avg_b=None,
                network_modal_a=None, network_modal_b=None,
                n_perm=5000, q=0.05, seed=42):
    """Test at individual, network, node levels. Returns dict.
    Inputs: avg_a/modal_a are 1D per-subject scalars or 2D (subjects x nodes) as needed.
    """
    out = {}
    # individual
    out["individual_avg"] = {**welch_ttest(avg_a, avg_b), **permutation_test(avg_a, avg_b, n_perm=n_perm, seed=seed)}
    out["individual_modal"] = {**welch_ttest(modal_a, modal_b), **permutation_test(modal_a, modal_b, n_perm=n_perm, seed=seed+1)}
    # network
    if network_avg_a is not None:
        # shape (n_nets, ) vs dict per net
        nets = list(network_avg_a.keys())
        p_avgs = []
        p_mods = []
        for net in nets:
            pa = permutation_test(np.asarray(network_avg_a[net]), np.asarray(network_avg_b[net]), n_perm=n_perm, seed=seed)
            pm = permutation_test(np.asarray(network_modal_a[net]), np.asarray(network_modal_b[net]), n_perm=n_perm, seed=seed)
            p_avgs.append(pa["p_perm"]); p_mods.append(pm["p_perm"])
        _, rej_a = bh_fdr(np.array(p_avgs), q=q)
        _, rej_m = bh_fdr(np.array(p_mods), q=q)
        out["network"] = {n: {"p_perm_avg": float(p_avgs[i]), "p_perm_modal": float(p_mods[i]),
                              "sig_avg_fdr": bool(rej_a[i]), "sig_modal_fdr": bool(rej_m[i])} for i,n in enumerate(nets)}
    return out
