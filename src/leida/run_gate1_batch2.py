#!/usr/bin/env python3
"""
exp/37 BATCH 2 GATE 1 — oracle screen. PREREG: SIEVE_TABLE.md batch-2 section (frozen).
Gate-0 outcome recorded in gate0_batch2.json BEFORE this run:
  DEAD: W2 (both geoms, mean SB<0). V1 g050 (0.305 borderline) -> per prereg scalar gate,
        V1/V2 SB>=0.30 pass to Gate 1 (all scalars recorded).
  ALIVE: W1 g080 (13/28 comps >= 0.30; full 28-vector still fitted per prereg "one variant
         definition = one gate triple" — the vector as defined goes forward; the per-comp
         reliability is recorded for interpretation), W1 g050 (4/28; recorded),
         V1 (0.305/0.322), V2 (0.433/0.373), C1 tce (0.484, 7/8 comps).
Targets: Inatt, Hyper (T-cohort n=335) AND ADHD-Index (sites 1/3/5, n=514) — preregistered.
Oracle: in-sample ridge, alpha grid, site-residualized r. DEAD if max oracle |r| < 0.15.
ALL variants scored and recorded; survivors -> Gate 2 (familywise over everything).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]

z = np.load(f"{OUT}/batch2_features.npz", allow_pickle=True)
archive_keys = [k for k in z.files if "__" in k]

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive", "ADHD Index"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure", "ADHD Index_n"]]

# T-cohort
keep_T = []
for i, sid in enumerate(ids_all):
    if sites_cc[i] not in (3, 5, 6) or sid not in phmap.index: continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2", "3"): continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]): continue
    keep_T.append(i)
keep_T = np.array(keep_T)
y_inatt_T = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep_T])
y_hyper_T = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep_T])
site_T = sites_cc[keep_T]

# ADHD-Index cohort (exp/28 convention: sites with >=10 labeled; exp28 used {1,3,5} n=514)
keep_A = []
for i, sid in enumerate(ids_all):
    if sid not in phmap.index: continue
    if pd.isna(phmap.loc[sid, "ADHD Index_n"]): continue
    keep_A.append(i)
keep_A = np.array(keep_A)
y_idx_A = np.array([float(phmap.loc[ids_all[i], "ADHD Index_n"]) for i in keep_A])
site_A = sites_cc[keep_A]
# restrict to sites with >=10 (exp28: {1,3,5})
uniq, cnts = np.unique(site_A, return_counts=True)
keep_sites = [int(s) for s, c in zip(uniq, cnts) if c >= 10]
m = np.isin(site_A, keep_sites)
keep_A = keep_A[m]; y_idx_A = y_idx_A[m]; site_A = site_A[m]
print(f"T-cohort n={len(keep_T)} sites {dict(zip(*np.unique(site_T, return_counts=True)))}")
print(f"ADHD-Index cohort n={len(keep_A)} sites {sorted(keep_sites)}")

COHORTS = {"T": (keep_T, {"Inatt": y_inatt_T, "Hyper": y_hyper_T}, site_T),
           "A": (keep_A, {"Idx": y_idx_A}, site_A)}

def site_center_r(y, site, pred):
    uniq = np.unique(site)
    D = np.column_stack([np.ones(len(site))] + [(site == s).astype(float) for s in uniq[1:]])
    def resid(v):
        b, *_ = np.linalg.lstsq(D, v, rcond=None)
        return v - D @ b
    return float(np.corrcoef(resid(y), resid(pred))[0, 1])

def oracle(X, y, site):
    msk = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xs, ys, ss = X[msk], y[msk], site[msk]
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-12
    Xz = (Xs - mu) / sd
    best = (0.0, None)
    S = Xz.T @ Xz; k = Xz.shape[1]
    for a in ALPHAS:
        w = np.linalg.solve(S + a * np.eye(k), Xz.T @ ys)
        pred = Xz @ w
        r = site_center_r(ys, ss, pred)
        if abs(r) > abs(best[0]): best = (r, a)
    return best[0], best[1], int(msk.sum())

results = {}
for cname, (keep, ydict, site) in COHORTS.items():
    for key in archive_keys:
        X = z[key][keep]
        for tname, y in ydict.items():
            r, a, n = oracle(X, y, site)
            results[f"{key}|{cname}|{tname}"] = {"oracle_r": r, "alpha": a, "n": n}
            flag = "ALIVE" if abs(r) >= 0.15 else "dead"
            print(f"{key:20s} {cname:1s} {tname:6s} oracle {r:+.3f} (a={a}) n={n}  {flag}")

with open(f"{OUT}/gate1_batch2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved gate1_batch2_results.json")
