#!/usr/bin/env python3
"""
exp/37 BATCH 5 GATE 1 — oracle screen. Prereg: SIEVE_TABLE.md batch-5 + Gate-0 outcome:
  ALIVE: H1 v_ab (SB 0.376, 21/28), H2 mag (0.453), H3 hvar (0.417, 8/8), K1 signed (0.563, 14/14)
  DEAD at Gate 0: P1 acf (0.167, 0/8) — NOT fitted.
  DROPPED (ill-posed): K2 CA (see invalidation note).
Estimator: pooled site-centered oracle ridge (original convention). Targets: Inatt, Hyper (T),
Idx (A). DEAD if max oracle |r|<0.15.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]

z = np.load(f"{OUT}/batch5_features.npz", allow_pickle=True)
archive_keys = ["H1__vab", "H2__mag", "H3__hvar", "K1__signed"]

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, _, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive", "ADHD Index"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure", "ADHD Index_n"]]

keep_T = []
for i, sid in enumerate(ids_all):
    if sites_cc[i] not in (3,5,6) or sid not in phmap.index: continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2","3"): continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]): continue
    keep_T.append(i)
keep_T = np.array(keep_T)
y_inatt_T = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep_T])
y_hyper_T = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep_T])
site_T = sites_cc[keep_T]

keep_A = []
for i, sid in enumerate(ids_all):
    if sid not in phmap.index or pd.isna(phmap.loc[sid, "ADHD Index_n"]): continue
    keep_A.append(i)
keep_A = np.array(keep_A)
y_idx_A = np.array([float(phmap.loc[ids_all[i], "ADHD Index_n"]) for i in keep_A])
site_A = sites_cc[keep_A]
uniq, cnts = np.unique(site_A, return_counts=True)
ks = [int(s) for s, c in zip(uniq, cnts) if c >= 10]
mA = np.isin(site_A, ks)
keep_A, y_idx_A, site_A = keep_A[mA], y_idx_A[mA], site_A[mA]

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
            print(f"{key:12s} {cname:1s} {tname:6s} oracle {r:+.3f} (a={a}) n={n}  {flag}")

with open(f"{OUT}/gate1_batch5_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved gate1_batch5_results.json")
