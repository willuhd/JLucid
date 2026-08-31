#!/usr/bin/env python3
"""
exp/37 GATE 1 — oracle ridge screen on T-cohort targets (PREREG exp/37).

Protocol (frozen in exp/37_leida/PREREG.md BEFORE this run):
- Cohort: T-cohort = sites 3/5/6, ADHD Measure in (2,3), Inattentive available (n=336 expected).
- Targets: Inattentive T (primary), Hyper/Impulsive T (secondary).
- Oracle = IN-SAMPLE ridge (alpha grid {0.1,1,10,100,1000,3000,1e4,3e5} per PREREG: we use
  the PREREG grid {0.1,1,10,100,1000,3000,1e4,3e4,1e5,3e5}) -> report best alpha's in-sample
  Pearson r on site-residualized y (site-centered r per exp/23-27 convention).
- Features: ALL variants from features.npz (Gate-0 survivors + base-enum bands + fused + controls).
  Rows with NaN dropped per-variant.
- DEAD if max oracle |r| < 0.15 (both targets) for a variant.
- This is a SCREEN: survivors go to Gate 2 (LOSO + familywise). No claims from Gate 1 alone.

Anti-hacking: every variant in the file is scored, results appended to sieve_table.json
(Dead or alive). The Gate-2 family = every variant that passes Gate 1, regardless of my priors.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]

# ---- load features ----
z = np.load(f"{OUT}/features.npz", allow_pickle=True)
keys = [k for k in z.files if "__" in k]
sites_all = z["sites"]; Ts = z["Ts"]
geom_names = sorted(set(k.split("__")[0] for k in keys))
var_names = sorted(set(k.split("__")[1] for k in keys))
print(f"geometries: {geom_names}")
print(f"variants: {var_names}")

# ---- load targets (exp/27-verbatim cohort definition) ----
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]
keep_idx = []
for i, sid in enumerate(ids_all):
    if sites_cc[i] not in (3, 5, 6) or sid not in phmap.index:
        continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2", "3"):
        continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]):
        continue
    keep_idx.append(i)
keep_idx = np.array(keep_idx)
y_inatt = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep_idx])
y_hyper = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep_idx])
site_t = sites_cc[keep_idx]
print(f"T-cohort n={len(keep_idx)} sites {dict(zip(*np.unique(site_t, return_counts=True)))}")
assert sites_all.shape[0] == len(ids_all)

def site_center_r(y, site, pred):
    # residualize both on site dummies, then Pearson
    uniq = np.unique(site)
    D = np.column_stack([np.ones(len(site))] + [(site == s).astype(float) for s in uniq[1:]])
    def resid(v):
        b, *_ = np.linalg.lstsq(D, v, rcond=None)
        return v - D @ b
    return float(np.corrcoef(resid(y), resid(pred))[0, 1])

def oracle(X, y, site):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xs, ys, ss = X[m], y[m], site[m]
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-12
    Xz = (Xs - mu) / sd
    best = (0.0, None)
    S = Xz.T @ Xz
    k = Xz.shape[1]
    for a in ALPHAS:
        w = np.linalg.solve(S + a * np.eye(k), Xz.T @ ys)
        pred = Xz @ w
        r = site_center_r(ys, ss, pred)
        if abs(r) > abs(best[0]):
            best = (r, a)
    return best[0], best[1], int(m.sum())

results = {}
archive_keys = [k for k in z.files if "__" in k]
for key in archive_keys:
    X = z[key][keep_idx]
    r_i, a_i, n = oracle(X, y_inatt, site_t)
    r_h, a_h, _ = oracle(X, y_hyper, site_t)
    results[key] = {"oracle_r_inatt": r_i, "alpha_inatt": a_i,
                    "oracle_r_hyper": r_h, "alpha_hyper": a_h, "n": n}
    flag = "ALIVE" if max(abs(r_i), abs(r_h)) >= 0.15 else "dead"
    print(f"{key:38s} Inatt {r_i:+.3f} (a={a_i})  Hyper {r_h:+.3f} (a={a_h})  n={n}  {flag}")

with open(f"{OUT}/gate1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved gate1_results.json")
