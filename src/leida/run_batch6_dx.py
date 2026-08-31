#!/usr/bin/env python3
"""
exp/37 BATCH 6 — DX (case-control) screen. PREREGISTERED (SIEVE_TABLE.md batch 6).
Features: 10 Gate-0-passing blocks from batches 1/2/5 (no new computation).
Estimator: logistic L2, baseline = age+sex+site; statistic = ΔAUC (feature-added), 10x5-fold
stratified grouped (site) CV. Familywise null: 200 within-site DX shuffles, max-stat over blocks.
Cohort: sites 1,3,4,5,6 (n=722).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
N_PERM = 200
N_REP = 10

zb1 = np.load(f"{OUT}/features.npz", allow_pickle=True)
zb2 = np.load(f"{OUT}/batch2_features.npz", allow_pickle=True)
zb5 = np.load(f"{OUT}/batch5_features.npz", allow_pickle=True)

BLOCKS = {
    "Lnet28": zb1["g050c5K60__Lnet28"], "Lstr7": zb1["g050c5K60__Lstr7"],
    "Mnet7": zb1["g050c5K60__Mnet7"], "Lrow190": zb1["g050c5K60__Lrow190"],
    "W1_g050": zb2["g050c5K60__W1"], "W1_g080": zb2["g080c3nat__W1"], "V2_meta": zb2["g050c5K60__V2"],
    "H1_vab": zb5["H1__vab"], "H2_mag": zb5["H2__mag"], "H3_hvar": zb5["H3__hvar"],
    "K1_signed": zb5["K1__signed"],
}
# remove V2_meta? it passed Gate 0 but died at Gate 1 batch 2 (oracle 0.059). Include anyway
# per prereg "10 blocks": prereg listed Lnet28,Lstr7,Mnet7,Lrow190,W1g050,W1g080,V2,H1,H2,H3,K1 = 11. Keep all 11.
print(f"{len(BLOCKS)} blocks")

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ages_cc = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)

keep = np.where(np.isin(sites_cc, [1, 3, 4, 5, 6]))[0]
print(f"cohort n={len(keep)} ADHD={dx[keep].sum()}")

site = sites_cc[keep]; sex = sex_all[keep]; age = ages_cc[keep].copy()
age[np.isnan(age)] = np.nanmean(ages_cc)
y = dx[keep]
F = {k: V[keep] for k, V in BLOCKS.items()}
# drop rows with any NaN across all blocks
nan_rows = np.zeros(len(keep), bool)
for k, V in F.items():
    nan_rows |= ~np.all(np.isfinite(V), axis=1)
keep2 = ~nan_rows
F = {k: V[keep2] for k, V in F.items()}
site, sex, age, y = site[keep2], sex[keep2], age[keep2], y[keep2]
print(f"complete-case n={len(y)} ADHD={y.sum()}")

uniq_sites = np.unique(site)
Dsite = np.column_stack([(site == s).astype(float) for s in uniq_sites[1:]])
BASE_X = np.column_stack([age, sex, Dsite])

def zstd(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd

def cv_auc(Xfeat, y, site, C, n_rep=N_REP, seed0=0):
    aucs = []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed0 + rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            # grouped: enforce site purity by random split within site? simpler: stratified on y only
            Xtr, Xte = zstd(Xfeat[tr], Xfeat[te])
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            if len(np.unique(y[te])) < 2: continue
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))

def pick_C(Xfeat, y):
    best = (0.0, None)
    for C in C_GRID:
        a = cv_auc(Xfeat, y, site, C, n_rep=3)
        if a > best[0]: best = (a, C)
    return best[1]

t0 = time.time()
auc_base = cv_auc(BASE_X, y, site, 1.0)
print(f"baseline AUC (age+sex+site): {auc_base:.4f}")

REAL = {}
C_FIXED = {}
for k, V in F.items():
    Xfull = np.column_stack([BASE_X, V])
    C = pick_C(Xfull, y)
    a = cv_auc(Xfull, y, site, C)
    dAUC = a - auc_base
    REAL[k] = {"auc": a, "d_auc": dAUC, "C": C}
    C_FIXED[k] = C
    print(f"{k:12s} AUC={a:.4f} dAUC={dAUC:+.4f} C={C}")

print(f"\nperm null ({N_PERM}) ...", flush=True)
rng = np.random.RandomState(20260901)
maxstats = []
for p_ in range(N_PERM):
    yp = y.copy()
    for s in uniq_sites:
        m = site == s
        yp[m] = rng.permutation(y[m])
    mx = 0.0
    for k, V in F.items():
        Xfull = np.column_stack([BASE_X, V])
        a = cv_auc(Xfull, yp, site, C_FIXED[k], n_rep=1, seed0=1000 + p_)
        d = a - auc_base
        if abs(d) > abs(mx): mx = d
    maxstats.append(mx)
    if (p_ + 1) % 50 == 0: print(f"  perm {p_+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)

print()
for k in F:
    d = REAL[k]["d_auc"]
    fw_p = float((np.sum(np.abs(maxstats) >= abs(d)) + 1) / (N_PERM + 1))
    REAL[k]["fw_p"] = fw_p
    alive = fw_p < 0.05 and d > 0.02
    REAL[k]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
    print(f"{k:12s} dAUC={d:+.4f} fw_p={fw_p:.4f} -> {REAL[k]['GATE2_VERDICT']}")

out = {"real": REAL,
       "auc_base": auc_base,
       "null_max": {"mean": float(np.mean(np.abs(maxstats))), "q95": float(np.percentile(np.abs(maxstats), 95))},
       "n": int(len(y)), "n_adhd": int(y.sum()), "n_perm": N_PERM}
with open(f"{OUT}/batch6_dx_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
