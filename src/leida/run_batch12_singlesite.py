#!/usr/bin/env python3
"""exp/37 BATCH 12 — single-site (site 5) discovery. Preregistered (SIEVE_TABLE.md)."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 200

zb1 = np.load(f"{OUT}/features.npz"); zb2 = np.load(f"{OUT}/batch2_features.npz"); zb5 = np.load(f"{OUT}/batch5_features.npz")
BLOCKS = {
    "Lnet28": zb1["g050c5K60__Lnet28"], "Lstr7": zb1["g050c5K60__Lstr7"],
    "Mnet7": zb1["g050c5K60__Mnet7"], "Lrow190": zb1["g050c5K60__Lrow190"],
    "W1_g050": zb2["g050c5K60__W1"], "W1_g080": zb2["g080c3nat__W1"], "V2_meta": zb2["g050c5K60__V2"],
    "H1_vab": zb5["H1__vab"], "H3_hvar": zb5["H3__hvar"],
    "K1_signed": zb5["K1__signed"],
}
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, _, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites = np.array(meta_all["sites"])
ages = np.array(meta_all["ages"], dtype=float)
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)
mm = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_maxmotion.tsv"), sep="\t")
mm["ScanDir ID"] = mm["ScanDir ID"].astype(str)
mmmap = dict(zip(mm["ScanDir ID"], mm["maxmotion_mm"]))

keep = []
for i, sid in enumerate(ids_all):
    if sites[i] != 5 or sid not in phmap.index: continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2","3"): continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]): continue
    keep.append(i)
keep = np.array(keep)
print(f"site-5 T-cohort n={len(keep)}")
y_inatt = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep])
y_hyper = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep])
age = ages[keep].copy(); age[np.isnan(age)] = np.nanmean(ages)
sex = sex_all[keep]
mot = np.array([float(mmmap.get(ids_all[i], np.nan)) for i in keep]); mot[np.isnan(mot)] = np.nanmean(mot)
CONF = np.column_stack([age, sex, mot])

def cv_r(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    preds = np.full(len(y), np.nan)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        best_a, best_s = ALPHAS[0], -1
        inner = np.array_split(rng.permutation(tr), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(tr, iv)
                mu = X[tr2].mean(0); sd = X[tr2].std(0) + 1e-12
                Xz = (X[tr2]-mu)/sd
                ww = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@y[tr2])
                pr = ((X[iv]-mu)/sd)@ww
                sc.append(np.corrcoef(pr, y[iv])[0,1] if np.std(pr) > 1e-12 else 0)
            s = np.mean(sc)
            if s > best_s: best_s, best_a = s, a
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-12
        Xz = (X[tr]-mu)/sd
        ww = np.linalg.solve(Xz.T@Xz + best_a*np.eye(Xz.shape[1]), Xz.T@y[tr])
        preds[f] = ((X[f]-mu)/sd)@ww
    return float(np.corrcoef(preds, y)[0,1])

t0 = time.time()
REAL = {}
for tname, y in [("Inatt", y_inatt), ("Hyper", y_hyper)]:
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), CONF]), y, rcond=None)[0]
    yr = y - np.column_stack([np.ones(len(y)), CONF]) @ coef
    REAL[tname] = {}
    for k, Xall in BLOCKS.items():
        X = Xall[keep]
        ok = np.isfinite(X).all(1) & np.isfinite(yr)
        r = cv_r(X[ok], yr[ok], seed=0)
        REAL[tname][k] = {"cv_r": r, "n": int(ok.sum())}
        print(f"[{tname:5s}] {k:12s} CVr={r:+.3f} n={ok.sum()}")

# confound baseline
for tname, y in [("Inatt", y_inatt), ("Hyper", y_hyper)]:
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), CONF]), y, rcond=None)[0]
    yr = y - np.column_stack([np.ones(len(y)), CONF]) @ coef
    r_conf = cv_r(CONF, y, seed=0)
    print(f"[{tname:5s}] conf baseline CVr={r_conf:+.3f}")
    REAL[tname]["_conf"] = r_conf

print(f"\nfamilywise null ({N_PERM}) ...", flush=True)
rng = np.random.RandomState(20260914)
maxstats = []
for p_ in range(N_PERM):
    mx = 0.0
    for tname, y0 in [("Inatt", y_inatt), ("Hyper", y_hyper)]:
        yp = rng.permutation(y0)
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(y0)), CONF]), yp, rcond=None)[0]
        ypr = yp - np.column_stack([np.ones(len(y0)), CONF]) @ coef
        for k, Xall in BLOCKS.items():
            X = Xall[keep]
            ok = np.isfinite(X).all(1)
            r = cv_r(X[ok], ypr[ok], seed=0)
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 50 == 0: print(f"  perm {p_+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)
print(f"null max q95={np.percentile(np.abs(maxstats),95):.3f}")

for tname in REAL:
    for k in BLOCKS:
        r = REAL[tname][k]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / (N_PERM + 1))
        REAL[tname][k]["fw_p"] = fw_p
        verdict = "ALIVE" if (fw_p < 0.05 and abs(r) > 0.20) else "dead"
        REAL[tname][k]["GATE2_VERDICT"] = verdict
        print(f"[{tname:5s}] {k:12s} CVr={r:+.3f} fw_p={fw_p:.4f} -> {verdict}")

with open(f"{OUT}/batch12_results.json", "w") as f:
    json.dump(REAL, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
