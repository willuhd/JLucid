#!/usr/bin/env python3
"""exp/37 BATCH 13 — amplitude-axis screen (preregistered). Gate 1 (oracle) then Gate 2 (LOSO CV)."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 200

z = np.load(f"{OUT}/batch13_amp_features.npz")
BLOCKS = {
    "a050__NA_mean7": z["a050__NA_mean7"], "a050__NA_std7": z["a050__NA_std7"], "a050__Avar7": z["a050__Avar7"],
    "a080__NA_mean7": z["a080__NA_mean7"], "a080__NA_std7": z["a080__NA_std7"],
    "a080__AEC21": z["a080__AEC21"], "a080__Avar7": z["a080__Avar7"],
}
sites = z["sites"]
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
ages = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
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

# T-cohort (sites 3,5,6 with T-scores)
keep_T, keep_DX = [], []
for i, sid in enumerate(ids_all):
    if sid not in phmap.index: continue
    meas = str(phmap.loc[sid, "ADHD Measure"])
    if sites[i] in (3, 5, 6) and meas in ("2", "3") and not pd.isna(phmap.loc[sid, "Inattentive_n"]):
        keep_T.append(i)
    if sites[i] in (1, 3, 4, 5, 6) and str(dx[i]) in ("0", "1") and i < len(dx):
        keep_DX.append(i)
keep_T = np.array(keep_T); keep_DX = np.array(keep_DX)
age_imputed = ages.copy(); age_imputed[np.isnan(age_imputed)] = np.nanmean(ages)
mot = np.array([float(mmmap.get(s, np.nan)) for s in ids_all]); mot[np.isnan(mot)] = np.nanmean(mot)

def site_demean(v, s, tr):
    out = v.copy()
    for st in np.unique(s[tr]):
        m = tr & (s == st)
        out[m] = v[m] - v[m].mean()
    return out
def rfp(Xtr, ytr, Xte, a):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-12
    Xz = (Xtr-mu)/sd
    w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@ytr)
    return ((Xte-mu)/sd)@w
def loso_r(X, y, s):
    preds = np.full(len(y), np.nan)
    for st in np.unique(s):
        te = s == st; tr = ~te
        mask_tr = np.zeros(len(y), bool); mask_tr[tr] = True
        ytr_c = site_demean(y, s, mask_tr)
        best_a, best_s = ALPHAS[0], -1
        rng = np.random.RandomState(0)
        inner = np.array_split(rng.permutation(np.where(tr)[0]), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(np.where(tr)[0], iv)
                m2 = np.zeros(len(y), bool); m2[tr2] = True
                ytr2_c = site_demean(y, s, m2)
                p = rfp(X[tr2], ytr2_c[tr2], X[iv], a)
                sc.append(np.corrcoef(p, y[iv])[0,1] if np.std(p) > 1e-12 else 0)
            sv = np.mean(sc)
            if sv > best_s: best_s, best_a = sv, a
        preds[te] = rfp(X[tr], ytr_c[tr], X[te], best_a)
    preds_c = preds.copy(); y_c = y.copy()
    for st in np.unique(s):
        m = s == st
        preds_c[m] -= preds_c[m].mean(); y_c[m] -= y_c[m].mean()
    return float(np.corrcoef(preds_c, y_c)[0,1])

t0 = time.time()
TARGETS = {}
yT_inatt = np.array([float(phmap.loc[ids_all[i], "Inattentive_n"]) for i in keep_T])
yT_hyper = np.array([float(phmap.loc[ids_all[i], "Hyper/Impulsive_n"]) for i in keep_T])
confT = np.column_stack([age_imputed[keep_T], sex_all[keep_T], mot[keep_T]])
TARGETS["Inatt"] = (keep_T, yT_inatt, confT, sites[keep_T])
TARGETS["Hyper"] = (keep_T, yT_hyper, confT, sites[keep_T])
confDX = np.column_stack([age_imputed[keep_DX], sex_all[keep_DX], mot[keep_DX]])
TARGETS["DX"] = (keep_DX, dx[keep_DX].astype(float), confDX, sites[keep_DX])

REAL = {}
for tname, (keep, y0, CONF, s) in TARGETS.items():
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y0)), CONF]), y0, rcond=None)[0]
    yr = y0 - np.column_stack([np.ones(len(y0)), CONF]) @ coef
    REAL[tname] = {}
    for k, Xall in BLOCKS.items():
        X = Xall[keep]
        ok = np.isfinite(X).all(1) & np.isfinite(yr)
        if ok.sum() < 50: continue
        r = loso_r(X[ok], yr[ok], s[ok])
        REAL[tname][k] = {"cv_r": r, "n": int(ok.sum())}
        print(f"[{tname:5s}] {k:16s} LOSO CVr={r:+.3f} n={ok.sum()}")
    # confound baseline
    r_conf = loso_r(CONF[ok], yr[ok], s[ok])
    REAL[tname]["_conf"] = r_conf
    print(f"[{tname:5s}] conf baseline      CVr={r_conf:+.3f}")

print(f"\nfamilywise null ({N_PERM}) ...", flush=True)
rng = np.random.RandomState(20260915)
maxstats = []
for p_ in range(N_PERM):
    mx = 0.0
    for tname, (keep, y0, CONF, s) in TARGETS.items():
        yp0 = rng.permutation(y0)
        for k, Xall in BLOCKS.items():
            X = Xall[keep]
            ok = np.isfinite(X).all(1)
            if ok.sum() < 50: continue
            r = loso_r(X[ok], yp0[ok], s[ok])
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 50 == 0: print(f"  perm {p_+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)
print(f"null max q95={np.percentile(np.abs(maxstats),95):.3f}")

for tname in REAL:
    for k in BLOCKS:
        if k not in REAL[tname]: continue
        r = REAL[tname][k]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / (N_PERM + 1))
        conf_r = REAL[tname].get("_conf", 0.0)
        verdict = "ALIVE" if (fw_p < 0.05 and abs(r) >= 0.15 and abs(r) > abs(conf_r)) else "dead"
        REAL[tname][k].update({"fw_p": fw_p, "conf_r": conf_r, "GATE2_VERDICT": verdict})
        print(f"[{tname:5s}] {k:16s} CVr={r:+.3f} fw_p={fw_p:.4f} conf={conf_r:+.3f} -> {verdict}")

with open(f"{OUT}/batch13_results.json", "w") as f:
    json.dump(REAL, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
