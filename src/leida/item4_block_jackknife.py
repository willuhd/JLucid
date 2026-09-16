#!/usr/bin/env python3
"""Block-jackknife of the certified 3-block index (fills the hole both sides never ran).

Recipe: recovered item-1 norm_z (HC>=30, MAD, clip3, motion-NaN->0.2, age-NaN->mean).
Families: full [H3,K1,Lnet] + leave-one-out x3. Masks: full-unmed (192/401).
Null: within-site label shuffle, 2000 perms, seed 410+k (stdout only; no saves).
Caveat: frozen-D null (certified construction); refit would shift all p up ~eref R11.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
zb1 = np.load(f"{OUT}/features.npz"); zb5 = np.load(f"{OUT}/batch5_features.npz")
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites = np.array(meta_all["sites"]); ages = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
ph = pd.read_csv(f"{BASE}/data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
phm = ph.set_index("ScanDir ID")
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)
mm = pd.read_csv(f"{BASE}/data/adhd200/adhd200_maxmotion.tsv", sep="\t")
mm["ScanDir ID"] = mm["ScanDir ID"].astype(str)
mmmap = dict(zip(mm["ScanDir ID"], mm["maxmotion_mm"]))
mot = np.nan_to_num(np.array([float(mmmap.get(s, np.nan)) for s in ids_all]), nan=0.2)
age_i = ages.copy(); age_i[np.isnan(age_i)] = np.nanmean(ages)
keep = np.array([i for i in range(len(ids_all)) if sites[i] in (1,3,4,5,6)])
s = sites[keep]; do = dx[keep]; ag = age_i[keep]; sx = sex_all[keep]
medk = np.array([str(phm.loc[ids_all[i], "Med Status"]) if ids_all[i] in phm.index and pd.notna(phm.loc[ids_all[i], "Med Status"]) else "nan" for i in keep])
hc = do == 0
BLOCKS = {"H3": zb5["H3__hvar"], "K1": zb5["K1__signed"], "Lnet": zb1["g050c5K60__Lnet28"]}
def norm_z(X):
    Z = np.full_like(X, np.nan, dtype=float)
    for st in np.unique(s):
        mhc = hc & (s == st); mall = s == st
        if mhc.sum() < 30: continue
        A = np.column_stack([np.ones(mhc.sum()), ag[mhc], sx[mhc]])
        for j in range(X.shape[1]):
            y = X[mhc, j]; ok = np.isfinite(y)
            if ok.sum() < 30: continue
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ coef
            mad = 1.4826*np.median(np.abs(r-np.median(r))) + 1e-12
            Aall = np.column_stack([np.ones(mall.sum()), ag[mall], sx[mall]])
            Z[mall, j] = (X[mall, j]-Aall@coef)/mad
    return np.clip(Z, -3, 3)
mz2 = {k: np.nanmean(norm_z(X[keep])**2, axis=1) for k, X in BLOCKS.items()}
unmed = (do==0) | ((do==1) & np.isin(medk, ["2","-999","nan"]))
m = unmed
rng = np.random.RandomState(410)
fams = {"full[H3+K1+Lnet]": ["H3","K1","Lnet"], "drop-H3": ["K1","Lnet"],
        "drop-K1": ["H3","Lnet"], "drop-Lnet": ["H3","K1"]}
for name, ks in fams.items():
    D = np.nanmean(np.column_stack([mz2[k] for k in ks]), axis=1)
    obs = np.nanmean(D[m&(do==1)]) - np.nanmean(D[m&(do==0)])
    nls = []
    for _ in range(2000):
        dp = do.copy()
        for st in np.unique(s):
            mm2 = s == st; dp[mm2] = rng.permutation(do[mm2])
        nls.append(np.nanmean(D[(dp==1)&m]) - np.nanmean(D[(dp==0)&m]))
    nls = np.array(nls)
    print(f"{name:20s} obs={obs:+.4f} within-site p={(np.sum(np.abs(nls)>=abs(obs))+1)/2001:.4f}")
