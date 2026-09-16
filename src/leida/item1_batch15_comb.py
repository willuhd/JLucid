#!/usr/bin/env python3
"""RECOVERED (transcribed) builder for batch15_adhd200_comb.npz.

Source: fork-session testimony (goal-session Round 4, 2026-08-30), items 1+3a context.
Original BASE was /Volumes/thinkplus/Code/JLucid2, OUT=exp/37_leida_sieve/gate1_oracle,
sys.path src-controllability, module adhd_controllability.datasets_cc200.
Adapted here ONLY for paths/module location (JLucid2->JLucid, OUT=results/,
src-controllability->src, adhd_controllability->controllability).
LOGIC IDENTICAL to recovered verbatim: 3-block comb (H3_hvar,K1_signed,Lnet28;
W1_g050 loaded, excluded), med mask isin(["2","-999","nan"]) (missing=unmedicated),
HC>=30 per site AND per component, MAD=1.4826*median(|r-median|)+1e-12, clip +-3,
motion NaN->0.2, age NaN->cohort mean, seed 41 (second test continues stream),
2000 pooled perms (k+1)/2001, VR on full-unmed mask.
NOTE: permutation here is POOLED (rng.permutation(do[m])), NOT within-site --
this is the certified p=0.0005 construction; see 00_recovered/NOTE_nulls.md.
Status: transcribed-pending-hash-verification (run this file, compare max|comb - saved|).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
zb1 = np.load(f"{OUT}/features.npz"); zb5 = np.load(f"{OUT}/batch5_features.npz"); zb2 = np.load(f"{OUT}/batch2_features.npz")
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
s = sites[keep]; do = dx[keep]; ag = age_i[keep]; sx = sex_all[keep]; mo = mot[keep]
medk = np.array([str(phm.loc[ids_all[i], "Med Status"]) if ids_all[i] in phm.index and pd.notna(phm.loc[ids_all[i], "Med Status"]) else "nan" for i in keep])
hc = do == 0

BLOCKS = {"H3_hvar": zb5["H3__hvar"], "K1_signed": zb5["K1__signed"],
          "Lnet28": zb1["g050c5K60__Lnet28"], "W1_g050": zb2["g050c5K60__W1"]}
def norm_z(X):
    n, d = X.shape
    Z = np.full_like(X, np.nan)
    for st in np.unique(s):
        mhc = hc & (s == st); mall = s == st
        if mhc.sum() < 30: continue
        A = np.column_stack([np.ones(mhc.sum()), ag[mhc], sx[mhc]])
        for j in range(d):
            y = X[mhc, j]; ok = np.isfinite(y)
            if ok.sum() < 30: continue
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ coef
            mad = 1.4826*np.median(np.abs(r-np.median(r))) + 1e-12
            Aall = np.column_stack([np.ones(mall.sum()), ag[mall], sx[mall]])
            Z[mall, j] = (X[mall, j]-Aall@coef)/mad
    return np.clip(Z, -3, 3)
Zs = {k: norm_z(X[keep]) for k, X in BLOCKS.items()}
mz2 = {k: np.nanmean(Z**2, axis=1) for k, Z in Zs.items()}
comb = np.nanmean(np.column_stack([mz2[k] for k in ["H3_hvar","K1_signed","Lnet28"]]), axis=1)

unmed = (do==0) | ((do==1) & np.isin(medk, ["2","-999","nan"]))
q1, q3 = np.percentile(mo[hc], [25, 75])
mm_mask = unmed & ((do==0) | ((do==1) & (mo>=q1) & (mo<=q3)))
m = mm_mask & np.isfinite(comb)
obs = comb[m&(do==1)].mean() - comb[m&(do==0)].mean()
rng = np.random.RandomState(41)
nulls = []
for _ in range(2000):
    yp = rng.permutation(do[m])
    nulls.append(comb[m][yp==1].mean() - comb[m][yp==0].mean())
nulls = np.array(nulls)
p = (np.sum(np.abs(nulls) >= abs(obs))+1)/2001
p1 = (np.sum(nulls >= obs)+1)/2001
print(f"ADHD-200 COMBINED disp (unmed + motion-IQR-matched): obs={obs:+.4f} two-sided p={p:.4f} one-sided p={p1:.4f}")
print(f"  nADHD={int(np.sum(m&(do==1)))}, nHC={int(np.sum(m&(do==0)))}")
m2 = unmed & np.isfinite(comb)
obs2 = comb[m2&(do==1)].mean() - comb[m2&(do==0)].mean()
nulls2 = []
for _ in range(2000):
    yp = rng.permutation(do[m2])
    nulls2.append(comb[m2][yp==1].mean() - comb[m2][yp==0].mean())
nulls2 = np.array(nulls2)
p2 = (np.sum(np.abs(nulls2) >= abs(obs2))+1)/2001
print(f"ADHD-200 COMBINED disp (unmed, all motion): obs={obs2:+.4f} two-sided p={p2:.4f} (nADHD={int(np.sum(m2&(do==1)))})")
va = np.var(comb[m2&(do==1)]); vc = np.var(comb[m2&(do==0)])
print(f"  var ratio ADHD/HC = {va/vc:.2f}")
print("VERIFY: max|comb - results/batch15_adhd200_comb.npz[comb]| =",
      float(np.nanmax(np.abs(comb - np.load(f"{OUT}/batch15_adhd200_comb.npz")["comb"]))))
