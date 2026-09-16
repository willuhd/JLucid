#!/usr/bin/env python3
"""exp/37 BATCH 15 — heterogeneity-first: dispersion, burden, cumulative score."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"

zb1 = np.load(f"{OUT}/features.npz"); zb2 = np.load(f"{OUT}/batch2_features.npz")
zb5 = np.load(f"{OUT}/batch5_features.npz"); zb13 = np.load(f"{OUT}/batch13_amp_features.npz")
BLOCKS = {
    "Lnet28": zb1["g050c5K60__Lnet28"], "Lstr7": zb1["g050c5K60__Lstr7"],
    "Mnet7": zb1["g050c5K60__Mnet7"], "Lrow190": zb1["g050c5K60__Lrow190"],
    "W1_g050": zb2["g050c5K60__W1"], "W1_g080": zb2["g080c3nat__W1"],
    "V2_meta": zb2["g050c5K60__V2"], "H1_vab": zb5["H1__vab"],
    "H3_hvar": zb5["H3__hvar"], "K1_signed": zb5["K1__signed"],
    "NAmean_a080": zb13["a080__NA_mean7"], "Avar_a080": zb13["a080__Avar7"],
    "NAstd_a080": zb13["a080__NA_std7"],
}
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites = np.array(meta_all["sites"])
ages = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
ph = pd.read_csv(f"{BASE}/data/adhd200/adhd200_preprocessed_phenotypics.tsv", sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)
mm = pd.read_csv(f"{BASE}/data/adhd200/adhd200_maxmotion.tsv", sep="\t")
mm["ScanDir ID"] = mm["ScanDir ID"].astype(str)
mmmap = dict(zip(mm["ScanDir ID"], mm["maxmotion_mm"]))
mot = np.array([float(mmmap.get(s, np.nan)) for s in ids_all])
age_i = ages.copy(); age_i[np.isnan(age_i)] = np.nanmean(ages)

t0 = time.time()
# DX cohort: sites 1,3,4,5,6 (>=30 TDC and ADHD present)
keep = np.array([i for i in range(len(ids_all)) if sites[i] in (1,3,4,5,6)])
s = sites[keep]; do = dx[keep]; ag = age_i[keep]; sx = sex_all[keep]; mo = mot[keep]
mo_k = np.nan_to_num(mo, nan=float(np.nanmean(mo)))
hc = do == 0
print(f"n={len(keep)} HC={hc.sum()} ADHD={(do==1).sum()}; per-site TDC:", {int(st): int((hc&(s==st)).sum()) for st in (1,3,4,5,6)})

# ---- (a) normative z per block (per-site TDC age/sex linear model, MAD scale) ----
def norm_z(X):
    n, d = X.shape
    Z = np.full_like(X, np.nan)
    for st in np.unique(s):
        mhc = hc & (s == st)
        mall = s == st
        if mhc.sum() < 30: continue
        # TDC design
        A = np.column_stack([np.ones(mhc.sum()), ag[mhc], sx[mhc]])
        for j in range(d):
            y = X[mhc, j]
            ok = np.isfinite(y)
            if ok.sum() < 30: continue
            coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ coef
            mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            # predict all subjects at this site
            Aall = np.column_stack([np.ones(mall.sum()), ag[mall], sx[mall]])
            zall = (X[mall, j] - Aall @ coef) / mad
            Z[mall, j] = zall
    return np.clip(Z, -3, 3)

Z = {}
for k, Xf in BLOCKS.items():
    X = Xf[keep]
    okr = np.isfinite(X).all(1)
    Xo = X.copy(); Xo[~okr] = np.nan
    Z[k] = norm_z(Xo)
    print(f"z: {k} ({time.time()-t0:.0f}s)", flush=True)

# ---- (a) DISPERSION test: z^2 ~ dx within site ----
res = {}
for k, Zk in Z.items():
    d = Zk.shape[1]
    # site-standardized z^2 stats
    z2 = Zk ** 2
    # within-site: ADHD mean z2 vs TDC mean z2 (Levene-ish), covary FD via residual on FD within group
    obs_list = []
    null_lists = []
    # observed: pooled mean difference of z2 (ADHD - HC), FD-residualized
    allz2 = []
    allfd = []
    alldx = []
    allst = []
    for j in range(d):
        m = np.isfinite(z2[:, j])
        allz2.append(z2[m, j]); allfd.append(mo_k[m]); alldx.append(do[m]); allst.append(s[m])
    allz2 = np.concatenate(allz2); allfd = np.concatenate(allfd); alldx = np.concatenate(alldx); allst = np.concatenate(allst)
    # residualize z2 on fd within site
    resid = allz2.copy()
    for st in np.unique(allst):
        mst = allst == st
        A = np.column_stack([np.ones(mst.sum()), allfd[mst]])
        coef, *_ = np.linalg.lstsq(A, allz2[mst], rcond=None)
        resid[mst] = allz2[mst] - A @ coef
    obs = resid[alldx == 1].mean() - resid[alldx == 0].mean()
    # permutation null: within-site dx shuffle (500)
    rng = np.random.RandomState(20260918)
    nulls = []
    for _ in range(500):
        dxp = alldx.copy()
        for st in np.unique(allst):
            mst = allst == st
            dxp[mst] = rng.permutation(alldx[mst])
        nulls.append(resid[dxp == 1].mean() - resid[dxp == 0].mean())
    nulls = np.array(nulls)
    p_perm = (np.sum(np.abs(nulls) >= abs(obs)) + 1) / 501
    res[k] = {"disp_obs": float(obs), "disp_p": float(p_perm), "n_comp": int(d)}
    print(f"[disp] {k:12s} obs={obs:+.4f} p={p_perm:.4f} (n_comp={d})")

# ---- (b) TAIL BURDEN ----
burden_res = {}
for k, Zk in Z.items():
    # per-subject count of |z|>2 (conservative mid threshold; sweep 1.64-3.1 later for survivors)
    abz = np.abs(Zk)
    cnt = np.nansum(abz > 2.0, axis=1)
    has = np.isfinite(Zk).all(1)
    c = cnt[has]; dxx = do[has]; stt = s[has]; fdd = mo_k[has]
    # NB-GLM approx: compare with within-site permutation, covarying FD
    obs = c[dxx == 1].mean() - c[dxx == 0].mean()
    rng = np.random.RandomState(20260919)
    nulls = []
    for _ in range(500):
        dxp = dxx.copy()
        for st in np.unique(stt):
            mst = stt == st
            dxp[mst] = rng.permutation(dxx[mst])
        nulls.append(c[dxp == 1].mean() - c[dxp == 0].mean())
    nulls = np.array(nulls)
    p_perm = (np.sum(np.abs(nulls) >= abs(obs)) + 1) / 501
    burden_res[k] = {"burden_obs": float(obs), "burden_p": float(p_perm)}
    print(f"[burd] {k:12s} obs={obs:+.4f} p={p_perm:.4f}")

# Holm over dispersion (primary) family
ps = np.array([res[k]["disp_p"] for k in res])
order = np.argsort(ps)
holm = np.empty(len(ps)); m = len(ps)
run = 1
for rank, oi in enumerate(order):
    holm[oi] = min(1.0, (m - rank) * ps[oi])
for oi, k in enumerate([k for k in res]):
    res[k]["holm_p"] = float(holm[oi])
    res[k]["DISP_VERDICT"] = "ALIVE" if holm[oi] < 0.05 and res[k]["disp_obs"] > 0 else "dead"
    print(f"[disp-holm] {k:12s} holm={holm[oi]:.4f} -> {res[k]['DISP_VERDICT']}")

# ---- (c) CUMULATIVE SCORE (polyneuro-style) ----
# z-features: all blocks concatenated -> subject-level matrix
Zmat = []
for k in BLOCKS:
    Zmat.append(Z[k])
Zall = np.concatenate(Zmat, axis=1)
okr = np.isfinite(Zall).all(1)
Zo = Zall[okr]
print(f"\ncumulative-score cohort n={Zo.shape[0]} p={Zo.shape[1]}")
# severity: within-site-standardized Inatt among T-cohort
keepT = []
for i_pos, i in enumerate(keep):
    sid = ids_all[i]
    if sites[i] in (3, 5, 6) and sid in phmap.index:
        if str(phmap.loc[sid, "ADHD Measure"]) in ("2", "3") and not pd.isna(phmap.loc[sid, "Inattentive_n"]):
            keepT.append(i_pos)
keepT = np.array(keepT)
y_inatt = np.array([float(phmap.loc[ids_all[keep[i_pos]], "Inattentive_n"]) for i_pos in keepT])
sT = s[keepT]
# within-site standardize severity
y_z = y_inatt.copy()
for st in np.unique(sT):
    m = sT == st
    y_z[m] = (y_inatt[m] - y_inatt[m].mean()) / (y_inatt[m].std() + 1e-12)
ZT = Zo[keepT]
okT = np.isfinite(y_z)
ZT = ZT[okT]; yz = y_z[okT]; sT2 = sT[okT]
nT = len(yz)
print(f"severity cohort n={nT}")

from sklearn.linear_model import ElasticNet
def enet_pred(Xtr, ytr, Xte, seed=0):
    best = (None, -1)
    for alpha in [0.01, 0.1, 1.0]:
        for l1r in [0.1, 0.5, 0.9]:
            e = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=5000, random_state=seed)
            e.fit(Xtr, ytr)
            # inner r via simple fit (in-sample for alpha pick; final test is held out)
            r = np.corrcoef(e.predict(Xtr), ytr)[0,1]
            if r > best[1]:
                best = (e, r)
    return best[0].predict(Xte)

# split-half discovery + test, 100 random splits
rng = np.random.RandomState(20260920)
rs_test = []
for rep in range(100):
    perm = rng.permutation(nT)
    A = perm[: nT // 2]; B = perm[nT // 2:]
    predB = enet_pred(ZT[A], yz[A], ZT[B], seed=rep)
    rB = float(np.corrcoef(predB, yz[B])[0,1])
    predA = enet_pred(ZT[B], yz[B], ZT[A], seed=rep + 1000)
    rA = float(np.corrcoef(predA, yz[A])[0,1])
    rs_test.append((rA + rB) / 2)
rs_test = np.array(rs_test)
print(f"split-half cumulative score: mean r={rs_test.mean():+.3f} median={np.median(rs_test):+.3f} P(r>0)={np.mean(rs_test>0):.3f}")

# permutation null for the mean-stat (100 perms)
nulls = []
for _ in range(100):
    yp = rng.permutation(yz)
    perm = rng.permutation(nT)
    A = perm[: nT // 2]; B = perm[nT // 2:]
    predB = enet_pred(ZT[A], yp[A], ZT[B], seed=0)
    rB = float(np.corrcoef(predB, yp[B])[0,1]) if np.std(predB) > 0 else 0
    predA = enet_pred(ZT[B], yp[B], ZT[A], seed=1)
    rA = float(np.corrcoef(predA, yp[A])[0,1]) if np.std(predA) > 0 else 0
    nulls.append((rA + rB) / 2)
nulls = np.array(nulls)
p_cum = (np.sum(np.abs(nulls) >= abs(rs_test.mean())) + 1) / 101
print(f"cumulative-score perm p={p_cum:.4f}")

out = {"dispersion": res, "burden": burden_res,
       "cumulative": {"mean_r": float(rs_test.mean()), "median_r": float(np.median(rs_test)),
                       "p_perm": float(p_cum), "p_pos": float(np.mean(rs_test > 0))},
       "n_dx": int(len(keep)), "n_sev": int(nT)}
with open(f"{OUT}/batch15_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
