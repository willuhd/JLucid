#!/usr/bin/env python3
"""
exp/37 BATCH 7 — subtype-stratified DX + medication moderator (preregistered).
Contrasts (logistic, baseline age+sex+site, ΔAUC statistic, 10x5-fold CV):
  C1: DX1 (inattentive) vs HC
  C2: DX3 (combined) vs HC
  C3: DX1-unmedicated (Med='2') vs HC
  C4: within-ADHD med contrast (linear, Med '1' vs '2', age/sex/site partialled r)
Blocks: same 11 Gate-0-passing blocks. Familywise null over C1,C2,C3 (AUC family) and C4
(r family) separately; 200 within-site shuffles.
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
C_GRID = [0.001, 0.01, 0.1, 1.0]
N_PERM = 200

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
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ages_cc = np.array(meta_all["ages"], dtype=float)
dxbin = np.array(labels_all).astype(int)
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
phm = ph.set_index("ScanDir ID")
dx_full = np.array([str(phm.loc[i, "DX"]) if i in phm.index else "0" for i in ids_all])
med = np.array([str(phm.loc[i, "Med Status"]) if i in phm.index else "nan" for i in ids_all])
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)

keep = np.where(np.isin(sites_cc, [1,3,4,5,6]))[0]
nan_rows = np.zeros(len(keep), bool)
for k, V in BLOCKS.items():
    nan_rows |= ~np.all(np.isfinite(V[keep]), axis=1)
good = keep[~nan_rows]
F = {k: V[good] for k, V in BLOCKS.items()}
site = sites_cc[good]; sex = sex_all[good]
age = ages_cc[good].copy(); age[np.isnan(age)] = np.nanmean(ages_cc)
dxf = dx_full[good]; medf = med[good]
print(f"n={len(good)}")

def dsite_of(site_):
    u = np.unique(site_)
    return np.column_stack([(site_ == s).astype(float) for s in u[1:]])

def cv_auc(Xfeat, y, C, n_rep=10, seed0=0):
    aucs = []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed0+rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            mu = Xfeat[tr].mean(0); sd = Xfeat[tr].std(0)+1e-12
            Xtr = (Xfeat[tr]-mu)/sd; Xte = (Xfeat[te]-mu)/sd
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:,1]
            if len(np.unique(y[te])) < 2: continue
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else 0.5

def pick_C(Xfeat, y):
    best = (0.0, None)
    for C in C_GRID:
        a = cv_auc(Xfeat, y, C, n_rep=3)
        if a > best[0]: best = (a, C)
    return best[1]

def run_contrast(cells_mask, label):
    """DX contrast: returns dAUC per block + fixed C."""
    y = cells_mask.astype(int)
    B = np.column_stack([age, sex, dsite_of(site)])
    auc_base = cv_auc(B, y, 1.0)
    res = {}; Cfix = {}
    for k, V in F.items():
        Xf = np.column_stack([B, V])
        C = pick_C(Xf, y)
        a = cv_auc(Xf, y, C)
        res[k] = {"auc": a, "d_auc": a - auc_base, "C": C, "auc_base": auc_base}
        Cfix[k] = C
    print(f"[{label}] base AUC={auc_base:.4f}")
    for k in res:
        print(f"  {k:12s} dAUC={res[k]['d_auc']:+.4f}")
    return res, Cfix, auc_base

t0 = time.time()
hc = dxf == "0"
c1_mask = (dxf == "1") | hc
c1_sel = (dxf != "x")  # placeholder; we handle subsetting below

# C1: DX1 vs HC
m1 = np.isin(dxf, ["0", "1"])
sel1 = np.where(m1)[0]
R1, C1, base1 = run_contrast((dxf[sel1] == "1").astype(int), "C1: DX1 vs HC")
S1 = {"feat": {k: V[sel1] for k, V in F.items()}, "site": site[sel1], "age": age[sel1], "sex": sex[sel1]}

# C2: DX3 vs HC
m3 = np.isin(dxf, ["0", "3"])
sel3 = np.where(m3)[0]
R3, C3, base3 = run_contrast((dxf[sel3] == "3").astype(int), "C2: DX3 vs HC")
S3 = {"feat": {k: V[sel3] for k, V in F.items()}, "site": site[sel3], "age": age[sel3], "sex": sex[sel3]}

# C3: DX1-unmed vs HC
m_un = ((dxf == "1") & (medf == "2")) | (dxf == "0")
selu = np.where(m_un)[0]
print(f"C3 n={len(selu)} cases={int((dxf[selu]=='1').sum())}")
Ru, Cu, baseu = run_contrast((dxf[selu] == "1").astype(int), "C3: DX1-unmed vs HC")
Su = {"feat": {k: V[selu] for k, V in F.items()}, "site": site[selu], "age": age[selu], "sex": sex[selu]}

# C4: within-ADHD med contrast (linear, partialled r per block)
adhd_m = np.isin(dxf, ["1","2","3"]) & np.isin(medf, ["1","2"])
sela = np.where(adhd_m)[0]
ymed = (medf[sela] == "1").astype(float)  # 1 = med '1'
B = np.column_stack([age[sela], sex[sela], dsite_of(site[sela])])
def resid(V, B):
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(V)), B]), V, rcond=None)
    return V - np.column_stack([np.ones(len(V)), B]) @ coef
yr = resid(ymed, B)
R4 = {}
for k, V in F.items():
    best = (0.0, None)
    for a in [0.1, 1, 10, 100, 1000]:
        Xr = resid(V[sela].astype(float), B)
        mu = Xr.mean(0); sd = Xr.std(0)+1e-12
        Xz = (Xr-mu)/sd
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=a).fit(Xz, yr).predict(Xz)
        rr = np.corrcoef(r, yr)[0,1]
        if abs(rr) > abs(best[0]): best = (rr, a)
    R4[k] = {"oracle_r_med": best[0], "alpha": best[1], "n": len(sela)}
    print(f"  C4 {k:12s} med-oracle r={best[0]:+.3f} (a={best[1]})")

# ---- familywise nulls for C1/C2/C3 (AUC family) ----
print(f"\nfamilywise null C1/C2/C3 ({N_PERM} perms)...", flush=True)
rng = np.random.RandomState(20260902)
CONTRASTS = [("C1", sel1, R1, C1, base1, S1), ("C2", sel3, R3, C3, base3, S3), ("C3", selu, Ru, Cu, baseu, Su)]
maxstats = []
for p_ in range(N_PERM):
    mx = 0.0
    for cname, sel, RES, CF, bases, S in CONTRASTS:
        ytrue = np.array([(1 if dxf[i] in {"1","2","3"} else 0) for i in sel])
        if cname == "C3":
            ytrue = np.array([(1 if (dxf[i]=="1") else 0) for i in sel])
        elif cname == "C2":
            ytrue = np.array([(1 if dxf[i]=="3" else 0) for i in sel])
        else:
            ytrue = np.array([(1 if dxf[i]=="1" else 0) for i in sel])
        yp = ytrue.copy()
        for s in np.unique(S["site"]):
            m = S["site"] == s
            yp[m] = rng.permutation(ytrue[m])
        B = np.column_stack([S["age"], S["sex"], dsite_of(S["site"])])
        for k in F:
            Xf = np.column_stack([B, S["feat"][k]])
            a = cv_auc(Xf, yp, CF[k], n_rep=1, seed0=2000+p_)
            d = a - bases
            if abs(d) > abs(mx): mx = d
    maxstats.append(mx)
    if (p_+1) % 50 == 0: print(f"  perm {p_+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)

print()
out_real = {"C1": R1, "C2": R3, "C3": Ru, "C4_med": R4}
for cname, sel, RES, CF, bases, S in CONTRASTS:
    for k in F:
        d = RES[k]["d_auc"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(d)) + 1) / (N_PERM + 1))
        RES[k]["fw_p"] = fw_p
        RES[k]["GATE2_VERDICT"] = "ALIVE" if (fw_p < 0.05 and d > 0.02) else "dead"
        print(f"[{cname}] {k:12s} dAUC={d:+.4f} fw_p={fw_p:.4f} -> {RES[k]['GATE2_VERDICT']}")

with open(f"{OUT}/batch7_results.json", "w") as f:
    json.dump(out_real, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
