#!/usr/bin/env python3
"""
exp/37 BATCH 5 GATE 2 — honest LOSO battery. Survivors of Gate 1 (batch 5):
  T-family: H1__vab, H3__hvar, K1__signed x {Inatt, Hyper}  (6 cells)
  A-family: H1__vab, K1__signed x {Idx}                      (2 cells)
Pooled-r estimator (original prereg convention). Inner-CV alpha fixed at real-data selection.
200 within-site perms; max-stat familywise per family. Criteria unchanged:
fw-p<0.05 AND |r_sc|>0.15 AND ΔR²>0 AND both-sex/all-site same-sign.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 200

z = np.load(f"{OUT}/batch5_features.npz", allow_pickle=True)
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, _, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ages_cc = np.array(meta_all["ages"], dtype=float)
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive", "ADHD Index"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure", "ADHD Index_n"]]
gmap = dict(zip(ph["ScanDir ID"], ph["Gender"].fillna(0).astype(int)))
sex_all = np.array([gmap.get(s, 0) for s in ids_all], dtype=float)
mm = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_maxmotion.tsv"), sep="\t")
mm["ScanDir ID"] = mm["ScanDir ID"].astype(str)
mmmap = dict(zip(mm["ScanDir ID"], mm["maxmotion_mm"]))

keep_T = []
for i, sid in enumerate(ids_all):
    if sites_cc[i] not in (3,5,6) or sid not in phmap.index: continue
    if str(phmap.loc[sid, "ADHD Measure"]) not in ("2","3"): continue
    if pd.isna(phmap.loc[sid, "Inattentive_n"]): continue
    keep_T.append(i)
keep_T = np.array(keep_T)
keep_A = []
for i, sid in enumerate(ids_all):
    if sid not in phmap.index or pd.isna(phmap.loc[sid, "ADHD Index_n"]): continue
    keep_A.append(i)
keep_A = np.array(keep_A)
site_A = sites_cc[keep_A]
uniq, cnts = np.unique(site_A, return_counts=True)
ks = [int(s) for s, c in zip(uniq, cnts) if c >= 10]
mA = np.isin(site_A, ks)
keep_A = keep_A[mA]

def get_y(keep, col):
    return np.array([float(phmap.loc[ids_all[i], col]) for i in keep])
def get_conf(keep):
    age = ages_cc[keep].copy()
    age[np.isnan(age)] = np.nanmean(ages_cc)
    sex = sex_all[keep]
    mmv = np.array([float(mmmap.get(ids_all[i], np.nan)) for i in keep])
    mmv[np.isnan(mmv)] = np.nanmean(mmv)
    return np.column_stack([age, sex, mmv])
def site_demean(v, s, tr):
    out = v.copy()
    for st in np.unique(s[tr]):
        mk = tr & (s == st)
        out[mk] = v[mk] - v[mk].mean()
    return out
def rfp(Xtr, ytr, Xte, a):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-12
    Xz = (Xtr-mu)/sd
    w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@ytr)
    return ((Xte-mu)/sd)@w
def loso_cv(X, y, site, fixed_alpha=None):
    preds = np.full(len(y), np.nan)
    alphas = []
    for st in np.unique(site):
        te = site == st; tr = ~te
        ytr_c = site_demean(y, site, tr)
        if fixed_alpha is None:
            rs = np.random.RandomState(42)
            scores = {a: [] for a in ALPHAS}
            tr_idx = np.where(tr)[0]
            folds = np.array_split(rs.permutation(tr_idx), 3)
            for f in range(3):
                va = folds[f]
                tr2 = np.setdiff1d(tr_idx, va)
                m2 = np.zeros(len(y), bool); m2[tr2] = True
                ytr2_c = site_demean(y, site, m2)
                for a in ALPHAS:
                    p = rfp(X[m2], ytr2_c[m2], X[va], a)
                    yc = y[va] - y[va].mean()
                    scores[a].append(np.corrcoef(p - p.mean(), yc)[0, 1] if np.std(p) > 1e-12 else 0.0)
            mean_scores = {a: np.nanmean(s) for a, s in scores.items()}
            mode_a = max(mean_scores, key=lambda a: mean_scores[a])
            alphas.append(mode_a)
            preds[te] = rfp(X[tr], ytr_c[tr], X[te], mode_a)
        else:
            alphas.append(fixed_alpha)
            preds[te] = rfp(X[tr], ytr_c[tr], X[te], fixed_alpha)
    preds_c = preds.copy(); y_c = y.copy()
    for st in np.unique(site):
        m = site == st
        preds_c[m] = preds[m] - preds[m].mean()
        y_c[m] = y[m] - y[m].mean()
    r_sc = float(np.corrcoef(preds_c, y_c)[0, 1])
    per_site = {}
    for st in np.unique(site):
        m = site == st
        per_site[int(st)] = float(np.corrcoef(preds_c[m], y_c[m])[0, 1]) if m.sum() > 5 else np.nan
    return preds_c, r_sc, per_site, alphas
def r2_oos(pred, y):
    return 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)

t0 = time.time()
CELLS_T = [("H1__vab","Inatt"),("H1__vab","Hyper"),("H3__hvar","Inatt"),("H3__hvar","Hyper"),
           ("K1__signed","Inatt"),("K1__signed","Hyper")]
CELLS_A = [("H1__vab","Idx"),("K1__signed","Idx")]

site_T = sites_cc[keep_T]
yT = {"Inatt": get_y(keep_T, "Inattentive_n"), "Hyper": get_y(keep_T, "Hyper/Impulsive_n")}
confT = get_conf(keep_T); sexT = sex_all[keep_T]
FT = {k: z[k][keep_T] for k, _ in CELLS_T}
nanT = np.zeros(len(keep_T), bool)
for k in set(k for k, _ in CELLS_T):
    nanT |= ~np.all(np.isfinite(FT[k]), axis=1)
mT = ~nanT
FT = {k: V[mT] for k, V in FT.items()}
for t in yT: yT[t] = yT[t][mT]
siteT2, sexT2, confT2 = site_T[mT], sexT[mT], confT[mT]
print(f"T n={mT.sum()}")

REAL_T = {}
for tname in ("Inatt", "Hyper"):
    pc, r_conf, _, _ = loso_cv(confT2, yT[tname], siteT2, fixed_alpha=100.0)
    REAL_T[tname] = {"conf_baseline_r": r_conf, "conf_r2": r2_oos(pc, yT[tname])}
    for k, t in CELLS_T:
        if t != tname: continue
        preds, r_sc, per_site, alphas = loso_cv(FT[k], yT[tname], siteT2)
        r2f = r2_oos(preds, yT[tname])
        sex_r = {}
        for sx in (0, 1):
            m = sexT2 == sx
            if m.sum() > 20:
                sex_r[int(sx)] = float(np.corrcoef(preds[m], yT[tname][m])[0, 1])
        REAL_T[tname][k] = {"r_sc": r_sc, "per_site_r": per_site, "alphas": alphas, "r2": r2f,
                            "dr2_vs_conf": r2f - r2_oos(pc, yT[tname]), "sex_r": sex_r}
        print(f"[T|{tname}] {k:12s} r_sc={r_sc:+.3f} per_site={ {kk: round(vv,3) for kk,vv in per_site.items()} } dr2={r2f - r2_oos(pc, yT[tname]):+.3f} sex={ {k2: round(v2,3) for k2,v2 in sex_r.items()} }")

site_A2 = sites_cc[keep_A]
yA = get_y(keep_A, "ADHD Index_n")
confA = get_conf(keep_A); sexA = sex_all[keep_A]
FA = {k: z[k][keep_A] for k, _ in CELLS_A}
nanA = np.zeros(len(keep_A), bool)
for k in set(k for k, _ in CELLS_A):
    nanA |= ~np.all(np.isfinite(FA[k]), axis=1)
mA2 = ~nanA
FA = {k: V[mA2] for k, V in FA.items()}
yA = yA[mA2]; siteA3, sexA2, confA2 = site_A2[mA2], sexA[mA2], confA[mA2]
print(f"A n={mA2.sum()}")

REAL_A = {}
pcA, r_confA, _, _ = loso_cv(confA2, yA, siteA3, fixed_alpha=100.0)
REAL_A["conf_baseline_r"] = r_confA; REAL_A["conf_r2"] = r2_oos(pcA, yA)
for k, t in CELLS_A:
    preds, r_sc, per_site, alphas = loso_cv(FA[k], yA, siteA3)
    r2f = r2_oos(preds, yA)
    sex_r = {}
    for sx in (0, 1):
        m = sexA2 == sx
        if m.sum() > 20:
            sex_r[int(sx)] = float(np.corrcoef(preds[m], yA[m])[0, 1])
    REAL_A[k] = {"r_sc": r_sc, "per_site_r": per_site, "alphas": alphas, "r2": r2f,
                 "dr2_vs_conf": r2f - r2_oos(pcA, yA), "sex_r": sex_r}
    print(f"[A|Idx] {k:12s} r_sc={r_sc:+.3f} per_site={ {kk: round(vv,3) for kk,vv in per_site.items()} } dr2={r2f - r2_oos(pcA, yA):+.3f} sex={ {k2: round(v2,3) for k2,v2 in sex_r.items()} }")

print(f"\nfamilywise nulls ({N_PERM} perms)...", flush=True)
rng = np.random.RandomState(20260831)

def run_perm_family(cells, FEATURES, ytargets, site, REALMAP):
    FIXED = {}
    for k, t in cells:
        if t == "Idx": FIXED[(k,t)] = REAL_A[k]["alphas"]
        else: FIXED[(k,t)] = REALMAP[t][k]["alphas"]
    maxstats = []
    for p in range(N_PERM):
        yperm = {}
        for t in set(t for _, t in cells):
            y0 = ytargets[t]
            yp = y0.copy()
            for st in np.unique(site):
                m = site == st
                yp[m] = rng.permutation(y0[m])
            yperm[t] = yp
        mx = 0.0
        for k, t in cells:
            X = FEATURES[k]
            yp = yperm[t]
            alphas = FIXED[(k,t)]
            preds = np.full(len(yp), np.nan)
            for ai, st in enumerate(np.unique(site)):
                te = site == st; tr = ~te
                ytr_c = site_demean(yp, site, tr)
                preds[te] = rfp(X[tr], ytr_c[tr], X[te], alphas[ai])
            preds_c = preds.copy()
            for st in np.unique(site):
                m = site == st
                preds_c[m] = preds[m] - preds[m].mean()
            r = float(np.corrcoef(preds_c, yp)[0, 1]) if np.std(preds_c) > 1e-12 else 0.0
            if abs(r) > abs(mx): mx = r
        maxstats.append(mx)
        if (p+1) % 50 == 0: print(f"  perm {p+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
    return np.array(maxstats)

null_T = run_perm_family(CELLS_T, FT, yT, siteT2, REAL_T)
null_A = run_perm_family(CELLS_A, FA, {"Idx": yA}, siteA3, REAL_A)

for tname in ("Inatt", "Hyper"):
    for k, t in CELLS_T:
        if t != tname: continue
        r = REAL_T[tname][k]["r_sc"]
        fw_p = float((np.sum(np.abs(null_T) >= abs(r)) + 1) / (N_PERM + 1))
        REAL_T[tname][k]["fw_p"] = fw_p
        v = REAL_T[tname][k]
        alive = (fw_p < 0.05 and abs(r) > 0.15 and v["dr2_vs_conf"] > 0
                 and len(v["sex_r"]) == 2 and all(np.sign(x) == np.sign(r) for x in v["sex_r"].values())
                 and all(np.sign(x) == np.sign(r) for x in v["per_site_r"].values() if not np.isnan(x)))
        REAL_T[tname][k]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[T|{tname}] {k:12s} r_sc={r:+.3f} fw_p={fw_p:.4f} dr2={v['dr2_vs_conf']:+.3f} -> {REAL_T[tname][k]['GATE2_VERDICT']}")
for k, t in CELLS_A:
    r = REAL_A[k]["r_sc"]
    fw_p = float((np.sum(np.abs(null_A) >= abs(r)) + 1) / (N_PERM + 1))
    REAL_A[k]["fw_p"] = fw_p
    v = REAL_A[k]
    alive = (fw_p < 0.05 and abs(r) > 0.15 and v["dr2_vs_conf"] > 0
             and len(v["sex_r"]) == 2 and all(np.sign(x) == np.sign(r) for x in v["sex_r"].values())
             and all(np.sign(x) == np.sign(r) for x in v["per_site_r"].values() if not np.isnan(x)))
    REAL_A[k]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
    print(f"[A|Idx] {k:12s} r_sc={r:+.3f} fw_p={fw_p:.4f} dr2={v['dr2_vs_conf']:+.3f} -> {REAL_A[k]['GATE2_VERDICT']}")

out = {"T_family": REAL_T, "A_family": REAL_A,
       "null_T_max": {"mean": float(np.mean(np.abs(null_T))), "q95": float(np.percentile(np.abs(null_T), 95))},
       "null_A_max": {"mean": float(np.mean(np.abs(null_A))), "q95": float(np.percentile(np.abs(null_A), 95))},
       "n_perm": N_PERM}
with open(f"{OUT}/gate2_batch5_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nDONE {time.time()-t0:.0f}s")
