#!/usr/bin/env python3
"""
exp/37 BATCH 3 GATE 2 — per-site estimator battery (preregistered in SIEVE_TABLE.md).
Estimator: LOSO predictions z-scored WITHIN site, then per-site Pearson r; statistic =
MEAN PER-SITE r. Confound baseline (age/sex/maxmotion) estimated with the SAME estimator.
Null: 200 within-site perms; max-stat over the batch-3 family per target-family.
Cells:
  T-family (Inatt, Hyper): X1 W1_g050, X2 W1_g080, X3 W2_g050, X4 Lstr7_g050 (batch-1 file), X5 C1_tce
  A-family (Idx): X1 W1_g050, X2 W1_g080
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 200

zb2 = np.load(f"{OUT}/batch2_features.npz", allow_pickle=True)
zb1 = np.load(f"{OUT}/features.npz", allow_pickle=True)

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

def loso_preds(X, y, site, fixed_alpha=None):
    """LOSO with inner-CV alpha; returns raw preds + alphas used."""
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
                # INNER EVAL WITH PER-SITE ESTIMATOR: mean per-site r on validation
                p_inner = np.full(len(y), np.nan)
                # fit on m2 train, predict all
                for a in ALPHAS:
                    p_all = np.full(len(y), np.nan)
                    # predict validation fold only
                    pv = rfp(X[m2], ytr2_c[m2], X[va], a)
                    # per-site r within va members' sites
                    rs_sites = np.unique(site[va])
                    rr = []
                    for s2 in rs_sites:
                        mk2 = site[va] == s2
                        if mk2.sum() > 5 and np.std(pv[mk2]) > 1e-12:
                            rr.append(np.corrcoef(pv[mk2], y[va][mk2])[0,1])
                    scores[a].append(np.mean(rr) if rr else 0.0)
            mean_scores = {a: np.nanmean(s) for a, s in scores.items()}
            mode_a = max(mean_scores, key=lambda a: mean_scores[a])
            alphas.append(mode_a)
            preds[te] = rfp(X[tr], ytr_c[tr], X[te], mode_a)
        else:
            alphas.append(fixed_alpha)
            preds[te] = rfp(X[tr], ytr_c[tr], X[te], fixed_alpha)
    return preds, alphas

def persite_stat(preds, y, site):
    rs = []
    for st in np.unique(site):
        mk = site == st
        if mk.sum() > 5 and np.std(preds[mk]) > 1e-12:
            rs.append(float(np.corrcoef(preds[mk], y[mk])[0,1]))
        else:
            rs.append(0.0)
    return float(np.mean(rs)), rs

def r2_persite(preds, y, site):
    # R2 under per-site z-scoring: z preds within site, then 1 - SSE/SST(pooled centered y)
    pz = preds.copy()
    for st in np.unique(site):
        mk = site == st
        pz[mk] = (preds[mk] - preds[mk].mean()) / (preds[mk].std() + 1e-12)
    yz = y.copy()
    for st in np.unique(site):
        mk = site == st
        yz[mk] = y[mk] - y[mk].mean()
    # scale pz to yz sd for R2
    pz = pz * (yz.std() / (pz.std() + 1e-12))
    return 1 - np.sum((yz - pz)**2) / np.sum((yz - yz.mean())**2)

t0 = time.time()
# ---- assemble features per cohort ----
FEAT_T = {
    "X1_W1_g050": zb2["g050c5K60__W1"][keep_T],
    "X2_W1_g080": zb2["g080c3nat__W1"][keep_T],
    "X3_W2_g050": zb2["g050c5K60__W2"][keep_T],
    "X4_Lstr7_g050": zb1["g050c5K60__Lstr7"][keep_T],
    "X5_C1_tce": zb2["C1__tce"][keep_T],
}
FEAT_A = {
    "X1_W1_g050": zb2["g050c5K60__W1"][keep_A],
    "X2_W1_g080": zb2["g080c3nat__W1"][keep_A],
}
confT = get_conf(keep_T); confA = get_conf(keep_A)
sexT = sex_all[keep_T]; sexA = sex_all[keep_A]

yT = {"Inatt": y_inatt_T, "Hyper": y_hyper_T}
yA = {"Idx": y_idx_A}

def battery(FEATURES, ytargets, site, conf, sex, cells):
    REAL = {}
    for tname, y in ytargets.items():
        REAL[tname] = {}
        pc, alc_conf = loso_preds(conf, y, site, fixed_alpha=100.0)
        mean_r_conf, _ = persite_stat(pc, y, site)
        r2_conf = r2_persite(pc, y, site)
        REAL[tname]["conf_baseline"] = {"mean_persite_r": mean_r_conf, "r2": r2_conf}
        for cellname in cells:
            X = FEATURES[cellname]
            msk = np.all(np.isfinite(X), axis=1)
            if msk.sum() < len(y) * 0.9:
                # per-variant NaN handling: drop and note
                pass
            preds, alphas = loso_preds(X[msk], y[msk], site[msk])
            mean_r, per_site = persite_stat(preds, y[msk], site[msk])
            r2f = r2_persite(preds, y[msk], site[msk])
            sex_r = {}
            for sx in (0,1):
                mk = sex[msk] == sx
                if mk.sum() > 20 and np.std(preds[mk]) > 1e-12:
                    sex_r[int(sx)] = float(np.corrcoef(preds[mk], y[msk][mk])[0,1])
            REAL[tname][cellname] = {"mean_persite_r": mean_r, "per_site_r": per_site,
                                      "alphas": alphas, "r2": r2f, "dr2_vs_conf": r2f - r2_conf,
                                      "sex_r": sex_r, "n_valid": int(msk.sum())}
            print(f"[{tname}] {cellname:16s} mean_r={mean_r:+.3f} per_site={ [round(x,3) for x in per_site] } dr2={r2f-r2_conf:+.3f} sex={ {k: round(v,3) for k,v in sex_r.items()} }")
    return REAL

print("=== T-family (per-site estimator) ===")
REAL_T = battery(FEAT_T, yT, site_T, confT, sexT, list(FEAT_T.keys()))
print("\n=== A-family (per-site estimator) ===")
REAL_A = battery(FEAT_A, yA, site_A, confA, sexA, list(FEAT_A.keys()))

# ---- familywise nulls ----
print(f"\nfamilywise nulls ({N_PERM} perms)...", flush=True)
rng = np.random.RandomState(20260830)

def perm_family(REAL, FEATURES, ytargets, site, cells):
    FIXED = {t: {c: REAL[t][c]["alphas"] for c in cells} for t in ytargets}
    maxstats = []
    for p in range(N_PERM):
        yperm = {}
        for tname, y0 in ytargets.items():
            yp = y0.copy()
            for st in np.unique(site):
                mk = site == st
                yp[mk] = rng.permutation(y0[mk])
            yperm[tname] = yp
        mx = 0.0
        for tname in ytargets:
            for c in cells:
                X = FEATURES[c]
                msk = np.all(np.isfinite(X), axis=1)
                yp = yperm[tname][msk]; sm = site[msk]
                preds = np.full(len(yp), np.nan)
                for ai, st in enumerate(np.unique(sm)):
                    te = sm == st; tr = ~te
                    ytr_c = site_demean(yp, sm, tr)
                    preds[te] = rfp(X[msk][tr], ytr_c[tr], X[msk][te], FIXED[tname][c][ai])
                mean_r, _ = persite_stat(preds, yp, sm)
                if abs(mean_r) > abs(mx): mx = mean_r
        maxstats.append(mx)
        if (p+1) % 50 == 0: print(f"  perm {p+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
    return np.array(maxstats)

null_T = perm_family(REAL_T, FEAT_T, yT, site_T, list(FEAT_T.keys()))
null_A = perm_family(REAL_A, FEAT_A, yA, site_A, list(FEAT_A.keys()))

for tname in yT:
    for c in FEAT_T:
        v = REAL_T[tname][c]; r = v["mean_persite_r"]
        fw_p = float((np.sum(np.abs(null_T) >= abs(r)) + 1) / (N_PERM + 1))
        v["fw_p"] = fw_p
        alive = (fw_p < 0.05 and abs(r) > 0.15 and v["dr2_vs_conf"] > 0
                 and len(v["sex_r"]) == 2 and all(np.sign(x) == np.sign(r) for x in v["sex_r"].values())
                 and all(np.sign(x) == np.sign(r) for x in v["per_site_r"]))
        v["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[T|{tname}] {c:16s} mean_r={r:+.3f} fw_p={fw_p:.4f} dr2={v['dr2_vs_conf']:+.3f} -> {v['GATE2_VERDICT']}")
for tname in yA:
    for c in FEAT_A:
        v = REAL_A[tname][c]; r = v["mean_persite_r"]
        fw_p = float((np.sum(np.abs(null_A) >= abs(r)) + 1) / (N_PERM + 1))
        v["fw_p"] = fw_p
        alive = (fw_p < 0.05 and abs(r) > 0.15 and v["dr2_vs_conf"] > 0
                 and len(v["sex_r"]) == 2 and all(np.sign(x) == np.sign(r) for x in v["sex_r"].values())
                 and all(np.sign(x) == np.sign(r) for x in v["per_site_r"]))
        v["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[A|{tname}] {c:16s} mean_r={r:+.3f} fw_p={fw_p:.4f} dr2={v['dr2_vs_conf']:+.3f} -> {v['GATE2_VERDICT']}")

out = {"T_family": REAL_T, "A_family": REAL_A,
       "null_T": {"mean": float(np.mean(np.abs(null_T))), "q95": float(np.percentile(np.abs(null_T),95))},
       "null_A": {"mean": float(np.mean(np.abs(null_A))), "q95": float(np.percentile(np.abs(null_A),95))},
       "n_perm": N_PERM}
with open(f"{OUT}/gate2_batch3_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nDONE {time.time()-t0:.0f}s")
