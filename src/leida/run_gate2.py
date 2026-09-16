#!/usr/bin/env python3
"""
exp/37 GATE 2 — honest battery per PREREG (frozen before running).

Survivors that reached Gate 2 (all 28 ALIVE variants from gate1_results.json; the
familywise null includes ALL of them, per PREREG anti-hacking rule #1):
  per geometry g in {g050c5K60, g080c3nat, g065c3nat, g095c3nat} x {Lnet28, Lstr7, Mnet7, Lrow190, Mrow190}
  plus FUSED__{Lnet28, Lstr7, Mnet7}

Battery per variant:
  1. 3-site LOSO (site-centered OOS r = r_sc): ridge alpha chosen by inner 3-fold CV mode
     within each LOSO train fold (exp/26-28 convention); features z-scored by train-fold stats;
     site-demeaned within train fold.
  2. Confound test: ΔR² > 0 vs age/sex/maxmotion baseline (same LOSO, baseline = ridge on 3 confounds).
  3. Both-sex same-sign; all-site same-sign (per-site r_sc).
  4. FAMILYWISE: 200 permutations, permute y WITHIN site, recompute the FULL LOSO battery for
     every variant, take max |r_sc| across the family -> null distribution for max-stat.
     Alpha fixed at the mode selected on real data (per PREREG: mode-of-per-fold selection on
     real data then FIXED for permutations).

DEAD unless: fw-p<0.05 AND |r_sc|>0.15 AND ΔR²>0 AND both-sex same-sign AND all-site same-sign.
Targets: Inattentive (primary), Hyper (secondary) — both reported.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 200
rng_seed = 20260829

z = np.load(f"{OUT}/features.npz", allow_pickle=True)
archive_keys = [k for k in z.files if "__" in k]

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites_cc = np.array(meta_all["sites"])
ages_cc = np.array(meta_all["ages"], dtype=float)
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
sex_t = sex_all[keep_idx]
age_t = ages_cc[keep_idx]
mm_t = np.array([float(mmmap.get(ids_all[i], np.nan)) for i in keep_idx])
mm_t[np.isnan(mm_t)] = np.nanmean(mm_t)
age_t[np.isnan(age_t)] = np.nanmean(age_t)
CONF = np.column_stack([age_t, sex_t, mm_t])
sites_uniq = np.unique(site_t)
N = len(keep_idx)
print(f"T-cohort n={N} sites {dict(zip(*np.unique(site_t, return_counts=True)))}")

# ---------------- LOSO machinery ----------------
def site_demean(v, s, train_mask):
    out = v.copy()
    for sname in np.unique(s[train_mask]):
        m = train_mask & (s == sname)
        out[m] = v[m] - v[m].mean()
    return out

def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-12
    Xz = (Xtr - mu) / sd
    k = Xz.shape[1]
    w = np.linalg.solve(Xz.T @ Xz + alpha * np.eye(k), Xz.T @ ytr)
    return ((Xte - mu) / sd) @ w

def loso_cv(X, y, site, fixed_alpha=None, inner=True):
    """Returns (oos_pred_centered, per_site_r, alphas_used). Site-centered within train."""
    preds = np.full(len(y), np.nan)
    alphas = []
    for st in np.unique(site):
        te = site == st
        tr = ~te
        ytr_c = site_demean(y, site, tr)
        if fixed_alpha is None:
            # inner 3-fold on train (subject split, site-stratified random)
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
                    p = ridge_fit_predict(X[m2], ytr2_c[m2], X[va], a)
                    # eval r within va (no site centering inside inner fold; simple Pearson)
                    yc = y[va] - y[va].mean()
                    scores[a].append(np.corrcoef(p - p.mean(), yc)[0, 1] if np.std(p) > 1e-12 else 0.0)
            mean_scores = {a: np.nanmean(s) for a, s in scores.items()}
            mode_a = max(mean_scores, key=lambda a: mean_scores[a])
            alphas.append(mode_a)
            preds[te] = ridge_fit_predict(X[tr], ytr_c[tr], X[te], mode_a)
        else:
            alphas.append(fixed_alpha)
            preds[te] = ridge_fit_predict(X[tr], ytr_c[tr], X[te], fixed_alpha)
    # site-center predictions and targets for r_sc
    preds_c = preds.copy()
    for st in np.unique(site):
        m = site == st
        preds_c[m] = preds[m] - preds[m].mean()
    y_c = y.copy()
    for st in np.unique(site):
        m = site == st
        y_c[m] = y[m] - y[m].mean()
    r_sc = float(np.corrcoef(preds_c, y_c)[0, 1])
    per_site = {}
    for st in np.unique(site):
        m = site == st
        per_site[int(st)] = float(np.corrcoef(preds_c[m], y_c[m])[0, 1]) if m.sum() > 5 else np.nan
    return preds_c, r_sc, per_site, alphas

def r2_oos(pred, y):
    sse = np.sum((y - pred) ** 2); sst = np.sum((y - y.mean()) ** 2)
    return 1 - sse / sst

t0 = time.time()
gate1 = json.load(open(f"{OUT}/gate1_results.json"))
GATE2_KEYS = [k for k, v in gate1.items() if max(abs(v["oracle_r_inatt"]), abs(v["oracle_r_hyper"])) >= 0.15]
print(f"Gate-2 family: {len(GATE2_KEYS)} variants")
print(GATE2_KEYS)

FEATURES = {k: z[k][keep_idx] for k in GATE2_KEYS}
# drop NaN rows per variant happens inside fit? Enforce global: rows with any NaN in any feature
# were set NaN for short runs. Check:
nan_rows = np.zeros(N, bool)
for k in GATE2_KEYS:
    nan_rows |= ~np.all(np.isfinite(FEATURES[k]), axis=1)
print(f"rows with NaN in some variant: {nan_rows.sum()} (dropped from ALL analyses for familywise integrity)")
KEEP = ~nan_rows
FEATURES = {k: V[KEEP] for k, V in FEATURES.items()}
y_inatt_m, y_hyper_m = y_inatt[KEEP], y_hyper[KEEP]
site_m, sex_m, conf_m = site_t[KEEP], sex_t[KEEP], CONF[KEEP]
print(f"final n={KEEP.sum()}")

TARGETS = {"Inatt": y_inatt_m, "Hyper": y_hyper_m}

# ---------------- real-data battery ----------------
REAL = {}
for tname, y in TARGETS.items():
    REAL[tname] = {}
    # confound baseline
    _, r_conf, ps_conf, _ = loso_cv(conf_m, y, site_m, fixed_alpha=100.0)
    REAL[tname]["conf_baseline_r"] = r_conf
    for k in GATE2_KEYS:
        preds, r_sc, per_site, alphas = loso_cv(FEATURES[k], y, site_m)
        r2f = r2_oos(preds, y)
        # ΔR² vs confounds: also LOSO with conf + feature
        Xc = np.hstack([FEATURES[k], conf_m])
        predc, r_sc_c, _, _ = loso_cv(Xc, y, site_m)
        r2c = r2_oos(predc, y)
        # sex/site splits
        sex_r = {}
        for sx in (0, 1):
            m = sex_m == sx
            if m.sum() > 20:
                sex_r[int(sx)] = float(np.corrcoef(preds[m], y[m])[0, 1])
        REAL[tname][k] = {"r_sc": r_sc, "per_site_r": per_site, "alphas": alphas,
                          "r2": r2f, "r2_plus_conf": r2c, "dr2_vs_conf": r2f - r_conf,
                          "sex_r": sex_r}
        print(f"[{tname}] {k:38s} r_sc={r_sc:+.3f} per_site={ {kk: round(vv,3) for kk,vv in per_site.items()} } dr2={r2f - r_conf:+.3f} sex={sex_r}")
print(f"real battery done {time.time()-t0:.0f}s")

# ---------------- familywise null (max-stat over family) ----------------
# Per PREREG: alpha mode fixed at real-data selection per (variant, fold); perms re-run LOSO with fixed alphas.
print(f"\nfamilywise null: {N_PERM} perms, permute y within site ...")
FIXED_ALPHAS = {tname: {k: REAL[tname][k]["alphas"] for k in GATE2_KEYS} for tname in TARGETS}
rng = np.random.RandomState(rng_seed)
maxstat_null = {tname: [] for tname in TARGETS}
perm_site_r = {tname: {k: [] for k in GATE2_KEYS} for tname in TARGETS}
for p in range(N_PERM):
    for tname, y0 in TARGETS.items():
        yp = y0.copy()
        for st in np.unique(site_m):
            m = site_m == st
            yp[m] = rng.permutation(y0[m])
        mx = 0.0
        for k in GATE2_KEYS:
            alphas = FIXED_ALPHAS[tname][k]
            preds = np.full(len(yp), np.nan)
            for ai, st in enumerate(np.unique(site_m)):
                te = site_m == st; tr = ~te
                ytr_c = site_demean(yp, site_m, tr)
                preds[te] = ridge_fit_predict(FEATURES[k][tr], ytr_c[tr], FEATURES[k][te], alphas[ai])
            preds_c = preds.copy()
            for st in np.unique(site_m):
                m = site_m == st
                preds_c[m] = preds[m] - preds[m].mean()
            r = float(np.corrcoef(preds_c, yp)[0, 1]) if np.std(preds_c) > 1e-12 else 0.0
            perm_site_r[tname][k].append(r)
            if abs(r) > abs(mx):
                mx = r
        maxstat_null[tname].append(mx)
    if (p + 1) % 20 == 0:
        print(f"  perm {p+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)

FW = {}
for tname in TARGETS:
    null = np.array(maxstat_null[tname])
    for k in GATE2_KEYS:
        r = REAL[tname][k]["r_sc"]
        fw_p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_PERM + 1))
        REAL[tname][k]["fw_p"] = fw_p
        REAL[tname][k]["perm_null_r_mean"] = float(np.mean(np.abs(perm_site_r[tname][k])))
        alive = (fw_p < 0.05 and abs(r) > 0.15 and REAL[tname][k]["dr2_vs_conf"] > 0
                 and len(REAL[tname][k]["sex_r"]) == 2 and
                 all(np.sign(v) == np.sign(r) for v in REAL[tname][k]["sex_r"].values()) and
                 all(np.sign(v) == np.sign(r) for v in REAL[tname][k]["per_site_r"].values()))
        REAL[tname][k]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[{tname}] {k:38s} r_sc={r:+.3f} fw_p={fw_p:.4f} dr2={REAL[tname][k]['dr2_vs_conf']:+.3f} -> {REAL[tname][k]['GATE2_VERDICT']}")
    FW[tname] = {"null_max_mean": float(np.mean(np.abs(null))),
                 "null_max_95": float(np.percentile(np.abs(null), 95)),
                 "n_perm": N_PERM}

out = {"gate2_family": GATE2_KEYS, "n_perm": N_PERM, "fw_summary": FW, "battery": REAL}
with open(f"{OUT}/gate2_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nDONE {time.time()-t0:.0f}s -> gate2_results.json")
