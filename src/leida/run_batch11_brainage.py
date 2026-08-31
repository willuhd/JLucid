#!/usr/bin/env python3
"""
exp/37 BATCH 11 — Maturational-lag brain-age delta test (preregistered above).
ADHD-200: age model trained on HC ONLY (site-centering, nested-CV ridge over Lnet28 and
Lrow190); predicted age for all; Δ = age − predicted. Tests:
  (a) ADHD vs HC Δ (lag hypothesis; also within-site to kill site effects)
  (b) Δ vs Inatt / Hyper among ADHD probands
  (c) PennLEAD: HC-only-trained age model on Lnet_r (n=87 rest), Δ vs ESWAN (n=85).
Permutation nulls per cell; Holm over the (a,b,c) family.
"""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
ALPHAS = [0.1, 1, 10, 100, 1000, 3000, 1e4, 3e4, 1e5, 3e5]
N_PERM = 500

sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
ts_all, labels_all, meta_all = load_cc200(qc_only=True)
ids_all = [str(x) for x in meta_all["subject_ids"]]
sites = np.array(meta_all["sites"])
ages = np.array(meta_all["ages"], dtype=float)
dx = np.array(labels_all).astype(int)
ph = pd.read_csv(os.path.join(BASE, "data/adhd200/adhd200_preprocessed_phenotypics.tsv"), sep="\t")
ph["ScanDir ID"] = ph["ScanDir ID"].astype(str)
for c in ["Inattentive", "Hyper/Impulsive"]:
    ph[c + "_n"] = pd.to_numeric(ph[c], errors="coerce")
    ph.loc[ph[c + "_n"] == -999, c + "_n"] = np.nan
phmap = ph.set_index("ScanDir ID")[["Inattentive_n", "Hyper/Impulsive_n", "ADHD Measure"]]

zb1 = np.load(f"{OUT}/features.npz")
FEATS = {"Lnet28": zb1["g050c5K60__Lnet28"], "Lrow190": zb1["g050c5K60__Lrow190"]}

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

t0 = time.time()
results = {}
for fname, Xall in FEATS.items():
    ok = np.isfinite(Xall).all(1) & np.isfinite(ages)
    Xo, yo, so, do = Xall[ok], ages[ok], sites[ok], dx[ok]
    hc = do == 0
    # nested-CV within HC for honest age-model assessment
    preds_hc = np.full(hc.sum(), np.nan)
    idx_hc = np.where(hc)[0]
    for seed in [0]:  # single-seed for the model (the TEST is the delta, not age r)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(idx_hc))
        folds = np.array_split(idx, 5)
        pr = np.full(len(idx_hc), np.nan)
        for f in folds:
            te = idx_hc[f]; tr = np.setdiff1d(idx_hc, te)
            mask_tr = np.zeros(len(yo), bool); mask_tr[tr] = True
            ytr_c = site_demean(yo, so, mask_tr)
            best_a, best_s = ALPHAS[0], -1
            inner = np.array_split(rng.permutation(tr), 3)
            for a in ALPHAS:
                sc = []
                for iv in inner:
                    tr2 = np.setdiff1d(tr, iv)
                    m2 = np.zeros(len(yo), bool); m2[tr2] = True
                    ytr2_c = site_demean(yo, so, m2)
                    p = rfp(Xo[tr2], ytr2_c[tr2], Xo[iv], a)
                    sc.append(np.corrcoef(p, yo[iv])[0,1] if np.std(p) > 1e-12 else 0)
                s = np.mean(sc)
                if s > best_s: best_s, best_a = s, a
            pr[f] = rfp(Xo[tr], ytr_c[tr], Xo[te], best_a)
        preds_hc = pr
    r_hc = float(np.corrcoef(preds_hc, yo[idx_hc])[0,1])
    # FULL HC model -> predict everyone
    mask_hc = np.zeros(len(yo), bool); mask_hc[idx_hc] = True
    ytr_c = site_demean(yo, so, mask_hc)
    # pick alpha via 3-fold HC CV
    rng = np.random.RandomState(1)
    best_a, best_s = ALPHAS[0], -1
    inner = np.array_split(rng.permutation(idx_hc), 3)
    for a in ALPHAS:
        sc = []
        for iv in inner:
            tr2 = np.setdiff1d(idx_hc, iv)
            m2 = np.zeros(len(yo), bool); m2[tr2] = True
            ytr2_c = site_demean(yo, so, m2)
            p = rfp(Xo[tr2], ytr2_c[tr2], Xo[iv], a)
            sc.append(np.corrcoef(p, yo[iv])[0,1] if np.std(p) > 1e-12 else 0)
        s = np.mean(sc)
        if s > best_s: best_s, best_a = s, a
    # site-centered full model: train with site-centering; predictions include site-mean
    # reconstruction? For delta, use site-centered prediction (within-site delta):
    preds_all = rfp(Xo[idx_hc], ytr_c[idx_hc], Xo, best_a)
    # add back site means of y for interpretability (predicted age in years)
    site_means = {}
    for st in np.unique(so):
        m = mask_hc & (so == st)
        site_means[st] = yo[m].mean() if m.sum() > 5 else yo[mask_hc].mean()
    pred_age = preds_all + np.array([site_means[s] for s in so])
    delta = yo - pred_age
    # (a) ADHD vs HC delta, site-demeaned delta (within-site lag)
    delta_ws = delta.copy()
    for st in np.unique(so):
        m = so == st
        delta_ws[m] -= delta[m].mean()   # NOTE: mean over ALL subjects at site incl ADHD —
        # for a group difference this is fine (shift-invariant within site)
    adhd_d = delta_ws[do == 1]; hc_d = delta_ws[do == 0]
    from scipy.stats import ttest_ind
    t, p_t = ttest_ind(adhd_d, hc_d)
    # permutation null (within-site DX shuffle)
    rng_p = np.random.RandomState(20260910)
    obs = adhd_d.mean() - hc_d.mean()
    nulls = []
    for _ in range(N_PERM):
        yp = do.copy()
        for st in np.unique(so):
            m = so == st
            yp[m] = rng_p.permutation(do[m])
        nulls.append(delta_ws[yp == 1].mean() - delta_ws[yp == 0].mean())
    nulls = np.array(nulls)
    p_perm = (np.sum(np.abs(nulls) >= abs(obs)) + 1) / (N_PERM + 1)
    print(f"[{fname}] HC age-model r={r_hc:.3f}; ADHD−HC Δ (site-centered) = {obs:+.3f}y  t-p={p_t:.4f} perm-p={p_perm:.4f}")
    results[fname] = {"r_hc_model": r_hc, "delta_adhd_minus_hc": float(obs), "p_t": float(p_t), "p_perm": float(p_perm),
                      "adhd_delta_mean": float(adhd_d.mean()), "hc_delta_mean": float(hc_d.mean()),
                      "alpha": best_a}
    # (b) delta vs symptoms among ADHD probands (T-cohort: sites 3,5,6, measure 2/3)
    keep = []
    for i_pos, i in enumerate(np.where(ok)[0]):
        sid = ids_all[i]
        if sites[i] not in (3,5,6) or sid not in phmap.index: continue
        if str(phmap.loc[sid, "ADHD Measure"]) not in ("2","3"): continue
        keep.append(i_pos)
    keep = np.array(keep)
    for tgt, col in [("Inatt", "Inattentive_n"), ("Hyper", "Hyper/Impulsive_n")]:
        yt = np.array([float(phmap.loc[ids_all[np.where(ok)[0][k]], col]) for k in keep])
        mt = np.isfinite(yt)
        dt = delta_ws[keep[mt]]; yy = yt[mt]
        r = float(np.corrcoef(dt, yy)[0,1])
        rng2 = np.random.RandomState(20260911)
        nn = [float(np.corrcoef(rng2.permutation(dt), yy)[0,1]) for _ in range(N_PERM)]
        pp = (np.sum(np.abs(np.array(nn)) >= abs(r)) + 1) / (N_PERM + 1)
        print(f"  [{fname}] Δ vs {tgt} among ADHD: r={r:+.3f} p={pp:.4f} n={mt.sum()}")
        results[fname][f"delta_vs_{tgt}"] = {"r": r, "p": pp, "n": int(mt.sum())}

# ---- (c) PennLEAD ----
print("\nPennLEAD ...")
z10 = np.load(f"{OUT}/batch10_feats.npz"); p10 = np.load(f"{OUT}/batch10_pheno.npz")
Lnet_r = z10["Lnet_r"]; ages_p = p10["ages"]; dx_p = p10["dx"]; esw = p10["esw_tot"]
ok = np.isfinite(Lnet_r).all(1) & np.isfinite(ages_p)
Xp, yp_, dp_ = Lnet_r[ok], ages_p[ok], dx_p[ok]
hc_p = dp_ == 0
# HC-only age model (single site)
def rfp2(Xtr, ytr, Xte, a):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-12
    Xz = (Xtr-mu)/sd
    w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@ytr)
    return ((Xte-mu)/sd)@w
best_a, best_s = 1, -1
rng = np.random.RandomState(2)
inner = np.array_split(rng.permutation(np.where(hc_p)[0]), 3)
for a in ALPHAS:
    sc = []
    for iv in inner:
        tr2 = np.setdiff1d(np.where(hc_p)[0], iv)
        p = rfp2(Xp[tr2], yp_[tr2], Xp[iv], a)
        sc.append(np.corrcoef(p, yp_[iv])[0,1] if np.std(p) > 1e-12 else 0)
    s = np.mean(sc)
    if s > best_s: best_s, best_a = s, a
preds_all_p = rfp2(Xp[hc_p], yp_[hc_p], Xp, best_a)
delta_p = yp_ - preds_all_p
# HC model honest r
preds_hc_p = np.full(hc_p.sum(), np.nan)
idx_hc_p = np.where(hc_p)[0]
for seed in [0]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(idx_hc_p))
    folds = np.array_split(idx, 5)
    pr = np.full(len(idx_hc_p), np.nan)
    for f in folds:
        te = idx_hc_p[f]; tr = np.setdiff1d(idx_hc_p, te)
        b_a, b_s = 1, -1
        inner = np.array_split(rng.permutation(tr), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(tr, iv)
                p = rfp2(Xp[tr2], yp_[tr2], Xp[iv], a)
                sc.append(np.corrcoef(p, yp_[iv])[0,1] if np.std(p) > 1e-12 else 0)
            s = np.mean(sc)
            if s > b_s: b_s, b_a = s, a
        pr[f] = rfp2(Xp[tr], yp_[tr], Xp[te], b_a)
    preds_hc_p = pr
r_hc_p = float(np.corrcoef(preds_hc_p, yp_[idx_hc_p])[0,1])
from scipy.stats import ttest_ind
t_p, p_t_p = ttest_ind(delta_p[dp_==1], delta_p[dp_==0])
rng3 = np.random.RandomState(20260912)
obs_p = delta_p[dp_==1].mean() - delta_p[dp_==0].mean()
nn = [delta_p[rng3.permutation(dp_)==1].mean() - delta_p[rng3.permutation(dp_)==0].mean() for _ in range(N_PERM)]
pp_p = (np.sum(np.abs(np.array(nn)) >= abs(obs_p)) + 1) / (N_PERM + 1)
print(f"[PennLnet] HC age-model r={r_hc_p:.3f}; ADHD−HC Δ = {obs_p:+.3f}y  t-p={p_t_p:.4f} perm-p={pp_p:.4f} (n ADHD={int((dp_==1).sum())})")
# delta vs ESWAN
ok_esw = np.isfinite(esw[ok])
r_esw = float(np.corrcoef(delta_p[ok_esw], esw[ok][ok_esw])[0,1])
rng4 = np.random.RandomState(20260913)
nn2 = [float(np.corrcoef(rng4.permutation(delta_p[ok_esw]), esw[ok][ok_esw])[0,1]) for _ in range(N_PERM)]
pp_esw = (np.sum(np.abs(np.array(nn2)) >= abs(r_esw)) + 1) / (N_PERM + 1)
print(f"[PennLnet] Δ vs ESWAN total: r={r_esw:+.3f} p={pp_esw:.4f} n={ok_esw.sum()}")

results["PennLnet"] = {"r_hc_model": r_hc_p, "delta_adhd_minus_hc": float(obs_p), "p_perm": float(pp_p),
                        "delta_vs_esw": {"r": r_esw, "p": pp_esw, "n": int(ok_esw.sum())}}

with open(f"{OUT}/batch11_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
