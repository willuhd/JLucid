#!/usr/bin/env python3
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats as st

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
z = np.load(f"{OUT}/batch10_feats.npz")
p = np.load(f"{OUT}/batch10_pheno.npz")

FEATS = {"W1_rest": z["W1_r"], "W1_nback": z["W1_n"], "dW1": z["dW1"],
         "Lnet_rest": z["Lnet_r"], "Lnet_nback": z["Lnet_n"], "dLnet": z["dLnet"]}
dx = p["dx"]; esw = p["esw_tot"]; ages = p["ages"]; sexes = p["sexes"]; fds = p["fds"]
N = len(dx)
B = np.column_stack([ages, sexes, fds])

ALPHAS = [1, 10, 100, 1000, 1e4]
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

def run_target(y, label):
    m = np.isfinite(y)
    yv = y[m]
    Bm = B[m]
    coef = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), Bm]), yv, rcond=None)[0]
    yr = yv - np.column_stack([np.ones(m.sum()), Bm]) @ coef
    cells = {}
    for fn, Xf in FEATS.items():
        X = Xf[m]; ok = np.isfinite(X).all(1)
        if ok.sum() < 40: continue
        cells[fn] = (X[ok], yr[ok])
    return cells

def auc_cv(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    aucs = []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        mu = X[tr].mean(0); sd = X[tr].std(0)+1e-12
        Xz = (X[tr]-mu)/sd
        # small alpha grid via inner split
        best_a, best_s = 0.01, -1
        inner = np.array_split(rng.permutation(tr), 3)
        for a in [0.001, 0.01, 0.1, 1.0]:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(tr, iv)
                mu2 = X[tr2].mean(0); sd2 = X[tr2].std(0)+1e-12
                Xz2 = (X[tr2]-mu2)/sd2
                from sklearn.linear_model import LogisticRegression as LR
                clf = LR(C=a, max_iter=2000)
                clf.fit(Xz2, y[tr2])
                pr = clf.predict_proba((X[iv]-mu2)/sd2)[:,1]
                if len(np.unique(y[iv])) == 2:
                    sc.append(roc_auc_score(y[iv], pr))
            s = np.mean(sc) if sc else 0.5
            if s > best_s: best_s, best_a = s, a
        clf = LogisticRegression(C=best_a, max_iter=2000)
        clf.fit(Xz, y[tr])
        pr = clf.predict_proba((X[f]-mu)/sd)[:,1]
        if len(np.unique(y[f])) == 2:
            aucs.append(roc_auc_score(y[f], pr))
    return float(np.mean(aucs)) if aucs else 0.5

TDATA = {"dx": run_target(dx, "dx"), "esw": run_target(esw, "esw")}

REAL = {"dx": {}, "esw": {}}
for tcol, cells in TDATA.items():
    for fn, (X, yr) in cells.items():
        r = cv_r(X, yr, seed=0)
        REAL[tcol][fn] = {"cv_r": r, "n": len(yr)}
        print(f"[{tcol:4s}] {fn:12s} CVr={r:+.3f} n={len(yr)}")

# DX AUC (baseline + features)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
m = np.isfinite(dx)
dxm = dx[m].astype(int)
Bm = B[m]
# find per-feature nan mask per feature
DXCELLS = {}
for fn, Xf in FEATS.items():
    X = Xf[m]; ok = np.isfinite(X).all(1)
    if ok.sum() < 40: continue
    DXCELLS[fn] = (X[ok], dxm[ok])
auc_base_cells = {}
for fn, (X, y) in DXCELLS.items():
    Xf = np.column_stack([B[ok_n], X]) if False else None
    ok_n = None
    pass
# simpler: use same ok mask per feature; baseline uses B rows for that mask
for fn, (X, y) in DXCELLS.items():
    # recompute ok for this feature
    X0 = FEATS[fn][m]; ok = np.isfinite(X0).all(1)
    Bf = B[m][ok]
    # baseline
    aucs_b = []
    aucs_f = []
    rng = np.random.RandomState(0)
    for rep in range(10):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            mu = Bf[tr].mean(0); sd = Bf[tr].std(0)+1e-12
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit((Bf[tr]-mu)/sd, y[tr])
            aucs_b.append(roc_auc_score(y[te], clf.predict_proba((Bf[te]-mu)/sd)[:,1]))
            Xf = np.column_stack([Bf, X])
            mu2 = Xf[tr].mean(0); sd2 = Xf[tr].std(0)+1e-12
            # pick C
            bestC, bestS = 0.01, -1
            for C in [0.001, 0.01, 0.1, 1.0]:
                scv = []
                skf2 = StratifiedKFold(3, shuffle=True, random_state=100+rep)
                for tr2, te2 in skf2.split(np.zeros(len(tr)), y[tr]):
                    mu3 = Xf[tr][tr2].mean(0); sd3 = Xf[tr][tr2].std(0)+1e-12
                    cl = LogisticRegression(C=C, max_iter=2000)
                    cl.fit((Xf[tr][tr2]-mu3)/sd3, y[tr][tr2])
                    scv.append(roc_auc_score(y[tr][te2], cl.predict_proba((Xf[tr][te2]-mu3)/sd3)[:,1]))
                s = np.mean(scv)
                if s > bestS: bestS, bestC = s, C
            clf2 = LogisticRegression(C=bestC, max_iter=2000)
            clf2.fit((Xf[tr]-mu2)/sd2, y[tr])
            aucs_f.append(roc_auc_score(y[te], clf2.predict_proba((Xf[te]-mu2)/sd2)[:,1]))
    ab = float(np.mean(aucs_b)); af = float(np.mean(aucs_f))
    REAL["dx"][fn]["auc_base"] = ab
    REAL["dx"][fn]["auc_feat"] = af
    REAL["dx"][fn]["d_auc"] = af - ab
    print(f"[dx  ] {fn:12s} AUC {af:.4f} vs base {ab:.4f} (d={af-ab:+.4f})")

# perm null over all cells (both stat types: r for esw, d_auc for dx)
print("\nperm null (1000)...", flush=True)
rng = np.random.RandomState(20260905)
maxstats = []
for p_ in range(1000):
    mx = 0.0
    # esw cells: r
    cells = TDATA["esw"]
    yp_all = rng.permutation(esw[np.isfinite(esw)])
    for fn, (X, yr) in cells.items():
        r = cv_r(X, rng.permutation(yr), seed=0)
        if abs(r) > abs(mx): mx = r
    # dx cells: d_auc (rough: use cv_r stat as proxy for null shape)
    for fn, (X, y) in DXCELLS.items():
        r = cv_r(X, rng.permutation(y.astype(float)), seed=0)
        if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 200 == 0: print(f"  perm {p_+1}", flush=True)
maxstats = np.array(maxstats)

for tcol in REAL:
    for fn in REAL[tcol]:
        stat = REAL[tcol][fn]["cv_r"] if tcol == "esw" else REAL[tcol][fn]["d_auc"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(stat)) + 1) / 1001)
        REAL[tcol][fn]["fw_p"] = fw_p
        REAL[tcol][fn]["GATE2_VERDICT"] = "ALIVE" if (fw_p < 0.05 and abs(stat) > 0.25) else "dead"
        print(f"[{tcol:4s}] {fn:12s} stat={stat:+.3f} fw_p={fw_p:.4f} -> {REAL[tcol][fn]['GATE2_VERDICT']}")

with open(f"{OUT}/batch10_results.json", "w") as f:
    json.dump(REAL, f, indent=2)
print("saved batch10_results.json")
