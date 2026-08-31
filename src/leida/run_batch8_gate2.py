#!/usr/bin/env python3
"""
exp/37 BATCH 8 GATE 2 — honest CV for PennLEAD discovery survivors.
All 9 blocks x 3 ESWAN targets (all passed Gate-1-style oracle; per prereg everything enters).
Estimator: 5-fold CV ridge (inner-fold alpha via nested CV on train), age/sex/FD partialled y.
Statistic: pooled CV r. Null: 1000 permutations of residualized y (complete shuffle — single
site), max-stat over ALL 27 cells. Criteria: fw-p<0.05 AND |r|>0.25 AND Spearman same-sign.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead

g1 = json.load(open(f"{OUT}/batch8_gate1_results.json"))
N_PERM = 1000

pts, plab, pmeta = load_pennlead(condition="rest", qc_fd_thresh=0.5)
Np = len(pts)
pids = pmeta["participant_ids"]
ages = np.array(pmeta["ages"], dtype=float)
sexes = np.array([1.0 if str(s).upper().startswith("M") else 0.0 for s in pmeta["sexes"]])
fds = np.array(pmeta["fd_means"], dtype=float)

# features were not saved — rebuild exactly as batch8 (deterministic, same code path)
TR = 0.8
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]
def kernel_len(TR, f, cycles, cap):
    sigma = cycles / (2 * np.pi * f)
    L = int(6 * sigma / TR); L3 = int(3.0 / f / TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap / TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap - 1
    if L > Lcap: L = Lcap
    return L
def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L // 2) * TR
    w = np.pi ** -0.25 * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t ** 2 / (2 * (cycles / (2 * np.pi * f)) ** 2))
    return w / np.sqrt(np.sum(np.abs(w) ** 2) + 1e-12), L
def phases_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    return np.angle(np.fft.ifft(np.fft.fft(ts, n=n_fft, axis=0) * np.fft.fft(w, n=n_fft)[:, None], axis=0)[L // 2:L // 2 + T])
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
from nilearn.datasets import fetch_atlas_schaefer_2018
sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=os.path.join(BASE, "data/atlas/schaefer_2018/nilearn"))
lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
net_code = {"Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6}
yeo = np.full(pmeta["N"], -1)
for i in range(min(400, pmeta["N"])):
    l = lab_names[i + 1] if i + 1 < len(lab_names) else ""
    for nm, code in net_code.items():
        if f"_{nm}_" in l: yeo[i] = code; break
net_idx = [np.where(yeo == k)[0] for k in range(7)]

t0 = time.time()
GEOMS = {"g050": (0.05, 5, 60.0), "g080": (0.08, 3, 90.0)}
feat = {}
for gname, (f, cyc, cap) in GEOMS.items():
    w, L = morlet(TR, f, cyc, cap)
    W1 = np.full((Np, 28), np.nan); H1v = np.full((Np, 28), np.nan); H3v = np.full((Np, 8), np.nan); Lnet = np.full((Np, 28), np.nan)
    for i, ts in enumerate(pts):
        T = ts.shape[0]
        if T < 100: continue
        ph = phases_of(ts, w, L)
        TH = np.zeros((T, 7))
        for k in range(7):
            z = np.exp(1j * ph[:, net_idx[k]]).sum(axis=1)
            TH[:, k] = np.angle(z)
        for kk, (a_, b_) in enumerate(UP):
            W1[i, kk] = wrap(TH[:, a_] - TH[:, b_]).var()
        cosd = np.cos(ph[:, :, None] - ph[:, None, :])
        for kk, (a_, b_) in enumerate(UP):
            ia, ib = net_idx[a_], net_idx[b_]
            if a_ == b_:
                mm = cosd[:, ia][:, :, ia]
                ii, jj = np.triu_indices(len(ia), k=1)
                series = mm[:, ii, jj].mean(axis=1)
            else:
                series = cosd[:, ia][:, :, ib].mean(axis=(1, 2))
            H1v[i, kk] = series.var()
        U = np.zeros((T, 7))
        for t in range(T):
            cc = np.cos(TH[t, :, None] - TH[t, None, :])
            evv, evec = np.linalg.eigh(cc)
            u = evec[:, -1]
            if u.sum() > 0: u = -u
            U[t] = u
        var7 = U.var(axis=0)
        H3v[i] = np.concatenate([[var7.mean()], var7])
        Lm = (U.T @ U) / T
        iu = np.triu_indices(7)
        Lnet[i] = Lm[iu]
    feat[f"{gname}__W1"] = W1; feat[f"{gname}__H1v"] = H1v; feat[f"{gname}__H3v"] = H3v; feat[f"{gname}__Lnet"] = Lnet
    print(f"{gname} rebuilt ({time.time()-t0:.0f}s)", flush=True)
K1 = np.full((Np, 14), np.nan)
for i, ts in enumerate(pts):
    T = ts.shape[0]
    if T < 100: continue
    cort = np.concatenate(net_idx)
    Xz = (ts[:, cort] - ts[:, cort].mean(0)) / (ts[:, cort].std(0) + 1e-12)
    FC = np.nan_to_num(np.corrcoef(Xz.T))
    lam = np.linalg.eigvalsh((FC + FC.T)/2)
    A_s = FC / (1.0 + float(lam.max()) + 1e-12) - np.eye(FC.shape[0])
    ev = np.linalg.eigvalsh((A_s + A_s.T)/2)
    if ev.max() >= 0:
        A_s = A_s - (ev.max() + 1e-6) * np.eye(FC.shape[0])
    vals, vecs = np.linalg.eigh((A_s + A_s.T)/2)
    phi = (vecs**2) @ (1.0/(-2.0*vals))
    mu = (vecs**2) @ (1.0 - np.exp(vals))
    masks = [[j for j in range(len(cort)) if yeo[cort[j]] == k] for k in range(7)]
    phi7 = np.array([phi[mk].mean() for mk in masks])
    mu7 = np.array([mu[mk].mean() for mk in masks])
    K1[i] = np.concatenate([phi7, mu7])
feat["K1_signed"] = K1
print(f"features rebuilt ({time.time()-t0:.0f}s)", flush=True)

cohort = pd.read_csv(os.path.join(BASE, "data/pennlead/pheno/cohort.csv"))
cohort["participant_id"] = cohort["participant_id"].astype(str)
ymaps = {c: dict(zip(cohort["participant_id"], pd.to_numeric(cohort[c], errors="coerce")))
         for c in ["eswan_adhd_inattention_total", "eswan_adhd_hyperactivity_impulsivity_total", "eswan_adhd_total_score"]}

ALPHAS = [1, 10, 100, 1000, 1e4]
def cv_r(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    preds = np.full(len(y), np.nan)
    alpha_used = []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        # inner 3-fold alpha
        best_a, best_s = ALPHAS[0], -1
        inner = np.array_split(rng.permutation(tr), 3)
        for a in ALPHAS:
            sc = []
            for iv in inner:
                tr2 = np.setdiff1d(tr, iv)
                mu = X[tr2].mean(0); sd = X[tr2].std(0) + 1e-12
                Xz = (X[tr2]-mu)/sd
                w = np.linalg.solve(Xz.T@Xz + a*np.eye(Xz.shape[1]), Xz.T@y[tr2])
                p = ((X[iv]-mu)/sd)@w
                sc.append(np.corrcoef(p, y[iv])[0,1] if np.std(p) > 1e-12 else 0)
            s = np.mean(sc)
            if s > best_s: best_s, best_a = s, a
        alpha_used.append(best_a)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-12
        Xz = (X[tr]-mu)/sd
        w = np.linalg.solve(Xz.T@Xz + best_a*np.eye(Xz.shape[1]), Xz.T@y[tr])
        preds[f] = ((X[f]-mu)/sd)@w
    return float(np.corrcoef(preds, y)[0,1]), preds, alpha_used

# assemble per-target data
TDATA = {}
for tcol, ymap in ymaps.items():
    y = np.array([ymap.get(p, np.nan) for p in pids])
    m = np.isfinite(y)
    if m.sum() < 40: continue
    yv = y[m]
    B = np.column_stack([ages[m], sexes[m], fds[m]])
    coef = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), B]), yv, rcond=None)[0]
    yr = yv - np.column_stack([np.ones(m.sum()), B]) @ coef
    feats_m = {}
    for fn, Xf in feat.items():
        X = Xf[m]; ok = np.isfinite(X).all(1)
        feats_m[fn] = (X[ok], yr[ok])
    TDATA[tcol] = feats_m

REAL = {}
for tcol, feats_m in TDATA.items():
    REAL[tcol] = {}
    for fn, (X, yr) in feats_m.items():
        r, preds, alphas = cv_r(X, yr, seed=0)
        rho = float(st.spearmanr(preds, yr)[0])
        REAL[tcol][fn] = {"cv_r": r, "spearman": rho, "alphas": alphas, "n": len(yr)}
        print(f"[{tcol[13:40]:28s}] {fn:12s} CVr={r:+.3f} sp={rho:+.3f} n={len(yr)}")

print(f"\nperm null ({N_PERM}) ...", flush=True)
rng = np.random.RandomState(20260903)
maxstats = []
for p_ in range(N_PERM):
    mx = 0.0
    for tcol, feats_m in TDATA.items():
        for fn, (X, yr) in feats_m.items():
            yp = rng.permutation(yr)
            r, _, _ = cv_r(X, yp, seed=0)
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 100 == 0: print(f"  perm {p_+1}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)

print()
for tcol in REAL:
    for fn in REAL[tcol]:
        r = REAL[tcol][fn]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / (N_PERM + 1))
        REAL[tcol][fn]["fw_p"] = fw_p
        alive = fw_p < 0.05 and abs(r) > 0.25 and np.sign(REAL[tcol][fn]["spearman"]) == np.sign(r)
        REAL[tcol][fn]["GATE2_VERDICT"] = "ALIVE" if alive else "dead"
        print(f"[{tcol[13:40]:28s}] {fn:12s} CVr={r:+.3f} fw_p={fw_p:.4f} -> {REAL[tcol][fn]['GATE2_VERDICT']}")

out = {"real": REAL, "null_max": {"mean": float(np.mean(np.abs(maxstats))), "q95": float(np.percentile(np.abs(maxstats), 95))}, "n_perm": N_PERM}
with open(f"{OUT}/batch8_gate2_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE {time.time()-t0:.0f}s")
