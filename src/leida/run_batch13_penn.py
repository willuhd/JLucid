#!/usr/bin/env python3
"""exp/37 BATCH 13b — PennLEAD amplitude x ESWAN (5 QC sites' worth: single site)."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/37_leida"
TR = 0.8
sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead
from nilearn.datasets import fetch_atlas_schaefer_2018

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
def amp_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    Wk = np.fft.fft(ts, n=n_fft, axis=0) * np.fft.fft(w, n=n_fft)[:, None]
    return np.abs(np.fft.ifft(Wk, axis=0)[L // 2:L // 2 + T]) * 2

t0 = time.time()
pts, plab, pmeta = load_pennlead(condition="rest", qc_fd_thresh=0.5)
Np = len(pts)
sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=os.path.join(BASE, "data/atlas/schaefer_2018/nilearn"))
lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
net_code = {"Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6}
yeo = np.full(pmeta["N"], -1)
for i in range(min(400, pmeta["N"])):
    l = lab_names[i + 1] if i + 1 < len(lab_names) else ""
    for nm, code in net_code.items():
        if f"_{nm}_" in l: yeo[i] = code; break
net_idx = [np.where(yeo == k)[0] for k in range(7)]

GEOMS = {"a050": (0.05, 5, 60.0), "a080": (0.08, 3, 90.0), "a120": (0.12, 3, 60.0)}
FEAT = {}
for gname, (f, cyc, cap) in GEOMS.items():
    w, L = morlet(TR, f, cyc, cap)
    NA_m = np.full((Np, 7), np.nan); NA_s = np.full((Np, 7), np.nan)
    AEC = np.full((Np, 21), np.nan); Av = np.full((Np, 7), np.nan)
    for i, ts in enumerate(pts):
        T = ts.shape[0]
        if T < L + 20: continue
        A = amp_of(ts, w, L)
        NA = np.zeros((T, 7))
        for k in range(7):
            NA[:, k] = A[:, net_idx[k]].mean(axis=1)
        NA_m[i] = NA.mean(axis=0); NA_s[i] = NA.std(axis=0)
        AEC[i] = np.corrcoef(NA.T)[np.triu_indices(7, k=1)]
        Av[i] = np.array([A[:, net_idx[k]].std(axis=0).mean() for k in range(7)])
    FEAT[f"{gname}__NA_mean7"] = NA_m
    FEAT[f"{gname}__NA_std7"] = NA_s
    FEAT[f"{gname}__AEC21"] = AEC
    FEAT[f"{gname}__Avar7"] = Av
    print(f"{gname} ({time.time()-t0:.0f}s)", flush=True)

pids = list(pmeta["participant_ids"])
cohort = pd.read_csv(os.path.join(BASE, "data/pennlead/pheno/cohort.csv"))
cohort["participant_id"] = cohort["participant_id"].astype(str)
cm = cohort.set_index("participant_id")
ages = np.array(pmeta["ages"], dtype=float)
sexes = np.array([1.0 if str(s).upper().startswith("M") else 0.0 for s in pmeta["sexes"]])
fds = np.array(pmeta["fd_means"], dtype=float)

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

REAL = {}
for tcol in ["eswan_adhd_inattention_total", "eswan_adhd_total_score"]:
    y = np.array([float(cm.loc[p, tcol]) if p in cm.index and pd.notna(cm.loc[p, tcol]) else np.nan for p in pids])
    m = np.isfinite(y)
    yv = y[m]
    B = np.column_stack([ages[m], sexes[m], fds[m]])
    coef = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), B]), yv, rcond=None)[0]
    yr = yv - np.column_stack([np.ones(m.sum()), B]) @ coef
    REAL[tcol] = {}
    for fn, Xf in FEAT.items():
        X = Xf[m]; ok = np.isfinite(X).all(1)
        if ok.sum() < 40: continue
        r = cv_r(X[ok], yr[ok], seed=0)
        REAL[tcol][fn] = {"cv_r": r, "n": int(ok.sum())}
        print(f"[{tcol[13:25]:14s}] {fn:16s} CVr={r:+.3f} n={ok.sum()}")

rng = np.random.RandomState(20260916)
maxstats = []
for p_ in range(300):
    mx = 0.0
    for tcol in ["eswan_adhd_inattention_total", "eswan_adhd_total_score"]:
        y = np.array([float(cm.loc[p, tcol]) if p in cm.index and pd.notna(cm.loc[p, tcol]) else np.nan for p in pids])
        m = np.isfinite(y)
        yv = y[m]
        B = np.column_stack([ages[m], sexes[m], fds[m]])
        coef = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), B]), yv, rcond=None)[0]
        yr = yv - np.column_stack([np.ones(m.sum()), B]) @ coef
        yp = rng.permutation(yr)
        for fn, Xf in FEAT.items():
            X = Xf[m]; ok = np.isfinite(X).all(1)
            if ok.sum() < 40: continue
            r = cv_r(X[ok], yp[ok], seed=0)
            if abs(r) > abs(mx): mx = r
    maxstats.append(mx)
    if (p_+1) % 100 == 0: print(f"  perm {p_+1} ({time.time()-t0:.0f}s)", flush=True)
maxstats = np.array(maxstats)
print(f"null q95={np.percentile(np.abs(maxstats),95):.3f}")
for tcol in REAL:
    for fn in REAL[tcol]:
        r = REAL[tcol][fn]["cv_r"]
        fw_p = float((np.sum(np.abs(maxstats) >= abs(r)) + 1) / 301)
        REAL[tcol][fn]["fw_p"] = fw_p
        verdict = "ALIVE" if (fw_p < 0.05 and abs(r) > 0.25) else "dead"
        REAL[tcol][fn]["GATE2_VERDICT"] = verdict
        print(f"[{tcol[13:25]:14s}] {fn:16s} CVr={r:+.3f} fw_p={fw_p:.4f} -> {verdict}")

np.savez_compressed(f"{OUT}/batch13b_penn_amp.npz", pids=np.array(pids), **FEAT)
with open(f"{OUT}/batch13b_results.json", "w") as f:
    json.dump(REAL, f, indent=2)
print(f"DONE {time.time()-t0:.0f}s")
