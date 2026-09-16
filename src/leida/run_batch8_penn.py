#!/usr/bin/env python3
"""
exp/37 BATCH 8 — PennLEAD as DISCOVERY screen (preregistered in SIEVE_TABLE.md).
Single-site, n=85-87, ESWAN dimensional scores. Feature families (same definitions as
ADHD-200 batches, PennLEAD TR=0.8s, cortical 400 parcels Yeo-7):
  L7/Lnet28 analogs (co-leadership from network-phase eigenvectors — approximated at network
    level: L = (1/T) Σ u_t u_tᵀ of the 7-network phase-coherence leading eigenvector),
  W1 netphase-var 28, H3 hvar, H1 v_ab (ROI-pairwise PL variance 28), K1 signed phi/mu7.
Targets: eswan_adhd_inattention_total (primary), eswan_adhd_total_score, eswan_hyper... if
present; age/sex/FD partialled; Spearman + Pearson; permutation p (5000, Freedman-Lane-ish:
residualized target permuted). Familywise: Holm over blocks x targets.
Statistic per block: max |partial r| over a small ridge grid (in-sample ORACLE first —
Gate 1 style — then CV r for survivors; honest 5-fold CV r reported for ALL blocks).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
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

sys.path.insert(0, f"{BASE}/src")
from pennlead.datasets import load_pennlead
from nilearn.datasets import fetch_atlas_schaefer_2018
sch = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, data_dir=os.path.join(BASE, "data/atlas/schaefer_2018/nilearn"))
lab_names = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
net_code = {"Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6}

pts, plab, pmeta = load_pennlead(condition="rest", qc_fd_thresh=0.5)
Np = len(pts); NROI = pmeta["N"]
yeo = np.full(NROI, -1)
for i in range(min(400, NROI)):
    l = lab_names[i + 1] if i + 1 < len(lab_names) else ""
    for nm, code in net_code.items():
        if f"_{nm}_" in l: yeo[i] = code; break
net_idx = [np.where(yeo == k)[0] for k in range(7)]
print(f"n={Np} rois={NROI} net sizes={[len(x) for x in net_idx]}")

t0 = time.time()
GEOMS = {"g050": (0.05, 5, 60.0), "g080": (0.08, 3, 90.0)}
feat = {}
for gname, (f, cyc, cap) in GEOMS.items():
    w, L = morlet(TR, f, cyc, cap)
    W1 = np.full((Np, 28), np.nan); H1v = np.full((Np, 28), np.nan); H3v = np.full((Np, 8), np.nan)
    Lnet = np.full((Np, 28), np.nan)
    for i, ts in enumerate(pts):
        T = ts.shape[0]
        if T < 100: continue
        ph = phases_of(ts, w, L)
        TH = np.zeros((T, 7))
        for k in range(7):
            z = np.exp(1j * ph[:, net_idx[k]]).sum(axis=1)
            TH[:, k] = np.angle(z)
        for kk, (a_, b_) in enumerate(UP):
            d = wrap(TH[:, a_] - TH[:, b_])
            W1[i, kk] = d.var()
        # H1 v_ab: ROI-pairwise cos(phase diff) mean per pair-of-networks, variance over time
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
        # network-level phase coherence matrix eigenvector (7x7 LEiDA analog) -> L net28
        C = np.cos(TH[:, :, None] - TH[:, None, :])
        u_all = np.zeros((T, 7))
        var_node7 = np.zeros((T, 7))
        U = np.zeros((T, 7))
        for t in range(T):
            cc = np.cos(TH[t, :, None] - TH[t, None, :])
            ev, evec = np.linalg.eigh(cc)
            u = evec[:, -1]
            if u.sum() > 0: u = -u
            U[t] = u
        # hvar analog at network level
        var7 = U.var(axis=0)
        H3v[i] = np.concatenate([[var7.mean()], var7])
        Lm = (U.T @ U) / T
        iu = np.triu_indices(7)
        Lnet[i] = Lm[iu]
    feat[f"{gname}__W1"] = W1
    feat[f"{gname}__H1v"] = H1v
    feat[f"{gname}__H3v"] = H3v
    feat[f"{gname}__Lnet"] = Lnet
    print(f"{gname} done ({time.time()-t0:.0f}s)", flush=True)

# K1 signed phi/mu from whole-run FC
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
    phi7 = np.array([phi[[j for j in range(len(cort)) if yeo[cort[j]]==k]].mean() for k in range(7)])
    mu7 = np.array([mu[[j for j in range(len(cort)) if yeo[cort[j]]==k]].mean() for k in range(7)])
    K1[i] = np.concatenate([phi7, mu7])
feat["K1_signed"] = K1
print(f"K1 done ({time.time()-t0:.0f}s)")

# targets
cohort = pd.read_csv(os.path.join(BASE, "data/pennlead/pheno/cohort.csv"))
cohort["participant_id"] = cohort["participant_id"].astype(str)
pids = pmeta["participant_ids"]
tgt_cols = [c for c in cohort.columns if "eswan" in c.lower()]
print("eswan cols:", tgt_cols)
ymaps = {}
for c in tgt_cols:
    ymaps[c] = dict(zip(cohort["participant_id"], pd.to_numeric(cohort[c], errors="coerce")))
ages = np.array(pmeta["ages"], dtype=float)
sexes = np.array([1.0 if str(s).upper().startswith("M") else 0.0 for s in pmeta["sexes"]])
fds = np.array(pmeta["fd_means"], dtype=float)

results = {}
n_tests = 0
for tcol, ymap in ymaps.items():
    y = np.array([ymap.get(p, np.nan) for p in pids])
    m = np.isfinite(y)
    if m.sum() < 40 or np.nanstd(y[m]) < 1e-6: continue
    yv = y[m]; n_t = m.sum()
    B = np.column_stack([ages[m], sexes[m], fds[m]])
    # residualize y on B
    coef = np.linalg.lstsq(np.column_stack([np.ones(n_t), B]), yv, rcond=None)[0]
    yr = yv - np.column_stack([np.ones(n_t), B]) @ coef
    for fname, Xf in feat.items():
        X = Xf[m]
        ok = np.isfinite(X).all(1)
        if ok.sum() < 40: continue
        Xv = X[ok]; yo = yr[ok]
        mu = Xv.mean(0); sd = Xv.std(0) + 1e-12
        Xz = (Xv - mu) / sd
        best = (0.0, None)
        for a in [1, 10, 100, 1000, 1e4]:
            w = np.linalg.solve(Xz.T @ Xz + a * np.eye(Xz.shape[1]), Xz.T @ yo)
            r = float(np.corrcoef(Xz @ w, yo)[0, 1])
            if abs(r) > abs(best[0]): best = (r, a)
        results[f"{fname}|{tcol}"] = {"oracle_r": best[0], "alpha": best[1], "n": int(ok.sum())}
        n_tests += 1
        print(f"{fname:14s} {tcol:38s} oracle {best[0]:+.3f} n={ok.sum()}")

with open(f"{OUT}/batch8_gate1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nn_tests={n_tests}; saved batch8_gate1_results.json ({time.time()-t0:.0f}s)")
