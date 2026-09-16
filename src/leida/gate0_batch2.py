#!/usr/bin/env python3
"""exp/37 BATCH 2 GATE 0 — split-half reliability of W1/W2/V1/V2/C1 features.
Reliability ONLY; no target touched. Split = first vs second half of run; SB=2r/(1+r).
For vector features: mean per-component SB + count of comps >= 0.30."""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

BASE = "/Volumes/thinkplus/Code/JLucid"
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200

OUT = f"{BASE}/results"
z = np.load(f"{OUT}/batch2_features.npz")
ts_all, _, meta = load_cc200(qc_only=True)
Ts = [t.shape[0] for t in ts_all]
sites = np.array(meta["sites"])

# rebuild halves
GEOMS = {"g050c5K60": (0.05, 5, 60.0), "g080c3nat": (0.08, 3, 90.0)}
TR = 2.0
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]
import json as _json
yeo = _json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]

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
def lam_series(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    return (tr + disc) / 2.0, (tr - disc) / 2.0

def worker_half(args):
    ts, f, cyc, cap = args
    w, L = morlet(TR, f, cyc, cap)
    ph = phases_of(ts, w, L)
    TH = np.zeros((ph.shape[0], 7))
    for k in range(7):
        z = np.exp(1j * ph[:, net_idx[k]]).sum(axis=1)
        TH[:, k] = np.angle(z)
    l1, _ = lam_series(ph)
    return TH, l1 / ph.shape[1]

def site_resid(X, s):
    uniq = np.unique(s)
    D = (s[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(s)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef
def sb(r): return 2 * r / (1 + abs(r)) if r > -1 else -1

results = {}
for gname, (f, cyc, cap) in GEOMS.items():
    with mp.Pool(4) as pool:
        outs = pool.map(worker_half, [(ts_all[i], f, cyc, cap) for i in range(len(ts_all))], chunksize=16)
    h1W1 = []; h2W1 = []; h1W2 = []; h2W2 = []; h1V1 = []; h2V1 = []; h1V2 = []; h2V2 = []
    keep = []
    for i, (TH, l1s) in enumerate(outs):
        T = Ts[i]
        if T < 60:
            keep.append(False); continue
        keep.append(True)
        h = T // 2
        def feats(THs, l1s_part):
            dphase = np.zeros((len(THs), 28)); plv = np.zeros((len(THs), 28))
            for kk, (a, b) in enumerate(UP):
                d = wrap(THs[:, a] - THs[:, b])
                dphase[:, kk] = d; plv[:, kk] = np.cos(d)
            r_t = np.abs(np.exp(1j * THs).sum(axis=1)) / 7.0
            return dphase.var(0), plv.std(0), l1s_part.std(), r_t.std()
        w1a, w2a, v1a, v2a = feats(TH[:h], l1s[:h])
        w1b, w2b, v1b, v2b = feats(TH[h:], l1s[h:])
        h1W1.append(w1a); h2W1.append(w1b); h1W2.append(w2a); h2W2.append(w2b)
        h1V1.append(v1a); h2V1.append(v1b); h1V2.append(v2a); h2V2.append(v2b)
    keep = np.array(keep)
    h1W1 = np.array(h1W1); h2W1 = np.array(h2W1)
    h1W2 = np.array(h1W2); h2W2 = np.array(h2W2)
    sv = sites[keep]
    for nm, (A, B) in [("W1", (h1W1, h2W1)), ("W2", (h1W2, h2W2))]:
        comps = []
        for j in range(28):
            r = np.corrcoef(A[:, j], B[:, j])[0, 1]
            comps.append({"pair": f"{NETS[UP[j][0]]}-{NETS[UP[j][1]]}", "sb": float(sb(r))})
        sbs = [c["sb"] for c in comps]
        results[f"{gname}__{nm}"] = {"mean_sb": float(np.mean(sbs)), "n_ge30": int(np.sum(np.array(sbs) >= 0.30)),
                                     "max_sb": float(np.max(sbs)), "per_comp": comps}
    for nm, (A, B) in [("V1", (np.array(h1V1), np.array(h2V1))), ("V2", (np.array(h1V2), np.array(h2V2)))]:
        r = np.corrcoef(A, B)[0, 1]
        results[f"{gname}__{nm}"] = {"sb": float(sb(r))}

# C1: rebuild halves? TCE depends on whole-run A and states; split-half for A-based features:
# compute TCE on first-half timeseries and second-half timeseries separately.
def worker_tce(ts):
    if ts is None or len(ts) < 30: return np.full(8, np.nan)
    T, N = ts.shape
    Xz = (ts - ts.mean(0)) / (ts.std(0) + 1e-12)
    FC = np.nan_to_num(np.corrcoef(Xz.T))
    Araw = np.abs(FC)
    lmax = float(np.max(np.linalg.eigvalsh((Araw + Araw.T) / 2)))
    A = Araw / (1.0 + lmax + 1e-12) - np.eye(N)
    A = (A + A.T) / 2
    net_z = np.zeros((T, 7))
    for k in range(7):
        net_z[:, k] = Xz[:, net_idx[k]].mean(axis=1)
    dom = net_z.argmax(axis=1)
    X = np.zeros((7, N))
    for k in range(7):
        m = dom == k
        X[k] = Xz[m].mean(axis=0) if m.sum() >= 3 else Xz.mean(axis=0)
    W = np.eye(N) + A @ A
    Winv = np.linalg.pinv(W + 1e-9 * np.eye(N))
    E = np.zeros((7, 7, N))
    for a in range(7):
        for b in range(7):
            if a == b: continue
            d = X[b] - X[a]
            E[a, b] = d * (Winv @ d)
    tce_node = E.sum(axis=(0, 1)) / 42.0
    tce7 = np.array([tce_node[net_idx[k]].mean() for k in range(7)])
    uni = [(0,1)]; het = [(a,b) for a in range(2,7) for b in range(a+1,7)]; bet = [(a,b) for a in range(2) for b in range(2,7)]
    e_het = np.mean([E[a,b].sum() for a,b in het]); e_bet = np.mean([E[a,b].sum() for a,b in bet])
    return np.concatenate([tce7, [np.log((e_bet + 1e-12) / (e_het + 1e-12))]])

with mp.Pool(4) as pool:
    th1 = pool.map(worker_tce, [t[:Ts[i]//2] if Ts[i] >= 60 else None for i, t in enumerate(ts_all)], chunksize=16)
    th2 = pool.map(worker_tce, [t[Ts[i]//2:] if Ts[i] >= 60 else None for i, t in enumerate(ts_all)], chunksize=16)
A = np.array([x if x is not None else np.full(8, np.nan) for x in th1])
B = np.array([x if x is not None else np.full(8, np.nan) for x in th2])
m = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
comps = []
for j in range(8):
    r = np.corrcoef(A[m, j], B[m, j])[0, 1]
    comps.append({"comp": j, "sb": float(sb(r))})
sbs = [c["sb"] for c in comps]
results["C1__tce"] = {"mean_sb": float(np.mean(sbs)), "n_ge30": int(np.sum(np.array(sbs) >= 0.30)), "per_comp": comps}

with open(f"{OUT}/gate0_batch2.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
for k, v in results.items():
    if "mean_sb" in v:
        print(f"{k:24s} meanSB={v['mean_sb']:.3f} n_ge30={v['n_ge30']}/{len(v['per_comp'])} max={v['max_sb'] if 'max_sb' in v else ''}")
    else:
        print(f"{k:24s} SB={v['sb']:.3f}")
