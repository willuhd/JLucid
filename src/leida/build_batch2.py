#!/usr/bin/env python3
"""
exp/37 BATCH 2 — VARIANCE-family phase dynamics + TCE controllability features.
PREREGISTERED in SIEVE_TABLE.md BEFORE any target fit. NO target is touched here.

Features per subject (all computed from timeseries; subject order = load_cc200):
  W1 netphase_var28 (per geometry g in {g050c5K60, g080c3nat}): Var_t[wrap(Θ_a − Θ_b)]
     where Θ_k(t) = network-mean phase = angle( Σ_{i∈net_k} exp(i θ_i(t)) ). 28 comps.
  W2 netplv_std28: std_t[ cos(Θ_a − Θ_b) ] — PLV fluctuation. 28 comps.
  V1 lam1_std: std_t λ1(t)/N (Farinha VAR). scalar.
  V2 meta_std: std_t r(t) over network-level Kuramoto order parameter r(t) =
     |Σ_k exp(iΘ_k)|/7. scalar.
  C1 tce7+asym (controllability agent rec #1): per-TR dominant network s(t)=argmax_k
     mean(z-BOLD, net k); state vectors x^(k) = mean BOLD over epochs where k dominant
     (fallback: global mean if never dominant); A = paper-6 FC system matrix
     A=|FC|/(1+λmax(|FC|))-I; transition energy E(k→k') via Gramian W=Σ A^τ A^τᵀ
     (T=1 horizon: E = (x'-x)ᵀ W⁺ (x'-x)); TCE_i = mean over ordered pairs of per-node
     energy contribution; network means (7) + hierarchy asymmetry log-ratio (1).
Output: batch2_features.npz
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
TR = 2.0
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]
GEOMS = {"g050c5K60": (0.05, 5, 60.0), "g080c3nat": (0.08, 3, 90.0)}

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

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def lam_series(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    return (tr + disc) / 2.0, (tr - disc) / 2.0

t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, _, meta = load_cc200(qc_only=True)
sites = np.array(meta["sites"]); Ts = [t.shape[0] for t in ts_all]
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]
print(f"loaded {len(ts_all)} ({time.time()-t0:.0f}s)", flush=True)

def worker_phases(args):
    ts, f, cyc, cap = args
    w, L = morlet(TR, f, cyc, cap)
    ph = phases_of(ts, w, L)
    # network-level phase: angle of resultant
    TH = np.zeros((ph.shape[0], 7))
    for k in range(7):
        z = np.exp(1j * ph[:, net_idx[k]]).sum(axis=1)
        TH[:, k] = np.angle(z)
    l1, l2 = lam_series(ph)
    return TH, l1 / ph.shape[1]

GEOM = {}
for gname, (f, cyc, cap) in GEOMS.items():
    t1 = time.time()
    with mp.Pool(4) as pool:
        outs = pool.map(worker_phases, [(ts_all[i], f, cyc, cap) for i in range(len(ts_all))], chunksize=16)
    print(f"{gname} phases done ({time.time()-t1:.0f}s)", flush=True)
    n = len(ts_all)
    W1 = np.full((n, 28), np.nan); W2 = np.full((n, 28), np.nan)
    V1 = np.full(n, np.nan); V2 = np.full(n, np.nan)
    for i, (TH, l1s) in enumerate(outs):
        T = Ts[i]
        if T < 60:
            continue
        dphase = np.zeros((T, 28))
        plv = np.zeros((T, 28))
        for kk, (a, b) in enumerate(UP):
            d = wrap(TH[:, a] - TH[:, b])
            dphase[:, kk] = d
            plv[:, kk] = np.cos(d)
        # phase-difference variance needs unwrapping per contiguous segment; Wang 2018 used
        # the variance of instantaneous phase differences (raw, not circular). Use both:
        W1[i] = dphase.var(axis=0)                       # raw variance (Wang form)
        W2[i] = plv.std(axis=0)                          # PLV fluctuation
        V1[i] = float(l1s.std())
        r_t = np.abs(np.exp(1j * TH).sum(axis=1)) / 7.0
        V2[i] = float(r_t.std())
    GEOM[gname] = {"W1": W1, "W2": W2, "V1": V1[:, None], "V2": V2[:, None]}
    print(f"  {gname} W1/W2/V1/V2 built ({time.time()-t1:.0f}s)", flush=True)

# ---- C1: TCE features (controllability) ----
print("C1 TCE ...", flush=True)
def worker_tce(ts):
    T, N = ts.shape
    if T < 60:
        return None
    Xz = (ts - ts.mean(0)) / (ts.std(0) + 1e-12)
    FC = np.nan_to_num(np.corrcoef(Xz.T))
    Araw = np.abs(FC)
    lmax = float(np.max(np.linalg.eigvalsh((Araw + Araw.T) / 2)))
    A = Araw / (1.0 + lmax + 1e-12) - np.eye(N)
    A = (A + A.T) / 2
    # dominant network per TR
    net_z = np.zeros((T, 7))
    for k in range(7):
        net_z[:, k] = Xz[:, net_idx[k]].mean(axis=1)
    dom = net_z.argmax(axis=1)
    # state vectors
    X = np.zeros((7, N))
    for k in range(7):
        m = dom == k
        X[k] = Xz[m].mean(axis=0) if m.sum() >= 3 else Xz.mean(axis=0)
    # Gramian T=1: W = Σ_{τ=0}^{1} A^τ A^τᵀ = I + A Aᵀ  (A symmetric -> I + A²)
    W = np.eye(N) + A @ A
    Winv = np.linalg.pinv(W + 1e-9 * np.eye(N))
    # transition energies for all ordered pairs
    E = np.zeros((7, 7, N))
    for a in range(7):
        for b in range(7):
            if a == b: continue
            d = X[b] - X[a]
            # full-control energy e = dᵀ W⁻¹ d; per-node contribution e_i = d_i (W⁻¹ d)_i
            wd = Winv @ d
            E[a, b] = d * wd
    # TCE_i = mean over ordered pairs
    tce_node = E.sum(axis=(0, 1)) / 42.0
    tce7 = np.array([tce_node[net_idx[k]].mean() for k in range(7)])
    # hierarchy: unimodal {VIS,SOM}=0,1; heteromodal {DAN,SAL,LIM,FPN,DMN}=2..6
    uni = [(0,1)]
    het = [(a,b) for a in range(2,7) for b in range(a+1,7)]
    # between: (uni, het) pairs
    bet = [(a,b) for a in range(2) for b in range(2,7)]
    e_uni = np.mean([E[a,b].sum() for a,b in uni])
    e_het = np.mean([E[a,b].sum() for a,b in het])
    e_bet = np.mean([E[a,b].sum() for a,b in bet])
    asym = np.log((e_bet + 1e-12) / (e_het + 1e-12))
    return np.concatenate([tce7, [asym], [e_uni]])

with mp.Pool(4) as pool:
    outs = pool.map(worker_tce, ts_all, chunksize=16)
C1 = np.array([o if o is not None else np.full(9, np.nan) for o in outs])
print(f"  C1 done ({time.time()-t0:.0f}s)", flush=True)

save = {f"{g}__{v}": F for g, d in GEOM.items() for v, F in d.items()}
save["C1__tce"] = C1
np.savez_compressed(f"{OUT}/batch2_features.npz", sites=sites, Ts=np.array(Ts), **save)
print(f"DONE {time.time()-t0:.0f}s -> batch2_features.npz", flush=True)
