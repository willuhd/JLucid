#!/usr/bin/env python3
"""
PILOT 4 — VERIFY the lam1 discrepancy (exp/36 Fact B says rho=-0.06; pilot2 says SB~0.6).
Reliability ONLY; no target touched. Pre-registered comparison:
  kernel:  c5K60 (exact cache: f=.05, 5 cycles, cap 60s -> truncated at 1.76 sigma)
           c3nat (f=.05, 3 cycles, cap 90s -> natural, no truncation, L=31TR)
  method:  A = lam1 series computed on FULL run, then series split into halves
           B = timeseries split into halves, phases recomputed per half (exp/36's likely method)
  feature: per-subject-half mean of lam1/N, and std.
If Method A on c5K60 reproduces ~-0.06, Fact B is kernel-truncation-specific and the
natural-kernel lam1 is LIVE. If not, my computation differs from exp/36 in some other way.
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
TR = 2.0

def kernel_len(TR, f, cycles, cap):
    sigma = cycles/(2*np.pi*f)
    L = int(6*sigma/TR); L3 = int(3.0/f/TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap/TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap-1
    if L > Lcap: L = Lcap
    return L

def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L//2) * TR
    w = np.pi**-0.25*np.exp(1j*2*np.pi*f*t)*np.exp(-t**2/(2*(cycles/(2*np.pi*f))**2))
    return w/np.sqrt(np.sum(np.abs(w)**2)+1e-12), L

def phases_of(ts, w, L, sl=None):
    if sl is not None:
        ts = ts[sl]
    T, N = ts.shape
    n_fft = 1
    while n_fft < T+L: n_fft <<= 1
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    conv = np.fft.ifft(Fd*Fk[:, None], axis=0)[L//2:L//2+T]
    return np.angle(conv)

def lam1_series(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2+4*b*b, 0.0))
    return ((a+d+disc)/2.0)/ph.shape[1]

def worker(args):
    ts, f, cycles, cap, mode = args
    w, L = morlet(TR, f, cycles, cap)
    T = ts.shape[0]
    if mode == "A":
        lam = lam1_series(phases_of(ts, w, L))
        h = T//2
        return (float(lam[:h].mean()), float(lam[h:].mean()),
                float(lam[:h].std()), float(lam[h:].std()))
    else:  # B: split the timeseries first
        h = T//2
        l1 = lam1_series(phases_of(ts, w, L, slice(0, h)))
        l2 = lam1_series(phases_of(ts, w, L, slice(h, T)))
        return (float(l1.mean()), float(l2.mean()), float(l1.std()), float(l2.std()))

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1

ts_all, _, meta = load_cc200(qc_only=True)
sites = meta["sites"]
results = {}
GEOMS = [("c5K60", 0.05, 5, 60.0), ("c3nat", 0.05, 3, 90.0), ("c1nat", 0.05, 1, 90.0)]
for gname, f, cyc, cap in GEOMS:
    for mode in ["A", "B"]:
        t0 = time.time()
        with mp.Pool(4) as pool:
            outs = pool.map(worker, [(ts_all[i], f, cyc, cap, mode) for i in range(len(ts_all))], chunksize=16)
        M = np.array([(o[0], o[1]) for o in outs])   # halves' means
        S = np.array([(o[2], o[3]) for o in outs])  # halves' stds
        r_mean = float(np.corrcoef(M[:, 0], M[:, 1])[0, 1])
        r_std = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
        r_mean_sr = float(np.corrcoef(site_resid(M[:, 0], sites), site_resid(M[:, 1], sites))[0, 1])
        key = f"{gname}_method{mode}"
        results[key] = {"r_mean": r_mean, "sb_mean": sb(r_mean), "sb_mean_siteres": sb(r_mean_sr),
                        "r_std": r_std, "sb_std": sb(r_std),
                        "mean_level": float(M.mean())}
        print(f"{key}: lam1.mean split-half r={r_mean:+.3f} SB={sb(r_mean):.3f} (siteres {sb(r_mean_sr):.3f}) | "
              f"lam1.std r={r_std:+.3f} SB={sb(r_std):.3f} | level={M.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

# Also: global-coherence via amplitude envelope (for context on the c5K60 cache lam1)
with open(f"{BASE}/results/pilot_geometry/pilot4_results.json", "w") as fjson:
    json.dump(results, fjson, indent=2, default=float)
print("DONE", flush=True)
