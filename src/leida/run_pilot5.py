#!/usr/bin/env python3
"""
PILOT 5 — reconcile the lam1 reliability conflict (exp/36 Fact B: r=-0.06 vs pilot4: r=+0.35).
Reliability ONLY; no target. Test alternative split/extractor methods that could have
produced -0.06, on the exact cache geometry (f=.05, 5 cycles, cap 60s -> L=29TR truncated):
  M-B  timeseries split, phases recomputed (pilot4 B) [reproduce]
  M-C  odd/even timepoint split, phases recomputed per half
  M-D  interleaved halves of the full-run lam1 series (odd/even windows, no recompute)
  M-E  Hilbert lam1: Butterworth 0.04-0.10 + analytic phase, timeseries split
  M-F  Hilbert lam1, odd/even series windows
  M-G  Pearson-FC windowed "leader coherence" (sliding-window FC leading eigenvalue, W=30TR)
Also: lam1 mean with 1st/2nd half AFTER site z-score, and Spearman instead of Pearson.
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
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr
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
    t = (np.arange(L)-L//2)*TR
    w = np.pi**-0.25*np.exp(1j*2*np.pi*f*t)*np.exp(-t**2/(2*(cycles/(2*np.pi*f))**2))
    return w/np.sqrt(np.sum(np.abs(w)**2)+1e-12), L

def phases_of(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T+L: n_fft <<= 1
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    return np.angle(np.fft.ifft(Fd*Fk[:, None], axis=0)[L//2:L//2+T])

def lam1_series(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2+4*b*b, 0.0))
    return ((a+d+disc)/2.0)/ph.shape[1]

def fc_leader(ts, W=30):
    # sliding-window Pearson FC leading eigenvalue (normalized by N), step 1
    T, N = ts.shape
    out = np.zeros(T)
    X = ts - ts.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-12
    Xn = X/sd
    for t in range(T):
        lo = max(0, t-W//2); hi = min(T, lo+W); lo = hi-W if hi == T else lo
        if hi-lo < W//2: out[t] = np.nan; continue
        C = np.corrcoef(ts[lo:hi].T)
        C = np.nan_to_num(C)
        vals = np.linalg.eigvalsh(C)
        out[t] = vals[-1]/N
    return out

def worker(args):
    ts, mode = args
    out = {}
    w, L = morlet(TR, 0.05, 5, 60.0)   # EXACT cache geometry
    T = ts.shape[0]
    h = T//2
    # M-B wavelet, ts split
    l1 = lam1_series(phases_of(ts[:h], w, L)); l2 = lam1_series(phases_of(ts[h:], w, L))
    out["MB"] = (l1.mean(), l2.mean(), l1.std(), l2.std())
    # M-C odd/even timepoints, phases recomputed
    o1 = lam1_series(phases_of(ts[0::2], w, L)); o2 = lam1_series(phases_of(ts[1::2], w, L))
    out["MC"] = (o1.mean(), o2.mean(), o1.std(), o2.std())
    # M-D interleaved windows of full-run lam1 series
    lam = lam1_series(phases_of(ts, w, L))
    out["MD"] = (lam[0::2].mean(), lam[1::2].mean(), lam[0::2].std(), lam[1::2].std())
    # M-E Hilbert wideband, ts split
    sos = butter(3, [0.04, 0.10], btype="band", fs=1.0/TR, output="sos")
    y = sosfiltfilt(sos, ts, axis=0)
    phh = np.angle(hilbert(y, axis=0))
    g1 = lam1_series(phh[:h]); g2 = lam1_series(phh[h:])
    out["ME"] = (g1.mean(), g2.mean(), g1.std(), g2.std())
    # M-F Hilbert interleaved windows
    lamh = lam1_series(phh)
    out["MF"] = (lamh[0::2].mean(), lamh[1::2].mean(), lamh[0::2].std(), lamh[1::2].std())
    # M-G sliding-window FC leading eigenvalue
    f1 = fc_leader(ts[:h]); f2 = fc_leader(ts[h:])
    out["MG"] = (np.nanmean(f1), np.nanmean(f2), np.nanstd(f1), np.nanstd(f2))
    return out

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1

ts_all, _, meta = load_cc200(qc_only=True)
sites = meta["sites"]
t0 = time.time()
with mp.Pool(4) as pool:
    outs = pool.map(worker, [(ts_all[i], "x") for i in range(len(ts_all))], chunksize=16)
print(f"computed in {time.time()-t0:.0f}s", flush=True)

results = {}
for m in ["MB", "MC", "MD", "ME", "MF", "MG"]:
    A = np.array([(o[m][0], o[m][1]) for o in outs])
    S = np.array([(o[m][2], o[m][3]) for o in outs])
    rP = float(np.corrcoef(A[:, 0], A[:, 1])[0, 1])
    rS = float(spearmanr(A[:, 0], A[:, 1]).statistic)
    rSr = float(np.corrcoef(site_resid(A[:, 0], sites), site_resid(A[:, 1], sites))[0, 1])
    rstd = float(np.corrcoef(S[:, 0], S[:, 1])[0, 1])
    results[m] = {"r_mean_pearson": rP, "sb_mean": sb(rP), "r_mean_spearman": rS,
                  "sb_mean_siteres": sb(rSr), "r_std": rstd, "sb_std": sb(rstd)}
    print(f"{m}: mean r={rP:+.3f} (SB {sb(rP):+.3f}) spearman={rS:+.3f} siteres SB={sb(rSr):+.3f} | std r={rstd:+.3f} (SB {sb(rstd):+.3f})", flush=True)

with open(f"{BASE}/results/pilot_geometry/pilot5_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("DONE", flush=True)
