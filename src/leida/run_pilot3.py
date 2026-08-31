#!/usr/bin/env python3
"""
PILOT 3 (reliability-only, Gate-0 preview): Hilbert-after-narrowband-filter, FIXED
segmentation (per-subject trimmed lengths respected). Two pre-committed bandwidths:
  hilb_W  0.040-0.100 Hz (BW=0.060) - wide, coherence time ~1/BW = 16.7s
  hilb_N  0.060-0.095 Hz (BW=0.035) - narrow, coherence time ~28.6s (prior point)
Edge trim = 1.5/BW seconds (filter transient), capped at T/6. NO target touched.
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
from sklearn.cluster import KMeans
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
TR = 2.0
K = 5
N_SUB_KM = 40000
BANDS = [("hilbW_040_100", 0.040, 0.100), ("hilbN_060_095", 0.060, 0.095)]

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1

def worker(args):
    ts, lo, hi = args
    T = ts.shape[0]
    sos = butter(3, [lo, hi], btype="band", fs=1.0/TR, output="sos")
    y = sosfiltfilt(sos, ts, axis=0)
    ph = np.angle(hilbert(y, axis=0))
    edge = int(min(np.ceil(1.5/(hi-lo)/TR), max(T//6, 1)))
    edge = max(edge, 2)
    if T - 2*edge < 30:
        return None  # unusable subject at this bandwidth
    ph = ph[edge:T-edge]
    # V1 closed form
    c = np.cos(ph); s = np.sin(ph)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2+4*b*b, 0.0))
    lam1 = (a+d+disc)/2.0
    u0 = b; u1 = lam1-a
    norm = np.sqrt(u0*u0+u1*u1); degen = norm < 1e-12
    u0 = np.where(degen, 1.0, u0/np.where(degen, 1.0, norm))
    u1 = np.where(degen, 0.0, u1/np.where(degen, 1.0, norm))
    V1 = c*u0[:, None]+s*u1[:, None]
    V1 /= (np.linalg.norm(V1, axis=1, keepdims=True)+1e-12)
    flip = (V1 > 0).sum(1) > 0.5*V1.shape[1]
    V1[flip] = -V1[flip]
    return V1.astype(np.float32)

t0 = time.time()
ts_all, _, meta = load_cc200(qc_only=True)
sites = meta["sites"]
results = {}
for name, lo, hi in BANDS:
    t1 = time.time()
    with mp.Pool(4) as pool:
        outs = pool.map(worker, [(ts_all[i], lo, hi) for i in range(len(ts_all))], chunksize=16)
    keep = [i for i, o in enumerate(outs) if o is not None]
    n_drop = len(ts_all) - len(keep)
    V_all = np.concatenate([outs[i] for i in keep], 0)
    lens = [outs[i].shape[0] for i in keep]
    rng = np.random.default_rng(0)
    idx = rng.choice(V_all.shape[0], min(N_SUB_KM, V_all.shape[0]), replace=False)
    km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(V_all[idx])
    lab = km.predict(V_all).astype(np.int16)
    labs_list, pos = [], 0
    for T in lens:
        labs_list.append(lab[pos:pos+T]); pos += T
    of, o1, o2 = [], [], []
    for l in labs_list:
        T = len(l)
        of.append(np.bincount(l, minlength=K)/max(T, 1))
        o1.append(np.bincount(l[:T//2], minlength=K)/max(T//2, 1))
        o2.append(np.bincount(l[T//2:], minlength=K)/max(T-T//2, 1))
    of, o1, o2 = np.array(of), np.array(o1), np.array(o2)
    sk = sites[keep]
    per = []
    for j in range(K):
        x1, x2 = o1[:, j], o2[:, j]
        if np.std(x1) < 1e-9 or np.std(x2) < 1e-9: continue
        r = float(np.corrcoef(x1, x2)[0, 1])
        rr = float(np.corrcoef(site_resid(x1, sk), site_resid(x2, sk))[0, 1])
        per.append({"state": j, "sb": sb(r), "sb_siteres": sb(rr)})
    msb = float(np.mean([p["sb"] for p in per])); msr = float(np.mean([p["sb_siteres"] for p in per]))
    results[name] = {"n_kept": len(keep), "n_dropped_too_short": n_drop,
                     "mean_sb": msb, "mean_sb_siteres": msr, "per_state": per,
                     "bw": hi-lo, "coherence_time_s": 1.0/(hi-lo)}
    print(f"{name}: kept {len(keep)}/{len(ts_all)} (dropped {n_drop}) mean SB={msb:.3f} "
          f"siteres={msr:.3f} per-state {[round(p['sb'],3) for p in per]} ({time.time()-t1:.0f}s)", flush=True)

with open(f"{BASE}/results/pilot_geometry/pilot3_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
