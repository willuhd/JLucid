#!/usr/bin/env python3
"""Gate 0 for K1/K2 (FC-based, halves) — fixed None guard."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass
BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results"
sys.path.insert(0, f"{BASE}/src")
from controllability.datasets_cc200 import load_cc200
import json as _json
yeo = _json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
net_idx = [np.array(yeo[n]) for n in NETS]

def worker_k_half(ts):
    if ts is None or len(ts) < 30: return None
    T, Nn = ts.shape
    Xz = (ts - ts.mean(0)) / (ts.std(0) + 1e-12)
    FC = np.nan_to_num(np.corrcoef(Xz.T))
    lam = np.linalg.eigvalsh((FC + FC.T)/2)
    A_s = FC / (1.0 + float(lam.max()) + 1e-12) - np.eye(Nn)
    ev = np.linalg.eigvalsh((A_s + A_s.T)/2)
    if ev.max() >= 0:
        A_s = A_s - (ev.max() + 1e-6) * np.eye(Nn)
    vals, vecs = np.linalg.eigh((A_s + A_s.T)/2)
    phi = (vecs**2) @ (1.0/(-2.0*vals))
    mu = (vecs**2) @ (1.0 - np.exp(vals))
    phi7 = np.array([phi[net_idx[k]].mean() for k in range(7)])
    mu7 = np.array([mu[net_idx[k]].mean() for k in range(7)])
    # K2 |A|-form intentionally NOT computed: documented discrete-UNSTABLE
    # (sieve-table batch 5 invalidation note; run_gate1_batch5.py DROPPED it,
    # and the discrete Lyapunov need not exist — LAPACK builds differ between
    # inf-filled W and LinAlgError). Gate 0 records K1__signed only.
    return np.concatenate([phi7, mu7]), None

ts_all, _, meta = load_cc200(qc_only=True)
Ts = [t.shape[0] for t in ts_all]
with mp.Pool(4) as pool:
    kh1 = pool.map(worker_k_half, [t[:Ts[i]//2] if Ts[i]>=60 else None for i, t in enumerate(ts_all)], chunksize=8)
    kh2 = pool.map(worker_k_half, [t[Ts[i]//2:] if Ts[i]>=60 else None for i, t in enumerate(ts_all)], chunksize=8)
def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1
res = json.load(open(f"{OUT}/gate0_batch5.json")) if os.path.exists(f"{OUT}/gate0_batch5.json") else {}
for fam, ci in [("K1__signed", 0)]:
    A = np.array([x[ci] if x else np.full(14, np.nan) for x in kh1])
    B = np.array([x[ci] if x else np.full(14, np.nan) for x in kh2])
    m = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    comps = [float(sb(np.corrcoef(A[m, j], B[m, j])[0,1])) for j in range(14)]
    res[fam] = {"mean_sb": float(np.mean(comps)), "n_ge30": int(np.sum(np.array(comps)>=0.30)), "per_comp_sb": comps}
    print(f"{fam:10s} meanSB={res[fam]['mean_sb']:.3f} n_ge30={res[fam]['n_ge30']}/14")
with open(f"{OUT}/gate0_batch5.json", "w") as f:
    json.dump(res, f, indent=2)
print("saved")
