#!/usr/bin/env python3
"""
PILOT 6b (Direction 3 addendum; reliability-only). Pre-committed BEFORE running:
fused 2-band co-leadership strength — does the pilot-1 occupancy-fusion trick
(0.05 c5K60 + 0.08 c3nat, z-fused, SB 0.48 vs 0.36 single-band) transfer to the
co-leadership second moment L?

  bands: b1 = EXACT cache geometry (f=0.05, 5cyc, cap 60s -> L=29 TR)
         b2 = f=0.08, 3cyc, cap 90s (pilot-1's best single band, L=19 TR)
  features: per subject-half, per band: s_i = rowmean(L_b), net-mean -> 7 comps,
            z-scored across subjects, averaged (no Hungarian needed: L-summary
            components are network-labeled, shared across bands).
  report: single-band SB (repro of pilot6), fused SB. NO target touched.
"""
import sys, json, time, warnings
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
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
BANDS = {"b1_cache": (0.05, 5, 60.0), "b2_080": (0.08, 3, 90.0)}

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

def v1_vec(phases):
    c = np.cos(phases); s = np.sin(phases)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    lam1 = (a + d + disc) / 2.0
    u0 = b; u1 = lam1 - a
    norm = np.sqrt(u0 * u0 + u1 * u1)
    degen = norm < 1e-12
    u0 = np.where(degen, 1.0, u0 / np.where(degen, 1.0, norm))
    u1 = np.where(degen, 0.0, u1 / np.where(degen, 1.0, norm))
    V1 = c * u0[:, None] + s * u1[:, None]
    V1 /= (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-12)
    flip = (V1 > 0).sum(1) > 0.5 * V1.shape[1]
    V1[flip] = -V1[flip]
    return V1.astype(np.float32)

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2 * r / (1 + abs(r)) if r > -1 else -1

t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, _, meta = load_cc200(qc_only=True)
sites = meta["sites"]; Ts = [t.shape[0] for t in ts_all]
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]
print(f"loaded {len(ts_all)} in {time.time()-t0:.0f}s", flush=True)

def worker(args):
    ts, f, cycles, cap = args
    w, L = morlet(TR, f, cycles, cap)
    return v1_vec(phases_of(ts, w, L))

results = {}
feat = {}
for bname, (f, cyc, cap) in BANDS.items():
    t1 = time.time()
    with mp.Pool(4) as pool:
        outs = pool.map(worker, [(ts_all[i], f, cyc, cap) for i in range(len(ts_all))], chunksize=16)
    print(f"{bname}: V1 done {time.time()-t1:.0f}s", flush=True)
    h1 = []; h2 = []
    for V1s, T in zip(outs, Ts):
        h = T // 2
        if T >= 60:
            L1 = (V1s[:h].T @ V1s[:h]) / h
            L2m = (V1s[h:].T @ V1s[h:]) / (T - h)
            s1 = L1.mean(axis=1); s2 = L2m.mean(axis=1)
            h1.append([s1[ix].mean() for ix in net_idx])
            h2.append([s2[ix].mean() for ix in net_idx])
    h1 = np.array(h1); h2 = np.array(h2)
    feat[bname] = (h1, h2)
    sv = sites[np.array([T >= 60 for T in Ts])]
    per = []
    for j in range(7):
        r = float(np.corrcoef(h1[:, j], h2[:, j])[0, 1])
        rres = float(np.corrcoef(site_resid(h1[:, j], sv), site_resid(h2[:, j], sv))[0, 1])
        per.append({"net": NETS[j], "r": r, "sb": sb(r), "sb_siteres": sb(rres)})
    results[bname] = {"per_net": per, "mean_sb": float(np.mean([p["sb"] for p in per])),
                      "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per]))}
    print(f"  {bname} meanSB={results[bname]['mean_sb']:.3f} siteres={results[bname]['mean_sb_siteres']:.3f} "
          f"per-net {[round(p['sb'],2) for p in per]}", flush=True)

# fused: z-score each band's halves across subjects, average
def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)
fused1 = 0.5 * (z(feat["b1_cache"][0]) + z(feat["b2_080"][0]))
fused2 = 0.5 * (z(feat["b1_cache"][1]) + z(feat["b2_080"][1]))
per = []
for j in range(7):
    r = float(np.corrcoef(fused1[:, j], fused2[:, j])[0, 1])
    rres = float(np.corrcoef(site_resid(fused1[:, j], sv), site_resid(fused2[:, j], sv))[0, 1])
    per.append({"net": NETS[j], "r": r, "sb": sb(r), "sb_siteres": sb(rres)})
results["FUSED_2band_Lstr7"] = {"per_net": per, "mean_sb": float(np.mean([p["sb"] for p in per])),
                                "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per]))}
print(f"  FUSED meanSB={results['FUSED_2band_Lstr7']['mean_sb']:.3f} "
      f"siteres={results['FUSED_2band_Lstr7']['mean_sb_siteres']:.3f} per-net {[round(p['sb'],2) for p in per]}", flush=True)

# also fused full 28-block for reference (block-labeled, band-shared)
h1a, h2a = feat["b1_cache"]; h1b, h2b = feat["b2_080"]
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
with open(f"{OUT}/pilot6b_d3_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"-> pilot6b_d3_results.json", flush=True)
