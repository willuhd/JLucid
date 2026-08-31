#!/usr/bin/env python3
"""
PILOT 7c (Direction 5, D5-A; reliability-only Gate-0 preview). NO target touched.
Pre-committed BEFORE running:
  Fused 3-band occupancy (pilot-2 recipe VERBATIM: bands f in {0.05, 0.07, 0.09}, 3 cycles,
  cap 90s; per-band KMeans k=5 n_init=10 rs=0 on <=40k rng(0) frames; predict all;
  per-band occ halves; Hungarian-align band 2/3 to band-1 by |between-subject occ corr|;
  z-score each component across subjects within each half; average bands)
  x (age_z), in two parameterizations:
    RAW      : occ_fused * age_z (recorded; contains the trivial mean_occ x age term)
    CENTERED : (occ_fused - cohort_mean) * age_z   <- the honest Gate-0 candidate
  Anchors: single-band locked-dict centered interaction (pilot7: 0.290),
  per-band occ SB, fused occ SB (expected ~0.505 per pilot-2).
Age is used ONLY as a feature multiplier; chronological, reliability = 1.0.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
N_SUB_KM = 40000
BANDS = {"b050": (0.05, 3, 90.0), "b070": (0.07, 3, 90.0), "b090": (0.09, 3, 90.0)}

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

def wavelet_phase(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    return np.angle(np.fft.ifft(Fd * Fk[:, None], axis=0)[L // 2:L // 2 + T])

def v1_lam1(phases):
    c = np.cos(phases); s = np.sin(phases)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    lam1 = (a + d + disc) / 2
    u0 = b; u1 = lam1 - a
    norm = np.sqrt(u0 * u0 + u1 * u1); degen = norm < 1e-12
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

def vec_stats(name, F1, F2, sites, mask, results):
    n, d = F1.shape
    per = []
    for j in range(d):
        x1, x2 = F1[mask, j], F2[mask, j]
        ok = np.isfinite(x1) & np.isfinite(x2)
        entry = {"comp": j, "n_used": int(ok.sum())}
        if ok.sum() >= 50 and np.std(x1[ok]) > 1e-12 and np.std(x2[ok]) > 1e-12:
            r = float(np.corrcoef(x1[ok], x2[ok])[0, 1])
            rr = float(np.corrcoef(site_resid(x1[ok], sites[mask][ok]), site_resid(x2[ok], sites[mask][ok]))[0, 1])
            entry.update({"r": r, "sb": sb(r), "sb_siteres": sb(rr)})
        per.append(entry)
    sbv = [p["sb"] for p in per if "sb" in p]
    results[name] = {"n_comp": len(sbv), "comps": d,
                     "mean_sb": float(np.mean(sbv)) if sbv else None,
                     "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per if "sb_siteres" in p])) if sbv else None,
                     "n_ge030": int(sum(1 for v in sbv if v >= 0.30)),
                     "per_comp": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in per]}
    if sbv:
        print(f"  {name:22s} d={d:3d} meanSB={results[name]['mean_sb']:.3f} "
              f"siteres={results[name]['mean_sb_siteres']:.3f} >=0.30: {results[name]['n_ge030']}/{len(sbv)} "
              f"per={[round(p['sb'],2) for p in per if 'sb' in p]}", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    print("loading ...", flush=True)
    ts_all, _, meta = load_cc200(qc_only=True)
    sites = np.array(meta["sites"]); ages = np.array(meta["ages"], float)
    ages = np.nan_to_num(ages, nan=np.nanmean(ages))
    Ts = [t.shape[0] for t in ts_all]
    nS = len(ts_all)
    mask60 = np.array([T >= 60 for T in Ts])
    age_z = (ages - ages.mean()) / (ages.std() + 1e-12)

    results = {}
    occ_h = {}  # band -> (of, o1, o2)
    for name, (f, c, cap) in BANDS.items():
        t1 = time.time()
        w, L = morlet(TR, f, c, cap)
        V1s = [v1_lam1(wavelet_phase(np.asarray(ts, float), w, L)) for ts in ts_all]
        Vall = np.concatenate(V1s, 0)
        rng_ = np.random.default_rng(0)
        idx = rng_.choice(Vall.shape[0], min(N_SUB_KM, Vall.shape[0]), replace=False)
        km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(Vall[idx])
        lab = km.predict(Vall).astype(np.int16)
        of, o1, o2 = [], [], []
        pos = 0
        for T in Ts:
            l = lab[pos:pos + T]; pos += T
            of.append(np.bincount(l, minlength=K) / max(T, 1))
            o1.append(np.bincount(l[:T // 2], minlength=K) / max(T // 2, 1))
            o2.append(np.bincount(l[T // 2:], minlength=K) / max(T - T // 2, 1))
        occ_h[name] = (np.array(of), np.array(o1), np.array(o2))
        vec_stats(f"occ_{name}", occ_h[name][1], occ_h[name][2], sites, mask60, results)
        print(f"    band {name} built in {time.time()-t1:.0f}s", flush=True)

    # pilot-2 fusion verbatim: Hungarian to band-1 on FULL occ corr, z-fuse halves
    ref = "b050"
    A = occ_h[ref][0]
    perms = {ref: np.arange(K)}
    for n_ in ["b070", "b090"]:
        B = occ_h[n_][0]
        C = np.array([[abs(float(np.corrcoef(A[:, i], B[:, j])[0, 1])) for j in range(K)] for i in range(K)])
        C = np.nan_to_num(C)
        ri, ci = linear_sum_assignment(-C)
        perms[n_] = ci
    def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)
    def fused(part):
        acc_f = z(occ_h[ref][part]); acc_1 = z(occ_h[ref][part - 1]) if part >= 1 else None
        # parts: 0=full,1=h1,2=h2
        idx_map = {0: 0, 1: 1, 2: 2}
        accs = {p: z(occ_h[ref][p]) for p in [0, 1, 2]}
        for n_ in ["b070", "b090"]:
            p = perms[n_]
            for q in [0, 1, 2]:
                accs[q] = accs[q] + z(occ_h[n_][q])[:, p]
        return accs[0] / 3, accs[1] / 3, accs[2] / 3
    ofF, o1F, o2F = fused(0)
    vec_stats("occFUSED_3band", o1F, o2F, sites, mask60, results)

    # D5-A on the fused basis
    omF = ofF.mean(0)
    C1 = (o1F - omF) * age_z[:, None]; C2 = (o2F - omF) * age_z[:, None]
    vec_stats("CxA_D5A_fused_centered", C1, C2, sites, mask60, results)
    R1 = o1F * age_z[:, None]; R2 = o2F * age_z[:, None]
    vec_stats("AxA_D5A_fused_raw", R1, R2, sites, mask60, results)

    results["wall_secs"] = int(time.time() - t0)
    with open(f"{OUT}/pilot7c_d5_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> pilot7c_d5_results.json", flush=True)
