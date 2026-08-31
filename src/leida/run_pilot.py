#!/usr/bin/env python3
"""
PILOT (reliability-only preview of Gate 0) for Direction-1 extractor geometry.
NO target is ever touched here (no Inatt/Hyper/IQ) - reliability only, per exp/37 PREREG.
Pre-committed grid below; every point reported. These are Gate-0 PREVIEWS, to be
re-run formally by the sieve; any survivor must carry the whole grid into the Gate-2 family.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)  # macOS default is spawn; workers are numpy-only
except Exception:
    pass
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
N_SUB_KM = 40000

# ---------------- geometry definitions (pre-committed) ----------------
# (name, freq, cycles, cap_secs)
WAVELETS = [
    ("g050c5K60", 0.05, 5, 60.0),   # EXACT current cache design (repro of exp/36 Fact A)
    ("g050c3nat", 0.05, 3, 90.0),   # envelope fix, same band, natural 3-cycle (57s)
    ("g060c5nat", 0.06, 5, 90.0),   # mild freq step, full 5-cycle (80s)
    ("g070c4nat", 0.07, 4, 90.0),   # 4-cycle at 57s support
    ("g080c3nat", 0.08, 3, 90.0),   # natural 3-cycle 37.5s
    ("g090c3nat", 0.09, 3, 90.0),   # 33.3s
    ("g100c3nat", 0.10, 3, 90.0),   # 30s, band-edge risk
]
# Hilbert transform-level point (axis closer): butter bandpass + analytic phase
HILB = ("hilb060_095", 0.060, 0.095)  # (name, lo, hi) Hz

def kernel_len(TR, f, cycles, cap):
    sigma = cycles / (2*np.pi*f)
    L = int(6*sigma/TR); L3 = int(3.0/f/TR); L = max(L, L3)
    if L % 2 == 0: L += 1
    Lcap = int(cap/TR); Lcap = Lcap if Lcap % 2 == 1 else Lcap - 1
    if L > Lcap: L = Lcap
    return L

def morlet(TR, f, cycles, cap):
    L = kernel_len(TR, f, cycles, cap)
    t = (np.arange(L) - L//2) * TR
    w = np.pi**-0.25 * np.exp(1j*2*np.pi*f*t) * np.exp(-t**2/(2*(cycles/(2*np.pi*f))**2))
    return w / np.sqrt(np.sum(np.abs(w)**2) + 1e-12), L

def wavelet_phase(ts, w, L):
    T, N = ts.shape
    n_fft = 1
    while n_fft < T + L: n_fft <<= 1
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    conv = np.fft.ifft(Fd * Fk[:, None], axis=0)[L//2:L//2+T]
    return np.angle(conv)

def v1_vec(phases):
    """exp/28 vectorized closed-form V1 (leading eigenvector of phase coherence 2x2)."""
    c = np.cos(phases); s = np.sin(phases)
    a = (c*c).sum(1); b = (c*s).sum(1); d = (s*s).sum(1)
    disc = np.sqrt(np.maximum((a-d)**2 + 4*b*b, 0.0))
    lam1 = (a + d + disc) / 2.0
    u0 = b; u1 = lam1 - a
    norm = np.sqrt(u0*u0 + u1*u1)
    degen = norm < 1e-12
    u0 = np.where(degen, 1.0, u0/np.where(degen, 1.0, norm))
    u1 = np.where(degen, 0.0, u1/np.where(degen, 1.0, norm))
    V1 = c*u0[:, None] + s*u1[:, None]
    V1 /= (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-12)
    flip = (V1 > 0).sum(1) > 0.5*V1.shape[1]
    V1[flip] = -V1[flip]
    return V1.astype(np.float32)

def worker_wavelet(args):
    ts, f, cycles, cap = args
    w, L = morlet(TR, f, cycles, cap)
    return v1_vec(wavelet_phase(ts, w, L)), L

def worker_hilb(args):
    ts, lo, hi = args
    T = ts.shape[0]
    sos = butter(3, [lo, hi], btype="band", fs=1.0/TR, output="sos")
    y = sosfiltfilt(sos, ts, axis=0)
    ph = np.angle(hilbert(y, axis=0))
    edge = int(min(1.5/(hi-lo) / TR, T/6.0))  # transient trim, capped
    edge = max(edge, 2)
    L_eff = int(1.0/(hi-lo)/TR)  # ~1/BW support proxy
    return v1_vec(ph[edge:T-edge]), L_eff, edge

# ---------------- stats helpers ----------------
def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2*r/(1+abs(r)) if r > -1 else -1

def half_occ_stats(labels_list, L_list, sites, k=K, trim=True, stride=1):
    """labels_list: per-subject label arrays (full, untrimmed). Returns dict of stats."""
    occ_f, occ_1, occ_2, dwell, nplac, Ts = [], [], [], [], [], []
    for lab, L in zip(labels_list, L_list):
        if trim:
            e = max(1, L//2)
            if len(lab) - 2*e < 24: 
                lab_t = lab  # too short to trim; keep raw (flagged via Ts)
                e_use = 0
            else:
                lab_t = lab[e:len(lab)-e]
        else:
            lab_t = lab
        lab_t = lab_t[::stride]
        T = len(lab_t)
        h1 = lab_t[:T//2]; h2 = lab_t[T//2:]
        of = np.bincount(lab_t, minlength=k)/max(T,1)
        o1 = np.bincount(h1, minlength=k)/max(len(h1),1)
        o2 = np.bincount(h2, minlength=k)/max(len(h2),1)
        occ_f.append(of); occ_1.append(o1); occ_2.append(o2)
        # mean run length
        if T > 1:
            chg = np.flatnonzero(np.diff(lab_t) != 0)
            runs = np.diff(np.concatenate([[-1], chg, [T-1]]))
            dwell.append(float(runs.mean()))
        else:
            dwell.append(0.0)
        nplac.append(T/max(L,1))
        Ts.append(T)
    occ_f = np.array(occ_f); occ_1 = np.array(occ_1); occ_2 = np.array(occ_2)
    Ts = np.array(Ts)
    valid = Ts >= 30
    res = {"n_valid": int(valid.sum()), "mean_T": float(Ts[valid].mean()),
           "mean_dwell_TR": float(np.mean(dwell)), "mean_N_placements": float(np.mean(nplac))}
    per_state = []
    for j in range(k):
        x1, x2 = occ_1[valid, j], occ_2[valid, j]
        if np.std(x1) < 1e-9 or np.std(x2) < 1e-9:
            per_state.append({"state": j, "r": np.nan}); continue
        r = float(np.corrcoef(x1, x2)[0, 1])
        r_res = float(np.corrcoef(site_resid(x1, sites[valid]), site_resid(x2, sites[valid]))[0, 1])
        d = x1 - x2
        sw2 = float(np.mean(d*d)/2.0)
        sf = occ_f[valid, j]
        st2 = max(float(np.var(sf)) - sw2/2.0, 1e-9)
        per_state.append({"state": j, "r": r, "sb": sb(r), "r_siteres": r_res, "sb_siteres": sb(r_res),
                          "sigma2_W": sw2, "sigma2_T": st2, "p_bar": float(sf.mean()),
                          "r_pred_anova": st2/(st2+sw2)})
    res["per_state"] = per_state
    ok = [s for s in per_state if "sb" in s]
    if ok:
        res["mean_sb"] = float(np.mean([s["sb"] for s in ok]))
        res["mean_sb_siteres"] = float(np.mean([s["sb_siteres"] for s in ok]))
        res["mean_sigma2_W"] = float(np.mean([s["sigma2_W"] for s in ok]))
        res["mean_sigma2_T"] = float(np.mean([s["sigma2_T"] for s in ok]))
    return res, occ_f, occ_1, occ_2

# ---------------- load data ----------------
t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, labels_all, meta = load_cc200(qc_only=True)
sites = meta["sites"]; Ts_loader = [t.shape[0] for t in ts_all]
print(f"loaded {len(ts_all)} subjects in {time.time()-t0:.0f}s, total_T={sum(Ts_loader)}", flush=True)

results = {}
occ_store = {}

import multiprocessing as mp

# ---------------- wavelet geometries ----------------
for name, f, cycles, cap in WAVELETS:
    t1 = time.time()
    L0 = kernel_len(TR, f, cycles, cap)
    print(f"\n=== {name} (f={f}, cycles={cycles}, cap={cap}s -> L={L0} TRs = {L0*TR:.0f}s) ===", flush=True)
    with mp.Pool(4) as pool:
        outs = pool.map(worker_wavelet, [(ts_all[i], f, cycles, cap) for i in range(len(ts_all))], chunksize=16)
    V1_all = np.concatenate([o[0] for o in outs], axis=0)
    Ls = [o[1] for o in outs]
    print(f"  V1 computed {V1_all.shape} in {time.time()-t1:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(V1_all.shape[0], min(N_SUB_KM, V1_all.shape[0]), replace=False)
    km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(V1_all[idx])
    lab_all = km.predict(V1_all).astype(np.int16)
    del V1_all
    labs_list, pos = [], 0
    for T in Ts_loader:
        labs_list.append(lab_all[pos:pos+T]); pos += T
    print(f"  kmeans+labels in {time.time()-t1:.0f}s", flush=True)
    r_notrim, of, o1, o2 = half_occ_stats(labs_list, Ls, sites, trim=False)
    r_trim, _, _, _ = half_occ_stats(labs_list, Ls, sites, trim=True)
    r_str2, _, _, _ = half_occ_stats(labs_list, Ls, sites, trim=False, stride=2)
    results[name] = {"L_TR": L0, "L_secs": L0*TR, "notrim": r_notrim, "trim": r_trim, "stride2_notrim": r_str2}
    occ_store[name] = {"occ_full": of, "occ_h1": o1, "occ_h2": o2}
    # binomial prediction with N_placements
    npl = r_notrim["mean_N_placements"]/2.0
    pbar = np.mean([s.get("p_bar", 0.2) for s in r_notrim["per_state"]])
    results[name]["binom_pred_sigma2W"] = float(pbar*(1-pbar)/max(npl, 1e-9))
    print(f"  mean SB(notrim)={r_notrim['mean_sb']:.3f} siteres={r_notrim['mean_sb_siteres']:.3f} | "
          f"SB(trim)={r_trim['mean_sb']:.3f} siteres={r_trim['mean_sb_siteres']:.3f} | "
          f"stride2 SB={r_str2['mean_sb']:.3f} | N_plac/run={r_notrim['mean_N_placements']:.1f} "
          f"dwell={r_notrim['mean_dwell_TR']:.1f}TR sigma2W={r_notrim['mean_sigma2_W']:.5f} "
          f"binom={results[name]['binom_pred_sigma2W']:.5f} ({time.time()-t1:.0f}s)", flush=True)

# ---------------- Hilbert geometry ----------------
name, lo, hi = HILB
t1 = time.time()
with mp.Pool(4) as pool:
    outs = pool.map(worker_hilb, [(ts_all[i], lo, hi) for i in range(len(ts_all))], chunksize=16)
V1_all = np.concatenate([o[0] for o in outs], axis=0)
Ls = [o[1] for o in outs]
edges = [o[2] for o in outs]
rng = np.random.default_rng(0)
idx = rng.choice(V1_all.shape[0], min(N_SUB_KM, V1_all.shape[0]), replace=False)
km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(V1_all[idx])
lab_all = km.predict(V1_all).astype(np.int16)
del V1_all
labs_list, pos = [], 0
for T in Ts_loader:
    labs_list.append(lab_all[pos:pos+T]); pos += T
# trimmed lengths (edge effects already handled inside worker: edges trimmed there)
r_notrim, of, o1, o2 = half_occ_stats(labs_list, Ls, sites, trim=False)
results[name] = {"lo": lo, "hi": hi, "L_TR": int(np.mean(Ls)), "L_secs": float(np.mean(Ls))*TR,
                 "edge_trim_mean_TR": float(np.mean(edges)),
                 "notrim": r_notrim, "trim": half_occ_stats(labs_list, Ls, sites, trim=True)[0]}
occ_store[name] = {"occ_full": of, "occ_h1": o1, "occ_h2": o2}
print(f"\n=== {name}: SB={r_notrim['mean_sb']:.3f} siteres={r_notrim['mean_sb_siteres']:.3f} ({time.time()-t1:.0f}s)", flush=True)

# ---------------- cache reproduction (exp/36 Fact A) ----------------
print("\n=== cache repro (V1_adhd200_wavelet.npy, locked dict) ===", flush=True)
V1c = np.load(f"{BASE}/results/V1_adhd200_wavelet.npy", mmap_mode="r")
cent = np.load(f"{BASE}/results/ADHD200_A1_centers.npy")
pos, labs_locked, labs_fresh = 0, [], []
sub_rows = []
for T in Ts_loader:
    sub_rows.append(np.asarray(V1c[pos:pos+T])); pos += T
for r in sub_rows:
    d = r @ cent.T
    labs_locked.append(np.argmax(np.abs(d), axis=1).astype(np.int16))
r_lock, ofl, o1l, o2l = half_occ_stats(labs_locked, [kernel_len(TR, .05, 5, 60)]*len(sub_rows), sites, trim=False)
results["CACHE_repro_locked_dict"] = r_lock
print(f"  locked-dict SB per state: {[round(s.get('sb',float('nan')),3) for s in r_lock['per_state']]}", flush=True)
print(f"  mean SB={r_lock['mean_sb']:.3f} siteres={r_lock['mean_sb_siteres']:.3f} (exp/36 Fact A: 0.32-0.51)", flush=True)
# striding diagnostic on locked labels
for s in [2, 3, 5]:
    rs, _, _, _ = half_occ_stats(labs_locked, [kernel_len(TR, .05, 5, 60)]*len(sub_rows), sites, trim=False, stride=s)
    results[f"CACHE_repro_stride{s}"] = {"mean_sb": rs["mean_sb"], "mean_dwell_TR": rs["mean_dwell_TR"], "n_valid": rs["n_valid"]}
    print(f"  stride {s}: SB={rs['mean_sb']:.3f} dwell={rs['mean_dwell_TR']:.1f}TR", flush=True)

# ---------------- band stacks ----------------
print("\n=== band stacks ===", flush=True)
def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)

def stack_stats(names, key):
    occ_f = np.hstack([occ_store[n]["occ_full"] for n in names])
    occ_1 = np.hstack([occ_store[n]["occ_h1"] for n in names])
    occ_2 = np.hstack([occ_store[n]["occ_h2"] for n in names])
    valid = np.ones(len(occ_f), bool)
    per = []
    for j in range(occ_f.shape[1]):
        x1, x2 = occ_1[:, j], occ_2[:, j]
        if np.std(x1) < 1e-9 or np.std(x2) < 1e-9: continue
        rr = float(np.corrcoef(x1, x2)[0, 1])
        rres = float(np.corrcoef(site_resid(x1, sites), site_resid(x2, sites))[0, 1])
        per.append({"comp": j, "r": rr, "sb": sb(rr), "sb_siteres": sb(rres)})
    msb = float(np.mean([p["sb"] for p in per])); msbres = float(np.mean([p["sb_siteres"] for p in per]))
    results[key] = {"components": len(per), "mean_sb": msb, "mean_sb_siteres": msbres, "per_component": per}
    print(f"  {key}: {len(per)} comps mean SB={msb:.3f} siteres={msbres:.3f}", flush=True)
    return occ_f, occ_1, occ_2

try:
    stack_stats(["g050c5K60", "g080c3nat"], "STACK_050_080_concat10")
    stack_stats(["g050c5K60", "g070c4nat", "g090c3nat"], "STACK_3band_concat15")
except KeyError as e:
    print(f"  stack skipped ({e})", flush=True)

# Hungarian-aligned per-state z-mean (fused 5-dim) for the 2-band stack
try:
    from itertools import product
    kmA = None
    # recompute centroids? use occ correlation alignment instead: align states by their
    # between-subject occ correlation across the two bands
    A = occ_store["g050c5K60"]["occ_full"]; B = occ_store["g080c3nat"]["occ_full"]
    Cmat = np.array([[abs(float(np.corrcoef(A[:, i], B[:, j])[0, 1])) for j in range(K)] for i in range(K)])
    ri, ci = linear_sum_assignment(-Cmat)
    fused_f = 0.5*(z(A) + z(B)[:, ci])
    f1 = 0.5*(z(occ_store["g050c5K60"]["occ_h1"]) + z(occ_store["g080c3nat"]["occ_h2"]*0+occ_store["g080c3nat"]["occ_h1"])[:, ci])
    f2 = 0.5*(z(occ_store["g050c5K60"]["occ_h2"]) + z(occ_store["g080c3nat"]["occ_h2"])[:, ci])
    per = []
    for j in range(K):
        x1, x2 = f1[:, j], f2[:, j]
        if np.std(x1) < 1e-9 or np.std(x2) < 1e-9: continue
        rr = float(np.corrcoef(x1, x2)[0, 1])
        rres = float(np.corrcoef(site_resid(x1, sites), site_resid(x2, sites))[0, 1])
        per.append({"state": j, "r": rr, "sb": sb(rr), "sb_siteres": sb(rres)})
    results["STACK_050_080_fused5"] = {"mean_sb": float(np.mean([p["sb"] for p in per])),
                                       "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per])),
                                       "per_state": per, "align_cost": float(Cmat[ri, ci].mean())}
    print(f"  STACK fused5: mean SB={results['STACK_050_080_fused5']['mean_sb']:.3f} "
          f"siteres={results['STACK_050_080_fused5']['mean_sb_siteres']:.3f} "
          f"align|C|={results['STACK_050_080_fused5']['align_cost']:.3f}", flush=True)
except Exception as e:
    print(f"  fused stack failed: {e}", flush=True)

with open(f"{OUT}/pilot_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
np.savez_compressed(f"{OUT}/pilot_occ.npz",
                    **{f"{n}_{w}": occ_store[n][w] for n in occ_store for w in ["occ_full", "occ_h1", "occ_h2"]},
                    sites=sites)
print(f"\nDONE in {time.time()-t0:.0f}s -> pilot_results.json", flush=True)
