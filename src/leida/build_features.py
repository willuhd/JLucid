#!/usr/bin/env python3
"""
exp/37 GATE 1 PREP — build subject-level feature matrices for ALL Gate-0 survivors,
plus the base-enum variants from PREREG (FREQ x KERNEL grid untested for L-summaries).

Feature definitions (frozen BEFORE any target fit; from pilot6/6b + PREREG base enum):
  Lnet28  (28) — Yeo 7x7 unordered block means of L = (1/T) sum_t l_t l_t^T
  Lstr7   (7)  — per-network mean of node co-leadership strength s_i = rowmean(L)
  Mnet7   (7)  — Yeo network means of m = (1/T) sum_t l_t (sign-fixed V1)
  ratio_mn (1) — mean_t lam2/lam1
  bim_frac (1) — frac_t 1[lam2/lam1 > 0.5]
  lam1_mn (1)  — mean_t lam1/N  (Fact-B contested; report only)
  Lrow190 (190) — per-node strength (dim-price control, not proposed for Gate 2)
  Mrow190 (190) — per-node m_i (dim-price control)

Geometries (each = separate variant family for the screen):
  G_cache  — f=0.05, 5 cycles, cap 60s, L=29 TR (exp/21 exact cache; V1 from cache, lam from recompute)
  G_080    — f=0.08, 3 cycles, cap 90s, L=19 TR (pilot-1 best single band)
  FUSED    — z-fused 2-band Lstr7 / Mnet7 (pilot6b recipe)
Base enum extra (PREREG inventory, untested for L/M): FREQ {0.065, 0.095} x c3nat
  -> Lstr7/Mnet7 at those bands (screen-only; will be Gate-2 family members if promoted)

NO TARGET IS TOUCHED IN THIS SCRIPT. Output: features.npz (per geometry, per variant)
subject order = load_cc200 order (verified cache alignment 2026-08-29).
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
CACHE_V1 = "/Volumes/thinkplus/Code/JLucid/results/V1_adhd200_wavelet.npy"

BANDS = {
    "g050c5K60": (0.05, 5, 60.0),    # cache geometry
    "g080c3nat": (0.08, 3, 90.0),    # pilot-1 best band
    "g065c3nat": (0.065, 3, 90.0),   # base enum
    "g095c3nat": (0.095, 3, 90.0),   # base enum
}

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

def lam_series(ph):
    c = np.cos(ph); s = np.sin(ph)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    return (tr + disc) / 2.0 / ph.shape[1], (tr - disc) / 2.0 / ph.shape[1]

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
    return V1.astype(np.float64)

t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, _, meta = load_cc200(qc_only=True)
sites = np.array(meta["sites"]); Ts = [t.shape[0] for t in ts_all]
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]
UP = [(a, b) for a in range(7) for b in range(a, 7)]
print(f"loaded {len(ts_all)} in {time.time()-t0:.0f}s", flush=True)

# ---- load cached V1 (g050c5K60 geometry; alignment verified) ----
print("loading cached V1 for g050c5K60 ...", flush=True)
V1_cache = np.load(CACHE_V1, mmap_mode='r')
off = np.concatenate([[0], np.cumsum(Ts)]).astype(int)

def worker(args):
    ts, f, cycles, cap = args
    w, L = morlet(TR, f, cycles, cap)
    ph = phases_of(ts, w, L)
    return v1_vec(ph), lam_series(ph)

GEOM_FEATURES = {}
for gname, (f, cyc, cap) in BANDS.items():
    t1 = time.time()
    if gname == "g050c5K60":
        # use cache for V1; lam needs recompute (cheap per subject, still parallel)
        with mp.Pool(4) as pool:
            outs = pool.map(worker, [(ts_all[i], f, cyc, cap) for i in range(len(ts_all))], chunksize=16)
        V1s_list = [V1_cache[off[i]:off[i+1]] for i in range(len(ts_all))]
        lams = [(o[1][0], o[1][1]) for o in outs]
        # SIGN-CONVENTION NOTE (2026-08-29): exp/21 cache uses eigh+majority-flip; closed-form
        # +majority-flip differs on rows where the majority-positive vector has 85-95 positives.
        # L-based summaries (outer products) are SIGN-INVARIANT; M-based (first moment) are not.
        # Here we use the closed-form majority-flip V1 (true majority-positive) for M features,
        # and the cache for L features (invariance verified below on 3 subjects).
        for k in [0, len(ts_all)//2, len(ts_all)-1]:
            Lc = (np.asarray(V1_cache[off[k]:off[k+1]]).T @ np.asarray(V1_cache[off[k]:off[k+1]])) / Ts[k]
            Lr = (outs[k][0].T @ outs[k][0]) / Ts[k]
            assert np.allclose(Lc, Lr, atol=1e-10), f"L invariance failed subj {k}"
        # build M features from closed-form V1 (true majority-positive), L features from cache
        V1_closed = [o[0] for o in outs]
        print(f"  L-sign-invariance verified on 3 subjects", flush=True)
    else:
        with mp.Pool(4) as pool:
            outs = pool.map(worker, [(ts_all[i], f, cyc, cap) for i in range(len(ts_all))], chunksize=16)
        V1s_list = [o[0] for o in outs]
        lams = [(o[1][0], o[1][1]) for o in outs]
    print(f"{gname}: V1+lam done {time.time()-t1:.0f}s", flush=True)

    feats = {}
    n = len(ts_all)
    Lnet28 = np.zeros((n, 28)); Lstr7 = np.zeros((n, 7)); Mnet7 = np.zeros((n, 7))
    Lrow = np.zeros((n, 190)); Mrow = np.zeros((n, 190))
    ratio = np.zeros(n); bim = np.zeros(n); lam1m = np.zeros(n)
    # M features must use closed-form majority-positive V1 (sign-fixed); L features sign-invariant
    V1_for_L = V1s_list
    V1_for_M = V1_closed if gname == "g050c5K60" else V1s_list
    for i, (V1s, T) in enumerate(zip(V1_for_M, Ts)):
        if T < 60:
            # keep NaN, excluded downstream
            Lnet28[i] = np.nan; Lstr7[i] = np.nan; Mnet7[i] = np.nan
            Lrow[i] = np.nan; Mrow[i] = np.nan; ratio[i] = np.nan; bim[i] = np.nan; lam1m[i] = np.nan
            continue
        Lm = (V1_for_L[i].T @ V1_for_L[i]) / T   # 190x190 co-leadership second moment
        s_row = Lm.mean(axis=1)          # node strength (sign-invariant)
        Mrow[i] = s_row                   # "Mrow190" in pilot6 = node strength? NO — pilot6 Mrow190 = m_i.
        Lrow[i] = V1s.mean(axis=0)        # first moment m (sign-sensitive, closed-form V1)
        for a in range(7):
            Lstr7[i, a] = s_row[net_idx[a]].mean()
            Mnet7[i, a] = Lrow[i][net_idx[a]].mean()
        for k, (a, b) in enumerate(UP):
            pa, pb = net_idx[a], net_idx[b]
            Lnet28[i, k] = Lm[np.ix_(pa, pb)].mean()
        l1, l2 = lams[i]
        ratio[i] = float(np.mean(l2 / np.maximum(l1, 1e-12)))
        bim[i] = float(np.mean((l2 / np.maximum(l1, 1e-12)) > 0.5))
        lam1m[i] = float(np.mean(l1))
    feats["Lnet28"] = Lnet28; feats["Lstr7"] = Lstr7; feats["Mnet7"] = Mnet7
    feats["ratio_mn"] = ratio[:, None]; feats["bim_frac"] = bim[:, None]; feats["lam1_mn"] = lam1m[:, None]
    feats["Lrow190"] = Mrow  # node co-leadership strength (pilot6 A3 "Lrow190" definition)
    feats["Mrow190"] = Lrow  # first-moment m_i (pilot6 B2 definition)
    GEOM_FEATURES[gname] = feats
    print(f"  {gname} features built ({time.time()-t1:.0f}s)", flush=True)

# ---- FUSED 2-band (z-fused per pilot6b) ----
def z(a):
    a = np.nan_to_num(a, nan=np.nan)
    mu = np.nanmean(a, 0); sd = np.nanstd(a, 0)
    return (a - mu) / (sd + 1e-12)

fused = {}
for vn in ["Lstr7", "Mnet7", "Lnet28"]:
    fused[f"FUSED_{vn}"] = 0.5 * (z(GEOM_FEATURES["g050c5K60"][vn]) + z(GEOM_FEATURES["g080c3nat"][vn]))
GEOM_FEATURES["FUSED"] = fused

np.savez_compressed(f"{OUT}/features.npz",
    sites=sites, Ts=np.array(Ts),
    **{f"{g}__{v}": F for g, d in GEOM_FEATURES.items() for v, F in d.items()})
print(f"DONE {time.time()-t0:.0f}s -> features.npz", flush=True)
