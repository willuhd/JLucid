#!/usr/bin/env python3
"""
PILOT 6 (Direction 3 — LEADERSHIP STRUCTURE; reliability-only Gate-0 previews).
NO target is ever touched here (no Inatt/Hyper/IQ) — reliability only, per exp/37 PREREG.
Pre-committed candidate list below; every point reported, including expected-dead ones.
These are Gate-0 PREVIEWS, to be re-run formally by the sieve; any survivor carries its
whole pre-committed grid into the Gate-2 family (same rule as pilots 1-3).

Cache design point used throughout: FREQ=0.05 Hz, 5 cycles, cap 60s -> L=29 TRs (TR=2.0),
i.e. EXACTLY the exp/21 V1/V2 cache geometry. Split-half = first vs second half of each
run; SB = 2r/(1+r) (Pearson across subjects, n with T>=60 per PREREG).
Site-residualized variant reported alongside raw (sites differ in T and TR).

Pre-committed Direction-3 candidate summaries (frozen before running):
  A. co-leadership second moment  L = (1/T) sum_t l_t l_t^T (190x190), sign-invariant:
     A1 Lnet28   — Yeo 7x7 unordered block means (28 comps)
     A2 Lstrnet7 — per-network mean of node co-leadership strength s_i = rowmean(L) (7)
     A3 Lrow190  — per-node strength s_i (190 comps)
  B. first moment (leader recruitment) m = (1/T) sum_t l_t (sign-fixed cache V1):
     B1 Mnet7    — Yeo network means of m (7)
     B2 Mrow190  — per-node m_i (190 comps)
  C. second axis (V2 cache; sign-invariant summaries only):
     C1 V2net28  — Yeo block means of L2 = (1/T) sum_t v2_t v2_t^T (28)
     C2 V2abs7   — Yeo network means of (1/T) sum_t |v2_t| (7)
     C3 bimod    — needs lam1/lam2 series (one extra wavelet pass, exact cache geometry):
                   bim_frac  = mean_t 1[lam2/lam1 > 0.5]      (scalar)
                   ratio_mn  = mean_t lam2/lam1              (scalar)
                   lam1_mn   = mean_t lam1/N                 (scalar; FACT-B contested, report only)
  D. leader-sequence persistence (locked exp/21 k=5 dict labels on cached V1):
     D1 pers_R   — mean run length over ALL states Rbar = T_h/N_runs (scalar, pooled)
     D2 pers_phi — lag-1 label persistence phi = P(l_t == l_{t+1}) (scalar)
     D3 dwell5   — per-state mean run length (5 comps; expected dead, documented)
  E. leader-pattern temporal autocorr (continuous, lam1-magnitude-free):
     E1-E4 Ak    — mean_t <l_t, l_{t+k}>, k in {1,4,8,16} (4 scalars)
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
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
LAGS = [1, 4, 8, 16]

# ---------------- exact cache geometry (f=.05, 5 cycles, cap 60s -> L=29) ----------------
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
    Fk = np.fft.fft(w, n=n_fft)
    Fd = np.fft.fft(ts, n=n_fft, axis=0)
    return np.angle(np.fft.ifft(Fd * Fk[:, None], axis=0)[L // 2:L // 2 + T])

def lam_series(ph):
    """lam1, lam2 of the 2x2 phase-coherence matrix per window, normalized by N."""
    c = np.cos(ph); s = np.sin(ph)
    a = (c * c).sum(1); b = (c * s).sum(1); d = (s * s).sum(1)
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    return (tr + disc) / 2.0 / ph.shape[1], (tr - disc) / 2.0 / ph.shape[1]

def worker_lam(args):
    ts, = args
    w, L = morlet(TR, 0.05, 5, 60.0)
    l1, l2 = lam_series(phases_of(ts, w, L))
    return l1.astype(np.float32), l2.astype(np.float32)

# ---------------- stats helpers (verbatim protocol from pilots 1-3) ----------------
def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2 * r / (1 + abs(r)) if r > -1 else -1

def vec_stats(name, F1, F2, Ff, sites, results, comps_name="per_comp"):
    """F1,F2,Ff: (n, d) half1/half2/full features. Per-component r/SB + mean; ANOVA noise."""
    n, d = F1.shape
    per = []
    ok_sb, ok_sb_res = [], []
    sw2s = []
    for j in range(d):
        x1, x2, xf = F1[:, j], F2[:, j], Ff[:, j]
        if np.std(x1) < 1e-12 or np.std(x2) < 1e-12:
            per.append({"comp": j, "r": None}); continue
        r = float(np.corrcoef(x1, x2)[0, 1])
        rres = float(np.corrcoef(site_resid(x1, sites), site_resid(x2, sites))[0, 1])
        dd = x1 - x2
        sw2 = float(np.mean(dd * dd) / 2.0)
        st2 = max(float(np.var(xf)) - sw2 / 2.0, 1e-12)
        per.append({"comp": j, "r": r, "sb": sb(r), "sb_siteres": sb(rres),
                    "sigma2_W": sw2, "sigma2_T": st2, "r_pred_anova": st2 / (st2 + sw2)})
        ok_sb.append(sb(r)); ok_sb_res.append(sb(rres)); sw2s.append(sw2)
    if ok_sb:
        results[name] = {"n_comp": len(ok_sb), "comps": d, "mean_sb": float(np.mean(ok_sb)),
                         "mean_sb_siteres": float(np.mean(ok_sb_res)),
                         "mean_sigma2_W": float(np.mean(sw2s)),
                         comps_name: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in per]}
        print(f"  {name:14s} d={d:3d} meanSB={results[name]['mean_sb']:.3f} "
              f"siteres={results[name]['mean_sb_siteres']:.3f} sigma2W={results[name]['mean_sigma2_W']:.5f}", flush=True)

def scal_stats(name, x1, x2, xf, sites, results):
    if np.std(x1) < 1e-12 or np.std(x2) < 1e-12:
        print(f"  {name:14s} DEGENERATE", flush=True); return
    r = float(np.corrcoef(x1, x2)[0, 1])
    rres = float(np.corrcoef(site_resid(x1, sites), site_resid(x2, sites))[0, 1])
    dd = x1 - x2
    sw2 = float(np.mean(dd * dd) / 2.0)
    st2 = max(float(np.var(xf)) - sw2 / 2.0, 1e-12)
    results[name] = {"r": r, "sb": sb(r), "sb_siteres": sb(rres), "sigma2_W": sw2,
                     "sigma2_T": st2, "r_pred_anova": st2 / (st2 + sw2)}
    print(f"  {name:14s} r={r:+.3f} SB={sb(r):.3f} siteres={sb(rres):.3f} "
          f"sw2={sw2:.5f} st2={st2:.5f} pred={st2/(st2+sw2):.3f}", flush=True)

# ---------------- load ----------------
t0 = time.time()
print("loading cc200 ...", flush=True)
ts_all, labels_all, meta = load_cc200(qc_only=True)
sites = meta["sites"]; Ts_loader = [t.shape[0] for t in ts_all]
print(f"loaded {len(ts_all)} subjects in {time.time()-t0:.0f}s total_T={sum(Ts_loader)}", flush=True)

V1c = np.load(f"{BASE}/results/V1_adhd200_wavelet.npy")
V2c = np.load(f"{BASE}/results/V2_adhd200_wavelet.npy")
cent = np.load(f"{BASE}/results/ADHD200_A1_centers.npy")
yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
net_idx = [np.array(yeo[n]) for n in NETS]
net_len = np.array([len(x) for x in net_idx])
print(f"V1 {V1c.shape} V2 {V2c.shape} | Yeo sizes {dict(zip(NETS, net_len.tolist()))} sum={net_len.sum()}", flush=True)

# split subjects, respecting concatenation order
subs = []; pos = 0
for T in Ts_loader:
    subs.append((pos, pos + T)); pos += T
assert pos == V1c.shape[0]

# labels from locked dictionary (verbatim run_pilot.py repro)
labs_all = np.argmax(np.abs(V1c @ cent.T), axis=1).astype(np.int16)

# lam1/lam2 series: one extra wavelet pass, exact cache geometry
print("lam pass (exact cache geometry, 4 cores) ...", flush=True)
t1 = time.time()
with mp.Pool(4) as pool:
    lam_out = pool.map(worker_lam, [(ts_all[i],) for i in range(len(ts_all))], chunksize=16)
print(f"lam pass done in {time.time()-t1:.0f}s", flush=True)

# ---------------- build half-features ----------------
print("\nbuilding half-features ...", flush=True)
store = {}
def append(store, key, arr):
    store.setdefault(key, []).append(arr)

for si, (a, b) in enumerate(subs):
    T = b - a
    v1 = V1c[a:b]; v2 = V2c[a:b]; lab = labs_all[a:b]
    l1s, l2s = lam_out[si]
    h = T // 2
    for tag, sl in (("h1", slice(0, h)), ("h2", slice(h, T))):
        V1h = v1[sl]; V2h = v2[sl]; labh = lab[sl]
        Th = V1h.shape[0]
        if Th < 20:
            # too short: record NaN rows to keep alignment, flagged by T>=60 mask anyway
            for key, d in (("Lnet28", 28), ("Lstrnet7", 7), ("Lrow190", 190), ("Mnet7", 7),
                           ("Mrow190", 190), ("V2net28", 28), ("V2abs7", 7),
                           ("dwell5", 5), ("pers_R", 1), ("pers_phi", 1), ("bim_frac", 1),
                           ("ratio_mn", 1), ("lam1_mn", 1), ("A1", 1), ("A4", 1), ("A8", 1), ("A16", 1)):
                append(store, key, np.full(d, np.nan))
            continue
        # A: co-leadership second moment (sign-invariant)
        L = (V1h.T @ V1h) / Th
        blocks = []
        for i in range(7):
            for j in range(i, 7):
                blocks.append(float(L[np.ix_(net_idx[i], net_idx[j])].mean()))
        append(store, "Lnet28", np.array(blocks))
        srow = L.mean(axis=1)
        append(store, "Lstrnet7", np.array([srow[ix].mean() for ix in net_idx]))
        append(store, "Lrow190", srow)
        # B: first moment (cache V1 sign-fixed: majority-positive)
        m = V1h.mean(axis=0)
        append(store, "Mnet7", np.array([m[ix].mean() for ix in net_idx]))
        append(store, "Mrow190", m)
        # C: second axis (sign-invariant)
        L2 = (V2h.T @ V2h) / Th
        blocks2 = []
        for i in range(7):
            for j in range(i, 7):
                blocks2.append(float(L2[np.ix_(net_idx[i], net_idx[j])].mean()))
        append(store, "V2net28", np.array(blocks2))
        a2 = np.abs(V2h).mean(axis=0)
        append(store, "V2abs7", np.array([a2[ix].mean() for ix in net_idx]))
        # C3: bimodality scalars
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(l1s[sl] > 1e-12, l2s[sl] / np.maximum(l1s[sl], 1e-12), np.nan)
            ratio = ratio[~np.isnan(ratio)]
        append(store, "ratio_mn", np.array([ratio.mean() if len(ratio) else np.nan]))
        append(store, "bim_frac", np.array([(ratio > 0.5).mean() if len(ratio) else np.nan]))
        append(store, "lam1_mn", np.array([float(l1s[sl].mean())]))
        # D: persistence on locked labels
        chg = np.flatnonzero(np.diff(labh) != 0)
        nruns = len(chg) + 1
        append(store, "pers_R", np.array([Th / nruns]))
        append(store, "pers_phi", np.array([1.0 - len(chg) / max(Th - 1, 1)]))
        # per-state dwell: mean run length of runs belonging to each state
        bounds = np.concatenate([[-1], chg, [Th - 1]])
        rlen = np.diff(bounds)
        rlab = labh[bounds[1:]]
        d5 = np.zeros(5)
        for st in range(5):
            sel = rlen[rlab == st]
            d5[st] = sel.mean() if len(sel) else 0.0
        append(store, "dwell5", d5)
        # E: leader-pattern autocorr lags (lam1-magnitude-free)
        for k in LAGS:
            if Th > k:
                ak = float((V1h[:-k] * V1h[k:]).sum(1).mean())
            else:
                ak = np.nan
            append(store, f"A{k}", np.array([ak]))

# assemble
F = {}
for key, lst in store.items():
    F[key] = np.array(lst)  # (n_subj, d)
n_all = F["Lnet28"].shape[0]

# full-run features (for ANOVA sigma2_T)
for si, (a, b) in enumerate(subs):
    pass  # computed inline below in full-features block

full = {}
for key in list(F.keys()):
    arr0 = np.array(F[key])
    full[key] = np.full((arr0.shape[1] and (arr0.shape[0] // 2), arr0.shape[1]), np.nan)
for si, (a, b) in enumerate(subs):
    T = b - a
    v1 = V1c[a:b]; v2 = V2c[a:b]; lab = labs_all[a:b]
    l1s, l2s = lam_out[si]
    if T < 20: continue
    L = (v1.T @ v1) / T
    blocks = [float(L[np.ix_(net_idx[i], net_idx[j])].mean()) for i in range(7) for j in range(i, 7)]
    full["Lnet28"][si] = blocks
    srow = L.mean(axis=1)
    full["Lstrnet7"][si] = [srow[ix].mean() for ix in net_idx]
    full["Lrow190"][si] = srow
    m = v1.mean(axis=0)
    full["Mnet7"][si] = [m[ix].mean() for ix in net_idx]
    full["Mrow190"][si] = m
    L2 = (v2.T @ v2) / T
    blocks2 = [float(L2[np.ix_(net_idx[i], net_idx[j])].mean()) for i in range(7) for j in range(i, 7)]
    full["V2net28"][si] = blocks2
    a2 = np.abs(v2).mean(axis=0)
    full["V2abs7"][si] = [a2[ix].mean() for ix in net_idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(l1s > 1e-12, l2s / np.maximum(l1s, 1e-12), np.nan)
        ratio = ratio[~np.isnan(ratio)]
    full["ratio_mn"][si] = ratio.mean() if len(ratio) else np.nan
    full["bim_frac"][si] = (ratio > 0.5).mean() if len(ratio) else np.nan
    full["lam1_mn"][si] = l1s.mean()
    chg = np.flatnonzero(np.diff(lab) != 0)
    full["pers_R"][si] = T / (len(chg) + 1)
    full["pers_phi"][si] = 1.0 - len(chg) / max(T - 1, 1)
    bounds = np.concatenate([[-1], chg, [T - 1]])
    rlen = np.diff(bounds); rlab = lab[bounds[1:]]
    d5 = np.zeros(5)
    for st in range(5):
        sel = rlen[rlab == st]
        d5[st] = sel.mean() if len(sel) else 0.0
    full["dwell5"][si] = d5
    for k in LAGS:
        if T > k:
            full[f"A{k}"][si] = float((v1[:-k] * v1[k:]).sum(1).mean())

# ---------------- Gate-0 preview stats (n with T>=60, per PREREG) ----------------
valid = np.array([T >= 60 for T in Ts_loader])
print(f"\nGate-0 previews (n={valid.sum()} of {n_all}, T>=60; halves; SB=2r/(1+r))", flush=True)
print("=" * 100, flush=True)
results = {"n_valid": int(valid.sum()), "protocol": "first/second half, Pearson, SB; siteres = site-demeaned halves"}

results_raw = {}
for key in list(F.keys()):
    arr = np.array(F[key])
    d = arr.shape[1]
    h1 = arr[0::2]; h2 = arr[1::2]; ff = np.array(full[key])
    v1_ = h1[valid]; v2_ = h2[valid]; ff_ = ff[valid]
    good = ~(np.isnan(v1_).any(1) | np.isnan(v2_).any(1) | np.isnan(ff_).any(1))
    if good.sum() < 100:
        print(f"  {key}: too few valid ({good.sum()})", flush=True); continue
    if d == 1:
        scal_stats(key, v1_[good, 0], v2_[good, 0], ff_[good, 0], sites[valid][good], results_raw)
    else:
        vec_stats(key, v1_[good], v2_[good], ff_[good], sites[valid][good], results_raw)

# ---------------- anchors: reproduce Fact A/B on this exact pipeline ----------------
print("\nanchor: locked-dict occ per-state SB (exp/36 Fact A: 0.32-0.51) ...", flush=True)
occ1 = []; occ2 = []; occf = []
for si, (a, b) in enumerate(subs):
    T = b - a; lab = labs_all[a:b]; h = T // 2
    if T < 60: continue
    l1h = lab[:h]; l2h = lab[h:]
    occ1.append(np.bincount(l1h, minlength=5) / h)
    occ2.append(np.bincount(l2h, minlength=5) / (T - h))
    occf.append(np.bincount(lab, minlength=5) / T)
occ1 = np.array(occ1); occ2 = np.array(occ2); occf = np.array(occf)
sv = sites[valid]
vec_stats("ANCHOR_occ5", occ1, occ2, occf, sv, results_raw)

print("\nanchor: dwell diagnostics")
dw = []
for si, (a, b) in enumerate(subs):
    T = b - a; lab = labs_all[a:b]
    chg = np.flatnonzero(np.diff(lab) != 0)
    dw.append(T / (len(chg) + 1))
dw = np.array(dw)
print(f"  locked-dict mean run length = {dw.mean():.2f} TR (pilot1 reported 9.49)", flush=True)
print(f"  changes/half mean = {np.mean([ (np.diff(labs_all[a:a+ (b-a)//2]) != 0).sum() for a,b in subs if (b-a)>=60 ]):.2f}", flush=True)

with open(f"{OUT}/pilot6_d3_results.json", "w") as f:
    json.dump(results_raw, f, indent=2, default=float)
print(f"\nDONE in {time.time()-t0:.0f}s -> pilot6_d3_results.json", flush=True)
