#!/usr/bin/env python3
"""
PILOT 7d (Direction 5 addendum 2; reliability-only + feature-only redundancy).
NO target touched. Pre-committed BEFORE running:

  PC1-RESCUE for the measured-dead D5-B state-conditional FC family:
    For hard gate and soft gate, per state s: PC1 (first principal component) of the
    28 block-mean dFC features, PCA fit on FULL-run features (no target), applied to
    h1/h2; report PC1 split-half SB. Reliability-first logic: if the 28 per-block
    halves share a common latent state-shift factor, PC1 averages 28 noisy probes
    and can pass where per-block comps die. Also PC1 of the 140-dim concatenation.
  REDUNDANCY (feature-only): R^2 of surviving D5 variants given the occupancy bases
    and static blocks, per pilot6 convention:
      R2[CS14 | occ5_locked], R2[CS14 | occFUSED_3band],
      R2[T10E5_15 | occ5_locked], R2[T10E5_15 | occFUSED_3band],
      R2[CxA_fused | occFUSED_3band],
      R2[dFC_pc1_* | FC28].
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from sklearn.cluster import KMeans
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
N_SUB_KM = 40000
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]
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

def v1_of(phases):
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

def blockmeans28(M, net_idx):
    out = np.zeros(28)
    for c, (a, b) in enumerate(UP):
        ia, ib = net_idx[a], net_idx[b]
        blk = M[np.ix_(ia, ib)]
        if a == b:
            iu = np.triu_indices(len(ia), k=1)
            out[c] = blk[iu].mean() if len(iu[0]) else 0.0
        else:
            out[c] = blk.mean()
    return out

def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2 * r / (1 + abs(r)) if r > -1 else -1

def scal_stats(name, x1, x2, sites, mask, results):
    ok = np.isfinite(x1) & np.isfinite(x2)
    ok &= mask
    if ok.sum() < 50 or np.std(x1[ok]) < 1e-12:
        print(f"  {name:26s} DEGENERATE", flush=True); return
    r = float(np.corrcoef(x1[ok], x2[ok])[0, 1])
    rr = float(np.corrcoef(site_resid(x1[ok], sites[ok]), site_resid(x2[ok], sites[ok]))[0, 1])
    results[name] = {"r": r, "sb": sb(r), "sb_siteres": sb(rr), "n_used": int(ok.sum())}
    print(f"  {name:26s} SB={sb(r):.3f} siteres={sb(rr):.3f} (n={ok.sum()})", flush=True)

def redundancy(name, F, FSTAT, blocks, results):
    r2s = []
    for j in range(F.shape[1]):
        y = F[:, j]
        ok = np.isfinite(y) & np.isfinite(FSTAT).all(1)
        if ok.sum() < 100 or np.std(y[ok]) < 1e-12:
            continue
        X = np.column_stack([np.ones(ok.sum()), FSTAT[ok]])
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        pred = X @ beta
        r2s.append(1.0 - float(np.var(y[ok] - pred)) / float(np.var(y[ok])))
    results[f"redundancy_{name}"] = {"blocks": blocks, "mean_R2": float(np.mean(r2s)) if r2s else None,
                                     "n_comp": len(r2s)}
    print(f"  R2[{name} | {blocks}] = {np.mean(r2s):.3f} (n={len(r2s)})", flush=True)

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

    V1c = np.load(f"{BASE}/results/V1_adhd200_wavelet.npy")
    cent = np.load(f"{BASE}/results/ADHD200_A1_centers.npy")
    yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
    net_idx = [np.array(yeo[n]) for n in NETS]
    FSTATIC = np.load(f"{BASE}/results/F_STATIC.npy")
    Z = np.load(f"{OUT}/pilot7_d5_raw.npz")
    occF_lock = Z["occF"]; CSF = Z["CSF"]; DDF = Z["DDF"]
    dFCH_F = Z["dFCF"]  # (nS, K, 28) hard-gate full-run

    subs = []; pos = 0
    for T in Ts:
        subs.append((pos, pos + T)); pos += T

    results = {}

    # ---------- recompute soft-gate dFC (h1/h2/full) ----------
    print("\nsoft-gate dFC pass ...", flush=True)
    t1 = time.time()
    dFCS = {k: np.full((nS, K, 28), np.nan) for k in ("h1", "h2", "full")}
    for i in range(nS):
        ts = np.asarray(ts_all[i], float)
        v1 = V1c[subs[i][0]:subs[i][1]]
        T = ts.shape[0]
        for tag, sl in (("h1", slice(0, T // 2)), ("h2", slice(T // 2, T)), ("full", slice(0, T))):
            X = ts[sl]; W = v1[sl]
            if X.shape[0] < 20:
                continue
            zall = np.arctanh(np.clip(np.nan_to_num(np.corrcoef(X.T)), -0.999999, 0.999999))
            ball_ = blockmeans28(zall, net_idx)
            sims = (W @ cent.T) ** 2
            for s in range(K):
                w = sims[:, s]; sw = w.sum()
                if sw < 1e-9:
                    continue
                mu = (w @ X) / sw
                Xc = X - mu
                S = (Xc * w[:, None]).T @ Xc / sw
                d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
                C = S / np.outer(d, d)
                z = np.arctanh(np.clip(np.nan_to_num(C), -0.999999, 0.999999))
                dFCS[tag][i, s] = blockmeans28(z, net_idx) - ball_
        if i % 400 == 0:
            print(f"  subj {i}/{nS} ({time.time()-t1:.0f}s)", flush=True)
    print(f"soft pass done {time.time()-t1:.0f}s", flush=True)

    # ---------- PC1 rescue: PCA fit on full-run, apply to halves ----------
    print("\nPC1 rescue (fit on full-run 28-blocks per state; applied to h1/h2) ...", flush=True)
    for gate, dH1, dH2, dFF in [("hard", Z["dFC1"], Z["dFC2"], dFCH_F),
                                 ("soft", dFCS["h1"], dFCS["h2"], dFCS["full"])]:
        for s in range(K):
            Ff = dFF[:, s, :]
            ok = np.isfinite(Ff).all(1)
            if ok.sum() < 100:
                scal_stats(f"pc1_{gate}_s{s}", np.full(nS, np.nan), np.full(nS, np.nan), sites, mask60, results)
                continue
            mu = Ff[ok].mean(0); sd = Ff[ok].std(0) + 1e-12
            Fz = (Ff - mu) / sd
            # PC1 from the covariance of z-scored full-run features (usable subjects)
            U, S_, Vt = np.linalg.svd(Fz[ok], full_matrices=False)
            pc1 = Vt[0]
            p1 = ((dH1[:, s, :] - mu) / sd) @ pc1
            p2 = ((dH2[:, s, :] - mu) / sd) @ pc1
            scal_stats(f"pc1_{gate}_s{s}", p1, p2, sites, mask60, results)
        # PC1 across the 4 adequately-usable states' blocks (4x28=112 dims); state-1 rows are
        # NaN for 80% of subjects (n_s>=8 both halves in only 20%), so all-140 is degenerate.
        keep_states = [0, 2, 3, 4]
        Ff = dFF[:, keep_states, :].reshape(nS, -1)
        okr = np.isfinite(Ff).all(1)
        mu = Ff[okr].mean(0); sd = Ff[okr].std(0) + 1e-12
        U, S_, Vt = np.linalg.svd((Ff[okr] - mu) / sd, full_matrices=False)
        pc1 = Vt[0]
        p1 = ((dH1[:, keep_states, :].reshape(nS, -1) - mu) / sd) @ pc1
        p2 = ((dH2[:, keep_states, :].reshape(nS, -1) - mu) / sd) @ pc1
        scal_stats(f"pc1_{gate}_all112", p1, p2, sites, mask60, results)
        results[f"pc1_{gate}_all112"]["var_expl"] = float(S_[0] ** 2 / np.sum(S_ ** 2))

    # ---------- fused 3-band occ (pilot-2 verbatim) for redundancy ----------
    print("\nfused 3-band occ for redundancy ...", flush=True)
    occ_h = {}
    for name, (f, c, cap) in BANDS.items():
        w, L = morlet(TR, f, c, cap)
        V1s = [v1_of(wavelet_phase(np.asarray(ts, float), w, L)) for ts in ts_all]
        Vall = np.concatenate(V1s, 0)
        rng_ = np.random.default_rng(0)
        idx = rng_.choice(Vall.shape[0], min(N_SUB_KM, Vall.shape[0]), replace=False)
        km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=100).fit(Vall[idx])
        lab = km.predict(Vall).astype(np.int16)
        of = []; pos = 0
        for T in Ts:
            of.append(np.bincount(lab[pos:pos + T], minlength=K) / max(T, 1)); pos += T
        occ_h[name] = np.array(of)
    from scipy.optimize import linear_sum_assignment
    ref = "b050"; A = occ_h[ref]
    perms = {ref: np.arange(K)}
    for n_ in ["b070", "b090"]:
        B = occ_h[n_]
        C = np.array([[abs(float(np.corrcoef(A[:, i], B[:, j])[0, 1])) for j in range(K)] for i in range(K)])
        C = np.nan_to_num(C)
        ri, ci = linear_sum_assignment(-C)
        perms[n_] = ci
    def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-12)
    ofF = z(occ_h[ref])
    for n_ in ["b070", "b090"]:
        ofF = ofF + z(occ_h[n_])[:, perms[n_]]
    ofF /= 3.0
    CxAF = ofF * age_z[:, None]

    print("\nredundancy of survivors vs occupancy bases and static blocks ...", flush=True)
    redundancy("CS14_vs_occ5lock", CSF, occF_lock, "occ5_locked", results)
    redundancy("CS14_vs_occFUSED", CSF, ofF, "occFUSED_3band", results)
    redundancy("T10E5_vs_occ5lock", DDF, occF_lock, "occ5_locked", results)
    redundancy("T10E5_vs_occFUSED", DDF, ofF, "occFUSED_3band", results)
    redundancy("CxAfused_vs_occFUSED", CxAF, ofF, "occFUSED_3band", results)
    redundancy("CS14_vs_occFC28", CSF, np.column_stack([occF_lock, FSTATIC[:, 0:28]]), "occ5+FC28", results)
    redundancy("T10E5_vs_occFC28", DDF, np.column_stack([occF_lock, FSTATIC[:, 0:28]]), "occ5+FC28", results)

    results["wall_secs"] = int(time.time() - t0)
    with open(f"{OUT}/pilot7d_d5_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> pilot7d_d5_results.json", flush=True)
